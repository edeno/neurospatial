# Designs

[← back to PLAN.md](PLAN.md) · [shared contracts](shared-contracts.md)

This file holds the algorithms behind the phases. Types and invariants are in
[shared-contracts.md](shared-contracts.md); this file doesn't repeat them.

- [Binning from a layout](#binning-from-a-layout) (phase 1, extended in 2a and 2b)
- [Data-backed locators](#data-backed-locators) (phase 2a)
- [Plotting from binning data](#plotting-from-binning-data) (phase 2b)
- [File format](#file-format) (phase 3)
- [Derivation algorithms](#derivation-algorithms) (phase 4)

---

## Binning from a layout

`binning_from_layout(layout, connectivity)` runs inside `_setup_from_layout`
(`src/neurospatial/environment/core.py:921-1041`), after
`validate_connectivity_graph` succeeds and before `_PublishedGeometry` is
replaced.

```python
def binning_from_layout(layout, connectivity):
    if hasattr(layout, "binning"):                       # BinningLayout facade (phase 3+)
        return layout.binning()
    edges = list(connectivity.edges(data=True))         # iteration order = pair order
    pairs = np.array([(u, v) for u, v, _ in edges], dtype=np.intp).reshape(-1, 2)
    distances = np.array([d["distance"] for _, _, d in edges], dtype=np.float64)
    centers = np.asarray(layout.bin_centers, dtype=np.float64)
    bins = SpatialBins.from_arrays(
        centers=centers,
        sizes=np.asarray(layout.bin_sizes(), dtype=np.float64),
        measure_dim=1 if layout.is_linearized_track else centers.shape[1],
        pairs=pairs,
        pair_distances=distances,
    )
    return Binning(
        kind=getattr(layout, "_layout_type_tag", type(layout).__name__),
        bins=bins,
        locator=_locator_of(layout),                    # phase 1: LayoutLocator(layout)
        faces=LayoutFaces(layout, connectivity),
        rectilinear=layout.rectilinear_grid() if isinstance(layout, RectilinearGridProvider) else None,
        linearization=_linearization_of(layout),         # phase 1: layout if TrackLinearizationProvider else None
        shapes=_shapes_of(layout),                       # phase 2b
        outline=_outline_of(layout),                     # phase 2b
    )
```

Detection of a capability uses the layout protocols **only here**, at
publication. In phase 2a, `_locator_of` returns `layout.locator()` when the
layout defines it, and `LayoutLocator(layout)` otherwise. Phase 2b's
`_shapes_of` and `_outline_of` follow the same pattern.

**Cost.** `bin_sizes()` and the edge loop now run once, at construction. Before
this change, `bin_sizes` was computed lazily on first use. Phase 1 measures
construction time to show the cost is negligible.

## Data-backed locators

Each locator's `locate` is the **single** implementation. The layout's
`point_to_bin_index` becomes a one-line call to its cached locator, so no logic
is duplicated. The bodies move as they are, and the error and warning messages
stay the same.

| Locator | Body moved from | Notes |
| --- | --- | --- |
| `RectilinearLocator(edges, shape, active_mask, flat_to_active)` | `layout/mixins.py:272-311`, which calls `_points_to_regular_grid_bin_ind` in `layout/helpers/regular_grid.py` | `flat_to_active` is `_GridMixin.active_mask_inverse_map` (`mixins.py:251`). Used by the regular, masked, image-mask, polygon and polar grids. |
| `HexLocator(offset_x, offset_y, radius, centers_shape, flat_to_active)` | `layout/engines/hexagonal.py:296-346`, which calls `_points_to_hex_bin_ind` | Keep the "not fully initialized" `RuntimeWarning` branch as a layout-level guard. A built locator always has its data. |
| `MeshLocator(points, simplices, simplex_to_active, boundary)` | `layout/engines/triangular_mesh.py:192-263` | `boundary` is the stored `_boundary_polygon_stored`, a shapely `Polygon` or `None`. After the simplex lookup, today's code (`triangular_mesh.py:249-260`) sets any point outside it to -1 using `shapely.contains_xy`; keep that rule unchanged. Without it, a point inside a hole of the polygon maps to a bin. The memo is `scipy.spatial.Delaunay(points)`. Check `np.array_equal(memo.simplices, simplices)` when the memo is built; if they differ, raise `ValueError`. |
| `NearestBinLocator(centers, max_distance=None)` | `layout/mixins.py:134-228` | Keep the dimension-mismatch `ValueError` text and the non-finite-row handling. The memo is `scipy.spatial.cKDTree(centers)`. `max_distance=None` gives today's behavior: no limit. |

Cache the locator on the layout at the end of `build()`, as `self._locator`.
`layout.locator()` returns it. A layout that's mutated after build isn't
supported today (`core.py:932-934`), so the cache is safe.

`TrackLinearization(graph, edge_order, edge_spacing)`: move the bodies of
`GraphLayout.to_linear` (`layout/engines/graph.py:328-352`) and `linear_to_nd`
(`graph.py:353-425`) into its methods unchanged. They read
`_build_params_used["graph_definition" | "edge_order" | "edge_spacing"]` today;
those become fields. The graph is a deep copy passed through `nx.freeze`.
`linear_to_nd` also needs the linear bin edges and active mask that it reads at
`graph.py:353-425`; store them as fields.

**Parity probes**, used by the phase 2a tests:
- the bin centers;
- 2,000 seeded uniform points over the bounding box widened by 20%;
- every grid edge coordinate, for points exactly on a boundary;
- the hexagon vertices;
- the triangle vertices and edge midpoints;
- for the mesh, points inside each hole of the boundary polygon and in each
  concave notch of the outer boundary (the mesh fixture must have both);
- rows containing NaN and inf;
- a single row;
- an empty `(0, n_dims)` array;
- a wrong-width array, which must raise the same error.

## Plotting from binning data

Move each layout `plot` body into a function in
`src/neurospatial/layout/_plotting.py`. The function takes data arguments
instead of `self`, and the layout's `plot` calls it. Output artists must be
identical.

| Function | Moved from | Data it reads |
| --- | --- | --- |
| `plot_grid(ax, *, title, bin_centers, connectivity, dimension_ranges, rectilinear, **kw)` | `layout/mixins.py:312-423` | grid edges, shape, active mask, centers, connectivity |
| `plot_polygon_grid(...)` | `layout/engines/shapely_polygon.py:146` (to the end of the method) | as `plot_grid`, plus `Outline` |
| `plot_hexagons(ax, *, title, shapes: HexagonShapes, dimension_ranges, **kw)` | `layout/engines/hexagonal.py:197-295` | centers, radius, orientation |
| `plot_triangles(ax, *, title, shapes: TriangleShapes, outline, connectivity, dimension_ranges, **kw)` | `layout/engines/triangular_mesh.py:276-434` | triangle points and simplices, boundary outline |
| `plot_track(ax, *, title, linearization, bin_centers, connectivity, grid_edges, **kw)` and `plot_track_linear(...)` | `layout/engines/graph.py:178-282` and `:283-327` | track graph, edge order and spacing, centers |

`BinningLayout.plot` picks the first that applies:
1. `linearization` → `plot_track`
2. `shapes` is `HexagonShapes` → `plot_hexagons`
3. `shapes` is `TriangleShapes` → `plot_triangles`
4. `rectilinear` → `plot_grid`, or `plot_polygon_grid` when `outline` is set
5. otherwise, a scatter of the bin centers, the same as `_SubsetLayout.plot`
   (`environment/transforms.py:565-593`)

**`plot_field` dispatches through components, not tags.** Today it chooses a
renderer by comparing `self.layout._layout_type_tag` with fixed names
(`environment/visualization.py:456-512`). A derived tag such as
`"MaskedGrid_transformed"` or `"subset"` then falls through to the scatter
renderer, even when grid or shape data was kept. Replace the comparison with the
following checks, in order:

1. `n_dims == 1` and not a track → line plot (unchanged)
2. `rectilinear is not None` → `_plot_grid_field`, reading edges, shape and mask
   from `env._binning.rectilinear`
3. `shapes` is `HexagonShapes` → `_plot_hex_field`, reading `env._binning.shapes`
4. `shapes` is `TriangleShapes` → `_plot_trimesh_field`, reading `env._binning.shapes`
5. `linearization is not None` → `_plot_1d_field`
6. otherwise → `_plot_scatter_field`

For every built environment this picks today's renderer:
- Grids, masks, polygons, image masks and polar grids all carry a rectilinear
  grid. Polar layouts are tagged `"MaskedGrid"` today and render with
  `QuadMesh`.
- 3D grids keep today's "pcolormesh requires 2D grids" error inside
  `_plot_grid_field`.

The reads of `env.layout.hex_radius_`, `env.layout._full_delaunay_tri` and
`env.layout.grid_*` in these renderers (`visualization.py:941-1179`) move to the
binning.

**`plot_1d` reads the linearization.** `plot_1d` calls
`self.layout.plot_linear_layout` when the layout has it
(`visualization.py:280-294`), and warns and returns `None` otherwise. Change it
to call `plot_track_linear` with `env._binning.linearization` and the bin data.
Keep the warning for a track without a linearization. `BinningLayout` also
defines `plot_linear_layout`, which delegates to `plot_track_linear` for any
direct caller.

## File format

**Files** (`to_file` and `from_file`, `src/neurospatial/io/files.py:239-480`):
- The schema becomes `"Environment-v2"`, replacing `_SCHEMA_VERSION` at
  `files.py:44`.
- The JSON keeps today's descriptive keys (name, `layout_type`,
  `layout_parameters`, units, frame, regions, graph) and adds `"binning"`: the
  `encode(binning, "binning")` meta.
- The NPZ keeps today's arrays and adds the binning arrays under their prefixed
  keys.
- `from_file` and `from_dict` decode the binning, rebuild the node-link graph,
  and construct the environment as follows:

```python
layout = BinningLayout(
    binning,
    connectivity=graph,
    dimension_ranges=meta["dimension_ranges"],
    layout_type_tag=meta["layout_type"],
    layout_type=meta["layout_category"],
    build_params=layout_parameters,
    is_linearized_track=meta["is_linearized_track"],
)
env = env_cls(meta["name"], layout, meta["layout_type"], layout_parameters,
              regions=regions, restored_from=source)
```

**One authority for duplicated data.** The binning is authoritative. The
facade derives `grid_edges`, `grid_shape` and `active_mask` from it, so these
aren't stored or passed separately:

| Binning has | `grid_edges` | `grid_shape` | `active_mask` |
| --- | --- | --- | --- |
| `rectilinear` | `rectilinear.edges` | `rectilinear.shape` | `rectilinear.active_mask` |
| `linearization` (a track) | the linear bin edges, as a 1-tuple | `(n_linear_bins,)` | the linear active mask |
| neither | `()` | `(n_bins,)` | all True (the fallback at `core.py:1002-1007`) |

The connectivity graph still carries node and edge attributes that the binning
doesn't hold (`source_grid_flat_index`, `original_grid_nd_index`, `vector`,
`edge_id`, `angle_2d`). The facade constructor runs the
[consistency checks](shared-contracts.md#binninglayout-facade) before the
environment publishes anything.

**Faces at save time.** `binning.faces.build()` is called when faces are
available and saved as `KnownFaces`. Otherwise `MissingFaces(reason)` is saved.
The reason text comes from today's message for the same environment, so a
loaded environment refuses with the same explanation.

**NWB** (`src/neurospatial/io/nwb/_environment.py:114` write, `:516` read):
- The environment record stores the same JSON meta as one description
  attribute on a standard one-row DynamicTable. NWB forbids slash-containing
  container names: `array_specs` maps canonical keys to safe column names,
  shapes and dtypes. Each typed array is stored as a ragged vector without
  padding, with its VectorIndex. This standard-container representation was
  verified by an HDF5 round trip and approved by the user.
- The node-link graph is stored like files store it, so nodes keep
  `original_grid_nd_index`. That fixes `boundary_bins` on NWB read-back.
- `TrackLinearization` is encoded, which fixes `to_linear` on NWB read-back.
- The record schema becomes `"2.0"`.

**Older formats.** `Environment-v1` files raise `ValueError`, and so do NWB
records with schema `"1.0"` or `"1.1"`. Today's writer emits `"1.1"`, and the
reader accepts both (`io/nwb/_environment.py:85-86`). The message names the
version and the fix: recreate the environment with a factory and save it again.
There are no users to migrate.

## Derivation algorithms

### Subset

```python
def subset_binning(b: Binning, keep: NDArray[np.bool_]) -> Binning:
    parent_ids = np.flatnonzero(keep)                       # ascending
    child_of = np.full(b.bins.n_bins, -1, np.intp); child_of[parent_ids] = np.arange(parent_ids.size)
    pair_kept = keep[b.bins.pairs[:, 0]] & keep[b.bins.pairs[:, 1]]
    bins = SpatialBins.from_arrays(
        centers=b.bins.centers[parent_ids], sizes=b.bins.sizes[parent_ids],
        measure_dim=b.bins.measure_dim,
        pairs=child_of[b.bins.pairs[pair_kept]], pair_distances=b.bins.pair_distances[pair_kept],
    )
    ...
```

- **Pair order** is the parent order restricted to the kept pairs. The legacy
  connectivity graph for the facade is built by the existing renumbering
  (`environment/transforms.py:596-632`), and its edge order must equal
  `bins.pairs`.
- **Faces:** if the parent has faces, the child gets
  `KnownFaces(FaceGeometry(edges=child pairs, A=A[pair_kept], distance=distance[pair_kept], volumes=sizes[parent_ids]))`.
  The arrays are read-only. Otherwise the child keeps the parent's faces source.
- **Locator:** `RectilinearLocator.subset(keep)` returns a new locator whose
  mask has the dropped cells set to False. Every other locator becomes
  `SubsetLocator(parent_locator, child_of)`. Its `locate` must mask `-1` before
  indexing, because `child_of[-1]` reads the last parent bin:

  ```python
  def locate(self, points):
      parent = self.base.locate(points)
      out = np.full(parent.shape, -1, dtype=np.intp)
      inside = parent >= 0
      out[inside] = self.parent_to_child[parent[inside]]
      return out
  ```
- **Rectilinear:** the mask is updated in the same way. Shapes are selected
  (`HexagonShapes` centers, `TriangleShapes` triangles). The outline is
  dropped. Track geometry, absolute linear coordinates and conversions are
  kept; the linearization active mask is restricted to the selected active
  bins, preserving their original full-grid IDs.
- **The legacy facade fields** use the updated rectilinear grid when there is
  one, and the point-based fallback otherwise (`core.py:1002-1007`).
  Drawing limits use full grid edge bounds or retained shape extents, so a
  single selected cell remains fully visible. Center extents are the fallback
  when no cell drawing component is retained.
- **Tags:** the facade tag is `"MaskedGrid"` when a rectilinear grid is kept and
  `"subset"` otherwise, matching today's tags.

### Transform

The transform is `x ↦ A x + t`. `m` is `measure_dim`, `n` is `n_dims`, and
`T` is the whole map.

| Quantity | Translation | Similarity `A = sQ` | Axis-aligned scaling `A = diag(a)` | Other `A` |
| --- | --- | --- | --- | --- |
| centers | `c + t` | `T(c)` | `T(c)` | `T(c)` |
| `pair_distances` | `‖T(c_u) − T(c_v)‖`, the same as `ops/transforms.py:1491-1498` | same | same | same |
| `sizes` | same | `× s**m` | `× abs(det A)` (needs `m == n`) | `× abs(det A)` when `m == n`; tracks refused |
| faces `A` | same | `× s**(m-1)` | for a face between bins that differ along axis `k`: `× abs(det A) / abs(a_k)`; requires the **parent** to have `rectilinear`, so negative `a_k` still keeps faces | `MissingFaces` |
| faces `distance` | same | `× s` | `× abs(a_k)` | — |
| faces `volumes` | same | `× s**m` | `× abs(det A)` | — |
| rectilinear | edges `+ t_k` | kept only if `A` is diagonal with **all entries positive**: edges `a_k e + t_k`, same order, same mask | kept only if every `a_k > 0`: same rule | dropped |
| locator | `RectilinearLocator` with shifted edges, or `TransformedLocator` | exact when the rectilinear grid is kept, `TransformedLocator` otherwise | same | `TransformedLocator` |
| shapes | shifted | hexagons only for a proper similarity (`det A > 0`): `s × radius`, orientation plus `atan2(Q[1,0], Q[0,0])`; hexagons dropped for reflections; triangles: points mapped | hexagons dropped; triangles: points mapped | hexagons dropped; triangles: points mapped |
| outline | `shapely.affinity.affine_transform` | same | same | same |
| linearization | node positions mapped | node positions mapped; edge spacing `× s` | refused for tracks | refused for tracks |

**Reflections stay off the fast path.** A negative scale factor would make the
grid fast path reverse the edges and flip the mask. That renumbers the bins:
on a 2×2 grid, lookup returns IDs `[2, 3, 0, 1]` where it should return
`[0, 1, 2, 3]`, while the center rows keep their order. Even with remapped IDs,
a reflection reverses each half-open interval, so a point on a boundary changes
owner. So any `a_k < 0` uses `TransformedLocator`, which undoes the transform
and keeps the parent's ownership exactly, and the child gets no rectilinear
component. Linear interpolation then refuses on reflected grids, as it does
today for every transformed environment. Faces are unaffected, because a
reflection is an isometry. Keeping a rectilinear grid in its own frame, so that
reflections and rotations keep the fast path, is roadmap stage E.

**Boundary ownership on the fast path.** Mapped edges are computed as
`a_k * e + t_k`, the same operation that maps a parent point `p` on edge `e`.
So `child.locate(T(p)) == parent.locate(p)` holds exactly, even on edges.
`TransformedLocator` inverts the map, so a point within rounding error of a
mapped edge can change owner. Invariance tests for wrapped locators therefore
use interior points, and the fast path also gets exact-edge tests.

**Why axis-aligned scaling keeps grid faces but not hexagon faces.** Scaling
by `diag(2, 1)` turns a 45° interface, whose center link was perpendicular to
it, into one that is 36.87° away from perpendicular. Hexagon and diagonal
interfaces lose the right angle that the two-point flux needs. Axis-aligned
grid faces keep it. The reason text in `MissingFaces` must say this, naming the
transform class.

**Classification tolerances.** "Translation" requires `A == I` exactly.
"Diagonal" requires every off-diagonal entry to be exactly 0. "Similarity" is
`np.allclose((A/s) @ (A/s).T, np.eye(n), rtol=1e-12, atol=1e-12)` with
`s = abs(det A) ** (1/n)`.
The normalized absolute tolerance permits ordinary rotation roundoff in the
off-diagonal entries without allowing small-magnitude anisotropy. Triangle
vertices transform exactly under every invertible affine map and are retained
for drawing. Regular hexagon shapes are dropped under non-similarities.
Retaining drawing geometry does not restore missing finite-volume faces.

**Regions** are transformed by the existing code (`ops/transforms.py:1622-1661`).
Units are copied; frame keeps the existing `"_transformed"` suffix behavior.
The facade tag is `f"{parent_tag}_transformed"`,
matching today's tag (`ops/transforms.py:1552`).
