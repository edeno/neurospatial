# Shared contracts

[← back to PLAN.md](PLAN.md)

These are the types and rules that more than one phase depends on. Each
contract lives only here. Phases link to the anchors below. Do not weaken an
invariant without updating this file and every phase that links to it.

- [Identity and ownership rules](#identity-and-ownership-rules)
- [SpatialBins](#spatialbins)
- [FacesSource](#facessource)
- [Locator](#locator)
- [Components](#components)
- [Binning](#binning)
- [Publication on the environment](#publication-on-the-environment)
- [BinningLayout facade](#binninglayout-facade)
- [Codec contract](#codec-contract)
- [Derivation rules](#derivation-rules)

---

## Identity and ownership rules

These rules apply to every type in this file.

1. **Identity semantics.** Every dataclass uses `@dataclass(frozen=True, eq=False)`.
   Equality and hashing are by identity. Never use the generated `__eq__` or
   `__hash__`: they compare or hash NumPy arrays, and hashing raises
   `TypeError`.
2. **Owned, read-only arrays.** A constructor copies each input array with
   `np.array(x, dtype=..., copy=True)` and then calls `setflags(write=False)`.
   Changing the caller's array after construction must not change the object.
3. **No hidden mutable state, except bounded memos.** A part may hold a private
   memo, such as a KD-tree or a Delaunay object, only if the memo can be
   rebuilt from the stored arrays. Memos are never saved.
4. **No global caches.** Analysis caches (kernels, eigenbases, operators) stay
   in the environment's existing holders: `versioned_cached_property`,
   `_kernel_cache` and `_gaussian_kernels`. A cache key includes every input
   that changes the result: bandwidth, mode, and coordinate system or metric
   where relevant.
5. **Dimensions are explicit.** `n_dims` is the number of coordinate columns.
   `measure_dim` is the dimension of a bin's measure. They differ for tracks:
   `n_dims == 2` and `measure_dim == 1`.

## SpatialBins

Module: `src/neurospatial/layout/_binning.py`.

```python
@dataclass(frozen=True, eq=False)
class SpatialBins:
    """Discrete bins: centers, sizes and neighbor pairs.

    Attributes
    ----------
    centers : NDArray[np.float64], shape (n_bins, n_dims)
        Bin centers in the environment's coordinates.
    sizes : NDArray[np.float64], shape (n_bins,)
        Bin measure in ``units ** measure_dim``: lengths for tracks, areas
        or volumes otherwise. Finite and strictly positive.
    measure_dim : int
        Dimension of ``sizes``. 1 for track bins; ``n_dims`` for grid,
        hexagonal, mesh and polar bins.
    pairs : NDArray[np.intp], shape (n_pairs, 2)
        Neighbor pairs in ``connectivity`` edge order and orientation.
    pair_distances : NDArray[np.float64], shape (n_pairs,)
        Navigation distance of each pair: the connectivity ``"distance"``.
        Finite and non-negative.
    """

    centers: NDArray[np.float64]
    sizes: NDArray[np.float64]
    measure_dim: int
    pairs: NDArray[np.intp]
    pair_distances: NDArray[np.float64]

    @classmethod
    def from_arrays(cls, centers, sizes, measure_dim, pairs, pair_distances) -> SpatialBins: ...

    @property
    def n_bins(self) -> int: ...

    @property
    def n_dims(self) -> int: ...
```

Invariants, checked by `from_arrays`, which raises `ValueError` with the
offending index:

- `centers` is 2D and finite.
- `sizes.shape == (n_bins,)`, and every size is finite and greater than 0.
- `1 <= measure_dim <= n_dims`.
- `pairs` has shape `(n_pairs, 2)`. Every index is in `[0, n_bins)`. There are
  no self pairs, and each unordered pair appears at most once.
- `pair_distances.shape == (n_pairs,)`, and every distance is finite and at
  least 0. This matches `validate_connectivity_graph`
  (`src/neurospatial/layout/validation.py:389-410`).

**Do not sort `pairs`.** Gradient and divergence index edge fields by
`env.connectivity` edge order and orientation. `FaceGeometry.from_graph`
(`src/neurospatial/layout/_diffusion.py:28`) depends on the same order.

**Two distances per pair exist and both are kept:**
- `pair_distances` is the navigation distance, used for paths and graph
  distances.
- `FaceGeometry.distance` is the flux distance, used for diffusion. On track
  junctions the flux distance is the along-track length and the navigation
  distance is the chord (`src/neurospatial/layout/engines/graph.py:457-494`).

## FacesSource

Module: `src/neurospatial/layout/_binning.py`. Face data reuses
`FaceGeometry` (`src/neurospatial/layout/_diffusion.py:28`) unchanged.

```python
class FacesSource(Protocol):
    """Where an environment's finite-volume faces come from."""

    @property
    def available(self) -> bool: ...

    def build(self) -> FaceGeometry:
        """Validated faces in pair order. Called once per environment by the
        environment's cached ``_face_geometry``; not memoized here."""

    def missing_error(self, operation: str) -> Exception:
        """Actionable error naming ``operation``, for when faces are unavailable."""
```

| Implementation | Introduced | Behavior |
| --- | --- | --- |
| `LayoutFaces(layout, connectivity)` | phase 1 | `available` is `isinstance(layout, FiniteVolumeGeometryProvider)`. `build()` runs today's checks from `src/neurospatial/environment/fields.py:798-812`. `missing_error` returns today's `_missing_geometry_error(layout, operation)` (`fields.py:57-89`). |
| `KnownFaces(faces: FaceGeometry)` | phase 3 | `available` is True; `build()` returns `faces`. |
| `MissingFaces(reason: str)` | phase 3 | `available` is False. `missing_error(op)` formats `f"{op} needs finite-volume cell geometry. {reason}"` with What / Why / Fix through `_format_error`. |

## Locator

Module: `src/neurospatial/layout/_binning.py` (protocol and `LayoutLocator`);
`src/neurospatial/layout/_locators.py` (the data-backed kinds).

```python
class Locator(Protocol):
    """Find the bin that contains each point."""

    def locate(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Bin index for each row of ``points``, or -1.

        ``points`` has shape (n_points, n_dims). Returns shape (n_points,),
        dtype np.intp, values in [-1, n_bins). Rows with a non-finite value
        return -1. A wrong number of columns raises ValueError.
        """
```

Every locator is deterministic and doesn't modify its input. A locator stores
everything it needs as arrays plus scalars, except a rebuildable memo (see the
[identity and ownership rules](#identity-and-ownership-rules)).

| Kind | Introduced | Data | Replaces |
| --- | --- | --- | --- |
| `LayoutLocator(layout)` | phase 1 | delegates to `layout.point_to_bin_index(points).astype(np.intp, copy=False)` | today's direct calls (`environment/cartesian.py:131`, `:177`) |
| `RectilinearLocator` | phase 2a | edges for each axis, grid shape, active mask, flat-to-active map | `_GridMixin.point_to_bin_index` (`layout/mixins.py:272`) |
| `HexLocator` | phase 2a | grid offsets x and y, hexagon radius, centers shape, flat-to-active map | `HexagonalLayout.point_to_bin_index` (`layout/engines/hexagonal.py:296`) |
| `MeshLocator` | phase 2a | Delaunay points, simplices, simplex-to-active map, and the boundary polygon (or `None`), applied with today's containment rule; Delaunay rebuilt as a memo | `TriangularMeshLayout.point_to_bin_index` (`layout/engines/triangular_mesh.py:192`) |
| `NearestBinLocator` | phase 2a | centers, optional `max_distance`; KD-tree is a memo | `_KDTreeMixin.point_to_bin_index` (`layout/mixins.py:134`) |
| `SubsetLocator(base, parent_to_child)` | phase 4 | base locator; `(n_parent_bins,)` map, -1 for dropped bins. `locate` masks the base's -1 results **before** indexing the map ([designs.md#subset](designs.md#subset)) | the subset KD-tree in `environment/transforms.py:493-531` |
| `TransformedLocator(base, inverse)` | phase 4 | base locator; owned inverse homogeneous matrix `(n_dims+1, n_dims+1)` | `TransformedLayout.point_to_bin_index` (`ops/transforms.py:1566-1574`) |

A locator may define `subset(keep) -> Locator` or `transformed(t) -> Locator`
for an exact, faster version, such as a `RectilinearLocator` with a new mask or
shifted edges. Otherwise derivation wraps it.

## Components

Optional structure. A missing component is `None`. Code that needs a component
checks for it and raises an actionable error. It never checks a layout's class
or `_layout_type_tag`.

| Component | Module | Introduced | Holds | Used by |
| --- | --- | --- | --- | --- |
| `RectilinearGrid` | `layout/_capabilities.py:20` (exists) | phase 1 | edges, shape, active mask | linear interpolation, linear time allocation, grid subsets |
| `TrackLinearization` | `layout/_binning.py` | phase 1 (layout-backed), phase 2a (data) | frozen copy of the track graph, edge order, edge spacing | `to_linear`, `linear_to_nd` |
| `HexagonShapes` | `layout/_components.py` | phase 2b | centers, radius, orientation | hexagonal plots |
| `TriangleShapes` | `layout/_components.py` | phase 2b | vertex points, active triangles `(n_bins, 3)` | mesh plots |
| `Outline` | `layout/_components.py` | phase 2b | a shapely `Polygon` or `MultiPolygon` | polygon and mesh outlines in plots |

In phase 1, `TrackLinearization` can be the layout itself, which satisfies
`TrackLinearizationProvider` (`layout/_capabilities.py:50`). Phase 2a replaces
it with the data class.

## Binning

Module: `src/neurospatial/layout/_binning.py`.

```python
@dataclass(frozen=True, eq=False)
class Binning:
    """Everything an environment computes from: bins, faces, point lookup, components."""

    kind: str                                   # layout type tag, used in messages only
    bins: SpatialBins
    locator: Locator
    faces: FacesSource
    rectilinear: RectilinearGrid | None = None
    linearization: TrackLinearizationProvider | None = None   # the layout in phase 1; TrackLinearization from phase 2a
    shapes: HexagonShapes | TriangleShapes | None = None   # phase 2b
    outline: Outline | None = None                          # phase 2b


def binning_from_layout(layout: LayoutEngine, connectivity: nx.Graph) -> Binning:
    """Build the binning of a built layout and its validated connectivity."""
```

`kind` is descriptive. No code may branch on it. Construction steps are in
[designs.md#binning-from-a-layout](designs.md#binning-from-a-layout).

## Publication on the environment

- `_PublishedGeometry` (`src/neurospatial/environment/core.py:71-85`) gains a
  `binning: Binning | None` field. The `__init__` placeholder (`core.py:439-446`)
  uses `None`. `_setup_from_layout` (`core.py:921-1041`) sets it in the same
  statement that replaces the geometry (`core.py:1018-1025`), after graph
  validation.
- `_BaseEnvironment._binning` is a read-only property. It returns
  `self._geometry.binning` and asserts it isn't `None`. Every reader is already
  behind `@check_fitted`.
- After publication, environment code reads capabilities only from the binning:
  `rectilinear`, `linearization`, `faces.available`, `shapes`. No
  `isinstance(self.layout, ...Provider)` checks remain in
  `src/neurospatial/environment/`.

## BinningLayout facade

Module: `src/neurospatial/layout/_binning_layout.py`. Introduced in phase 3.

This is a `LayoutEngine`-compatible view of a `Binning`. It lets the about 80
remaining `env.layout` readers (visualization, animation, encoding, metrics,
NWB) work for environments that were loaded or derived instead of built. It's a
bridge, removed at roadmap stage F (see
[overview.md#roadmap-after-phase-4](overview.md#roadmap-after-phase-4)).

```python
class BinningLayout:
    def __init__(
        self,
        binning: Binning,
        *,
        connectivity: nx.Graph,                 # with the required node and edge attributes
        dimension_ranges: Sequence[tuple[float, float]] | None,
        layout_type_tag: str,                   # e.g. "RegularGrid", "subset", "Hexagonal_transformed"
        layout_type: str,                       # the category: "grid", "mask", "hexagonal", ...
        build_params: dict[str, Any],           # descriptive record; never re-run
        is_linearized_track: bool,
        coordinate_kind: Literal["cartesian", "polar"] = "cartesian",
    ) -> None: ...

    def binning(self) -> Binning: ...           # binning_from_layout returns this directly
    def build(self, **kwargs: Any) -> None: ... # no-op
    def point_to_bin_index(self, points): ...   # self._binning.locator.locate(points)
    def bin_sizes(self): ...                    # self._binning.bins.sizes
    def plot(self, ax=None, **kwargs): ...      # chosen by component; see designs.md#plotting-from-binning-data
    def plot_linear_layout(self, ax=None, **kwargs): ...  # tracks only: plot_track_linear
    # grid_edges, grid_shape, active_mask: read-only properties derived from the
    # binning (designs.md#file-format, "One authority for duplicated data")
```

**Consistency checks.** The constructor raises `ValueError`, naming the first
mismatch, unless all of these hold. They run before `_setup_from_layout`
publishes anything. `validate_connectivity_graph` checks only structure, and
would accept changed positions or distances.

1. The graph has nodes `0..n_bins-1`, and node `i`'s `"pos"` equals
   `bins.centers[i]` exactly.
2. `list(connectivity.edges())` equals `bins.pairs` in order and orientation.
3. Each edge's `"distance"` equals `bins.pair_distances` exactly.
4. When `faces` is `KnownFaces`: `faces.edges == bins.pairs` and
   `faces.volumes == bins.sizes`.
5. Component sizes agree with the bins:
   - `rectilinear.active_mask.sum() == n_bins`;
   - `HexagonShapes.centers == bins.centers`;
   - `TriangleShapes.triangles.shape[0] == n_bins`;
   - the active bins of the linearization number `n_bins`.
6. A data locator and its optional grid, shape and track components agree
   exactly on shared geometry, active masks and ordering. There is one authority
   for grid edges and masks; changing a component without its locator is invalid.
   Bin centers and sizes must agree with that geometry.
   Polar grid measures include the radial bin center; the facade receives the
   saved coordinate kind to distinguish these from Cartesian cell measures.
7. `locator.locate(bins.centers) == arange(n_bins)`. Every bin center must
   locate to its own bin. This is one lookup of `n_bins` points.

Derivations (phase 4) build the graph's `"pos"` and `"distance"` from the
derived bins, so these checks hold by construction.

The facade defines **none** of the capability methods: `finite_volume_geometry`,
`rectilinear_grid`, `to_linear`. Environment code reads capabilities from the
binning (see [publication](#publication-on-the-environment)), so a facade can't
falsely claim a capability through `isinstance`.

## Codec contract

Module: `src/neurospatial/layout/_codecs.py`. Codecs are pure: no file I/O.
Introduced in phase 3; phase 4 adds the wrapper codecs.

```python
Payload = tuple[dict[str, Any], dict[str, NDArray[Any]]]   # (JSON-safe meta, arrays)

def encode(obj: object, prefix: str) -> Payload: ...
def decode(meta: dict[str, Any], arrays: Mapping[str, NDArray[Any]], prefix: str) -> object: ...
```

- Every meta dict has `"codec"` (the kind name, such as `"RectilinearLocator"`)
  and `"codec_version"` (an int, starting at 1).
- Array keys are `f"{prefix}/{field}"`. Nested parts extend the prefix: for
  example `binning/locator/base/edges_0` for a `SubsetLocator`'s base.
- Built-in kinds are registered in a module-level `dict[str, Codec]` filled at
  import. A kind that isn't registered raises
  `ValueError("cannot save <kind>: no codec is registered for it ...")` with a fix
  line. Third-party registration (through entry points) is roadmap stage B.
- **Decoding never runs a layout builder.** It may rebuild memos: KD-trees, and
  Delaunay objects from stored points. For Delaunay it checks that the rebuilt
  simplices equal the stored ones, and raises `ValueError` if they differ.
- The encoding is lossless: `decode(encode(x))` gives arrays equal to `x`'s
  under `np.array_equal`, and identical metadata.

## Derivation rules

Introduced in phase 4. `subset_binning(binning, keep)` and
`transform_binning(binning, transform)` live in
`src/neurospatial/layout/_derive.py`. The algorithms and formulas are in
[designs.md#derivation-algorithms](designs.md#derivation-algorithms).

Guaranteed-preservation classes:

| Transform class | Test on the linear part `A` (n×n) | Faces kept? | Rectilinear kept? |
| --- | --- | --- | --- |
| translation | `A == I` exactly | yes, unchanged | yes, edges shifted |
| similarity | `(A/s) @ (A/s).T == I` (rtol and atol `1e-12`), with `s > 0` | yes, scaled | only if `A` is diagonal with every entry positive |
| axis-aligned scaling | `A` diagonal, unequal entries | only when the **parent** has `rectilinear` (every face is axis-aligned) | only if every entry is positive; edges scaled |
| any other invertible `A` | — | no: `MissingFaces` naming the transform | no |

**Reflections (any negative diagonal entry, or `det A < 0`)** never take the
rectilinear fast path. Reversing edges would renumber the bins and move
boundary ownership. The child uses `TransformedLocator`, has no rectilinear
component, and keeps faces by the rules above.

Track binnings (`measure_dim < n_dims`) accept translation and similarity only.
Any other class raises `ValueError`, because the new bin lengths would depend on
each bin's direction.

Triangle drawing vertices are retained under every invertible affine map.
This is independent of finite-volume face validity: a sheared or unequally
scaled mesh still refuses smoothing when its faces lose the required right
angles. Regular hexagon shapes remain limited to proper similarities.

The normalized Gram check allows floating-point off-diagonal roundoff in a
rotation. A zero absolute tolerance would reject ordinary rotations despite
their unit diagonal. Normalization keeps the bound relative to scale, so a tiny
anisotropic transform does not pass because of its absolute magnitude.

`subset_binning` always keeps faces among the kept bins when the parent has
them. Clipping bins to a region is **not** a subset; it's roadmap stage A.
