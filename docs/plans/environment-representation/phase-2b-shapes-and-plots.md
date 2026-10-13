# Phase 2b — Bin shapes, outlines and plotting from binning data

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#plotting-from-binning-data)

Hexagon and triangle shapes, and polygon and mesh outlines, become binning
components. Every layout plot becomes a function of data. This is what lets a
loaded environment (phase 3) or a derived one (phase 4) plot correctly without
its original layout object. Outputs don't change.

**Inputs to read first:**

- [src/neurospatial/layout/mixins.py:312-423](../../../src/neurospatial/layout/mixins.py#L312-L423) — the grid plot.
- [src/neurospatial/layout/engines/shapely_polygon.py:146](../../../src/neurospatial/layout/engines/shapely_polygon.py#L146) — the polygon grid plot, which also draws `polygon_definition_` (stored at line 103).
- [src/neurospatial/layout/engines/hexagonal.py:197-295](../../../src/neurospatial/layout/engines/hexagonal.py#L197-L295) — the hexagon plot, which reads `hex_radius_` and `hex_orientation_`.
- [src/neurospatial/layout/engines/triangular_mesh.py:276-434](../../../src/neurospatial/layout/engines/triangular_mesh.py#L276-L434) — the mesh plot, which reads `_full_delaunay_tri`, `_active_original_simplex_indices` and `_boundary_polygon_stored`.
- [src/neurospatial/layout/engines/graph.py:178-327](../../../src/neurospatial/layout/engines/graph.py#L178-L327) — the track plot and `plot_linear_layout`.
- [src/neurospatial/environment/visualization.py:456-512](../../../src/neurospatial/environment/visualization.py#L456-L512) — `plot_field`'s renderer dispatch, which compares the layout tag with fixed names and reads private layout attributes.
- [src/neurospatial/environment/visualization.py:941-1179](../../../src/neurospatial/environment/visualization.py#L941-L1179) — the grid, hexagon, mesh, 1D and scatter renderers.
- [src/neurospatial/environment/visualization.py:280-294](../../../src/neurospatial/environment/visualization.py#L280-L294) — `plot_1d`, which calls `layout.plot_linear_layout` when it exists.

**Contracts referenced:**

- [Components](shared-contracts.md#components) — `HexagonShapes`, `TriangleShapes`, `Outline`.
- [Binning](shared-contracts.md#binning) — fill in `shapes` and `outline`.

**Designs referenced:** [designs.md#plotting-from-binning-data](designs.md#plotting-from-binning-data).

## Tasks

- **Record the baseline**, as in phase 1, on the base commit. Also record the
  output of `env.plot()` and `env.plot_field(field)` for the ten environment
  types:
  - **Artists** (committed): for every collection and patch on the axes, its
    class name, path vertices, face colors and z-order, saved to
    `tests/environment/data/plot_artists.npz`.
  - **PNGs** (not committed): rendered at `dpi=72` for a manual same-machine
    pixel comparison after the change.
- **Create `src/neurospatial/layout/_components.py`** with `HexagonShapes`,
  `TriangleShapes` and `Outline`. They're frozen `eq=False` dataclasses with
  read-only arrays. `Outline` holds a shapely geometry, which is immutable.
- **Give the layouts methods that return their components:**
  - `HexagonalLayout.bin_shapes() -> HexagonShapes`;
  - `TriangularMeshLayout.bin_shapes() -> TriangleShapes`, covering active
    triangles only, in bin order;
  - `ShapelyPolygonLayout.outline()` and `TriangularMeshLayout.outline()`.

  `binning_from_layout`'s `_shapes_of` and `_outline_of` call these when they
  exist.
- **Create `src/neurospatial/layout/_plotting.py`** and move the six plot bodies
  listed in [designs.md](designs.md#plotting-from-binning-data) into its
  functions unchanged. Each layout's `plot` and `plot_linear_layout` becomes a
  single call to its function. The matplotlib import stays lazy, inside the
  functions.
- **Dispatch `plot_field` through components.** Replace the tag comparison at
  `visualization.py:456-512` with the
  [component dispatch](designs.md#plotting-from-binning-data).
  - The renderers (`visualization.py:941-1179`) read grid data from
    `env._binning.rectilinear` and shapes from `env._binning.shapes`.
  - The "missing attributes" `RuntimeError`s become checks for missing
    components, worded in scientific terms, keeping the error type.
  - Remove every read of `hex_radius_`, `hex_orientation_`,
    `_full_delaunay_tri`, `_active_original_simplex_indices` and `layout.grid_*`
    from `environment/visualization.py`'s `plot_field` path.
- **Route `plot_1d` through the linearization.** At `visualization.py:280-294`,
  call `plot_track_linear` with `env._binning.linearization`. Keep today's
  warning when a track has no linearization.
- **Tests:** new `tests/layout/test_components.py` and
  `tests/environment/test_plot_from_binning.py`.
- **Compare the baseline** (zero differences). On the same machine, compare
  the PNGs byte for byte and report the result in the PR.

## Deliberately not in this phase

- **`BinningLayout`, which uses these plot functions,** is phase 3.
- **Shapes for grid bins.** Grids plot from `RectilinearGrid`, and per-bin
  boxes would cost memory on large 3D grids.
- **Shapes for track bins** come later, with the arena (roadmap stage A).
- **Changing what any plot draws.** The artists must be identical.
- **Other `env.layout` readers in animation** (`animation/rendering.py`,
  `animation/backends/napari_backend.py`) are roadmap stage F.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_hexagon_shapes_match_layout` | `shapes.radius == layout.hex_radius_`; `shapes.orientation == layout.hex_orientation_`; `shapes.centers == env.bin_centers` |
| `test_triangle_shapes_cover_active_bins` | `shapes.triangles.shape == (n_bins, 3)`; triangle centroids equal `env.bin_centers` |
| `test_outline_matches_polygon` | For `from_polygon`, the `Outline` geometry `.equals()` the input polygon |
| `test_plot_artists_unchanged` (parametrized over the ten types; `@pytest.mark.slow`) | Class names, path vertices, face colors and z-order of every artist from `env.plot()` and `plot_field()` equal the committed base-commit record |
| `test_plot_field_renderer_by_component` (parametrized over the ten types) | The renderer's artist class is unchanged from today: `QuadMesh` for the grid, polygon, mask, pixel-mask and polar types; `PatchCollection` for the hexagonal type; `PolyCollection` for the mesh; `FillBetweenPolyCollection` plus `Line2D` for the track. This holds even when `_layout_type_tag` is monkeypatched to an unknown string, which proves the dispatch ignores tags |
| `test_plot_1d_uses_linearization` | `plot_1d()` on the track returns axes with the same artists as before, and emits no warning |
| `test_plot_field_reads_binning_shapes` | `plot_field` on a hexagonal environment draws `n_bins` hexagon patches with the radius from `env._binning.shapes` |
| baseline compare (manual) | zero differences |
| full suite, doctests, `tests/docs`, API snapshot, ruff, mypy | all pass |

## Fixtures

- Reuse `built_environment`.
- Commit `tests/environment/data/plot_artists.npz`, recorded on the base
  commit by a one-off script (describe the command in the PR). Keep the figure
  sizes and environments small.
- Render under the Agg backend.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or an equivalent
independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored: no scope creep into
  adjacent phases.
- The validation slice tests pass, and slow or integration tests are marked.
- The tests aren't trivial. They exercise the asserted behavior, not
  tautologies: no `assert True`, and no assertions that only check the mock the
  test just configured. Shared setup is in fixtures, not copied between tests.
  (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names and module names don't reference this plan or its
  phases.
- Old code paths flagged for removal in this phase are actually removed: the
  private-attribute reads in `environment/visualization.py`, and the plot bodies
  in the layouts.
- No user-facing documentation changes are needed beyond error-message text.
