# Phase 5 — Core primitives coverage

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add behavioral tests for `map_points_to_bins` (currently 105 lines of mostly imports for a 783-LOC function), dedicated test files for the 5 layout engines that have none, `ops/transforms.py` quick-helpers (zero references in test tree), polar-egocentric angular seam, and heading edge cases at the ±π wrap.

**Inputs to read first:**

- [src/neurospatial/ops/binning.py:90-99](../../../src/neurospatial/ops/binning.py) — `map_points_to_bins(points, env, *, tie_break=TieBreakStrategy.LOWEST_INDEX, return_dist=False, max_distance=None, max_distance_factor=None)`. `tie_break` accepts `TieBreakStrategy` enum or the literal strings `"lowest_index"` / `"closest_center"` (lowercase). Read the full function to understand the `return_dist`, `max_distance`, `max_distance_factor` branches and the `_estimate_typical_bin_spacing` heuristic at lines 56-87. The "10 × typical bin spacing" default is at lines 314-320.
- [src/neurospatial/layout/engines/](../../../src/neurospatial/layout/engines/) — `masked_grid.py`, `image_mask.py`, `shapely_polygon.py`, `hexagonal.py`, `graph.py`. Each has a `point_to_bin_index(point) -> int` (returns -1 for out-of-env points), `bin_centers` array, and connectivity. The protocol is in `src/neurospatial/layout/_layout_engine.py` or similar — read first.
- [src/neurospatial/ops/transforms.py:1051+](../../../src/neurospatial/ops/transforms.py) — `flip_y_data(data)`, `convert_to_cm(data, units)`, `convert_to_pixels(data_cm, scale_factor)`. Quick helpers. Zero test references currently.
- [tests/environment/test_polar_egocentric.py:97-222](../../../tests/environment/test_polar_egocentric.py) — existing tests cover circular connectivity; missing the angular-seam distance-field test.
- [src/neurospatial/ops/egocentric.py:617-700](../../../src/neurospatial/ops/egocentric.py) — `heading_from_velocity(positions, dt, *, min_speed, ...)`. Read the westward-π convention comment around line 617-618 and the low-speed interpolation around 692-696.

## Tasks

### 1. Behavioral tests for `map_points_to_bins`

In [tests/ops/test_binning.py](../../../tests/ops/test_binning.py), keep the existing 105 lines (the import-callable tests are low-value but harmless), then add `TestMapPointsToBinsBehavior` with the following:

- `test_returns_dist_finite_for_inside_points`: `return_dist=True`. Points inside the env get finite distances `< env.bin_size`. Pin one point to its expected distance (`np.linalg.norm(point - bin_center)` for the chosen bin) at `atol=1e-10`.
- `test_returns_dist_inf_for_outside_points`: a point far outside the env returns `inf` distance when `return_dist=True`.
- `test_max_distance_marks_outside`: a point at distance `1.5 * env.bin_size` from the nearest center. With `max_distance=env.bin_size`, the point is marked as outside (bin index = -1). With `max_distance=2*env.bin_size`, it's marked inside.
- `test_max_distance_factor_relative_to_bin_spacing`: `max_distance_factor=0.5` should be equivalent to `max_distance = 0.5 * typical_bin_spacing`. Verify by computing both ways.
- `test_max_distance_and_factor_together_raises`: `pytest.raises(ValueError, match="Cannot specify both")`.
- `test_max_distance_negative_raises`: `max_distance=-1.0`, `pytest.raises(ValueError, match="positive|non-negative")`.
- `test_default_threshold_is_10x_typical_spacing`: monkey-patch `_estimate_typical_bin_spacing` to return a known value `K`, run on a point at `9.5 * K` and `10.5 * K`. Assert the 9.5 case is inside, 10.5 case is outside. Pins the "10x" heuristic so silent changes are caught.
- `test_closest_center_vs_lowest_index_on_tie`: construct a point exactly equidistant from two bin centers. `tie_break="closest_center"` should return either bin (document the deterministic choice — see docstring subtask 1a below); `tie_break="lowest_index"` must return the lower bin index. Parametrize over both the enum form (`TieBreakStrategy.LOWEST_INDEX`) and the literal-string form to confirm both call paths work.

#### 1a. Documentation subtask — `tie_break` semantics

Read the `tie_break` docstring at [src/neurospatial/ops/binning.py:90-99](../../../src/neurospatial/ops/binning.py). If the docstring does not explicitly state which bin `tie_break="closest_center"` returns on a true tie (it likely says "the closest" without specifying tie-break order, since both are equally close), add a one-sentence clarification to the parameter docstring stating the implementation's actual choice (e.g., "On exact ties between equidistant centers, returns the first bin encountered by the underlying KDTree query — order is implementation-defined and may differ between platforms."). The clarification ships with the test in this phase, not as a follow-up.
- `test_nan_in_points_marked_outside`: input `points = [[np.nan, np.nan], [50, 50]]`. NaN row should return -1 (outside). Pin to current behavior — read source first to verify this isn't a raise.
- `test_returns_correct_bin_on_regular_grid`: regular grid `bin_size=2.0`, point `(7.3, 11.7)`. Expected bin: `(round(7.3/2), round(11.7/2)) = (4, 6)`. Compute the expected flat bin index from `env.bin_at`-equivalent math and assert.

### 2. Dedicated test files for 5 layout engines

Create one file per engine, each ~150 LOC. All tests should drive the engine directly (not through `Environment`), so that engine-specific edge cases are exercised independently.

**[tests/layout/test_masked_grid_layout.py](../../../tests/layout/test_masked_grid_layout.py)** (new):
- `test_point_inside_bbox_but_masked_returns_negative_one`: construct a `MaskedGridLayout` where a region inside the bounding box is masked out. Query a point in the masked region. Assert `point_to_bin_index(point) == -1`.
- `test_point_in_active_region`: query a point in an active region; returns the corresponding bin index.
- `test_point_on_mask_boundary`: query a point exactly on the boundary between active and masked. Pin to current behavior (read source); document.
- `test_n_bins_excludes_masked`: assert `layout.n_bins` equals the count of `True` in the mask, not the total grid size.

**[tests/layout/test_image_mask_layout.py](../../../tests/layout/test_image_mask_layout.py)** (new):
- `test_loads_from_synthetic_mask`: synthesize a 16×16 boolean mask in `conftest.py`, build an `ImageMaskLayout` from it, assert `n_bins == mask.sum()`.
- `test_point_outside_image_bounds_returns_negative_one`.
- `test_point_in_active_pixel_returns_valid_bin`.
- `test_pixel_size_scaling`: changing `pixel_size` scales `bin_centers` linearly.

**[tests/layout/test_shapely_polygon_layout.py](../../../tests/layout/test_shapely_polygon_layout.py)** (new):
- `test_point_inside_polygon`: simple convex polygon, point in interior → valid bin index.
- `test_point_outside_polygon_returns_negative_one`.
- `test_point_in_polygon_hole_returns_negative_one`: polygon with interior hole (use `shapely.Polygon(exterior, [hole])`); point in hole returns -1.
- `test_point_on_polygon_edge`: pin to current behavior — Shapely's `contains` typically returns False for boundary points. Document.
- `test_irregular_polygon_active_bins_inside`: L-shape or U-shape polygon; assert no active bins fall outside the polygon's convex hull where it shouldn't.

Gate all tests in this file behind `pytest.importorskip("shapely")`.

**[tests/layout/test_hexagonal_layout.py](../../../tests/layout/test_hexagonal_layout.py)** (new):
- `test_neighbors_have_six_for_interior_cells`: a hex grid with at least 3×3 hexes; pick an interior cell; assert `layout.connectivity.degree[interior_cell] == 6`.
- `test_neighbors_have_fewer_at_boundary`: edge cells have fewer than 6 neighbors.
- `test_point_to_bin_round_trip`: for each `bin_center`, `point_to_bin_index(bin_center) == bin_index`.
- `test_hexagonal_spacing`: assert distance between adjacent bin centers matches the hex spacing parameter.

**[tests/layout/test_graph_layout.py](../../../tests/layout/test_graph_layout.py)** (new):
- `test_1d_track_linearization`: build a `GraphLayout` from a simple linear-track graph (Y-maze or similar). Assert `n_bins == n_nodes` and linearization is monotonic along each segment.
- `test_point_to_bin_on_node`: query a point at a node position → returns that node's bin.
- `test_point_off_graph_returns_negative_one`: query a point far from any edge.
- `test_is_linearized_track_true`: `layout.is_linearized_track` should return True.

### 3. Tests for `ops/transforms.py` quick helpers

Create [tests/ops/test_transforms_helpers.py](../../../tests/ops/test_transforms_helpers.py) (new file, ~80 LOC).

#### 3a. Read source and enumerate supported units

Before writing tests, read [src/neurospatial/ops/transforms.py:1051+](../../../src/neurospatial/ops/transforms.py) and produce a definitive list (paste into the PR description) of:
- `flip_y_data`'s exact axis convention (which column or axis is negated; whether input is `(n, 2)`, `(n_time, n, 2)`, or both).
- `convert_to_cm`'s supported `units` argument values and the scale factor for each (likely some subset of `{"m", "mm", "cm", "inch", "in", "ft"}` — verify before assuming).
- `convert_to_pixels`'s parameter name for the scale factor and the direction of conversion (`cm → pixels` or `pixels → cm`).

The test cases below assume conventions; the executor must confirm and adjust assertions if the source differs.

#### 3b. Test cases

- `test_flip_y_data_inverts_y`: input `[[1, 2], [3, 4]]`; expected output `[[1, -2], [3, -4]]` (adjust per subtask 3a finding).
- `test_flip_y_data_preserves_x`.
- `test_convert_to_cm_known_unit`: parametrize over each unit listed in subtask 3a. For each `(value, unit, expected_cm)` triple: input `1.0 unit`, assert output equals `expected_cm` at `atol=1e-10`. E.g. `(1.0, "m", 100.0)`, `(1.0, "inch", 2.54)`.
- `test_convert_to_cm_unknown_unit_raises`: `pytest.raises(ValueError)` for an undocumented unit string.
- `test_convert_to_pixels_round_trip`: pick a value in cm, convert to pixels with `scale=K`, convert back, assert exact recovery (`atol=1e-10`). The expected round-trip identity depends on the conversion direction — confirm via subtask 3a.

#### 3c. Documentation subtask

If subtask 3a reveals that the supported `units` list is not explicitly enumerated in the `convert_to_cm` docstring, add the enumeration as a `units` parameter sub-line. This ships with the test PR.

### 4. Polar-egocentric angular seam distance field

In [tests/environment/test_polar_egocentric.py](../../../tests/environment/test_polar_egocentric.py), add `TestPolarEgocentricAngularSeam`:

- `test_distance_field_wraps_across_seam`: build `env = Environment.from_polar_egocentric(distance_range=(0, 50), n_distance_bins=10, n_direction_bins=12, circular_angle=True)`. Choose a source bin near `angle ≈ +π` (the seam). Compute `compute_distance_field(env.connectivity, sources=[source_bin])`. Find the bin at `angle ≈ -π`. Assert its distance is **smaller** than the distance to a bin at `angle ≈ 0`. This pins the load-bearing correctness property of `circular_angle=True`.
- `test_distance_field_non_circular_does_not_wrap`: same setup with `circular_angle=False`. The `-π` and `+π` bins should now be far apart in the graph (each at the end of a separate angular spine), so the distance from source at `+π` to a bin at `-π` is larger than to a bin at `angle=0`.

### 5. Pin westward-π convention for `heading_from_velocity`

In [tests/ops/test_reference_frames.py](../../../tests/ops/test_reference_frames.py) or [tests/ops/test_calibration.py](../../../tests/ops/test_calibration.py), add:

- `test_heading_from_velocity_westward_returns_pi_not_negative_pi`: positions traversing `x` from 100 down to 0 in equal steps (purely westward), constant `y`. Call `heading_from_velocity(positions, dt=0.1, min_speed=5.0)`. Assert `np.allclose(headings, np.pi, atol=1e-6)`. Per the source docstring at [egocentric.py:617-618](../../../src/neurospatial/ops/egocentric.py), this is the documented behavior — but currently no test guards it.
- `test_heading_at_pi_boundary_with_brief_stop`: animal moves westward, stops briefly (zero velocity for 5 timesteps), then continues westward. Heading should remain at `+π` across the stop (interpolated), not flip to `-π`. This is analogous to the existing body-orientation test that the audit cited but is missing for the velocity path.

### 6. Pin `allocentric_to_egocentric` batch path sign convention

In [tests/ops/test_reference_frames.py](../../../tests/ops/test_reference_frames.py), strengthen the batch-path tests (audit cited line 118-120 having only an absolute-value spot check).

Add `TestAllocentricToEgocentricSignConvention` with a parametrized test over `{heading=0, π/2, π, -π/2} × {East target (1,0), North target (0,1), West target (-1,0), South target (0,-1)}`. For each combination, compute the expected egocentric `(x, y)` by hand (one-line trig per case) and assert the batch function returns those exact values within `atol=1e-10`. 16 assertions total.

This pins the rotation matrix sign convention end-to-end and would catch any coordinated sign-flip that round-trip tests miss.

## Deliberately not in this phase

- **No fix for the audit-cited subset KDTree cache implementation-coupling.** `tests/environment/test_subset_kdtree_cache.py` uses `unittest.mock.patch` on `cKDTree`; the audit recommended replacing this with a behavioral test. Out of scope here; revisit if it causes a real CI failure.
- **No removal of `MockEnvironment` in `tests/ops/test_ops_alignment.py:21-28`.** Phase 8 handles mock removal.
- **No new viewshed/visibility behavior tests** beyond what already exists. The audit cited a NaN-contract gap at `test_visibility.py:348-373` — captured by Phase 7's threshold-tightening sweep.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/ops/test_binning.py::TestMapPointsToBinsBehavior` (10 tests) | `return_dist`, `max_distance`, `max_distance_factor`, NaN handling, default-threshold behavior, tie-breaking all pinned. |
| `tests/layout/test_masked_grid_layout.py` (4 tests) | Masked points return -1; n_bins counts active only. |
| `tests/layout/test_image_mask_layout.py` (4 tests) | Loads from mask, pixel_size scales, bounds enforced. |
| `tests/layout/test_shapely_polygon_layout.py` (5 tests) | Hole, edge, exterior, irregular polygon all handled. **`importorskip("shapely")`.** |
| `tests/layout/test_hexagonal_layout.py` (4 tests) | Hex degree=6 interior, fewer at boundary; round-trip. |
| `tests/layout/test_graph_layout.py` (4 tests) | 1D linearization monotonic; node round-trip; off-graph returns -1. |
| `tests/ops/test_transforms_helpers.py` (5+ tests) | flip_y, convert_to_cm with known units, convert_to_pixels round-trip. |
| `tests/environment/test_polar_egocentric.py::TestPolarEgocentricAngularSeam` (2 tests) | `circular_angle=True` connects the seam; `=False` does not. |
| `tests/ops/test_*::test_heading_from_velocity_westward_*` (2 tests) | Westward returns +π; brief stops don't flip the sign. |
| `tests/ops/test_reference_frames.py::TestAllocentricToEgocentricSignConvention` (parametrized 16×) | Sign convention pinned across all cardinal-heading × cardinal-target combinations. |

## Fixtures

In `tests/layout/conftest.py`:
- `synthetic_16x16_mask`: deterministic 16×16 boolean mask (e.g., a centered circle of radius 6) for `ImageMaskLayout` tests.
- `l_shaped_polygon`: Shapely L-shape polygon for `ShapelyPolygonLayout` tests.
- `polygon_with_hole`: Shapely polygon with an interior hole.
- `simple_y_maze_graph`: networkx graph for `GraphLayout` tests.

In `tests/ops/conftest.py`:
- `westward_trajectory`: positions for the heading-from-velocity test (decreasing x, constant y).
- `regular_grid_env_2x2`: a small regular-grid env for `map_points_to_bins` tests with known bin centers.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 7 (threshold sweep) or Phase 8 (mock removal).
- Validation slice tests pass; slow / integration tests are marked (none expected here; layout tests should be fast).
- Tests aren't trivial — every layout-engine test drives the engine directly and asserts on a concrete bin index or distance, not just "is not None". (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (none).
- User-facing documentation listed as tasks is updated, not deferred (none in this phase — public API behavior is unchanged).
