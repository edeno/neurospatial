# Stratified Cell Placement Implementation Plan

**Status:** Not started.

**Goal:** Every simulated cell population spreads its tuning parameters by
stratified (jittered) random draws, so a population of each cell type decodes
the variable it encodes everywhere in that variable's space, and replicate
sessions differ.

**Architecture:** One private sampler module gives stratified draws over the
five parameter spaces the cell types use:
- an interval
- the circle
- the torus or unit square (2-D phases, and (distance, direction) pairs)
- the bins of an environment, area-weighted, with geodesic strata
- arc length along the walls

`simulate_session` and the example sessions build every population from it.
`coverage="stratified"` replaces the deterministic `"uniform"` layout as the
default, and `"random"` stays as the independent-draw baseline.

The plan uses three kinds of stratum:
- **Bins:** geodesic Voronoi cells of a farthest-point seeding. This reuses
  `_spread_bins`, so walls separate strata as they do today.
- **Torus:** a randomly shifted Kronecker (R2) lattice. This is a
  low-discrepancy set for any `n`, unlike Sobol, which wants powers of two
  and warns otherwise; the test config turns that warning into an error.
- **Circle and interval:** one draw inside each of `n` equal strata.

Acceptance is a decoding test per cell type. It works on the true tuning, so
it measures coverage, not spike estimation: draw Poisson counts from the
models' true rates at each state, decode with `neurospatial.decoding.decode_position`,
and require the worst state's median error to stay below a bound that
`coverage="random"` misses.

**Tech stack:** NumPy, SciPy (`scipy.sparse.csgraph.dijkstra` with
`min_only=True`), shapely (through `simulation/_walls.py`), and the package's
own decoder.

**Out of scope:** see "Deliberately not in this plan".

## Decisions (settled with the maintainer, 2026-11-03)

1. `coverage="stratified"` is the default for every session builder. The
   deterministic `"uniform"` layout is **removed** (no alias), and
   `"random"` stays. An int `rng` still reproduces a session exactly.
2. `simulate_session(cell_type=...)` gains `"head_direction"`,
   `"boundary_vector"`, `"object_vector"` and `"spatial_view"`, alongside
   `"place"`, `"border"`, `"grid"` and `"mixed"`.
3. Grid populations default to several modules. Passing `grid_spacing=`
   gives a single module.
4. No literature citations for the module count or spacing ratio. The papers
   could not be read locally, and the maintainer's rule is to cite only what
   was read. The ratio is a modelling choice, calibrated by the
   grid-decoding test (Task 5), and the docstring says so.

Choices this plan makes. Each one is stated in its docstring:
- **Session headings** are the movement direction
  (`behavior.heading_from_velocity`), held over samples below the speed gate,
  so head-direction and spatial-view cells keep firing at rest. Real head
  direction departs from movement direction; the docstring says the
  simulator does not model that.
- **Range of boundary-vector preferred distances:** `[0, 2 * max wall
  depth]`, where max wall depth is the largest bin-to-wall distance from
  `_bin_depths`. In a square of side `L` this gives `[0, L]`, the range a
  BVC population needs to tile the arena.
- **Range of object-vector preferred distances:** `[0, the largest
  object-to-bin distance]`.

## Inputs to read first

- [src/neurospatial/simulation/session.py:534-602](../../../../src/neurospatial/simulation/session.py) — `_field_centers`: the `coverage` switch to replace. The track branch (cumulative bin length) is reused.
- [src/neurospatial/simulation/session.py:418-531](../../../../src/neurospatial/simulation/session.py) — `UNIFORM_WALL_INSET`, `_bin_depths`, `_interior_bins`, `_spread_bins`: reused as the stratum seeds and the wall inset.
- [src/neurospatial/simulation/session.py:362-411](../../../../src/neurospatial/simulation/session.py) — `BORDER_MAX_PREFERRED_DISTANCE_CM` and `_border_cells`: independent uniform draws to stratify.
- [src/neurospatial/simulation/session.py:344-359](../../../../src/neurospatial/simulation/session.py) — `_MODEL_PARAMETERS` / `_RATE_PARAMETERS`: keyword routing to each cell type; extend it for the new types.
- [src/neurospatial/simulation/session.py:605-1060](../../../../src/neurospatial/simulation/session.py) — `simulate_session` / `_simulate_session`: cell counts, the place/grid shared layout and its `mixed` permutation (around :1020-1050), and the streams (`_stream_seeds(rng, 3)`, `field_rng`).
- [src/neurospatial/simulation/session.py:55-95](../../../../src/neurospatial/simulation/session.py) — `SimulationSession` fields and `_rebuild_session` (pickle state); `headings` is added here.
- [src/neurospatial/simulation/examples.py:568-766](../../../../src/neurospatial/simulation/examples.py) — `border_cell_session` (`_border_cells` at :739, `_field_centers(..., "uniform", ...)` at :742).
- [src/neurospatial/simulation/examples.py:769-950](../../../../src/neurospatial/simulation/examples.py) — `grid_cell_session`: bin-stride phases (:921-926), uniform orientations (:927), metadata `"coverage": "uniform"` (:947).
- [src/neurospatial/simulation/examples.py](../../../../src/neurospatial/simulation/examples.py) — `coverage="uniform"` hard-coded at :174 (`open_field_session`), :342 (`linear_track_session`) and :550 (`tmaze_alternation_session`).
- [src/neurospatial/simulation/models/base.py:304-331](../../../../src/neurospatial/simulation/models/base.py) — `_draw_rng` / `_keep_draw`: how a single model consumes `rng`.
- [src/neurospatial/simulation/models/place_cells.py:264-271](../../../../src/neurospatial/simulation/models/place_cells.py) — single-model center draw: `integers(0, env.n_bins)` is uniform over bins, not over space.
- [src/neurospatial/simulation/models/grid_cells.py:172-229](../../../../src/neurospatial/simulation/models/grid_cells.py) — `grid_spacing`, `grid_orientation`, `phase_offset`; the lattice points along `grid_orientation` and `grid_orientation + pi/3`, with period `grid_spacing`.
- [src/neurospatial/simulation/models/boundary_cells.py:414-449](../../../../src/neurospatial/simulation/models/boundary_cells.py) — `BoundaryVectorCellModel` fields and its direction draw.
- [src/neurospatial/simulation/models/head_direction_cells.py:202-213](../../../../src/neurospatial/simulation/models/head_direction_cells.py) — `HeadDirectionCellModel` fields and its direction draw.
- [src/neurospatial/simulation/models/object_vector_cells.py:279-291](../../../../src/neurospatial/simulation/models/object_vector_cells.py) — `ObjectVectorCellModel` fields (`object_positions` required, `preferred_direction=None` means omnidirectional).
- [src/neurospatial/simulation/models/spatial_view_cells.py:263-273](../../../../src/neurospatial/simulation/models/spatial_view_cells.py) — `SpatialViewCellModel` fields (`preferred_view_location` required, `gaze_model="ray_cast"`).
- [src/neurospatial/simulation/_walls.py:30-136](../../../../src/neurospatial/simulation/_walls.py) — `Walls.enclosing(env)`: the outline and internal wall segments, for arc-length strata.
- [src/neurospatial/simulation/spikes.py:245-255](../../../../src/neurospatial/simulation/spikes.py) — `generate_population_spikes(models, times, positions, *, headings=None, ...)`.
- [src/neurospatial/behavior/trajectory.py:466-476](../../../../src/neurospatial/behavior/trajectory.py) — `heading_from_velocity(times, positions, *, min_speed=..., ...)`.
- `.claude/rules/public-api.md` — the `rng` rule (an int and the Generator it seeds give the same result) and the result contract (`SimulationSession` gains a field).

## Tasks

### 1. Stratified samplers — new `src/neurospatial/simulation/_stratified.py`

These are private functions. Each takes a `np.random.Generator` and returns
draws in a random order (a final `rng.permutation`), so unit index does not
sort by tuning.

```python
_PLASTIC = 1.324717957244746  # real root of x**3 = x + 1; R2 lattice constant
_R2_STEP = np.array([1.0 / _PLASTIC, 1.0 / _PLASTIC**2])


def stratified_interval(n: int, low: float, high: float, rng) -> NDArray:
    """One uniform draw in each of n equal strata of [low, high), shuffled."""
    u = (np.arange(n) + rng.uniform(size=n)) / n
    return rng.permutation(low + (high - low) * u)


def stratified_circle(n: int, rng) -> NDArray:
    """n angles in [-pi, pi), one per equal arc, shuffled (radians)."""
    return stratified_interval(n, -np.pi, np.pi, rng)


def stratified_square(n: int, rng) -> NDArray:
    """n points of [0, 1)^2: an R2 lattice shifted uniformly at random (mod 1).

    Low-discrepancy on the torus for any n, and each point is marginally
    uniform, so it serves 2-D phases and (distance, direction) pairs alike.
    """
    points = (rng.uniform(size=2) + np.arange(1, n + 1)[:, None] * _R2_STEP) % 1.0
    return points[rng.permutation(n)]
```

`stratified_bins(env, n, rng, *, candidates) -> NDArray[np.intp]`:
- **Track or 1-D environment:** reuse the cumulative-length coordinate of
  `_field_centers`'s track branch (`along`). Use the targets
  `stratified_interval(n, 0, along_total, rng)` and take the bin whose
  length interval contains each target.
- **Otherwise:**
  1. Seeds: `_spread_bins(env, candidates, n, depths)`, which is
     deterministic.
  2. Strata: the geodesic Voronoi cells of the seeds over `candidates`.
     Compute them as `dijkstra(_bin_graph(env), indices=seeds, min_only=True,
     return_predecessors=True)`, which returns `sources`. Each candidate
     belongs to its nearest seed.
  3. One bin per stratum, drawn with probability proportional to
     `env.bin_sizes`, so the draw is uniform over area.

  Candidates are `_interior_bins` when `n` is at most their count, which
  keeps today's wall inset; otherwise all bins. Strata may share no bin, so
  the result is `n` distinct bins whenever `n <= env.n_bins`, and more than
  that raises as today.

`stratified_wall_points(env, n, rng) -> NDArray, shape (n, 2)`:
1. Get the segments from `Walls.enclosing(env)`.
2. Draw the targets `stratified_interval(n, 0, total_length, rng)` along the
   concatenated segments.
3. Interpolate each target to its point.

None from `Walls.enclosing` (a track) raises `ValueError` through
`_format_error`: spatial-view cells need 2-D walls.

### 2. Coverage over bins — `_field_centers`, single place cells

- `_field_centers(env, n, coverage, rng)` takes `coverage: Literal["stratified", "random"]`:
  - `"stratified"` → `env.bin_centers[stratified_bins(...)]`.
  - `"random"` → `rng.choice(env.n_bins, n, p=env.bin_sizes / env.bin_sizes.sum())`,
    which is uniform over space, with replacement.
- Delete the deterministic farthest-point layout as a coverage mode. Keep
  `_spread_bins`, which now only seeds the strata.
- Delete the track evenly-spaced branch as a layout; it lives on as the
  stratified track draw.
- `PlaceCellModel` with `center=None`: replace `integers(0, env.n_bins)` with
  `choice(env.n_bins, p=bin_sizes/sum)`. Update its docstring ("uniform over
  the environment's area").
- Update `validate_choice(coverage, ...)` and every `coverage` docstring in
  `session.py` and `examples.py`:
  - `open_field_session`, `linear_track_session`,
    `tmaze_alternation_session`: stop passing `"uniform"` and use the
    default.
  - The docstring passages that describe the old layout: `examples.py:147`,
    `:300`, `:690`.

### 3. Session headings

- Add `headings: NDArray[np.float64] | None`, shape (n_time,), radians,
  allocentric, to `SimulationSession`. Include it in `_rebuild_session`'s
  state and in the pickle round trip.
- `_simulate_session` computes it when any model `requires_headings`:
  1. `heading_from_velocity(times, positions, min_speed=<2 cm/s via
     default_length>)`.
  2. Forward-fill NaN, using the first valid heading at the start.
  3. Pass it to `generate_population_spikes(headings=...)`.
- Otherwise `headings` is None.

Document the choices in the class and `simulate_session` docstrings:
- movement direction
- held at rest
- the 2 cm/s gate as a modelling choice

### 4. New cell types in `simulate_session`

Add to the `cell_type` Literal, `validate_choice`, the docstring and
`_MODEL_PARAMETERS`:

| `cell_type` | Model | Stratified parameters (unless passed) | Routed keywords |
| --- | --- | --- | --- |
| `"head_direction"` | `HeadDirectionCellModel` | `preferred_direction` = `stratified_circle` | `concentration` |
| `"boundary_vector"` | `BoundaryVectorCellModel` | `(preferred_distance, preferred_direction)` = `stratified_square` mapped to `[0, 2 * max(_bin_depths)] × [-pi, pi)` | `distance_tolerance`, `direction_tolerance`, `boundary_source` |
| `"object_vector"` | `ObjectVectorCellModel` | `(preferred_distance, preferred_direction)` = `stratified_square` mapped to `[0, max object-to-bin distance] × [-pi, pi)` | `object_positions` (**required**; refuse with What/Why/Fix if missing), `distance_tolerance`, `direction_concentration`, `object_selectivity` |
| `"spatial_view"` | `SpatialViewCellModel` | `preferred_view_location` = `stratified_wall_points` | `view_field_width`, `gaze_model`, `view_distance`, `fov` |

- 2-D-only types (`grid`, `boundary_vector`, `object_vector`,
  `spatial_view`) keep the existing "requires 2D environment" refusal,
  generalized to name the type.
- `"mixed"` keeps 60/20/20 place/border/grid, with each type now drawn from
  its own stratified sampler. Delete the shared place/grid layout and its
  permutation (around :1032-1050).

### 5. Grid modules — `_grid_cells(env, n_cells, rng, params)` in `session.py`

- With `grid_spacing` in `params`: one module of that spacing.
- Otherwise: `GRID_N_MODULES` modules with spacing
  `default_length(env, GRID_SPACING_CM) * GRID_MODULE_RATIO ** m`.
  - Start at `GRID_N_MODULES = 4` and `GRID_MODULE_RATIO = np.sqrt(2)`.
  - Calibrate with the grid-decoding test: the smallest ratio and fewest
    modules that pass it in the 100 cm and 150 cm arenas.
  - Write the measured result in the constants' comments.
- Split cells across modules with `np.array_split`.
- Each module gets one orientation, `rng.uniform(0, pi/3)`, unless
  `grid_orientation` is passed.
- Each cell's phase is
  `origin + u * a1 + v * a2`, with:
  - `(u, v)` from `stratified_square(n_module, rng)`
  - `a1 = s * (cos o, sin o)` and `a2 = s * (cos(o + pi/3), sin(o + pi/3))`
  - `origin` the mean of `env.bin_centers`
- `grid_cell_session`:
  - Replace its bin-stride phases and per-cell orientations with
    `_grid_cells`.
  - Change `grid_spacing: float = 41.0` to `grid_spacing: float | None =
    None`. None means modules, a value means one module.
  - Record `coverage` truthfully in the metadata.

### 6. Border cells

In `_border_cells`, replace the independent draws with one
`stratified_square(n, rng)` sample, mapped to
`[0, BORDER_MAX_PREFERRED_DISTANCE_CM in env units] × [-pi, pi)`. Passing
`preferred_distance` or `preferred_direction` still fixes that coordinate.
`border_cell_session` uses `coverage="stratified"` for its place cells.

### 7. Population decoding tests — new `tests/simulation/test_population_decoding.py`

Write one helper, `decoding_error(tuning, states_env, dt, rng)`:
1. `tuning` is `(n_cells, n_states)` true rates in Hz.
2. For each state, draw `K` Poisson count vectors with mean `tuning[:, s] * dt`.
3. Decode with `decode_position(states_env, counts, tuning, dt=dt)`.
4. Return the per-state median error: Euclidean in cm, or circular in rad for
   headings.

Compute each type's tuning from the models' public `firing_rate`:

| Type | States | Tuning |
| --- | --- | --- |
| place, grid, boundary-vector, object-vector | `env.bin_centers` of a 100 cm square, 4 cm bins | `firing_rate(times, env.bin_centers)` |
| head-direction | 72 headings, on a 1-D environment over `[-pi, pi)` | position fixed, `headings` = the 72 states |
| spatial-view | points along the walls | animal at the arena center, heading toward each wall point, so the ray-cast gaze lands on it; states are arc-length positions on a 1-D environment |
| border | near-wall bins only (depth below 15 cm) | decode error checked on those bins only |

For each type, compare `coverage="stratified"` with `"random"` over 10
seeds, using the same cell count:
- the worst state's median error under stratified stays below a stated bound
  (calibrate once, then record it)
- stratified's worst-state error is at most random's on every seed

The grid test also checks that one module alone cannot decode the arena
(error above half the spacing) while the default modules can. This is the
calibration for Task 5.

If `decode_position` refuses a 1-D angle environment, report it and stop.
Do not add a decoder path in this plan.

### 8. Re-pin tests that encoded the deterministic layout

Measured pins found by the survey:
- `tests/simulation/test_session.py`:
  - :419 `test_simulate_session_coverage_uniform`
  - :453-594: L-shape, internal wall, plus-maze, radial arms
  - :522 "deterministic" (asserts seeds 0 and 1 give identical layouts): **delete**, since the layout is now random by design
  - :572 "grow one at a time": **delete**, since stratified layouts do not nest
  - :605 mixed coverage, :636 random coverage, :226 and :279 border
- `tests/simulation/test_examples.py`: :150 track coverage, :381 spacing, :605 border-session place centers (≈20/80 cm), :914 grid phase/orientation stream.
- `tests/simulation/test_rng.py:194-205` `_expected_draws`: place now draws `choice(..., p=)`.
- Seeded `open_field_session(rng=0)` downstream pins:
  - `tests/encoding/test_spatial_cell_type_significance.py:191`
  - `tests/encoding/test_label_cell_types.py:206`
  - `tests/test_result_contracts.py:31` (pinned p-values)
  - `tests/conftest.py:243` `open_field_120s` and its users: `encoding/test_discarded_data_warnings.py`, `decoding/test_decode_input_guards.py`, `environment/test_occupancy.py`
- Also: `tests/simulation/test_validation_sim.py`, `test_session_duration.py`, `test_lap_realism.py`, `test_integration.py`, `tests/test_error_guidance.py`, plus doctests in `session.py` and `examples.py`.

How to re-pin:
- **Structural assertions** (coverage of every arm, the wall inset, counts):
  keep them, and assert them for several seeds.
- **Literal pins** (p-values, centers): recompute, and say in the commit
  message which outputs moved and why.
- **A downstream test that only needed some field in range:** prefer a
  structural assertion over a new literal.

### 9. Docs

- **CHANGELOG** "Unreleased" gets one section:
  - Breaking: `coverage="uniform"` removed; `grid_cell_session(grid_spacing=None)`; seeded session outputs changed.
  - Added: four cell types, `SimulationSession.headings`, grid modules.
- `docs/user-guide/simulated-mazes.md` and `docs/examples/05_simulation_and_validation.py` / `.ipynb`: every `coverage=` mention and the explanation of the layout. Re-run the executed pages (`.claude/rules/executable-docs.md`), and keep the `.py` and `.ipynb` copies in sync.
- `tests/data/public_api.txt`: regenerate (signature changes to `simulate_session` and `grid_cell_session`) and review the diff.

## Deliberately not in this plan

- **Head direction separate from movement direction** (head scanning,
  noise). It belongs to the trajectory simulator; revisit when a test or
  tutorial needs head direction that departs from movement.
- **Egocentric object-vector and egocentric boundary-vector populations.**
  The table uses the allocentric frame, the model default; egocentric needs
  heading-dependent decoding.
- **Placement proportional to the animal's occupancy or to reward sites.**
  That couples field placement to the trajectory and is a separate modelling
  question.
- **A decoder for circular variables.** The head-direction test uses
  `decode_position` on a 1-D angle environment. If that environment is
  refused, stop and report; don't build one here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_stratified_interval_one_per_stratum` | for n in {1, 7, 50}: sorted draws fall one in each `[k/n, (k+1)/n)`; an int seed reproduces the draws; different seeds differ |
| `test_stratified_square_low_discrepancy` | n = 50: every cell of a 5×5 grid on [0,1)² holds 1–3 points (random draws fail this on most seeds; checked over 20 seeds) |
| `test_stratified_bins_area_weighted` | on bins of unequal size, the empirical draw frequency per stratum is proportional to bin area (chi-square over 2000 draws, p > 0.01) |
| `test_stratified_bins_respect_walls` | an L-shaped arena and a plus maze, several seeds: every arm gets centers; no stratum's bins straddle a declared internal wall (geodesic Voronoi) |
| `test_stratified_wall_points_on_walls` | every point lies on a wall segment (distance < 1e-9); along the total length, one point per stratum |
| `test_place_center_uniform_over_area` | the single-model `center=None` draw over unequal bins matches area weights (chi-square) |
| `test_session_headings_follow_movement` | `headings` equal `heading_from_velocity` where moving, and hold the last value at rest; a pickle round trip keeps them |
| `test_new_cell_types_build_sessions` | each new `cell_type` returns `n_cells` models of the right class with stratified parameters; `object_vector` without `object_positions` raises with What/Why/Fix; 2-D-only types refuse a track |
| `test_grid_modules_layout` | the default makes `GRID_N_MODULES` spacings at the stated ratio; phases tile each module's rhombus (the reduced (u, v) pass the 5×5 test); `grid_spacing=` gives one module |
| `test_population_decodes_its_variable[type]` (10 seeds; mark `slow` if > 10 s total) | the worst-state median decoding error under stratified is below the recorded bound, and at most random's |
| `test_one_grid_module_aliases` | one module's error > spacing / 2; the default modules' error is below the bound |
| `test_int_and_generator_rng_agree` | `simulate_session(rng=0)` equals `simulate_session(rng=default_rng(0))` for every cell type |
| the re-pinned suites of Task 8 | pass, with structural assertions over several seeds |

## Fixtures

All simulated; no real data. Module-scoped fixtures in
`tests/simulation/conftest.py`:
- a 100 cm square, 4 cm bins, `units="cm"`
- an L-shaped arena with a declared internal wall
- a plus maze (`simulation.mazes`)
- a 1-D angle environment

Keep each decoding test under a few seconds by decoding true tuning, not
simulated spike trains.

## Review

Before opening the PR, dispatch `code-reviewer` (or another independent
reviewer) against the diff, giving it the diff and not the author's
conclusions. For the samplers and the decoding test, also run the
`scientific-code-change-audit` skill. Confirm:
- Every task is implemented as specified, and nothing in "Deliberately not
  in this plan" was added.
- The validation-slice tests pass, and slow tests are marked.
- Tests aren't trivial. The decoding bounds were calibrated once and
  recorded, not tuned per seed. The stratified-vs-random comparison is the
  real check.
- Docstrings, test names and module names don't reference this plan.
- The full default suite, the slow tier and the doctests pass. Do not merge
  on green tests alone; the review is required.
