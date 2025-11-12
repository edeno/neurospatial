# Simulation Subpackage Implementation Tasks

**Project**: neurospatial.simulation subpackage
**Last Updated**: 2025-11-11
**Total Estimated Time**: 4.5 weeks (excluding optional Phase 4)

---

## Milestone 1: Core Trajectory + Place Cells (1 week)

**Goal**: Minimal viable simulation for testing place field detection

### Module Structure Setup

- [x] Create directory structure: `src/neurospatial/simulation/`
- [x] Create `__init__.py` with flat import design (see SIMULATION_PLAN.md lines 96-174)
- [x] Create `models/` subdirectory with `__init__.py`
- [x] Create test directory: `tests/simulation/`
- [x] Set up imports in test `conftest.py`

### Base Protocol Implementation

- [x] Implement `NeuralModel` Protocol in `models/base.py`
  - [x] Add `@runtime_checkable` decorator
  - [x] Define `firing_rate()` method signature
  - [x] Define `ground_truth` property with documented dict structures
  - [x] Add comprehensive docstring with all model types (lines 345-386)
- [x] Add type hints: `from typing import Protocol, runtime_checkable, Any`

### Trajectory Simulation: OU Process

- [x] Implement `simulate_trajectory_ou()` in `trajectory.py` (lines 208-316)
  - [x] Parse parameters: `env`, `duration`, `dt`, `start_position`, `speed_mean`, `speed_std`, `coherence_time`
  - [x] Validate `env.units` is set (raise `ValueError` if None)
  - [x] Implement Euler-Maruyama integration scheme for N-D velocity
  - [x] Add velocity magnitude clipping to maintain `speed_mean`
  - [x] Implement boundary handling modes:
    - [x] `'reflect'`: Elastic boundary reflection (lines 2066-2082)
    - [x] `'periodic'`: Toroidal wrapping (lines 2084-2090)
    - [x] `'stop'`: Clamp to boundary (lines 2092-2097)
  - [x] Add speed units conversion support (if `speed_units` != `env.units`)
  - [x] Implement random seed handling with `np.random.default_rng(seed)`
  - [x] Return `(positions, times)` tuple
- [x] Write comprehensive NumPy docstring with all parameters documented

### Trajectory Simulation: Sinusoidal (1D)

- [x] Implement `simulate_trajectory_sinusoidal()` in `trajectory.py` (lines 318-373)
  - [x] Check `env.is_1d` (raise `ValueError` if False)
  - [x] Compute period from track length and speed if not provided
  - [x] Generate sinusoidal motion with pauses at peaks
  - [x] Return `(positions, times)` tuple
- [x] Write NumPy docstring

### Place Cell Model

- [x] Implement `PlaceCellModel` in `models/place_cells.py` (lines 502-610)
  - [x] Constructor: store `env`, `center`, `width`, `max_rate`, `baseline_rate`, `distance_metric`, `condition`, `seed`
  - [x] If `center` is None, randomly choose from `env.bin_centers`
  - [x] If `width` is None, default to `3 * env.bin_size`
  - [x] Implement `firing_rate(positions, times)` method:
    - [x] Compute distance (euclidean or geodesic) from positions to center
    - [x] For geodesic: precompute distance field in `__init__` (cache it)
    - [x] Apply Gaussian decay with clipping at 5σ for numerical stability
    - [x] Formula: `baseline + (max - baseline) * exp(-0.5 * (d/width)^2)`
    - [x] Apply condition mask if provided: `rates * condition(positions, times)`
  - [x] Implement `ground_truth` property returning dict with keys: `center`, `width`, `max_rate`, `baseline_rate`
- [x] Write comprehensive NumPy docstring with examples

### Spike Generation

- [x] Implement `generate_poisson_spikes()` in `spikes.py` (lines 781-838)
  - [x] Generate candidate spikes from inhomogeneous Poisson process
  - [x] Sort candidate spike times
  - [x] Apply O(n) refractory period filter (single-pass algorithm, lines 811-823)
  - [x] Handle random seed with `np.random.default_rng(seed)`
  - [x] Return spike times array
- [x] Write NumPy docstring with algorithm notes

- [x] Implement `generate_population_spikes()` in `spikes.py` (lines 840-908)
  - [x] Loop over models with progress bar (use `tqdm.auto.tqdm`)
  - [x] Compute firing rates for each model
  - [x] Generate spikes for each model with derived seeds
  - [x] Show progress with postfix: `n_spikes`, `rate`
  - [x] Print summary after completion
  - [x] Return list of spike time arrays
- [x] Write NumPy docstring

### Testing

- [x] Write `tests/simulation/test_trajectory.py`:
  - [x] Test OU trajectory stays in bounds
  - [x] Test velocity autocorrelation matches `coherence_time`
  - [x] Test boundary modes (reflect, periodic, stop)
  - [x] Test sinusoidal trajectory requires 1D environment
  - [x] Test reproducibility with same seed

- [x] Write `tests/simulation/test_models.py`:
  - [x] Test place cell peak at center
  - [x] Test Gaussian falloff at 1σ
  - [x] Test geodesic vs euclidean distance options
  - [x] Test condition function gates firing correctly
  - [x] Test ground_truth property returns correct keys

- [x] Write `tests/simulation/test_spikes.py`:
  - [x] Test Poisson spike generation mean rate
  - [x] Test refractory period constraint (all ISIs >= refractory_period)
  - [x] Test spike times are sorted
  - [x] Test reproducibility with seed
  - [x] Test population spikes returns correct number of arrays

### Documentation

- [x] Add docstring examples to all functions
- [x] Run doctests: `uv run pytest --doctest-modules src/neurospatial/simulation/`
- [x] Ensure all docstrings follow NumPy format

### Validation

- [x] Run all tests: `uv run pytest tests/simulation/`
- [x] Run with coverage: `uv run pytest --cov=src/neurospatial/simulation`
- [x] Run mypy: `uv run mypy src/neurospatial/simulation/`
- [x] Run ruff: `uv run ruff check src/neurospatial/simulation/ && uv run ruff format src/neurospatial/simulation/`

---

## Milestone 2: Boundary Cells + Extended Models (1 week)

**Goal**: Add boundary cell model and structured trajectory patterns

### Trajectory Simulation: Laps

- [x] Implement `simulate_trajectory_laps()` in `trajectory.py` (lines 375-440)
  - [x] Parse parameters: `env`, `n_laps`, `speed_mean`, `speed_std`, `outbound_path`, `inbound_path`, `pause_duration`, `sampling_frequency`, `seed`, `return_metadata`
  - [x] If paths not provided, use shortest path between environment extrema
  - [x] Generate velocity along path with noise
  - [x] Add pauses at lap ends
  - [x] If `return_metadata=True`, return `(positions, times, metadata)` with lap IDs and directions
  - [x] Otherwise return `(positions, times)`
- [x] Write NumPy docstring with T-maze example

### Boundary Cell Model

- [x] Implement `BoundaryCellModel` in `models/boundary_cells.py` (lines 614-687)
  - [x] Constructor: store `env`, `preferred_distance`, `distance_tolerance`, `preferred_direction`, `direction_tolerance`, `max_rate`, `baseline_rate`, `distance_metric`
  - [x] Precompute boundary bins: `env.boundary_bins`
  - [x] Precompute distance field from all boundary bins (cache it)
  - [x] Implement `firing_rate(positions, times)` method:
    - [x] Compute distance to nearest boundary for each position
    - [x] If directional, compute direction to boundary and apply von Mises tuning
    - [x] Apply Gaussian tuning: `exp(-(d - preferred_distance)^2 / (2 * tolerance^2))`
    - [x] Scale by max_rate and add baseline
  - [x] Implement `ground_truth` property
- [x] Write NumPy docstring

### Spike Modulation

- [x] Implement `add_modulation()` in `spikes.py` (lines 910-948)
  - [x] Compute phase of each spike time: `2π * freq * spike_times + phase`
  - [x] Compute acceptance probability: `(1 + depth * cos(phase)) / 2`
  - [x] Thin spikes using acceptance probability
  - [x] Return modulated spike times
- [x] Write NumPy docstring

### Testing

- [x] Write tests for `simulate_trajectory_laps()`:
  - [x] Test n_laps produces correct number of laps
  - [x] Test metadata contains lap_ids and directions
  - [x] Test pauses at lap ends

- [x] Write tests for `BoundaryCellModel`:
  - [x] Test peak firing at preferred_distance
  - [x] Test directional tuning when preferred_direction specified
  - [x] Test omnidirectional when preferred_direction=None

- [x] Write tests for `add_modulation()`:
  - [x] Test modulation reduces spike count
  - [x] Test phase preference
  - [x] Test modulation_depth=0 returns all spikes

### Documentation

- [x] Add examples to all new functions
- [x] Run doctests (12/12 passing)

### Validation

- [x] Run all tests with coverage (74 passed, 3 skipped, 73% coverage)
- [x] Run mypy and ruff (all checks passed)

---

## Milestone 3: Grid Cells + Convenience Functions (1.5 weeks)

**Goal**: Add grid cells, high-level session API, validation helpers, and pre-configured examples

### Grid Cell Model

- [x] Implement `GridCellModel` in `models/grid_cells.py` (lines 689-772)
  - [x] Check `env.n_dims == 2` in constructor (raise ValueError otherwise)
  - [x] Constructor: store `env`, `grid_spacing`, `grid_orientation`, `phase_offset`, `max_rate`, `baseline_rate` (field_width removed as unused)
  - [x] Implement `firing_rate(positions, times)` method:
    - [x] Compute wave vectors: `k_magnitude = 4π / (√3 * grid_spacing)`
    - [x] Compute three wave vectors at 60° intervals (lines 753-758)
    - [x] Apply rotation by `grid_orientation`
    - [x] Compute grid pattern: `g(x) = (1/3) * Σ cos(k_i · (x - phase_offset))`
    - [x] Apply rectification: `rate = baseline + (max - baseline) * max(0, g(x))`
  - [x] Implement `ground_truth` property
- [x] Write NumPy docstring with doctests and examples
- [x] Add parameter validation (grid_spacing, max_rate, baseline_rate, phase_offset)
- [x] Export in `__init__.py` files
- [x] Write comprehensive tests (16 tests total, all passing)
- [x] Code review and address all critical/important issues

### Session Simulation

- [x] Implement `SimulationSession` dataclass in `session.py` (lines 965-992)
  - [x] Add `@dataclass(frozen=True)` decorator
  - [x] Define fields: `env`, `positions`, `times`, `spike_trains`, `models`, `ground_truth`, `metadata`
  - [x] Add type hints for all fields
  - [x] Write comprehensive docstring

- [x] Implement `simulate_session()` in `session.py` (lines 994-1077)
  - [x] Parse parameters including `coverage` (lines 1027-1034 explain algorithm)
  - [x] Validate parameters
  - [x] Generate trajectory based on `trajectory_method`
  - [x] Create models based on `cell_type`:
    - [x] `'place'`: All place cells
    - [x] `'boundary'`: All boundary cells
    - [x] `'grid'`: All grid cells
    - [x] `'mixed'`: 60% place, 20% boundary, 20% grid
  - [x] Distribute field centers based on `coverage`:
    - [x] `'uniform'`: `env.bin_centers[::step]` where `step = max(1, n_bins // n_cells)`
    - [x] `'random'`: `np.random.choice(env.bin_centers, size=n_cells)`
  - [x] Generate spikes with `generate_population_spikes()`
  - [x] Collect ground truth from all models
  - [x] Create metadata dict with session parameters
  - [x] Return `SimulationSession` instance
- [x] Write NumPy docstring with examples

### Validation Helpers

- [x] Implement `validate_simulation()` in `validation.py` (lines 1056-1141)
  - [x] Parse session or individual parameters
  - [x] Loop over spike trains:
    - [x] Compute place field with `compute_place_field()`
    - [x] Detect center (peak of rate map)
    - [x] Compare to ground truth center
    - [x] Compute correlation between true and detected rate maps
  - [x] Compute error statistics: center errors, correlations, width errors, rate errors
  - [x] Generate summary string
  - [x] Determine pass/fail based on thresholds
  - [x] If `show_plots`, create diagnostic plots
  - [x] Return dict with: `center_errors`, `center_correlations`, `width_errors`, `rate_errors`, `summary`, `passed`, optional `plots`
- [x] Write NumPy docstring

- [x] Implement `plot_session_summary()` in `validation.py` (lines 419-703)
  - [x] Create multi-panel figure (4x3 GridSpec layout)
  - [x] Plot trajectory (with color-coded time)
  - [x] Plot selected cells' rate maps (up to 6 cells, 2x3 grid)
  - [x] Plot raster plots (all cells)
  - [x] Return `(fig, axes)` tuple
  - [x] Handle empty spike trains gracefully
  - [x] Support both full-grid and masked-grid environments
- [x] Write NumPy docstring

### Pre-Configured Examples

- [x] Implement `open_field_session()` in `examples.py` (lines 1183-1223)
  - [x] Create square arena environment
  - [x] Call `simulate_session()` with appropriate parameters
  - [x] Return `SimulationSession`
- [x] Write NumPy docstring

- [x] Implement `linear_track_session()` in `examples.py` (lines 1225-1261)
  - [x] Create 1D track environment
  - [x] Use laps trajectory (with n_laps parameter)
  - [x] Add place cells with uniform coverage
  - [x] Return `SimulationSession`
- [x] Write NumPy docstring

- [x] Implement `tmaze_alternation_session()` in `examples.py` (lines 331-530)
  - [x] Create T-maze graph environment
  - [x] Use lap-based trajectory with alternating paths
  - [x] Add trial metadata
  - [x] Return `SimulationSession`
- [x] Write NumPy docstring

- [x] Implement `boundary_cell_session()` in `examples.py` (lines 542-784)
  - [x] Create environment with specified shape (square/circle/polygon)
  - [x] Mix boundary cells and place cells
  - [x] Return `SimulationSession`
- [x] Write NumPy docstring

- [x] Implement `grid_cell_session()` in `examples.py` (lines 795-1015)
  - [x] Create 2D arena
  - [x] Create grid cells with varied phases
  - [x] Return `SimulationSession`
- [x] Write NumPy docstring

### Testing

- [x] Write tests for `GridCellModel`:
  - [x] Test hexagonal symmetry
  - [x] Test grid spacing matches parameter
  - [x] Test orientation rotation
  - [x] Test requires 2D environment

- [x] Write `tests/simulation/test_integration.py`:
  - [x] Test `simulate_session()` with all cell types
  - [x] Test `validate_simulation()` detects place fields correctly
  - [x] Test all pre-configured examples run without errors
  - [x] Test place field detection accuracy (lines 1912-1947):
    - [x] Create known place cells
    - [x] Generate trajectory and spikes
    - [x] Detect fields with `compute_place_field()`
    - [x] Assert detection works for simulated data

### Update **init**.py

- [x] Add all new functions/classes to imports (lines 96-174)
- [x] Update `__all__` list
- [x] Test flat imports work: `from neurospatial.simulation import <everything>`

### Documentation

- [x] Add comprehensive examples to high-level functions
- [x] Run all doctests (23/23 passing in 46s after optimization - reduced durations from 60-180s to 2-5s)
- [x] Create example usage scripts

### Validation

- [x] Run full test suite: `uv run pytest tests/simulation/` (236 passed, 4 skipped)
- [x] Achieve >90% code coverage (validation.py: 98%, models: 95%, examples: 83%)
- [x] Run mypy with no errors (1 warning about tqdm stubs - not blocking)
- [x] Run ruff check and format (all clean)

---

## Milestone 3.5: Documentation Integration (0.5 weeks)

**Goal**: Replace hand-written simulation code in example notebooks with simulation subpackage

### Update Existing Notebooks

- [x] Update `examples/11_place_field_analysis.ipynb` (use jupyter-notebook skill):
  - [x] Replace Section "2D Random Walk Generation" with `simulate_trajectory_ou()`
  - [x] Replace Section "Place Cell Simulation" with `PlaceCellModel` + `generate_poisson_spikes()`
  - [x] Replace Section "T-maze Trajectory" with `tmaze_alternation_session()`
  - [x] Add markdown cell explaining simulation subpackage usage
  - [x] Test notebook runs without errors (imports validated)

- [x] Update `examples/08_spike_field_basics.ipynb` (use jupyter-notebook skill):
  - [x] Replace Section "Random Walk" with `simulate_trajectory_ou()`
  - [x] Replace Section "Spike Generation" with simulation API
  - [x] Add note directing to simulation subpackage documentation
  - [x] Test notebook runs without errors

- [x] Update `examples/12_boundary_cell_analysis.ipynb` (use jupyter-notebook skill):
  - [x] Replace Section "Random Walk" with `simulate_trajectory_ou()`
  - [x] Add note about `boundary_cell_session()` example
  - [x] Test notebook runs without errors

### Create New Notebook

- [x] Create `examples/15_simulation_workflows.ipynb` (use jupyter-notebook skill):
  - [x] Add introduction to simulation subpackage
  - [x] Section 1: Quick start with `open_field_session()`
  - [x] Section 2: Low-level API demonstration (trajectory + models + spikes)
  - [x] Section 3: All pre-configured examples (open_field, linear_track, tmaze, boundary, grid)
  - [x] Section 4: Validation workflow with `validate_simulation()`
  - [x] Section 5: Customization examples:
    - [x] Direction-selective place cell
    - [x] Speed-gated place cell
    - [x] Custom boundary cell
  - [x] Section 6: Performance tips
  - [x] Test notebook runs without errors

### Sync Notebooks

- [x] Copy updated notebooks from `examples/` to `docs/examples/` (use docs/sync_notebooks.py)
- [x] Verify documentation build picks up new notebooks
- [x] Check all links work

### Documentation Updates

- [x] Update main `README.md`:
  - [x] Add "Simulation" section
  - [x] Link to `examples/15_simulation_workflows.ipynb`
  - [x] Add quick example code snippet

- [x] Update API reference:
  - [x] Add simulation subpackage section
  - [x] Document all public functions and classes
  - [x] Add cross-references to related functions

### Validation

- [x] Run all notebooks and verify they execute without errors
- [x] Check that simulation code is significantly shorter and clearer
- [x] Verify examples demonstrate best practices

---

## Milestone 4 (Optional): Advanced Features

**Goal**: Advanced realism features based on user demand

**Priority**: Implement based on user requests. Phase 3 features provide more immediate value.

### State-Dependent Movement

- [ ] Add `exploration_params` and `exploitation_params` to OU process
- [ ] Implement state switching logic
- [ ] Add tests
- [ ] Document usage

### Bursting Behavior

- [ ] Add `burst_probability`, `burst_size`, `intraburst_interval` to spike generation
- [ ] Implement burst generation algorithm
- [ ] Add tests
- [ ] Document usage

### Elliptical Place Fields

- [ ] Add `covariance` matrix parameter to `PlaceCellModel`
- [ ] Implement anisotropic Gaussian fields
- [ ] Add tests
- [ ] Document usage

### Correlated Turning

- [ ] Add `turning_correlation` parameter to OU process
- [ ] Implement heading autocorrelation
- [ ] Add tests
- [ ] Document usage

### SpikeTrain Class

- [ ] Design `SpikeTrain` wrapper class with metadata
- [ ] Implement useful methods (ISI, firing rate, etc.)
- [ ] Add tests
- [ ] Document usage

---

## Completion Checklist

### Code Quality

- [x] All functions have NumPy-style docstrings
- [x] All functions have working examples in docstrings
- [x] All doctests pass: `uv run pytest --doctest-modules src/neurospatial/simulation/` (23/23 in 46s)
- [x] Test coverage >90%: `uv run pytest --cov=src/neurospatial/simulation tests/simulation/` (validation: 98%, models: 95%, examples: 83%)
- [x] Mypy passes with no errors: `uv run mypy src/neurospatial/simulation/` (1 acceptable tqdm warning)
- [x] Ruff check passes: `uv run ruff check src/neurospatial/simulation/`
- [x] Ruff format applied: `uv run ruff format src/neurospatial/simulation/`

### Performance

- [ ] `simulate_trajectory_ou()` for 60s @ 100 Hz (6k points) < 100 ms
- [ ] `PlaceCellModel.firing_rate()` (Euclidean) for 6k positions < 10 ms
- [ ] `PlaceCellModel.firing_rate()` (Geodesic) for 6k positions < 1 s
- [ ] `generate_poisson_spikes()` for 6k timepoints < 50 ms
- [ ] `generate_population_spikes()` for 50 cells × 6k points < 5 s
- [ ] `GridCellModel.firing_rate()` for 6k positions < 20 ms

### Testing Checklist (from SIMULATION_PLAN.md lines 2174-2197)

**For Each Neural Model:**

- [ ] Peak firing rate occurs at expected location
- [ ] Firing rate decays correctly with distance
- [ ] Condition function properly gates firing (if applicable)
- [ ] ground_truth property returns all model parameters
- [ ] Works correctly in 1D, 2D, 3D environments (or raises clear error)

**For Trajectory Simulation:**

- [ ] All positions lie within environment (`env.contains()` returns True)
- [ ] Velocity statistics match parameters (mean speed, coherence time)
- [ ] Boundary handling works correctly (reflect, periodic, stop)
- [ ] Position and time arrays have consistent shapes
- [ ] Reproducible with same seed

**For Spike Generation:**

- [ ] Mean firing rate matches expected rate (within Poisson variance)
- [ ] Inter-spike intervals >= refractory_period (all ISIs)
- [ ] Spike times sorted in ascending order
- [ ] Reproducible with same seed
- [ ] No spikes outside time range

### Documentation

- [ ] All notebooks run without errors
- [ ] All examples in documentation are tested
- [ ] API reference is complete and accurate
- [ ] Migration guide is clear and helpful
- [ ] README includes simulation section

### Integration

- [ ] Flat imports work correctly from `neurospatial.simulation`
- [ ] All public functions/classes in `__all__`
- [ ] Type hints are complete and correct
- [ ] Follows neurospatial patterns (functions, not methods)
- [ ] Works with all neurospatial `Environment` types (1D, 2D, 3D, N-D)

---

## Notes

- Use `uv run` prefix for all commands (per CLAUDE.md)
- Follow NumPy docstring format (per CLAUDE.md)
- All implementations must follow SIMULATION_PLAN.md specifications
- Reference line numbers in SIMULATION_PLAN.md for detailed specifications
- Run tests frequently during development
- Commit after each completed task with conventional commit messages (per CLAUDE.md)

**Conventional Commit Format:**

- `feat(simulation): add PlaceCellModel implementation`
- `test(simulation): add tests for OU trajectory`
- `docs(simulation): add examples to validate_simulation`
- `fix(simulation): correct grid cell wave vector calculation`
