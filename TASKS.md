# Package Reorganization Tasks

**Goal**: Reorganize neurospatial into a clean, domain-centric structure following the [PLAN.md](PLAN.md) specification.

**Success Criteria**:

- All tests pass (`uv run pytest`)
- Type checking passes (`uv run mypy src/neurospatial/`)
- Linting passes (`uv run ruff check . && uv run ruff format .`)
- No circular imports
- All doctest examples work (`uv run pytest --doctest-modules src/neurospatial/`)

---

## Milestone 1: Create Directory Structure

**Goal**: Create the new package structure without moving any code yet.

**Dependencies**: None (can start immediately)

### Tasks

- [x] Create `src/neurospatial/encoding/` directory with `__init__.py`
- [x] Create `src/neurospatial/behavior/` directory with `__init__.py`
- [x] Create `src/neurospatial/io/` directory with `__init__.py`
- [x] Create `src/neurospatial/ops/` directory with `__init__.py`
- [x] Create `src/neurospatial/stats/` directory with `__init__.py`

**Verification**:

```bash
ls -la src/neurospatial/{encoding,behavior,io,ops,stats}/__init__.py
```

**Note**: Also moved `io.py` → `io/files.py` early (from M3) to avoid import conflict.

---

## Milestone 2: Move ops/ Modules (Tier 1-3 - No Domain Dependencies)

**Goal**: Move low-level utilities to `ops/`. These have minimal internal dependencies.

**Dependencies**: Milestone 1 complete

**Why first?** Per PLAN.md dependency graph, ops/ is at Tier 3 and is imported by all domain modules. Moving it first creates a stable foundation.

### Tasks

#### 2.1: Move and Rename Files

- [x] Move `spatial.py` → `ops/binning.py`
- [x] Move `distance.py` → `ops/distance.py`
- [x] Move `field_ops.py` → `ops/normalize.py`
- [x] Move `kernels.py` → `ops/smoothing.py`
- [x] Move `primitives.py` → `ops/graph.py`
- [x] Move `differential.py` → `ops/calculus.py`
- [x] Move `transforms.py` → `ops/transforms.py` (then merge calibration.py into it)
- [x] Merge `calibration.py` into `ops/transforms.py`
- [x] Move `alignment.py` → `ops/alignment.py`
- [ ] Move `reference_frames.py` → `ops/egocentric.py`
- [ ] Move `visibility.py` → `ops/visibility.py`
- [ ] Move `basis.py` → `ops/basis.py`

#### 2.2: Update ops/**init**.py

- [ ] Export public API from `ops/__init__.py`
- [ ] Ensure all moved functions are importable from new paths

#### 2.3: Update Internal Imports

For each moved file:

```bash
# Find old imports
grep -r "from neurospatial.spatial import" src/ tests/
grep -r "from neurospatial.field_ops import" src/ tests/
# ... repeat for each moved file
```

- [ ] Update imports for `spatial` → `ops.binning`
- [ ] Update imports for `distance` → `ops.distance`
- [ ] Update imports for `field_ops` → `ops.normalize`
- [ ] Update imports for `kernels` → `ops.smoothing`
- [ ] Update imports for `primitives` → `ops.graph`
- [ ] Update imports for `differential` → `ops.calculus`
- [x] Update imports for `transforms` → `ops.transforms`
- [x] Update imports for `calibration` → `ops.transforms`
- [x] Update imports for `alignment` → `ops.alignment`
- [ ] Update imports for `reference_frames` → `ops.egocentric`
- [ ] Update imports for `visibility` → `ops.visibility`
- [ ] Update imports for `basis` → `ops.basis`

**Verification**:

```bash
uv run pytest tests/ -x -v
uv run mypy src/neurospatial/
```

---

## Milestone 3: Move io/ Module

**Goal**: Consolidate I/O functionality into `io/` package.

**Dependencies**: Milestone 1 complete

### Tasks

- [ ] Move `io.py` → `io/files.py`
- [ ] Move `nwb/` → `io/nwb/`
- [ ] Update `io/__init__.py` with re-exports: `to_file`, `from_file`, `to_dict`, `from_dict`
- [ ] Update internal imports for `io` → `io.files`
- [ ] Update internal imports for `nwb` → `io.nwb`

**Verification**:

```bash
uv run pytest tests/ -x -v
uv run pytest tests/nwb/ -v  # If NWB tests exist
```

---

## Milestone 4: Move stats/ Module

**Goal**: Consolidate statistical methods into `stats/` package.

**Dependencies**: Milestone 1 complete

### Tasks

#### 4.1: Create stats/circular.py

- [ ] Move circular functions from `metrics/circular.py` → `stats/circular.py`
- [ ] Move circular basis functions from `metrics/circular_basis.py` → `stats/circular.py`
- [ ] Make private functions public: `_circular_mean` → `circular_mean`, etc.
- [ ] Move `wrap_angle()` from `metrics/vte.py` → `stats/circular.py`

#### 4.2: Create stats/shuffle.py

- [ ] Move shuffle functions from `decoding/shuffle.py` → `stats/shuffle.py`
- [ ] Add new functions: `shuffle_trials()`, `shuffle_spikes_isi()` (per PLAN.md)

#### 4.3: Create stats/surrogates.py

- [ ] Extract surrogate functions from `decoding/shuffle.py` → `stats/surrogates.py`
- [ ] Add: `generate_jittered_spikes()` (per PLAN.md)

#### 4.4: Update Internal Imports

- [ ] Update imports for `metrics.circular` → `stats.circular`
- [ ] Update imports for `metrics.circular_basis` → `stats.circular_basis`
- [ ] Update imports for `decoding.shuffle` → `stats.shuffle`

**Verification**:

```bash
uv run pytest tests/ -x -v
```

---

## Milestone 5: Move behavior/ Module

**Goal**: Consolidate behavioral analysis into `behavior/` package.

**Dependencies**: Milestones 2, 4 complete (behavior imports ops/ and stats/)

### Tasks

#### 5.1: Create behavior/trajectory.py

- [ ] Move functions from `metrics/trajectory.py` → `behavior/trajectory.py`
- [ ] Move `compute_trajectory_curvature()` from `behavioral.py` → `behavior/trajectory.py`

#### 5.2: Create behavior/segmentation.py

- [ ] Move all from `segmentation/` → `behavior/segmentation.py` (flatten the directory)

#### 5.3: Create behavior/navigation.py

- [ ] Move navigation functions from `behavioral.py` → `behavior/navigation.py`
- [ ] Move functions from `metrics/path_efficiency.py` → `behavior/navigation.py`
- [ ] Move functions from `metrics/goal_directed.py` → `behavior/navigation.py`

#### 5.4: Create behavior/decisions.py

- [ ] Move functions from `metrics/decision_analysis.py` → `behavior/decisions.py`
- [ ] Move functions from `metrics/vte.py` → `behavior/decisions.py`

#### 5.5: Create behavior/reward.py

- [ ] Move functions from `reward.py` → `behavior/reward.py`

#### 5.6: Update Internal Imports

- [ ] Update imports for `metrics.trajectory` → `behavior.trajectory`
- [ ] Update imports for `segmentation` → `behavior.segmentation`
- [ ] Update imports for `behavioral` → `behavior.navigation`
- [ ] Update imports for `metrics.path_efficiency` → `behavior.navigation`
- [ ] Update imports for `metrics.goal_directed` → `behavior.navigation`
- [ ] Update imports for `metrics.decision_analysis` → `behavior.decisions`
- [ ] Update imports for `metrics.vte` → `behavior.decisions`
- [ ] Update imports for `reward` → `behavior.reward`

**Verification**:

```bash
uv run pytest tests/ -x -v
```

---

## Milestone 6: Move encoding/ Module

**Goal**: Consolidate neural encoding analysis into `encoding/` package.

**Dependencies**: Milestones 2, 4 complete (encoding imports ops/ and stats/)

### Tasks

#### 6.1: Create encoding/place.py

- [ ] Move functions from `spike_field.py` → `encoding/place.py`
- [ ] Move functions from `metrics/place_fields.py` → `encoding/place.py`

#### 6.2: Create encoding/grid.py

- [ ] Move functions from `metrics/grid_cells.py` → `encoding/grid.py`

#### 6.3: Create encoding/head_direction.py

- [ ] Move functions from `metrics/head_direction.py` → `encoding/head_direction.py`
- [ ] Add re-exports from `stats.circular`: `rayleigh_test`, `mean_resultant_length`, `circular_mean`

#### 6.4: Create encoding/border.py

- [ ] Move functions from `metrics/boundary_cells.py` → `encoding/border.py`

#### 6.5: Create encoding/object_vector.py

- [ ] Move functions from `object_vector_field.py` → `encoding/object_vector.py`
- [ ] Move functions from `metrics/object_vector_cells.py` → `encoding/object_vector.py`

#### 6.6: Create encoding/spatial_view.py

- [ ] Move functions from `spatial_view_field.py` → `encoding/spatial_view.py`
- [ ] Move functions from `metrics/spatial_view_cells.py` → `encoding/spatial_view.py`
- [ ] Add re-exports from `ops.visibility`: `compute_viewed_location`, `compute_viewshed`, `visibility_occupancy`, `FieldOfView`

#### 6.7: Create encoding/phase_precession.py

- [ ] Move functions from `metrics/phase_precession.py` → `encoding/phase_precession.py`

#### 6.8: Create encoding/population.py

- [ ] Move functions from `metrics/population.py` → `encoding/population.py`

#### 6.9: Update Internal Imports

- [ ] Update imports for `spike_field` → `encoding.place`
- [ ] Update imports for `metrics.place_fields` → `encoding.place`
- [ ] Update imports for `metrics.grid_cells` → `encoding.grid`
- [ ] Update imports for `metrics.head_direction` → `encoding.head_direction`
- [ ] Update imports for `metrics.boundary_cells` → `encoding.border`
- [ ] Update imports for `object_vector_field` → `encoding.object_vector`
- [ ] Update imports for `metrics.object_vector_cells` → `encoding.object_vector`
- [ ] Update imports for `spatial_view_field` → `encoding.spatial_view`
- [ ] Update imports for `metrics.spatial_view_cells` → `encoding.spatial_view`
- [ ] Update imports for `metrics.phase_precession` → `encoding.phase_precession`
- [ ] Update imports for `metrics.population` → `encoding.population`

**Verification**:

```bash
uv run pytest tests/ -x -v
```

---

## Milestone 7: Reorganize decoding/ Module

**Goal**: Clean up decoding/ after shuffle functions moved to stats/.

**Dependencies**: Milestone 4 complete

### Tasks

- [ ] Remove `decoding/shuffle.py` (contents moved to `stats/shuffle.py` and `stats/surrogates.py`)
- [ ] Update `decoding/__init__.py` to re-export from `stats.shuffle` and `stats.surrogates`
- [ ] Verify decoding/ structure matches PLAN.md specification

**Verification**:

```bash
uv run pytest tests/ -x -v
```

---

## Milestone 8: Consolidate animation/ Module

**Goal**: Move visualization config to animation/, delete empty visualization/.

**Dependencies**: None (can run in parallel with other milestones)

### Tasks

- [ ] Move `visualization/scale_bar.py` → `animation/config.py`
- [ ] Update imports for `visualization.scale_bar` → `animation.config`
- [ ] Delete empty `visualization/` directory
- [ ] Verify animation/ structure matches PLAN.md specification

**Verification**:

```bash
uv run pytest tests/ -x -v
```

---

## Milestone 9: Update Top-Level **init**.py

**Goal**: Reduce top-level exports to sparse set (5 core exports).

**Dependencies**: All move milestones complete (2-8)

### Tasks

- [ ] Update `src/neurospatial/__init__.py` to export only:
  - `Environment`
  - `EnvironmentNotFittedError`
  - `Region`
  - `Regions`
  - `CompositeEnvironment`
- [ ] Remove all other top-level exports
- [ ] Update module docstring with new import patterns

**Verification**:

```bash
uv run python -c "from neurospatial import Environment, Region, Regions, CompositeEnvironment, EnvironmentNotFittedError; print('Core imports OK')"
uv run pytest tests/ -x -v
```

---

## Milestone 10: Delete Old Files

**Goal**: Remove empty/migrated files and directories.

**Dependencies**: All milestones 2-9 complete, all tests passing

### Tasks

Delete files (in order):

- [ ] Delete `src/neurospatial/behavioral.py`
- [ ] Delete `src/neurospatial/spatial.py`
- [ ] Delete `src/neurospatial/distance.py` (if moved, not if original)
- [ ] Delete `src/neurospatial/field_ops.py`
- [ ] Delete `src/neurospatial/kernels.py`
- [ ] Delete `src/neurospatial/primitives.py`
- [ ] Delete `src/neurospatial/differential.py`
- [ ] Delete `src/neurospatial/transforms.py` (original, not ops/)
- [ ] Delete `src/neurospatial/alignment.py` (original, not ops/)
- [ ] Delete `src/neurospatial/calibration.py`
- [ ] Delete `src/neurospatial/basis.py` (original, not ops/)
- [ ] Delete `src/neurospatial/reward.py`
- [ ] Delete `src/neurospatial/reference_frames.py`
- [ ] Delete `src/neurospatial/spike_field.py`
- [ ] Delete `src/neurospatial/object_vector_field.py`
- [ ] Delete `src/neurospatial/spatial_view_field.py`
- [ ] Delete `src/neurospatial/visibility.py` (original, not ops/)
- [ ] Delete `src/neurospatial/io.py` (original, not io/files.py)

Delete directories:

- [ ] Delete `src/neurospatial/visualization/` directory
- [ ] Delete `src/neurospatial/segmentation/` directory (after flattening to behavior/)
- [ ] Delete `src/neurospatial/metrics/` directory (after distributing contents)

**Verification**:

```bash
uv run pytest tests/ -x -v
uv run mypy src/neurospatial/
```

---

## Milestone 11: Update Documentation

**Goal**: Update all documentation to reflect new import paths.

**Dependencies**: Milestone 10 complete

### Tasks

#### 11.1: Update CLAUDE.md

- [ ] Update "Most Common Patterns" section with new import paths
- [ ] Update "Quick Navigation" tables
- [ ] Update "Architecture Overview" section

#### 11.2: Update .claude/ Documentation

- [ ] Update `.claude/QUICKSTART.md` with new import paths
- [ ] Update `.claude/API_REFERENCE.md` with new module structure
- [ ] Update `.claude/PATTERNS.md` if needed
- [ ] Update `.claude/TROUBLESHOOTING.md` if needed
- [ ] Update `.claude/ADVANCED.md` if needed
- [ ] Update `.claude/ARCHITECTURE.md` with new structure

#### 11.3: Update mkdocs.yml

- [ ] Update navigation structure in `mkdocs.yml`
- [ ] Verify API reference generation points to new paths

#### 11.4: Update Docstrings

- [ ] Update docstring import examples in moved modules
- [ ] Run doctests to verify: `uv run pytest --doctest-modules src/neurospatial/`

**Verification**:

```bash
uv run pytest --doctest-modules src/neurospatial/
```

---

## Milestone 12: Update Example Notebooks

**Goal**: Update all example notebooks with new import paths.

**Dependencies**: Milestone 10 complete

### Tasks

#### 12.1: Key Notebooks Requiring Updates

- [ ] Update `examples/08_spike_field_basics.ipynb` (spike_field → encoding.place)
- [ ] Update `examples/11_place_field_analysis.ipynb` (metrics.place_fields → encoding.place)
- [ ] Update `examples/12_boundary_cell_analysis.ipynb` (metrics.boundary_cells → encoding.border)
- [ ] Update `examples/13_trajectory_analysis.ipynb` (metrics.trajectory → behavior.trajectory)
- [ ] Update `examples/14_behavioral_segmentation.ipynb` (segmentation → behavior.segmentation)
- [ ] Update `examples/20_bayesian_decoding.ipynb` (verify decoding exports)
- [ ] Update `examples/22_spatial_view_cells.ipynb` (spatial_view_field → encoding.spatial_view)

#### 12.2: Update Remaining Notebooks

- [ ] Grep all notebooks for old import paths and update
- [ ] Update corresponding `docs/examples/` notebooks (mirrors)

#### 12.3: Verify Notebooks Execute

```bash
for nb in examples/*.ipynb; do
    uv run jupyter nbconvert --to notebook --execute "$nb" --output /dev/null
done
```

- [ ] All notebooks execute without import errors
- [ ] Clear and re-execute notebooks for clean state

**Verification**:

```bash
# Run each notebook to verify
uv run jupyter nbconvert --to notebook --execute examples/08_spike_field_basics.ipynb --output /dev/null
```

---

## Milestone 13: Create Import Test Suite

**Goal**: Add tests to verify new import structure is correct and stable.

**Dependencies**: Milestone 10 complete

### Tasks

- [ ] Create `tests/test_imports.py` with tests for:
  - All new import paths work
  - Core exports from top-level work
  - No circular imports
  - Re-exports work (stats functions from encoding modules)
- [ ] Add test for each domain module's `__all__` exports
- [ ] Verify doctests still pass

**Verification**:

```bash
uv run pytest tests/test_imports.py -v
uv run pytest --doctest-modules src/neurospatial/
```

---

## Milestone 14: Final Verification

**Goal**: Comprehensive verification that reorganization is complete and correct.

**Dependencies**: All previous milestones complete

### Tasks

- [ ] Run full test suite: `uv run pytest`
- [ ] Run type checking: `uv run mypy src/neurospatial/`
- [ ] Run linting: `uv run ruff check . && uv run ruff format .`
- [ ] Run doctests: `uv run pytest --doctest-modules src/neurospatial/`
- [ ] Verify no circular imports (import neurospatial in fresh Python)
- [ ] Verify example usage from PLAN.md works:

```python
import neurospatial as ns
from neurospatial import Environment, Region

# Create environment
env = Environment.from_samples(positions, bin_size=2.0)

# Neural encoding
from neurospatial.encoding import place, grid, head_direction
rate_map = place.compute_place_field(env, spikes, times, positions)

# Neural decoding
from neurospatial.decoding import decode_position

# Behavioral analysis
from neurospatial.behavior import segmentation, navigation, trajectory

# Events
from neurospatial.events import peri_event_histogram

# Low-level ops
from neurospatial.ops import distance, normalize
```

**Success Criteria**:

- [ ] All tests pass
- [ ] All type checks pass
- [ ] All linting passes
- [ ] All notebooks execute
- [ ] Documentation is up-to-date
- [ ] Example usage from PLAN.md works

---

## Notes for Implementation

### Git Strategy

Consider creating a feature branch for this reorganization:

```bash
git checkout -b refactor/package-reorganization
```

Commit after each milestone completion:

```bash
git commit -m "refactor(M2): move ops/ modules

Move low-level utilities to ops/ package:
- spatial.py → ops/binning.py
- distance.py → ops/distance.py
- ... etc
"
```

### Testing Between Tasks

Run tests frequently during implementation:

```bash
# Quick smoke test
uv run pytest tests/test_environment.py -x -v

# Full test after each milestone
uv run pytest -x -v
```

### Import Update Strategy

Use ripgrep to find all imports needing update:

```bash
# Example: find all imports of old module
rg "from neurospatial.spatial import" src/ tests/
rg "from neurospatial import.*map_points_to_bins" src/ tests/
```

### Handling Circular Imports

If circular imports occur:

1. Check dependency tier in PLAN.md
2. Ensure lower-tier modules don't import from higher tiers
3. Use re-exports at the `__init__.py` level, not direct imports

---

## Progress Tracking

| Milestone | Status | Notes |
|-----------|--------|-------|
| M1: Directory Structure | Complete | Also moved io.py → io/files.py early |
| M2: ops/ Modules | Not Started | |
| M3: io/ Module | Partial | io.py → io/files.py done in M1; nwb/ still pending |
| M4: stats/ Module | Not Started | |
| M5: behavior/ Module | Not Started | |
| M6: encoding/ Module | Not Started | |
| M7: decoding/ Cleanup | Not Started | |
| M8: animation/ Consolidation | Not Started | |
| M9: Top-Level **init**.py | Not Started | |
| M10: Delete Old Files | Not Started | |
| M11: Update Documentation | Not Started | |
| M12: Update Notebooks | Not Started | |
| M13: Import Tests | Not Started | |
| M14: Final Verification | Not Started | |
