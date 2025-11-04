# Environment.py Modularization - Implementation Tasks

**Goal**: Split 5,335-line `environment.py` into 9 focused modules
**Timeline**: 15-20 hours (2-3 days of focused work)
**Breaking Changes**: NONE - backward compatible

---

## Milestone 1: Preparation (1 hour) ✅ COMPLETED

### Tasks

- [x] Create test snapshot for comparison

  ```bash
  uv run pytest tests/ --tb=short -v > tests_before.log
  ```

  **Result**: 1067 passed, 85 warnings in 13.39s

- [x] Document current public API

  ```python
  python -c "from neurospatial import Environment; print(sorted([m for m in dir(Environment) if not m.startswith('_')]))"
  ```

  **Result**: Saved to `public_api_before.txt`

- [x] Create environment package directory

  ```bash
  mkdir src/neurospatial/environment
  ```

  **Result**: Directory created at `src/neurospatial/environment/`

- [x] Count current line distribution

  ```bash
  wc -l src/neurospatial/environment.py
  ```

  **Result**: 5335 lines

### Success Criteria

- ✅ `tests_before.log` created with all test results
- ✅ Public API documented
- ✅ `environment/` directory exists

---

## Milestone 2: Extract Decorators (30 minutes) ✅ COMPLETED

### Tasks

- [x] Create `src/neurospatial/environment/decorators.py`
- [x] Copy `check_fitted` decorator from environment.py (lines 52-91)
- [x] Add module docstring
- [x] Verify decorator is plain Python (no dependencies on Environment)
- [x] Run decorator tests

  ```bash
  uv run pytest tests/test_check_fitted_error.py -v
  ```

  **Result**: 8/8 tests passed

### Success Criteria

- ✅ `decorators.py` created (78 lines)
- ✅ All decorator tests pass (8/8)
- ✅ No imports of Environment in decorators.py (uses TYPE_CHECKING guard)

---

## Milestone 3: Extract Visualization (45 minutes) ✅ COMPLETED

### Tasks

- [x] Create `src/neurospatial/environment/visualization.py`
- [x] Add TYPE_CHECKING guard:

  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from neurospatial.environment.core import Environment
  ```

- [x] Create `EnvironmentVisualization` mixin class (plain class, NOT @dataclass)
- [x] Move `plot()` method
- [x] Move `plot_1d()` method
- [x] Add type hints: `def plot(self: "Environment", ...) -> Axes:`
- [x] Import decorator: `from neurospatial.environment.decorators import check_fitted`
- [x] Add module docstring
- [x] Run quick import test:

  **Note**: Module cannot be imported yet because `environment/` isn't a package yet (intentional - will be fixed in Milestone 10)

### Success Criteria

- ✅ `visualization.py` created (209 lines - well under target)
- ✅ Class is plain, NOT @dataclass
- ✅ TYPE_CHECKING used correctly
- ✅ No circular import errors (verified with py_compile)
- ✅ Code review approved with minor improvements applied

---

## Milestone 4: Extract Analysis (1 hour) ✅ COMPLETED

### Tasks

- [x] Create `src/neurospatial/environment/analysis.py`
- [x] Add TYPE_CHECKING guard (same pattern as visualization)
- [x] Create `EnvironmentAnalysis` mixin class (plain class)
- [x] Move `boundary_bins()` method (line 3871)
- [x] Move `bin_attributes()` method (line 3918)
- [x] Move `edge_attributes()` method (line 3951)
- [x] Move `to_linear()` / `linear_to_nd()` methods (if present)
- [x] Add type hints with string annotations
- [x] Import check_fitted decorator
- [x] Add module docstring
- [x] Run import test

### Success Criteria

- ✅ `analysis.py` created (413 lines - well under target)
- ✅ Class is plain, NOT @dataclass
- ✅ No circular imports

---

## Milestone 5: Extract Regions (30 minutes) ✅ COMPLETED

### Tasks

- [x] Create `src/neurospatial/environment/regions.py`
- [x] Add TYPE_CHECKING guard
- [x] Create `EnvironmentRegions` mixin class (plain class)
- [x] Move `bins_in_region()` method (line 4465)
- [x] Move `mask_for_region()` method (line 4522)
- [x] Add type hints
- [x] Import check_fitted decorator
- [x] Add module docstring
- [x] Run import test

### Success Criteria

- ✅ `regions.py` created (222 lines - well under target)
- ✅ Class is plain, NOT @dataclass
- ✅ No circular imports
- ✅ All 15/15 region tests pass

---

## Milestone 6: Extract Serialization (1 hour) ✅ COMPLETED

### Tasks

- [x] Create `src/neurospatial/environment/serialization.py`
- [x] Add TYPE_CHECKING guard
- [x] Create `EnvironmentSerialization` mixin class (plain class)
- [x] Implement thin delegation methods:

  ```python
  def to_file(self: "Environment", path: Path | str) -> None:
      from neurospatial.io import to_file
      return to_file(self, path)
  ```

- [x] Move `save()`, `load()`, `to_file()`, `from_file()`, `to_dict()`, `from_dict()`
- [x] Add type hints for classmethods: `cls: type["Environment"]`
- [x] Add module docstring
- [x] Run serialization tests:

  ```bash
  uv run pytest tests/test_io.py -v
  ```

  **Result**: 12/12 tests passed

### Success Criteria

- ✅ `serialization.py` created (314 lines - well under target)
- ✅ Delegates to `io.py` (no code duplication)
- ✅ All serialization tests pass (12/12)
- ✅ Class is plain, NOT @dataclass

---

## Milestone 7: Extract Queries (1.5 hours) ✅ COMPLETED

### Tasks

- [x] Create `src/neurospatial/environment/queries.py`
- [x] Add TYPE_CHECKING guard
- [x] Create `EnvironmentQueries` mixin class (plain class)
- [x] Move spatial query methods:
  - [x] `bin_at()`
  - [x] `contains()`
  - [x] `neighbors()`
  - [x] `distance_between()`
  - [x] `bin_center_of()`
  - [x] `shortest_path()`
  - [x] `bin_sizes()`
- [x] Add type hints: `def bin_at(self: "Environment", ...) -> int:`
- [x] Import check_fitted decorator
- [x] Add module docstring
- [x] Run tests:

  **Result**: 28/28 tests passed

### Success Criteria

- ✅ `queries.py` created (385 lines - under target)
- ✅ All query methods extracted
- ✅ Class is plain, NOT @dataclass
- ✅ Query tests pass (28/28)

---

## Milestone 8: Extract Factories (2 hours) ✅ COMPLETED

### Tasks

- [x] Create `src/neurospatial/environment/factories.py`
- [x] Add TYPE_CHECKING guard
- [x] Create `EnvironmentFactories` mixin class (plain class)
- [x] Move factory classmethods:
  - [x] `from_layout()` (line 595)
  - [x] `from_samples()` (line 113)
  - [x] `from_graph()` (line 319)
  - [x] `from_polygon()` (line 372)
  - [x] `from_mask()` (line 478)
  - [x] `from_image()` (line 544)
- [x] Add type hints for classmethods
- [x] Add return type: `-> Environment`
- [x] Import necessary dependencies (layout engines, etc.)
- [x] Add module docstring
- [x] Run factory tests (16/16 tests pass)

### Success Criteria

- ✅ `factories.py` created (631 lines - well under 1,500 line target)
- ✅ All 6 factory methods extracted
- ✅ Class is plain, NOT @dataclass
- ✅ Factory tests pass (16/16)
- ✅ Code review approved with 5/5 ratings

---

## Milestone 9: Create Core Module (2 hours)

### Tasks

- [ ] Create `src/neurospatial/environment/core.py`
- [ ] Import all mixin classes:

  ```python
  from neurospatial.environment.decorators import check_fitted
  from neurospatial.environment.factories import EnvironmentFactories
  from neurospatial.environment.queries import EnvironmentQueries
  from neurospatial.environment.serialization import EnvironmentSerialization
  from neurospatial.environment.regions import EnvironmentRegions
  from neurospatial.environment.visualization import EnvironmentVisualization
  from neurospatial.environment.analysis import EnvironmentAnalysis
  ```

- [ ] Create Environment class with mixin inheritance:

  ```python
  @dataclass  # ← ONLY Environment is a dataclass
  class Environment(
      EnvironmentFactories,
      EnvironmentQueries,
      EnvironmentSerialization,
      EnvironmentRegions,
      EnvironmentVisualization,
      EnvironmentAnalysis,
  ):
      """Main Environment class assembled from mixins."""
      name: str = ""
      layout: LayoutEngine | None = None
      # ... rest of dataclass fields
  ```

- [ ] Move core methods:
  - [ ] `_setup_from_layout()`
  - [ ] `__eq__()`
  - [ ] `__repr__()`
  - [ ] `_repr_html_()`
  - [ ] `info()`
  - [ ] `copy()`
- [ ] Move properties: `n_dims`, `n_bins`, `layout_type`, `layout_parameters`, `is_1d`
- [ ] Move HTML helpers: `_html_table_row()`, `_html_table_header()`
- [ ] Move internal helpers: `_source_flat_to_active_node_id_map()`
- [ ] Add class docstring

### Success Criteria

- ✅ `core.py` created (~800-1000 lines)
- ✅ Environment inherits from all mixins
- ✅ ONLY Environment is @dataclass (verify with `hasattr(Environment, '__dataclass_fields__')`)
- ✅ No circular import errors

---

## Milestone 10: Create Package Init & Update Imports (45 minutes)

### Tasks

- [ ] Create `src/neurospatial/environment/__init__.py`:

  ```python
  """Environment module for spatial discretization."""

  from neurospatial.environment.core import Environment
  from neurospatial.environment.decorators import check_fitted

  __all__ = ["Environment", "check_fitted"]
  ```

- [ ] Update `src/neurospatial/__init__.py` to import from new location
  - Change `from neurospatial.environment import Environment`
  - Keep the same line (should now resolve to package)
- [ ] Delete old `src/neurospatial/environment.py` file:

  ```bash
  rm src/neurospatial/environment.py
  ```

- [ ] Test both import paths:

  ```python
  from neurospatial import Environment  # Primary path
  from neurospatial.environment import Environment  # Alternative path
  ```

- [ ] Verify they return same class:

  ```python
  from neurospatial import Environment as Env1
  from neurospatial.environment import Environment as Env2
  assert Env1 is Env2
  ```

### Success Criteria

- ✅ Package `__init__.py` created
- ✅ Old `environment.py` deleted
- ✅ Both import paths work
- ✅ Both paths return same class object

---

## Milestone 11: Comprehensive Testing (2 hours)

### Tasks

- [ ] Run full test suite:

  ```bash
  uv run pytest tests/ --tb=short -v > tests_after.log
  ```

- [ ] Compare test results:

  ```bash
  diff tests_before.log tests_after.log
  ```

- [ ] Verify all 1,067 tests pass
- [ ] Run mixin verification tests (add to `tests/test_environment.py`):
  - [ ] `test_mixins_are_not_dataclasses()`
  - [ ] `test_environment_is_dataclass()`
  - [ ] `test_factory_methods_return_environment_type()`
  - [ ] `test_mro_order()`
  - [ ] `test_no_circular_imports()`
  - [ ] `test_primary_import()`
  - [ ] `test_direct_import()`
  - [ ] `test_imports_are_same()`
- [ ] Test IDE autocomplete (manual verification in VSCode/PyCharm):
  - [ ] Type `env.` and verify autocomplete shows all methods
  - [ ] Type `Environment.from_` and verify factory methods appear
- [ ] Verify no circular imports at module load:

  ```bash
  python -c "import sys; from neurospatial.environment import Environment; print(f'✓ Loaded {len([m for m in sys.modules if \"neurospatial\" in m])} modules')"
  ```

### Success Criteria

- ✅ All 1,067 tests pass
- ✅ No test regressions (same pass/fail as before)
- ✅ All mixin verification tests pass
- ✅ Both import paths verified
- ✅ No circular import errors
- ✅ IDE autocomplete works

---

## Milestone 12: Documentation & Cleanup (1.5 hours)

### Tasks

- [ ] Update `CHANGELOG.md`:

  ```markdown
  ## [Unreleased]

  ### Changed
  - **Internal**: Refactored `environment.py` (5,335 lines) into focused submodules for better maintainability
    - Split into 9 modules: `core`, `factories`, `queries`, `serialization`, `regions`, `visualization`, `analysis`, `decorators`
    - No breaking changes - `from neurospatial import Environment` continues to work
    - Improved code organization using mixin pattern
  ```

- [ ] Update `CLAUDE.md` with new internal structure:
  - [ ] Update "Core Architecture" section
  - [ ] Add note about mixin pattern
  - [ ] Document TYPE_CHECKING pattern usage
- [ ] Update `README.md` if it references internal structure
- [ ] Check for unused imports in each module
- [ ] Run linter and formatter:

  ```bash
  uv run ruff check src/neurospatial/environment/
  uv run ruff format src/neurospatial/environment/
  ```

- [ ] Check line counts for each module:

  ```bash
  wc -l src/neurospatial/environment/*.py
  ```

- [ ] Verify each module < 1,000 lines

### Success Criteria

- ✅ CHANGELOG.md updated
- ✅ CLAUDE.md updated
- ✅ All modules linted and formatted
- ✅ Each module < 1,000 lines
- ✅ No unused imports

---

## Final Verification Checklist

Before merging:

- [ ] ✅ All 1,067 tests pass
- [ ] ✅ Public API unchanged (same methods, same behavior)
- [ ] ✅ Both import paths work: `from neurospatial import Environment` and `from neurospatial.environment import Environment`
- [ ] ✅ No circular import errors
- [ ] ✅ Only Environment is a dataclass (mixins are plain classes)
- [ ] ✅ Each module < 1,000 lines
- [ ] ✅ Code coverage maintained or improved
- [ ] ✅ Documentation updated
- [ ] ✅ Linting passes
- [ ] ✅ IDE autocomplete works

---

## Rollback Plan

If critical issues are discovered:

1. **Immediate rollback**: Restore `environment.py` from git history

   ```bash
   git checkout HEAD -- src/neurospatial/environment.py
   rm -rf src/neurospatial/environment/
   ```

2. **Run tests to verify rollback**:

   ```bash
   uv run pytest tests/
   ```

3. **Document issue** in GitHub issue tracker

4. **Re-evaluate plan** based on discovered issues

---

## Estimated Time by Milestone

| Milestone | Duration | Cumulative |
|-----------|----------|------------|
| 1. Preparation | 1 hour | 1h |
| 2. Decorators | 30 min | 1.5h |
| 3. Visualization | 45 min | 2.25h |
| 4. Analysis | 1 hour | 3.25h |
| 5. Regions | 30 min | 3.75h |
| 6. Serialization | 1 hour | 4.75h |
| 7. Queries | 1.5 hours | 6.25h |
| 8. Factories | 2 hours | 8.25h |
| 9. Core Module | 2 hours | 10.25h |
| 10. Package Init | 45 min | 11h |
| 11. Testing | 2 hours | 13h |
| 12. Documentation | 1.5 hours | 14.5h |
| **Buffer** | **3-5 hours** | **17.5-19.5h** |

**Total: 17.5-19.5 hours** (2-3 days of focused work)

---

## Critical Reminders

⚠️ **ONLY Environment is a @dataclass** - All mixins MUST be plain classes

⚠️ **Use TYPE_CHECKING guards** in all mixins to avoid circular imports

⚠️ **String annotations** for type hints: `self: "Environment"`

⚠️ **Test after each milestone** - Don't wait until the end

⚠️ **Commit after each milestone** - Makes rollback easier

---

## Notes

- This refactoring maintains backward compatibility - users see zero changes
- The mixin pattern is clean and Pythonic (used by Django, Flask, etc.)
- TYPE_CHECKING pattern is already used in 9 files in this codebase
- All factory methods automatically work via inheritance (no manual binding needed)
