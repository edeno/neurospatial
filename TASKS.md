# neurospatial Implementation Tasks

**Last Updated**: 2025-11-14
**Status**: Active Development
**Reference**: See [PLAN.md](PLAN.md) for detailed specifications

This file tracks concrete implementation tasks for the code review action items. Each task references the corresponding PLAN.md section for full context.

---

## Milestone 1: Critical Security & Stability Fixes (v1.0-alpha)

**Target**: Week 1-2 | **Estimated**: 25 hours | **Status**: Not Started

**Goal**: Eliminate all critical security vulnerabilities and numerical stability issues. Required before any production deployment.

### 1.1 Security: Path Traversal Vulnerability [CRITICAL]

**Ref**: PLAN.md §1.1 | **Effort**: 2h | **Files**: `src/neurospatial/io.py`

- [x] Add path validation in `to_file()` (check for `..` in path parts)
- [x] Add path validation in `from_file()`
- [x] Add test `test_to_file_rejects_path_traversal()` in `tests/test_io.py`
- [x] Add test for symlink attack vector
- [x] Update docstrings to document path restrictions
- [x] Run security audit: `uv run pytest tests/test_io.py -v`

**Verification**: `grep -r "Path traversal" tests/` should show new test
**Status**: ✅ COMPLETED (2025-11-14) - Commit: 5a8cd24

---

### 1.2 Numerical: Division by Zero in Trajectory [CRITICAL]

**Ref**: PLAN.md §1.2 | **Effort**: 3h | **Files**: `src/neurospatial/environment/trajectory.py`

- [x] Add EPSILON constant (1e-12) at module level in `trajectory.py`
- [x] Replace magic numbers with EPSILON in two locations (lines ~1154, ~1172)
- [x] Add test `test_occupancy_ray_parallel_to_edge()` in `tests/test_linear_occupancy.py`
- [x] Add test `test_occupancy_very_small_ray_direction()`
- [x] Add test `test_occupancy_near_epsilon_threshold()`
- [x] Add test `test_occupancy_perfectly_stationary_linear()`
- [x] Verify no inf/nan: All tests pass with `np.all(np.isfinite(occupancy))`
- [x] Run full trajectory test suite: All 45 occupancy tests PASS
- [x] Code review: APPROVED by code-reviewer agent
- [x] Enhanced docstring to document numerical stability

**Acceptance**: All occupancy tests pass with `np.all(np.isfinite(occupancy))`
**Status**: ✅ COMPLETED (2025-11-14) - Ready to commit

---

### 1.3 Numerical: Float Comparison in Hexagonal [CRITICAL]

**Ref**: PLAN.md §1.3 | **Effort**: 2h | **Files**: `src/neurospatial/layout/helpers/hexagonal.py`

- [x] Add MIN_HEX_RADIUS constant (1e-10) at module level in `hexagonal.py`
- [x] Replace `hex_radius == 0` with `np.isclose(hex_radius, 0.0, atol=1e-12)` at line ~220
- [x] Add MIN_HEX_RADIUS validation (1e-10) in `_cartesian_to_fractional_cube()` (lines 217-222)
- [x] Add test `test_cartesian_to_cube_very_small_radius()` in `tests/layout/test_hex_grid_utils.py`
- [x] Add test `test_cartesian_to_cube_zero_radius_handling()`
- [x] Add test `test_cartesian_to_cube_preserves_constraint()` (bonus test for mathematical invariant)
- [x] Enhanced docstring with Raises section documenting validation
- [x] Run hex tests: All 12 tests PASS (9 existing + 3 new)
- [x] Code review: APPROVED by code-reviewer agent
- [x] Ruff and mypy pass

**Verification**: No direct `==` comparisons on floats in hexagonal.py
**Status**: ✅ COMPLETED (2025-11-14) - Ready to commit

---

### 1.4 Correctness: Region Metadata Mutability [CRITICAL]

**Ref**: PLAN.md §1.4 | **Effort**: 2h | **Files**: `src/neurospatial/regions/core.py`

- [x] Import `copy` module at top of `regions/core.py` (line 9)
- [x] Replace `dict(self.metadata)` with `copy.deepcopy(self.metadata)` in `Region.__post_init__` (line 67)
- [x] Updated comment to explain deep copy vs shallow copy (lines 65-66)
- [x] Add test `test_metadata_isolated_from_external_modification()` in `tests/regions/test_core.py`
- [x] Add test `test_nested_metadata_isolated_from_external_modification()` (critical nested dict test)
- [x] Add test `test_metadata_empty_dict_default()` (bonus test)
- [x] Run region tests: All 33 core tests PASS (30 existing + 3 new)
- [x] Code review: APPROVED by code-reviewer agent
- [x] Ruff and mypy pass

**Verification**: Test should verify external mutation doesn't affect Region
**Status**: ✅ COMPLETED (2025-11-14) - Commit: f74e6e7

---

### 1.5 Testing: 3D Environment Coverage [CRITICAL]

**Ref**: PLAN.md §1.5 | **Effort**: 4h | **Files**: `tests/conftest.py`, `tests/test_environment.py`

- [x] Add `simple_3d_env` fixture to `tests/conftest.py`
- [x] Create `TestEnvironment3D` class in `tests/test_environment.py`
- [x] Add test `test_creation_3d()`
- [x] Add test `test_bin_at_3d()`
- [x] Add test `test_neighbors_3d_connectivity()` (verify 6-26 neighbors)
- [x] Add test `test_distance_between_3d()`
- [x] Add test `test_serialization_roundtrip_3d()`
- [x] Add test `test_3d_occupancy()`
- [x] Run 3D tests: All 6 tests PASS
- [x] Verify 2D tests still pass: All 32 tests PASS (zero regressions)
- [x] Code review: APPROVED by code-reviewer agent

**Acceptance**: 3D test class has ≥7 tests, all passing ✅ (6 comprehensive tests)
**Status**: ✅ COMPLETED (2025-11-14) - Ready to commit

---

### 1.6 UX: Progress Feedback [CRITICAL]

**Ref**: PLAN.md §1.6 | **Effort**: 6h | **Files**: `spike_field.py`, `trajectory.py`, `pyproject.toml`

- [ ] Add `tqdm>=4.66.0` to `[project.optional-dependencies]` in `pyproject.toml`
- [ ] Add tqdm import with try/except in `spike_field.py`
- [ ] Add `show_progress=True` parameter to `compute_place_field()`
- [ ] Add progress bar to KDE iterations in `compute_place_field()`
- [ ] Add tqdm import to `trajectory.py`
- [ ] Add `show_progress=True` parameter to `occupancy()`
- [ ] Add progress bar to linear allocation loop in `occupancy()`
- [ ] Add test `test_compute_place_field_with_progress()` in `tests/test_spike_field.py`
- [ ] Add test `test_progress_disabled_when_show_progress_false()`
- [ ] Update docstrings with `show_progress` parameter documentation
- [ ] Test without tqdm: `pip uninstall tqdm && uv run pytest tests/test_spike_field.py`
- [ ] Install tqdm: `uv add --optional progress tqdm`
- [ ] Test with tqdm: `uv run pytest tests/test_spike_field.py -v`

**Verification**: Progress bar appears for long operations, gracefully disabled without tqdm

---

### 1.7 UX: Units Validation [CRITICAL]

**Ref**: PLAN.md §1.7 | **Effort**: 4h | **Files**: `environment/factories.py`, `environment/core.py`

- [ ] Add `units: str | None = None` parameter to `from_samples()`
- [ ] Add units warning in `from_samples()` if None
- [ ] Add `units` parameter to `from_polygon()`
- [ ] Add `units` parameter to `from_graph()`
- [ ] Add `units` parameter to `from_mask()`
- [ ] Add `units` parameter to `from_image()`
- [ ] Add `units` parameter to `from_layout()`
- [ ] Update all factory method docstrings with units parameter
- [ ] Update all examples in docstrings to show `units='cm'`
- [ ] Add test `test_from_samples_warns_without_units()` in `tests/test_environment.py`
- [ ] Add test `test_from_samples_sets_units_when_provided()`
- [ ] Update CLAUDE.md Quick Reference examples with units
- [ ] Run: `uv run pytest tests/test_environment.py -k units -v`

**Verification**: Warning appears when units not provided, examples show units

---

### 1.8 API: Import Consistency [CRITICAL]

**Ref**: PLAN.md §1.8 | **Effort**: 2h | **Files**: `src/neurospatial/__init__.py`

- [x] Add imports: `from neurospatial.io import from_dict, from_file, to_dict, to_file`
- [x] Add imports: `from neurospatial.regions import Region, Regions`
- [x] Add imports: `from neurospatial.spatial import clear_kdtree_cache`
- [x] Update `__all__` list (alphabetically within groups)
- [x] Verify against CLAUDE.md: all documented imports work
- [x] Run: `uv run python -c "from neurospatial import to_file, Region, clear_kdtree_cache"`
- [x] Run all tests: `uv run pytest tests/test_api.py -v`

**Verification**: All imports in CLAUDE.md §13 (Import Patterns) work at top level
**Status**: ✅ COMPLETED (2025-11-14) - Commit: 56286f9

---

### Milestone 1 Completion Checklist

- [ ] All critical tasks completed (1.1-1.8)
- [ ] Full test suite passes: `uv run pytest`
- [ ] No new mypy errors: `uv run mypy src/neurospatial/`
- [ ] No new ruff errors: `uv run ruff check .`
- [ ] Security audit clean (no path traversal vulnerabilities)
- [ ] Documentation updated (CLAUDE.md, docstrings)
- [ ] Git commit: `git commit -m "fix: critical security and stability issues (M1)"`
- [ ] Create release branch: `git checkout -b release/v1.0-alpha`

---

## Milestone 2: Code Quality & Performance (v1.0-beta)

**Target**: Week 3-5 | **Estimated**: 45 hours | **Status**: Not Started

**Goal**: Improve code quality, reduce duplication, add performance tests, improve API consistency.

### 2.1 Refactor: Graph Connectivity Helper [HIGH]

**Ref**: PLAN.md §2.1 | **Effort**: 8h | **Files**: `layout/helpers/`

- [ ] Create `src/neurospatial/layout/helpers/graph_building.py` (new file)
- [ ] Implement `_create_connectivity_graph_generic()` function
- [ ] Add comprehensive docstring with NumPy format
- [ ] Refactor `regular_grid.py` to use new helper
- [ ] Refactor `hexagonal.py` to use new helper
- [ ] Add unit tests for `_create_connectivity_graph_generic()`
- [ ] Verify no performance regression: run benchmarks
- [ ] Run layout tests: `uv run pytest tests/layout/ -v`

**Acceptance**: Code duplication reduced from ~300 lines to <50 lines

---

### 2.2 Optimize: region_membership() Performance [HIGH]

**Ref**: PLAN.md §2.2 | **Effort**: 3h | **Files**: `environment/regions.py`

- [ ] Hoist `shapely_points()` call outside loop in `region_membership()`
- [ ] Add benchmark test `test_region_membership_performance()` in `tests/test_performance.py`
- [ ] Measure speedup (expect ~90% for 10+ regions)
- [ ] Verify correctness: `uv run pytest tests/regions/ -v`

**Verification**: Benchmark shows >5x speedup for 10 regions

---

### 2.3 Testing: Performance Regression Suite [HIGH]

**Ref**: PLAN.md §2.3 | **Effort**: 6h | **Files**: `tests/test_performance.py` (new)

- [ ] Create `tests/test_performance.py`
- [ ] Add pytest marker `@pytest.mark.slow` to all performance tests
- [ ] Add test `test_large_environment_creation_time()` (1M points)
- [ ] Add test `test_kdtree_batch_query_performance()` (10k queries)
- [ ] Add test `test_shortest_path_large_graph()`
- [ ] Add test `test_occupancy_large_trajectory()`
- [ ] Add to pytest.ini: `markers = slow: marks tests as slow`
- [ ] Document how to run: `uv run pytest -m slow`
- [ ] Create CI performance tracking (optional)

**Acceptance**: Performance tests exist and pass with reasonable thresholds

---

### 2.4 Testing: Property-Based Tests [HIGH]

**Ref**: PLAN.md §2.4 | **Effort**: 8h | **Files**: `tests/test_properties.py` (new)

- [ ] Add `hypothesis>=6.92.0` to `[project.optional-dependencies.test]`
- [ ] Create `tests/test_properties.py`
- [ ] Add test `test_bin_centers_within_data_range()` (hypothesis)
- [ ] Add test `test_rotation_composition_property()`
- [ ] Add test `test_distance_triangle_inequality()`
- [ ] Add test `test_straightness_bounds()` (0 ≤ straightness ≤ 1)
- [ ] Add test `test_normalized_field_sums_to_one()`
- [ ] Configure hypothesis settings (max_examples, deadline)
- [ ] Run: `uv run pytest tests/test_properties.py -v`

**Acceptance**: ≥5 property-based tests covering key mathematical invariants

---

### 2.5 UX: Standardize Parameter Naming [HIGH]

**Ref**: PLAN.md §2.5 | **Effort**: 6h | **Files**: Multiple

- [ ] Add `positions` parameter to `from_samples()` (preferred name)
- [ ] Add `data_samples` as deprecated alias with warning
- [ ] Update all `occupancy()` calls to use `positions`
- [ ] Update all `compute_place_field()` calls to use `positions`
- [ ] Update all docstrings to use `positions`
- [ ] Update all examples to use `positions`
- [ ] Add test `test_data_samples_deprecated_warning()`
- [ ] Add test `test_positions_parameter_works()`
- [ ] Update CLAUDE.md to use `positions` throughout
- [ ] Run: `uv run pytest -W error::DeprecationWarning` (should fail on old usage)

**Acceptance**: All code uses `positions`, deprecation warning on `data_samples`

---

### 2.6 Documentation: Error Code System [HIGH]

**Ref**: PLAN.md §2.6 | **Effort**: 4h | **Files**: Multiple, `docs/errors.md` (new)

- [ ] Create `docs/errors.md` with error reference structure
- [ ] Define error code constants (E1001, E1002, etc.) in relevant modules
- [ ] Add E1001 to "No active bins" error in `regular_grid.py`
- [ ] Add E1002 to bin_size validation errors
- [ ] Add E1003 to dimension mismatch errors
- [ ] Add documentation link to error messages
- [ ] Document 5 most common errors in `docs/errors.md`
- [ ] Add test verifying error codes appear in messages

**Acceptance**: Top 5 errors have codes and documented solutions

---

### 2.7 Refactor: Split Long Methods [MEDIUM]

**Ref**: PLAN.md §2.7 | **Effort**: 4h | **Files**: `layout/helpers/regular_grid.py`

- [ ] Extract `_validate_grid_inputs()` from `_create_regular_grid()`
- [ ] Extract `_compute_dimension_ranges()` from `_create_regular_grid()`
- [ ] Extract `_compute_grid_structure()` from `_create_regular_grid()`
- [ ] Extract `_generate_bin_centers()` from `_create_regular_grid()`
- [ ] Update `_create_regular_grid()` to orchestrate helper functions
- [ ] Verify `_create_regular_grid()` is now <50 lines
- [ ] Run: `uv run pytest tests/layout/test_regular_grid_utils.py -v`

**Acceptance**: Main function <50 lines, cyclomatic complexity <10

---

### 2.8 Type Safety: Fix SubsetLayout [MEDIUM]

**Ref**: PLAN.md §2.8 | **Effort**: 2h | **Files**: `environment/transforms.py`

- [ ] Remove `self: SelfEnv` from `SubsetLayout.__init__()`
- [ ] Add proper type hints to all SubsetLayout methods
- [ ] Run mypy: `uv run mypy src/neurospatial/environment/transforms.py`
- [ ] Verify no type errors
- [ ] Run tests: `uv run pytest tests/test_transforms.py -v`

**Verification**: Mypy passes with no errors on transforms.py

---

### 2.9 Documentation: API Overview Docstring [MEDIUM]

**Ref**: PLAN.md §2.9 | **Effort**: 2h | **Files**: `src/neurospatial/__init__.py`

- [ ] Add comprehensive module-level docstring to `__init__.py`
- [ ] Document core classes (Environment, Region, CompositeEnvironment)
- [ ] Document key functions by category (spatial, trajectory, fields, etc.)
- [ ] Add import patterns section
- [ ] Add "See Also" references
- [ ] Verify formatting: `uv run python -c "import neurospatial; help(neurospatial)"`

**Acceptance**: `help(neurospatial)` shows comprehensive API overview

---

### 2.10 Consistency: Scale Parameter Naming [LOW]

**Ref**: PLAN.md §2.10 | **Effort**: 2h | **Files**: `transforms.py`, `alignment.py`

- [ ] Audit all uses of `scale`, `scale_factor`, `sx`/`sy`
- [ ] Standardize on `scale` for uniform scaling
- [ ] Standardize on `sx`, `sy`, `sz` for per-axis scaling
- [ ] Update docstrings
- [ ] Add deprecation aliases if needed
- [ ] Run: `uv run pytest tests/test_transforms.py tests/test_alignment.py -v`

**Acceptance**: Consistent naming across all transform functions

---

### 2.11 Memory: Cache Management [MEDIUM]

**Ref**: PLAN.md §2.11 | **Effort**: 3h | **Files**: `environment/core.py`, `spatial.py`

- [ ] Add `clear_cache()` instance method to Environment
- [ ] Add memory usage warning when cache exceeds threshold
- [ ] Update `clear_kdtree_cache()` to work with new system
- [ ] Add docstrings explaining cache behavior
- [ ] Add test `test_clear_cache_method()`
- [ ] Run: `uv run pytest tests/test_environment.py -k cache -v`

**Acceptance**: `env.clear_cache()` clears all caches, warnings appear

---

### 2.12 Cleanup: Remove Dead Code [LOW]

**Ref**: PLAN.md §2.12 | **Effort**: 1h | **Files**: `layout/helpers/utils.py`

- [ ] Remove commented code at lines 629-642 in `utils.py`
- [ ] Check for other commented code blocks
- [ ] Remove any other dead code found
- [ ] Run: `uv run pytest tests/layout/ -v`

**Verification**: No commented-out code blocks in production files

---

### Milestone 2 Completion Checklist

- [ ] All high priority tasks completed (2.1-2.12)
- [ ] Full test suite passes: `uv run pytest`
- [ ] Performance tests pass: `uv run pytest -m slow`
- [ ] Property tests pass: `uv run pytest tests/test_properties.py`
- [ ] Code duplication <10%
- [ ] Mypy clean: `uv run mypy src/neurospatial/`
- [ ] Documentation complete
- [ ] Git commit: `git commit -m "refactor: code quality improvements (M2)"`
- [ ] Tag release: `git tag v1.0-beta`

---

## Milestone 3: Features & UX Polish (v1.0-rc)

**Target**: Week 6-10 | **Estimated**: 40 hours | **Status**: Not Started

**Goal**: Add quality-of-life features, workflow helpers, improve documentation.

### 3.1 Feature: env.info() Method [MEDIUM]

**Ref**: PLAN.md §3.1 | **Effort**: 3h

- [ ] Add `info()` method to `environment/core.py` or create `environment/info.py` mixin
- [ ] Implement formatted output with tree structure
- [ ] Include: bins, dims, ranges, units, layout, regions
- [ ] Add test `test_info_method_output()`
- [ ] Update docstring with example output
- [ ] Run: `uv run python -c "from neurospatial import Environment; ..."`

**Acceptance**: `env.info()` returns human-readable summary

---

### 3.2 Feature: Workflow Helpers [MEDIUM]

**Ref**: PLAN.md §3.2 | **Effort**: 6h

- [ ] Create `src/neurospatial/workflows.py`
- [ ] Implement `quick_place_field_analysis()`
- [ ] Implement `analyze_trajectory()`
- [ ] Implement `compare_sessions()`
- [ ] Add comprehensive docstrings with examples
- [ ] Add unit tests for each workflow
- [ ] Update `__init__.py` to export workflow functions
- [ ] Run: `uv run pytest tests/test_workflows.py -v`

**Acceptance**: 3 workflow functions with examples

---

### 3.3-3.10 Additional Medium Priority

**See PLAN.md §3.3-§3.10 for details**

- [ ] 3.3: Transform visualization helper
- [ ] 3.4: Multi-session alignment
- [ ] 3.5: Builder pattern exploration
- [ ] 3.6: Integration/workflow tests
- [ ] 3.7: Visual regression tests
- [ ] 3.8: Mathematical formulations in docstrings
- [ ] 3.9: Distance field caching
- [ ] 3.10: GraphLayout decoupling

---

### Milestone 3 Completion Checklist

- [ ] Features implemented and tested
- [ ] User workflows streamlined
- [ ] Documentation comprehensive
- [ ] Integration tests passing
- [ ] Git commit: `git commit -m "feat: UX improvements and workflows (M3)"`
- [ ] Tag release: `git tag v1.0-rc1`

---

## Milestone 4: Production Readiness (v1.0)

**Target**: Week 11-12 | **Estimated**: 20 hours | **Status**: Not Started

**Goal**: Final polish, documentation review, performance verification.

### 4.1 Final Review Tasks

- [ ] Run full test suite with coverage: `uv run pytest --cov=src/neurospatial --cov-report=html`
- [ ] Review coverage report: `open htmlcov/index.html`
- [ ] Ensure coverage >85%
- [ ] Run performance benchmarks and verify within 10% of baseline
- [ ] Review all error messages for clarity
- [ ] Review all docstrings for completeness
- [ ] Update CHANGELOG.md with all changes
- [ ] Update README.md with new features
- [ ] Run linters: `uv run ruff check . && uv run ruff format .`
- [ ] Run type checker: `uv run mypy src/neurospatial/`
- [ ] Build documentation: `uv run mkdocs build` (if applicable)

---

### 4.2 Release Preparation

- [ ] Update version in `pyproject.toml` to `1.0.0`
- [ ] Create release notes in `RELEASE_NOTES.md`
- [ ] Tag final release: `git tag v1.0.0`
- [ ] Push tags: `git push origin --tags`
- [ ] Create GitHub release with notes
- [ ] Announce release (if applicable)

---

## Future Enhancements (Backlog)

**See PLAN.md Phase 4 for details**

These are tracked but not scheduled:

- [ ] Add `__version__` attribute
- [ ] Schema migration framework
- [ ] Atomic file writes
- [ ] Replace magic numbers with constants
- [ ] Fuzzing tests
- [ ] Mutation testing (mutmut)
- [ ] Contract tests for LayoutEngine
- [ ] Parallel test execution
- [ ] Benchmark suite
- [ ] Fixture data extraction

---

## Quick Commands Reference

```bash
# Run all tests
uv run pytest

# Run specific milestone tests
uv run pytest -k "test_to_file_rejects_path_traversal or test_occupancy_ray_parallel"

# Run with coverage
uv run pytest --cov=src/neurospatial --cov-report=term-missing

# Run slow tests
uv run pytest -m slow

# Run type checking
uv run mypy src/neurospatial/

# Run linting
uv run ruff check .

# Format code
uv run ruff format .

# Run security checks
uv run bandit -r src/neurospatial/

# Check for updates
git status
git diff TASKS.md
```

---

## Progress Tracking

**Milestone 1**: ☐ 0/8 tasks (0%)
**Milestone 2**: ☐ 0/12 tasks (0%)
**Milestone 3**: ☐ 0/10 tasks (0%)
**Milestone 4**: ☐ 0/2 tasks (0%)

**Overall**: ☐ 0/32 major tasks (0%)

---

## Notes & Decisions

### Open Questions (from PLAN.md)

1. **Units**: Require (error) or optional (warning)? → **Decision**: Warning for now
2. **Progress bars**: Default on or off? → **Decision**: Default on (can disable)
3. **Deprecation timeline**: data_samples → positions? → **Decision**: 2 minor versions
4. **Python version**: Minimum supported? → **Decision**: 3.13+ (current)

### Blockers

- None currently

### Completed Milestones

- None yet

---

**Last Updated**: 2025-11-14
**Next Review**: After Milestone 1 completion
