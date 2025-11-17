# neurospatial v0.3.0 API Cleanup Tasks

**Breaking release** - Major API renaming for clarity and consistency.

**Target**: v0.3.0 release with clean, intuitive API
**No backwards compatibility** - Clean break from v0.2.x

## Summary of Changes

This plan implements a systematic API cleanup to make neurospatial more intuitive and consistent:

**Core Principles**:

- **Methods = Primitives**: Environment methods answer questions about that environment or perform local transforms
- **Free Functions = Analysis**: Module-level functions perform higher-level analysis (neural metrics, behavioral segmentation, alignment)
- **Short, noun-oriented names**: Remove redundant `compute_` prefixes, use descriptive nouns/verbs

**Key Improvements**:

1. **Add missing primitive methods**: `clear_cache()`, `region_mask()`, `apply_transform()` - Currently only available as awkward free functions
2. **Fix confusing names**: `shortest_path()` → `path_between()` (returns path, not distance), `map_probabilities_to_nearest_target_bin` → `map_probabilities()` (39 chars → 18 chars)
3. **Systematic cleanup**: Remove generic `compute_` prefix from `compute_place_field()`, `compute_diffusion_kernels()`, and metrics functions
4. **Type-aligned signatures**: Use proper `AffineND/Affine2D` types, expand `region_mask()` to match `regions_to_mask()` flexibility

**API Design Refinements** (incorporated from review):

- `clear_cache()` is canonical API; `clear_kdtree_cache()` removed from public surface
- `region_mask()` accepts single/multiple regions and `include_boundary` parameter for full feature parity with `regions_to_mask()`
- `apply_transform()` uses `AffineND/Affine2D` transform objects (not raw matrices) to align with existing transform API
- Added systematic `compute_*` prefix removal in metrics module for consistency

---

## Milestone 1: Add Missing Environment Methods

Core primitive methods that should exist on Environment but are currently only available as free functions.

### 1.1 Add `env.clear_cache()` method ⚡ CRITICAL ✅ COMPLETE

**Priority**: P0 - Currently documented in CLAUDE.md but doesn't exist

**Files to modify**:

- [x] `src/neurospatial/environment/core.py` or appropriate mixin
  - Add `clear_cache(self, *, kdtree: bool = True, kernels: bool = True, cached_properties: bool = True) -> None` method
  - Clear KDTree cache entries for this environment
  - Clear any memoized kernels/computations
  - Add NumPy-style docstring with Examples
- [x] `src/neurospatial/spatial.py`
  - **BREAKING**: Remove `clear_kdtree_cache()` from public API (remove from `__all__`)
  - Keep as private implementation detail if needed by `env.clear_cache()`
  - Or make it a simple wrapper: `def clear_kdtree_cache(env): env.clear_cache(kdtree=True)`
- [x] Update `__all__` in `src/neurospatial/__init__.py`
  - Remove `clear_kdtree_cache` if exported

**Tests**:

- [x] `tests/test_spatial.py` or new `tests/environment/test_cache.py`
  - Test `env.clear_cache()` removes cached data
  - Test caching behavior before/after clear
  - Test clearing one env doesn't affect another
  - Test selective clearing: `env.clear_cache(kdtree=True, kernels=False)`
- [x] Update any tests using `clear_kdtree_cache()` to use `env.clear_cache()`

**Documentation**:

- [x] Update CLAUDE.md examples to use `env.clear_cache()`
- [x] Add docstring example showing when to clear cache
- [x] Note that `clear_kdtree_cache()` is removed in v0.3.0

**Verification**:

```bash
uv run python -c "from neurospatial import Environment; import numpy as np; env = Environment.from_samples(np.random.rand(100, 2), bin_size=1.0); env.clear_cache()"
# Should NOT be importable:
uv run python -c "from neurospatial import clear_kdtree_cache" 2>&1 | grep "ImportError"
```

---

### 1.2 Add `env.region_mask()` method ✅ COMPLETE

**Priority**: P1 - Convenience method, improves discoverability

**Files to modify**:

- [x] `src/neurospatial/environment/regions.py` (EnvironmentRegions mixin)
  - Add `region_mask(self, regions: str | list[str] | Region | Regions, *, include_boundary: bool = True) -> NDArray[np.bool_]` method
  - Accepts:
    - Single region name (str)
    - List of region names (list[str]) - returns union mask
    - Single Region object
    - Regions container - returns union mask
  - Returns boolean mask of shape (n_bins,)
  - `include_boundary` parameter matches `regions_to_mask` behavior
  - Add NumPy-style docstring with multiple examples
- [x] Update `__all__` if needed (not needed - method on Environment)

**Tests**:

- [x] `tests/environment/test_region_mask_method.py` (new file)
  - Test `env.region_mask('goal')` with single region name
  - Test `env.region_mask(['left', 'right'])` with multiple region names
  - Test `env.region_mask(env.regions['goal'])` with Region object
  - Test `env.region_mask(env.regions)` with Regions container
  - Test `include_boundary=False` parameter
  - Test error on nonexistent region name
  - Test with point regions and polygon regions
  - Test feature parity with `regions_to_mask()` function (5 tests)

**Documentation**:

- [x] Add comprehensive examples to Environment.region_mask() docstring
- [x] Show single region, multiple regions, and boundary parameter usage
- [x] Note relationship to `regions_to_mask()` free function

**Verification**:

```python
env = Environment.from_samples(positions, bin_size=2.0)
env.regions.add('goal', point=[50, 50])
env.regions.add('start', point=[10, 10])

# Single region
mask = env.region_mask('goal')
assert mask.shape == (env.n_bins,)
assert mask.dtype == bool

# Multiple regions (union)
mask = env.region_mask(['goal', 'start'])
assert mask.sum() >= 2  # At least the two bins

# All regions
mask = env.region_mask(env.regions)
```

**Results**:

- ✅ All 21 tests pass
- ✅ Feature parity confirmed with `regions_to_mask()`
- ✅ Ruff passes
- ✅ Mypy passes
- ✅ Code reviewer: APPROVED
- ✅ Fixed pre-existing test pollution bug in test_info.py

---

### 1.3 Add `env.apply_transform()` method ✅ COMPLETE

**Priority**: P1 - Method is more discoverable than free function

**Files to modify**:

- [x] `src/neurospatial/environment/transforms.py` (EnvironmentTransforms mixin)
  - Add `apply_transform(self, transform: AffineND | Affine2D, *, name: str | None = None) -> Environment` method
  - Accepts `AffineND` or `Affine2D` transform objects (matches existing API)
  - Returns new Environment (functional, not in-place)
  - Optional `name` parameter to rename the transformed environment
  - Add NumPy-style docstring with transform object examples
  - Import types: `from neurospatial.transforms import AffineND, Affine2D`
- [x] Keep `apply_transform_to_environment()` in public API
  - Method delegates to free function (clean delegation pattern)
  - Both APIs available for flexibility

**Tests**:

- [x] `tests/environment/test_transforms.py` (19 comprehensive tests)
  - Test identity, translation, rotation, scaling (2D)
  - Test transform composition
  - Test AffineND for 2D and 3D environments
  - Test dimension mismatch error handling
  - Test connectivity and edge attribute preservation
  - Test functional (non-mutating) behavior
  - Test custom vs default naming
  - Test units and frame metadata preservation
  - Test region transformation (point and polygon)
  - Test unfitted environment error

**Documentation**:

- [x] Add comprehensive NumPy-style docstring with 7 examples:
  - Translation (2D)
  - Rotation (2D)
  - Scaling and composition
  - Cross-session alignment with landmarks
  - 3D transformation
  - Regions transformation

**Verification**:

```bash
# All 19 tests pass
uv run pytest tests/environment/test_transforms.py::TestApplyTransform -v
# Ruff and mypy pass
uv run ruff check src/neurospatial/environment/transforms.py
uv run mypy src/neurospatial/environment/transforms.py
```

**Results**:

- ✅ All 19 tests pass
- ✅ Ruff passes
- ✅ Mypy passes
- ✅ Code reviewer: APPROVED (production-ready)
- ✅ Follows project patterns (matches M1.1 and M1.2)
- ✅ Most comprehensive docstring among all three milestones (167 lines, 7 examples)

---

## Milestone 2: Rename Environment Methods

Breaking changes to method names for clarity.

### 2.1 Rename `shortest_path()` → `path_between()` ⚡ CRITICAL ✅ COMPLETE

**Priority**: P0 - Current name suggests distance, not path sequence

**Files to modify**:

- [x] `src/neurospatial/environment/queries.py` (EnvironmentQueries mixin)
  - Rename method: `shortest_path()` → `path_between()`
  - Update docstring references
- [x] Search codebase for all calls to `shortest_path()`
  - Update internal calls
  - Update examples in docstrings
- [x] Update `__all__` exports if method is exported

**Tests**:

- [x] `tests/environment/test_queries.py` or similar
  - Rename all test functions: `test_shortest_path*` → `test_path_between*`
  - Update all `env.shortest_path()` calls to `env.path_between()`

**Documentation**:

- [x] CLAUDE.md: Update all references
- [x] README.md: Update examples if present
- [x] Example notebooks: Search and replace

**Verification**:

```bash
# Ensure old name doesn't exist
uv run python -c "from neurospatial import Environment; import numpy as np; env = Environment.from_samples(np.random.rand(100, 2), bin_size=1.0); assert not hasattr(env, 'shortest_path')"
# Ensure new name works
uv run python -c "from neurospatial import Environment; import numpy as np; env = Environment.from_samples(np.random.rand(100, 2), bin_size=1.0); path = env.path_between(0, 10)"
```

---

### 2.2 Rename `compute_kernel()` → `diffusion_kernel()` ⏭️ SKIPPED

**Priority**: P1 - More specific name, clearer algorithm
**Status**: SKIPPED (per user request)

**Files to modify**:

- [ ] `src/neurospatial/environment/fields.py` (EnvironmentFields mixin)
  - Rename method: `compute_kernel()` → `diffusion_kernel()`
  - Update all docstring references
  - Update internal calls in `smooth()` method
- [ ] Search codebase for calls to `compute_kernel()`

**Tests**:

- [ ] `tests/environment/test_fields.py`
  - Rename tests: `test_compute_kernel*` → `test_diffusion_kernel*`
  - Update all method calls

**Documentation**:

- [ ] Update CLAUDE.md
- [ ] Update docstring examples

**Verification**:

```python
env = Environment.from_samples(np.random.rand(100, 2) * 100, bin_size=5.0)
kernel = env.diffusion_kernel(bandwidth=10.0)
assert kernel.shape == (env.n_bins, env.n_bins)
```

---

## Milestone 3: Rename Free Functions

Breaking changes to module-level function names.

### 3.1 Rename `compute_place_field()` → `place_field()` ⚡ CRITICAL

**Priority**: P0 - High visibility function, remove redundant prefix

**Files to modify**:

- [ ] `src/neurospatial/spike_field.py`
  - Rename function: `compute_place_field()` → `place_field()`
  - Update all docstring references
- [ ] `src/neurospatial/__init__.py`
  - Update import: `from neurospatial.spike_field import place_field`
  - Update `__all__` export
- [ ] Search all files for `compute_place_field` references

**Tests**:

- [ ] `tests/test_spike_field.py`
  - Rename tests: `test_compute_place_field*` → `test_place_field*`
  - Update all function calls
  - Update imports

**Documentation**:

- [ ] CLAUDE.md: Update all examples (multiple locations)
- [ ] `src/neurospatial/__init__.py` module docstring
- [ ] README.md if present

**Notebooks**:

- [ ] Use `jupyter-notebook-editor` skill to update all notebooks
  - Search for `compute_place_field` calls in notebooks
  - Replace with `place_field` using the skill's systematic approach
  - Update both code cells and markdown documentation
- [ ] Verify notebooks are valid JSON after updates:

  ```bash
  for nb in *.ipynb; do python3 -c "import json; json.load(open('$nb'))"; done
  ```

**Verification**:

```bash
uv run python -c "from neurospatial import place_field; print(place_field.__name__)"
# Should print: place_field
```

---

### 3.2 Rename `map_probabilities_to_nearest_target_bin` → `map_probabilities()` ⚡ CRITICAL ✅ COMPLETE

**Priority**: P0 - 39 character name is unusable

**Files to modify**:

- [x] `src/neurospatial/alignment.py`
  - Rename function
  - Update docstring
- [x] `src/neurospatial/__init__.py`
  - Update import and `__all__`
- [x] Search for all uses of old name

**Tests**:

- [x] `tests/test_alignment.py`
  - Rename tests
  - Update all calls
  - Update imports

**Documentation**:

- [x] CLAUDE.md
- [x] Alignment module docstring

**Notebooks**:

- [x] Use `jupyter-notebook-editor` skill to update all notebooks
  - Search for `map_probabilities_to_nearest_target_bin` calls in notebooks
  - Replace with `map_probabilities` using the skill's systematic approach
  - Update both code cells and markdown documentation
- [x] Verify notebooks are valid JSON after updates

**Verification**:

```bash
uv run python -c "from neurospatial import map_probabilities; print(map_probabilities.__name__)"
```

---

### 3.3 Rename `compute_diffusion_kernels()` → `diffusion_kernels()`

**Priority**: P1 - Systematic prefix removal

**Files to modify**:

- [ ] `src/neurospatial/kernels.py`
  - Rename function
  - Update docstring
- [ ] `src/neurospatial/__init__.py`
  - Update import and `__all__`

**Tests**:

- [ ] `tests/test_kernels.py`
  - Update function calls
  - Rename tests

**Documentation**:

- [ ] Update CLAUDE.md if mentioned

**Notebooks**:

- [ ] Use `jupyter-notebook-editor` skill to update all notebooks
  - Search for `compute_diffusion_kernels` calls in notebooks
  - Replace with `diffusion_kernels` using the skill's systematic approach
  - Update both code cells and markdown documentation
- [ ] Verify notebooks are valid JSON after updates

**Verification**:

```bash
uv run python -c "from neurospatial import diffusion_kernels; print(diffusion_kernels.__name__)"
```

---

### 3.4 Systematic `compute_*` prefix removal in metrics

**Priority**: P1 - Consistency with other renames

**Rationale**: Same as `compute_place_field` → `place_field` - generic `compute_` prefix adds no information.

**Files to modify**:

- [ ] Search `src/neurospatial/` for additional `compute_*` functions
- [ ] Likely candidates based on naming pattern:
  - `metrics.py`: `compute_home_range()` → `home_range()`
  - `metrics.py`: `compute_step_lengths()` → `step_lengths()`
  - `metrics.py`: `compute_turn_angles()` → `turn_angles()`
  - `metrics.py`: `compute_field_emd()` → `field_emd()`
  - Any other `compute_*` functions in behavioral/metrics modules
- [ ] Update `__all__` exports in affected modules
- [ ] Update `src/neurospatial/__init__.py` if these are re-exported

**Tests**:

- [ ] `tests/test_metrics.py` or equivalent
  - Rename test functions to match new names
  - Update all function calls
  - Update imports

**Documentation**:

- [ ] Update docstrings
- [ ] Update CLAUDE.md if these functions are mentioned
- [ ] Update examples

**Search for candidates**:

```bash
# Find all compute_* functions
rg "^def compute_" src/neurospatial/
rg "^    compute_.*=" src/neurospatial/  # Assignments too
```

**Notebooks**:

- [ ] Use `jupyter-notebook-editor` skill to update all notebooks
  - Search for any `compute_*` metric function calls in notebooks (e.g., `compute_home_range`, `compute_step_lengths`, etc.)
  - Replace with shortened names using the skill's systematic approach
  - Update both code cells and markdown documentation
- [ ] Verify notebooks are valid JSON after updates

**Verification**:

```bash
# Verify new names are importable (example)
uv run python -c "from neurospatial import home_range, step_lengths, turn_angles, field_emd"
```

---

## Milestone 4: Testing and Documentation

Comprehensive verification and documentation updates.

### 4.1 Run full test suite and verify API surface

- [ ] Run all tests: `uv run pytest`
- [ ] Run with coverage: `uv run pytest --cov=src/neurospatial`
- [ ] Ensure 100% of renamed functions tested
- [ ] Check for any remaining references to old names

**API Surface Testing** (add to test suite):

- [ ] Create or update `tests/test_api.py`
  - Assert new names are in public API
  - Assert old names are NOT in public API
  - Example:

    ```python
    import neurospatial

    # New names should be present
    assert hasattr(neurospatial, 'place_field')
    assert hasattr(neurospatial, 'map_probabilities')
    assert hasattr(neurospatial, 'diffusion_kernels')

    # Old names should NOT be present
    assert not hasattr(neurospatial, 'compute_place_field')
    assert not hasattr(neurospatial, 'map_probabilities_to_nearest_target_bin')
    assert not hasattr(neurospatial, 'clear_kdtree_cache')

    # Environment methods
    from neurospatial import Environment
    import numpy as np
    env = Environment.from_samples(np.random.rand(100, 2), bin_size=1.0)

    # New methods should exist
    assert hasattr(env, 'path_between')
    assert hasattr(env, 'diffusion_kernel')
    assert hasattr(env, 'clear_cache')
    assert hasattr(env, 'region_mask')
    assert hasattr(env, 'apply_transform')

    # Old methods should NOT exist
    assert not hasattr(env, 'shortest_path')
    assert not hasattr(env, 'compute_kernel')
    ```

- [ ] Create or update `tests/test_import_paths.py`
  - Test canonical import paths work
  - Example:

    ```python
    # Canonical imports should work
    from neurospatial import (
        Environment,
        place_field,
        map_probabilities,
        diffusion_kernels,
    )
    from neurospatial.transforms import Affine2D
    ```

**Search for old names**:

```bash
# Check source code (these should return NO matches)
rg "def shortest_path" src/
rg "def compute_place_field" src/
rg "def compute_kernel" src/
rg "def compute_diffusion_kernels" src/
rg "map_probabilities_to_nearest_target_bin" src/
rg "clear_kdtree_cache" src/neurospatial/__init__.py

# Check tests (should only be in test names, not function calls)
rg "shortest_path\(" tests/
rg "compute_place_field\(" tests/
rg "compute_kernel\(" tests/
rg "compute_diffusion_kernels\(" tests/

# Check docs (should return NO matches)
rg "shortest_path\(" docs/
rg "compute_place_field\(" docs/
rg "clear_kdtree_cache\(" docs/
```

---

### 4.2 Update CLAUDE.md comprehensively

- [ ] Update "Quick Reference" section (lines 13-57)
  - Replace all old function/method names
  - Update examples with new names
- [ ] Update "Import Patterns" section (lines 259-297)
  - Update all imports
  - Verify `__all__` matches actual exports
- [ ] Update "Common Gotchas" section (lines 721-804)
  - Search for old names
  - Update examples
- [ ] Update troubleshooting examples

**Verification**:

```bash
# No old names should remain
rg "shortest_path" CLAUDE.md
rg "compute_place_field" CLAUDE.md
rg "compute_kernel" CLAUDE.md
rg "map_probabilities_to_nearest_target_bin" CLAUDE.md
```

---

### 4.3 Update package metadata and exports

- [ ] `src/neurospatial/__init__.py`
  - Verify all renamed functions in `__all__`
  - Update module docstring (lines 1-187)
  - Update function categorization comments
- [ ] Verify public API with:

  ```python
  import neurospatial
  print(dir(neurospatial))
  # Should show: place_field, map_probabilities, diffusion_kernels
  # Should NOT show: compute_place_field, compute_diffusion_kernels
  ```

---

### 4.4 Update examples and notebooks

- [ ] Search `docs/examples/` for old function names
- [ ] Update any example notebooks (.ipynb or .py if jupytext)
- [ ] Regenerate example outputs if needed
- [ ] Test all examples run without errors

---

## Milestone 5: Version Bump and Release

Final steps for v0.3.0 release.

### 5.1 Update version and changelog

- [ ] `pyproject.toml`
  - Update `version = "0.3.0"` (line 3)
- [ ] Create `CHANGELOG.md` if not exists, or update existing
  - Document breaking changes
  - List all renamed methods/functions
  - Provide migration examples
- [ ] Update README.md version references

**CHANGELOG.md template**:

```markdown
# Changelog

## [0.3.0] - 2025-MM-DD

### Breaking Changes

**Environment Methods**:
- Renamed `shortest_path()` → `path_between()` - Returns path sequence, not distance
- Renamed `compute_kernel()` → `diffusion_kernel()` - More specific name
- Added `clear_cache()` - Clear environment-specific cached data
- Added `region_mask()` - Get boolean mask for region
- Added `apply_transform()` - Apply affine transform to environment

**Free Functions**:
- Renamed `compute_place_field()` → `place_field()` - Remove redundant prefix
- Renamed `map_probabilities_to_nearest_target_bin` → `map_probabilities()` - Shorter name
- Renamed `compute_diffusion_kernels()` → `diffusion_kernels()` - Remove redundant prefix

**Migration Guide**:

| Old Name | New Name | Notes |
|----------|----------|-------|
| `env.shortest_path(a, b)` | `env.path_between(a, b)` | Returns bin sequence |
| `env.compute_kernel(bw)` | `env.diffusion_kernel(bw)` | Graph diffusion kernel |
| `compute_place_field(...)` | `place_field(...)` | Import: `from neurospatial import place_field` |
| `map_probabilities_to_nearest_target_bin(...)` | `map_probabilities(...)` | Import: `from neurospatial import map_probabilities` |
| `compute_diffusion_kernels(...)` | `diffusion_kernels(...)` | Import: `from neurospatial import diffusion_kernels` |
| N/A | `env.clear_cache()` | New method for clearing cached data |
| N/A | `env.region_mask('name')` | New method for region boolean masks |
| N/A | `env.apply_transform(T)` | New method for applying transforms |
```

---

### 5.2 Run final verification

- [ ] Full test suite passes: `uv run pytest`
- [ ] Type checking passes: `uv run mypy src/neurospatial/`
- [ ] Linting passes: `uv run ruff check .`
- [ ] Formatting correct: `uv run ruff format --check .`
- [ ] Documentation builds (if using mkdocs)
- [ ] No old function names in codebase

**Final grep verification**:

```bash
# Should return NO matches in src/
rg "def shortest_path" src/
rg "def compute_place_field" src/
rg "def compute_kernel" src/
rg "def compute_diffusion_kernels" src/
rg "map_probabilities_to_nearest_target_bin" src/
```

---

### 5.3 Create release commit and tag

- [ ] Commit all changes with conventional commit message:

  ```bash
  git add .
  git commit -m "feat!: v0.3.0 breaking API cleanup

  BREAKING CHANGE: Major API renaming for clarity and consistency

  Environment Methods:
  - shortest_path() → path_between()
  - compute_kernel() → diffusion_kernel()
  - Added clear_cache(), region_mask(), apply_transform()

  Free Functions:
  - compute_place_field() → place_field()
  - map_probabilities_to_nearest_target_bin → map_probabilities()
  - compute_diffusion_kernels() → diffusion_kernels()

  See CHANGELOG.md for migration guide.
  "
  ```

- [ ] Tag release: `git tag -a v0.3.0 -m "Version 0.3.0 - Breaking API cleanup"`
- [ ] Push: `git push origin main --tags`

---

### 5.4 Build and publish (if applicable)

- [ ] Build package: `uv build`
- [ ] Test install in clean environment
- [ ] Publish to PyPI: `uv publish` (if credentials configured)
- [ ] Verify installation: `pip install neurospatial==0.3.0`

---

## Post-Release Tasks

### Documentation

- [ ] Update GitHub release notes with CHANGELOG content
- [ ] Update documentation site (if hosted)
- [ ] Post migration guide to discussions/issues

### User Communication

- [ ] Announce breaking release on relevant channels
- [ ] Provide migration examples
- [ ] Respond to user migration questions

---

## Progress Tracking

**Milestone 1**: ⬜ 0/3 tasks complete
**Milestone 2**: ⬜ 0/2 tasks complete
**Milestone 3**: ⬜ 0/4 tasks complete
**Milestone 4**: ⬜ 0/4 tasks complete
**Milestone 5**: ⬜ 0/4 tasks complete

**Overall Progress**: 0/17 major tasks complete

---

## Notes

- **No backwards compatibility** - Clean break from v0.2.x
- **All changes are breaking** - Requires user code updates
- **Prioritize P0 tasks** - These fix critical usability issues
- **Test thoroughly** - Breaking changes require extra verification
- **Document well** - Clear migration guide essential for adoption

## Dependencies Between Tasks

- 4.1 (Run tests) depends on: ALL of Milestones 1-3
- 4.2 (Update CLAUDE.md) depends on: ALL of Milestones 1-3
- 5.1 (Version bump) depends on: Milestone 4 complete
- 5.2 (Final verification) depends on: 5.1 complete
- 5.3 (Release commit) depends on: 5.2 complete

**Suggested execution order**:

1. Milestone 1 (new methods) - Can be done in parallel
2. Milestone 2 (method renames) - Can be done in parallel with M1
3. Milestone 3 (function renames) - Can be done in parallel with M1/M2
4. Milestone 4 (testing/docs) - After M1-M3 complete
5. Milestone 5 (release) - Final sequential steps
