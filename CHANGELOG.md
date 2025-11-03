# Changelog

All notable changes to the neurospatial project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-03

### Added

#### New Features
- **Regions.update_region()**: New method to update existing regions while preserving metadata ([#68e49a4](https://github.com/edeno/neurospatial/commit/68e49a4))
  - Supports updating point or polygon geometry
  - Preserves metadata when not explicitly provided
  - Maintains immutable Region design pattern

- **Environment.__repr__()**: Concise single-line representation showing name, dimensions, bins, and layout type ([#71e6e72](https://github.com/edeno/neurospatial/commit/71e6e72))
  - Example: `<Environment 'PlusMaze' (2D, 100 bins, regular_grid)>`
  - Improves interactive debugging experience

- **Environment._repr_html_()**: Rich HTML representation for Jupyter notebooks ([#71e6e72](https://github.com/edeno/neurospatial/commit/71e6e72))
  - Styled table showing all environment properties
  - Better visualization in notebook environments

- **Environment.info()**: Detailed diagnostic method for environment inspection ([#3d6ab57](https://github.com/edeno/neurospatial/commit/3d6ab57))
  - Shows comprehensive environment details: dimensions, extent, bin sizes, connectivity
  - Includes linearization status and region count
  - Formatted for readability in terminal output

#### Documentation Improvements
- **Comprehensive README.md**: Complete project documentation with installation, quickstart, and examples ([#c055c32](https://github.com/edeno/neurospatial/commit/c055c32))
  - Project overview and key features
  - Installation instructions for both users and developers
  - Core concepts explanation (bins, active bins, connectivity, layout engines)
  - Four common use case examples with verified code
  - All examples tested and guaranteed to run

- **Factory Method Selection Guide**: Decision tree in Environment docstring helping users choose the right factory method ([#724b4ad](https://github.com/edeno/neurospatial/commit/724b4ad))

- **See Also Cross-References**: All factory methods now cross-reference related methods for better discoverability ([#639c803](https://github.com/edeno/neurospatial/commit/639c803))

- **Active Bins Terminology**: Added terminology section explaining "active bins" concept and scientific motivation ([#d1f2de2](https://github.com/edeno/neurospatial/commit/d1f2de2))

- **Units Clarification**: All factory methods now explicitly document bin_size units matching data coordinate system ([#34f7dc7](https://github.com/edeno/neurospatial/commit/34f7dc7))

- **Scientific Terms Defined**: Parenthetical definitions for domain-specific terms:
  - "Place fields" in alignment module
  - "Geodesic distance" in spatial methods
  - "Linearization" in 1D track methods
  - ([#e9e2539](https://github.com/edeno/neurospatial/commit/e9e2539))

- **Common Pitfalls Sections**: Added to `from_samples()` and `CompositeEnvironment.__init__()` with actionable guidance ([#13356a4](https://github.com/edeno/neurospatial/commit/13356a4))

- **Doctests**: All 20 docstring examples now pass pytest doctest validation ([#d852ac8](https://github.com/edeno/neurospatial/commit/d852ac8))

### Changed

#### API Changes
- **BREAKING: bin_size now required**: Removed default values from all factory methods requiring explicit bin_size ([#11e3fda](https://github.com/edeno/neurospatial/commit/11e3fda))
  - Affects: `from_samples()`, `from_polygon()`, `from_mask()`, `from_image()`
  - **Migration**: Add explicit `bin_size` parameter to all method calls
  - **Rationale**: Prevents accidental use of arbitrary defaults; forces intentional parameter selection

#### Error Message Improvements
- **"No active bins" error**: Comprehensive diagnostic error with WHAT/WHY/HOW pattern ([#429c823](https://github.com/edeno/neurospatial/commit/429c823))
  - Shows data range, grid shape, bin_size, thresholds
  - Explains 3 common causes
  - Provides 4 specific actionable fixes
  - Special handling for all-NaN data edge case

- **Standardized error messages**: All validation errors now show actual invalid values ([#0c678eb](https://github.com/edeno/neurospatial/commit/0c678eb))
  - Pattern: `"parameter must be constraint (got actual_value)"`
  - Affected modules: `layout/helpers/utils.py`, `layout/helpers/regular_grid.py`, `layout/helpers/hexagonal.py`, `layout/helpers/graph.py`, `calibration.py`

- **Enhanced @check_fitted error**: Now includes correct and incorrect usage examples ([#e38e61a](https://github.com/edeno/neurospatial/commit/e38e61a))

- **Improved dimension mismatch error**: CompositeEnvironment now explains common causes and how to fix ([#015a55e](https://github.com/edeno/neurospatial/commit/015a55e))

- **Better type validation**: Try-except blocks with helpful error messages for type conversion failures ([#01fc73a](https://github.com/edeno/neurospatial/commit/01fc73a))
  - Separate validation for NaN/Inf vs type errors
  - Preserves original exceptions with `from e`
  - Clear guidance on expected types vs received types

### Fixed
- **Regions documentation bug**: Fixed `__setitem__` error message that referenced non-existent `update()` method ([#68e49a4](https://github.com/edeno/neurospatial/commit/68e49a4))
  - Now correctly references `update_region()` or `add()` depending on context

- **Metadata preservation**: `update_region()` now preserves existing metadata when not explicitly provided ([#68e49a4](https://github.com/edeno/neurospatial/commit/68e49a4))

### Testing
- **Type validation test suite**: 32 new tests verifying helpful error messages for invalid inputs ([#01fc73a](https://github.com/edeno/neurospatial/commit/01fc73a))
  - Tests for strings, mixed types, NaN, Inf, None, dicts, booleans, complex numbers
  - Validates error message quality and exception chaining

- **Error message content tests**: 10 tests for "no active bins" error message completeness ([#429c823](https://github.com/edeno/neurospatial/commit/429c823))

- **Regions update tests**: 8 tests for `update_region()` behavior including edge cases ([#68e49a4](https://github.com/edeno/neurospatial/commit/68e49a4))

- **Representation tests**: Tests for `__repr__()` and `_repr_html_()` output format ([#71e6e72](https://github.com/edeno/neurospatial/commit/71e6e72))

- **Info method tests**: Tests for `.info()` output with various configurations ([#3d6ab57](https://github.com/edeno/neurospatial/commit/3d6ab57))

- **README examples verification**: All 5 README examples tested programmatically ([#c055c32](https://github.com/edeno/neurospatial/commit/c055c32))

- **Doctest suite**: 20 passing doctests across all modules ([#d852ac8](https://github.com/edeno/neurospatial/commit/d852ac8))

- **Test coverage**: All Milestones 1-4 complete with 526 tests passing, >95% coverage of new code ([#6a06424](https://github.com/edeno/neurospatial/commit/6a06424))


---

## Usage Guide

### Important: bin_size Parameter

The `bin_size` parameter is **required** (has no default value) in these factory methods:
- `Environment.from_samples()`
- `Environment.from_polygon()`
- `Environment.from_mask()`
- `Environment.from_image()`

**Example:**
```python
# Must explicitly specify bin_size
env = Environment.from_samples(data, bin_size=2.0)  # 2.0 cm (or whatever your data units are)
```

**Why**: Forcing explicit bin_size selection ensures users make intentional choices about spatial discretization and prevents accidental use of arbitrary defaults.

**Best practice**: Consider your data's coordinate system units (cm, meters, pixels, etc.) when choosing bin_size.

### New Features in 0.1.0

#### Regions.update_region() Method

Update existing regions without deleting and recreating them:

```python
# Create a region
env.regions.add("goal", point=np.array([10.0, 10.0]))

# Update it later (preserves metadata)
env.regions.update_region("goal", point=np.array([15.0, 15.0]))
```

#### Environment Inspection Methods

Better debugging and exploration:

```python
# Quick overview
env  # <Environment 'PlusMaze' (2D, 100 bins, regular_grid)>

# Detailed information
env.info()  # Shows extent, bin sizes, connectivity, regions, etc.
```

#### Jupyter Notebook Rich Display

Automatic rich HTML rendering in Jupyter notebooks via `_repr_html_()`.

---

## Contributors

- Eric Denovellis ([@edeno](https://github.com/edeno))

## Acknowledgments

This UX refactoring (Milestones 1-4) was completed with assistance from Claude Code to improve the user experience and documentation quality of the neurospatial package.
