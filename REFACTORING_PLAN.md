# Environment.py Modularization Plan

**Status**: Planning Phase
**Goal**: Split the 5,335-line `environment.py` into focused, maintainable modules
**Priority**: Medium (code quality improvement, not a blocker)
**Estimated Effort**: 8-12 hours + testing
**Breaking Changes**: NONE - `from neurospatial import Environment` will continue to work

---

## Problem Statement

The `environment.py` file has grown to **5,335 lines**, making it:

- Difficult to navigate and maintain
- Slower to load and parse
- Hard for new contributors to understand
- A potential merge conflict hotspot

---

## Proposed Module Structure

```
src/neurospatial/
└── environment/
    ├── __init__.py              # Public API exports only
    ├── core.py                  # Environment class (core attributes & state)
    ├── factories.py             # Factory methods (from_samples, from_graph, etc.)
    ├── queries.py               # Spatial queries (bin_at, contains, neighbors, etc.)
    ├── serialization.py         # Save/load/to_dict/from_dict
    ├── regions.py               # Region-related methods (bins_in_region, mask_for_region)
    ├── visualization.py         # Plotting methods
    ├── analysis.py              # Analysis methods (occupancy, smooth, etc.)
    └── decorators.py            # check_fitted decorator
```

**Note**: The old `environment.py` file will be completely replaced by the `environment/` package, but the public import path `from neurospatial import Environment` will be maintained via `__init__.py`.

---

## Detailed Module Breakdown

### 1. `environment/__init__.py` (10-20 lines)

**Purpose**: Public API exports only

```python
"""Environment module for spatial discretization."""

from neurospatial.environment.core import Environment
from neurospatial.environment.decorators import check_fitted

__all__ = ["Environment", "check_fitted"]
```

**Exports**:

- `Environment` class (fully assembled with all methods)
- `check_fitted` decorator

**Import Pattern**:

- Primary: `from neurospatial import Environment` (maintained for backward compatibility)
- Alternative: `from neurospatial.environment import Environment` (also works)

---

### 2. `environment/decorators.py` (~50 lines)

**Purpose**: Utility decorators

**Contains**:

- `check_fitted()` decorator (lines 52-91 of current file)

**Dependencies**: None (pure utility)

**Rationale**:

- Small, self-contained
- Reusable across methods
- Clear separation of concerns

---

### 3. `environment/core.py` (~800-1000 lines)

**Purpose**: Core Environment class definition

**Contains**:

- Class definition and dataclass fields
- `__init__` (indirect via `_setup_from_layout`)
- `_setup_from_layout()` method
- `__eq__`, `__repr__`, `_repr_html_()`, `info()`
- Properties: `n_dims`, `n_bins`, `layout_type`, `layout_parameters`
- HTML helper methods: `_html_table_row()`, `_html_table_header()`
- Internal helpers: `_source_flat_to_active_node_id_map()`
- State management: `copy()`

**Dependencies**:

- Standard library (dataclasses, typing)
- NumPy, pandas, NetworkX
- `decorators.py` (for @check_fitted)

**Rationale**:

- Core class structure and state
- Representation methods belong with the class
- Properties that expose core attributes

---

### 4. `environment/factories.py` (~1200-1500 lines)

**Purpose**: Factory methods for creating Environment instances

**Contains**:

- `from_layout()` (line 813)
- `from_samples()` (line 1017)
- `from_graph()` (line 1071)
- `from_polygon()` (line 1153)
- `from_mask()` (line 1227)
- `from_image()` (line 1294)

**Dependencies**:

- `core.py` (Environment class)
- Layout engines
- NumPy, NetworkX, Shapely

**Rationale**:

- Clear conceptual grouping: "How to create an Environment"
- Already marked with `@classmethod`
- Large, self-contained methods
- ~25-30% of total file size

**Implementation Pattern**:

When using mixin inheritance (recommended approach), factory methods are automatically available - no manual binding needed:

```python
# In factories.py
class EnvironmentFactories:
    @classmethod
    def from_samples(cls, data_samples, bin_size, **kwargs):
        """Create Environment from sample data.

        Note: cls will be Environment when called as Environment.from_samples()
        due to Python's method resolution order.
        """
        # Implementation here
        ...

# In core.py
@dataclass
class Environment(EnvironmentFactories, EnvironmentQueries, ...):
    """Main Environment class assembled from mixins."""
    name: str = ""
    # ... rest of dataclass fields
    # from_samples is automatically available via inheritance
```

---

### 5. `environment/queries.py` (~600-800 lines)

**Purpose**: Spatial query methods

**Contains**:

- `bin_at()` - Map points to bins
- `contains()` - Check if points are in environment
- `neighbors()` - Get neighboring bins
- `distance_between()` - Compute distance between bins
- `bin_center_of()` - Get center coordinates
- `shortest_path()` - Find shortest path between bins
- `bin_sizes()` - Get bin sizes/areas/volumes

**Dependencies**:

- `core.py` (Environment class)
- `decorators.py` (@check_fitted)
- NumPy, NetworkX

**Rationale**:

- Clear conceptual grouping: "How to query an Environment"
- Methods that answer spatial questions
- All use `@check_fitted` decorator
- ~12-15% of total file size

---

### 6. `environment/serialization.py` (~400-600 lines)

**Purpose**: Saving and loading environments

**Contains**:

- `save()` (deprecated pickle) (line 4300)
- `load()` (deprecated pickle) (line 4324)
- `to_file()` (JSON + NPZ) (line 4361)
- `from_file()` (JSON + NPZ) (line 4394)
- `to_dict()` (line 4423)
- `from_dict()` (line 4441)

**Dependencies**:

- `core.py` (Environment class)
- `io` module (for to_file/from_file implementation)
- Standard library (pickle, json, pathlib)
- NumPy

**Rationale**:

- Clear conceptual grouping: "How to persist an Environment"
- Mix of instance and class methods
- Already has some separation via `io.py` module
- ~8-10% of total file size

**Serialization Strategy**: Keep thin delegation methods in `EnvironmentSerialization` mixin, implementation stays in `io.py`:

```python
# In environment/serialization.py
class EnvironmentSerialization:
    def to_file(self: "Environment", path: Path | str) -> None:
        """Save environment to disk."""
        from neurospatial.io import to_file
        return to_file(self, path)

    @classmethod
    def from_file(cls: type["Environment"], path: Path | str) -> "Environment":
        """Load environment from disk."""
        from neurospatial.io import from_file
        return from_file(path)
```

This approach:
- Maintains `Environment.to_file()` API for users
- Avoids code duplication
- Keeps implementation details in `io.py`

---

### 7. `environment/regions.py` (~300-400 lines)

**Purpose**: Region-related operations

**Contains**:

- `bins_in_region()` (line 4465)
- `mask_for_region()` (line 4522)
- Region update/manipulation methods (if any)

**Dependencies**:

- `core.py` (Environment class)
- `decorators.py` (@check_fitted)
- `regions/` module (Region, Regions classes)

**Rationale**:

- Focused on region operations
- Small, cohesive set of methods
- ~5-6% of total file size

---

### 8. `environment/visualization.py` (~600-800 lines)

**Purpose**: Plotting and visualization methods

**Contains**:

- `plot()` - Main plotting method
- `plot_1d()` - 1D linearized plotting
- Helper methods for visualization

**Dependencies**:

- `core.py` (Environment class)
- `decorators.py` (@check_fitted)
- Matplotlib

**Rationale**:

- Clear conceptual grouping: "How to visualize an Environment"
- Heavy matplotlib dependency
- Optional functionality (not core to Environment)
- ~12-15% of total file size

---

### 9. `environment/analysis.py` (~800-1200 lines)

**Purpose**: Analysis and computation methods

**Contains**:

- `boundary_bins()` (line 3871)
- `bin_attributes()` (line 3918)
- `edge_attributes()` (line 3951)
- `to_linear()` / `linear_to_nd()` (1D methods)
- Occupancy, smoothing, diffusion methods (if present)
- Kernel computation methods

**Dependencies**:

- `core.py` (Environment class)
- `decorators.py` (@check_fitted)
- NumPy, pandas, SciPy

**Rationale**:

- Computational and analytical methods
- Often involve complex calculations
- ~15-20% of total file size

---

## Migration Strategy

### Phase 1: Preparation (1 hour)

1. **Create test snapshot**

   ```bash
   uv run pytest tests/ --tb=short -v > tests_before.log
   ```

2. **Create environment/ directory**

   ```bash
   mkdir src/neurospatial/environment
   ```

3. **Document current public API** (for verification later)

   ```python
   # List all public methods and attributes
   dir(Environment)
   ```

### Phase 2: Extract Decorators (30 minutes)

**Low Risk, High Confidence**

1. Create `environment/decorators.py`
2. Move `check_fitted` decorator
3. Run tests: `uv run pytest tests/test_check_fitted_error.py`

### Phase 3: Extract Specialized Modules (2-3 hours)

**Low Risk, Parallel Work Possible**

Extract in order of independence:

1. Create `environment/visualization.py` - Move `plot()`, `plot_1d()`
2. Create `environment/analysis.py` - Move `boundary_bins()`, `bin_attributes()`, `edge_attributes()`, linearization methods
3. Create `environment/regions.py` - Move `bins_in_region()`, `mask_for_region()`
4. Create `environment/serialization.py` - Move `save()`, `load()`, `to_file()`, `from_file()`, `to_dict()`, `from_dict()`

Run tests after each: `uv run pytest tests/test_environment.py`

### Phase 4: Extract Queries (1-2 hours)

**Medium Risk, Medium Value**

1. Create `environment/queries.py`
2. Move spatial query methods: `bin_at()`, `contains()`, `neighbors()`, `distance_between()`, `bin_center_of()`, `shortest_path()`, `bin_sizes()`
3. Bind methods using mixin pattern
4. Run tests: `uv run pytest tests/test_environment.py`

### Phase 5: Extract Factories (2-3 hours)

**Medium Risk, High Value**

1. Create `environment/factories.py`
2. Move all `@classmethod` factory methods: `from_layout()`, `from_samples()`, `from_graph()`, `from_polygon()`, `from_mask()`, `from_image()`
3. Use mixin pattern to bind to Environment
4. Run tests: `uv run pytest tests/test_environment.py`

### Phase 6: Create Core Module (1-2 hours)

**High Risk, High Value**

1. Create `environment/core.py`
2. Move core class definition with mixin inheritance
3. Include: dataclass fields, `_setup_from_layout()`, `__eq__`, `__repr__`, `_repr_html_()`, `info()`, properties, HTML helpers, `copy()`
4. Assemble all mixins

### Phase 7: Create Package Init & Update Top-Level Import (30 minutes)

1. Create `environment/__init__.py` with exports
2. Update `src/neurospatial/__init__.py` to import from new location:

   ```python
   # Change from:
   from neurospatial.environment import Environment
   # To:
   from neurospatial.environment.core import Environment
   # Or keep as:
   from neurospatial.environment import Environment  # Works either way
   ```

3. Delete old `environment.py` file
4. Test both import paths:
   - `from neurospatial import Environment`
   - `from neurospatial.environment import Environment`
5. Verify they return the same class object

### Phase 8: Testing & Validation (1-2 hours)

1. **Run full test suite**

   ```bash
   uv run pytest tests/ --tb=short -v
   ```

2. **Compare with snapshot**

   ```bash
   diff tests_before.log tests_after.log
   ```

3. **Verify import works**

   ```python
   from neurospatial.environment import Environment
   ```

4. **Check documentation builds** (if applicable)

### Phase 9: Cleanup & Documentation (1 hour)

1. Update `CHANGELOG.md` with breaking change notice
2. Update documentation to reflect new import path
3. Update CLAUDE.md with new import patterns
4. Clean up any unused imports

---

## Method Binding Pattern

Since Python doesn't have true partial classes, use this pattern:

```python
# environment/factories.py
class EnvironmentFactories:
    @classmethod
    def from_samples(cls, data_samples, bin_size, **kwargs):
        """Create Environment from sample data."""
        # Implementation
        ...

# environment/core.py
from neurospatial.environment.factories import EnvironmentFactories
from neurospatial.environment.queries import EnvironmentQueries
# ... other imports

@dataclass
class Environment:
    """Main Environment class."""
    # Core attributes
    name: str = ""
    # ...

    # Bind factory methods
    from_samples = EnvironmentFactories.from_samples
    from_graph = EnvironmentFactories.from_graph

    # Bind query methods
    bin_at = EnvironmentQueries.bin_at
    contains = EnvironmentQueries.contains
    # ... etc
```

**Advantages**:

- Methods stay with their conceptual groups
- Easy to test individual modules
- Clear separation of concerns

**Disadvantages**:

- Slightly more complex binding
- IDE autocomplete may need configuration

**Recommended Pattern** (Mixins - Simpler without backward compatibility):

```python
# environment/factories.py
class EnvironmentFactories:
    @classmethod
    def from_samples(cls, data_samples, bin_size, **kwargs):
        """Create Environment from sample data."""
        # Implementation
        ...

# environment/queries.py
class EnvironmentQueries:
    def bin_at(self, points):
        """Map points to bins."""
        # Implementation
        ...

# environment/core.py
from neurospatial.environment.decorators import check_fitted
from neurospatial.environment.factories import EnvironmentFactories
from neurospatial.environment.queries import EnvironmentQueries
from neurospatial.environment.serialization import EnvironmentSerialization
from neurospatial.environment.regions import EnvironmentRegions
from neurospatial.environment.visualization import EnvironmentVisualization
from neurospatial.environment.analysis import EnvironmentAnalysis

@dataclass
class Environment(
    EnvironmentFactories,
    EnvironmentQueries,
    EnvironmentSerialization,
    EnvironmentRegions,
    EnvironmentVisualization,
    EnvironmentAnalysis,
):
    """Main Environment class assembled from mixins.

    This approach is cleaner without backward compatibility constraints:
    - All methods naturally belong to Environment
    - IDE autocomplete works perfectly
    - Type hints work correctly
    - No manual method binding needed
    """
    # Core attributes
    name: str = ""
    # ... rest of dataclass fields
```

**Why Mixins Work Better Here**:

- No backward compatibility needed, so we can use clean inheritance
- Methods automatically become part of Environment
- IDE tooling (autocomplete, type checking) works seamlessly
- Less code than manual binding pattern
- Each mixin can be tested independently

---

## CRITICAL: Dataclass and Type Checking Patterns

### Dataclass Usage

**IMPORTANT**: Only `Environment` in `core.py` should use the `@dataclass` decorator. All mixin classes MUST be plain classes.

**Why**: Python's dataclass field inheritance with multiple inheritance can cause conflicts and MRO errors.

**Correct Pattern**:

```python
# environment/queries.py
class EnvironmentQueries:  # ← Plain class, NOT @dataclass
    """Spatial query methods mixin."""

    def bin_at(self, points):
        """Map points to bins."""
        # Can safely access self.layout, self.connectivity, etc.
        # because Environment (the dataclass) provides these attributes
        ...

# environment/core.py
@dataclass  # ← ONLY Environment is a dataclass
class Environment(
    EnvironmentFactories,
    EnvironmentQueries,
    # ... other mixins
):
    """Main Environment class."""
    name: str = ""
    layout: LayoutEngine | None = None
    # ... rest of dataclass fields
```

### Circular Import Prevention

**Problem**: `core.py` imports all mixins, but mixins may need to reference `Environment` for type hints.

**Solution**: Use `TYPE_CHECKING` guards (already used in 9 files in this codebase).

**Correct Pattern**:

```python
# In any mixin file (queries.py, factories.py, etc.)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

class EnvironmentQueries:
    def bin_at(self: "Environment", points) -> int:
        # Use string annotation for type hint
        # At runtime, self is Environment instance
        # At type-checking time, type checkers see the hint
        ...

    @check_fitted
    def contains(self: "Environment", points) -> NDArray[np.bool_]:
        # Type hint with string annotation
        ...
```

**Key Points**:

1. Import `Environment` only inside `if TYPE_CHECKING:` block
2. Use string annotations (`"Environment"`) for type hints that reference `Environment`:
   - `self` parameter: `def bin_at(self: "Environment", ...) -> int:`
   - `cls` parameter in classmethods: `def from_file(cls: type["Environment"], ...) -> "Environment":`
   - Return types that reference `Environment`: `-> "Environment"`
   - Other parameters don't need string annotations: `points: NDArray` works fine
3. Never import `Environment` at module level in mixins
4. At runtime, `TYPE_CHECKING` is `False`, so no circular import occurs
5. Type checkers (mypy, pyright) see the string annotation and resolve it

### Decorator Imports

Each mixin that uses `@check_fitted` should import from `decorators.py`:

```python
# In queries.py, analysis.py, regions.py, visualization.py
from neurospatial.environment.decorators import check_fitted

class EnvironmentQueries:
    @check_fitted
    def bin_at(self, points):
        ...
```

---

## Testing Strategy

### Unit Tests

- Each module should have corresponding tests
- `tests/environment/test_factories.py`
- `tests/environment/test_queries.py`
- etc.

### Integration Tests

- Keep existing `tests/test_environment.py` as integration tests
- Ensure all functionality still works end-to-end

### Import Tests

```python
def test_primary_import():
    """Test that primary import path still works."""
    from neurospatial import Environment
    assert Environment is not None

def test_direct_import():
    """Test that direct import path also works."""
    from neurospatial.environment import Environment
    assert Environment is not None

def test_imports_are_same():
    """Test that both import paths return the same class."""
    from neurospatial import Environment as Env1
    from neurospatial.environment import Environment as Env2
    assert Env1 is Env2

def test_check_fitted_import():
    """Test that decorator can be imported."""
    from neurospatial.environment import check_fitted
    assert check_fitted is not None
```

### Mixin Verification Tests

```python
def test_mixins_are_not_dataclasses():
    """Ensure mixins are plain classes, not dataclasses."""
    from neurospatial.environment.factories import EnvironmentFactories
    from neurospatial.environment.queries import EnvironmentQueries
    from neurospatial.environment.serialization import EnvironmentSerialization
    from neurospatial.environment.regions import EnvironmentRegions
    from neurospatial.environment.visualization import EnvironmentVisualization
    from neurospatial.environment.analysis import EnvironmentAnalysis

    mixins = [
        EnvironmentFactories,
        EnvironmentQueries,
        EnvironmentSerialization,
        EnvironmentRegions,
        EnvironmentVisualization,
        EnvironmentAnalysis,
    ]

    for mixin in mixins:
        assert not hasattr(mixin, '__dataclass_fields__'), \
            f"{mixin.__name__} should NOT be a dataclass"

def test_environment_is_dataclass():
    """Ensure Environment itself IS a dataclass."""
    from neurospatial.environment import Environment
    assert hasattr(Environment, '__dataclass_fields__'), \
        "Environment should be a dataclass"

def test_factory_methods_return_environment_type():
    """Verify factory classmethods return Environment, not mixin type."""
    from neurospatial.environment import Environment
    import numpy as np

    data = np.random.rand(100, 2)
    env = Environment.from_samples(data, bin_size=2.0)

    assert isinstance(env, Environment)
    assert type(env).__name__ == "Environment"

def test_mro_order():
    """Verify Method Resolution Order is correct."""
    from neurospatial.environment import Environment
    from neurospatial.environment.factories import EnvironmentFactories
    from neurospatial.environment.queries import EnvironmentQueries

    mro = Environment.__mro__
    assert mro[0] is Environment
    assert EnvironmentFactories in mro
    assert EnvironmentQueries in mro

def test_no_circular_imports():
    """Ensure no circular import errors at module load (runtime check)."""
    import sys

    # Clear neurospatial modules to test fresh import
    modules_to_clear = [mod for mod in sys.modules if 'neurospatial' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]

    # Try importing - should not raise ImportError
    from neurospatial.environment import Environment
    from neurospatial import Environment as Env2

    assert Environment is not None
    assert Environment is Env2
```

---

## Risks & Mitigation

### Risk 1: Dataclass with Multiple Inheritance

**Likelihood**: High (will occur if not careful)
**Impact**: High (runtime errors, MRO conflicts)
**Mitigation**:

- **Only `Environment` in `core.py` uses `@dataclass`**
- All mixins are plain classes (no `@dataclass` decorator)
- Document this requirement clearly in each mixin file
- Add verification test to ensure mixins are not dataclasses

### Risk 2: Circular Import Dependencies

**Likelihood**: Medium (if type hints not handled correctly)
**Impact**: High (import errors at runtime)
**Mitigation**:

- **Use `TYPE_CHECKING` guards in all mixins** (pattern already used in 9 codebase files)
- Import `Environment` only inside `if TYPE_CHECKING:` blocks in mixins
- Use string annotations (`self: "Environment"`) for type hints
- Never import `Environment` at module level in mixin files
- Test imports explicitly in Phase 7

### Risk 3: IDE/Tooling Issues

**Likelihood**: Medium
**Impact**: Low
**Mitigation**:

- **Use explicit type hints in mixin methods** to help IDEs:
  ```python
  def bin_at(self: "Environment", points: NDArray) -> int:
      # IDE sees "Environment" type hint and can autocomplete self.layout
      ...
  ```
- Test with common IDEs (VSCode, PyCharm)
- Verify autocomplete works for inherited methods
- Update type stubs if needed
- Document any known issues

**Why Medium Likelihood**: IDEs need to resolve `self` type across files and follow inheritance chains. Explicit type hints significantly improve the experience.

### Risk 4: Documentation Sync

**Likelihood**: High
**Impact**: Low
**Mitigation**:

- Update docstrings during refactor
- Regenerate API documentation
- Add migration guide

---

## Success Criteria

1. ✅ All 1,067 tests pass
2. ✅ Public API preserved (same methods, same behavior, same import path)
3. ✅ Each module < 1,000 lines
4. ✅ Clear module boundaries with minimal coupling
5. ✅ Both import paths work:
   - `from neurospatial import Environment` ✓
   - `from neurospatial.environment import Environment` ✓
6. ✅ No circular import errors (TYPE_CHECKING pattern used correctly)
7. ✅ Only Environment is a dataclass (mixins are plain classes)
8. ✅ Documentation updated (CHANGELOG, README, CLAUDE.md)
9. ✅ Code coverage maintained or improved
10. ✅ Factory methods return Environment type (not mixin types)

---

## Future Improvements

After successful modularization:

1. **Type Hints**: Add comprehensive type hints to each module
2. **Protocol Classes**: Extract interfaces/protocols for extensibility
3. **Plugin System**: Enable custom Environment behaviors via plugins
4. **Lazy Loading**: Only import visualization/analysis when needed
5. **Async Support**: Add async versions of I/O methods

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Preparation | 1 hour | 1h |
| Extract Decorators | 30 min | 1.5h |
| Extract Specialized Modules | 2-3 hours | 4.5h |
| Extract Queries | 1-2 hours | 6.5h |
| Extract Factories | 2-3 hours | 9.5h |
| Create Core | 1-2 hours | 11.5h |
| Create Package Init | 30 min | 12h |
| Testing & Validation | 1-2 hours | 14h |
| Cleanup & Documentation | 1 hour | 15h |

**Total: 11-15 hours** (1.5-2 days of focused work)

**Realistic estimate with buffer**: 15-20 hours (2-3 days of focused work)

**Additional time accounts for**:

- Debugging circular imports or dataclass issues (+2-3 hours)
- Testing after each extraction phase (+1-2 hours)
- Handling edge cases in factory methods (+1-2 hours)
- IDE verification and type hint adjustments (+1 hour)

**Note**: Under-promise and over-deliver is better than the reverse. Budget 20 hours to be safe.

---

## Decision: Proceed?

**Recommendation**: YES, as a **non-breaking refactoring** (suitable for minor/patch version)

**Rationale**:

- Significantly improves code maintainability (5,335 lines → 9 focused modules)
- **NO breaking changes** - `from neurospatial import Environment` continues to work
- Clean mixin pattern is more Pythonic and IDE-friendly
- Better module organization makes contributing easier
- Internal refactoring only - users won't notice any changes

**Version Recommendation**:

Since this is a non-breaking internal refactoring:

- v0.2.1 (patch) or v0.3.0 (minor) - user's choice
- No major version bump needed
- CHANGELOG entry: "refactor: modularize environment.py into focused submodules"

**Next Steps**:

1. Create refactoring branch from main
2. Follow phased migration strategy (9 phases, ~8-12 hours)
3. Ensure all 1,067 tests pass
4. Update CHANGELOG.md with refactoring note
5. Update CLAUDE.md with new internal structure
6. Release as minor/patch version

**Files That DO NOT Need Updates**:

Since `from neurospatial import Environment` continues to work, **NO user-facing files need updates**:

- ✓ Tests continue to work (no changes needed)
- ✓ Documentation examples continue to work (no changes needed)
- ✓ Notebooks continue to work (no changes needed)
- ✓ User code continues to work (no changes needed)

**Internal Files to Update**:

Only internal files may need adjustments:

1. `src/neurospatial/__init__.py` - Update to import from `environment/` package
2. Internal imports may need adjustment if they reference `environment.py` directly

**Key Benefit**: Users can upgrade seamlessly without any code changes!

---

## References

- Python import system: <https://docs.python.org/3/reference/import.html>
- Package design patterns: <https://packaging.python.org/guides/>
- Refactoring large codebases: <https://martinfowler.com/books/refactoring.html>
