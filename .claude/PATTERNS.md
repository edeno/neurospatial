# Design Patterns & Constraints

Design patterns you **must** follow when extending the codebase.

---

## Graph Metadata Requirements

The connectivity graph (`nx.Graph`) has **mandatory node and edge attributes**:

### Node Attributes

Enforced by layout engines:

- `'pos'`: Tuple[float, ...] - N-D coordinates
- `'source_grid_flat_index'`: int - Flat index in original grid
- `'original_grid_nd_index'`: Tuple[int, ...] - N-D grid index

### Edge Attributes

Enforced by layout engines:

- `'distance'`: float - Euclidean distance between bin centers
- `'vector'`: Tuple[float, ...] - Displacement vector
- `'edge_id'`: int - Unique edge ID
- `'angle_2d'`: Optional[float] - Angle for 2D layouts

---

## Mixin Pattern for Environment

The `Environment` class uses **mixin inheritance** to organize 6,000+ lines of functionality.

### Key Constraints

- **ONLY `Environment` is a `@dataclass`** - All mixins MUST be plain classes
- Mixins use `TYPE_CHECKING` guards to avoid circular imports
- All mixin methods annotate `self: "Environment"` for type checking
- Public API unchanged: `from neurospatial import Environment` still works

### Example Mixin

```python
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

class EnvironmentMixin:
    """Mixin providing methods for Environment class.

    Methods in this mixin expect to be called on an Environment instance
    and will have access to all Environment attributes.
    """

    def method_name(self: "Environment", param: int) -> NDArray[np.float64]:
        """Use string annotation 'Environment' for self parameter."""
        # Mypy understands self has Environment attributes
        return self.bin_centers  # ✓ No error
```

---

## Mypy Type Checking Requirements

**IMPORTANT: Mypy runs in pre-commit hooks and should pass without errors.**

### Configuration

Pragmatic configuration in `pyproject.toml` suited for scientific Python:

- **Type annotations encouraged**: Public APIs and mixin methods should have type hints
- **Lenient for scientific code**: Allows untyped defs, untyped calls, incomplete defs
- **Pre-commit enforcement**: Mypy runs automatically on commit

**Key settings** (`[tool.mypy]` in `pyproject.toml`):

- `disallow_untyped_defs = false` - Allows functions without type annotations
- `check_untyped_defs = false` - Doesn't check inside untyped functions
- `allow_untyped_calls = true` - Allows calling untyped library functions
- `warn_unused_ignores = true` - Warns about unnecessary `type: ignore` comments

### Guidelines

1. **Prefer proper typing over suppressions** - Add type hints when possible rather than using `type: ignore`
2. **Mixin methods should be typed** - Use proper type annotations for all public mixin methods
3. **Avoid skipping mypy** - Let pre-commit run mypy normally; only skip if absolutely necessary

### Mixin Type Annotation Pattern

Since mixins access Environment attributes that don't exist on the mixin class itself:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

class EnvironmentMixin:
    """Mixin providing methods for Environment class."""

    def method_name(self: "Environment", param: int) -> NDArray[np.float64]:
        """Use string annotation 'Environment' for self parameter."""
        return self.bin_centers  # ✓ Mypy understands self has Environment attributes
```

### Common Mypy Issues and Fixes

- **Missing attribute errors**: Ensure `self: "Environment"` annotation is present
- **Type mismatches**: Ensure return types match exactly between mixin and Environment
- **Cache type annotations**: Use precise Literal types for cache keys
- **Import TYPE_CHECKING**: Always use `if TYPE_CHECKING:` guard for Environment import

### Using Protocol for Mixin Type Safety

This project uses the `Protocol` pattern (PEP 544) for proper type checking of mixins.

**Pattern** (`src/neurospatial/environment/_protocols.py`):

```python
from typing import Protocol

class EnvironmentProtocol(Protocol):
    """Interface that Environment provides to mixins."""
    name: str
    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    # ... all attributes mixins need

    def bin_at(self, points: NDArray[np.float64]) -> NDArray[np.int_]: ...
```

**Mixin Usage:**

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol

class EnvironmentFields:
    def smooth(self: EnvironmentProtocol, field: NDArray, ...) -> NDArray:
        # Mypy validates Protocol at call sites, not definition sites
        return self.compute_kernel(...) @ field
```

**Why this works:** Mypy checks that `Environment` satisfies `EnvironmentProtocol` at usage sites. Mixins can safely access Protocol-defined attributes without "erased type" errors.

**Run mypy manually:**

```bash
# Check all files
uv run mypy src/neurospatial/

# Check specific file
uv run mypy --ignore-missing-imports src/neurospatial/environment/fields.py
```

---

## Protocol-Based Design

Layout engines implement the `LayoutEngine` Protocol, **not inheritance**.

### When Creating New Engines

**Required attributes:**

- `bin_centers`: NDArray[np.float64]
- `connectivity`: nx.Graph
- `dimension_ranges`: list[tuple[float, float]]
- `is_1d`: bool
- `_layout_type_tag`: str
- `_build_params_used`: dict

**Required methods:**

- `build()` - Construct layout
- `point_to_bin_index()` - Map point to bin
- `bin_sizes()` - Compute bin sizes
- `plot()` - Visualize layout

**Optional grid metadata:**

- `grid_edges`: tuple of NDArray
- `grid_shape`: tuple[int, ...]
- `active_mask`: NDArray[np.bool_]

---

## Fitted State Pattern

`Environment` uses a `_is_fitted` flag set after `_setup_from_layout()` completes.

- Methods requiring fitted state use the `@check_fitted` decorator
- Prevents calling spatial query methods before initialization
- Factory methods automatically set fitted state

**Example:**

```python
@check_fitted
def bin_at(self, points: NDArray) -> NDArray[np.int_]:
    """Requires fitted state - decorator enforces."""
    # Implementation
```

---

## Regions are Immutable

`Region` objects are immutable dataclasses - create new instances rather than modifying existing ones.

### Regions Container Semantics

The `Regions` container uses dict-like semantics:

- Use `regions.add()` to create and insert (raises `KeyError` if name already exists)
- Use `del regions[name]` or `regions.remove(name)` to delete
- Assignment to existing keys succeeds but emits a `UserWarning` to prevent accidental overwrites
- Use `regions.update_region()` to update regions without warnings

**Correct usage:**

```python
# Add new region
env.regions.add("goal", point=(50.0, 50.0))

# Update region (creates new Region, no warning)
env.regions.update_region("goal", point=(55.0, 55.0))

# Delete region
del env.regions["goal"]
# or
env.regions.remove("goal")
```

**Avoid:**

```python
# ✗ Wrong - immutable
env.regions['goal'].point = new_point  # AttributeError

# ⚠️ Works but emits UserWarning
env.regions['goal'] = new_region  # UserWarning: Overwriting existing region
```

---

## 1D vs N-D Environments

Environments can be 1D (linearized tracks) or N-D (grids):

- **1D**: `GraphLayout` with `is_1d=True`, provides `to_linear()` and `linear_to_nd()`
- **N-D**: Grid-based layouts with spatial queries in original coordinate space

**Always check before linearization:**

```python
if env.is_1d:
    linear_pos = env.to_linear(nd_position)
else:
    bin_idx = env.bin_at(position)  # Use N-D queries
```

---

## Methods vs Free Functions

**When to use Environment methods vs module-level functions:**

### Environment Methods

Answer questions about that environment or perform local transforms:

- **Examples**: `env.bin_at()`, `env.neighbors()`, `env.distance_between()`, `env.rebin()`
- **Use when**: Working with a single environment's structure and properties

### Free Functions

Take environments/graphs/fields as input and perform higher-level analysis:

- **Examples**: `distance_field()`, `map_points_to_bins()`, `estimate_transform()`, `compute_place_field()`
- **Use when**: Cross-environment operations, neural/behavioral analysis, or batch processing

**If you're unsure:** Start from the object you have (Environment, field array, graph) and use autocomplete. If it's about cross-environment, neural, or behavioral analysis, look under the free functions in `neurospatial.__init__`.

**Design principle:** This separation keeps the `Environment` class focused on spatial structure while providing specialized functions for domain-specific analyses (neuroscience, navigation, etc.).

---

## Animation Overlay Coordinate Conventions

All overlay data uses **environment coordinates** (x, y):

- System automatically transforms to napari pixel space
- Transformations include: (x,y) to (row,col) swap and Y-axis inversion
- For EventOverlay: positions are interpolated from trajectory or provided explicitly

**Correct usage:**

```python
# Your data is in environment coordinates (x, y)
positions = np.array([[10.0, 20.0], [15.0, 25.0]])

# Pass directly - transformation happens internally
overlay = PositionOverlay(data=positions)
env.animate_fields(fields, frame_times=frame_times, overlays=[overlay])
```

**Wrong:**

```python
# Don't manually swap x,y or invert
positions_wrong = np.column_stack([y_coords, x_coords])  # Manual swap
overlay = PositionOverlay(data=positions_wrong)  # Display will be wrong!
```

---

## Skeleton Class Pattern

The `Skeleton` class defines anatomical structure for pose tracking:

```python
from neurospatial.animation.skeleton import Skeleton

skeleton = Skeleton(
    name="mouse",
    nodes=("nose", "body", "tail"),
    edges=(("nose", "body"), ("body", "tail")),
)

# Graph traversal via adjacency property
skeleton.adjacency["body"]  # Returns ['nose', 'tail']
```

**Key features:**

- **Edge canonicalization**: Edges stored in lexicographic order `(min, max)`
- **Deduplication**: Reversed duplicates automatically removed
- **Adjacency property**: Precomputed O(1) access for graph traversal
- **Immutable**: Frozen dataclass with slots for performance
- **Factory methods**: `from_edge_list()`, `from_ndx_pose()`, `from_movement()`

---

## Cache Management

Environment instances cache expensive computations (KDTree, kernels, properties).

**Clear caches before parallel rendering:**

```python
# IMPORTANT: Makes environment pickle-able for multiprocessing
env.clear_cache()

# Then safe to use parallel rendering
env.animate_fields(fields, frame_times=frame_times, backend="video", n_workers=4, save_path="output.mp4")
```

**Selective cache clearing:**

```python
env.clear_cache(kdtree=True, kernels=False, cached_properties=False)
```

---

## Error Message Diagnostics

When validation fails, error messages include actual invalid values to help debugging.

**Example:**

```
ValueError: bin_size must be positive (got -2.0)
ValueError: No active bins found. Data range: [0.0, 100.0], bin_size: 200.0
```

**Use these diagnostics** to identify the problem immediately rather than guessing.
