# Environment Refactoring Plan

**Goal**: Transform Environment from a "God Object" (~70 methods via mixins) to a "Data Container" (~25 methods/properties) following Rhodes-Hettinger design principles.

**Status**: Pending (execute after PLAN.md completion)
**Breaking Change**: Yes - Clean break, no deprecation period (no current users)
**Target Version**: v1.0.0 or v2.0.0

---

## Design Decisions (Locked)

These decisions have been validated and should not be revisited:

1. **Frozen Dataclass with Interior Mutability**
   - `Environment` is `@dataclass(frozen=True)` - geometry cannot change
   - `Regions` remains mutable - annotations can be added/edited
   - Rationale: Geometry changes invalidate caches; annotations don't

2. **Factory Methods as Facades**
   - Keep `Environment.from_samples()`, `from_polygon()`, etc. as class methods
   - Implementation lives in `neurospatial/factories/` module
   - User benefit: Autocomplete works; architecture benefit: no file bloat

3. **Domain-Specific Re-exports**
   - `from neurospatial.encoding import smooth_field`
   - `from neurospatial.behavior import compute_occupancy`
   - Preserves discoverability within scientific domains

4. **Extend Existing `ops/` Subpackage** (CORRECTED)
   - `ops/` already exists with 11 modules - extend, don't recreate
   - Add missing graph algorithms to existing files
   - Mathematical primitives stay in `neurospatial/ops/`
   - Top-level namespaces reserved for scientific domains

5. **Cache Strategy** (REVISED - see Cache Strategy section)
   - Keep `env._kernel_cache` dict approach (Environment not hashable for `@lru_cache`)
   - Move KDTree cache clearing to `ops.binning.clear_kdtree_cache(env)`
   - Provide `neurospatial.clear_all_caches(env)` convenience function

---

## Existing ops/ Module Inventory

**IMPORTANT**: These modules already exist and should be EXTENDED, not replaced:

```
src/neurospatial/ops/
├── __init__.py          # 219 lines - comprehensive re-exports
├── alignment.py         # map_probabilities, apply_similarity_transform
├── basis.py             # spatial_basis, geodesic_rbf_basis, heat_kernel_wavelet_basis
├── binning.py           # map_points_to_bins, clear_kdtree_cache, regions_to_mask
├── calculus.py          # gradient, divergence, compute_differential_operator ← ALREADY EXISTS
├── distance.py          # geodesic_distance_matrix, distance_field, pairwise_distances ← ALREADY EXISTS
├── egocentric.py        # heading_from_velocity, compute_egocentric_bearing
├── graph.py             # convolve, neighbor_reduce ← EXTEND with path algorithms
├── normalize.py         # normalize_field, clamp, combine_fields
├── smoothing.py         # compute_diffusion_kernels, apply_kernel ← EXTEND
├── transforms.py        # Affine2D, calibrate_from_landmarks ← EXTEND with subset/rebin
└── visibility.py        # compute_viewshed, compute_view_field
```

**Functions to ADD to existing modules:**

| Target File | Functions to Add | Source |
|-------------|------------------|--------|
| `ops/graph.py` | `shortest_path`, `reachable_from`, `connected_components`, `graph_rings` | `environment/queries.py` |
| `ops/distance.py` | `distance_to_region` | `environment/queries.py` |
| `ops/smoothing.py` | `smooth_field` (wrapper for apply_kernel) | `environment/fields.py` |
| `ops/transforms.py` | `subset_environment`, `rebin_environment` | `environment/transforms.py` |

---

## Naming Convention (Standardized)

**Rule**: All public functions use `verb_noun` pattern for consistency.

| Category | Pattern | Examples |
|----------|---------|----------|
| Compute operations | `compute_*` | `compute_occupancy`, `compute_kernel`, `compute_transition_matrix` |
| Get operations | `get_*` | `get_bins_in_region`, `get_region_mask` |
| Plot operations | `plot_*` | `plot_environment`, `plot_field` |
| Transform operations | verb + target | `smooth_field`, `interpolate_field`, `subset_environment` |

**Migration renaming:**

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| `bin_sequence` | `compute_bin_sequence` | Add verb prefix |
| `distance_to_region` | `compute_distance_to_region` | Add verb prefix |
| `geodesic_distance` | `compute_geodesic_distance` | Add verb prefix |
| `shortest_path` | `compute_shortest_path` | Add verb prefix |
| `reachable_from` | `compute_reachable_bins` | Add verb + clarify |
| `connected_components` | `compute_connected_components` | Add verb prefix |
| `graph_rings` | `compute_graph_rings` | Add verb prefix |

---

## Cache Strategy (Revised)

### Problem with `@lru_cache`

`@lru_cache` requires hashable arguments. `Environment` is NOT hashable because:

- Contains mutable `Regions`
- Contains `nx.Graph` (unhashable)
- Even with `frozen=True`, interior mutability breaks hashing

```python
# This WILL NOT work:
@lru_cache(maxsize=128)
def compute_kernel(env: Environment, bandwidth: float) -> NDArray:
    ...  # TypeError: unhashable type: 'Environment'
```

### Solution: Keep Instance-Level Cache

**Keep current pattern** - Environment holds its own cache dict:

```python
@dataclass(frozen=True)
class Environment:
    # Use object.__setattr__ to set on frozen dataclass during __post_init__
    _kernel_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        # Frozen dataclass workaround for mutable default
        object.__setattr__(self, '_kernel_cache', {})
```

**Provide convenience functions for cache management:**

```python
# neurospatial/ops/cache.py (NEW)
def clear_kernel_cache(env: Environment) -> None:
    """Clear kernel cache for this environment."""
    env._kernel_cache.clear()

def clear_kdtree_cache(env: Environment) -> None:
    """Clear KDTree cache (already in ops/binning.py)."""
    from neurospatial.ops.binning import clear_kdtree_cache
    clear_kdtree_cache(env)

def clear_all_caches(env: Environment) -> None:
    """Clear all caches for this environment."""
    clear_kernel_cache(env)
    clear_kdtree_cache(env)
```

**Top-level convenience export:**

```python
# neurospatial/__init__.py
from neurospatial.ops.cache import clear_all_caches
```

**User experience:**

```python
# Before parallel rendering (common use case)
from neurospatial import clear_all_caches
clear_all_caches(env)
animate_fields(env, fields, backend="video", n_workers=4)
```

---

## Final Environment Class Definition

```python
@dataclass(frozen=True)
class Environment:
    """
    The spatial arena definition.
    A pure data container for geometry and topology.
    """
    # === Core State ===
    layout: LayoutEngine
    regions: Regions = field(default_factory=Regions)  # Mutable container
    name: str = ""
    units: str | None = None
    frame: str | None = None

    # === Internal (not user-facing) ===
    _kernel_cache: dict = field(default_factory=dict, repr=False, compare=False)

    # === Facade Factories (dispatch to neurospatial.factories) ===
    @classmethod
    def from_samples(cls, positions, bin_size, **kwargs) -> "Environment": ...
    @classmethod
    def from_polygon(cls, polygon, bin_size, **kwargs) -> "Environment": ...
    @classmethod
    def from_graph(cls, graph, **kwargs) -> "Environment": ...
    @classmethod
    def from_mask(cls, mask, bin_size, **kwargs) -> "Environment": ...
    @classmethod
    def from_image(cls, image_path, bin_size, **kwargs) -> "Environment": ...
    @classmethod
    def from_polar_egocentric(cls, **kwargs) -> "Environment": ...
    @classmethod
    def from_layout(cls, layout, **kwargs) -> "Environment": ...
    @classmethod
    def from_nwb(cls, nwb_file, **kwargs) -> "Environment": ...  # Facade to io.from_nwb

    # === Identity & Introspection ===
    def info(self) -> str: ...
    def copy(self, deep: bool = True) -> "Environment": ...
    def __repr__(self) -> str: ...
    def _repr_html_(self) -> str: ...  # Jupyter display

    # === Core Geometry (delegated to layout) ===
    @property
    def connectivity(self) -> nx.Graph: ...
    @property
    def bin_centers(self) -> np.ndarray: ...
    @property
    def n_bins(self) -> int: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def is_1d(self) -> bool: ...
    @property
    def dimension_ranges(self) -> list[tuple[float, float]]: ...
    @property
    def layout_type(self) -> str: ...
    @property
    def layout_parameters(self) -> dict: ...

    # === Fundamental Verbs (delegated to layout) ===
    def bin_at(self, point) -> int: ...
    def contains(self, point) -> bool: ...
    def neighbors(self, idx) -> list[int]: ...
    def to_linear(self, point) -> float: ...      # 1D only
    def linear_to_nd(self, x) -> np.ndarray: ...  # 1D only
```

**Total: ~25 public methods/properties** (down from ~70)

---

## Complete Migration Table

### Visualization (→ `animation/`)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.animate_fields(...)` | `animate_fields(env, ...)` | `from neurospatial.animation import animate_fields` |
| `env.plot(...)` | `plot_environment(env, ...)` | `from neurospatial.animation import plot_environment` |
| `env.plot_1d(...)` | `plot_1d_environment(env, ...)` | `from neurospatial.animation import plot_1d_environment` |
| `env.plot_field(...)` | `plot_field(env, field, ...)` | `from neurospatial.animation import plot_field` |

### Behavior (→ `behavior/`)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.occupancy(...)` | `compute_occupancy(env, ...)` | `from neurospatial.behavior import compute_occupancy` |
| `env.bin_sequence(...)` | `compute_bin_sequence(env, ...)` | `from neurospatial.behavior import compute_bin_sequence` |
| `env.transitions(...)` | `compute_transition_matrix(env, ...)` | `from neurospatial.behavior import compute_transition_matrix` |
| `env._empirical_transitions()` | `_empirical_impl()` | Internal to `behavior/transitions.py` |
| `env._random_walk_transitions()` | `_random_walk_impl()` | Internal to `behavior/transitions.py` |
| `env._diffusion_transitions()` | `_diffusion_impl()` | Internal to `behavior/transitions.py` |
| `env._allocate_time_linear()` | `_allocate_time_linear()` | Internal to `behavior/trajectory.py` |
| `env._compute_ray_grid_intersections()` | `_compute_ray_grid_intersections()` | Internal to `behavior/trajectory.py` |
| `env._position_to_flat_index()` | `_position_to_flat_index()` | Internal to `behavior/trajectory.py` |

### Fields / Smoothing (→ `ops/smoothing.py`)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.smooth(...)` | `smooth_field(env, field, ...)` | `from neurospatial.ops import smooth_field` |
| `env.compute_kernel(...)` | `compute_kernel(env, ...)` | `from neurospatial.ops import compute_kernel` |
| `env.interpolate(...)` | `interpolate_field(env, field, ...)` | `from neurospatial.ops import interpolate_field` |
| `env._interpolate_nearest()` | `_interpolate_nearest_impl()` | Internal to `ops/smoothing.py` |
| `env._interpolate_linear()` | `_interpolate_linear_impl()` | Internal to `ops/smoothing.py` |

### Calculus (→ `ops/calculus.py` - ALREADY EXISTS)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.differential_operator` | `compute_differential_operator(env)` | `from neurospatial.ops import compute_differential_operator` |

**Note**: `gradient()` and `divergence()` already exist in `ops/calculus.py`.

### Graph Algorithms (→ `ops/graph.py` - EXTEND)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.path_between(...)` | `compute_shortest_path(env, ...)` | `from neurospatial.ops import compute_shortest_path` |
| `env.reachable_from(...)` | `compute_reachable_bins(env, ...)` | `from neurospatial.ops import compute_reachable_bins` |
| `env.components(...)` | `compute_connected_components(env)` | `from neurospatial.ops import compute_connected_components` |
| `env.rings(...)` | `compute_graph_rings(env, ...)` | `from neurospatial.ops import compute_graph_rings` |

### Distance (→ `ops/distance.py` - EXTEND)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.distance_between(...)` | `compute_geodesic_distance(env, ...)` | `from neurospatial.ops import compute_geodesic_distance` |
| `env.distance_to(...)` | `compute_distance_to_region(env, ...)` | `from neurospatial.ops import compute_distance_to_region` |

**Note**: `geodesic_distance_matrix()`, `distance_field()`, `pairwise_distances()` already exist.

### Queries (→ KEEP on Environment)

| Current Method | Status | Rationale |
|----------------|--------|-----------|
| `env.bin_at(...)` | **KEEP** | Core geometry query |
| `env.contains(...)` | **KEEP** | Core geometry query |
| `env.neighbors(...)` | **KEEP** | Core geometry query |
| `env.bin_center_of(...)` | **KEEP** | Core geometry query |
| `env.bin_sizes` | **KEEP** | Core geometry property |

### Metrics (→ `ops/metrics.py` NEW or KEEP as properties)

| Current Method | New Location | Import Path |
|----------------|--------------|-------------|
| `env.boundary_bins` | `compute_boundary_bins(env)` | `from neurospatial.ops import compute_boundary_bins` |
| `env.bin_attributes` | `compute_bin_attributes(env)` | `from neurospatial.ops import compute_bin_attributes` |
| `env.edge_attributes` | `compute_edge_attributes(env)` | `from neurospatial.ops import compute_edge_attributes` |
| `env.linearization_properties` | **KEEP** as property | 1D-specific, tied to layout |
| `env.to_linear(...)` | **KEEP** | Core 1D geometry |
| `env.linear_to_nd(...)` | **KEEP** | Core 1D geometry |

### Regions (→ `regions/` - ADD functions)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.bins_in_region(...)` | `get_bins_in_region(env, ...)` | `from neurospatial.regions import get_bins_in_region` |
| `env.mask_for_region(...)` | `get_region_mask(env, ...)` | `from neurospatial.regions import get_region_mask` |
| `env.region_membership(...)` | `get_region_membership(env)` | `from neurospatial.regions import get_region_membership` |
| `env.region_mask(...)` | `get_all_region_masks(env)` | `from neurospatial.regions import get_all_region_masks` |

### Transforms (→ `ops/transforms.py` - EXTEND)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.subset(...)` | `subset_environment(env, ...)` | `from neurospatial.ops import subset_environment` |
| `env.rebin(...)` | `rebin_environment(env, ...)` | `from neurospatial.ops import rebin_environment` |
| `env.apply_transform(...)` | `apply_transform_to_environment(env, ...)` | `from neurospatial.ops import apply_transform_to_environment` |

**Note**: `apply_transform_to_environment` may already exist in `ops/transforms.py`.

### Serialization (→ `io/`)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.to_file(...)` | `save_environment(env, path)` | `from neurospatial.io import save_environment` |
| `env.to_dict()` | `environment_to_dict(env)` | `from neurospatial.io import environment_to_dict` |
| `Environment.from_file(...)` | `load_environment(path)` | `from neurospatial.io import load_environment` |
| `Environment.from_dict(...)` | `environment_from_dict(data)` | `from neurospatial.io import environment_from_dict` |
| `env.save(...)` | `save_environment_pickle(env, path)` | `from neurospatial.io import save_environment_pickle` |
| `Environment.load(...)` | `load_environment_pickle(path)` | `from neurospatial.io import load_environment_pickle` |
| `env.to_nwb(...)` | `save_environment_nwb(env, nwb)` | `from neurospatial.io import save_environment_nwb` |
| `Environment.from_nwb(...)` | `load_environment_nwb(nwb)` | `from neurospatial.io import load_environment_nwb` |

### Cache Management (→ `ops/cache.py` NEW)

| Current Method | New Function | Import Path |
|----------------|--------------|-------------|
| `env.clear_cache()` | `clear_all_caches(env)` | `from neurospatial import clear_all_caches` |
| `env._clear_explicit_caches()` | `clear_kernel_cache(env)` | `from neurospatial.ops import clear_kernel_cache` |

---

## Phase 1: Extend Existing ops/ (NOT create)

### Step 1: Audit and extend ops/graph.py

1. **Read existing** `ops/graph.py` (currently: `convolve`, `neighbor_reduce`)
2. **Add** these functions from `environment/queries.py`:

   ```python
   def compute_shortest_path(env, source, target) -> list[int]: ...
   def compute_reachable_bins(env, source, max_distance) -> NDArray: ...
   def compute_connected_components(env) -> list[set[int]]: ...
   def compute_graph_rings(env, center, max_rings) -> list[set[int]]: ...
   ```

3. **Update** `ops/__init__.py` to export new functions
4. **Test**: `uv run pytest tests/ops/test_graph.py`

### Step 2: Extend ops/distance.py

1. **Read existing** `ops/distance.py` (has `geodesic_distance_matrix`, `distance_field`)
2. **Add** from `environment/queries.py`:

   ```python
   def compute_distance_to_region(env, region_name) -> NDArray: ...
   ```

3. **Update** `ops/__init__.py`
4. **Test**: `uv run pytest tests/ops/test_distance.py`

### Step 3: Extend ops/smoothing.py

1. **Read existing** `ops/smoothing.py` (has `compute_diffusion_kernels`, `apply_kernel`)
2. **Add** from `environment/fields.py`:

   ```python
   def smooth_field(env, field, bandwidth, method="diffusion") -> NDArray: ...
   def compute_kernel(env, bandwidth, mode="transition") -> NDArray: ...
   def interpolate_field(env, field, points, method="nearest") -> NDArray: ...
   ```

3. **Update** `ops/__init__.py`
4. **Test**: `uv run pytest tests/ops/test_smoothing.py`

### Step 4: Extend ops/transforms.py

1. **Read existing** `ops/transforms.py` (has `Affine2D`, calibration functions)
2. **Add** from `environment/transforms.py`:

   ```python
   def subset_environment(env, bins) -> Environment: ...
   def rebin_environment(env, new_bin_size) -> Environment: ...
   ```

3. **Update** `ops/__init__.py`
4. **Test**: `uv run pytest tests/ops/test_transforms.py`

### Step 5: Create ops/cache.py (NEW)

```python
# ops/cache.py
def clear_kernel_cache(env: Environment) -> None: ...
def clear_all_caches(env: Environment) -> None: ...
```

### Step 6: Create ops/metrics.py (NEW)

Move from `environment/metrics.py`:

```python
def compute_boundary_bins(env) -> NDArray: ...
def compute_bin_attributes(env) -> pd.DataFrame: ...
def compute_edge_attributes(env) -> pd.DataFrame: ...
```

---

## Phase 2: Create New Modules

### Step 7: Create behavior/ module

```
src/neurospatial/behavior/
├── __init__.py
├── trajectory.py    # compute_occupancy, compute_bin_sequence
└── transitions.py   # compute_transition_matrix
```

### Step 8: Create animation/ module

```
src/neurospatial/animation/
├── __init__.py
├── animate.py       # animate_fields
├── plot.py          # plot_environment, plot_1d_environment, plot_field
└── overlays.py      # Move from visualization/overlays.py
```

### Step 9: Create io/ module

```
src/neurospatial/io/
├── __init__.py
├── file.py          # save_environment, load_environment, dict conversion
└── nwb.py           # NWB integration
```

### Step 10: Create factories/ module

```
src/neurospatial/factories/
├── __init__.py
├── from_samples.py
├── from_polygon.py
├── from_graph.py
├── from_mask.py
├── from_image.py
├── from_layout.py
└── from_nwb.py      # Delegates to io/nwb.py
```

---

## Phase 3: Slim Environment Class

### Step 11: Create slim Environment

1. Create new `environment/core.py` with frozen dataclass
2. Remove mixin imports from `__init__.py`
3. Add factory method facades
4. Keep only core methods/properties (~25 total)
5. Keep `_kernel_cache` for caching

### Step 12: Update regions/ module

Add standalone functions:

```python
def get_bins_in_region(env, region_name) -> NDArray: ...
def get_region_mask(env, region_name) -> NDArray: ...
def get_region_membership(env) -> dict: ...
def get_all_region_masks(env) -> dict: ...
```

---

## Phase 4: Integration

### Step 13: Update domain re-exports

```python
# encoding/__init__.py
from neurospatial.ops.smoothing import smooth_field, compute_kernel

# behavior/__init__.py
from neurospatial.behavior.trajectory import compute_occupancy, compute_bin_sequence
from neurospatial.behavior.transitions import compute_transition_matrix

# decoding/__init__.py
# (already has decode_position)
```

### Step 14: Update top-level **init**.py

```python
# neurospatial/__init__.py
from neurospatial.environment import Environment
from neurospatial.ops.cache import clear_all_caches

# Convenience re-exports for common operations
from neurospatial.animation import animate_fields, plot_environment
from neurospatial.behavior import compute_occupancy
```

### Step 15: Update tests

1. Update all test files to use new import paths
2. Add tests for new function signatures
3. Verify all edge cases covered

### Step 16: Update documentation

1. Update CLAUDE.md with new patterns
2. Update QUICKSTART.md examples
3. Update API_REFERENCE.md imports
4. Create MIGRATION.md guide

---

## Verification Checklist

After each step:

- [ ] `uv run pytest` - All tests pass
- [ ] `uv run ruff check .` - No linting errors
- [ ] `uv run mypy src/neurospatial/` - No type errors
- [ ] `uv run python -c "from neurospatial import Environment"` - Import works

Final verification:

- [ ] Environment class has ~25 public methods/properties
- [ ] All moved functions are importable from new locations
- [ ] Domain re-exports work (`from neurospatial.encoding import smooth_field`)
- [ ] Factory methods work (`Environment.from_samples(...)`)
- [ ] Regions mutation works on frozen Environment
- [ ] Cache clearing works (`clear_all_caches(env)`)

---

## Risk Mitigation

### Risk: Breaking user code

**Mitigation**: No current users - clean break is acceptable.

### Risk: Missing method during migration

**Mitigation**: Complete audit performed (see migration table above - 70+ methods catalogued).

### Risk: Circular imports

**Mitigation**:

```python
# factories/from_samples.py
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurospatial.environment import Environment

def from_samples(...) -> Any:  # Use Any at runtime for return type
    from neurospatial.environment import Environment  # Import at call time
    return Environment(...)
```

### Risk: Performance regression from cache changes

**Mitigation**:

- Keep `env._kernel_cache` dict approach (proven, no hashability issues)
- Benchmark before/after on place field computation
- Add performance regression tests

### Risk: Frozen dataclass with mutable cache

**Mitigation**: Use `object.__setattr__` in `__post_init__` to set mutable fields on frozen dataclass:

```python
def __post_init__(self):
    object.__setattr__(self, '_kernel_cache', {})
```

---

## Success Criteria

1. **Environment is a dataclass**: `@dataclass(frozen=True)` with ~25 methods
2. **No mixins**: All mixin files deleted (queries.py, fields.py, trajectory.py, etc.)
3. **Functional API**: All operations are `function(env, ...)` not `env.method(...)`
4. **Tests pass**: Full test suite green
5. **Docs updated**: CLAUDE.md reflects new patterns
6. **Existing ops/ preserved**: Extended, not replaced
