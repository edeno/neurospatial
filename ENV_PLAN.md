# Environment Refactoring Plan

**Goal**: Transform Environment from a "God Object" (~200 methods via mixins) to a "Data Container" (<15 methods/attributes) following Rhodes-Hettinger design principles.

**Status**: Pending (execute after PLAN.md completion)
**Breaking Change**: Yes - Clean break, no deprecation period
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
   - `from neurospatial.behavior import occupancy`
   - Preserves discoverability within scientific domains

4. **`ops` as Subpackage**
   - Mathematical primitives in `neurospatial/ops/`
   - Top-level namespaces reserved for scientific domains

5. **Cache Strategy**
   - Delete `env.clear_cache()` method
   - Kernel caching via `@lru_cache` on `compute_kernel()` function
   - KDTree as lazy property on `LayoutEngine`

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

    # === Identity & Introspection ===
    def info(self) -> str: ...
    def copy(self, deep: bool = True) -> "Environment": ...
    def __repr__(self) -> str: ...

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

    # === Fundamental Verbs (delegated to layout) ===
    def bin_at(self, point) -> int: ...
    def contains(self, point) -> bool: ...
    def neighbors(self, idx) -> list[int]: ...
    def to_linear(self, point) -> float: ...      # 1D only
    def linear_to_nd(self, x) -> np.ndarray: ...  # 1D only
```

**Total: ~15 methods/properties** (down from ~200)

---

## Migration Map

### Phase 1: Create New Module Structure

Create the following directory structure:

```
src/neurospatial/
├── __init__.py                    # Update exports
├── environment/
│   ├── __init__.py
│   └── core.py                    # Slim Environment dataclass
├── factories/
│   ├── __init__.py                # Re-export all factory functions
│   ├── from_samples.py
│   ├── from_polygon.py
│   ├── from_graph.py
│   ├── from_mask.py
│   └── from_image.py
├── ops/
│   ├── __init__.py                # Re-export common operations
│   ├── graph.py                   # shortest_path, reachable_from, connected_components, rings
│   ├── distance.py                # geodesic_distance, distance_to_region
│   ├── smoothing.py               # smooth_field, compute_kernel
│   ├── calculus.py                # gradient, divergence, laplacian
│   └── transforms.py              # subset_environment, rebin_environment
├── animation/
│   ├── __init__.py
│   ├── animate.py                 # animate_fields
│   ├── plot.py                    # plot_environment, plot_1d_environment
│   └── overlays.py                # Existing overlay classes
├── behavior/
│   ├── __init__.py
│   ├── trajectory.py              # compute_occupancy, bin_sequence
│   └── transitions.py             # empirical_transition_matrix
├── io/
│   ├── __init__.py
│   ├── file.py                    # to_file, from_file, to_dict, from_dict
│   └── nwb.py                     # NWB integration (optional)
├── encoding/                      # Already exists - add re-exports
│   └── __init__.py                # Add: smooth_field, compute_kernel
├── decoding/                      # Already exists - add re-exports
│   └── __init__.py                # Add: decode_position
└── regions/                       # Already exists - no changes needed
```

### Phase 2: Method Migration Table

| Current Method | New Location | New Signature |
|----------------|--------------|---------------|
| **Visualization** | | |
| `env.animate_fields()` | `animation.animate_fields` | `animate_fields(env, fields, frame_times, ...)` |
| `env.plot()` | `animation.plot_environment` | `plot_environment(env, field=None, ...)` |
| `env.plot_1d()` | `animation.plot_1d_environment` | `plot_1d_environment(env, field=None, ...)` |
| **Behavior** | | |
| `env.occupancy()` | `behavior.compute_occupancy` | `compute_occupancy(env, positions, times, ...)` |
| `env.bin_sequence()` | `behavior.bin_sequence` | `bin_sequence(env, positions)` |
| `env._empirical_transition_matrix()` | `behavior.empirical_transition_matrix` | `empirical_transition_matrix(env, positions)` |
| **Smoothing** | | |
| `env.smooth_field()` | `ops.smooth_field` | `smooth_field(env, field, bandwidth, ...)` |
| `env.compute_kernel()` | `ops.compute_kernel` | `compute_kernel(env, bandwidth, ...)` |
| **Calculus** | | |
| `env.diffusion_operator` | `ops.compute_diffusion_operator` | `compute_diffusion_operator(env)` |
| **Graph** | | |
| `env.distance_matrix()` | `ops.geodesic_distance` | `geodesic_distance(env, source=None)` |
| `env.path_between()` | `ops.shortest_path` | `shortest_path(env, source, target)` |
| `env.distance_to_region()` | `ops.distance_to_region` | `distance_to_region(env, region_name)` |
| `env.reachable_from()` | `ops.reachable_from` | `reachable_from(env, source, max_distance)` |
| `env.connected_components()` | `ops.connected_components` | `connected_components(env)` |
| `env.rings()` | `ops.graph_rings` | `graph_rings(env, center, max_rings)` |
| `env._random_walk_matrix()` | `ops.random_walk_matrix` | `random_walk_matrix(env)` |
| **Regions** | | |
| `env.bins_in_region()` | `regions.get_bins_in_region` | `get_bins_in_region(env, region_name)` |
| `env.mask_for_region()` | `regions.get_region_mask` | `get_region_mask(env, region_name)` |
| `env.region_membership()` | `regions.get_region_membership` | `get_region_membership(env)` |
| **I/O** | | |
| `env.to_file()` | `io.to_file` | `to_file(env, path)` |
| `env.to_dict()` | `io.to_dict` | `to_dict(env)` |
| `Environment.from_file()` | `io.from_file` | `from_file(path) -> Environment` |
| `Environment.from_dict()` | `io.from_dict` | `from_dict(data) -> Environment` |
| **Transforms** | | |
| `env.subset()` | `ops.subset_environment` | `subset_environment(env, bins)` |
| `env.rebin()` | `ops.rebin_environment` | `rebin_environment(env, new_bin_size)` |
| **Deleted** | | |
| `env.clear_cache()` | N/A | Use `compute_kernel.cache_clear()` |
| `env._kernel_cache` | N/A | Internal to `ops.smoothing` |

### Phase 3: Implementation Order

Execute in dependency order (leaf modules first):

#### Step 1: Create `ops/` subpackage (no dependencies on other new modules)

1. Create `src/neurospatial/ops/__init__.py`
2. Create `src/neurospatial/ops/graph.py`
   - Move `shortest_path`, `reachable_from`, `connected_components`, `graph_rings`, `random_walk_matrix`
   - Each function takes `env` as first argument
3. Create `src/neurospatial/ops/distance.py`
   - Move `geodesic_distance`, `distance_to_region`
4. Create `src/neurospatial/ops/smoothing.py`
   - Move `smooth_field`, `compute_kernel`
   - Add `@lru_cache` to `compute_kernel` (replaces `env._kernel_cache`)
5. Create `src/neurospatial/ops/calculus.py`
   - Move `compute_diffusion_operator` (was property `diffusion_operator`)
6. Create `src/neurospatial/ops/transforms.py`
   - Move `subset_environment`, `rebin_environment`
7. Update `src/neurospatial/ops/__init__.py` with re-exports
8. **Test**: `uv run pytest tests/ops/` (create test directory)

#### Step 2: Create `behavior/` module

1. Create `src/neurospatial/behavior/__init__.py`
2. Create `src/neurospatial/behavior/trajectory.py`
   - Move `compute_occupancy`, `bin_sequence`
3. Create `src/neurospatial/behavior/transitions.py`
   - Move `empirical_transition_matrix`
4. **Test**: `uv run pytest tests/behavior/`

#### Step 3: Create `animation/` module

1. Create `src/neurospatial/animation/__init__.py`
2. Create `src/neurospatial/animation/animate.py`
   - Move `animate_fields` function
3. Create `src/neurospatial/animation/plot.py`
   - Move `plot_environment`, `plot_1d_environment`
4. Move existing `overlays.py` if not already there
5. **Test**: `uv run pytest tests/animation/`

#### Step 4: Create `io/` module

1. Create `src/neurospatial/io/__init__.py`
2. Create `src/neurospatial/io/file.py`
   - Move `to_file`, `from_file`, `to_dict`, `from_dict`
3. Move NWB integration if exists
4. **Test**: `uv run pytest tests/io/`

#### Step 5: Create `factories/` module

1. Create `src/neurospatial/factories/__init__.py`
2. Create individual factory files:
   - `from_samples.py`
   - `from_polygon.py`
   - `from_graph.py`
   - `from_mask.py`
   - `from_image.py`
3. **Test**: `uv run pytest tests/factories/`

#### Step 6: Update `regions/` module

1. Add `get_bins_in_region(env, region_name)` function
2. Add `get_region_mask(env, region_name)` function
3. Add `get_region_membership(env)` function
4. **Test**: `uv run pytest tests/regions/`

#### Step 7: Slim down Environment class

1. Create new `src/neurospatial/environment/core.py` with frozen dataclass
2. Remove all mixin imports
3. Add factory method facades (dispatch to `factories/`)
4. Keep only: `info()`, `copy()`, properties, fundamental verbs
5. Delete `env.clear_cache()` method
6. **Test**: `uv run pytest tests/test_environment.py`

#### Step 8: Update domain re-exports

1. Update `encoding/__init__.py`:
   - Add `from neurospatial.ops.smoothing import smooth_field, compute_kernel`
2. Update `behavior/__init__.py`:
   - Add `from neurospatial.behavior.trajectory import compute_occupancy`
3. Update `decoding/__init__.py`:
   - Ensure `decode_position` is exported

#### Step 9: Update top-level `__init__.py`

1. Export new modules: `ops`, `animation`, `behavior`, `io`, `factories`
2. Ensure backward-compatible imports still work for transition

#### Step 10: Update tests

1. Update all test files to use new import paths
2. Remove tests for deleted methods
3. Add tests for new function signatures

#### Step 11: Update documentation

1. Update CLAUDE.md with new patterns
2. Update QUICKSTART.md examples
3. Update API_REFERENCE.md imports
4. Create migration guide in docs/

---

## Verification Checklist

After each step, verify:

- [ ] `uv run pytest` - All tests pass
- [ ] `uv run ruff check .` - No linting errors
- [ ] `uv run mypy src/neurospatial/` - No type errors
- [ ] `uv run python -c "from neurospatial import Environment; print(Environment)"` - Import works

Final verification:

- [ ] Environment class has <15 public methods/properties
- [ ] All moved functions are importable from new locations
- [ ] Domain re-exports work (`from neurospatial.encoding import smooth_field`)
- [ ] Factory methods work (`Environment.from_samples(...)`)
- [ ] Regions mutation works on frozen Environment

---

## Risk Mitigation

### Risk: Breaking user code

**Mitigation**: Migration guide with exact old→new mappings (see table above)

### Risk: Missing method during migration

**Mitigation**: Use grep to find all `def` in current mixins before starting

### Risk: Circular imports

**Mitigation**:

- Factories import Environment only for type hints (use `TYPE_CHECKING`)
- Environment imports factories at call time (inside method body)

### Risk: Performance regression from removed caches

**Mitigation**:

- `@lru_cache` on `compute_kernel` matches current behavior
- Benchmark before/after on place field computation

---

## Success Criteria

1. **Environment is a dataclass**: `@dataclass(frozen=True)` with <15 methods
2. **No mixins**: All mixin files deleted
3. **Functional API**: All operations are `function(env, ...)` not `env.method(...)`
4. **Tests pass**: Full test suite green
5. **Docs updated**: CLAUDE.md reflects new patterns
