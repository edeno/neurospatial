# Layout Engines

Layout engines define how continuous space is discretized into bins. This page helps you choose the right engine for your analysis.

## Quick Decision Guide

**Most common scenarios:**

1. **Standard open field experiment** → Use `Environment.from_samples()` (defaults to RegularGridLayout)
2. **Circular arena** → Use `Environment.from_polygon()` (ShapelyPolygonLayout)
3. **T-maze or track** → Use `Environment.from_graph()` (GraphLayout)
4. **Need uniform neighbor distances** → Use `layout_type="hexagonal"` (HexagonalLayout)

## Available Engines

### RegularGridLayout
**Standard rectangular/cuboid grids** - fastest and most common.

**When to use**: Default choice for rectangular environments
**Performance**: ⭐⭐⭐⭐⭐ (fastest)
**Memory**: ⭐⭐⭐⭐⭐ (most efficient)
**Setup complexity**: ⭐⭐⭐⭐⭐ (easiest)

```python
env = Environment.from_samples(positions, bin_size=2.5)
# or explicitly:
env = Environment.from_samples(positions, bin_size=2.5, layout_type="regular")
```

### HexagonalLayout
**Hexagonal tessellation** with uniform neighbor distances.

**When to use**: When analyzing directional patterns or need isotropic representation
**Performance**: ⭐⭐⭐⭐ (very fast)
**Memory**: ⭐⭐⭐⭐ (efficient, ~15% more bins)
**Setup complexity**: ⭐⭐⭐⭐ (easy)

```python
env = Environment.from_samples(positions, bin_size=2.5, layout_type="hexagonal")
```

**Trade-off**: All 6 neighbors equidistant, but requires 15% more bins for same coverage.

### GraphLayout
**1D linearized track** for mazes and structured environments.

**When to use**: T-maze, plus maze, linear track experiments
**Performance**: ⭐⭐ (slower lookups)
**Memory**: ⭐⭐ (higher usage)
**Setup complexity**: ⭐⭐ (requires graph definition)

```python
import networkx as nx

G = nx.Graph()
G.add_node(0, pos=(0, 0))
G.add_node(1, pos=(50, 0))
G.add_edge(0, 1, edge_id=0, distance=50.0)

env = Environment.from_graph(G, edge_order=[(0, 1)], bin_size=2.0)
```

**Key feature**: Converts 2D positions to 1D linear coordinates with `env.to_linear()`

### MaskedGridLayout
**Grid with active/inactive regions** - automatically used when `infer_active_bins=True`.

**When to use**: Sparse data, need to exclude walls/unvisited areas
**Performance**: ⭐⭐⭐⭐ (fast)
**Memory**: ⭐⭐⭐⭐⭐ (very efficient for sparse environments)
**Setup complexity**: ⭐⭐⭐ (automatic with parameter tuning)

```python
env = Environment.from_samples(
    positions,
    bin_size=2.5,
    infer_active_bins=True,
    dilate=True,
    fill_holes=True
)
```

### ShapelyPolygonLayout
**Grid bounded by polygon** for geometric arenas.

**When to use**: Circular, elliptical, or custom-shaped arenas
**Performance**: ⭐⭐⭐ (moderate)
**Memory**: ⭐⭐⭐⭐ (efficient)
**Setup complexity**: ⭐⭐⭐ (need polygon definition)

```python
from shapely.geometry import Point
import numpy as np

# Circular arena
theta = np.linspace(0, 2*np.pi, 100)
boundary = np.column_stack([50*np.cos(theta), 50*np.sin(theta)])
from shapely.geometry import Polygon
arena = Polygon(boundary)

env = Environment.from_polygon(arena, bin_size=2.5)
```

### TriangularMeshLayout
**Triangular tessellation** - alternative grid structure.

**When to use**: Alternative to hexagonal, specific geometric needs
**Performance**: ⭐⭐⭐ (moderate)
**Memory**: ⭐⭐⭐⭐ (efficient)
**Setup complexity**: ⭐⭐⭐⭐ (easy)

```python
env = Environment.from_samples(positions, bin_size=2.5, layout_type="triangular")
```

### ImageMaskLayout
**Binary image-based** boundaries from video analysis.

**When to use**: Extracting arena from video frames
**Performance**: ⭐⭐⭐ (moderate)
**Memory**: ⭐⭐⭐ (moderate)
**Setup complexity**: ⭐⭐ (requires image preprocessing)

```python
env = Environment.from_image(
    image_path="arena_mask.png",
    bin_size=2.5,
    dimension_ranges=[(0, 100), (0, 100)]
)
```

## Performance Comparison

Benchmark: 100x100 cm arena, 2.5 cm bins, 10,000 position lookups

| Engine | Setup (ms) | Lookups/sec | Memory (MB) | Bins |
|--------|-----------|-------------|-------------|------|
| Regular | 5 | 1,000,000 | 0.5 | 1,600 |
| Hexagonal | 8 | 900,000 | 0.6 | 1,840 |
| Masked | 15 | 800,000 | 0.3 | ~1,100 |
| Polygon | 50 | 600,000 | 0.6 | ~1,250 |
| Triangular | 10 | 700,000 | 0.6 | ~1,700 |
| Graph | 100 | 100,000 | 1.0 | 50 nodes |

**Practical impact**: For typical experiments (10k-100k positions), all engines complete in <1 second. Choose based on spatial structure needs, not performance.

## Choosing Bin Size

Bin size has more impact than engine choice for most analyses.

### Guidelines by Arena Size

**Rat (100x100 cm arena):**
- Coarse: 5-10 cm → 100-400 bins
- Standard: 2-5 cm → 400-2,500 bins
- Fine: 1-2 cm → 2,500-10,000 bins

**Mouse (40x40 cm arena):**
- Coarse: 2-4 cm → 100-400 bins
- Standard: 1-2 cm → 400-1,600 bins
- Fine: 0.5-1 cm → 1,600-6,400 bins

### Rule of Thumb

Target **10-100 samples per bin** for stable statistics:

```python
# Estimate good bin_size
n_samples = len(positions)
arena_area = 100 * 100  # cm²
target_samples_per_bin = 50

target_bins = n_samples / target_samples_per_bin
bin_size = np.sqrt(arena_area / target_bins)
```

## Decision Tree

```
What's your environment?
│
├─ Rectangular/square arena
│  ├─ Need direction-independent analysis? → Hexagonal
│  └─ Standard analysis → Regular (default)
│
├─ Circular/elliptical arena → Polygon
│
├─ Track/maze with branches → Graph
│
└─ Sparse data with walls/obstacles
   ├─ Have binary image → ImageMask
   ├─ Have polygon boundary → Polygon
   └─ Just position data → Masked (infer_active_bins=True)
```

## Common Patterns

### Pattern 1: Let neurospatial choose

```python
# Simplest - neurospatial picks appropriate engine
env = Environment.from_samples(positions, bin_size=2.5)
```

### Pattern 2: Explicit control

```python
# Choose specific engine when you know what you need
env = Environment.from_samples(
    positions,
    bin_size=2.5,
    layout_type="hexagonal"  # Override default
)
```

### Pattern 3: Complex boundaries

```python
# Use factory method for specific geometry
env = Environment.from_polygon(polygon, bin_size=2.5)
# or
env = Environment.from_image(image_path, bin_size=2.5, dimension_ranges=[(0,100), (0,100)])
```

## Summary

**For most users:**
- Start with `Environment.from_samples()` and default settings
- Adjust `bin_size` based on your data density
- Only change engine if you have specific geometric or analytical needs

**Key factors:**
1. **Environment shape** (rectangular vs. circular vs. track)
2. **Data density** (affects bin size, not engine)
3. **Analysis needs** (directional uniformity, 1D linearization)

## See Also

- [Environments Guide](environments.md): Using different factory methods
- [Examples](../examples/02_layout_engines.ipynb): Visual comparisons
- [API Reference](../api/neurospatial/layout/index.md)
