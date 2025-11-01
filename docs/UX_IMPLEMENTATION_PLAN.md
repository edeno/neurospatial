# UX Improvement Implementation Plan

**Based on**: UX Review dated 2025-11-01
**Target**: Transform neurospatial from NEEDS_POLISH to USER_READY
**Estimated Total Effort**: 5-6 days
**Priority**: High (blocks broader adoption)

---

## Overview

This implementation plan addresses the UX issues identified in the comprehensive review. Work is organized into three sprints with clear success criteria and measurable outcomes.

**Key Principles**:
- Fix critical blockers first (empty README, documentation bugs)
- Improve error messages systematically (WHAT/WHY/HOW pattern)
- Enhance discoverability (cross-references, selection guides)
- Maintain excellent existing patterns

---

## Sprint 1: Critical Fixes (Days 1-2)

**Goal**: Make library discoverable and fix blocking issues
**Duration**: 1-2 days
**Success Metric**: New user can install and create first Environment in < 10 minutes

### Task 1.1: Write Comprehensive README ⚠️ BLOCKER

**File**: `README.md`
**Effort**: 2-3 hours
**Priority**: 🔴 CRITICAL

**Requirements**:

1. **Project Overview** (2-3 paragraphs)
   - What is neurospatial?
   - Who is it for? (neuroscientists, spatial data analysts)
   - What problems does it solve?

2. **Key Features** (bulleted list)
   - N-dimensional spatial discretization
   - Multiple layout engines (regular grid, hexagonal, graph-based)
   - Connectivity graphs for path analysis
   - Region of interest (ROI) management
   - Alignment and transformation tools

3. **Installation**
   ```bash
   pip install neurospatial
   # or for development
   git clone https://github.com/edeno/neurospatial.git
   cd neurospatial
   uv sync
   ```

4. **Quickstart Example** (copy-pasteable)
   ```python
   import numpy as np
   from neurospatial import Environment

   # Simulate animal position data in a 100x100 cm arena
   positions = np.random.rand(1000, 2) * 100

   # Create environment with 5cm bins
   env = Environment.from_samples(
       data_samples=positions,
       bin_size=5.0,
       name="arena"
   )

   # Inspect the environment
   print(f"Dimensions: {env.n_dims}")
   print(f"Number of bins: {env.n_bins}")
   print(f"Spatial extent: {env.dimension_ranges}")

   # Find which bin contains a point
   point = np.array([50.0, 50.0])
   bin_idx = env.bin_at(point)
   print(f"Point {point} is in bin {bin_idx}")

   # Get neighbors of a bin
   neighbors = env.neighbors(bin_idx)
   print(f"Bin {bin_idx} has {len(neighbors)} neighbors")
   ```

5. **Core Concepts** (brief explanations)
   - **Bins/Nodes**: Discrete spatial units
   - **Active Bins**: Bins included in analysis (filtered by occupancy)
   - **Connectivity Graph**: NetworkX graph of spatial relationships
   - **Layout Engines**: Different discretization strategies

6. **Common Use Cases** (with links to examples)
   - Discretizing position tracking data
   - Creating environments from geometric boundaries
   - Analyzing spatial navigation on complex tracks
   - Computing shortest paths and distances

7. **Documentation Links**
   - API Reference (when available)
   - Examples gallery (when available)
   - Contributing guidelines

8. **Citation** (if applicable)
   ```bibtex
   @software{neurospatial,
     author = {Denovellis, Eric},
     title = {neurospatial: Spatial discretization for neuroscience},
     year = {2025},
     url = {https://github.com/edeno/neurospatial}
   }
   ```

**Acceptance Criteria**:
- [ ] README is 100-200 lines (sufficient depth, not overwhelming)
- [ ] Quickstart example runs without modification
- [ ] All links are valid
- [ ] Installation instructions tested on clean environment

---

### Task 1.2: Fix Regions.update() Documentation Bug

**File**: `src/neurospatial/regions/core.py`
**Effort**: 30 minutes
**Priority**: 🔴 CRITICAL

**Option A: Implement update() method** (RECOMMENDED)

```python
def update(self, key: str, value: Region) -> None:
    """Update an existing region, replacing its value.

    Parameters
    ----------
    key : str
        The name of the region to update. Must already exist.
    value : Region
        The new Region object. Its name must match the key.

    Raises
    ------
    KeyError
        If the region does not exist (use `regions[key] = value` to add new).
    ValueError
        If key does not match value.name.

    Examples
    --------
    >>> from neurospatial.regions import Region, Regions
    >>> regions = Regions()
    >>> regions.add("roi1", coords=[[0, 0], [1, 1]])
    >>> # Update the region
    >>> new_region = Region(name="roi1", coords=[[0, 0], [2, 2]], kind="polygon")
    >>> regions.update("roi1", new_region)
    """
    if key not in self._store:
        raise KeyError(
            f"Region {key!r} does not exist. Use 'regions[{key!r}] = value' "
            "to add a new region instead."
        )
    if key != value.name:
        raise ValueError(
            f"Key {key!r} must match Region.name {value.name!r}. "
            "Region objects are immutable and their name must match the key."
        )
    self._store[key] = value
```

**Option B: Fix error message** (if update() not desired)

```python
def __setitem__(self, key: str, value: Region) -> None:
    if key in self._store:
        raise KeyError(
            f"Region {key!r} already exists. To replace it, first delete it:\n"
            f"  del regions[{key!r}]  # Remove existing region\n"
            f"  regions[{key!r}] = new_region  # Add replacement"
        )
    if key != value.name:
        raise ValueError("Key must match Region.name")
    self._store[key] = value
```

**Recommendation**: Implement Option A (update() method) for full dict-like API

**Acceptance Criteria**:
- [ ] Method implemented with full docstring
- [ ] Error messages reference existing methods only
- [ ] Tests added for update() behavior
- [ ] Tests for error conditions (KeyError, ValueError)

---

### Task 1.3: Improve "No Active Bins Found" Error

**File**: `src/neurospatial/layout/engines/regular_grid.py` (line 144-146)
**Effort**: 1 hour
**Priority**: 🔴 CRITICAL

**Current Code**:
```python
if not np.any(self.active_mask):
    raise ValueError(
        "No active bins found. Check your data_samples and bin_size.",
    )
```

**Improved Code**:
```python
if not np.any(self.active_mask):
    # Gather diagnostic information
    diagnostics = []
    if data_samples is not None:
        data_range = np.ptp(data_samples, axis=0)
        diagnostics.append(f"  Data range per dimension: {data_range}")
        diagnostics.append(f"  Data min: {np.min(data_samples, axis=0)}")
        diagnostics.append(f"  Data max: {np.max(data_samples, axis=0)}")
    diagnostics.append(f"  bin_size: {bin_size}")
    diagnostics.append(f"  bin_count_threshold: {bin_count_threshold}")
    diagnostics.append(f"  Grid shape: {self.grid_shape}")

    # Calculate suggested bin_size
    if data_samples is not None:
        suggested_bin_size = np.array(bin_size) / 2
        diagnostics.append(f"\nSuggested bin_size: {suggested_bin_size}")

    diagnostics_str = "\n".join(diagnostics)

    raise ValueError(
        f"No active bins found. This typically means:\n"
        f"  1. bin_size is too large relative to your data range\n"
        f"  2. bin_count_threshold is too high\n"
        f"  3. data_samples fall outside inferred dimension_ranges\n"
        f"\n"
        f"Diagnostics:\n"
        f"{diagnostics_str}\n"
        f"\n"
        f"Solutions to try:\n"
        f"  - Reduce bin_size (currently {bin_size})\n"
        f"  - Set bin_count_threshold=0 (currently {bin_count_threshold})\n"
        f"  - Check that data_samples contains valid coordinates (no NaN/Inf)\n"
        f"  - Explicitly provide dimension_ranges if auto-inference fails"
    )
```

**Acceptance Criteria**:
- [ ] Error message shows actual data ranges
- [ ] Error message shows current parameter values
- [ ] Error message provides specific suggestions
- [ ] Calculation of diagnostics doesn't fail (handle None, edge cases)
- [ ] Test added that triggers this error and validates message content

---

### Task 1.4: Add "Active Bins" Definition

**File**: `src/neurospatial/environment.py` (Environment class docstring)
**Effort**: 20 minutes
**Priority**: 🔴 CRITICAL

**Location**: Add after Attributes section, before Methods

**Addition**:
```python
    Terminology
    -----------
    active bins : array
        Bins within the environment's discretized space that are included in the
        analysis. For grid layouts, these are bins containing sufficient data
        samples (controlled by `bin_count_threshold`) or inferred through
        morphological operations (`dilate`, `fill_holes`, `close_gaps`).
        Inactive bins are excluded from `bin_centers` and the `connectivity` graph.

        This filtering allows the environment to represent only the spatially
        relevant regions. For example, when analyzing animal behavior, active bins
        represent the area the animal actually explored, rather than the entire
        bounding box of the tracking arena.

        To include all bins regardless of data coverage, set `infer_active_bins=False`
        when creating the environment.
```

**Acceptance Criteria**:
- [ ] Definition added to Environment class docstring
- [ ] Cross-references infer_active_bins parameter
- [ ] Explains scientific motivation (not just technical detail)
- [ ] Docstring renders correctly in help() and documentation tools

---

### Task 1.5: Add bin_size Units Clarification

**Files**: All factory methods in `src/neurospatial/environment.py`
**Effort**: 30 minutes
**Priority**: 🟠 HIGH

**Affected Methods**:
- `from_samples()` (line 354)
- `from_polygon()` (line 520)
- `from_mask()` (line 575)
- `from_image()` (line 644)
- `from_graph()` (line 476)

**Enhancement Pattern**:
```python
bin_size : float or sequence of floats
    Length of each bin side. Units must match your input data coordinates.
    For example, if data_samples are in centimeters, bin_size=5.0 creates
    5cm bins.

    For RegularGrid: Side length of square bins.
    For Hexagonal: Hexagon width (flat-to-flat distance).

    Can be a single float (isotropic bins) or a sequence with one value per
    dimension (anisotropic bins).
```

**Acceptance Criteria**:
- [ ] All factory methods updated
- [ ] Hexagonal interpretation clarified with warning symbol
- [ ] Examples show units in comments
- [ ] Docstrings render correctly

---

## Sprint 2: Clarity Improvements (Days 3-4)

**Goal**: Reduce confusion and improve discoverability
**Duration**: 2-3 days
**Success Metric**: 80% reduction in "which method should I use?" questions

### Task 2.1: Add Factory Method Selection Guide

**File**: `src/neurospatial/environment.py` (Environment class docstring)
**Effort**: 30 minutes
**Priority**: 🟠 HIGH

**Location**: Add after class summary, before Attributes

**Addition**:
```python
    Factory Method Selection Guide
    -------------------------------
    Choose the factory method that matches your input data:

    **Have position/tracking data?**
        Use `from_samples(data, bin_size=5.0)`
        Most common for neuroscience: discretize animal position into spatial bins.
        Automatically infers spatial extent and filters bins by occupancy.

    **Have shapely polygon boundary?**
        Use `from_polygon(polygon, bin_size=5.0)`
        Define environment by arbitrary geometric boundary.
        Useful when experimental arena has specific shape (e.g., circular, L-shaped).

    **Have pre-computed boolean mask?**
        Use `from_mask(mask, bin_size=5.0)`
        Use existing mask array where True = active bin, False = inactive.
        Efficient when mask is already computed from other processing.

    **Have binary image (PNG/JPG)?**
        Use `from_image(image_path, bin_size=5.0)`
        Load environment from image file (white = active, black = inactive).
        Convenient for manually drawn environments or image-based analysis.

    **Have 1D track graph structure?**
        Use `from_graph(graph, edge_order, edge_spacing, bin_size)`
        Linearize complex track geometries (e.g., figure-8 maze, T-maze).
        Requires track-linearization package and pre-built graph structure.

    **Need custom layout engine?**
        Use `from_layout(kind, params, name)`
        Advanced: directly specify layout engine and parameters.
        For extending neurospatial with custom discretization strategies.

    See individual factory method docstrings for detailed parameter descriptions
    and usage examples.
```

**Acceptance Criteria**:
- [ ] Guide added to Environment class docstring
- [ ] Each method has clear use case description
- [ ] Ordered by frequency of use (most common first)
- [ ] Cross-references to individual method docstrings

---

### Task 2.2: Define Scientific Terms

**Files**: Multiple locations
**Effort**: 1 hour
**Priority**: 🟡 MEDIUM

**Locations to Update**:

1. **"Place fields"** in `src/neurospatial/alignment.py` (line 27)
   ```python
   # Current
   * Comparing probability distributions (e.g., place fields, occupancy maps)

   # Enhanced
   * Comparing probability distributions (e.g., place fields - spatial regions
     where a neuron preferentially fires, or occupancy maps - time spent in
     each location)
   ```

2. **"Geodesic distance"** in `src/neurospatial/environment.py` (line 1104)
   ```python
   # Current
   """Calculate the geodesic distance between two points in the environment.

   # Enhanced
   """Calculate the geodesic distance between two points in the environment.

   Geodesic distance is the shortest path along the environment's connectivity
   graph, following actual spatial relationships. This differs from Euclidean
   distance (straight-line "as the crow flies") when obstacles or boundaries
   prevent direct paths.
   ```

3. **"Linearization"** in `src/neurospatial/environment.py` (line 1212)
   ```python
   # Current
   """Convert N-D coordinates to linearized 1D positions.

   # Enhanced
   """Convert N-D coordinates to linearized 1D positions along the track path.

   Linearization maps spatial coordinates from a track environment (which may
   curve and branch in 2D/3D space) to a 1D distance along the track's path.
   This is useful for analyzing neural activity as a function of track position,
   where position "10cm" has consistent meaning regardless of 2D coordinates.

   Only available for 1D track environments created with `from_graph()`.
   ```

**Acceptance Criteria**:
- [ ] All scientific terms have brief definitions
- [ ] Definitions use accessible language (avoid jargon in definitions)
- [ ] Cross-references to related concepts where appropriate

---

### Task 2.3: Standardize Error Messages with Actual Values

**Files**: Multiple locations (~20 files)
**Effort**: 2-3 hours
**Priority**: 🟡 MEDIUM

**Pattern to Apply**:
```python
# Before
raise ValueError("parameter must be constraint")

# After
raise ValueError(f"parameter must be constraint (got {actual_value})")
```

**Files to Update** (in priority order):

1. `src/neurospatial/layout/helpers/utils.py`
   - Line 96: bin_size validation
   - Line 148: bin_count_threshold validation

2. `src/neurospatial/calibration.py`
   - Line 48-50: offset_px validation

3. `src/neurospatial/layout/helpers/regular_grid.py`
   - Parameter validations

4. `src/neurospatial/layout/helpers/hexagonal.py`
   - Parameter validations

5. `src/neurospatial/layout/helpers/graph.py`
   - Parameter validations

6. `src/neurospatial/layout/mixins.py`
   - State validation errors

**Script for Systematic Update**:
```python
# Create a helper script to find all simple ValueError patterns
import re
import pathlib

def find_improvable_errors(src_dir):
    """Find ValueError raises that don't show actual values."""
    pattern = re.compile(r'raise ValueError\(["\']([^"\']*must[^"\']*)["\']\)')
    results = []

    for pyfile in pathlib.Path(src_dir).rglob("*.py"):
        content = pyfile.read_text()
        for match in pattern.finditer(content):
            results.append({
                'file': pyfile,
                'message': match.group(1),
                'line': content[:match.start()].count('\n') + 1
            })

    return results

# Run: python find_errors.py
```

**Acceptance Criteria**:
- [ ] All parameter validation errors show actual values
- [ ] Format is consistent: `f"{param} must be {constraint} (got {value})"`
- [ ] Type information included where helpful: `(got {type(value).__name__}: {value})`
- [ ] Long values are truncated: `(got {str(value)[:100]}...)`

---

### Task 2.4: Add "See Also" Cross-References

**File**: `src/neurospatial/environment.py` (all factory methods)
**Effort**: 1 hour
**Priority**: 🟡 MEDIUM

**Pattern to Apply** (add before Examples section):

```python
    See Also
    --------
    from_polygon : Create environment from shapely polygon boundary
    from_mask : Create environment from pre-computed boolean mask
    from_layout : Advanced: create environment with custom layout engine
    Environment : Main class documentation and attribute reference
```

**Methods to Update**:
- `from_samples()` → reference polygon, mask, layout
- `from_polygon()` → reference samples, mask, image
- `from_mask()` → reference samples, polygon, image
- `from_image()` → reference mask, polygon, samples
- `from_graph()` → reference samples, layout
- `from_layout()` → reference all specialized methods

**Acceptance Criteria**:
- [ ] All factory methods have "See Also" sections
- [ ] Cross-references are bidirectional (A references B, B references A)
- [ ] Brief descriptions explain relationship
- [ ] NumPy docstring format maintained

---

### Task 2.5: Audit and Standardize None Defaults

**File**: `src/neurospatial/environment.py` (factory methods)
**Effort**: 1 hour
**Priority**: 🟡 MEDIUM

**Current Inconsistencies**:
- `from_samples()`: `bin_size=2.0` (default)
- `from_polygon()`: `bin_size=2.0` (default)
- `from_graph()`: `bin_size` (required, no default)
- `from_mask()`: `bin_size` (check current state)
- `from_image()`: `bin_size` (check current state)

**Decision**: Make bin_size required for all methods (explicit is better than implicit)

**Rationale**:
- bin_size=2.0 is arbitrary and may not suit user's data scale
- Forcing explicit specification prevents "forgot to set bin_size" bugs
- Scientific software should require explicit parameter choices

**Changes**:
```python
# Before
def from_samples(cls, data_samples, bin_size: float = 2.0, ...):

# After
def from_samples(cls, data_samples, bin_size: float, ...):
```

**Update Docstrings**:
```python
bin_size : float or sequence of floats
    Length of each bin side. **Required parameter** - must be explicitly
    specified to match your data's spatial scale.
```

**Update Examples**:
- All examples must show explicit bin_size
- Add comment explaining scale: `bin_size=5.0  # 5cm bins`

**Acceptance Criteria**:
- [ ] All factory methods have consistent parameter requirements
- [ ] Breaking change documented in docstrings
- [ ] All examples updated to include explicit bin_size
- [ ] Tests updated to provide explicit bin_size

---

## Sprint 3: Polish (Days 5-6)

**Goal**: Professional finish and convenience features
**Duration**: 1-2 days
**Success Metric**: Zero "how do I debug this?" questions on common errors

### Task 3.1: Add "Common Pitfalls" Sections

**File**: `src/neurospatial/environment.py`
**Effort**: 1 hour
**Priority**: 🟡 MEDIUM

**Methods to Enhance**:

1. **from_samples()** - Add before Examples section

```python
    Common Pitfalls
    ---------------
    **bin_size too large**
        If bin_size is large relative to your data range, you may get only one
        bin or a "No active bins found" error. As a rule of thumb, bin_size
        should be ≤ 10% of the data range per dimension.

        Example: If data spans 0-100cm, avoid bin_size > 10cm.

    **bin_count_threshold too high**
        Setting bin_count_threshold too high will filter out most or all bins.
        Start with bin_count_threshold=0 and increase gradually only if you
        need to remove sparsely sampled regions.

        Example: With 1000 samples and bin_count_threshold=100, only bins with
        ≥10% of all samples will be active.

    **Mismatched units**
        Ensure bin_size units match your data_samples. If data is in meters but
        you assume centimeters, bin_size=5.0 creates 5-meter bins (500cm), not
        5cm bins.

        Fix: Check data_samples units and scale bin_size accordingly.

    **Missing morphological operations**
        If your active region has holes or narrow gaps, the default behavior
        (no morphological operations) preserves these. Set fill_holes=True and
        close_gaps=True to smooth the region.

        Example: Animal briefly passed through a location but didn't spend time
        there - bin may be inactive unless you enable filling/closing.
```

2. **CompositeEnvironment.__init__()** in `src/neurospatial/composite.py`

```python
    Common Pitfalls
    ---------------
    **Dimension mismatch**
        All sub-environments must have the same n_dims. You cannot combine a
        2D environment with a 3D environment. Check that all data_samples used
        to create sub-environments have consistent dimensionality.

    **No bridge edges**
        If `add_bridge_edges=True` but no bridges are inferred, the sub-environments
        remain disconnected. This happens when environments don't have bins close
        enough to each other. Try increasing `max_distance_between_nodes` or
        manually specify bridges.

    **Overlapping bins**
        CompositeEnvironment does not check for spatial overlap between sub-environments.
        If bins from different sub-environments occupy the same physical space,
        they will be treated as distinct bins in the composite graph.
```

**Acceptance Criteria**:
- [ ] Common Pitfalls sections added to key methods
- [ ] Each pitfall includes explanation and fix
- [ ] Examples are concrete and actionable
- [ ] Ordered by frequency (most common first)

---

### Task 3.2: Add Visual Examples of Morphological Operations

**Location**: Documentation or notebooks
**Effort**: 2 hours
**Priority**: 🟢 LOW (can be deferred)

**Deliverable**: Jupyter notebook or documentation page

**Content**:

1. **Setup Example Data**
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from neurospatial import Environment

   # Create data with holes and gaps
   np.random.seed(42)
   positions = []
   # Dense region 1
   positions.append(np.random.rand(400, 2) * 30 + 10)
   # Dense region 2 (separated by gap)
   positions.append(np.random.rand(400, 2) * 30 + 60)
   # Sparse connections
   positions.append(np.random.rand(50, 2) * np.array([80, 10]) + np.array([10, 45]))
   positions = np.vstack(positions)
   ```

2. **Visualize Each Operation**
   - Figure 1: Original (no operations)
   - Figure 2: With dilate=True
   - Figure 3: With fill_holes=True
   - Figure 4: With close_gaps=True
   - Figure 5: With all operations combined

3. **Code for Each Figure**
   ```python
   fig, axes = plt.subplots(2, 3, figsize=(15, 10))

   configs = [
       ("Original", {}),
       ("Dilate", {"dilate": True}),
       ("Fill Holes", {"fill_holes": True}),
       ("Close Gaps", {"close_gaps": True}),
       ("All Combined", {"dilate": True, "fill_holes": True, "close_gaps": True}),
   ]

   for ax, (title, ops) in zip(axes.flat, configs):
       env = Environment.from_samples(positions, bin_size=5.0, **ops)
       env.plot(ax=ax)
       ax.set_title(title)
   ```

**Acceptance Criteria**:
- [ ] Notebook runs without errors
- [ ] Visual differences are clear and labeled
- [ ] Each operation's effect is explained
- [ ] Link to notebook added in from_samples() docstring

---

### Task 3.3: Improve Type Validation for Sequences

**Files**: Multiple parameter validation locations
**Effort**: 1 hour
**Priority**: 🟢 LOW

**Pattern to Apply**:

```python
# Current (in layout helpers)
bin_size_arr = np.asarray(bin_size, dtype=float)

# Enhanced with explicit error handling
try:
    bin_size_arr = np.asarray(bin_size, dtype=float)
except (ValueError, TypeError) as e:
    raise TypeError(
        f"bin_size must contain numeric values (int or float), "
        f"got {type(bin_size).__name__}.\n"
        f"Received: {bin_size}\n"
        f"Ensure all elements are numbers, not strings or other types."
    ) from e

# Additional validation
if not np.all(np.isfinite(bin_size_arr)):
    raise ValueError(
        f"bin_size must contain finite values (no NaN or Inf), "
        f"got {bin_size_arr}"
    )
```

**Locations to Update**:
1. `layout/helpers/utils.py` - bin_size handling
2. `layout/helpers/regular_grid.py` - dimension_ranges handling
3. `environment.py` - data_samples validation

**Acceptance Criteria**:
- [ ] Type errors caught with helpful messages
- [ ] NaN/Inf validated separately
- [ ] Original exception preserved with `from e`
- [ ] Tests added for invalid inputs (strings, mixed types, NaN)

---

### Task 3.4: Add Custom __repr__ and .info() Methods

**File**: `src/neurospatial/environment.py`
**Effort**: 1 hour
**Priority**: 🟢 LOW

**Implementation**:

```python
def __repr__(self) -> str:
    """Return concise string representation of the environment.

    Returns
    -------
    str
        String showing key environment properties.

    Examples
    --------
    >>> env = Environment.from_samples(data, bin_size=5.0, name="arena")
    >>> env
    Environment(name='arena', n_dims=2, n_bins=256, layout='RegularGrid')
    """
    return (
        f"Environment(name={self.name!r}, n_dims={self.n_dims}, "
        f"n_bins={self.n_bins}, layout={self.layout._layout_type_tag!r})"
    )


def info(self) -> str:
    """Return detailed summary of this environment.

    Returns
    -------
    str
        Multi-line formatted summary of environment properties.

    Examples
    --------
    >>> env = Environment.from_samples(data, bin_size=5.0, name="arena")
    >>> print(env.info())
    Environment: arena
    Dimensions: 2
    Bins: 256
    Layout: RegularGrid
    Spatial extent: [(0.0, 100.0), (0.0, 100.0)]
    Bin sizes: [5.0, 5.0]
    1D linearized: False
    Regions: 0
    """
    ranges_str = ", ".join(
        f"({r[0]:.2f}, {r[1]:.2f})" for r in self.dimension_ranges
    )

    bin_sizes = self.layout.bin_sizes()
    if len(bin_sizes) == 0:
        bin_sizes_str = "N/A"
    else:
        # Get first bin size as representative (assuming uniform)
        bin_sizes_str = str(bin_sizes[0])

    return (
        f"Environment: {self.name or '(unnamed)'}\n"
        f"Dimensions: {self.n_dims}\n"
        f"Bins: {self.n_bins}\n"
        f"Layout: {self.layout._layout_type_tag}\n"
        f"Spatial extent: [{ranges_str}]\n"
        f"Bin sizes: {bin_sizes_str}\n"
        f"1D linearized: {self.is_1d}\n"
        f"Regions: {len(self.regions)}"
    )
```

**Acceptance Criteria**:
- [ ] __repr__ returns single-line summary
- [ ] info() returns multi-line detailed summary
- [ ] Both handle edge cases (empty name, no regions, etc.)
- [ ] Tests added for both methods
- [ ] Docstrings include examples

---

### Task 3.5: Add Validation Warnings for Unusual Parameters

**File**: `src/neurospatial/environment.py` (from_samples method)
**Effort**: 1 hour
**Priority**: 🟢 LOW (may be noisy, evaluate carefully)

**Implementation**:

```python
@classmethod
def from_samples(cls, data_samples, bin_size, bin_count_threshold=0, ...):
    """..."""
    data_samples = np.asarray(data_samples, dtype=float)

    # Validate and warn for unusual parameters
    data_range = np.ptp(data_samples, axis=0)
    bin_size_arr = np.atleast_1d(bin_size)

    # Warning 1: bin_size is large relative to data
    if np.any(bin_size_arr > data_range * 0.2):
        warnings.warn(
            f"bin_size {bin_size} is large (>20%) relative to data range "
            f"{data_range}. This will result in coarse discretization. "
            f"Consider reducing bin_size for finer spatial resolution.",
            UserWarning,
            stacklevel=2
        )

    # Warning 2: bin_count_threshold is high
    n_samples = len(data_samples)
    if bin_count_threshold > n_samples / 10:
        warnings.warn(
            f"bin_count_threshold={bin_count_threshold} is high relative to "
            f"sample count ({n_samples}). Many bins may be filtered out. "
            f"Consider reducing bin_count_threshold.",
            UserWarning,
            stacklevel=2
        )

    # Warning 3: No morphological operations with sparse data
    if (bin_count_threshold > 0 and
        not dilate and not fill_holes and not close_gaps and
        n_samples < 500):
        warnings.warn(
            f"You have sparse data ({n_samples} samples) with bin filtering "
            f"but no morphological operations. Active region may be fragmented. "
            f"Consider setting dilate=True, fill_holes=True, or close_gaps=True.",
            UserWarning,
            stacklevel=2
        )

    # Continue with normal processing...
```

**Decision**: Implement conservatively
- Only warn for clearly problematic parameter combinations
- Use `stacklevel=2` to show warning at user's call site
- Make warnings informative, not alarming
- Consider adding `suppress_warnings=False` parameter for advanced users

**Acceptance Criteria**:
- [ ] Warnings are helpful, not noisy
- [ ] stacklevel set correctly (points to user code)
- [ ] Can be suppressed with warnings.filterwarnings()
- [ ] Tests verify warnings are issued correctly

---

## Testing Strategy

### Unit Tests (add alongside implementation)

For each task, add corresponding tests:

1. **README**: Manual verification (can copy-paste and run)

2. **Regions.update()**:
   - Test successful update
   - Test KeyError when region doesn't exist
   - Test ValueError when key doesn't match name

3. **Error messages**:
   - Test that each improved error is raised correctly
   - Test that error message contains expected diagnostic info
   - Use `pytest.raises()` with `match` parameter

4. **Docstring additions**:
   - Run doctests: `pytest --doctest-modules`
   - Verify rendering: `python -c "from neurospatial import Environment; help(Environment)"`

5. **Type validation**:
   - Test invalid types (strings, mixed types)
   - Test NaN/Inf handling
   - Test helpful error messages

6. **__repr__ and info()**:
   - Test with various environment configurations
   - Test edge cases (empty name, no regions)
   - Verify output format

### Integration Tests

Create new test file: `tests/test_ux_improvements.py`

```python
"""Integration tests for UX improvements."""
import numpy as np
import pytest
from neurospatial import Environment


def test_first_run_experience():
    """Simulate first-time user workflow from README."""
    # This should "just work" without surprises
    positions = np.random.rand(1000, 2) * 100
    env = Environment.from_samples(
        data_samples=positions,
        bin_size=5.0,
        name="arena"
    )
    assert env.n_dims == 2
    assert env.n_bins > 0

    # Should be able to query immediately
    point = np.array([50.0, 50.0])
    bin_idx = env.bin_at(point)
    assert bin_idx is not None


def test_error_messages_are_helpful():
    """Verify error messages follow WHAT/WHY/HOW pattern."""
    positions = np.random.rand(100, 2) * 100

    # Trigger "no active bins" error
    with pytest.raises(ValueError) as exc_info:
        Environment.from_samples(
            data_samples=positions,
            bin_size=1000.0,  # Way too large
        )

    error_msg = str(exc_info.value)
    # WHAT: mentions "no active bins"
    assert "No active bins" in error_msg
    # WHY: explains likely causes
    assert "bin_size" in error_msg
    # HOW: suggests solutions
    assert "Reduce" in error_msg or "Try" in error_msg


def test_factory_method_discovery():
    """Users can find the right factory method."""
    # Check that selection guide exists
    help_text = Environment.__doc__
    assert "Factory Method Selection Guide" in help_text
    assert "from_samples" in help_text
    assert "from_polygon" in help_text
```

### Manual QA Checklist

Before marking sprints complete, verify:

**Sprint 1**:
- [ ] Install fresh environment and follow README quickstart
- [ ] Time how long it takes (should be < 10 minutes)
- [ ] README example runs without modification
- [ ] Trigger "no active bins" error and verify message is helpful
- [ ] Try to update a region and verify it works

**Sprint 2**:
- [ ] Read factory method guide and verify it's clear which to use
- [ ] Check that scientific terms are defined when first used
- [ ] Trigger 5 different errors and verify all show actual values
- [ ] Use help(Environment.from_samples) and verify See Also section

**Sprint 3**:
- [ ] Read Common Pitfalls and verify they're actionable
- [ ] Create environment and check __repr__ output
- [ ] Call env.info() and verify output is readable
- [ ] Provide invalid bin_size and verify error is clear

---

## Success Metrics

### Quantitative Metrics

**Before UX Improvements**:
- Time to first successful Environment creation: Unknown (no README)
- Error messages with diagnostic info: ~40%
- Methods with "See Also" sections: 0%
- Docstrings with "Common Pitfalls": 0%

**After Sprint 1** (Critical Fixes):
- ✅ Time to first successful Environment creation: < 10 minutes
- ✅ Error messages with diagnostic info: > 60%
- ✅ README exists and is comprehensive
- ✅ Zero documentation bugs

**After Sprint 2** (Clarity):
- ✅ Time to first successful Environment creation: < 5 minutes
- ✅ Error messages with diagnostic info: > 80%
- ✅ Methods with "See Also" sections: 100% (factory methods)
- ✅ Scientific terms defined: 100%

**After Sprint 3** (Polish):
- ✅ Time to first successful Environment creation: < 3 minutes
- ✅ Error messages with diagnostic info: > 90%
- ✅ Docstrings with "Common Pitfalls": 100% (key methods)
- ✅ Convenience methods available: __repr__, info()

### Qualitative Metrics

**User Feedback Questions**:
1. "How easy was it to get started?" → Target: "Very easy"
2. "When you encountered errors, were they helpful?" → Target: "Yes, I knew what to fix"
3. "Could you find the right method to use?" → Target: "Yes, the guide was clear"
4. "Did you understand what the library does?" → Target: "Yes, README explained it well"

**Internal Quality**:
- Code review approval for all PRs
- No regressions in existing tests
- New tests achieve >95% coverage of new code
- Documentation renders correctly in all tools (help(), Sphinx, etc.)

---

## Risk Management

### Potential Issues and Mitigation

**Risk 1: Breaking Changes from Removing Defaults**
- **Impact**: Task 2.5 removes bin_size=2.0 default
- **Mitigation**:
  - Document as breaking change
  - Add deprecation warning in intermediate version
  - Update all examples and tests
- **Alternative**: Keep defaults but add prominent warning in docstring

**Risk 2: Verbose Error Messages**
- **Impact**: Improved errors may be too long/overwhelming
- **Mitigation**:
  - Keep core message on first line
  - Use indentation for details
  - Test with real users
- **Alternative**: Add `verbose_errors=True` parameter

**Risk 3: Validation Warnings Too Noisy**
- **Impact**: Task 3.5 warnings may annoy power users
- **Mitigation**:
  - Conservative thresholds (only warn for clearly wrong)
  - Easy to suppress (standard warnings module)
  - Consider making opt-in
- **Alternative**: Skip this task if too risky

**Risk 4: Time Estimation**
- **Impact**: Work takes longer than estimated
- **Mitigation**:
  - Prioritize critical tasks (Sprint 1) first
  - Sprint 2-3 can be deferred if needed
  - Track actual time and adjust
- **Alternative**: Split into more sprints

---

## Rollout Plan

### Phase 1: Internal Review (Before Sprint 1)
- [ ] Review implementation plan with maintainer
- [ ] Agree on breaking changes (bin_size default removal)
- [ ] Confirm priority ordering
- [ ] Set up tracking (GitHub issues, project board)

### Phase 2: Development (Sprints 1-3)
- [ ] Create feature branch: `feature/ux-improvements`
- [ ] Implement tasks in order
- [ ] Add tests alongside implementation
- [ ] Run full test suite after each task
- [ ] Commit frequently with clear messages

### Phase 3: Review and Polish
- [ ] Self-review using QA checklist
- [ ] Run manual tests with fresh environment
- [ ] Create draft PR for each sprint
- [ ] Address review feedback
- [ ] Update CHANGELOG.md

### Phase 4: Documentation
- [ ] Ensure README renders correctly on GitHub
- [ ] Verify help() output is readable
- [ ] Check that examples run
- [ ] Update any external documentation

### Phase 5: Release
- [ ] Merge feature branch to main
- [ ] Tag release (suggest v0.2.0 for breaking changes)
- [ ] Update PyPI package
- [ ] Announce improvements in release notes

---

## Maintenance Plan

### Post-Release Monitoring

**Week 1**: Watch for issue reports
- Monitor GitHub issues for confusion
- Check if error messages are actually helpful
- Gather feedback on README clarity

**Month 1**: Collect metrics
- Time-to-first-success for new users
- Most common error messages encountered
- Questions about factory method selection

**Quarter 1**: Iterate
- Address any emergent issues
- Refine error messages based on real usage
- Add examples for common workflows

### Long-Term Improvements

**Future Enhancements** (beyond this plan):
1. Interactive tutorial (Jupyter notebook)
2. Examples gallery with real neuroscience data
3. Video walkthrough
4. API reference documentation (Sphinx)
5. Contribution guide for custom layout engines

---

## Appendix: File Change Summary

### Files to Modify

**High Priority**:
- [ ] `README.md` - Complete rewrite
- [ ] `src/neurospatial/environment.py` - Major docstring additions
- [ ] `src/neurospatial/regions/core.py` - Add update() method
- [ ] `src/neurospatial/layout/engines/regular_grid.py` - Improve error
- [ ] `src/neurospatial/layout/helpers/utils.py` - Multiple error improvements

**Medium Priority**:
- [ ] `src/neurospatial/alignment.py` - Define place fields
- [ ] `src/neurospatial/composite.py` - Improve error, add pitfalls
- [ ] `src/neurospatial/calibration.py` - Show actual values in errors
- [ ] All layout helpers - Standardize error messages

**Low Priority**:
- [ ] `docs/examples/morphological_operations.ipynb` - New file
- [ ] `tests/test_ux_improvements.py` - New file
- [ ] Various test files - Update for breaking changes

### Estimated Line Changes

| Sprint | Files Modified | Lines Added | Lines Removed | Net Change |
|--------|---------------|-------------|---------------|------------|
| 1 | 5 | ~350 | ~50 | +300 |
| 2 | 10 | ~400 | ~100 | +300 |
| 3 | 8 | ~300 | ~50 | +250 |
| **Total** | **23** | **~1050** | **~200** | **~850** |

---

## Conclusion

This implementation plan provides a **systematic path from NEEDS_POLISH to USER_READY** status. The work is front-loaded with critical fixes (Sprint 1) that immediately unblock users, followed by clarity improvements (Sprint 2) and polish (Sprint 3).

**Key Principles**:
- ✅ Fix critical blockers first
- ✅ Maintain existing excellent patterns
- ✅ Every error message answers WHAT/WHY/HOW
- ✅ Documentation is comprehensive and discoverable
- ✅ Changes are testable and measurable

**Expected Outcome**: After completing this plan, neurospatial will be a **user-ready library** with excellent documentation, helpful error messages, and clear guidance for new users. The foundation is already excellent - this plan adds the final layer of polish that makes the library truly accessible.

**Timeline**: 5-6 days of focused work, can be split across multiple weeks if needed.

---

**Next Steps**:
1. Review this plan with maintainer
2. Create GitHub issues for each task
3. Begin Sprint 1 (Critical Fixes)
4. Ship early, ship often (merge after each sprint)

**Questions or Concerns**: Contact [maintainer] or open GitHub issue with label `ux-improvement`
