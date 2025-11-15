# neurospatial: Technical Implementation Details

This document explains **how** each capability in neurospatial is implemented, focusing on algorithms, data structures, and architectural patterns.

---

## Table of Contents

1. [Core Environment System](#1-core-environment-system)
2. [Spatial Discretization (Layout Engines)](#2-spatial-discretization-layout-engines)
3. [Place Field Analysis](#3-place-field-analysis)
4. [Boundary & Grid Cell Metrics](#4-boundary--grid-cell-metrics)
5. [Trajectory Analysis](#5-trajectory-analysis)
6. [Population-Level Metrics](#6-population-level-metrics)
7. [Spatial Operations](#7-spatial-operations)
8. [Distance & Graph Operations](#8-distance--graph-operations)
9. [Field Smoothing & Kernels](#9-field-smoothing--kernels)
10. [Segmentation & Trial Analysis](#10-segmentation--trial-analysis)
11. [Simulation & Synthetic Data](#11-simulation--synthetic-data)
12. [Environment Transformation & Alignment](#12-environment-transformation--alignment)
13. [Composite Environments](#13-composite-environments)
14. [Regions of Interest (ROIs)](#14-regions-of-interest-rois)
15. [Serialization & I/O](#15-serialization--io)

---

## 1. Core Environment System

### Architecture: Mixin Pattern with Protocol-Based Design

**File**: `src/neurospatial/environment/core.py`

```python
@dataclass
class Environment(
    EnvironmentFactories,      # Factory classmethods
    EnvironmentQueries,         # Spatial query methods
    EnvironmentSerialization,   # Save/load methods
    EnvironmentRegions,         # Region operations
    EnvironmentVisualization,   # Plotting methods
    EnvironmentMetrics,         # Metrics and properties
    EnvironmentFields,          # Spatial field operations
    EnvironmentTrajectory,      # Trajectory analysis
    EnvironmentTransforms,      # Rebin/subset operations
):
    """Main Environment class assembled from mixins."""
    name: str = ""
    layout: LayoutEngine | None = None
    bin_centers: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    connectivity: nx.Graph = field(default_factory=nx.Graph)
    regions: Regions = field(default_factory=Regions)
    # ... additional fields
```

**Key Implementation Details**:

1. **Only `Environment` is a `@dataclass`** - all mixins are plain classes to avoid field inheritance conflicts
2. **Mixin methods use `TYPE_CHECKING` guards**:
   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from neurospatial.environment.core import Environment

   class EnvironmentQueries:
       def bin_at(self: "Environment", points) -> NDArray[np.int_]:
           return self.layout.point_to_bin_index(points)
   ```
3. **Protocol-based validation** via `EnvironmentProtocol` ensures type safety
4. **Fitted state pattern**: `_is_fitted` flag + `@check_fitted` decorator

### Data Structure: The Three Core Attributes

```python
env.bin_centers: NDArray[np.float64]  # Shape (n_bins, n_dims)
# Continuous coordinates of each spatial bin center
# Example: [[1.0, 1.0], [1.0, 3.0], [3.0, 1.0], ...]

env.connectivity: nx.Graph
# Nodes: integers 0 to n_bins-1 (indices into bin_centers)
# Node attributes:
#   - 'pos': tuple of coordinates (matches bin_centers[i])
#   - 'source_grid_flat_index': original grid flat index
#   - 'original_grid_nd_index': original grid N-D index
# Edge attributes:
#   - 'distance': Euclidean distance between centers
#   - 'vector': displacement vector
#   - 'edge_id': unique edge identifier
#   - 'angle_2d': angle for 2D layouts (optional)

env.layout: LayoutEngine
# Protocol-based interface for spatial discretization
# Handles point-to-bin mapping via KDTree or grid indexing
```

### Factory Methods: Delegation to Layout Engines

**Implementation** (`src/neurospatial/environment/factories.py`):

```python
@classmethod
def from_samples(
    cls,
    samples: NDArray[np.float64],
    bin_size: float,
    **kwargs
) -> Environment:
    """Factory: discretize continuous position samples."""
    # 1. Create layout engine
    layout = RegularGridLayout()

    # 2. Build layout (populate bin_centers, connectivity)
    layout.build(
        samples=samples,
        bin_size=bin_size,
        infer_active_bins=True,  # Default
        **kwargs
    )

    # 3. Instantiate Environment from layout
    return cls.from_layout(layout)

@classmethod
def from_layout(cls, layout: LayoutEngine, **kwargs) -> Environment:
    """Factory: create Environment from pre-built layout."""
    env = cls(**kwargs)
    env._setup_from_layout(layout)
    return env
```

**`_setup_from_layout` implementation**:
```python
def _setup_from_layout(self, layout: LayoutEngine) -> None:
    """Copy layout attributes into Environment."""
    self.layout = layout
    self.bin_centers = layout.bin_centers
    self.connectivity = layout.connectivity
    self.dimension_ranges = layout.dimension_ranges
    # ... copy all layout attributes

    # Validate graph structure
    validate_connectivity_graph(self.connectivity)

    # Mark as fitted
    self._is_fitted = True
```

---

## 2. Spatial Discretization (Layout Engines)

### Protocol-Based Design

**File**: `src/neurospatial/layout/base.py`

All layout engines implement the `LayoutEngine` Protocol (not inheritance):

```python
@runtime_checkable
class LayoutEngine(Protocol):
    bin_centers: NDArray[np.float64]
    connectivity: nx.Graph
    dimension_ranges: Sequence[tuple[float, float]] | None
    grid_edges: tuple[NDArray[np.float64], ...] | None
    grid_shape: tuple[int, ...] | None
    active_mask: NDArray[np.bool_] | None
    _layout_type_tag: str
    _build_params_used: dict[str, Any]

    def build(self, **kwargs) -> None: ...
    def point_to_bin_index(self, points: NDArray) -> NDArray[np.int_]: ...
    def bin_sizes(self) -> NDArray[np.float64]: ...
```

### RegularGridLayout Implementation

**File**: `src/neurospatial/layout/engines/regular_grid.py`

**Algorithm**:
```python
def build(self, samples, bin_size, infer_active_bins=True, **kwargs):
    # 1. Compute data extent
    data_min = samples.min(axis=0)  # Shape (n_dims,)
    data_max = samples.max(axis=0)

    # 2. Create grid edges
    grid_edges = []
    for dim in range(n_dims):
        edges = np.arange(
            data_min[dim],
            data_max[dim] + bin_size,
            bin_size
        )
        grid_edges.append(edges)

    # 3. Compute bin centers via meshgrid
    centers_1d = [edges[:-1] + bin_size/2 for edges in grid_edges]
    mesh = np.meshgrid(*centers_1d, indexing='ij')
    bin_centers_full = np.stack(
        [m.ravel() for m in mesh], axis=1
    )  # Shape (n_full_bins, n_dims)

    # 4. Infer active bins (bins with data)
    if infer_active_bins:
        # Histogram samples into grid
        hist, _ = np.histogramdd(samples, bins=grid_edges)

        # Threshold by count
        active_mask = hist >= bin_count_threshold

        # Optional: morphological operations
        if dilate:
            active_mask = binary_dilation(active_mask)
        if fill_holes:
            active_mask = binary_fill_holes(active_mask)

        # Extract active bins
        active_indices = np.where(active_mask.ravel())[0]
        self.bin_centers = bin_centers_full[active_indices]
    else:
        # All bins active
        self.bin_centers = bin_centers_full
        active_mask = np.ones(grid_shape, dtype=bool)

    # 5. Build connectivity graph
    self.connectivity = _build_grid_connectivity(
        grid_shape, active_mask, bin_centers
    )
```

**Point-to-bin mapping** (KDTree):
```python
def point_to_bin_index(self, points):
    # Build KDTree from bin_centers (cached)
    if not hasattr(self, '_kdtree'):
        from scipy.spatial import KDTree
        self._kdtree = KDTree(self.bin_centers)

    # Query nearest neighbor
    distances, indices = self._kdtree.query(points)

    # Invalidate points outside max_distance threshold
    max_distance = bin_size * sqrt(n_dims) / 2
    indices[distances > max_distance] = -1

    return indices
```

### HexagonalLayout Implementation

**File**: `src/neurospatial/layout/engines/hexagonal.py`

**Algorithm**:
```python
def build(self, samples, bin_size, **kwargs):
    # 1. Hexagonal grid geometry
    # Horizontal spacing
    dx = bin_size
    # Vertical spacing (equilateral triangles)
    dy = bin_size * np.sqrt(3) / 2

    # 2. Generate hexagonal lattice
    x_coords = np.arange(x_min, x_max, dx)
    y_coords = np.arange(y_min, y_max, dy)

    centers = []
    for row_idx, y in enumerate(y_coords):
        # Offset every other row by dx/2
        x_offset = (dx / 2) if row_idx % 2 == 1 else 0
        for x in x_coords:
            centers.append([x + x_offset, y])

    self.bin_centers = np.array(centers)

    # 3. Build connectivity (6 neighbors per hex)
    # Use Delaunay triangulation to find adjacencies
    from scipy.spatial import Delaunay
    tri = Delaunay(self.bin_centers)

    # Extract unique edges from simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)

    # Build graph
    G = nx.Graph()
    for i, center in enumerate(self.bin_centers):
        G.add_node(i, pos=tuple(center), ...)
    for u, v in edges:
        distance = np.linalg.norm(
            self.bin_centers[v] - self.bin_centers[u]
        )
        # Keep only edges ≈ bin_size (6 nearest neighbors)
        if distance < bin_size * 1.2:
            G.add_edge(u, v, distance=distance, ...)
```

### GraphLayout (1D Linearization)

**File**: `src/neurospatial/layout/engines/graph.py`

**Requires**: `track-linearization` package

**Algorithm**:
```python
def build(self, graph, edge_order, bin_size, **kwargs):
    # 1. Use track-linearization to linearize 2D graph
    from track_linearization import make_track_graph

    track_graph = make_track_graph(graph, edge_order)

    # 2. Create linear bins along track
    track_length = track_graph.get_track_length()
    n_bins = int(np.ceil(track_length / bin_size))
    linear_positions = np.linspace(0, track_length, n_bins)

    # 3. Map linear positions back to 2D coordinates
    bin_centers_2d = []
    for linear_pos in linear_positions:
        x, y = track_graph.linearize_to_2d(linear_pos)
        bin_centers_2d.append([x, y])

    self.bin_centers = np.array(bin_centers_2d)

    # 4. Build 1D chain connectivity
    G = nx.Graph()
    for i in range(n_bins):
        G.add_node(i, pos=tuple(self.bin_centers[i]), ...)
        if i > 0:
            G.add_edge(i-1, i, distance=bin_size, ...)

    self.connectivity = G
    self.is_1d = True

    # 5. Store linearization mapping
    self._track_graph = track_graph
```

**Linearization methods**:
```python
def to_linear(self, nd_positions):
    """Convert 2D positions to 1D linear positions."""
    return self._track_graph.get_linear_position(nd_positions)

def linear_to_nd(self, linear_positions):
    """Convert 1D linear positions to 2D positions."""
    return self._track_graph.linearize_to_2d(linear_positions)
```

---

## 3. Place Field Analysis

### Estimating Place Fields: compute_place_field()

**File**: `src/neurospatial/field_ops.py`

**Three Methods**:

#### Method 1: Diffusion KDE (default, graph-aware)

**Algorithm**:
```python
def compute_place_field(env, spike_times, times, positions,
                        method="diffusion_kde", bandwidth=5.0):
    # 1. Map positions to bins
    bin_indices = env.bin_at(positions)

    # 2. Compute occupancy (time spent in each bin)
    dt = np.median(np.diff(times))
    occupancy = np.zeros(env.n_bins)
    for bin_idx in bin_indices:
        if bin_idx >= 0:
            occupancy[bin_idx] += dt

    # 3. Compute spike counts per bin
    spike_counts = np.zeros(env.n_bins)
    spike_bin_indices = np.interp(
        spike_times, times, bin_indices
    ).astype(int)
    for bin_idx in spike_bin_indices:
        if 0 <= bin_idx < env.n_bins:
            spike_counts[bin_idx] += 1

    # 4. Smooth using graph diffusion
    # Convert bandwidth to diffusion time
    sigma_bins = bandwidth / np.mean(env.bin_sizes)
    diffusion_time = sigma_bins**2 / (2 * env.n_dims)

    # Compute graph Laplacian
    L = nx.laplacian_matrix(env.connectivity).asarray()

    # Diffusion: exp(-t*L) @ field
    from scipy.sparse.linalg import expm_multiply
    spike_counts_smooth = expm_multiply(
        -diffusion_time * L, spike_counts
    )
    occupancy_smooth = expm_multiply(
        -diffusion_time * L, occupancy
    )

    # 5. Compute firing rate
    firing_rate = spike_counts_smooth / (occupancy_smooth + 1e-10)

    return firing_rate
```

**Why diffusion KDE?**
- **Respects boundaries**: Diffusion doesn't leak across walls (disconnected graph nodes)
- **Adaptive smoothing**: Uses connectivity structure, not just Euclidean distance
- **Mathematically principled**: Solves heat equation on graph

#### Method 2: Gaussian KDE (classic)

**Algorithm**:
```python
def compute_place_field_gaussian_kde(env, spike_times, times, positions,
                                      bandwidth=5.0):
    # 1-3. Same: occupancy and spike counts
    ...

    # 4. Smooth using Gaussian KDE
    from scipy.ndimage import gaussian_filter

    # For grid layouts, can use fast FFT-based convolution
    if hasattr(env.layout, 'grid_shape'):
        # Reshape to grid
        spike_grid = spike_counts.reshape(env.layout.grid_shape)
        occupancy_grid = occupancy.reshape(env.layout.grid_shape)

        # Gaussian filter
        sigma_pixels = bandwidth / env.layout.bin_size
        spike_smooth = gaussian_filter(spike_grid, sigma=sigma_pixels)
        occ_smooth = gaussian_filter(occupancy_grid, sigma=sigma_pixels)

        firing_rate = (spike_smooth / (occ_smooth + 1e-10)).ravel()
    else:
        # Fall back to KDE over bin_centers
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        # ... (slower, for non-grid layouts)

    return firing_rate
```

#### Method 3: Binned (legacy, no smoothing)

```python
def compute_place_field_binned(env, spike_times, times, positions):
    # No smoothing - just raw counts / occupancy
    firing_rate = spike_counts / (occupancy + 1e-10)
    return firing_rate
```

### Place Field Detection: detect_place_fields()

**File**: `src/neurospatial/metrics/place_fields.py`

**Algorithm** (neurocode iterative peak method):

```python
def detect_place_fields(firing_rate, env, threshold=0.2, min_size=9):
    # 1. Interneuron exclusion
    if np.mean(firing_rate) > 10.0:  # Hz
        return []  # Too high - likely interneuron

    fields = []
    rate_map = firing_rate.copy()

    while True:
        # 2. Find global peak
        peak_idx = np.argmax(rate_map)
        peak_rate = rate_map[peak_idx]

        if peak_rate == 0:
            break  # No more fields

        # 3. Threshold at fraction of peak
        field_threshold = threshold * peak_rate

        # 4. Find connected component containing peak
        # Use graph BFS from peak bin
        field_bins = _connected_component_bfs(
            env.connectivity, peak_idx,
            lambda bin_idx: rate_map[bin_idx] >= field_threshold
        )

        # 5. Check size
        if len(field_bins) >= min_size:
            fields.append(np.array(field_bins))

        # 6. Remove field from map and iterate
        rate_map[field_bins] = 0

    return fields

def _connected_component_bfs(graph, start_node, condition_func):
    """BFS to find connected bins satisfying condition."""
    visited = {start_node}
    queue = [start_node]
    component = []

    while queue:
        node = queue.pop(0)
        if condition_func(node):
            component.append(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return component
```

### Place Field Metrics

**File**: `src/neurospatial/metrics/place_fields.py`

#### Skaggs Information (bits/spike)

**Formula**: `I = Σ pᵢ * (rᵢ / r̄) * log₂(rᵢ / r̄)`

```python
def skaggs_information(firing_rate, occupancy):
    """Spatial information content (bits/spike)."""
    # Normalize occupancy to probability
    prob = occupancy / occupancy.sum()

    # Mean firing rate
    mean_rate = (firing_rate * prob).sum()

    # Information
    info = 0.0
    for i in range(len(firing_rate)):
        if prob[i] > 0 and firing_rate[i] > 0:
            ratio = firing_rate[i] / mean_rate
            info += prob[i] * ratio * np.log2(ratio)

    return info  # bits/spike
```

#### Sparsity

**Formula**: `S = (Σ pᵢ rᵢ)² / Σ pᵢ rᵢ²`

```python
def sparsity(firing_rate, occupancy):
    """Skaggs sparsity measure (0 = dense, 1 = sparse)."""
    prob = occupancy / occupancy.sum()

    mean_rate = (firing_rate * prob).sum()
    mean_rate_sq = (firing_rate**2 * prob).sum()

    if mean_rate_sq == 0:
        return 0.0

    return (mean_rate**2) / mean_rate_sq
```

#### Selectivity (max/mean ratio)

```python
def selectivity(firing_rate):
    """Peak to mean ratio."""
    mean_rate = np.mean(firing_rate)
    if mean_rate == 0:
        return 0.0
    return np.max(firing_rate) / mean_rate
```

---

## 4. Boundary & Grid Cell Metrics

### Boundary Vector Cells

**File**: `src/neurospatial/metrics/boundary_cells.py`

**Algorithm**:
```python
def boundary_vector_score(firing_rate, env):
    """Compute boundary vector cell score."""
    # 1. Find boundary bins
    boundary_bins = env.boundary_bins()

    # 2. For each bin, compute vector to nearest boundary
    boundary_vectors = []
    for i in range(env.n_bins):
        bin_center = env.bin_centers[i]

        # Find nearest boundary bin
        boundary_centers = env.bin_centers[boundary_bins]
        distances = np.linalg.norm(
            boundary_centers - bin_center, axis=1
        )
        nearest_idx = np.argmin(distances)
        nearest_boundary = boundary_centers[nearest_idx]

        # Vector from bin to boundary
        vector = nearest_boundary - bin_center
        distance = distances[nearest_idx]
        angle = np.arctan2(vector[1], vector[0])

        boundary_vectors.append({
            'distance': distance,
            'angle': angle,
            'rate': firing_rate[i]
        })

    # 3. Compute tuning to distance and direction
    # Group by distance bins
    distance_tuning = ...
    direction_tuning = ...

    # 4. Score is product of distance and direction selectivity
    score = distance_selectivity * direction_selectivity
    return score
```

### Grid Cells

**File**: `src/neurospatial/metrics/grid_cells.py`

#### Grid Score (autocorrelation method)

**Algorithm**:
```python
def grid_score(firing_rate, env):
    """Compute grid score via spatial autocorrelation."""
    # 1. Compute spatial autocorrelation
    autocorr = _spatial_autocorrelation_2d(firing_rate, env)

    # 2. Find central peak
    center_idx = len(autocorr) // 2

    # 3. Mask out center
    radius = 3  # bins
    autocorr_masked = autocorr.copy()
    autocorr_masked[
        center_idx - radius:center_idx + radius,
        center_idx - radius:center_idx + radius
    ] = 0

    # 4. Find 6 peaks (hexagonal lattice)
    peaks = _find_local_maxima(autocorr_masked, n_peaks=6)

    # 5. Compute rotational symmetry
    # Grid score = corr(0°) + corr(120°) + corr(240°) - corr(60°) - corr(180°) - corr(300°)
    angles = [np.arctan2(p[1] - center_idx, p[0] - center_idx) for p in peaks]

    # Check for 60° rotational symmetry
    rotated_60 = _rotate_field(autocorr, 60)
    rotated_120 = _rotate_field(autocorr, 120)

    corr_60 = np.corrcoef(autocorr.ravel(), rotated_60.ravel())[0, 1]
    corr_120 = np.corrcoef(autocorr.ravel(), rotated_120.ravel())[0, 1]

    grid_score = min(corr_60, corr_120)
    return grid_score

def _spatial_autocorrelation_2d(firing_rate, env):
    """Compute 2D spatial autocorrelation."""
    # Reshape to grid (requires grid layout)
    rate_grid = firing_rate.reshape(env.layout.grid_shape)

    # FFT-based autocorrelation
    from scipy.signal import correlate2d
    autocorr = correlate2d(rate_grid, rate_grid, mode='same')

    # Normalize
    autocorr /= autocorr.max()

    return autocorr
```

---

## 5. Trajectory Analysis

### Occupancy Computation

**File**: `src/neurospatial/environment/trajectory.py`

```python
def occupancy(self, positions, times=None, sampling_frequency=None):
    """Compute time spent in each bin."""
    # 1. Map positions to bins
    bin_indices = self.bin_at(positions)

    # 2. Infer time step
    if times is not None:
        dt = np.median(np.diff(times))
    elif sampling_frequency is not None:
        dt = 1.0 / sampling_frequency
    else:
        dt = 1.0  # Assume 1 sample = 1 time unit

    # 3. Accumulate time per bin
    occupancy = np.zeros(self.n_bins)
    for bin_idx in bin_indices:
        if bin_idx >= 0:
            occupancy[bin_idx] += dt

    return occupancy  # seconds
```

### Transition Matrix

```python
def transitions(self, positions, normalize=True):
    """Compute bin-to-bin transition counts."""
    # 1. Get bin sequence
    bin_sequence = self.bin_at(positions)

    # 2. Build transition matrix
    T = np.zeros((self.n_bins, self.n_bins))
    for i in range(len(bin_sequence) - 1):
        bin_from = bin_sequence[i]
        bin_to = bin_sequence[i + 1]
        if bin_from >= 0 and bin_to >= 0:
            T[bin_from, bin_to] += 1

    # 3. Normalize rows to probabilities
    if normalize:
        row_sums = T.sum(axis=1, keepdims=True)
        T = T / (row_sums + 1e-10)

    return T  # Shape (n_bins, n_bins)
```

### Speed Computation

```python
def speed(self, positions, times):
    """Compute instantaneous speed."""
    # Compute displacement
    displacement = np.diff(positions, axis=0)
    dt = np.diff(times)

    # Speed = distance / time
    speeds = np.linalg.norm(displacement, axis=1) / dt

    # Prepend 0 for first sample
    speeds = np.concatenate([[0], speeds])

    return speeds  # units/second
```

---

## 6. Population-Level Metrics

### Population Vector Correlation

**File**: `src/neurospatial/metrics/population.py`

```python
def population_vector_correlation(rate_maps_1, rate_maps_2):
    """Correlation between population vectors in two sessions.

    Parameters
    ----------
    rate_maps_1 : array, shape (n_cells, n_bins)
    rate_maps_2 : array, shape (n_cells, n_bins)

    Returns
    -------
    pv_corr : array, shape (n_bins,)
        Correlation at each spatial bin
    """
    n_cells, n_bins = rate_maps_1.shape

    correlations = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        # Population vector at this bin
        pv1 = rate_maps_1[:, bin_idx]  # Shape (n_cells,)
        pv2 = rate_maps_2[:, bin_idx]

        # Pearson correlation
        if pv1.std() > 0 and pv2.std() > 0:
            correlations[bin_idx] = np.corrcoef(pv1, pv2)[0, 1]
        else:
            correlations[bin_idx] = 0.0

    return correlations
```

### Co-firing (Joint Probability)

```python
def coactivity_map(spike_train_1, spike_train_2, env, positions, times):
    """Compute spatial map of co-firing between two neurons."""
    # 1. Map positions to bins
    bin_indices = env.bin_at(positions)

    # 2. Detect coincident spikes (within time window)
    window = 0.025  # 25 ms

    coactivity = np.zeros(env.n_bins)
    for t in range(len(times)):
        bin_idx = bin_indices[t]
        if bin_idx < 0:
            continue

        # Check if both neurons spiked near this time
        spikes_1 = np.abs(spike_train_1 - times[t]) < window
        spikes_2 = np.abs(spike_train_2 - times[t]) < window

        if np.any(spikes_1) and np.any(spikes_2):
            coactivity[bin_idx] += 1

    return coactivity
```

---

## 7. Spatial Operations

### Smoothing Fields

**File**: `src/neurospatial/environment/fields.py`

```python
def smooth(self, field, method='graph_diffusion', sigma=1.0):
    """Smooth a spatial field."""
    if method == 'graph_diffusion':
        # Use graph Laplacian
        L = nx.laplacian_matrix(self.connectivity).asarray()

        # Diffusion time from sigma
        diffusion_time = sigma**2 / (2 * self.n_dims)

        # Matrix exponential: exp(-t*L) @ field
        from scipy.sparse.linalg import expm_multiply
        field_smooth = expm_multiply(-diffusion_time * L, field)

    elif method == 'gaussian':
        # Requires grid layout
        if not hasattr(self.layout, 'grid_shape'):
            raise ValueError("Gaussian smoothing requires grid layout")

        from scipy.ndimage import gaussian_filter
        field_grid = field.reshape(self.layout.grid_shape)
        field_smooth = gaussian_filter(field_grid, sigma=sigma)
        field_smooth = field_smooth.ravel()

    return field_smooth
```

### Interpolation

```python
def interpolate(self, field, query_points, method='linear'):
    """Interpolate field values at arbitrary points."""
    from scipy.interpolate import LinearNDInterpolator

    # Create interpolator from bin_centers and field values
    interpolator = LinearNDInterpolator(
        self.bin_centers, field, fill_value=0.0
    )

    # Evaluate at query points
    values = interpolator(query_points)

    return values
```

---

## 8. Distance & Graph Operations

### Geodesic Distance

**File**: `src/neurospatial/distance.py`

```python
def distance_field(connectivity, sources, edge_weight='distance'):
    """Compute geodesic distances from source bins to all bins.

    Uses Dijkstra's algorithm via NetworkX.
    """
    # Multi-source Dijkstra
    distances = nx.multi_source_dijkstra_path_length(
        connectivity,
        sources=sources,
        weight=edge_weight
    )

    # Convert to array
    n_bins = len(connectivity)
    dist_array = np.full(n_bins, np.inf)
    for node, dist in distances.items():
        dist_array[node] = dist

    return dist_array
```

### Shortest Path

**File**: `src/neurospatial/environment/queries.py`

```python
def shortest_path(self, source_bin, target_bin, weight='distance'):
    """Find shortest path between two bins."""
    try:
        path = nx.shortest_path(
            self.connectivity,
            source=source_bin,
            target=target_bin,
            weight=weight
        )
        return path  # List of bin indices
    except nx.NetworkXNoPath:
        return None  # No path exists
```

### Distance Between Points

```python
def distance_between(self, point1, point2, edge_weight='distance'):
    """Geodesic distance between two continuous points."""
    # 1. Map points to bins
    bin1 = self.bin_at(np.atleast_2d(point1))[0]
    bin2 = self.bin_at(np.atleast_2d(point2))[0]

    if bin1 < 0 or bin2 < 0:
        return np.inf  # Outside environment

    # 2. Graph shortest path distance
    try:
        distance = nx.shortest_path_length(
            self.connectivity,
            source=bin1,
            target=bin2,
            weight=edge_weight
        )
        return distance
    except nx.NetworkXNoPath:
        return np.inf
```

---

## 9. Field Smoothing & Kernels

### Graph-Based Kernels

**File**: `src/neurospatial/kernels.py`

```python
def compute_kernel(self, kernel_type='gaussian_graph', bandwidth=1.0):
    """Compute smoothing kernel matrix.

    Returns
    -------
    K : array, shape (n_bins, n_bins)
        Kernel matrix where K[i, j] = weight from bin j to bin i
    """
    if kernel_type == 'gaussian_graph':
        # Geodesic distance-based Gaussian
        # 1. Compute all-pairs distances
        from neurospatial.distance import pairwise_distances
        D = pairwise_distances(self.connectivity, metric='distance')

        # 2. Gaussian kernel: K(i,j) = exp(-D²/(2σ²))
        K = np.exp(-D**2 / (2 * bandwidth**2))

        # 3. Normalize rows
        K = K / K.sum(axis=1, keepdims=True)

    elif kernel_type == 'diffusion':
        # Graph diffusion kernel
        # K = exp(-t * L) where L is graph Laplacian
        L = nx.laplacian_matrix(self.connectivity).asarray()

        from scipy.linalg import expm
        K = expm(-bandwidth * L)

    return K
```

**Usage for smoothing**:
```python
# Compute kernel once
K = env.compute_kernel('gaussian_graph', bandwidth=5.0)

# Smooth any field
field_smooth = K @ field  # Matrix multiplication
```

---

## 10. Segmentation & Trial Analysis

### Lap Detection

**File**: `src/neurospatial/segmentation/laps.py`

```python
def detect_laps(env, positions, times, lap_bins):
    """Detect laps based on passing through lap bins.

    Parameters
    ----------
    lap_bins : array of int
        Bin indices defining lap start/end
    """
    # 1. Get bin sequence
    bin_sequence = env.bin_at(positions)

    # 2. Find crossings of lap bins
    crossings = []
    for i in range(1, len(bin_sequence)):
        if bin_sequence[i] in lap_bins and bin_sequence[i-1] not in lap_bins:
            crossings.append(i)

    # 3. Define laps as segments between crossings
    laps = []
    for i in range(len(crossings) - 1):
        lap_start = crossings[i]
        lap_end = crossings[i + 1]
        laps.append({
            'start_idx': lap_start,
            'end_idx': lap_end,
            'start_time': times[lap_start],
            'end_time': times[lap_end],
            'duration': times[lap_end] - times[lap_start]
        })

    return laps
```

### Trial Segmentation

**File**: `src/neurospatial/segmentation/trials.py`

```python
def segment_by_trials(positions, times, trial_starts, trial_ends):
    """Segment trajectory into trials."""
    trials = []
    for start_time, end_time in zip(trial_starts, trial_ends):
        # Find indices in time range
        mask = (times >= start_time) & (times < end_time)

        trials.append({
            'positions': positions[mask],
            'times': times[mask],
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        })

    return trials
```

---

## 11. Simulation & Synthetic Data

### Ornstein-Uhlenbeck Trajectory

**File**: `src/neurospatial/simulation/trajectory.py`

**Algorithm** (2D rotational OU):

```python
def simulate_trajectory_ou(env, duration, dt=0.01, speed_mean=0.08,
                           rotational_velocity_std=120*np.pi/180,
                           coherence_time=0.7):
    """Simulate trajectory using OU process."""
    # Initialize
    rng = np.random.default_rng(seed)
    position = env.bin_centers[rng.integers(0, env.n_bins)]
    velocity = rng.standard_normal(2)
    velocity = velocity / np.linalg.norm(velocity) * speed_mean

    # OU parameters for rotational velocity
    theta_rot = 1.0 / coherence_time
    sigma_rot = rotational_velocity_std * np.sqrt(2 * theta_rot)

    rotational_velocity = 0.0

    # Time array
    n_steps = int(duration / dt)
    positions = np.zeros((n_steps, 2))
    positions[0] = position

    # Euler-Maruyama integration
    for i in range(1, n_steps):
        # Update rotational velocity (OU process)
        dw = rng.standard_normal() * np.sqrt(dt)
        rotational_velocity = (
            rotational_velocity
            - theta_rot * rotational_velocity * dt
            + sigma_rot * dw
        )

        # Rotate velocity vector
        dtheta = rotational_velocity * dt
        rotation_matrix = np.array([
            [np.cos(dtheta), -np.sin(dtheta)],
            [np.sin(dtheta), np.cos(dtheta)]
        ])
        velocity = rotation_matrix @ velocity

        # Maintain constant speed
        velocity = velocity / np.linalg.norm(velocity) * speed_mean

        # Update position
        position = position + velocity * dt

        # Handle boundaries (reflect)
        if not env.contains(position):
            # Reflect velocity
            nearest_bin = env.bin_at(position)[0]
            if nearest_bin >= 0:
                position = env.bin_centers[nearest_bin]
            # Reverse velocity normal to boundary
            normal = ...  # Compute boundary normal
            velocity = velocity - 2 * np.dot(velocity, normal) * normal

        positions[i] = position

    return positions, times
```

### Coverage-Ensuring Trajectory

**File**: `src/neurospatial/simulation/trajectory.py`

**Algorithm**:

```python
def simulate_trajectory_coverage(env, duration, coverage_bias=2.0):
    """Generate trajectory ensuring >90% coverage."""
    # Track occupancy counts
    occupancy = np.zeros(env.n_bins, dtype=int)
    current_bin = np.random.randint(env.n_bins)
    occupancy[current_bin] = 1

    trajectory_bins = [current_bin]

    while len(trajectory_bins) < n_samples:
        # Get neighbors
        neighbors = list(env.connectivity.neighbors(current_bin))

        # Weight by inverse occupancy
        weights = 1.0 / (1.0 + occupancy[neighbors]) ** coverage_bias
        weights /= weights.sum()

        # Choose next bin
        next_bin = np.random.choice(neighbors, p=weights)

        # Add to trajectory
        trajectory_bins.append(next_bin)
        occupancy[next_bin] += 1
        current_bin = next_bin

    # Convert bins to continuous positions
    positions = env.bin_centers[trajectory_bins]

    # Add jitter
    jitter = np.random.randn(*positions.shape) * bin_size * 0.2
    positions += jitter

    return positions, times
```

### Place Cell Simulation

**File**: `src/neurospatial/simulation/models.py`

```python
class PlaceCellModel:
    """Simulate place cell with Gaussian tuning."""

    def __init__(self, env, center, width, peak_rate=10.0):
        self.env = env
        self.center = center  # 2D position
        self.width = width    # Gaussian std
        self.peak_rate = peak_rate

    def firing_rate(self, positions):
        """Compute firing rate at positions."""
        # Gaussian tuning
        distances = np.linalg.norm(positions - self.center, axis=1)
        rates = self.peak_rate * np.exp(-(distances**2) / (2 * self.width**2))
        return rates

    def generate_spikes(self, positions, times):
        """Generate Poisson spikes."""
        rates = self.firing_rate(positions)
        dt = np.median(np.diff(times))

        spike_times = []
        for i, rate in enumerate(rates):
            # Poisson probability
            p = rate * dt
            if np.random.rand() < p:
                spike_times.append(times[i])

        return np.array(spike_times)
```

---

## 12. Environment Transformation & Alignment

### Rebinning

**File**: `src/neurospatial/environment/transforms.py`

```python
def rebin(self, new_bin_size):
    """Create new environment with different bin size."""
    # 1. Get original samples (if from_samples was used)
    if not hasattr(self.layout, '_build_params_used'):
        raise ValueError("Cannot rebin: no build params")

    samples = self.layout._build_params_used.get('samples')
    if samples is None:
        raise ValueError("Cannot rebin: not created from samples")

    # 2. Create new environment with new bin_size
    new_env = Environment.from_samples(
        samples,
        bin_size=new_bin_size,
        **other_params
    )

    return new_env
```

### Coordinate Transformations

**File**: `src/neurospatial/transforms.py`

```python
class Affine2D:
    """2D affine transformation matrix."""

    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(3)  # Identity
        else:
            self.matrix = matrix

    def translate(self, dx, dy):
        """Apply translation."""
        T = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])
        self.matrix = T @ self.matrix
        return self

    def rotate(self, angle_rad):
        """Apply rotation."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        self.matrix = R @ self.matrix
        return self

    def scale(self, sx, sy):
        """Apply scaling."""
        S = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0,  0, 1]
        ])
        self.matrix = S @ self.matrix
        return self

    def apply(self, points):
        """Apply transformation to points."""
        # Homogeneous coordinates
        n_points = points.shape[0]
        points_h = np.hstack([points, np.ones((n_points, 1))])

        # Transform
        points_transformed = (self.matrix @ points_h.T).T

        # Back to Cartesian
        return points_transformed[:, :2]
```

**Estimating transformations from landmarks**:

```python
def estimate_transform(src_landmarks, dst_landmarks, kind='rigid'):
    """Estimate transformation from corresponding points.

    Parameters
    ----------
    src_landmarks : array, shape (n_points, 2)
    dst_landmarks : array, shape (n_points, 2)
    kind : 'rigid', 'similarity', or 'affine'

    Returns
    -------
    transform : Affine2D
    """
    if kind == 'rigid':
        # Rotation + translation (no scaling)
        # 1. Center points
        src_center = src_landmarks.mean(axis=0)
        dst_center = dst_landmarks.mean(axis=0)

        src_centered = src_landmarks - src_center
        dst_centered = dst_landmarks - dst_center

        # 2. Compute rotation via SVD
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 3. Compute translation
        t = dst_center - R @ src_center

        # 4. Build transformation matrix
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t

        return Affine2D(T)

    elif kind == 'affine':
        # Full affine (6 DOF)
        # Solve least squares: dst = A @ src + b
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(src_landmarks, dst_landmarks)

        T = np.eye(3)
        T[:2, :2] = reg.coef_
        T[:2, 2] = reg.intercept_

        return Affine2D(T)
```

---

## 13. Composite Environments

**File**: `src/neurospatial/composite.py`

**Purpose**: Merge multiple disjoint environments with automatic bridge detection

```python
class CompositeEnvironment:
    """Multiple environments connected by bridges."""

    def __init__(self, environments, bridges=None):
        self.environments = environments

        if bridges is None:
            # Auto-detect bridges via mutual nearest neighbors
            bridges = self._infer_bridges()

        self.bridges = bridges

        # Build combined graph
        self.global_graph = self._build_global_graph()

    def _infer_bridges(self):
        """Detect bridges between environments."""
        bridges = []

        for i in range(len(self.environments)):
            for j in range(i + 1, len(self.environments)):
                env_i = self.environments[i]
                env_j = self.environments[j]

                # Find mutual nearest neighbors
                for bin_i in range(env_i.n_bins):
                    pos_i = env_i.bin_centers[bin_i]

                    # Nearest bin in env_j
                    distances = np.linalg.norm(
                        env_j.bin_centers - pos_i, axis=1
                    )
                    nearest_j = np.argmin(distances)
                    min_dist = distances[nearest_j]

                    # Check if mutual nearest neighbor
                    pos_j = env_j.bin_centers[nearest_j]
                    distances_back = np.linalg.norm(
                        env_i.bin_centers - pos_j, axis=1
                    )
                    nearest_i_from_j = np.argmin(distances_back)

                    if nearest_i_from_j == bin_i and min_dist < threshold:
                        # Mutual nearest neighbor - create bridge
                        bridges.append({
                            'env_i': i,
                            'bin_i': bin_i,
                            'env_j': j,
                            'bin_j': nearest_j,
                            'distance': min_dist
                        })

        return bridges

    def _build_global_graph(self):
        """Build combined connectivity graph."""
        # 1. Merge all environment graphs with offset node IDs
        G_global = nx.Graph()

        node_offset = 0
        for env in self.environments:
            # Add nodes with global IDs
            for node in env.connectivity.nodes():
                global_id = node + node_offset
                G_global.add_node(global_id, **env.connectivity.nodes[node])

            # Add edges
            for u, v in env.connectivity.edges():
                G_global.add_edge(
                    u + node_offset,
                    v + node_offset,
                    **env.connectivity[u][v]
                )

            node_offset += env.n_bins

        # 2. Add bridge edges
        node_offset = 0
        for bridge in self.bridges:
            env_i_idx = bridge['env_i']
            env_j_idx = bridge['env_j']

            # Compute offsets
            offset_i = sum(self.environments[k].n_bins for k in range(env_i_idx))
            offset_j = sum(self.environments[k].n_bins for k in range(env_j_idx))

            # Add bridge edge
            G_global.add_edge(
                bridge['bin_i'] + offset_i,
                bridge['bin_j'] + offset_j,
                distance=bridge['distance'],
                is_bridge=True
            )

        return G_global
```

---

## 14. Regions of Interest (ROIs)

**File**: `src/neurospatial/regions/core.py`

```python
@dataclass(frozen=True)  # Immutable
class Region:
    """Immutable region definition."""
    name: str
    kind: Literal["point", "polygon"]
    point: NDArray[np.float64] | None = None
    polygon: Any | None = None  # Shapely Polygon

    def __post_init__(self):
        # Validation
        if self.kind == "point" and self.point is None:
            raise ValueError("Point region requires point")
        if self.kind == "polygon" and self.polygon is None:
            raise ValueError("Polygon region requires polygon")

class Regions:
    """Container for multiple regions (dict-like)."""

    def __init__(self):
        self._regions: dict[str, Region] = {}

    def add(self, name, *, point=None, polygon=None):
        """Add new region (raises KeyError if exists)."""
        if name in self._regions:
            raise KeyError(f"Region '{name}' already exists")

        if point is not None:
            region = Region(name, "point", point=np.asarray(point))
        elif polygon is not None:
            region = Region(name, "polygon", polygon=polygon)
        else:
            raise ValueError("Must provide point or polygon")

        self._regions[name] = region

    def update_region(self, name, **kwargs):
        """Update existing region (creates new immutable Region)."""
        if name not in self._regions:
            raise KeyError(f"Region '{name}' not found")

        old_region = self._regions[name]

        # Create new Region with updated attributes
        new_region = Region(
            name=name,
            kind=old_region.kind,
            point=kwargs.get('point', old_region.point),
            polygon=kwargs.get('polygon', old_region.polygon)
        )

        self._regions[name] = new_region

    def __getitem__(self, name):
        return self._regions[name]

    def __delitem__(self, name):
        del self._regions[name]
```

**Environment integration**:

```python
# In EnvironmentRegions mixin
def bins_in_region(self, region_name):
    """Find bins inside a region."""
    region = self.regions[region_name]

    if region.kind == "point":
        # Single bin containing point
        bin_idx = self.bin_at(region.point[np.newaxis, :])[0]
        return np.array([bin_idx]) if bin_idx >= 0 else np.array([])

    elif region.kind == "polygon":
        # All bins whose centers are inside polygon
        from shapely.geometry import Point

        inside = []
        for i, center in enumerate(self.bin_centers):
            point = Point(center)
            if region.polygon.contains(point):
                inside.append(i)

        return np.array(inside)
```

---

## 15. Serialization & I/O

**File**: `src/neurospatial/environment/serialization.py`

### Save Environment

```python
def to_file(self, base_path):
    """Save environment to .json + .npz files."""
    import json

    # 1. Serialize to dict
    data = self.to_dict()

    # 2. Extract numpy arrays
    arrays = {}
    metadata = {}

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
            metadata[key] = {'type': 'ndarray', 'dtype': str(value.dtype)}
        else:
            metadata[key] = value

    # 3. Save metadata as JSON
    with open(f"{base_path}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # 4. Save arrays as NPZ
    np.savez_compressed(f"{base_path}.npz", **arrays)
```

### to_dict() Implementation

```python
def to_dict(self):
    """Convert environment to dictionary."""
    # Graph serialization
    graph_data = nx.node_link_data(self.connectivity)

    data = {
        'version': '0.2.0',
        'name': self.name,
        'bin_centers': self.bin_centers,
        'connectivity': graph_data,
        'dimension_ranges': self.dimension_ranges,
        'units': self.units,
        'frame': self.frame,

        # Layout info
        'layout_type': self.layout._layout_type_tag,
        'layout_params': self.layout._build_params_used,

        # Regions
        'regions': self.regions.to_dict() if self.regions else {},
    }

    # Handle grid-specific attributes
    if hasattr(self.layout, 'grid_shape'):
        data['grid_shape'] = self.layout.grid_shape
        data['grid_edges'] = self.layout.grid_edges
        data['active_mask'] = self.layout.active_mask

    return data
```

### Load Environment

```python
@classmethod
def from_file(cls, base_path):
    """Load environment from .json + .npz files."""
    import json

    # 1. Load metadata
    with open(f"{base_path}.json", 'r') as f:
        metadata = json.load(f)

    # 2. Load arrays
    arrays = np.load(f"{base_path}.npz", allow_pickle=True)

    # 3. Merge
    data = {}
    for key, value in metadata.items():
        if isinstance(value, dict) and value.get('type') == 'ndarray':
            data[key] = arrays[key]
        else:
            data[key] = value

    # 4. Reconstruct environment
    return cls.from_dict(data)
```

---

## Summary: Key Architectural Patterns

1. **Protocol-Based Layout Engines**: Duck typing with runtime validation
2. **Mixin Pattern for Environment**: Modular functionality across 9 mixin classes
3. **Graph-Centric Design**: NetworkX graphs for connectivity, distances, paths
4. **KDTree for Spatial Queries**: Fast nearest-neighbor lookup for point-to-bin mapping
5. **Graph Diffusion for Smoothing**: Boundary-aware smoothing using Laplacian
6. **Immutable Regions**: Dataclass-based with copy-on-update semantics
7. **Fitted State Pattern**: `@check_fitted` decorator prevents invalid operations
8. **Factory Methods**: Clean API via classmethods hiding layout complexity
9. **Type Safety**: Protocol + TYPE_CHECKING guards for mixin type checking
10. **Validation**: Comprehensive graph validation ensures data integrity

This architecture balances **flexibility** (protocol-based layouts), **performance** (KDTree, sparse matrices), and **scientific rigor** (validated metrics, graph-based methods).
