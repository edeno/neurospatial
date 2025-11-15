# neurospatial: Complete Capabilities Overview

## What is neurospatial?

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs. It provides comprehensive tools for spatial neuroscience analysis - particularly for analyzing place cells, boundary cells, grid cells, trajectories, and spatial navigation.

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

The `Environment` class is the foundation of neurospatial.

### Creating Environments

```python
from neurospatial import Environment
import numpy as np

# From position samples (automatic bin detection)
positions = np.random.uniform(0, 100, (1000, 2))
env = Environment.from_samples(positions, bin_size=2.0)
env.units = "cm"

# From 1D graph (linearized track)
env = Environment.from_graph(graph, edge_order, bin_size=1.0)

# From polygon
from shapely.geometry import Polygon
polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
env = Environment.from_polygon(polygon, bin_size=2.0)

# From binary image mask
env = Environment.from_image("arena.png", bin_size=2.0)

# From boolean mask
mask = np.random.rand(50, 50) > 0.3
env = Environment.from_mask(mask, bin_size=2.0)
```

### Key Properties

```python
env.n_bins              # Number of active bins
env.n_dims              # Dimensionality (1, 2, 3, or higher)
env.bin_centers         # Continuous coordinates of bin centers
env.connectivity        # NetworkX graph with edges between adjacent bins
env.is_1d               # True if 1D linearized environment
env.boundary_bins()     # Bins on the environment boundary
env.units               # Spatial units ("cm", "m", "pixels", etc.)
env.frame               # Coordinate frame identifier
```

### Spatial Queries

```python
# Map continuous positions to bins
bin_indices = env.bin_at(positions)

# Check if points are inside environment
is_inside = env.contains(positions)

# Get neighbors of a bin
neighbors = env.neighbors(bin_idx)

# Shortest path between bins
path = env.shortest_path(bin_a, bin_b)

# Distance between positions (geodesic along graph)
distance = env.distance_between(pos_a, pos_b)
```

---

## 2. Spatial Discretization (Layout Engines)

Multiple layout engines for different environment types:

### Available Layouts

```python
from neurospatial import list_available_layouts

layouts = list_available_layouts()
# ['regular_grid', 'hexagonal', 'graph', 'masked_grid',
#  'image_mask', 'shapely_polygon', 'triangular_mesh']
```

### 1. **Regular Grid** (default for `from_samples()`)
- Standard rectangular grid
- N-dimensional support
- Morphological operations (dilation, hole filling)

### 2. **Hexagonal Tessellation**
```python
from neurospatial.layout.engines import HexagonalLayout
layout = HexagonalLayout(bin_size=2.0)
env = Environment.from_layout(layout, ...)
```

### 3. **Graph Layout** (1D linearized tracks)
```python
# For T-mazes, linear tracks, figure-8 tracks
env = Environment.from_graph(graph, edge_order, bin_size=1.0)

# Linearization
if env.is_1d:
    linear_pos = env.to_linear(nd_position)
    nd_pos = env.linear_to_nd(linear_position)
```

### 4. **Masked Grid**
```python
# Arbitrary active/inactive regions
active_mask = (x_grid**2 + y_grid**2) < radius**2
env = Environment.from_mask(active_mask, bin_size=2.0)
```

### 5. **Image Mask**
```python
# Binary image-based layouts
env = Environment.from_image("maze.png", bin_size=5.0, threshold=0.5)
```

### 6. **Shapely Polygon**
```python
# Polygon-bounded grids
polygon = Polygon([(0, 0), (100, 0), (50, 100)])
env = Environment.from_polygon(polygon, bin_size=2.0)
```

### 7. **Triangular Mesh**
```python
# Triangular tessellations
from neurospatial.layout.engines import TriangularMeshLayout
```

---

## 3. Place Field Analysis

### Estimating Place Fields from Spikes

```python
from neurospatial import compute_place_field

# From spike times
firing_rate = compute_place_field(
    env,
    spike_times,       # Spike times (seconds)
    times,             # Position time vector
    positions,         # Position coordinates
    method="diffusion_kde",  # Graph-based KDE (default)
    bandwidth=5.0,     # Smoothing bandwidth (cm)
)

# Alternative methods
# method="gaussian_kde"  # Classic Gaussian KDE
# method="binned"        # Simple binned histogram
```

### Place Field Detection

```python
from neurospatial.metrics import detect_place_fields

# Detect fields from rate map
fields = detect_place_fields(
    firing_rate,
    env,
    threshold=0.2,        # 20% of max rate
    min_field_size=5,     # Minimum field size (bins)
    min_peak_rate=1.0     # Minimum peak firing rate (Hz)
)
# Returns: list of binary masks for each field
```

### Place Field Metrics (Validated vs. Literature)

```python
from neurospatial.metrics import (
    skaggs_information,      # Spatial information (Skaggs 1993)
    sparsity,                # Sparsity metric (Skaggs 1996)
    selectivity,             # Selectivity (peak vs mean)
    field_size,              # Field size in cm²
    field_centroid,          # Field center of mass
    field_stability,         # Correlation between sessions
    rate_map_coherence,      # Spatial coherence (neighbor correlation)
    information_per_second,  # Information rate
    mutual_information,      # MI between position and spikes
)

# Spatial information (bits/spike)
info = skaggs_information(firing_rate, occupancy)

# Sparsity (0-1, higher = more selective)
spars = sparsity(firing_rate, occupancy)

# Field size (area in environment units²)
size = field_size(field_mask, env)

# Field centroid (continuous coordinates)
center = field_centroid(firing_rate, field_mask, env)

# Field stability (correlation between two sessions)
stability = field_stability(rate_map_1, rate_map_2)

# Spatial coherence
coherence = rate_map_coherence(firing_rate, env)
```

### Field Shape Analysis

```python
from neurospatial.metrics import field_shape_metrics

metrics = field_shape_metrics(firing_rate, field_mask, env)
# Returns: {
#   'area': float,
#   'perimeter': float,
#   'circularity': float,
#   'compactness': float,
#   'aspect_ratio': float,
# }
```

### Field Comparison

```python
from neurospatial.metrics import (
    field_shift_distance,   # Distance between field centers
    compute_field_emd,      # Earth Mover's Distance
    in_out_field_ratio,     # Ratio of in-field to out-field firing
)

# Distance between field centers
shift = field_shift_distance(rate_1, rate_2, env)

# Earth Mover's Distance (optimal transport)
emd = compute_field_emd(rate_1, rate_2, env)
```

---

## 4. Boundary & Grid Cell Metrics

### Boundary/Border Cells

```python
from neurospatial.metrics import border_score, compute_region_coverage

# Border score (Solstad 2008)
score = border_score(
    firing_rate,
    env,
    threshold=0.3,        # Firing rate threshold (fraction of max)
    distance_metric="geodesic"  # or "euclidean"
)
# Returns: -1 to 1 (higher = stronger border cell)

# Region coverage
coverage = compute_region_coverage(firing_rate, env, region_mask)
```

### Grid Cells

```python
from neurospatial.metrics import (
    grid_score,              # Gridness (Hafting 2005)
    spatial_autocorrelation, # Spatial autocorrelation map
    periodicity_score,       # Periodicity metric
)

# Grid score (-1 to 1, higher = more hexagonal)
gridness = grid_score(firing_rate, env)

# Spatial autocorrelation
autocorr_map = spatial_autocorrelation(firing_rate, env, max_distance=100)

# Periodicity score
periodicity = periodicity_score(firing_rate, env)
```

---

## 5. Trajectory Analysis

### Basic Trajectory Metrics

```python
from neurospatial.metrics import (
    compute_turn_angles,     # Turn angles (radians)
    compute_step_lengths,    # Step lengths (distance between bins)
    compute_home_range,      # Home range (95% activity area)
    mean_square_displacement # Mean squared displacement
)

# Convert trajectory to bin sequence
trajectory_bins = env.bin_sequence(times, positions)

# Turn angles
angles = compute_turn_angles(trajectory_bins, env)

# Step lengths (geodesic or Euclidean)
steps = compute_step_lengths(trajectory_bins, env, method="geodesic")

# Home range (MCP or kernel density)
home_range = compute_home_range(trajectory_bins, env, method="mcp", percent=95)

# Mean squared displacement
msd = mean_square_displacement(trajectory_bins, env, time_lags=[1, 2, 5, 10])
```

### Occupancy Mapping

```python
# Time spent in each bin
occupancy = env.occupancy(times, positions)

# With speed filtering (remove stationary periods)
occupancy = env.occupancy(
    times, positions,
    speed_threshold=2.5,  # cm/s
    speed_smoothing_std=0.1  # Smooth speed estimate
)

# Normalized occupancy (probability distribution)
from neurospatial import normalize_field
occupancy_prob = normalize_field(occupancy)
```

### Empirical Transition Matrix

```python
# Compute transition probabilities between bins
transition_matrix = env.empirical_transition_matrix(
    times, positions,
    adjacency_filter=True  # Only count transitions between neighbors
)
# Returns: (n_bins, n_bins) matrix of transition probabilities
```

---

## 6. Population-Level Metrics

```python
from neurospatial.metrics import (
    population_coverage,           # Fraction of bins covered by any cell
    count_place_cells,             # Number of cells with place fields
    field_density_map,             # Density of fields per bin
    field_overlap,                 # Overlap between two fields
    population_vector_correlation  # PV correlation matrix
)

# Population coverage
all_fields = [detect_place_fields(rate, env) for rate in all_rate_maps]
coverage = population_coverage(all_fields, env.n_bins)
# Returns: fraction of bins with at least one field

# Count place cells
n_place_cells = count_place_cells(
    all_rate_maps,
    env,
    min_peak_rate=1.0,
    min_spatial_info=0.5
)

# Field density map
density = field_density_map(all_fields, env.n_bins)
# Returns: (n_bins,) array of field counts per bin

# Field overlap
overlap = field_overlap(field_1, field_2)
# Returns: Jaccard index (0-1)

# Population vector correlation
pv_corr = population_vector_correlation(population_matrix)
# population_matrix: (n_cells, n_bins)
```

---

## 7. Spatial Operations

### Field Operations

```python
from neurospatial import (
    normalize_field,    # Normalize to sum=1 or range [0,1]
    clamp,             # Clamp values to range
    combine_fields,    # Weighted sum of fields
    kl_divergence,     # KL divergence between distributions
)

# Normalize field
normalized = normalize_field(field, method="sum")  # or "max", "std"

# Clamp to range
clamped = clamp(field, min_val=0.0, max_val=10.0)

# Combine multiple fields
combined = combine_fields([field1, field2, field3], weights=[0.5, 0.3, 0.2])

# KL divergence
kl_div = kl_divergence(distribution_p, distribution_q)
```

### Field Interpolation

```python
# Evaluate field at arbitrary points
values = env.interpolate(field, query_points, method="nearest")
# method: "nearest", "linear" (for grids)
```

### Field Resampling

```python
from neurospatial import resample_field

# Resample field to new grid
new_field = resample_field(
    field,
    source_bin_centers,
    target_bin_centers,
    method="nearest"  # or "linear", "cubic"
)
```

---

## 8. Distance & Graph Operations

### Distance Fields

```python
from neurospatial import distance_field

# Distance from specific bins to all other bins
distances = distance_field(
    env.connectivity,
    sources=[goal_bin_id],
    method="dijkstra"  # or "shortest_path"
)
# Returns: (n_bins,) array of distances
```

### Pairwise Distances

```python
from neurospatial import pairwise_distances

# Distances between two sets of bins
dist_matrix = pairwise_distances(
    env.connectivity,
    source_bins=[1, 2, 3],
    target_bins=[10, 20, 30]
)
# Returns: (len(source_bins), len(target_bins))
```

### K-Hop Neighborhoods

```python
from neurospatial import neighbors_within

# All bins within k hops
neighbors = neighbors_within(
    env.connectivity,
    source_bin,
    max_distance=5  # 5 hops
)
# Returns: set of bin indices
```

### Connected Components

```python
import networkx as nx

# Find disconnected regions
components = list(nx.connected_components(env.connectivity))
n_components = len(components)
```

---

## 9. Field Smoothing & Kernels

### Diffusion Kernel Smoothing

```python
from neurospatial import compute_diffusion_kernels, apply_kernel

# Compute diffusion kernels (graph-based)
kernels = compute_diffusion_kernels(
    env,
    bandwidth=5.0,  # Smoothing scale (cm)
    volume_correct=True  # Correct for bin volumes
)

# Apply kernel to smooth a field
smoothed = apply_kernel(kernels, field)
```

### Environment Smoothing Methods

```python
# Smooth field using environment's method
smoothed = env.smooth(
    field,
    bandwidth=5.0,
    method="diffusion",  # or "gaussian"
    volume_correct=True
)
```

### Convolution

```python
from neurospatial import convolve

# Convolve field with kernel
result = convolve(env.connectivity, field, kernel)
```

### Neighbor Reduction

```python
from neurospatial import neighbor_reduce

# Apply reduction operation over neighbors
reduced = neighbor_reduce(
    env.connectivity,
    field,
    operation="mean"  # "mean", "sum", "max", "min"
)
```

---

## 10. Segmentation & Trial Analysis

### Lap Detection

```python
from neurospatial.segmentation import (
    detect_laps,           # Detect lap boundaries
    segment_trajectory_by_laps  # Split trajectory into laps
)

# Detect laps (back-and-forth movement)
lap_boundaries = detect_laps(
    trajectory_bins,
    method="extrema"  # or "graph_cycles", "peak_detection"
)

# Segment trajectory
laps = segment_trajectory_by_laps(times, positions, lap_boundaries)
# Returns: list of (times, positions) for each lap
```

### Trial Segmentation

```python
from neurospatial.segmentation import (
    segment_by_region_entries,  # Split by region entries
    segment_by_time_gaps,       # Split by temporal gaps
    segment_by_markers          # Split by explicit markers
)

# By region entries (e.g., reward well visits)
trials = segment_by_region_entries(
    times, positions, env,
    region_mask,
    min_dwell_time=0.5  # seconds
)

# By time gaps
segments = segment_by_time_gaps(
    times, positions,
    max_gap=10.0  # Split if gap > 10 seconds
)
```

### Similarity Analysis

```python
from neurospatial.segmentation import (
    compute_trajectory_similarity,  # Compare trajectories
    cluster_trajectories           # Group similar trajectories
)

# Compare two trajectories (DTW, Hausdorff, Fréchet)
similarity = compute_trajectory_similarity(
    traj_1, traj_2,
    method="dtw"  # "hausdorff", "frechet"
)

# Cluster trajectories
labels = cluster_trajectories(
    all_trajectories,
    method="kmeans",
    n_clusters=5
)
```

---

## 11. Simulation & Synthetic Data

**Full simulation module for generating test data and validating algorithms.**

### Trajectory Simulation

```python
from neurospatial.simulation import (
    simulate_trajectory_ou,         # Ornstein-Uhlenbeck process
    simulate_trajectory_sinusoidal, # Sinusoidal (1D)
    simulate_trajectory_laps,       # Structured laps
)

# Realistic random exploration (OU process)
positions, times = simulate_trajectory_ou(
    env,
    duration=180.0,        # seconds
    dt=0.01,              # 10ms timesteps
    speed_mean=0.08,      # 8 cm/s
    coherence_time=0.7,   # velocity autocorrelation time
    seed=42
)

# Sinusoidal motion (1D tracks)
positions, times = simulate_trajectory_sinusoidal(
    env,
    duration=120.0,
    running_speed=10.0
)

# Structured laps
positions, times, metadata = simulate_trajectory_laps(
    env,
    n_laps=20,
    speed_mean=0.1,
    return_metadata=True  # Get lap IDs, directions
)
```

### Neural Models

```python
from neurospatial.simulation import (
    PlaceCellModel,      # Gaussian place fields
    BoundaryCellModel,   # Distance-tuned boundary cells
    GridCellModel        # Hexagonal grid cells
)

# Place cell
pc = PlaceCellModel(
    env,
    center=[50, 75],     # Field center (cm)
    width=10.0,          # Field width (cm)
    max_rate=25.0        # Peak firing rate (Hz)
)
rates = pc.firing_rate(positions)

# Boundary cell
bc = BoundaryCellModel(
    env,
    preferred_distance=5.0,  # Fires 5 cm from walls
    max_rate=15.0
)
rates = bc.firing_rate(positions)

# Grid cell (2D only)
gc = GridCellModel(
    env,
    grid_spacing=50.0,       # Distance between peaks (cm)
    grid_orientation=0.0,    # Rotation (radians)
    max_rate=20.0
)
rates = gc.firing_rate(positions)
```

### Spike Generation

```python
from neurospatial.simulation import (
    generate_poisson_spikes,       # Poisson spike train
    generate_population_spikes,    # Multiple cells
    add_modulation                 # Rhythmic modulation (theta)
)

# Generate spikes from firing rate
spike_times = generate_poisson_spikes(
    rates,
    times,
    refractory_period=0.002  # 2ms refractory period
)

# Population spikes
all_spike_times = generate_population_spikes(
    [pc1, pc2, pc3],
    positions,
    times
)

# Add theta modulation
theta_modulated_spikes = add_modulation(
    spike_times,
    modulation_freq=8.0,      # 8 Hz theta
    modulation_depth=0.5
)
```

### High-Level Session API

```python
from neurospatial.simulation import (
    open_field_session,        # Open field exploration
    linear_track_session,      # Linear track
    tmaze_alternation_session, # T-maze
    boundary_cell_session,     # Boundary cell example
    grid_cell_session          # Grid cell example
)

# Complete simulated session
session = open_field_session(
    env,
    n_cells=50,
    duration=300.0,
    seed=42
)
# Returns: {
#   'positions': positions,
#   'times': times,
#   'spike_trains': [spike_times_1, spike_times_2, ...],
#   'ground_truth': [model_1, model_2, ...]
# }
```

### Validation Utilities

```python
from neurospatial.simulation import validate_simulation, plot_session_summary

# Validate detection against ground truth
results = validate_simulation(session, env)
# Returns detection errors, coverage, etc.

# Visualize session
plot_session_summary(session, env)
```

---

## 12. Environment Transformation & Alignment

### Coordinate Transforms

```python
from neurospatial import estimate_transform, apply_transform_to_environment
from neurospatial.transforms import Affine2D, translate, rotate, scale

# Estimate transform from landmark pairs
transform = estimate_transform(
    source_landmarks,
    target_landmarks,
    kind="rigid"  # "rigid", "affine", "similarity"
)

# Apply to environment
aligned_env = apply_transform_to_environment(env, transform)

# Manual transforms (composable)
from neurospatial.transforms import translate, rotate, scale

T = translate(10, 20) @ rotate(np.pi/4) @ scale(1.5)
transformed_env = apply_transform_to_environment(env, T)
```

### Probability Distribution Alignment

```python
from neurospatial import map_probabilities_to_nearest_target_bin

# Map probability distribution between environments
mapped_probs = map_probabilities_to_nearest_target_bin(
    source_env,
    target_env,
    source_probs,
    transform
)
```

### Environment Operations

```python
# Subset environment
subset_env = env.subset_environment(bin_mask)

# Rebin (change bin size)
rebinned_env = env.rebin(new_bin_size=4.0)

# Crop to region
cropped_env = env.crop_to_polygon(polygon)
```

---

## 13. Composite Environments

**Merge multiple environments with automatic bridge inference.**

```python
from neurospatial import CompositeEnvironment

# Create composite from multiple environments
composite = CompositeEnvironment(
    environments=[env1, env2, env3],
    names=["arena1", "arena2", "arena3"]
)

# Automatic bridge detection (mutual nearest neighbors)
composite.infer_bridges(max_distance=10.0)

# Manual bridge creation
composite.add_bridge("arena1", "arena2", bin_1, bin_2)

# Query across environments
global_bin = composite.local_to_global("arena1", local_bin)
env_name, local_bin = composite.global_to_local(global_bin)

# Global connectivity graph
global_graph = composite.get_global_connectivity()
```

---

## 14. Regions of Interest (ROIs)

```python
from neurospatial.regions import Region, Regions

# Create regions
goal_region = Region(
    point=[80, 80],        # Point region
    name="goal",
    metadata={"type": "reward"}
)

start_region = Region(
    polygon=Polygon(...),  # Polygon region
    name="start_box"
)

# Add to environment
env.regions.add(goal_region)
env.regions.add(start_region)

# Query regions
goal_bins = env.regions.get_bins("goal", env)
is_in_goal = env.regions.contains_points("goal", positions)

# Region operations
buffered = env.regions.buffer("goal", distance=10.0)
area = env.regions.area("goal", env)
center = env.regions.region_center("goal")
```

---

## 15. Serialization & I/O

### Save/Load Environments

```python
# Save environment (JSON metadata + NPZ arrays)
env.to_file("my_environment")
# Creates: my_environment.json, my_environment.npz

# Load environment
from neurospatial.io import from_file
loaded_env = from_file("my_environment")

# Export to dict
env_dict = env.to_dict()

# Import from dict
env = Environment.from_dict(env_dict)
```

### Validation

```python
from neurospatial import validate_environment

# Validate environment structure
validate_environment(env, strict=True)
# Checks: connectivity, metadata, units, etc.
```

---

## Advanced Features

### Differential Operators

```python
from neurospatial import gradient, divergence

# Gradient of scalar field
grad = gradient(env, scalar_field)
# Returns: (n_bins, n_dims) gradient vectors

# Divergence of vector field
div = divergence(env, vector_field)
# Returns: (n_bins,) divergence values
```

### Reward Fields

```python
from neurospatial import goal_reward_field, region_reward_field

# Exponential decay from goal
reward = goal_reward_field(
    env,
    goal_bin,
    decay_rate=0.1,
    max_reward=1.0
)

# Reward in region
reward = region_reward_field(env, region_mask, reward_value=1.0)
```

### Map Points to Bins (Batch)

```python
from neurospatial import map_points_to_bins

# Efficient batch mapping with KDTree caching
bin_indices = map_points_to_bins(
    points,
    env,
    tie_break="lowest_index"  # How to handle equidistant bins
)
```

---

## Summary of Validated Metrics

neurospatial includes implementations validated against field-standard packages:

### ✅ Validated Against:
- **opexebo** (5 exact matches < 1e-10 error)
- **neurocode** (5 exact matches)
- **buzcode** (via neurocode/FMAToolbox)
- **Traja** (trajectory metrics)
- **yupi** (trajectory metrics)
- **adehabitatHR** (home range)

### Metrics Coverage (~44% of standard hippocampal analyses):
- Skaggs information ✓
- Sparsity ✓
- Border score ✓
- Grid score ✓
- Field detection ✓
- Field stability ✓
- Turn angles ✓
- Step lengths ✓
- Home range ✓
- Population coverage ✓
- ...and more

---

## What neurospatial Does **NOT** Have (Yet)

From the Brandon Rhodes review and ecosystem comparison:

### Missing Features (Future Work):
1. **Bayesian decoder** - Population decoding
2. **Head direction support** - HD cell analysis (Priority 1)
3. **Shuffle tests** - Statistical significance testing
4. **Spatial coherence** - Alternate coherence metrics
5. **Online analysis** - Real-time streaming
6. **LFP/EEG analysis** - Theta, ripples (out of scope)

### By Design (Use Other Tools):
- **Spike sorting** → Use Kilosort, MountainSort
- **RL training** → Use RatInABox, Stable-Baselines3
- **Continuous simulation** → Use RatInABox

---

## Integration with Other Tools

### Complementary Packages:
- **RatInABox**: Generate synthetic data → analyze with neurospatial
- **track-linearization**: 1D linearization (used internally)
- **Shapely**: Polygon operations (used internally)
- **NetworkX**: Graph algorithms (used internally)

---

## Key Design Principles

1. **N-D Native**: All operations work in 1D, 2D, 3D, or higher dimensions
2. **Graph-Based**: Connectivity graph enables geodesic distances, path finding
3. **Validated**: Metrics match published algorithms (< 1e-10 error)
4. **Protocol-Based**: Duck typing with type safety (no inheritance hierarchies)
5. **Immutable Regions**: Regions are immutable dataclasses
6. **Fitted State**: `@check_fitted` decorator ensures proper initialization

---

## Quick Decision Tree: What Should I Use?

| Task | Use This |
|------|----------|
| Create environment from trajectory | `Environment.from_samples()` |
| 1D linearized track | `Environment.from_graph()` |
| Polygon-bounded arena | `Environment.from_polygon()` |
| Estimate place fields | `compute_place_field()` |
| Detect place fields | `detect_place_fields()` |
| Spatial information | `skaggs_information()` |
| Border score | `border_score()` |
| Grid score | `grid_score()` |
| Trajectory metrics | `compute_turn_angles()`, `compute_step_lengths()` |
| Generate test data | `neurospatial.simulation.*` |
| Merge environments | `CompositeEnvironment()` |
| Transform environments | `estimate_transform()`, `apply_transform_to_environment()` |
| Smooth fields | `env.smooth()` or `apply_kernel()` |

---

## Getting Started

```python
# Minimal complete example
import numpy as np
from neurospatial import Environment, compute_place_field
from neurospatial.metrics import skaggs_information, detect_place_fields

# 1. Create environment
positions = np.random.uniform(0, 100, (1000, 2))
env = Environment.from_samples(positions, bin_size=2.0)
env.units = "cm"

# 2. Compute place field
spike_times = np.array([0.5, 1.2, 1.8, 2.3])  # Example spikes
times = np.linspace(0, 10, 1000)
rate_map = compute_place_field(env, spike_times, times, positions)

# 3. Detect fields
fields = detect_place_fields(rate_map, env, threshold=0.2)

# 4. Compute metrics
occupancy = env.occupancy(times, positions)
info = skaggs_information(rate_map, occupancy)

print(f"Spatial information: {info:.3f} bits/spike")
print(f"Number of fields: {len(fields)}")
```

---

**That's neurospatial!** A comprehensive toolkit for spatial neuroscience analysis with validated metrics, flexible layouts, and powerful simulation capabilities.
