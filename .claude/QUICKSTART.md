# Quick Start Guide

Essential patterns for daily use. Copy-paste and modify for your needs.

---

## Your First Environment

```python
from neurospatial import Environment
import numpy as np

# Generate sample position data
positions = np.random.rand(100, 2) * 100  # 100 points in 2D space

# Create environment (bin_size is REQUIRED)
env = Environment.from_samples(positions, bin_size=2.0)

# Set metadata (recommended)
env.units = "cm"
env.frame = "session1"

# Query the environment
bin_idx = env.bin_at([50.0, 50.0])
neighbors = env.neighbors(bin_idx)
```

---

## Essential Commands

```bash
# Run all tests
uv run pytest

# Lint and format
uv run ruff check . && uv run ruff format .

# Type checking
uv run mypy src/neurospatial/
```

---

## Common Patterns

### Environment Creation

**From position data (most common):**

```python
from neurospatial import Environment

# Basic usage
env = Environment.from_samples(positions, bin_size=2.0)
env.units = "cm"
env.frame = "session1"

# Save and load
env.to_file("my_environment")  # Creates .json + .npz files
loaded_env = Environment.from_file("my_environment")

# Validate
from neurospatial import validate_environment
validate_environment(env, strict=True)  # Warns if units/frame missing
```

**From polygon boundary:**

```python
from shapely.geometry import Polygon

# Define arena boundary
boundary = Polygon([(0, 0), (100, 0), (100, 80), (0, 80)])
env = Environment.from_polygon(boundary, bin_size=2.0)
env.units = "cm"
```

**1D linearized track:**

```python
import networkx as nx

# Create track graph (nodes = track points, edges = connections)
G = nx.Graph()
G.add_nodes_from([(0, {"pos": (0, 0)}), (1, {"pos": (50, 0)}), (2, {"pos": (100, 0)})])
G.add_edges_from([(0, 1), (1, 2)])

env = Environment.from_graph(G, bin_size=2.0)
print(env.is_1d)  # True

# Linearization methods available
linear_pos = env.to_linear(nd_position)
nd_pos = env.linear_to_nd(linear_pos)
```

### Neural Analysis

**Place fields:**

```python
from neurospatial import compute_place_field

# Compute place field for one neuron
firing_rate = compute_place_field(
    env, spike_times, times, positions,
    method="diffusion_kde",  # Default: graph-based boundary-aware KDE
    bandwidth=5.0  # Smoothing bandwidth (cm)
)
# Methods: "diffusion_kde" (default), "gaussian_kde", "binned" (legacy)
```

**Bayesian decoding:**

```python
from neurospatial import decode_position, decoding_error

# Build encoding models (one per neuron)
encoding_models = np.array([
    compute_place_field(env, spike_times_list[i], times, positions, bandwidth=8.0)
    for i in range(n_neurons)
])  # Shape: (n_neurons, n_bins)

# Bin spikes for decoding
dt = 0.025  # 25 ms time bins
time_bins = np.arange(0, times[-1], dt)
spike_counts = np.zeros((len(time_bins) - 1, n_neurons), dtype=np.int64)
for i, spikes in enumerate(spike_times_list):
    spike_counts[:, i], _ = np.histogram(spikes, bins=time_bins)

# Decode position
result = decode_position(env, spike_counts, encoding_models, dt, times=time_bins[:-1] + dt/2)

# Access results (lazy-computed, cached)
posterior = result.posterior       # (n_time_bins, n_bins) probability
map_pos = result.map_position      # (n_time_bins, n_dims) MAP estimates
errors = decoding_error(map_pos, actual_positions)  # Per-bin error

# Visualize
result.plot(show_map=True, colorbar=True)
```

**Trajectory analysis:**

```python
from neurospatial.segmentation import segment_trials

# Segment trials from trajectory
trials = segment_trials(
    trajectory_bins, times, env,
    start_region="home",
    end_regions=["reward_left", "reward_right"],
)

# Analyze each trial
for t in trials:
    print(f"{t.start_region} -> {t.end_region}: {'success' if t.success else 'timeout'}")
```

### Visualization & Animation

**Animate spatial fields:**

```python
# IMPORTANT: frame_times is REQUIRED - provides temporal structure
frame_times = np.arange(len(fields)) / 30.0  # 30 Hz timestamps

# Interactive napari viewer (best for 100K+ frames)
env.animate_fields(fields, frame_times=frame_times, backend="napari")

# Video export with parallel rendering (requires ffmpeg)
env.clear_cache()  # Required before parallel rendering
env.animate_fields(
    fields, frame_times=frame_times, speed=1.0,
    backend="video", save_path="animation.mp4", n_workers=4
)

# Speed control (speed = playback multiplier)
# speed=1.0: real-time, speed=0.1: slow motion, speed=2.0: fast forward
env.animate_fields(fields, frame_times=frame_times, speed=0.1)  # 10% speed
```

**Add trajectory overlays:**

```python
from neurospatial import PositionOverlay, BodypartOverlay, HeadDirectionOverlay

# Position overlay with trail
position_overlay = PositionOverlay(
    data=trajectory,  # Shape: (n_frames, n_dims) in environment (x, y) coordinates
    color="red",
    size=12.0,
    trail_length=10  # Show last 10 frames as decaying trail
)
env.animate_fields(fields, frame_times=frame_times, overlays=[position_overlay])

# Pose tracking with skeleton
bodypart_overlay = BodypartOverlay(
    data={"nose": nose_traj, "body": body_traj, "tail": tail_traj},
    skeleton=[("tail", "body"), ("body", "nose")],
    colors={"nose": "yellow", "body": "red", "tail": "blue"},
)
env.animate_fields(fields, frame_times=frame_times, overlays=[bodypart_overlay])

# Multi-animal tracking
animal1 = PositionOverlay(data=traj1, color="red", trail_length=10)
animal2 = PositionOverlay(data=traj2, color="blue", trail_length=10)
env.animate_fields(fields, frame_times=frame_times, overlays=[animal1, animal2])
```

**Video overlay:**

```python
from neurospatial.animation import calibrate_video, VideoOverlay

# Calibrate video to environment coordinates
calibration = calibrate_video("session.mp4", env, cm_per_px=0.25)

# Create video overlay
video_overlay = VideoOverlay(
    source="session.mp4",
    calibration=calibration,
    alpha=0.5,  # 50% blend (0.3=field dominant, 0.7=video dominant)
)
env.animate_fields(fields, frame_times=frame_times, overlays=[video_overlay])
```

### Working with Regions

```python
# Add regions
env.regions.add("goal", point=(50.0, 50.0))
env.regions.add("start", point=(10.0, 10.0))

# Query regions
bins_in_goal = env.bins_in_region("goal")
is_in_start = env.point_in_region((12.0, 12.0), "start")

# Update region (don't modify in place - regions are immutable)
env.regions.update_region("goal", point=(55.0, 55.0))  # No warning
```

### Spatial Basis Functions for GLMs

Create maze-aware basis functions that respect walls and barriers for spatial regression:

```python
from neurospatial import Environment, geodesic_rbf_basis, spatial_basis

# Create environment
env = Environment.from_samples(positions, bin_size=2.0)
env.units = "cm"

# Option 1: Automatic (recommended for most users)
basis = spatial_basis(env, n_features=100)  # Automatic parameter selection

# Option 2: Manual control with geodesic RBF
basis = geodesic_rbf_basis(
    env,
    n_centers=50,        # Number of basis centers
    sigma=[5.0, 10.0],   # Bandwidths in cm (multi-scale)
)  # Shape: (n_centers * n_sigmas, n_bins)

# Create GLM design matrix from trajectory
bin_indices = env.bin_sequence(trajectory, times)
X_spatial = basis[:, bin_indices].T  # Shape: (n_times, n_basis)

# Fit GLM (example with statsmodels)
import statsmodels.api as sm
X = sm.add_constant(X_spatial)
model = sm.GLM(spike_counts, X, family=sm.families.Poisson())
result = model.fit()

# Visualize fitted place field
beta_spatial = result.params[1:]  # Spatial coefficients
place_field = beta_spatial @ basis  # Project back to space
env.plot_field(place_field, title="Fitted Place Field")
```

**Three basis types available:**

- `geodesic_rbf_basis`: RBF using shortest-path distances (start here)
- `heat_kernel_wavelet_basis`: Diffusion-based multi-scale (rooms/corridors)
- `chebyshev_filter_basis`: Polynomial filters (fast, large environments)

### NWB Integration (Optional)

Requires: `pip install neurospatial[nwb-full]`

```python
from pynwb import NWBHDF5IO
from neurospatial.nwb import (
    read_position, environment_from_position,
    write_place_field, write_environment,
)

# Read position data
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()
    positions, timestamps = read_position(nwbfile)
    env = environment_from_position(nwbfile, bin_size=2.0, units="cm")

# Write analysis results
with NWBHDF5IO("session.nwb", "r+") as io:
    nwbfile = io.read()
    write_place_field(nwbfile, env, place_field, name="cell_001")
    write_environment(nwbfile, env, name="linear_track")
    io.write(nwbfile)
```

**For complete NWB API, see [ADVANCED.md](ADVANCED.md#nwb-integration).**

---

## Critical Rules

1. **ALWAYS use `uv run`** before Python commands: `uv run pytest`, `uv run python script.py`
2. **NEVER modify bare `Environment()`** - use factory methods: `Environment.from_samples()`
3. **bin_size is required** for all Environment creation
4. **NumPy docstring format** for all documentation

---

## Next Steps

- **Common issues?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Import patterns?** See [API_REFERENCE.md](API_REFERENCE.md)
- **Extend the codebase?** See [PATTERNS.md](PATTERNS.md)
- **Architecture details?** See [ARCHITECTURE.md](ARCHITECTURE.md)
