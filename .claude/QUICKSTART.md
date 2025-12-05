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
    smoothing_method="diffusion_kde",  # Default: graph-based boundary-aware KDE
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

### Behavioral Trajectory Metrics

Analyze path efficiency, goal-directed behavior, and vicarious trial and error (VTE).

**Path efficiency:**

```python
from neurospatial.metrics import compute_path_efficiency

# Compute path efficiency for a trajectory
result = compute_path_efficiency(
    env, positions, times, goal_position,
    metric="geodesic",       # Respects walls/obstacles
    reference_speed=20.0,    # Optional: for time efficiency
)

# Access results
print(f"Path efficiency: {result.efficiency:.1%}")
print(f"Traveled: {result.traveled_length:.1f} cm")
print(f"Shortest: {result.shortest_length:.1f} cm")
print(result.summary())

# Quick classification
if result.is_efficient(threshold=0.8):
    print("Efficient navigation!")
```

**Goal-directed metrics:**

```python
from neurospatial.metrics import compute_goal_directed_metrics, goal_bias

# Compute full goal-directed analysis
result = compute_goal_directed_metrics(env, positions, times, goal_position)

# Access results
print(f"Goal bias: {result.goal_bias:.2f}")  # Range [-1, 1]
print(f"Approach rate: {result.mean_approach_rate:.1f} cm/s")
print(result.summary())

if result.is_goal_directed(threshold=0.3):
    print("Goal-directed navigation detected!")

# Quick goal bias calculation
bias = goal_bias(positions, times, goal_position, min_speed=5.0)
# bias > 0: approaching goal; bias < 0: moving away
```

**VTE (Vicarious Trial and Error) detection:**

```python
from neurospatial.metrics import compute_vte_session, compute_vte_trial

# Analyze VTE behavior at decision points across a session
result = compute_vte_session(
    positions, times, trials,
    decision_region="center",  # Region name in env.regions
    env=env,
    window_duration=1.0,       # Pre-decision window (seconds)
    vte_threshold=0.5,         # Classification threshold
)

# Session-level summary
print(f"VTE trials: {result.n_vte_trials}/{len(result.trial_results)}")
print(f"VTE fraction: {result.vte_fraction:.1%}")
print(result.summary())

# Per-trial analysis
for trial in result.trial_results:
    if trial.is_vte:
        print(f"  VTE at {trial.window_end:.1f}s: IdPhi={trial.idphi:.2f} rad")

# Single trial analysis (no z-scoring)
single_result = compute_vte_trial(
    positions, times,
    entry_time=5.0,        # Time of decision region entry
    window_duration=1.0,
    min_speed=5.0,
)
print(f"Head sweep: {single_result.head_sweep_magnitude:.2f} rad")
```

**Decision point analysis:**

```python
from neurospatial.metrics import (
    compute_decision_analysis,
    compute_pre_decision_metrics,
    geodesic_voronoi_labels,
)

# Full decision analysis for a trial
result = compute_decision_analysis(
    env, positions, times,
    decision_region="center",
    goal_regions=["left", "right"],
    pre_window=1.0,
)

# Pre-decision metrics
print(f"Entry time: {result.entry_time:.2f}s")
print(f"Heading variance: {result.pre_decision.heading_circular_variance:.2f}")
if result.pre_decision.suggests_deliberation():
    print("High heading variance + low speed → possible deliberation")

# Decision boundary crossings
if result.boundary is not None:
    print(f"Boundary crossings: {result.boundary.n_crossings}")

# Geodesic Voronoi partition (label bins by nearest goal)
goal_bins = [env.bin_at(left_goal), env.bin_at(right_goal)]
labels = geodesic_voronoi_labels(env, goal_bins)
# Each bin labeled 0 or 1 based on nearest goal
```

### Egocentric Reference Frames

Transform between allocentric (world-centered) and egocentric (animal-centered) coordinates:

```python
from neurospatial import (
    allocentric_to_egocentric,
    egocentric_to_allocentric,
    compute_egocentric_bearing,
    compute_egocentric_distance,
    heading_from_velocity,
    heading_from_body_orientation,
    EgocentricFrame,
)
import numpy as np

# Compute heading from trajectory
positions = np.column_stack([x, y])  # Shape: (n_time, 2)
dt = times[1] - times[0]
headings = heading_from_velocity(positions, dt, min_speed=2.0, smoothing_sigma=3.0)

# Or from pose tracking keypoints
headings = heading_from_body_orientation(nose_positions, tail_positions)

# Transform landmarks to egocentric coordinates
landmarks = np.array([[50.0, 30.0], [80.0, 60.0]])  # 2 landmarks
ego_landmarks = allocentric_to_egocentric(landmarks, positions, headings)
# ego_landmarks shape: (n_time, n_landmarks, 2)
# ego_landmarks[t, i, :] = landmark i in animal's reference frame at time t

# Compute bearing (angle) to targets
bearings = compute_egocentric_bearing(landmarks, positions, headings)
# bearings shape: (n_time, n_landmarks)
# 0=ahead, pi/2=left, -pi/2=right, +/-pi=behind

# Compute distances (Euclidean or geodesic)
distances = compute_egocentric_distance(
    landmarks, positions, headings,
    metric="euclidean"  # or "geodesic" with env parameter
)

# For geodesic distances (respects walls/obstacles)
distances = compute_egocentric_distance(
    landmarks, positions, headings,
    metric="geodesic", env=env
)
```

**Coordinate conventions:**

- Allocentric: 0=East, π/2=North (standard math convention)
- Egocentric: 0=ahead, π/2=left, -π/2=right, ±π=behind

**Create egocentric polar environment (for object-vector cells):**

```python
# Create polar grid in egocentric space
ego_env = Environment.from_polar_egocentric(
    distance_range=(0, 50),     # 0-50 cm from animal
    angle_range=(-np.pi, np.pi), # Full 360° around animal
    distance_bin_size=5.0,       # 5 cm radial bins
    angle_bin_size=np.pi / 8,    # 22.5° angular bins
    circular_angle=True,         # Wrap angles at ±π
)
# ego_env.bin_centers[:, 0] = distances
# ego_env.bin_centers[:, 1] = angles
```

### Object-Vector Cells

Analyze cells that encode distance and direction to objects in egocentric coordinates:

**Compute object-vector field:**

```python
from neurospatial import compute_object_vector_field

# Define object positions in allocentric (world) coordinates
object_positions = np.array([[50.0, 30.0], [80.0, 60.0]])  # 2 objects

# Compute object-vector field (egocentric polar representation)
result = compute_object_vector_field(
    spike_times=spike_times,          # Spike times for one neuron
    times=times,                       # Trajectory timestamps
    positions=positions,               # Animal positions (n_time, 2)
    headings=headings,                 # Animal heading angles (radians)
    object_positions=object_positions,
    max_distance=50.0,                 # Max distance to consider (cm)
    n_distance_bins=10,                # Number of radial bins
    n_direction_bins=12,               # Number of angular bins
    smoothing_method="diffusion_kde",            # or "binned"
)

# Access results
field = result.field           # Firing rate in egocentric polar space
ego_env = result.ego_env       # Egocentric polar environment
occupancy = result.occupancy   # Time spent in each bin

# Find preferred distance and direction
peak_idx = np.nanargmax(field)
preferred_distance = ego_env.bin_centers[peak_idx, 0]
preferred_direction = ego_env.bin_centers[peak_idx, 1]
print(f"Preferred: {preferred_distance:.1f} cm at {np.degrees(preferred_direction):.0f}°")
```

**Classify object-vector cells:**

```python
from neurospatial.metrics import (
    compute_object_vector_tuning,
    is_object_vector_cell,
    plot_object_vector_tuning,
)

# Compute tuning metrics
metrics = compute_object_vector_tuning(
    spike_times=spike_times,
    times=times,
    positions=positions,
    headings=headings,
    object_positions=object_positions,
)

# Check classification
print(metrics.interpretation())  # Human-readable summary
print(f"OVC score: {metrics.object_vector_score:.3f}")
print(f"Peak rate: {metrics.peak_rate:.1f} Hz")

# Classify
if is_object_vector_cell(metrics, threshold=0.3, min_peak_rate=5.0):
    print("This is an object-vector cell!")

# Plot polar tuning curve
fig, ax = plot_object_vector_tuning(metrics, mark_peak=True, colorbar=True)
```

**Simulate object-vector cells:**

```python
from neurospatial.simulation import ObjectVectorCellModel, generate_poisson_spikes

# Create OVC model
ovc = ObjectVectorCellModel(
    env=env,
    object_positions=object_positions,
    preferred_distance=20.0,        # Peak at 20 cm
    distance_width=8.0,             # Gaussian width
    preferred_direction=np.pi/4,    # 45° left (optional)
    direction_kappa=4.0,            # Direction tuning sharpness
    max_rate=30.0,                  # Peak firing rate (Hz)
)

# Generate firing rates along trajectory
rates = ovc.firing_rate(positions, headings=headings)

# Generate spikes
spike_times = generate_poisson_spikes(rates, times, seed=42)

# Access ground truth parameters
print(ovc.ground_truth)
```

**Animate with object-vector overlay:**

```python
from neurospatial.animation import ObjectVectorOverlay

# Create overlay showing vectors to objects
overlay = ObjectVectorOverlay(
    object_positions=object_positions,  # Static object locations
    animal_positions=trajectory,         # Animal trajectory (n_frames, 2)
    times=times,                          # Optional: for interpolation
    firing_rates=rates,                   # Optional: modulate line appearance
    color="cyan",
    linewidth=2.0,
    show_objects=True,                   # Mark object locations
)

env.animate_fields(fields, frame_times=frame_times, overlays=[overlay])
```

### Spatial View Cells

Analyze cells that fire when the animal is *looking at* a specific location (not *at* that location):

**Compute spatial view field:**

```python
from neurospatial import compute_spatial_view_field

# Compute view field (binned by *viewed location*, not position)
result = compute_spatial_view_field(
    env=env,
    spike_times=spike_times,
    times=times,
    positions=positions,
    headings=headings,
    view_distance=15.0,      # Fixed viewing distance
    gaze_model="fixed_distance",  # or "ray_cast", "boundary"
    smoothing_method="diffusion_kde",  # or "binned", "gaussian_kde"
)

# Access results
view_field = result.field          # Firing rate by viewed location
view_occupancy = result.view_occupancy  # Time viewing each bin

# Compare to place field (binned by position)
from neurospatial import compute_place_field
place_field = compute_place_field(env, spike_times, times, positions)

# For spatial view cells: view_field differs from place_field
```

**Classify spatial view cells:**

```python
from neurospatial import (
    spatial_view_cell_metrics,
    is_spatial_view_cell,
    SpatialViewMetrics,
)

# Compute metrics comparing view field vs place field
metrics = spatial_view_cell_metrics(
    env=env,
    spike_times=spike_times,
    times=times,
    positions=positions,
    headings=headings,
    view_distance=15.0,
)

# Human-readable interpretation
print(metrics.interpretation())
# Shows: view info, place info, correlation, classification reasoning

# Access individual metrics
print(f"View field info: {metrics.view_field_skaggs_info:.3f} bits/spike")
print(f"Place field info: {metrics.place_field_skaggs_info:.3f} bits/spike")
print(f"View-place correlation: {metrics.view_place_correlation:.3f}")

# Quick classification
if is_spatial_view_cell(env, spike_times, times, positions, headings):
    print("Spatial view cell detected!")
```

**Simulate spatial view cells:**

```python
from neurospatial import SpatialViewCellModel
from neurospatial.simulation import generate_poisson_spikes

# Create model that fires when looking at specific location
svc = SpatialViewCellModel(
    env=env,
    preferred_view_location=np.array([50.0, 50.0]),  # Fires when viewing here
    view_field_width=10.0,   # Gaussian tuning width
    view_distance=15.0,       # Fixed viewing distance
    gaze_model="fixed_distance",
    max_rate=20.0,
)

# Generate firing rates
rates = svc.firing_rate(positions, times=times, headings=headings)

# Generate spikes
spike_times = generate_poisson_spikes(rates, times, seed=42)
```

**Visibility and gaze analysis:**

```python
from neurospatial import (
    compute_viewed_location,
    compute_viewshed,
    FieldOfView,
    visible_cues,
)

# Compute where animal is looking
viewed_locations = compute_viewed_location(
    positions, headings, view_distance=15.0,
    method="fixed_distance",  # or "ray_cast", "boundary"
)

# Species-specific field of view
fov = FieldOfView.rat()   # ~320° total FOV
fov = FieldOfView.primate()  # ~180° total FOV

# Compute visible bins from a position
viewshed = compute_viewshed(
    env, position=np.array([50, 50]),
    heading=0.0,  # Facing East
    fov=fov,
    n_rays=360,
)
print(f"Visible bins: {viewshed.n_visible_bins}")

# Check which cues/landmarks are visible
cue_positions = np.array([[80, 50], [20, 80]])
visible, distances, bearings = visible_cues(
    env, observer_position=np.array([50, 50]),
    observer_heading=0.0, cue_positions=cue_positions
)
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

### Circular Basis Functions for GLMs

For circular predictors (head direction, theta phase, running direction):

```python
from neurospatial.metrics import (
    circular_basis,
    circular_basis_metrics,
    plot_circular_basis_tuning,
)
import statsmodels.api as sm

# Create design matrix from head direction angles
result = circular_basis(head_direction_angles)  # angles in radians
X = result.design_matrix  # Shape: (n_samples, 2) = [sin(θ), cos(θ)]

# Fit GLM
X_with_const = sm.add_constant(X)
model = sm.GLM(spike_counts, X_with_const, family=sm.families.Poisson())
fit = model.fit()

# Extract tuning parameters
beta_sin, beta_cos = fit.params[1], fit.params[2]
amplitude, preferred_direction, p_value = circular_basis_metrics(
    beta_sin, beta_cos,
    var_sin=fit.cov_params()[1, 1],
    var_cos=fit.cov_params()[2, 2],
    cov_sin_cos=fit.cov_params()[1, 2],
)

# Visualize tuning curve
plot_circular_basis_tuning(beta_sin, beta_cos, projection="polar")
```

**Related functions:**

- `head_direction_metrics()`: Complete head direction cell analysis
- `phase_precession()`: Detect theta phase precession
- `rayleigh_test()`: Test for circular uniformity

### Events and Peri-Event Analysis

**Peri-event time histogram (PSTH):**

```python
from neurospatial import peri_event_histogram, plot_peri_event_histogram

# Compute PSTH around reward events
result = peri_event_histogram(
    spike_times,          # Array of spike times
    reward_times,         # Array of event times
    window=(-1.0, 2.0),   # -1s before to 2s after event
    bin_size=0.025,       # 25 ms bins
)

# Access results
print(f"Peak firing rate: {result.firing_rate().max():.1f} Hz")
print(f"Number of events: {result.n_events}")

# Plot PSTH
plot_peri_event_histogram(result, show_sem=True, as_rate=True)
```

**Population PSTH across multiple neurons:**

```python
from neurospatial import population_peri_event_histogram

# Analyze multiple neurons
spike_trains = [neuron1_spikes, neuron2_spikes, neuron3_spikes]
result = population_peri_event_histogram(
    spike_trains, event_times, window=(-1.0, 2.0), bin_size=0.025
)

print(f"Population mean shape: {result.mean_histogram.shape}")
```

**GLM regressors from events:**

```python
from neurospatial import time_to_nearest_event, event_indicator, event_count_in_window

# Time to nearest event (for peri-event time covariate)
peri_event_time = time_to_nearest_event(
    sample_times, reward_times,
    max_time=5.0,   # Limit to 5s from event
    signed=True,    # Negative before, positive after
)

# Binary indicator of event presence
is_near_reward = event_indicator(
    sample_times, reward_times, window=(-0.5, 1.0)
)

# Count events in sliding window
n_rewards = event_count_in_window(
    sample_times, reward_times, window=(-2.0, 0.0)  # Events in past 2s
)
```

**Spatial distance regressors:**

```python
from neurospatial import distance_to_reward, distance_to_boundary

# Distance to reward location (requires Environment)
dist_to_reward = distance_to_reward(
    env, positions, times, reward_times,
    mode="next",      # Distance to upcoming reward
    metric="geodesic" # Respects walls/obstacles
)

# Distance to environment boundaries
dist_to_wall = distance_to_boundary(
    env, positions, boundary_type="edge"
)

# Distance to named region boundary
dist_to_goal = distance_to_boundary(
    env, positions, boundary_type="region", region_name="goal"
)

# Build GLM design matrix
import numpy as np
X = np.column_stack([
    time_to_nearest_event(times, reward_times, max_time=5.0),
    dist_to_reward,
    dist_to_wall,
])
```

**Add spatial positions to events:**

```python
from neurospatial import add_positions
import pandas as pd

# Create events DataFrame (must have 'timestamp' column)
events = pd.DataFrame({
    "timestamp": [1.5, 3.2, 5.8],
    "label": ["reward", "lick", "reward"],
})

# Add x, y columns by interpolation from trajectory
events_with_pos = add_positions(events, trajectory, trajectory_times)
print(events_with_pos.columns)  # ['timestamp', 'label', 'x', 'y']
```

**Interval utilities:**

```python
from neurospatial import intervals_to_events, events_to_intervals, filter_by_intervals

# Convert intervals to point events
intervals = pd.DataFrame({"start_time": [0, 10], "stop_time": [5, 15]})
events = intervals_to_events(intervals, which="start")  # Get start times

# Filter events to keep only those within intervals
valid_events = filter_by_intervals(
    all_events, intervals, include=True  # Set to False to exclude
)
```

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
