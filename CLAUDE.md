# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Last Updated**: 2025-11-27 (v0.9.0 - Track graph annotation for 1D environments)

## Table of Contents

- [Quick Reference](#quick-reference)
- [Project Overview](#project-overview)
- [Package Management with uv](#package-management-with-uv)
- [Core Architecture](#core-architecture)
- [Import Patterns](#import-patterns)
- [Important Patterns & Constraints](#important-patterns--constraints)
- [Development Commands](#development-commands)
- [Key Implementation Notes](#key-implementation-notes)
- [Testing Structure](#testing-structure)
- [Documentation Style](#documentation-style)
- [Common Gotchas](#common-gotchas)
- [Troubleshooting](#troubleshooting)

## Quick Reference

**Most Common Commands**:

```bash
# Run all tests (from project root)
uv run pytest

# Run specific test
uv run pytest tests/test_environment.py::test_function_name -v

# Run with coverage
uv run pytest --cov=src/neurospatial

# Lint and format
uv run ruff check . && uv run ruff format .

# Run doctests
uv run pytest --doctest-modules src/neurospatial/

# Run performance benchmarks (slow tests)
uv run pytest -m slow -v -s

# Run all tests except performance benchmarks
uv run pytest -m "not slow"
```

**Most Common Patterns**:

```python
# Create environment from data
env = Environment.from_samples(positions, bin_size=2.0)  # bin_size is required

# Add units and coordinate frame (v0.1.0+)
env.units = "cm"
env.frame = "session1"

# Save and load environments (v0.1.0+)
env.to_file("my_environment")  # Creates .json + .npz files
loaded_env = Environment.from_file("my_environment")

# Map points to bins with KDTree caching (v0.1.0+)
from neurospatial import map_points_to_bins
bin_indices = map_points_to_bins(points, env, tie_break="lowest_index")

# Estimate transform from corresponding points (v0.1.0+)
from neurospatial import estimate_transform, apply_transform_to_environment
transform = estimate_transform(src_landmarks, dst_landmarks, kind="rigid")
aligned_env = apply_transform_to_environment(env, T)

# Compute distance fields (v0.1.0+)
from neurospatial import distance_field
distances = distance_field(env.connectivity, sources=[goal_bin_id])

# Compute place fields from spike data (v0.2.0+)
from neurospatial import compute_place_field
firing_rate = compute_place_field(
    env, spike_times, times, positions,
    method="diffusion_kde",  # Default: graph-based boundary-aware KDE
    bandwidth=5.0  # Smoothing bandwidth (cm)
)
# Methods: "diffusion_kde" (default), "gaussian_kde", "binned" (legacy)

# Bayesian Position Decoding (v0.12.0+)
from neurospatial import DecodingResult, decode_position, decoding_error, median_decoding_error

# Build encoding models from place fields (one per neuron)
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

# Decode position from population activity
result = decode_position(
    env, spike_counts, encoding_models, dt,
    prior=None,  # Uniform prior (or custom prior array)
    times=time_bins[:-1] + dt/2,  # Time bin centers
)

# Access results (lazy-computed, cached)
posterior = result.posterior       # (n_time_bins, n_bins) probability distribution
map_pos = result.map_position      # (n_time_bins, n_dims) MAP position estimates
mean_pos = result.mean_position    # (n_time_bins, n_dims) posterior mean
uncertainty = result.uncertainty   # (n_time_bins,) entropy in bits

# Evaluate decoding accuracy
from neurospatial.decoding import decoding_error, median_decoding_error
errors = decoding_error(map_pos, actual_positions)  # Per-time-bin error
median_err = median_decoding_error(map_pos, actual_positions)  # Summary statistic

# Visualization
result.plot()  # Posterior heatmap with optional MAP overlay
result.plot(show_map=True, colorbar=True)  # With MAP trajectory and colorbar
df = result.to_dataframe()  # Export to pandas DataFrame

# Trajectory analysis (for replay detection)
from neurospatial.decoding import (
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    detect_trajectory_radon,
)

# Fit monotonic trajectory (isotonic regression)
iso_result = fit_isotonic_trajectory(posterior, times, method="expected")
print(f"R²={iso_result.r_squared:.3f}, direction={iso_result.direction}")

# Fit linear trajectory with uncertainty (Monte Carlo sampling)
lin_result = fit_linear_trajectory(env, posterior, times, method="sample", rng=42)
print(f"Slope={lin_result.slope:.1f} bins/s ± {lin_result.slope_std:.2f}")

# Detect trajectory angle with Radon transform (requires scikit-image)
radon_result = detect_trajectory_radon(posterior)
print(f"Detected angle: {radon_result.angle_degrees:.1f}°")

# Shuffle-based significance testing
from neurospatial.decoding import (
    shuffle_time_bins,
    shuffle_cell_identity,
    compute_shuffle_pvalue,
    ShuffleTestResult,
)

# Test if sequential structure is significant
observed_score = iso_result.r_squared
null_scores = []
for shuffled in shuffle_time_bins(spike_counts, n_shuffles=1000, rng=42):
    shuffled_result = decode_position(env, shuffled, encoding_models, dt)
    null_fit = fit_isotonic_trajectory(shuffled_result.posterior, times)
    null_scores.append(null_fit.r_squared)

p_value = compute_shuffle_pvalue(observed_score, np.array(null_scores))
print(f"Sequence p-value: {p_value:.4f}")

# Validate environment (v0.1.0+)
from neurospatial import validate_environment
validate_environment(env, strict=True)  # Warns if units/frame missing

# Update a region (don't modify in place)
env.regions.update_region("goal", point=new_point)

# Check if 1D before using linearization
if env.is_1d:
    linear_pos = env.to_linear(nd_position)

# Always use factory methods, not bare Environment()
env = Environment.from_samples(...)  # ✓ Correct
env = Environment()  # ✗ Wrong - won't be fitted

# Memory safety: automatic warnings for large grids (v0.2.1+)
# Warns at 100MB estimated memory (but creation still proceeds)
positions = np.random.uniform(0, 10000, (1000, 2))
env = Environment.from_samples(positions, bin_size=1.0)  # May warn, but will succeed

# Disable warning if intentional (v0.2.1+)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))

# Segment trials from trajectory data (v0.7.0+)
from neurospatial.segmentation import segment_trials

trials = segment_trials(
    trajectory_bins, times, env,
    start_region="home",
    end_regions=["reward_left", "reward_right"],
)

# Each trial has start_region and end_region
for t in trials:
    print(f"{t.start_region} -> {t.end_region}: {'success' if t.success else 'timeout'}")

# Behavioral & Goal-Directed Metrics (v0.8.0+)
from neurospatial import (
    path_progress,              # Normalized progress (0→1) along path
    distance_to_region,         # Distance to goal region over time
    cost_to_goal,               # RL cost with terrain/avoidance
    time_to_goal,               # Time until goal arrival
    compute_trajectory_curvature,  # Continuous curvature analysis
    graph_turn_sequence,        # Discrete turn labels
    trials_to_region_arrays,    # Helper for trial arrays
)

# Path progress for multiple trials (vectorized)
trials = segment_trials(trajectory_bins, times, env,
                        start_region="home", end_regions=["goal"])
start_bins, goal_bins = trials_to_region_arrays(trials, times, env)
progress = path_progress(env, trajectory_bins, start_bins, goal_bins)

# Distance to goal over time
goal_bin = env.bins_in_region('reward_zone')[0]
dist = distance_to_region(env, trajectory_bins, goal_bin)

# Cost-to-goal with learned avoidance
cost_map = np.ones(env.n_bins)
cost_map[punishment_bins] = 10.0  # Avoid punishment zone
cost = cost_to_goal(env, trajectory_bins, goal_bins, cost_map=cost_map)

# Time remaining until goal arrival
ttg = time_to_goal(times, trials)

# Trajectory curvature (for GLM regressors)
curvature = compute_trajectory_curvature(trajectory_positions, times)
is_turning = np.abs(curvature) > np.pi / 4  # Sharp turns (>45°)

# Turn sequence classification
for trial in trials:
    mask = (times >= trial.start_time) & (times <= trial.end_time)
    turn_seq = graph_turn_sequence(
        env, trajectory_bins[mask],
        start_bin=env.bins_in_region(trial.start_region)[0],
        end_bin=env.bins_in_region(trial.end_region)[0]
    )
    print(f"Trial {trial.start_region} → {trial.end_region}: {turn_seq}")  # e.g., "left-right"

# Directional Place Fields (v0.10.0+)
from neurospatial import (
    compute_directional_place_fields,
    goal_pair_direction_labels,
    heading_direction_labels,
)

# For trialized tasks (T-maze, Y-maze, linear track)
trials = segment_trials(trajectory_bins, times, env,
                        start_region="home", end_regions=["goal"])
labels = goal_pair_direction_labels(times, trials)
result = compute_directional_place_fields(
    env, spike_times, times, positions, labels, bandwidth=5.0
)
outbound_field = result.fields["home→goal"]
inbound_field = result.fields["goal→home"]

# For open fields (heading-based)
labels = heading_direction_labels(positions, times, n_directions=8)
result = compute_directional_place_fields(
    env, spike_times, times, positions, labels, bandwidth=5.0
)

# Animate spatial fields over time (v0.3.0+)
from neurospatial.animation import subsample_frames

# IMPORTANT: frame_times is REQUIRED - provides temporal structure from your data
# speed controls playback: 1.0 = real-time, 0.1 = slow motion, 2.0 = fast forward

# Interactive Napari viewer (best for 100K+ frames)
# Example: 30 Hz position tracking data
frame_times = np.arange(len(fields)) / 30.0  # 30 Hz timestamps
env.animate_fields(fields, frame_times=frame_times, backend="napari")

# Replay decoding: 500 Hz data, view at 10% speed (slow motion)
# playback_fps = 500 * 0.1 = 50 fps (within 60 fps limit)
decode_times = np.arange(len(posterior_fields)) / 500.0  # 500 Hz timestamps
env.animate_fields(posterior_fields, frame_times=decode_times, speed=0.1, backend="napari")

# Video export with parallel rendering (requires ffmpeg)
env.animate_fields(
    fields, frame_times=frame_times, speed=1.0,
    backend="video", save_path="animation.mp4", n_workers=4
)

# HTML standalone player (max 500 frames)
env.animate_fields(fields, frame_times=frame_times, backend="html", save_path="animation.html")

# Jupyter widget (notebook integration)
env.animate_fields(fields, frame_times=frame_times, backend="widget")

# Auto-select backend based on save_path or context
env.animate_fields(fields, frame_times=frame_times, save_path="animation.mp4")

# Subsample large datasets for video export (e.g., 250 Hz → 30 fps)
subsampled_fields = subsample_frames(fields, source_fps=250, target_fps=30)

# IMPORTANT: Clear caches before parallel rendering (pickle-ability requirement)
env.clear_cache()  # Makes environment pickle-able for multiprocessing
env.animate_fields(
    fields, frame_times=frame_times, backend="video", n_workers=4, save_path="output.mp4"
)

# Large session helpers (v0.x.x+)
from neurospatial.animation import (
    estimate_colormap_range_from_subset,
    large_session_napari_config,
)

# Pre-compute colormap range from subset (~10K frames) instead of scanning all data
# Essential for large sessions to avoid napari scanning millions of values
vmin, vmax = estimate_colormap_range_from_subset(fields, seed=42)

# Get recommended napari settings based on session size
napari_config = large_session_napari_config(n_frames=500_000, sample_rate_hz=250)
# Returns: {'speed': 1.0, 'chunk_size': 1000, 'max_chunks': 50}

# Combine for large session workflow
session_times = np.arange(500_000) / 250.0  # 250 Hz, ~33 minutes
env.animate_fields(
    fields,
    frame_times=session_times,
    backend="napari",
    vmin=vmin,       # Pre-computed from subset (fast)
    vmax=vmax,       # Pre-computed from subset (fast)
    **napari_config, # Optimized speed, chunk_size, max_chunks
)

# Scale bars on visualizations (v0.11.0+)
from neurospatial import ScaleBarConfig

# Static plots with scale bar
ax = env.plot_field(field, scale_bar=True)  # Auto-sized based on extent
ax = env.plot(scale_bar=True)  # Works with plot() too

# Custom scale bar configuration
config = ScaleBarConfig(length=20, position="lower left", color="white")
ax = env.plot_field(field, scale_bar=config)

# Scale bars in animations
env.animate_fields(fields, frame_times=frame_times, scale_bar=True, backend="napari")
env.animate_fields(fields, frame_times=frame_times, scale_bar=True, save_path="video.mp4")

# Note: scale_bar is different from calibrate_video(scale_bar=...) which is for
# calibrating video coordinates. This scale_bar adds visual scale bars to plots.

# Animation overlays (v0.4.0+)
from neurospatial import PositionOverlay, BodypartOverlay, HeadDirectionOverlay, EventOverlay, TimeSeriesOverlay

# Position overlay with trail
# NOTE: All overlay coordinates use environment space (x, y) - automatic napari conversion
position_overlay = PositionOverlay(
    data=trajectory,  # Shape: (n_frames, n_dims) in environment (x, y) coordinates
    color="red",
    size=12.0,
    trail_length=10  # Show last 10 frames as decaying trail
)
env.animate_fields(fields, frame_times=frame_times, overlays=[position_overlay], backend="napari")

# Pose tracking with skeleton
bodypart_overlay = BodypartOverlay(
    data={"nose": nose_traj, "body": body_traj, "tail": tail_traj},
    skeleton=[("tail", "body"), ("body", "nose")],
    colors={"nose": "yellow", "body": "red", "tail": "blue"},
    skeleton_color="white",
    skeleton_width=2.0
)
env.animate_fields(fields, frame_times=frame_times, overlays=[bodypart_overlay], backend="napari")

# Head direction arrows
head_direction_overlay = HeadDirectionOverlay(
    data=head_angles,  # Shape: (n_frames,) radians OR (n_frames, n_dims) unit vectors
    color="yellow",
    length=15.0  # Arrow length in environment units
)
env.animate_fields(fields, frame_times=frame_times, overlays=[head_direction_overlay], backend="napari")

# Multi-animal tracking (multiple overlays)
animal1 = PositionOverlay(data=traj1, color="red", trail_length=10)
animal2 = PositionOverlay(data=traj2, color="blue", trail_length=10)
env.animate_fields(fields, frame_times=frame_times, overlays=[animal1, animal2], backend="napari")

# Event overlays (v0.13.0+) - visualize spikes, licks, rewards at spatial positions
from neurospatial import EventOverlay, SpikeOverlay  # SpikeOverlay is alias

# Single neuron spikes at animal position (trajectory mode)
spike_overlay = EventOverlay(
    event_times=spike_times,           # When spikes occurred
    positions=trajectory,              # Animal trajectory (n_samples, 2)
    position_times=timestamps,         # Trajectory timestamps
    color="red",
    size=8.0,
    decay_frames=5,                    # Fade over 5 frames (0 = instant)
)
env.animate_fields(fields, overlays=[spike_overlay], frame_times=frame_times)

# Multiple neurons with auto-assigned colors
spike_overlay = EventOverlay(
    event_times={
        "cell_001": spikes_cell1,
        "cell_002": spikes_cell2,
        "cell_003": spikes_cell3,
    },
    positions=trajectory,
    position_times=timestamps,
    colors=None,  # Auto-assign from tab10 colormap
    size=8.0,
    decay_frames=5,
)

# Fixed-location events (explicit positions mode) - rewards, stimuli
reward_overlay = EventOverlay(
    event_times=reward_times,
    event_positions=np.array([[50.0, 25.0]]),  # Single position broadcasts to all
    color="gold",
    size=15.0,
    decay_frames=10,
)

# Multiple event types with different positions
event_overlay = EventOverlay(
    event_times={"reward": reward_times, "punishment": punishment_times},
    event_positions={
        "reward": np.array([[50.0, 25.0]]),      # Reward location
        "punishment": np.array([[75.0, 50.0]]),  # Punishment location
    },
    colors={"reward": "gold", "punishment": "red"},
    markers={"reward": "o", "punishment": "s"},  # Circle vs square (video only)
)

# Mixed-rate temporal alignment (120 Hz position → 10 Hz fields)
overlay = PositionOverlay(
    data=trajectory_120hz,
    times=timestamps_120hz,  # Overlay timestamps
    color="red",
    trail_length=15
)
env.animate_fields(
    fields_10hz,
    overlays=[overlay],
    frame_times=timestamps_10hz,  # Field timestamps - automatic interpolation
    backend="napari"
)

# Show regions with overlays
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[position_overlay],
    show_regions=True,  # Show all regions, or ["region1", "region2"] for specific
    region_alpha=0.3,   # 30% transparent
    backend="napari"
)

# Time series overlays (v0.14.0+) - display continuous variables alongside spatial field
from neurospatial import TimeSeriesOverlay

# Single variable time series (e.g., speed, head direction, LFP)
speed_overlay = TimeSeriesOverlay(
    data=speed_cm_s,              # Shape: (n_samples,) continuous variable
    times=timestamps,              # Shape: (n_samples,) timestamps
    label="Speed (cm/s)",          # Y-axis label
    color="cyan",
    window_seconds=2.0,            # Scrolling window width
    linewidth=1.5,
    show_cursor=True,              # Vertical line at current time
    cursor_color="red",
)
env.animate_fields(fields, overlays=[speed_overlay], frame_times=frame_times)

# Multiple time series stacked (separate rows)
speed_overlay = TimeSeriesOverlay(data=speed, times=times, label="Speed", color="cyan")
accel_overlay = TimeSeriesOverlay(data=accel, times=times, label="Accel", color="orange")
env.animate_fields(
    fields,
    overlays=[speed_overlay, accel_overlay],  # Stacked vertically
    frame_times=frame_times,
    backend="napari"
)

# Multiple time series overlaid (same axes) - use group parameter
left_lfp = TimeSeriesOverlay(
    data=lfp_left, times=times, label="Left LFP", color="blue", group="lfp"
)
right_lfp = TimeSeriesOverlay(
    data=lfp_right, times=times, label="Right LFP", color="red", group="lfp"
)
env.animate_fields(
    fields,
    overlays=[left_lfp, right_lfp],  # Same group → overlaid on same axes
    frame_times=frame_times,
    backend="video",
    save_path="animation.mp4"
)

# Normalized time series (scales to [0, 1] for comparison)
speed_overlay = TimeSeriesOverlay(
    data=speed, times=times, label="Speed", color="cyan", normalize=True, group="kinematics"
)
accel_overlay = TimeSeriesOverlay(
    data=accel, times=times, label="Accel", color="orange", normalize=True, group="kinematics"
)

# Fixed y-axis limits (manual control)
temp_overlay = TimeSeriesOverlay(
    data=temperature, times=times, label="Temp (°C)",
    vmin=20.0, vmax=40.0,  # Fixed limits
    color="red"
)

# Backend capabilities:
# - Napari/Video/Widget: All overlays including VideoOverlay, EventOverlay, TimeSeriesOverlay ✓
# - HTML: Position + EventOverlay (instant only) + regions (Video/TimeSeries skipped with warning) ⚠️

# Video overlay - display recorded video behind/above spatial fields (v0.5.0+)
from neurospatial.animation import VideoOverlay, calibrate_video
from neurospatial.transforms import calibrate_from_landmarks, VideoCalibration

# Create calibration from known landmarks (video pixels → environment cm)
landmarks_px = np.array([[100, 200], [500, 200], [500, 400]])  # Video pixels
landmarks_cm = np.array([[0, 0], [100, 0], [100, 50]])          # Environment cm
transform = calibrate_from_landmarks(landmarks_px, landmarks_cm, frame_size_px=(640, 480))
calib = VideoCalibration(transform, frame_size_px=(640, 480))

# Video overlay with calibration
video_overlay = VideoOverlay(
    source="experiment.mp4",     # Or pre-loaded array (n_frames, H, W, 3)
    calibration=calib,           # Pixel → cm transform
    alpha=0.5,                   # 50% blend (default) - equal video/field visibility
    z_order="above",             # Render on top of field (default)
)
env.animate_fields(fields, frame_times=frame_times, overlays=[video_overlay], backend="napari")

# Adjust alpha to control video/field balance:
# - alpha=0.3: 30% video, 70% field (field dominant)
# - alpha=0.5: 50% video, 50% field (balanced, default)
# - alpha=0.7: 70% video, 30% field (video dominant)

# Convenience function: calibrate_video() (recommended approach)
from neurospatial.animation import calibrate_video

# Method 1: Scale bar calibration
calibration = calibrate_video(
    "session.mp4",
    env,
    scale_bar=((100, 200), (300, 200), 50.0),  # Two points + known length in cm
)

# Method 2: Landmark correspondences (arena corners)
corners_px = np.array([[50, 50], [590, 50], [590, 430], [50, 430]])  # Video pixels
corners_env = np.array([[0, 0], [100, 0], [100, 80], [0, 80]])       # Environment cm
calibration = calibrate_video(
    "session.mp4",
    env,
    landmarks_px=corners_px,
    landmarks_env=corners_env,
)

# Method 3: Direct scale factor (most common)
calibration = calibrate_video("session.mp4", env, cm_per_px=0.25)

# The flip_y parameter controls Y-axis convention (default: True for scientific data)
# - flip_y=True (default): Environment uses Y-up (scientific/Cartesian convention)
# - flip_y=False: Environment uses Y-down (image/pixel convention)

# If overlay appears inverted, toggle flip_y
calibration = calibrate_video("session.mp4", env, cm_per_px=0.25, flip_y=False)

# Use calibration with VideoOverlay
video = VideoOverlay(source="session.mp4", calibration=calibration)
env.animate_fields(fields, frame_times=frame_times, overlays=[video], backend="napari")
```

**Video Overlay Best Practices:**

| Goal | Settings | Result |
|------|----------|--------|
| Balanced view | `alpha=0.5, z_order="above"` | Equal video/field visibility (default) |
| Field dominant | `alpha=0.3, z_order="above"` | Field shows through video |
| Video dominant | `alpha=0.7, z_order="above"` | Video shows through field |
| Video as background | `z_order="below"` | Only works if field has transparent regions |

**Video Calibration Coordinate Conventions:**

| Environment Units | `flip_y` | When to Use |
|-------------------|----------|-------------|
| cm (most common)  | `True` (default) | Scientific tracking data (DeepLabCut, SLEAP, etc.) |
| meters            | `True` (default) | Same as cm, just convert: `cm_per_px = m_per_px * 100` |
| pixels            | `False` | Environment already in image coordinates |

The video itself serves as visual validation - if the overlay appears inverted, toggle `flip_y`.

**Coordinate Conventions for Overlays:**

Environment and napari use different coordinate systems:

| System | X-axis | Y-axis | Origin |
|--------|--------|--------|--------|
| Environment | Horizontal (columns) | Vertical (rows), up is positive | Bottom-left |
| Napari | Column index | Row index, down is positive | Top-left |

When providing overlay data (PositionOverlay, BodypartOverlay, HeadDirectionOverlay, EventOverlay):
- Use **environment coordinates** (same as your position data)
- The animation system automatically transforms to napari pixel space
- Transformations include: (x,y) to (row,col) swap and Y-axis inversion
- For EventOverlay: positions are interpolated from trajectory or provided explicitly

```python
# Your data is in environment coordinates (x, y)
positions = np.array([[10.0, 20.0], [15.0, 25.0]])  # (x, y) format

# Pass directly - transformation happens internally
overlay = PositionOverlay(data=positions)
env.animate_fields(fields, frame_times=frame_times, overlays=[overlay])
# Napari displays correctly with Y-axis increasing upward
```

**Video Annotation (v0.6.0+)**:

```python
# Annotate video frames interactively
from neurospatial import annotate_video, regions_from_labelme, regions_from_cvat

# Interactive napari annotation - draw environment boundary and regions
result = annotate_video("experiment.mp4", bin_size=2.0)
env = result.environment  # Environment from boundary polygon
regions = result.regions   # Named regions

# With calibration (pixel -> cm coordinates)
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
transform = calibrate_from_scale_bar((0, 0), (200, 0), 100.0, (640, 480))
calib = VideoCalibration(transform, (640, 480))
result = annotate_video("experiment.mp4", calibration=calib, bin_size=2.0)

# Import from external annotation tools
regions = regions_from_labelme("labelme_export.json", calibration=calib)
regions = regions_from_cvat("cvat_export.xml", calibration=calib)

# Annotation modes
result = annotate_video("experiment.mp4", mode="both", bin_size=2.0)      # Default: boundary + regions
result = annotate_video("experiment.mp4", mode="environment", bin_size=2.0)  # Only boundary
result = annotate_video("experiment.mp4", mode="regions")                    # Only regions (no bin_size needed)

# Simplify hand-drawn polygons (removes jagged edges)
result = annotate_video("experiment.mp4", bin_size=2.0, simplify_tolerance=1.0)

# Boundary seeding from position data (v0.9.0+)
# Pre-draw boundary from trajectory - edit rather than draw from scratch
from neurospatial.annotation import BoundaryConfig, boundary_from_positions

# Simple: pass positions directly (uses sensible defaults)
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,  # Auto-infer with alpha_shape, 2% buffer, 1% simplify
)

# With config for fine-tuning
config = BoundaryConfig(method="convex_hull", buffer_fraction=0.05)
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,
    boundary_config=config,
)

# Show trajectory as reference while editing
result = annotate_video(
    "experiment.mp4",
    bin_size=2.0,
    initial_boundary=positions,
    show_positions=True,  # Semi-transparent cyan Points layer
)

# Composable: create boundary explicitly for more control
boundary = boundary_from_positions(
    positions,
    method="alpha_shape",  # or "convex_hull"
    alpha=0.05,
    buffer_fraction=0.03,
)
result = annotate_video("experiment.mp4", bin_size=2.0, initial_boundary=boundary)

# Available boundary inference methods:
# - "alpha_shape" (default): Concave hull, tighter fit. Install: pip install alphashape
#   Falls back to convex_hull if alphashape not installed.
# - "convex_hull": Fast, always convex, no extra dependencies
```

**Annotation Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| `E` | Set mode to draw environment boundary (cyan) |
| `R` | Set mode to draw named region (yellow) |
| `3` | Move shape mode |
| `4` | Edit vertices mode |
| `Delete` | Remove selected shape |

**Track Graph Annotation (v0.9.0+)**:

```python
# Annotate track graphs interactively for 1D linearized environments
from neurospatial.annotation import annotate_track_graph, TrackGraphResult

# From video file - opens napari for interactive annotation
result = annotate_track_graph("maze.mp4")

# From static image
import matplotlib.pyplot as plt
img = plt.imread("track_photo.png")
result = annotate_track_graph(image=img)

# With calibration (convert pixels to cm)
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar
transform = calibrate_from_scale_bar((0, 0), (200, 0), 100.0, (640, 480))
calib = VideoCalibration(transform, (640, 480))
result = annotate_track_graph("maze.mp4", calibration=calib)

# Result contains everything needed for Environment.from_graph()
print(result.track_graph)      # NetworkX Graph with 'pos', 'distance', 'edge_id'
print(result.node_positions)   # List of (x, y) tuples (cm if calibrated)
print(result.edges)            # Edge list as (node_i, node_j) tuples
print(result.edge_order)       # Ordered edges for linearization
print(result.edge_spacing)     # Spacing between edges
print(result.pixel_positions)  # Original pixel coordinates (preserved)

# Create 1D linearized environment from result
env = result.to_environment(bin_size=2.0)
print(env.is_1d)  # True - ready for linearization

# Override edge spacing if needed
env = result.to_environment(bin_size=2.0, edge_spacing=[0.0, 5.0, 0.0])

# With initial data (for editing existing track graphs)
result = annotate_track_graph(
    "maze.mp4",
    initial_nodes=np.array([[100, 200], [300, 200], [300, 400]]),
    initial_edges=[(0, 1), (1, 2)],
    initial_node_labels=["start", "junction", "goal"],
)

# Workflow: annotate → create environment → analyze
result = annotate_track_graph("linear_track.mp4", calibration=calib)
env = result.to_environment(bin_size=2.0, name="linear_track")
env.units = "cm"

# Add regions from node labels
for i, label in enumerate(result.node_labels):
    if label:  # Non-empty label
        env.regions.add(label, point=result.node_positions[i])
```

**Track Graph Builder Keyboard Shortcuts:**

| Key | Action |
|-----|--------|
| `E` | Switch to Add Edge mode (click two nodes to connect them) |
| `Shift+S` | Set selected node as start node (select node first with napari's `3` key) |
| `Ctrl+Z` | Undo last action |
| `Ctrl+Shift+Z` | Redo |
| `Escape` | Cancel edge creation in progress |

**Note**: Use napari's native layer shortcuts for node manipulation:

- Press `2` for Add mode (click to add nodes)
- Press `3` for Select mode (click to select nodes for deletion or start node)
- Use the widget buttons for Delete Node/Delete Edge operations

**Track Graph Builder UI Components:**

- **Mode Selector**: Radio buttons for Add Node / Add Edge / Delete modes
- **Node List**: Shows all nodes with labels and start marker (★)
- **Edge List**: Shows all edges as (node_i, node_j)
- **Edge Order**: Reorderable list with Move Up/Down/Reset buttons
- **Edge Spacing**: Input field to set gap between edges
- **Preview Button**: Shows 1D linearization preview (matplotlib)
- **Validation Status**: Green (valid), Orange (warnings), Red (errors)
- **Save Button**: Validates and closes the viewer

**NWB Integration (v0.7.0+)**:

```python
# Install NWB support (optional dependencies)
# pip install neurospatial[nwb]       # Basic NWB support
# pip install neurospatial[nwb-full]  # Full NWB support with extensions

from pynwb import NWBHDF5IO
from neurospatial.nwb import (
    # Reading functions
    read_position,              # Position → (positions, timestamps)
    read_head_direction,        # CompassDirection → (angles, timestamps)
    read_pose,                  # PoseEstimation → (bodyparts, timestamps, skeleton)
    read_events,                # EventsTable → DataFrame
    read_intervals,             # TimeIntervals → DataFrame (trials, epochs)
    read_trials,                # Trials table → DataFrame
    read_environment,           # scratch/ → Environment

    # Writing functions
    write_place_field,          # Write place field to analysis/
    write_occupancy,            # Write occupancy map to analysis/
    write_laps,                 # Write lap events to processing/behavior/
    write_trials,               # Write trials to intervals/trials/
    write_region_crossings,     # Write region crossing events
    write_environment,          # Write Environment to scratch/

    # Factory functions
    environment_from_position,  # Create Environment from NWB Position
    position_overlay_from_nwb,  # PositionOverlay from NWB Position
    bodypart_overlay_from_nwb,  # BodypartOverlay from ndx-pose
    head_direction_overlay_from_nwb,  # HeadDirectionOverlay from NWB
)

# Read position data from NWB file
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()

    # Read position data
    positions, timestamps = read_position(nwbfile)

    # Create environment from position data
    env = environment_from_position(nwbfile, bin_size=2.0, units="cm")

    # Create overlays for animation
    position_overlay = position_overlay_from_nwb(nwbfile, color="red", trail_length=10)
    bodypart_overlay = bodypart_overlay_from_nwb(nwbfile, pose_estimation_name="DLC")
    head_direction = head_direction_overlay_from_nwb(nwbfile, color="yellow")

# Write analysis results to NWB
with NWBHDF5IO("session.nwb", "r+") as io:
    nwbfile = io.read()

    # Write place field to analysis/
    write_place_field(nwbfile, env, place_field, name="cell_001")

    # Write occupancy to analysis/
    write_occupancy(nwbfile, env, occupancy, unit="seconds")

    # Write laps to processing/behavior/
    write_laps(nwbfile, lap_times, lap_types=lap_directions)

    # Write region crossings to processing/behavior/
    write_region_crossings(
        nwbfile, crossing_times,
        region_names=["goal", "goal", "start"],
        event_types=["enter", "exit", "enter"]
    )

    # Write environment for round-trip (to scratch/)
    write_environment(nwbfile, env, name="linear_track")

    # Write trials from Trial objects (from segment_trials)
    write_trials(nwbfile, trials)

    # Or write trials from raw arrays
    write_trials(
        nwbfile,
        start_times=start_times,
        stop_times=stop_times,
        start_regions=start_regions,
        end_regions=end_regions,
        successes=successes,
        overwrite=True,  # Replace existing trials
    )

    io.write(nwbfile)

# Read trials back from NWB
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()
    trials_df = read_trials(nwbfile)
    # DataFrame with: start_time, stop_time, start_region, end_region, success

# Round-trip: Read environment from NWB
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()
    loaded_env = read_environment(nwbfile, name="linear_track")
    # loaded_env is fully reconstructed with bin_centers, connectivity, regions

# Alternative: Use Environment methods for NWB integration
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()

    # Create from position data
    env = Environment.from_nwb(nwbfile, bin_size=2.0, units="cm")

    # Or load from scratch space
    env = Environment.from_nwb(nwbfile, scratch_name="linear_track")

# Write environment using method
with NWBHDF5IO("session.nwb", "r+") as io:
    nwbfile = io.read()
    env.to_nwb(nwbfile, name="my_environment")
    io.write(nwbfile)
```

**NWB Data Locations:**

| Data Type | NWB Location | Function |
|-----------|--------------|----------|
| Place fields | `analysis/` | `write_place_field()` |
| Occupancy | `analysis/` | `write_occupancy()` |
| Trials | `intervals/trials/` | `write_trials()` |
| Lap events | `processing/behavior/` | `write_laps()` |
| Region crossings | `processing/behavior/` | `write_region_crossings()` |
| Environment | `scratch/` | `write_environment()` |

**NWB Dependencies:**

| Extra | Packages | Use Case |
|-------|----------|----------|
| `nwb` | pynwb, hdmf | Basic NWB support |
| `nwb-pose` | pynwb, ndx-pose | Pose estimation data |
| `nwb-events` | pynwb, ndx-events | EventsTable support |
| `nwb-full` | All above | Full NWB support |

**Type Checking Support (v0.2.1+)**:

This package now includes a `py.typed` marker, enabling type checkers like mypy to use the library's type annotations:

```python
# Your IDE and mypy will now see neurospatial types!
from neurospatial import Environment
import numpy as np

positions: np.ndarray = np.random.rand(100, 2)
env: Environment = Environment.from_samples(positions, bin_size=5.0)
# mypy will validate types ✓
```

**Commit Message Format**:

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): description` - New features
- `fix(scope): description` - Bug fixes
- `docs(scope): description` - Documentation changes
- `test(scope): description` - Test additions/fixes
- `chore(scope): description` - Maintenance tasks

Examples: `feat(M3): add .info() method`, `fix: correct version reference`

## Project Overview

**neurospatial** is a Python library for discretizing continuous N-dimensional spatial environments into bins/nodes with connectivity graphs. It provides tools for spatial analysis, particularly for neuroscience applications involving place fields, position tracking, and spatial navigation.

## Package Management with uv

**CRITICAL: This project uses `uv` for package management and virtual environment handling.**

- Python version: 3.13 (specified in `.python-version`)
- **ALWAYS** use `uv run` to execute Python commands in the correct virtual environment
- **NEVER** use bare `python`, `pip`, or `pytest` commands - always prefix with `uv run`

### Why uv?

`uv` automatically manages the virtual environment and ensures all commands run in the correct Python environment without manual activation. It reads `.python-version` and handles environment setup transparently.

## Core Architecture

### Three-Layer Design

The codebase follows a three-layer architecture:

1. **Layout Engines** (`src/neurospatial/layout/`)
   - Protocol-based design with `LayoutEngine` interface ([layout/base.py](src/neurospatial/layout/base.py))
   - Multiple concrete implementations in `layout/engines/`:
     - `RegularGridLayout` - Standard rectangular grids
     - `HexagonalLayout` - Hexagonal tessellations
     - `GraphLayout` - 1D linearized track representations (requires `track-linearization` package)
     - `MaskedGridLayout` - Grids with arbitrary active/inactive regions
     - `ImageMaskLayout` - Binary image-based layouts
     - `ShapelyPolygonLayout` - Polygon-bounded grids
     - `TriangularMeshLayout` - Triangular tessellations
   - Factory pattern via `create_layout()` in [layout/factories.py](src/neurospatial/layout/factories.py:126-177)
   - All engines produce: `bin_centers`, `connectivity` graph, `dimension_ranges`, and optional grid metadata

2. **Environment** (`src/neurospatial/environment/`)
   - Main user-facing class wrapping a `LayoutEngine` instance
   - **Modular package structure** (as of v0.2.1) using mixin pattern:
     - [core.py](src/neurospatial/environment/core.py) - Core `Environment` dataclass with state and properties (1,023 lines)
     - [factories.py](src/neurospatial/environment/factories.py) - Factory classmethods for creating instances (630 lines)
     - [queries.py](src/neurospatial/environment/queries.py) - Spatial query methods (897 lines)
     - [trajectory.py](src/neurospatial/environment/trajectory.py) - Trajectory analysis (occupancy, bin_sequence, transitions) (1,222 lines)
     - [transforms.py](src/neurospatial/environment/transforms.py) - Rebin/subset operations (634 lines)
     - [fields.py](src/neurospatial/environment/fields.py) - Spatial field operations (compute_kernel, smooth, interpolate) (564 lines)
     - [metrics.py](src/neurospatial/environment/metrics.py) - Environment metrics and properties (boundary_bins, bin_attributes, linearization) (469 lines)
     - [regions.py](src/neurospatial/environment/regions.py) - Region operations (398 lines)
     - [serialization.py](src/neurospatial/environment/serialization.py) - Save/load methods (315 lines)
     - [visualization.py](src/neurospatial/environment/visualization.py) - Plotting methods (211 lines)
     - [decorators.py](src/neurospatial/environment/decorators.py) - `@check_fitted` decorator (77 lines)
   - Factory methods for common use cases:
     - `Environment.from_samples()` - Discretize point data into bins
     - `Environment.from_graph()` - Create 1D track-based environments
     - `Environment.from_polygon()` - Grid masked by Shapely polygon
     - `Environment.from_mask()` - Pre-defined N-D boolean mask
     - `Environment.from_image()` - Binary image mask
     - `Environment.from_layout()` - Direct layout specification
   - Provides spatial queries: `bin_at()`, `contains()`, `neighbors()`, `distance_between()`, `path_between()`
   - Integrates `Regions` for defining named ROIs (regions of interest)
   - Uses `@check_fitted` decorator ([environment/decorators.py](src/neurospatial/environment/decorators.py)) to ensure methods are only called after initialization

3. **Regions** (`src/neurospatial/regions/`)
   - Immutable `Region` dataclass for points or polygons ([regions/core.py:36-125](src/neurospatial/regions/core.py#L36-L125))
   - `Regions` mapping container (dict-like interface)
   - JSON serialization support with versioned schema
   - Operations: `add()`, `buffer()`, `area()`, `region_center()`

### Supporting Modules

- **CompositeEnvironment** ([composite.py](src/neurospatial/composite.py)) - Merges multiple `Environment` instances with automatic mutual-nearest-neighbor bridge inference
- **Alignment** ([alignment.py](src/neurospatial/alignment.py)) - Transforms and maps probability distributions between environments (rotation, scaling, translation)
- **Transforms** ([transforms.py](src/neurospatial/transforms.py)) - 2D affine transformations with composable API

## Import Patterns

Standard import patterns for this package:

```python
# Main classes
from neurospatial import Environment
from neurospatial.regions import Region, Regions

# v0.1.0+ Public API functions
from neurospatial import (
    validate_environment,          # Validate environment structure
    map_points_to_bins,            # Batch point-to-bin mapping with KDTree
    estimate_transform,             # Estimate transform from point pairs
    apply_transform_to_environment, # Transform entire environment
    distance_field,                # Multi-source geodesic distances
    pairwise_distances,            # Distances between node subsets
)

# Serialization (v0.1.0+)
from neurospatial.io import to_file, from_file, to_dict, from_dict

# Animation (v0.3.0+)
from neurospatial.animation import subsample_frames

# Cache management (v0.3.0+)
# Use env.clear_cache() for all cache clearing operations
# Example: env.clear_cache(kdtree=True, kernels=False, cached_properties=False)
# IMPORTANT: Call env.clear_cache() before parallel rendering (makes environment pickle-able)

# Layout engines and factories
from neurospatial.layout.factories import create_layout, list_available_layouts
from neurospatial.layout.engines.regular_grid import RegularGridLayout

# Utility functions
from neurospatial.alignment import get_2d_rotation_matrix, map_probabilities
from neurospatial.transforms import Affine2D, translate, scale_2d

# NWB integration (v0.7.0+ - requires optional dependencies)
# Install with: pip install neurospatial[nwb-full]
from neurospatial.nwb import (
    # Reading
    read_position,           # Position → (positions, timestamps)
    read_head_direction,     # CompassDirection → (angles, timestamps)
    read_pose,               # PoseEstimation → (bodyparts, timestamps, skeleton)
    read_events,             # EventsTable → DataFrame
    read_intervals,          # TimeIntervals → DataFrame
    read_trials,             # Trials table → DataFrame
    read_environment,        # scratch/ → Environment

    # Writing
    write_place_field,       # Write to analysis/
    write_occupancy,         # Write to analysis/
    write_trials,            # Write to intervals/trials/
    write_laps,              # Write to processing/behavior/
    write_region_crossings,  # Write to processing/behavior/
    write_environment,       # Write to scratch/

    # Factories
    environment_from_position,
    position_overlay_from_nwb,
    bodypart_overlay_from_nwb,
    head_direction_overlay_from_nwb,
)

# Bayesian decoding (v0.12.0+)
from neurospatial import (
    DecodingResult,          # Result container class
    decode_position,         # Main entry point
    decoding_error,          # Per-time-bin position error
    median_decoding_error,   # Summary statistic
)

# Full decoding API (from subpackage)
from neurospatial.decoding import (
    # Likelihood computation
    log_poisson_likelihood,
    poisson_likelihood,

    # Posterior estimation
    normalize_to_posterior,

    # Point estimates
    map_estimate,
    map_position,
    mean_position,
    entropy,
    credible_region,

    # Trajectory analysis
    fit_isotonic_trajectory,
    fit_linear_trajectory,
    detect_trajectory_radon,  # Requires scikit-image
    IsotonicFitResult,
    LinearFitResult,
    RadonDetectionResult,

    # Quality metrics
    confusion_matrix,
    decoding_correlation,

    # Shuffle-based significance testing
    shuffle_time_bins,
    shuffle_time_bins_coherent,
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
    generate_poisson_surrogates,
    generate_inhomogeneous_poisson_surrogates,
    compute_shuffle_pvalue,
    compute_shuffle_zscore,
    ShuffleTestResult,
)
```

## Important Patterns & Constraints

### Graph Metadata Requirements

The connectivity graph (`nx.Graph`) has **mandatory node and edge attributes**:

**Node attributes** (enforced by layout engines):

- `'pos'`: Tuple[float, ...] - N-D coordinates
- `'source_grid_flat_index'`: int - Flat index in original grid
- `'original_grid_nd_index'`: Tuple[int, ...] - N-D grid index

**Edge attributes** (enforced by layout engines):

- `'distance'`: float - Euclidean distance between bin centers
- `'vector'`: Tuple[float, ...] - Displacement vector
- `'edge_id'`: int - Unique edge ID
- `'angle_2d'`: Optional[float] - Angle for 2D layouts

### Mixin Pattern for Environment

The `Environment` class uses **mixin inheritance** to organize its 6,000+ lines of functionality into focused modules:

```python
# In src/neurospatial/environment/core.py
@dataclass  # Only Environment is a dataclass
class Environment(
    EnvironmentFactories,      # Factory classmethods
    EnvironmentQueries,         # Spatial query methods
    EnvironmentSerialization,   # Save/load methods
    EnvironmentRegions,         # Region operations
    EnvironmentVisualization,   # Plotting methods
    EnvironmentMetrics,         # Metrics and spatial properties
    EnvironmentFields,          # Spatial field operations
    EnvironmentTrajectory,      # Trajectory analysis
    EnvironmentTransforms,      # Rebin/subset operations
):
    """Main Environment class assembled from mixins."""
    name: str = ""
    layout: LayoutEngine | None = None
    # ... rest of dataclass fields
```

**Key constraints:**

- **ONLY `Environment` is a `@dataclass`** - All mixins MUST be plain classes
- Mixins use `TYPE_CHECKING` guards to avoid circular imports:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from neurospatial.environment.core import Environment

  class EnvironmentQueries:
      def bin_at(self: "Environment", points) -> int:
          # Use string annotation for type hint
          ...
  ```
- All mixin methods have access to `self` attributes (from Environment dataclass)
- Public API unchanged: `from neurospatial import Environment` still works

### Mypy Type Checking Requirements

**IMPORTANT: Mypy runs in pre-commit hooks and should pass without errors.**

This project uses mypy for type checking with a **pragmatic configuration** suited for scientific Python code. The configuration (in `pyproject.toml`) balances type safety with practicality:

- **Type annotations encouraged**: Public APIs and mixin methods should have type hints
- **Lenient for scientific code**: Allows untyped defs, untyped calls, and incomplete defs (common in scientific libraries)
- **Pre-commit enforcement**: Mypy runs automatically on commit with basic checks
- **Key settings** (see `[tool.mypy]` in `pyproject.toml`):
  - `disallow_untyped_defs = false` - Allows functions without type annotations
  - `check_untyped_defs = false` - Doesn't check inside untyped functions
  - `allow_untyped_calls = true` - Allows calling untyped library functions
  - `warn_unused_ignores = true` - Warns about unnecessary `type: ignore` comments

**Guidelines:**

1. **Prefer proper typing over suppressions** - Add type hints when possible rather than using `type: ignore`
2. **Mixin methods should be typed** - Use proper type annotations for all public mixin methods (see pattern below)
3. **Avoid skipping mypy** - Let pre-commit run mypy normally; only skip if absolutely necessary

**Mixin Type Annotation Pattern:**

The mixin pattern requires special care for mypy. Since mixins access Environment attributes that don't exist on the mixin class itself, use these patterns:

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol
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

**Common mypy issues and fixes:**

- **Missing attribute errors**: Ensure `self: "Environment"` annotation is present
- **Type mismatches**: Ensure return types match exactly between mixin and Environment
- **Cache type annotations**: Use precise Literal types for cache keys
- **Import TYPE_CHECKING**: Always use `if TYPE_CHECKING:` guard for Environment import

**Pre-commit mypy configuration** (`.pre-commit-config.yaml`):

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.13.0
  hooks:
    - id: mypy
      args: [--ignore-missing-imports, --warn-unused-ignores]
```

**Using Protocol for Mixin Type Safety:**

This project uses the `Protocol` pattern (PEP 544) to enable proper type checking for mixins. See [mypy docs on mixins](https://mypy.readthedocs.io/en/latest/more_types.html#mixin-classes).

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

**Running mypy manually:**

```bash
# Check all files (same args as pre-commit)
uv run mypy --ignore-missing-imports --warn-unused-ignores src/neurospatial/

# Check specific file
uv run mypy --ignore-missing-imports --warn-unused-ignores src/neurospatial/environment/fields.py

# Check with pyproject.toml config (recommended)
uv run mypy src/neurospatial/
```

### Protocol-Based Design

Layout engines implement the `LayoutEngine` Protocol ([layout/base.py:10-166](src/neurospatial/layout/base.py#L10-L166)), not inheritance. When creating new engines:

- Implement required attributes: `bin_centers`, `connectivity`, `dimension_ranges`, `is_1d`, `_layout_type_tag`, `_build_params_used`
- Implement required methods: `build()`, `point_to_bin_index()`, `bin_sizes()`, `plot()`
- Optionally provide grid-specific attributes: `grid_edges`, `grid_shape`, `active_mask`

### Fitted State Pattern

`Environment` uses a `_is_fitted` flag set after `_setup_from_layout()` completes. Methods requiring fitted state use the `@check_fitted` decorator. This prevents calling spatial query methods before the environment is properly initialized.

### Regions are Immutable

`Region` objects are immutable dataclasses - create new instances rather than modifying existing ones. The `Regions` container uses dict-like semantics:

- Use `regions.add()` to create and insert (raises `KeyError` if name already exists)
- Use `del regions[name]` or `regions.remove(name)` to delete
- Assignment to existing keys succeeds but emits a `UserWarning` to prevent accidental overwrites
- Use `regions.update_region()` to update regions without warnings

### 1D vs N-D Environments

Environments can be 1D (linearized tracks) or N-D (grids):

- 1D: `GraphLayout` with `is_1d=True`, provides `to_linear()` and `linear_to_nd()` methods
- N-D: Grid-based layouts with spatial queries in original coordinate space

Check `env.is_1d` before calling linearization methods.

### Methods vs Free Functions

**When to use Environment methods vs module-level functions:**

- **Methods on Environment** answer questions about that environment or perform local transforms.
  - Examples: `env.bin_at()`, `env.neighbors()`, `env.distance_between()`, `env.rebin()`
  - Use when: Working with a single environment's structure and properties

- **Free functions** take environments/graphs/fields as input and perform higher-level analysis (neural metrics, segmentation, alignment).
  - Examples: `distance_field()`, `map_points_to_bins()`, `estimate_transform()`, `compute_place_field()`
  - Use when: Cross-environment operations, neural/behavioral analysis, or batch processing

**If you're unsure:** Start from the object you have (Environment, field array, graph) and use autocomplete. If it's about cross-environment, neural, or behavioral analysis, look under the free functions in `neurospatial.__init__`.

**Design principle:** This separation keeps the `Environment` class focused on spatial structure while providing specialized functions for domain-specific analyses (neuroscience, navigation, etc.).

## Development Commands

**IMPORTANT: All commands below MUST be prefixed with `uv run` to ensure they execute in the correct virtual environment. Run all commands from the project root directory.**

### Environment Setup

```bash
# Initialize/sync the virtual environment (uv handles this automatically)
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Install the package in editable mode (usually not needed with uv)
uv pip install -e .
```

### Testing

```bash
# Run all tests (most common - use this for verification)
uv run pytest

# Run specific test module
uv run pytest tests/test_environment.py

# Run tests for a specific layout engine
uv run pytest tests/layout/test_regular_grid_utils.py

# Run with verbose output (helpful for debugging)
uv run pytest -v

# Run with coverage report
uv run pytest --cov=src/neurospatial

# Run a specific test function
uv run pytest tests/test_environment.py::test_function_name -v

# Run doctests (validate docstring examples)
uv run pytest --doctest-modules src/neurospatial/

# Run tests matching a pattern
uv run pytest -k "test_bin_size"

# Run NWB integration tests (requires nwb-full extra)
uv run pytest tests/nwb/ -v

# Run NWB tests for specific module
uv run pytest tests/nwb/test_environment.py -v
```

### Running the Package

```bash
# Run any Python script (from project root)
uv run python path/to/script.py

# Import in interactive session
uv run python -c "from neurospatial import Environment; print(Environment)"
```

### Python REPL

```bash
# Start interactive Python session with package available
uv run python

# Or use IPython if installed (recommended for exploration)
uv run ipython
```

### Code Quality

```bash
# Run ruff linter (check for issues)
uv run ruff check .

# Run ruff formatter (auto-format code)
uv run ruff format .

# Run both check and format (common workflow)
uv run ruff check . && uv run ruff format .

# Check specific file
uv run ruff check src/neurospatial/environment.py
```

### Napari Performance Monitoring

Use napari's built-in performance monitoring (perfmon) to profile animation backends:

```bash
# Enable perfmon with environment variable
NAPARI_PERFMON=1 uv run python your_script.py

# Or output trace to file for detailed analysis
NAPARI_PERFMON=/tmp/perfmon.json uv run python your_script.py
```

**Viewing Trace Data:**
- Chrome: Open `chrome://tracing` and drag-drop the JSON file
- Speedscope: Upload to https://www.speedscope.app/ for flame graphs

**Programmatic Timing (in code):**

```python
from napari.utils.perf import perf_timer, add_instant_event

# Time a code block
with perf_timer("my_operation"):
    expensive_function()

# Mark specific moments in traces
add_instant_event("checkpoint_reached")
```

**Configuration File (optional):**

Create a JSON config file for fine-grained control:

```json
{
    "trace_qt_events": true,
    "trace_file_on_start": "/tmp/latest.json",
    "trace_callables": ["animation"],
    "callable_lists": {
        "animation": [
            "neurospatial.animation.napari_backend._build_skeleton_vectors",
            "neurospatial.animation.napari_backend._render_bodypart_overlay"
        ]
    }
}
```

Then run with: `NAPARI_PERFMON=/path/to/config.json uv run python script.py`

**Key hotspots to profile in this codebase:**
- `_build_skeleton_vectors` - skeleton overlay construction
- `_render_bodypart_overlay` - bodypart rendering per frame
- `_render_head_direction_overlay` - head direction arrows
- Layer update callbacks during playback

See: https://napari.org/stable/howtos/perfmon.html

## Key Implementation Notes

### Creating New Layout Engines

1. Implement the `LayoutEngine` protocol in `src/neurospatial/layout/engines/`
2. Populate required attributes in your `build()` method
3. Add to `_LAYOUT_MAP` in [layout/factories.py:17-25](src/neurospatial/layout/factories.py#L17-L25)
4. Ensure graph nodes/edges have mandatory metadata
5. Test boundary detection works with your layout in `layout/helpers/utils.py`
6. Add tests in `tests/layout/` following existing patterns

### Dependencies

Core dependencies:

- `numpy` - Array operations and numerical computing
- `pandas` - Data structures and analysis
- `matplotlib` - Plotting and visualization
- `networkx` - Graph data structures for connectivity
- `scipy` - Scientific computing (KDTree, morphological operations)
- `scikit-learn` - Machine learning utilities (KDTree)
- `shapely` - Geometric operations and polygon support
- `track-linearization` - 1D track linearization for GraphLayout

Development dependencies:

- `pytest` - Testing framework
- `pytest-cov` - Test coverage reporting
- `ruff` - Fast Python linter and formatter
- `ipython` - Enhanced interactive Python shell

Optional NWB dependencies (v0.7.0+):

- `pynwb` - NWB file format support
- `hdmf` - Hierarchical Data Modeling Framework (pynwb dependency)
- `ndx-pose` - Pose estimation extension (for PoseEstimation data)
- `ndx-events` - Events extension (for EventsTable data)

Install with: `pip install neurospatial[nwb-full]` or `uv add neurospatial[nwb-full]`

Optional trajectory analysis dependencies (v0.12.0+):

- `scikit-image` - Radon transform for trajectory detection (`detect_trajectory_radon`)

Install with: `pip install neurospatial[trajectory]` or `uv add neurospatial[trajectory]`

### Animation Playback Control (v0.15.0+)

The animation system uses a speed-based API that separates data sample rate from playback speed:

**Core Concepts:**

- **`frame_times`** (required): Timestamps for each frame in seconds, defining the temporal structure of your data. Use timestamps from your data source (e.g., position timestamps, decoding time bins).

- **`speed`** (default 1.0): Playback speed multiplier relative to real-time:
  - `speed=1.0`: Real-time (1 second of data = 1 second viewing)
  - `speed=0.1`: 10% speed (slow motion, good for replay analysis)
  - `speed=2.0`: 2× speed (fast forward)

**How playback fps is computed:**

```python
# System computes sample rate from frame_times
sample_rate_hz = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

# Playback fps is derived from speed
playback_fps = min(sample_rate_hz * speed, MAX_PLAYBACK_FPS)  # Capped at 60 fps
```

**Use Case Examples:**

| Analysis Type | Data Rate | Speed | Playback fps | Notes |
|---------------|-----------|-------|--------------|-------|
| Replay decoding | 500 Hz | 0.1 | 50 fps | See trajectory unfold |
| Theta sequences | 30 Hz | 1.0 | 30 fps | Natural dynamics |
| Place fields | 30 Hz | 2.0 | 60 fps | Quick preview |
| High-speed replay | 500 Hz | 1.0 | 60 fps (capped) | Warning emitted |

**Speed Capping Warning:**

When requested speed exceeds display limits, a `UserWarning` is emitted:

```python
# 500 Hz data at real-time would need 500 fps (impossible)
env.animate_fields(posterior_fields, frame_times=decode_times, speed=1.0)
# UserWarning: Requested speed=1.00x would require 500 fps.
#              Capped to 60 fps (effective speed=0.12x).
```

**Advanced: Override max fps:**

For high-refresh displays (120/144 Hz), use `max_playback_fps`:

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    speed=2.0,
    max_playback_fps=120  # For 120 Hz displays
)
```

### Animation Overlay Architecture (v0.4.0+)

The overlay system provides three public dataclasses for visualizing animal behavior alongside spatial fields:

**Public API** (`src/neurospatial/animation/overlays.py`):
- `PositionOverlay` - Trajectory tracking with decaying trails
- `BodypartOverlay` - Pose tracking with skeleton rendering (dict of bodypart trajectories)
- `HeadDirectionOverlay` - Orientation arrows (angles or unit vectors)

**Conversion funnel**:
1. User creates overlay dataclasses with behavioral data
2. `_convert_overlays_to_data()` validates and aligns overlays to field frames
3. Temporal interpolation handles mixed sampling rates (e.g., 120 Hz → 10 Hz)
4. Outputs pickle-safe `OverlayData` for backend rendering

**Backend support**:
- Napari: Full support (all overlay types + regions)
- Video: Full support (all overlay types + regions)
- HTML: Partial support (position + regions only; warns for bodypart/head direction)
- Widget: Full support (reuses video backend renderer)

**Key patterns**:
- Overlays accept optional `times` parameter for temporal alignment
- `frame_times` parameter on `animate_fields()` enables mixed-rate synchronization
- Multi-animal: Pass multiple overlay instances in a list
- Regions: Use `show_regions=True` or `show_regions=["region1", "region2"]`
- Parallel rendering: Call `env.clear_cache()` before `n_workers > 1`

### Skeleton Class (v0.4.0+)

The `Skeleton` class defines anatomical structure for pose tracking:

```python
from neurospatial.animation.skeleton import Skeleton, SIMPLE_SKELETON

# Create a skeleton
skeleton = Skeleton(
    name="mouse",
    nodes=("nose", "body", "tail"),
    edges=(("nose", "body"), ("body", "tail")),  # Edges normalized automatically
)

# Edge canonicalization: ("body", "nose") becomes ("body", "nose") since "body" < "nose"
# Duplicates removed: ("a", "b") and ("b", "a") become single ("a", "b")

# Graph traversal via adjacency property
skeleton.adjacency["body"]  # Returns ['nose', 'tail'] - neighbors sorted
```

**Key features**:
- **Edge canonicalization**: Edges stored in lexicographic order `(min, max)`
- **Deduplication**: Reversed duplicates automatically removed
- **Adjacency property**: Precomputed O(1) access for graph traversal
- **Immutable**: Frozen dataclass with slots for performance
- **Factory methods**: `from_edge_list()`, `from_ndx_pose()`, `from_movement()`

## Testing Structure

Tests mirror source structure:

- `tests/test_environment.py` - Core `Environment` class tests
- `tests/test_composite.py` - `CompositeEnvironment` tests
- `tests/test_alignment.py` - Alignment/transformation tests
- `tests/layout/` - Layout engine-specific tests
- `tests/regions/` - Region functionality tests
- `tests/nwb/` - NWB integration tests (requires nwb-full extra)
- `tests/conftest.py` - Shared fixtures (plus maze, sample environments)

Fixtures in `conftest.py` provide common test environments (plus maze graphs, sample data).

NWB fixtures in `tests/nwb/conftest.py` use `pytest.importorskip()` for graceful skipping when NWB dependencies are not installed.

## Documentation Style

### Docstring Format

**All docstrings MUST follow NumPy docstring format.** This is the standard for scientific Python projects and ensures consistency with the broader ecosystem.

#### NumPy Docstring Structure

```python
def function_name(param1, param2):
    """
    Short one-line summary ending with a period.

    Optional longer description providing more context about what the
    function does, its behavior, and any important implementation details.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ValueError
        Description of when ValueError is raised.
    TypeError
        Description of when TypeError is raised.

    See Also
    --------
    related_function : Brief description of relation.

    Notes
    -----
    Additional technical information, implementation notes, or mathematical
    details.

    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result)
    expected_output
    """
```

#### Key NumPy Docstring Guidelines

1. **Section Headers**: Use underlines with dashes (`---`) matching section name length
2. **Type Annotations**: Include types after parameter names with colon separator
3. **Section Order**: Parameters → Returns → Raises → See Also → Notes → Examples
4. **Blank Lines**: One blank line between sections
5. **Examples**: Use `>>>` for interactive examples, show expected output
6. **Cross-references**: Use backticks for code elements: `Environment`, `bin_centers`
7. **Math**: Use LaTeX notation in Notes section when needed
8. **Arrays**: Specify shape in type: `NDArray[np.float64], shape (n_samples, n_dims)`

#### Common Patterns in This Codebase

**Class docstrings**:

```python
class Environment:
    """
    Short summary of the class.

    Longer description of purpose, key features, and design patterns.

    Attributes
    ----------
    name : str
        Description of name attribute.
    layout : LayoutEngine
        Description of layout attribute.

    See Also
    --------
    CompositeEnvironment : Related class.

    Examples
    --------
    >>> env = Environment.from_samples(positions, bin_size=2.0)
    >>> print(env.n_bins)
    100
    """
```

**Protocol/Interface docstrings**: Document expected behavior, not implementation
**Property docstrings**: Focus on what the property represents, include type info
**Factory method docstrings**: Emphasize parameters and typical usage patterns

#### Resources

- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [numpydoc package](https://github.com/numpy/numpydoc) for validation

## Common Gotchas

### 1. Always use `uv run`

**Problem**: Running Python commands directly uses the wrong environment.

❌ Wrong:

```bash
python script.py
pytest
pip install package
```

✅ Right:

```bash
uv run python script.py
uv run pytest
uv add package
```

### 2. Check `_is_fitted` state

**Problem**: Calling spatial query methods on unfitted Environment raises error.

❌ Wrong:

```python
env = Environment()  # Not fitted!
env.bin_at([10.0, 5.0])  # RuntimeError
```

✅ Right:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Factory methods fit automatically
env.bin_at([10.0, 5.0])  # Works
```

### 3. Graph metadata is mandatory

**Problem**: Missing node/edge attributes cause failures in spatial queries.

**Required node attributes**: `'pos'`, `'source_grid_flat_index'`, `'original_grid_nd_index'`
**Required edge attributes**: `'distance'`, `'vector'`, `'edge_id'`, `'angle_2d'` (optional)

All layout engines must populate these. If creating custom graphs, ensure all attributes present.

### 4. Regions are immutable

**Problem**: Trying to modify Region objects in place fails.

❌ Wrong:

```python
env.regions['goal'].point = new_point  # AttributeError - immutable
```

⚠️ Discouraged (emits warning):

```python
env.regions['goal'] = new_region  # UserWarning - overwriting existing region
```

✅ Right:

```python
env.regions.update_region('goal', point=new_point)  # Creates new Region, no warning
env.regions.add('new_goal', point=point)  # Add new region
del env.regions['old_goal']  # Delete existing
```

### 5. Check `is_1d` before linearization

**Problem**: Calling `to_linear()` on N-D environments fails.

❌ Wrong:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Creates 2D grid
linear_pos = env.to_linear(position)  # AttributeError
```

✅ Right:

```python
if env.is_1d:
    linear_pos = env.to_linear(position)
else:
    # Use N-D spatial queries instead
    bin_idx = env.bin_at(position)
```

### 6. Protocol, not inheritance

**Problem**: Layout engines don't inherit from a base class.

❌ Wrong:

```python
class MyLayout(LayoutEngine):  # LayoutEngine is a Protocol, not a class
    pass
```

✅ Right:

```python
class MyLayout:
    """Implements LayoutEngine protocol."""
    def build(self, ...): ...
    def point_to_bin_index(self, ...): ...
    # Implement all required methods and attributes
```

### 7. NumPy docstrings required

**Problem**: Using Google or reStructuredText style docstrings inconsistent with codebase.

❌ Wrong:

```python
def foo(x, y):
    """Does foo.

    Args:
        x: First parameter
        y: Second parameter
    """
```

✅ Right:

```python
def foo(x, y):
    """Does foo.

    Parameters
    ----------
    x : type
        First parameter
    y : type
        Second parameter
    """
```

### 8. bin_size is required

**Problem**: Forgetting bin_size parameter causes TypeError.

❌ Wrong:

```python
env = Environment.from_samples(data)  # TypeError: missing required argument
```

✅ Right:

```python
env = Environment.from_samples(positions, bin_size=2.0)  # Explicit is better
```

**Tip**: Choose bin_size based on your data's spatial scale and units (cm, meters, pixels, etc.)

### 9. Error messages show diagnostics

**What this means**: When validation fails, error messages include the actual invalid values to help debugging. Use these diagnostics to understand what went wrong.

Example:

```
ValueError: bin_size must be positive (got -2.0)
ValueError: No active bins found. Data range: [0.0, 100.0], bin_size: 200.0
```

The diagnostic values help identify the problem immediately.

### 10. Memory safety checks (v0.2.1+)

**Problem**: Creating very large grids can cause unexpected memory usage.

**Solution**: Grid creation now includes automatic memory estimation and warnings:

- **Warning at 100MB**: Large grid detected, creation proceeds but you're informed of memory usage

⚠️ Creates large grid (will warn but succeed):

```python
positions = np.random.uniform(0, 100000, (1000, 2))
env = Environment.from_samples(positions, bin_size=1.0)  # ResourceWarning, but succeeds
```

✅ Better (reduce grid size to avoid warning):

```python
# Option 1: Increase bin_size
env = Environment.from_samples(positions, bin_size=10.0)  # Smaller grid, no warning

# Option 2: Filter active bins
env = Environment.from_samples(positions, bin_size=1.0, infer_active_bins=True)

# Option 3: Disable warning (if intentional and you have RAM)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))
```

**Tip**: Warning messages include estimated memory and suggestions for reducing usage.

### 11. Overlay temporal alignment (v0.4.0+)

**Problem**: Overlays and fields at different sampling rates need explicit timestamps.

❌ Wrong (missing frame_times or overlay times for mixed-rate data):

```python
# Position tracked at 120 Hz, fields at 10 Hz - without proper timestamps
position_overlay = PositionOverlay(data=trajectory_120hz)  # No times!
# frame_times is required - this would raise TypeError
# env.animate_fields(fields_10hz, overlays=[position_overlay])
```

✅ Right (provide timestamps for alignment):

```python
# Position tracked at 120 Hz, fields at 10 Hz - with timestamps
position_overlay = PositionOverlay(
    data=trajectory_120hz,
    times=timestamps_120hz  # Overlay timestamps
)
env.animate_fields(
    fields_10hz,
    frame_times=timestamps_10hz,  # Field timestamps - REQUIRED
    overlays=[position_overlay],  # Auto-interpolated to frame_times
)
```

**Tip**: Linear interpolation automatically aligns overlay to field frames.

### 12. HTML backend overlay limitations (v0.4.0+)

**Problem**: HTML backend only supports position and region overlays.

⚠️ Will warn (HTML doesn't support bodypart/head direction):

```python
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[bodypart_overlay, head_direction_overlay],  # Not supported in HTML!
    backend="html"
)
# UserWarning: HTML backend does not support bodypart overlays. Use video or napari backend.
```

✅ Right (use supported overlays or different backend):

```python
# Option 1: Use only position + regions with HTML
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[position_overlay],
    show_regions=True,
    backend="html"
)

# Option 2: Use video/napari for full overlay support
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[bodypart_overlay, head_direction_overlay],
    backend="napari"  # or "video" or "widget"
)
```

**Backend capability matrix**:

- Napari: All overlays including VideoOverlay, EventOverlay (with decay), TimeSeriesOverlay ✓
- Video: All overlays including VideoOverlay, EventOverlay (with decay), TimeSeriesOverlay ✓
- Widget: All overlays including VideoOverlay, EventOverlay (with decay), TimeSeriesOverlay ✓
- HTML: Position + EventOverlay (instant only) + regions (Video/TimeSeries skipped with warning) ⚠️

### 13. Overlay coordinates use environment space

**Problem**: Manually converting coordinates for napari causes inverted display.

❌ Wrong:

```python
# Don't manually swap x,y or invert before passing to overlay
positions_wrong = np.column_stack([y_coords, x_coords])  # Manual swap
overlay = PositionOverlay(data=positions_wrong)  # Display will be wrong!
```

✅ Right:

```python
# Pass coordinates in environment (x, y) format - system handles conversion
positions = np.column_stack([x_coords, y_coords])  # (x, y) format
overlay = PositionOverlay(data=positions)  # System transforms internally
```

**Why**: The animation system automatically transforms environment coordinates to napari pixel space, including axis swap and Y-inversion. Manual pre-transformation causes double-conversion.

### 14. VideoOverlay requires 2D environments (v0.5.0+)

**Problem**: Using VideoOverlay with 1D linearized or 3D environments fails.

❌ Wrong (1D environment):

```python
# 1D linearized track environments don't support video overlay
env_1d = Environment.from_graph(track_graph, ...)  # Creates 1D environment
video = VideoOverlay(source="session.mp4", calibration=calib)
# frame_times is required, and 1D environment raises ValueError
# env_1d.animate_fields(fields, frame_times=frame_times, overlays=[video])  # ValueError!
```

✅ Right (2D environment):

```python
# VideoOverlay works with any 2D environment
env_2d = Environment.from_samples(positions, bin_size=2.0)  # 2D grid
video = VideoOverlay(source="session.mp4", calibration=calib)
env_2d.animate_fields(
    fields, frame_times=frame_times, overlays=[video], backend="napari"
)  # ✓ Works

# Also works with polygon and masked environments
env_polygon = Environment.from_polygon(arena_boundary, bin_size=2.0)
env_polygon.animate_fields(
    fields, frame_times=frame_times, overlays=[video]
)  # ✓ Works with fallback warning
```

**Why**: Video frames are 2D images that map to 2D spatial coordinates. 1D linearized tracks have no meaningful 2D extent for video alignment.

**VideoOverlay support matrix**:

- 2D grid environments: Full support ✓
- 2D polygon/masked environments: Works with fallback warning ⚠️
- 1D linearized tracks: Not supported ✗

### 15. Animation API migration (v0.15.0+)

**Breaking changes**: The animation API now uses speed-based playback control.

**Changes:**

| Old API | New API | Notes |
|---------|---------|-------|
| `fps=30` | `speed=1.0` | Speed is relative to real-time |
| `frame_times=None` | `frame_times=timestamps` | Now **required** |

❌ Old code (v0.14.x and earlier):

```python
# Old: fps parameter controlled playback directly
env.animate_fields(fields, fps=30)
env.animate_fields(fields, fps=30, frame_times=times)  # frame_times was optional
```

✅ New code (v0.15.0+):

```python
# New: frame_times is required, speed controls playback relative to data rate
frame_times = np.arange(len(fields)) / 30.0  # Create timestamps from data
env.animate_fields(fields, frame_times=frame_times)  # speed=1.0 (real-time)
env.animate_fields(fields, frame_times=frame_times, speed=0.1)  # 10% speed (slow motion)
env.animate_fields(fields, frame_times=frame_times, speed=2.0)  # 2× speed (fast forward)
```

**Migration steps:**

1. Add `frame_times` parameter - use timestamps from your data source
2. Replace `fps=X` with `speed=Y` where `speed = X / sample_rate_hz`
3. For slow motion: use `speed < 1.0` (e.g., `speed=0.1` for 10%)
4. For fast forward: use `speed > 1.0` (e.g., `speed=2.0` for 2×)

**Why this change?** The old API conflated data sample rate with playback speed. The new API lets you view 500 Hz replay data in slow motion (`speed=0.1`) while preserving all frames, rather than dropping frames to achieve a display-compatible fps.

## Troubleshooting

### `ModuleNotFoundError: No module named 'neurospatial'`

**Cause**: Dependencies not installed or wrong Python environment.

**Solution**:

```bash
# Sync dependencies (run from project root)
uv sync

# Verify environment
uv run python -c "import neurospatial; print(neurospatial.__file__)"
```

### Tests fail with import errors

**Cause**: Running pytest without `uv run` prefix.

**Solution**:

```bash
# Wrong
pytest

# Right
uv run pytest
```

### `RuntimeError: Environment must be fitted before calling this method`

**Cause**: Calling spatial query methods on unfitted Environment.

**Solution**: Use factory methods, not bare `Environment()`:

```python
# Wrong
env = Environment()
env.bin_at([10, 5])

# Right
env = Environment.from_samples(positions, bin_size=2.0)
env.bin_at([10, 5])
```

### UserWarning when overwriting a region

**Cause**: Using assignment to overwrite an existing region.

**Solution**: Assignment works but emits a `UserWarning`. Use `update_region()` to suppress the warning:

```python
# Works but emits UserWarning
env.regions['goal'] = new_region  # UserWarning: Overwriting existing region 'goal'

# Preferred - no warning
env.regions.update_region('goal', point=new_point)
```

**Note**: This warning follows standard dict semantics while preventing accidental overwrites. To suppress the warning without using `update_region()`, use Python's warnings filter.

### `AttributeError: 'Environment' object has no attribute 'to_linear'`

**Cause**: Calling `to_linear()` on N-D environment. Only 1D (GraphLayout) environments support linearization.

**Solution**: Check `is_1d` first:

```python
if env.is_1d:
    linear_pos = env.to_linear(position)
else:
    bin_idx = env.bin_at(position)  # Use this for N-D
```

### `ValueError: No active bins found`

**Cause**: bin_size too large, threshold too high, or data too sparse.

**Solution**: Read the detailed error message - it provides diagnostics:

- Data range and extent
- Grid shape and bin_size used
- Suggested fixes (reduce bin_size, lower threshold, enable morphological operations)

Example fix:

```python
# If bin_size is too large
env = Environment.from_samples(positions, bin_size=1.0)  # Reduce from 10.0

# If threshold is too high
env = Environment.from_samples(positions, bin_size=2.0, bin_count_threshold=1)

# If data is sparse
env = Environment.from_samples(positions, bin_size=2.0, dilate=True, fill_holes=True)
```

### Pre-commit hooks fail on commit

**Cause**: Linting or formatting issues in code.

**Solution**: Let hooks auto-fix, then commit again:

```bash
git commit -m "message"
# Hooks run and fix files
git add .  # Stage the fixes
git commit -m "message"  # Commit again
```

Or manually run checks before committing:

```bash
uv run ruff check . && uv run ruff format .
git add .
git commit -m "message"
```

### Slow test execution

**Cause**: Running tests without parallelization.

**Solution**: Install pytest-xdist and use parallel execution:

```bash
uv add --dev pytest-xdist
uv run pytest -n auto  # Use all CPU cores
```

### `ResourceWarning: Creating large grid` (v0.2.1+)

**Cause**: Grid estimated to use >100MB memory (warning threshold).

**Solution**: This is a warning, not an error. Grid creation will proceed, but you're being informed about memory usage. Consider:

- Is this grid size intentional?
- Can you increase `bin_size` to reduce resolution?
- Would `infer_active_bins=True` help filter unused bins?

**Common fixes:**

```python
# Fix 1: Increase bin_size (most common)
env = Environment.from_samples(positions, bin_size=10.0)  # Instead of 1.0

# Fix 2: Enable active bin filtering
env = Environment.from_samples(
    positions,
    bin_size=1.0,
    infer_active_bins=True,
    bin_count_threshold=1
)

# Fix 3: Disable warning (if intentional and you have sufficient RAM)
env = Environment.from_samples(positions, bin_size=1.0, warn_threshold_mb=float('inf'))
```

**To suppress the warning globally:**

```python
import warnings
warnings.filterwarnings('ignore', category=ResourceWarning)
```

**Note**: The memory estimate is conservative but approximate. Actual usage may vary by ±20%.

### Type errors despite correct code

**Cause**: May be using outdated type stubs or IDE not recognizing runtime checks.

**Note**: This project includes a `py.typed` marker (v0.2.1+) for type checking support. If you encounter type errors, ensure you're using the latest version. IDE warnings may be false positives that can be ignored if tests pass.

### `ImportError: pynwb is required for NWB integration` (v0.7.0+)

**Cause**: NWB dependencies are optional and not installed.

**Solution**: Install the appropriate NWB extras:

```bash
# Basic NWB support (position, head direction, environment)
pip install neurospatial[nwb]

# With pose estimation support (ndx-pose)
pip install neurospatial[nwb-pose]

# With events support (ndx-events)
pip install neurospatial[nwb-events]

# Full NWB support (all extensions)
pip install neurospatial[nwb-full]

# Or with uv
uv add neurospatial[nwb-full]
```

### `KeyError: No Position found` when reading NWB (v0.7.0+)

**Cause**: NWB file doesn't contain Position data in expected location.

**Solution**: Check where Position data is stored and specify explicitly:

```python
from neurospatial.nwb import read_position

# Check what's in the NWB file
print(nwbfile.processing.keys())           # Processing modules
print(nwbfile.acquisition.keys())          # Acquisition containers

# If Position is in a custom processing module
positions, timestamps = read_position(nwbfile, processing_module="custom_module")

# If there are multiple SpatialSeries, specify which one
positions, timestamps = read_position(nwbfile, position_name="position_xy")
```

### `KeyError: No PoseEstimation found` when reading NWB (v0.7.0+)

**Cause**: NWB file doesn't contain PoseEstimation (ndx-pose) data.

**Solution**: Verify ndx-pose data exists and specify the name if multiple exist:

```python
from neurospatial.nwb import read_pose

# List available PoseEstimation containers (look for ndx-pose data)
# PoseEstimation is typically in processing/behavior/

# If multiple PoseEstimation exist, specify by name
bodyparts, timestamps, skeleton = read_pose(nwbfile, pose_estimation_name="DLC_pose")
```

### `ValueError: Place field '{name}' already exists` when writing to NWB (v0.7.0+)

**Cause**: Attempting to write a place field with a name that already exists.

**Solution**: Use `overwrite=True` to replace existing data:

```python
from neurospatial.nwb import write_place_field

# Replace existing place field
write_place_field(nwbfile, env, field, name="cell_001", overwrite=True)

# Or use a different name
write_place_field(nwbfile, env, field, name="cell_001_v2")
```

### Environment round-trip loses some properties (v0.7.0+)

**Cause**: NWB storage uses reconstructed layout, not original layout engine.

**Solution**: This is expected behavior. Round-trip preserves:

- ✓ `bin_centers` (exact)
- ✓ `connectivity` graph structure and edge weights
- ✓ `dimension_ranges`
- ✓ `units` and `frame` metadata
- ✓ Regions (points and polygons)

Not preserved:

- ✗ Original layout engine type (reconstructed as generic layout)
- ✗ Grid-specific metadata (`grid_shape`, `grid_edges`, `active_mask`)
- ✗ Layout engine's `is_1d` property (always `False` for reconstructed)

```python
# Original environment
env = Environment.from_samples(positions, bin_size=2.0)
env.layout.__class__.__name__  # 'RegularGridLayout'

# After round-trip
loaded = read_environment(nwbfile)
loaded.layout.__class__.__name__  # '_ReconstructedLayout'
# But spatial queries still work identically
```

### `RuntimeError: Environment must be fitted` when writing to NWB (v0.7.0+)

**Cause**: Attempting to write an unfitted Environment.

**Solution**: Use factory methods to create fitted environments:

```python
# Wrong - unfitted environment
env = Environment()
write_environment(nwbfile, env)  # RuntimeError!

# Right - use factory methods
env = Environment.from_samples(positions, bin_size=2.0)
write_environment(nwbfile, env)  # Works
```

### "No start node set" warning in track graph annotation (v0.9.0+)

**Cause**: Saving a track graph without explicitly setting a start node.

**Solution**: This is a warning, not an error. The annotation will proceed with Node 0 as the default start. To set a specific start node:

```python
# In the napari widget:
# 1. Select a node in the Node List
# 2. Press Shift+S to set it as start
# 3. The start node is marked with ★ in the list

# Or set programmatically if loading initial data:
# The first node in initial_nodes becomes the start by default
```

**Note**: The start node affects edge order when using `infer_edge_layout()` - DFS traversal starts from this node.

### Track graph edge order seems wrong (v0.9.0+)

**Cause**: Automatic edge ordering via DFS may not match expected linearization.

**Solution**: The edge order is inferred by depth-first search from the start node. You can:

1. **Set a different start node**: DFS traversal order depends on which node you start from
2. **Manually reorder edges**: Use the Edge Order list in the widget with Move Up/Down buttons
3. **Reset to auto**: Click "Reset to Auto" to re-run `infer_edge_layout()`
4. **Preview first**: Click "Preview Linearization" to see the 1D layout before saving

```python
# After annotation, you can also override edge order:
env = result.to_environment(
    bin_size=2.0,
    edge_spacing=[0.0, 10.0, 0.0],  # Custom spacing
)
```

### `ValueError: Cannot create Environment: no track graph` (v0.9.0+)

**Cause**: Calling `result.to_environment()` when the track graph is empty or invalid.

**Solution**: The annotation requires at least 2 nodes and 1 edge to create a valid track graph:

```python
from neurospatial.annotation import annotate_track_graph

result = annotate_track_graph("maze.mp4")

# Check if track graph exists before creating Environment
if result.track_graph is not None:
    env = result.to_environment(bin_size=2.0)
else:
    print("No valid track graph created")
    print(f"Nodes: {len(result.node_positions)}")
    print(f"Edges: {len(result.edges)}")
```

### Track graph coordinates don't match video (v0.9.0+)

**Cause**: Calibration not applied or incorrect calibration parameters.

**Solution**: Check calibration settings:

```python
# Without calibration: coordinates are in pixels (video pixel coordinates)
result = annotate_track_graph("maze.mp4")
print(result.node_positions)  # Pixel values, e.g., [(100.0, 200.0), ...]
print(result.pixel_positions)  # Same as node_positions when no calibration

# With calibration: coordinates are transformed to cm
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar

# Create calibration from a known scale bar in the video
transform = calibrate_from_scale_bar(
    point1=(100, 300),    # First scale bar endpoint (pixels)
    point2=(300, 300),    # Second scale bar endpoint (pixels)
    length_cm=50.0,       # Known distance between points (cm)
    frame_size_px=(640, 480),
)
calib = VideoCalibration(transform, (640, 480))

result = annotate_track_graph("maze.mp4", calibration=calib)
print(result.node_positions)   # Transformed to cm
print(result.pixel_positions)  # Original pixel positions preserved
```

### TimeSeriesOverlay not showing in HTML backend (v0.14.0+)

**Cause**: HTML backend does not support TimeSeriesOverlay.

**Solution**: TimeSeriesOverlay requires a separate panel with scrolling time series plots, which cannot be rendered in static HTML images. Use video or napari backend instead:

```python
# Will warn - HTML doesn't support TimeSeriesOverlay
env.animate_fields(
    fields,
    frame_times=frame_times,  # Required
    overlays=[timeseries_overlay],
    backend="html"  # Warning emitted, time series skipped
)

# Right - use video or napari backend
env.animate_fields(
    fields,
    frame_times=frame_times,  # Required for time series
    overlays=[timeseries_overlay],
    backend="video",
    save_path="animation.mp4"
)

# Or napari for interactive viewing
env.animate_fields(
    fields,
    frame_times=frame_times,
    overlays=[timeseries_overlay],
    backend="napari"
)
```

**Note**: Other overlays (position, bodypart, events, regions) still render in HTML. Only TimeSeriesOverlay and VideoOverlay are skipped with warning.

### TimeSeriesOverlay synchronization

**Note**: `frame_times` is required for `animate_fields()`. For TimeSeriesOverlay, this enables proper synchronization:

```python
# frame_times is required - provides temporal structure for synchronization
frame_times = np.linspace(0, duration, len(fields))
env.animate_fields(
    fields,
    frame_times=frame_times,  # Required for all animate_fields calls
    overlays=[timeseries_overlay],
    backend="napari"
)
```

### Time series data and times must be same length

**Cause**: TimeSeriesOverlay data and times arrays have different lengths.

**Solution**: Ensure data and times arrays match:

```python
# Wrong - mismatched lengths
data = np.linspace(0, 100, 1000)  # 1000 points
times = np.linspace(0, 10, 500)   # 500 points
overlay = TimeSeriesOverlay(data=data, times=times)  # ValueError!

# Right - matched lengths
data = np.linspace(0, 100, 1000)
times = np.linspace(0, 10, 1000)  # Same length
overlay = TimeSeriesOverlay(data=data, times=times)  # Works
```
