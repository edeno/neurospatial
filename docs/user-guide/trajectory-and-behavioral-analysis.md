# Trajectory Metrics & Behavioral Segmentation

This guide covers trajectory characterization metrics and automatic behavioral epoch detection using the `neurospatial.behavior.trajectory`, `neurospatial.behavior.segmentation`, and `neurospatial.behavior.visits` modules.

Import trajectory operations from `neurospatial.behavior.trajectory`.
Missing or outside positions produce `NaN` geodesic steps. Check observation
coverage before reducing movement metrics; a missing interval is not zero movement.

## Overview

Neurospatial provides a comprehensive suite of tools for analyzing animal trajectories and automatically detecting behavioral epochs:

**Trajectory Metrics**:
- Distance traveled across recordings, and resampling at equal distances
- Turn angles between movement vectors
- Step lengths along the path
- Home range estimation (core territory)
- Mean square displacement (diffusion classification)

**Behavioral Segmentation**:
- Region-based segmentation (crossings, runs between regions)
- Lap detection (circular track analysis)
- Trial segmentation (task-based epochs)
- Trajectory similarity (pattern comparison, goal-directed behavior)

---

## Table of Contents

1. [Trajectory Characterization](#trajectory-characterization)
   - [Distance Traveled Across Recordings and Distance Resampling](#distance-traveled-across-recordings-and-distance-resampling)
   - [Turn Angles](#turn-angles)
   - [Step Lengths](#step-lengths)
   - [Home Range](#home-range)
   - [Mean Square Displacement](#mean-square-displacement)
2. [Region-Based Segmentation](#region-based-segmentation)
   - [Detecting Region Crossings](#detecting-region-crossings)
   - [Observed Region Visits and Transitions](#observed-region-visits-and-transitions)
   - [Detecting Runs Between Regions](#detecting-runs-between-regions)
   - [Velocity-Based Segmentation](#velocity-based-segmentation)
3. [Lap Detection](#lap-detection)
   - [Auto Template Detection](#auto-template-detection)
   - [Reference Method](#reference-method)
   - [Region Method](#region-method)
   - [Direction Detection](#direction-detection)
4. [Trial Segmentation](#trial-segmentation)
   - [T-maze Trials](#t-maze-trials)
   - [Y-maze Trials](#y-maze-trials)
   - [Radial Arm Maze](#radial-arm-maze)
5. [Trajectory Similarity](#trajectory-similarity)
   - [Jaccard Overlap](#jaccard-overlap)
   - [Correlation](#correlation)
   - [Hausdorff Distance](#hausdorff-distance)
   - [Dynamic Time Warping](#dynamic-time-warping)
6. [Goal-Directed Behavior](#goal-directed-behavior)
   - [Directedness Score](#directedness-score)
   - [Replay Analysis](#replay-analysis)
7. [Complete Workflows](#complete-workflows)
   - [Circular Track Lap-by-Lap Analysis](#circular-track-lap-by-lap-analysis)
   - [T-maze Trial Analysis](#t-maze-trial-analysis)
   - [Exploration to Goal-Directed Transition](#exploration-to-goal-directed-transition)

---

## Trajectory Characterization

### Distance Traveled Across Recordings and Distance Resampling

**The question.** Did the animal take the same route in two sessions? One
session was filmed at 30 Hz while the animal ran at 22 cm/s; the other at
120 Hz while it ran at 47 cm/s. Each session is two recordings joined into one
array, and the tracker lost the animal for 0.3 s in each. Comparing the
sessions frame by frame or second by second mixes route differences with
frame-rate and speed differences, and the joins and dropouts add distance the
animal never ran.

Two choices answer it. First, count distance only over intervals that were
actually observed: lost frames are `NaN`, and the first frame of a new
recording is marked with `break_before`, so neither adds distance. Second,
resample each path at equal distances along the route (here every 2 cm), so
sample *k* is the same distance from the run's start in both sessions,
whatever the speed or frame rate.

The simulation runs the animal counterclockwise around a circular track of
40 cm radius. The two recordings share one clock, as when files are
concatenated, and the animal is put back on the track at a random place for
the second one.

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.behavior import compute_cumulative_distance, traveled_path_length

rng = np.random.default_rng(0)
radius = 40.0  # cm, circular track


def simulate_session(frame_rate, running_speed):
    """Two 20 s recordings on one clock, with a 0.3 s tracking dropout.

    The animal runs counterclockwise at a constant speed (cm/s). The second
    recording starts wherever the animal was put back on the track.
    """
    times = np.arange(int(40 * frame_rate)) / frame_rate  # seconds
    recording_id = (times >= 20.0).astype(int)
    put_back_angle = rng.uniform(0.0, 2 * np.pi)
    angle = np.where(
        recording_id == 0,
        running_speed * times / radius,
        put_back_angle + running_speed * (times - 20.0) / radius,
    )
    positions = radius * np.column_stack([np.cos(angle), np.sin(angle)])  # cm
    dropout_start = rng.uniform(5.0, 15.0)
    lost = (times >= dropout_start) & (times < dropout_start + 0.3)
    positions[lost] = np.nan  # lost frames are missing, not zero
    return times, positions, recording_id


sessions = {
    "30 Hz, 22 cm/s": simulate_session(frame_rate=30.0, running_speed=22.0),
    "120 Hz, 47 cm/s": simulate_session(frame_rate=120.0, running_speed=47.0),
}

for name, (times, positions, recording_id) in sessions.items():
    # True at the first frame of the second recording
    new_recording = np.r_[False, np.diff(recording_id) != 0]
    distance = compute_cumulative_distance(times, positions, break_before=new_recording)
    observed_time = np.diff(distance.run_time_bounds, axis=1).sum()
    total = traveled_path_length(times, positions, break_before=new_recording)
    runs = ", ".join(f"{length:.1f}" for length in distance.run_path_lengths)
    print(f"{name}: runs of {runs} cm")
    print(f"  total {total:.1f} cm in {observed_time:.2f} s observed")
    print(f"  without break_before: {traveled_path_length(times, positions):.1f} cm")
```

Each session splits into three **runs**, stretches of continuous observation:
the dropout ends the first and the new recording ends the second. The 30 Hz
session prints runs of 168.7, 263.3 and 439.3 cm and a total of 871.2 cm in
39.60 s of observed time, which is 22 cm/s times 39.60 s: nothing was added
across the dropout or the join, and the lost frames did not turn the total
into `NaN`. The 120 Hz session gives 1864.7 cm, likewise 47 cm/s times its
observed time. `compute_cumulative_distance` returns the distance at every
frame (`run_distance` restarts at 0 in each run, `cumulative_distance` keeps
adding across runs); `traveled_path_length` returns only the total.

Without `break_before` the totals are 898.5 and 1929.8 cm. The clock runs
straight from one file into the next, so nothing marks the join, and the
27.3 cm and 65.1 cm jumps to where the animal was put back are counted as
running. A break is seen only when you declare it: a time gap longer than
`max_gap` (0.5 s by default), `break_before`, or `epochs`.

Now resample both sessions every 2 cm and compare them. The spacing sets the
distance scale of the comparison: choose it larger than your tracking jitter
and smaller than the turns you want to resolve. The example rebuilds the same
two sessions so that it runs on its own.

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial.behavior import (
    compute_planar_curvature,
    compute_speed,
    compute_trajectory_curvature,
    resample_by_distance,
)

rng = np.random.default_rng(0)
radius = 40.0  # cm, circular track


def simulate_session(frame_rate, running_speed):
    """Two 20 s recordings on one clock, with a 0.3 s tracking dropout.

    The animal runs counterclockwise at a constant speed (cm/s). The second
    recording starts wherever the animal was put back on the track.
    """
    times = np.arange(int(40 * frame_rate)) / frame_rate  # seconds
    recording_id = (times >= 20.0).astype(int)
    put_back_angle = rng.uniform(0.0, 2 * np.pi)
    angle = np.where(
        recording_id == 0,
        running_speed * times / radius,
        put_back_angle + running_speed * (times - 20.0) / radius,
    )
    positions = radius * np.column_stack([np.cos(angle), np.sin(angle)])  # cm
    dropout_start = rng.uniform(5.0, 15.0)
    lost = (times >= dropout_start) & (times < dropout_start + 0.3)
    positions[lost] = np.nan  # lost frames are missing, not zero
    return times, positions, recording_id


sessions = {
    "30 Hz, 22 cm/s": simulate_session(frame_rate=30.0, running_speed=22.0),
    "120 Hz, 47 cm/s": simulate_session(frame_rate=120.0, running_speed=47.0),
}

resampled = {}
for name, (times, positions, recording_id) in sessions.items():
    new_recording = np.r_[False, np.diff(recording_id) != 0]
    path = resample_by_distance(
        times, positions, spacing=2.0, break_before=new_recording
    )
    resampled[name] = path
    print(f"{name}: {len(times)} frames -> {path.times.size} samples 2 cm apart")

# Both sessions start at the same place. Compare them 1 s after the start ...
(t_slow, xy_slow, _), (t_fast, xy_fast, _) = sessions.values()
apart = np.linalg.norm(
    xy_slow[np.searchsorted(t_slow, 1.0)] - xy_fast[np.searchsorted(t_fast, 1.0)]
)
print(f"Same time (1 s): {apart:.1f} cm apart")
# ... and at the same distances along the first run
slow, fast = (path.positions[path.run_ids == 0] for path in resampled.values())
n_shared = min(len(slow), len(fast))
apart = np.linalg.norm(slow[:n_shared] - fast[:n_shared], axis=1).max()
print(f"Same distance ({n_shared} samples): at most {apart:.4f} cm apart")

for name, (times, positions, recording_id) in sessions.items():
    path = resampled[name]
    # compute_speed has no break_before: compute it in each recording
    speed = np.concatenate(
        [
            compute_speed(times[recording_id == k], positions[recording_id == k])
            for k in np.unique(recording_id)
        ]
    )
    speed_at_samples = path.interpolate_signal(speed)
    # Feed the resampled path onward with its own breaks and no time-gap limit
    curvature = compute_planar_curvature(
        path.times, path.positions, max_gap=None, break_before=path.break_before
    )
    turn_per_frame = compute_trajectory_curvature(positions, times)
    print(f"{name}:")
    print(f"  median speed at the samples: {np.median(speed_at_samples):.1f} cm/s")
    print(
        f"  median curvature: {np.nanmedian(curvature):.4f} /cm "
        f"(radius {1 / np.nanmedian(curvature):.2f} cm)"
    )
    print(f"  legacy median turn per frame: {np.nanmedian(turn_per_frame):.4f} rad")
```

**Equal distances, not equal times.** The 1200 frames at 30 Hz become 437
samples and the 4800 frames at 120 Hz become 934: the faster animal covered
more track, and the frame rate no longer sets the number of samples. One
second after the start the two animals are 24.6 cm apart on the track, but the
85 resampled samples of the first run (up to the dropout of the 30 Hz
session) agree to within 0.0022 cm. Each run is resampled from its own start,
so samples never bridge the dropout or the join. Each resampled sample keeps
its time on the original clock (`path.times`) and the two original frames
around it (`source_left`, `source_right`, `weights`); `path.samples_table()`
lists them in one table.

**Carrying a signal onto the samples.** `interpolate_signal` combines a
per-frame signal from the two frames around each sample. The median speed at
the samples is 22.0 and 46.7 cm/s. The second is slightly below 47 cm/s
because `compute_speed` smooths the velocity over 0.1 s, and averaging
velocity directions around a curve shortens it. Use `interpolate_signal` for
continuous signals such as speed or LFP amplitude; take labels and counts at
`source_left`.

**Path shape.** `compute_planar_curvature` measures how sharply the path
bends, in 1/cm: 1/40 = 0.025 /cm on this track. On the resampled paths the
medians are 0.0248 /cm at 30 Hz and 0.0250 /cm at 120 Hz (radii 40.39 and
40.02 cm), so the two sessions agree within 1% although their speed and frame
rate differ. The small residual comes from resampling: a sample that falls
between two frames is placed on the straight line joining them, slightly
inside the circle, and the longer steps between frames at 30 Hz put it
further inside. The legacy `compute_trajectory_curvature` instead returns the
turn per frame in radians (despite its name, it is not a curvature): 0.0183
rad at 30 Hz and 0.0098 rad at 120 Hz, the speed divided by the radius and by
the frame rate. It differs between the sessions although the route is the
same, so do not compare it across frame rates or speeds.

The resampled path is passed on with `max_gap=None` and
`break_before=path.break_before`. Resampled samples are not evenly spaced in
time (a slow animal can take more than `max_gap` to cover 2 cm), so the time-gap
limit would break the path where the animal was merely slow; the run starts in
`path.break_before` keep the real breaks.

**Using your own recording.**

- Pass `times` in seconds on one clock, strictly increasing. If each recording
  restarts its clock at 0, map them to one clock first or analyze them in
  separate calls.
- Set dropped or low-confidence frames to `NaN` in `positions`; never fill them
  with zeros or interpolate across them.
- Build `break_before` from your recording or epoch IDs:
  `break_before = np.r_[False, recording_id[1:] != recording_id[:-1]]`. A gap
  in the clock longer than `max_gap` breaks the path on its own.
- Choose `spacing` in the units of your positions (cm here), at the distance
  scale of your question.
- When feeding a resampled path into another time-and-position function, pass
  `max_gap=None` and `break_before=path.break_before`.

---

### Turn Angles

Turn angles quantify changes in movement direction at each position along the trajectory.

```python
from neurospatial.behavior.trajectory import compute_turn_angles
import numpy as np

# Compute turn angles from continuous positions
# positions: (n_samples, 2) array of x,y coordinates
turn_angles = compute_turn_angles(positions)

# Analyze turning behavior
mean_turn = np.abs(np.mean(turn_angles))  # Average turning magnitude
turn_variance = np.var(turn_angles)  # Variability in turning

print(f"Mean turn angle: {np.degrees(mean_turn):.1f}°")
print(f"Turn variance: {turn_variance:.3f}")
```

**Key Points**:
- **Uses continuous positions** for sub-bin precision (ecology standard)
- Returns angles in radians `[-π, π]`
- Positive = left turns, Negative = right turns, 0 = straight
- Meaningful for 2-D (x, y) positions only. With more columns the cross
  product uses x and y but the dot product uses every column, so the angle is
  neither the horizontal-plane nor the 3-D angle; pass `positions[:, :2]`
- Stationary periods (consecutive duplicate positions) are filtered out, so the
  output is shorter than `positions` and not aligned with your frames. It also
  ignores time gaps, dropped frames and recording breaks. For one angle per
  frame that is `NaN` next to a dropout, gap or break, use
  `compute_planar_turn_angles(times, positions)`
- Uses vectorized NumPy operations for efficiency

**Applications**:
- **Path straightness**: Low turn variance indicates directed movement
- **Circling behavior**: Consistent positive or negative angles
- **Random search**: High turn variance, zero mean

**Reference**: Traja package (behavioral trajectory analysis)

---

### Step Lengths

Step lengths are the distances between consecutive positions. Supports both Euclidean (straight-line) and geodesic (graph-based) distances.

```python
from neurospatial.behavior.trajectory import compute_step_lengths

# Euclidean distance (default, ecology standard)
step_lengths = compute_step_lengths(positions, metric="euclidean")

# OR: Geodesic distance for constrained environments (requires env)
# step_lengths_geo = compute_step_lengths(positions, metric="geodesic", env=env)

# Analyze movement statistics
mean_step = np.mean(step_lengths)
step_cv = np.std(step_lengths) / mean_step  # Coefficient of variation

print(f"Mean step length: {mean_step:.2f} cm")
print(f"Step length CV: {step_cv:.3f}")
```

**Key Points**:
- **Euclidean (default)**: Straight-line distance in physical space, uses continuous positions
- **Geodesic**: Graph shortest-path distance, respects walls/obstacles, requires `env`
- Stationary periods (duplicate positions) return 0.0
- For geodesic: disconnected bins return `np.inf`

**Applications**:
- **Movement speed** (step_length / dt): Velocity estimation
- **Step variability**: High CV suggests variable exploration
- **Path efficiency**: Compare actual steps to direct distance

---

### Home Range

Home range is the set of bins containing a specified percentile of time spent. This metric uses discretized bins (occupancy-based).

<!-- docs-test: run -->
```python
from neurospatial.behavior.trajectory import compute_home_range
from neurospatial import Environment

# Create environment and map positions to bins
env = Environment.from_samples(positions, bin_size=5.0)
position_bins = env.bin_at(positions)

# Compute core territory (95th percentile)
home_range_bins = compute_home_range(position_bins, times=times, percentile=95.0)

# Compute home range area
home_range_area = env.bin_sizes[home_range_bins].sum()

print(f"Home range: {home_range_area:.1f} cm²")
core_bins = compute_home_range(position_bins, times=times, percentile=50.0)
print(f"Core area (50%): {env.bin_sizes[core_bins].sum():.1f} cm²")
```

**Key Points**:
- Uses occupancy-based selection (ecology standard)
- Returns sorted bin indices (most visited first)
- `percentile=50` gives core area, `percentile=95` gives full range
- `percentile=100` gives all visited bins

**Applications**:
- **Territory size**: Total area used by animal
- **Core vs periphery**: Compare 50% vs 95% areas
- **Space use patterns**: Identify preferred locations
- **Habitat selection**: Overlap with resource locations

**Reference**: adehabitatHR (R ecology package)

---

### Mean Square Displacement

Mean square displacement (MSD) classifies diffusion processes based on scaling exponent α. **Critical**: Uses continuous positions for accurate diffusion exponent estimation.

```python
from neurospatial.behavior.trajectory import mean_square_displacement

# Compute MSD from continuous positions (Euclidean distance, ecology standard)
_msd = mean_square_displacement(
    times, positions, metric="euclidean", max_tau=10.0
)
tau_values, msd_values = _msd.lags, _msd.msd

# OR: Geodesic distance for constrained environments
# tau_geo, msd_geo = mean_square_displacement(
#     positions, times, metric="geodesic", env=env, max_tau=10.0
# )

# Fit power law: MSD ~ τ^α
log_tau = np.log(tau_values[1:])
log_msd = np.log(msd_values[1:])
alpha = np.polyfit(log_tau, log_msd, 1)[0]

print(f"Diffusion exponent α = {alpha:.2f}")
if alpha < 0.8:
    print("Subdiffusive (confined)")
elif alpha < 1.2:
    print("Normal diffusion (random walk)")
else:
    print("Superdiffusive (directed/ballistic)")
```

**Mathematical Definition**:

$$
\text{MSD}(\tau) = \langle |r(t + \tau) - r(t)|^2 \rangle
$$

**Diffusion Classification**:
- α < 1: Subdiffusive (confined, obstacles)
- α ≈ 1: Normal diffusion (random walk)
- α > 1: Superdiffusive (directed, persistent)
- α ≈ 2: Ballistic motion (constant velocity)

**Key Points**:
- Uses "all pairs" estimator (standard for stationary processes)
- Handles disconnected bins gracefully (skips invalid pairs)
- Returns an ``MSDResult`` dataclass with ``lags`` (tau values) and
  ``msd`` (MSD values) fields, both shape (n_lags,)

**Applications**:
- **Movement classification**: Random vs directed search
- **Learning dynamics**: Transition from exploration (α≈1) to exploitation (α>1)
- **Spatial memory**: Decreased diffusion after learning

**Reference**: Sims et al. (2014), yupi package

---

## Region-Based Segmentation

### Detecting Region Crossings

Detect when the animal enters or exits named regions.

<!-- docs-test: run setup=docs_trajectory_region_crossings -->
```python
from neurospatial.behavior.segmentation import detect_region_crossings

# Detect all crossings (entries and exits)
crossings = detect_region_crossings(
    position_bins, times, env, region_name="goal", direction="both"
)

# Filter by direction
entries = detect_region_crossings(
    position_bins, times, env, region_name="goal", direction="entry"
)
exits = detect_region_crossings(
    position_bins, times, env, region_name="goal", direction="exit"
)

# Analyze crossing events
for crossing in crossings:
    print(f"{crossing.direction} at t={crossing.time:.2f}s, bin={crossing.bin_index}")
```

**Returns**: List of `Crossing` objects
- `time`: Timestamp of crossing event
- `direction`: "entry" or "exit"
- `bin_index`: Bin index where crossing occurred

**Applications**:
- **Dwell time analysis**: Time between entry and exit
- **Visit frequency**: Count entries to region
- **Approach behavior**: Timing of region approaches
- **Task performance**: Correct vs incorrect entries

---

### Observed Region Visits and Transitions

Use `detect_region_visits` when you want an observation-based event table across
several regions, without assigning task trials, rewards, or behavioral success.
It retains visits that start inside a region or end before an exit is observed,
with `entry_observed` and `exit_observed` flags identifying those censored bounds.
The selected regions must have disjoint bin-center memberships; overlaps raise
an error instead of choosing a region by its order in the argument.

<!-- docs-test: run -->
```python
from neurospatial import Regions

import numpy as np
import pandas as pd
from shapely.geometry import box
from neurospatial import Environment
from neurospatial.behavior import detect_region_visits, region_transitions

# A known corridor: A, travel, travel, B, travel, C.
env = Environment.from_grid_mask(
    active_mask=np.ones((6, 1), dtype=bool),
    grid_edges=(np.arange(7, dtype=float), np.array([0.0, 1.0])),
)
env = env.with_metadata(units="cm")
region_builder = Regions(env.regions.values())
for name, left in [("A", 0), ("B", 3), ("C", 5)]:
    region_builder.add(name, polygon=box(left, 0, left + 1, 1))
env = env.with_regions(region_builder)

position_bins = np.array([0, 1, 0, 1, 3, 4, 5])
times = np.arange(len(position_bins), dtype=float) / 10.0
visits = detect_region_visits(
    position_bins, times, env, region_names=["A", "B", "C"], max_gap=0.5
)
transitions = region_transitions(visits)
visit_table = pd.DataFrame(visits)
transition_table = pd.DataFrame(transitions)
print(visit_table[["region_name", "duration", "entry_observed", "exit_observed"]])
print(transition_table[["source_region", "target_region", "duration"]])
assert not visits[0].entry_observed  # Starting inside is not an observed entry.
assert [(t.source_region, t.target_region) for t in transitions] == [
    ("A", "B"), ("B", "C")
]
```

`RegionVisit` and `RegionTransition` are immutable scalar records. Their visit
and run identifiers are local to this detection call, while their sample
indices refer to the original arrays. A visit's `[start_index:stop_index]`
slice contains its inside-region samples. A transition's `start_index` and
`end_index` are entry samples: include both when plotting its path with
`position_bins[start_index:end_index + 1]`.

`region_transitions` pairs consecutive visits in the same observed run when both
entries were observed and the names differ. For `A, A, B`, only the last
`A -> B` is retained; the visits themselves remain separate. Build transitions
before filtering visits, since omitted visit identifiers are not bridged.
Entry-to-entry `duration` includes source dwell time. Use
`detect_runs_between_regions` below for source-exit to target-entry intervals.

Invalid bins (`-1` or outside the environment's bin range), recording gaps, and
excluded epochs break observation continuity; they do not fabricate exits or
entries. Valid bins outside all selected regions are observed travel. Singleton
observations are retained as zero-duration, doubly censored visits. Empty event
lists remain empty; check for an empty list before grouping a DataFrame made
from it.

Epochs are half-open `[start, stop)` for events: entry exactly at `stop` is
excluded. A fully observed sampling interval ending at that boundary can still
contribute dwell time, with a censored visit end. Visit duration uses complete
sampling intervals and never extrapolates across a gap or fractional epoch
boundary. Shared entry timestamps can terminate one transition and start the
next; use half-open intervals when turning transitions into analysis epochs.

---

### Detecting Runs Between Regions

Detect runs from a source region to a target region, tracking success and failure.

```python
from neurospatial.behavior.segmentation import detect_runs_between_regions

# Detect runs from start to goal
runs = detect_runs_between_regions(
    positions,
    times,
    env,
    source="start",
    target="goal",
    min_duration=1.0,
    max_duration=10.0,
    min_speed=5.0,  # Optional speed filter
)

# Analyze successful vs failed runs
successful = [r for r in runs if r.success]
failed = [r for r in runs if not r.success]

print(f"Success rate: {len(successful) / len(runs):.1%}")
print(f"Mean duration (successful): {np.mean([r.end_time - r.start_time for r in successful]):.2f}s")
```

**Returns**: List of `Run` objects
- `start_time`: Time of source region exit
- `end_time`: Time of target entry or timeout
- `bins`: Bin sequence for the run
- `success`: True if reached target, False if timeout

**Applications**:
- **T-maze alternation**: Runs from start to left/right arms
- **Goal-directed navigation**: Efficiency of paths to goal
- **Learning curves**: Improvement in success rate over trials
- **Strategy analysis**: Compare path patterns for successful vs failed runs

---

### Velocity-Based Segmentation

Segment trajectory into movement vs rest periods based on velocity.

```python
from neurospatial.behavior.segmentation import segment_by_velocity

# Segment into movement epochs
movement_epochs = segment_by_velocity(
    times,
    positions,
    min_speed=10.0,  # cm/s
    min_duration=0.5,  # seconds
    hysteresis=2.0,  # ratio for exit threshold
    smooth_window=0.1,  # seconds (Gaussian SD of the velocity smoothing)
)

# Analyze movement statistics. segment_by_velocity returns list[Run]
# (matching detect_runs_between_regions, detect_laps, ...). For
# velocity epochs, ``bins`` is empty and ``success`` is True; only
# start_time / end_time vary.
total_movement_time = sum(
    run.end_time - run.start_time for run in movement_epochs
)
print(f"Movement time: {total_movement_time:.1f}s ({total_movement_time/times[-1]:.1%})")
```

**Hysteresis Thresholding**:
- Enter movement: velocity > min_speed
- Exit movement: velocity < min_speed / hysteresis
- Prevents rapid flickering between states

**Key Parameters**:
- `min_speed`: Velocity threshold (same units as positions/time)
- `min_duration`: Filter brief epochs
- `hysteresis`: Exit threshold ratio (default 2.0)
- `smooth_window`: Gaussian standard deviation (seconds) of the velocity
  smoothing, the same speed as `compute_speed`; it keeps tracking jitter at
  rest from reading as movement

**Applications**:
- **Rest periods**: Identify stationary epochs
- **Active behavior**: Movement bouts analysis
- **Sleep classification**: Low velocity periods
- **Grooming detection**: Brief stationary periods

---

## Lap Detection

Lap detection identifies repeating circular paths (e.g., circular track running).

### Auto Template Detection

Automatically extracts template lap from trajectory.

```python
from neurospatial.behavior.segmentation import detect_laps

# Auto-detect laps using first 10% as template
laps = detect_laps(
    position_bins,
    times,
    env,
    method="auto",
    min_overlap=0.8,  # Jaccard similarity threshold
    direction="both",  # or "clockwise", "counter-clockwise"
)

# Analyze lap properties
for i, lap in enumerate(laps):
    print(f"Lap {i+1}: {lap.start_time:.1f}-{lap.end_time:.1f}s, "
          f"direction={lap.direction}, overlap={lap.overlap_score:.3f}")
```

**Algorithm**:
1. Extract template from first 10% of trajectory
2. Sliding window search (70%-130% of template length)
3. Compute Jaccard overlap for each window
4. Detect laps exceeding `min_overlap` threshold

---

### Reference Method

Use a user-provided reference lap as template.

```python
# Provide reference lap from previous session
reference_lap = np.array([10, 15, 20, 25, 30, 35], dtype=np.int64)

laps = detect_laps(
    position_bins,
    times,
    env,
    method="reference",
    reference_lap=reference_lap,
    min_overlap=0.7,
)
```

**Use Cases**:
- Compare across sessions with canonical lap
- Control for atypical first lap
- Test if animal follows trained route

---

### Region Method

Define laps as segments between start region crossings.

```python
from neurospatial import Regions

# Add start region (crossing point on track)
from shapely.geometry import Point
region_builder = Regions(env.regions.values())
region_builder.add("start", polygon=Point(50, 50).buffer(5.0))
env = env.with_regions(region_builder)

# Detect laps as inter-crossing intervals
laps = detect_laps(
    position_bins,
    times,
    env,
    method="region",
    start_region="start",
    direction="both",
)
```

**Suitable For**:
- Well-controlled tasks with clear start zone
- Trials with explicit start cue
- Behavioral shaping with defined checkpoints

---

### Direction Detection

Automatically classifies laps as clockwise or counter-clockwise.

**Algorithm**: Uses shoelace formula for signed area
- Positive area → counter-clockwise
- Negative area → clockwise
- Non-2D or degenerate → "unknown"

**Filtering by Direction**:
```python
# Only counter-clockwise laps
ccw_laps = detect_laps(
    position_bins, times, env,
    method="auto",
    direction="counter-clockwise"
)

# Only clockwise laps
cw_laps = detect_laps(
    position_bins, times, env,
    method="auto",
    direction="clockwise"
)
```

**Applications**:
- **Lap-by-lap learning**: Performance improvement over laps
- **Strategy consistency**: Stereotypy of paths
- **Trajectory variability**: Spatial consistency analysis
- **Hippocampal replay**: Lap templates for sequence detection

**References**: Barnes et al. (1997), Dupret et al. (2010)

---

## Trial Segmentation

Segment behavioral tasks into discrete trials based on region entries.

### T-maze Trials

```python
from neurospatial import Regions

from neurospatial.behavior.segmentation import segment_trials

# Define regions
region_builder = Regions(env.regions.values())
region_builder.add("start", polygon=Point(50, 10).buffer(8))
region_builder.add("left", polygon=Point(20, 90).buffer(8))
region_builder.add("right", polygon=Point(80, 90).buffer(8))
env = env.with_regions(region_builder)

# Segment into trials
trials = segment_trials(
    position_bins,
    times,
    env,
    start_region="start",
    end_regions=["left", "right"],
    min_duration=1.0,
    max_duration=15.0,
)

# Analyze outcomes
left_trials = [t for t in trials if t.outcome == "left"]
right_trials = [t for t in trials if t.outcome == "right"]

print(f"Left choices: {len(left_trials)}")
print(f"Right choices: {len(right_trials)}")
print(f"Timeouts: {sum(1 for t in trials if not t.success)}")
```

**Returns**: List of `Trial` objects
- `start_time`: Departure from the start region (the last sample inside it)
- `end_time`: End region entry, timeout, or return to the start region
- `end_region`: Name of end region reached (or None if not reached)
- `success`: True if end region reached, False if timed out or aborted

---

### Y-maze Trials

Y-maze spontaneous alternation task.

```python
from neurospatial import Regions

# Define Y-maze arms
region_builder = Regions(env.regions.values())
region_builder.add("center", polygon=Point(50, 50).buffer(10))
region_builder.add("arm_a", polygon=Point(50, 90).buffer(8))
region_builder.add("arm_b", polygon=Point(20, 30).buffer(8))
region_builder.add("arm_c", polygon=Point(80, 30).buffer(8))
env = env.with_regions(region_builder)

# Segment into arm visits
trials = segment_trials(
    position_bins,
    times,
    env,
    start_region="center",
    end_regions=["arm_a", "arm_b", "arm_c"],
    min_duration=0.5,
    max_duration=10.0,
)

# Compute spontaneous alternation score
outcomes = [t.outcome for t in trials if t.success]
alternations = sum(1 for i in range(len(outcomes)-1) if outcomes[i] != outcomes[i+1])
alternation_rate = alternations / (len(outcomes) - 1)

print(f"Spontaneous alternation: {alternation_rate:.1%}")
```

**Use Case**: Working memory and exploration strategy assessment

---

### Radial Arm Maze

8-arm radial maze for reference/working memory.

```python
from neurospatial import Regions

# Define center and 8 arms
region_builder = Regions(env.regions.values())
region_builder.add("center", polygon=Point(50, 50).buffer(10))
for i in range(8):
    angle = i * 2 * np.pi / 8
    x = 50 + 40 * np.cos(angle)
    y = 50 + 40 * np.sin(angle)
    region_builder.add(f"arm_{i}", polygon=Point(x, y).buffer(8))
env = env.with_regions(region_builder)

# Segment into arm choices
trials = segment_trials(
    position_bins,
    times,
    env,
    start_region="center",
    end_regions=[f"arm_{i}" for i in range(8)],
    min_duration=0.5,
    max_duration=20.0,
)

# Analyze working memory errors (re-entries)
visited_arms = []
errors = 0
for trial in trials:
    if trial.success and trial.end_region:
        if trial.end_region in visited_arms:
            errors += 1  # Re-entry error
        visited_arms.append(trial.end_region)

print(f"Working memory errors: {errors}")
```

**Scientific Applications**:
- **Spatial working memory**: Re-entry errors
- **Reference memory**: Pre-baited vs never-baited arms
- **Optimal foraging**: Efficiency of arm selection
- **Aging/disease models**: Memory impairment quantification

**References**: Olton & Samuelson (1976), Wood et al. (2000)

---

## Trajectory Similarity

Compare spatial and temporal patterns between trajectories.

### Jaccard Overlap

Spatial overlap (set-based, order-independent).

```python
from neurospatial.behavior.segmentation import trajectory_similarity

# Compare two trajectories
similarity = trajectory_similarity(
    position1_bins,
    position2_bins,
    env,
    method="jaccard"
)

print(f"Jaccard similarity: {similarity:.3f}")
```

**Formula**:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Range**: [0, 1] where 1.0 = identical, 0.0 = disjoint

**Applications**:
- **Replay analysis**: Compare awake trajectory to sleep reactivation
- **Learning**: Increasing similarity to optimal path
- **Spatial overlap**: Shared territory between animals

---

### Correlation

Sequential correlation (order-sensitive).

```python
similarity = trajectory_similarity(
    position1_bins,
    position2_bins,
    env,
    method="correlation"
)
```

**Computes**: Pearson correlation between bin index sequences

**Sensitive To**:
- Sequential order (unlike Jaccard)
- Timing of visits

**Applications**:
- **Temporal pattern matching**: Order matters
- **Sequence learning**: Following trained route

---

### Hausdorff Distance

Maximum deviation between paths.

```python
similarity = trajectory_similarity(
    position1_bins,
    position2_bins,
    env,
    method="hausdorff"
)
```

**Formula**:

$$
d_H(A, B) = \max\{\sup_{a \in A} \inf_{b \in B} d(a, b), \sup_{b \in B} \inf_{a \in A} d(a, b)\}
$$

Converted to similarity: $1 - d_H / d_{\max}$

**Applications**:
- **Path comparison**: Worst-case deviation
- **Route fidelity**: Consistency of paths
- **Outlier detection**: Atypical trajectories

---

### Dynamic Time Warping

Optimal alignment allowing temporal shifts.

```python
similarity = trajectory_similarity(
    position1_bins,
    position2_bins,
    env,
    method="dtw"
)
```

**Algorithm**: Finds optimal time-shifted alignment between trajectories

**Allows**:
- Speed variations
- Temporal stretching/compression
- Pauses at different times

**Applications**:
- **Same path, different speeds**: Normalize for speed
- **Hesitation points**: Compare decision locations
- **Learning dynamics**: Same route with improving speed

**Complexity**: O(n₁ × n₂) time and space

**Reference**: Davidson et al. (2009)

---

## Goal-Directed Behavior

### Directedness Score

Quantifies how efficiently an animal moves toward a goal.

```python
from neurospatial.behavior.segmentation import detect_goal_directed_runs

# Detect efficient paths toward goal
goal_runs = detect_goal_directed_runs(
    position_bins,
    times,
    env,
    goal_region="goal",
    directedness_threshold=0.7,
    min_progress=20.0,  # cm
)

# Analyze directedness
for run in goal_runs:
    duration = run.end_time - run.start_time
    path_length = len(run.bins)
    print(f"Run: duration={duration:.1f}s, path_length={path_length} bins")
```

**Directedness Formula** (Pfeiffer & Foster, 2013):

$$
\text{directedness} = \frac{d_{\text{start}} - d_{\text{end}}}{\text{path length}}
$$

**Where**:
- $d_{\text{start}}$: Distance from start position to goal
- $d_{\text{end}}$: Distance from end position to goal
- path_length: Actual path traveled

**Interpretation**:
- directedness = 1.0: Perfectly efficient (straight line)
- directedness = 0.0: No net progress toward goal
- directedness < 0.0: Moving away from goal

---

### Replay Analysis

Detect hippocampal replay sequences that match awake trajectories.

```python
# Awake exploration
awake_trajectory = position_bins[:1000]

# Sleep/rest epochs (detected separately)
sleep_epochs = [(1000, 1200), (1500, 1800), (2000, 2300)]

# Detect replays matching awake trajectory
replays = []
for start, end in sleep_epochs:
    sleep_segment = position_bins[start:end]
    similarity = trajectory_similarity(
        awake_trajectory,
        sleep_segment,
        env,
        method="jaccard"
    )
    if similarity > 0.5:  # Threshold for significant replay
        replays.append((start, end, similarity))

print(f"Detected {len(replays)} replay events")
```

**Use Cases**:
- **Memory consolidation**: Sleep replay of learned routes
- **Planning**: Forward replay toward goals
- **Reverse replay**: Backward sequences from goal
- **Preplay**: Replay before first experience (rodents)

**References**: Pfeiffer & Foster (2013), Davidson et al. (2009), Wilson & McNaughton (1993)

---

## Complete Workflows

### Circular Track Lap-by-Lap Analysis

Analyze learning and performance across laps.

```python
# Step 1: Detect laps
laps = detect_laps(
    position_bins,
    times,
    env,
    method="auto",
    min_overlap=0.8,
    direction="both"
)

# Step 2: Compute metrics per lap using continuous positions
lap_metrics = []
for lap in laps:
    mask = (times >= lap.start_time) & (times <= lap.end_time)
    lap_positions = positions[mask]  # Use continuous positions

    turn_angles = compute_turn_angles(lap_positions)
    step_lengths = compute_step_lengths(lap_positions, metric="euclidean")

    lap_metrics.append({
        "duration": lap.end_time - lap.start_time,
        "mean_turn": np.abs(np.mean(turn_angles)),
        "mean_step": np.mean(step_lengths),
        "path_length": np.sum(step_lengths),
        "overlap": lap.overlap_score,
    })

# Step 3: Analyze learning curve
durations = [m["duration"] for m in lap_metrics]
overlaps = [m["overlap"] for m in lap_metrics]

print(f"First lap duration: {durations[0]:.1f}s")
print(f"Last lap duration: {durations[-1]:.1f}s")
print(f"Learning improvement: {(durations[0] - durations[-1]) / durations[0]:.1%}")

# Step 4: Compare lap similarity
for i in range(len(laps)-1):
    mask1 = (times >= laps[i].start_time) & (times <= laps[i].end_time)
    mask2 = (times >= laps[i+1].start_time) & (times <= laps[i+1].end_time)

    similarity = trajectory_similarity(
        position_bins[mask1],
        position_bins[mask2],
        env,
        method="jaccard"
    )
    print(f"Lap {i+1} vs {i+2}: similarity = {similarity:.3f}")
```

**Applications**:
- **Spatial learning**: Decreasing duration, increasing consistency
- **Trajectory stereotypy**: Increasing lap-to-lap similarity
- **Performance variability**: Within-session stability

---

### T-maze Trial Analysis

Analyze choice behavior and learning in T-maze.

```python
# Step 1: Segment trials
trials = segment_trials(
    position_bins,
    times,
    env,
    start_region="start",
    end_regions=["left", "right"],
    min_duration=1.0,
    max_duration=15.0,
)

# Step 2: Separate by outcome
left_trials = [t for t in trials if t.outcome == "left" and t.success]
right_trials = [t for t in trials if t.outcome == "right" and t.success]

# Step 3: Analyze path similarity within choice type
left_similarities = []
for i in range(len(left_trials)-1):
    mask1 = (times >= left_trials[i].start_time) & (times <= left_trials[i].end_time)
    mask2 = (times >= left_trials[i+1].start_time) & (times <= left_trials[i+1].end_time)

    sim = trajectory_similarity(
        position_bins[mask1],
        position_bins[mask2],
        env,
        method="jaccard"
    )
    left_similarities.append(sim)

print(f"Left choice stereotypy: {np.mean(left_similarities):.3f}")

# Step 4: Detect goal-directed behavior
for trial in trials[:5]:  # First 5 trials
    if trial.end_region:
        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_bins = position_bins[mask]
        trial_times = times[mask]

        goal_runs = detect_goal_directed_runs(
            trial_bins,
            trial_times,
            env,
            goal_region=trial.end_region,
            directedness_threshold=0.5,
            min_progress=10.0,
        )
        print(f"Trial {trials.index(trial)+1}: {len(goal_runs)} goal-directed runs")
```

**Applications**:
- **Choice consistency**: Similarity within choice type
- **Strategy development**: Increasing directedness over trials
- **Decision-making**: Latency to choice, hesitation points

---

### Exploration to Goal-Directed Transition

Detect behavioral shift from exploration to exploitation.

```python
# Compute MSD in sliding windows using continuous positions
window_size = 100  # samples
hop_size = 20
alphas = []

for i in range(0, len(positions) - window_size, hop_size):
    window_positions = positions[i:i+window_size]
    window_times = times[i:i+window_size]

    # Use continuous positions for accurate diffusion exponent
    _msd = mean_square_displacement(
        window_times,
        window_positions,
        metric="euclidean",
        max_tau=5.0
    )
    tau_vals, msd_vals = _msd.lags, _msd.msd

    if len(tau_vals) > 3:
        log_tau = np.log(tau_vals[1:])
        log_msd = np.log(msd_vals[1:])
        alpha = np.polyfit(log_tau, log_msd, 1)[0]
        alphas.append(alpha)

# Detect transition (α crossing threshold)
threshold = 1.2  # Superdiffusive
transition_idx = next((i for i, a in enumerate(alphas) if a > threshold), None)

if transition_idx:
    transition_time = times[transition_idx * hop_size]
    print(f"Exploration → goal-directed transition at t={transition_time:.1f}s")

    # Compare spatial coverage before/after (home range still uses bins)
    env = Environment.from_samples(positions, bin_size=5.0)
    position_bins = env.bin_at(positions)

    explore_bins = position_bins[:transition_idx * hop_size]
    goal_bins = position_bins[transition_idx * hop_size:]

    explore_home = compute_home_range(explore_bins, percentile=95)
    goal_home = compute_home_range(goal_bins, percentile=95)

    print(f"Exploration home range: {len(explore_home)} bins")
    print(f"Goal-directed home range: {len(goal_home)} bins")
```

**Applications**:
- **Learning dynamics**: Transition from exploration to exploitation
- **Spatial memory formation**: Decreasing diffusion after learning
- **Behavioral flexibility**: Switching between strategies

---

## Best Practices

### Choosing Similarity Metrics

- **Jaccard**: Spatial overlap, order-independent → Replay analysis, territory overlap
- **Correlation**: Sequential patterns → Route learning, temporal matching
- **Hausdorff**: Maximum deviation → Path consistency, outlier detection
- **DTW**: Timing-invariant alignment → Same path at different speeds

### Lap Detection Methods

- **Auto**: Default, robust for typical circular trajectories
- **Reference**: Cross-session comparisons, control for atypical first lap
- **Region**: Well-controlled tasks with explicit start zones

### Trial Duration Filters

- **min_duration**: Exclude brief entries (false starts, pass-throughs)
- **max_duration**: Timeout threshold (typically 2-3x median duration)
- Start with permissive thresholds, refine based on task demands

### Directedness Thresholds

- **High threshold (>0.7)**: Strict goal-directed behavior
- **Medium threshold (0.4-0.7)**: Moderately efficient paths
- **Low threshold (<0.4)**: Permissive, includes exploration

### Velocity Segmentation

- **Hysteresis**: Prevents rapid flickering (recommended ratio 2.0)
- **Smooth window**: Match to animal movement timescale (0.1-0.5s typical)
- **Threshold**: Set based on species and environment scale

---

## References

**Trajectory Metrics**:
- Traja: [https://github.com/traja-team/traja](https://github.com/traja-team/traja)
- Sims, D. W., et al. (2014). "Hierarchical random walks in trace fossils and the origin of optimal search behavior." *PNAS*, 111(30), 11073-11078.

**Lap Detection**:
- Barnes, C. A., et al. (1997). "Comparison of spatial and temporal characteristics of neuronal activity in sequential stages of hippocampal processing." *Progress in Brain Research*, 83, 287-300.
- Dupret, D., et al. (2010). "The reorganization and reactivation of hippocampal maps predict spatial memory performance." *Nature Neuroscience*, 13(8), 995-1002.

**Trial Segmentation**:
- Olton, D. S., & Samuelson, R. J. (1976). "Remembrance of places passed: Spatial memory in rats." *Journal of Experimental Psychology: Animal Behavior Processes*, 2(2), 97-116.
- Wood, E. R., et al. (2000). "Hippocampal neurons encode information about different types of memory episodes occurring in the same location." *Neuron*, 27(3), 623-633.

**Trajectory Similarity & Replay**:
- Pfeiffer, B. E., & Foster, D. J. (2013). "Hippocampal place-cell sequences depict future paths to remembered goals." *Nature*, 497(7447), 74-79.
- Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). "Hippocampal replay of extended experience." *Neuron*, 63(4), 497-507.
- Wilson, M. A., & McNaughton, B. L. (1993). "Dynamics of the hippocampal ensemble code for space." *Science*, 261(5124), 1055-1058.

---

## See Also

- [Neuroscience Metrics](neuroscience-metrics.md): Place field analysis, boundary cells
- [Differential Operators](differential-operators.md): Gradient, divergence for flow analysis
- [Signal Processing Primitives](signal-processing-primitives.md): Spatial filtering operations
- [Spike Field Primitives](spike-field-primitives.md): Converting spike trains to firing rate maps
