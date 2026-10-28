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
   - [Joined Recordings](#joined-recordings)
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
array, and the tracker lost the animal for 1.5 s in each. Comparing the
sessions frame by frame or second by second mixes route differences with
frame-rate and speed differences, and the joins and dropouts add distance the
animal never ran.

Two choices answer it. First, count distance only over intervals that were
actually observed: lost frames are `NaN`, and the first frame of a new
recording is marked with `break_before`, so neither adds distance. Second,
resample each path at equal distances along the route (here every 2 cm), so
sample *k* is the same distance from the run's start in both sessions,
whatever the speed or frame rate. Sample *k* is then the same place only if
the runs start at the same place; here the first run of each session starts
at the same point on the track.

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
    """Two 20 s recordings on one clock, with a 1.5 s tracking loss.

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
    lost = (times >= dropout_start) & (times < dropout_start + 1.5)
    positions[lost] = np.nan  # lost frames are missing, not zero
    return times, positions, recording_id


sessions = {
    "30 Hz, 22 cm/s": simulate_session(frame_rate=30.0, running_speed=22.0),
    "120 Hz, 47 cm/s": simulate_session(frame_rate=120.0, running_speed=47.0),
}

quoted = {}
for name, (times, positions, recording_id) in sessions.items():
    # True at the first frame of the second recording
    new_recording = np.r_[False, np.diff(recording_id) != 0]
    distance = compute_cumulative_distance(times, positions, break_before=new_recording)
    observed_time = np.diff(distance.run_time_bounds, axis=1).sum()
    total = traveled_path_length(times, positions, break_before=new_recording)
    runs = ", ".join(f"{length:.1f}" for length in distance.run_path_lengths)
    print(f"{name}: runs of {runs} cm")
    print(f"  total {total:.1f} cm in {observed_time:.2f} s observed")
    unbroken = traveled_path_length(times, positions)
    print(f"  without break_before: {unbroken:.1f} cm")
    quoted[name] = (distance.run_path_lengths, total, observed_time, unbroken)

# The numbers the text below quotes (atol = 0.6 * 10**-decimals)
runs_30, total_30, time_30, unbroken_30 = quoted["30 Hz, 22 cm/s"]
assert np.allclose(runs_30, [168.7, 236.9, 439.3], atol=0.06)
assert np.allclose([total_30, unbroken_30], [844.8, 872.1], atol=0.06)
assert np.isclose(time_30, 38.40, atol=0.006)
_, total_120, _, unbroken_120 = quoted["120 Hz, 47 cm/s"]
assert np.allclose([total_120, unbroken_120], [1808.3, 1873.4], atol=0.06)
```

Each session splits into three **runs**, stretches of continuous observation:
the tracking loss ends the first and the new recording ends the second. Lost
frames are skipped as if their rows were deleted, so a loss counts as a break
only when the time between the frames on either side exceeds `max_gap`; a
shorter one, such as a few low-confidence frames, is bridged by the straight
step between them. The 30 Hz session prints runs of 168.7, 236.9 and 439.3 cm
and a total of 844.8 cm in 38.40 s of observed time, which is 22 cm/s times
38.40 s: nothing was added across the loss or the join, and the lost frames
did not turn the total into `NaN`. The 120 Hz session gives 1808.3 cm,
likewise 47 cm/s times its observed time. `compute_cumulative_distance` returns the distance at every
frame (`run_distance` restarts at 0 in each run, `cumulative_distance` keeps
adding across runs); `traveled_path_length` returns only the total.

Without `break_before` the totals are 872.1 and 1873.4 cm. The clock runs
straight from one file into the next, so nothing marks the join, and the
27.3 cm and 65.1 cm jumps to where the animal was put back are counted as
running. A break is seen only when you declare it: a time gap longer than
`max_gap` (1.0 s by default), `break_before`, or `epochs`.

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
    """Two 20 s recordings on one clock, with a 1.5 s tracking loss.

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
    lost = (times >= dropout_start) & (times < dropout_start + 1.5)
    positions[lost] = np.nan  # lost frames are missing, not zero
    return times, positions, recording_id


sessions = {
    "30 Hz, 22 cm/s": simulate_session(frame_rate=30.0, running_speed=22.0),
    "120 Hz, 47 cm/s": simulate_session(frame_rate=120.0, running_speed=47.0),
}

resampled, counts = {}, []
for name, (times, positions, recording_id) in sessions.items():
    new_recording = np.r_[False, np.diff(recording_id) != 0]
    path = resample_by_distance(
        times, positions, spacing=2.0, break_before=new_recording
    )
    resampled[name] = path
    print(f"{name}: {len(times)} frames -> {path.times.size} samples 2 cm apart")
    counts.append((len(times), path.times.size))
assert counts == [(1200, 424), (4800, 906)]

# Both sessions start at the same place. Compare them 1 s after the start ...
(t_slow, xy_slow, _), (t_fast, xy_fast, _) = sessions.values()
apart = np.linalg.norm(
    xy_slow[np.searchsorted(t_slow, 1.0)] - xy_fast[np.searchsorted(t_fast, 1.0)]
)
print(f"Same time (1 s): {apart:.1f} cm apart")
assert np.isclose(apart, 24.6, atol=0.06)
# ... and at the same distances along the first run
slow, fast = (path.positions[path.run_ids == 0] for path in resampled.values())
n_shared = min(len(slow), len(fast))
apart = np.linalg.norm(slow[:n_shared] - fast[:n_shared], axis=1).max()
print(f"Same distance ({n_shared} samples): at most {apart:.4f} cm apart")
assert n_shared == 85 and np.isclose(apart, 0.0022, atol=0.00006)

stats = {}
for name, (times, positions, recording_id) in sessions.items():
    path = resampled[name]
    new_recording = np.r_[False, np.diff(recording_id) != 0]
    speed = compute_speed(times, positions, break_before=new_recording)
    speed_at_samples = path.interpolate_signal(speed)
    # Feed the resampled path onward with its own breaks and no time-gap limit
    curvature = compute_planar_curvature(
        path.times, path.positions, max_gap=None, break_before=path.break_before
    )
    turn_per_frame = compute_trajectory_curvature(
        times, positions, break_before=new_recording
    )
    print(f"{name}:")
    print(f"  median speed at the samples: {np.median(speed_at_samples):.1f} cm/s")
    for label, statistic in [("median", np.nanmedian), ("mean", np.nanmean)]:
        value = statistic(curvature)
        print(f"  {label} curvature: {value:.4f} /cm (radius {1 / value:.2f} cm)")
    legacy_turn = np.nanmedian(turn_per_frame)
    print(f"  legacy median turn per frame: {legacy_turn:.4f} rad")
    stats[name] = (
        np.median(speed_at_samples),
        np.nanmean(curvature),
        np.nanmedian(curvature),
        legacy_turn,
    )

# The numbers the text below quotes (atol = 0.6 * 10**-decimals)
slow_stats, fast_stats = stats.values()
assert np.allclose([slow_stats[0], fast_stats[0]], [22.0, 46.7], atol=0.06)
assert np.allclose([1 / slow_stats[1], 1 / fast_stats[1]], 40.00, atol=0.006)
assert np.allclose([1 / slow_stats[2], 1 / fast_stats[2]], [40.39, 40.02], atol=0.006)
assert np.allclose([slow_stats[3], fast_stats[3]], [0.0183, 0.0098], atol=6e-5)
```

**Equal distances, not equal times.** The 1200 frames at 30 Hz become 424
samples and the 4800 frames at 120 Hz become 906: the faster animal covered
more track, and the frame rate no longer sets the number of samples. One
second after the start the two animals are 24.6 cm apart on the track, but the
85 resampled samples of the first run (up to the tracking loss of the
30 Hz session) agree to within 0.0022 cm. Each run is resampled from its own
start, so samples never bridge the loss or the join. Each resampled sample keeps
its time on the original clock (`path.times`) and the two original frames
around it (`source_left`, `source_right`, `weights`); `path.samples_table()`
lists them in one table.

**Carrying a signal onto the samples.** `interpolate_signal` combines a
per-frame signal from the two frames around each sample. The median speed at
the samples is 22.0 and 46.7 cm/s. The second is slightly below 47 cm/s
because `compute_speed` smooths the velocity over 0.1 s, and averaging
velocity directions around a curve shortens it. Use `interpolate_signal` for
continuous signals such as speed or LFP amplitude; take labels and counts at
`source_left`. `compute_speed` takes the same `break_before`, so speed is never
smoothed across the join.

**Path shape.** `compute_planar_curvature` measures how sharply the path
bends, in 1/cm: 1/40 = 0.025 /cm on this track. On the resampled paths the
mean curvature gives a radius of 40.00 cm in both sessions, although their
speed and frame rate differ. The medians give radii of 40.39 cm at 30 Hz and
40.02 cm at 120 Hz, within 1% of each other. The small difference comes from
resampling: a sample that falls between two frames is placed on the straight
line joining them, so the resampled path is made of straight pieces that bend
only at the original frames. The bend measured at a sample then depends on
where the frames fall between it and its neighbors: some samples measure
slightly less bend than the circle and others more. At 30 Hz the straight
pieces are longer, so this scatter is larger and the median lands on a
slightly larger radius. The mean averages the scatter out and recovers the
track.

The legacy `compute_trajectory_curvature` instead returns the turn per frame
in radians (despite its name, it is not a curvature). It takes the same
`break_before`, and the tracking loss splits the path, so no turn is
computed or smoothed across the join or the lost frames. The turn per frame still depends
on the frame rate and the running speed: 0.0183 rad at 30 Hz and 0.0098 rad at
120 Hz, the speed divided by the radius and by the frame rate. It differs
between the sessions although the route is the same, so do not compare it
across frame rates or speeds.

The resampled path is passed on with `max_gap=None` and
`break_before=path.break_before`. Resampled samples are not evenly spaced in
time (a slow animal can take more than `max_gap` to cover 2 cm), so the time-gap
limit would break the path where the animal was merely slow; the run starts in
`path.break_before` keep the real breaks.

**Using your own recording.**

- Pass `times` in seconds on one clock, strictly increasing. If each recording
  restarts its clock at 0, map them to one clock first or analyze them in
  separate calls.
- Set dropped or low-confidence frames to `NaN` in `positions`. Do not fill
  them with zeros, and do not interpolate across them first: interpolated
  frames add distance that was never observed.
- To compare runs sample by sample, start each run at a shared landmark, for
  example with one `epochs` window per lap or trial that begins where the lap
  or trial begins.
- Build `break_before` from your recording or epoch IDs:
  `break_before = np.r_[False, recording_id[1:] != recording_id[:-1]]`. Pass
  the same array to every function that takes `break_before`, such as
  `compute_speed`, `traveled_path_length`, `compute_path_efficiency`,
  `compute_cumulative_distance`, `resample_by_distance`, the planar turning
  functions, `compute_home_range` and the detectors in
  [Joined Recordings](#joined-recordings). A gap in the clock longer than `max_gap` breaks the path on its own.
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
  output is shorter than `positions` and not aligned with your frames. It does
  not use times, so it bridges time gaps and recording breaks. For one angle per
  frame that is `NaN` at a missing frame and next to a long tracking loss or a break, use
  `compute_planar_turn_angles(times, positions)`
- Uses vectorized NumPy operations for efficiency

**Applications**:
- **Path straightness**: Low turn variance indicates directed movement
- **Circling behavior**: Consistent positive or negative angles
- **Random search**: High turn variance, zero mean

**Reference**: Traja package (behavioral trajectory analysis)

---

### Step Lengths

Step lengths are the distances between consecutive positions. Supports both Euclidean (straight-line) and geodesic (shortest path inside the environment) distances.

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
# _msd_geo = mean_square_displacement(times, positions, metric="geodesic", env=env, max_tau=10.0)

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

A sample outside the environment (bin `-1`) is missing: it is skipped as if
its row were deleted, and the gap around it is bridged when the valid samples on
either side are no more than `max_gap` apart. Only a longer gap, a declared break
(`break_before`, see [Joined Recordings](#joined-recordings)) or an epoch edge
breaks observation continuity; none of them fabricates an exit or an entry. Valid bins outside all selected regions are observed travel. Singleton
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

# Detect runs from start to goal (bin indices, from env.bin_at(positions))
runs = detect_runs_between_regions(
    env.bin_at(positions),
    times,
    env,
    source="start",
    target="goal",
    min_duration=1.0,
    max_duration=10.0,
    min_speed=5.0,  # optional: drop runs whose mean speed is below 5 cm/s
)

# Success rate among runs whose result was observed: a censored run was
# cut off by a tracking gap, epoch edge or the end of the recording.
observed = [r for r in runs if r.outcome != "censored"]
successful = [r for r in observed if r.success]  # outcome == "reached"

print(f"Success rate: {len(successful) / len(observed):.1%}")
print(f"Mean duration (successful): {np.mean([r.end_time - r.start_time for r in successful]):.2f}s")
```

**Returns**: List of `Run` objects
- `start_time`: First sample outside the source region
- `end_time`: When the run ended (see `outcome`)
- `source_region`, `target_region`: Region names
- `outcome`: `"reached"` (entered the target within `max_duration`),
  `"timeout"`, `"returned"` (went back to the source first) or `"censored"`
  (observation stopped first, so the result is unknown)
- `success`: Shorthand for `outcome == "reached"`
- `bins`: Bin sequence for the run

**Applications**:
- **T-maze alternation**: Runs from start to left/right arms
- **Goal-directed navigation**: Efficiency of paths to goal
- **Learning curves**: Improvement in success rate over trials
- **Strategy analysis**: Compare path patterns for successful vs failed runs

---

### Velocity-Based Segmentation

Find movement epochs, stretches where the smoothed speed stays above a
threshold, as an `(n_epochs, 2)` array of `[start, stop)` times.

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

# movement_epochs is an (n_epochs, 2) array of [start, stop) times in
# seconds; pass it as epochs= to restrict another analysis to movement.
total_movement_time = np.sum(movement_epochs[:, 1] - movement_epochs[:, 0])
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

### Labels to analysis epochs

Direction and trial functions such as `laps_to_direction_labels` and
`goal_pair_direction_labels` give one label per frame. `labels_to_epochs`
turns them into `[start, stop)` windows for each label, which you pass as
`epochs=` to a rate map, decoder or other analysis:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.behavior import Lap, labels_to_epochs, laps_to_direction_labels

times = np.arange(0, 10, 0.5)  # 2 Hz frames, for a short printout
laps = [Lap(0.0, 4.0, "clockwise", 1.0), Lap(6.0, 9.5, "counter-clockwise", 1.0)]
labels = laps_to_direction_labels(laps, times)  # None outside every lap
conditions = labels_to_epochs(times, labels)
print(conditions.epochs_table()[["label", "start", "stop", "duration"]])
assert conditions["clockwise"].tolist() == [[0.0, 4.0]]
assert conditions["counter-clockwise"].tolist() == [[6.0, 9.5]]
```

The time between two frames takes the label of the frame at its start. A lap
covers `[start_time, end_time)`, so its last frame, whose interval leaves the
lap, is labeled `None`, and each window is exactly its lap: 0–4 s and
6–9.5 s. Pass `valid=` to leave out frames with lost tracking.
[Place fields on outbound and inbound runs](workflows.md#place-fields-on-outbound-and-inbound-runs)
uses these windows to compare rate maps between conditions.

### Joined Recordings

Sessions are often analyzed as several recordings joined on one clock, for
example two runs on the same track with the animal carried back to the start
in between. If the clock has no pause at the join, nothing in `times` tells
the detectors that the recordings are separate: a trial that ends one
recording seems to finish at wherever the animal was put down for the next.
Mark the first sample of each later recording with `break_before`, and the
region, run, lap and trial detectors analyze each recording on its own:

<!-- docs-test: run -->
```python
import numpy as np
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior import detect_region_visits, segment_trials

# A 100 cm linear track with a home zone and a reward zone at its ends
track = np.c_[np.linspace(0.0, 100.0, 101), np.zeros(101)]
env = Environment.from_samples(track, bin_size=5.0).with_metadata(units="cm")
regions = Regions(env.regions.values())
regions.add("home", polygon=box(-1, -5, 10, 5))
regions.add("reward", polygon=box(90, -5, 101, 5))
env = env.with_regions(regions)

# Recording 1 ends halfway down the track. Recording 2 starts with the
# animal placed at the reward well. Both are on one 30 Hz clock.
x = np.r_[np.linspace(5.0, 50.0, 90), np.full(60, 95.0)]
positions = np.c_[x, np.zeros_like(x)]
times = np.arange(x.size) / 30  # seconds, no time between the recordings
new_recording = np.arange(x.size) == 90  # first sample of recording 2
position_bins = env.bin_at(positions)

trial_options = {"start_region": "home", "end_regions": ["reward"], "min_duration": 0.5}
joined = segment_trials(position_bins, times, env, **trial_options)
split = segment_trials(
    position_bins, times, env, break_before=new_recording, **trial_options
)
print(f"without break_before: {joined[0].outcome}, ends at {joined[0].end_time:.2f} s")
print(f"with break_before:    {split[0].outcome}, ends at {split[0].end_time:.2f} s")

(visit,) = detect_region_visits(
    position_bins, times, env, region_names=["reward"], break_before=new_recording
)
print(f"reward visit entry observed: {visit.entry_observed}")
assert joined[0].outcome == "reached" and split[0].outcome == "censored"
assert split[0].end_time == times[89]
assert not visit.entry_observed
```

Without `break_before` the trial is `"reached"` at 3.00 s, the moment the
animal was placed at the reward well. With it, the trial is `"censored"` at
2.97 s, the last sample of the first recording: its result is unknown, not a
failure. The reward visit that opens the
second recording has no observed entry: the animal was already there when the
recording started. A declared break has the same effect as a gap in the clock
longer than `max_gap`; it does not need the clock to pause.

Build `break_before` from your recording or epoch IDs, for example
`np.r_[False, recording_id[1:] != recording_id[:-1]]`, and pass the same array
to every call. The region, run, velocity, lap and trial detectors on this page
take it, as do `compute_home_range`, the goal-directed metrics,
`heading_direction_labels`, `heading_from_body_orientation`,
`Environment.occupancy`, `Environment.bin_sequence`,
`Environment.bin_sequence_with_runs` and `Environment.transitions`. Passing each
recording's `[start, stop)` as `epochs` does not do this: windows that touch
merge into one.

---

## Lap Detection

Lap detection identifies repeating circular paths (e.g., circular track running).

### Auto Template Detection

Uses the first complete circuit of the recording as the template lap.

```python
from neurospatial.behavior.segmentation import detect_laps

# Auto-detect laps; the first circuit is the template and the first lap
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
1. Template: from the first sample to the first return to its bin after a
   trip to the far side of the track (a bin at least half the greatest path
   distance away)
2. Candidate lap boundaries: each pass through the template's first bin (or
   a neighbor), at the sample nearest that bin
3. A lap ends at the first boundary where the bins visited since an earlier
   boundary overlap the template's bins (both widened by one ring of
   neighbors) with Jaccard index >= `min_overlap`, and starts at the latest
   such boundary; laps run at any speed are found whole
4. An aborted run that turns back through the start line belongs to no lap:
   the next lap starts where it returned. Without turn backs, laps tile the
   recording as `[boundary, next boundary)`

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

A lap is a trip out of a start region, round the loop and back into it. Draw
the start region across the track: it cuts the loop into a stretch on each side.
A trip counts as a lap when it leaves on one side and comes back on the other,
which means it passed the far side of the loop. A trip that comes back the way
it went (an aborted run, however far it got) is not a lap, and its time belongs
to no lap; tracking jitter across the region's edge does not split a lap. Each
lap runs from the entry before its trip to the entry that ends it; if the
recording starts inside the start region, the first lap starts at the first
sample. Region laps have `overlap_score = NaN` (no template). The region method
needs a loop of connected bins: in an open arena use `method="auto"` or
`"reference"`, and on a back-and-forth track use `segment_trials`.

```python
from neurospatial import Regions

# Add start region (drawn across the track)
from shapely.geometry import Point
region_builder = Regions(env.regions.values())
region_builder.add("start", polygon=Point(50, 50).buffer(5.0))
env = env.with_regions(region_builder)

# Detect laps as trips round the loop between start-region entries
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

**Algorithm**: Shoelace formula for the signed area of the lap path closed
back to its start, so the answer does not depend on the coordinate origin or
on where the lap starts
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

# Analyze choices and outcomes
left_trials = [t for t in trials if t.end_region == "left"]
right_trials = [t for t in trials if t.end_region == "right"]

print(f"Left choices: {len(left_trials)}")
print(f"Right choices: {len(right_trials)}")
print(f"Timeouts: {sum(t.outcome == 'timeout' for t in trials)}")
print(f"Censored (result unknown): {sum(t.outcome == 'censored' for t in trials)}")
```

**Returns**: List of `Trial` objects
- `start_time`: Departure from the start region (the last sample inside it)
- `end_time`: When the trial ended (see `outcome`)
- `end_region`: Name of the end region reached (None unless `outcome == "reached"`)
- `outcome`: `"reached"`, `"timeout"`, `"returned"` (went back to the start
  region first) or `"censored"` (observation stopped first: a tracking gap
  longer than `max_gap`, an epochs edge, a declared break or the end of the
  recording). Leave censored trials out of fraction-correct measures.
- `success`: Shorthand for `outcome == "reached"`

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
arms = [t.end_region for t in trials if t.success]
alternations = sum(1 for i in range(len(arms)-1) if arms[i] != arms[i+1])
alternation_rate = alternations / (len(arms) - 1)

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

where $A$ and $B$ are the visited bins and $|\cdot|$ is their area
(`env.bin_sizes`), so bins of different sizes count by the space they cover.

**Range**: [0, 1] where 1.0 = identical, 0.0 = disjoint

**Applications**:
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

**Computes**: Pearson correlation between the positions along the track of
the visited bins. It needs a 1-D environment or a linearized track; for a 2-D
path, use DTW or Hausdorff below.

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

# goal_runs is an (n_runs, 2) array of [start, stop) times in seconds
for start, stop in goal_runs:
    print(f"Goal-directed run: {start:.1f}-{stop:.1f} s ({stop - start:.1f} s)")
```

**Directedness Formula**:

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
left_trials = [t for t in trials if t.end_region == "left"]
right_trials = [t for t in trials if t.end_region == "right"]

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
# Compute MSD in sliding 20 s windows using continuous positions. Each window
# must be longer than max_tau, the longest lag the MSD is measured at.
dt = np.median(np.diff(times))
window_size = int(round(20.0 / dt))  # samples in 20 s
hop_size = max(1, window_size // 5)
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

if transition_idx is not None:
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

- **Jaccard**: Spatial overlap, order-independent → Territory overlap, route coverage
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

**Trial Segmentation**:
- Olton, D. S., & Samuelson, R. J. (1976). "Remembrance of places passed: Spatial memory in rats." *Journal of Experimental Psychology: Animal Behavior Processes*, 2(2), 97-116.
- Wood, E. R., et al. (2000). "Hippocampal neurons encode information about different types of memory episodes occurring in the same location." *Neuron*, 27(3), 623-633.

---

## See Also

- [Neuroscience Metrics](neuroscience-metrics.md): Place field analysis, boundary cells
- [Differential Operators](differential-operators.md): Gradient, divergence for flow analysis
- [Signal Processing Primitives](signal-processing-primitives.md): Spatial filtering operations
- [Spike Field Primitives](spike-field-primitives.md): Converting spike trains to firing rate maps
