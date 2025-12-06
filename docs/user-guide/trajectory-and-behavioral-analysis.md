# Trajectory Metrics & Behavioral Segmentation

This guide covers trajectory characterization metrics and automatic behavioral epoch detection using the `neurospatial.behavior.trajectory` and `neurospatial.segmentation` modules.

> **Note**: Trajectory functions have been moved to `neurospatial.behavior.trajectory`. Old imports from `neurospatial.metrics.trajectory` still work for backward compatibility.

## Overview

Neurospatial provides a comprehensive suite of tools for analyzing animal trajectories and automatically detecting behavioral epochs:

**Trajectory Metrics**:
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
   - [Turn Angles](#turn-angles)
   - [Step Lengths](#step-lengths)
   - [Home Range](#home-range)
   - [Mean Square Displacement](#mean-square-displacement)
2. [Region-Based Segmentation](#region-based-segmentation)
   - [Detecting Region Crossings](#detecting-region-crossings)
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
- Stationary periods (consecutive duplicate positions) are filtered out
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
step_lengths = compute_step_lengths(positions, distance_type="euclidean")

# OR: Geodesic distance for constrained environments (requires env)
# step_lengths_geo = compute_step_lengths(positions, distance_type="geodesic", env=env)

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

```python
from neurospatial.behavior.trajectory import compute_home_range
from neurospatial import Environment

# Create environment and map positions to bins
env = Environment.from_samples(positions, bin_size=5.0)
trajectory_bins = env.bin_at(positions)

# Compute core territory (95th percentile)
home_range_bins = compute_home_range(trajectory_bins, percentile=95.0)

# Compute home range area
home_range_area = len(home_range_bins) * env.bin_area

print(f"Home range: {home_range_area:.1f} cm²")
print(f"Core area (50%): {len(compute_home_range(trajectory_bins, percentile=50.0))}")
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
tau_values, msd_values = mean_square_displacement(
    positions, times, distance_type="euclidean", max_tau=10.0
)

# OR: Geodesic distance for constrained environments
# tau_geo, msd_geo = mean_square_displacement(
#     positions, times, distance_type="geodesic", env=env, max_tau=10.0
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
- Returns (tau_values, msd_values) for plotting

**Applications**:
- **Movement classification**: Random vs directed search
- **Learning dynamics**: Transition from exploration (α≈1) to exploitation (α>1)
- **Spatial memory**: Decreased diffusion after learning

**Reference**: Sims et al. (2014), yupi package

---

## Region-Based Segmentation

### Detecting Region Crossings

Detect when the animal enters or exits named regions.

```python
from neurospatial.segmentation import detect_region_crossings

# Detect all crossings (entries and exits)
crossings = detect_region_crossings(
    trajectory_bins, times, region_name="goal", env=env, direction="both"
)

# Filter by direction
entries = detect_region_crossings(
    trajectory_bins, times, "goal", env, direction="entry"
)
exits = detect_region_crossings(
    trajectory_bins, times, "goal", env, direction="exit"
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

### Detecting Runs Between Regions

Detect runs from a source region to a target region, tracking success and failure.

```python
from neurospatial.segmentation import detect_runs_between_regions

# Detect runs from start to goal
runs = detect_runs_between_regions(
    trajectory_positions,
    times,
    env,
    source="start",
    target="goal",
    min_duration=1.0,
    max_duration=10.0,
    velocity_threshold=5.0,  # Optional speed filter
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
from neurospatial.segmentation import segment_by_velocity

# Segment into movement epochs
movement_epochs = segment_by_velocity(
    trajectory_positions,
    times,
    threshold=10.0,  # cm/s
    min_duration=0.5,  # seconds
    hysteresis=2.0,  # ratio for exit threshold
    smooth_window=0.2,  # seconds
)

# Analyze movement statistics
total_movement_time = sum(end - start for start, end in movement_epochs)
print(f"Movement time: {total_movement_time:.1f}s ({total_movement_time/times[-1]:.1%})")
```

**Hysteresis Thresholding**:
- Enter movement: velocity > threshold
- Exit movement: velocity < threshold / hysteresis
- Prevents rapid flickering between states

**Key Parameters**:
- `threshold`: Velocity threshold (same units as positions/time)
- `min_duration`: Filter brief epochs
- `hysteresis`: Exit threshold ratio (default 2.0)
- `smooth_window`: Velocity smoothing window (seconds)

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
from neurospatial.segmentation import detect_laps

# Auto-detect laps using first 10% as template
laps = detect_laps(
    trajectory_bins,
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
    trajectory_bins,
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
# Add start region (crossing point on track)
from shapely.geometry import Point
env.regions.add("start", polygon=Point(50, 50).buffer(5.0))

# Detect laps as inter-crossing intervals
laps = detect_laps(
    trajectory_bins,
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
    trajectory_bins, times, env,
    method="auto",
    direction="counter-clockwise"
)

# Only clockwise laps
cw_laps = detect_laps(
    trajectory_bins, times, env,
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
from neurospatial.segmentation import segment_trials

# Define regions
env.regions.add("start", polygon=Point(50, 10).buffer(8))
env.regions.add("left", polygon=Point(20, 90).buffer(8))
env.regions.add("right", polygon=Point(80, 90).buffer(8))

# Segment into trials
trials = segment_trials(
    trajectory_bins,
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
- `start_time`: Start region entry time
- `end_time`: End region entry or timeout
- `outcome`: Name of end region reached (or None if timeout)
- `success`: True if end region reached, False if timeout

---

### Y-maze Trials

Y-maze spontaneous alternation task.

```python
# Define Y-maze arms
env.regions.add("center", polygon=Point(50, 50).buffer(10))
env.regions.add("arm_a", polygon=Point(50, 90).buffer(8))
env.regions.add("arm_b", polygon=Point(20, 30).buffer(8))
env.regions.add("arm_c", polygon=Point(80, 30).buffer(8))

# Segment into arm visits
trials = segment_trials(
    trajectory_bins,
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
# Define center and 8 arms
env.regions.add("center", polygon=Point(50, 50).buffer(10))
for i in range(8):
    angle = i * 2 * np.pi / 8
    x = 50 + 40 * np.cos(angle)
    y = 50 + 40 * np.sin(angle)
    env.regions.add(f"arm_{i}", polygon=Point(x, y).buffer(8))

# Segment into arm choices
trials = segment_trials(
    trajectory_bins,
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
from neurospatial.segmentation import trajectory_similarity

# Compare two trajectories
similarity = trajectory_similarity(
    trajectory1_bins,
    trajectory2_bins,
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
    trajectory1_bins,
    trajectory2_bins,
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
    trajectory1_bins,
    trajectory2_bins,
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
    trajectory1_bins,
    trajectory2_bins,
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
from neurospatial.segmentation import detect_goal_directed_runs

# Detect efficient paths toward goal
goal_runs = detect_goal_directed_runs(
    trajectory_bins,
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
awake_trajectory = trajectory_bins[:1000]

# Sleep/rest epochs (detected separately)
sleep_epochs = [(1000, 1200), (1500, 1800), (2000, 2300)]

# Detect replays matching awake trajectory
replays = []
for start, end in sleep_epochs:
    sleep_segment = trajectory_bins[start:end]
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
    trajectory_bins,
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
    step_lengths = compute_step_lengths(lap_positions, distance_type="euclidean")

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
        trajectory_bins[mask1],
        trajectory_bins[mask2],
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
    trajectory_bins,
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
        trajectory_bins[mask1],
        trajectory_bins[mask2],
        env,
        method="jaccard"
    )
    left_similarities.append(sim)

print(f"Left choice stereotypy: {np.mean(left_similarities):.3f}")

# Step 4: Detect goal-directed behavior
for trial in trials[:5]:  # First 5 trials
    if trial.end_region:
        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_bins = trajectory_bins[mask]
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
    tau_vals, msd_vals = mean_square_displacement(
        window_positions,
        window_times,
        distance_type="euclidean",
        max_tau=5.0
    )

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
    trajectory_bins = env.bin_at(positions)

    explore_bins = trajectory_bins[:transition_idx * hop_size]
    goal_bins = trajectory_bins[transition_idx * hop_size:]

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
