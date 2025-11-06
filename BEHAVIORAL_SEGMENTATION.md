# Behavioral Segmentation: Spatial Primitives

**Date**: 2025-11-06
**Status**: Proposed addition to implementation plan

---

## Overview

**Behavioral segmentation** is the process of identifying discrete behavioral epochs from continuous trajectories. Examples include:
- **Goal-directed runs**: Trajectories from nest to goal
- **Laps**: Complete loops around a track
- **Trials**: Task-specific behavioral units (T-maze left/right)
- **Movement periods**: Active exploration vs. quiescence
- **State transitions**: Exploration → exploitation, foraging → returning

**Why critical**: Most neuroscience analyses require behavioral segmentation:
- Place field analysis per trial type
- Replay detection (reactivation of specific routes)
- Learning dynamics (performance across trials)
- Goal-directed coding (trajectory-specific firing)
- Behavioral state dependence (run vs. rest, left vs. right)

---

## What Existing Packages Provide

### neurocode (MATLAB)

**Lap Detection**:
```matlab
% NSMAFindGoodLaps.m - Detect laps in circular/figure-8 tracks
laps = NSMAFindGoodLaps(position, track_geometry)

% FindLapsNSMAadapted.m - Adapted for specific track types
[start_times, end_times] = FindLapsNSMAadapted(position)
```

**Movement Periods**:
```matlab
% MovementPeriods.m - Detect active movement
movement = MovementPeriods(velocity, threshold=10)  % cm/s

% QuietPeriods.m - Detect rest/quiet periods
quiet = QuietPeriods(velocity, threshold=2)
```

**Zone/Region Operations**:
```matlab
% DefineZone.m - Define spatial ROI
zone = DefineZone(vertices)

% IsInZone.m - Check if position in zone
in_zone = IsInZone(position, zone)
```

### vandermeerlab (MATLAB)

**Task-Specific Segmentation**:
```matlab
% Task-specific directories with segmentation
tasks/
├── Alyssa_Tmaze/          # T-maze trial segmentation
├── Julien_linear_track/   # Linear track laps
├── Eric_square_maze/      # Square maze runs
└── Replay_Analysis/       # SWR event segmentation
```

**Approach**: Manual trial definition based on task structure.

### pynapple (Python)

**IntervalSet Infrastructure** ✅:
```python
# Define behavioral epochs
trials = nap.IntervalSet(start=trial_starts, end=trial_ends)

# Restrict data to epochs
spikes_trial = spikes.restrict(trials)
position_trial = position.restrict(trials)

# Get data between intervals
inter_trial = position.get_intervals_complement()
```

**Strengths**:
- ✅ **Excellent representation** - IntervalSet is perfect for behavioral epochs
- ✅ **Easy restriction** - restrict() is intuitive
- ✅ **Complement operations** - get time between trials

**Limitations**:
- ❌ **No automatic detection** - User must provide start/end times
- ❌ **No spatial criteria** - No "detect runs from A to B"
- ❌ **No trajectory analysis** - No lap detection, path matching

### neurospatial (Current)

**Regions** ✅:
```python
from neurospatial.regions import Region

# Define goal regions
env.regions.add('nest', point=[10, 10])
env.regions.add('goal', point=[90, 90])

# Check if bins in region
in_goal = env.regions.contains('goal', bin_indices)
```

**Trajectory Analysis** ✅:
```python
# Map trajectory to bins
trajectory_bins = env.bin_at(trajectory_positions)

# Compute transitions
T = env.transitions(trajectory_positions, times)
```

**Limitations**:
- ❌ **No epoch detection** - Can't automatically segment runs
- ❌ **No region-based segmentation** - Can't detect "runs from nest to goal"
- ❌ **No lap detection** - Can't identify complete loops
- ❌ **No velocity gating** - Can't separate movement from rest

---

## Proposed Primitives for Behavioral Segmentation

### 1. Region Entry/Exit Detection

**Primitive**: Identify when trajectory enters/exits spatial regions.

```python
from neurospatial.segmentation import detect_region_crossings

# Detect when trajectory enters/exits goal region
crossings = detect_region_crossings(
    trajectory_bins,
    region='goal',
    env=env,
    direction='entry'  # 'entry', 'exit', or 'both'
)

# Returns: List[Crossing]
# Crossing(time, bin_index, direction='entry'/'exit')
```

**Algorithm**:
1. Map trajectory to bins
2. Check which bins are in region (via `env.regions.contains()`)
3. Find transitions: outside→inside (entry), inside→outside (exit)
4. Return crossing times and bins

**Use case**: Identify goal arrivals, nest departures, zone entries.

---

### 2. Detect Runs Between Regions

**Primitive**: Identify trajectory segments from region A to region B.

```python
from neurospatial.segmentation import detect_runs_between_regions

# Detect runs from nest to goal
runs = detect_runs_between_regions(
    trajectory_positions,
    times,
    env=env,
    source='nest',
    target='goal',
    min_duration=0.5,      # Minimum run duration (s)
    max_duration=10.0,     # Maximum run duration (s)
    velocity_threshold=5.0, # Minimum velocity (cm/s)
)

# Returns: List[Run]
# Run(
#     start_time, end_time, duration,
#     trajectory_bins,     # Bin sequence
#     path_length,         # Geodesic distance
#     success=True/False   # Reached target?
# )
```

**Algorithm**:
1. Detect entries to source region → potential run starts
2. For each start, search forward until:
   - Reach target region (success)
   - Exceed max_duration (failure)
   - Return to source (aborted)
3. Filter by min/max duration and velocity
4. Return valid runs

**Use case**:
- Goal-directed behavior analysis
- Learning curves (success rate over time)
- Trajectory-specific place fields

---

### 3. Lap Detection

**Primitive**: Identify complete loops/laps in circular or figure-8 tracks.

```python
from neurospatial.segmentation import detect_laps

# Detect laps (automatically finds lap start/end point)
laps = detect_laps(
    trajectory_bins,
    times,
    env=env,
    method='auto',          # 'auto', 'reference', or 'region'
    min_overlap=0.8,        # Minimum spatial overlap with reference
    direction='both',       # 'clockwise', 'counter', or 'both'
)

# Returns: List[Lap]
# Lap(
#     start_time, end_time, duration,
#     trajectory_bins,
#     direction='clockwise'/'counter',
#     overlap_score=0.95   # Similarity to canonical lap
# )
```

**Algorithm** (auto method):
1. Find spatial bins visited in first 10% of trajectory → candidate lap template
2. For each subsequent segment, compute spatial overlap with template
3. When overlap < threshold, previous segment is one lap
4. Repeat with next lap as new template
5. Return all detected laps

**Algorithm** (reference method):
1. User provides reference lap trajectory
2. Compute correlation between sliding windows and reference
3. Peaks in correlation = lap completions

**Algorithm** (region method):
1. User provides lap start/end region
2. Detect region crossings
3. Segments between consecutive crossings = laps

**Use case**:
- Lap-by-lap firing rate analysis
- Within-session learning
- Theta sequences per lap

---

### 4. Movement vs Rest Segmentation

**Primitive**: Segment trajectory by velocity threshold.

```python
from neurospatial.segmentation import segment_by_velocity

# Detect movement periods
movement_epochs = segment_by_velocity(
    trajectory_positions,
    times,
    threshold=5.0,          # cm/s
    min_duration=0.5,       # Minimum epoch duration (s)
    hysteresis=2.0,         # Hysteresis range (cm/s)
    smooth_window=0.2,      # Smoothing window (s)
)

# Returns: IntervalSet (pynapple-compatible)
# or List[Epoch] if pynapple not available
```

**Algorithm**:
1. Compute velocity via finite differences
2. Smooth velocity (Gaussian window)
3. Apply hysteresis thresholding:
   - Movement starts when velocity > threshold + hysteresis
   - Movement ends when velocity < threshold - hysteresis
4. Merge short gaps (< min_duration)
5. Return epochs

**Use case**:
- Place field analysis during active movement
- Rest-period analysis (SWR detection)
- Movement-modulated cells

---

### 5. Trial Segmentation (Task-Specific)

**Primitive**: Detect trials based on task structure (e.g., T-maze left/right).

```python
from neurospatial.segmentation import segment_trials

# T-maze: Detect left vs right trials
trials = segment_trials(
    trajectory_bins,
    times,
    env=env,
    trial_type='tmaze',     # 'tmaze', 'ymaze', 'radial_arm', 'custom'
    start_region='center',  # Trial start
    end_regions={           # Trial outcomes
        'left': 'left_arm',
        'right': 'right_arm',
    },
    min_duration=1.0,
    max_duration=15.0,
)

# Returns: List[Trial]
# Trial(
#     start_time, end_time, duration,
#     trajectory_bins,
#     outcome='left'/'right'/'timeout',
#     success=True/False
# )
```

**Algorithm**:
1. Detect entries to start_region → trial starts
2. For each start, search forward until:
   - Reach an end_region (trial outcome)
   - Exceed max_duration (timeout)
3. Label trial by which end_region reached
4. Return trials with outcomes

**Use case**:
- Choice-selective place cells (left vs right)
- Decision point analysis
- Learning dynamics (correct vs error trials)

---

### 6. Path Similarity/Clustering

**Primitive**: Measure similarity between trajectory segments.

```python
from neurospatial.segmentation import trajectory_similarity

# Compare two trajectory segments
similarity = trajectory_similarity(
    trajectory1_bins,
    trajectory2_bins,
    env=env,
    method='jaccard',  # 'jaccard', 'correlation', 'hausdorff', 'dtw'
)

# Returns: float [0, 1] (1 = identical, 0 = no overlap)
```

**Methods**:

**Jaccard similarity** (spatial overlap):
```
J = |bins1 ∩ bins2| / |bins1 ∪ bins2|
```

**Correlation** (sequential similarity):
```
1. Create occupancy vectors (time in each bin)
2. Compute Pearson correlation
```

**Hausdorff distance** (maximum deviation):
```
H = max(max_dist(bins1, bins2), max_dist(bins2, bins1))
where max_dist = maximum distance from any bin in set1 to nearest bin in set2
```

**Dynamic Time Warping** (temporal alignment):
```
DTW aligns sequences allowing stretching/compression
```

**Use case**:
- Cluster trials by trajectory type
- Detect trajectory-specific replay
- Identify stereotyped vs exploratory behavior

---

### 7. Detect Goal-Directed Runs (Advanced)

**Primitive**: Identify runs where trajectory moves toward a goal.

```python
from neurospatial.segmentation import detect_goal_directed_runs

# Detect runs moving toward goal
runs = detect_goal_directed_runs(
    trajectory_bins,
    times,
    env=env,
    goal_region='goal',
    directedness_threshold=0.7,  # [0, 1] - how direct is the path?
    min_progress=20.0,           # Minimum distance traveled toward goal (cm)
)

# Returns: List[Run]
# Run(
#     start_time, end_time,
#     directedness=0.85,           # How directed toward goal?
#     progress=35.0,               # Distance moved closer to goal (cm)
#     reached_goal=True/False
# )
```

**Algorithm**:
1. Compute goal location (region center)
2. For each trajectory segment:
   - Compute distance to goal at start and end
   - Progress = distance_start - distance_end
   - Directedness = progress / path_length (0 = random walk, 1 = straight line)
3. Filter by thresholds
4. Return directed runs

**Metrics**:

**Directedness** (path efficiency):
```
directedness = (dist_start_to_goal - dist_end_to_goal) / path_length
```

**Progress** (net distance toward goal):
```
progress = dist_start_to_goal - dist_end_to_goal
```

**Use case**:
- Goal-coding place cells
- Distinguish goal-directed from random exploration
- Evaluate planning/learning

---

## Implementation Plan Addition

### Phase 2.5: Behavioral Segmentation (NEW) - 2 weeks

**Week 1: Core Primitives**

```python
# src/neurospatial/segmentation/regions.py
def detect_region_crossings(trajectory_bins, region, env, direction='both')
def detect_runs_between_regions(positions, times, env, source, target, ...)
def segment_by_velocity(positions, times, threshold, ...)
```

**Tests**:
- Region entry/exit detection
- Goal-directed runs (nest → goal)
- Velocity segmentation (movement vs rest)

**Week 2: Advanced Segmentation**

```python
# src/neurospatial/segmentation/laps.py
def detect_laps(trajectory_bins, times, env, method='auto', ...)

# src/neurospatial/segmentation/trials.py
def segment_trials(trajectory_bins, times, env, trial_type, ...)

# src/neurospatial/segmentation/similarity.py
def trajectory_similarity(traj1, traj2, env, method='jaccard')
def detect_goal_directed_runs(trajectory_bins, times, env, goal, ...)
```

**Tests**:
- Lap detection on circular track
- T-maze trial segmentation
- Trajectory similarity measures
- Goal-directedness scoring

**Integration**:
```python
# Example: Complete workflow
from neurospatial import Environment
from neurospatial.segmentation import (
    detect_runs_between_regions,
    segment_by_velocity,
)

# 1. Create environment
env = Environment.from_samples(positions, bin_size=2.0)
env.regions.add('nest', point=[10, 10])
env.regions.add('goal', point=[90, 90])

# 2. Segment by velocity (only analyze movement)
movement_epochs = segment_by_velocity(positions, times, threshold=5.0)

# 3. Detect goal-directed runs
runs = detect_runs_between_regions(
    positions[movement_epochs],  # Only during movement
    times[movement_epochs],
    env=env,
    source='nest',
    target='goal',
)

# 4. Analyze per-run place fields
for run in runs:
    run_spikes = spikes.restrict(run.start_time, run.end_time)
    tuning_curve = compute_tuning_curve(run_spikes, run.trajectory_bins)
```

---

## Comparison with Existing Packages

| Capability | neurocode | vandermeerlab | pynapple | neurospatial (proposed) |
|-----------|-----------|---------------|----------|------------------------|
| **Lap detection** | ✅ NSMAFindGoodLaps | ⚠️ Manual | ❌ | ✅ detect_laps |
| **Region crossings** | ✅ IsInZone | ⚠️ Manual | ❌ | ✅ detect_region_crossings |
| **Movement/rest** | ✅ MovementPeriods | ⚠️ Manual | ❌ | ✅ segment_by_velocity |
| **Runs between regions** | ⚠️ Manual | ⚠️ Manual | ❌ | ✅ detect_runs_between_regions |
| **Trial segmentation** | ⚠️ Task-specific | ⚠️ Task-specific | ❌ | ✅ segment_trials |
| **Trajectory similarity** | ❌ | ❌ | ❌ | ✅ trajectory_similarity |
| **Goal-directedness** | ❌ | ❌ | ❌ | ✅ detect_goal_directed_runs |
| **Epoch representation** | ✅ IntervalArray | ✅ iv | ✅ IntervalSet ⭐ | ✅ pynapple-compatible |

**Key insight**: Most packages require **manual segmentation** or have **task-specific** scripts. neurospatial can provide **general-purpose spatial segmentation primitives**.

---

## Integration with pynapple ⭐

**Strategy**: Return pynapple `IntervalSet` objects when pynapple available.

```python
# src/neurospatial/segmentation/base.py

def _to_intervalset(epochs):
    """Convert epochs to pynapple IntervalSet if available."""
    try:
        import pynapple as nap
        starts = [e.start_time for e in epochs]
        ends = [e.end_time for e in epochs]
        return nap.IntervalSet(start=starts, end=ends)
    except ImportError:
        # Return simple list of (start, end) tuples
        return [(e.start_time, e.end_time) for e in epochs]

# All segmentation functions return IntervalSet-compatible output
runs = detect_runs_between_regions(...)  # Returns IntervalSet (if pynapple) or list
```

**Benefits**:
- ✅ **Seamless integration** - neurospatial segments → pynapple restricts
- ✅ **Best of both** - Spatial segmentation + time-series infrastructure
- ✅ **Familiar API** - Users already know IntervalSet

**Example workflow**:
```python
import pynapple as nap
from neurospatial import Environment
from neurospatial.segmentation import detect_runs_between_regions

# 1. Load data (pynapple)
spikes = nap.load_file('spikes.nwb')
position = nap.load_file('position.nwb')

# 2. Segment trajectories (neurospatial)
env = Environment.from_samples(position.values, bin_size=2.0)
runs = detect_runs_between_regions(
    position.values, position.index.values,
    env, source='nest', target='goal'
)  # Returns nap.IntervalSet

# 3. Restrict data (pynapple)
spikes_runs = spikes.restrict(runs)
position_runs = position.restrict(runs)

# 4. Compute tuning curves per run
for i, run in enumerate(runs):
    spikes_run = spikes.restrict(run)
    tc = nap.compute_2d_tuning_curves(spikes_run, position_runs, bins=20)
```

---

## Validation Strategy

### 1. Unit Tests

**Test cases**:
```python
def test_detect_region_crossings():
    """Test entry/exit detection."""
    # Linear trajectory: outside → goal → outside
    trajectory = np.array([[0, 0], [5, 5], [10, 10], [15, 15], [20, 20]])
    env.regions.add('goal', point=[10, 10], radius=3.0)

    crossings = detect_region_crossings(trajectory, 'goal', env, direction='both')

    assert len(crossings) == 2  # One entry, one exit
    assert crossings[0].direction == 'entry'
    assert crossings[1].direction == 'exit'

def test_detect_runs_between_regions():
    """Test goal-directed run detection."""
    # 3 runs: nest → goal, goal → nest, nest → goal
    runs = detect_runs_between_regions(positions, times, env, 'nest', 'goal')

    assert len(runs) == 2  # Two nest → goal runs
    assert all(r.success for r in runs)  # Both reached goal

def test_lap_detection():
    """Test lap detection on circular track."""
    # Simulate 3 laps around circle
    laps = detect_laps(circular_trajectory, times, env)

    assert len(laps) == 3
    assert all(0.9 < lap.overlap_score < 1.0 for lap in laps)  # High overlap
```

### 2. Validation Against neurocode

**Compare lap detection**:
```matlab
% MATLAB (neurocode)
laps_matlab = NSMAFindGoodLaps(position, track);

% Python (neurospatial)
laps_python = detect_laps(trajectory_bins, times, env)

% Compare start/end times (should match within 1 time bin)
```

### 3. Example Datasets

**Use RatInABox to generate ground truth**:
```python
from ratinabox import Environment as RatEnv, Agent
from neurospatial.segmentation import detect_laps

# 1. Simulate laps with RatInABox
rat_env = RatEnv(params={'scale': 1.0, 'boundary': 'circular'})
agent = Agent(rat_env)
agent.update(dt=0.01, T=60)  # 60 seconds

# 2. Detect laps with neurospatial
env = Environment.from_samples(agent.history['pos'], bin_size=0.05)
laps = detect_laps(agent.history['pos'], agent.history['t'], env)

# 3. Validate: Should detect ~6-8 laps (agent runs ~10 laps/min)
assert 5 < len(laps) < 10
```

---

## Use Cases and Examples

### Example 1: Place Fields by Trial Type

```python
# Segment T-maze trials
trials = segment_trials(
    trajectory_bins, times, env,
    trial_type='tmaze',
    start_region='center',
    end_regions={'left': 'left_arm', 'right': 'right_arm'},
)

# Separate left vs right trials
left_trials = [t for t in trials if t.outcome == 'left']
right_trials = [t for t in trials if t.outcome == 'right']

# Compute place fields separately
from neurospatial.metrics import detect_place_fields

left_fields = detect_place_fields(
    firing_rate_left,  # Only during left trials
    env,
)

right_fields = detect_place_fields(
    firing_rate_right,  # Only during right trials
    env,
)

# Test for choice-selective cells
selectivity = compute_trial_type_selectivity(left_fields, right_fields)
```

### Example 2: Lap-by-Lap Learning

```python
# Detect laps
laps = detect_laps(trajectory_bins, times, env)

# Compute performance per lap
for i, lap in enumerate(laps):
    # Extract data for this lap
    lap_spikes = spikes.restrict(lap.start_time, lap.end_time)
    lap_trajectory = trajectory_bins_in_lap

    # Compute place field stability
    tuning_curve = compute_tuning_curve(lap_spikes, lap_trajectory)

    # Track across laps
    stability[i] = correlation(tuning_curve, reference_tuning_curve)

# Plot learning curve
plt.plot(stability)
plt.xlabel('Lap')
plt.ylabel('Place field stability')
```

### Example 3: Goal-Coding Analysis

```python
# Detect runs to two different goals
runs_goal1 = detect_runs_between_regions(
    positions, times, env, source='nest', target='goal1'
)

runs_goal2 = detect_runs_between_regions(
    positions, times, env, source='nest', target='goal2'
)

# Compute place fields per goal
fields_goal1 = detect_place_fields(firing_rate_goal1, env)
fields_goal2 = detect_place_fields(firing_rate_goal2, env)

# Test for goal-selective cells
# (Do fields shift location depending on which goal is target?)
goal_coding_score = spatial_correlation(fields_goal1, fields_goal2)
# Low correlation → goal-coding cell
```

### Example 4: Replay Detection

```python
# During rest periods, detect SWR events
movement_epochs = segment_by_velocity(positions, times, threshold=5.0)
rest_epochs = complement(movement_epochs)  # Time NOT moving

# Detect ripples during rest (user's ripple_detection package)
from ripple_detection import Kay_ripple_detector
ripples = Kay_ripple_detector(lfp.restrict(rest_epochs))

# For each ripple, decode trajectory
from replay_trajectory_classification import SortedSpikesClassifier

for ripple in ripples:
    # Decode spatial trajectory during ripple
    decoded = classifier.predict(spikes, ripple.times)

    # Compare to behavioral runs
    similarity_to_runs = [
        trajectory_similarity(decoded.bins, run.trajectory_bins, env)
        for run in runs_goal1
    ]

    # Is this replay of a goal-directed run?
    best_match = max(similarity_to_runs)
    if best_match > 0.8:
        print(f"Ripple at {ripple.start_time:.2f}s replays run {argmax(similarity_to_runs)}")
```

---

## Updated Implementation Plan

### Full Timeline: 15 weeks → 17 weeks

**Phase 1: Differential Operators** (3 weeks) - unchanged
**Phase 2: Signal Processing** (6 weeks) - unchanged
**Phase 2.5: Behavioral Segmentation** (2 weeks) - **NEW**
**Phase 3: Metrics Module** (5 weeks) - unchanged
**Phase 4: Polish & Validation** (1 week) - unchanged

---

## Conclusion

**Behavioral segmentation is a CRITICAL primitive** for spatial neuroscience:

1. ✅ **Complements existing capabilities** - Regions + trajectory analysis + segmentation
2. ✅ **Fills ecosystem gap** - Most packages require manual segmentation
3. ✅ **Integrates with pynapple** - Return IntervalSet objects
4. ✅ **Enables key analyses** - Trial-type selectivity, learning curves, goal coding
5. ✅ **Supports user's workflow** - Segment runs → analyze replay

**Recommendation**: Add Phase 2.5 (2 weeks) for behavioral segmentation primitives.

**Total timeline**: 17 weeks (up from 15 weeks)

**Risk**: LOW - Algorithms are well-defined, no complex mathematics, similar to existing trajectory analysis.

**Impact**: HIGH - Enables per-trial analysis, learning dynamics, goal-coding studies.
