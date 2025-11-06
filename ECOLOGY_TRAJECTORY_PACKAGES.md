# Animal Movement & Behavior Packages: Ecology & Ethology Focus

**Date**: 2025-11-06
**Domain**: Movement ecology, behavioral ecology, trajectory analysis
**Languages**: Python (Traja, yupi, PyRAT) + R (ctmm, moveHMM, adehabitatHR, amt)

---

## Overview

This analysis covers packages from **movement ecology** and **behavioral analysis** - different from the neuroscience packages we've analyzed. These focus on:
- GPS tracking and migration
- Home range estimation
- Behavioral state classification
- Trajectory statistics
- Movement models (diffusion, Lévy walks, etc.)

**Key difference from neuroscience packages**:
- Neuroscience: Place cells, grid cells, replay, spatial discretization
- Ecology: Home ranges, migration patterns, foraging behavior, habitat selection

---

## Python Packages

### 1. Traja ⭐⭐⭐

**GitHub**: https://github.com/traja-team/traja
**Stars**: 114, **Forks**: 25
**Last updated**: October 23, 2025
**License**: MIT

**Purpose**: "Python tools for spatial trajectory and time-series data analysis"

#### Core Capabilities

**Data structure**: Extends pandas DataFrame for trajectories
```python
import traja

# Load trajectory
df = traja.read_file('trajectory.csv')
# TrajaDataFrame with columns: x, y, time
```

**Trajectory analysis**:
```python
# Derivatives
df = traja.calc_derivatives(df)  # dx, dy, speed, acceleration

# Turn angles
df = traja.calc_turn_angles(df)  # Angle relative to x-axis

# Resample to regular time intervals
df = traja.resample_time(df, step_time='1s')

# Rediscretize to regular step lengths
df = traja.rediscretize_points(df, R=1.0)  # 1 unit steps
```

**Grid-based analysis**:
```python
# Discretize into grid bins
df = traja.grid_coordinates(df, bins=(50, 50))

# Transition matrix (first-order Markov)
transitions = traja.transitions(df)
# Returns: n_bins × n_bins transition probability matrix
```

**Deep learning integration**:
```python
# Data augmentation
augmented = traja.augment(df,
    rotation=True,
    noise=0.1,
    scaling=0.9-1.1,
)

# Convert to PyTorch tensors
tensors = traja.to_tensors(df)
```

**Visualization**:
```python
# Interactive plots
traja.plot(df, interactive=True)

# Heatmap
traja.plot_heatmap(df, bins=50)

# Quiver plot (flow field)
traja.plot_quiver(df, bins=20)
```

#### What It Provides

✅ Trajectory statistics (turn angles, step lengths)
✅ Grid discretization (bins)
✅ Transition matrices (Markov models)
✅ Resampling and regularization
✅ Deep learning support (PyTorch)
✅ Interactive visualization

#### What It Lacks

❌ No spatial metrics (Skaggs info, coherence)
❌ No place field detection
❌ No behavioral segmentation (runs, laps)
❌ No irregular graphs (regular grids only)
❌ Basic grid binning (no connectivity graph)

#### Comparison with neurospatial

| Feature | Traja | neurospatial |
|---------|-------|--------------|
| **Grid discretization** | ✅ Regular grids | ✅ Any topology |
| **Transition matrices** | ✅ Markov models | ✅ `env.transitions()` |
| **Turn angles** | ✅ | ❌ |
| **Step length analysis** | ✅ | ❌ |
| **Graph connectivity** | ❌ | ✅ |
| **Behavioral segmentation** | ❌ | ✅ Planned |
| **Spatial metrics** | ❌ | ✅ Planned |

**Complementary**: Traja for trajectory statistics, neurospatial for spatial neuroscience.

---

### 2. yupi ⭐⭐

**GitHub**: https://github.com/yupidevs/yupi
**Purpose**: "Generation, Tracking and Analysis of Trajectory data in Python"
**Published**: SoftwareX, 2023

#### Core Capabilities

**Three main modules**:

1. **Trajectory Generation** (synthetic data)
   ```python
   from yupi import LangevinGenerator

   # Generate trajectories from Langevin equation
   gen = LangevinGenerator(
       dt=0.01,
       T=100,
       dim=2,
       noise_scale=0.1,
   )
   trajectory = gen.generate()
   ```

2. **Trajectory Tracking** (from video)
   ```python
   from yupi import TrackingScenario

   # Track objects in video
   scenario = TrackingScenario.from_video(
       'video.mp4',
       color_range=(hue_min, hue_max),
   )
   trajectories = scenario.track()
   ```

3. **Trajectory Analysis**
   ```python
   # Speed distribution
   speeds = trajectory.v  # Velocity array

   # Turning angles
   angles = trajectory.turning_angles

   # Mean Square Displacement (MSD)
   msd = trajectory.msd()  # Indicates diffusion type

   # Kurtosis (tail heaviness)
   kurt = trajectory.kurtosis()
   ```

#### Analysis Methods

**Mean Square Displacement (MSD)**:
```python
msd = trajectory.msd()
# MSD ~ t^α
# α = 1: normal diffusion
# α < 1: subdiffusion
# α > 1: superdiffusion (Lévy walk)
```

**Applications**:
- Fluid pollution
- Animal migrations
- Environmental modeling
- N-dimensional trajectories

#### What It Provides

✅ Trajectory generation (simulations)
✅ Video tracking
✅ MSD analysis (diffusion classification)
✅ N-dimensional trajectories
✅ Statistical analysis (speed, angles, kurtosis)

#### What It Lacks

❌ No spatial discretization
❌ No behavioral states
❌ No neural integration
❌ Basic analysis (no advanced spatial metrics)

**Focus**: General-purpose trajectory analysis (physics, ecology, etc.)

---

### 3. PyRAT ⭐

**GitHub**: https://github.com/pyratlib/pyrat
**Published**: Frontiers in Neuroscience, 2022
**Purpose**: "User friendly library to analyze data from DeepLabCut"

#### Core Capabilities

**Behavior classification**:
```python
import pyratlib as rat

# Load DLC data
df = rat.load_dlc('tracking.h5')

# Classify behaviors (unsupervised clustering)
behaviors = rat.classify_behaviors(
    df,
    method='hierarchical',  # t-SNE + agglomerative clustering
    n_clusters=5,
)
```

**Motion metrics**:
```python
# Speed, acceleration, distance
metrics = rat.MotionMetrics(df)
# Returns: speed, acceleration, traveled_distance
```

**Area occupancy**:
```python
# Time spent in regions
occupancy = rat.area_occupancy(df, regions=roi_dict)
```

**Neural integration**:
```python
# Synchronize neural data with behavior
synchronized = rat.sync_neural_behavior(
    neural_data=spikes,
    behavior_data=df,
)
```

**Batch processing**:
```python
# Process multiple videos
results = rat.batch_process(
    video_files=['video1.mp4', 'video2.mp4'],
    config='config.yaml',
)
```

#### What It Provides

✅ Behavior classification (unsupervised)
✅ DLC integration
✅ Motion metrics (speed, acceleration)
✅ Area occupancy
✅ Neural synchronization
✅ User-friendly (non-programmers)

#### What It Lacks

❌ No spatial discretization (graphs)
❌ No place fields
❌ No behavioral segmentation (runs, laps)
❌ Basic analysis only

**Focus**: Accessible tool for DLC users, not comprehensive spatial analysis.

---

## R Packages (Movement Ecology)

### Major Review: "Navigating through the R packages for movement"

**Paper**: Joo et al., 2020, Journal of Animal Ecology
**Scope**: Review of **58 R packages** for animal movement analysis
**Domain**: GPS tracking, migration, home ranges, habitat selection

#### The "Big Four" Packages

### 1. ctmm (Continuous-Time Movement Modeling) ⭐⭐⭐

**Purpose**: Analyze animal relocation data as continuous-time stochastic process

**Capabilities**:
- **Continuous-time models** (not discrete time bins)
- **Home range estimation** (autocorrelation-aware)
- **Trajectory simulation**
- **Diffusion models** (Ornstein-Uhlenbeck, Brownian motion)

**Key feature**: Accounts for temporal autocorrelation in GPS data

**User satisfaction**: 80% rate documentation as good/excellent

### 2. moveHMM (Hidden Markov Models) ⭐⭐⭐

**Purpose**: Statistical modeling of behavioral states using HMMs

**Capabilities**:
```r
library(moveHMM)

# Fit HMM with 2 states (e.g., foraging vs traveling)
hmm <- fitHMM(
  data = tracking_data,
  nbStates = 2,
  stepDist = "gamma",  # Step length distribution
  angleDist = "vm",     # Turn angle distribution (von Mises)
)

# Decode most likely state sequence
states <- viterbi(hmm)
```

**Applications**:
- Behavioral state classification
- Foraging vs. traveling
- Migration detection

**User satisfaction**: 89.5% rate documentation as good/excellent

### 3. adehabitatHR (Home Range Estimation) ⭐⭐⭐

**Purpose**: Estimate home ranges and spatial utilization

**Methods**:
- Minimum Convex Polygon (MCP)
- Kernel Density Estimation (KDE)
- Brownian Bridge Movement Model (BBMM)
- Local Convex Hull (LoCoH)

**Key feature**: Industry standard for home range analysis

**User satisfaction**: 83.2% rate as "Important" or "Essential"

### 4. amt (Animal Movement Tools) ⭐⭐

**Purpose**: Modern tidyverse-compatible movement analysis

**Capabilities**:
- Habitat selection analysis
- Step selection functions (SSF)
- Resource selection functions (RSF)
- Home range estimation
- Integration with modern R tools

**Key feature**: Tidyverse-friendly (like animovement)

---

## What These Packages Provide (That Neurospatial Doesn't)

### 1. Trajectory Statistics ⭐⭐⭐

**Turn angles**:
```python
# Traja
angles = traja.calc_turn_angles(df)
# Distribution of turning behavior
```

**Step lengths**:
```python
# Analyze step-by-step movement
steps = traja.step_lengths(df)
# Lévy walk vs. Brownian motion
```

**Why relevant**: Could add to neurospatial metrics
- Useful for trajectory analysis
- Complement spatial metrics

**Recommendation**: ⚠️ Consider adding in Phase 4 (metrics)
```python
# neurospatial.metrics.trajectory
def compute_turn_angles(trajectory_bins, env):
    """Compute turn angles on graph."""
    # Use graph edges to compute angles
```

### 2. Mean Square Displacement (MSD) ⭐⭐

**What it is**:
```
MSD(t) = <|r(t) - r(0)|²>
```

**Classification**:
- MSD ~ t^α
- α = 1: Normal diffusion (random walk)
- α < 1: Subdiffusion (constrained)
- α > 1: Superdiffusion (Lévy flight, ballistic)

**Why relevant**: Characterize exploration patterns
- Is animal doing random walk?
- Is movement ballistic (goal-directed)?
- Is movement confined (home range)?

**Recommendation**: ⚠️ Consider for advanced metrics
```python
# neurospatial.metrics.diffusion
def mean_square_displacement(trajectory_positions, times):
    """Compute MSD to classify movement pattern."""
```

### 3. Hidden Markov Models (HMM) ⭐⭐⭐

**What moveHMM does**:
- Classify behavioral states (foraging, traveling, resting)
- Model state transitions
- Decode most likely state sequence

**Why relevant**: Behavioral segmentation
- Detect behavioral states from movement
- Complement region-based segmentation

**Relationship to neurospatial**:
```python
# moveHMM approach (R)
states = fitHMM(trajectory, nbStates=3)
# States: 1=foraging, 2=traveling, 3=resting

# neurospatial approach (spatial)
runs = detect_runs_between_regions(trajectory, env, 'nest', 'goal')
# Spatial criteria: traveled from A to B

# Could combine!
behavioral_runs = [
    run for run in runs
    if run.behavioral_state == 'traveling'  # From HMM
]
```

**Recommendation**: ⚠️ Optional integration (Phase 6+)
- HMMs are complex (separate package)
- Could integrate with pomegranate or hmmlearn
- Low priority (neurospatial focuses on spatial)

### 4. Home Range Estimation ⭐⭐

**What adehabitatHR does**:
- Kernel Density Estimation (KDE)
- 95% contour = home range
- Utilization distribution

**Neurospatial equivalent**:
```python
# Already possible with occupancy
occupancy = env.occupancy(times, positions)
occupancy_normalized = occupancy / occupancy.sum()

# 95% home range = bins where cumulative occupancy < 0.95
sorted_bins = np.argsort(occupancy_normalized)[::-1]
cumsum = np.cumsum(occupancy_normalized[sorted_bins])
home_range_bins = sorted_bins[cumsum < 0.95]
```

**Recommendation**: ✅ Add helper function (Phase 4)
```python
# neurospatial.metrics.occupancy
def compute_home_range(occupancy, threshold=0.95):
    """Compute home range (bins containing X% of time)."""
```

### 5. Continuous-Time Models ⭐

**What ctmm does**:
- Model trajectories as continuous stochastic process
- Account for temporal autocorrelation
- Ornstein-Uhlenbeck process
- Brownian motion

**Why different from neurospatial**:
- ctmm: Continuous time + space
- neurospatial: Discrete space (bins)

**Recommendation**: ❌ Out of scope
- Continuous-time modeling is complex
- neurospatial focuses on discrete bins
- Users needing this should use ctmm

---

## Comparison Matrix

| Package | Language | Domain | Grid/Bins | Transition Matrix | Behavioral States | Spatial Metrics | Neural Integration |
|---------|----------|--------|-----------|-------------------|-------------------|-----------------|-------------------|
| **Traja** | Python | Ecology | ✅ Regular | ✅ Markov | ❌ | ❌ | ❌ |
| **yupi** | Python | Physics/Ecology | ❌ | ❌ | ❌ | ❌ | ❌ |
| **PyRAT** | Python | Neuroscience | ⚠️ Basic | ❌ | ✅ Clustering | ❌ | ⚠️ Basic |
| **moveHMM** | R | Ecology | ❌ | ✅ HMM | ✅ HMM | ❌ | ❌ |
| **ctmm** | R | Ecology | ❌ | ❌ | ⚠️ Continuous | ❌ | ❌ |
| **adehabitatHR** | R | Ecology | ⚠️ KDE | ❌ | ❌ | ❌ | ❌ |
| **amt** | R | Ecology | ⚠️ Habitat | ✅ SSF | ⚠️ Habitat | ❌ | ❌ |
| **neurospatial** | Python | Neuroscience | ✅ Any graph | ✅ | ✅ Planned | ✅ Planned | ✅ Core |

---

## Key Lessons for neurospatial

### 1. Turn Angles and Step Lengths ⭐⭐⭐

**What Traja provides**:
```python
angles = traja.calc_turn_angles(df)
steps = traja.step_lengths(df)
```

**For neurospatial** (on graphs):
```python
# neurospatial.metrics.trajectory
def compute_turn_angles(trajectory_bins, env):
    """
    Compute turn angles on graph.

    For each triplet (bin_i, bin_j, bin_k):
    - Vector v1 from i→j
    - Vector v2 from j→k
    - Angle = arccos(v1·v2 / |v1||v2|)
    """
    angles = []
    for i in range(len(trajectory_bins) - 2):
        b1, b2, b3 = trajectory_bins[i:i+3]

        # Get bin centers
        p1 = env.bin_centers[b1]
        p2 = env.bin_centers[b2]
        p3 = env.bin_centers[b3]

        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Angle
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)

    return np.array(angles)
```

**Recommendation**: ✅ **Add in Phase 4.5** (trajectory metrics)
- Effort: 2-3 days
- Benefit: Complement spatial metrics with trajectory statistics

### 2. Mean Square Displacement (MSD) ⭐⭐

**For neurospatial**:
```python
# neurospatial.metrics.diffusion
def mean_square_displacement(trajectory_bins, env, max_lag=100):
    """
    Compute MSD on graph.

    MSD(lag) = <d²(bin[t], bin[t+lag])>
    where d = graph distance
    """
    msd = []
    for lag in range(1, max_lag):
        squared_displacements = []
        for t in range(len(trajectory_bins) - lag):
            b1 = trajectory_bins[t]
            b2 = trajectory_bins[t + lag]

            # Graph distance
            dist = env.distance_between(b1, b2)
            squared_displacements.append(dist ** 2)

        msd.append(np.mean(squared_displacements))

    return np.array(msd)
```

**Recommendation**: ⚠️ **Consider for Phase 5** (advanced metrics)
- Less common in neuroscience
- More relevant for ecology
- Low priority

### 3. Home Range Estimation ⭐⭐

**For neurospatial**:
```python
# neurospatial.metrics.occupancy
def compute_home_range(occupancy, threshold=0.95):
    """
    Compute home range bins (containing X% of time).

    Returns bins where animal spends most time.
    """
    occ_normalized = occupancy / occupancy.sum()
    sorted_bins = np.argsort(occ_normalized)[::-1]
    cumsum = np.cumsum(occ_normalized[sorted_bins])
    home_range_bins = sorted_bins[cumsum < threshold]

    return home_range_bins
```

**Recommendation**: ✅ **Add in Phase 4** (metrics module)
- Simple to implement
- Useful for neuroscience (identify "preferred" regions)
- Effort: 1 day

### 4. Transition Matrices ⭐⭐⭐

**Traja provides**:
```python
transitions = traja.transitions(df)
```

**neurospatial ALREADY provides** ✅:
```python
T = env.transitions(times, positions)
```

**No action needed** - already implemented!

### 5. HMM Behavioral States ⭐

**moveHMM provides**:
- Hidden Markov Models for state classification
- Foraging, traveling, resting

**For neurospatial**:
- Too complex for core package
- Could integrate with existing HMM libraries (hmmlearn, pomegranate)
- Out of scope for now

**Recommendation**: ❌ **Not a priority**
- neurospatial focuses on spatial segmentation
- Behavioral state HMMs are different domain

---

## Updated Implementation Plan

### Phase 4.5: Trajectory Metrics (NEW) - 1 week

**Week 12.5: Trajectory Statistics**

```python
# src/neurospatial/metrics/trajectory.py

def compute_turn_angles(
    trajectory_bins: NDArray[np.int64],
    env: Environment,
) -> NDArray[np.float64]:
    """
    Compute turn angles from bin sequence.

    Returns array of angles (radians) for each triplet.
    """

def compute_step_lengths(
    trajectory_bins: NDArray[np.int64],
    env: Environment,
) -> NDArray[np.float64]:
    """
    Compute step lengths (graph distances between consecutive bins).
    """

def compute_home_range(
    occupancy: NDArray[np.float64],
    threshold: float = 0.95,
) -> NDArray[np.int64]:
    """
    Compute home range bins (containing X% of time).
    """

def mean_square_displacement(
    trajectory_bins: NDArray[np.int64],
    env: Environment,
    max_lag: int = 100,
) -> NDArray[np.float64]:
    """
    Compute MSD on graph for diffusion classification.
    """
```

**Effort**: 1 week (5 days)
**Risk**: Low (well-defined algorithms)

**Updated timeline**: 16 weeks → **17 weeks**

---

## Recommendations

### High Priority ✅

1. **Turn angles** - Add in Phase 4.5
   - Commonly used metric
   - Easy to implement on graphs
   - Complements spatial metrics

2. **Home range** - Add in Phase 4
   - Simple occupancy-based calculation
   - Useful for identifying preferred regions
   - 1 day effort

### Medium Priority ⚠️

3. **Step lengths** - Add in Phase 4.5
   - Graph distances between consecutive bins
   - Useful for trajectory characterization

4. **MSD** - Consider for Phase 5
   - Less common in neuroscience
   - More ecology-focused
   - Optional advanced metric

### Low Priority ❌

5. **HMM behavioral states** - Out of scope
   - Too complex for core package
   - Different domain (continuous-time models)
   - Users can integrate with hmmlearn separately

---

## Conclusion

**Animal movement packages from ecology provide valuable algorithms**:
- ✅ Turn angles (Traja)
- ✅ Step lengths (Traja)
- ✅ Home range (adehabitatHR)
- ✅ MSD (yupi)
- ⚠️ HMM states (moveHMM) - different domain

**Most relevant for neurospatial**:
1. **Turn angles** - Graph-based implementation
2. **Home range** - Occupancy-based calculation
3. **Step lengths** - Graph distances

**Not relevant**:
- Continuous-time models (ctmm) - neurospatial is discrete
- GPS-specific algorithms - different scale
- Migration/habitat selection - different domain

**Updated plan**:
- Add Phase 4.5: Trajectory metrics (1 week)
- Total: 16 weeks → **17 weeks**

**Package count**: 16 + 8 (ecology packages) = **24 packages analyzed** ✅

---

**Package comparison summary** (24 packages analyzed):

| Domain | Packages | Key Focus |
|--------|----------|-----------|
| **Neuroscience** | opexebo, buzcode, pynapple, neurocode, vandermeerlab, nelpy, neurospatial | Place cells, grid cells, replay |
| **Tracking** | movement, animovement | DLC/SLEAP data cleaning |
| **Trajectory** | Traja, yupi, PyRAT | Turn angles, MSD, clustering |
| **Ecology (R)** | ctmm, moveHMM, adehabitatHR, amt | GPS, home ranges, HMM states |
| **Infrastructure** | SpikeInterface, Elephant, NetworkX, RatInABox, track_linearization | Preprocessing, simulation |
| **User's** | replay_trajectory_classification, ripple_detection | Complete workflow |

**Ecosystem survey: COMPLETE** ✅
