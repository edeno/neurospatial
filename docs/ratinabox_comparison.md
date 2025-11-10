# RatInABox vs. neurospatial: Comprehensive Comparison

## Repository Information

**RatInABox**: https://github.com/RatInABox-Lab/RatInABox
**Paper**: George et al. (2023), eLife - "RatInABox: A toolkit for modelling locomotion and neuronal activity in continuous environments"
**Language**: Python (~6.3k LOC)
**Purpose**: Simulation toolkit for synthetic behavioral and neural data generation

**neurospatial**: Python library for spatial discretization and analysis
**Language**: Python (~30k LOC)
**Purpose**: Discretize continuous N-D environments with graph connectivity for neuroscience analysis

---

## Executive Summary

**RatInABox** and **neurospatial** are **complementary tools** that address different parts of the spatial neuroscience workflow:

- **RatInABox**: **Generative** - Simulates agent movement and neural activity (forward model)
- **neurospatial**: **Analytical** - Analyzes recorded data and extracts metrics (inverse model)

**Key insight**: They solve opposite problems:
- RatInABox: "Given an environment, generate realistic trajectories and neural data"
- neurospatial: "Given recorded data, discretize space and compute spatial metrics"

---

## Core Philosophy Differences

### RatInABox: Continuous Space Simulation

```python
# RatInABox operates in CONTINUOUS space
Env = Environment(params={'scale': 1.0, 'aspect': 1.5})
Ag = Agent(Env)
PCs = PlaceCells(Ag, params={'n': 100})

# Agent moves smoothly through continuous coordinates
for _ in range(1000):
    Ag.update(dt=0.01)  # Updates position as float coordinates
    PCs.update()         # Calculates firing rates on-the-fly

# NO discretization - everything is continuous
```

**Key features**:
- Position is continuous (floats)
- Firing rates computed analytically from position
- No binning or discretization
- Forward model: environment → trajectory → neural activity

### neurospatial: Discretized Space Analysis

```python
# neurospatial operates in DISCRETIZED space
env = Environment.from_samples(positions, bin_size=2.0)

# Space is divided into bins/nodes with graph connectivity
bin_indices = env.bin_at(positions)  # Maps continuous → discrete
firing_rate = spikes_to_field(env, spike_times, times, positions)

# Discretized spatial analysis
fields = detect_place_fields(firing_rate, env)
info = skaggs_information(firing_rate, occupancy)
```

**Key features**:
- Position is discretized (bin indices)
- Firing rates estimated from data (occupancy-normalized)
- Graph-based connectivity between bins
- Inverse model: recorded data → spatial discretization → metrics

---

## What RatInABox Has (neurospatial doesn't)

### 1. Agent Movement Simulation ⭐⭐⭐

**RatInABox's killer feature**: Biologically realistic agent movement

```python
# Ornstein-Uhlenbeck process for smooth random motion
Agent.speed_mean = 0.08  # m/s (fitted to Sargolini et al. 2006 data)
Agent.speed_coherence_time = 0.7  # seconds
Agent.rotational_velocity_std = 120 * (np.pi / 180)  # degrees/s

# Agent explores environment smoothly
for t in range(10000):  # 60 seconds at dt=0.01
    Ag.update(dt=0.01)
```

**What this enables**:
- Generate synthetic trajectories for testing
- Reinforcement learning (policy control)
- Simulate experiments without animal data
- Test decoding algorithms with ground truth

**neurospatial equivalent**: ❌ None - assumes you have recorded trajectory data

---

### 2. Neural Activity Generation ⭐⭐⭐

**Built-in neural models**:

```python
# Place cells
PCs = PlaceCells(Ag, params={'n': 100, 'widths': 0.2})

# Grid cells
GCs = GridCells(Ag, params={'n': 50, 'gridscale': 0.3})

# Boundary vector cells (egocentric or allocentric)
BVCs = BoundaryVectorCells(Ag, params={'n': 100})

# Head direction cells
HDCs = HeadDirectionCells(Ag, params={'n': 60})

# Velocity/speed cells
VCs = VelocityCells(Ag, params={'n': 50})
SpCs = SpeedCells(Ag, params={'n': 20})

# Object vector cells
OVCs = ObjectVectorCells(Ag, params={'n': 80})

# Field of view neurons
FoVBVCs = FieldOfViewBVCs(Ag, params={'n': 100})

# Successor features (for RL)
SFs = SuccessorFeatures(Ag, params={'n': 200})
```

**Firing rate calculation** (continuous):
```python
# Place cell example (simplified)
def get_state(self, evaluate_at='agent', **kwargs):
    if evaluate_at == 'agent':
        pos = self.Agent.pos
    else:
        pos = kwargs['pos']

    # Gaussian place fields
    distances = np.linalg.norm(pos - self.place_cell_centres, axis=1)
    firing_rates = self.max_fr * np.exp(-(distances**2) / (2 * self.widths**2))

    return firing_rates
```

**neurospatial equivalent**: ❌ None - analyzes recorded neural data, doesn't generate it

---

### 3. Reinforcement Learning Support ⭐⭐

**Policy control**:
```python
# Manual control of agent
Ag.velocity = np.array([vx, vy])  # Set velocity directly

# Reinforcement learning integration
from ratinabox.contribs import TaskEnvironment

# Create RL task environment (OpenAI Gym-compatible)
task_env = TaskEnvironment(Env, params={
    'reward_locations': [[0.5, 0.5]],
    'reward_radius': 0.1
})

# Use with RL algorithms (Stable-Baselines3, RLlib, etc.)
```

**Value neurons and successor features**:
```python
# Value function approximation
VN = ValueNeuron(Ag, params={'discount_factor': 0.9})

# Successor features for RL
SFs = SuccessorFeatures(Ag, params={'n': 200})
```

**neurospatial equivalent**: ⚠️ Partial - has reward.py with basic value functions, but not RL-focused

---

### 4. Continuous Space (No Discretization) ⭐⭐

**RatInABox strength**: Everything is continuous

```python
# Position is float coordinates (not bin indices)
Ag.pos  # array([0.37294810, 0.61829374])

# Firing rates computed analytically
PCs.firingrate  # array([0.234, 5.789, 0.012, ...]) - computed from gaussians

# No discretization overhead
# Simulating 60s with 100 place cells: ~2 seconds on laptop
```

**Benefits**:
- No discretization artifacts
- Smooth trajectories
- Fast simulation (no grid lookups)
- Float precision

**neurospatial approach**: Discretized by design (bins/nodes with graph connectivity)

---

### 5. Spiking Neurons ⭐

**Poisson spiking**:
```python
# Convert rates to spikes
PCs.update()  # Updates firing rates
spikes = np.random.poisson(PCs.firingrate * dt)  # Poisson spikes

# Or use built-in spike generation
PCs.history['spikes']  # Automatically tracked
```

**neurospatial equivalent**: ❌ None - works with spike times, not generating them

---

### 6. Real-Time Animation ⭐

**Built-in visualization**:
```python
# Animate agent movement
Ag.animate_trajectory()

# Animate firing rates
PCs.plot_rate_timeseries(spikes=True)

# Animate heat maps
Ag.plot_position_heatmap(animate=True)
```

**neurospatial equivalent**: ⚠️ Partial - has `plot_field()` but no animation framework

---

## What neurospatial Has (RatInABox doesn't)

### 1. Graph-Based Spatial Discretization ⭐⭐⭐

**neurospatial's killer feature**: Graph connectivity for irregular environments

```python
# Regular grid
env = Environment.from_samples(positions, bin_size=2.0)

# Irregular graph (maze, 1D track)
env = Environment.from_graph(graph, edge_order, bin_size=0.5)

# Polygon-masked environment
env = Environment.from_polygon(polygon, bin_size=2.0)

# Image mask (binary)
env = Environment.from_image("arena.png", bin_size=5.0)

# Graph connectivity
neighbors = env.connectivity.neighbors(bin_idx)
shortest_path = env.shortest_path(bin_a, bin_b)
```

**Why this matters**:
- Handles mazes, T-mazes, complex arenas
- Geodesic distances (true path length, not Euclidean)
- 1D linearization of 2D tracks
- 3D environments

**RatInABox equivalent**: ⚠️ Partial - has walls/boundaries but no graph structure

---

### 2. Place Field Detection ⭐⭐⭐

**Automatic field detection**:
```python
# Detect place fields from data
firing_rate = spikes_to_field(env, spike_times, times, positions)
fields = detect_place_fields(firing_rate, env, threshold=0.2)

# Field geometry
for field in fields:
    size = field_size(field, env)
    centroid = field_centroid(firing_rate, field, env)
```

**RatInABox equivalent**: ❌ None - generates place cells, doesn't detect them from data

---

### 3. Spatial Information Metrics ⭐⭐⭐

**Standard neuroscience metrics** (validated against neurocode, opexebo):
```python
# Spatial information (Skaggs 1993)
info = skaggs_information(firing_rate, occupancy)  # bits/spike

# Sparsity (Skaggs 1996)
spars = sparsity(firing_rate, occupancy)

# Field stability
stability = field_stability(rate_map_1, rate_map_2)

# Border score (Solstad 2008)
score = border_score(firing_rate, env)

# Gridness score (Hafting 2005)
gridness = gridness_score(firing_rate, env)
```

**RatInABox equivalent**: ❌ None - doesn't compute these metrics from data

---

### 4. N-D Support (1D/2D/3D/Arbitrary) ⭐⭐

**Works in any dimensionality**:
```python
# 1D track
env_1d = Environment.from_graph(graph_1d, edge_order, bin_size=0.5)

# 2D arena
env_2d = Environment.from_samples(positions_2d, bin_size=2.0)

# 3D volumetric
env_3d = Environment.from_samples(positions_3d, bin_size=1.0)

# All operations (bin_at, occupancy, smooth, etc.) work in any dimension
```

**RatInABox equivalent**: ⚠️ Partial - supports 1D and 2D, but not 3D

---

### 5. Trajectory Analysis Metrics ⭐⭐

**Behavioral analysis**:
```python
# Trajectory metrics
turn_angles = compute_turn_angles(trajectory_bins, env)
step_lengths = compute_step_lengths(trajectory_bins, env)
home_range = home_range(trajectory_bins, env)
msd = mean_square_displacement(trajectory_bins, env)

# Segmentation
laps = detect_laps(trajectory_bins, env, method='graph_cycles')
trials = segment_trials(times, positions, env, trial_markers)
```

**RatInABox equivalent**: ❌ None - generates trajectories, doesn't analyze them

---

### 6. Population Metrics ⭐⭐

**Multi-cell analysis**:
```python
# Population coverage
coverage = population_coverage(all_fields, env.n_bins)

# Field density map
density = field_density_map(all_fields, env.n_bins)

# Population vector correlation
corr_matrix = population_vector_correlation(population_matrix)

# Field overlap
overlap = field_overlap(field_i, field_j)
```

**RatInABox equivalent**: ❌ None

---

### 7. Geodesic vs Euclidean Distances ⭐⭐

**Flexible distance metrics**:
```python
# Geodesic (true path length through maze)
score = border_score(firing_rate, env, distance_metric='geodesic')

# Euclidean (straight-line distance)
score = border_score(firing_rate, env, distance_metric='euclidean')

# Distance fields
distances = distance_field(env.connectivity, sources=[goal_bin])
```

**RatInABox equivalent**: ❌ Only Euclidean distances

---

### 8. Environment Transformations ⭐

**Alignment and transforms**:
```python
# Estimate transform between sessions
transform = estimate_transform(src_landmarks, dst_landmarks, kind='rigid')

# Apply to environment
aligned_env = apply_transform_to_environment(env, transform)

# Map probability distributions between environments
mapped_probs = map_probabilities_to_nearest_target_bin(
    source_env, target_env, source_probs, transform
)
```

**RatInABox equivalent**: ❌ None

---

### 9. Validation Against Literature ⭐⭐

**Rigorous validation**:
- 5 EXACT MATCHES with neurocode (< 1e-10 error)
- Validated against opexebo, Traja, yupi
- 43 validation tests
- Published algorithms (Skaggs, Solstad, Hafting)

**RatInABox equivalent**: Validated against Sargolini et al. (2006) rat locomotion data

---

## What Both Have (Overlapping Features)

| Feature | RatInABox | neurospatial |
|---------|-----------|-------------|
| **2D environments** | ✅ Square, rectangular, polygon | ✅ Grid, polygon, mask, image |
| **Walls** | ✅ Add walls dynamically | ✅ Implicit in graph structure |
| **Boundary conditions** | ✅ Solid, periodic | ✅ Implicit in layout |
| **Place cells** | ✅ Generate synthetic | ✅ Detect from data |
| **Grid cells** | ✅ Generate synthetic | ✅ Gridness score |
| **Boundary cells** | ✅ Generate synthetic (BVCs) | ✅ Border score |
| **Head direction** | ✅ Generate synthetic | ❌ Not yet (Priority 1) |
| **1D environments** | ✅ Linear tracks | ✅ Graph-based linearization |
| **Visualization** | ✅ Rate maps, trajectories | ✅ Field plots, occupancy |
| **Occupancy** | ✅ Position heatmaps | ✅ Occupancy-normalized |

---

## What Neither Has

### Missing in Both
1. **Bayesian decoder** - Population decoding (neurospatial Priority 2, RatInABox not focused on this)
2. **LFP/EEG analysis** - Theta, ripples, spindles (both out of scope)
3. **Spike sorting** - Waveform clustering (pre-processing, not analysis)
4. **Online analysis** - Real-time streaming (both are offline)

---

## Complementary Use Cases

### Use RatInABox when you need to:

1. **Generate synthetic data**
   - Test algorithms with ground truth
   - Augment real datasets
   - Simulate experiments

2. **Reinforcement learning**
   - Train RL agents in spatial tasks
   - Successor features
   - Value function approximation

3. **Test hypotheses with simulations**
   - "What if grid spacing changes?"
   - "How do BVCs respond to new walls?"
   - Forward models

4. **Education and demos**
   - Teaching spatial coding concepts
   - Interactive demonstrations
   - Quick prototyping

### Use neurospatial when you need to:

1. **Analyze recorded data**
   - Extract spatial metrics from experiments
   - Detect place fields
   - Compute spatial information

2. **Complex/irregular environments**
   - Mazes with graph structure
   - 3D volumetric recordings
   - 1D track linearization

3. **Population analysis**
   - Multi-cell coverage
   - Remapping analysis
   - Population vector correlations

4. **Rigorous statistics**
   - Validated metrics (exact matches with literature)
   - Reproducible analysis pipeline
   - Publication-ready results

---

## Potential Integration Strategies

### Strategy 1: RatInABox → neurospatial Pipeline

**Use RatInABox to generate data, neurospatial to analyze it**:

```python
# 1. Generate data with RatInABox
Env = ratinabox.Environment(params={'scale': 1.0})
Ag = ratinabox.Agent(Env)
PCs = ratinabox.PlaceCells(Ag, params={'n': 100})

for _ in range(10000):
    Ag.update(dt=0.01)
    PCs.update()

# Extract trajectory and spikes
positions = np.array(Ag.history['pos'])
times = np.array(Ag.history['t'])
spike_times = []  # Extract from PCs.history['spikes']

# 2. Analyze with neurospatial
env = neurospatial.Environment.from_samples(positions, bin_size=0.05)
firing_rate = neurospatial.spikes_to_field(env, spike_times, times, positions)
fields = neurospatial.detect_place_fields(firing_rate, env)
info = neurospatial.skaggs_information(firing_rate, env.occupancy(times, positions))
```

**Benefits**:
- Test neurospatial metrics on ground truth data (RatInABox knows true place field centers)
- Validate metric implementations
- Generate synthetic datasets for benchmarking

---

### Strategy 2: Hybrid Environment Definition

**Use neurospatial's graph structure in RatInABox simulation**:

```python
# Define complex maze in neurospatial
env = neurospatial.Environment.from_polygon(maze_polygon, bin_size=0.02)

# Extract geometry for RatInABox
walls = extract_walls_from_graph(env.connectivity)
Env = ratinabox.Environment(params={'walls': walls})

# Simulate in RatInABox
Ag = ratinabox.Agent(Env)
# ... run simulation

# Analyze results in neurospatial
env2 = neurospatial.Environment.from_samples(Ag.history['pos'], bin_size=0.05)
# ... compute metrics
```

---

### Strategy 3: Bidirectional Validation

**Cross-validate implementations**:

```python
# RatInABox: Generate place cell with known parameters
PC_center = np.array([0.5, 0.5])
PC_width = 0.2

# neurospatial: Detect the field
env = neurospatial.Environment.from_samples(positions, bin_size=0.02)
fields = neurospatial.detect_place_fields(firing_rate, env)
detected_center = neurospatial.field_centroid(firing_rate, fields[0], env)

# Validate
assert np.linalg.norm(PC_center - detected_center) < 0.05  # Within 5cm
```

---

## What neurospatial Could Adopt from RatInABox

### 🟢 High Value Additions

1. **Movement simulation for testing** (Medium effort, High value)
   ```python
   # Adopt Ornstein-Uhlenbeck random motion model
   from neurospatial.simulation import simulate_trajectory

   positions, times = simulate_trajectory(
       env, duration=60.0, dt=0.01,
       speed_mean=0.08, speed_coherence_time=0.7
   )
   ```
   - **Use case**: Test place field detection without real data
   - **Effort**: 1-2 days (port random motion model)

2. **Continuous→Discrete conversion utilities** (Low effort, High value)
   ```python
   # Helper to convert RatInABox data to neurospatial format
   from neurospatial.io import from_ratinabox

   env, spike_data = from_ratinabox(
       ratinabox_agent=Ag,
       ratinabox_neurons=PCs,
       bin_size=0.05
   )
   ```
   - **Use case**: Seamless interoperability
   - **Effort**: 1 day

3. **Forward models for validation** (Medium effort, High value)
   ```python
   # Generate synthetic place cells for testing
   from neurospatial.models import PlaceCellModel

   pc_model = PlaceCellModel(env, n_cells=100, widths=0.2)
   synthetic_rates = pc_model.firing_rate(positions)
   ```
   - **Use case**: Test detection algorithms with ground truth
   - **Effort**: 2-3 days

### 🟡 Medium Value Additions

4. **Animation framework** (Medium effort, Medium value)
   - Adopt RatInABox's matplotlib animation patterns
   - `env.animate_trajectory()`, `env.animate_field()`
   - **Effort**: 2-3 days

5. **RL integration utilities** (Low priority for neurospatial)
   - neurospatial is analysis-focused, not RL-focused
   - Refer users to RatInABox for RL tasks

---

## What RatInABox Could Adopt from neurospatial

### 🟢 High Value Additions

1. **Graph-based environment representation** (High effort, High value)
   - Would enable geodesic distances in mazes
   - Better handling of 1D linearized tracks
   - **Challenge**: Conflicts with continuous philosophy
   - **Hybrid approach**: Keep continuous motion, add graph for distance calculations

2. **Spatial metrics from literature** (Medium effort, High value)
   ```python
   # Add to RatInABox analysis tools
   info = PCs.compute_spatial_information()  # Skaggs 1993
   border_scores = BVCs.compute_border_score()  # Solstad 2008
   gridness = GCs.compute_gridness()  # Hafting 2005
   ```
   - **Use case**: Validate generated data against literature
   - **Effort**: 2-3 days (port neurospatial implementations)

3. **Place field detection from generated data** (Low effort, Medium value)
   ```python
   # Detect fields in generated rate maps
   fields = PCs.detect_fields(threshold=0.2)
   ```
   - **Use case**: Analyze generated place cells
   - **Effort**: 1 day

### 🟡 Medium Value Additions

4. **N-D support (3D)** (High effort, Medium value)
   - Would enable 3D simulations
   - **Challenge**: Visualization becomes harder
   - **Priority**: Low (most neuroscience is 2D)

5. **Geodesic distance option** (Medium effort, Medium value)
   - Add graph connectivity for geodesic distance calculations
   - **Use case**: Realistic distance calculations in mazes
   - **Effort**: 3-4 days

---

## Key Design Differences

| Aspect | RatInABox | neurospatial |
|--------|-----------|-------------|
| **Philosophy** | Continuous simulation | Discrete analysis |
| **Primary use** | Data generation | Data analysis |
| **Space representation** | Float coordinates | Bin indices + graph |
| **Position** | `pos` (float array) | `bin_idx` (integer) |
| **Movement** | Simulated (OU process) | Recorded (imported) |
| **Neural data** | Generated (models) | Analyzed (recorded) |
| **Firing rates** | Continuous functions | Binned estimates |
| **Environment** | Walls + boundaries | Graph connectivity |
| **Distances** | Euclidean only | Geodesic + Euclidean |
| **Dimensionality** | 1D, 2D | 1D, 2D, 3D, N-D |
| **Speed** | Very fast (continuous) | Fast (optimized numpy) |
| **Validation** | vs. rat behavior | vs. literature metrics |

---

## Citation and Community

### RatInABox
- **Paper**: George et al. (2023), eLife
- **GitHub stars**: ~300
- **Downloads**: ~50k (PyPI)
- **Active development**: Yes (2021-present)
- **Community**: Computational neuroscience, RL

### neurospatial
- **Paper**: None yet (v0.2.0)
- **GitHub**: Active development
- **Community**: Experimental neuroscience
- **Focus**: Production-ready analysis tools

---

## Recommendation: Complementary Ecosystem

**RatInABox** and **neurospatial** should be viewed as **complementary tools** in the spatial neuroscience ecosystem:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Spatial Neuroscience Analysis Pipeline             │
│                                                     │
│  ┌──────────────┐         ┌──────────────┐        │
│  │  RatInABox   │────────▶│  neurospatial│        │
│  │  (Generate)  │         │   (Analyze)  │        │
│  └──────────────┘         └──────────────┘        │
│        │                          │                │
│        │                          │                │
│   Synthetic data            Real data              │
│   Ground truth              Experiments            │
│   Forward models            Inverse models         │
│   Continuous space          Discrete space         │
│   Agent simulation          Metric extraction      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Integration Roadmap

**Phase 1**: Interoperability (1 week)
- `neurospatial.io.from_ratinabox()` converter
- Example notebook showing RatInABox→neurospatial pipeline

**Phase 2**: Cross-validation (2 weeks)
- Use RatInABox to generate test data for neurospatial
- Validate place field detection against ground truth
- Benchmark spatial info metrics

**Phase 3**: Hybrid features (1 month)
- Add movement simulation to neurospatial (for testing)
- Add spatial metrics to RatInABox (for validation)
- Shared benchmark suite

---

## Conclusion

**RatInABox** and **neurospatial** solve opposite problems:

- **RatInABox**: "I need to generate realistic spatial data" (forward model)
- **neurospatial**: "I have recorded data, now analyze it" (inverse model)

**They are NOT competitors** - they are complementary tools that enable a complete workflow:

1. **Design experiment** (RatInABox: test with simulations)
2. **Collect data** (experimental recording)
3. **Analyze data** (neurospatial: extract metrics)
4. **Compare to model** (RatInABox: validate hypotheses)

**Recommendation**:
- Use **RatInABox** for RL, simulation, and hypothesis testing
- Use **neurospatial** for analysis of recorded data with complex environments
- **Integrate** them for validation and testing workflows

Both packages would benefit from tighter integration - the synthetic→analysis pipeline is powerful for methods development and validation.
