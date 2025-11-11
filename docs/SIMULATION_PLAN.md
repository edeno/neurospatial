# Simulation Subpackage Plan

**Goal**: Create a `neurospatial.simulation` subpackage for generating synthetic spatial data, neural activity, and spike trains for testing, validation, and education.

**Inspiration**: Existing simulation code + RatInABox movement models + neurospatial's N-D architecture

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Structure](#module-structure)
3. [Trajectory Simulation](#trajectory-simulation)
4. [Neural Models](#neural-models)
5. [Spike Generation](#spike-generation)
6. [Integration with Environment](#integration-with-environment)
7. [API Examples](#api-examples)
8. [Implementation Priorities](#implementation-priorities)
9. [Testing Strategy](#testing-strategy)

---

## Architecture Overview

### Design Principles

1. **Environment-centric**: All simulations operate on `Environment` instances
2. **N-D native**: Support 1D, 2D, 3D, and arbitrary dimensions
3. **Composable**: Separate trajectory, neural model, and spike generation
4. **Validated**: Use ground truth to validate neurospatial's analysis functions
5. **Extensible**: Protocol-based design for custom neural models

### Three-Layer Design

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Trajectory Simulation                 │
│  - OU process for realistic movement            │
│  - Task-based trajectories (laps, trials)       │
│  - Position → bin_sequence mapping              │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  Layer 2: Neural Models                         │
│  - Place cells (Gaussian fields)                │
│  - Boundary cells (distance-tuned)              │
│  - Grid cells (periodic patterns)               │
│  - Head direction cells (circular tuning)       │
│  → Output: firing_rate(position, time)          │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  Layer 3: Spike Generation                      │
│  - Poisson process (inhomogeneous)              │
│  - Refractory period constraints                │
│  - Rate modulation (theta, state-dependent)     │
│  → Output: spike_times arrays                   │
└─────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/neurospatial/simulation/
├── __init__.py                    # Public API exports
├── trajectory.py                  # Trajectory generation (OU process, laps, etc.)
├── models/                        # Neural models
│   ├── __init__.py
│   ├── base.py                    # NeuralModel protocol
│   ├── place_cells.py             # Gaussian place fields
│   ├── boundary_cells.py          # Distance-tuned fields
│   ├── grid_cells.py              # Periodic grid patterns
│   ├── head_direction.py          # Circular directional tuning
│   └── speed_cells.py             # Speed-tuned neurons
├── spikes.py                      # Spike generation (Poisson, etc.)
├── utils.py                       # Helper functions
└── examples.py                    # Pre-configured examples

tests/simulation/
├── test_trajectory.py
├── test_models.py
├── test_spikes.py
└── test_integration.py            # End-to-end validation tests
```

---

## Trajectory Simulation

### 1. Ornstein-Uhlenbeck Process (from RatInABox)

**Purpose**: Generate realistic, smooth random exploration trajectories

```python
# src/neurospatial/simulation/trajectory.py

def simulate_trajectory_ou(
    env: Environment,
    duration: float,
    dt: float = 0.01,
    start_position: NDArray[np.float64] | None = None,
    speed_mean: float = 0.08,  # m/s (8 cm/s)
    speed_std: float = 0.04,
    coherence_time: float = 0.7,  # seconds
    boundary_mode: Literal["reflect", "periodic", "stop"] = "reflect",
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate trajectory using Ornstein-Uhlenbeck process.

    Generates smooth, realistic random exploration in N-D space with
    biologically plausible movement statistics (fitted to Sargolini et al. 2006).

    Parameters
    ----------
    env : Environment
        The environment to simulate within.
    duration : float
        Simulation duration in seconds.
    dt : float, optional
        Time step in seconds (default: 0.01 = 10ms).
    start_position : NDArray[np.float64], shape (n_dims,), optional
        Starting position. If None, randomly chosen from bin centers.
    speed_mean : float, optional
        Mean speed in units/second (default: 0.08 m/s = 8 cm/s).
    speed_std : float, optional
        Speed standard deviation (default: 0.04).
    coherence_time : float, optional
        Time scale for velocity autocorrelation in seconds (default: 0.7s).
        Larger values = smoother, more persistent movement.
    boundary_mode : {'reflect', 'periodic', 'stop'}, optional
        How to handle environment boundaries (default: 'reflect').
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Continuous position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.

    Notes
    -----
    The OU process implements:
        dv = -θ(v - v_mean)dt + σ dW
    where θ = 1/coherence_time controls mean reversion rate.

    Examples
    --------
    >>> env = Environment.from_samples(arena_data, bin_size=2.0)
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)
    >>> print(positions.shape)  # (6000, 2) at 100 Hz

    See Also
    --------
    simulate_trajectory_laps : Structured back-and-forth movement
    simulate_trajectory_sinusoidal : Simple periodic motion (1D)
    """
    pass
```

### 2. Structured Trajectories

**Purpose**: Generate task-specific movement patterns (laps, trials, etc.)

```python
def simulate_trajectory_sinusoidal(
    env: Environment,
    duration: float,
    sampling_frequency: float = 500.0,
    running_speed: float = 10.0,
    pause_duration: float = 0.0,
    pause_at_peaks: bool = True,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate sinusoidal trajectory (for 1D/linear tracks).

    Generalizes existing `simulate_position()` to work with 1D neurospatial
    environments (GraphLayout).

    Parameters
    ----------
    env : Environment
        Must be 1D (env.is_1d == True).
    duration : float
        Simulation duration in seconds.
    sampling_frequency : float, optional
        Samples per second (default: 500 Hz).
    running_speed : float, optional
        Period of sinusoidal motion (default: 10.0).
    pause_duration : float, optional
        Pause at track ends in seconds (default: 0.0).
    pause_at_peaks : bool, optional
        Whether to pause at track extrema (default: True).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, 1)
        1D position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.

    Examples
    --------
    >>> env_1d = Environment.from_graph(track_graph, edge_order, bin_size=0.5)
    >>> positions, times = simulate_trajectory_sinusoidal(env_1d, duration=120.0)
    """
    if not env.is_1d:
        raise ValueError("Sinusoidal trajectory only works for 1D environments")
    pass


def simulate_trajectory_laps(
    env: Environment,
    n_laps: int,
    speed_mean: float = 0.1,
    speed_std: float = 0.02,
    outbound_path: list[int] | None = None,
    inbound_path: list[int] | None = None,
    pause_duration: float = 0.5,
    sampling_frequency: float = 500.0,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_]]:
    """Simulate structured lap-based trajectory.

    Useful for alternation tasks, T-mazes, linear tracks.

    Parameters
    ----------
    env : Environment
        The environment.
    n_laps : int
        Number of complete laps (outbound + inbound).
    speed_mean : float, optional
        Mean speed in units/second.
    speed_std : float, optional
        Speed variability.
    outbound_path : list[int] | None, optional
        Sequence of bin indices for outbound trajectory.
        If None, uses shortest path from start to end of environment.
    inbound_path : list[int] | None, optional
        Sequence of bin indices for inbound trajectory.
        If None, reverses outbound_path.
    pause_duration : float, optional
        Pause at lap ends in seconds (default: 0.5s).
    sampling_frequency : float, optional
        Samples per second.
    seed : int | None, optional
        Random seed.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points.
    lap_ids : NDArray[np.int_], shape (n_time,)
        Lap number for each time point.

    Examples
    --------
    >>> # T-maze alternation
    >>> env = Environment.from_graph(tmaze_graph, edge_order, bin_size=1.0)
    >>> left_path = [0, 1, 2, 5, 8]  # Start → center → left arm
    >>> right_path = [0, 1, 2, 6, 9]  # Start → center → right arm
    >>> positions, times, laps = simulate_trajectory_laps(
    ...     env, n_laps=10, outbound_path=left_path, inbound_path=right_path
    ... )
    """
    pass
```

---

## Neural Models

### Base Protocol

```python
# src/neurospatial/simulation/models/base.py

from typing import Protocol, runtime_checkable
from numpy.typing import NDArray
import numpy as np

@runtime_checkable
class NeuralModel(Protocol):
    """Protocol for neural models that generate firing rates.

    All neural models must implement:
    - firing_rate(positions, times) → rates
    - ground_truth property (returns dict with model parameters)
    """

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute firing rate at given positions.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, n_dims)
            Position coordinates.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points (for time-varying models).

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position/time.
        """
        ...

    @property
    def ground_truth(self) -> dict:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict
            Model parameters (e.g., {'center': [50.0, 75.0], 'width': 10.0})
        """
        ...
```

### Place Cells

```python
# src/neurospatial/simulation/models/place_cells.py

class PlaceCellModel:
    """Gaussian place field model.

    Generalizes existing `simulate_place_field_firing_rate()` to work
    with neurospatial's N-D environments.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    center : NDArray[np.float64], shape (n_dims,), optional
        Place field center in continuous coordinates.
        If None, randomly chosen from env.bin_centers.
    width : float | NDArray[np.float64], optional
        Place field width (standard deviation of Gaussian).
        Can be scalar (isotropic) or array (anisotropic per dimension).
        If None, defaults to 3 * env bin_size.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0).
    baseline_rate : float, optional
        Baseline firing rate outside field (default: 0.001 Hz).
    condition : Callable[[NDArray, NDArray], NDArray] | None, optional
        Optional condition function: (positions, times) → bool mask.
        Example: lambda pos, t: pos[:, 0] > 50  # Only fire in right half

    Attributes
    ----------
    ground_truth : dict
        Contains 'center', 'width', 'max_rate', 'baseline_rate'

    Examples
    --------
    >>> # Simple place cell
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> pc = PlaceCellModel(env, center=[50.0, 75.0], width=10.0, max_rate=25.0)
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)
    >>> rates = pc.firing_rate(positions)

    >>> # Direction-selective place cell
    >>> def outbound_only(positions, times):
    ...     velocity = np.gradient(positions[:, 0])
    ...     return velocity > 0
    >>> pc_directional = PlaceCellModel(
    ...     env, center=[50.0], width=5.0, condition=outbound_only
    ... )

    >>> # Multiple place cells
    >>> pcs = [
    ...     PlaceCellModel(env, center=c, width=8.0)
    ...     for c in env.bin_centers[::10]  # Every 10th bin
    ... ]
    >>> all_rates = np.column_stack([pc.firing_rate(positions) for pc in pcs])
    """

    def __init__(
        self,
        env: Environment,
        center: NDArray[np.float64] | None = None,
        width: float | NDArray[np.float64] | None = None,
        max_rate: float = 20.0,
        baseline_rate: float = 0.001,
        condition: Callable | None = None,
        seed: int | None = None,
    ):
        pass

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute Gaussian place field firing rate.

        Implementation:
        1. Compute Euclidean distance from positions to center
        2. Apply Gaussian: exp(-distance^2 / (2 * width^2))
        3. Scale by max_rate, add baseline_rate
        4. Apply condition mask if provided
        """
        pass

    @property
    def ground_truth(self) -> dict:
        return {
            'center': self.center,
            'width': self.width,
            'max_rate': self.max_rate,
            'baseline_rate': self.baseline_rate,
        }
```

### Boundary Cells

```python
# src/neurospatial/simulation/models/boundary_cells.py

class BoundaryCellModel:
    """Distance-to-boundary tuned firing model.

    Models boundary vector cells (BVCs) or border cells.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    preferred_distance : float
        Preferred distance from boundary in environment units (default: 5.0).
    distance_tolerance : float
        Tuning width (default: 3.0).
    preferred_direction : float | None, optional
        Preferred allocentric direction in radians (2D only).
        If None, responds to all directions (classic border cell).
        If specified, responds to boundaries in that direction (BVC).
    direction_tolerance : float, optional
        Direction tuning width in radians (default: π/4).
    max_rate : float, optional
        Peak firing rate (default: 15.0 Hz).
    baseline_rate : float, optional
        Baseline rate (default: 0.001 Hz).
    distance_metric : Literal["geodesic", "euclidean"], optional
        Distance calculation method (default: "geodesic").

    Examples
    --------
    >>> # Classic border cell (all boundaries)
    >>> env = Environment.from_samples(arena_data, bin_size=2.0)
    >>> bc = BoundaryCellModel(env, preferred_distance=5.0)
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)
    >>> rates = bc.firing_rate(positions)

    >>> # Boundary vector cell (south wall)
    >>> bvc = BoundaryCellModel(
    ...     env,
    ...     preferred_distance=10.0,
    ...     preferred_direction=-np.pi/2,  # South
    ...     direction_tolerance=np.pi/6,
    ... )
    """

    def __init__(
        self,
        env: Environment,
        preferred_distance: float = 5.0,
        distance_tolerance: float = 3.0,
        preferred_direction: float | None = None,
        direction_tolerance: float = np.pi / 4,
        max_rate: float = 15.0,
        baseline_rate: float = 0.001,
        distance_metric: Literal["geodesic", "euclidean"] = "geodesic",
    ):
        pass

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute boundary-tuned firing rate.

        Implementation:
        1. For each position, compute distance to nearest boundary
        2. If preferred_direction specified, compute direction to boundary
        3. Apply Gaussian tuning: exp(-(distance - preferred_distance)^2 / (2 * tolerance^2))
        4. If directional, multiply by direction tuning
        """
        pass
```

### Grid Cells

```python
# src/neurospatial/simulation/models/grid_cells.py

class GridCellModel:
    """Periodic hexagonal grid pattern (2D only).

    Parameters
    ----------
    env : Environment
        The spatial environment (must be 2D).
    grid_spacing : float
        Distance between grid peaks in environment units (default: 50.0 cm).
    grid_orientation : float, optional
        Rotation of grid in radians (default: 0.0).
    phase_offset : NDArray[np.float64], shape (2,), optional
        Spatial phase offset (default: [0.0, 0.0]).
    max_rate : float, optional
        Peak firing rate (default: 20.0 Hz).
    baseline_rate : float, optional
        Baseline rate (default: 0.1 Hz).
    field_width : float, optional
        Width of individual grid bumps (default: grid_spacing / 3).

    Examples
    --------
    >>> env = Environment.from_samples(arena_2d, bin_size=2.0)
    >>> gc = GridCellModel(env, grid_spacing=50.0, grid_orientation=0.0)
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)
    >>> rates = gc.firing_rate(positions)

    >>> # Grid cell module (multiple cells with same spacing, different phases)
    >>> module_spacing = 50.0
    >>> phases = [
    ...     [0, 0], [10, 5], [20, 10], [30, 15]  # Different phase offsets
    ... ]
    >>> gcs = [
    ...     GridCellModel(env, grid_spacing=module_spacing, phase_offset=p)
    ...     for p in phases
    ... ]
    """

    def __init__(
        self,
        env: Environment,
        grid_spacing: float = 50.0,
        grid_orientation: float = 0.0,
        phase_offset: NDArray[np.float64] | None = None,
        max_rate: float = 20.0,
        baseline_rate: float = 0.1,
        field_width: float | None = None,
    ):
        if env.n_dims != 2:
            raise ValueError("GridCellModel only works for 2D environments")
        pass

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute hexagonal grid firing rate.

        Implementation (Hafting et al. 2005):
        Uses sum of 3 cosines with 60° spacing:
            rate = Σ_i cos(k_i · (pos - phase))
        where k_i are wave vectors at 0°, 60°, 120°
        """
        pass
```

---

## Spike Generation

```python
# src/neurospatial/simulation/spikes.py

def generate_poisson_spikes(
    firing_rate: NDArray[np.float64],
    times: NDArray[np.float64],
    refractory_period: float = 0.002,  # 2ms
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate Poisson spike train from firing rate time series.

    Generalizes existing `simulate_poisson_spikes()`.

    Parameters
    ----------
    firing_rate : NDArray[np.float64], shape (n_time,)
        Instantaneous firing rate in Hz at each time point.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.
    refractory_period : float, optional
        Absolute refractory period in seconds (default: 2ms).
        Prevents spikes within this window.
    seed : int | None, optional
        Random seed.

    Returns
    -------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of generated spikes in seconds.

    Examples
    --------
    >>> # Generate spikes from place cell
    >>> pc = PlaceCellModel(env, center=[50.0, 75.0])
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)
    >>> rates = pc.firing_rate(positions)
    >>> spike_times = generate_poisson_spikes(rates, times)
    >>> print(f"Generated {len(spike_times)} spikes")
    """
    pass


def generate_population_spikes(
    models: list[NeuralModel],
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    refractory_period: float = 0.002,
    seed: int | None = None,
) -> list[NDArray[np.float64]]:
    """Generate spike trains for population of neurons.

    Parameters
    ----------
    models : list[NeuralModel]
        List of neural models (PlaceCellModel, BoundaryCellModel, etc.).
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Position trajectory.
    times : NDArray[np.float64], shape (n_time,)
        Time points.
    refractory_period : float, optional
        Refractory period for each neuron (default: 2ms).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    spike_times_list : list[NDArray[np.float64]]
        List of spike time arrays, one per model.

    Examples
    --------
    >>> # Create population of place cells
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> place_cells = [
    ...     PlaceCellModel(env, center=c, width=8.0, max_rate=np.random.uniform(15, 30))
    ...     for c in env.bin_centers[::5]
    ... ]
    >>>
    >>> # Generate trajectory and spikes
    >>> positions, times = simulate_trajectory_ou(env, duration=120.0)
    >>> spike_trains = generate_population_spikes(place_cells, positions, times)
    >>>
    >>> # Analyze with neurospatial
    >>> for i, spike_times in enumerate(spike_trains):
    ...     rate_map = compute_place_field(env, spike_times, times, positions)
    ...     detected_center = field_centroid(rate_map, env)
    ...     true_center = place_cells[i].ground_truth['center']
    ...     error = np.linalg.norm(detected_center - true_center)
    ...     print(f"Cell {i}: detection error = {error:.2f} cm")
    """
    pass


def add_modulation(
    spike_times: NDArray[np.float64],
    modulation_freq: float,
    modulation_depth: float = 0.5,
    modulation_phase: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Add rhythmic modulation to spike train (e.g., theta oscillation).

    Parameters
    ----------
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Original spike times.
    modulation_freq : float
        Modulation frequency in Hz (e.g., 8 Hz for theta).
    modulation_depth : float, optional
        Modulation strength from 0 (none) to 1 (full) (default: 0.5).
    modulation_phase : float, optional
        Phase offset in radians (default: 0.0).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    modulated_spike_times : NDArray[np.float64], shape (n_modulated_spikes,)
        Spike times after rhythmic modulation (subset of original).

    Notes
    -----
    Implements non-homogeneous thinning: randomly removes spikes with
    probability proportional to phase of modulation cycle.

    Examples
    --------
    >>> # Add theta modulation to place cell
    >>> spike_times = generate_poisson_spikes(rates, times)
    >>> theta_modulated = add_modulation(spike_times, modulation_freq=8.0, modulation_depth=0.7)
    """
    pass
```

---

## Integration with Environment

### Factory Methods (optional)

Add convenience methods to `Environment` class:

```python
# In src/neurospatial/environment/factories.py (new mixin or additions)

class EnvironmentSimulation:
    """Mixin for simulation-related methods (optional design)."""

    def simulate_trajectory(
        self: "Environment",
        method: Literal["ou", "sinusoidal", "laps"] = "ou",
        **kwargs,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Simulate trajectory in this environment.

        Convenience wrapper for simulation.trajectory functions.

        Parameters
        ----------
        method : {'ou', 'sinusoidal', 'laps'}
            Trajectory generation method.
        **kwargs
            Passed to trajectory function (duration, speed, etc.).

        Returns
        -------
        positions : NDArray[np.float64], shape (n_time, n_dims)
        times : NDArray[np.float64], shape (n_time,)

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> positions, times = env.simulate_trajectory(method='ou', duration=60.0)
        """
        from neurospatial.simulation import (
            simulate_trajectory_ou,
            simulate_trajectory_sinusoidal,
            simulate_trajectory_laps,
        )

        if method == "ou":
            return simulate_trajectory_ou(self, **kwargs)
        elif method == "sinusoidal":
            return simulate_trajectory_sinusoidal(self, **kwargs)
        elif method == "laps":
            return simulate_trajectory_laps(self, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def create_place_cells(
        self: "Environment",
        n_cells: int,
        coverage: Literal["uniform", "random", "clustered"] = "uniform",
        **kwargs,
    ) -> list[PlaceCellModel]:
        """Create population of place cells covering this environment.

        Parameters
        ----------
        n_cells : int
            Number of place cells to create.
        coverage : {'uniform', 'random', 'clustered'}
            Spatial distribution of field centers.
        **kwargs
            Passed to PlaceCellModel (width, max_rate, etc.).

        Returns
        -------
        place_cells : list[PlaceCellModel]
            List of place cell models.

        Examples
        --------
        >>> env = Environment.from_samples(data, bin_size=2.0)
        >>> pcs = env.create_place_cells(n_cells=50, coverage='uniform', width=10.0)
        >>> positions, times = env.simulate_trajectory('ou', duration=120.0)
        >>> spike_trains = generate_population_spikes(pcs, positions, times)
        """
        from neurospatial.simulation.models import PlaceCellModel

        if coverage == "uniform":
            # Use evenly spaced bin centers
            indices = np.linspace(0, self.n_bins - 1, n_cells).astype(int)
            centers = self.bin_centers[indices]
        elif coverage == "random":
            # Random bin centers
            indices = np.random.choice(self.n_bins, n_cells, replace=False)
            centers = self.bin_centers[indices]
        elif coverage == "clustered":
            # TODO: Implement clustering (k-means or manual regions)
            raise NotImplementedError("Clustered coverage not yet implemented")
        else:
            raise ValueError(f"Unknown coverage: {coverage}")

        return [PlaceCellModel(self, center=c, **kwargs) for c in centers]
```

---

## API Examples

### Example 1: Basic Place Cell Simulation

```python
import numpy as np
from neurospatial import Environment
from neurospatial.simulation import (
    simulate_trajectory_ou,
    generate_population_spikes,
)
from neurospatial.simulation.models import PlaceCellModel
from neurospatial.fields import compute_place_field

# Create environment
env = Environment.from_samples(recorded_positions, bin_size=2.0)

# Create place cells with known ground truth
place_cells = [
    PlaceCellModel(env, center=[50, 75], width=10.0, max_rate=25.0),
    PlaceCellModel(env, center=[120, 45], width=8.0, max_rate=30.0),
    PlaceCellModel(env, center=[80, 100], width=12.0, max_rate=20.0),
]

# Generate synthetic trajectory
positions, times = simulate_trajectory_ou(
    env, duration=120.0, dt=0.01, speed_mean=0.08, coherence_time=0.7
)

# Generate spikes
spike_trains = generate_population_spikes(place_cells, positions, times)

# Validate neurospatial's place field detection
for i, spike_times in enumerate(spike_trains):
    # Compute place field using neurospatial
    rate_map = compute_place_field(
        env, spike_times, times, positions, method="diffusion_kde", bandwidth=5.0
    )

    # Compare to ground truth
    true_center = place_cells[i].ground_truth['center']
    detected_center = field_centroid(rate_map, env)
    error = np.linalg.norm(detected_center - true_center)

    print(f"Cell {i}: center error = {error:.2f} cm")
    # Expected: error < 2 * bin_size with good sampling
```

### Example 2: Direction-Selective Place Cell

```python
from neurospatial import Environment
from neurospatial.simulation import simulate_trajectory_sinusoidal
from neurospatial.simulation.models import PlaceCellModel

# 1D track environment
env = Environment.from_graph(track_graph, edge_order, bin_size=0.5)

# Direction-selective condition
def outbound_only(positions, times):
    velocity = np.gradient(positions[:, 0], axis=0)
    return velocity > 0

# Create directional place cell
pc_outbound = PlaceCellModel(
    env,
    center=[85.0],  # 1D center at 85 cm
    width=5.0,
    max_rate=30.0,
    condition=outbound_only,
)

pc_inbound = PlaceCellModel(
    env,
    center=[85.0],
    width=5.0,
    max_rate=25.0,
    condition=lambda pos, t: np.gradient(pos[:, 0], axis=0) < 0,  # Inbound
)

# Generate laps
positions, times = simulate_trajectory_sinusoidal(
    env, duration=240.0, running_speed=10.0, pause_duration=0.5
)

# Generate spikes
spike_times_out = generate_poisson_spikes(
    pc_outbound.firing_rate(positions, times), times
)
spike_times_in = generate_poisson_spikes(
    pc_inbound.firing_rate(positions, times), times
)

# Analyze by direction
direction = get_trajectory_direction(positions[:, 0])
# Use neurospatial to compute separate rate maps for each direction
```

### Example 3: Boundary Cell Simulation

```python
from neurospatial.simulation.models import BoundaryCellModel

# Create arena environment
env = Environment.from_samples(arena_data, bin_size=2.0)

# Classic border cell (responds to all walls)
bc = BoundaryCellModel(
    env,
    preferred_distance=5.0,  # Fires 5 cm from walls
    distance_tolerance=3.0,
    max_rate=15.0,
)

# Boundary vector cell (prefers south wall)
bvc_south = BoundaryCellModel(
    env,
    preferred_distance=10.0,
    preferred_direction=-np.pi / 2,  # South
    direction_tolerance=np.pi / 6,
    max_rate=18.0,
)

# Generate trajectory and spikes
positions, times = simulate_trajectory_ou(env, duration=180.0)
bc_spikes = generate_poisson_spikes(bc.firing_rate(positions), times)
bvc_spikes = generate_poisson_spikes(bvc_south.firing_rate(positions), times)

# Validate neurospatial's border score
from neurospatial.metrics import border_score

bc_rate_map = compute_place_field(env, bc_spikes, times, positions)
score = border_score(bc_rate_map, env, distance_metric="geodesic")
print(f"Border score: {score:.3f}")  # Expect > 0.5 for border cell
```

### Example 4: RatInABox Integration (Future)

```python
# Convert RatInABox simulation to neurospatial
from neurospatial.io import from_ratinabox
import ratinabox

# Generate data with RatInABox
Env_rib = ratinabox.Environment(params={'scale': 1.0})
Ag = ratinabox.Agent(Env_rib)
PCs = ratinabox.PlaceCells(Ag, params={'n': 100})

for _ in range(10000):
    Ag.update(dt=0.01)
    PCs.update()

# Convert to neurospatial format
env, spike_data = from_ratinabox(Ag, PCs, bin_size=0.05)

# Analyze with neurospatial
for cell_id, spike_times in enumerate(spike_data['spike_times']):
    rate_map = compute_place_field(
        env, spike_times, spike_data['times'], spike_data['positions']
    )
    info = skaggs_information(rate_map, env.occupancy(...))
    print(f"Cell {cell_id}: spatial info = {info:.3f} bits/spike")
```

---

## Implementation Priorities

### Phase 1: Core Trajectory + Place Cells (1 week)

**Goal**: Minimal viable simulation for testing place field detection

**Tasks**:
1. ✅ Create module structure (`simulation/` directory)
2. ✅ Implement `simulate_trajectory_ou()` (OU process from RatInABox)
3. ✅ Implement `simulate_trajectory_sinusoidal()` (generalize existing code)
4. ✅ Implement `PlaceCellModel` (Gaussian fields)
5. ✅ Implement `generate_poisson_spikes()`
6. ✅ Implement `generate_population_spikes()`
7. ✅ Write tests validating against existing simulation code
8. ✅ Document API with examples

**Validation**:
- Generate synthetic data
- Detect place fields with neurospatial
- Measure detection error vs ground truth (should be < 2 * bin_size)

### Phase 2: Boundary Cells + Extended Models (1 week)

**Tasks**:
1. ✅ Implement `BoundaryCellModel`
2. ✅ Validate against border_score metric
3. ✅ Implement `simulate_trajectory_laps()` for structured tasks
4. ✅ Add condition functions for direction-selective cells
5. ✅ Add `add_modulation()` for theta rhythms
6. ✅ Write integration tests (end-to-end validation)

### Phase 3: Grid Cells + Advanced Features (1 week)

**Tasks**:
1. ✅ Implement `GridCellModel` (hexagonal patterns)
2. ✅ Validate against gridness_score metric
3. ✅ Add HeadDirectionCellModel (if HD support added to neurospatial)
4. ✅ Add SpeedCellModel
5. ✅ Implement `Environment.create_place_cells()` convenience method
6. ✅ Add examples notebook

### Phase 4: RatInABox Integration (2 weeks)

**Tasks**:
1. ✅ Implement `neurospatial.io.from_ratinabox()` converter
2. ✅ Test RatInABox → neurospatial pipeline
3. ✅ Benchmark neurospatial metrics against RatInABox ground truth
4. ✅ Document integration workflow

---

## Testing Strategy

### Unit Tests

```python
# tests/simulation/test_trajectory.py

def test_ou_trajectory_stays_in_bounds():
    """OU process should respect environment boundaries."""
    env = Environment.from_samples(arena_data, bin_size=2.0)
    positions, times = simulate_trajectory_ou(env, duration=60.0, boundary_mode='reflect')

    # All positions should be inside environment
    bin_indices = env.bin_at(positions)
    assert np.all(bin_indices >= 0), "Some positions outside environment"


def test_ou_trajectory_coherence_time():
    """Velocity autocorrelation should decay with coherence_time."""
    env = Environment.from_samples(arena_data, bin_size=2.0)
    coherence_time = 0.7
    positions, times = simulate_trajectory_ou(
        env, duration=120.0, dt=0.01, coherence_time=coherence_time
    )

    # Compute velocity
    velocity = np.diff(positions, axis=0) / np.diff(times)[:, np.newaxis]

    # Compute autocorrelation at lag = coherence_time
    lag_samples = int(coherence_time / 0.01)
    autocorr = np.corrcoef(velocity[:-lag_samples, 0], velocity[lag_samples:, 0])[0, 1]

    # Should be around exp(-1) ≈ 0.37
    assert 0.2 < autocorr < 0.5, f"Autocorr = {autocorr:.3f}, expected ~0.37"


def test_sinusoidal_trajectory_1d_only():
    """Sinusoidal trajectory should require 1D environment."""
    env_2d = Environment.from_samples(arena_2d, bin_size=2.0)

    with pytest.raises(ValueError, match="only works for 1D"):
        simulate_trajectory_sinusoidal(env_2d, duration=60.0)
```

```python
# tests/simulation/test_models.py

def test_place_cell_peak_at_center():
    """Place cell should fire maximally at field center."""
    env = Environment.from_samples(arena_data, bin_size=2.0)
    center = np.array([50.0, 75.0])
    pc = PlaceCellModel(env, center=center, width=10.0, max_rate=25.0)

    # Evaluate at center
    rate_at_center = pc.firing_rate(center[np.newaxis, :])

    assert np.isclose(rate_at_center[0], 25.0, atol=0.1), "Peak rate should be at center"


def test_place_cell_gaussian_falloff():
    """Place cell should have Gaussian distance falloff."""
    env = Environment.from_samples(arena_data, bin_size=2.0)
    center = np.array([50.0, 75.0])
    width = 10.0
    pc = PlaceCellModel(env, center=center, width=width, max_rate=25.0, baseline_rate=0.0)

    # Test at 1 standard deviation away
    test_pos = center + np.array([width, 0.0])
    rate = pc.firing_rate(test_pos[np.newaxis, :])

    # Should be max_rate * exp(-0.5) ≈ 0.606 * max_rate
    expected = 25.0 * np.exp(-0.5)
    assert np.isclose(rate[0], expected, rtol=0.05), f"Rate = {rate[0]:.2f}, expected {expected:.2f}"


def test_boundary_cell_prefers_distance():
    """Boundary cell should fire at preferred distance from walls."""
    env = Environment.from_samples(arena_data, bin_size=2.0)
    bc = BoundaryCellModel(env, preferred_distance=5.0, distance_tolerance=2.0, max_rate=20.0)

    # Find positions exactly 5 cm from boundary
    boundary_bins = env.boundary_bins()
    # ... test at preferred_distance from boundary
    # Rate should be maximal
```

### Integration Tests

```python
# tests/simulation/test_integration.py

def test_place_field_detection_accuracy():
    """Validate neurospatial place field detection on synthetic data."""
    env = Environment.from_samples(arena_data, bin_size=2.0)

    # Create place cells with known centers
    true_centers = [
        np.array([50.0, 75.0]),
        np.array([120.0, 45.0]),
        np.array([80.0, 100.0]),
    ]
    place_cells = [
        PlaceCellModel(env, center=c, width=10.0, max_rate=25.0)
        for c in true_centers
    ]

    # Generate trajectory
    positions, times = simulate_trajectory_ou(env, duration=180.0, dt=0.01)

    # Generate spikes
    spike_trains = generate_population_spikes(place_cells, positions, times)

    # Detect fields
    for i, spike_times in enumerate(spike_trains):
        rate_map = compute_place_field(env, spike_times, times, positions)
        fields = detect_place_fields(rate_map, env, threshold=0.2)

        assert len(fields) >= 1, f"Cell {i}: No field detected"

        # Centroid should be close to true center
        detected_center = field_centroid(rate_map, fields[0], env)
        error = np.linalg.norm(detected_center - true_centers[i])

        # Error should be less than 2 bin widths with good sampling
        max_error = 2 * env.bin_size
        assert error < max_error, f"Cell {i}: center error {error:.2f} > {max_error:.2f} cm"


def test_border_score_validation():
    """Validate border_score metric on synthetic boundary cells."""
    env = Environment.from_samples(arena_data, bin_size=2.0)

    # Create boundary cell
    bc = BoundaryCellModel(env, preferred_distance=5.0, max_rate=20.0)

    # Generate trajectory
    positions, times = simulate_trajectory_ou(env, duration=240.0)

    # Generate spikes
    spike_times = generate_poisson_spikes(bc.firing_rate(positions), times)

    # Compute rate map
    rate_map = compute_place_field(env, spike_times, times, positions)

    # Compute border score
    score = border_score(rate_map, env, distance_metric="geodesic")

    # Boundary cell should have high border score (> 0.5)
    assert score > 0.5, f"Border score {score:.3f} too low for boundary cell"


def test_skaggs_information_validation():
    """Validate spatial information metric on synthetic place cells."""
    env = Environment.from_samples(arena_data, bin_size=2.0)

    # Create place cell with narrow field (high information)
    pc_narrow = PlaceCellModel(env, center=[50, 75], width=5.0, max_rate=30.0)

    # Create place cell with broad field (low information)
    pc_broad = PlaceCellModel(env, center=[50, 75], width=20.0, max_rate=30.0)

    # Generate trajectory
    positions, times = simulate_trajectory_ou(env, duration=180.0)

    # Generate spikes and compute information
    for pc, expected_info in [(pc_narrow, "high"), (pc_broad, "low")]:
        spike_times = generate_poisson_spikes(pc.firing_rate(positions), times)
        rate_map = compute_place_field(env, spike_times, times, positions)
        occupancy = env.occupancy(times, positions)
        info = skaggs_information(rate_map, occupancy)

        if expected_info == "high":
            assert info > 1.0, f"Narrow field should have info > 1.0, got {info:.3f}"
        else:
            assert info < 1.0, f"Broad field should have info < 1.0, got {info:.3f}"
```

---

## Summary

### Key Design Decisions

1. **Environment-centric**: All simulations operate on `Environment` instances (not standalone)
2. **Protocol-based**: `NeuralModel` protocol allows extensibility
3. **Composable layers**: Separate trajectory, models, and spike generation
4. **Ground truth tracking**: All models expose `.ground_truth` for validation
5. **N-D native**: Works with any dimensionality (1D, 2D, 3D, N-D)

### Benefits

1. **Testing**: Generate synthetic data to validate all neurospatial metrics
2. **Education**: Teach spatial coding with interactive simulations
3. **Benchmarking**: Test algorithm performance on controlled data
4. **Integration**: Bridge to RatInABox for forward-inverse model pipeline

### Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize Phase 1** (OU trajectory + place cells)
3. **Implement** following structure outlined above
4. **Validate** against existing simulation code (port + extend)
5. **Document** with examples and tutorials

---

## References

- **RatInABox**: George et al. (2023), eLife - OU process implementation
- **Existing code**: `simulate_position()`, `simulate_place_field_firing_rate()`, etc.
- **Neuroscience**: Sargolini et al. (2006) for locomotion parameters
- **Neurospatial**: Validated metrics (Skaggs 1993, Solstad 2008, Hafting 2005)
