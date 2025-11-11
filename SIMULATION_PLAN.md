# Simulation Subpackage Plan

**Goal**: Create a `neurospatial.simulation` subpackage for generating synthetic spatial data, neural activity, and spike trains for testing, validation, and education.

**Inspiration**: Existing simulation code + biologically-realistic movement models + neurospatial's N-D architecture

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
├── session.py                     # High-level session simulation
├── validation.py                  # Validation and analysis helpers
├── utils.py                       # Helper functions
└── examples.py                    # Pre-configured example datasets

tests/simulation/
├── test_trajectory.py
├── test_models.py
├── test_spikes.py
└── test_integration.py            # End-to-end validation tests
```

### Public API Design (Flat Imports)

**Design Principle**: "Flat is better than nested" (Zen of Python)

All public functions and classes should be importable directly from `neurospatial.simulation`:

```python
# src/neurospatial/simulation/__init__.py

"""Simulation subpackage for generating synthetic spatial data."""

# Trajectory simulation
from neurospatial.simulation.trajectory import (
    simulate_trajectory_ou,
    simulate_trajectory_sinusoidal,
    simulate_trajectory_laps,
)

# Neural models
from neurospatial.simulation.models import (
    NeuralModel,  # Protocol
    PlaceCellModel,
    BoundaryCellModel,
    GridCellModel,
    HeadDirectionCellModel,
    SpeedCellModel,
)

# Spike generation
from neurospatial.simulation.spikes import (
    generate_poisson_spikes,
    generate_population_spikes,
    add_modulation,
)

# High-level session API
from neurospatial.simulation.session import (
    SimulationSession,  # Dataclass
    simulate_session,
)

# Validation and plotting
from neurospatial.simulation.validation import (
    validate_simulation,
    plot_session_summary,
)

# Pre-configured examples
from neurospatial.simulation.examples import (
    open_field_session,
    linear_track_session,
    tmaze_alternation_session,
    boundary_cell_session,
    grid_cell_session,
)

__all__ = [
    # Trajectory
    "simulate_trajectory_ou",
    "simulate_trajectory_sinusoidal",
    "simulate_trajectory_laps",
    # Models
    "NeuralModel",
    "PlaceCellModel",
    "BoundaryCellModel",
    "GridCellModel",
    "HeadDirectionCellModel",
    "SpeedCellModel",
    # Spikes
    "generate_poisson_spikes",
    "generate_population_spikes",
    "add_modulation",
    # Session
    "SimulationSession",
    "simulate_session",
    # Validation
    "validate_simulation",
    "plot_session_summary",
    # Examples
    "open_field_session",
    "linear_track_session",
    "tmaze_alternation_session",
    "boundary_cell_session",
    "grid_cell_session",
]
```

**Benefits**:

- ✓ Discoverable: `from neurospatial.simulation import <TAB>` shows everything
- ✓ Consistent: One obvious way to import
- ✓ IDE-friendly: Autocomplete works across the entire API
- ✓ Type-checker friendly: All exports in one place

**Usage Example**:

```python
# ✓ Recommended: Flat import (one source of truth)
from neurospatial.simulation import (
    simulate_trajectory_ou,
    PlaceCellModel,
    generate_poisson_spikes,
    open_field_session,
)

# ✗ Discouraged: Deep imports (harder to discover, inconsistent)
from neurospatial.simulation.trajectory import simulate_trajectory_ou
from neurospatial.simulation.models.place_cells import PlaceCellModel
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
    speed_units: str | None = None,
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
        Controls movement smoothness:
        - 0.3-0.5s: Jittery, erratic movement (stressed/novel environment)
        - 0.6-0.8s: Natural exploration (default, from Sargolini 2006)
        - 0.9-1.2s: Smooth, persistent movement (familiar environment)
        Larger values = slower direction changes, more persistent trajectories.
    boundary_mode : {'reflect', 'periodic', 'stop'}, optional
        How to handle environment boundaries (default: 'reflect').
        - 'reflect': Trajectories bounce off walls elastically. Most realistic
          for typical experiments (open field arenas, water mazes).
        - 'periodic': Opposite boundaries wrap around (torus topology). Use
          only for specialized analyses. NOT biologically realistic.
        - 'stop': Trajectory halts at boundaries. Can create unrealistic
          accumulation at walls.
    speed_units : str | None, optional
        Units for speed_mean/speed_std. If None, assumes same units as env.units.
        If specified and different from env.units, will auto-convert.
        Example: speed_units="m" when env.units="cm" converts m/s to cm/s.
    seed : int | None, optional
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If env.units is not set. Call env.units = "cm" before simulation.
    ValueError
        If speed_units are incompatible with env.units.

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Continuous position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.

    Notes
    -----
    The OU process implements an N-dimensional velocity-based random walk:

    For each spatial dimension i:
        dv_i = -θ v_i dt + σ dW_i
        dx_i = v_i dt

    where:
    - θ = 1/coherence_time controls mean reversion rate (velocity decorrelation)
    - σ = speed_std * sqrt(2*θ) ensures stationary speed distribution
    - dW_i are independent Wiener increments
    - Velocity magnitude is clipped to ensure |v| ≈ speed_mean

    The stationary distribution of velocity has zero mean and variance speed_std².
    This produces realistic movement with controlled speed and smoothness.

    Boundary Handling Implementation:
    - 'reflect': Upon boundary crossing, reflect position across boundary and
      reverse velocity component normal to boundary: v_perp → -v_perp
    - 'periodic': Wrap position using modulo on dimension_ranges, keep velocity
    - 'stop': Clamp position to nearest valid bin center, zero velocity in
      crossing direction

    References: George et al. (2023) eLife, Sargolini et al. (2006) Science.

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
    speed: float = 10.0,
    period: float | None = None,
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
    speed : float, optional
        Running speed in environment units/second (default: 10.0 cm/s).
    period : float | None, optional
        Period of sinusoidal motion in seconds. If None, computed from
        track length and speed: period = 2 * track_length / speed.
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
    return_metadata: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[NDArray[np.float64], NDArray[np.float64], dict]:
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
    return_metadata : bool, optional
        If True, returns metadata dict with lap_ids, lap_boundaries, and
        direction arrays (default: False).

    Returns
    -------
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_time,)
        Time points.
    metadata : dict, optional (if return_metadata=True)
        Contains:
        - 'lap_ids': NDArray[np.int_], shape (n_time,) - Lap number for each time
        - 'lap_boundaries': NDArray[np.int_] - Indices where laps start
        - 'direction': NDArray[str] - 'outbound' or 'inbound' for each time

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

from typing import Protocol, runtime_checkable, Any
from numpy.typing import NDArray
import numpy as np

@runtime_checkable
class NeuralModel(Protocol):
    """Protocol for neural models that generate firing rates.

    All neural models must implement:
    - firing_rate(positions, times) → rates
    - ground_truth property (returns dict with model parameters)

    The structure of ground_truth dict depends on model type:

    **PlaceCell**:
        - 'center': NDArray[np.float64] - field center coordinates
        - 'width': float | NDArray[np.float64] - field width (isotropic or per-dimension)
        - 'max_rate': float - peak firing rate in Hz
        - 'baseline_rate': float - baseline firing rate in Hz

    **BoundaryCell**:
        - 'preferred_distance': float - preferred distance from boundary
        - 'distance_tolerance': float - tuning width
        - 'preferred_direction': float | None - direction (radians, or None for omnidirectional)
        - 'max_rate': float - peak firing rate in Hz
        - 'baseline_rate': float - baseline firing rate in Hz

    **GridCell**:
        - 'grid_spacing': float - distance between grid peaks
        - 'grid_orientation': float - grid rotation in radians
        - 'phase_offset': NDArray[np.float64] - spatial phase offset
        - 'max_rate': float - peak firing rate in Hz
        - 'baseline_rate': float - baseline firing rate in Hz
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
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Model-specific parameters. Structure varies by model type
            (see class docstring for details).
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
    distance_metric : {'euclidean', 'geodesic'}, optional
        Method for computing distance from positions to field center.
        - 'euclidean': Straight-line distance (default). Fast but ignores
          barriers. Use for open field environments or when performance matters.
        - 'geodesic': Path distance through environment connectivity graph.
          Biologically accurate for complex environments with barriers/walls.
          ~100x slower than euclidean.
    condition : Callable[[NDArray, NDArray], NDArray[bool]] | None, optional
        Optional condition function: (positions, times) → bool mask.
        Allows direction-selective, speed-gated, or context-dependent firing.
        Common patterns:
        - Direction-selective: lambda pos, t: np.gradient(pos[:, 0]) > 0
        - Speed-gated: lambda pos, t: compute_speed(pos, t) > threshold
        - Region-specific: lambda pos, t: (pos[:, 0] > x_min) & (pos[:, 0] < x_max)
        - Trial-type: lambda pos, t: trial_labels[t] == "correct"

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
        distance_metric: Literal["euclidean", "geodesic"] = "euclidean",
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
        1. Compute distance from positions to center (euclidean or geodesic)
        2. Apply Gaussian with numerical stability:
           - Clip distances > 5*width (contribution < 1e-6)
           - rate = baseline + (max - baseline) * exp(-0.5 * (d/width)^2)
        3. Apply condition mask if provided (multiply rates by mask)

        For geodesic distance:
        - Map positions to bin indices using env.bin_at()
        - Precompute distance field from center bin
        - Lookup distances (O(1) per position)
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

        Implementation (Hafting et al. 2005, Fuhs & Touretzky 2006):

        Wave vectors in rotated coordinate system:
            k_magnitude = 4π / (√3 * grid_spacing)
            k_1 = k_magnitude * [cos(θ), sin(θ)]
            k_2 = k_magnitude * [cos(θ + 60°), sin(θ + 60°)]
            k_3 = k_magnitude * [cos(θ + 120°), sin(θ + 120°)]

        where θ = grid_orientation

        Grid pattern (range [-1, 1]):
            g(x) = (1/3) * Σ_i cos(k_i · (x - phase_offset))

        Final firing rate:
            rate(x) = baseline + (max_rate - baseline) * max(0, g(x))

        This produces hexagonal symmetry with peaks at grid vertices.
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

    Notes
    -----
    Algorithm (absolute refractory period implementation):

    1. Generate candidate spikes from inhomogeneous Poisson process:
       - For each time step: spike if rand() < rate[i] * dt
    2. Sort candidate spike times
    3. Apply refractory period filter (single pass, O(n)):
       - Initialize last_spike_time = -inf
       - For each candidate in order:
         - If candidate >= last_spike_time + refractory_period:
           - Keep spike, update last_spike_time = candidate
         - Else: discard spike

    This ensures minimum inter-spike interval (ISI) >= refractory_period,
    matching biological absolute refractory period dynamics.

    Examples
    --------
    >>> # Generate spikes from place cell
    >>> pc = PlaceCellModel(env, center=[50.0, 75.0])
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0)
    >>> rates = pc.firing_rate(positions)
    >>> spike_times = generate_poisson_spikes(rates, times)
    >>> print(f"Generated {len(spike_times)} spikes")
    >>> # Verify refractory period
    >>> isi = np.diff(spike_times)
    >>> assert np.all(isi >= 0.002), "All ISIs should be >= 2ms"
    """
    pass


def generate_population_spikes(
    models: list[NeuralModel],
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    refractory_period: float = 0.002,
    seed: int | None = None,
    show_progress: bool = True,
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
    show_progress : bool, optional
        Show progress bar during spike generation (default: True).
        Set to False for quiet operation in scripts.

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

    Notes
    -----
    Implementation shows progress bar with tqdm:

    from tqdm.auto import tqdm
    iterator = tqdm(models, desc="Generating spikes", disable=not show_progress)
    for model in iterator:
        rates = model.firing_rate(positions, times)
        spikes = generate_poisson_spikes(rates, times, refractory_period, seed)
        spike_trains.append(spikes)
        iterator.set_postfix({'n_spikes': len(spikes), 'rate': f'{len(spikes)/times[-1]:.1f} Hz'})

    Prints summary after completion:
    "Generated 50 cells, 12,450 total spikes (avg 249 spikes/cell), mean rate 2.3 Hz"
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

## High-Level Convenience Functions

### Session Simulation

**Purpose**: One-call workflow for common simulation patterns

```python
# src/neurospatial/simulation/session.py

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SimulationSession:
    """Complete simulation session result.

    Attributes
    ----------
    env : Environment
        The spatial environment.
    positions : NDArray[np.float64], shape (n_time, n_dims)
        Trajectory positions.
    times : NDArray[np.float64], shape (n_time,)
        Time points in seconds.
    spike_trains : list[NDArray[np.float64]]
        List of spike time arrays, one per cell.
    models : list[NeuralModel]
        Neural model instances used to generate spikes.
    ground_truth : dict[str, Any]
        True parameters for each cell (for validation).
    metadata : dict[str, Any]
        Session parameters and configuration.
    """
    env: Environment
    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    spike_trains: list[NDArray[np.float64]]
    models: list[NeuralModel]
    ground_truth: dict[str, Any]
    metadata: dict[str, Any]


def simulate_session(
    env: Environment,
    duration: float,
    n_cells: int = 50,
    cell_type: Literal["place", "boundary", "grid", "mixed"] = "place",
    trajectory_method: Literal["ou", "sinusoidal", "laps"] = "ou",
    coverage: Literal["uniform", "random"] = "uniform",
    show_progress: bool = True,
    seed: int | None = None,
    **kwargs,
) -> SimulationSession:
    """Simulate complete recording session with trajectory and neural activity.

    High-level convenience function that combines trajectory simulation,
    neural model creation, and spike generation in one call.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    duration : float
        Session duration in seconds.
    n_cells : int, optional
        Number of neurons to simulate (default: 50).
    cell_type : {'place', 'boundary', 'grid', 'mixed'}, optional
        Type of neurons to create (default: 'place').
        - 'place': All place cells
        - 'boundary': All boundary cells
        - 'grid': All grid cells (requires 2D environment)
        - 'mixed': 60% place, 20% boundary, 20% grid
    trajectory_method : {'ou', 'sinusoidal', 'laps'}, optional
        Trajectory generation method (default: 'ou').
    coverage : {'uniform', 'random'}, optional
        Spatial distribution of field centers (default: 'uniform').
        - 'uniform': Evenly space field centers across environment using
          env.bin_centers[::step] where step = max(1, n_bins // n_cells).
          Ensures full coverage without clustering.
        - 'random': Randomly sample field centers from env.bin_centers
          using np.random.choice(env.bin_centers, size=n_cells).
          May result in clustering or gaps in coverage.
    show_progress : bool, optional
        Show progress bars during generation (default: True).
    seed : int | None, optional
        Random seed for reproducibility.
    **kwargs
        Additional parameters passed to trajectory and model functions.
        Examples: speed_mean, coherence_time, max_rate, etc.

    Returns
    -------
    session : SimulationSession
        Dataclass containing:
        - env: Environment instance
        - positions: NDArray, shape (n_time, n_dims) - trajectory
        - times: NDArray, shape (n_time,) - time points
        - spike_trains: list[NDArray] - spike times per cell
        - models: list[NeuralModel] - neural model instances
        - ground_truth: dict - true parameters for each cell
        - metadata: dict - session parameters

    Examples
    --------
    >>> # Quick place cell session
    >>> env = Environment.from_samples(arena_data, bin_size=2.0)
    >>> session = simulate_session(env, duration=120.0, n_cells=50)
    >>> positions = session.positions  # ✓ Typed access
    >>> spike_trains = session.spike_trains  # ✓ IDE autocomplete
    >>> ground_truth = session.ground_truth  # ✓ Discoverable

    >>> # Mixed cell types with custom parameters
    >>> session = simulate_session(
    ...     env,
    ...     duration=180.0,
    ...     n_cells=100,
    ...     cell_type='mixed',
    ...     trajectory_method='ou',
    ...     speed_mean=0.1,
    ...     coherence_time=0.8,
    ... )

    >>> # Validate detected vs true place fields
    >>> from neurospatial.simulation import validate_simulation
    >>> report = validate_simulation(session)
    >>> print(report['summary'])

    See Also
    --------
    simulate_trajectory_ou : Low-level trajectory simulation
    PlaceCellModel : Individual place cell model
    generate_population_spikes : Low-level spike generation
    validate_simulation : Validate simulation results
    """
    pass
```

### Validation Helpers

**Purpose**: Automated validation of simulation results against ground truth

```python
# src/neurospatial/simulation/validation.py

def validate_simulation(
    session: SimulationSession | None = None,
    spike_trains: list[NDArray] | None = None,
    ground_truth: list[dict] | None = None,
    env: Environment | None = None,
    positions: NDArray | None = None,
    times: NDArray | None = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    show_plots: bool = False,
) -> dict:
    """Validate simulation by comparing detected fields to ground truth.

    Computes place fields from spike trains and compares to known true
    parameters. Returns comprehensive diagnostic report.

    Parameters
    ----------
    session : SimulationSession | None, optional
        Session from simulate_session(). If provided, other parameters
        are extracted from session. Otherwise, must provide individual params.
    spike_trains : list[NDArray] | None, optional
        List of spike time arrays (if session not provided).
    ground_truth : list[dict] | None, optional
        List of ground truth parameter dicts (if session not provided).
    env : Environment | None, optional
        Environment instance (if session not provided).
    positions : NDArray | None, optional
        Position trajectory (if session not provided).
    times : NDArray | None, optional
        Time points (if session not provided).
    method : str, optional
        Place field computation method (default: 'diffusion_kde').
    show_plots : bool, optional
        Show diagnostic plots (default: False).

    Returns
    -------
    report : dict
        Validation report containing:
        - 'center_errors': NDArray - distance between detected and true centers (cm)
        - 'center_correlations': NDArray - correlation of rate maps
        - 'width_errors': NDArray - difference in field widths
        - 'rate_errors': NDArray - difference in peak firing rates
        - 'summary': str - formatted summary statistics
        - 'passed': bool - True if all errors within acceptable thresholds
        - 'plots': dict - matplotlib figures (if show_plots=True)

    Examples
    --------
    >>> # Validate session
    >>> session = simulate_session(env, duration=120.0, n_cells=50)
    >>> report = validate_simulation(session, show_plots=True)
    >>> print(report['summary'])
    '''
    Validation Report
    =================
    Cells analyzed: 50
    Mean center error: 1.85 cm (threshold: 4.0 cm)
    Mean correlation: 0.94 (threshold: 0.80)
    Mean width error: 0.23 cm (threshold: 2.0 cm)
    Passed: 48/50 cells (96%)
    Status: PASS
    '''

    >>> # Manual validation
    >>> report = validate_simulation(
    ...     spike_trains=spike_trains,
    ...     ground_truth=ground_truth,
    ...     env=env,
    ...     positions=positions,
    ...     times=times,
    ... )

    >>> # Check individual cell errors
    >>> for i, error in enumerate(report['center_errors']):
    ...     if error > 4.0:  # threshold
    ...         print(f"Cell {i} failed: center error = {error:.2f} cm")

    See Also
    --------
    simulate_session : Generate simulation data
    compute_place_field : Compute place fields from spikes
    """
    pass


def plot_session_summary(
    session: SimulationSession,
    cell_ids: list[int] | None = None,
    figsize: tuple[float, float] = (15, 10),
) -> tuple:
    """Create comprehensive visualization of simulation session.

    Parameters
    ----------
    session : SimulationSession
        Session from simulate_session().
    cell_ids : list[int] | None, optional
        Specific cells to plot. If None, plots first 6 cells.
    figsize : tuple, optional
        Figure size (default: (15, 10)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    axes : np.ndarray
        Array of axis handles.

    Examples
    --------
    >>> session = simulate_session(env, duration=120.0, n_cells=50)
    >>> fig, axes = plot_session_summary(session, cell_ids=[0, 5, 10, 15])
    >>> plt.show()
    """
    pass
```

### Pre-Configured Examples

**Purpose**: Ready-to-use simulation datasets for testing and education

```python
# src/neurospatial/simulation/examples.py

def open_field_session(
    duration: float = 180.0,
    arena_size: float = 100.0,
    bin_size: float = 2.0,
    n_place_cells: int = 50,
    seed: int | None = None,
) -> SimulationSession:
    """Generate standard open field arena simulation.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 180.0).
    arena_size : float, optional
        Square arena side length in cm (default: 100.0).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    n_place_cells : int, optional
        Number of place cells (default: 50).
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    session : SimulationSession
        Complete session (see simulate_session).

    Examples
    --------
    >>> # Quick start for testing
    >>> session = open_field_session(duration=60.0, n_place_cells=20)
    >>> env = session.env
    >>> spike_trains = session.spike_trains

    >>> # Validate neurospatial's place field detection
    >>> from neurospatial.simulation import validate_simulation
    >>> report = validate_simulation(session)
    >>> assert report['passed'], "Place field detection failed!"
    """
    pass


def linear_track_session(
    duration: float = 240.0,
    track_length: float = 200.0,
    bin_size: float = 1.0,
    n_place_cells: int = 40,
    n_laps: int = 20,
    seed: int | None = None,
) -> SimulationSession:
    """Generate linear track with sinusoidal running.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 240.0).
    track_length : float, optional
        Track length in cm (default: 200.0).
    bin_size : float, optional
        Spatial bin size in cm (default: 1.0).
    n_place_cells : int, optional
        Number of place cells (default: 40).
    n_laps : int, optional
        Number of back-and-forth laps (default: 20).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    session : SimulationSession
        Complete session with 1D environment.

    Examples
    --------
    >>> session = linear_track_session(n_laps=10, duration=120.0)
    >>> # Session includes directional place cells
    >>> assert session.env.is_1d
    """
    pass


def tmaze_alternation_session(
    duration: float = 300.0,
    n_trials: int = 20,
    n_place_cells: int = 60,
    seed: int | None = None,
) -> SimulationSession:
    """Generate T-maze spatial alternation task.

    Parameters
    ----------
    duration : float, optional
        Session duration in seconds (default: 300.0).
    n_trials : int, optional
        Number of alternation trials (default: 20).
    n_place_cells : int, optional
        Number of place cells (default: 60).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    session : SimulationSession
        Session with structured lap trajectories.
        Includes trial metadata (left/right choices).

    Examples
    --------
    >>> session = tmaze_alternation_session(n_trials=10)
    >>> metadata = session.metadata
    >>> trial_choices = metadata['trial_choices']  # ['left', 'right', 'left', ...]
    """
    pass


def boundary_cell_session(
    duration: float = 180.0,
    arena_shape: Literal["square", "circle", "polygon"] = "square",
    arena_size: float = 100.0,
    n_boundary_cells: int = 30,
    n_place_cells: int = 20,
    seed: int | None = None,
) -> SimulationSession:
    """Generate session with boundary/border cells.

    Parameters
    ----------
    duration : float, optional
        Session duration (default: 180.0).
    arena_shape : {'square', 'circle', 'polygon'}, optional
        Arena shape (default: 'square').
    arena_size : float, optional
        Approximate arena size in cm (default: 100.0).
    n_boundary_cells : int, optional
        Number of boundary cells (default: 30).
    n_place_cells : int, optional
        Number of place cells (default: 20).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    session : SimulationSession
        Session with boundary and place cells.

    Examples
    --------
    >>> session = boundary_cell_session(arena_shape='circle', n_boundary_cells=20)
    >>> # Validate border score metric
    >>> from neurospatial.metrics import border_score
    >>> for i, model in enumerate(session.models):
    ...     if isinstance(model, BoundaryCellModel):
    ...         # Compute rate map from spikes
    ...         rate_map = compute_place_field(session.env, session.spike_trains[i],
    ...                                         session.times, session.positions)
    ...         score = border_score(rate_map, session.env)
    ...         print(f"Cell {i} border score: {score:.3f}")
    """
    pass


def grid_cell_session(
    duration: float = 300.0,
    arena_size: float = 150.0,
    grid_spacing: float = 50.0,
    n_grid_cells: int = 40,
    seed: int | None = None,
) -> SimulationSession:
    """Generate session with grid cells (2D only).

    Parameters
    ----------
    duration : float, optional
        Session duration (default: 300.0).
    arena_size : float, optional
        Square arena size (default: 150.0 cm).
    grid_spacing : float, optional
        Grid spacing for module (default: 50.0 cm).
    n_grid_cells : int, optional
        Number of grid cells (default: 40).
    seed : int | None, optional
        Random seed.

    Returns
    -------
    session : SimulationSession
        Session with grid cells at various phases.

    Examples
    --------
    >>> session = grid_cell_session(grid_spacing=40.0, n_grid_cells=20)
    >>> # Validate gridness score
    >>> from neurospatial.metrics import gridness_score
    >>> for i in range(len(session.spike_trains)):
    ...     # Compute rate map from spikes
    ...     rate_map = compute_place_field(session.env, session.spike_trains[i],
    ...                                     session.times, session.positions)
    ...     score = gridness_score(rate_map, session.env)
    ...     print(f"Cell {i} gridness: {score:.3f}")
    """
    pass
```

---

## API Examples

### Example 0: Quick Start with Pre-Configured Sessions (Recommended)

```python
from neurospatial.simulation import (
    open_field_session,
    validate_simulation,
    plot_session_summary,
)

# Generate complete simulation session
session = open_field_session(duration=180.0, n_place_cells=50, seed=42)

# Validate place field detection
report = validate_simulation(session, show_plots=True)
print(report['summary'])
# Output:
# Validation Report
# =================
# Cells analyzed: 50
# Mean center error: 1.85 cm (threshold: 4.0 cm)
# Mean correlation: 0.94 (threshold: 0.80)
# Passed: 48/50 cells (96%)
# Status: PASS

# Visualize session
fig, axes = plot_session_summary(session, cell_ids=[0, 5, 10, 15])
plt.show()

# Access session components (typed attributes)
env = session.env
positions = session.positions
spike_trains = session.spike_trains
ground_truth = session.ground_truth
```

### Example 1: Basic Place Cell Simulation (Low-Level API)

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
env.units = "cm"  # Required for trajectory simulation

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

    # Compare to ground truth (simple peak detection)
    true_center = place_cells[i].ground_truth['center']
    detected_center = env.bin_centers[np.argmax(rate_map)]
    error = np.linalg.norm(detected_center - true_center)

    print(f"Cell {i}: center error = {error:.2f} cm")
    # Expected: error < 2 * bin_size with good sampling
```

### Example 1b: Using High-Level Session API

```python
from neurospatial import Environment
from neurospatial.simulation import simulate_session, validate_simulation

# Create environment
env = Environment.from_samples(recorded_positions, bin_size=2.0)
env.units = "cm"

# Simulate complete session with one call
session = simulate_session(
    env,
    duration=120.0,
    n_cells=50,
    cell_type='place',
    trajectory_method='ou',
    speed_mean=0.08,
    coherence_time=0.7,
    seed=42,
)

# Validate automatically
report = validate_simulation(session)
if report['passed']:
    print(f"✓ Validation passed: {report['summary']}")
else:
    print(f"✗ Validation failed")
    for i, error in enumerate(report['center_errors']):
        if error > 4.0:
            print(f"  Cell {i}: center error = {error:.2f} cm")
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
import numpy as np
from neurospatial import Environment
from neurospatial.simulation import (
    simulate_trajectory_ou,
    generate_poisson_spikes,
)
from neurospatial.simulation.models import BoundaryCellModel
from neurospatial.fields import compute_place_field

# Create arena environment
env = Environment.from_samples(arena_data, bin_size=2.0)
env.units = "cm"

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

# Validate with simple boundary proximity metric
bc_rate_map = compute_place_field(env, bc_spikes, times, positions)
# Note: border_score() is a metric function in neurospatial
# that compares firing near vs far from boundaries
print(f"Boundary cell validated with rate map peak near walls")
```

### Example 4: Multi-Environment Testing with Pre-Configured Datasets

```python
from neurospatial.simulation import (
    open_field_session,
    linear_track_session,
    boundary_cell_session,
    grid_cell_session,
    validate_simulation,
)

# Test different environment types
sessions = {
    'open_field': open_field_session(duration=120.0, n_place_cells=40, seed=1),
    'linear_track': linear_track_session(duration=120.0, n_place_cells=30, seed=2),
    'boundary': boundary_cell_session(duration=120.0, n_boundary_cells=20, seed=3),
    'grid': grid_cell_session(duration=180.0, n_grid_cells=30, seed=4),
}

# Validate all sessions
results = {}
for name, session in sessions.items():
    report = validate_simulation(session)
    results[name] = report['passed']
    print(f"{name}: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"  Mean error: {report['center_errors'].mean():.2f} cm")
    print(f"  Mean correlation: {report['center_correlations'].mean():.3f}")

# Ensure all pass
assert all(results.values()), "Some validations failed!"
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

### Phase 3: Grid Cells + Convenience Functions (1.5 weeks)

**Core Tasks**:
1. ✅ Implement `GridCellModel` (hexagonal patterns)
2. ✅ Validate against gridness_score metric
3. ✅ Add HeadDirectionCellModel (if HD support added to neurospatial)
4. ✅ Add SpeedCellModel

**New High-Value Features** (from reviewer feedback):
5. ✅ Implement `simulate_session()` - One-call workflow for common patterns
6. ✅ Implement `validate_simulation()` - Automated ground truth validation
7. ✅ Create pre-configured examples:
   - `open_field_session()`
   - `linear_track_session()`
   - `tmaze_alternation_session()`
   - `boundary_cell_session()`
   - `grid_cell_session()`
8. ✅ Add `plot_session_summary()` visualization
9. ✅ Implement batch trajectory generation (n_trials parameter)
10. ✅ Add examples notebook showcasing convenience functions

### Phase 4: Advanced Realism Features (1 week, optional)

**Goal**: Enhanced biological realism for specialized applications

**Optional Enhancements**:
1. **State-dependent movement** - Exploration vs exploitation modes
   - `exploration_params` and `exploitation_params` in OU process
   - Automatic state switching with configurable rate
2. **Bursting behavior** - Realistic CA1/CA3 spike patterns
   - Add `burst_probability`, `burst_size`, `intraburst_interval` to spike generation
3. **Elliptical place fields** - Non-circular Gaussian fields
   - Add `covariance` matrix parameter to PlaceCellModel
4. **Correlated turning** - Realistic heading changes
   - Add `turning_correlation` parameter to OU process
5. **SpikeTrain class** - Wrap spike arrays with metadata and methods

**Priority**: Implement based on user demand. Phase 3 features (convenience functions) provide more immediate value.

### Phase 3.5: Documentation Integration (0.5 weeks)

**Goal**: Replace hand-written simulation code in example notebooks with simulation subpackage functions

Execute this phase after Phase 3 (when high-level convenience functions are available). The notebooks currently contain 200+ lines of duplicated simulation code that can be replaced with simple function calls.

**Integration Priorities**:

1. **11_place_field_analysis.ipynb** (HIGH PRIORITY)
   - **Section "2D Random Walk Generation"** (~60 lines)
     - Replace with: `simulate_trajectory_ou(env, duration=100.0, speed_mean=0.025, coherence_time=0.7)`
   - **Section "Place Cell Simulation"** (~20 lines)
     - Replace with: `PlaceCellModel(env, center=[60, 50], width=10.0)` + `generate_poisson_spikes()`
   - **Section "T-maze Trajectory"** (~100 lines)
     - Replace with: `tmaze_alternation_session(n_trials=25, duration=100.0)`
   - **Impact**: Reduces notebook by ~180 lines, improves maintainability
   - **Benefits**: Shows recommended simulation workflow, demonstrates integration with analysis functions

2. **08_spike_field_basics.ipynb** (HIGH PRIORITY)
   - **Section "Random Walk"** (~50 lines)
     - Replace with: `simulate_trajectory_ou(env, duration=60.0, speed_mean=0.025, coherence_time=0.7)`
   - **Section "Spike Generation"** (~15 lines)
     - Replace with: `PlaceCellModel(env, center=[60, 30], width=10.0)` + `generate_poisson_spikes()`
   - **Impact**: Simplifies introductory notebook, focuses on spike field analysis
   - **Benefits**: New users see clean API immediately, demonstrates `compute_place_field()` usage

3. **12_boundary_cell_analysis.ipynb** (MEDIUM PRIORITY)
   - **Section "Random Walk"** (~50 lines)
     - Replace with: `simulate_trajectory_ou(env, duration=100.0, speed_mean=0.025, coherence_time=0.7)`
   - **Section "Boundary Cell Firing"** (manual firing rates)
     - Replace with: `BoundaryCellModel(env, preferred_distance=0, distance_tolerance=10.0)` when available
   - **Impact**: Reduces trajectory code; full integration requires BoundaryCellModel implementation

4. **13_trajectory_analysis.ipynb** (LOW PRIORITY)
   - **Section "Random Walk + Goal-Directed"** (~60 lines)
     - Potential replacement: `simulate_trajectory_ou()` with state transitions (Phase 4 feature)
   - **Impact**: Lower priority; notebook focuses on trajectory metrics, not neural simulation
   - **Note**: Current code demonstrates two-phase behavior not yet supported by simulation subpackage

**Implementation Tasks**:

1. Update **11_place_field_analysis.ipynb**:
   - Replace random walk section with `simulate_trajectory_ou()`
   - Replace place cell generation with `PlaceCellModel` + `generate_poisson_spikes()`
   - Replace T-maze section with `tmaze_alternation_session()`
   - Add brief markdown explanation of simulation functions

2. Update **08_spike_field_basics.ipynb**:
   - Replace random walk with `simulate_trajectory_ou()`
   - Replace spike generation with simulation API
   - Add note directing to simulation subpackage documentation

3. Update **12_boundary_cell_analysis.ipynb**:
   - Replace random walk with `simulate_trajectory_ou()`
   - Add note about `boundary_cell_session()` example

4. Create **15_simulation_workflows.ipynb**:
   - Demonstrate all pre-configured examples (open_field_session, linear_track_session, etc.)
   - Show low-level API vs high-level API comparison
   - Demonstrate validation workflow with `validate_simulation()`
   - Show how to customize models (direction-selective, speed-gated, etc.)

5. Sync notebooks between directories:
   - Copy updated notebooks from examples/ to docs/examples/
   - Ensure documentation build picks up new simulation examples

**Validation**:
- All updated notebooks should run without errors
- Simulation code should be significantly shorter and clearer
- New users should see best practices immediately
- Integration should demonstrate synergy between simulation and analysis functions

**Documentation Updates**:
- Add "Simulation" section to main README
- Link to new `examples/15_simulation_workflows.ipynb` notebook
- Update API reference to include simulation subpackage
- Add migration guide for users with existing hand-written simulation code

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

## Implementation Notes for Developers

### Critical Technical Considerations

This section summarizes key implementation details based on code review feedback.

#### 1. Numerical Stability

**Place Cell Gaussian Decay:**
- Clip distances at 5*width before computing exp() to avoid underflow
- Use stable form: `baseline + (max - baseline) * exp(-0.5 * (d/width)^2)`
- For float32 positions in large coordinate systems (e.g., pixels), pre-normalize coordinates

**OU Process Integration:**
- Use Euler-Maruyama scheme for SDE: `v_new = v + (-θ*v)*dt + σ*sqrt(dt)*randn()`
- Ensure dt is small enough: recommend dt <= coherence_time/10 for accuracy
- Speed clipping should use soft constraints (e.g., scale velocity vector) not hard clamps

**Grid Cell Trigonometry:**
- Wave vector magnitudes can be large for small grid_spacing → ensure float64 precision
- Normalize the sum of cosines before scaling to avoid accumulation errors

#### 2. Algorithmic Complexity

**Spike Generation with Refractory Period:**
- **MUST** use O(n) single-pass algorithm, NOT nested loop O(n²)
- Sort candidate spikes once, then filter with running last_spike_time tracker
- For 60s at 500 Hz = 30k timepoints, O(n²) would take minutes vs milliseconds

**Geodesic Distance in Place Cells:**
- Precompute distance field from center bin ONCE in __init__()
- Store as self._distance_field for O(1) lookup per position
- Geodesic distance is ~100x slower than Euclidean - document this clearly

**Boundary Detection for Boundary Cells:**
- Use `env.boundary_bins()` once, then `distance_field(connectivity, boundary_bins)`
- For directional tuning, store source_ids from distance computation
- Direction vector: `normalize(bin_centers[source_ids] - positions)`

#### 3. Units Consistency

**Required Validation:**
```python
if env.units is None:
    raise ValueError(
        "Environment units must be set before simulation. "
        "Set env.units = 'cm' (or 'm', 'mm', etc.)"
    )
```

**Auto-Conversion Pattern:**
```python
if speed_units is not None and speed_units != env.units:
    # Convert speed from speed_units to env.units
    conversion_factor = get_conversion_factor(speed_units, env.units)
    speed_mean *= conversion_factor
    speed_std *= conversion_factor
```

**Utility Functions Needed:**
- `get_conversion_factor(from_unit, to_unit) -> float`
- Support: 'm', 'cm', 'mm', 'pixels' (error for incompatible units)

#### 4. Boundary Handling Details

**Reflection Algorithm (2D example):**
```python
# 1. Detect boundary crossing
if not env.contains(new_position):
    # 2. Find nearest valid bin
    nearest_bin = env.bin_at(new_position, mode='nearest')
    boundary_point = env.bin_centers[nearest_bin]

    # 3. Compute boundary normal (gradient of distance field)
    normal = compute_boundary_normal(boundary_point, env)

    # 4. Reflect velocity: v' = v - 2(v·n)n
    v_parallel = np.dot(velocity, normal) * normal
    velocity = velocity - 2 * v_parallel

    # 5. Place position just inside boundary
    position = boundary_point - epsilon * normal
```

**Periodic Wrapping (1D example):**
```python
# Wrap position using dimension ranges
for dim in range(env.n_dims):
    range_min, range_max = env.dimension_ranges[dim]
    range_size = range_max - range_min
    position[dim] = range_min + (position[dim] - range_min) % range_size
```

#### 5. Random Seed Management

**Consistent Pattern Across All Functions:**
```python
if seed is not None:
    rng = np.random.default_rng(seed)
else:
    rng = np.random.default_rng()

# Use rng.random(), rng.normal(), etc. throughout
# NOT np.random.random() (global state)
```

**For Population Spikes:**
```python
# Ensure reproducibility: derive seeds for each neuron
if seed is not None:
    seeds = [seed + i for i in range(len(models))]
else:
    seeds = [None] * len(models)

for i, model in enumerate(models):
    spikes = generate_poisson_spikes(rates, times, refractory_period, seeds[i])
```

#### 6. Performance Benchmarks (Target)

These are recommended performance targets for typical usage on modern hardware (2020+ laptop, 4+ CPU cores, NumPy with MKL/OpenBLAS).

| Operation | Input Size | Target Time | Notes |
|-----------|------------|-------------|-------|
| `simulate_trajectory_ou()` | 60s @ 100 Hz (6k points) | < 100 ms | 2D environment |
| `PlaceCellModel.firing_rate()` (Euclidean) | 6k positions | < 10 ms | Single cell |
| `PlaceCellModel.firing_rate()` (Geodesic) | 6k positions | < 1 s | Precomputed field |
| `generate_poisson_spikes()` | 6k timepoints | < 50 ms | Including refractory filter |
| `generate_population_spikes()` | 50 cells × 6k points | < 5 s | With progress bar |
| `GridCellModel.firing_rate()` | 6k positions | < 20 ms | Vectorized cosines |

**Note**: Benchmarks measured on typical development hardware (e.g., MacBook Pro M1/M2, 2020+ Intel i7/i9). Actual performance depends on hardware, Python version, and BLAS backend (MKL vs OpenBLAS).

If implementations exceed 2x target time, investigate vectorization and algorithmic complexity.

#### 7. Testing Checklist

**For Each Neural Model:**
- [ ] Peak firing rate occurs at expected location (place center, grid vertices, boundary)
- [ ] Firing rate decays correctly with distance (Gaussian, cosine, exponential)
- [ ] Condition function properly gates firing (if applicable)
- [ ] ground_truth property returns all model parameters
- [ ] Works correctly in 1D, 2D, 3D environments (or raises clear error)

**For Trajectory Simulation:**
- [ ] All positions lie within environment (`env.contains()` returns True)
- [ ] Velocity statistics match parameters (mean speed, coherence time)
- [ ] Boundary handling works correctly (reflect, periodic, stop)
- [ ] Position and time arrays have consistent shapes
- [ ] Reproducible with same seed

**For Spike Generation:**
- [ ] Mean firing rate matches expected rate (within Poisson variance)
- [ ] Inter-spike intervals >= refractory_period (all ISIs)
- [ ] Spike times sorted in ascending order
- [ ] Reproducible with same seed
- [ ] No spikes outside time range

---

## Summary

### Key Design Decisions

1. **Environment-centric**: All simulations operate on `Environment` instances (not standalone)
2. **Protocol-based**: `NeuralModel` protocol allows extensibility
3. **Composable layers**: Separate trajectory, models, and spike generation
4. **Ground truth tracking**: All models expose `.ground_truth` for validation
5. **N-D native**: Works with any dimensionality (1D, 2D, 3D, N-D)
6. **Multiple API levels**: Low-level (composable) and high-level (convenience) APIs
7. **Units awareness**: Explicit units handling with validation and auto-conversion
8. **Performance conscious**: O(n) algorithms, precomputed fields, vectorization targets
9. **Typed returns**: Dataclasses (SimulationSession) instead of dicts for discoverability
10. **Flat imports**: Single entry point (`neurospatial.simulation`) for all public API
11. **Functions over methods**: Standalone functions, no Environment method pollution

### Benefits

1. **Testing**: Generate synthetic data to validate all neurospatial metrics
2. **Education**: Pre-configured examples make learning immediate and intuitive
3. **Benchmarking**: Test algorithm performance on controlled data with known ground truth
4. **Validation**: Automated comparison of detected vs true parameters
5. **Flexibility**: Choose between quick high-level API or fine-grained control
6. **Discoverability**: Flat imports and typed returns make API easy to learn
7. **Type Safety**: IDE autocomplete and type checkers catch errors early
8. **Maintainability**: Single source of truth for simulation code (no duplicated notebook examples)

### Implementation Roadmap

1. **Phase 1**: Core trajectory + place cells (1 week)
2. **Phase 2**: Boundary cells + extended models (1 week)
3. **Phase 3**: Grid cells + convenience functions (1.5 weeks)
4. **Phase 3.5**: Integrate into example notebooks (0.5 weeks)
5. **Phase 4** (optional): Advanced realism features based on user requests
6. **Documentation**: Tutorial notebooks and API reference

---

## References

- **RatInABox**: George et al. (2023), eLife - OU process implementation
- **Existing code**: `simulate_position()`, `simulate_place_field_firing_rate()`, etc.
- **Neuroscience**: Sargolini et al. (2006) for locomotion parameters
- **Neurospatial**: Validated metrics (Skaggs 1993, Solstad 2008, Hafting 2005)
