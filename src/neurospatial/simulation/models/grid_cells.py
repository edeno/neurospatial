"""Grid cell model with hexagonal firing patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment


class GridCellModel:
    """Periodic hexagonal grid pattern model (2D only).

    Implements grid cells with hexagonal symmetry as observed in rodent
    medial entorhinal cortex (Hafting et al. 2005).

    Parameters
    ----------
    env : Environment
        The spatial environment (must be 2D).
    grid_spacing : float
        Distance between grid peaks in environment units (default: 50.0 cm).
        This is the characteristic spatial scale of the grid pattern.
    grid_orientation : float, optional
        Rotation of grid in radians (default: 0.0).
        Different grid cells have different orientations, creating
        diverse representations across the population.
    phase_offset : NDArray[np.float64], shape (2,), optional
        Spatial phase offset in (x, y) coordinates (default: [0.0, 0.0]).
        Controls the position of grid vertices. Different phase offsets
        create complementary grid representations.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0).
    baseline_rate : float, optional
        Baseline firing rate outside peaks (default: 0.1 Hz).

    Attributes
    ----------
    ground_truth : dict
        Contains 'grid_spacing', 'grid_orientation', 'phase_offset',
        'max_rate', 'baseline_rate'.

    Raises
    ------
    ValueError
        If environment is not 2D (grid cells require 2D space).

    Notes
    -----
    The hexagonal grid pattern is generated using three plane waves
    at 60° intervals (Fuhs & Touretzky 2006):

    .. math::
        g(x) = (1/3) * \\sum_{i=1}^{3} \\cos(k_i \\cdot (x - phase\\_offset))

    where wave vectors are:

    .. math::
        k_{magnitude} = 4\\pi / (\\sqrt{3} * grid\\_spacing)
        k_1 = k_{magnitude} * [\\cos(\\theta), \\sin(\\theta)]
        k_2 = k_{magnitude} * [\\cos(\\theta + 60°), \\sin(\\theta + 60°)]
        k_3 = k_{magnitude} * [\\cos(\\theta + 120°), \\sin(\\theta + 120°)]

    The final firing rate is:

    .. math::
        rate(x) = baseline + (max\\_rate - baseline) * \\max(0, g(x))

    This produces hexagonal symmetry with peaks at grid vertices.

    **Grid Spacing Selection**: Choose grid_spacing based on environment size
    and analysis goals:

    - Small arenas (< 1m): Use 10-30 cm spacing
    - Medium arenas (1-2m): Use 30-60 cm spacing
    - Large arenas (> 2m): Use 60-100 cm spacing

    Grid cells typically have multiple scales (modules) in biological systems.
    For realistic populations, create multiple GridCellModel instances with
    spacing ratios of approximately √2 between scales.

    **Performance**: Grid pattern computation is O(n_positions) with low
    constant factor (3 cosine evaluations per position). Much faster than
    geodesic distance metrics in PlaceCellModel or BoundaryCellModel.

    **Numerical Stability**: The cosine gratings are bounded [-1, 1], so
    grid_pattern is bounded [-1, 1]. After rectification (max(0, g(x))),
    the pattern is bounded [0, 1], preventing overflow/underflow.

    References
    ----------
    - Hafting et al. (2005). "Microstructure of a spatial map in the
      entorhinal cortex." Nature 436(7052): 801-806.
    - Fuhs & Touretzky (2006). "A spin glass model of path integration
      in rat medial entorhinal cortex." Journal of Neuroscience.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_ou, GridCellModel
    >>> # Create 2D environment
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Create grid cell
    >>> gc = GridCellModel(env, grid_spacing=50.0, grid_orientation=0.0)
    >>>
    >>> # Generate trajectory and compute rates
    >>> positions, times = simulate_trajectory_ou(env, duration=60.0, seed=42)
    >>> rates = gc.firing_rate(positions)
    >>> rates.shape == positions.shape[:1]
    True
    >>> bool(np.all(rates >= gc.baseline_rate))
    True

    >>> # Grid cell module (same spacing, different phases)
    >>> module_spacing = 50.0
    >>> phases = [[0, 0], [10, 5], [20, 10], [30, 15]]
    >>> gcs = [
    ...     GridCellModel(env, grid_spacing=module_spacing, phase_offset=np.array(p))
    ...     for p in phases
    ... ]
    >>> len(gcs)
    4
    """

    def __init__(
        self,
        env: Environment,
        grid_spacing: float = 50.0,
        grid_orientation: float = 0.0,
        phase_offset: NDArray[np.float64] | None = None,
        max_rate: float = 20.0,
        baseline_rate: float = 0.1,
    ) -> None:
        """Initialize grid cell model."""
        # Validate 2D environment
        if env.n_dims != 2:
            raise ValueError(
                f"GridCellModel only works for 2D environments. "
                f"Got {env.n_dims}D environment. "
                f"Grid cells exhibit hexagonal spatial periodicity that "
                f"requires exactly 2 spatial dimensions."
            )

        # Validate parameters
        if grid_spacing <= 0:
            msg = f"grid_spacing must be positive (got {grid_spacing})"
            raise ValueError(msg)

        if max_rate <= 0:
            msg = f"max_rate must be positive (got {max_rate})"
            raise ValueError(msg)

        if baseline_rate < 0:
            msg = f"baseline_rate must be non-negative (got {baseline_rate})"
            raise ValueError(msg)

        if baseline_rate >= max_rate:
            msg = (
                f"baseline_rate ({baseline_rate}) must be less than "
                f"max_rate ({max_rate})"
            )
            raise ValueError(msg)

        self.env = env
        self.grid_spacing = grid_spacing
        self.grid_orientation = grid_orientation
        self.max_rate = max_rate
        self.baseline_rate = baseline_rate

        # Default phase offset at origin
        if phase_offset is None:
            self.phase_offset = np.array([0.0, 0.0])
        else:
            self.phase_offset = np.asarray(phase_offset, dtype=np.float64)
            # Validate phase_offset shape
            if self.phase_offset.shape != (2,):
                msg = (
                    f"phase_offset must be shape (2,), "
                    f"got shape {self.phase_offset.shape}"
                )
                raise ValueError(msg)

    def firing_rate(
        self,
        positions: NDArray[np.float64],
        times: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute hexagonal grid firing rate.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, 2)
            Position coordinates in 2D space.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points (not used for grid cells, included for protocol).

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position.

        Notes
        -----
        Implements hexagonal grid pattern using three plane waves:

        1. Compute wave vector magnitude:
           k = 4π / (√3 * grid_spacing)

        2. Compute three wave vectors at 60° intervals:
           k_1 = k * [cos(θ), sin(θ)]
           k_2 = k * [cos(θ + 60°), sin(θ + 60°)]
           k_3 = k * [cos(θ + 120°), sin(θ + 120°)]

           where θ = grid_orientation

        3. Compute grid pattern:
           g(x) = (1/3) * Σ cos(k_i · (x - phase_offset))

        4. Apply rectification and scaling:
           rate(x) = baseline + (max - baseline) * max(0, g(x))

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.simulation import GridCellModel
        >>> samples = np.random.uniform(0, 100, (500, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0)
        >>> env.units = "cm"
        >>> gc = GridCellModel(env, grid_spacing=40.0)
        >>> positions = np.array([[50.0, 50.0], [55.0, 50.0]])
        >>> rates = gc.firing_rate(positions)
        >>> rates.shape == (2,)
        True
        >>> bool(np.all(rates >= gc.baseline_rate))
        True
        >>> bool(np.all(rates <= gc.max_rate))
        True
        """
        # Ensure positions are 2D array
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        # Relative positions from phase offset
        rel_pos = positions - self.phase_offset

        # Wave vector magnitude for hexagonal grid
        # k = 4π / (√3 * λ) where λ is grid spacing
        k_magnitude = (4.0 * np.pi) / (np.sqrt(3.0) * self.grid_spacing)

        # Three wave vectors at 60° intervals, rotated by grid_orientation
        angles = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])  # 0°, 60°, 120°
        angles += self.grid_orientation

        # Compute wave vectors: k_i = k_magnitude * [cos(angle_i), sin(angle_i)]
        wave_vectors = k_magnitude * np.column_stack([np.cos(angles), np.sin(angles)])

        # Compute grid pattern as sum of three cosine gratings
        # g(x) = (1/3) * Σ cos(k_i · x)
        grid_pattern = np.zeros(len(rel_pos))
        for k_vec in wave_vectors:
            # Dot product: k · x
            phase = np.dot(rel_pos, k_vec)
            grid_pattern += np.cos(phase)

        grid_pattern /= 3.0  # Average of three gratings

        # Rectify: only positive parts contribute to firing
        grid_pattern = np.maximum(0.0, grid_pattern)

        # Scale to firing rate range
        rates = self.baseline_rate + (self.max_rate - self.baseline_rate) * grid_pattern

        return rates

    @property
    def ground_truth(self) -> dict:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict
            Contains:
            - 'grid_spacing': float - distance between peaks
            - 'grid_orientation': float - grid rotation in radians
            - 'phase_offset': NDArray - spatial phase offset
            - 'max_rate': float - peak firing rate
            - 'baseline_rate': float - baseline firing rate
        """
        return {
            "grid_spacing": self.grid_spacing,
            "grid_orientation": self.grid_orientation,
            "phase_offset": self.phase_offset,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }
