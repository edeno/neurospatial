"""Grid cell model with hexagonal firing patterns."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._units import default_length
from neurospatial.geometry import require_cartesian
from neurospatial.simulation.models.base import (
    _check_angle,
    _check_point,
    _check_positive,
    _check_rates,
    _model_inputs,
)

if TYPE_CHECKING:
    from neurospatial.environment import Environment


#: Default grid spacing in centimeters: the middle of the 39.1-43.0 cm spacing
#: of dorsal-most rat grid cells in Hafting et al. (2005).
GRID_SPACING_CM = 41.0


@dataclass(frozen=True, eq=False)
class GridCellModel:
    """Periodic hexagonal grid pattern model (2D only).

    Implements grid cells with hexagonal symmetry as observed in rodent
    medial entorhinal cortex (Hafting et al. 2005).

    Parameters
    ----------
    env : Environment
        The spatial environment (must be 2D and allocentric Cartesian).
    grid_spacing : float, optional
        Distance between grid peaks in environment units. This is the
        characteristic spatial scale of the grid pattern. If None, 41 cm
        expressed in ``env.units``: the middle of the 39.1-43.0 cm spacing of
        the dorsal-most grid cells in Hafting et al. (2005). An environment
        without metric length units needs an explicit ``grid_spacing``.
    grid_orientation : float, optional
        Orientation of the field lattice in radians (default: 0.0): the
        direction from a firing field to a neighboring field, from +x (East)
        counterclockwise toward +y (North). The lattice repeats every pi/3,
        so values are equivalent modulo pi/3. ``grid_properties`` reports the
        same angle for this cell's rate map.
        Different grid cells have different orientations, creating
        diverse representations across the population.
    phase_offset : NDArray[np.float64], shape (2,), optional
        Spatial phase offset in (x, y) coordinates (default: [0.0, 0.0]):
        the position of one firing field. Different phase offsets
        create complementary grid representations.
    max_rate : float, optional
        Peak firing rate in Hz (default: 20.0). An illustrative value; it is
        not taken from a published measurement.
    baseline_rate : float, optional
        Firing rate between the fields, in Hz (default: 0.01). An
        illustrative value for a cell nearly silent between its fields.

    Attributes
    ----------
    ground_truth : dict[str, Any]
        Contains 'cell_type', 'grid_spacing', 'grid_orientation',
        'phase_offset', 'max_rate', 'baseline_rate'.

    Raises
    ------
    ValueError
        If the environment is not 2D (grid cells require 2D space); if
        ``grid_spacing`` is not finite and positive, ``grid_orientation`` is
        not finite, or ``phase_offset`` is not two finite coordinates; if a
        rate is not finite, is negative, or ``baseline_rate`` exceeds
        ``max_rate``; or if ``grid_spacing`` is None and ``env.units`` is not
        a metric length unit.
    CartesianOnlyError
        If the environment holds egocentric polar coordinates.

    Notes
    -----
    The hexagonal grid pattern is the mean of three plane waves at 60°
    intervals (Solstad, Moser & Einevoll 2006):

    .. math::
        g(x) = (1/3) * \\sum_{i=1}^{3} \\cos(k_i \\cdot (x - phase\\_offset))

    where wave vectors are:

    .. math::
        k_{magnitude} = 4\\pi / (\\sqrt{3} * grid\\_spacing)
        k_1 = k_{magnitude} * [\\cos(\\alpha), \\sin(\\alpha)]
        k_2 = k_{magnitude} * [\\cos(\\alpha + 60°), \\sin(\\alpha + 60°)]
        k_3 = k_{magnitude} * [\\cos(\\alpha + 120°), \\sin(\\alpha + 120°)]

    with ``α = grid_orientation - 30°``, which points the field lattice
    along ``grid_orientation`` (see ``firing_rate``). The final firing rate
    is:

    .. math::
        rate(x) = baseline + (max\\_rate - baseline) * \\max(0, g(x))

    This produces hexagonal symmetry with peaks at grid vertices.

    **Grid Spacing Selection**: Hafting et al. (2005) measured spacings of
    39 to 73 cm across rat grid cells, increasing from dorsal to ventral
    medial entorhinal cortex; dorsal-most cells were spaced 39.1-43.0 cm
    apart. The arena must span several spacings for the hexagonal pattern
    to show.

    Grid cells typically have multiple scales (modules) in biological systems.
    For realistic populations, create multiple GridCellModel instances with
    spacing ratios of approximately √2 between scales.

    **Range**: the mean of the three cosines lies in [-0.5, 1]: it is 1 at
    a field and -0.5 at the points farthest from the fields. After
    rectification, ``max(0, g(x))`` lies in [0, 1], so the rate lies
    between ``baseline_rate`` and ``max_rate``.

    References
    ----------
    - Hafting, T., Fyhn, M., Molden, S., Moser, M.-B., & Moser, E. I.
      (2005). Microstructure of a spatial map in the entorhinal cortex.
      Nature, 436(7052), 801-806.
    - Solstad, T., Moser, E. I., & Einevoll, G. T. (2006). From grid cells
      to place cells: a mathematical model. Hippocampus, 16(12), 1026-1031.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_ou, GridCellModel
    >>> # Create 2D environment
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env = env.with_metadata(units="cm")
    >>>
    >>> # Create grid cell
    >>> gc = GridCellModel(env, grid_spacing=50.0, grid_orientation=0.0)
    >>>
    >>> # Generate trajectory and compute rates
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=60.0, seed=42, speed_units="cm"
    ... )
    >>> rates = gc.firing_rate(times, positions)
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

    env: Environment = field(repr=False)
    _: KW_ONLY
    grid_spacing: float | None = None
    grid_orientation: float = 0.0
    phase_offset: NDArray[np.float64] | None = None
    max_rate: float = 20.0
    baseline_rate: float = 0.01
    _wave_vectors: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        env = self.env
        if env.n_dims != 2:
            raise ValueError(
                f"GridCellModel only works for 2D environments. "
                f"Got {env.n_dims}D environment. "
                f"Grid cells exhibit hexagonal spatial periodicity that "
                f"requires exactly 2 spatial dimensions."
            )
        require_cartesian(
            env.coordinate_system,
            "GridCellModel()",
            why="A grid cell's hexagonal lattice lies in the animal's "
            "allocentric (x, y) plane, not in (distance, angle) coordinates.",
            fix="simulate on the allocentric environment the animal moves in",
        )
        spacing = self.grid_spacing
        if spacing is None:
            spacing = default_length(
                env, GRID_SPACING_CM, name="grid_spacing", call="GridCellModel()"
            )
        _check_positive("grid_spacing", spacing)
        _check_angle("grid_orientation", self.grid_orientation)
        phase = _check_point(
            "phase_offset",
            [0.0, 0.0] if self.phase_offset is None else self.phase_offset,
            2,
        )
        _check_rates(self.max_rate, self.baseline_rate)

        # Three wave vectors at 60° intervals. The field lattice lies 30°
        # from the first wave (see firing_rate Notes), so the waves start at
        # grid_orientation - 30° to put the lattice at grid_orientation.
        k_magnitude = (4.0 * np.pi) / (np.sqrt(3.0) * spacing)
        angles = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
        angles += self.grid_orientation - np.pi / 6.0
        wave_vectors = k_magnitude * np.column_stack([np.cos(angles), np.sin(angles)])
        wave_vectors.setflags(write=False)
        for name, value in (
            ("grid_spacing", float(spacing)),
            ("grid_orientation", float(self.grid_orientation)),
            ("phase_offset", phase),
            ("max_rate", float(self.max_rate)),
            ("baseline_rate", float(self.baseline_rate)),
            ("_wave_vectors", wave_vectors),
        ):
            object.__setattr__(self, name, value)

    @property
    def requires_headings(self) -> bool:
        """False: a grid cell's rate depends on position only."""
        return False

    def firing_rate(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute hexagonal grid firing rate.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_time,)
            Sample times in seconds; the rate does not depend on them.
        positions : NDArray[np.float64], shape (n_time, 2)
            Position coordinates in 2D space.
        headings : NDArray[np.float64], shape (n_time,), optional
            Unused; accepted for the ``NeuralModel`` interface.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each position; NaN for a missing position.

        Raises
        ------
        ValueError
            If ``positions`` is not one row of 2 coordinates per time.

        Notes
        -----
        Implements hexagonal grid pattern using three plane waves:

        1. Compute wave vector magnitude:
           k = 4π / (√3 * grid_spacing)

        2. Compute three wave vectors at 60° intervals:
           k_1 = k * [cos(α), sin(α)]
           k_2 = k * [cos(α + 60°), sin(α + 60°)]
           k_3 = k * [cos(α + 120°), sin(α + 120°)]

           where α = grid_orientation - 30°. A field sits where every
           k_i · x is a multiple of 2π. The vector a of length grid_spacing at
           angle α + 30° gives k_1 · a = k_2 · a = k |a| cos(30°) = 2π and
           k_3 · a = 0, so the field lattice points along α + 30° =
           grid_orientation (and every 60° from it).

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
        >>> env = env.with_metadata(units="cm")
        >>> gc = GridCellModel(env, grid_spacing=40.0)
        >>> positions = np.array([[50.0, 50.0], [55.0, 50.0]])
        >>> rates = gc.firing_rate(np.array([0.0, 0.1]), positions)
        >>> rates.shape == (2,)
        True
        >>> bool(np.all(rates >= gc.baseline_rate))
        True
        >>> bool(np.all(rates <= gc.max_rate))
        True
        """
        _times, positions = _model_inputs(
            times, positions, 2, call="GridCellModel.firing_rate()"
        )
        # The mean of three cosine gratings, g(x) = (1/3) Σ cos(k_i · x):
        # rel_pos @ wave_vectors.T has shape (n_time, 3), each phase k_i · x.
        assert self.phase_offset is not None
        rel_pos = positions - self.phase_offset
        grid_pattern = np.cos(rel_pos @ self._wave_vectors.T).mean(axis=1)

        # Rectify: only positive parts contribute to firing
        grid_pattern = np.maximum(0.0, grid_pattern)
        rates: NDArray[np.float64] = (
            self.baseline_rate + (self.max_rate - self.baseline_rate) * grid_pattern
        )
        return rates

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Contains:
            - 'cell_type': 'grid'
            - 'grid_spacing': float - distance between peaks
            - 'grid_orientation': float - field lattice angle in radians,
              from +x counterclockwise, modulo pi/3
            - 'phase_offset': NDArray - spatial phase offset (read-only)
            - 'max_rate': float - peak firing rate
            - 'baseline_rate': float - baseline firing rate
        """
        return {
            "cell_type": "grid",
            "grid_spacing": self.grid_spacing,
            "grid_orientation": self.grid_orientation,
            "phase_offset": self.phase_offset,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }
