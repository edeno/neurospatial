"""Place cell models for simulating spatial firing patterns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._units import default_length
from neurospatial.environment import Environment
from neurospatial.geometry import require_cartesian
from neurospatial.ops._types import DistanceMetric
from neurospatial.ops.distance import distance_field
from neurospatial.simulation.models.base import (
    _check_metric,
    _check_point,
    _check_rates,
    _model_inputs,
    _read_only,
    geodesic_lookup_bins,
)

#: Default place field width (Gaussian standard deviation) in centimeters,
#: derived from the median rat CA1 open-field field size of Mizuseki et al.
#: (2012); see ``PlaceCellModel`` Notes.
PLACE_FIELD_WIDTH_CM = 11.1

#: Default peak rate in Hz: rat CA1 place cells in the open field (Mizuseki
#: et al. 2012).
PLACE_FIELD_PEAK_HZ = 12.4

#: Place field width in centimeters on a linear track, derived from the
#: 75 cm CA1 field length of Mizuseki et al. (2012) on their 250 cm track;
#: see ``PlaceCellModel`` Notes. The track session helpers use it.
TRACK_PLACE_FIELD_WIDTH_CM = 17.5

#: Peak rate in Hz of the same CA1 cells on the linear track.
TRACK_PLACE_FIELD_PEAK_HZ = 14.5


@dataclass(frozen=True, eq=False)
class PlaceCellModel:
    """Gaussian place field model.

    A place field in an N-D environment, with Euclidean or geodesic
    distances to its center and an optional firing gate (direction-selective,
    speed-gated, etc.). The model is immutable: build a new one to change a
    parameter.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    center : NDArray[np.float64], shape (n_dims,), optional
        Place field center in continuous coordinates.
        If None, randomly chosen from env.bin_centers with ``seed``.
    width : float | NDArray[np.float64], optional
        Place field width (standard deviation of Gaussian), in environment
        units. Can be scalar (isotropic) or one value per dimension
        (anisotropic), shape (n_dims,).
        If None, 11.1 cm expressed in ``env.units``: the size of a median
        rat CA1 open-field place field (see Notes). It does not depend on the
        environment's bins. An environment without metric length units
        needs an explicit ``width``.
    max_rate : float, optional
        Peak firing rate in Hz (default: 12.4, the peak rate of rat CA1
        place cells in the open field; Mizuseki et al. 2012).
    baseline_rate : float, optional
        Firing rate far from the field, in Hz (default: 0.01). An
        illustrative value for a cell that is nearly silent outside its
        field; it is not taken from a published measurement.
    metric : {'euclidean', 'geodesic'}, optional
        Method for computing distance from positions to field center.

        - 'euclidean': Straight-line distance (default); ignores barriers.
          Use for open field environments.
          On an egocentric polar environment, ``center`` and positions are
          ``(distance, angle)`` rows and the distance between them is measured
          in the animal's plane, ``(d cos(angle), d sin(angle))``, so ``width``
          is a length; a per-axis ``width`` is refused there.
        - 'geodesic': Path distance through environment connectivity graph,
          for environments with barriers/walls. The distance field is
          computed once, at construction.
    condition : Callable[[NDArray, NDArray], NDArray[bool]] | None, optional
        Optional condition function: (positions, times) → bool mask, one
        entry per sample. Where the mask is False the rate is 0.

        Common patterns:

        - Direction-selective: ``lambda pos, t: np.gradient(pos[:, 0]) > 0``
        - Speed-gated: ``lambda pos, t: compute_speed(t, pos) > threshold``
        - Region-specific: ``lambda pos, t: (pos[:, 0] > x_min) & (pos[:, 0] < x_max)``
        - Trial-type: ``lambda pos, t: is_correct_trial``, where
          ``is_correct_trial`` is a bool array with one entry per sample
    seed : int | None, optional
        Random seed for reproducible random center selection.

    Attributes
    ----------
    env : Environment
        The spatial environment.
    center : NDArray[np.float64], shape (n_dims,)
        Place field center coordinates (read-only).
    width : float | NDArray[np.float64]
        Place field width(s); a per-dimension width is read-only.
    max_rate : float
        Peak firing rate in Hz.
    baseline_rate : float
        Baseline firing rate in Hz.
    metric : str
        Distance metric used ('euclidean' or 'geodesic').
    condition : Callable | None
        Condition function for gated firing.

    Raises
    ------
    ValueError
        If ``center`` is not ``n_dims`` finite coordinates; if ``width`` is
        not finite and strictly positive (it is the Gaussian standard
        deviation, so a zero width gives an undefined 0/0 rate at the
        center), or a per-dimension ``width`` does not have ``n_dims``
        entries; if a rate is not finite, is negative, or ``baseline_rate``
        exceeds ``max_rate``; if ``metric`` is unknown; if a per-dimension
        ``width`` is combined with ``metric='geodesic'``; or if ``width`` is
        None and ``env.units`` is not a metric length unit.
    CartesianOnlyError
        If ``metric='geodesic'`` or a per-dimension ``width`` is used on an
        egocentric polar environment.

    Examples
    --------
    Simple place cell in 2D arena:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import PlaceCellModel, simulate_trajectory_ou
    >>> import numpy as np
    >>>
    >>> # Create environment
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env = env.with_metadata(units="cm")
    >>>
    >>> # Create place cell at center of environment
    >>> center = env.bin_centers[len(env.bin_centers) // 2]
    >>> pc = PlaceCellModel(env, center=center, width=10.0, max_rate=25.0)
    >>>
    >>> # Generate trajectory and compute firing rates
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=10.0, seed=42, speed_units="cm"
    ... )
    >>> rates = pc.firing_rate(times, positions)
    >>> rates.shape[0] == len(times)
    True
    >>> bool(rates.max() > 0.0)  # Should have some firing
    True

    Direction-selective place cell (fires only when moving right):

    >>> def rightward_only(positions, times):
    ...     if len(positions) < 2:
    ...         return np.ones(len(positions), dtype=bool)
    ...     velocity = np.gradient(positions[:, 0])
    ...     return velocity > 0
    >>>
    >>> pc_directional = PlaceCellModel(
    ...     env,
    ...     center=[50.0, 50.0],
    ...     width=10.0,
    ...     condition=rightward_only,
    ... )
    >>> rates_directional = pc_directional.firing_rate(times, positions)

    Multiple place cells tiling the environment:

    >>> # Create place cells at every 10th bin center
    >>> place_cells = [
    ...     PlaceCellModel(env, center=c, width=8.0) for c in env.bin_centers[::10]
    ... ]
    >>>
    >>> # Compute all firing rates
    >>> all_rates = np.column_stack(
    ...     [pc.firing_rate(times, positions) for pc in place_cells]
    ... )
    >>> all_rates.shape[0] == len(times)  # (n_time, n_cells)
    True
    >>> all_rates.shape[1] == len(place_cells)
    True

    Geodesic distance for environment with barriers:

    >>> pc_geodesic = PlaceCellModel(
    ...     env,
    ...     center=[50.0, 50.0],
    ...     width=10.0,
    ...     metric="geodesic",
    ... )
    >>> rates_geodesic = pc_geodesic.firing_rate(times, positions)

    See Also
    --------
    BorderCellModel : Wall-distance tuned cells
    GridCellModel : Hexagonal grid patterns
    simulate_trajectory_ou : Generate realistic trajectories
    generate_poisson_spikes : Convert rates to spike times

    Notes
    -----
    **Default field**: Mizuseki et al. (2012) report, for rat CA1 pyramidal
    cells in a 180 × 180 cm open field, a peak firing rate of 12.4 Hz and a
    median place field size of 1,775 cm², a field being the region where the
    rate exceeds 10% of its peak. A Gaussian field exceeds 10% of its peak
    inside a disk of radius ``σ √(2 ln 10)``, of area ``2π ln(10) σ²``, so
    that median area gives ``σ = √(1775 / (2π ln 10)) = 11.1 cm``. On their
    250 cm linear track the same cells had longer fields (75 cm above 10% of
    peak, ``σ = 75 / (2 √(2 ln 10)) = 17.5 cm``) and a 14.5 Hz peak; pass
    those values to model a track.

    **Geodesic Distance**: For geodesic metric, the distance field is
    computed at construction from graph shortest paths, and each rate looks
    up its position's bin in it.

    **Condition Function**: The condition function receives the full trajectory
    and should return a boolean mask. It's called once per firing_rate() invocation,
    so it can use temporal information (e.g., velocity, acceleration).

    References
    ----------
    Mizuseki, K., Royer, S., Diba, K., & Buzsáki, G. (2012). Activity dynamics
        and behavioral correlates of CA3 and CA1 hippocampal pyramidal neurons.
        Hippocampus, 22(8), 1659-1680.
    """

    env: Environment = field(repr=False)
    _: KW_ONLY
    center: NDArray[np.float64] | None = None
    width: float | NDArray[np.float64] | None = None
    max_rate: float = PLACE_FIELD_PEAK_HZ
    baseline_rate: float = 0.01
    metric: DistanceMetric = "euclidean"
    condition: (
        Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.bool_]] | None
    ) = None
    seed: InitVar[int | None] = None
    _center_place: NDArray[np.float64] = field(init=False, repr=False)
    _distance_field: NDArray[np.float64] | None = field(init=False, repr=False)

    def __post_init__(self, seed: int | None) -> None:
        env = self.env
        _check_rates(self.max_rate, self.baseline_rate)
        _check_metric(self.metric, "PlaceCellModel()")
        object.__setattr__(self, "max_rate", float(self.max_rate))
        object.__setattr__(self, "baseline_rate", float(self.baseline_rate))
        if self.center is None:
            center_idx = np.random.default_rng(seed).integers(0, env.n_bins)
            center = _read_only(env.bin_centers[center_idx])
        else:
            center = _check_point("center", self.center, env.n_dims)
        object.__setattr__(self, "center", center)

        # A field's size belongs to the cell, not to the analysis bins.
        width_given = (
            default_length(
                env, PLACE_FIELD_WIDTH_CM, name="width", call="PlaceCellModel()"
            )
            if self.width is None
            else self.width
        )
        width_check = np.asarray(width_given, dtype=np.float64)
        if not np.all(np.isfinite(width_check)) or np.any(width_check <= 0.0):
            raise ValueError(
                _format_error(
                    f"width must be finite and strictly positive, got {width_given!r}.",
                    why="width is the Gaussian standard deviation of the place "
                    "field; a zero or negative width gives an undefined (0/0) "
                    "firing rate at the field center.",
                    fix="pass the field's standard deviation in the "
                    "environment's units, e.g. width=10.0",
                )
            )
        if width_check.ndim > 0 and width_check.shape != (env.n_dims,):
            raise ValueError(
                _format_error(
                    f"A per-dimension width needs {env.n_dims} values, one per "
                    f"dimension, got shape {width_check.shape}.",
                    fix="pass one width for an isotropic field, or "
                    f"{env.n_dims} widths in the order of the environment's axes",
                )
            )
        width: float | NDArray[np.float64]
        if width_check.ndim == 0 or env.n_dims == 1:
            width = float(width_check.reshape(-1)[0])
        else:
            width = _read_only(width_check)
            if env.coordinate_kind == "polar":
                require_cartesian(
                    env.coordinate_system,
                    "PlaceCellModel(width=<one per axis>)",
                    why="A per-axis width scales distance and angle separately, "
                    "but the field is a Gaussian of the physical distance "
                    "between places.",
                    fix="give one width in the environment's length units, for "
                    "example the distance-axis width",
                    example="neurospatial.simulation.PlaceCellModel(env, "
                    "center=center, width=float(width[0]))",
                )
            if self.metric == "geodesic":
                raise ValueError(
                    _format_error(
                        "A per-dimension width is not supported with "
                        "metric='geodesic'.",
                        why="a geodesic distance is one path length through "
                        "the bins, with no per-axis components.",
                        fix="pass one width, or metric='euclidean'",
                    )
                )
        object.__setattr__(self, "width", width)

        # Distances are measured between places: on an egocentric polar
        # environment, (distance, angle) rows are placed in the animal's plane.
        object.__setattr__(
            self,
            "_center_place",
            env.coordinate_system.cartesian_embedding(center[None, :])[0],
        )

        distances: NDArray[np.float64] | None = None
        if self.metric == "geodesic":
            require_cartesian(
                env.coordinate_system,
                "PlaceCellModel(metric='geodesic')",
                why="It locates the field center and the animal's allocentric "
                "Cartesian positions in the environment's bins.",
                fix="simulate on the allocentric environment the animal moves in",
            )
            center_bin = int(geodesic_lookup_bins(env, center[None, :])[0])
            distances = distance_field(env.connectivity, sources=[center_bin])
            distances.setflags(write=False)
        object.__setattr__(self, "_distance_field", distances)

    @property
    def requires_headings(self) -> bool:
        """False: a place cell's rate depends on position only."""
        return False

    def firing_rate(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute Gaussian place field firing rate.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_time,)
            Sample times in seconds, passed to ``condition``.
        positions : NDArray[np.float64], shape (n_time, n_dims)
            Position coordinates in the same units as the environment.
        headings : NDArray[np.float64], shape (n_time,), optional
            Unused; accepted for the ``NeuralModel`` interface.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each sample; NaN for a missing position.

        Raises
        ------
        ValueError
            If ``positions`` is not one row of ``env.n_dims`` coordinates per
            time, or ``condition`` does not return one value per sample.

        Notes
        -----
        Implementation:

        1. Compute distance from positions to center (euclidean or geodesic)
        2. Apply the Gaussian:
           rate = baseline + (max - baseline) * exp(-0.5 * (d/width)^2)

        3. Apply condition mask if provided (multiply rates by mask)

        For geodesic distance, each position takes its containing bin, or
        the nearest bin center for a position outside, and reads the
        distance field computed at construction.
        """
        times, positions = _model_inputs(
            times, positions, self.env.n_dims, call="PlaceCellModel.firing_rate()"
        )
        if isinstance(self.width, np.ndarray):
            # Anisotropic (Cartesian only, so the center is its own place): the
            # distance in standard-deviation units, sqrt(sum_i ((x_i - c_i) / w_i)^2).
            sigmas = np.linalg.norm(
                (positions - self._center_place) / self.width, axis=1
            )
        elif self.metric == "euclidean":
            # Straight-line distance between places (in the animal's plane on
            # an egocentric polar environment).
            embedded = self.env.coordinate_system.cartesian_embedding(positions)
            sigmas = np.linalg.norm(embedded - self._center_place, axis=1) / self.width
        else:
            assert self._distance_field is not None
            bin_indices = geodesic_lookup_bins(self.env, positions)
            # A missing position has no distance, so its rate stays NaN.
            distances = np.where(
                bin_indices >= 0, self._distance_field[bin_indices], np.nan
            )
            sigmas = distances / self.width

        # Far from the field the exponential underflows to 0 (and an
        # unreachable geodesic distance, inf, gives exactly 0), so the rate
        # settles on the baseline without any clipping.
        gaussian = np.exp(-0.5 * sigmas**2)
        rates: NDArray[np.float64] = (
            self.baseline_rate + (self.max_rate - self.baseline_rate) * gaussian
        )

        if self.condition is not None:
            mask = np.asarray(self.condition(positions, times))
            if mask.shape != rates.shape:
                raise ValueError(
                    _format_error(
                        f"PlaceCellModel condition returned shape {mask.shape} "
                        f"for {len(times)} samples.",
                        fix="return one bool per sample from condition("
                        "positions, times)",
                    )
                )
            rates = rates * mask.astype(np.float64)
        return rates

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Dictionary with keys:

            - 'cell_type': 'place'
            - 'center': NDArray[np.float64] - field center coordinates
              (read-only)
            - 'width': float | NDArray[np.float64] - field width
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz
            - 'metric': str - distance metric
            - 'condition': Callable | None - firing gate

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.simulation import PlaceCellModel
        >>> samples = np.random.uniform(0, 100, (100, 2))
        >>> env = Environment.from_samples(samples, bin_size=2.0)
        >>> env = env.with_metadata(units="cm")
        >>> pc = PlaceCellModel(env, center=[50.0, 50.0], width=10.0, max_rate=20.0)
        >>> np.allclose(pc.ground_truth["center"], [50.0, 50.0])
        True
        >>> pc.ground_truth["width"]
        10.0
        >>> pc.ground_truth["max_rate"]
        20.0
        """
        return {
            "cell_type": "place",
            "center": self.center,
            "width": self.width,
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
            "metric": self.metric,
            "condition": self.condition,
        }
