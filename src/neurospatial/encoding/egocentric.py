"""Object-vector rate maps in allocentric and egocentric reference frames.

Allocentric functions measure animal-to-object direction in world coordinates
(0 = East, +pi/2 = North), following Høydal et al. (2019). Egocentric functions
require headings and measure bearing relative to the animal (0 = ahead,
+pi/2 = left), following Wang et al. (2018). Both use nearest-object distance,
Euclidean or geodesic, and return ObjectVectorRateResult/ObjectVectorRatesResult
with a required ``direction_frame``. Add pi and wrap to obtain the reverse,
object-to-animal vector. Firing by distance and direction to walls
(boundary-vector and egocentric boundary cells; Alexander et al., 2020) is in
:mod:`neurospatial.encoding.boundary_vector`.

Examples
--------
>>> import numpy as np
>>> from neurospatial.encoding import compute_object_vector_rate
>>> rng = np.random.default_rng(42)
>>> times = np.arange(0, 40, 0.04)
>>> positions = rng.uniform(10, 90, (len(times), 2))
>>> spikes = np.sort(rng.uniform(0, 39.9, 100))
>>> result = compute_object_vector_rate(
...     None,
...     spikes,
...     times,
...     positions,
...     [[50, 50]],
...     distance_range=(0.0, 50.0),
...     distance_bin_size=5.0,
... )
>>> result.direction_frame
'allocentric'

References
----------
Høydal, Ø. A., et al. (2019). Object-vector coding in the medial entorhinal
    cortex. Nature, 568, 400-404. doi:10.1038/s41586-019-1077-7.
Wang, C., et al. (2018). Egocentric coding of external items in the lateral
    entorhinal cortex. Science, 362, 945-949. doi:10.1126/science.aau4940.
Deshmukh, S. S., & Knierim, J. J. (2011). Representation of non-spatial and
    spatial information in the lateral entorhinal cortex. Frontiers in
    Behavioral Neuroscience, 5, 69.
Alexander, A. S., et al. (2020). Egocentric boundary vector tuning of the
    retrosplenial cortex. Science Advances, 6, eaaz2322.

See Also
--------
neurospatial.encoding.spatial : Spatial rate computation for place cells
neurospatial.ops.egocentric : Egocentric coordinate transforms
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    hold_last_valid,
    interval_valid_mask,
    resolve_time_windows,
    run_time_bounds,
    start_allocated_occupancy,
)
from neurospatial._observation import validate_break_before
from neurospatial._results import _copy_analysis_settings, _copy_unit_table
from neurospatial.encoding._base import (
    PopulationRateMixin,
    SingleUnitRateMixin,
    _to_numpy,
)
from neurospatial.encoding._polar_results import (
    DEFAULT_DIRECTION_BIN_SIZE,
    _PolarRateMixin,
    polar_summary_frame,
    require_known_bandwidth,
    resolve_polar_grid,
)
from neurospatial.encoding._significance import (
    _SHUFFLE_DEFAULTS,
    check_criterion,
    check_mode_keywords,
    resolve_shuffle_settings,
)
from neurospatial.encoding._smoothing import _warn_if_fully_masked
from neurospatial.encoding._spatial_counts import (
    _warn_if_population_silent,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.projections.polar import PolarAxes

    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.environment import Environment
    from neurospatial.geometry import EgocentricPolarSystem
    from neurospatial.ops._types import BackendType, DistanceMetric, ResolvedBackend

    PolarEnvironment = Environment[EgocentricPolarSystem]
    from neurospatial.stats.shuffle import ShuffleTestResult

# Object coordinates, fixed or per sample, or names of ``env.landmarks``.
ObjectPositions: TypeAlias = ArrayLike | Sequence[str]


__all__ = [
    # Result classes
    "ObjectVectorRateResult",
    "ObjectVectorRatesResult",
    # Compute functions
    "compute_egocentric_rate",
    "compute_egocentric_rates",
    "compute_object_vector_rate",
    "compute_object_vector_rates",
    # Convenience functions
    "egocentric_object_vector_cell_significance",
    "is_egocentric_object_vector_cell",
    "is_object_vector_cell",
    "object_vector_cell_significance",
    "object_vector_score",
    "plot_object_vector_tuning",
]


OBJECT_VECTOR_THRESHOLDS = MappingProxyType({"min_info": 0.3})


def _check_direction_frame(direction_frame: str, *, context: str) -> None:
    """Reject a direction frame that downstream code would misread."""
    if direction_frame not in ("allocentric", "egocentric"):
        raise ValueError(
            _format_error(
                f"{context}: unknown direction_frame={direction_frame!r}.",
                why=(
                    "Why: preferred directions, plots and exports interpret "
                    "angles in the recorded frame."
                ),
                fix='pass direction_frame="allocentric" or direction_frame="egocentric"',
            )
        )


def _object_vector_classify(
    result: ObjectVectorRateResult | ObjectVectorRatesResult,
    *,
    min_info: float,
    bias: NDArray[np.float64] | None = None,
) -> NDArray[np.bool_]:
    """The shared rule: bias-corrected information at or above ``min_info``."""
    from neurospatial.encoding._metrics import _debiased_information

    return _debiased_information(result, bias) >= min_info


def _object_vector_summary_frame(
    result: ObjectVectorRateResult | ObjectVectorRatesResult,
    *,
    index: Sequence[Hashable],
) -> pd.DataFrame:
    """Build identical metric columns for single and population results.

    ``spatial_information`` and ``spatial_information_bias`` are the two
    terms of the information screen behind ``is_object_vector_cell``.
    """
    from neurospatial.encoding._metrics import _information_bias

    bias = _information_bias(result)
    return polar_summary_frame(
        result,
        index=index,
        extra={
            "spatial_information": np.atleast_1d(
                np.asarray(_to_numpy(result.spatial_information()), dtype=np.float64)
            ),
            "spatial_information_bias": bias,
        },
        classification={
            "is_object_vector_cell": _object_vector_classify(
                result, **OBJECT_VECTOR_THRESHOLDS, bias=bias
            )
        },
        extra_units={
            "spatial_information": "bits/spike",
            "spatial_information_bias": "bits/spike",
        },
        thresholds=OBJECT_VECTOR_THRESHOLDS,
    )


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class ObjectVectorRateResult(_PolarRateMixin, SingleUnitRateMixin):
    """Result of object-vector rate computation for a single neuron.

    This class wraps an object-vector firing rate map (firing rate by distance
    and direction to object) with its associated metadata. Object-vector cells
    fire when the animal is at a specific distance and direction from an object.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate in polar coordinates in the recorded frame in Hz. Shape is (n_bins,)
        where n_bins is the number of active bins in the polar environment.
        The polar environment represents a polar grid with distance on one
        axis and direction on another. Can contain NaN for bins with insufficient
        occupancy.
    occupancy : ArrayLike
        Time spent in each polar bin in seconds. Shape is (n_bins,).
    env : Environment
        The polar environment used for the computation. This is
        typically created via ``Environment.from_polar_egocentric()`` and
        represents the (distance, direction) space indexing distance and direction to objects.
    direction_frame : {"allocentric", "egocentric"}
        Required reference frame for the direction to the object.
        Allocentric: 0 = East, +pi/2 = North. Egocentric: 0 = ahead, +pi/2 = left.
    method : {"binned", "diffusion_kde", "gaussian_kde"}, default="binned"
        Estimator used. ``"binned"`` is the raw rate in each polar bin.
    bandwidth : float or None, default=None
        Kernel bandwidth in the environment's length units for the kernel
        methods; ``None`` for ``"binned"``, which does not smooth.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate by polar coordinates in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The egocentric polar environment.
    distance_range : tuple[float, float]
        (min, max) distance covered by the bins; read from ``env``.
    n_distance_bins : int
        Number of distance bins; read from ``env``.
    n_direction_bins : int
        Number of direction bins; read from ``env``.
    unit_id : int or str or None
        Identifier for this unit. Set automatically when indexing/iterating a
        population result (``rates[i].unit_id == rates.unit_ids[i]``); ``None``
        for a standalone single-unit computation.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        True when the unit labels were generated (no ``unit_ids`` or
        ``unit_id`` supplied), so pairing with spike inputs stays positional.
        Set by the compute functions and by indexing; not an analysis choice.
    n_spikes_outside_intervals : int or ndarray of int64, shape (n_units,), or None
        Each unit's spikes outside the analyzed time intervals
        (``analysis_settings["observed_intervals"]``), which are in no rate:
        an int for one unit, one count per unit for a population. ``None``
        when unknown, as for a result built by hand. Also a
        ``summary_table()`` column.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to be in ``spatial_support``: the
        threshold the rates were estimated with. ``analysis_settings`` only
        records it.
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        family-specific settings. Preserved by indexing and table/xarray
        exports. None for manually constructed results without provenance.

    Notes
    -----
    ``spatial_support`` is a boolean mask over this result's bins. A bin must
    have positive occupancy, be occupied for at least ``min_occupancy`` and have
    finite rates for every unit. Angular and polar results use their own bin
    coordinates; unsupported bins are not evidence of neural silence.

    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Reference frame**: ``direction_frame`` records how direction to the
    object was computed: allocentric (0 = East, +pi/2 = North) or egocentric
    (0 = ahead, +pi/2 = left). The object-to-animal vector is reversed: add
    pi to ``preferred_direction()`` and wrap to [-pi, pi].

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import compute_egocentric_rate

    >>> # Build a result from a small, seeded trajectory + spike train
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = np.sort(rng.uniform(0, 100, 100))
    >>> result = compute_egocentric_rate(
    ...     None,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     object_positions,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape
    (120,)
    >>> result.distance_range
    (0.0, 50.0)

    See Also
    --------
    ObjectVectorRatesResult : Batch version for multiple neurons
    compute_egocentric_rate : Function to compute this result
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    env: PolarEnvironment
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    method: str = field(default="binned", kw_only=True)
    bandwidth: float | None = field(default=None, kw_only=True)
    unit_id: int | str | None = None

    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: int | None = field(default=None, kw_only=True)
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )
    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def _check_family(self) -> None:
        _check_direction_frame(self.direction_frame, context="ObjectVectorRateResult")

    def summary_table(self) -> pd.DataFrame:
        """Per-unit metrics with the same columns as the population table.

        Returns
        -------
        pandas.DataFrame
            One row indexed by this unit label, or <NA> when no label was supplied.

        Notes
        -----
        Labels are fixed-threshold heuristics; the family defaults are in
        ``df.attrs["classification_thresholds"]``. For shuffle significance,
        call object_vector_cell_significance / egocentric_object_vector_cell_significance with the raw arrays.
        See the family predicate's Notes for information/statistic bias.
        The direction frame is in ``df.attrs["direction_frame"]``.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> table = result[0].summary_table()
        >>> len(table)
        1
        """
        return _object_vector_summary_frame(self, index=self._row_unit_ids().tolist())

    def _headline_metrics(self) -> dict[str, float]:
        """Cheap, NaN-safe metrics for the singular summary."""
        if not np.any(np.isfinite(_to_numpy(self.firing_rate))):
            return {
                "preferred_distance": float("nan"),
                "preferred_direction": float("nan"),
            }
        return {
            "preferred_distance": float(self.preferred_distance()),
            "preferred_direction": float(self.preferred_direction()),
        }

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the object-vector rate map (firing rate by distance/direction).

        Delegates to the polar environment's plot_field method for
        consistent visualization across the codebase.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - add_colorbar : bool, default=True

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Notes
        -----
        The object-vector rate map shows firing rate indexed by (distance,
        direction) relative to the object. Distance is the first dimension,
        direction is the second dimension.

        Examples
        --------
        >>> ax = result.plot()  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

        >>> fig, ax = plt.subplots()  # doctest: +SKIP
        >>> result.plot(ax=ax, cmap="viridis", vmax=20.0)  # doctest: +SKIP

        See Also
        --------
        preferred_distance : Get distance component of peak response
        preferred_direction : Get direction component of peak response
        """
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

    def preferred_distance(self) -> float:
        """Distance to object at peak firing rate.

        Returns the distance component (first dimension) of the egocentric
        bin where the neuron shows maximum firing rate.

        Returns
        -------
        float
            Distance to the object at the peak firing rate, in the
            environment's length units; NaN when no bin has a rate.

        Notes
        -----
        For object-vector cells, this represents the preferred distance to
        the object. With positions in cm, a cell with preferred_distance=20
        fires most when the object is 20 cm away from the animal.

        The distance is extracted from the polar environment's bin
        centers. The first component (index 0) represents distance.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> dist = result.preferred_distance()
        >>> print(f"Preferred distance: {dist:.1f} cm")
        Preferred distance: 2.5 cm

        See Also
        --------
        preferred_direction : Get direction component of peak response
        plot : Visualize the object-vector rate map
        """
        return float(self._peak_coordinates(0)[0])

    def preferred_direction(self) -> float:
        """Direction from the animal to the object at peak firing rate.

        The result's ``direction_frame`` sets the convention: allocentric
        0 = East and +pi/2 = North; egocentric 0 = ahead and +pi/2 = left.
        The reverse, object-to-animal vector adds pi and wraps to [-pi, pi].

        Returns
        -------
        float
            Peak animal-to-object direction in radians, in the recorded
            frame; NaN when no bin has a rate.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> direction = result.preferred_direction()
        >>> print(f"Preferred direction: {np.degrees(direction):.1f}")
        Preferred direction: -45.0

        See Also
        --------
        preferred_distance : Get distance component of peak response
        plot : Visualize the object-vector rate map
        """
        return float(self._peak_coordinates(1)[0])

    def spatial_information(self) -> float:
        """Compute spatial information in the recorded frame (bits per spike).

        Quantifies distance/direction selectivity using Skaggs information and
        the occupancy of polar bins in ``result.direction_frame``.

        Returns
        -------
        float
            Spatial information in the recorded frame, in bits per spike. Returns 0.0
            for uniform firing (no spatial selectivity).

        Notes
        -----
        **Formula (Skaggs et al. 1993)**:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        where :math:`p_i` is occupancy probability in polar bin :math:`i`,
        :math:`r_i` is firing rate in that bin, and :math:`\\bar{r}` is mean
        firing rate.

        **Interpretation**: higher values mean more selective tuning to
        distance and direction; zero means the same rate in every bin.
        Plug-in information is biased upward for few spikes, so compare it
        with a circular-shift test before calling a cell tuned.

        This metric uses polar occupancy in the recorded frame (time spent
        at each distance/direction combination), which differs from standard spatial
        information that uses allocentric position occupancy.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> info = result.spatial_information()
        >>> print(f"Egocentric spatial info: {info:.2f} bits/spike")
        Egocentric spatial info: 0.83 bits/spike

        See Also
        --------
        is_object_vector_cell : Classify as object-vector cell based on this metric
        """
        from neurospatial.encoding._metrics import spatial_information

        firing_rate = _to_numpy(self.firing_rate)
        occupancy = _to_numpy(self.occupancy)
        return spatial_information(firing_rate, occupancy)

    def spatial_information_bias(self) -> float:
        """Expected spatial information of an untuned cell (bits/spike).

        The information a cell firing at this unit's mean rate in every bin is
        expected to show on this occupancy with this estimator, from Poisson
        noise alone. ``spatial_information() - spatial_information_bias()``
        is what the object-vector screen compares with its cutoff. The derivation
        and its validation are in
        :meth:`~neurospatial.encoding.spatial.SpatialRateResult.spatial_information_bias`.

        Returns
        -------
        float
            Expected untuned information in bits/spike. NaN for a silent unit
            and, with a ``UserWarning``, when the unit has fewer spikes than
            the map has effectively independent bins.
        """
        from neurospatial.encoding._metrics import _information_bias

        return float(_information_bias(self)[0])

    def is_object_vector_cell(self, *, min_info: float | None = None) -> bool:
        """Classify as object-vector cell based on spatial information in the recorded frame.

        Tests object-vector tuning in ``result.direction_frame``.

        A neuron is classified as a candidate in the recorded frame if its
        bias-corrected spatial information meets or exceeds the minimum
        threshold. OVCs fire when the
        animal is at a specific distance and direction from an object.

        Parameters
        ----------
        min_info : float or None, default=None
            Minimum spatial information in the recorded frame threshold in bits/spike.


        Returns
        -------
        bool
            True if spatial_information() - spatial_information_bias()
            >= min_info, False otherwise.

        Notes
        -----
        The 0.3 bits/spike default is a screening convention. The screen
        compares bias-corrected information, ``spatial_information() -
        spatial_information_bias()``, with it: the information an untuned cell
        shows from Poisson noise alone is subtracted first (derivation in
        :meth:`~neurospatial.encoding.spatial.SpatialRateResult.spatial_information_bias`). For the default
        ``binned`` map that bias is (K - 1) / (2 ln(2) N_spikes) for K occupied
        polar bins, and units with fewer spikes than K fail, with a warning.
        The default grid covers the arena, so K grows with it: with the object
        at the center of a 100 x 100 cm box, 150-170 of the 348 polar bins (29
        rings x 12 directions) were occupied in 10 minutes. In 40 untuned
        Poisson units per spike count in that box (10-minute sessions,
        allocentric and egocentric), uncorrected information passed 0.3
        bits/spike in 100% of units at about 20, 60, 120 and 250 spikes and 0%
        at 600; corrected, none passed. An object-vector cell with a 2 Hz peak
        fired about 70 spikes in those 10 minutes, too few for the screen; pass
        a narrower distance_range or wider bins, or use the shuffle test.
        For publication, report a circular-shift test and its assumptions.
        None thresholds resolve through OBJECT_VECTOR_THRESHOLDS.
        For a shuffle test, call object_vector_cell_significance(...) (allocentric) or
        egocentric_object_vector_cell_significance(...) (egocentric)
        or is_egocentric_object_vector_cell(..., criterion="shuffle") with the raw
        arrays; a result does not keep the arrays it was computed from.

        **Object-vector vs place cells**: with one fixed object, a place
        field is also a field at one distance and direction from the object,
        so an allocentric object-vector map cannot tell the two apart within
        a session: in simulation both pass this screen and score alike on
        ``object_vector_score``. The control is
        to move the object: an object-vector field keeps its vector to the
        object, so vector maps from the two placements correlate, while a
        place field stays put (Høydal et al., 2019). An egocentric map is
        appropriate only for heading-relative bearing.

        References
        ----------
        .. [1] Hoydal, O. A., et al. (2019). Object-vector coding in the medial
               entorhinal cortex. Nature, 568(7752), 400-404.
        .. [2] Deshmukh, S. S., & Knierim, J. J. (2011). Representation of
               non-spatial and spatial information in the lateral entorhinal
               cortex. Frontiers in Behavioral Neuroscience, 5, 69.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rate
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = np.sort(rng.uniform(0, 100, 100))
        >>> result = compute_egocentric_rate(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> bool(result.spatial_information() >= 0.3)
        True
        >>> import warnings
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")
        ...     result.is_object_vector_cell()
        False

        These uniform random spikes pass the uncorrected 0.3 bits/spike
        cutoff from finite-count bias alone. The screen fails them: 100
        spikes are fewer than the map's roughly 110 independent bins, so it
        warns and refuses. The allocentric circular-shift verdict tests the
        same untuned train:

        >>> from neurospatial.encoding import is_object_vector_cell
        >>> is_object_vector_cell(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     object_positions,
        ...     criterion="shuffle",
        ...     n_shuffles=50,
        ...     rng=0,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        False

        See Also
        --------
        spatial_information : Compute the metric used for classification
        """
        min_info = (
            OBJECT_VECTOR_THRESHOLDS["min_info"] if min_info is None else min_info
        )
        return bool(_object_vector_classify(self, min_info=min_info)[0])


@_copy_unit_table
@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class ObjectVectorRatesResult(
    _PolarRateMixin, PopulationRateMixin[ObjectVectorRateResult]
):
    """Result of object-vector rate computation for multiple neurons.

    This class wraps object-vector firing rate maps for a population of neurons
    with shared metadata (occupancy, polar environment, bin parameters).
    It supports iteration and indexing to access individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rates in polar coordinates in the recorded frame for all neurons in Hz.
        Shape is (n_neurons, n_bins) where n_bins is the number of active
        bins in the polar environment.
    occupancy : ArrayLike
        Time spent in each polar bin in seconds. Shape is (n_bins,).
        This is shared across all neurons since the animal's trajectory
        (and thus polar occupancy in the recorded frame) is the same for all neurons.
    env : Environment
        The polar environment used for the computation.
    direction_frame : {"allocentric", "egocentric"}
        Required reference frame for the direction to the object.
        Allocentric: 0 = East, +pi/2 = North. Egocentric: 0 = ahead, +pi/2 = left.
    method : {"binned", "diffusion_kde", "gaussian_kde"}, default="binned"
        Estimator used. ``"binned"`` is the raw rate in each polar bin.
    bandwidth : float or None, default=None
        Kernel bandwidth in the environment's length units for the kernel
        methods; ``None`` for ``"binned"``, which does not smooth.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rates for all neurons. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time in each bin in seconds. Shape is (n_bins,). Shared.
    env : Environment
        The egocentric polar environment.
    distance_range : tuple[float, float]
        (min, max) distance covered by the bins; read from ``env``.
    n_distance_bins : int
        Number of distance bins; read from ``env``.
    n_direction_bins : int
        Number of direction bins; read from ``env``.
    unit_ids : NDArray, shape (n_units,)
        Identifier for each unit (row), e.g. from ``read_units`` or passed via
        ``unit_ids=``. Defaults to ``np.arange(n_units)``. Carried into
        indexed/iterated single-unit results and into xarray exports.
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata aligned to ``unit_ids`` (e.g. region,
        quality, depth, inclusion flags), one row per unit; ``None`` when not
        provided. A unit_id column or named/non-default index aligns rows
        by identity; an unnamed default RangeIndex uses unit order.
        Mismatched or duplicate labels raise ValueError.

    Other Parameters
    ----------------
    _unit_ids_generated : bool, default=False
        True when the unit labels were generated (no ``unit_ids`` or
        ``unit_id`` supplied), so pairing with spike inputs stays positional.
        Set by the compute functions and by indexing; not an analysis choice.
    n_spikes_outside_intervals : int or ndarray of int64, shape (n_units,), or None
        Each unit's spikes outside the analyzed time intervals
        (``analysis_settings["observed_intervals"]``), which are in no rate:
        an int for one unit, one count per unit for a population. ``None``
        when unknown, as for a result built by hand. Also a
        ``summary_table()`` column.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to be in ``spatial_support``: the
        threshold the rates were estimated with. ``analysis_settings`` only
        records it.
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        family-specific settings. Preserved by indexing and table/xarray
        exports. None for manually constructed results without provenance.

    Notes
    -----
    ``spatial_support`` is a boolean mask over this result's bins. A bin must
    have positive occupancy, be occupied for at least ``min_occupancy`` and have
    finite rates for every unit. Angular and polar results use their own bin
    coordinates; unsupported bins are not evidence of neural silence.

    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration interface**: Supports ``len()``, indexing with ``[]``, and
    iteration with ``for``. Each element is an ``ObjectVectorRateResult`` for
    one neuron.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import (
    ...     ObjectVectorRateResult,
    ...     compute_egocentric_rates,
    ... )

    >>> # Build a batch result from a small, seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> times = np.linspace(0, 100, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0]])
    >>> spike_times = [
    ...     np.sort(rng.uniform(0, 100, 100)),
    ...     np.sort(rng.uniform(0, 100, 150)),
    ...     np.sort(rng.uniform(0, 100, 50)),
    ... ]
    >>> result = compute_egocentric_rates(
    ...     None,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     object_positions,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )

    >>> # Access fields
    >>> len(result)
    3
    >>> isinstance(result[0], ObjectVectorRateResult)  # First neuron
    True

    >>> # Iterate over neurons
    >>> rates = [float(single.firing_rate.max()) for single in result]
    >>> len(rates)
    3

    See Also
    --------
    ObjectVectorRateResult : Single-neuron version
    compute_egocentric_rates : Function to compute this result
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    env: PolarEnvironment
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    method: str = field(default="binned", kw_only=True)
    bandwidth: float | None = field(default=None, kw_only=True)
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)

    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: NDArray[np.int64] | None = field(
        default=None, kw_only=True
    )
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )
    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def _check_family(self) -> None:
        _check_direction_frame(self.direction_frame, context="ObjectVectorRatesResult")

    def __getitem__(self, idx: int) -> ObjectVectorRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Unit position (0-based), the row of :attr:`firing_rates`. Select
            by label with ``sel(unit_id=...)``.

        Returns
        -------
        ObjectVectorRateResult
            Object-vector rate result in the same frame for the specified neuron.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> single = result[0]
        >>> isinstance(single, ObjectVectorRateResult)
        True
        """
        return self._unit_result(
            ObjectVectorRateResult,
            idx,
            env=self.env,
            direction_frame=self.direction_frame,
            method=self.method,
            bandwidth=self.bandwidth,
        )

    def plot(self, idx: int, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the object-vector rate map for a specific neuron.

        Delegates to the polar environment's plot_field method for
        consistent visualization across the codebase.

        Parameters
        ----------
        idx : int
            Index of the neuron to plot (0-indexed).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - add_colorbar : bool, default=True

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Notes
        -----
        The object-vector rate map shows firing rate indexed by (distance,
        direction) relative to the object. Distance is the first dimension,
        direction is the second dimension.

        Examples
        --------
        >>> # Plot the first neuron's object-vector rate map
        >>> ax = result.plot(idx=0)  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

        >>> # Plot neuron 3 with custom colormap
        >>> fig, ax = plt.subplots()  # doctest: +SKIP
        >>> result.plot(idx=3, ax=ax, cmap="viridis", vmax=20.0)  # doctest: +SKIP

        See Also
        --------
        preferred_distances : Get distance preferences for all neurons
        ObjectVectorRateResult.plot : Plot for single-neuron result
        """
        return self.env.plot_field(
            _to_numpy(self.firing_rates[idx]),  # type: ignore[index]
            ax=ax,
            **kwargs,
        )

    def preferred_distances(self) -> NDArray[np.float64]:
        """Preferred distances to object for all neurons.

        Returns the distance component (first dimension) of the egocentric
        bin where each neuron shows maximum firing rate.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Distance to the object at each neuron's peak firing rate, in
            the environment's length units; NaN for a unit with no rate.

        Notes
        -----
        For object-vector cells, this represents the preferred distance to
        the object. With positions in cm, a cell with preferred_distance=20
        fires most when the object is 20 cm away from the animal.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> distances = result.preferred_distances()
        >>> distances.shape
        (3,)
        >>> print(f"Neuron 0 prefers distance: {distances[0]:.1f} cm")
        Neuron 0 prefers distance: 2.5 cm

        See Also
        --------
        ObjectVectorRateResult.preferred_distance : Single-neuron version
        preferred_directions : Get direction preferences for all neurons
        """
        return self._peak_coordinates(0)

    def preferred_directions(self) -> NDArray[np.float64]:
        """Directions from the animal to the object at peak firing rate.

        The result's ``direction_frame`` sets the convention: allocentric
        0 = East and +pi/2 = North; egocentric 0 = ahead and +pi/2 = left.
        The reverse, object-to-animal vector adds pi and wraps to [-pi, pi].

        Returns
        -------
        NDArray[np.float64], shape (n_neurons,)
            Peak animal-to-object direction in radians, in the recorded
            frame; NaN for a unit with no rate.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> directions = result.preferred_directions()
        >>> directions.shape
        (3,)
        >>> print(f"Neuron 0 prefers direction: {np.degrees(directions[0]):.1f}")
        Neuron 0 prefers direction: -45.0

        See Also
        --------
        ObjectVectorRateResult.preferred_direction : Single-neuron version
        preferred_distances : Get distance preferences for all neurons
        """
        return self._peak_coordinates(1)

    def spatial_information(self) -> NDArray[np.float64]:
        """Spatial information in the recorded frame for all neurons (bits per spike).

        Quantifies distance/direction selectivity in ``result.direction_frame``
        for each neuron.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Spatial information in the recorded frame, in bits/spike per neuron.
            Always non-negative. Returns 0.0 for uniform firing.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula with polar occupancy in the
        recorded frame.
        This is computed by delegating to the batch spatial information
        function in ``_metrics.py``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> info = result.spatial_information()
        >>> info.shape
        (3,)
        >>> print(f"Neuron with highest info: {np.argmax(info)}")
        Neuron with highest info: 2

        See Also
        --------
        ObjectVectorRateResult.spatial_information : Single-neuron version
        classify : Classify neurons based on this metric
        """
        from neurospatial.encoding._metrics import batch_spatial_information

        return batch_spatial_information(
            _to_numpy(self.firing_rates), _to_numpy(self.occupancy)
        )

    def spatial_information_bias(self) -> NDArray[np.float64]:
        """Expected spatial information of an untuned cell, per unit (bits/spike).

        The information a cell firing at each unit's mean rate in every bin is
        expected to show on this occupancy with this estimator, from Poisson
        noise alone. ``spatial_information() - spatial_information_bias()``
        is what :meth:`classify` compares with its cutoff. The derivation and
        its validation are in
        :meth:`~neurospatial.encoding.spatial.SpatialRateResult.spatial_information_bias`.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Expected untuned information in bits/spike. NaN for silent units
            and, with one ``UserWarning``, for units with fewer spikes than
            the maps have effectively independent bins.
        """
        from neurospatial.encoding._metrics import _information_bias

        return _information_bias(self)

    def classify(self, *, min_info: float | None = None) -> NDArray[np.bool_]:
        """Classify neurons as object-vector cells.

        Tests object-vector tuning in ``result.direction_frame``.

        A neuron is classified as a candidate in the recorded frame if its
        bias-corrected spatial information meets or exceeds the minimum
        threshold. This is the
        single-type boolean predicate ("is this an OVC") for the batch result.

        Parameters
        ----------
        min_info : float or None, default=None
            Minimum spatial information in the recorded frame threshold in bits/spike.
            See ObjectVectorRateResult.is_object_vector_cell() for threshold rationale.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Boolean array where True indicates the neuron is classified
            as an object-vector cell.

        Notes
        -----
        The 0.3 bits/spike default is a screening convention. The screen
        compares bias-corrected information, ``spatial_information() -
        spatial_information_bias()``, with it: the information an untuned cell
        shows from Poisson noise alone is subtracted first (derivation in
        :meth:`~neurospatial.encoding.spatial.SpatialRateResult.spatial_information_bias`). For the default
        ``binned`` map that bias is (K - 1) / (2 ln(2) N_spikes) for K occupied
        polar bins, and units with fewer spikes than K fail, with a warning.
        The default grid covers the arena, so K grows with it: with the object
        at the center of a 100 x 100 cm box, 150-170 of the 348 polar bins (29
        rings x 12 directions) were occupied in 10 minutes. In 40 untuned
        Poisson units per spike count in that box (10-minute sessions,
        allocentric and egocentric), uncorrected information passed 0.3
        bits/spike in 100% of units at about 20, 60, 120 and 250 spikes and 0%
        at 600; corrected, none passed. An object-vector cell with a 2 Hz peak
        fired about 70 spikes in those 10 minutes, too few for the screen; pass
        a narrower distance_range or wider bins, or use the shuffle test.
        For publication, report a circular-shift test and its assumptions.
        None thresholds resolve through OBJECT_VECTOR_THRESHOLDS.
        For a shuffle test, call object_vector_cell_significance(...) (allocentric) or
        egocentric_object_vector_cell_significance(...) (egocentric)
        or is_egocentric_object_vector_cell(..., criterion="shuffle") with the raw
        arrays; a result does not keep the arrays it was computed from.

        Uses vectorized computation of spatial_information() for
        efficiency with large populations.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> is_object_vector_cell = result.classify()  # untuned: warns, none pass
        >>> print(f"Found {is_object_vector_cell.sum()} OVCs")
        Found 0 OVCs

        >>> # Use stricter threshold
        >>> is_object_vector_cell = result.classify(min_info=0.5)

        See Also
        --------
        ObjectVectorRateResult.is_object_vector_cell : Single-neuron classification
        spatial_information : The metric used for classification
        """
        min_info = (
            OBJECT_VECTOR_THRESHOLDS["min_info"] if min_info is None else min_info
        )
        return _object_vector_classify(self, min_info=min_info)

    def summary_table(
        self,
        unit_ids: Sequence[str | int] | None = None,
    ) -> pd.DataFrame:
        """Per-unit scalar summary: one row per unit, ``unit_id``-indexed.

        Computes all object-vector metrics and returns one row per unit, indexed
        by ``unit_id``, with scalar metric columns. This is the per-unit
        summary for filtering, sorting, and population tables. For the dense
        per-bin frame (one row per ``(unit, bin)``) use :meth:`to_dataframe`.

        Parameters
        ----------
        unit_ids : sequence of str or int, optional
            Identity labels for the index, one per unit. If ``None``, the
            result's own :attr:`unit_ids` are used.

        Returns
        -------
        pd.DataFrame
            One row per unit, indexed by ``unit_id``, with columns:

            - peak_firing_rate: maximum firing rate (Hz)
            - preferred_distance: preferred distance to the object, in the
              environment's length units (``df.attrs["units"]``)
            - preferred_direction_deg: preferred direction (degrees in
              ``result.direction_frame``)
            - preferred_direction: preferred direction (radians, same frame)
            - spatial_information: Skaggs information of the polar map
              (bits/spike)
            - spatial_information_bias: the information an untuned cell
              would show (bits/spike); the screen compares
              ``spatial_information - spatial_information_bias`` with its
              cutoff, and a NaN bias (too few spikes) fails it
            - n_spikes_outside_intervals: spikes outside the analyzed time
            - is_object_vector_cell: whether the default threshold screen
              passes

        Raises
        ------
        ValueError
            If unit_ids has a different length than the number of units, or
            repeats a label.

        Notes
        -----
        Labels are fixed-threshold heuristics; the family defaults are in
        ``df.attrs["classification_thresholds"]``. For shuffle significance,
        call object_vector_cell_significance / egocentric_object_vector_cell_significance with the raw arrays.
        See the family predicate's Notes for information/statistic bias.
        The direction frame is in ``df.attrs["direction_frame"]``.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.

        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``preferred_distances()``, ``classify()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["is_object_vector_cell"] == True]``
        - Sort: ``df.sort_values("preferred_distance")``
        - Top-N: ``df.nlargest(10, "peak_firing_rate")``

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.egocentric import compute_egocentric_rates
        >>> rng = np.random.default_rng(0)
        >>> times = np.linspace(0, 100, 1000)
        >>> positions = rng.uniform(10, 90, (1000, 2))
        >>> headings = rng.uniform(-np.pi, np.pi, 1000)
        >>> object_positions = np.array([[50.0, 50.0]])
        >>> spike_times = [
        ...     np.sort(rng.uniform(0, 100, 100)),
        ...     np.sort(rng.uniform(0, 100, 150)),
        ...     np.sort(rng.uniform(0, 100, 50)),
        ... ]
        >>> result = compute_egocentric_rates(
        ...     None,
        ...     spike_times,
        ...     times,
        ...     positions,
        ...     headings,
        ...     object_positions,
        ...     distance_range=(0.0, 50.0),
        ...     distance_bin_size=5.0,
        ... )
        >>> df = result.summary_table()
        >>> list(df.columns)
        ['peak_firing_rate', 'preferred_distance', 'preferred_direction_deg', 'preferred_direction', 'spatial_information', 'spatial_information_bias', 'n_spikes_outside_intervals', 'is_object_vector_cell']
        >>> len(df)
        3
        >>> df.index.name
        'unit_id'

        >>> # Filter for OVCs only
        >>> ovcs = df[df["is_object_vector_cell"]]

        >>> # Sort by preferred distance
        >>> sorted_df = df.sort_values("preferred_distance")

        >>> # Custom unit identifiers
        >>> df = result.summary_table(unit_ids=["unit_0", "unit_1", "unit_2"])
        >>> list(df.index)
        ['unit_0', 'unit_1', 'unit_2']

        See Also
        --------
        to_dataframe : Dense per-bin frame (one row per (unit, bin)).
        classify : OVC classification
        preferred_distances : Batch preferred distance computation
        preferred_directions : Batch preferred direction computation
        """
        n_neurons = len(self)

        if unit_ids is None:
            index_ids: list[str | int] = list(np.asarray(self.unit_ids))
        else:
            from neurospatial._results import resolve_unit_ids

            # Validate as an object array so mixed int/str labels are
            # neither coerced to strings nor merged; keep them as given.
            index_ids = list(unit_ids)
            resolve_unit_ids(
                np.asarray(index_ids, dtype=object),
                n_neurons,
                context="ObjectVectorRatesResult.summary_table",
            )

        return _object_vector_summary_frame(self, index=index_ids)


def _raw_polar_rate(
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    min_occupancy: float,
) -> NDArray[np.float64]:
    """Raw firing rate (spikes / occupancy) for an egocentric polar grid.

    Graph-diffusion smoothing on the polar environment bleeds rate across
    *distance* rings (the env connects adjacent distance bins radially), which
    erases the distance tuning object-vector cells encode. The ``binned``
    method therefore computes the bin rate directly, with no graph smoothing.

    Parameters
    ----------
    spike_counts : ndarray of shape (n_bins,), dtype float64
        Spike counts per polar bin.
    occupancy : ndarray of shape (n_bins,), dtype float64
        Time spent in each polar bin, in seconds.
    min_occupancy : float
        Bins with occupancy below this value are treated as unvisited.

    Returns
    -------
    ndarray of shape (n_bins,), dtype float64
        Firing rate per bin in Hz. Unvisited bins and bins below
        ``min_occupancy`` are NaN (undefined, not zero).

    Notes
    -----
    Masking convention (shared with ``_smoothing._apply_min_occupancy_mask``):
    a bin is valid iff its raw occupancy is positive and at least
    ``min_occupancy``, so ``result.occupancy < min_occupancy`` is exactly the
    masked set.
    """
    occ = np.asarray(occupancy, dtype=np.float64)
    counts = np.asarray(spike_counts, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = counts / occ
    valid = (occ > 0.0) & (occ >= min_occupancy)
    return np.where(valid, rate, np.nan)


def _egocentric_firing_rate(
    polar_env: PolarEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float | None,
    min_occupancy: float,
    backend: ResolvedBackend,
) -> ArrayLike:
    """Egocentric polar firing rate: raw for ``binned``, smoothed otherwise.

    Parameters
    ----------
    polar_env : Environment
        The egocentric polar grid the rate is computed over.
    spike_counts : ndarray of shape (n_bins,), dtype float64
        Spike counts per polar bin.
    occupancy : ndarray of shape (n_bins,), dtype float64
        Time spent in each polar bin, in seconds.
    method : {"diffusion_kde", "gaussian_kde", "binned"}
        ``"binned"`` returns the raw bin rate (see ``_raw_polar_rate``); the
        kernel methods smooth via ``smooth_rate_map``.
    bandwidth : float or None
        Smoothing bandwidth, in environment units; ``None`` for ``"binned"``.
    min_occupancy : float
        Bins with occupancy below this value are treated as unvisited.
    backend : {"numpy", "jax"}
        Array backend for the returned rate.

    Returns
    -------
    ArrayLike of shape (n_bins,)
        Firing rate per bin in Hz, as a NumPy or JAX array per ``backend``.
    """
    from neurospatial.encoding._backend import is_jax_available

    if method == "binned":
        _warn_if_fully_masked(occupancy, min_occupancy)
        rate = _raw_polar_rate(spike_counts, occupancy, min_occupancy)
        if backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            jax_rate: ArrayLike = jnp.asarray(rate, dtype=jnp.float64)
            return jax_rate
        return rate

    from neurospatial.encoding._smoothing import smooth_rate_map

    assert bandwidth is not None  # resolved for every kernel method
    return smooth_rate_map(
        polar_env,
        spike_counts,
        occupancy,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
    )


def compute_egocentric_rate(
    env: Environment | None,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: BackendType = "numpy",
) -> ObjectVectorRateResult:
    """Compute egocentric firing rate for one neuron.

    This function computes a firing rate map in egocentric polar
    coordinates (distance and direction to nearest object). This is the key
    metric for identifying object-vector cells (OVCs).

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal position coordinates at each time sample. NaN values (in
        positions or headings) are treated as missing data and excluded from
        occupancy and firing-rate computation; callers do not need to
        pre-filter tracking dropouts.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, **allocentric
        world-frame convention**: 0 = East, π/2 = North, π = West,
        -π/2 = South, wrapped to [-π, π]). The allocentric→egocentric
        transform is applied internally; pass world-frame headings, not
        animal-frame angles.
    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.

    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    Returns
    -------
    ObjectVectorRateResult
        Result object containing:

        - ``firing_rate``: Firing rate by egocentric coordinates in Hz,
          shape (n_bins,)
        - ``occupancy``: Time in each egocentric bin in seconds,
          shape (n_bins,)
        - ``env``: The egocentric polar environment
        - ``distance_range``: Distance range used
        - ``n_distance_bins``: Number of distance bins
        - ``n_direction_bins``: Number of direction bins

    Raises
    ------
    ValueError
        If ``metric="geodesic"`` but ``env`` is None.
        If ``metric`` is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_egocentric_rates : Batch version for multiple neurons
    ObjectVectorRateResult : Result class with convenience methods
    compute_spatial_rate : Standard spatial rate (by animal position)

    Notes
    -----
    An interval is analyzed only if it passes the gap and bounds checks
    and lies inside ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    **Algorithm**:

    1. Compute egocentric coordinates (distance, bearing) to nearest object
       at each behavioral frame
    2. Bin spikes by egocentric coordinates at spike time
    3. Compute egocentric occupancy (time spent at each distance/direction)
    4. Divide spike counts by occupancy in each bin; the kernel methods
       smooth counts and occupancy first

    **Coordinate convention**: Direction uses egocentric (animal-centered)
    coordinates where 0=ahead, +π/2=left, -π/2=right.

    **Place cells vs object-vector cells**: For place cells, firing is
    determined by allocentric position. For OVCs, firing is determined by
    egocentric relationship to objects. Computing place field using
    ``compute_spatial_rate`` and OVC field using this function, then comparing
    spatial information, can help distinguish cell types.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import compute_egocentric_rate

    >>> # Create trajectory and objects
    >>> rng = np.random.default_rng(42)
    >>> times = np.linspace(0, 40, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0], [25.0, 75.0]])
    >>> spike_times = np.sort(rng.uniform(0, 40, 100))

    >>> # Compute egocentric rate
    >>> result = compute_egocentric_rate(
    ...     None,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     object_positions,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )

    >>> # Access results
    >>> result.firing_rate.shape
    (120,)
    >>> pref_dist = result.preferred_distance()
    >>> pref_dir = result.preferred_direction()
    >>> info = result.spatial_information()
    >>> is_object_vector_cell = result.is_object_vector_cell()

    >>> # Plot the egocentric rate map
    >>> ax = result.plot()

    References
    ----------
    .. [1] Wang, C., et al. (2018). Egocentric coding of external items in the
           lateral entorhinal cortex. Science, 362, 945-949.
           doi:10.1126/science.aau4940.
    """
    if headings is None:
        raise ValueError(
            _format_error(
                "compute_egocentric_rate: headings is required for egocentric bearing.",
                why="Why: animal-relative direction needs the heading at each sample",
                fix="pass headings, or use compute_object_vector_rate without headings",
            )
        )
    return _object_vector_rate(
        env,
        spike_times,
        times,
        positions,
        headings,
        object_positions,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
        context="compute_egocentric_rate",
    )


def compute_object_vector_rate(
    env: Environment | None,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: BackendType = "numpy",
) -> ObjectVectorRateResult:
    """Compute allocentric firing rate for one neuron.

    This function computes a firing rate map in allocentric polar
    coordinates (distance and direction to nearest object). This is the key
    metric for identifying object-vector cells (OVCs).

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal position coordinates at each time sample. NaN values (in
        positions) are treated as missing data and excluded from
        occupancy and firing-rate computation; callers do not need to
        pre-filter tracking dropouts.
    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.

    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    Returns
    -------
    ObjectVectorRateResult
        Result object containing:

        - ``firing_rate``: Firing rate by allocentric coordinates in Hz,
          shape (n_bins,)
        - ``occupancy``: Time in each allocentric bin in seconds,
          shape (n_bins,)
        - ``env``: The allocentric polar environment
        - ``distance_range``: Distance range used
        - ``n_distance_bins``: Number of distance bins
        - ``n_direction_bins``: Number of direction bins

    Raises
    ------
    ValueError
        If ``metric="geodesic"`` but ``env`` is None.
        If ``metric`` is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_object_vector_rates : Batch version for multiple neurons
    ObjectVectorRateResult : Result class with convenience methods
    compute_spatial_rate : Standard spatial rate (by animal position)

    Notes
    -----
    An interval is analyzed only if it passes the gap and bounds checks
    and lies inside ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    **Algorithm**:

    1. Compute allocentric coordinates (distance, bearing) to nearest object
       at each behavioral frame
    2. Bin spikes by allocentric coordinates at spike time
    3. Compute allocentric occupancy (time spent at each distance/direction)
    4. Divide spike counts by occupancy in each bin; the kernel methods
       smooth counts and occupancy first

    **Coordinate convention**: Direction uses allocentric (world-centered)
    coordinates where 0=East, +π/2=North, -π/2=South.

    **Place cells vs object-vector cells**: with one fixed object, a place
    field lies at one distance and direction from it, so this map shows a
    place cell as a localized vector field; comparing its information with
    ``compute_spatial_rate`` does not separate them. Record a session with
    the object moved and correlate the two vector maps: an object-vector
    field follows the object, a place field does not (Høydal et al., 2019).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import compute_object_vector_rate
    >>> rng = np.random.default_rng(42)
    >>> times = np.arange(0.0, 40.0, 0.04)
    >>> positions = rng.uniform(10, 90, (len(times), 2))
    >>> objects = np.array([[50.0, 50.0]])
    >>> train = np.sort(rng.uniform(0, 39.9, 100))
    >>> result = compute_object_vector_rate(
    ...     None,
    ...     train,
    ...     times,
    ...     positions,
    ...     objects,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )
    >>> result.direction_frame
    'allocentric'
    >>> result.firing_rate.shape
    (120,)

    References
    ----------
    .. [1] Høydal, Ø. A., et al. (2019). Object-vector coding in the medial
           entorhinal cortex. Nature, 568, 400-404.
           doi:10.1038/s41586-019-1077-7.
    """
    return _object_vector_rate(
        env,
        spike_times,
        times,
        positions,
        None,
        object_positions,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
        context="compute_object_vector_rate",
    )


def _validate_object_vector_metric(
    env: Environment | None, metric: str, *, context: str
) -> None:
    """Validate public metric choices before coordinate construction."""
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            _format_error(
                f"{context}: Invalid metric: {metric!r}; expected 'euclidean' or 'geodesic'.",
                why="Why: nearest-object distance uses a straight line or the environment graph",
                fix="pass metric='euclidean', or metric='geodesic' with an env",
            )
        )
    if metric == "geodesic" and env is None:
        raise ValueError(
            _format_error(
                f"{context}: metric='geodesic' requires env parameter.",
                why="Why: geodesic distance needs the environment's connectivity graph",
                fix="pass an env, or use metric='euclidean' with env=None",
            )
        )


def _require_planar_env(env: Environment, operation: str, *, measures: str) -> None:
    """Refuse a polar, 1-D or 3-D environment for a map built from (x, y) positions.

    Parameters
    ----------
    env : Environment
        The environment passed with the positions.
    operation : str
        The refused call, e.g. ``"compute_object_vector_rate()"``.
    measures : str
        What the map measures from the positions, e.g. ``"distance and
        direction to the nearest object"``.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar.
    ValueError
        If ``env`` is not two-dimensional.
    """
    from neurospatial.geometry import require_cartesian

    require_cartesian(
        env.coordinate_system,
        operation,
        why=f"It measures {measures} from the animal's allocentric positions.",
        fix="pass the allocentric environment the positions were recorded in",
    )
    if env.n_dims != 2:
        raise ValueError(
            _format_error(
                f"{operation} needs a 2-D environment; got a {env.n_dims}-D one.",
                why=f"Why: {measures} are measured in the plane, from (x, y) positions",
                fix="build the environment from the animal's 2-D (x, y) positions",
            )
        )


def _check_object_vector_env(
    env: Environment | None, headings: NDArray[np.float64] | None, *, context: str
) -> None:
    """Validate ``env`` when one was supplied.

    ``env`` is optional for object-vector rates: ``None`` is permitted with the
    euclidean metric, and the geodesic metric raises its own error.

    Parameters
    ----------
    env : Environment or None
        Environment passed to the object-vector entry point.
    headings : ndarray of shape (n_time,) or None
        Headings; ``None`` selects the allocentric frame and shortens the
        argument list in the error message.
    context : str
        Public function name used in error messages.

    Raises
    ------
    TypeError
        If ``env`` is given and is not an environment.
    CartesianOnlyError
        If ``env`` is polar.
    ValueError
        If ``env`` is not two-dimensional.
    """
    from neurospatial.encoding._validation import _validate_env_argument

    if env is not None:
        _validate_env_argument(
            env,
            context=context,
            arguments=(
                "spike_times, times, positions, object_positions"
                if headings is None
                else "spike_times, times, positions, headings, object_positions"
            ),
        )
        _require_planar_env(
            env,
            f"{context}()",
            measures="distance and direction to the nearest object",
        )


def _largest_object_distance(
    env: Environment,
    object_positions: NDArray[np.float64],
    metric: DistanceMetric,
) -> float:
    """The farthest the animal can be from the nearest object inside ``env``.

    Euclidean: the diagonal of the environment's bounding box, which bounds
    the distance between any two points inside it whatever the objects'
    placement, so maps of one arena with the object moved share a grid.
    Geodesic: the largest path distance from the nearest object's bin to any
    reachable bin; no placement-free bound exists for path lengths.
    """
    if metric == "geodesic":
        from neurospatial.ops.distance import distance_field

        object_bins = env.bin_at(np.asarray(object_positions).reshape(-1, 2))
        sources = [int(b) for b in object_bins if b >= 0]
        if sources:
            field_ = distance_field(env.connectivity, sources=sources)
            reachable = field_[np.isfinite(field_)]
            if reachable.size:
                return float(reachable.max())
    if env.dimension_ranges is not None:
        spans = np.array([stop - start for start, stop in env.dimension_ranges])
    else:
        spans = np.ptp(env.bin_centers, axis=0)
    return float(np.hypot(*spans))


def _object_vector_grid(
    env: Environment | None,
    object_positions: NDArray[np.float64],
    *,
    metric: DistanceMetric,
    distance_range: tuple[float, float] | None,
    distance_bin_size: float | None,
    direction_bin_size: float,
    context: str,
) -> tuple[tuple[float, float], int, int]:
    """Resolve the object-vector polar grid; see :func:`resolve_polar_grid`.

    The default range covers every distance to the nearest object inside
    ``env`` (:func:`_largest_object_distance`).

    Raises
    ------
    ValueError
        If the default range has no ``env`` to measure, or the widths or
        range are invalid.
    """
    if distance_range is None and env is None:
        raise ValueError(
            _format_error(
                f"{context}: distance_range is required when env is None.",
                why="Why: the default range covers the environment, and "
                "there is no environment to measure",
                fix="pass distance_range=(0.0, max_distance) in the units "
                "of positions, or the env the positions were recorded in",
            )
        )
    return resolve_polar_grid(
        env.units if env is not None else None,
        distance_range=distance_range,
        largest_distance=(
            None
            if env is None
            else partial(_largest_object_distance, env, object_positions, metric)
        ),
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        context=context,
    )


def _warn_if_directions_unsampled(
    occupancy: ArrayLike,
    *,
    n_distance_bins: int,
    n_direction_bins: int,
    direction_frame: str,
    units: str | None,
    extended_landmarks: bool,
    context: str,
) -> None:
    """Warn when some directions to the object were never sampled at any distance.

    A mean resultant length over part of the circle reads even flat firing
    as directional, so an arc with no occupancy biases direction tuning
    whatever the cell does. Line and polygon landmarks, such as a monitor on
    a wall, are seen from one side by construction and are not checked.
    """
    if extended_landmarks:
        return
    occupancy = np.asarray(_to_numpy(occupancy), dtype=np.float64).reshape(
        n_distance_bins, n_direction_bins
    )
    sampled: NDArray[np.bool_] = np.asarray((occupancy > 0).any(axis=0))
    if sampled.all() or not sampled.any():
        return
    # Contiguous unsampled arcs, starting just after a sampled bin so an arc
    # across the +/-180 degree seam is reported once.
    width = 360.0 / n_direction_bins
    first = int(np.flatnonzero(sampled)[0])
    order = np.roll(np.arange(n_direction_bins), -first)
    arcs: list[str] = []
    start: int | None = None
    for k in [*order.tolist(), first]:
        if not sampled[k] and start is None:
            start = k
        elif sampled[k] and start is not None:
            low = -180.0 + width * start
            high = -180.0 + width * k if k > start else 180.0 + width * k
            arcs.append(f"[{low:g}, {high:g})")
            start = None
    warnings.warn(
        _format_error(
            f"{context}: no time was spent with the nearest object in the "
            f"{direction_frame} directions {', '.join(arcs)} degrees, at any "
            "distance.",
            why="Why: direction tuning then covers only the sampled arc, "
            "preferred_direction is pulled toward it, and object_vector_score "
            "is NaN. This happens when the object lies outside the arena or "
            "against its wall, or when object_positions are not in the frame "
            "and units of positions",
            fix=f"check object_positions against the arena (units={units!r}); "
            "for an object that really is at the edge, read direction tuning "
            "over the sampled arc only",
        ),
        UserWarning,
        stacklevel=3,
    )


def _warn_if_peak_at_range_edge(
    firing_rates: ArrayLike,
    *,
    n_distance_bins: int,
    n_direction_bins: int,
    distance_range: tuple[float, float],
    time_beyond: float,
    time_in_range: float,
    unit_ids: Sequence[Any],
    units: str | None,
    context: str,
) -> None:
    """Warn when a unit peaks in the outermost ring and the range cut time off.

    A rate map's peak is a local maximum only away from the domain's edge. A
    peak in the last ring, while the animal also spent time beyond it, cannot
    be told from a field that continues past ``distance_range``.
    """
    if time_beyond <= 0:
        return
    rates = np.atleast_2d(np.asarray(_to_numpy(firing_rates), dtype=np.float64))
    observed = np.isfinite(rates).any(axis=1)
    if not observed.any():
        return
    peak_ring = np.full(len(rates), -1)
    peak_ring[observed] = np.nanargmax(rates[observed], axis=1) // n_direction_bins
    edge = np.flatnonzero(peak_ring == n_distance_bins - 1)
    if edge.size == 0:
        return
    labels = [unit_ids[i] for i in edge[:5]]
    more = f" and {edge.size - 5} more" if edge.size > 5 else ""
    share = time_beyond / (time_beyond + time_in_range)
    warnings.warn(
        _format_error(
            f"{context}: unit(s) {labels}{more} peak in the outermost ring of "
            f"distance_range={distance_range} {units or ''}, and the nearest "
            f"object was beyond {distance_range[1]:g} for {time_beyond:.1f} s "
            f"({share:.0%} of the analyzed time).",
            why="Why: a field cut off by the range looks the same as a peak at "
            "its edge, so preferred_distance may be wrong",
            fix="widen distance_range (the default covers the whole environment)",
        ),
        UserWarning,
        stacklevel=3,
    )


class _ObjectVectorCounts(NamedTuple):
    """Unsmoothed object-vector counts with the geometry and provenance used."""

    spike_counts: NDArray[np.float64]  # (n_units, n_bins), unsmoothed
    occupancy: NDArray[np.float64]  # (n_bins,) seconds
    polar_env: PolarEnvironment  # polar geometry (distance in env units, angle in rad)
    bandwidth: float | None  # resolved kernel bandwidth; None for "binned"
    analysis_settings: dict[str, Any]
    n_spikes_outside_intervals: NDArray[np.int64]  # (n_units,) not analyzed
    time_beyond: float  # seconds the nearest object was beyond the range


def _count_object_vectors(
    spike_trains: Sequence[NDArray[np.float64]],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: NDArray[np.float64],
    object_names: list[str] | None,
    env: Environment | None,
    *,
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
    metric: DistanceMetric,
    max_gap: float | None,
    epochs: Any,
    spike_window: Any,
    method: str,
    bandwidth: float | None,
    min_occupancy: float,
    backend: str,
    n_jobs: int = 1,
    frame_bins: NDArray[np.int64] | None = None,
    polar_env: PolarEnvironment | None = None,
    break_before: NDArray[np.bool_] | None = None,
) -> _ObjectVectorCounts:
    """Bin egocentric coordinates, count spikes and build provenance.

    Parameters
    ----------
    spike_trains : sequence of ndarray of shape (n_spikes_i,)
        Validated spike times, one array per unit.
    times : ndarray of shape (n_time,)
        Trajectory timestamps in seconds.
    positions : ndarray of shape (n_time, n_dims)
        Trajectory positions.
    headings : ndarray of shape (n_time,) or None
        Heading in radians, or ``None`` for the allocentric frame.
    object_positions : ndarray
        Resolved object coordinates.
    object_names : sequence of str or None
        Landmark names recorded in the provenance.
    env : Environment or None
        Environment for geodesic distances and units.
    distance_range, n_distance_bins, n_direction_bins, metric
        Polar binning settings.
    max_gap : float or None
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs, spike_window
        Resolved time windows.
    method, bandwidth, min_occupancy, backend
        Smoothing settings; ``bandwidth`` may be ``None`` for the default.
    n_jobs : int, default=1
        Parallel jobs for spike counting.
    frame_bins, polar_env : optional
        Trajectory-only geometry reused from an earlier call.
    break_before : ndarray of bool, shape (n_time,), or None
        Validated declared recording breaks.

    Returns
    -------
    spike_counts : ndarray of shape (n_units, n_bins)
        Spike counts per polar bin.
    occupancy : ndarray of shape (n_bins,)
        Seconds spent in each polar bin.
    polar_env : PolarEnvironment
        Polar geometry carrying the environment's length units.
    bandwidth : float or None
        Resolved kernel bandwidth; ``None`` for ``"binned"``.
    analysis_settings : dict
        Provenance settings for the result.
    """
    from neurospatial.encoding._egocentric_counts import prepare_egocentric_binning
    from neurospatial.encoding._provenance import (
        rate_analysis_settings,
        spikes_outside_intervals,
    )
    from neurospatial.encoding._smoothing import resolve_spatial_bandwidth

    # Keep the prepared polar geometry separate from the allocentric env.
    prepared, polar_env, time_beyond = prepare_egocentric_binning(
        times,
        positions,
        headings,
        object_positions,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        env=env,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        frame_bins=frame_bins,
        polar_env=polar_env,
    )
    spike_counts, occupancy = prepared.count(spike_trains, n_jobs=n_jobs)
    time_in_range = float(occupancy.sum())
    analyzed = time_in_range + time_beyond

    # Polar graph distances are physical lengths; angles remain radians.
    polar_env = polar_env.with_metadata(units=env.units if env is not None else None)
    bandwidth = (
        None if method == "binned" else resolve_spatial_bandwidth(polar_env, bandwidth)
    )
    settings = rate_analysis_settings(
        prepared.support,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
        direction_frame="allocentric" if headings is None else "egocentric",
        distance_range=[float(value) for value in distance_range],
        n_distance_bins=int(n_distance_bins),
        n_direction_bins=int(n_direction_bins),
        metric=metric,
        objects=object_names,
        time_beyond_distance_range=time_beyond,
        fraction_of_time_in_distance_range=(
            time_in_range / analyzed if analyzed > 0 else float("nan")
        ),
    )
    return _ObjectVectorCounts(
        spike_counts,
        occupancy,
        polar_env,
        bandwidth,
        settings,
        spikes_outside_intervals(spike_trains, prepared.support),
        time_beyond,
    )


def _object_vector_rate(
    env: Environment | None,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: BackendType = "numpy",
    context: str,
) -> ObjectVectorRateResult:
    """Compute either frame through the shared binning and smoothing path."""
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    from neurospatial.encoding._backend import (
        is_jax_available,
        validate_and_resolve_backend,
    )
    from neurospatial.encoding._egocentric_counts import resolve_object_positions
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
    )
    from neurospatial.encoding._validation import (
        validate_spike_times,
        validate_trajectory,
    )

    _check_object_vector_env(env, headings, context=context)

    # Raises ValueError for unknown names and ImportError if jax is unavailable;
    # "auto" resolves to "numpy" or "jax".
    resolved_backend = validate_and_resolve_backend(backend)

    _validate_object_vector_metric(env, metric, context=context)
    units = env.units if env is not None else None
    require_known_bandwidth(units, bandwidth, method, context=context)

    _validate_smoothing_parameters(method, 1.0 if bandwidth is None else bandwidth)

    # Convert inputs to arrays (1D required for spike_times/times/headings)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = None if headings is None else np.asarray(headings, dtype=np.float64)

    validate_trajectory(
        times,
        positions=positions,
        headings=headings,
        context=context,
        n_dims=env.n_dims if env is not None else None,
    )
    # Coordinates, or each named landmark's nearest point at every sample.
    object_positions, object_names = resolve_object_positions(
        object_positions, env=env, positions=positions, metric=metric
    )
    distance_range, n_distance_bins, n_direction_bins = _object_vector_grid(
        env,
        object_positions,
        metric=metric,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        context=context,
    )
    validate_spike_times(
        spike_times,
        context=context,
    )
    breaks = validate_break_before(break_before, len(times), call=context)

    # Reuse the batch binning path for the single-neuron API so egocentric
    # coordinates are computed once and shared by spike counts and occupancy.
    counts = _count_object_vectors(
        [spike_times],
        times,
        positions,
        headings,
        object_positions,
        object_names,
        env,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
        break_before=breaks,
    )
    spike_counts = counts.spike_counts[0]

    firing_rate = _egocentric_firing_rate(
        counts.polar_env,
        spike_counts,
        counts.occupancy,
        method=method,
        bandwidth=counts.bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    _warn_if_directions_unsampled(
        counts.occupancy,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        direction_frame="allocentric" if headings is None else "egocentric",
        units=units,
        extended_landmarks=object_names is not None and object_positions.ndim == 3,
        context=context,
    )
    _warn_if_peak_at_range_edge(
        firing_rate,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        distance_range=distance_range,
        time_beyond=counts.time_beyond,
        time_in_range=float(counts.occupancy.sum()),
        unit_ids=[0],
        units=units,
        context=context,
    )

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rate is already JAX from smooth_rate_map)
    occupancy_out: ArrayLike = counts.occupancy
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy_out = jnp.asarray(counts.occupancy, dtype=jnp.float64)

    # Return result
    return ObjectVectorRateResult(
        min_occupancy=min_occupancy,
        direction_frame="allocentric" if headings is None else "egocentric",
        firing_rate=firing_rate,
        occupancy=occupancy_out,
        env=counts.polar_env,
        method=method,
        bandwidth=counts.bandwidth,
        n_spikes_outside_intervals=int(counts.n_spikes_outside_intervals[0]),
        spike_window=resolved_spike_window,
        analysis_settings=counts.analysis_settings,
    )


def compute_egocentric_rates(
    env: Environment | None,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> ObjectVectorRatesResult:
    """Compute egocentric firing rates for multiple neurons.

    This is the batch version of ``compute_egocentric_rate`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (egocentric coordinates, occupancy) once and optionally
    parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : sequence of arrays, 2D array, or pynapple TsGroup
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        All formats are coerced to per-neuron spike trains via
        ``_as_spike_trains_with_ids()``.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal position coordinates at each time sample. NaN values (in
        positions or headings) are treated as missing data and excluded from
        occupancy and firing-rate computation; callers do not need to
        pre-filter tracking dropouts.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, **allocentric
        world-frame convention**: 0 = East, π/2 = North, π = West,
        -π/2 = South, wrapped to [-π, π]). The allocentric→egocentric
        transform is applied internally; pass world-frame headings, not
        animal-frame angles.
    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.

    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.

    Returns
    -------
    ObjectVectorRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each egocentric bin in seconds, shape ``(n_bins,)``
        - ``env``: The egocentric polar environment
        - ``distance_range``: Distance range used
        - ``n_distance_bins``: Number of distance bins
        - ``n_direction_bins``: Number of direction bins

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    Raises
    ------
    ValueError
        If ``metric="geodesic"`` but ``env`` is None.
        If ``metric`` is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_egocentric_rate : Single-neuron version
    ObjectVectorRatesResult : Result class with batch methods
    compute_spatial_rates : Standard spatial rates (by animal position)

    Warns
    -----
    UserWarning
        When at least five units are all silent for at least 60 seconds of
        continuously tracked time and ``spike_window`` was not supplied.
        This is a heuristic for possible recording outages: it cannot detect
        an outage for a single unit or one shorter than 60 seconds, and its
        absence is not proof that recording coverage is correct.

    Notes
    -----
    An interval is analyzed only if it passes the gap and bounds checks
    and lies inside ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    **Efficiency advantages over calling ``compute_egocentric_rate`` in a loop**:

    1. Egocentric coordinates (distance, bearing to nearest object) are
       computed once and shared across all neurons
    2. Occupancy is computed once and shared
    3. Diffusion kernel (for ``diffusion_kde`` method) is computed once
    4. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      efficiency matters. The overhead of precomputing shared quantities
      is amortized over multiple neurons.
    - **Single** (``compute_egocentric_rate``): Processing 1-2 neurons, or when
      you need fine-grained control over individual neurons.

    **Coordinate convention**: Direction uses egocentric (animal-centered)
    coordinates where 0=ahead, +pi/2=left, -pi/2=right.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import compute_egocentric_rates

    >>> # Create trajectory and objects
    >>> rng = np.random.default_rng(42)
    >>> times = np.linspace(0, 40, 1000)
    >>> positions = rng.uniform(10, 90, (1000, 2))
    >>> headings = rng.uniform(-np.pi, np.pi, 1000)
    >>> object_positions = np.array([[50.0, 50.0], [25.0, 75.0]])

    >>> # Spike times for 3 neurons
    >>> spike_times = [
    ...     np.sort(rng.uniform(0, 40, 100)),  # Neuron 0
    ...     np.sort(rng.uniform(0, 40, 150)),  # Neuron 1
    ...     np.sort(rng.uniform(0, 40, 50)),  # Neuron 2
    ... ]

    >>> # Compute egocentric rates for all neurons
    >>> result = compute_egocentric_rates(
    ...     None,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     headings,
    ...     object_positions,
    ...     n_jobs=2,  # Parallel spike binning,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )

    >>> # Access results
    >>> print(f"Number of neurons: {len(result)}")
    Number of neurons: 3
    >>> print(f"Firing rates shape: {result.firing_rates.shape}")
    Firing rates shape: (3, 120)

    >>> # Iterate over neurons
    >>> for i, single in enumerate(result):
    ...     pref_dist = single.preferred_distance()
    ...     pref_dir = single.preferred_direction()
    ...     print(f"Neuron {i}: {pref_dist:.1f} cm at {np.degrees(pref_dir):.0f} deg")
    Neuron 0: 2.5 cm at 15 deg
    Neuron 1: 7.5 cm at 45 deg
    Neuron 2: 17.5 cm at 135 deg

    >>> # Per-unit scalar summary (one row per unit)
    >>> summary = result.summary_table()
    >>> len(summary)
    3
    >>> # Dense per-bin frame (one row per (unit, bin))
    >>> df = result.to_dataframe()
    >>> len(df) == 3 * result.env.n_bins
    True

    >>> # Use 2D array with NaN padding
    >>> spike_times_2d = np.array(
    ...     [
    ...         [0.1, 0.5, 1.0, np.nan],
    ...         [0.2, 0.3, 0.8, 1.2],
    ...     ]
    ... )
    >>> result2 = compute_egocentric_rates(
    ...     None,
    ...     spike_times_2d,
    ...     times,
    ...     positions,
    ...     headings,
    ...     object_positions,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )
    >>> len(result2)
    2

    References
    ----------
    .. [1] Wang, C., et al. (2018). Egocentric coding of external items in the
           lateral entorhinal cortex. Science, 362, 945-949.
           doi:10.1126/science.aau4940.
    """
    if headings is None:
        raise ValueError(
            _format_error(
                "compute_egocentric_rates: headings is required for egocentric bearing.",
                why="Why: animal-relative direction needs the heading at each sample",
                fix="pass headings, or use compute_object_vector_rates without headings",
            )
        )
    return _object_vector_rates(
        env,
        spike_times,
        times,
        positions,
        headings,
        object_positions,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        n_jobs=n_jobs,
        backend=backend,
        unit_ids=unit_ids,
        context="compute_egocentric_rates",
    )


def compute_object_vector_rates(
    env: Environment | None,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> ObjectVectorRatesResult:
    """Compute allocentric firing rates for multiple neurons.

    This is the batch version of ``compute_object_vector_rate(None)`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (allocentric coordinates, occupancy) once and optionally
    parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : sequence of arrays, 2D array, or pynapple TsGroup
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        All formats are coerced to per-neuron spike trains via
        ``_as_spike_trains_with_ids()``.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Animal position coordinates at each time sample. NaN values (in
        positions) are treated as missing data and excluded from
        occupancy and firing-rate computation; callers do not need to
        pre-filter tracking dropouts.
    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.

    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.

    Returns
    -------
    ObjectVectorRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each allocentric bin in seconds, shape ``(n_bins,)``
        - ``env``: The allocentric polar environment
        - ``distance_range``: Distance range used
        - ``n_distance_bins``: Number of distance bins
        - ``n_direction_bins``: Number of direction bins

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    Raises
    ------
    ValueError
        If ``metric="geodesic"`` but ``env`` is None.
        If ``metric`` is not one of the valid options.
        If inputs have mismatched lengths.

    See Also
    --------
    compute_object_vector_rate : Single-neuron version
    ObjectVectorRatesResult : Result class with batch methods
    compute_spatial_rates : Standard spatial rates (by animal position)

    Warns
    -----
    UserWarning
        When at least five units are all silent for at least 60 seconds of
        continuously tracked time and ``spike_window`` was not supplied.
        This is a heuristic for possible recording outages: it cannot detect
        an outage for a single unit or one shorter than 60 seconds, and its
        absence is not proof that recording coverage is correct.

    Notes
    -----
    An interval is analyzed only if it passes the gap and bounds checks
    and lies inside ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    **Efficiency advantages over calling ``compute_object_vector_rate(None)`` in a loop**:

    1. Object-vector coordinates (distance, bearing to nearest object) are
       computed once and shared across all neurons
    2. Occupancy is computed once and shared
    3. Diffusion kernel (for ``diffusion_kde`` method) is computed once
    4. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      efficiency matters. The overhead of precomputing shared quantities
      is amortized over multiple neurons.
    - **Single** (``compute_object_vector_rate``): Processing 1-2 neurons, or when
      you need fine-grained control over individual neurons.

    **Coordinate convention**: Direction uses allocentric (world-centered)
    coordinates where 0=East, +pi/2=North, -pi/2=South.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import compute_object_vector_rates
    >>> rng = np.random.default_rng(42)
    >>> times = np.arange(0.0, 40.0, 0.04)
    >>> positions = rng.uniform(10, 90, (len(times), 2))
    >>> objects = np.array([[50.0, 50.0]])
    >>> train = np.sort(rng.uniform(0, 39.9, 100))
    >>> result = compute_object_vector_rates(
    ...     None,
    ...     [train, train],
    ...     times,
    ...     positions,
    ...     objects,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )
    >>> result.direction_frame
    'allocentric'
    >>> result.firing_rates.shape
    (2, 120)

    References
    ----------
    .. [1] Høydal, Ø. A., et al. (2019). Object-vector coding in the medial
           entorhinal cortex. Nature, 568, 400-404.
           doi:10.1038/s41586-019-1077-7.
    """
    return _object_vector_rates(
        env,
        spike_times,
        times,
        positions,
        None,
        object_positions,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        n_jobs=n_jobs,
        backend=backend,
        unit_ids=unit_ids,
        context="compute_object_vector_rates",
    )


def _object_vector_rates(
    env: Environment | None,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    context: str,
    frame_bins: NDArray[np.int64] | None = None,
    polar_env: PolarEnvironment | None = None,
    diagnose: bool = True,
) -> ObjectVectorRatesResult:
    """Compute either frame through the shared binning and smoothing path.

    ``frame_bins`` and ``polar_env`` reuse trajectory-only geometry computed
    by a caller that bins many spike sets on one trajectory
    (the shuffle null), which also passes ``diagnose=False`` so the input
    warnings are not repeated for every shuffle.
    """
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    from neurospatial._spikes import as_labelled_spike_trains
    from neurospatial.encoding._backend import (
        is_jax_available,
        validate_and_resolve_backend,
    )
    from neurospatial.encoding._egocentric_counts import resolve_object_positions
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        smooth_rate_maps_batch,
    )
    from neurospatial.encoding._validation import (
        validate_spike_trains,
        validate_trajectory,
    )

    _check_object_vector_env(env, headings, context=context)

    # Raises ValueError for unknown names and ImportError if jax is unavailable;
    # "auto" resolves to "numpy" or "jax".
    resolved_backend = validate_and_resolve_backend(backend)

    _validate_smoothing_parameters(method, 1.0 if bandwidth is None else bandwidth)

    _validate_object_vector_metric(env, metric, context=context)
    units = env.units if env is not None else None
    require_known_bandwidth(units, bandwidth, method, context=context)

    # Normalize spike times to canonical list-of-arrays format and resolve the
    # per-unit identity labels (defaults to arange). A labelled input keeps its
    # own labels; a differing unit_ids= raises.
    spike_times_list, resolved_unit_ids, labels_supplied = as_labelled_spike_trains(
        spike_times, unit_ids, context=context
    )
    n_neurons = len(spike_times_list)

    # Convert inputs to arrays (1D required for times/headings)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = None if headings is None else np.asarray(headings, dtype=np.float64)

    validate_trajectory(
        times,
        positions=positions,
        headings=headings,
        context=context,
        n_dims=env.n_dims if env is not None else None,
    )
    # Coordinates, or each named landmark's nearest point at every sample.
    object_positions, object_names = resolve_object_positions(
        object_positions, env=env, positions=positions, metric=metric
    )
    distance_range, n_distance_bins, n_direction_bins = _object_vector_grid(
        env,
        object_positions,
        metric=metric,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        context=context,
    )
    validate_spike_trains(spike_times_list, context=context)
    breaks = validate_break_before(break_before, len(times), call=context)

    # Recording coverage uses tracked runs, independently of speed or frame bins.
    _warn_if_population_silent(
        spike_times_list,
        times,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=breaks,
    )

    # Retain the exact binning support for result provenance.
    counts = _count_object_vectors(
        spike_times_list,
        times,
        positions,
        headings,
        object_positions,
        object_names,
        env,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
        n_jobs=n_jobs,
        frame_bins=frame_bins,
        polar_env=polar_env,
        break_before=breaks,
    )

    # Compute firing rates. The "binned" method uses the raw bin rate (no graph
    # smoothing): diffusion over the polar env bleeds rate across distance rings
    # and erases distance tuning (see _raw_polar_rate). Other methods smooth.
    firing_rates: ArrayLike
    if n_neurons == 0:
        firing_rates = np.empty((0, counts.polar_env.n_bins), dtype=np.float64)
    elif method == "binned":
        _warn_if_fully_masked(counts.occupancy, min_occupancy)
        firing_rates = np.stack(
            [
                _raw_polar_rate(unit_counts, counts.occupancy, min_occupancy)
                for unit_counts in counts.spike_counts
            ]
        )
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            firing_rates = jnp.asarray(firing_rates, dtype=jnp.float64)
    else:
        assert counts.bandwidth is not None  # resolved for every kernel method
        # smooth_rate_maps_batch dispatches to JAX or NumPy based on backend
        firing_rates = smooth_rate_maps_batch(
            counts.polar_env,
            counts.spike_counts,
            counts.occupancy,
            method=method,
            bandwidth=counts.bandwidth,
            min_occupancy=min_occupancy,
            backend=resolved_backend,
        )

    if diagnose:
        _warn_if_directions_unsampled(
            counts.occupancy,
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
            direction_frame="allocentric" if headings is None else "egocentric",
            units=units,
            extended_landmarks=(
                object_names is not None and object_positions.ndim == 3
            ),
            context=context,
        )
        _warn_if_peak_at_range_edge(
            firing_rates,
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
            distance_range=distance_range,
            time_beyond=counts.time_beyond,
            time_in_range=float(counts.occupancy.sum()),
            unit_ids=list(resolved_unit_ids),
            units=units,
            context=context,
        )

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rates is already JAX from smooth_rate_maps_batch)
    occupancy_out: ArrayLike = counts.occupancy
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy_out = jnp.asarray(counts.occupancy, dtype=jnp.float64)
        firing_rates = jnp.asarray(firing_rates, dtype=jnp.float64)

    # Return result
    return ObjectVectorRatesResult(
        min_occupancy=min_occupancy,
        direction_frame="allocentric" if headings is None else "egocentric",
        firing_rates=firing_rates,
        occupancy=occupancy_out,
        env=counts.polar_env,
        method=method,
        bandwidth=counts.bandwidth,
        unit_ids=resolved_unit_ids if labels_supplied else None,
        unit_table=getattr(spike_times, "unit_table", None),
        n_spikes_outside_intervals=counts.n_spikes_outside_intervals,
        spike_window=resolved_spike_window,
        analysis_settings=counts.analysis_settings,
    )


# ==============================================================================
# Convenience Functions for Object-Vector Cell Analysis
# ==============================================================================


def _mean_resultant_length(
    angles: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute mean resultant length of circular data.

    Parameters
    ----------
    angles : array of float
        Angles in radians.
    weights : array of float, optional
        Weights for each angle. If None, uniform weights.

    Returns
    -------
    float
        Mean resultant length in [0, 1].
    """
    if weights is None:
        weights = np.ones_like(angles)

    weights = weights / np.sum(weights)
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))

    return float(np.sqrt(x**2 + y**2))


def object_vector_score(rate_map: ObjectVectorRateResult | ArrayLike) -> float:
    """Combined distance and direction selectivity of an object-vector map.

    .. math::

        s_{OV} = \\frac{1 - a_d}{1 - 1/K} \\, R_\\theta,
        \\qquad a_d = \\frac{\\bar{f}^2}{\\overline{f^2}}

    Both profiles are taken over the complete rings, the :math:`K` rings
    with a rate in every direction bin; around an object in open space they
    form the disc out to the nearest wall.

    - The distance profile :math:`f` is each complete ring's mean rate;
      :math:`a_d` is its sparsity, from :math:`1/K` (all firing in one ring)
      to 1 (flat), so the distance factor runs from 0 (flat) to 1 (one ring).
    - The direction profile is each direction bin's mean rate over the
      complete rings; :math:`R_\\theta` is its mean resultant length, from 0
      (flat) to 1 (one direction).

    A map tuned to direction alone has a flat distance profile and scores
    0, as does a map tuned to distance alone. Because both profiles cover
    the same rings and every direction, neither reads missing coverage as
    tuning: a ring sampled only on one side of the object (beyond a wall)
    is left out rather than biasing either profile.

    Parameters
    ----------
    rate_map : ObjectVectorRateResult or array-like
        A single-unit result, or its firing rate reshaped distance-major to
        shape ``(n_distance_bins, n_direction_bins)``:
        ``result.firing_rate.reshape(result.n_distance_bins,
        result.n_direction_bins)``. An array's direction bins are taken to
        span [-pi, pi) evenly, as the encoders build them.

    Returns
    -------
    float
        Score in [0, 1]; higher means sharper tuning to one distance and
        one direction. NaN when fewer than two rings are complete (the
        distance factor compares rings) or the complete rings are silent
        (0 / 0).

    Raises
    ------
    ValueError
        If ``rate_map`` is not 2-D.

    Notes
    -----
    The score describes the complete rings only. Near a wall or in a corner
    they end close to the object, and a field beyond them does not count:
    a real object-vector cell tuned 20 cm from an object near a wall can
    score near 0. Read the map, or place the object away from the walls.

    This is a descriptive score with no null distribution; compare it with
    shuffled spike trains before calling a cell object-vector tuned. Within
    one session a place field near the object scores like an object-vector
    field; only a vector that follows the object when it is moved (vector
    maps correlated across object positions, Høydal et al., 2019) tells
    them apart.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import object_vector_score
    >>> # One sharp (distance, direction) bin over a low baseline
    >>> tc = np.full((10, 12), 0.01)
    >>> tc[5, 6] = 20.0
    >>> bool(object_vector_score(tc) > 0.8)
    True
    >>> # Tuned to direction alone: no distance selectivity
    >>> by_direction = np.tile(np.linspace(1.0, 5.0, 12), (10, 1))
    >>> float(object_vector_score(by_direction))
    0.0

    See Also
    --------
    is_object_vector_cell : Classify neuron as OVC
    ObjectVectorRateResult.is_object_vector_cell : Classifier method on result object
    """
    if isinstance(rate_map, ObjectVectorRateResult):
        tuning_curve = np.asarray(
            _to_numpy(rate_map.firing_rate), dtype=np.float64
        ).reshape(rate_map.n_distance_bins, rate_map.n_direction_bins)
        edges = np.asarray(rate_map.env.grid_edges[1], dtype=np.float64)
        dir_bin_centers = (edges[:-1] + edges[1:]) / 2
    else:
        tuning_curve = np.asarray(rate_map, dtype=np.float64)
        if tuning_curve.ndim != 2:
            raise ValueError(
                _format_error(
                    "object_vector_score needs a (distance, direction) rate map "
                    f"of shape (n_distance_bins, n_direction_bins); got shape "
                    f"{tuning_curve.shape}.",
                    fix="pass the ObjectVectorRateResult itself, or "
                    "result.firing_rate.reshape(result.n_distance_bins, "
                    "result.n_direction_bins)",
                )
            )
        n_dir = tuning_curve.shape[1]
        direction_edges = np.linspace(-np.pi, np.pi, n_dir + 1)
        dir_bin_centers = (direction_edges[:-1] + direction_edges[1:]) / 2

    disc = tuning_curve[np.isfinite(tuning_curve).all(axis=1)]
    n_rings = len(disc)
    if n_rings < 2:
        return float("nan")
    distance_profile = disc.mean(axis=1)
    mean_square = float(np.mean(distance_profile**2))
    if mean_square == 0:
        return float("nan")
    sparsity = float(np.mean(distance_profile)) ** 2 / mean_square
    distance_selectivity = (1.0 - sparsity) / (1.0 - 1.0 / n_rings)
    direction_selectivity = _mean_resultant_length(
        dir_bin_centers, weights=disc.mean(axis=0)
    )
    return float(np.clip(distance_selectivity * direction_selectivity, 0.0, 1.0))


def is_object_vector_cell(
    env: Environment | None,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    criterion: Literal["threshold", "shuffle"] = "threshold",
    min_info: float | None = None,
    alpha: float | None = None,
    n_shuffles: int | None = None,
    min_shift: float | None = None,
    rng: np.random.Generator | int | None = None,
    unit_id: Hashable | None = None,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: BackendType = "numpy",
) -> bool:
    """Classify one neuron by the object_vector_cell screen or circular-shift test.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    positions : ndarray, shape (n_samples, n_dims)
        Raw animal coordinates aligned with times, in environment units.

    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.

    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    criterion : {"threshold", "shuffle"}, default="threshold"
        Screen the observed statistic or test circular-shift significance.
    min_info : float or None, default=None
        Inclusive screen cutoff; None resolves to the family threshold constant.
    alpha : float or None, default=None
        P-value level for ``criterion="shuffle"``; None resolves to 0.05.
    n_shuffles : int or None, default=None
        Number of circular shifts in shuffle mode; None resolves to 1000.
    min_shift : float or None, default=None
        Minimum shift in analyzed seconds; None resolves to 20.0.
    rng : numpy.random.Generator, int or None, default=None
        Shuffle random source; an integer seed with the same unit label is stable
        across single and population calls.
    unit_id : hashable or None, default=None
        Shuffle stream label; None uses label 0. Match the population's label.

    Returns
    -------
    bool
        Whether the chosen criterion is met.

    Raises
    ------
    ValueError
        If the criterion, inputs, or mode-specific keywords are invalid.

    Notes
    -----
    The 0.3 bits/spike default is a screening convention. The screen
    compares bias-corrected information, ``spatial_information() -
    spatial_information_bias()``, with it: the information an untuned cell
    shows from Poisson noise alone is subtracted first (derivation in
    :meth:`~neurospatial.encoding.spatial.SpatialRateResult.spatial_information_bias`). For the default
    ``binned`` map that bias is (K - 1) / (2 ln(2) N_spikes) for K occupied
    polar bins, and units with fewer spikes than K fail, with a warning.
    The default grid covers the arena, so K grows with it: with the object
    at the center of a 100 x 100 cm box, 150-170 of the 348 polar bins (29
    rings x 12 directions) were occupied in 10 minutes. In 40 untuned
    Poisson units per spike count in that box (10-minute sessions,
    allocentric and egocentric), uncorrected information passed 0.3
    bits/spike in 100% of units at about 20, 60, 120 and 250 spikes and 0%
    at 600; corrected, none passed. An object-vector cell with a 2 Hz peak
    fired about 70 spikes in those 10 minutes, too few for the screen; pass
    a narrower distance_range or wider bins, or use the shuffle test.
    For publication, report a circular-shift test and its assumptions.

    Circular shifting costs about n_shuffles recomputes of the plural map.
    The shuffled statistic is the polar map's Skaggs information in this
    frame (``spatial_information()``), the same value the screen thresholds.
    Its null assumes stable firing statistics on the joined analyzed clock;
    recording gaps and excluded epochs are never shift destinations.
    Compare p_value < alpha; a significant association alone does not establish
    cell identity. Results keep no raw arrays or recompute closures.
    Threshold keywords belong only to the screen; shuffle keywords belong
    only to the shuffle. Passing a keyword for the other mode raises.

    See Also
    --------
    object_vector_cell_significance : Population significance on raw arrays.
    compute_object_vector_rate : Compute the map without classification.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import is_object_vector_cell
    >>> rng = np.random.default_rng(42)
    >>> times = np.arange(0, 40, 0.04)
    >>> positions = rng.uniform(10, 90, (len(times), 2))
    >>> objects = np.array([[50.0, 50.0]])
    >>> spikes = np.sort(rng.uniform(0, 39.9, 100))
    >>> result = is_object_vector_cell(
    ...     None,
    ...     spikes,
    ...     times,
    ...     positions,
    ...     objects,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )
    >>> type(result)
    <class 'bool'>
    """
    check_criterion(criterion, ("threshold", "shuffle"), call="is_object_vector_cell")
    check_mode_keywords(
        criterion,
        threshold={"min_info": min_info},
        shuffle={
            "n_shuffles": n_shuffles,
            "min_shift": min_shift,
            "rng": rng,
            "unit_id": unit_id,
            "alpha": alpha,
        },
        call="is_object_vector_cell",
    )
    break_before = validate_break_before(
        break_before, len(times), call="is_object_vector_cell"
    )
    if criterion == "shuffle":
        label = 0 if unit_id is None else unit_id
        level = _SHUFFLE_DEFAULTS["alpha"] if alpha is None else alpha
        shuffle_result = object_vector_cell_significance(
            env,
            [spike_times],
            times,
            positions,
            object_positions,
            unit_ids=[label],
            **resolve_shuffle_settings(n_shuffles, min_shift),
            rng=rng,
            distance_range=distance_range,
            distance_bin_size=distance_bin_size,
            direction_bin_size=direction_bin_size,
            metric=metric,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
            spike_window=spike_window,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            backend=backend,
        )[label]
        return shuffle_result.p_value < level
    min_info = OBJECT_VECTOR_THRESHOLDS["min_info"] if min_info is None else min_info
    result = compute_object_vector_rate(
        env,
        spike_times,
        times,
        positions,
        object_positions,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
    )
    return result.is_object_vector_cell(min_info=min_info)


def is_egocentric_object_vector_cell(
    env: Environment | None,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    criterion: Literal["threshold", "shuffle"] = "threshold",
    min_info: float | None = None,
    alpha: float | None = None,
    n_shuffles: int | None = None,
    min_shift: float | None = None,
    rng: np.random.Generator | int | None = None,
    unit_id: Hashable | None = None,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: BackendType = "numpy",
) -> bool:
    """Classify one neuron by the egocentric_object_vector_cell screen or circular-shift test.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    positions : ndarray, shape (n_samples, n_dims)
        Raw animal coordinates aligned with times, in environment units.

    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, **allocentric
        world-frame convention**: 0 = East, π/2 = North, π = West,
        -π/2 = South, wrapped to [-π, π]). The allocentric→egocentric
        transform is applied internally; pass world-frame headings, not
        animal-frame angles.
    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.

    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy

    criterion : {"threshold", "shuffle"}, default="threshold"
        Screen the observed statistic or test circular-shift significance.
    min_info : float or None, default=None
        Inclusive screen cutoff; None resolves to the family threshold constant.
    alpha : float or None, default=None
        P-value level for ``criterion="shuffle"``; None resolves to 0.05.
    n_shuffles : int or None, default=None
        Number of circular shifts in shuffle mode; None resolves to 1000.
    min_shift : float or None, default=None
        Minimum shift in analyzed seconds; None resolves to 20.0.
    rng : numpy.random.Generator, int or None, default=None
        Shuffle random source; an integer seed with the same unit label is stable
        across single and population calls.
    unit_id : hashable or None, default=None
        Shuffle stream label; None uses label 0. Match the population's label.

    Returns
    -------
    bool
        Whether the chosen criterion is met.

    Raises
    ------
    ValueError
        If the criterion, inputs, or mode-specific keywords are invalid.

    Notes
    -----
    The 0.3 bits/spike default is a screening convention. The screen
    compares bias-corrected information, ``spatial_information() -
    spatial_information_bias()``, with it: the information an untuned cell
    shows from Poisson noise alone is subtracted first (derivation in
    :meth:`~neurospatial.encoding.spatial.SpatialRateResult.spatial_information_bias`). For the default
    ``binned`` map that bias is (K - 1) / (2 ln(2) N_spikes) for K occupied
    polar bins, and units with fewer spikes than K fail, with a warning.
    The default grid covers the arena, so K grows with it: with the object
    at the center of a 100 x 100 cm box, 150-170 of the 348 polar bins (29
    rings x 12 directions) were occupied in 10 minutes. In 40 untuned
    Poisson units per spike count in that box (10-minute sessions,
    allocentric and egocentric), uncorrected information passed 0.3
    bits/spike in 100% of units at about 20, 60, 120 and 250 spikes and 0%
    at 600; corrected, none passed. An object-vector cell with a 2 Hz peak
    fired about 70 spikes in those 10 minutes, too few for the screen; pass
    a narrower distance_range or wider bins, or use the shuffle test.
    For publication, report a circular-shift test and its assumptions.

    Circular shifting costs about n_shuffles recomputes of the plural map.
    The shuffled statistic is the polar map's Skaggs information in this
    frame (``spatial_information()``), the same value the screen thresholds.
    Its null assumes stable firing statistics on the joined analyzed clock;
    recording gaps and excluded epochs are never shift destinations.
    Compare p_value < alpha; a significant association alone does not establish
    cell identity. Results keep no raw arrays or recompute closures.
    Threshold keywords belong only to the screen; shuffle keywords belong
    only to the shuffle. Passing a keyword for the other mode raises.

    See Also
    --------
    egocentric_object_vector_cell_significance : Population significance on raw arrays.
    compute_egocentric_rate : Compute the map without classification.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import is_egocentric_object_vector_cell
    >>> rng = np.random.default_rng(42)
    >>> times = np.arange(0, 40, 0.04)
    >>> positions = rng.uniform(10, 90, (len(times), 2))
    >>> headings = rng.uniform(-np.pi, np.pi, len(times))
    >>> objects = np.array([[50.0, 50.0]])
    >>> spikes = np.sort(rng.uniform(0, 39.9, 100))
    >>> result = is_egocentric_object_vector_cell(
    ...     None,
    ...     spikes,
    ...     times,
    ...     positions,
    ...     headings,
    ...     objects,
    ...     distance_range=(0.0, 50.0),
    ...     distance_bin_size=5.0,
    ... )
    >>> type(result)
    <class 'bool'>
    """
    check_criterion(
        criterion, ("threshold", "shuffle"), call="is_egocentric_object_vector_cell"
    )
    check_mode_keywords(
        criterion,
        threshold={"min_info": min_info},
        shuffle={
            "n_shuffles": n_shuffles,
            "min_shift": min_shift,
            "rng": rng,
            "unit_id": unit_id,
            "alpha": alpha,
        },
        call="is_egocentric_object_vector_cell",
    )
    break_before = validate_break_before(
        break_before, len(times), call="is_egocentric_object_vector_cell"
    )
    if criterion == "shuffle":
        label = 0 if unit_id is None else unit_id
        level = _SHUFFLE_DEFAULTS["alpha"] if alpha is None else alpha
        shuffle_result = egocentric_object_vector_cell_significance(
            env,
            [spike_times],
            times,
            positions,
            headings,
            object_positions,
            unit_ids=[label],
            **resolve_shuffle_settings(n_shuffles, min_shift),
            rng=rng,
            distance_range=distance_range,
            distance_bin_size=distance_bin_size,
            direction_bin_size=direction_bin_size,
            metric=metric,
            max_gap=max_gap,
            epochs=epochs,
            break_before=break_before,
            spike_window=spike_window,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            backend=backend,
        )[label]
        return shuffle_result.p_value < level
    min_info = OBJECT_VECTOR_THRESHOLDS["min_info"] if min_info is None else min_info
    result = compute_egocentric_rate(
        env,
        spike_times,
        times,
        positions,
        headings,
        object_positions,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
    )
    return result.is_object_vector_cell(min_info=min_info)


def plot_object_vector_tuning(
    result: ObjectVectorRateResult,
    ax: Axes | PolarAxes | None = None,
    *,
    show_peak: bool = True,
    add_colorbar: bool = False,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Axes | PolarAxes:
    """Plot object-vector tuning curve as polar heatmap.

    Creates a polar plot where:
    - Radial axis = distance from object
    - Angular axis = direction to object in result.direction_frame

    Parameters
    ----------
    result : ObjectVectorRateResult
        Result from either frame encoder; direction_frame sets the orientation.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure with polar projection.
    show_peak : bool, default=True
        If True, mark the peak location with a marker.
    add_colorbar : bool, default=False
        If True, add a colorbar.
    cmap : str, default='viridis'
        Colormap name.
    **kwargs : dict
        Additional keyword arguments passed to pcolormesh.

    Returns
    -------
    matplotlib.axes.Axes or matplotlib.projections.polar.PolarAxes
        The axes object with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import (
    ...     compute_egocentric_rate,
    ...     plot_object_vector_tuning,
    ... )
    >>> # Compute egocentric rate field
    >>> result = compute_egocentric_rate(None, ...)  # doctest: +SKIP
    >>> ax = plot_object_vector_tuning(result)  # doctest: +SKIP

    See Also
    --------
    ObjectVectorRateResult.plot : Basic plotting on result object
    """
    import matplotlib.pyplot as plt
    from matplotlib.projections.polar import PolarAxes as MPLPolarAxes

    # Reshape firing rate to 2D grid (distance x direction)
    firing_rate = np.asarray(result.firing_rate, dtype=np.float64)
    tuning_curve = firing_rate.reshape(result.n_distance_bins, result.n_direction_bins)

    # Create bin edges
    dist_min, dist_max = result.distance_range
    distance_bins = np.linspace(dist_min, dist_max, result.n_distance_bins + 1)
    direction_bins = np.linspace(-np.pi, np.pi, result.n_direction_bins + 1)

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Create mesh grid for polar plot
    theta, r = np.meshgrid(direction_bins, distance_bins)

    # Plot heatmap
    mesh = ax.pcolormesh(theta, r, tuning_curve, cmap=cmap, shading="flat", **kwargs)

    # Configure polar plot
    if isinstance(ax, MPLPolarAxes):
        if result.direction_frame == "allocentric":
            ax.set_theta_zero_location("E")
            ax.set_xlabel("direction to object (allocentric, 0 = East)")
        else:
            ax.set_theta_zero_location("N")  # 0 degrees at top (ahead)
        # Counter-clockwise: +π/2 = left of the animal is drawn on the left.
        ax.set_theta_direction(1)

    # Mark peak if requested
    if show_peak:
        valid_mask = np.isfinite(tuning_curve)
        if np.any(valid_mask):
            # Find peak
            peak_idx = np.unravel_index(np.nanargmax(tuning_curve), tuning_curve.shape)
            dist_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
            dir_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

            peak_r = dist_centers[peak_idx[0]]
            peak_theta = dir_centers[peak_idx[1]]

            ax.scatter(
                [peak_theta],
                [peak_r],
                color="red",
                s=100,
                marker="*",
                zorder=5,
                label="Peak",
            )

    # Add colorbar if requested
    if add_colorbar:
        plt.colorbar(mesh, ax=ax, label="Firing rate (Hz)")

    return ax


def _object_vector_significance(
    env: Environment | None,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: ObjectPositions,
    *,
    context: str,
    direction_frame: Literal["allocentric", "egocentric"],
    distance_range: tuple[float, float] | None,
    distance_bin_size: float | None,
    direction_bin_size: float,
    metric: DistanceMetric,
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
    spike_window: Any,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float | None,
    min_occupancy: float,
    n_jobs: int,
    backend: BackendType,
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    n_shuffles: int,
    min_shift: float,
    rng: np.random.Generator | int | None,
) -> dict[Hashable, ShuffleTestResult]:
    """Shared body of the allocentric and egocentric object-vector tests.

    ``direction_frame`` names the analysis; it is never inferred from whether
    ``headings`` is missing, so an egocentric call without headings raises.
    """
    egocentric = direction_frame == "egocentric"
    if egocentric and headings is None:
        raise ValueError(
            _format_error(
                f"{context}: headings is required for egocentric bearing.",
                why="Why: animal-relative direction needs the heading at each sample",
                fix=(
                    "pass headings, or use object_vector_cell_significance "
                    "without headings"
                ),
            )
        )
    arguments = (
        "spike_times, times, positions, headings, object_positions"
        if egocentric
        else "spike_times, times, positions, object_positions"
    )
    from neurospatial._intervals import resolve_time_windows
    from neurospatial._results import resolve_unit_ids
    from neurospatial._spikes import _as_spike_trains_with_ids
    from neurospatial.encoding._egocentric_counts import (
        _create_egocentric_environment,
        object_frame_bins,
        observed_frames,
        resolve_object_positions,
    )
    from neurospatial.encoding._significance import (
        run_shuffle_test,
        snapshot_options,
        to_shuffle_results,
    )
    from neurospatial.encoding._validation import (
        _validate_env_argument,
        validate_spike_times,
        validate_trajectory,
    )

    trains, input_ids = _as_spike_trains_with_ids(spike_times)
    trains = [np.array(train, dtype=np.float64, copy=True) for train in trains]
    times = np.array(times, dtype=np.float64, copy=True)
    positions = np.array(positions, dtype=np.float64, copy=True)
    if egocentric:
        headings = np.array(headings, dtype=np.float64, copy=True)
    ids = np.array(
        resolve_unit_ids(
            unit_ids,
            len(trains),
            input_ids=input_ids,
            context=context,
        ),
        copy=True,
    )
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    options: dict[str, Any] = {
        "distance_range": distance_range,
        "distance_bin_size": distance_bin_size,
        "direction_bin_size": direction_bin_size,
        "metric": metric,
        "max_gap": max_gap,
        "method": method,
        "bandwidth": bandwidth,
        "min_occupancy": min_occupancy,
        "n_jobs": n_jobs,
        "backend": backend,
    }
    options = snapshot_options(
        options,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        unit_ids=ids,
    )
    if env is not None:
        _validate_env_argument(
            env,
            context=context,
            arguments=arguments,
        )
        _require_planar_env(
            env,
            f"{context}()",
            measures="distance and direction to the nearest object",
        )
    validate_trajectory(times, positions=positions, headings=headings, context=context)
    for train in trains:
        validate_spike_times(train, context=context)
    breaks = validate_break_before(break_before, len(times), call=context)
    options["break_before"] = breaks
    _validate_object_vector_metric(env, metric, context=context)
    units = env.units if env is not None else None
    require_known_bandwidth(units, bandwidth, method, context=context)
    object_positions, object_names = resolve_object_positions(
        object_positions, env=env, positions=positions, metric=metric
    )
    distance_range, n_distance_bins, n_direction_bins = _object_vector_grid(
        env,
        object_positions,
        metric=metric,
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        context=context,
    )
    # Every shuffle rebuilds exactly this grid.
    options["distance_range"] = distance_range
    options["distance_bin_size"] = (
        distance_range[1] - distance_range[0]
    ) / n_distance_bins
    options["direction_bin_size"] = 2 * np.pi / n_direction_bins
    frame_bins = object_frame_bins(
        positions,
        headings,
        object_positions,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        env=env,
    )
    tracked = observed_frames(positions, headings, object_positions)
    mask = interval_valid_mask(
        times,
        start_bin=frame_bins,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=options["break_before"],
        sample_valid=tracked,
    )
    windows = run_time_bounds(times, mask)
    _warn_if_directions_unsampled(
        start_allocated_occupancy(
            hold_last_valid(frame_bins, tracked, fill=-1),
            np.diff(times),
            mask,
            n_distance_bins * n_direction_bins,
        ),
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        direction_frame=direction_frame,
        units=units,
        extended_landmarks=object_names is not None and object_positions.ndim == 3,
        context=context,
    )

    # The polar grid and each frame's polar bin depend only on the trajectory,
    # so every shuffle reuses them instead of rebuilding them.
    polar_env = _create_egocentric_environment(
        distance_range, n_distance_bins, n_direction_bins
    )

    def statistic(shifted: list[NDArray[np.float64]]) -> ArrayLike:
        return _object_vector_rates(
            env,
            shifted,
            times,
            positions,
            headings,
            object_positions,
            **options,
            context=(
                "compute_egocentric_rates"
                if egocentric
                else "compute_object_vector_rates"
            ),
            frame_bins=frame_bins,
            polar_env=polar_env,
            diagnose=False,
        ).spatial_information()

    observed, null = run_shuffle_test(
        statistic,
        trains,
        windows,
        ids,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        rng=rng,
    )
    return to_shuffle_results(observed, null, ids)


def object_vector_cell_significance(
    env: Environment | None,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> dict[Hashable, ShuffleTestResult]:
    """Test object vector tuning against circular spike-time shifts.

    The statistic is each unit's Skaggs information, in bits per spike, of
    its polar rate map (``ObjectVectorRatesResult.spatial_information()``).
    Recompute the population map for the supplied arrays and for each shifted
    train, using exactly the observed map's valid intervals. Shuffles preserve
    counts on the compressed analyzed clock and never enter recording gaps.
    This costs about n_shuffles population-map recomputes. The result's
    is_significant property uses 0.05; compare p_value < alpha for another level.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : sequence of arrays, 2D array, or pynapple TsGroup
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        All formats are coerced to per-neuron spike trains via
        ``_as_spike_trains_with_ids()``.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    positions : ndarray, shape (n_samples, n_dims)
        Raw animal coordinates aligned with times, in environment units.

    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.
    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.
    n_shuffles : int, default=1000
        Positive number of null map recomputes.
    min_shift : float, default=20.0
        Minimum shift in either direction, seconds of analyzed time.
    rng : numpy.random.Generator, int or None, default=None
        Integer seeds give stable per-unit shifts regardless of population order.

    Returns
    -------
    dict[Hashable, ShuffleTestResult]
        Per-unit observed score, null scores, corrected p-value and z-score,
        keyed by unit label in input order. Invalid observed scores yield NaN p-values.

    Raises
    ------
    ValueError
        For invalid inputs/settings, duplicate/conflicting unit labels, or
        insufficient analyzed time.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import object_vector_cell_significance
    >>> rng = np.random.default_rng(7)
    >>> times = np.arange(0.0, 60.0, 0.1)
    >>> xx, yy = np.meshgrid(np.arange(0, 101, 5), np.arange(0, 101, 5))
    >>> env = Environment.from_samples(
    ...     np.c_[xx.ravel(), yy.ravel()], bin_size=5.0, units="cm"
    ... )
    >>> positions = np.c_[50 + 20 * np.sin(times / 3), 50 + 20 * np.cos(times / 4)]
    >>> headings = np.sin(times / 5)
    >>> objects = np.array([[50.0, 50.0]])
    >>> trains = [np.sort(rng.uniform(0, 59.9, 40))]
    >>> results = object_vector_cell_significance(
    ...     env, trains, times, positions, objects, n_shuffles=20, rng=0
    ... )
    >>> list(results)
    [0]
    >>> results[0].n_shuffles
    20
    """
    return _object_vector_significance(
        env,
        spike_times,
        times,
        positions,
        None,
        object_positions,
        context="object_vector_cell_significance",
        direction_frame="allocentric",
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        n_jobs=n_jobs,
        backend=backend,
        unit_ids=unit_ids,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        rng=rng,
    )


def egocentric_object_vector_cell_significance(
    env: Environment | None,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] | None = None,
    distance_bin_size: float | None = None,
    direction_bin_size: float = DEFAULT_DIRECTION_BIN_SIZE,
    metric: DistanceMetric = "euclidean",
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> dict[Hashable, ShuffleTestResult]:
    """Test egocentric object vector tuning against circular spike-time shifts.

    The statistic is each unit's Skaggs information, in bits per spike, of
    its polar rate map (``ObjectVectorRatesResult.spatial_information()``).
    Recompute the population map for the supplied arrays and for each shifted
    train, using exactly the observed map's valid intervals. Shuffles preserve
    counts on the compressed analyzed clock and never enter recording gaps.
    This costs about n_shuffles population-map recomputes. The result's
    is_significant property uses 0.05; compare p_value < alpha for another level.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment the positions were recorded in. Required
        when ``metric="geodesic"`` (used to compute distances around
        obstacles). When given, a sample whose position lies outside its
        bins adds no occupancy and no spikes, as in place maps. May be
        ``None`` with
        ``metric="euclidean"``; then every tracked sample counts.
    spike_times : sequence of arrays, 2D array, or pynapple TsGroup
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        All formats are coerced to per-neuron spike trains via
        ``_as_spike_trains_with_ids()``.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    positions : ndarray, shape (n_samples, n_dims)
        Raw animal coordinates aligned with times, in environment units.

    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, **allocentric
        world-frame convention**: 0 = East, π/2 = North, π = West,
        -π/2 = South, wrapped to [-π, π]). The allocentric→egocentric
        transform is applied internally; pass world-frame headings, not
        animal-frame angles.
    object_positions : array-like, str or sequence of str
        The objects, as allocentric positions with shape ``(n_objects, 2)``,
        as positions at every sample with shape ``(n_time, n_objects, 2)``, or
        as names of ``env.landmarks``. A line or polygon landmark contributes
        its nearest point at each sample, so a monitor is measured from its
        closest edge rather than its center; names are recorded in the
        result's ``analysis_settings["objects"]``. The firing rate is
        computed relative to the *nearest* object at each timepoint.
    distance_range : tuple of float or None, default=None
        (min, max) distance to the object, in the positions' length units.
        Distances outside it are not included in the rate map. None covers
        every distance inside ``env`` from 0, in whole rings: the diagonal of
        its bounding box with ``metric="euclidean"`` (the same grid wherever
        the object is placed), or the farthest reachable bin from the nearest
        object with ``metric="geodesic"``. Required when ``env`` is None.
        ``analysis_settings["fraction_of_time_in_distance_range"]`` records
        the share of analyzed time the map covers, and a unit whose peak lies
        in the outermost ring while time was cut off raises a warning.
    distance_bin_size : float or None, default=None
        Width of each distance ring, in the positions' length units. None
        uses 5 cm converted to ``env.units``; pass it when the units are
        unknown. An explicit ``distance_range`` is divided into the whole
        number of rings nearest this width.
    direction_bin_size : float, default=pi/6
        Width of each direction bin in radians (30 degrees); the circle is
        divided into ``round(2 pi / direction_bin_size)`` bins. The 5 cm and
        30 degree defaults suit unsmoothed maps of 5-15 minute sessions;
        Høydal et al. (2019) used 2 cm by 5 degree bins smoothed with a
        2-bin Gaussian.
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="binned"
        Smoothing method to use:

        - **binned** (default): spikes divided by occupancy in each bin,
          with no smoothing.
        - **diffusion_kde**: diffusion over the polar bin graph.
        - **gaussian_kde**: Gaussian kernel over the polar bin graph.

        The kernel methods smooth across neighboring distance rings as well
        as directions, which blurs the distance tuning an object-vector map
        is meant to show; the default leaves each bin's rate unsmoothed.
    bandwidth : float or None, default=None
        Kernel width in the positions' length units. None uses 5 cm
        converted to ``env.units``; with no env, or units that are not a
        known length, a kernel method needs it explicitly. Unused for
        ``method="binned"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included. Bins with
        occupancy below this threshold are set to NaN.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {'numpy', 'jax', 'auto'}, default='numpy'
        Computation backend.

        - 'numpy': Use NumPy (always available)
        - 'jax': Use JAX for rate computation (requires JAX installation)
        - 'auto': Use JAX if available, otherwise NumPy
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.
    n_shuffles : int, default=1000
        Positive number of null map recomputes.
    min_shift : float, default=20.0
        Minimum shift in either direction, seconds of analyzed time.
    rng : numpy.random.Generator, int or None, default=None
        Integer seeds give stable per-unit shifts regardless of population order.

    Returns
    -------
    dict[Hashable, ShuffleTestResult]
        Per-unit observed score, null scores, corrected p-value and z-score,
        keyed by unit label in input order. Invalid observed scores yield NaN p-values.

    Raises
    ------
    ValueError
        For invalid inputs/settings, duplicate/conflicting unit labels, or
        insufficient analyzed time.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import egocentric_object_vector_cell_significance
    >>> rng = np.random.default_rng(7)
    >>> times = np.arange(0.0, 60.0, 0.1)
    >>> xx, yy = np.meshgrid(np.arange(0, 101, 5), np.arange(0, 101, 5))
    >>> env = Environment.from_samples(
    ...     np.c_[xx.ravel(), yy.ravel()], bin_size=5.0, units="cm"
    ... )
    >>> positions = np.c_[50 + 20 * np.sin(times / 3), 50 + 20 * np.cos(times / 4)]
    >>> headings = np.sin(times / 5)
    >>> objects = np.array([[50.0, 50.0]])
    >>> trains = [np.sort(rng.uniform(0, 59.9, 40))]
    >>> results = egocentric_object_vector_cell_significance(
    ...     env, trains, times, positions, headings, objects, n_shuffles=20, rng=0
    ... )
    >>> list(results)
    [0]
    >>> results[0].n_shuffles
    20
    """
    return _object_vector_significance(
        env,
        spike_times,
        times,
        positions,
        headings,
        object_positions,
        context="egocentric_object_vector_cell_significance",
        direction_frame="egocentric",
        distance_range=distance_range,
        distance_bin_size=distance_bin_size,
        direction_bin_size=direction_bin_size,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        n_jobs=n_jobs,
        backend=backend,
        unit_ids=unit_ids,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        rng=rng,
    )
