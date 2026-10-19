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
>>> result = compute_object_vector_rate(None, spikes, times, positions, [[50, 50]])
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

from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    interval_valid_mask,
    resolve_time_windows,
    run_time_bounds,
)
from neurospatial._results import UnitSelectionMixin, _copy_analysis_settings, label_at
from neurospatial.encoding._base import SpatialResultMixin, _to_numpy
from neurospatial.encoding._polar_results import _PolarRateMixin, polar_summary_frame
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
    rates: NDArray, occupancy: NDArray, *, min_info: float
) -> NDArray[np.bool_]:
    """The shared information-threshold rule for this rate family."""
    from neurospatial.encoding._metrics import batch_spatial_information

    return np.asarray(batch_spatial_information(rates, occupancy)) >= min_info


def _object_vector_summary_frame(
    result: ObjectVectorRateResult | ObjectVectorRatesResult,
    *,
    index: Sequence[Hashable],
) -> pd.DataFrame:
    """Build identical metric columns for single and population results."""
    rates = np.atleast_2d(_to_numpy(result._get_rates()))
    occupancy = _to_numpy(result.occupancy)
    return polar_summary_frame(
        result,
        index=index,
        extra={
            "is_object_vector_cell": _object_vector_classify(
                rates, occupancy, **OBJECT_VECTOR_THRESHOLDS
            )
        },
        extra_units={},
        thresholds=OBJECT_VECTOR_THRESHOLDS,
    )


@_copy_analysis_settings
@dataclass(frozen=True, repr=False)
class ObjectVectorRateResult(_PolarRateMixin, SpatialResultMixin):
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
    distance_range : tuple[float, float]
        Range of distances (min, max) covered by the polar environment.
    n_distance_bins : int
        Number of distance bins in the polar grid.
    n_direction_bins : int
        Number of direction bins in the polar grid.

    direction_frame : {"allocentric", "egocentric"}
        Required reference frame for the direction to the object.
        Allocentric: 0 = East, +pi/2 = North. Egocentric: 0 = ahead, +pi/2 = left.

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate by polar coordinates in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The egocentric polar environment.
    distance_range : tuple[float, float]
        Distance range (min, max).
    n_distance_bins : int
        Number of distance bins.
    n_direction_bins : int
        Number of direction bins.
    unit_id : int or str or None
        Identifier for this unit. Set automatically when indexing/iterating a
        population result (``rates[i].unit_id == rates.unit_ids[i]``); ``None``
        for a standalone single-unit computation.

    Other Parameters
    ----------------
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        family-specific settings. Preserved by indexing and table/xarray
        exports. None for manually constructed results without provenance.

    Notes
    -----
    ``spatial_support`` is a boolean mask over this result's bins. A bin must
    have positive occupancy, meet the recorded minimum occupancy and have
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
    ...     None, spike_times, times, positions, headings, object_positions
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
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    unit_id: int | str | None = None

    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _check_direction_frame(self.direction_frame, context="ObjectVectorRateResult")
        self._own_arrays()

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
        Physical units are in ``df.attrs["units"]``; a constant estimator
        is in ``df.attrs["method"]`` when recorded.

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
        ...     None, spike_times, times, positions, headings, object_positions
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
            Distance to object at peak firing rate, in the same units as
            the environment (typically cm). Uses nanargmax to handle NaN
            values in the firing rate map.

        Notes
        -----
        For object-vector cells, this represents the preferred distance to
        the object. A cell with preferred_distance=20 fires most when the
        object is 20 cm away from the animal.

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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> dist = result.preferred_distance()
        >>> print(f"Preferred distance: {dist:.1f} cm")
        Preferred distance: 2.5 cm

        See Also
        --------
        preferred_direction : Get direction component of peak response
        plot : Visualize the object-vector rate map
        """
        firing_rate = _to_numpy(self.firing_rate)
        peak_bin = np.nanargmax(firing_rate)
        bin_centers: NDArray[np.float64] = self.env.bin_centers
        return float(bin_centers[peak_bin, 0])

    def preferred_direction(self) -> float:
        """Direction from the animal to the object at peak firing rate.

        The result's ``direction_frame`` sets the convention: allocentric
        0 = East and +pi/2 = North; egocentric 0 = ahead and +pi/2 = left.
        The reverse, object-to-animal vector adds pi and wraps to [-pi, pi].

        Returns
        -------
        float
            Peak animal-to-object direction in radians, in the recorded frame.

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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> direction = result.preferred_direction()
        >>> print(f"Preferred direction: {np.degrees(direction):.1f}")
        Preferred direction: -45.0

        See Also
        --------
        preferred_distance : Get distance component of peak response
        plot : Visualize the object-vector rate map
        """
        firing_rate = _to_numpy(self.firing_rate)
        peak_bin = np.nanargmax(firing_rate)
        bin_centers: NDArray[np.float64] = self.env.bin_centers
        return float(bin_centers[peak_bin, 1])

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

        **Interpretation**:

        - Object-vector cells typically have 0.5-2.0+ bits/spike
        - Higher values indicate more selective tuning to distance/direction
        - Zero means uniform firing (no distance/direction selectivity)

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
        ...     None, spike_times, times, positions, headings, object_positions
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

    def is_object_vector_cell(self, *, min_info: float | None = None) -> bool:
        """Classify as object-vector cell based on spatial information in the recorded frame.

        Tests object-vector tuning in ``result.direction_frame``.

        A neuron is classified as a candidate in the recorded frame if its
        spatial information meets or exceeds the minimum threshold. OVCs fire when the
        animal is at a specific distance and direction from an object.

        Parameters
        ----------
        min_info : float or None, default=None
            Minimum spatial information in the recorded frame threshold in bits/spike.


        Returns
        -------
        bool
            True if spatial_information() >= min_info, False otherwise.

        Notes
        -----
        The 0.3 bits/spike default is this library's screening heuristic.
        Plug-in information is biased upward by approximately
        (n_bins - 1) / (2 ln(2) N_spikes). In 20 untuned 0.5 Hz Poisson units,
        egocentric 10 x 12 polar maps had median information 2.12, 1.41, 0.70,
        0.41 and 0.21 bits/spike at 1, 2, 5, 10 and 20 minutes (about 30, 60,
        150, 300 and 600 spikes). The screen flagged 20/20 at 1-10 minutes
        and 0/20 at 20 minutes. The allocentric screen also flagged 20/20
        in the seeded 10-minute fixture. Low counts can resemble tuning.
        For publication, report a circular-shift test and its assumptions.
        None thresholds resolve through OBJECT_VECTOR_THRESHOLDS.
        For a shuffle test, call object_vector_cell_significance(...) (allocentric) or
        egocentric_object_vector_cell_significance(...) (egocentric)
        or is_egocentric_object_vector_cell(..., criterion="shuffle") with the raw
        arrays; a result does not keep the arrays it was computed from.

        **Object-vector vs place cells**: Both may show high spatial
        information. Compare polar tuning in the appropriate direction frame
        with a Cartesian position map; an egocentric map is appropriate only
        for heading-relative bearing. A threshold screen alone does not
        distinguish object-vector tuning from a place-cell control.

        For more rigorous classification, consider also using:

        - Stability across sessions
        - Multiple objects (OVCs should generalize)
        - Shuffling controls

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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> result.is_object_vector_cell()
        True
        >>> result.is_object_vector_cell(min_info=0.5)
        True

        Uniform random spikes passing this screen illustrate finite-count
        information bias, rather than an object-vector cell identity.
        The allocentric circular-shift verdict tests the same untuned train:

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
        ... )
        False

        See Also
        --------
        spatial_information : Compute the metric used for classification
        """
        min_info = (
            OBJECT_VECTOR_THRESHOLDS["min_info"] if min_info is None else min_info
        )
        return bool(
            _object_vector_classify(
                np.atleast_2d(_to_numpy(self.firing_rate)),
                _to_numpy(self.occupancy),
                min_info=min_info,
            )[0]
        )


@_copy_analysis_settings
@dataclass(frozen=True, repr=False)
class ObjectVectorRatesResult(_PolarRateMixin, SpatialResultMixin, UnitSelectionMixin):
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
    distance_range : tuple[float, float]
        Range of distances (min, max) covered by the polar environment.
    n_distance_bins : int
        Number of distance bins in the polar grid.
    n_direction_bins : int
        Number of direction bins in the polar grid.

    direction_frame : {"allocentric", "egocentric"}
        Required reference frame for the direction to the object.
        Allocentric: 0 = East, +pi/2 = North. Egocentric: 0 = ahead, +pi/2 = left.

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rates for all neurons. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time in each bin in seconds. Shape is (n_bins,). Shared.
    env : Environment
        The egocentric polar environment.
    distance_range : tuple[float, float]
        Distance range (min, max).
    n_distance_bins : int
        Number of distance bins.
    n_direction_bins : int
        Number of direction bins.
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
    analysis_settings : dict or None
        Resolved analysis windows, retained time intervals, smoothing and
        family-specific settings. Preserved by indexing and table/xarray
        exports. None for manually constructed results without provenance.

    Notes
    -----
    ``spatial_support`` is a boolean mask over this result's bins. A bin must
    have positive occupancy, meet the recorded minimum occupancy and have
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
    ...     None, spike_times, times, positions, headings, object_positions
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
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int
    direction_frame: Literal["allocentric", "egocentric"] = field(kw_only=True)
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)

    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        from neurospatial._results import align_unit_table, resolve_unit_ids

        _check_direction_frame(self.direction_frame, context="ObjectVectorRatesResult")
        n_units = int(np.asarray(self.firing_rates).shape[0])
        object.__setattr__(
            self,
            "unit_ids",
            resolve_unit_ids(self.unit_ids, n_units),
        )
        object.__setattr__(
            self,
            "unit_table",
            align_unit_table(
                self.unit_table, self.unit_ids, context="ObjectVectorRatesResult"
            ),
        )
        self._own_arrays()

    def __len__(self) -> int:
        """Return the number of units.

        Returns
        -------
        int
            Number of neurons in the batch.

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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> len(result)
        3
        """
        return len(self.firing_rates)  # type: ignore[arg-type]

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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> single = result[0]
        >>> isinstance(single, ObjectVectorRateResult)
        True
        """
        idx = self._unit_position(idx)
        return ObjectVectorRateResult(
            firing_rate=self.firing_rates[idx],  # type: ignore[index]
            occupancy=self.occupancy,
            env=self.env,
            distance_range=self.distance_range,
            n_distance_bins=self.n_distance_bins,
            n_direction_bins=self.n_direction_bins,
            unit_id=label_at(self.unit_ids, idx),
            spike_window=self.spike_window,
            analysis_settings=self._analysis_attrs()["analysis_settings"],
            direction_frame=self.direction_frame,
        )

    def __iter__(self) -> Iterator[ObjectVectorRateResult]:
        """Iterate over single-neuron results.

        Yields
        ------
        ObjectVectorRateResult
            Object-vector rate result in the same frame for each neuron in order.

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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> peaks = [float(single.firing_rate.max()) for single in result]
        >>> len(peaks)
        2
        """
        for i in range(len(self)):
            yield self[i]

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
            Distance to object at peak firing rate for each neuron, in the
            same units as the environment (typically cm).

        Notes
        -----
        For object-vector cells, this represents the preferred distance to
        the object. A cell with preferred_distance=20 fires most when the
        object is 20 cm away from the animal.

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
        ...     None, spike_times, times, positions, headings, object_positions
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
        firing_rates = _to_numpy(self.firing_rates)
        bin_centers: NDArray[np.float64] = self.env.bin_centers

        # Peak (max-firing) bin per neuron, ignoring NaNs; then read off the
        # distance component (column 0 of bin_centers).
        peak_idx = np.nanargmax(firing_rates, axis=1)
        distances: NDArray[np.float64] = bin_centers[peak_idx, 0]
        return distances

    def preferred_directions(self) -> NDArray[np.float64]:
        """Directions from the animal to the object at peak firing rate.

        The result's ``direction_frame`` sets the convention: allocentric
        0 = East and +pi/2 = North; egocentric 0 = ahead and +pi/2 = left.
        The reverse, object-to-animal vector adds pi and wraps to [-pi, pi].

        Returns
        -------
        NDArray[np.float64], shape (n_neurons,)
            Peak animal-to-object direction in radians, in the recorded frame.

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
        ...     None, spike_times, times, positions, headings, object_positions
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
        firing_rates = _to_numpy(self.firing_rates)
        bin_centers: NDArray[np.float64] = self.env.bin_centers

        # Peak (max-firing) bin per neuron, ignoring NaNs; then read off the
        # direction component (column 1 of bin_centers).
        peak_idx = np.nanargmax(firing_rates, axis=1)
        directions: NDArray[np.float64] = bin_centers[peak_idx, 1]
        return directions

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
        ...     None, spike_times, times, positions, headings, object_positions
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

    def classify(self, *, min_info: float | None = None) -> NDArray[np.bool_]:
        """Classify neurons as object-vector cells.

        Tests object-vector tuning in ``result.direction_frame``.

        A neuron is classified as a candidate in the recorded frame if its
        spatial information meets or exceeds the minimum threshold. This is the
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
        The 0.3 bits/spike default is this library's screening heuristic.
        Plug-in information is biased upward by approximately
        (n_bins - 1) / (2 ln(2) N_spikes). In 20 untuned 0.5 Hz Poisson units,
        egocentric 10 x 12 polar maps had median information 2.12, 1.41, 0.70,
        0.41 and 0.21 bits/spike at 1, 2, 5, 10 and 20 minutes (about 30, 60,
        150, 300 and 600 spikes). The screen flagged 20/20 at 1-10 minutes
        and 0/20 at 20 minutes. The allocentric screen also flagged 20/20
        in the seeded 10-minute fixture. Low counts can resemble tuning.
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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> is_object_vector_cell = result.classify()
        >>> print(f"Found {is_object_vector_cell.sum()} OVCs")
        Found 3 OVCs

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
        return _object_vector_classify(
            _to_numpy(self.firing_rates), _to_numpy(self.occupancy), min_info=min_info
        )

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

            - preferred_distance: preferred distance to object (cm)
            - preferred_direction: preferred direction to object (radians in result.direction_frame)
            - preferred_direction_deg: preferred direction (degrees in the same frame)
            - peak_rate: maximum firing rate (Hz)
            - is_object_vector_cell: whether classified as OVC (using default threshold)

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
        Physical units are in ``df.attrs["units"]``; a constant estimator
        is in ``df.attrs["method"]`` when recorded.

        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``preferred_distances()``, ``classify()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["is_object_vector_cell"] == True]``
        - Sort: ``df.sort_values("preferred_distance")``
        - Top-N: ``df.nlargest(10, "peak_rate")``

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
        ...     None, spike_times, times, positions, headings, object_positions
        ... )
        >>> df = result.summary_table()
        >>> list(df.columns)
        ['peak_rate', 'preferred_distance', 'preferred_direction_deg', 'preferred_direction', 'is_object_vector_cell']
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
    bandwidth: float,
    min_occupancy: float,
    backend: Literal["numpy", "jax"],
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
    bandwidth : float
        Smoothing bandwidth, in environment units. Unused for ``"binned"``.
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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> ObjectVectorRateResult:
    """Compute egocentric firing rate for one neuron.

    This function computes a smoothed firing rate map in egocentric polar
    coordinates (distance and direction to nearest object). This is the key
    metric for identifying object-vector cells (OVCs).

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for egocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the egocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the egocentric polar grid. Covers the
        full circle (-π to π).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for egocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
    An interval is analyzed only if it passes the gap, speed and bounds
    checks and lies inside ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    The function uses the egocentric binning layer (``_egocentric_counts.py``)
    to convert spike times to spike counts based on distance and direction to
    the nearest object, then the smoothing layer (``_smoothing.py``) to compute
    the smoothed firing rate.

    **Algorithm**:

    1. Compute egocentric coordinates (distance, bearing) to nearest object
       at each behavioral frame
    2. Bin spikes by egocentric coordinates at spike time
    3. Compute egocentric occupancy (time spent at each distance/direction)
    4. Apply smoothing (method-dependent, see ``_smoothing.py``)

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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> ObjectVectorRateResult:
    """Compute allocentric firing rate for one neuron.

    This function computes a smoothed firing rate map in allocentric polar
    coordinates (distance and direction to nearest object). This is the key
    metric for identifying object-vector cells (OVCs).

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for allocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the allocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the allocentric polar grid. Covers the
        full circle (-π to π).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for allocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
    An interval is analyzed only if it passes the gap, speed and bounds
    checks and lies inside ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    The function uses the allocentric binning layer (``_egocentric_counts.py``)
    to convert spike times to spike counts based on distance and direction to
    the nearest object, then the smoothing layer (``_smoothing.py``) to compute
    the smoothed firing rate.

    **Algorithm**:

    1. Compute allocentric coordinates (distance, bearing) to nearest object
       at each behavioral frame
    2. Bin spikes by allocentric coordinates at spike time
    3. Compute allocentric occupancy (time spent at each distance/direction)
    4. Apply smoothing (method-dependent, see ``_smoothing.py``)

    **Coordinate convention**: Direction uses allocentric (world-centered)
    coordinates where 0=East, +π/2=North, -π/2=South.

    **Place cells vs object-vector cells**: For place cells, firing is
    determined by allocentric position. For OVCs, firing is determined by
    allocentric relationship to objects. Computing place field using
    ``compute_spatial_rate`` and OVC field using this function, then comparing
    spatial information, can help distinguish cell types.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import compute_object_vector_rate
    >>> rng = np.random.default_rng(42)
    >>> times = np.arange(0.0, 40.0, 0.04)
    >>> positions = rng.uniform(10, 90, (len(times), 2))
    >>> objects = np.array([[50.0, 50.0]])
    >>> train = np.sort(rng.uniform(0, 39.9, 100))
    >>> result = compute_object_vector_rate(None, train, times, positions, objects)
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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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


class _ObjectVectorCounts(NamedTuple):
    """Unsmoothed object-vector counts with the geometry and provenance used."""

    spike_counts: NDArray[np.float64]  # (n_units, n_bins), unsmoothed
    occupancy: NDArray[np.float64]  # (n_bins,) seconds
    polar_env: PolarEnvironment  # polar geometry (distance in env units, angle in rad)
    bandwidth: float  # resolved smoothing bandwidth
    analysis_settings: dict[str, Any]


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
    metric: Literal["euclidean", "geodesic"],
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
        Maximum timestamp gap, in seconds, bridged within a tracked run.
    epochs, spike_window
        Resolved time windows.
    method, bandwidth, min_occupancy, backend
        Smoothing settings; ``bandwidth`` may be ``None`` for the default.
    n_jobs : int, default=1
        Parallel jobs for spike counting.
    frame_bins, polar_env : optional
        Trajectory-only geometry reused from an earlier call.

    Returns
    -------
    spike_counts : ndarray of shape (n_units, n_bins)
        Spike counts per polar bin.
    occupancy : ndarray of shape (n_bins,)
        Seconds spent in each polar bin.
    polar_env : PolarEnvironment
        Polar geometry carrying the environment's length units.
    bandwidth : float
        Resolved smoothing bandwidth.
    analysis_settings : dict
        Provenance settings for the result.
    """
    from neurospatial.encoding._egocentric_counts import prepare_egocentric_binning
    from neurospatial.encoding._provenance import rate_analysis_settings
    from neurospatial.encoding._smoothing import resolve_spatial_bandwidth

    # Keep the prepared polar geometry separate from the allocentric env.
    prepared, polar_env = prepare_egocentric_binning(
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
        spike_window=spike_window,
        frame_bins=frame_bins,
        polar_env=polar_env,
    )
    spike_counts, occupancy = prepared.count(spike_trains, n_jobs=n_jobs)

    # Polar graph distances are physical lengths; angles remain radians.
    polar_env = polar_env.with_metadata(units=env.units if env is not None else None)
    bandwidth = (
        resolve_spatial_bandwidth(polar_env, bandwidth)
        if method != "binned"
        else (1.0 if bandwidth is None else bandwidth)
    )
    settings = rate_analysis_settings(
        prepared.support,
        method=method,
        bandwidth=bandwidth if method != "binned" else None,
        min_occupancy=min_occupancy,
        backend=backend,
        direction_frame="allocentric" if headings is None else "egocentric",
        distance_range=[float(value) for value in distance_range],
        n_distance_bins=int(n_distance_bins),
        n_direction_bins=int(n_direction_bins),
        metric=metric,
        objects=object_names,
    )
    return _ObjectVectorCounts(spike_counts, occupancy, polar_env, bandwidth, settings)


def _object_vector_rate(
    env: Environment | None,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64] | None,
    object_positions: ObjectPositions,
    *,
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
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
    validate_spike_times(
        spike_times,
        context=context,
    )

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

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rate is already JAX from smooth_rate_map)
    occupancy_out: ArrayLike = counts.occupancy
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy_out = jnp.asarray(counts.occupancy, dtype=jnp.float64)

    # Return result
    return ObjectVectorRateResult(
        direction_frame="allocentric" if headings is None else "egocentric",
        firing_rate=firing_rate,
        occupancy=occupancy_out,
        env=counts.polar_env,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> ObjectVectorRatesResult:
    """Compute egocentric firing rates for multiple neurons.

    This is the batch version of ``compute_egocentric_rate(None)`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (egocentric coordinates, occupancy) once and optionally
    parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for egocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the egocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the egocentric polar grid. Covers the
        full circle (-pi to pi).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for egocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
    An interval is analyzed only if it passes the gap, speed and bounds
    checks and lies inside ``epochs ∩ spike_window``. The same intervals
    are removed from the spike counts and the occupancy.

    **Efficiency advantages over calling ``compute_egocentric_rate(None)`` in a loop**:

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
    ...     n_jobs=2,  # Parallel spike binning
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
    ...     None, spike_times_2d, times, positions, headings, object_positions
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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
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
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for allocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the allocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the allocentric polar grid. Covers the
        full circle (-pi to pi).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for allocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
    An interval is analyzed only if it passes the gap, speed and bounds
    checks and lies inside ``epochs ∩ spike_window``. The same intervals
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
    ...     None, [train, train], times, positions, objects
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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    context: str,
    frame_bins: NDArray[np.int64] | None = None,
    polar_env: PolarEnvironment | None = None,
) -> ObjectVectorRatesResult:
    """Compute either frame through the shared binning and smoothing path.

    ``frame_bins`` and ``polar_env`` reuse trajectory-only geometry computed by
    a caller that bins many spike sets on one trajectory (the shuffle null).
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

    # Normalize spike times to canonical list-of-arrays format and resolve the
    # per-unit identity labels (defaults to arange). A labelled input keeps its
    # own labels; a differing unit_ids= raises.
    spike_times_list, resolved_unit_ids, _ = as_labelled_spike_trains(
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
    validate_spike_trains(spike_times_list, context=context)

    # Recording coverage uses tracked runs, independently of speed or frame bins.
    _warn_if_population_silent(
        spike_times_list,
        times,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
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

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rates is already JAX from smooth_rate_maps_batch)
    occupancy_out: ArrayLike = counts.occupancy
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy_out = jnp.asarray(counts.occupancy, dtype=jnp.float64)
        firing_rates = jnp.asarray(firing_rates, dtype=jnp.float64)

    # Return result
    return ObjectVectorRatesResult(
        direction_frame="allocentric" if headings is None else "egocentric",
        firing_rates=firing_rates,
        occupancy=occupancy_out,
        env=counts.polar_env,
        distance_range=distance_range,
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        unit_ids=resolved_unit_ids,
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


def object_vector_score(
    tuning_curve: NDArray[np.float64],
    *,
    max_distance_selectivity: float = 10.0,
) -> float:
    """Compute combined object-vector selectivity score.

    The score combines distance selectivity and direction selectivity
    following the formula:

        s_OV = ((s_d - 1) / (s_d* - 1)) * s_theta

    where:
    - s_d = peak / mean (distance selectivity)
    - s_d* = max_distance_selectivity (normalization constant)
    - s_theta = mean resultant length (direction selectivity)

    Parameters
    ----------
    tuning_curve : NDArray[np.float64], shape (n_dist, n_dir)
        2D firing rate tuning curve in egocentric polar coordinates.
    max_distance_selectivity : float, default=10.0
        Maximum expected distance selectivity for normalization.
        Must be > 1.

    Returns
    -------
    float
        Object-vector score in [0, 1]. Higher scores indicate sharper
        tuning to a specific distance and direction.

    Raises
    ------
    ValueError
        If max_distance_selectivity <= 1.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.egocentric import object_vector_score
    >>> # Sharp tuning at one location
    >>> tc = np.zeros((10, 12)) + 0.1
    >>> tc[5, 6] = 20.0
    >>> score = object_vector_score(tc)
    >>> score > 0.5
    True

    See Also
    --------
    is_object_vector_cell : Classify neuron as OVC
    ObjectVectorRateResult.is_object_vector_cell : Classifier method on result object
    """
    if max_distance_selectivity <= 1.0:
        raise ValueError(
            f"max_distance_selectivity must be > 1, got {max_distance_selectivity}"
        )

    tuning_curve = np.asarray(tuning_curve, dtype=np.float64)

    # Handle NaN values
    valid_mask = np.isfinite(tuning_curve)
    if not np.any(valid_mask):
        return float(np.nan)

    valid_rates = tuning_curve[valid_mask]

    # Compute distance selectivity
    peak_rate = float(np.max(valid_rates))
    mean_rate = float(np.mean(valid_rates))

    if mean_rate == 0:
        return 0.0

    distance_selectivity = peak_rate / mean_rate

    # Normalize distance selectivity to [0, 1]
    normalized_dist_sel = (distance_selectivity - 1.0) / (
        max_distance_selectivity - 1.0
    )
    normalized_dist_sel = float(np.clip(normalized_dist_sel, 0.0, 1.0))

    # Compute direction selectivity (mean resultant length)
    n_dir = tuning_curve.shape[1]
    direction_bins = np.linspace(-np.pi, np.pi, n_dir + 1)
    dir_bin_centers = (direction_bins[:-1] + direction_bins[1:]) / 2

    # Marginalize over distance
    direction_tuning = np.nansum(tuning_curve, axis=0)
    total = np.sum(direction_tuning)

    if total == 0:
        direction_selectivity = 0.0
    else:
        direction_selectivity = _mean_resultant_length(
            dir_bin_centers, weights=direction_tuning
        )

    # Combined score
    score = normalized_dist_sel * direction_selectivity

    return float(np.clip(score, 0.0, 1.0))


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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> bool:
    """Classify one neuron by the object_vector_cell screen or circular-shift test.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for allocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the allocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the allocentric polar grid. Covers the
        full circle (-π to π).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for allocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
    The 0.3 bits/spike default is this library's screening heuristic.
    Plug-in information is biased upward by approximately
    (n_bins - 1) / (2 ln(2) N_spikes). In 20 untuned 0.5 Hz Poisson units,
    egocentric 10 x 12 polar maps had median information 2.12, 1.41, 0.70,
    0.41 and 0.21 bits/spike at 1, 2, 5, 10 and 20 minutes (about 30, 60,
    150, 300 and 600 spikes). The screen flagged 20/20 at 1-10 minutes
    and 0/20 at 20 minutes. The allocentric screen also flagged 20/20
    in the seeded 10-minute fixture. Low counts can resemble tuning.
    For publication, report a circular-shift test and its assumptions.

    Circular shifting costs about n_shuffles recomputes of the plural map.
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
    >>> result = is_object_vector_cell(None, spikes, times, positions, objects)
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
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
            metric=metric,
            max_gap=max_gap,
            epochs=epochs,
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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
) -> bool:
    """Classify one neuron by the egocentric_object_vector_cell screen or circular-shift test.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for egocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the egocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the egocentric polar grid. Covers the
        full circle (-π to π).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for egocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.

    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
    The 0.3 bits/spike default is this library's screening heuristic.
    Plug-in information is biased upward by approximately
    (n_bins - 1) / (2 ln(2) N_spikes). In 20 untuned 0.5 Hz Poisson units,
    egocentric 10 x 12 polar maps had median information 2.12, 1.41, 0.70,
    0.41 and 0.21 bits/spike at 1, 2, 5, 10 and 20 minutes (about 30, 60,
    150, 300 and 600 spikes). The screen flagged 20/20 at 1-10 minutes
    and 0/20 at 20 minutes. The allocentric screen also flagged 20/20
    in the seeded 10-minute fixture. Low counts can resemble tuning.
    For publication, report a circular-shift test and its assumptions.

    Circular shifting costs about n_shuffles recomputes of the plural map.
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
    ...     None, spikes, times, positions, headings, objects
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
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
            metric=metric,
            max_gap=max_gap,
            epochs=epochs,
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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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
    distance_range: tuple[float, float],
    n_distance_bins: int,
    n_direction_bins: int,
    metric: Literal["euclidean", "geodesic"],
    max_gap: float | None,
    epochs: Any,
    spike_window: Any,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float | None,
    min_occupancy: float,
    n_jobs: int,
    backend: Literal["numpy", "jax", "auto"],
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
        _compute_object_coords,
        _coords_to_flat_bin_idx,
        _create_egocentric_environment,
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
        "n_distance_bins": n_distance_bins,
        "n_direction_bins": n_direction_bins,
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
    validate_trajectory(times, positions=positions, headings=headings, context=context)
    for train in trains:
        validate_spike_times(train, context=context)
    _validate_object_vector_metric(env, metric, context=context)
    object_positions, _ = resolve_object_positions(
        object_positions, env=env, positions=positions, metric=metric
    )
    distances, bearings = _compute_object_coords(
        positions, headings, object_positions, metric=metric, env=env
    )
    frame_bins = _coords_to_flat_bin_idx(
        distances.ravel(),
        bearings.ravel(),
        distance_range,
        n_distance_bins,
        n_direction_bins,
    )
    mask = interval_valid_mask(
        times,
        start_bin=frame_bins,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
    )
    windows = run_time_bounds(times, mask)

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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> dict[Hashable, ShuffleTestResult]:
    """Test object vector tuning against circular spike-time shifts.

    Recompute the population map for the supplied arrays and for each shifted
    train, using exactly the observed map's valid intervals. Shuffles preserve
    counts on the compressed analyzed clock and never enter recording gaps.
    This costs about n_shuffles population-map recomputes. The result's
    is_significant property uses 0.05; compare p_value < alpha for another level.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for allocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the allocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the allocentric polar grid. Covers the
        full circle (-pi to pi).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for allocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.
    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
        For invalid inputs/settings, duplicate/conflicting unit labels,
        insufficient analyzed time, or unsupported method="glm".

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import object_vector_cell_significance
    >>> rng = np.random.default_rng(7)
    >>> times = np.arange(0.0, 60.0, 0.1)
    >>> xx, yy = np.meshgrid(np.arange(0, 101, 5), np.arange(0, 101, 5))
    >>> env = Environment.from_samples(np.c_[xx.ravel(), yy.ravel()], bin_size=5.0)
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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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
    distance_range: tuple[float, float] = (0.0, 50.0),
    n_distance_bins: int = 10,
    n_direction_bins: int = 12,
    metric: Literal["euclidean", "geodesic"] = "euclidean",
    max_gap: float | None = 0.5,
    epochs: Any = None,
    spike_window: Any = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "binned",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    n_jobs: int = 1,
    backend: Literal["numpy", "jax", "auto"] = "numpy",
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> dict[Hashable, ShuffleTestResult]:
    """Test egocentric object vector tuning against circular spike-time shifts.

    Recompute the population map for the supplied arrays and for each shifted
    train, using exactly the observed map's valid intervals. Shuffles preserve
    counts on the compressed analyzed clock and never enter recording gaps.
    This costs about n_shuffles population-map recomputes. The result's
    is_significant property uses 0.05; compare p_value < alpha for another level.

    Parameters
    ----------
    env : Environment or None
        The allocentric environment. Required when
        ``metric="geodesic"`` (used to compute distances around
        obstacles). May be ``None`` when ``metric="euclidean"``.
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
    distance_range : tuple of float, default=(0.0, 50.0)
        (min_distance, max_distance) for egocentric binning. Distances outside
        this range are not included in the rate map.
    n_distance_bins : int, default=10
        Number of distance bins in the egocentric polar grid.
    n_direction_bins : int, default=12
        Number of direction bins in the egocentric polar grid. Covers the
        full circle (-pi to pi).
    metric : {"euclidean", "geodesic"}, default="euclidean"
        Distance metric for computing distance to objects:

        - **euclidean**: Straight-line distance.
        - **geodesic**: Path distance respecting environment boundaries.
          Requires ``env`` parameter.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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

        - **binned** (default): Raw rate computation without smoothing.
          Appropriate for egocentric polar grids where boundary-aware
          smoothing may not apply.
        - **diffusion_kde**: Graph-based boundary-aware KDE.
        - **gaussian_kde**: Standard Euclidean KDE.
    bandwidth : float or None, default=None
        Physical smoothing length in environment units. None uses 5 cm
        converted to declared metric units, or 2.5 median graph-edge lengths
        for unspecified/custom units. The polar environment inherits ``env.units``
        when an environment is supplied. Unused for ``method="binned"``.
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
        For invalid inputs/settings, duplicate/conflicting unit labels,
        insufficient analyzed time, or unsupported method="glm".

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import egocentric_object_vector_cell_significance
    >>> rng = np.random.default_rng(7)
    >>> times = np.arange(0.0, 60.0, 0.1)
    >>> xx, yy = np.meshgrid(np.arange(0, 101, 5), np.arange(0, 101, 5))
    >>> env = Environment.from_samples(np.c_[xx.ravel(), yy.ravel()], bin_size=5.0)
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
        n_distance_bins=n_distance_bins,
        n_direction_bins=n_direction_bins,
        metric=metric,
        max_gap=max_gap,
        epochs=epochs,
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
