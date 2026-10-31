"""Rank-order correlation between place-field order and firing order in an event.

A replay event activates place cells in the order of their fields on the
track, forward or reversed. The rank-order correlation (Foster & Wilson 2006;
Diba & Buzsáki 2007) is the Spearman correlation between each active cell's
field position and when it fired in the event. It needs no decoding, only
spike times and each cell's place field.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from neurospatial._exceptions import _format_error
from neurospatial._pynapple import interval_rows, is_pynapple_type
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._validation import validate_integer_at_least
from neurospatial.stats import _ALTERNATIVES

if TYPE_CHECKING:
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment
    from neurospatial.stats.shuffle import ShuffleTestResult

_SPIKE_STATISTICS = ("first", "median", "mean", "all_spikes")
_FIELD_ORDERS = ("peak", "center")
_MULTI_FIELD_CELLS = ("keep", "exclude")
# When a cell fired in an event, for each spike_statistic but "all_spikes".
_SPIKE_SUMMARIES: dict[str, Callable[..., Any]] = {
    "first": np.min,
    "median": np.median,
    "mean": np.mean,
}
# Largest number of (shuffle, pair) values ranked at once.
_BLOCK_VALUES = 2_000_000


@dataclass(frozen=True, repr=False, eq=False)
class RankOrderResult(ReadOnlyArraysMixin, ResultMixin):
    """Rank-order correlation of one candidate event.

    Attributes
    ----------
    score : float
        Spearman correlation between field position and firing time, from -1
        to 1: positive when cells fire in the order of their fields along the
        track (toward larger positions), negative for the reverse order. NaN
        when fewer than ``min_active_cells`` cells fired, or when the active
        cells' field positions or firing times are all equal.
    analytic_p_value : float
        Two-sided p-value of ``score`` assuming independent pairs: exact (all
        orderings enumerated) for up to 9 pairs, and the t-approximation
        above that. Foster & Wilson (2006) used the t-approximation for
        every number of pairs (``FOSTER_WILSON_2006`` uses
        ``test="t_distribution"``), so for up to 9 pairs this differs from
        theirs. MATLAB's ``corr``, used in the Bendor-lab analyses, also
        uses the exact distribution for small samples.
        ``spike_statistic="all_spikes"`` breaks the independence (a cell's
        spikes share one field position): Takigawa et al. (2024) found
        about 17% false positives with every spike when spikes were
        permuted one by one, which, like this p-value, treats them as
        independent. Use
        :func:`rank_order_significance` instead.
    n_active_cells : int
        Cells with a place field that fired at least once in the event.
    active_units : NDArray[np.intp], shape (n_active_cells,)
        Index of each active cell in ``spike_times``.
    spike_statistic : str
        When each cell fired: ``"first"``, ``"median"`` or ``"mean"`` spike
        time, or ``"all_spikes"``.
    field_order : str
        How fields were placed on the track: ``"peak"`` or ``"center"``.
    multi_field_cells : str
        ``"keep"`` or ``"exclude"``: whether cells with more than one place
        field were left out of the template.
    excluded_units : NDArray[np.intp], shape (n_excluded_cells,)
        Index in ``spike_times`` of each cell left out for having more than
        one place field, whether or not it fired in the event; empty with
        ``multi_field_cells="keep"``.

    See Also
    --------
    rank_order_correlation : Function that produces this result.
    """

    score: float
    analytic_p_value: float
    n_active_cells: int
    active_units: NDArray[np.intp]
    spike_statistic: str
    field_order: str
    multi_field_cells: str
    excluded_units: NDArray[np.intp]

    def __post_init__(self) -> None:
        """Own read-only copies of ``active_units`` and ``excluded_units``."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "score": self.score,
            "analytic_p_value": self.analytic_p_value,
            "n_active_cells": self.n_active_cells,
            "spike_statistic": self.spike_statistic,
            "field_order": self.field_order,
            "multi_field_cells": self.multi_field_cells,
            "n_excluded_cells": int(self.excluded_units.size),
        }


@dataclass(frozen=True, repr=False, eq=False)
class RankOrderSignificanceResult(ResultMixin):
    """Rank-order correlation of one event tested against cell-order shuffles.

    Attributes
    ----------
    observed : RankOrderResult
        The event's rank-order correlation.
    shuffle : ShuffleTestResult
        The observed correlation, the correlation of every shuffle, the Monte
        Carlo p-value ``(k + 1) / (n + 1)`` and the z-score.
    p_value : float
        The shuffle test's p-value.
    alternative : {"greater", "less", "two-sided"}
        The side of the null distribution tested.

    See Also
    --------
    rank_order_significance : Function that produces this result.
    """

    observed: RankOrderResult
    shuffle: ShuffleTestResult
    alternative: str

    @property
    def p_value(self) -> float:
        """The shuffle test's p-value."""
        return self.shuffle.p_value

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            **self.observed.summary(),
            "p_value": self.p_value,
            "z_score": self.shuffle.z_score,
        }


def rank_order_correlation(
    env: Environment,
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    spike_statistic: Literal["first", "median", "mean", "all_spikes"] = "median",
    field_order: Literal["peak", "center"] = "peak",
    min_active_cells: int = 5,
    multi_field_cells: Literal["keep", "exclude"] = "keep",
    place_fields: Sequence[Any] | None = None,
) -> RankOrderResult:
    """Rank-order correlation between field order and firing order in an event.

    Parameters
    ----------
    env : Environment
        The environment the rate maps are on: a 1-D environment or a
        linearized track. Field positions are measured along the track.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the same order as the
        rows of ``encoding_models``. Any input
        :func:`~neurospatial.decoding.bin_spikes_in_time` accepts.
    interval : (start, stop) or one-row IntervalSet
        The candidate event, in seconds. Spikes at ``start <= t < stop``
        count.
    encoding_models : array of shape (n_cells, n_bins), or SpatialRatesResult
        Each cell's rate map on ``env``, in Hz. NaN marks bins without
        training data, which are ignored. A cell with no finite positive rate
        has no field and is left out.
    spike_statistic : {"median", "first", "mean", "all_spikes"}, default="median"
        When each cell fired in the event:

        - ``"median"``: median spike time (Tirole et al. 2022). Takigawa et
          al. (2024) found it held the false-positive rate near 5%.
        - ``"first"``: first spike (Diba & Buzsáki 2007).
        - ``"mean"``: mean spike time (Chenani et al. 2019).
        - ``"all_spikes"``: every spike, each paired with its cell's field
          position (Silva et al. 2015). Takigawa et al. found a
          false-positive rate near 17%
          with it when spikes were permuted one by one, which, like the
          analytic p-value, treats each spike as independent; the
          cell-order permutation test of :func:`rank_order_significance`
          keeps a cell's spikes together and holds its false-positive rate.
    field_order : {"peak", "center"}, default="peak"
        Each cell's field position: the track position of its peak rate
        (most studies) or the rate-weighted center of its rate map. Chenani
        et al. (2019) order cells by "place field centers" without defining
        them; the rate-weighted center is this function's reading. A cell
        with several fields is placed between them by ``"center"``, and on
        a branching track the center can fall between arms; use ``"peak"``
        for such cells, or leave them out with
        ``multi_field_cells="exclude"``.
    min_active_cells : int, default=5
        Fewest active cells for a correlation (Chenani et al. 2019; Diba &
        Buzsáki 2007 required 5 cells or 30% of place cells, whichever was
        greater, when detecting events). With fewer, ``score`` is NaN.
    multi_field_cells : {"keep", "exclude"}, default="keep"
        Cells with more than one place field have no single place in the
        order (Tingley & Peyrache 2020: "Typically, such neurons are
        excluded"; their method comparison used only single-field cells).

        - ``"keep"``: every cell takes its position by ``field_order``.
        - ``"exclude"``: each cell with more than one field is left out of
          the template, as if it had no field. Fields are found by
          :func:`~neurospatial.encoding.detect_place_fields` with its default
          settings and the rate result's ``bandwidth``, or taken from
          ``place_fields``. The excluded cells are listed in the result's
          ``excluded_units``. A cell with no detected field is kept and
          placed by ``field_order``.
    place_fields : sequence, optional
        Each cell's place fields, one entry per row of ``encoding_models``,
        such as ``[detect_place_fields(env, rate, bandwidth=5.0) for rate in
        rates]``; a cell with more than one entry is excluded. Only with
        ``multi_field_cells="exclude"``, where it is required for rate maps
        given as an array, since the smoothing bandwidth that sets the
        smallest field is not known. With a rate result, pass it to choose
        other detection settings.

    Returns
    -------
    RankOrderResult
        The correlation, its analytic p-value, the active cells and the
        cells excluded for having several fields.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if
        ``encoding_models`` does not have one row per cell and one column per
        bin, has negative rates, or is a rate result computed on another
        environment, if ``interval`` is not a finite ``(start, stop)`` with
        ``start < stop``, if ``spike_statistic``, ``field_order`` or
        ``multi_field_cells`` is not one of the options, if
        ``min_active_cells`` is less than 3, or if ``place_fields`` is given
        without ``multi_field_cells="exclude"``, does not have one entry
        per cell, or is missing for rate maps given as an array.

    See Also
    --------
    rank_order_significance : Test the correlation against shuffles.

    Notes
    -----
    The correlation is Spearman's rho with average ranks for ties: the
    Pearson correlation of the ranks of the field positions and the ranks of
    the firing times. Two cells whose fields peak in the same bin tie.

    Tingley & Peyrache (2020) showed that random sequences of 10 cells reach
    "significant" rank-order correlations with a random template 5-12% of the
    time in sets of 100 events (up to 20% at five cells), so judge the
    proportion of significant events against chance rather than single
    events. Those rates come from the parametric p-value of MATLAB's
    ``corr`` (Pearson's r of the orders with the t-distribution) and the
    binomial spread of the count across sets of 100 events
    (``rankOrder_falsePositive_tests.m``): per event, p < 0.05 for about
    5.4% of random orders of 10 cells, 8.3% of 4 or 5 cells and 33% of 3
    cells, while a permutation test, such as the Monte Carlo cell-order
    shuffles of :func:`rank_order_significance` or the exact
    ``analytic_p_value`` for up to 9 pairs, stays at or below 5% for every
    number of cells. The
    ``replay_RankOrder.m`` function in their RnR_methods toolbox correlates
    sort indices, not ranks; this function computes Spearman's rho.

    References
    ----------
    Foster, D. J., & Wilson, M. A. (2006). Reverse replay of behavioural
    sequences in hippocampal place cells during the awake state. *Nature*,
    440(7084), 680-683.

    Diba, K., & Buzsáki, G. (2007). Forward and reverse hippocampal
    place-cell sequences during ripples. *Nature Neuroscience*, 10(10),
    1241-1242.

    Takigawa, M., Huelin Gorriz, M., Tirole, M., & Bendor, D. (2024).
    Evaluating hippocampal replay without a ground truth. *eLife*, 13,
    e85635.

    Examples
    --------
    Six cells with fields along a 120 cm track fire in field order during a
    100 ms event:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import rank_order_correlation
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> fields = np.linspace(10.0, 110.0, 6)
    >>> rates = 20.0 * np.exp(-0.5 * ((x - fields[:, None]) / 8.0) ** 2)
    >>> spike_times = [np.array([1.0 + 0.015 * i, 1.004 + 0.015 * i]) for i in range(6)]
    >>> result = rank_order_correlation(env, spike_times, (1.0, 1.1), rates)
    >>> result.score, result.n_active_cells
    (1.0, 6)
    >>> rank_order_correlation(env, spike_times[::-1], (1.0, 1.1), rates).score
    -1.0

    Cell 2 also fires at the end of the track, so it has two fields. Leave
    it out of the order:

    >>> from neurospatial.encoding import detect_place_fields
    >>> rates[2] += 20.0 * np.exp(-0.5 * ((x - 115.0) / 8.0) ** 2)
    >>> fields = [detect_place_fields(env, rate, bandwidth=0.0) for rate in rates]
    >>> [len(cell_fields) for cell_fields in fields]
    [1, 1, 2, 1, 1, 1]
    >>> result = rank_order_correlation(
    ...     env,
    ...     spike_times,
    ...     (1.0, 1.1),
    ...     rates,
    ...     multi_field_cells="exclude",
    ...     place_fields=fields,
    ... )
    >>> result.excluded_units, result.n_active_cells
    (array([2]), 5)
    """
    pairs = _event_pairs(
        env,
        spike_times,
        interval,
        encoding_models,
        spike_statistic=spike_statistic,
        field_order=field_order,
        min_active_cells=min_active_cells,
        operation="rank_order_correlation()",
        multi_field_cells=multi_field_cells,
        place_fields=place_fields,
    )
    return _result(pairs, spike_statistic, field_order)


def rank_order_significance(
    env: Environment,
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    spike_statistic: Literal["first", "median", "mean", "all_spikes"] = "median",
    field_order: Literal["peak", "center"] = "peak",
    min_active_cells: int = 5,
    multi_field_cells: Literal["keep", "exclude"] = "keep",
    place_fields: Sequence[Any] | None = None,
    alternative: Literal["greater", "less", "two-sided"] = "two-sided",
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> RankOrderSignificanceResult:
    """Test an event's rank-order correlation against cell-order shuffles.

    Each shuffle permutes the field positions among the event's active cells
    (cell-order permutation), which keeps when each cell fired and which
    field positions occur but breaks their pairing (Diba & Buzsáki 2007,
    500 shuffles; Silva et al. 2015, 5000).

    Parameters
    ----------
    env, spike_times, interval, encoding_models
        As for :func:`rank_order_correlation`.
    spike_statistic, field_order, min_active_cells
        As for :func:`rank_order_correlation`.
    multi_field_cells, place_fields
        As for :func:`rank_order_correlation`. Excluded cells are left out
        of the shuffles too.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        ``"two-sided"`` counts forward and reverse sequences alike (Diba &
        Buzsáki 2007): twice the smaller one-sided p-value, capped at 1.
        ``"greater"`` tests forward order only, ``"less"`` reverse only.
    n_shuffles : int, default=1000
        Number of cell-order permutations. A p-value is at least
        ``1 / (n_shuffles + 1)``, or ``2 / (n_shuffles + 1)`` with
        ``alternative="two-sided"``.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed. Pass a seed for a reproducible
        result.

    Returns
    -------
    RankOrderSignificanceResult
        The event's correlation and its permutation test.

    Raises
    ------
    ValueError
        For inputs :func:`rank_order_correlation` refuses; if the event's
        correlation is NaN (too few active cells, or no variation); if
        ``alternative`` is not one of the options; or if ``n_shuffles`` is
        not a positive integer.

    See Also
    --------
    rank_order_correlation : The correlation alone.

    Notes
    -----
    The p-value is ``(k + 1) / (n + 1)``, where ``k`` counts permutations at
    least as extreme as the event (Phipson & Smyth 2010). It holds its
    false-positive rate for every ``spike_statistic``, because a cell's
    spikes move together. Foster & Wilson (2006) used an analytic Spearman
    p-value per event instead, from the t-distribution for every number of
    pairs (``FOSTER_WILSON_2006``, ``test="t_distribution"``).
    :class:`RankOrderResult` reports a similar ``analytic_p_value``, which is
    exact for up to 9 pairs. The z-score of the shuffle test approximates the
    spatial similarity index of Chenani et al. (2019), who z-scored the
    correlation against 100,000 random permutations of the same number of
    cells (``testsession.m`` subtracts their mean and divides by their
    standard deviation).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import rank_order_significance
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> fields = np.linspace(10.0, 110.0, 8)
    >>> rates = 20.0 * np.exp(-0.5 * ((x - fields[:, None]) / 8.0) ** 2)
    >>> spike_times = [np.array([1.0 + 0.01 * i]) for i in range(8)]
    >>> result = rank_order_significance(
    ...     env, spike_times, (1.0, 1.1), rates, n_shuffles=999, rng=0
    ... )
    >>> print(f"rho = {result.observed.score:.2f}, p = {result.p_value:.3f}")
    rho = 1.00, p = 0.002
    """

    operation = "rank_order_significance()"
    validate_integer_at_least(
        n_shuffles,
        1,
        name="n_shuffles",
        owner=operation,
        hint="Diba & Buzsáki used 500 permutations per event, Silva et al. 5000",
    )
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"{operation}: alternative must be one of {_ALTERNATIVES}, got "
            f"{alternative!r}."
        )
    pairs = _event_pairs(
        env,
        spike_times,
        interval,
        encoding_models,
        spike_statistic=spike_statistic,
        field_order=field_order,
        min_active_cells=min_active_cells,
        operation=operation,
        multi_field_cells=multi_field_cells,
        place_fields=place_fields,
    )
    observed = _result(pairs, spike_statistic, field_order)
    if not np.isfinite(observed.score):
        raise ValueError(
            _format_error(
                f"{operation}: the event's rank-order correlation is undefined "
                f"({observed.n_active_cells} active cells; at least "
                f"{min_active_cells} needed, with differing field positions and "
                "firing times).",
                why="a correlation that is not a number cannot be compared with "
                "the shuffles",
                fix="skip this event, or lower min_active_cells (at least 3)",
            )
        )
    test = _cell_order_test(
        pairs, observed.score, n_shuffles, alternative, np.random.default_rng(rng)
    )
    return RankOrderSignificanceResult(
        observed=observed,
        shuffle=test,
        alternative=alternative,
    )


@dataclass(frozen=True)
class _EventPairs:
    """The (field position, firing time) pairs of one event.

    ``cell`` maps each pair to its position among the active cells, so a
    permutation of the active cells' field positions permutes the pairs.
    """

    field_positions: NDArray[np.float64]  # (n_active,)
    times: NDArray[np.float64]  # (n_pairs,)
    cell: NDArray[np.intp]  # (n_pairs,)
    active_units: NDArray[np.intp]  # (n_active,)
    multi_field_cells: str
    excluded_units: NDArray[np.intp]  # (n_excluded,)


def _event_pairs(
    env: Environment,
    spike_times: Any,
    interval: Any,
    encoding_models: Any,
    *,
    spike_statistic: str,
    field_order: str,
    min_active_cells: int,
    operation: str,
    multi_field_cells: str = "keep",
    place_fields: Sequence[Any] | None = None,
) -> _EventPairs:
    from neurospatial._spikes import as_spike_trains

    if spike_statistic not in _SPIKE_STATISTICS:
        raise ValueError(
            f"{operation}: spike_statistic must be one of {_SPIKE_STATISTICS}, "
            f"got {spike_statistic!r}."
        )
    if field_order not in _FIELD_ORDERS:
        raise ValueError(
            f"{operation}: field_order must be one of {_FIELD_ORDERS}, got "
            f"{field_order!r}."
        )
    validate_integer_at_least(
        min_active_cells,
        3,
        name="min_active_cells",
        owner=operation,
        hint="a rank correlation of two cells is always +1 or -1",
    )
    start, stop = _validated_interval(interval, operation)
    positions = _field_positions(env, encoding_models, field_order, operation)
    excluded = _multi_field_units(
        env, encoding_models, multi_field_cells, place_fields, operation
    )
    positions[excluded] = np.nan
    trains = as_spike_trains(spike_times)
    _require_same_cells(len(trains), len(positions), operation)
    in_event = _spikes_in(trains, start, stop)
    active = np.array(
        [
            unit
            for unit, spikes in enumerate(in_event)
            if spikes.size and np.isfinite(positions[unit])
        ],
        dtype=np.intp,
    )
    times, cell = _pooled_spikes(in_event, active, spike_statistic)
    if active.size < min_active_cells:
        times, cell = np.empty(0), np.empty(0, dtype=np.intp)
    return _EventPairs(
        field_positions=positions[active],
        times=times,
        cell=cell,
        active_units=active,
        multi_field_cells=multi_field_cells,
        excluded_units=excluded,
    )


def _result(
    pairs: _EventPairs,
    spike_statistic: str,
    field_order: str,
    *,
    analytic_p_value: bool = True,
) -> RankOrderResult:
    """The event's correlation; ``analytic_p_value=False`` leaves the p-value NaN.

    The exact p-value enumerates every ordering of up to 9 pairs, so callers
    that test the score otherwise skip it.
    """
    score, p_value = np.nan, np.nan
    if pairs.times.size:
        x = pairs.field_positions[pairs.cell]
        # The shuffles' formula: ranks are half-integers, so the sums are exact
        # and a shuffle with the event's ranks gives the same correlation.
        score = float(_spearman_rows(x[np.newaxis], pairs.times)[0])
        if analytic_p_value and np.isfinite(score):
            p_value = _spearman_p_value(x, pairs.times, score)
    return RankOrderResult(
        score=score,
        analytic_p_value=p_value,
        n_active_cells=int(pairs.active_units.size),
        active_units=pairs.active_units,
        spike_statistic=spike_statistic,
        field_order=field_order,
        multi_field_cells=pairs.multi_field_cells,
        excluded_units=pairs.excluded_units,
    )


def _multi_field_units(
    env: Environment,
    encoding_models: Any,
    multi_field_cells: str,
    place_fields: Sequence[Any] | None,
    operation: str,
) -> NDArray[np.intp]:
    """Index of each cell to exclude for having more than one place field.

    Run after :func:`_field_positions` has validated ``encoding_models``.
    """
    if multi_field_cells not in _MULTI_FIELD_CELLS:
        raise ValueError(
            f"{operation}: multi_field_cells must be one of {_MULTI_FIELD_CELLS}, "
            f"got {multi_field_cells!r}."
        )
    if multi_field_cells == "keep":
        if place_fields is not None:
            raise ValueError(
                _format_error(
                    f"{operation}: place_fields was given with "
                    "multi_field_cells='keep'.",
                    why="place_fields only decides which cells "
                    "multi_field_cells='exclude' leaves out",
                    fix="pass multi_field_cells='exclude' to leave out cells with "
                    "more than one field, or leave out place_fields",
                )
            )
        return np.empty(0, dtype=np.intp)
    rates = _supported_rates(encoding_models)
    if place_fields is None:
        if not hasattr(encoding_models, "firing_rates"):
            raise ValueError(
                _format_error(
                    f"{operation}: multi_field_cells='exclude' needs place_fields "
                    "for rate maps given as an array.",
                    why="the smallest place field depends on the bandwidth the "
                    "maps were smoothed with, which an array does not carry",
                    fix="pass place_fields=[detect_place_fields(env, rate, "
                    "bandwidth=...) for rate in rates], or pass the "
                    "SpatialRatesResult itself",
                )
            )
        from neurospatial.encoding.spatial import detect_place_fields

        bandwidth = getattr(encoding_models, "bandwidth", None)
        place_fields = [
            detect_place_fields(
                env, rate, bandwidth=0.0 if bandwidth is None else bandwidth
            )
            for rate in rates
        ]
    if len(place_fields) != len(rates):
        raise ValueError(
            _format_error(
                f"{operation}: place_fields has {len(place_fields)} entries but "
                f"encoding_models has {len(rates)} rate maps.",
                why="each cell's fields decide whether that cell is excluded",
                fix="pass one entry per cell, in the order of the rate maps",
            )
        )
    n_fields = np.array([len(cell_fields) for cell_fields in place_fields])
    return np.flatnonzero(n_fields > 1).astype(np.intp)


_EXACT_MAX_PAIRS = 9


def _spearman_p_value(
    x: NDArray[np.float64], times: NDArray[np.float64], score: float
) -> float:
    """Two-sided Spearman p-value: exact for few pairs, else the t-approximation."""
    import math

    from neurospatial.replay.sequence_match import _all_permutations

    n = x.size
    if n > _EXACT_MAX_PAIRS:
        # t = rho sqrt((n - 2) / (1 - rho^2)) on n - 2 degrees of freedom, as
        # scipy.stats.spearmanr. It reaches 0 for |rho| = 1; no ordering of n
        # distinct values is rarer than 2 / n!, computed in log space because
        # n! overflows a float from n = 171 (the bound then underflows to 0).
        with np.errstate(divide="ignore"):
            t = abs(score) * np.sqrt(
                (n - 2) / np.float64((1.0 - score) * (1.0 + score))
            )
        approximate = float(2.0 * stats.t.sf(t, n - 2))
        return max(approximate, 2.0 * math.exp(-math.lgamma(n + 1)))
    orders = _all_permutations(n)
    # Blocks of orderings bound the memory of the rank arrays (n = 9 has 362,880).
    block = _BLOCK_VALUES // (8 * n)
    rho = np.concatenate(
        [
            _spearman_rows(x[orders[start : start + block]], times)
            for start in range(0, len(orders), block)
        ]
    )
    return float(np.mean(np.abs(rho) >= abs(score)))


def _spearman_rows(
    x: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Spearman correlation of each row of ``x``, shape (n_rows, n), with ``values``.

    Average ranks for ties; NaN where a row, or ``values``, does not vary.
    """
    v_ranks = stats.rankdata(values)
    v_dev = v_ranks - v_ranks.mean()
    x_ranks = stats.rankdata(x, axis=1)
    x_dev = x_ranks - x_ranks.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (x_dev @ v_dev) / np.sqrt((x_dev**2).sum(axis=1) * (v_dev**2).sum())
    varies = (np.ptp(x, axis=1) > 0) & (np.ptp(values) > 0)
    return np.where(varies, rho, np.nan)


def _cell_order_test(
    pairs: _EventPairs,
    score: float,
    n_shuffles: int,
    alternative: Literal["greater", "less", "two-sided"],
    rng: np.random.Generator,
) -> ShuffleTestResult:
    """Test an event's correlation against permutations of its fields among cells."""
    from neurospatial.stats.shuffle import ShuffleTestResult

    n_active = pairs.active_units.size
    permutations = np.argsort(rng.random((n_shuffles, n_active)), axis=1)
    return ShuffleTestResult.from_null(
        score,
        _permuted_correlations(pairs, permutations),
        shuffle_type="cell_order",
        alternative=alternative,
    )


def _permuted_correlations(
    pairs: _EventPairs, permutations: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Spearman correlation under each permutation of the active cells' fields."""
    correlations = np.empty(len(permutations))
    # Rank the pairs in blocks of shuffles to bound memory with many spikes.
    block = max(1, _BLOCK_VALUES // max(pairs.cell.size, 1))
    for start in range(0, len(permutations), block):
        chunk = permutations[start : start + block]
        correlations[start : start + block] = _spearman_rows(
            pairs.field_positions[chunk][:, pairs.cell], pairs.times
        )
    return correlations


def _validated_interval(interval: Any, operation: str) -> tuple[float, float]:
    """One event's half-open ``(start, stop)`` in seconds.

    A one-row pynapple ``IntervalSet`` is read with its inclusive end moved to
    the next float, so a spike at ``end`` stays in the event.
    """
    if is_pynapple_type(interval, "IntervalSet"):
        rows = interval_rows(interval)
        if len(rows) != 1:
            raise ValueError(
                _format_error(
                    f"{operation}: interval has {len(rows)} intervals, but it "
                    "is one event.",
                    why="each call scores the spikes of one candidate event",
                    fix="pass one row, e.g. intervals[k], and call once per event",
                )
            )
        interval = rows[0]
    values = np.asarray(interval, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
        raise ValueError(
            _format_error(
                f"{operation}: interval must be a finite (start, stop) in seconds "
                f"with start < stop, got {interval!r}.",
                why="spikes at start <= t < stop belong to the event",
                fix="pass the candidate event's start and stop times, e.g. "
                "(event_start, event_stop)",
            )
        )
    return float(values[0]), float(values[1])


def _template_fields(
    env: Environment,
    encoding_models: Any,
    field_order: str,
    min_peak_rate: float | None,
    operation: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The template cells' rate maps and field positions, checked against env.

    Returns the supported rate maps, shape (n_cells, n_bins), and each
    cell's field position, shape (n_cells,). A cell whose highest rate is
    below ``min_peak_rate`` is not a template cell: its rate map and field
    position are NaN, as for a cell without a field.
    """
    positions = _field_positions(env, encoding_models, field_order, operation)
    rates = _supported_rates(encoding_models)
    if min_peak_rate is not None:
        keep = np.where(np.isfinite(rates), rates, 0.0).max(axis=1) >= min_peak_rate
        rates = np.where(keep[:, None], rates, np.nan)
        positions = np.where(keep, positions, np.nan)
    return rates, positions


def _template_items(
    env: Environment | Mapping[str, Environment],
    encoding_models: Any,
    template: str | None,
    operation: str,
    *,
    single_template_method: str | None = None,
) -> list[tuple[str | None, Environment, Any]]:
    """Each tested template's name, environment and rate maps.

    ``encoding_models`` is one set of rate maps, named None, or a mapping of
    template name to rate maps, of which ``template`` picks one and None
    tests every one. ``env`` is one environment or a mapping with each tested
    template's environment. ``single_template_method`` names a method that
    tests one template and has no rule to choose one, so it needs
    ``template=`` with a mapping.
    """
    if not isinstance(encoding_models, Mapping):
        if template is not None:
            raise ValueError(
                _format_error(
                    f"{operation}: template={template!r} needs a mapping of rate maps.",
                    why="a template names one of several rate-map sets",
                    fix="pass encoding_models={'left': rates_left, 'right': "
                    "rates_right}, or leave out template=",
                )
            )
        if isinstance(env, Mapping):
            raise ValueError(
                _format_error(
                    f"{operation}: a mapping of environments needs a mapping of "
                    "rate maps with the same templates.",
                    why="each template's fields are read on its own environment",
                    fix="pass encoding_models as a mapping keyed like env, or one "
                    "environment",
                )
            )
        return [(None, env, encoding_models)]
    names = list(encoding_models)
    if not names:
        raise ValueError(
            _format_error(
                f"{operation}: encoding_models is an empty mapping.",
                why="each template's rate maps place the cells' fields",
                fix="pass at least one set of rate maps, such as "
                "encoding_models={'left': rates_left}, or one array of rate maps",
            )
        )
    if template is not None:
        if template not in encoding_models:
            raise ValueError(
                _format_error(
                    f"{operation}: template {template!r} is not one of {names}.",
                    why="template= names one of the rate-map sets in encoding_models",
                    fix=f"pass template= one of {names}",
                )
            )
        names = [template]
    elif single_template_method is not None:
        raise ValueError(
            _format_error(
                f"{operation}: method {single_template_method!r} needs template= "
                "with several rate-map sets.",
                why="it tests one template and has no rule to choose one",
                fix=f"pass template= one of {names}",
            )
        )
    if isinstance(env, Mapping):
        missing = [name for name in names if name not in env]
        if missing:
            raise ValueError(
                _format_error(
                    f"{operation}: templates {missing} have rate maps but no "
                    f"environment in env, whose templates are {list(env)}.",
                    why="each template's fields are read on its own environment",
                    fix="pass a mapping of environments with the keys of "
                    "encoding_models, or one environment",
                )
            )
        return [(name, env[name], encoding_models[name]) for name in names]
    return [(name, env, encoding_models[name]) for name in names]


def _template_rngs(
    rng: np.random.Generator | int | None, names: Sequence[str | None]
) -> list[np.random.Generator]:
    """The random stream of each template in ``names``, in their order.

    Several templates draw from streams derived from ``rng`` and their names,
    as in every other replay test, so a template's shuffles do not depend on
    the other templates or their order. One template draws from a stream
    seeded by ``rng``.
    """
    from neurospatial.replay.significance import _template_streams

    if len(names) > 1:
        streams = _template_streams(rng, [str(name) for name in names])
        return [streams[str(name)] for name in names]
    return [np.random.default_rng(np.random.default_rng(rng).integers(2**63))]


def _require_same_cells(n_trains: int, n_maps: int, operation: str) -> None:
    """Refuse spike trains and rate maps of different numbers of cells."""
    if n_trains != n_maps:
        raise ValueError(
            _format_error(
                f"{operation}: spike_times has {n_trains} cells but "
                f"encoding_models has {n_maps} rate maps.",
                why="each cell's firing is compared with its own place field",
                fix="pass spike times and rate maps for the same cells, in the "
                "same order",
            )
        )


def _spikes_in(
    trains: Sequence[NDArray[np.float64]], start: float, stop: float
) -> list[NDArray[np.float64]]:
    """Each train's spikes at ``start <= t < stop``, in the train's order."""
    return [train[(train >= start) & (train < stop)] for train in trains]


def _spikes_by_window(
    trains: Sequence[NDArray[np.float64]], windows: Any
) -> Iterator[list[NDArray[np.float64]]]:
    """:func:`_spikes_in` for each ``(start, stop)`` row of ``windows``, in turn.

    Each train is sorted once and cut by binary search instead of compared
    with every window. The spikes keep the train's order, so a summary such
    as their mean adds them in the same order as :func:`_spikes_in`.
    """
    bounds = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
    cuts = []
    for train in trains:
        order = (
            None
            if np.all(train[:-1] <= train[1:])
            else np.argsort(train, kind="stable")
        )
        lo, hi = np.searchsorted(
            train if order is None else train[order], bounds.T, side="left"
        )
        cuts.append((order, lo, hi))
    for window in range(len(bounds)):
        yield [
            train[lo[window] : hi[window]]
            if order is None
            else train[np.sort(order[lo[window] : hi[window]])]
            for train, (order, lo, hi) in zip(trains, cuts, strict=True)
        ]


def _pooled_spikes(
    in_event: Sequence[NDArray[np.float64]],
    cells: NDArray[np.intp],
    spike_statistic: str,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """When each of ``cells`` fired in an event, and each time's position in ``cells``.

    ``"all_spikes"`` gives every spike of ``in_event[cell]``, cell after cell
    in the order of ``cells``; the other statistics give one time per cell,
    its first, median or mean spike.
    """
    if spike_statistic == "all_spikes":
        times = np.concatenate([np.empty(0), *(in_event[cell] for cell in cells)])
        cell = np.repeat(np.arange(cells.size), [in_event[c].size for c in cells])
        return times, cell.astype(np.intp)
    summarize = _SPIKE_SUMMARIES[spike_statistic]
    times = np.array([summarize(in_event[cell]) for cell in cells], dtype=np.float64)
    return times, np.arange(cells.size, dtype=np.intp)


def _supported_rates(encoding_models: Any) -> NDArray[np.float64]:
    """Rate maps as an array, NaN outside a rate result's ``spatial_support``.

    A :class:`~neurospatial.encoding.spatial.SpatialRatesResult` can hold
    finite rates in bins with too little occupancy; those bins have no
    defensible rate, so a field's peak or center must not fall there.
    """
    rates = np.asarray(
        getattr(encoding_models, "firing_rates", encoding_models), dtype=np.float64
    )
    support = getattr(encoding_models, "spatial_support", None)
    if support is None:
        return rates
    return np.where(np.asarray(support, dtype=bool), rates, np.nan)


def _field_positions(
    env: Environment, encoding_models: Any, field_order: str, operation: str
) -> NDArray[np.float64]:
    """Each cell's field position along the track; NaN for a cell without a field."""
    from neurospatial.replay.correlation import _track_bin_centers

    centers = _track_bin_centers(env, operation)
    source = getattr(encoding_models, "env", None)
    if source is not None and source is not env:
        from neurospatial._exceptions import IncompatibleEnvironmentError
        from neurospatial.environment import _environment_differences

        bins, topology = _environment_differences(source, env)
        mismatch = bins or topology
        if mismatch is not None:
            raise IncompatibleEnvironmentError(
                f"{operation}: the rate result's environment differs from env "
                f"in {mismatch}.",
                fix="pass the environment the rates were computed on, or "
                "recompute the rates on env.",
                first=source,
                second=env,
            )
    rates = _supported_rates(encoding_models)
    if rates.ndim != 2 or rates.shape[1] != env.n_bins:
        raise ValueError(
            f"{operation}: encoding_models must have shape (n_cells, {env.n_bins}), "
            f"one rate map per cell on env, got {rates.shape}."
        )
    if np.any(rates < 0):
        raise ValueError(
            _format_error(
                f"{operation}: encoding_models has negative rates.",
                why="a firing rate is a count per second and cannot be negative",
                fix="pass rate maps in Hz; mark bins without training data with "
                "NaN, not a negative value",
            )
        )
    finite = np.isfinite(rates)
    clean = np.where(finite, rates, 0.0)
    has_field = clean.max(axis=1) > 0
    if field_order == "peak":
        positions = centers[np.argmax(clean, axis=1)]
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            positions = (clean @ centers) / clean.sum(axis=1)
    return np.where(has_field, positions, np.nan)
