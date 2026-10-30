"""Rank-order correlation between place-field order and firing order in an event.

A replay event activates place cells in the order of their fields on the
track, forward or reversed. The rank-order correlation (Foster & Wilson 2006;
Diba & Buzsáki 2007) is the Spearman correlation between each active cell's
field position and when it fired in the event. It needs no decoding, only
spike times and each cell's place field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin

if TYPE_CHECKING:
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment
    from neurospatial.stats.shuffle import ShuffleTestResult

_SPIKE_STATISTICS = ("first", "median", "mean", "all_spikes")
_FIELD_ORDERS = ("peak", "center")
_ALTERNATIVES = ("greater", "less", "two-sided")
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
        orderings enumerated) for up to 9 pairs, and the t-approximation of
        Foster & Wilson (2006) above that. MATLAB's ``corr``, used in the
        Bendor-lab analyses, also uses the exact distribution for small
        samples. ``spike_statistic="all_spikes"`` breaks
        the independence (a cell's spikes share one field position):
        Takigawa et al. (2024) found about 17% false positives with every
        spike when spikes were permuted one by one, which, like this
        p-value, treats them as independent. Use
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

    def __post_init__(self) -> None:
        """Own a read-only copy of ``active_units``."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "score": self.score,
            "analytic_p_value": self.analytic_p_value,
            "n_active_cells": self.n_active_cells,
            "spike_statistic": self.spike_statistic,
            "field_order": self.field_order,
        }


@dataclass(frozen=True, repr=False)
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
    p_value: float
    alternative: str

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
    interval : (start, stop)
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
        for such cells.
    min_active_cells : int, default=5
        Fewest active cells for a correlation (Chenani et al. 2019; Diba &
        Buzsáki 2007 required 5 cells or 30% of place cells, whichever was
        greater, when detecting events). With fewer, ``score`` is NaN.

    Returns
    -------
    RankOrderResult
        The correlation, its analytic p-value and the active cells.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if
        ``encoding_models`` does not have one row per cell and one column per
        bin, has negative rates, or is a rate result computed on another
        environment, if ``interval`` is not a finite ``(start, stop)`` with
        ``start < stop``, if ``spike_statistic`` or ``field_order`` is not
        one of the options, or if ``min_active_cells`` is less than 3.

    See Also
    --------
    rank_order_significance : Test the correlation against shuffles.

    Notes
    -----
    The correlation is Spearman's rho with average ranks for ties: the
    Pearson correlation of the ranks of the field positions and the ranks of
    the firing times. Two cells whose fields peak in the same bin tie.

    Tingley & Peyrache (2020) showed that short random sequences reach
    "significant" rank-order correlations with a random template 5-12% of the
    time (up to 20% at five cells), so judge the proportion of significant
    events against chance rather than single events. The
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
    spikes move together. Foster & Wilson (2006) used the analytic Spearman
    p-value per event instead, which :class:`RankOrderResult` reports as
    ``analytic_p_value``. The z-score of the shuffle test approximates the
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
    from neurospatial.stats import _ensure_rng
    from neurospatial.stats.shuffle import (
        ShuffleTestResult,
        compute_shuffle_pvalue,
        compute_shuffle_zscore,
    )

    operation = "rank_order_significance()"
    if (
        isinstance(n_shuffles, bool)
        or not isinstance(n_shuffles, (int, np.integer))
        or n_shuffles < 1
    ):
        raise ValueError(
            f"{operation} needs n_shuffles to be a positive integer, got "
            f"{n_shuffles!r}. Diba & Buzsáki used 500 permutations per event, "
            "Silva et al. 5000."
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
    generator = _ensure_rng(rng)
    n_active = observed.n_active_cells
    permutations = np.argsort(generator.random((n_shuffles, n_active)), axis=1)
    null = _permuted_correlations(pairs, permutations)
    test = ShuffleTestResult(
        observed_score=observed.score,
        null_scores=null,
        # Ties in rho differ by rounding between spearmanr and the matrix
        # formula of the shuffles.
        p_value=compute_shuffle_pvalue(
            round(observed.score, 12), np.round(null, 12), tail=alternative
        ),
        z_score=compute_shuffle_zscore(observed.score, null),
        shuffle_type="cell_order",
        n_shuffles=n_shuffles,
    )
    return RankOrderSignificanceResult(
        observed=observed,
        shuffle=test,
        p_value=test.p_value,
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
    if (
        isinstance(min_active_cells, bool)
        or not isinstance(min_active_cells, (int, np.integer))
        or min_active_cells < 3
    ):
        raise ValueError(
            f"{operation}: min_active_cells must be an integer of at least 3, got "
            f"{min_active_cells!r}; a rank correlation of two cells is always "
            "+1 or -1."
        )
    start, stop = _validated_interval(interval, operation)
    positions = _field_positions(env, encoding_models, field_order, operation)
    trains = as_spike_trains(spike_times)
    if len(trains) != len(positions):
        raise ValueError(
            _format_error(
                f"{operation}: spike_times has {len(trains)} cells but "
                f"encoding_models has {len(positions)} rate maps.",
                why="each cell's firing is compared with its own place field",
                fix="pass spike times and rate maps for the same cells, in the "
                "same order",
            )
        )
    in_event = [train[(train >= start) & (train < stop)] for train in trains]
    active = np.array(
        [
            unit
            for unit, spikes in enumerate(in_event)
            if spikes.size and np.isfinite(positions[unit])
        ],
        dtype=np.intp,
    )
    if spike_statistic == "all_spikes":
        times = (
            np.concatenate([in_event[unit] for unit in active])
            if active.size
            else np.empty(0)
        )
        cell = np.repeat(
            np.arange(active.size), [in_event[unit].size for unit in active]
        )
    else:
        summaries = [in_event[unit] for unit in active]
        if spike_statistic == "first":
            times = np.array([spikes.min() for spikes in summaries], dtype=np.float64)
        elif spike_statistic == "median":
            times = np.array(
                [np.median(spikes) for spikes in summaries], dtype=np.float64
            )
        else:
            times = np.array([spikes.mean() for spikes in summaries], dtype=np.float64)
        cell = np.arange(active.size)
    if active.size < min_active_cells:
        times, cell = np.empty(0), np.empty(0, dtype=np.intp)
    return _EventPairs(
        field_positions=positions[active],
        times=np.asarray(times, dtype=np.float64),
        cell=np.asarray(cell, dtype=np.intp),
        active_units=active,
    )


def _result(
    pairs: _EventPairs, spike_statistic: str, field_order: str
) -> RankOrderResult:
    score, p_value = np.nan, np.nan
    if pairs.times.size:
        x = pairs.field_positions[pairs.cell]
        if np.ptp(x) > 0 and np.ptp(pairs.times) > 0:
            test = stats.spearmanr(x, pairs.times)
            score = float(test.statistic)
            p_value = _spearman_p_value(x, pairs.times, score, float(test.pvalue))
    return RankOrderResult(
        score=score,
        analytic_p_value=p_value,
        n_active_cells=int(pairs.active_units.size),
        active_units=pairs.active_units,
        spike_statistic=spike_statistic,
        field_order=field_order,
    )


_EXACT_MAX_PAIRS = 9


def _spearman_p_value(
    x: NDArray[np.float64], times: NDArray[np.float64], score: float, approximate: float
) -> float:
    """Two-sided Spearman p-value: exact for few pairs, else the t-approximation."""
    import math
    from itertools import permutations

    n = x.size
    if n > _EXACT_MAX_PAIRS:
        # The t-approximation reaches 0 for |rho| = 1; no ordering of n
        # distinct values is rarer than 2 / n!.
        return max(approximate, 2.0 / math.factorial(n))
    x_ranks = stats.rankdata(x)
    t_ranks = stats.rankdata(times)
    orders = np.array(list(permutations(range(n))), dtype=np.intp)
    permuted = x_ranks[orders] - x_ranks.mean()
    t_dev = t_ranks - t_ranks.mean()
    rho = (permuted @ t_dev) / np.sqrt((permuted[0] ** 2).sum() * (t_dev**2).sum())
    return float(np.mean(np.abs(np.round(rho, 12)) >= round(abs(score), 12)))


def _permuted_correlations(
    pairs: _EventPairs, permutations: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Spearman correlation under each permutation of the active cells' fields."""
    t_ranks = stats.rankdata(pairs.times)
    t_dev = t_ranks - t_ranks.mean()
    correlations = np.empty(len(permutations))
    # Rank the pairs in blocks of shuffles to bound memory with many spikes.
    block = max(1, _BLOCK_VALUES // max(pairs.cell.size, 1))
    for start in range(0, len(permutations), block):
        chunk = permutations[start : start + block]
        x = pairs.field_positions[chunk][:, pairs.cell]  # (n_chunk, n_pairs)
        x_ranks = stats.rankdata(x, axis=1)
        x_dev = x_ranks - x_ranks.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations[start : start + block] = (x_dev @ t_dev) / np.sqrt(
                (x_dev**2).sum(axis=1) * (t_dev**2).sum()
            )
    return correlations


def _validated_interval(interval: Any, operation: str) -> tuple[float, float]:
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
    rates = np.asarray(
        getattr(encoding_models, "firing_rates", encoding_models), dtype=np.float64
    )
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
