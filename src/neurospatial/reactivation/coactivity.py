"""Co-activity measures built on assembly patterns and pair correlations.

- :func:`assembly_member_cofiring`: the Pearson correlation of every pair of
  a pattern's members in two periods, for example over the sharp-wave
  ripples (SWRs) of sleep before and after learning (El-Gaby et al. 2021).
- :func:`assembly_similarity`: the absolute cosine between the weights of
  patterns detected in two sessions, and each pattern's best match
  (El-Gaby et al. 2021).
- :func:`assembly_event_response`: the activation strength R(t) averaged
  around events such as SWR onsets, and its mean in a response window
  (the Fernández-Ruiz and Oliva labs: Harvey et al. 2023, Liu et al. 2023,
  Chang et al. 2025, Robinson et al. 2026).
- :func:`explained_variance_contributions`: each pair's contribution to the
  explained variance, EV with all pairs minus EV without that pair (Oliva
  et al. 2020).
- :func:`ev_rev_contrast`: (EV - REV) / (EV + REV) (Karaba et al. 2024).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin, resolve_unit_ids
from neurospatial.reactivation.assemblies import (
    AssemblyDetectionResult,
    AssemblyPattern,
    _columns_by_label,
    _label_order,
    _n_units_from_pairs,
    _one_control_ev,
    _pairwise_correlations,
    _selected_pairs,
    _validated_counts,
    explained_variance_reactivation,
)

__all__ = [
    "AssemblyEventResponseResult",
    "AssemblyMemberCofiringResult",
    "AssemblySimilarityResult",
    "ExplainedVarianceContributionResult",
    "assembly_event_response",
    "assembly_member_cofiring",
    "assembly_similarity",
    "ev_rev_contrast",
    "explained_variance_contributions",
]


# =============================================================================
# Member co-firing
# =============================================================================


@dataclass(frozen=True, repr=False, eq=False)
class AssemblyMemberCofiringResult(ReadOnlyArraysMixin, ResultMixin):
    """Correlations between a pattern's members in a control and a match period.

    Attributes
    ----------
    member_unit_ids : ndarray, shape (n_members,)
        Labels of the pattern's members, in ``member_indices`` order.
    control_correlations, match_correlations : ndarray, shape (n_member_pairs_all,)
        Pearson correlation of each pair of members over the period's bins
        (for example one bin per SWR), in ``np.triu_indices(n_members, k=1)``
        order; NaN for a pair with a member constant in that period.
    control_mean_correlation, match_mean_correlation : float
        Mean correlation over the pairs defined in both periods, so both
        means use the same pairs; NaN when there is none.
    mean_correlation_difference : float
        ``match_mean_correlation - control_mean_correlation``.
    n_member_pairs : int
        Pairs defined in both periods.
    n_control_bins, n_match_bins : int
        Rows of the two count matrices (events, when each event is a bin).
    """

    member_unit_ids: NDArray[Any]
    control_correlations: NDArray[np.float64]
    match_correlations: NDArray[np.float64]
    control_mean_correlation: float
    match_mean_correlation: float
    mean_correlation_difference: float
    n_member_pairs: int
    n_control_bins: int
    n_match_bins: int

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "mean_correlation_difference": self.mean_correlation_difference,
            "control_mean_correlation": self.control_mean_correlation,
            "match_mean_correlation": self.match_mean_correlation,
            "n_member_pairs": self.n_member_pairs,
            "n_control_bins": self.n_control_bins,
            "n_match_bins": self.n_match_bins,
        }


def assembly_member_cofiring(
    control_counts: NDArray[np.float64],
    match_counts: NDArray[np.float64],
    pattern: AssemblyPattern,
    *,
    control_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    match_unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> AssemblyMemberCofiringResult:
    """Correlate a pattern's members in a control and a match period.

    El-Gaby et al. (2021) measured offline reactivation as the co-firing of a
    pattern's members during sleep: "we used SWR events as activity bins and
    calculated the Pearson correlations between pairs of pattern members
    separately across each sleep session". This function computes the
    correlation of every pair of members (``pattern.member_indices``) with
    :func:`pairwise_correlations` in each period and the mean over the pairs
    defined in both.

    Parameters
    ----------
    control_counts : NDArray[np.float64], shape (n_control_bins, n_neurons)
        Spike counts of the control period, for example one row per SWR of
        sleep before learning; finite and non-negative.
    match_counts : NDArray[np.float64], shape (n_match_bins, n_neurons)
        Spike counts of the match period, for example one row per SWR of
        sleep after learning.
    pattern : AssemblyPattern
        The pattern whose members are correlated, usually from
        ``detect_assemblies(..., member_rule="positive_z")`` on the task.
    control_unit_ids, match_unit_ids : array-like, shape (n_neurons,), optional
        Labels of the columns of each period, to match them to the pattern's
        units by label, as in
        :func:`~neurospatial.reactivation.assembly_reactivation`. Pass both
        or neither.

    Returns
    -------
    AssemblyMemberCofiringResult
        Each pair's correlation in both periods, their means over the pairs
        defined in both and the difference.

    Raises
    ------
    ValueError
        If the counts hold NaN, infinite or negative values, do not have one
        column per pattern weight, or only one of the label arguments is
        given.

    Warns
    -----
    UserWarning
        If the pattern has fewer than two members (no pair; the means are
        NaN), or members are constant in a period (their pairs are NaN
        there).

    Notes
    -----
    **Event bins and their durations.** With one row per event, each count
    is a number of spikes in an interval whose length varies from event to
    event. Every unit fires more in a longer event, so the counts of every
    pair share a duration component that raises all pair correlations
    alike. It cancels in the difference only when the two periods' events
    have similar durations. Compare the durations of the two periods'
    events, or divide counts by duration before the call; El-Gaby et al.
    (2021) do not say whether they did.

    **No test.** El-Gaby et al. (2021) report means with bootstrap
    confidence intervals across units, without stating whether the unit is
    a pair, a member or a pattern; this function reports the per-pair
    values so either pooling is possible.

    Examples
    --------
    Units 0, 1 and 2 are members. They fire together in the events after
    learning and independently before it:

    >>> import numpy as np
    >>> from neurospatial.reactivation import (
    ...     AssemblyPattern,
    ...     assembly_member_cofiring,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> pattern = AssemblyPattern(
    ...     np.array([0.6, 0.6, 0.5, 0.1, 0.1]), np.array([0, 1, 2]), 0.4
    ... )
    >>> before = rng.poisson(2.0, (300, 5)).astype(float)  # one row per SWR
    >>> after = rng.poisson(2.0, (300, 5)).astype(float)
    >>> after[:, :3] += rng.poisson(2.0, (300, 1))
    >>> result = assembly_member_cofiring(before, after, pattern)
    >>> result.n_member_pairs
    3
    >>> bool(result.mean_correlation_difference > 0.3)
    True
    """
    call = "assembly_member_cofiring()"
    if (control_unit_ids is None) != (match_unit_ids is None):
        raise ValueError(
            _format_error(
                f"{call} needs control_unit_ids= and match_unit_ids= together.",
                why="Why: one period's columns would be matched by label and "
                "the other's by position.",
                fix="pass the labels of both periods' columns, or neither",
            )
        )
    periods = []
    for counts, labels, argument, labels_argument in (
        (control_counts, control_unit_ids, "control_counts", "control_unit_ids"),
        (match_counts, match_unit_ids, "match_counts", "match_unit_ids"),
    ):
        checked = _columns_by_label(
            _validated_counts(counts, call=call, argument=argument),
            labels,
            pattern,
            call=call,
            argument=argument,
            labels_argument=labels_argument,
        )
        if checked.shape[1] != pattern.weights.size:
            raise ValueError(
                f"{call}: {argument} has {checked.shape[1]} neurons but the "
                f"pattern has {pattern.weights.size} weights. They must match."
            )
        periods.append(checked)
    members = pattern.member_indices
    member_labels = np.asarray(pattern.unit_ids)[members]
    correlations = []
    for counts, period in zip(periods, ("control_counts", "match_counts"), strict=True):
        if members.size < 2:
            correlations.append(np.zeros(0))
            continue
        pairs, constant = _pairwise_correlations(counts[:, members])
        if np.any(constant):
            warnings.warn(
                f"{call}: members {member_labels[constant].tolist()} have the "
                f"same count in every bin of {period}, so their pairs have no "
                "correlation there (NaN) and are left out of both means.",
                UserWarning,
                stacklevel=2,
            )
        correlations.append(pairs)
    control, match = correlations
    if members.size < 2:
        warnings.warn(
            f"{call}: the pattern has {members.size} member(s), so it has no "
            "pair of members to correlate; the means are NaN.",
            UserWarning,
            stacklevel=2,
        )
    both = np.isfinite(control) & np.isfinite(match)
    control_mean = float(control[both].mean()) if both.any() else np.nan
    match_mean = float(match[both].mean()) if both.any() else np.nan
    return AssemblyMemberCofiringResult(
        member_unit_ids=member_labels,
        control_correlations=control,
        match_correlations=match,
        control_mean_correlation=control_mean,
        match_mean_correlation=match_mean,
        mean_correlation_difference=match_mean - control_mean,
        n_member_pairs=int(both.sum()),
        n_control_bins=int(periods[0].shape[0]),
        n_match_bins=int(periods[1].shape[0]),
    )


# =============================================================================
# Pattern similarity
# =============================================================================


@dataclass(frozen=True, repr=False, eq=False)
class AssemblySimilarityResult(ReadOnlyArraysMixin, ResultMixin):
    """Absolute cosine similarity between two sets of patterns.

    Attributes
    ----------
    similarity : ndarray, shape (n_patterns, n_reference_patterns)
        ``|cos|`` of the angle between the weights of each pattern and each
        reference pattern, in [0, 1]; 1 for the same pattern up to sign.
    best_match_indices : ndarray of int, shape (n_patterns,)
        Position of each pattern's most similar reference pattern (the
        first on ties).
    best_match_similarity : ndarray, shape (n_patterns,)
        Each pattern's largest similarity, its "maximum similarity"
        (El-Gaby et al. 2021).
    """

    similarity: NDArray[np.float64]
    best_match_indices: NDArray[np.int64]
    best_match_similarity: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        n_patterns, n_reference = self.similarity.shape
        return {
            "n_patterns": n_patterns,
            "n_reference_patterns": n_reference,
            "mean_best_match_similarity": float(self.best_match_similarity.mean()),
        }


def _pattern_weights(
    patterns: Sequence[AssemblyPattern] | AssemblyDetectionResult, argument: str
) -> tuple[NDArray[np.float64], NDArray[Any], bool]:
    """Unit-length weights of a set of patterns that share their units.

    Returns
    -------
    weights : ndarray, shape (n_patterns, n_neurons)
    unit_ids : ndarray, shape (n_neurons,)
    generated : bool
        Whether the labels are generated column positions.
    """
    call = "assembly_similarity()"
    items = tuple(
        patterns.patterns if isinstance(patterns, AssemblyDetectionResult) else patterns
    )
    if not items or not all(isinstance(item, AssemblyPattern) for item in items):
        raise ValueError(
            _format_error(
                f"{call}: {argument} must hold at least one AssemblyPattern.",
                fix="pass detect_assemblies(...).patterns, or the result itself",
            )
        )
    labels = np.asarray(items[0].unit_ids)
    if any(not np.array_equal(np.asarray(item.unit_ids), labels) for item in items):
        raise ValueError(
            f"{call}: every pattern of {argument} must carry the same unit_ids, "
            "as the patterns of one detect_assemblies() result do."
        )
    weights = np.array([item.weights for item in items], dtype=np.float64)
    norms = np.linalg.norm(weights, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"{call}: a pattern of {argument} has only zero weights.")
    return weights / norms[:, None], labels, items[0]._unit_ids_generated


def assembly_similarity(
    patterns: Sequence[AssemblyPattern] | AssemblyDetectionResult,
    reference_patterns: Sequence[AssemblyPattern] | AssemblyDetectionResult,
) -> AssemblySimilarityResult:
    """Absolute cosine similarity of patterns detected in two sessions.

    El-Gaby et al. (2021) compared the patterns of two sessions recorded
    with the same units: "We computed the cosine similarity between the
    weight vector representing a given pattern detected in one session ...
    and the weight vectors representing each individual pattern detected in
    another task session ... For each pattern, we thus identified its
    'maximum similarity' value with a pattern (i.e., 'the best match') of
    another session". Their preprint states that the similarity is "the
    absolute value of the cosine similarity of their normalised weight
    vectors", since a pattern's sign is arbitrary.

    Parameters
    ----------
    patterns, reference_patterns : sequence of AssemblyPattern, or AssemblyDetectionResult
        The two sets, each from one :func:`detect_assemblies` call. With
        real labels (``detect_assemblies(..., unit_ids=...)``) both sets must
        hold the same units, in any order; with generated labels the
        weights are paired by position and must have the same length.

    Returns
    -------
    AssemblySimilarityResult
        The similarity of every pair of patterns and each pattern's best
        match among the reference patterns.

    Raises
    ------
    ValueError
        If a set is empty or mixes units, or the two sets hold different
        units.

    Notes
    -----
    El-Gaby et al. (2021) classified patterns with these similarities:
    "contingency-discriminating" patterns have a best match in the other
    contingency below the 90th percentile of the best matches between the
    task and a control enclosure, pooled over all recording days. The
    thresholds pool sessions, so the classification is a step for your own
    analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reactivation import AssemblyPattern, assembly_similarity
    >>> day = [
    ...     AssemblyPattern(np.array([0.8, 0.6, 0.0]), np.array([0]), 0.3),
    ...     AssemblyPattern(np.array([0.0, 0.0, 1.0]), np.array([2]), 0.2),
    ... ]
    >>> other = [AssemblyPattern(np.array([0.0, -0.6, -0.8]), np.array([2]), 0.3)]
    >>> result = assembly_similarity(day, other)
    >>> result.similarity.round(2).tolist()
    [[0.36], [0.8]]
    >>> result.best_match_indices.tolist()
    [0, 0]
    """
    call = "assembly_similarity()"
    weights, labels, generated = _pattern_weights(patterns, "patterns")
    reference, reference_labels, reference_generated = _pattern_weights(
        reference_patterns, "reference_patterns"
    )
    if generated != reference_generated:
        raise ValueError(
            _format_error(
                f"{call}: one set of patterns has unit labels and the other has "
                "generated column positions.",
                fix="pass unit_ids= to both detect_assemblies() calls",
            )
        )
    if generated:
        if reference.shape[1] != weights.shape[1]:
            raise ValueError(
                f"{call}: the patterns have {weights.shape[1]} weights and the "
                f"reference patterns {reference.shape[1]}; without unit labels "
                "they must match by position."
            )
    else:
        order, only_reference, only_patterns = _label_order(reference_labels, labels)
        if only_reference or only_patterns:
            raise ValueError(
                _format_error(
                    f"{call}: the two sets hold different units; only in "
                    f"patterns: {only_reference}, only in reference_patterns: "
                    f"{only_patterns}.",
                    why="Why: a cosine compares the weights of the same cells.",
                    fix="detect both sets on the units recorded in both sessions",
                )
            )
        reference = reference[:, order]
    similarity = np.clip(np.abs(weights @ reference.T), 0.0, 1.0)
    best = np.argmax(similarity, axis=1)
    return AssemblySimilarityResult(
        similarity=similarity,
        best_match_indices=best.astype(np.int64),
        best_match_similarity=similarity[np.arange(len(best)), best],
    )


# =============================================================================
# Event-triggered activation strength
# =============================================================================


@dataclass(frozen=True, repr=False, eq=False)
class AssemblyEventResponseResult(ReadOnlyArraysMixin, ResultMixin):
    """Activation strength around events and each event's response.

    Attributes
    ----------
    lags : ndarray, shape (n_lags,)
        Times relative to each event, seconds, multiples of the sample step
        inside ``window``.
    peri_event_mean : ndarray, shape (n_lags,)
        Mean activation strength at each lag over the events with a sample
        there; NaN where none has.
    event_responses : ndarray, shape (n_events,)
        Mean activation strength of each event over the lags in
        ``response_window``; NaN for an event with no sample there. In input
        order.
    baseline_responses : ndarray, shape (n_events,)
        The same over ``baseline_window``; all NaN without one.
    response : float
        ``reduce="mean"``: the mean of ``event_responses``. ``reduce="peak"``:
        the largest ``peri_event_mean`` over the response lags.
    p_value : float
        Two-sided Wilcoxon signed-rank test of ``event_responses`` against
        ``baseline_responses`` over the events with both; NaN without a
        baseline or without such an event, 1 when every difference is 0.
    n_events : int
        Events with a defined response.
    window, response_window : tuple of float
        The windows used, seconds relative to the events.
    baseline_window : tuple of float or None
        The baseline window, if any.
    reduce : {"mean", "peak"}
        How ``response`` was computed.
    """

    lags: NDArray[np.float64]
    peri_event_mean: NDArray[np.float64]
    event_responses: NDArray[np.float64]
    baseline_responses: NDArray[np.float64]
    response: float
    p_value: float
    n_events: int
    window: tuple[float, float]
    response_window: tuple[float, float]
    baseline_window: tuple[float, float] | None
    reduce: Literal["mean", "peak"]

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "response": self.response,
            "p_value": self.p_value,
            "n_events": self.n_events,
        }


def _checked_event_windows(
    window: Any,
    response_window: Any,
    baseline_window: Any,
    owner: str,
    step: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float] | None]:
    """Check the peri-event windows of an event response.

    Each window must be ``(start, stop)`` seconds with ``start < stop``, and
    the response and baseline windows must lie inside ``window``.

    Parameters
    ----------
    window, response_window : array-like, shape (2,)
    baseline_window : array-like, shape (2,), or None
    owner : str
        Start of each message: ``"EventResponse"`` or the function call.
    step : float, optional
        Sample step of the R(t) trace a function call measures. Given, the
        response and baseline windows must also hold one of its lags, and
        the messages name the events and the step.

    Returns
    -------
    window, response_window, baseline_window : tuple of float, or None
    """
    relative = "" if step is None else " relative to the events"

    def pair(name: str, value: Any) -> tuple[float, float]:
        array = np.asarray(value, dtype=np.float64)
        if (
            array.shape != (2,)
            or not np.all(np.isfinite(array))
            or array[0] >= array[1]
        ):
            raise ValueError(
                f"{owner}: {name} must be (start, stop) in seconds{relative} with "
                f"start < stop, got {value!r}."
            )
        return float(array[0]), float(array[1])

    full = pair("window", window)
    response = pair("response_window", response_window)
    baseline = (
        None if baseline_window is None else pair("baseline_window", baseline_window)
    )
    for name, value, part in (
        ("response_window", response_window, response),
        ("baseline_window", baseline_window, baseline),
    ):
        if part is None:
            continue
        inside = full[0] <= part[0] and part[1] <= full[1]
        if step is None:
            if not inside:
                raise ValueError(
                    f"{owner}: {name}={value} must lie inside window={window}."
                )
        elif not inside or not _window_holds_lag(full, part, step):
            raise ValueError(
                _format_error(
                    f"{owner}: {name}={part} must lie inside window={full} and "
                    f"hold at least one lag of the {step:g} s sample step.",
                    fix=f"widen window= or choose a {name} inside it",
                )
            )
    return full, response, baseline


def _window_holds_lag(
    window: tuple[float, float], part: tuple[float, float], step: float
) -> bool:
    """Whether ``part`` holds one of the lags of ``window`` sampled every ``step``."""
    return bool(_in_lag_window(_lags(window, step), part, step).any())


def _lags(window: tuple[float, float], step: float) -> NDArray[np.float64]:
    """Multiples of ``step`` in ``[window[0], window[1])``."""
    tolerance = 1e-9
    first = int(np.ceil(window[0] / step - tolerance))
    stop = int(np.ceil(window[1] / step - tolerance))
    lags: NDArray[np.float64] = np.arange(first, stop, dtype=np.float64) * step
    return lags


def _in_lag_window(
    lags: NDArray[np.float64], window: tuple[float, float], step: float
) -> NDArray[np.bool_]:
    """Lags in the half-open ``[window[0], window[1])``, up to rounding."""
    tolerance = 1e-9 * step
    return (lags >= window[0] - tolerance) & (lags < window[1] - tolerance)


def _peri_event_values(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    step: float,
    event_times: NDArray[np.float64],
    lags: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Each event's samples at ``event + lag``.

    A query time takes the sample whose bin ``[t - step / 2, t + step / 2)``
    holds it; a query in a gap between samples, or outside them, is NaN.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Sample times (bin centers), increasing.
    values : ndarray, shape (n_samples, n_series)
        Samples.
    step : float
        Sample spacing, seconds.
    event_times : ndarray, shape (n_events,)
    lags : ndarray, shape (n_lags,)

    Returns
    -------
    ndarray, shape (n_events, n_lags, n_series)
    """
    queries = event_times[:, None] + lags[None, :]
    # The last bin starting at or before each query (within rounding, so a
    # query on a bin edge belongs to the bin that starts there); the query
    # is in it unless it lies at or past the bin's end.
    tolerance = 1e-9 * step
    index = np.searchsorted(times - step / 2, queries + tolerance, side="right") - 1
    found = index >= 0
    index = np.clip(index, 0, times.size - 1)
    inside = found & (queries < times[index] + step / 2 - tolerance)
    out = values[index].astype(np.float64, copy=False)
    out[~inside] = np.nan
    return out


def _nanmean(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Mean over the finite values along ``axis``, NaN where none is."""
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean: NDArray[np.float64] = np.where(count > 0, total / count, np.nan)
    return mean


def _mean_over_events(responses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean of each series' defined responses, shape (n_series,)."""
    return _nanmean(responses, axis=0)


def _mean_of_defined_events(responses: NDArray[np.float64]) -> NDArray[np.float64]:
    """:func:`_mean_over_events`, averaging each series' defined responses alone.

    Summed without the undefined events, so it can differ in the last bit.
    """
    means = []
    for series in responses.T:
        defined = series[np.isfinite(series)]
        means.append(defined.mean() if defined.size else np.nan)
    return np.array(means, dtype=np.float64)


class _PeriEventResponses(NamedTuple):
    """Peri-event measures of R(t) series, from :func:`_peri_event_responses`.

    Attributes
    ----------
    lags : ndarray, shape (n_lags,)
    peri_event_mean : ndarray, shape (n_lags, n_series)
        Mean over the events at each lag.
    responses : ndarray, shape (n_events, n_series)
        Each event's mean over the response lags.
    summary : ndarray, shape (n_series,)
        The ``event_mean`` of the responses, or with ``reduce="peak"`` the
        largest finite peri-event mean over the response lags.
    baselines : ndarray, shape (n_events, n_series), or None
        Each event's mean over the baseline lags.
    """

    lags: NDArray[np.float64]
    peri_event_mean: NDArray[np.float64]
    responses: NDArray[np.float64]
    summary: NDArray[np.float64]
    baselines: NDArray[np.float64] | None


def _peri_event_responses(
    times: NDArray[np.float64],
    activation: NDArray[np.float64],
    event_times: NDArray[np.float64],
    step: float,
    window: tuple[float, float],
    response_window: tuple[float, float],
    baseline_window: tuple[float, float] | None,
    reduce: Literal["mean", "peak"],
    event_mean: Callable[
        [NDArray[np.float64]], NDArray[np.float64]
    ] = _mean_over_events,
) -> _PeriEventResponses:
    """Event responses of R(t) series sampled every ``step``.

    The windows must already be checked (:func:`_checked_event_windows`).

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Sample times (bin centers), increasing.
    activation : ndarray, shape (n_samples, n_series)
    event_times : ndarray, shape (n_events,)
    step : float
        Sample spacing, seconds.
    window, response_window, baseline_window : tuple of float
        Seconds relative to the events, half-open; ``baseline_window`` may be
        None.
    reduce : {"mean", "peak"}
    event_mean : callable
        Maps the responses, shape (n_events, n_series), to their mean over
        the defined events, shape (n_series,); formulations round
        differently.

    Returns
    -------
    _PeriEventResponses
    """
    lags = _lags(window, step)
    in_response = _in_lag_window(lags, response_window, step)
    # Shape (n_events, n_lags, n_series).
    values = _peri_event_values(times, activation, step, event_times, lags)
    peri_event_mean = _nanmean(values, axis=0)
    responses = _nanmean(values[:, in_response], axis=1)
    if reduce == "mean":
        summary = event_mean(responses)
    else:
        peak = peri_event_mean[in_response]
        finite = np.isfinite(peak)
        summary = np.where(
            finite.any(axis=0), np.max(np.where(finite, peak, -np.inf), axis=0), np.nan
        )
    baselines = (
        None
        if baseline_window is None
        else _nanmean(values[:, _in_lag_window(lags, baseline_window, step)], axis=1)
    )
    return _PeriEventResponses(lags, peri_event_mean, responses, summary, baselines)


def _signed_rank_p(
    responses: NDArray[np.float64], baselines: NDArray[np.float64]
) -> tuple[float, float]:
    """Two-sided signed-rank p-value and median difference of paired values.

    Returns
    -------
    p_value : float
        NaN without a pair defined in both, 1 when every difference is 0.
    median_difference : float
    """
    both = np.isfinite(responses) & np.isfinite(baselines)
    differences = responses[both] - baselines[both]
    if not differences.size:
        return np.nan, np.nan
    median = float(np.median(differences))
    if not np.any(differences != 0):
        return 1.0, median
    return float(stats.wilcoxon(differences).pvalue), median


def assembly_event_response(
    times: ArrayLike,
    activation: ArrayLike,
    event_times: ArrayLike,
    window: tuple[float, float],
    *,
    response_window: tuple[float, float] = (0.0, 0.1),
    baseline_window: tuple[float, float] | None = None,
    reduce: Literal["mean", "peak"] = "mean",
) -> AssemblyEventResponseResult:
    """Activation strength around events, such as sharp-wave ripple onsets.

    The Fernández-Ruiz and Oliva labs measure a pattern's reactivation as its
    activation strength R(t) around SWR onsets in sleep: "Assembly responses
    to SWRs were quantified as the mean reactivation strength within 0–100
    ms of SWR onset" (Robinson et al. 2026). This function aligns R(t) to
    each event, averages it over events at each lag, and takes each event's
    mean over ``response_window``.

    Parameters
    ----------
    times : array-like, shape (n_samples,)
        Time of each sample of ``activation`` (the bin center), seconds,
        strictly increasing; spaced by a regular step within each recorded
        stretch, with gaps between stretches allowed.
    activation : array-like, shape (n_samples,)
        Activation strength R(t), for example
        :func:`~neurospatial.reactivation.assembly_activation` of counts
        binned with :func:`~neurospatial.decoding.bin_spikes_in_time`. NaN
        marks a missing sample.
    event_times : array-like, shape (n_events,)
        Event times, for example SWR onsets, on the clock of ``times``.
    window : tuple of float
        ``(start, stop)`` of the peri-event lags, seconds relative to each
        event, half-open; for example ``(-0.5, 0.5)``.
    response_window : tuple of float, default=(0.0, 0.1)
        Lags averaged into each event's response, half-open, inside
        ``window``. The default is the 0–100 ms after SWR onset of Liu et
        al. (2023), Chang et al. (2025) and Robinson et al. (2026); Harvey
        et al. (2023) took the peak over 0–50 ms.
    baseline_window : tuple of float, optional
        Lags of each event's baseline, inside ``window``, for a signed-rank
        test of the responses against it; Chang et al. (2025) used
        ``(-0.5, -0.1)``.
    reduce : {"mean", "peak"}, default="mean"
        ``"mean"``: ``response`` is the mean of the events' responses.
        ``"peak"``: the largest value of the peri-event mean over the
        response lags (Harvey et al. 2023).

    Returns
    -------
    AssemblyEventResponseResult
        Lags, peri-event mean, each event's response (and baseline), the
        response and the test's p-value.

    Raises
    ------
    ValueError
        If ``times`` is not strictly increasing, the arrays have different
        lengths or non-finite times, a window is not ``start < stop``, the
        response or baseline window is outside ``window`` or holds no lag,
        or ``reduce`` is unknown.

    Notes
    -----
    **Samples and lags.** The sample step is the median spacing of
    ``times``; the lags are its multiples in ``window``. Each lag takes the
    sample whose bin ``[t - step / 2, t + step / 2)`` holds ``event +
    lag``: an event between two bin centers is assigned the bin that holds
    it, without interpolation. Lags in a gap between stretches, or beyond
    the samples, are NaN and left out of the means.

    **Z-scoring.** R(t) uses z-scored counts, and the choice of samples
    changes what a rate change does. The labs z-score pre- and post-task
    sleep together: compute R(t) once over both periods' counts, stacked.
    Then members that fire more after the task, without firing together
    more, have higher z-scores in every event after it, and the response
    rises. Z-scored each period on its own, their larger variance after the
    task shrinks their z-scores in events, and the response falls. Neither
    choice removes a rate change; compare the members' rates in both
    periods.

    Examples
    --------
    R(t) sampled every 10 ms, with a bump 20–60 ms after each of three
    events:

    >>> import numpy as np
    >>> from neurospatial.reactivation import assembly_event_response
    >>> times = np.arange(0.005, 10.0, 0.01)
    >>> events = np.array([2.0, 5.0, 8.0])
    >>> activation = np.zeros(times.size)
    >>> for event in events:
    ...     activation[(times > event + 0.02) & (times < event + 0.06)] = 4.0
    >>> result = assembly_event_response(times, activation, events, (-0.5, 0.5))
    >>> result.event_responses.round(2).tolist()
    [1.6, 1.6, 1.6]
    >>> result.lags[:3].round(2).tolist()
    [-0.5, -0.49, -0.48]
    """
    call = "assembly_event_response()"
    sample_times = np.asarray(times, dtype=np.float64)
    strength = np.asarray(activation, dtype=np.float64)
    events = np.asarray(event_times, dtype=np.float64)
    if sample_times.ndim != 1 or sample_times.size < 2:
        raise ValueError(f"{call}: times must be 1-D with at least 2 samples.")
    if strength.shape != sample_times.shape:
        raise ValueError(
            f"{call}: activation must have one value per time, got shape "
            f"{strength.shape} for {sample_times.size} times."
        )
    if not np.all(np.isfinite(sample_times)) or np.any(np.diff(sample_times) <= 0):
        raise ValueError(
            _format_error(
                f"{call}: times must be finite and strictly increasing.",
                fix="pass the bin centers of the activation's time bins, in order",
            )
        )
    if events.ndim != 1 or not np.all(np.isfinite(events)):
        raise ValueError(f"{call}: event_times must be a 1-D array of finite times.")
    if reduce not in ("mean", "peak"):
        raise ValueError(f"{call}: reduce must be 'mean' or 'peak', got {reduce!r}.")
    step = float(np.median(np.diff(sample_times)))
    full, response, baseline = _checked_event_windows(
        window, response_window, baseline_window, call, step
    )
    measured = _peri_event_responses(
        sample_times,
        strength[:, None],
        events,
        step,
        full,
        response,
        baseline,
        reduce,
        event_mean=_mean_of_defined_events,
    )
    event_responses = measured.responses[:, 0]
    if measured.baselines is None:
        baseline_responses = np.full(events.size, np.nan)
        p_value = np.nan
    else:
        baseline_responses = measured.baselines[:, 0]
        p_value = _signed_rank_p(event_responses, baseline_responses)[0]
    return AssemblyEventResponseResult(
        lags=measured.lags,
        peri_event_mean=measured.peri_event_mean[:, 0],
        event_responses=event_responses,
        baseline_responses=baseline_responses,
        response=float(measured.summary[0]),
        p_value=p_value,
        n_events=int(np.isfinite(event_responses).sum()),
        window=full,
        response_window=response,
        baseline_window=baseline,
        reduce=reduce,
    )


# =============================================================================
# Explained variance: per-pair contributions and the EV/REV contrast
# =============================================================================


@dataclass(frozen=True, repr=False, eq=False)
class ExplainedVarianceContributionResult(ReadOnlyArraysMixin, ResultMixin):
    """Each pair's and each unit's contribution to the explained variance.

    Attributes
    ----------
    explained_variance : float
        EV with every valid pair, as
        :func:`~neurospatial.reactivation.explained_variance_reactivation`
        computes it.
    pair_contributions : ndarray, shape (n_pairs,)
        EV with every valid pair minus EV without that pair, in
        ``np.triu_indices(n_units, k=1)`` order; NaN for pairs left out
        (NaN in a period or not in ``pair_mask``).
    unit_contributions : ndarray, shape (n_units,)
        Mean contribution of the valid pairs each unit belongs to; NaN for a
        unit in none.
    unit_ids : ndarray, shape (n_units,)
        Label of each unit.
    n_pairs : int
        Valid pairs.
    """

    explained_variance: float
    pair_contributions: NDArray[np.float64]
    unit_contributions: NDArray[np.float64]
    unit_ids: NDArray[Any]
    n_pairs: int

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {"explained_variance": self.explained_variance, "n_pairs": self.n_pairs}


def _leave_one_out_correlations(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pearson correlation of ``x`` and ``y`` without each element in turn."""
    n = x.size - 1
    # Centering first keeps the sums small; it does not change correlations.
    x = x - x.mean()
    y = y - y.mean()
    sx, sy = x.sum() - x, y.sum() - y
    sxx, syy, sxy = (x**2).sum() - x**2, (y**2).sum() - y**2, (x * y).sum() - x * y
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation: NDArray[np.float64] = (n * sxy - sx * sy) / np.sqrt(
            (n * sxx - sx**2) * (n * syy - sy**2)
        )
    return correlation


def explained_variance_contributions(
    template_correlations: NDArray[np.float64],
    match_correlations: NDArray[np.float64],
    *,
    control_correlations: NDArray[np.float64] | None = None,
    pair_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> ExplainedVarianceContributionResult:
    """Each pair's contribution to the explained variance, leaving it out.

    Oliva et al. (2020): "For assessing the relative contribution to the
    explained variance of a single pair of neurons, the EV was calculated for
    all the pairs without including the pair of interest. Subtracting this
    value from the total EV value, a relative contribution for the
    corresponding pair of neurons was calculated. To calculate the average
    contribution of each neuron to the EV, the relative contribution for all
    pairs in which a particular neuron participated was averaged."

    Parameters
    ----------
    template_correlations, match_correlations : ndarray, shape (n_pairs,)
        Pair correlations of the template and match periods from
        :func:`~neurospatial.reactivation.pairwise_correlations`, with the
        same units in the same column order.
    control_correlations : ndarray, shape (n_pairs,), optional
        Pair correlations of one control period; EV is then the squared
        partial correlation (Kudrimoti et al. 1999), else the squared plain
        correlation.
    pair_mask : array-like of bool, shape (n_pairs,), optional
        Pairs to use, in ``np.triu_indices(n_units, k=1)`` order.
    unit_ids : array-like, shape (n_units,), optional
        Label of each unit; default ``np.arange(n_units)``.

    Returns
    -------
    ExplainedVarianceContributionResult
        EV, each pair's contribution and each unit's mean contribution.

    Raises
    ------
    ValueError
        If the vectors have different lengths, ``n_pairs`` is not
        ``n_units * (n_units - 1) / 2``, ``pair_mask`` is not one boolean per
        pair, or fewer than 4 pairs are valid.

    Notes
    -----
    A pair that fits the template-match relation better than the others
    has a positive contribution; removing it lowers EV. Pairs share units,
    so the contributions of a unit's pairs are not independent.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reactivation import (
    ...     explained_variance_contributions,
    ...     pairwise_correlations,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> task = rng.poisson(1.0, (3000, 8)).astype(float)
    >>> task[:, :3] += rng.poisson(0.5, (3000, 1))
    >>> sleep = rng.poisson(1.0, (3000, 8)).astype(float)
    >>> sleep[:, :3] += rng.poisson(0.5, (3000, 1))
    >>> result = explained_variance_contributions(
    ...     pairwise_correlations(task), pairwise_correlations(sleep)
    ... )
    >>> int(np.argmax(result.unit_contributions)) in (0, 1, 2)
    True
    """
    call = "explained_variance_contributions()"
    template = np.asarray(template_correlations, dtype=np.float64)
    match = np.asarray(match_correlations, dtype=np.float64)
    vectors = [template, match]
    if control_correlations is not None:
        vectors.append(np.asarray(control_correlations, dtype=np.float64))
    n_pairs = template.size
    if any(vector.shape != (n_pairs,) for vector in vectors):
        raise ValueError(
            f"{call}: the correlation vectors must be 1-D with the same length, "
            f"got shapes {[vector.shape for vector in vectors]}."
        )
    n_units = _n_units_from_pairs(n_pairs)
    if n_units is None:
        raise ValueError(
            f"{call}: {n_pairs} pairs is not n_units * (n_units - 1) / 2 for any "
            "number of units; pass pairwise_correlations() output."
        )
    labels = resolve_unit_ids(unit_ids, n_units, context=call)
    valid = _selected_pairs(n_pairs, None, pair_mask, call=call)
    for vector in vectors:
        valid &= np.isfinite(vector)
    n_valid = int(valid.sum())
    if n_valid < 4:
        raise ValueError(
            f"{call}: {n_valid} valid pairs; leaving one out needs at least 4."
        )
    total = explained_variance_reactivation(
        template,
        match,
        control_correlations=control_correlations,
        pair_mask=valid,
    ).explained_variance
    t, m = template[valid], match[valid]
    r_tm = _leave_one_out_correlations(t, m)
    if control_correlations is None:
        without = r_tm**2
    else:
        c = vectors[2][valid]
        r_tc = _leave_one_out_correlations(t, c)
        r_mc = _leave_one_out_correlations(m, c)
        without = _one_control_ev(r_tm, r_tc, r_mc)
    pair_contributions = np.full(n_pairs, np.nan)
    pair_contributions[valid] = total - without
    first, second = np.triu_indices(n_units, k=1)
    # Each pair counts toward both of its units.
    units = np.concatenate([first[valid], second[valid]])
    sums = np.bincount(
        units, weights=np.tile(pair_contributions[valid], 2), minlength=n_units
    )
    counts = np.bincount(units, minlength=n_units)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_contributions = np.where(counts > 0, sums / counts, np.nan)
    return ExplainedVarianceContributionResult(
        explained_variance=float(total),
        pair_contributions=pair_contributions,
        unit_contributions=unit_contributions,
        unit_ids=labels,
        n_pairs=n_valid,
    )


def ev_rev_contrast(
    explained_variance: ArrayLike, reversed_explained_variance: ArrayLike
) -> Any:
    """The explained variance ratio (EV - REV) / (EV + REV).

    Karaba et al. (2024) report a "CA1 explained variance ratio
    [(EV−REV)/(EV+REV)]" in post-learning sleep: 1 when the template
    pattern appears only after the task, 0 when before and after alike, -1
    when only before.

    Parameters
    ----------
    explained_variance, reversed_explained_variance : float or array-like
        EV and REV, for example the columns of
        :func:`~neurospatial.reactivation.compare_reactivation_methods`.

    Returns
    -------
    float or ndarray
        The contrast, in [-1, 1]; NaN where EV + REV is 0 or either is NaN.

    Examples
    --------
    >>> from neurospatial.reactivation import ev_rev_contrast
    >>> ev_rev_contrast(0.75, 0.25)
    0.5
    """
    ev = np.asarray(explained_variance, dtype=np.float64)
    rev = np.asarray(reversed_explained_variance, dtype=np.float64)
    total = ev + rev
    with np.errstate(invalid="ignore", divide="ignore"):
        contrast = np.where(
            total > 0, (ev - rev) / np.where(total > 0, total, 1.0), np.nan
        )
    return float(contrast) if contrast.ndim == 0 else contrast
