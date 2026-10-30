"""Pairwise firing-order bias of spike sequences (Roth 2016; Drieu et al. 2018).

A sequence's order is summarized pair by pair, without place fields or
decoding. For cells ``i`` and ``j`` that both fire in an event, the bias
``B(i, j) = n(i before j) / (n_i n_j)`` is the share of their spike pairs in
which ``i`` fires first, and the skew bias ``S(i, j) = 2 B(i, j) - 1`` runs
from -1 (``j`` always first) to 1 (``i`` always first). Two sequences are
compared by the cosine between their skew-bias matrices over the pairs both
define. Liu et al. (2023, after Drieu, Todorova & Zugaro 2018) compared each
sleep SWR's matrix with the mean matrix of the theta cycles of the task, and
tested it against the event's spikes with their cell labels permuted.

:func:`pairwise_bias_matrix` computes an event's skew-bias matrix,
:func:`mean_pairwise_bias` the mean over many events (the template),
:func:`bias_similarity` the cosine between two matrices and
:func:`bias_similarity_significance` its shuffle test.
:class:`BiasSequenceMethod` with :func:`detect_bias_sequence_replay` applies
a published test to one candidate event, and
:func:`~neurospatial.replay.compare_replay_methods` applies it with the
other replay methods (``template_bias=``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.replay.significance import _ALTERNATIVES

if TYPE_CHECKING:
    from neurospatial.replay.methods import ReplayCriterion, ReplayDetectionResult

__all__ = [
    "BiasSequenceMethod",
    "BiasSimilarityResult",
    "PairwiseBiasResult",
    "bias_similarity",
    "bias_similarity_significance",
    "detect_bias_sequence_replay",
    "mean_pairwise_bias",
    "pairwise_bias_matrix",
]

# Tolerance of a skew-bias matrix's skew symmetry, S(i, j) = -S(j, i).
_SKEW_TOLERANCE = 1e-9


@dataclass(frozen=True, repr=False, eq=False)
class PairwiseBiasResult(ReadOnlyArraysMixin, ResultMixin):
    """Skew-bias matrix of the spikes in one event.

    Attributes
    ----------
    bias : NDArray[np.float64], shape (n_units, n_units)
        Skew bias ``S(i, j) = 2 n(i before j) / (n_i n_j) - 1``, in [-1, 1]:
        1 when every spike of unit ``i`` precedes every spike of unit ``j``,
        -1 for the reverse. Skew-symmetric (``S(j, i) = -S(i, j)``), with 0
        on the diagonal of units that fired. NaN where either unit is silent.
    n_spikes : NDArray[np.int64], shape (n_units,)
        Spikes of each unit in the event.
    unit_ids : NDArray, shape (n_units,)
        Label of each unit, in the order of ``spike_times``.

    See Also
    --------
    pairwise_bias_matrix : Function that produces this result.
    """

    bias: NDArray[np.float64]
    n_spikes: NDArray[np.int64]
    unit_ids: NDArray[Any]

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        upper = np.triu(np.isfinite(self.bias), 1)
        return {
            "n_units": int(self.n_spikes.size),
            "n_active_cells": int(np.sum(self.n_spikes > 0)),
            "n_pairs": int(np.sum(upper)),
        }


@dataclass(frozen=True, repr=False, eq=False)
class BiasSimilarityResult(ReadOnlyArraysMixin, ResultMixin):
    """Similarity of an event's firing order to a template, against shuffles.

    Attributes
    ----------
    similarity : float
        Cosine between the event's and the template's skew-bias matrices
        over the pairs both define, in [-1, 1]: 1 for the template's order,
        -1 for its reverse. NaN when no pair is defined in both, or when
        either matrix is zero on those pairs.
    p_value : float
        Monte Carlo p-value ``(k + 1) / (n + 1)`` over the ``n`` shuffles
        with a defined similarity, ``k`` of them at least as extreme as the
        event (on ``|similarity|`` with ``use_absolute``).
    null_similarities : NDArray[np.float64], shape (n_shuffles,)
        The signed similarity of each shuffle; NaN where undefined.
    n_pairs : int
        Pairs of units defined in both matrices.
    n_active_cells : int
        Units that fired in the event.
    use_absolute : bool
        Whether ``|similarity|`` was tested.
    alternative : {"greater", "less", "two-sided"}
        Tail of the test.

    See Also
    --------
    bias_similarity_significance : Function that produces this result.
    """

    similarity: float
    p_value: float
    null_similarities: NDArray[np.float64]
    n_pairs: int
    n_active_cells: int
    use_absolute: bool
    alternative: str

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "similarity": self.similarity,
            "p_value": self.p_value,
            "n_shuffles_used": int(np.sum(np.isfinite(self.null_similarities))),
            "n_pairs": self.n_pairs,
            "n_active_cells": self.n_active_cells,
            "use_absolute": self.use_absolute,
            "alternative": self.alternative,
        }


@dataclass(frozen=True)
class BiasSequenceMethod:
    """A published pairwise-bias sequence test (Liu et al. 2023).

    Each candidate event's skew-bias matrix is compared with a template's
    (:func:`bias_similarity`), and the similarity is tested against the
    event's spikes with their cell labels permuted
    (:func:`bias_similarity_significance`).

    Attributes
    ----------
    name : str
        Short name, for example ``"liu_2023_bias"``.
    reference : str
        The paper the method follows.
    n_shuffles : int
        Label permutations per event.
    alpha : float
        Significance level: the p-value must be below it.
    alternative : {"greater", "less", "two-sided"}
        Tail of the shuffle test.
    use_absolute : bool
        Test ``|similarity|``, so the template's order and its reverse both
        count.
    min_active_cells : int
        Fewest units that fire in the event, at least 2.
    notes : tuple of str
        Choices the paper does not specify, and analyses not applied.
    """

    name: str
    reference: str
    n_shuffles: int = 100
    alpha: float = 0.05
    alternative: Literal["greater", "less", "two-sided"] = "greater"
    use_absolute: bool = True
    min_active_cells: int = 2
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse settings that cannot run."""
        prefix = f"BiasSequenceMethod {self.name!r}"
        object.__setattr__(self, "notes", tuple(self.notes))
        _check_test(prefix, self.n_shuffles, self.alternative, self.use_absolute)
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float, np.integer, np.floating))
            or not 0 < self.alpha < 1
        ):
            raise ValueError(f"{prefix}: alpha must be in (0, 1), got {self.alpha!r}.")
        value = self.min_active_cells
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 2
        ):
            raise ValueError(
                f"{prefix}: min_active_cells must be an integer of at least 2, "
                f"got {value!r}; a bias compares the spikes of two cells."
            )

    @property
    def criteria(self) -> tuple[ReplayCriterion, ...]:
        """The active-cell threshold, as a criterion."""
        from neurospatial.replay.methods import ReplayCriterion

        return (ReplayCriterion("n_active_cells", ">=", self.min_active_cells),)


def pairwise_bias_matrix(
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    *,
    unit_ids: Any = None,
) -> PairwiseBiasResult:
    """Skew-bias matrix of the spikes in one event (Roth 2016).

    For units ``i`` and ``j`` that fire in the event, ``n(i before j)``
    counts the pairs of a spike of ``i`` and a spike of ``j`` in which the
    spike of ``i`` comes first, so the bias ``B(i, j) = n(i before j) /
    (n_i n_j)`` is in [0, 1] and the skew bias ``S(i, j) = 2 B(i, j) - 1``
    in [-1, 1] (Liu et al. 2023, after Drieu et al. 2018).

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per unit.
    interval : (start, stop)
        The event, in seconds; spikes at ``start <= t < stop`` belong to it.
    unit_ids : array-like or None, default=None
        Label of each unit, one per train; None numbers them from 0.

    Returns
    -------
    PairwiseBiasResult
        The skew-bias matrix, each unit's spike count and the unit labels.

    Raises
    ------
    ValueError
        If ``interval`` is not a finite (start, stop) with start < stop, or
        ``unit_ids`` does not have one label per unit.

    See Also
    --------
    mean_pairwise_bias : Mean matrix over many events, such as theta cycles.
    bias_similarity : Cosine between two matrices.

    Notes
    -----
    Liu et al. print the unwrapped matrix as ``2 B - 2``, which runs from -2
    to 0; their stated range, [-1, 1], and the lab's code
    (``neuro_py.ensemble.pairwise_bias_correlation.skew_bias_matrix``, after
    Roth 2016) give ``2 B - 1``, used here. Two spikes at the same time count
    half to each order, so ``S`` is exactly skew-symmetric; neuro_py counts
    such a pair as the lower-numbered unit first. Spike-time ties between
    units are rare at 20 kHz sampling.

    References
    ----------
    Roth, Z. (2016). Analysis of neuronal sequences using pairwise biases.
    arXiv:1603.02916.

    Drieu, C., Todorova, R., & Zugaro, M. (2018). Nested sequences of
    hippocampal assemblies during behavior support subsequent sleep
    replay. *Science*, 362(6415), 675-679.

    Liu, C., Todorova, R., Tang, W., Oliva, A., & Fernandez-Ruiz, A. (2023).
    Associative and predictive hippocampal codes support memory-guided
    behaviors. *Science*, 382(6668), eadi8237.

    Examples
    --------
    Unit 0 fires before unit 1 in all four of their spike pairs, and unit 2
    is silent:

    >>> import numpy as np
    >>> from neurospatial.replay import pairwise_bias_matrix
    >>> spikes = [np.array([0.01, 0.02]), np.array([0.03, 0.04]), np.array([])]
    >>> result = pairwise_bias_matrix(spikes, (0.0, 0.1))
    >>> result.bias
    array([[ 0.,  1., nan],
           [-1.,  0., nan],
           [nan, nan, nan]])
    """
    operation = "pairwise_bias_matrix()"
    times, labels, n_units = _event_spikes(spike_times, interval, operation)
    ids = _unit_ids(unit_ids, n_units, operation)
    bias, counts = _skew_bias(times, labels, n_units)
    return PairwiseBiasResult(bias=bias, n_spikes=counts, unit_ids=ids)


def mean_pairwise_bias(
    spike_times: Any, intervals: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Mean skew-bias matrix over many events, such as a task's theta cycles.

    Each pair's mean is over the events in which both units fired, so a
    unit absent from an event does not pull its pairs toward 0.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per unit.
    intervals : NDArray[np.float64], shape (n_events, 2)
        Each event's (start, stop), in seconds; spikes at ``start <= t <
        stop`` belong to it.

    Returns
    -------
    NDArray[np.float64], shape (n_units, n_units)
        Mean skew bias of each pair; NaN for a pair that never fired
        together in an event. Skew-symmetric.

    Raises
    ------
    ValueError
        If ``intervals`` is not an (n_events, 2) array of finite intervals
        with start < stop, or is empty.

    See Also
    --------
    pairwise_bias_matrix : One event's matrix.

    Notes
    -----
    Liu et al. (2023) "computed the mean bias matrix of all sequences taking
    place in theta cycles during the task" without saying how a cell absent
    from a cycle enters the mean; their statement that "the activity of
    cells that are not common to the two sequences is ignored" is read here
    as averaging each pair over the cycles where both cells fired. The lab's
    code (neuro_py ``shuffled_significance``) instead computes one matrix
    over all the task's theta spikes, with silent pairs set to 0.

    Examples
    --------
    Unit 0 fires before unit 1 in one cycle and after it in two:

    >>> import numpy as np
    >>> from neurospatial.replay import mean_pairwise_bias
    >>> spikes = [np.array([0.01, 0.13, 0.23]), np.array([0.02, 0.11, 0.21])]
    >>> cycles = np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]])
    >>> mean_pairwise_bias(spikes, cycles).round(3)
    array([[ 0.   , -0.333],
           [ 0.333,  0.   ]])
    """
    operation = "mean_pairwise_bias()"
    times, labels, n_units = _sorted_spikes(spike_times, operation)
    bounds = np.asarray(intervals, dtype=np.float64)
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or len(bounds) == 0
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError(
            _format_error(
                f"{operation}: intervals must be a non-empty (n_events, 2) array "
                f"of finite (start, stop) with start < stop, got shape "
                f"{bounds.shape}.",
                why="each row is one event, such as a theta cycle, whose spikes "
                "give one skew-bias matrix",
                fix="pass np.column_stack([starts, stops]) in seconds",
            )
        )
    total = np.zeros((n_units, n_units))
    count = np.zeros((n_units, n_units))
    edges = np.searchsorted(times, bounds, side="left")
    for first, last in edges:
        bias, _ = _skew_bias(times[first:last], labels[first:last], n_units)
        defined = np.isfinite(bias)
        total[defined] += bias[defined]
        count += defined
    mean: NDArray[np.float64] = np.divide(
        total, count, out=np.full_like(total, np.nan), where=count > 0
    )
    return mean


def bias_similarity(
    template_bias: NDArray[np.float64], event_bias: NDArray[np.float64]
) -> float:
    """Cosine between two skew-bias matrices over the pairs both define.

    Parameters
    ----------
    template_bias : NDArray[np.float64], shape (n_units, n_units)
        A skew-bias matrix, such as :func:`mean_pairwise_bias` of the task's
        theta cycles: values in [-1, 1], skew-symmetric, NaN for undefined
        pairs.
    event_bias : NDArray[np.float64], shape (n_units, n_units)
        Another, such as :func:`pairwise_bias_matrix` of a sleep event
        (its ``bias``), with the units in the same order.

    Returns
    -------
    float
        The cosine, in [-1, 1]: 1 for the same order, -1 for the reverse,
        near 0 for unrelated orders. NaN when no pair is defined in both,
        or either matrix is zero on those pairs.

    Raises
    ------
    ValueError
        If either matrix is not square, the two differ in shape, or a matrix
        has values outside [-1, 1] or is not skew-symmetric.

    See Also
    --------
    bias_similarity_significance : The cosine against label shuffles.

    Notes
    -----
    Liu et al. (2023) estimated "the correlation between two sequences ...
    as the cosine between their skew-bias matrices as each bias matrix is
    unwrapped into a vector", ignoring "the activity of cells that are not
    common to the two sequences". The cosine here runs over the pairs
    defined in both matrices (the pairs above the diagonal: the matrices
    are skew-symmetric, so the whole matrices give the same cosine). The
    lab's code (neuro_py ``cosine_similarity_matrices``) sets undefined
    pairs to 0 and keeps them in the vectors, which leaves the dot product
    unchanged but enlarges the norms, so its cosines are smaller in size.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.replay import bias_similarity
    >>> template = np.array([[0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, 0.0]])
    >>> bias_similarity(template, -template)
    -1.0
    """
    operation = "bias_similarity()"
    template = _validated_bias(template_bias, "template_bias", operation)
    event = _validated_bias(event_bias, "event_bias", operation)
    if template.shape != event.shape:
        raise ValueError(
            _format_error(
                f"{operation}: template_bias has shape {template.shape} but "
                f"event_bias has shape {event.shape}.",
                why="the two matrices are compared unit pair by unit pair",
                fix="compute both from spike trains of the same units, in the "
                "same order",
            )
        )
    return float(_cosines(template, event[np.newaxis])[0])


def bias_similarity_significance(
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    template_bias: NDArray[np.float64],
    *,
    n_shuffles: int = 100,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    use_absolute: bool = True,
    rng: np.random.Generator | int | None = None,
) -> BiasSimilarityResult:
    """Test an event's bias similarity to a template against label shuffles.

    Each shuffle permutes the unit labels of the event's spikes, so every
    unit keeps its number of spikes and the event keeps its spike times,
    and the skew-bias matrix and its similarity to the template are
    computed again (Liu et al. 2023: "shuffling the spikes sequence n = 100
    times").

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per unit, in the order of the
        template's rows.
    interval : (start, stop)
        The event, in seconds; spikes at ``start <= t < stop`` belong to it.
    template_bias : NDArray[np.float64], shape (n_units, n_units)
        The template's skew-bias matrix, such as :func:`mean_pairwise_bias`
        of the task's theta cycles.
    n_shuffles : int, default=100
        Label permutations (Liu et al. 2023: 100). A p-value is at least
        ``1 / (n_shuffles + 1)``.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        Tail of the test, on ``|similarity|`` with ``use_absolute``.
    use_absolute : bool, default=True
        Test ``|similarity|``, so the template's order and its reverse
        both count (Liu et al.: "correlations whose absolute value
        exceeded 5% of the shuffled distribution").
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed.

    Returns
    -------
    BiasSimilarityResult
        The similarity, the p-value, every shuffle's similarity, and the
        numbers of pairs and active units.

    Raises
    ------
    ValueError
        For inputs :func:`pairwise_bias_matrix` or :func:`bias_similarity`
        refuses; if ``spike_times`` has not one train per template row; if
        ``n_shuffles`` is not a positive integer, ``alternative`` not one of
        its choices, or ``use_absolute=True`` with an alternative other than
        "greater".

    See Also
    --------
    detect_bias_sequence_replay : The published test, with its criteria.

    Notes
    -----
    The p-value is ``(k + 1) / (n + 1)`` with ``k`` the shuffles whose
    ``|similarity|`` is at least the event's. With 100 shuffles, p < 0.05
    allows at most 4 such shuffles, where "exceeded 5% of the shuffled
    distribution" could allow 5. The cell set, and so the pairs compared,
    is the same in every shuffle.

    Examples
    --------
    Twelve units fire twice in the template's order. For units ``i < j``,
    the second spike of ``i`` comes after the first of ``j``, so 3 of their
    4 spike pairs are in order and every skew bias is 0.5, not 1; the
    order is still the template's, and the cosine is 1:

    >>> import numpy as np
    >>> from neurospatial.replay import bias_similarity_significance
    >>> template = np.sign(np.subtract.outer(np.arange(12), np.arange(12))).T * 1.0
    >>> spikes = [np.array([0.005 * i, 0.06 + 0.005 * i]) for i in range(12)]
    >>> result = bias_similarity_significance(spikes, (0.0, 0.2), template, rng=0)
    >>> round(result.similarity, 2), bool(result.p_value < 0.05)
    (1.0, True)
    """
    operation = "bias_similarity_significance()"
    _check_test(operation, n_shuffles, alternative, use_absolute)
    template = _validated_bias(template_bias, "template_bias", operation)
    times, labels, n_units = _event_spikes(spike_times, interval, operation)
    if n_units != len(template):
        raise ValueError(
            _format_error(
                f"{operation}: spike_times has {n_units} units but template_bias "
                f"has {len(template)} rows.",
                why="each unit's spikes are compared with its row of the template",
                fix="pass spike times of the template's units, in its order",
            )
        )
    return _significance(
        times,
        labels,
        template,
        n_shuffles=int(n_shuffles),
        alternative=alternative,
        use_absolute=bool(use_absolute),
        rng=rng,
    )


def detect_bias_sequence_replay(
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    template_bias: NDArray[np.float64],
    *,
    bias_sequence_method: BiasSequenceMethod,
    rng: np.random.Generator | int | None = None,
) -> ReplayDetectionResult:
    """Apply a published pairwise-bias sequence test to one candidate event.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per unit, in the order of the
        template's rows.
    interval : (start, stop)
        The candidate event, in seconds.
    template_bias : NDArray[np.float64], shape (n_units, n_units)
        The template's skew-bias matrix, such as :func:`mean_pairwise_bias`
        of the task's theta cycles.
    bias_sequence_method : BiasSequenceMethod
        The method, for example ``neurospatial.replay.methods.LIU_2023_BIAS``.
    rng : np.random.Generator, int, or None
        Random number generator or seed for the shuffles.

    Returns
    -------
    ReplayDetectionResult
        ``score`` is the signed similarity, ``p_value`` its shuffle test,
        ``criteria`` the active-unit rule (over every unit given), and
        ``n_shuffles_used`` the shuffles with a defined similarity. The
        verdict is False when the similarity is undefined (no pair active
        in both).

    Raises
    ------
    ValueError
        For inputs :func:`bias_similarity_significance` refuses.

    See Also
    --------
    compare_replay_methods : Several methods on several events.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.replay import detect_bias_sequence_replay
    >>> from neurospatial.replay.methods import LIU_2023_BIAS
    >>> template = np.sign(np.subtract.outer(np.arange(12), np.arange(12))).T * 1.0
    >>> reverse = [np.array([0.06 - 0.005 * i]) for i in range(12)]
    >>> result = detect_bias_sequence_replay(
    ...     reverse, (0.0, 0.1), template, bias_sequence_method=LIU_2023_BIAS, rng=0
    ... )
    >>> result.is_replay, result.score
    (True, -1.0)
    """
    from neurospatial.replay.methods import ReplayDetectionResult
    from neurospatial.stats.shuffle import compute_shuffle_zscore

    method = bias_sequence_method
    tested = bias_similarity_significance(
        spike_times,
        interval,
        template_bias,
        n_shuffles=method.n_shuffles,
        alternative=method.alternative,
        use_absolute=method.use_absolute,
        rng=rng,
    )
    n_active = float(tested.n_active_cells)
    criterion = method.criteria[0]
    passes = {criterion.label: criterion.evaluate(n_active, np.nan)}
    null = tested.null_similarities
    defined = null[np.isfinite(null)]
    z_score = compute_shuffle_zscore(tested.similarity, defined)
    return ReplayDetectionResult(
        method=method.name,
        is_replay=bool(tested.p_value < method.alpha) and all(passes.values()),
        score=tested.similarity,
        p_value=tested.p_value,
        criteria=passes,
        values={criterion.label: n_active},
        n_shuffles_used=int(defined.size),
        z_scores={"spike_label_permutation": z_score},
    )


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _check_test(
    prefix: str, n_shuffles: Any, alternative: Any, use_absolute: Any
) -> None:
    if (
        isinstance(n_shuffles, bool)
        or not isinstance(n_shuffles, (int, np.integer))
        or n_shuffles < 1
    ):
        raise ValueError(
            f"{prefix}: n_shuffles must be a positive integer, got "
            f"{n_shuffles!r}. Liu et al. (2023) used 100."
        )
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"{prefix}: alternative must be one of {list(_ALTERNATIVES)}, got "
            f"{alternative!r}."
        )
    if not isinstance(use_absolute, (bool, np.bool_)):
        raise TypeError(f"{prefix}: use_absolute must be a bool, got {use_absolute!r}.")
    if use_absolute and alternative != "greater":
        raise ValueError(
            _format_error(
                f"{prefix}: use_absolute=True needs alternative='greater', got "
                f"{alternative!r}.",
                why="an absolute similarity is extreme only when large",
                fix="use alternative='greater', or use_absolute=False to test "
                "the signed similarity in either tail",
            )
        )


def _interval(interval: Any, operation: str) -> tuple[float, float]:
    from neurospatial.replay.rank_order import _validated_interval

    return _validated_interval(interval, operation)


def _unit_ids(unit_ids: Any, n_units: int, operation: str) -> NDArray[Any]:
    if unit_ids is None:
        return np.arange(n_units)
    ids = np.asarray(unit_ids)
    if ids.shape != (n_units,):
        raise ValueError(
            f"{operation}: unit_ids has shape {ids.shape} but spike_times has "
            f"{n_units} units; pass one label per spike train."
        )
    return ids


def _sorted_spikes(
    spike_times: Any, operation: str, interval: tuple[float, float] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.intp], int]:
    """Every spike's time and unit, in time order, and the number of units.

    With ``interval``, only the spikes at ``start <= t < stop``.
    """
    from neurospatial._spikes import as_spike_trains
    from neurospatial.replay.rank_order import _pooled_spikes, _spikes_in

    trains = as_spike_trains(spike_times)
    if interval is not None:
        trains = _spikes_in(trains, *interval)
    times, labels = _pooled_spikes(trains, np.arange(len(trains)), "all_spikes")
    if not np.isfinite(times).all():
        raise ValueError(
            f"{operation}: spike_times holds NaN or Inf; pass finite spike "
            "times in seconds."
        )
    order = np.argsort(times, kind="stable")
    return times[order], labels[order], len(trains)


def _event_spikes(
    spike_times: Any, interval: Any, operation: str
) -> tuple[NDArray[np.float64], NDArray[np.intp], int]:
    """The event's spikes, in time order, with their units."""
    return _sorted_spikes(spike_times, operation, _interval(interval, operation))


def _skew_bias(
    times: NDArray[np.float64], labels: NDArray[np.intp], n_units: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Skew-bias matrix of sorted spikes, and each unit's spike count.

    ``N[i, j]`` sums, over the spikes ``b`` of unit ``j``, the spikes of
    unit ``i`` before ``b``, with spikes at the same time counted half.
    """
    one_hot = np.zeros((times.size, n_units))
    one_hot[np.arange(times.size), labels] = 1.0
    cumulative = np.vstack([np.zeros(n_units), np.cumsum(one_hot, axis=0)])
    left = np.searchsorted(times, times, side="left")
    right = np.searchsorted(times, times, side="right")
    before = (cumulative[left] + cumulative[right]) / 2.0  # (n_spikes, n_units)
    pairs = before.T @ one_hot
    counts = one_hot.sum(axis=0)
    both = np.outer(counts, counts)
    bias: NDArray[np.float64] = np.divide(
        2.0 * pairs, both, out=np.full_like(both, np.nan), where=both > 0
    )
    bias -= 1.0
    return bias, counts.astype(np.int64)


def _validated_bias(matrix: Any, name: str, operation: str) -> NDArray[np.float64]:
    values = np.asarray(matrix, dtype=np.float64)
    problem = None
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        problem = f"has shape {values.shape}, not (n_units, n_units)"
    elif np.any(np.isinf(values)) or np.any(np.abs(values[np.isfinite(values)]) > 1):
        problem = "has values outside [-1, 1]"
    elif not np.array_equal(np.isnan(values), np.isnan(values.T)) or np.any(
        np.abs(np.nan_to_num(values + values.T)) > _SKEW_TOLERANCE
    ):
        problem = "is not skew-symmetric (S[j, i] = -S[i, j])"
    if problem is not None:
        raise ValueError(
            _format_error(
                f"{operation}: {name} {problem}.",
                why="a skew-bias matrix holds S(i, j) = 2 B(i, j) - 1 for each "
                "pair, with S(j, i) = -S(i, j), and NaN for undefined pairs",
                fix="compute it with pairwise_bias_matrix(...).bias or "
                "mean_pairwise_bias(...)",
            )
        )
    return values


def _cosines(
    template: NDArray[np.float64], events: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Cosine of each event matrix with the template over the pairs both define.

    ``events`` has shape (n_events, n_units, n_units). Rounded so that
    equal cosines summed in different orders compare equal.
    """
    from neurospatial.replay.trajectory import _SCORE_DECIMALS

    upper = np.triu(np.ones(template.shape, dtype=bool), 1)
    pairs = upper & np.isfinite(template) & np.isfinite(events)
    a = np.where(pairs, template, 0.0)
    b = np.where(pairs, events, 0.0)
    dot = (a * b).sum(axis=(1, 2))
    norms = np.sqrt((a * a).sum(axis=(1, 2)) * (b * b).sum(axis=(1, 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = dot / norms
    rounded = np.clip(np.round(cosine, _SCORE_DECIMALS), -1.0, 1.0)
    return np.where(norms > 0, rounded, np.nan)


def _significance(
    times: NDArray[np.float64],
    labels: NDArray[np.intp],
    template: NDArray[np.float64],
    *,
    n_shuffles: int,
    alternative: Literal["greater", "less", "two-sided"],
    use_absolute: bool,
    rng: np.random.Generator | int | None,
) -> BiasSimilarityResult:
    from neurospatial.stats import _ensure_rng
    from neurospatial.stats.shuffle import compute_shuffle_pvalue

    n_units = len(template)
    bias, counts = _skew_bias(times, labels, n_units)
    observed = float(_cosines(template, bias[np.newaxis])[0])
    generator = _ensure_rng(rng)
    shuffled = np.stack(
        [
            _skew_bias(times, generator.permutation(labels), n_units)[0]
            for _ in range(n_shuffles)
        ]
    )
    null = _cosines(template, shuffled)
    defined = null[np.isfinite(null)]
    if np.isfinite(observed) and defined.size:
        statistic = np.abs if use_absolute else np.asarray
        p_value = compute_shuffle_pvalue(
            float(statistic(observed)), statistic(defined), tail=alternative
        )
    else:
        p_value = np.nan
    upper = np.triu(np.isfinite(template) & np.isfinite(bias), 1)
    return BiasSimilarityResult(
        similarity=observed,
        p_value=float(p_value),
        null_similarities=null,
        n_pairs=int(upper.sum()),
        n_active_cells=int(np.sum(counts > 0)),
        use_absolute=use_absolute,
        alternative=alternative,
    )
