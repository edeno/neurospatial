"""Published template-matching replay tests: Lee & Wilson (2002), Ji & Wilson (2007).

:class:`TemplateMatchMethod` records one paper's test of firing order
against the order of the cells' fields, which needs spikes and rate maps
but no decoding. :func:`detect_template_match_replay` gives the verdict on
one candidate event, and :func:`~neurospatial.replay.compare_replay_methods`
compares these methods with the decoded and rank-order methods. The papers'
claims rest on data-set tests, which :func:`template_match_dataset_test`
runs: Lee & Wilson's binomial counts of matching pairs, triplets and
low-probability words, and Ji & Wilson's chance and template-shuffle
significance of the number of replaying frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial.replay.sequence_match import (
    _FIELD_ORDERS,
    _MATCH_ORDERS,
    _MAX_EXACT_LETTERS,
    _active_firing_times,
    _check_seconds,
    _dense,
    _event_letters,
    _exact_distribution,
    _firing_order,
    _index_test_size,
    _inversion_p_values,
    _is_count,
    _KeyDistribution,
    _match_keys,
    _matching_index,
    _opposite_pairs,
    _passes,
    _perfect_key,
    _random_orderings,
    _sampled_distribution,
    _sequence_match,
    _sorted_trains,
    _template_ranks,
    _trial_group,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.environment import Environment
    from neurospatial.replay.methods import ReplayCriterion, ReplayDetectionResult

__all__ = [
    "JI_WILSON_2007",
    "LEE_WILSON_2002",
    "TemplateMatchDatasetResult",
    "TemplateMatchMethod",
    "detect_template_match_replay",
    "template_match_dataset_test",
]

_STATISTICS = ("best_match", "matching_index")
# Lee & Wilson's trial groups and the chance probability of a match in each;
# the low-probability group's probability is the method's trial_probability.
_PAIR, _TRIPLET, _LOW = "pair", "triplet", "low_probability"
_FRAMES = "candidate_frames"


@dataclass(frozen=True)
class TemplateMatchMethod:
    """A published template-matching replay test of firing order.

    The template is the order of the cells' fields along the track (one
    template per running direction). An event's score needs no decoding:

    - ``statistic="best_match"`` (Lee & Wilson 2002): the event's letters
      (bursts of each template cell) form a word, scored by its best
      ``(x, y)`` match to the template; the p-value is the share of the
      orderings of its letters with a match at least as good
      (:func:`~neurospatial.replay.best_sequence_match`). The score is
      ``x``, the letters matched in order.
    - ``statistic="matching_index"`` (Ji & Wilson 2007): one firing time per
      active cell, the matching index ``I`` of their order, and its exact
      p-value over the orders of the active cells
      (:func:`~neurospatial.replay.sequence_matching_index`). The score is
      ``I``.

    Both p-values are exact probabilities, not shuffle counts, and take a
    few discrete values for a given number of cells.

    Attributes
    ----------
    name : str
        Short name, for example ``"ji_wilson_2007"``.
    reference : str
        The paper the method follows.
    statistic : {"best_match", "matching_index"}
        The score and its p-value, as above.
    field_order : {"peak", "center"}
        Each cell's place on the track: the peak of its rate map or its
        rate-weighted center.
    min_peak_rate : float or None
        Template cells: cells whose rate map peaks at this rate or more, in
        Hz. None makes every cell with a field a template cell.
    min_template_cells : int or None
        Fewest cells in a template; a smaller template is refused (Ji &
        Wilson: 5). None for no limit.
    max_isi : float or None
        ``"best_match"``: burst interval in seconds; a cell's spikes closer
        than this are one letter (Lee & Wilson: 0.05).
    max_gap : float or None
        ``"best_match"``: the gap in seconds above which Lee & Wilson ended
        a word (0.1). Not used to score an event, which is one word; pass
        it to :func:`~neurospatial.replay.spike_words` to cut the events.
    match_order : {"interruptions_first", "length_first"}
        ``"best_match"``: how matches are ranked (see
        :func:`~neurospatial.replay.best_sequence_match`).
    trial_probability : float or None
        ``"best_match"``: ``P`` of the low-probability trials (Lee & Wilson:
        1/24).
    exact_max_letters : int
        ``"best_match"``: words with up to this many letters enumerate every
        ordering; longer words sample ``n_samples`` orderings. At most 10.
    n_samples : int
        ``"best_match"``: random orderings for longer words.
    bandwidth : float or None
        ``"matching_index"``: kernel standard deviation in seconds for each
        cell's firing time (Ji & Wilson: 0.18 for hippocampus, 0.4 for
        cortex).
    min_active_cells : int
        Fewest template cells that fired in the event (for
        ``"best_match"``, distinct letters); with fewer, the score and
        p-value are NaN and the event is not replay. At least 2.
    alpha : float
        Significance level.
    p_comparison : {"<", "<="}
        How the p-value is compared with ``alpha``: Lee & Wilson count a
        word with p equal to 1/24 (a perfect four-letter word) as a match,
        Ji & Wilson require p < 0.05. Equality is judged within round-off.
    template_choice : {"largest_score"} or None
        With several templates and no ``template=``: test the event against
        the template with the largest score among those on which enough
        cells fired, then the smaller p-value, then the first (Ji & Wilson
        assigned a frame to the template with the higher matching index).
        None needs ``template=``.
    n_shuffles : int
        Template shuffles of :func:`template_match_dataset_test`.
    notes : tuple of str
        Choices the paper does not specify, and rules not applied.
    """

    name: str
    reference: str
    statistic: Literal["best_match", "matching_index"]
    field_order: Literal["peak", "center"] = "peak"
    min_peak_rate: float | None = None
    min_template_cells: int | None = None
    max_isi: float | None = None
    max_gap: float | None = None
    match_order: Literal["interruptions_first", "length_first"] = "interruptions_first"
    trial_probability: float | None = None
    exact_max_letters: int = 10
    n_samples: int = 100_000
    bandwidth: float | None = None
    min_active_cells: int = 2
    alpha: float = 0.05
    p_comparison: Literal["<", "<="] = "<"
    template_choice: Literal["largest_score"] | None = None
    n_shuffles: int = 1000
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse settings that cannot run."""
        object.__setattr__(self, "notes", tuple(self.notes))
        prefix = f"TemplateMatchMethod {self.name!r}"
        choices: dict[str, tuple[Any, ...]] = {
            "statistic": _STATISTICS,
            "field_order": _FIELD_ORDERS,
            "match_order": _MATCH_ORDERS,
            "p_comparison": ("<", "<="),
            "template_choice": (None, "largest_score"),
        }
        for field_name, allowed in choices.items():
            if getattr(self, field_name) not in allowed:
                raise ValueError(
                    f"{prefix}: {field_name} must be one of {list(allowed)}, got "
                    f"{getattr(self, field_name)!r}."
                )
        own = {
            "best_match": ("max_isi", "max_gap", "trial_probability"),
            "matching_index": ("bandwidth",),
        }
        needed = own[self.statistic]
        other = own[
            "matching_index" if self.statistic == "best_match" else "best_match"
        ]
        missing = [name for name in needed if getattr(self, name) is None]
        extra = [name for name in other if getattr(self, name) is not None]
        if missing or extra:
            raise ValueError(
                _format_error(
                    f"{prefix}: statistic={self.statistic!r} needs "
                    f"{list(needed)} and takes no {list(other)}; missing "
                    f"{missing}, given {extra}.",
                    why="Lee & Wilson's best match cuts bursts and words, and Ji "
                    "& Wilson's index smooths each cell's spikes",
                    fix="set the options of the chosen statistic only",
                )
            )
        operation = prefix
        if self.statistic == "best_match":
            _check_seconds(self.max_isi, "max_isi", operation, allow_zero=True)
            _check_seconds(self.max_gap, "max_gap", operation, allow_zero=False)
            assert self.max_isi is not None and self.max_gap is not None
            if self.max_isi > self.max_gap:
                raise ValueError(
                    f"{prefix}: max_isi ({self.max_isi:g} s) must be at most "
                    f"max_gap ({self.max_gap:g} s), as in Lee & Wilson (2002)."
                )
            probability = self.trial_probability
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float, np.integer, np.floating))
                or not 0 < probability < 1
            ):
                raise ValueError(
                    f"{prefix}: trial_probability must be in (0, 1), got "
                    f"{probability!r}."
                )
        else:
            _check_seconds(self.bandwidth, "bandwidth", operation, allow_zero=False)
        counts: tuple[tuple[str, int, int | None], ...] = (
            ("min_active_cells", 2, None),
            ("n_shuffles", 1, None),
            ("n_samples", 1, None),
            ("exact_max_letters", 1, _MAX_EXACT_LETTERS),
        )
        for field_name, minimum, maximum in counts:
            value = getattr(self, field_name)
            if not _is_count(value, minimum, maximum):
                bound = (
                    f"from {minimum} to {maximum}"
                    if maximum
                    else f"of at least {minimum}"
                )
                raise ValueError(
                    f"{prefix}: {field_name} must be an integer {bound}, got {value!r}."
                )
        if self.min_template_cells is not None and not _is_count(
            self.min_template_cells, 2
        ):
            raise ValueError(
                f"{prefix}: min_template_cells must be an integer of at least 2 or "
                f"None, got {self.min_template_cells!r}."
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"{prefix}: alpha must be in (0, 1), got {self.alpha!r}.")
        rate = self.min_peak_rate
        if rate is not None and (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float, np.integer, np.floating))
            or not np.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError(
                f"{prefix}: min_peak_rate must be a positive rate in Hz or None, "
                f"got {rate!r}."
            )

    @property
    def criteria(self) -> tuple[ReplayCriterion, ...]:
        """The active-cell threshold, as a criterion."""
        from neurospatial.replay.methods import ReplayCriterion

        return (ReplayCriterion("n_active_cells", ">=", self.min_active_cells),)


def detect_template_match_replay(
    env: Environment | Mapping[str, Environment],
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: Any,
    *,
    template_match_method: TemplateMatchMethod,
    template: str | None = None,
    rng: np.random.Generator | int | None = None,
) -> ReplayDetectionResult:
    """Apply one published template-matching method to one candidate event.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment of the rate maps: a 1-D environment or a linearized
        track; or one environment per template.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the order of the rows
        of ``encoding_models``.
    interval : (start, stop)
        The candidate event in seconds (Lee & Wilson's word, Ji & Wilson's
        frame). Spikes at ``start <= t < stop`` count.
    encoding_models : array of shape (n_cells, n_bins), SpatialRatesResult, or Mapping
        Each cell's rate map on ``env``, in Hz; or a mapping of such rate
        maps, one per template (such as each running direction), with the
        cells in the same order. A cell with a NaN rate map is not in that
        template.
    template_match_method : TemplateMatchMethod
        The method, for example
        ``neurospatial.replay.methods.JI_WILSON_2007``.
    template : str or None, default=None
        With a mapping of rate maps, the template to test; None lets the
        method's ``template_choice`` choose.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed for the sampled orderings of words
        longer than the method's ``exact_max_letters``; otherwise unused.

    Returns
    -------
    ReplayDetectionResult
        The verdict, the score (letters matched in order, or the matching
        index), the p-value, the active-cell criterion and, with several
        templates, the tested ``template``. With fewer active template
        cells than the method's ``min_active_cells`` the score and p-value
        are NaN and the event is not replay. ``n_shuffles_used`` is None:
        the p-values are exact.

    Raises
    ------
    ValueError
        If a mapping of rate maps is given without ``template=`` to a
        method without a ``template_choice``, or ``template`` is not one of
        its keys; if a template has fewer cells than the method's
        ``min_template_cells``, or two of its cells have their field at the
        same position; or for inputs
        :func:`~neurospatial.replay.rank_order_correlation` refuses.

    See Also
    --------
    neurospatial.replay.best_sequence_match : Lee & Wilson's score.
    neurospatial.replay.sequence_matching_index : Ji & Wilson's score.
    template_match_dataset_test : The papers' data-set tests.
    neurospatial.replay.compare_replay_methods : Several methods on several
        events.

    Examples
    --------
    Six cells fire in field order within 300 ms:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import detect_template_match_replay
    >>> from neurospatial.replay.methods import JI_WILSON_2007
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> rates = 20.0 * np.exp(
    ...     -0.5 * ((x - np.linspace(15.0, 105.0, 6)[:, None]) / 8.0) ** 2
    ... )
    >>> spikes = [np.array([1.0 + 0.05 * i]) for i in range(6)]
    >>> result = detect_template_match_replay(
    ...     env, spikes, (1.0, 1.3), rates, template_match_method=JI_WILSON_2007
    ... )
    >>> result.is_replay, result.score, round(result.p_value * 720)
    (True, 1.0, 1)
    """
    from neurospatial._spikes import as_spike_trains
    from neurospatial.replay.rank_order import _validated_interval

    operation = "detect_template_match_replay()"
    method = template_match_method
    names = _template_names(encoding_models, template, method, operation)
    start, stop = _validated_interval(interval, operation)
    trains = _sorted_trains(as_spike_trains(spike_times))
    generator = np.random.default_rng(rng)
    seeds = generator.integers(2**63, size=len(names))
    results = []
    for name, seed in zip(names, seeds, strict=True):
        ranks = _method_ranks(
            method, env, encoding_models, name, len(trains), operation
        )
        results.append(
            _verdict(
                method, ranks, trains, start, stop, name, np.random.default_rng(seed)
            )
        )
    if len(results) == 1:
        return results[0]
    return _chosen(results, method)


def _template_names(
    encoding_models: Any,
    template: str | None,
    method: TemplateMatchMethod,
    operation: str,
) -> list[str | None]:
    if not isinstance(encoding_models, Mapping):
        return [None]
    if template is not None:
        if template not in encoding_models:
            raise ValueError(
                f"{operation}: template {template!r} is not one of "
                f"{list(encoding_models)}."
            )
        return [template]
    if method.template_choice is None:
        raise ValueError(
            _format_error(
                f"{operation}: method {method.name!r} needs template= with several "
                "rate-map sets.",
                why="it tests one template and has no rule to choose one",
                fix=f"pass template= one of {list(encoding_models)}",
            )
        )
    return list(encoding_models)


def _method_ranks(
    method: TemplateMatchMethod,
    env: Any,
    encoding_models: Any,
    name: str | None,
    n_cells: int,
    operation: str,
) -> NDArray[np.intp]:
    """One template's ranks, checked against the method and the spike trains."""
    if isinstance(env, Mapping):
        if name is None:
            raise ValueError(
                f"{operation}: a mapping of environments needs a mapping of rate "
                "maps with the same templates."
            )
        if name not in env:
            raise ValueError(
                f"{operation}: template {name!r} has rate maps but no environment "
                f"in env, whose templates are {list(env)}."
            )
        env = env[name]
    models = encoding_models if name is None else encoding_models[name]
    ranks = _template_ranks(
        env,
        models,
        field_order=method.field_order,
        min_peak_rate=method.min_peak_rate,
        operation=operation,
    )
    if ranks.size != n_cells:
        raise ValueError(
            _format_error(
                f"{operation}: spike_times has {n_cells} cells but encoding_models "
                f"has {ranks.size} rate maps.",
                why="each cell's firing is compared with its own place field",
                fix="pass spike times and rate maps for the same cells, in the "
                "same order",
            )
        )
    n_template = int(np.sum(ranks >= 0))
    if method.min_template_cells is not None and n_template < method.min_template_cells:
        label = "the template" if name is None else f"template {name!r}"
        raise ValueError(
            _format_error(
                f"{operation}: {label} has {n_template} cells, fewer than method "
                f"{method.name!r} allows ({method.min_template_cells}).",
                why="the method's paper analyzed only templates with at least "
                f"{method.min_template_cells} cells",
                fix="leave this template out, or record more cells",
            )
        )
    return ranks


def _verdict(
    method: TemplateMatchMethod,
    ranks: NDArray[np.intp],
    trains: list[NDArray[np.float64]],
    start: float,
    stop: float,
    template: str | None,
    rng: np.random.Generator,
) -> ReplayDetectionResult:
    """One template's verdict on one event."""
    from neurospatial.replay.methods import ReplayDetectionResult

    if method.statistic == "best_match":
        assert method.max_isi is not None and method.trial_probability is not None
        match = _sequence_match(
            ranks,
            trains,
            start,
            stop,
            max_isi=method.max_isi,
            match_order=method.match_order,
            trial_probability=method.trial_probability,
            exact_max_letters=method.exact_max_letters,
            n_samples=method.n_samples,
            rng=rng,
        )
        n_active = match.n_distinct_letters
        score, p_value = float(match.matched_letters), match.match_probability
        if n_active < method.min_active_cells:
            score, p_value = np.nan, np.nan
    else:
        assert method.bandwidth is not None
        index = _matching_index(
            ranks,
            trains,
            start,
            stop,
            bandwidth=method.bandwidth,
            min_active_cells=method.min_active_cells,
        )
        n_active = index.n_active_cells
        score, p_value = index.score, index.match_probability
    criterion = method.criteria[0]
    passes = criterion.evaluate(float(n_active), np.nan)
    return ReplayDetectionResult(
        method=method.name,
        is_replay=bool(passes and _passes(p_value, method.alpha, method.p_comparison)),
        score=float(score),
        p_value=float(p_value),
        criteria={criterion.label: passes},
        values={criterion.label: float(n_active)},
        template=template,
        n_shuffles_used=None,
    )


def _chosen(
    results: list[ReplayDetectionResult], method: TemplateMatchMethod
) -> ReplayDetectionResult:
    """The template with the largest score, then the smaller p, then the first."""
    scored = [result for result in results if np.isfinite(result.score)]
    if scored:
        return max(scored, key=lambda result: (result.score, -result.p_value))
    label = method.criteria[0].label
    return max(results, key=lambda result: result.values[label])


# --------------------------------------------------------------------------
# Data-set tests
# --------------------------------------------------------------------------


@dataclass(frozen=True, repr=False, eq=False)
class TemplateMatchDatasetResult(ReadOnlyArraysMixin, ResultMixin):
    """A template-matching method's data-set test: counts of matches against chance.

    Each row counts the trials of one group in one template, or in every
    template together (``template`` None): Lee & Wilson's pairs, triplets
    and low-probability words, or Ji & Wilson's candidate frames.

    Attributes
    ----------
    method : str
        The method's name.
    statistic : {"best_match", "matching_index"}
        The method's statistic.
    row_templates : tuple of (str or None)
        Each row's template; None for the rows of every template together.
    row_groups : tuple of str
        Each row's group: ``"pair"``, ``"triplet"``, ``"low_probability"``
        or ``"candidate_frames"``.
    n_trials : NDArray[np.int64], shape (n_rows,)
        Trials: words of the group, or candidate frames (at least
        ``min_active_cells`` active cells, assigned to one template).
    n_matches : NDArray[np.int64], shape (n_rows,)
        Trials that match: p at most the group's probability (pairs in
        template order, triplets in order, low-probability words with
        ``p <= trial_probability``), or replaying frames (the method's
        verdict).
    probability : NDArray[np.float64], shape (n_rows,)
        The chance probability of a match per trial: 1/2, 1/6 or the
        ``trial_probability``; NaN for frames, whose probability depends on
        their number of cells.
    expected : NDArray[np.float64], shape (n_rows,)
        Expected matches under chance: ``n_trials * probability``, or for
        frames the sum of each frame's exact test size (the largest p-value
        at its number of cells that passes alpha).
    z_score : NDArray[np.float64], shape (n_rows,)
        ``(n_matches - expected) / sd``: the binomial sd
        ``sqrt(N P (1 - P))`` (Lee & Wilson; a lower bound for the
        low-probability words), or the Poisson sd ``sqrt(expected)`` (Ji &
        Wilson). NaN without trials.
    z_score_upper : NDArray[np.float64], shape (n_rows,)
        Lee & Wilson's upper-bound z: the expected count and variance from
        each word's attainable size (the largest probability of a match
        that is at most ``trial_probability`` for its letters), which can
        be well below ``trial_probability``. Equal to ``z_score`` for pairs
        and triplets; NaN for frames.
    p_value : NDArray[np.float64], shape (n_rows,)
        One-sided p-value as the paper computed it: the normal
        approximation for pairs and frames, the exact binomial for
        low-probability words and (unstated in the paper) triplets.
    p_value_exact : NDArray[np.float64], shape (n_rows,)
        The exact one-sided p-value: binomial for word groups, the sum of
        independent Bernoulli trials with each frame's test size for frames.
    shuffle_group : str
        The group the template shuffles test: ``"low_probability"`` or
        ``"candidate_frames"``.
    observed_statistic : float
        The shuffled statistic for the real templates: the pooled
        low-probability ``z_score`` (Lee & Wilson) or the pooled number of
        replaying frames (Ji & Wilson).
    shuffle_statistics : NDArray[np.float64], shape (n_shuffles,)
        The statistic with every template's cell order permuted.
    shuffle_p_value : float
        ``(k + 1) / (n + 1)``, where ``k`` shuffles reach the observed
        statistic.
    shuffle_z_score : float
        The observed statistic in standard deviations of the shuffles (Lee
        & Wilson's 7.9).

    See Also
    --------
    template_match_dataset_test : Function that produces this result.
    """

    method: str
    statistic: str
    row_templates: tuple[str | None, ...]
    row_groups: tuple[str, ...]
    n_trials: NDArray[np.int64]
    n_matches: NDArray[np.int64]
    probability: NDArray[np.float64]
    expected: NDArray[np.float64]
    z_score: NDArray[np.float64]
    z_score_upper: NDArray[np.float64]
    p_value: NDArray[np.float64]
    p_value_exact: NDArray[np.float64]
    shuffle_group: str
    observed_statistic: float
    shuffle_statistics: NDArray[np.float64]
    shuffle_p_value: float
    shuffle_z_score: float

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        object.__setattr__(self, "row_templates", tuple(self.row_templates))
        object.__setattr__(self, "row_groups", tuple(self.row_groups))
        self._own_arrays()

    def groups_table(self) -> pd.DataFrame:
        """One row per (template, group): trials, matches and their tests.

        Columns ``template`` (None for every template together), ``group``,
        ``n_trials``, ``n_matches``, ``match_ratio``, ``probability``,
        ``expected``, ``z_score``, ``z_score_upper``, ``p_value`` and
        ``p_value_exact``; ``attrs`` holds ``method`` and ``statistic``.
        """
        import pandas as pd

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = self.n_matches / self.n_trials
        table = pd.DataFrame(
            {
                "template": list(self.row_templates),
                "group": list(self.row_groups),
                "n_trials": self.n_trials,
                "n_matches": self.n_matches,
                "match_ratio": np.where(self.n_trials > 0, ratio, np.nan),
                "probability": self.probability,
                "expected": self.expected,
                "z_score": self.z_score,
                "z_score_upper": self.z_score_upper,
                "p_value": self.p_value,
                "p_value_exact": self.p_value_exact,
            }
        )
        table.attrs.update(method=self.method, statistic=self.statistic)
        return table

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result.

        The pooled row of the shuffled group, and the shuffle test.
        """
        row = next(
            index
            for index, (template, group) in enumerate(
                zip(self.row_templates, self.row_groups, strict=True)
            )
            if template is None and group == self.shuffle_group
        )
        return {
            "method": self.method,
            "group": self.shuffle_group,
            "n_trials": int(self.n_trials[row]),
            "n_matches": int(self.n_matches[row]),
            "expected": float(self.expected[row]),
            "z_score": float(self.z_score[row]),
            "p_value": float(self.p_value[row]),
            "shuffle_p_value": self.shuffle_p_value,
            "shuffle_z_score": self.shuffle_z_score,
        }


def template_match_dataset_test(
    env: Environment | Mapping[str, Environment],
    spike_times: Any,
    intervals: NDArray[np.float64] | Mapping[str, NDArray[np.float64]],
    encoding_models: Any,
    *,
    template_match_method: TemplateMatchMethod,
    rng: np.random.Generator | int | None = None,
) -> TemplateMatchDatasetResult:
    """Test a data set's matches against chance and template shuffles.

    Lee & Wilson (2002) and Ji & Wilson (2007) drew their conclusions from
    the number of matching events in a data set, not from single events.

    - ``statistic="best_match"`` (Lee & Wilson): each word is a pair (two
      letters of distinct cells), a triplet (three) or a low-probability
      trial (its best possible match has probability at most ``P``, the
      ``trial_probability``). A pair matches in template order (chance
      1/2), a triplet in order (1/6), a low-probability word when its p is
      at most ``P``. Each group's ``Z = (M - N P) / sqrt(N P (1 - P))``.
      The template shuffles permute every template's cell order together
      and recompute the pooled low-probability Z.
    - ``statistic="matching_index"`` (Ji & Wilson): a frame is a candidate
      for a template when at least ``min_active_cells`` of its cells fired,
      and is assigned to the candidate template with the larger matching
      index. The number of replaying frames ``L`` is compared with chance,
      ``Normal(a, sqrt(a))`` with ``a`` the sum of each frame's test size,
      and with the counts for templates whose cell orders are permuted
      (counts combined over templates). Each shuffle assigns frames to the
      template with the larger shuffled index again, so the null repeats
      the choice.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment of the rate maps, or one per template.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the order of the rows
        of ``encoding_models``.
    intervals : array of shape (n_events, 2), or Mapping
        The events, ``(start, stop)`` in seconds. For ``"best_match"`` with
        several templates, a mapping of each template's words, keyed like
        ``encoding_models`` (Lee & Wilson cut each template's cells into
        words separately: ``spike_words(..., cells=...).intervals``). For
        ``"matching_index"``, one array of frames for every template.
    encoding_models : array of shape (n_cells, n_bins), SpatialRatesResult, or Mapping
        Each cell's rate map on ``env``, in Hz, or a mapping of them, one
        per template, with the cells in the same order.
    template_match_method : TemplateMatchMethod
        The method, for example
        ``neurospatial.replay.methods.LEE_WILSON_2002``.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed for the template shuffles and the
        sampled orderings of long words. Pass a seed for a reproducible
        result.

    Returns
    -------
    TemplateMatchDatasetResult
        Each group's counts and tests (``groups_table()``) and the template
        shuffle test.

    Raises
    ------
    ValueError
        If ``intervals`` is not ``(start, stop)`` rows with ``start <
        stop``, or a mapping whose keys differ from those of
        ``encoding_models``, or a mapping for ``"matching_index"``; or for
        templates :func:`detect_template_match_replay` refuses.

    See Also
    --------
    detect_template_match_replay : The verdict on one event.
    neurospatial.replay.spike_words : Lee & Wilson's words.

    Notes
    -----
    Lee & Wilson reported a lower-bound Z for POST (with ``P``) and an
    upper-bound Z for PRE without saying how they computed the latter;
    ``z_score_upper`` uses each word's attainable size. Their POST
    low-probability trials (35 matches of 270) give Z = 7.2 and an exact
    binomial p of 3.8e-9; against shuffled templates (number not stated;
    ``n_shuffles`` here) the Z was 7.9 standard deviations above the
    shuffles' mean (``shuffle_z_score``). Ji & Wilson divided the count of
    shuffles reaching ``L`` by the number of shuffles; ``(k + 1) / (n + 1)``
    is used here. The chance test's size per frame is the exact test size
    at its number of cells (0.028 to 0.049 at alpha 0.05), which Ji &
    Wilson called the cutoff threshold. With several templates the chance
    test is anti-conservative: a frame is tested against the template it
    matches better, so on random frames with a template and its reverse
    about twice the size replay. The shuffle test repeats that choice.

    Lee & Wilson's shuffles rescore every word that is not a pair or a
    triplet. A word's exact distribution depends only on how many letters
    each cell gives it, in template order, and is computed once per Python
    session, so the cost depends on word length. Measured: 1000 shuffles of
    1000 words of 5 to 8 letters take about 15 s; of 20 ten-letter words
    with repeated cells about 35 s, and of 200 such words about 1 min.
    Every pattern of 9 letters together takes about 3 s, every pattern of
    10 letters about 1 min (ten distinct cells alone about 2.5 s); longer
    words are sampled. Ji & Wilson's shuffles are vectorized and take well
    under a second.

    Examples
    --------
    Twenty frames, each with six cells firing in field order:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import template_match_dataset_test
    >>> from neurospatial.replay.methods import JI_WILSON_2007
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> rates = 20.0 * np.exp(
    ...     -0.5 * ((x - np.linspace(15.0, 105.0, 6)[:, None]) / 8.0) ** 2
    ... )
    >>> onsets = np.arange(20) * 2.0
    >>> spikes = [onsets + 0.05 * i for i in range(6)]
    >>> frames = np.column_stack([onsets, onsets + 0.3])
    >>> result = template_match_dataset_test(
    ...     env, spikes, frames, rates, template_match_method=JI_WILSON_2007, rng=0
    ... )
    >>> row = result.groups_table().iloc[0]
    >>> (
    ...     int(row["n_matches"]),
    ...     round(float(row["expected"]), 2),
    ...     round(result.shuffle_p_value, 3),
    ... )
    (20, 0.56, 0.024)

    All 20 frames replay, against 0.56 expected by chance. The shuffle test
    is weaker: every frame has the same order, and 2.8% of the 720 orders
    of six cells also pass p < 0.05 against it.
    """
    from neurospatial._spikes import as_spike_trains

    operation = "template_match_dataset_test()"
    method = template_match_method
    names: list[str | None] = (
        list(encoding_models) if isinstance(encoding_models, Mapping) else [None]
    )
    trains = _sorted_trains(as_spike_trains(spike_times))
    ranks = {
        name: _method_ranks(method, env, encoding_models, name, len(trains), operation)
        for name in names
    }
    generator = np.random.default_rng(rng)
    if method.statistic == "best_match":
        events = _template_intervals(intervals, names, operation)
        return _word_groups_test(method, ranks, trains, events, generator)
    if isinstance(intervals, Mapping):
        raise ValueError(
            _format_error(
                f"{operation}: method {method.name!r} tests frames against every "
                "template, so intervals must be one array, not a mapping.",
                why="Ji & Wilson assigned each frame to the template it matched better",
                fix="pass the frames as an array of shape (n_frames, 2)",
            )
        )
    frames = _checked_intervals(intervals, operation, "intervals")
    return _frame_test(method, ranks, trains, frames, generator)


def _checked_intervals(
    intervals: Any, operation: str, name: str
) -> NDArray[np.float64]:
    values = np.asarray(intervals, dtype=np.float64)
    if values.size == 0:
        return values.reshape(0, 2)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or not np.isfinite(values).all()
        or np.any(values[:, 0] >= values[:, 1])
    ):
        raise ValueError(
            _format_error(
                f"{operation}: {name} must be finite (start, stop) rows in seconds "
                f"with start < stop, got shape {values.shape}.",
                why="each row is one event, and spikes at start <= t < stop belong "
                "to it",
                fix="pass an array of shape (n_events, 2), such as "
                "spike_words(...).intervals",
            )
        )
    return values


def _template_intervals(
    intervals: Any, names: list[str | None], operation: str
) -> dict[str | None, NDArray[np.float64]]:
    if names == [None]:
        if isinstance(intervals, Mapping):
            raise ValueError(
                f"{operation}: intervals is a mapping but encoding_models is one "
                "set of rate maps; pass an array of shape (n_events, 2)."
            )
        return {None: _checked_intervals(intervals, operation, "intervals")}
    if not isinstance(intervals, Mapping) or set(intervals) != set(names):
        given = list(intervals) if isinstance(intervals, Mapping) else "an array"
        raise ValueError(
            _format_error(
                f"{operation}: intervals must map each template {names} to its "
                f"words, got {given}.",
                why="Lee & Wilson cut each template's cells into words separately",
                fix="pass {name: spike_words(spike_times, max_isi=..., max_gap=..., "
                "cells=template_cells).intervals for each template}",
            )
        )
    return {
        name: _checked_intervals(intervals[name], operation, f"intervals[{name!r}]")
        for name in names
    }


def _binomial_tests(
    n_trials: int, n_matches: int, probability: float, sizes: NDArray[np.float64]
) -> tuple[float, float, float, float, float]:
    """Expected count, z, upper z, normal p and exact binomial p of one group."""
    from scipy import stats

    if n_trials == 0:
        return 0.0, np.nan, np.nan, np.nan, np.nan
    expected = n_trials * probability
    z = (n_matches - expected) / np.sqrt(expected * (1.0 - probability))
    variance = float(np.sum(sizes * (1.0 - sizes)))
    z_upper = (
        (n_matches - float(sizes.sum())) / np.sqrt(variance) if variance > 0 else np.nan
    )
    exact = float(stats.binom.sf(n_matches - 1, n_trials, probability))
    return expected, float(z), float(z_upper), float(stats.norm.sf(z)), exact


@dataclass
class _Word:
    """One word: its letters' cells and times, and its template."""

    template: str | None
    units: NDArray[np.intp]
    times: NDArray[np.float64]
    n_distinct: int

    def letter_ranks(self, ranks: NDArray[np.intp]) -> NDArray[np.intp]:
        """The letters' template ranks in firing order, ties ordered against ``ranks``.

        Re-sorted for each template, real or shuffled, so a tie never helps a
        match.
        """
        return ranks[self.units[_firing_order(self.units, self.times, ranks)]]


class _Distributions:
    """Best-match key distributions per multiset of letters, within one test."""

    def __init__(self, method: TemplateMatchMethod, rng: np.random.Generator) -> None:
        self.method = method
        self.rng = rng
        self.orderings: dict[int, NDArray[np.intp]] = {}
        self.sampled: dict[tuple[int, ...], _KeyDistribution] = {}

    def get(self, dense: NDArray[np.int16]) -> _KeyDistribution:
        pattern = tuple(int(value) for value in np.sort(dense))
        method = self.method
        if dense.size <= method.exact_max_letters:
            return _exact_distribution(pattern, method.match_order)
        if pattern not in self.sampled:
            n = dense.size
            if n not in self.orderings:
                # Common random orderings for every word of this length.
                self.orderings[n] = _random_orderings(n, method.n_samples, self.rng)
            self.sampled[pattern] = _sampled_distribution(
                pattern, method.match_order, self.orderings[n]
            )
        return self.sampled[pattern]


def _word_outcomes(
    words: list[_Word],
    ranks: dict[str | None, NDArray[np.intp]],
    distributions: _Distributions,
) -> tuple[list[str | None], NDArray[np.float64], NDArray[np.float64]]:
    """Each word's group, p-value and attainable low-probability size."""
    method = distributions.method
    assert method.trial_probability is not None
    groups: list[str | None] = []
    p_values = np.full(len(words), np.nan)
    sizes = np.zeros(len(words))
    keys = _word_keys(words, ranks, method.match_order)
    for index, word in enumerate(words):
        n = word.units.size
        if n < 2 or word.n_distinct < 2:
            groups.append(None)
            continue
        dense = _dense(ranks[word.template][word.units])
        distribution = distributions.get(dense)
        best = distribution.count(_perfect_key(n, word.n_distinct, method.match_order))
        group = _trial_group(
            n, word.n_distinct, best / distribution.total, method.trial_probability
        )
        groups.append(group)
        p_values[index] = distribution.p_value(int(keys[index]))
        if group == _LOW:
            sizes[index] = distribution.attainable_size(method.trial_probability)
    return groups, p_values, sizes


def _word_keys(
    words: list[_Word], ranks: dict[str | None, NDArray[np.intp]], match_order: str
) -> NDArray[np.int64]:
    """Each word's best-match key, scored in one batch per word length."""
    keys = np.full(len(words), -1, dtype=np.int64)
    lengths = np.array([word.units.size for word in words], dtype=np.intp)
    for n in np.unique(lengths[lengths >= 2]):
        members = np.flatnonzero(lengths == n)
        rows = np.stack(
            [_dense(words[i].letter_ranks(ranks[words[i].template])) for i in members]
        )
        keys[members] = _match_keys(rows, match_order)
    return keys


def _word_groups_test(
    method: TemplateMatchMethod,
    ranks: dict[str | None, NDArray[np.intp]],
    trains: list[NDArray[np.float64]],
    events: dict[str | None, NDArray[np.float64]],
    rng: np.random.Generator,
) -> TemplateMatchDatasetResult:
    """Lee & Wilson's binomial tests of pairs, triplets and low-probability words."""
    from neurospatial.stats.shuffle import compute_shuffle_zscore

    assert method.max_isi is not None and method.trial_probability is not None
    probability = method.trial_probability
    words = []
    for name, rows in events.items():
        for start, stop in rows:
            units, times = _event_letters(
                trains, ranks[name], start, stop, method.max_isi
            )
            words.append(_Word(name, units, times, int(np.unique(units).size)))
    distributions = _Distributions(method, rng)
    groups, p_values, sizes = _word_outcomes(words, ranks, distributions)
    chance = {_PAIR: 0.5, _TRIPLET: 1.0 / 6.0, _LOW: probability}
    names = list(events)
    row_templates: list[str | None] = []
    row_groups: list[str] = []
    columns: dict[str, list[float]] = {
        key: []
        for key in (
            "n_trials",
            "n_matches",
            "probability",
            "expected",
            "z_score",
            "z_score_upper",
            "p_value",
            "p_value_exact",
        )
    }
    word_templates = [word.template for word in words]
    scopes: list[str | None] = [*names, None] if names != [None] else [None]
    for scope in scopes:
        for group, group_probability in chance.items():
            members = np.array(
                [
                    groups[i] == group and (scope is None or word_templates[i] == scope)
                    for i in range(len(words))
                ],
                dtype=bool,
            )
            n_trials = int(members.sum())
            matched = _passes(p_values[members], group_probability, "<=")
            n_matches = int(np.sum(matched))
            group_sizes = (
                sizes[members]
                if group == _LOW
                else np.full(n_trials, group_probability)
            )
            expected, z, z_upper, normal_p, exact_p = _binomial_tests(
                n_trials, n_matches, group_probability, group_sizes
            )
            row_templates.append(scope)
            row_groups.append(group)
            columns["n_trials"].append(n_trials)
            columns["n_matches"].append(n_matches)
            columns["probability"].append(group_probability)
            columns["expected"].append(expected)
            columns["z_score"].append(z)
            columns["z_score_upper"].append(z if group != _LOW else z_upper)
            # Lee & Wilson: normal approximation for pairs, exact binomial for
            # low-probability trials; triplets are not stated (exact here).
            columns["p_value"].append(normal_p if group == _PAIR else exact_p)
            columns["p_value_exact"].append(exact_p)
    observed = _pooled_low_probability_z(groups, p_values, probability)
    # Pairs, triplets and words of one cell keep their group under any
    # template; only the others can be low-probability trials.
    open_words = [
        word
        for word, group in zip(words, groups, strict=True)
        if group not in (_PAIR, _TRIPLET) and word.n_distinct >= 2
    ]
    shuffled = np.empty(method.n_shuffles)
    for shuffle in range(method.n_shuffles):
        permuted = {
            name: _permuted_ranks(template_ranks, rng)
            for name, template_ranks in ranks.items()
        }
        shuffle_groups, shuffle_p, _ = _word_outcomes(
            open_words, permuted, distributions
        )
        shuffled[shuffle] = _pooled_low_probability_z(
            shuffle_groups, shuffle_p, probability
        )
    return _dataset_result(
        method,
        row_templates,
        row_groups,
        columns,
        shuffle_group=_LOW,
        observed=observed,
        shuffled=shuffled,
        zscore=compute_shuffle_zscore,
    )


def _pooled_low_probability_z(
    groups: list[str | None], p_values: NDArray[np.float64], probability: float
) -> float:
    low = np.array([group == _LOW for group in groups], dtype=bool)
    n_trials = int(low.sum())
    if n_trials == 0:
        return np.nan
    n_matches = int(np.sum(_passes(p_values[low], probability, "<=")))
    expected = n_trials * probability
    return float((n_matches - expected) / np.sqrt(expected * (1.0 - probability)))


def _permuted_ranks(
    ranks: NDArray[np.intp], rng: np.random.Generator
) -> NDArray[np.intp]:
    """The template with its cells' places permuted; non-template cells stay -1."""
    template = np.flatnonzero(ranks >= 0)
    permuted: NDArray[np.intp] = ranks.copy()
    permuted[template] = rng.permutation(ranks[template])
    return permuted


def _dataset_result(
    method: TemplateMatchMethod,
    row_templates: list[str | None],
    row_groups: list[str],
    columns: dict[str, list[float]],
    *,
    shuffle_group: str,
    observed: float,
    shuffled: NDArray[np.float64],
    zscore: Any,
) -> TemplateMatchDatasetResult:
    finite = shuffled[np.isfinite(shuffled)]
    if np.isfinite(observed) and finite.size:
        shuffle_p = float((np.sum(finite >= observed) + 1) / (finite.size + 1))
        shuffle_z = float(zscore(observed, shuffled))
    else:
        shuffle_p, shuffle_z = np.nan, np.nan
    return TemplateMatchDatasetResult(
        method=method.name,
        statistic=method.statistic,
        row_templates=tuple(row_templates),
        row_groups=tuple(row_groups),
        n_trials=np.asarray(columns["n_trials"], dtype=np.int64),
        n_matches=np.asarray(columns["n_matches"], dtype=np.int64),
        probability=np.asarray(columns["probability"], dtype=np.float64),
        expected=np.asarray(columns["expected"], dtype=np.float64),
        z_score=np.asarray(columns["z_score"], dtype=np.float64),
        z_score_upper=np.asarray(columns["z_score_upper"], dtype=np.float64),
        p_value=np.asarray(columns["p_value"], dtype=np.float64),
        p_value_exact=np.asarray(columns["p_value_exact"], dtype=np.float64),
        shuffle_group=shuffle_group,
        observed_statistic=float(observed),
        shuffle_statistics=shuffled,
        shuffle_p_value=shuffle_p,
        shuffle_z_score=shuffle_z,
    )


def _poisson_binomial_sf(probabilities: NDArray[np.float64], k: int) -> float:
    """P(at least ``k`` successes) of independent trials with these probabilities."""
    from scipy import stats

    distribution = np.ones(1)
    values, counts = np.unique(probabilities, return_counts=True)
    for value, count in zip(values, counts, strict=True):
        pmf = stats.binom.pmf(np.arange(count + 1), count, value)
        distribution = np.convolve(distribution, pmf)
    return float(min(1.0, distribution[k:].sum())) if k > 0 else 1.0


def _frame_test(
    method: TemplateMatchMethod,
    ranks: dict[str | None, NDArray[np.intp]],
    trains: list[NDArray[np.float64]],
    frames: NDArray[np.float64],
    rng: np.random.Generator,
) -> TemplateMatchDatasetResult:
    """Ji & Wilson's chance and template-shuffle tests of replaying frames."""
    from scipy import stats

    from neurospatial.stats.shuffle import compute_shuffle_zscore

    assert method.bandwidth is not None
    names = list(ranks)
    n_shuffles = method.n_shuffles
    # Every template's cell order permuted together, n_shuffles times.
    permutations = {
        name: np.stack(
            [_permuted_ranks(template_ranks, rng) for _ in range(n_shuffles)]
        )
        for name, template_ranks in ranks.items()
    }
    any_template = np.any(np.stack(list(ranks.values())) >= 0, axis=0)
    union = np.where(any_template, 0, -1).astype(np.intp)
    sizes = {}
    assigned = np.full(len(frames), -1, dtype=np.intp)
    assigned_cells = np.zeros(len(frames), dtype=np.intp)
    replaying = np.zeros(len(frames), dtype=bool)
    shuffle_counts = np.zeros(n_shuffles)
    for index, (start, stop) in enumerate(frames):
        active, times = _active_firing_times(
            trains, union, start, stop, method.bandwidth
        )
        scores = np.full((len(names), n_shuffles + 1), -np.inf)
        p_values = np.full((len(names), n_shuffles + 1), np.inf)
        cells = np.zeros(len(names), dtype=np.intp)
        for row, name in enumerate(names):
            mine = ranks[name][active] >= 0
            m = int(mine.sum())
            cells[row] = m
            if m < method.min_active_cells:
                continue
            units = active[mine]
            observed_ranks = ranks[name][units][None, :]
            all_ranks = np.vstack([observed_ranks, permutations[name][:, units]])
            opposite = np.asarray(_opposite_pairs(times[mine], all_ranks))
            pairs = m * (m - 1) // 2
            scores[row] = (pairs - 2 * opposite) / pairs
            p_values[row] = _inversion_p_values(m)[opposite]
        if not np.any(cells >= method.min_active_cells):
            continue
        chosen = _assignments(scores, p_values)
        chosen_p = p_values[chosen, np.arange(n_shuffles + 1)]
        passing = _passes(chosen_p, method.alpha, method.p_comparison)
        assigned[index] = chosen[0]
        assigned_cells[index] = cells[chosen[0]]
        replaying[index] = bool(passing[0])
        shuffle_counts += passing[1:]
        m = int(cells[chosen[0]])
        if m not in sizes:
            sizes[m] = _index_test_size(m, method.alpha, method.p_comparison)
    row_templates: list[str | None] = []
    columns: dict[str, list[float]] = {
        key: []
        for key in (
            "n_trials",
            "n_matches",
            "probability",
            "expected",
            "z_score",
            "z_score_upper",
            "p_value",
            "p_value_exact",
        )
    }
    scopes: list[str | None] = [*names, None] if names != [None] else [None]
    for scope in scopes:
        members = assigned >= 0
        if scope is not None:
            members &= assigned == names.index(scope)
        frame_sizes = np.array([sizes[m] for m in assigned_cells[members]], dtype=float)
        n_trials = int(members.sum())
        n_matches = int(replaying[members].sum())
        expected = float(frame_sizes.sum())
        if expected > 0:
            z = (n_matches - expected) / np.sqrt(expected)
            normal_p = float(stats.norm.sf(z))
        else:
            z, normal_p = np.nan, np.nan
        row_templates.append(scope)
        columns["n_trials"].append(n_trials)
        columns["n_matches"].append(n_matches)
        columns["probability"].append(np.nan)
        columns["expected"].append(expected)
        columns["z_score"].append(float(z))
        columns["z_score_upper"].append(np.nan)
        columns["p_value"].append(normal_p)
        columns["p_value_exact"].append(
            _poisson_binomial_sf(frame_sizes, n_matches) if n_trials else np.nan
        )
    return _dataset_result(
        method,
        row_templates,
        [_FRAMES] * len(row_templates),
        columns,
        shuffle_group=_FRAMES,
        observed=float(replaying[assigned >= 0].sum()),
        shuffled=shuffle_counts,
        zscore=compute_shuffle_zscore,
    )


def _assignments(
    scores: NDArray[np.float64], p_values: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Each column's template: the largest score, then the smaller p, then the first.

    ``scores`` and ``p_values`` are (n_templates, n_columns); a template that
    is not a candidate has score -inf.
    """
    best = scores.max(axis=0)
    tied = scores == best
    smallest = np.where(tied, p_values, np.inf).min(axis=0)
    chosen: NDArray[np.intp] = np.argmax(tied & (p_values == smallest), axis=0)
    return chosen


# --------------------------------------------------------------------------
# Published methods
# --------------------------------------------------------------------------

LEE_WILSON_2002 = TemplateMatchMethod(
    name="lee_wilson_2002",
    reference=(
        "Lee, A. K., & Wilson, M. A. (2002). Memory of sequential experience in "
        "the hippocampus during slow wave sleep. Neuron, 36(6), 1183-1194."
    ),
    statistic="best_match",
    field_order="peak",
    max_isi=0.05,
    max_gap=0.1,
    match_order="interruptions_first",
    trial_probability=1 / 24,
    exact_max_letters=10,
    n_samples=100_000,
    min_active_cells=2,
    alpha=1 / 24,
    p_comparison="<=",
    n_shuffles=1000,
    notes=(
        "Template: the order of the peaks of each cell's smoothed directional "
        "rate field (Gaussian 5, 10 or 15 cm gave the same order), one template "
        "per running direction. Cells: at least 1 spike per lap in that "
        "direction after removing spikes at the track ends and during stops, "
        "single-peaked, firing through the session; apply these through the "
        "rate maps passed (NaN rows). A template with two fields in one bin is "
        "refused.",
        "Letters: a cell's spikes less than 50 ms apart are one letter at the "
        "first spike. Words: the letters of one template's cells, split where "
        "adjacent letters are more than 100 ms apart, within slow-wave sleep: "
        "spike_words(max_isi=0.05, max_gap=0.1, cells=..., epochs=sws). A given "
        "interval is scored as one word; bursts are found on the whole trains.",
        "Best match: the best (x, y) on the list (K, 0) > (K, 1) > (K - 1, 0) > "
        "... > (2, 0); p is the share of the n! orderings of the word's letters "
        "with a match at least as good. A word is replay when p <= 1/24, "
        "inclusive (a perfect four-letter word); a word whose best possible "
        "match has p above 1/24 can never pass. Forward order only; the paper's "
        "reversed-sequence control is the reversed template. Letters at the same "
        "time are ordered against the template (unstated).",
        "Exact enumeration up to 10 letters, else 100,000 random orderings "
        "((k + 1) / (n + 1)); the paper does not say how it enumerated. Lee & "
        "Wilson (2004, J Neurophysiol 92:2555) describe the method in full and "
        "were not consulted.",
        "Data-set test (template_match_dataset_test): pairs (P 1/2, normal "
        "approximation), triplets (P 1/6, exact binomial here; the paper does "
        "not say) and low-probability trials (P 1/24, exact binomial); Z = "
        "(M - N P) / sqrt(N P (1 - P)), a lower bound with P and an upper bound "
        "with each word's attainable size (the paper's upper bound is not "
        "described). Template shuffles: 1000 here (not stated); the observed "
        "pooled Z is reported in standard deviations of the shuffles, as the "
        "paper's 7.9.",
        "Not applied: the pair-bias reweighting, the repeated-letter "
        "constraint, wraparound sequences, cluster manipulations and the "
        "template-overlap compression analysis with cell-shuffled templates "
        "(controls, pp. 1187-1188 and 1193). REM was excluded by a theta/total "
        "power ratio; sleep staging is not done here.",
    ),
)

JI_WILSON_2007 = TemplateMatchMethod(
    name="ji_wilson_2007",
    reference=(
        "Ji, D., & Wilson, M. A. (2007). Coordinated memory replay in the visual "
        "cortex and hippocampus during sleep. Nature Neuroscience, 10(1), 100-107."
    ),
    statistic="matching_index",
    field_order="peak",
    min_template_cells=5,
    bandwidth=0.18,
    min_active_cells=4,
    alpha=0.05,
    p_comparison="<",
    template_choice="largest_score",
    n_shuffles=1000,
    notes=(
        "Template: cells ordered by the peak of their firing on a linearized "
        "trajectory (2 cm bins, rate = spikes / occupancy, dominant peak); cells "
        "with more than 0.8 bit/spike and a stable dominant field (peak at the "
        "same place in at least 70% of laps); CA1 RUN rate 0.2 to 4 Hz, cortex "
        "at least 0.5 Hz; templates of at least 5 cells. Apply the cell rules "
        "through the rate maps passed (NaN rows).",
        "Firing time in a frame: the peak of the sum of Gaussian kernels on the "
        "cell's spikes in the frame, sigma 180 ms for hippocampus; use "
        "dataclasses.replace(JI_WILSON_2007, bandwidth=0.4) for cortex. Located "
        "on a 1 ms grid and refined to the exact maximum here (unstated). Cells "
        "with the same firing time count as a pair in opposite order.",
        "I = (m - n) / (m + n) over the active template cells' pairs; p = "
        "P(I >= observed) with every order equally likely, exact for every "
        "number of cells here (the paper sampled 10,000 orders from 11 cells; "
        "its 14-cell cutoff of 30 opposite pairs is p = 0.0505 exactly, so those "
        "frames are not replay here). Replay when p < 0.05; frames with fewer "
        "than 4 active cells cannot reach it. A frame that is a candidate for "
        "both trajectories is tested against the one with the larger I (ties: "
        "smaller p, then the first; unstated).",
        "Data-set test (template_match_dataset_test): chance significance from "
        "Normal(a, sqrt(a)), a = the sum of each candidate frame's exact test "
        "size at its number of cells; shuffle significance from 1000 "
        "permutations of every template, counts combined over templates, each "
        "shuffle reassigning frames to the template with the larger shuffled I "
        "(whether the paper did is not stated). The paper divides by 1000 "
        "without the +1.",
        "Frames (not detected here): multiunit spikes in 10 ms bins smoothed "
        "with a 30 ms Gaussian, silent below a threshold T, frames closer than G "
        "merged (T 0.5-0.8, G 70-90 ms by animal, Supplementary Fig. 12); SWS "
        "of the last hour of PRE and the first hour of POST.",
    ),
)
