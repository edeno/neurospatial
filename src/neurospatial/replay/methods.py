"""Published replay detection methods as explicit, comparable recipes.

Each :class:`ReplayMethod` records one paper's replay test: the score, the
shuffles and significance level, how time bins without spikes are handled,
and the thresholds an event must meet. :func:`detect_replay` applies a method
to one decoded event and :func:`compare_replay_methods` applies several
methods to several events, so the same data can be scored the way each lab
scored it.

A recipe records the paper's decoding settings (time-bin and position-bin
sizes) for reference but does not decode: methods are compared on the same
decoded posterior, so differences come from the replay test alone. Choices a
paper does not specify, and bugs in the authors' code that are not
reproduced, are listed in each recipe's ``notes``.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ResultMixin
from neurospatial.replay.correlation import weighted_correlation
from neurospatial.replay.trajectory import (
    detect_trajectory_radon,
    fit_isotonic_trajectory,
    fit_linear_trajectory,
)

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.decoding import DecodingResult
    from neurospatial.environment import Environment

__all__ = [
    "DAVIDSON_2009",
    "KARLSSON_FRANK_2009",
    "MICHON_2019",
    "PUBLISHED_METHODS",
    "SILVA_2015",
    "TIROLE_2022",
    "WU_2017",
    "ReplayCriterion",
    "ReplayDetectionResult",
    "ReplayMethod",
    "compare_replay_methods",
    "detect_replay",
]

_OPERATION = "detect_replay()"
_DECODING_NULLS = ("unit_identity", "spike_train_circular", "place_field_circular")
_COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class ReplayCriterion:
    """One threshold an event must meet to count as replay.

    Attributes
    ----------
    quantity : str
        ``"score"`` (the method's score), ``"n_active_cells"`` (cells with a
        spike in the event), or a scalar of
        :class:`~neurospatial.replay.TrajectoryMetrics`, such as
        ``"max_map_jump"``.
    comparison : {"<", "<=", ">", ">="}
        How the value is compared with the threshold, as the paper states it.
    threshold : float
        The threshold, in environment units for distances unless
        ``relative_to`` is set.
    relative_to : {"track_length"} or None
        Scale the threshold by the event's track length, for thresholds the
        paper gives as a fraction of the track.
    absolute : bool
        Compare the absolute value, for signed quantities such as the
        weighted correlation.
    """

    quantity: str
    comparison: Literal["<", "<=", ">", ">="]
    threshold: float
    relative_to: Literal["track_length"] | None = None
    absolute: bool = False

    def __post_init__(self) -> None:
        """Refuse unknown comparisons and scales."""
        if self.comparison not in _COMPARISONS:
            raise ValueError(
                f"ReplayCriterion: comparison must be one of "
                f"{list(_COMPARISONS)}, got {self.comparison!r}."
            )
        if self.relative_to not in (None, "track_length"):
            raise ValueError(
                f"ReplayCriterion: relative_to must be None or 'track_length', "
                f"got {self.relative_to!r}."
            )

    @property
    def label(self) -> str:
        """A readable form, such as ``"|score| > 0.6"``."""
        name = f"|{self.quantity}|" if self.absolute else self.quantity
        scale = " x track_length" if self.relative_to else ""
        return f"{name} {self.comparison} {self.threshold:g}{scale}"

    def evaluate(self, value: float, track_length: float) -> bool:
        """Whether ``value`` meets this criterion; False when either is NaN."""
        threshold = self.threshold * (track_length if self.relative_to else 1.0)
        value = abs(value) if self.absolute else value
        if not (np.isfinite(value) and np.isfinite(threshold)):
            return False
        return bool(_COMPARISONS[self.comparison](value, threshold))


@dataclass(frozen=True)
class ReplayMethod:
    """A published replay test: score, shuffles, and thresholds.

    Attributes
    ----------
    name : str
        Short name, for example ``"davidson_2009"``.
    reference : str
        The paper the method follows.
    score : callable or None
        The replay score, a function ``score(env, posterior, times,
        **score_options)`` returning a result with ``score``, such as
        :func:`~neurospatial.replay.weighted_correlation`. None for a method
        that uses criteria only.
    score_options : Mapping[str, Any]
        Options for ``score``. ``"n_samples_per_time_bin"`` is converted to
        ``n_samples`` for each event, for samplers like
        :func:`~neurospatial.replay.fit_linear_trajectory` whose papers
        give a count per time bin.
    nulls : tuple of str
        Shuffles the score is tested against (see
        :func:`~neurospatial.replay.replay_significance`). Empty for a method
        without a per-event shuffle test.
    n_shuffles : int
        Shuffles per null.
    alpha : float
        Significance level: the event's largest p-value must be below it.
    alternative : {"greater", "less", "two-sided"}
        Tail of the shuffle test.
    use_absolute : bool or None
        Test the absolute score; None uses the default of
        :func:`~neurospatial.replay.replay_significance`.
    empty_time_bins : {"keep", "drop", "zero"}
        Time bins without spikes: keep them, drop them with their times
        (Karlsson & Frank 2009; Wu et al. 2017), or set their posterior to
        zero (the Bendor-lab code).
    criteria : tuple of ReplayCriterion
        Thresholds the event must meet as well.
    decoding : Mapping[str, Any]
        The paper's decoding settings, for reference (not enforced).
    notes : tuple of str
        Choices the paper does not specify, and reference-code bugs not
        reproduced.
    """

    name: str
    reference: str
    score: Callable[..., Any] | None
    score_options: Mapping[str, Any] = field(default_factory=dict)
    nulls: tuple[str, ...] = ()
    n_shuffles: int = 1000
    alpha: float = 0.05
    alternative: Literal["greater", "less", "two-sided"] = "greater"
    use_absolute: bool | None = None
    empty_time_bins: Literal["keep", "drop", "zero"] = "keep"
    criteria: tuple[ReplayCriterion, ...] = ()
    decoding: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the mappings and refuse settings that cannot run."""
        object.__setattr__(
            self, "score_options", MappingProxyType(dict(self.score_options))
        )
        object.__setattr__(self, "decoding", MappingProxyType(dict(self.decoding)))
        object.__setattr__(self, "nulls", tuple(self.nulls))
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.score is None and (
            self.nulls or any(c.quantity == "score" for c in self.criteria)
        ):
            raise ValueError(
                f"ReplayMethod {self.name!r}: a method without a score cannot "
                "test or threshold one."
            )
        if self.empty_time_bins not in ("keep", "drop", "zero"):
            raise ValueError(
                f"ReplayMethod {self.name!r}: empty_time_bins must be 'keep', "
                f"'drop' or 'zero', got {self.empty_time_bins!r}."
            )
        from neurospatial.replay.significance import _NULLS

        unknown = [name for name in self.nulls if name not in _NULLS]
        if unknown:
            raise ValueError(
                f"ReplayMethod {self.name!r}: unknown nulls {unknown}; choose "
                f"from {list(_NULLS)}."
            )
        if self.alternative not in ("greater", "less", "two-sided"):
            raise ValueError(
                f"ReplayMethod {self.name!r}: alternative must be 'greater', "
                f"'less' or 'two-sided', got {self.alternative!r}."
            )
        if not 0 < self.alpha < 1:
            raise ValueError(
                f"ReplayMethod {self.name!r}: alpha must be in (0, 1), got "
                f"{self.alpha!r}."
            )
        if isinstance(self.n_shuffles, bool) or int(self.n_shuffles) < 1:
            raise ValueError(
                f"ReplayMethod {self.name!r}: n_shuffles must be a positive "
                f"integer, got {self.n_shuffles!r}."
            )

    def __hash__(self) -> int:
        """Hash by name and reference; the option mappings are not hashable."""
        return hash((self.name, self.reference))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        return (
            type(self),
            (
                self.name,
                self.reference,
                self.score,
                dict(self.score_options),
                self.nulls,
                self.n_shuffles,
                self.alpha,
                self.alternative,
                self.use_absolute,
                self.empty_time_bins,
                self.criteria,
                dict(self.decoding),
                self.notes,
            ),
        )


@dataclass(frozen=True, repr=False, eq=False)
class ReplayDetectionResult(ResultMixin):
    """One method's verdict on one event.

    Attributes
    ----------
    method : str
        The method's name.
    is_replay : bool
        True when the event meets every criterion and, for a method with
        shuffles, its largest p-value is below ``alpha``.
    score : float
        The method's score (NaN for a criteria-only method).
    p_value : float
        The largest p-value over the method's nulls (NaN without shuffles).
    criteria : Mapping[str, bool]
        Whether each criterion holds, keyed by its label.
    values : Mapping[str, float]
        The value each criterion was compared on, keyed by its label.
    """

    method: str
    is_replay: bool
    score: float
    p_value: float
    criteria: Mapping[str, bool]
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        """Freeze the mappings."""
        object.__setattr__(self, "criteria", MappingProxyType(dict(self.criteria)))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        return (
            type(self),
            (
                self.method,
                self.is_replay,
                self.score,
                self.p_value,
                dict(self.criteria),
                dict(self.values),
            ),
        )

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "method": self.method,
            "is_replay": self.is_replay,
            "score": self.score,
            "p_value": self.p_value,
            **{f"passes[{label}]": ok for label, ok in self.criteria.items()},
        }


def detect_replay(
    env: Environment | Mapping[str, Environment],
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    replay_method: ReplayMethod,
    spike_counts: NDArray[np.int64] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: Any = None,
    template: str | None = None,
    pseudo_event_pool: NDArray[np.float64] | None = None,
    rng: np.random.Generator | int | None = None,
) -> ReplayDetectionResult:
    """Apply one published replay method to one decoded event.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the posterior was decoded on; for one template's
        share of a joint decode over separate tracks, the mapping of
        environments passed to :func:`~neurospatial.decoding.decode_jointly`,
        with ``template`` naming the share.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        The event's posterior, as decoded.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds.
    replay_method : ReplayMethod
        The method, for example ``neurospatial.replay.methods.DAVIDSON_2009``.
    spike_counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons)
        The event's spike counts. Needed by methods that drop or zero time
        bins without spikes, count active cells, or decode shuffled counts.
    encoding_models, dt, prior, template
        The decoding inputs, for the nulls that decode again (see
        :func:`~neurospatial.replay.replay_significance`), including one
        template's share of a joint decode (``template=``).
    pseudo_event_pool : NDArray[np.float64] or None
        Posterior rows of the session's candidate events, for the
        pseudo-event null.
    rng : np.random.Generator, int, or None
        Random number generator or seed.

    Returns
    -------
    ReplayDetectionResult
        The verdict, score, p-value and each criterion.

    Raises
    ------
    ValueError
        If the method needs an input that is not given (spike counts,
        decoding inputs or a pseudo-event pool), or for inputs the score,
        the shuffles or :func:`~neurospatial.replay.trajectory_metrics`
        refuse.

    See Also
    --------
    compare_replay_methods : Several methods on several events.
    neurospatial.replay.replay_significance : The shuffle test.
    neurospatial.replay.trajectory_metrics : The quantities criteria use.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import detect_replay
    >>> from neurospatial.replay.methods import SILVA_2015
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
    ... )
    >>> posterior = np.zeros((20, env.n_bins))
    >>> posterior[np.arange(20), np.arange(20)] = 1.0  # 4 cm per time bin
    >>> result = detect_replay(
    ...     env, posterior, np.arange(20) * 0.02, replay_method=SILVA_2015
    ... )
    >>> result.is_replay, round(result.score, 2)
    (True, 1.0)
    """
    from neurospatial.replay.criteria import trajectory_metrics
    from neurospatial.replay.significance import replay_significance

    if isinstance(env, Mapping):
        if template is None:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: a mapping of environments needs template=.",
                    why="the posterior is one template's share of a joint "
                    "decode, scored on that template's environment",
                    fix="pass template=name, the share's key in env",
                )
            )
        track_env = env[template]
    else:
        track_env = env
    posterior = np.asarray(posterior, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    counts = None if spike_counts is None else np.asarray(spike_counts)
    if counts is not None and counts.ndim == 1:
        counts = counts[:, np.newaxis]
    needs_counts = replay_method.empty_time_bins != "keep" or any(
        c.quantity == "n_active_cells" for c in replay_method.criteria
    )
    if needs_counts and counts is None:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: method {replay_method.name!r} needs spike_counts.",
                why="it drops or zeroes time bins without spikes, or counts "
                "the cells active in the event",
                fix="pass the event's spike_counts, one row per posterior row",
            )
        )
    if replay_method.empty_time_bins == "drop":
        assert counts is not None
        keep = counts.sum(axis=1) > 0
        posterior, times, counts = posterior[keep], times[keep], counts[keep]

    options = dict(replay_method.score_options)
    if "n_samples_per_time_bin" in options:
        options["n_samples"] = int(options.pop("n_samples_per_time_bin")) * len(
            posterior
        )

    score = np.nan
    p_value = np.nan
    if replay_method.score is not None and replay_method.nulls:
        # The test needs the counts only to decode shuffles again or to zero
        # empty time bins; given without the rate maps they would be refused.
        decodes = any(name in _DECODING_NULLS for name in replay_method.nulls)
        result = replay_significance(
            env if decodes else track_env,
            posterior,
            times,
            score=replay_method.score,
            score_options=options,
            nulls=replay_method.nulls,
            spike_counts=counts
            if decodes or replay_method.empty_time_bins == "zero"
            else None,
            encoding_models=encoding_models if decodes else None,
            dt=dt if decodes else None,
            prior=prior,
            template=template if decodes else None,
            pseudo_event_pool=pseudo_event_pool,
            alternative=replay_method.alternative,
            use_absolute=replay_method.use_absolute,
            empty_time_bins="zero"
            if replay_method.empty_time_bins == "zero"
            else "keep",
            n_shuffles=replay_method.n_shuffles,
            rng=rng,
        )
        score, p_value = float(result.observed.score), float(result.p_value)
    elif replay_method.score is not None:
        scored = posterior
        if replay_method.empty_time_bins == "zero":
            assert counts is not None
            scored = np.where(counts.sum(axis=1)[:, None] == 0, 0.0, posterior)
        score = float(replay_method.score(track_env, scored, times, **options).score)

    needs_metrics = any(
        c.quantity not in ("score", "n_active_cells") for c in replay_method.criteria
    ) or any(c.relative_to for c in replay_method.criteria)
    metrics = (
        trajectory_metrics(track_env, posterior, times, spike_counts=counts)
        if needs_metrics
        else None
    )
    track_length = metrics.track_length if metrics is not None else np.nan
    passes: dict[str, bool] = {}
    values: dict[str, float] = {}
    for criterion in replay_method.criteria:
        if criterion.quantity == "score":
            value = score
        elif criterion.quantity == "n_active_cells":
            assert counts is not None
            value = float(np.sum(counts.sum(axis=0) > 0))
        else:
            value = float(getattr(metrics, criterion.quantity))
        values[criterion.label] = value
        passes[criterion.label] = criterion.evaluate(value, track_length)
    significant = bool(p_value < replay_method.alpha) if replay_method.nulls else True
    return ReplayDetectionResult(
        method=replay_method.name,
        is_replay=significant and all(passes.values()),
        score=score,
        p_value=p_value,
        criteria=passes,
        values=values,
    )


def compare_replay_methods(
    env: Environment | Mapping[str, Environment],
    events: Sequence[DecodingResult],
    *,
    replay_methods: Sequence[ReplayMethod],
    spike_counts: Sequence[NDArray[np.int64]] | None = None,
    encoding_models: Any = None,
    dt: float | None = None,
    prior: Any = None,
    template: str | None = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Apply several published replay methods to several decoded events.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment the events were decoded on, or the mapping of
        environments of a joint decode (see :func:`detect_replay`).
    events : sequence of DecodingResult
        The decoded candidate events, each with ``times``.
    replay_methods : sequence of ReplayMethod
        The methods to compare.
    spike_counts : sequence of arrays or None
        Each event's spike counts, for methods that need them.
    encoding_models, dt, prior, template
        The decoding inputs, for the nulls that decode again (see
        :func:`detect_replay`).
    rng : numpy.random.Generator, int, or None
        Random number generator or seed for the shuffles. Each (event, method)
        pair draws from its own stream, derived from ``rng``, the event's
        position and the method's name, so adding, removing or reordering
        methods does not change the other pairs' results.

    Returns
    -------
    pandas.DataFrame
        One row per (event, method): ``event``, ``method``, ``is_replay``,
        ``score``, ``p_value``, one ``passes[...]`` column per criterion, and
        ``error``, empty unless the event has fewer than two time bins left
        to score (after the method drops time bins without spikes), in which
        case ``is_replay`` is False and the score and p-value are NaN. The pseudo-event null draws
        from the posterior rows of every event, as Davidson et al. (2009) and
        Michon et al. (2019) drew from the complete set of candidate events.

    Raises
    ------
    ValueError
        If an event has no ``times``, or if ``spike_counts`` does not have one
        entry per event.

    See Also
    --------
    detect_replay : One method on one event.
    """
    import zlib

    import pandas as pd

    if spike_counts is not None and len(spike_counts) != len(events):
        raise ValueError(
            f"compare_replay_methods(): spike_counts has {len(spike_counts)} "
            f"entries but there are {len(events)} events."
        )
    for index, event in enumerate(events):
        if event.times is None:
            raise ValueError(
                _format_error(
                    f"compare_replay_methods(): event {index} has no times.",
                    why="replay scores and segments need each time bin's time",
                    fix="decode each event with times=, e.g. "
                    "decode_position(env, counts, rates, dt=dt, times=grid)",
                )
            )
    pool = None
    if any("pseudo_event" in method.nulls for method in replay_methods):
        pool = np.vstack([event.posterior for event in events])
    entropy = (
        int(rng.integers(2**63))
        if isinstance(rng, np.random.Generator)
        else np.random.SeedSequence(rng).entropy
    )
    rows = []
    for index, event in enumerate(events):
        for method in replay_methods:
            stream = np.random.default_rng(
                np.random.SeedSequence(
                    entropy, spawn_key=(index, zlib.crc32(method.name.encode()))
                )
            )
            counts = None if spike_counts is None else np.asarray(spike_counts[index])
            usable = np.asarray(event.times)
            if method.empty_time_bins == "drop" and counts is not None:
                usable = usable[counts.reshape(len(usable), -1).sum(axis=1) > 0]
            if np.unique(usable).size < 2:
                rows.append(
                    {
                        "event": index,
                        "method": method.name,
                        "is_replay": False,
                        "score": np.nan,
                        "p_value": np.nan,
                        "error": "fewer than two time bins to score",
                    }
                )
                continue
            result = detect_replay(
                env,
                event.posterior,
                np.asarray(event.times),
                replay_method=method,
                spike_counts=counts,
                encoding_models=encoding_models,
                dt=dt,
                prior=prior,
                template=template,
                pseudo_event_pool=pool,
                rng=stream,
            )
            rows.append({"event": index, **result.summary(), "error": ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Published methods
# --------------------------------------------------------------------------

DAVIDSON_2009 = ReplayMethod(
    name="davidson_2009",
    reference=(
        "Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal "
        "replay of extended experience. Neuron, 63(4), 497-507."
    ),
    score=detect_trajectory_radon,
    nulls=("column_cycle", "unit_identity", "pseudo_event"),
    n_shuffles=1500,
    alpha=0.01,
    decoding={"time_bin": 0.02},
    notes=(
        "Band of 15 cm and speeds up to 30 m/s, as in detect_trajectory_radon.",
        "Significant when the p-value of every null is below 0.01.",
        "The paper decoded position x direction jointly in 20 ms windows and "
        "summed over direction; decode the same way to match it.",
        "Candidate events of at least 100 ms; event selection is not applied here.",
        "Pseudo-events draw from the complete set of candidate events.",
    ),
)

KARLSSON_FRANK_2009 = ReplayMethod(
    name="karlsson_frank_2009",
    reference=(
        "Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote "
        "experiences in the hippocampus. Nature Neuroscience, 12(7), 913-918."
    ),
    score=fit_linear_trajectory,
    score_options={"method": "sample", "n_samples_per_time_bin": 10_000},
    nulls=("time_bin_permutation",),
    n_shuffles=10_000,
    alpha=0.05,
    empty_time_bins="drop",
    criteria=(ReplayCriterion("n_active_cells", ">=", 5),),
    decoding={"time_bin": 0.015, "position_bin": 2.0, "prior": "uniform"},
    notes=(
        "10,000 posterior samples per decoded time bin, one pooled regression "
        "of position on time; R² against 10,000 shuffles of time-bin order.",
        "Time bins without spikes cannot be decoded and are dropped.",
        "At least 5 active place cells; place-cell selection is not applied here.",
        "The paper regresses on bin number; whether the remaining bins are "
        "renumbered after dropping is not stated. Here they keep their times.",
        "The paper's p-value is the share of shuffles with a larger R² (no "
        "+1); here it is (k + 1) / (n + 1).",
    ),
)

WU_2017 = ReplayMethod(
    name="wu_2017",
    reference=(
        "Wu, C.-T., Haggerty, D., Kemere, C., & Ji, D. (2017). Hippocampal "
        "awake replay in fear memory retrieval. Nature Neuroscience, 20(4), "
        "571-580."
    ),
    score=fit_linear_trajectory,
    score_options={"method": "map"},
    nulls=("time_bin_permutation",),
    n_shuffles=1000,
    alpha=0.05,
    empty_time_bins="drop",
    criteria=(ReplayCriterion("n_active_cells", ">=", 4),),
    decoding={"time_bin": 0.02, "time_step": 0.01},
    notes=(
        "Linear regression of the MAP position on time over time bins with "
        "at least one spike; R² against 1,000 shuffles of decoded positions "
        "in time.",
        "At least 4 active template cells (single-field place cells); the "
        "template-cell selection is not applied here.",
        "Dropped time bins keep their times; the paper regresses on time-bin "
        "number and does not say whether bins are renumbered.",
        "The paper's p-value has no +1; here it is (k + 1) / (n + 1).",
    ),
)

SILVA_2015 = ReplayMethod(
    name="silva_2015",
    reference=(
        "Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events "
        "across hippocampal place cells require previous experience. Nature "
        "Neuroscience, 18(12), 1772-1779."
    ),
    score=weighted_correlation,
    criteria=(
        ReplayCriterion("score", ">", 0.6, absolute=True),
        ReplayCriterion("max_map_jump", "<", 0.4, relative_to="track_length"),
    ),
    decoding={"time_bin": 0.02},
    notes=(
        "Thresholds on |weighted correlation| and the maximum jump between "
        "peak decoded positions (fraction of the track). The paper also used "
        "(0.7, 0.4) and (0.7, 0.3).",
        "Significance was tested on the number of events passing, against "
        "5,000 column-cycle shuffled datasets, not per event.",
    ),
)

MICHON_2019 = ReplayMethod(
    name="michon_2019",
    reference=(
        "Michon, F., Sun, J.-J., Kim, C. Y., Ciliberti, D., & Kloosterman, F. "
        "(2019). Post-learning hippocampal replay selectively reinforces "
        "spatial memory for highly rewarded locations. Current Biology. "
        "https://doi.org/10.1016/j.cub.2019.03.048"
    ),
    score=fit_isotonic_trajectory,
    score_options={"method": "map"},
    nulls=("pseudo_event",),
    n_shuffles=500,
    alpha=0.01,
    decoding={"time_bin": 0.01},
    notes=(
        "Weighted isotonic regression of the MAP position, score = mean MAP "
        "probability x R², against 500 pseudo-events drawn from the complete "
        "set of candidate events.",
        "The paper fits each arm of the maze separately; restrict the "
        "posterior to one arm to match it.",
        "Clusterless decoding in the paper; any decode can be scored here.",
    ),
)

TIROLE_2022 = ReplayMethod(
    name="tirole_2022",
    reference=(
        "Tirole, M., Huelin Gorriz, M., Takigawa, M., Kukovska, L., & Bendor, "
        "D. (2022). Experience-driven rate modulation is reinstated during "
        "hippocampal replay. eLife, 11, e79031."
    ),
    score=weighted_correlation,
    score_options={"row_normalization": "none"},
    nulls=("spike_train_circular", "place_field_circular", "column_cycle"),
    n_shuffles=1000,
    alpha=0.05,
    use_absolute=True,
    empty_time_bins="zero",
    criteria=(ReplayCriterion("n_active_cells", ">=", 5),),
    decoding={"time_bin": 0.02, "position_bin": 10.0},
    notes=(
        "Weighted correlation of the posterior with time bins without spikes "
        "set to zero, against three shuffles; significant when above the 95th "
        "percentile of all three. The paper does not say |r|; the Bendor-lab "
        "code (weighted_correlation.m) returns abs(r).",
        "The paper normalizes the posterior across both tracks: pass one "
        "track's share, decode_jointly(...).joint(name), with every track's "
        "rate maps as encoding_models and template=name, so shuffles are "
        "decoded jointly too.",
        "Candidate events of 100-750 ms with at least 5 time bins; event "
        "selection is not applied here.",
        "Events split in halves at the multi-unit minimum (tested at "
        "p < 0.025) are not included; split the event first.",
        "The Bendor-lab code zeroes the original event's empty time bins in "
        "shuffles that move time bins; here each shuffle zeroes its own.",
    ),
)

PUBLISHED_METHODS: tuple[ReplayMethod, ...] = (
    DAVIDSON_2009,
    KARLSSON_FRANK_2009,
    SILVA_2015,
    WU_2017,
    MICHON_2019,
    TIROLE_2022,
)
"""Every published method recipe in this module."""
