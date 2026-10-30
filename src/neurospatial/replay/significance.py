"""Test a replay score against shuffled versions of the event.

:func:`replay_significance` scores a decoded candidate event with any replay
score, scores ``n_shuffles`` shuffled versions of the event with the same
score and search, and reports a Monte Carlo p-value for each null. An event
is called significant when every null rejects it, so the reported p-value is
the largest among the nulls run (Davidson et al. 2009; Tirole et al. 2022).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ResultMixin
from neurospatial.replay.correlation import (
    WeightedCorrelationResult,
    _track_bin_centers,
    _validated_times,
    _validated_weights,
    _weighted_correlations,
    weighted_correlation,
)
from neurospatial.replay.trajectory import (
    _best_line,
    _prepare_replay_lines,
    detect_trajectory_radon,
)

if TYPE_CHECKING:
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment
    from neurospatial.stats.shuffle import ShuffleTestResult

_OPERATION = "replay_significance()"
_NULLS = (
    "column_cycle",
    "unit_identity",
    "pseudo_event",
    "spike_train_circular",
    "place_field_circular",
    "time_bin_permutation",
)
# Run when nulls=None, if their inputs are given: Davidson et al. (2009).
_DEFAULT_NULLS = ("column_cycle", "unit_identity", "pseudo_event")
# Nulls after which a time bin no longer holds its own spikes.
_ROW_MOVING_NULLS = ("pseudo_event", "spike_train_circular", "time_bin_permutation")
# Nulls that decode shuffled spike counts again.
_DECODING_NULLS = ("unit_identity", "spike_train_circular", "place_field_circular")
_ALTERNATIVES = ("greater", "less", "two-sided")


@dataclass(frozen=True, repr=False, eq=False)
class ReplaySignificanceResult(ResultMixin):
    """A decoded event's replay score tested against shuffled events.

    Attributes
    ----------
    observed : result
        The score's result for the event itself, for example a
        :class:`RadonDetectionResult` (speed, start position and replay
        score) or a :class:`WeightedCorrelationResult`. Its ``score`` is the
        tested value.
    shuffles : Mapping[str, ShuffleTestResult]
        One shuffle test per null that was run, keyed by the null's name.
        Each holds the observed statistic, the statistic of every shuffle and
        the Monte Carlo p-value ``(k + 1) / (n + 1)``.
    p_value : float
        The largest p-value among ``shuffles``: an event is significant when
        every null rejects it.
    alternative : {"greater", "less", "two-sided"}
        The side of the null distribution tested.
    use_absolute : bool
        Whether the absolute value of the score was tested.

    Results compare equal only to themselves.

    See Also
    --------
    replay_significance : Function that produces this result.
    """

    observed: Any
    shuffles: Mapping[str, ShuffleTestResult]
    p_value: float
    alternative: str
    use_absolute: bool

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            **self.observed.summary(),
            "p_value": self.p_value,
            **{f"{name}_p_value": test.p_value for name, test in self.shuffles.items()},
            "alternative": self.alternative,
            "use_absolute": self.use_absolute,
        }


def replay_significance(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    score: Callable[..., Any] = detect_trajectory_radon,
    score_options: Mapping[str, Any] | None = None,
    nulls: str | Sequence[str] | None = None,
    spike_counts: NDArray[np.int64] | None = None,
    encoding_models: NDArray[np.float64] | SpatialRatesResult | None = None,
    dt: float | None = None,
    prior: NDArray[np.float64] | None = None,
    pseudo_event_pool: NDArray[np.float64] | None = None,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    use_absolute: bool | None = None,
    n_shuffles: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> ReplaySignificanceResult:
    """Test a decoded event's replay score against shuffled events.

    The event is scored with ``score``. Each null then builds ``n_shuffles``
    shuffled events, each scored with the same function and options, so the
    test accounts for any search the score does (such as the best line of
    :func:`detect_trajectory_radon`). Each null controls for a different way
    a high score can arise without replay:

    - ``"column_cycle"``: each time bin's posterior is shifted circularly
      along the track by a random distance, which keeps each decoded
      distribution but breaks the alignment of positions along a trajectory
      (Davidson et al. 2009; "place bin circular shift" in Tirole et al.
      2022). Needs only the posterior.
    - ``"unit_identity"``: spike trains are reassigned to randomly permuted
      units' rate maps and decoded again, which keeps the spiking dynamics but
      breaks the place code. Needs ``spike_counts``, ``encoding_models`` and
      ``dt`` (and ``prior`` if the decode used one).
    - ``"pseudo_event"``: each time bin's posterior is replaced by one drawn at
      random from other candidate events, which keeps the decoder's bias
      toward particular locations. Needs ``pseudo_event_pool``.
    - ``"spike_train_circular"``: each cell's spike counts are shifted
      circularly in time within the event and decoded again, which keeps each
      cell's spikes but breaks the order in which cells fire (Tirole et al.
      2022). Needs the inputs of ``"unit_identity"``.
    - ``"place_field_circular"``: each cell's rate map is shifted circularly
      along the track and the event decoded again, which keeps each cell's
      firing but moves its field (Grosmark & Buzsáki 2016; Tirole et al.
      2022). Needs the inputs of ``"unit_identity"``. Maps are shifted over
      their supported bins only, so bins without training data stay
      excluded; smoothed maps are shifted as they are (Grosmark & Buzsáki
      shifted unsmoothed maps and smoothed them again).
    - ``"time_bin_permutation"``: the posterior's time bins are put in random
      order (Karlsson & Frank 2009; Takigawa et al. 2024). Needs only the
      posterior. The decoder treats time bins independently, so with a prior
      that is the same in every time bin this equals permuting the spike
      counts and decoding again; with a time-varying prior it also moves each
      time bin's prior.

    Tirole et al. (2022) call an event significant when it beats the
    ``"spike_train_circular"``, ``"place_field_circular"`` and
    ``"column_cycle"`` nulls; Davidson et al. (2009) used ``"column_cycle"``,
    ``"unit_identity"`` and ``"pseudo_event"``. Use the nulls together: on
    events with no sequence, a single null that decodes again, or the
    column cycle, rejects more often than its nominal rate, because a
    shuffled decode is flatter than the event's own posterior. Requiring
    every null to reject brings the rate back to about nominal (Takigawa et
    al. 2024 measured the false-positive rates of these combinations).

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior mass over spatial bins for each time bin of the event.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds; strictly increasing for
        :func:`detect_trajectory_radon`.
    score : callable, default=detect_trajectory_radon
        The replay score: a function called as ``score(env, posterior, times,
        **score_options)`` that returns a result with a ``score`` attribute,
        larger for a better trajectory. :func:`detect_trajectory_radon`,
        :func:`weighted_correlation`, :func:`fit_linear_trajectory` and
        :func:`fit_isotonic_trajectory` all qualify; the first two are
        scored in batches (also when wrapped in :func:`functools.partial`).
        A score that takes ``rng`` gets one derived from ``rng`` unless
        ``score_options`` sets it, so sampled fits are reproducible too; the
        event and every shuffle then draw the same uniforms.
    score_options : Mapping[str, Any] or None, default=None
        Keyword arguments for ``score``, for example ``{"band": 10.0}`` for
        :func:`detect_trajectory_radon`.
    nulls : str, sequence of str, or None, default=None
        The nulls to run, by the names above. ``None`` runs the nulls of
        Davidson et al. (2009) whose inputs are given: ``"column_cycle"``
        always, ``"unit_identity"`` with the decoding inputs and
        ``"pseudo_event"`` with a pool. For one template's share of a
        :func:`~neurospatial.decoding.decode_jointly` posterior, whose rows
        sum to less than 1, only ``"column_cycle"`` applies.
    spike_counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons)
        Spike counts of the event, one row per posterior row, for the nulls
        that decode again.
    encoding_models : array of shape (n_neurons, n_bins), SpatialRatesResult, or None
        The rate maps ``posterior`` was decoded with, as passed to
        :func:`~neurospatial.decoding.decode_position`.
    dt : float or None
        Time-bin duration in seconds that ``posterior`` was decoded with.
    prior : NDArray[np.float64] or None
        The prior ``posterior`` was decoded with, if not the default.
    pseudo_event_pool : NDArray[np.float64] or None, shape (n_pool, n_bins)
        Posterior rows of the session's other candidate events, stacked, for
        the pseudo-event null. Each pseudo-event draws its time bins at
        random, with replacement, from these rows. Leave out the tested
        event's own rows.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        The side of the null distribution tested. ``"greater"`` asks whether
        the event scores higher than its shuffles.
    use_absolute : bool or None, default=None
        Test the absolute value of the score, so forward and reverse
        sequences count alike. ``None`` tests the absolute value for
        :func:`weighted_correlation`, whose sign is the replay direction, as
        Grosmark & Buzsáki (2016) and Tirole et al. (2022) do, and the score
        itself otherwise (the replay line score and R² are not signed).
    n_shuffles : int, default=1000
        Shuffles per null. A p-value is at least ``1 / (n_shuffles + 1)``.
        Davidson et al. (2009) used 1500, Tirole et al. (2022) 1000.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed; the nulls draw from it in the order
        listed above. Pass a seed for a reproducible result.

    Returns
    -------
    ReplaySignificanceResult
        The event's score result, one shuffle test per null run, and the
        largest of their p-values.

    Raises
    ------
    ValueError
        For inputs ``score`` refuses; if a null in ``nulls`` is unknown or
        lacks its inputs; if only some of ``spike_counts``,
        ``encoding_models`` and ``dt`` are given; if decoding
        ``spike_counts`` with ``encoding_models``, ``dt`` and ``prior`` does
        not reproduce ``posterior``; if ``pseudo_event_pool`` does not hold
        probability rows with one column per bin; if ``n_shuffles`` is not a
        positive integer; or if the event's score is NaN.
    TypeError
        If ``score_options`` holds an argument ``score`` does not take.

    See Also
    --------
    detect_trajectory_radon : The Davidson et al. (2009) replay line.
    weighted_correlation : Weighted correlation of time and position.
    neurospatial.replay.shuffle_posterior_circular : The column-cycle shuffle.
    neurospatial.replay.shuffle_cell_identity : The unit-identity shuffle.

    Notes
    -----
    The p-values are ``(k + 1) / (n + 1)``, where ``k`` counts shuffles at
    least as extreme as the event (a Monte Carlo p-value; Davison & Hinkley
    1997; Phipson & Smyth 2010). Shuffles whose score is NaN are left out.
    Requiring every null to reject an event, and reporting the largest
    p-value, follows Davidson et al. (2009), who called an event significant
    replay when all three p-values were below 0.01, and Tirole et al. (2022)
    and Takigawa et al. (2024), who use the same rule.

    The unit-identity null decodes the shuffled counts the way ``posterior``
    was decoded, which is checked by decoding the unshuffled counts first.

    References
    ----------
    Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal
    replay of extended experience. *Neuron*, 63(4), 497-507.

    Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote
    experiences in the hippocampus. *Nature Neuroscience*, 12(7), 913-918.

    Grosmark, A. D., & Buzsáki, G. (2016). Diversity in neural firing
    dynamics supports both rigid and learned hippocampal sequences.
    *Science*, 351(6280), 1440-1443.

    Tirole, M., Huelin Gorriz, M., Takigawa, M., Kukovska, L., & Bendor, D.
    (2022). Experience-driven rate modulation is reinstated during
    hippocampal replay. *eLife*, 11, e79031.

    Takigawa, M., Huelin Gorriz, M., Tirole, M., & Bendor, D. (2024).
    Evaluating hippocampal replay without a ground truth. *eLife*, 13,
    e85635.

    Examples
    --------
    A posterior that sweeps along a 100 cm track, tested with the replay line
    of Davidson et al. and with the weighted correlation:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import replay_significance, weighted_correlation
    >>>
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=2.0, units="cm"
    ... )
    >>> times = np.arange(25) * 0.02
    >>> posterior = np.zeros((25, env.n_bins))
    >>> posterior[np.arange(25), 2 * np.arange(25)] = 1.0  # 4 cm per 20 ms
    >>> result = replay_significance(env, posterior, times, n_shuffles=199, rng=0)
    >>> print(f"{result.observed.speed:.0f} cm/s, p = {result.p_value:.3f}")
    202 cm/s, p = 0.005
    >>> result = replay_significance(
    ...     env, posterior, times, score=weighted_correlation, n_shuffles=199, rng=0
    ... )
    >>> print(f"r = {result.observed.score:.2f}, p = {result.p_value:.3f}")
    r = 1.00, p = 0.005
    """
    from neurospatial.replay.nulls import (
        shuffle_cell_identity,
        shuffle_place_fields_circular,
        shuffle_posterior_circular,
        shuffle_spike_counts_circular,
    )
    from neurospatial.stats import _ensure_rng
    from neurospatial.stats.shuffle import (
        ShuffleTestResult,
        compute_shuffle_pvalue,
        compute_shuffle_zscore,
    )

    if (
        isinstance(n_shuffles, bool)
        or not isinstance(n_shuffles, (int, np.integer))
        or n_shuffles < 1
    ):
        raise ValueError(
            f"{_OPERATION} needs n_shuffles to be a positive integer, got "
            f"{n_shuffles!r}. Davidson et al. used 1500 per null, Tirole et al. "
            "1000."
        )
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"{_OPERATION}: alternative must be one of {_ALTERNATIVES}, got "
            f"{alternative!r}."
        )
    score, options = _unwrap_partial(score, dict(score_options or {}))
    if use_absolute is None:
        use_absolute = score is weighted_correlation
    generator = _ensure_rng(rng)
    if _takes_rng(score) and "rng" not in options:
        options["rng"] = int(generator.integers(2**63 - 1))
    scorer = _scorer_for(score, env, posterior, times, options)
    run = _nulls_to_run(nulls, spike_counts, encoding_models, dt, pseudo_event_pool)
    moving = [name for name in run if name in _ROW_MOVING_NULLS]
    if "spike_counts" in options and moving:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: score_options holds spike_counts, but the "
                f"{moving} null{'s' if len(moving) > 1 else ''} change which "
                "time bins have spikes in each shuffle.",
                why="score_options is passed unchanged to every shuffle, so "
                "the spike counts would no longer match the shuffled rows",
                fix="drop the time bins without spikes, with their times, "
                "before testing, and leave spike_counts out of score_options",
            )
        )
    if "pseudo_event" in run:
        pool = _validated_pool(pseudo_event_pool, env)

    observed_result = scorer.observe()
    observed = float(_statistic(np.asarray([observed_result.score]), use_absolute)[0])
    if not np.isfinite(observed):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the event's score is {observed_result.score}.",
                why="a score that is not a number cannot be compared with the "
                "shuffles; the weighted correlation, for example, is undefined "
                "when all of the posterior mass is at one position or in one "
                "time bin",
                fix="check the event's posterior, or choose a score defined for it",
            )
        )
    posterior = scorer.posterior
    null_scores: dict[str, NDArray[np.float64]] = {}

    if "column_cycle" in run:
        shifted = shuffle_posterior_circular(
            env, posterior, n_shuffles=n_shuffles, rng=generator
        )
        null_scores["column_cycle"] = scorer.scores(shifted, n_shuffles)

    if any(name in run for name in _DECODING_NULLS):
        # _nulls_to_run runs these nulls only when all three inputs are given.
        assert encoding_models is not None and dt is not None
        counts = np.asarray(spike_counts)
        _require_reproducible_decode(env, counts, encoding_models, dt, prior, posterior)

        from neurospatial.decoding import decode_position

        def decode(
            event_counts: Any, models: Any = encoding_models, event_prior: Any = prior
        ) -> Any:
            """Decode shuffled counts or rate maps as the posterior was decoded."""
            return decode_position(
                env, event_counts, models, dt=dt, prior=event_prior
            ).posterior

    if "unit_identity" in run:
        rates = np.asarray(getattr(encoding_models, "firing_rates", encoding_models))
        permuted = (
            shuffled
            for shuffled, _ in shuffle_cell_identity(
                counts, rates, n_shuffles=n_shuffles, rng=generator
            )
        )
        null_scores["unit_identity"] = scorer.scores(map(decode, permuted), n_shuffles)

    if "pseudo_event" in run:
        draws = generator.integers(len(pool), size=(n_shuffles, len(posterior)))
        null_scores["pseudo_event"] = scorer.scores(
            (pool[rows] for rows in draws), n_shuffles
        )

    if "spike_train_circular" in run:
        shifted_counts = shuffle_spike_counts_circular(
            counts, n_shuffles=n_shuffles, rng=generator
        )
        null_scores["spike_train_circular"] = scorer.scores(
            map(decode, shifted_counts), n_shuffles
        )

    if "place_field_circular" in run:
        rates, support_prior = _supported_rates(env, encoding_models, prior)
        shifted_maps = shuffle_place_fields_circular(
            env, rates, n_shuffles=n_shuffles, rng=generator
        )
        null_scores["place_field_circular"] = scorer.scores(
            (
                decode(counts, np.nan_to_num(maps), support_prior)
                for maps in shifted_maps
            ),
            n_shuffles,
        )

    if "time_bin_permutation" in run:
        null_scores["time_bin_permutation"] = scorer.scores(
            (
                posterior[generator.permutation(len(posterior))]
                for _ in range(n_shuffles)
            ),
            n_shuffles,
        )

    shuffles = {}
    for name, scores in null_scores.items():
        statistic = _statistic(scores, use_absolute)
        shuffles[name] = ShuffleTestResult(
            observed_score=observed,
            null_scores=statistic,
            p_value=compute_shuffle_pvalue(observed, statistic, tail=alternative),
            z_score=compute_shuffle_zscore(observed, statistic),
            shuffle_type=name,
            n_shuffles=n_shuffles,
        )
    return ReplaySignificanceResult(
        observed=observed_result,
        shuffles=MappingProxyType(shuffles),
        p_value=max(test.p_value for test in shuffles.values()),
        alternative=alternative,
        use_absolute=use_absolute,
    )


def _statistic(scores: NDArray[np.float64], use_absolute: bool) -> NDArray[np.float64]:
    values = np.asarray(scores, dtype=np.float64)
    return np.abs(values) if use_absolute else values


def _nulls_to_run(
    nulls: Sequence[str] | None,
    spike_counts: Any,
    encoding_models: Any,
    dt: Any,
    pseudo_event_pool: Any,
) -> tuple[str, ...]:
    """The nulls to run, in the order the random numbers are drawn."""
    unit_inputs = {
        "spike_counts": spike_counts,
        "encoding_models": encoding_models,
        "dt": dt,
    }
    missing = [name for name, value in unit_inputs.items() if value is None]
    if 0 < len(missing) < len(unit_inputs):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the nulls that decode the event again "
                "(unit_identity, spike_train_circular, place_field_circular) "
                f"need spike_counts, encoding_models and dt, but "
                f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                "missing.",
                why="these nulls decode shuffled spike counts or shifted rate "
                "maps the way the posterior was decoded",
                fix="pass all three (and prior if the decode used one), or none "
                "to run only the nulls that need the posterior alone",
            )
        )
    available = {
        "column_cycle": True,
        "unit_identity": not missing,
        "pseudo_event": pseudo_event_pool is not None,
        "spike_train_circular": not missing,
        "place_field_circular": not missing,
        "time_bin_permutation": True,
    }
    if nulls is None:
        return tuple(name for name in _DEFAULT_NULLS if available[name])
    requested = [nulls] if isinstance(nulls, str) else list(nulls)
    unknown = [name for name in requested if name not in _NULLS]
    if unknown or not requested:
        raise ValueError(
            f"{_OPERATION}: nulls must name at least one of {_NULLS}, got "
            f"{requested!r}."
        )
    lacking = [name for name in requested if not available[name]]
    if lacking:
        decoding = "spike_counts, encoding_models and dt"
        needs = {
            "unit_identity": decoding,
            "pseudo_event": "pseudo_event_pool",
            "spike_train_circular": decoding,
            "place_field_circular": decoding,
        }
        raise ValueError(
            _format_error(
                f"{_OPERATION}: nulls asks for {lacking} without their inputs.",
                why="; ".join(f"{name} needs {needs[name]}" for name in lacking),
                fix="pass those inputs, or leave these nulls out",
            )
        )
    return tuple(name for name in _NULLS if name in requested)


def _require_reproducible_decode(
    env: Environment,
    counts: NDArray[Any],
    encoding_models: Any,
    dt: float,
    prior: Any,
    posterior: NDArray[np.float64],
) -> None:
    """Refuse decoding inputs that do not give back the event's posterior."""
    from neurospatial.decoding import decode_position

    decoded = decode_position(
        env, counts, encoding_models, dt=dt, prior=prior
    ).posterior
    if decoded.shape != posterior.shape or not np.allclose(
        decoded, posterior, rtol=1e-5, atol=1e-7
    ):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: decoding spike_counts with encoding_models, dt "
                "and prior does not reproduce posterior.",
                why="the nulls that decode shuffled counts or rate maps are "
                "valid only if they decode them the way the posterior was "
                "decoded",
                fix="pass the counts, rate maps (or rate result), dt and prior "
                "that produced posterior",
            )
        )


def _supported_rates(
    env: Environment, encoding_models: Any, prior: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rate maps with NaN outside training support, and the matching prior.

    Decoding the zero-filled maps with this prior equals decoding
    ``encoding_models`` with ``prior``: bins without training support get
    no posterior mass.
    """
    rates = np.array(
        getattr(encoding_models, "firing_rates", encoding_models), dtype=np.float64
    )
    support = np.isfinite(rates).all(axis=0)
    spatial_support = getattr(encoding_models, "spatial_support", None)
    if spatial_support is not None:
        support &= np.asarray(spatial_support, dtype=bool)
    rates[:, ~support] = np.nan
    base = np.asarray(env.bin_sizes if prior is None else prior, dtype=np.float64)
    return rates, base * support


def _validated_pool(pool: Any, env: Environment) -> NDArray[np.float64]:
    from neurospatial.replay.trajectory import _require_probability_rows

    rows = np.asarray(pool, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != env.n_bins or len(rows) == 0:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: pseudo_event_pool has shape {rows.shape}, but "
                f"needs (n_pool, {env.n_bins}): posterior rows of other "
                "candidate events on this environment.",
                why="each pseudo-event draws its time bins from these rows",
                fix="stack the posteriors of the session's other candidate "
                "events, e.g. np.vstack([r.posterior for r in others])",
            )
        )
    return _require_probability_rows(rows, _OPERATION)


def _unwrap_partial(
    score: Callable[..., Any], options: dict[str, Any]
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Turn ``functools.partial(score, **kw)`` into ``score`` and its options."""
    if not isinstance(score, functools.partial):
        return score, options
    if score.args:
        raise TypeError(
            f"{_OPERATION}: score is a functools.partial with positional "
            "arguments; give the score's options as keywords, or in "
            "score_options."
        )
    return score.func, {**score.keywords, **options}


def _takes_rng(score: Callable[..., Any]) -> bool:
    try:
        return "rng" in inspect.signature(score).parameters
    except (TypeError, ValueError):
        return False


def _scorer_for(
    score: Callable[..., Any],
    env: Environment,
    posterior: Any,
    times: Any,
    options: dict[str, Any],
) -> _Scorer:
    """A scorer for ``score``: batched for the scores that support it."""
    batched: dict[
        Callable[..., Any], tuple[Callable[..., _Scorer], tuple[str, ...]]
    ] = {
        detect_trajectory_radon: (_RadonScorer, ("band", "max_speed")),
        weighted_correlation: (_WeightedCorrelationScorer, ("row_normalization",)),
    }
    if score in batched:
        scorer_class, accepted = batched[score]
        unknown = sorted(set(options) - set(accepted))
        if unknown:
            raise TypeError(
                f"{_OPERATION}: score_options has {unknown}, which "
                f"{score.__name__} does not take; it takes {list(accepted)}."
            )
        return scorer_class(env, posterior, times, **options)
    if not callable(score):
        raise TypeError(
            f"{_OPERATION}: score must be a function such as "
            f"detect_trajectory_radon, got {type(score).__name__}."
        )
    return _CallScorer(score, env, posterior, times, options)


class _Scorer:
    """Scores an event and its shuffles with one score and its options."""

    posterior: NDArray[np.float64]

    def observe(self) -> Any:
        """The score's result for the event itself."""
        raise NotImplementedError

    def scores(
        self, posteriors: Iterable[NDArray[np.float64]], n_shuffles: int
    ) -> NDArray[np.float64]:
        """The score of each shuffled posterior."""
        raise NotImplementedError


class _RadonScorer(_Scorer):
    def __init__(
        self,
        env: Environment,
        posterior: Any,
        times: Any,
        *,
        band: float | None = None,
        max_speed: float | None = None,
    ) -> None:
        self.lines, self.posterior = _prepare_replay_lines(
            env, posterior, times, band=band, max_speed=max_speed, operation=_OPERATION
        )

    def observe(self) -> Any:
        return _best_line(self.lines, self.posterior)

    def scores(
        self, posteriors: Iterable[NDArray[np.float64]], n_shuffles: int
    ) -> NDArray[np.float64]:
        return self.lines.best_scores_of(posteriors, n_shuffles)


class _WeightedCorrelationScorer(_Scorer):
    _BATCH = 256

    def __init__(
        self,
        env: Environment,
        posterior: Any,
        times: Any,
        *,
        row_normalization: str = "per_time_bin",
    ) -> None:
        self.positions = _track_bin_centers(env, _OPERATION)
        self.posterior = np.asarray(posterior, dtype=np.float64)
        self.row_normalization = row_normalization
        self.weights = _validated_weights(self.posterior, env.n_bins, row_normalization)
        self.times = _validated_times(times, len(self.weights))

    def observe(self) -> WeightedCorrelationResult:
        score = _weighted_correlations(self.weights[None], self.times, self.positions)
        return WeightedCorrelationResult(
            score=float(score[0]), row_normalization=self.row_normalization
        )

    def scores(
        self, posteriors: Iterable[NDArray[np.float64]], n_shuffles: int
    ) -> NDArray[np.float64]:
        out = np.empty(n_shuffles)
        batch: list[NDArray[np.float64]] = []
        filled = 0
        for shuffled in posteriors:
            batch.append(
                _validated_weights(
                    shuffled, len(self.positions), self.row_normalization
                )
            )
            if len(batch) == self._BATCH:
                out[filled : filled + len(batch)] = _weighted_correlations(
                    np.stack(batch), self.times, self.positions
                )
                filled += len(batch)
                batch = []
        if batch:
            out[filled : filled + len(batch)] = _weighted_correlations(
                np.stack(batch), self.times, self.positions
            )
        return out


class _CallScorer(_Scorer):
    def __init__(
        self,
        score: Callable[..., Any],
        env: Environment,
        posterior: Any,
        times: Any,
        options: dict[str, Any],
    ) -> None:
        self.score = score
        self.env = env
        self.posterior = np.asarray(posterior, dtype=np.float64)
        self.times = times
        self.options = options

    def observe(self) -> Any:
        result = self.score(self.env, self.posterior, self.times, **self.options)
        if not hasattr(result, "score"):
            raise TypeError(
                f"{_OPERATION}: score returned {type(result).__name__}, which has "
                "no score attribute; return a result with a score."
            )
        return result

    def scores(
        self, posteriors: Iterable[NDArray[np.float64]], n_shuffles: int
    ) -> NDArray[np.float64]:
        return np.fromiter(
            (
                self.score(self.env, shuffled, self.times, **self.options).score
                for shuffled in posteriors
            ),
            dtype=np.float64,
            count=n_shuffles,
        )
