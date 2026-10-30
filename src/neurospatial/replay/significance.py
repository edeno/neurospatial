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
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ResultMixin, reduce_through_constructor
from neurospatial.replay.correlation import (
    WeightedCorrelationResult,
    _track_bin_centers,
    _validated_times,
    _validated_weights,
    _weighted_correlations,
    weighted_correlation,
)
from neurospatial.replay.trajectory import (
    _TRACK_ALTERNATIVE,
    RegressionLineScoreResult,
    _best_line,
    _prepare_replay_lines,
    _regression_line_scores,
    _require_probability_rows,
    _require_times,
    detect_trajectory_radon,
    regression_line_score,
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
    "spike_time_uniform",
    "column_cycle_weighted",
)
# Run when nulls=None, if their inputs are given: Davidson et al. (2009).
_DEFAULT_NULLS = ("column_cycle", "unit_identity", "pseudo_event")
# Nulls after which a time bin no longer holds its own spikes.
_ROW_MOVING_NULLS = (
    "pseudo_event",
    "spike_train_circular",
    "time_bin_permutation",
    "spike_time_uniform",
)
# Nulls that decode shuffled spike counts again.
_DECODING_NULLS = (
    "unit_identity",
    "spike_train_circular",
    "place_field_circular",
    "spike_time_uniform",
)
# Nulls that move each cell's spikes to other time bins of the event, so
# the time bins without spikes differ from shuffle to shuffle.
_COUNT_CHANGING_NULLS = ("spike_train_circular", "spike_time_uniform")
# The options each null takes in null_options, with their defaults.
_NULL_OPTIONS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "column_cycle": MappingProxyType({"exclude_zero_shift": False}),
        "column_cycle_weighted": MappingProxyType({"edge_exclusion": 0.0}),
        "unit_identity": MappingProxyType({"active_only": False}),
    }
)
_ALTERNATIVES = ("greater", "less", "two-sided")


class _UndefinedScoreError(ValueError):
    """The event's own score is undefined (NaN), so it cannot be tested."""


class _UndefinedNullError(ValueError):
    """Every shuffle of a null has an undefined score, so it gives no p-value."""

    def __init__(self, message: str, *, score: float, null: str) -> None:
        super().__init__(message)
        self.score = score
        self.null = null


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
        Each holds the observed statistic, the statistic of every shuffle
        (NaN where a shuffle's score is undefined) and the Monte Carlo
        p-value ``(k + 1) / (n + 1)`` over the shuffles with a score.
    p_value : float
        The largest p-value among ``shuffles``: an event is significant when
        every null rejects it.
    alternative : {"greater", "less", "two-sided"}
        The side of the null distribution tested.
    use_absolute : bool
        Whether the absolute value of the score was tested.
    template : str or None
        For several templates decoded separately (``posterior`` given as a
        mapping), the template with the largest score, whose result is
        ``observed``; None for one posterior.

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
    template: str | None = None

    def __post_init__(self) -> None:
        """Freeze the mapping of shuffle tests."""
        object.__setattr__(self, "shuffles", MappingProxyType(dict(self.shuffles)))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mapping cannot pickle."""
        return reduce_through_constructor(self)

    @property
    def n_shuffles_used(self) -> Mapping[str, int]:
        """Shuffles with a defined score per null: those its p-value counts.

        Equal to ``n_shuffles`` unless some shuffles' scores are undefined
        (NaN); those are left out.
        """
        return MappingProxyType(
            {
                name: int(np.isfinite(test.null_scores).sum())
                for name, test in self.shuffles.items()
            }
        )

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result.

        Starts with the observed result's ``summary()``, or only its
        ``score`` for a custom score whose result has no ``summary()``. Each
        null adds its p-value, ``{null}_p_value``, and the shuffles it counts,
        ``{null}_n_shuffles_used``. Separately decoded templates add the
        ``template`` with the largest score.
        """
        summarize = getattr(self.observed, "summary", None)
        observed = (
            summarize() if callable(summarize) else {"score": self.observed.score}
        )
        used = self.n_shuffles_used
        values = {
            **observed,
            "p_value": self.p_value,
            **{f"{name}_p_value": test.p_value for name, test in self.shuffles.items()},
            **{f"{name}_n_shuffles_used": used[name] for name in self.shuffles},
            "alternative": self.alternative,
            "use_absolute": self.use_absolute,
        }
        if self.template is not None:
            values["template"] = self.template
        return values


def replay_significance(
    env: Environment | Mapping[str, Environment],
    posterior: NDArray[np.float64] | Mapping[str, NDArray[np.float64]],
    times: NDArray[np.float64],
    *,
    score: Callable[..., Any] = detect_trajectory_radon,
    score_options: Mapping[str, Any] | None = None,
    nulls: str | Sequence[str] | None = None,
    null_options: Mapping[str, Mapping[str, Any]] | None = None,
    spike_counts: NDArray[np.int64] | None = None,
    encoding_models: NDArray[np.float64] | SpatialRatesResult | None = None,
    dt: float | None = None,
    prior: NDArray[np.float64] | Mapping[str, NDArray[np.float64]] | None = None,
    template: str | None = None,
    pseudo_event_pool: NDArray[np.float64] | None = None,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    use_absolute: bool | None = None,
    empty_time_bins: Literal["keep", "zero"] = "keep",
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
      2022). Needs the inputs of ``"unit_identity"``; ``times`` must be at
      most ``dt`` apart, since a gap between time bins would make a shift
      by one time bin longer than one time step and leave part of the event
      without spikes. Pass every time bin of the event, including those
      without spikes.
    - ``"place_field_circular"``: each cell's rate map is shifted circularly
      along the track and the event decoded again, which keeps each cell's
      firing but moves its field (Grosmark & Buzsáki 2016; Tirole et al.
      2022). Needs the inputs of ``"unit_identity"``. Maps are shifted over
      their supported bins only, so bins without training data stay
      excluded; smoothed maps are shifted as they are (Grosmark & Buzsáki
      shifted unsmoothed maps and smoothed them again).
    - ``"time_bin_permutation"``: the posterior's time bins are put in random
      order (Karlsson & Frank 2009; Takigawa et al. 2024). This is the "time
      swap" of Silva et al. (2015) and Maboudi et al. (2018) and the
      "time-bin shuffle" of Grosmark & Buzsáki (2016) and Farooq et al.
      (2019). Needs only the posterior. The decoder treats time bins
      independently, so with a prior that is the same in every time bin this
      equals permuting the spike counts and decoding again (Maboudi et al.'s
      form); with a time-varying prior it also moves each time bin's prior.
    - ``"spike_time_uniform"``: each cell's spikes in the event are placed
      uniformly at random over the event window and the event decoded
      again, the "spike jittering" of Stella et al. (2019). Each cell keeps
      its count; all timing is broken: the order of cells, bursts and which
      cells fire together. On equal time bins this spreads each cell's
      count over the time bins as Multinomial(count, 1 / n_time_bins). Needs
      the inputs of ``"unit_identity"``; ``times`` must be spaced exactly
      ``dt`` apart (up to round-off): overlapping windows share spikes, and
      a gap between time bins is part of the event that could receive no
      spikes. Pass every time bin of the event, including those without
      spikes.
    - ``"column_cycle_weighted"``: as ``"column_cycle"``, but each time bin's
      peak is moved to a position drawn from the session's mean posterior,
      ``pseudo_event_pool.mean(axis=0)``, and the whole row is shifted
      circularly to put it there (Farooq et al. 2019). Peaks never land
      within ``edge_exclusion`` (see ``null_options``; default 0) of either
      end of the track; Farooq et al. used 10 cm. Needs
      ``pseudo_event_pool``.

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
    env : Environment or Mapping[str, Environment]
        The environment the posterior was decoded on: a 1-D environment or a
        linearized track. For one template's share of a joint decode over
        separate tracks, the mapping of environments passed to
        :func:`~neurospatial.decoding.decode_jointly` (see ``template``).
        For separately decoded templates on different environments, a
        mapping keyed like ``posterior``.
    posterior : NDArray[np.float64] or Mapping[str, NDArray[np.float64]]
        Posterior mass over spatial bins for each time bin of the event,
        shape (n_time_bins, n_bins). Or, for several templates decoded
        separately, such as the two running directions each decoded with
        :func:`~neurospatial.decoding.decode_position` on its own rate maps
        (not jointly), a mapping of template name to posterior: the tested
        statistic is then the largest over the templates, in the event and
        in every shuffle (Muessig et al. 2019; see Notes).
        ``encoding_models`` and ``prior`` are then mappings keyed like it,
        or one value for every template; ``template`` and
        ``pseudo_event_pool`` are not used, and ``alternative`` must be
        ``"greater"``.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds; strictly increasing for
        :func:`detect_trajectory_radon`.
    score : callable, default=detect_trajectory_radon
        The replay score: a function called as ``score(env, posterior, times,
        **score_options)`` that returns a result with a ``score`` attribute,
        larger for a better trajectory. :func:`detect_trajectory_radon`,
        :func:`weighted_correlation`, :func:`fit_linear_trajectory` and
        :func:`fit_isotonic_trajectory` all qualify; the first two are
        scored in batches (also when wrapped in :func:`functools.partial`),
        as is :func:`~neurospatial.replay.fit_circular_sequence`.
        A score that takes ``rng`` gets one derived from ``rng`` unless
        ``score_options`` sets it, so sampled fits are reproducible too; the
        event and every shuffle then draw the same uniforms; whether it takes
        ``rng`` is read from its signature.
    score_options : Mapping[str, Any] or None, default=None
        Keyword arguments for ``score``, for example ``{"band": 10.0}`` for
        :func:`detect_trajectory_radon`.
    nulls : str, sequence of str, or None, default=None
        The nulls to run, by the names above. ``None`` runs the nulls of
        Davidson et al. (2009) whose inputs are given: ``"column_cycle"``
        always, ``"unit_identity"`` with the decoding inputs and
        ``"pseudo_event"`` with a pool.
    null_options : Mapping[str, Mapping[str, Any]] or None, default=None
        Options of the nulls run, by null name:

        - ``"column_cycle"``: ``exclude_zero_shift=True`` draws each time
          bin's shift from 1 to ``n_bins - 1`` bins, so no time bin stays in
          place, as nelpy does (Maboudi et al. 2018); the default draws
          every circular shift, 0 included.
        - ``"column_cycle_weighted"``: ``edge_exclusion``, the distance in
          the environment's units within which no peak lands at either end
          of the track, for example
          ``{"column_cycle_weighted": {"edge_exclusion": 10.0}}`` (Farooq
          et al. 2019, in cm).
        - ``"unit_identity"``: ``active_only=True`` permutes only the cells
          that fire in the event among themselves (Bush et al. 2022);
          the default permutes every cell.
    spike_counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons)
        Spike counts of the event, one row per posterior row, for the nulls
        that decode again.
    encoding_models : array, SpatialRatesResult, Mapping of them, or None
        The rate maps ``posterior`` was decoded with, as passed to
        :func:`~neurospatial.decoding.decode_position`; or, for one
        template's share of a joint decode, the mapping of every template's
        rate maps passed to :func:`~neurospatial.decoding.decode_jointly`,
        with ``template`` naming the share. Shuffles are then decoded
        jointly too.
    dt : float or None
        Time-bin duration in seconds that ``posterior`` was decoded with.
    prior : array, Mapping of arrays, or None
        The prior ``posterior`` was decoded with, if not the default.
    template : str or None
        For a joint decode: the template whose share,
        ``decode_jointly(...).joint(template)``, is ``posterior``. ``env``
        may then be one environment or a mapping, as for
        :func:`~neurospatial.decoding.decode_jointly`.
    pseudo_event_pool : NDArray[np.float64] or None, shape (n_pool, n_bins)
        Posterior rows of the session's candidate events, stacked, for the
        pseudo-event null. Each pseudo-event draws its time bins at random,
        with replacement, from these rows. Davidson et al. (2009) and Michon
        et al. (2019) drew from the complete set of candidate events,
        including the tested one. The weighted column cycle draws each peak
        from the mean of these rows.
    alternative : {"greater", "less", "two-sided"}, default="greater"
        The side of the null distribution tested. ``"greater"`` asks whether
        the event scores higher than its shuffles.
    use_absolute : bool or None, default=None
        Test the absolute value of the score, so forward and reverse
        sequences count alike. ``None`` tests the absolute value for
        :func:`weighted_correlation`, whose sign is the replay direction, as
        Grosmark & Buzsáki (2016) and Tirole et al. (2022) do, and the score
        itself otherwise (the replay line score and R² are not signed).
    empty_time_bins : {"keep", "zero"}, default="keep"
        ``"zero"`` sets the posterior of every time bin without spikes to
        zero before scoring, in the event and in every shuffle, as the
        Bendor-lab analyses do (Tirole et al. 2022): for the nulls that decode
        again, the time bins without spikes are those of the shuffled counts,
        and for the column-cycle and time-bin nulls the zeroed rows move with
        their rows (under the weighted column cycle a zeroed row stays
        zero). Needs ``spike_counts`` and a score that accepts rows
        without mass, such as :func:`weighted_correlation` or
        :func:`detect_trajectory_radon` with ``zero_rows="line_median"``;
        pass the posterior as decoded, not zeroed. Not available with
        ``"pseudo_event"``, whose pool rows carry no spike counts.
    n_shuffles : int, default=1000
        Shuffles per null. A p-value is at least ``1 / (n_shuffles + 1)``,
        or ``2 / (n_shuffles + 1)`` with ``alternative="two-sided"``.
        Davidson et al. (2009) used 1500, Tirole et al. (2022) 1000.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed; the nulls draw from it in the order
        listed above. Pass a seed for a reproducible result. Separately
        decoded templates each draw from their own stream, derived from
        ``rng`` and the template's name, so the result does not depend on
        the order of the templates.

    Returns
    -------
    ReplaySignificanceResult
        The event's score result, one shuffle test per null run, and the
        largest of their p-values; for separately decoded templates, also
        the template with the largest score.

    Raises
    ------
    ValueError
        For inputs ``score`` refuses; if a null in ``nulls`` is unknown or
        lacks its inputs; if ``null_options`` names a null that is not run or
        an option the null does not take; if ``"spike_time_uniform"`` is run
        on time bins less than ``dt`` apart, or it or
        ``"spike_train_circular"`` on time bins more than ``dt`` apart; if
        only some of ``spike_counts``, ``encoding_models`` and ``dt`` are
        given; if decoding
        ``spike_counts`` with ``encoding_models``, ``dt`` and ``prior`` does
        not reproduce ``posterior``; if ``pseudo_event_pool`` does not hold
        probability rows with one column per bin; if ``n_shuffles`` is not a
        positive integer; if the event's score is NaN; or if every shuffle
        of a null has a NaN score.
    TypeError
        If ``score_options`` holds an argument ``score`` does not take, or
        the signature of ``score`` cannot be read.

    See Also
    --------
    detect_trajectory_radon : The Davidson et al. (2009) replay line.
    weighted_correlation : Weighted correlation of time and position.
    neurospatial.replay.shuffle_posterior_circular : The column-cycle shuffle.
    neurospatial.replay.shuffle_cell_identity : The unit-identity shuffle.
    neurospatial.replay.shuffle_spike_counts_uniform : The spike-time-uniform
        shuffle.
    neurospatial.replay.shuffle_posterior_weighted_circular : The weighted
        column-cycle shuffle.

    Notes
    -----
    The p-values are ``(k + 1) / (n + 1)``, where ``k`` counts shuffles at
    least as extreme as the event (a Monte Carlo p-value; Davison & Hinkley
    1997; Phipson & Smyth 2010). Shuffles whose score is undefined (NaN),
    such as the weighted correlation of a shuffle whose decoded position does
    not change, are left out of ``n``, ``k`` and the z-score, with a warning
    that names the null; ``n_shuffles_used`` reports how many each null
    kept. Counting them as less extreme than the event would make the
    p-value too small.
    Requiring every null to reject an event, and reporting the largest
    p-value, follows Davidson et al. (2009), who called an event significant
    replay when all three p-values were below 0.01, and Tirole et al. (2022)
    and Takigawa et al. (2024), who use the same rule.

    The unit-identity null decodes the shuffled counts the way ``posterior``
    was decoded, which is checked by decoding the unshuffled counts first.

    For templates decoded separately, the event's statistic is the largest
    over the templates, and each shuffle's statistic is the largest over
    the templates' shuffles, each template shuffled independently: the
    shuffles are not yoked to the template that fits the event best (Silva
    et al. 2015; Muessig et al. 2019). Taking the largest of several scores
    raises the event's statistic even without replay, so the null must take
    the same largest; comparing the event's largest score with one
    template's shuffles calls too many events replay (up to ``1 - (1 -
    alpha)**k`` of events without replay for ``k`` independent templates).
    A shuffle whose score is undefined in any template is left out.

    References
    ----------
    Davidson, T. J., Kloosterman, F., & Wilson, M. A. (2009). Hippocampal
    replay of extended experience. *Neuron*, 63(4), 497-507.

    Karlsson, M. P., & Frank, L. M. (2009). Awake replay of remote
    experiences in the hippocampus. *Nature Neuroscience*, 12(7), 913-918.

    Grosmark, A. D., & Buzsáki, G. (2016). Diversity in neural firing
    dynamics supports both rigid and learned hippocampal sequences.
    *Science*, 351(6280), 1440-1443.

    Silva, D., Feng, T., & Foster, D. J. (2015). Trajectory events across
    hippocampal place cells require previous experience. *Nature
    Neuroscience*, 18(12), 1772-1779.

    Maboudi, K., Ackermann, E., de Jong, L. W., Pfeiffer, B. E., Foster, D.,
    Diba, K., & Kemere, C. (2018). Uncovering temporal structure in
    hippocampal output patterns. *eLife*, 7, e34467.

    Farooq, U., Sibille, J., Liu, K., & Dragoi, G. (2019). Strengthened
    temporal coordination within pre-existing sequential cell assemblies
    supports trajectory replay. *Neuron*, 103(4), 719-733.

    Stella, F., Baracskay, P., O'Neill, J., & Csicsvari, J. (2019).
    Hippocampal reactivation of random trajectories resembling Brownian
    diffusion. *Neuron*, 102(2), 450-461.

    Tirole, M., Huelin Gorriz, M., Takigawa, M., Kukovska, L., & Bendor, D.
    (2022). Experience-driven rate modulation is reinstated during
    hippocampal replay. *eLife*, 11, e79031.

    Takigawa, M., Huelin Gorriz, M., Tirole, M., & Bendor, D. (2024).
    Evaluating hippocampal replay without a ground truth. *eLife*, 13,
    e85635.

    Muessig, L., Lasek, M., Varsavsky, I., Cacucci, F., & Wills, T. J.
    (2019). Coordinated emergence of hippocampal replay and theta sequences
    during post-natal development. *Current Biology*, 29(5), 834-840.

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
        shuffle_posterior_circular,
        shuffle_posterior_weighted_circular,
        shuffle_spike_counts_circular,
        shuffle_spike_counts_uniform,
    )
    from neurospatial.stats import _ensure_rng

    if isinstance(posterior, Mapping):
        return _max_over_templates(
            env,
            posterior,
            times,
            score=score,
            score_options=score_options,
            nulls=nulls,
            null_options=null_options,
            spike_counts=spike_counts,
            encoding_models=encoding_models,
            dt=dt,
            prior=prior,
            template=template,
            pseudo_event_pool=pseudo_event_pool,
            alternative=alternative,
            use_absolute=use_absolute,
            empty_time_bins=empty_time_bins,
            n_shuffles=n_shuffles,
            rng=rng,
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
    if empty_time_bins not in ("keep", "zero"):
        raise ValueError(
            f"{_OPERATION}: empty_time_bins must be 'keep' or 'zero', got "
            f"{empty_time_bins!r}."
        )
    decoded_posterior = np.asarray(posterior, dtype=np.float64)
    if empty_time_bins == "zero":
        if spike_counts is None:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: empty_time_bins='zero' needs spike_counts.",
                    why="the time bins whose posterior is set to zero are those "
                    "without spikes",
                    fix="pass the event's spike_counts, one row per posterior row",
                )
            )
        posterior = _zero_empty_rows(decoded_posterior, np.asarray(spike_counts))
    score, options = _unwrap_partial(score, dict(score_options or {}))
    if use_absolute is None:
        use_absolute = score is weighted_correlation
    generator = _ensure_rng(rng)
    if _takes_rng(score) and "rng" not in options:
        options["rng"] = int(generator.integers(2**63 - 1))
    joint = isinstance(encoding_models, Mapping)
    if joint != (template is not None):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: template and a mapping of encoding_models go together.",
                why="a joint decode's share is decoded again from every "
                "template's rate maps, and template names the share",
                fix="pass encoding_models={name: rates, ...} with template=name, "
                "or one set of rate maps without template",
            )
        )
    if isinstance(env, Mapping):
        # template is given exactly when the decode is joint (checked above).
        if template is None:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: a mapping of environments needs a joint decode.",
                    why="one environment per template describes a joint decode over "
                    "separate tracks",
                    fix="pass every template's rate maps as encoding_models and "
                    "template=name, or the one environment the posterior is on",
                )
            )
        track_env = env[template]
    else:
        track_env = env
    # One template's share of a joint decode keeps its joint weights only for
    # a score built for them; any other score sees the conditional posterior.
    conditional = template is not None and not _uses_joint_weights(score, options)
    if conditional:
        posterior = _conditional_rows(posterior)
    scorer = _scorer_for(score, track_env, posterior, times, options)
    run = _nulls_to_run(nulls, spike_counts, encoding_models, dt, pseudo_event_pool)
    options_of = _checked_null_options(null_options, run, _OPERATION)
    spreading = [name for name in run if name in _COUNT_CHANGING_NULLS]
    if spreading:
        assert dt is not None  # _nulls_to_run needs the decoding inputs.
        _require_contiguous_time_bins(
            times, dt, spreading, overlap="spike_time_uniform" in run
        )
    moving = [name for name in run if name in _ROW_MOVING_NULLS]
    if "spike_counts" in options and moving:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: score_options holds spike_counts, but the "
                f"{moving} null{'s' if len(moving) > 1 else ''} change which "
                "time bins have spikes in each shuffle.",
                why="score_options is passed unchanged to every shuffle, so "
                "the spike counts would no longer match the shuffled rows",
                fix="leave spike_counts out of score_options; to leave out "
                "time bins without spikes, drop them with their times before "
                "testing (except for "
                f"{' and '.join(_COUNT_CHANGING_NULLS)}, which spread the "
                "spikes over every time bin of the event)",
            )
        )
    if "pseudo_event" in run and empty_time_bins == "zero":
        raise ValueError(
            _format_error(
                f"{_OPERATION}: empty_time_bins='zero' cannot be used with "
                "the pseudo_event null.",
                why="pseudo-events draw posterior rows from other events, "
                "whose spike counts are not given, so their empty time bins "
                "cannot be zeroed",
                fix="leave pseudo_event out of nulls, or use empty_time_bins='keep'",
            )
        )
    if "pseudo_event" in run or "column_cycle_weighted" in run:
        pool = _validated_pool(pseudo_event_pool, track_env)

    observed_result = scorer.observe()
    observed = float(_statistic(np.asarray([observed_result.score]), use_absolute)[0])
    if not np.isfinite(observed):
        example = (
            "the weighted correlation is undefined when all of the posterior "
            "mass is at one position or in one time bin"
            if score is weighted_correlation
            else f"{getattr(score, '__name__', 'the score')} is undefined for "
            "this event, for example when the decoded position does not change"
        )
        raise _UndefinedScoreError(
            _format_error(
                f"{_OPERATION}: the event's score is {observed_result.score}.",
                why="a score that is not a number cannot be compared with the "
                f"shuffles; {example}",
                fix="check the event's posterior, or choose a score defined for it",
            )
        )
    posterior = scorer.posterior
    null_scores: dict[str, NDArray[np.float64]] = {}

    if "column_cycle" in run:
        shifted = shuffle_posterior_circular(
            track_env,
            posterior,
            n_shuffles=n_shuffles,
            rng=generator,
            **options_of["column_cycle"],
        )
        null_scores["column_cycle"] = scorer.scores(shifted, n_shuffles)

    if any(name in run for name in _DECODING_NULLS):
        # _nulls_to_run runs these nulls only when all three inputs are given.
        assert encoding_models is not None and dt is not None
        counts = np.asarray(spike_counts)
        decoder = _EventDecoder(env, encoding_models, dt, prior, template)
        _require_reproducible_decode(decoder.decode(counts), decoded_posterior)

        def decode(event_counts: Any, models: Any = None) -> Any:
            """Decode shuffled counts or rate maps as the posterior was decoded."""
            decoded = decoder.decode(event_counts, models)
            if conditional:
                decoded = _conditional_rows(decoded)
            if empty_time_bins == "zero":
                return _zero_empty_rows(decoded, np.asarray(event_counts))
            return decoded

    if "unit_identity" in run:
        permuted = (
            shuffled
            for shuffled, _ in shuffle_cell_identity(
                counts,
                decoder.first_rates,
                **options_of.get("unit_identity", {}),
                n_shuffles=n_shuffles,
                rng=generator,
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
        null_scores["place_field_circular"] = scorer.scores(
            (
                decode(counts, models)
                for models in decoder.shifted_models(n_shuffles, generator)
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

    if "spike_time_uniform" in run:
        spread_counts = shuffle_spike_counts_uniform(
            counts, n_shuffles=n_shuffles, rng=generator
        )
        null_scores["spike_time_uniform"] = scorer.scores(
            map(decode, spread_counts), n_shuffles
        )

    if "column_cycle_weighted" in run:
        weighted = shuffle_posterior_weighted_circular(
            track_env,
            posterior,
            pool.mean(axis=0),
            **options_of["column_cycle_weighted"],
            n_shuffles=n_shuffles,
            rng=generator,
        )
        null_scores["column_cycle_weighted"] = scorer.scores(weighted, n_shuffles)

    shuffles = _shuffle_tests(
        observed,
        {
            name: _statistic(scores, use_absolute)
            for name, scores in null_scores.items()
        },
        score=float(observed_result.score),
        alternative=alternative,
        n_shuffles=n_shuffles,
        stacklevel=3,
    )
    return ReplaySignificanceResult(
        observed=observed_result,
        shuffles=shuffles,
        p_value=max(test.p_value for test in shuffles.values()),
        alternative=alternative,
        use_absolute=use_absolute,
    )


def _shuffle_tests(
    observed: float,
    null_statistics: Mapping[str, NDArray[np.float64]],
    *,
    score: float,
    alternative: Literal["greater", "less", "two-sided"],
    n_shuffles: int,
    stacklevel: int,
) -> dict[str, ShuffleTestResult]:
    """One shuffle test per null, leaving out shuffles with an undefined score.

    ``observed`` and ``null_statistics`` are the tested statistic (the
    absolute score when it is tested); ``score`` is the event's score,
    reported when a null has no defined shuffle. ``stacklevel`` points the
    warning about undefined shuffles at the caller of the public function.
    """
    from neurospatial.stats.shuffle import ShuffleTestResult

    shuffles = {}
    for name, statistic in null_statistics.items():
        defined = np.isfinite(statistic)
        n_used = int(defined.sum())
        if n_used == 0:
            raise _UndefinedNullError(
                _format_error(
                    f"{_OPERATION}: all {n_shuffles} {name} shuffles have an "
                    "undefined (NaN) score.",
                    why="the p-value compares the event with its shuffles, and "
                    "this null left none to compare with",
                    fix=f"leave {name} out of nulls, or check that the score is "
                    "defined on shuffled versions of this event",
                ),
                score=score,
                null=name,
            )
        if n_used < n_shuffles:
            warnings.warn(
                f"{_OPERATION}: {n_shuffles - n_used} of {n_shuffles} {name} "
                "shuffles have an undefined (NaN) score and are left out; the "
                f"{name} p-value counts the other {n_used} (n_shuffles_used).",
                UserWarning,
                stacklevel=stacklevel,
            )
        shuffles[name] = ShuffleTestResult.from_null(
            observed, statistic, shuffle_type=name, tail=alternative
        )
    return shuffles


def _max_over_templates(
    env: Environment | Mapping[str, Environment],
    posteriors: Mapping[str, NDArray[np.float64]],
    times: NDArray[np.float64],
    *,
    score: Callable[..., Any],
    score_options: Mapping[str, Any] | None,
    nulls: str | Sequence[str] | None,
    null_options: Mapping[str, Mapping[str, Any]] | None,
    spike_counts: NDArray[np.int64] | None,
    encoding_models: Any,
    dt: float | None,
    prior: Any,
    template: str | None,
    pseudo_event_pool: NDArray[np.float64] | None,
    alternative: Literal["greater", "less", "two-sided"],
    use_absolute: bool | None,
    empty_time_bins: Literal["keep", "zero"],
    n_shuffles: int,
    rng: np.random.Generator | int | None,
) -> ReplaySignificanceResult:
    """The largest score over separately decoded templates, tested likewise.

    Each template is tested on its own random stream. The event's statistic
    is the largest of the templates' statistics, and shuffle ``i`` of the
    null is the largest of the templates' ``i``-th shuffles, which are drawn
    independently: not yoked to the template that fits the event best
    (Silva et al. 2015; Muessig et al. 2019).
    """
    names, envs, models, priors = _separate_inputs(
        env, posteriors, encoding_models, prior, _OPERATION
    )
    refused = [
        name
        for name, given in (
            ("template", template),
            ("pseudo_event_pool", pseudo_event_pool),
        )
        if given is not None
    ]
    if refused:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: {refused} cannot be used with templates decoded "
                "separately.",
                why="every template is tested and the largest score kept; a "
                "pseudo-event pool holds the rows of one environment's posterior",
                fix=f"leave out {refused}",
            )
        )
    if alternative != "greater":
        raise ValueError(
            _format_error(
                f"{_OPERATION}: templates decoded separately need "
                f"alternative='greater', got {alternative!r}.",
                why="the event's statistic is the largest over the templates, "
                "which is the best fit only when larger scores are better",
                fix="use alternative='greater' (with use_absolute=True for a "
                "signed score such as the weighted correlation)",
            )
        )
    if use_absolute is None:
        base, _ = _unwrap_partial(score, dict(score_options or {}))
        use_absolute = base is weighted_correlation
    streams = _template_streams(rng, names)
    results: dict[str, ReplaySignificanceResult] = {}
    scores: dict[str, float] = {}
    undefined: _UndefinedNullError | None = None
    with warnings.catch_warnings():
        # Undefined shuffles are counted, and warned about, after the maximum.
        warnings.filterwarnings("ignore", message=".*undefined \\(NaN\\) score")
        for name in names:
            try:
                results[name] = replay_significance(
                    envs[name],
                    posteriors[name],
                    times,
                    score=score,
                    score_options=score_options,
                    nulls=nulls,
                    null_options=null_options,
                    spike_counts=spike_counts,
                    encoding_models=models[name],
                    dt=dt,
                    prior=priors[name],
                    pseudo_event_pool=pseudo_event_pool,
                    alternative="greater",
                    use_absolute=use_absolute,
                    empty_time_bins=empty_time_bins,
                    n_shuffles=n_shuffles,
                    rng=streams[name],
                )
            except _UndefinedNullError as error:
                undefined = error
                scores[name] = error.score
            else:
                scores[name] = float(results[name].observed.score)
    statistics = {
        name: float(_statistic(np.asarray([value]), use_absolute)[0])
        for name, value in scores.items()
    }
    # The first template in the mapping's order wins a tie.
    best = max(names, key=statistics.__getitem__)
    if undefined is not None:
        # Every maximum over the templates' shuffles of that null is undefined.
        raise _UndefinedNullError(
            str(undefined), score=scores[best], null=undefined.null
        )
    shuffles = _shuffle_tests(
        statistics[best],
        {
            null: np.max(
                [results[name].shuffles[null].null_scores for name in names], axis=0
            )
            for null in results[best].shuffles
        },
        score=scores[best],
        alternative="greater",
        n_shuffles=n_shuffles,
        stacklevel=4,
    )
    return ReplaySignificanceResult(
        observed=results[best].observed,
        shuffles=shuffles,
        p_value=max(test.p_value for test in shuffles.values()),
        alternative="greater",
        use_absolute=use_absolute,
        template=best,
    )


def _separate_inputs(
    env: Any,
    posteriors: Mapping[str, Any],
    encoding_models: Any,
    prior: Any,
    operation: str,
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Each separately decoded template's name, environment, rate maps and prior.

    ``env``, ``encoding_models`` and ``prior`` are mappings keyed like
    ``posteriors`` or one value for every template; rate maps must be a
    mapping, since templates decoded on the same maps would be the same.
    """
    names = tuple(posteriors)
    shapes = {name: np.shape(posteriors[name]) for name in names}
    if (
        not names
        or any(not isinstance(name, str) for name in names)
        or any(len(shape) != 2 for shape in shapes.values())
    ):
        raise ValueError(
            _format_error(
                f"{operation}: separately decoded templates need a mapping of "
                f"template name to posterior, shape (n_time_bins, n_bins); got "
                f"shapes {shapes}.",
                why="each template's posterior is scored on its own",
                fix="pass {name: decode_position(...).posterior, ...}, one per "
                "template, all for the same time bins",
            )
        )
    if len({shape[0] for shape in shapes.values()}) > 1:
        raise ValueError(
            _format_error(
                f"{operation}: the templates' posteriors have different numbers "
                f"of time bins: {shapes}.",
                why="separately decoded templates decode the same event's time bins",
                fix="decode every template from the same spike counts and times",
            )
        )
    if encoding_models is not None and not isinstance(encoding_models, Mapping):
        raise ValueError(
            _format_error(
                f"{operation}: templates decoded separately need a mapping of "
                "encoding_models.",
                why="each template was decoded with its own rate maps, and the "
                "nulls that decode again decode each template with them",
                fix=f"pass encoding_models={{name: rates, ...}} keyed like "
                f"posterior, {list(names)}",
            )
        )
    per_template = []
    for argument, value in (
        ("env", env),
        ("encoding_models", encoding_models),
        ("prior", prior),
    ):
        if isinstance(value, Mapping) and set(value) != set(names):
            raise ValueError(
                _format_error(
                    f"{operation}: {argument} has keys {sorted(value)} but the "
                    f"templates are {sorted(names)}.",
                    why=f"each template is tested with its own {argument}",
                    fix=f"key {argument} by the template names of posterior, "
                    "or pass one value for every template",
                )
            )
        per_template.append(
            {name: value[name] for name in names}
            if isinstance(value, Mapping)
            else dict.fromkeys(names, value)
        )
    envs, models, priors = per_template
    return names, envs, models, priors


def _template_streams(
    rng: np.random.Generator | int | None, names: Sequence[str]
) -> dict[str, np.random.Generator]:
    """One random stream per template, derived from ``rng`` and its name.

    A template's stream does not depend on the other templates or their
    order.
    """
    import zlib

    entropy = (
        int(rng.integers(2**63))
        if isinstance(rng, np.random.Generator)
        else np.random.SeedSequence(rng).entropy
    )
    return {
        name: np.random.default_rng(
            np.random.SeedSequence(entropy, spawn_key=(zlib.crc32(name.encode()),))
        )
        for name in names
    }


def _zero_empty_rows(
    posterior: NDArray[np.float64], spike_counts: NDArray[Any]
) -> NDArray[np.float64]:
    """The posterior with the rows of time bins without spikes set to zero."""
    counts = np.asarray(spike_counts)
    if counts.ndim != 2 or len(counts) != len(posterior):
        raise ValueError(
            f"{_OPERATION}: spike_counts must have one row per posterior row "
            f"({len(posterior)}), got shape {counts.shape}."
        )
    empty = counts.sum(axis=1) == 0
    zeroed: NDArray[np.float64] = np.where(empty[:, np.newaxis], 0.0, posterior)
    return zeroed


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
    requested_names = (
        None if nulls is None else ([nulls] if isinstance(nulls, str) else list(nulls))
    )
    wants_decoding = requested_names is None or any(
        name in _DECODING_NULLS for name in requested_names
    )
    if 0 < len(missing) < len(unit_inputs) and wants_decoding:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the nulls that decode the event again "
                f"({', '.join(_DECODING_NULLS)}) "
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
        "spike_time_uniform": not missing,
        "column_cycle_weighted": pseudo_event_pool is not None,
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
            "spike_time_uniform": decoding,
            "column_cycle_weighted": "pseudo_event_pool, whose mean row "
            "weights the peaks",
        }
        raise ValueError(
            _format_error(
                f"{_OPERATION}: nulls asks for {lacking} without their inputs.",
                why="; ".join(f"{name} needs {needs[name]}" for name in lacking),
                fix="pass those inputs, or leave these nulls out",
            )
        )
    return tuple(name for name in _NULLS if name in requested)


def _checked_null_options(
    null_options: Mapping[str, Mapping[str, Any]] | None,
    nulls: Sequence[str],
    owner: str,
) -> dict[str, dict[str, Any]]:
    """Every option of ``nulls``, with defaults, after refusing unknown ones.

    Parameters
    ----------
    null_options : Mapping[str, Mapping[str, Any]] or None
        Options by null name, as given.
    nulls : sequence of str
        The nulls that are run.
    owner : str
        The function or method the options were given to, for errors.

    Returns
    -------
    dict[str, dict[str, Any]]
        For each null in ``nulls`` that takes options, all of its options.

    Raises
    ------
    ValueError
        If ``null_options`` is not a mapping of mappings, names a null that
        is not run or takes no options, or an option the null does not
        take, or gives an option an invalid value.
    """
    given = {} if null_options is None else null_options
    if not isinstance(given, Mapping) or not all(
        isinstance(options, Mapping) for options in given.values()
    ):
        raise ValueError(
            f"{owner}: null_options must map null names to mappings of "
            "options, such as {'column_cycle_weighted': {'edge_exclusion': "
            f"10.0}}}}, got {null_options!r}."
        )
    not_run = [name for name in given if name not in nulls]
    if not_run:
        raise ValueError(
            _format_error(
                f"{owner}: null_options names {not_run}, which "
                f"{'is' if len(not_run) == 1 else 'are'} not among the nulls "
                f"run, {list(nulls)}.",
                why="options of a null that is not run would be ignored silently",
                fix="add the null to nulls, or remove its options",
            )
        )
    checked = {}
    for name in nulls:
        defaults = _NULL_OPTIONS.get(name, {})
        options = dict(given.get(name, {}))
        unknown = sorted(set(options) - set(defaults))
        if unknown:
            takes = (
                f"it takes {sorted(defaults)}" if defaults else "it takes no options"
            )
            raise ValueError(f"{owner}: null_options gives {name} {unknown}; {takes}.")
        if name in _NULL_OPTIONS:
            checked[name] = {**defaults, **options}
    edge = checked.get("column_cycle_weighted", {}).get("edge_exclusion", 0.0)
    if (
        isinstance(edge, bool)
        or not isinstance(edge, (int, float, np.integer, np.floating))
        or not np.isfinite(edge)
        or edge < 0
    ):
        raise ValueError(
            f"{owner}: the edge_exclusion of column_cycle_weighted must be a "
            f"finite distance >= 0 in the environment's units, got {edge!r}."
        )
    for name, option in (
        ("unit_identity", "active_only"),
        ("column_cycle", "exclude_zero_shift"),
    ):
        flag = checked.get(name, {}).get(option, False)
        if not isinstance(flag, (bool, np.bool_)):
            raise ValueError(
                f"{owner}: the {option} of {name} must be True or False, got {flag!r}."
            )
    return checked


def _require_contiguous_time_bins(
    times: Any, dt: float, nulls: list[str], *, overlap: bool
) -> None:
    """Refuse time bins that leave gaps in the event, or overlap.

    The nulls in ``nulls`` move each cell's spikes over the event's time
    bins, so the bins must tile the event: time bins further apart than
    ``dt`` leave stretches of the event that cannot receive spikes. With
    ``overlap``, time bins closer than ``dt`` are refused too: their counts
    share spikes, so they cannot be spread as separate spikes.
    """
    centers = np.sort(np.asarray(times, dtype=np.float64).ravel())
    spacing = np.diff(centers)
    if not spacing.size:
        return
    plural = "s" if len(nulls) > 1 else ""
    # Allow round-off in times built as start + k * step.
    if overlap and np.min(spacing) < dt * (1.0 - 1e-6):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the spike_time_uniform null needs time bins that "
                f"do not overlap, but times are as little as {np.min(spacing):g} s "
                f"apart with dt={dt:g} s.",
                why="overlapping windows count the same spikes, so their counts "
                "cannot be spread over the event as separate spikes",
                fix="decode the event in time bins that do not overlap (a time "
                "step equal to dt) for this null, or make null events from the "
                "spike times with null_events(kind='spike_time_uniform')",
            )
        )
    if np.max(spacing) > dt * (1.0 + 1e-6):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: the {nulls} null{plural} need{'' if plural else 's'} "
                "time bins that cover the event without gaps, but times are up "
                f"to {np.max(spacing):g} s apart with dt={dt:g} s.",
                why="these nulls move each cell's spikes over the time bins "
                "given, so stretches of the event between them could receive "
                "no spikes, and a shift by one time bin would not be one time "
                "step",
                fix="pass every time bin of the event, including those without "
                "spikes (set their posterior to zero with "
                "empty_time_bins='zero' rather than dropping them), decoded in "
                "time bins a step of dt apart",
            )
        )


def _require_reproducible_decode(
    decoded: NDArray[np.float64], posterior: NDArray[np.float64]
) -> None:
    """Refuse decoding inputs that do not give back the event's posterior."""
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
                "that produced posterior; for one template's share of a joint "
                "decode, pass every template's rate maps and template=",
            )
        )


class _EventDecoder:
    """Decodes an event's counts the way its posterior was decoded.

    One set of rate maps decodes with :func:`decode_position`; a mapping of
    templates decodes with :func:`decode_jointly` and keeps ``template``'s
    share.
    """

    def __init__(
        self,
        env: Any,
        encoding_models: Any,
        dt: float,
        prior: Any,
        template: str | None,
    ) -> None:
        self.env = env
        self.encoding_models = encoding_models
        self.dt = dt
        self.prior = prior
        self.template = template

    @property
    def first_rates(self) -> NDArray[np.float64]:
        """Rate maps whose rows name the neurons, for the unit-identity shuffle."""
        models = self.encoding_models
        if self.template is not None:
            models = models[self.template]
        return np.asarray(getattr(models, "firing_rates", models))

    def decode(self, counts: Any, models: Any = None) -> NDArray[np.float64]:
        """The posterior of ``counts``, with ``models`` replacing the rate maps."""
        from neurospatial.decoding import decode_jointly, decode_position

        if self.template is None:
            event_models, event_prior = (
                models
                if models is not None
                else (
                    self.encoding_models,
                    self.prior,
                )
            )
            decoded: NDArray[np.float64] = decode_position(
                self.env, counts, event_models, dt=self.dt, prior=event_prior
            ).posterior
            return decoded
        import warnings

        with warnings.catch_warnings():
            # Shifted maps keep each template's untrained bins as NaN, which
            # decode_jointly excludes, as it did for the event itself.
            warnings.filterwarnings("ignore", message=".*zero posterior probability")
            result = decode_jointly(
                self.env,
                counts,
                self.encoding_models if models is None else models,
                dt=self.dt,
                prior=self.prior,
            )
        return np.asarray(result.joint(self.template))

    def shifted_models(
        self, n_shuffles: int, generator: np.random.Generator
    ) -> Iterator[Any]:
        """Rate maps for the place-field shuffle, shifted over supported bins."""
        from neurospatial.replay.nulls import shuffle_place_fields_circular

        if self.template is None:
            rates, support_prior = _supported_rates(
                self.env, self.encoding_models, self.prior
            )
            for maps in shuffle_place_fields_circular(
                self.env, rates, n_shuffles=n_shuffles, rng=generator
            ):
                yield np.nan_to_num(maps), support_prior
            return
        names = list(self.encoding_models)
        envs = (
            self.env
            if isinstance(self.env, Mapping)
            else dict.fromkeys(names, self.env)
        )
        streams = {
            name: shuffle_place_fields_circular(
                envs[name],
                _supported_rates(envs[name], self.encoding_models[name], None)[0],
                n_shuffles=n_shuffles,
                rng=generator,
            )
            for name in names
        }
        for _ in range(n_shuffles):
            yield {name: next(streams[name]) for name in names}


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


def _uses_joint_weights(score: Callable[..., Any], options: Mapping[str, Any]) -> bool:
    """Whether a score reads a template's joint weights rather than its rows.

    The weighted correlation with ``row_normalization="none"`` weighs each
    time bin by the template's share of it (Tirole et al. 2022). Every other
    score needs rows that sum to 1.
    """
    return score is weighted_correlation and options.get("row_normalization") == "none"


def _conditional_rows(posterior: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows divided by their sums: the posterior given the template.

    A row without mass (a time bin zeroed for having no spikes) stays zero.
    """
    posterior = np.asarray(posterior, dtype=np.float64)
    sums = posterior.sum(axis=1, keepdims=True)
    conditional: NDArray[np.float64] = np.divide(
        posterior, sums, out=np.zeros_like(posterior), where=sums > 0
    )
    return conditional


def _takes_rng(score: Callable[..., Any]) -> bool:
    """Whether ``score`` takes an ``rng`` keyword."""
    try:
        return "rng" in inspect.signature(score).parameters
    except (TypeError, ValueError) as error:
        raise TypeError(
            _format_error(
                f"{_OPERATION}: cannot read the signature of score {score!r} "
                f"({error}).",
                why="a score that takes rng gets one derived from the test's "
                "rng, so that a seeded test is reproducible, and only the "
                "signature tells whether it takes one",
                fix="wrap the score in a Python function, def score(env, "
                "posterior, times, **options), with an rng keyword if it "
                "draws random numbers",
            )
        ) from error


def _scorer_for(
    score: Callable[..., Any],
    env: Environment,
    posterior: Any,
    times: Any,
    options: dict[str, Any],
) -> _Scorer:
    """A scorer for ``score``: batched for the scores that support it."""
    # circular.py builds its scorer on _Scorer, so it is imported here.
    from neurospatial.replay.circular import (
        _CircularSequenceScorer,
        fit_circular_sequence,
    )

    batched: dict[
        Callable[..., Any], tuple[Callable[..., _Scorer], tuple[str, ...]]
    ] = {
        detect_trajectory_radon: (
            _RadonScorer,
            (
                "band",
                "max_speed",
                "min_speed",
                "wrap",
                "line_ends",
                "off_track",
                "zero_rows",
                "n_lines",
                "line_seed",
            ),
        ),
        weighted_correlation: (_WeightedCorrelationScorer, ("row_normalization",)),
        regression_line_score: (_RegressionLineScorer, ("band", "wrap_band")),
        fit_circular_sequence: (_CircularSequenceScorer, ()),
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
        min_speed: float = 0.0,
        wrap: bool = False,
        line_ends: str = "any",
        off_track: str = "band_median",
        zero_rows: str = "refuse",
        n_lines: int | None = None,
        line_seed: int = 0,
    ) -> None:
        self.lines, self.posterior = _prepare_replay_lines(
            env,
            posterior,
            times,
            band=band,
            max_speed=max_speed,
            min_speed=min_speed,
            wrap=wrap,
            line_ends=line_ends,
            off_track=off_track,
            zero_rows=zero_rows,
            n_lines=n_lines,
            line_seed=line_seed,
            operation=_OPERATION,
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


class _RegressionLineScorer(_Scorer):
    _BATCH = 256

    def __init__(
        self,
        env: Environment,
        posterior: Any,
        times: Any,
        *,
        band: float | None = None,
        wrap_band: bool = False,
    ) -> None:
        from neurospatial.environment import _require_track_extents

        # Validates the inputs and resolves the default band.
        self.observed = regression_line_score(
            env, posterior, times, band=band, wrap_band=wrap_band
        )
        starts, stops = _require_track_extents(
            env, _OPERATION, alternative=_TRACK_ALTERNATIVE
        )
        self.order = np.argsort(starts, kind="stable")  # as in _track_positions
        self.starts, self.stops = starts[self.order], stops[self.order]
        self.posterior = np.asarray(posterior, dtype=np.float64)
        self.times = _require_times(times, len(self.posterior), _OPERATION)

    def observe(self) -> RegressionLineScoreResult:
        return self.observed

    def _score_batch(self, batch: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        return _regression_line_scores(
            np.stack(batch),
            self.times,
            self.starts,
            self.stops,
            self.observed.band,
            self.observed.wrap_band,
        )[0]

    def scores(
        self, posteriors: Iterable[NDArray[np.float64]], n_shuffles: int
    ) -> NDArray[np.float64]:
        out = np.empty(n_shuffles)
        batch: list[NDArray[np.float64]] = []
        filled = 0
        for shuffled in posteriors:
            rows = _require_probability_rows(shuffled, _OPERATION, allow_zero_rows=True)
            batch.append(rows[:, self.order])
            if len(batch) == self._BATCH:
                out[filled : filled + len(batch)] = self._score_batch(batch)
                filled += len(batch)
                batch = []
        if batch:
            out[filled : filled + len(batch)] = self._score_batch(batch)
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
