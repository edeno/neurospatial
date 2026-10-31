"""Test a replay score against shuffled versions of the event.

:func:`replay_significance` scores a decoded candidate event with any replay
score, scores ``n_shuffles`` shuffled versions of the event with the same
score and search, and reports a Monte Carlo p-value for each null. An event
is called significant when every null rejects it, so the reported p-value is
the largest among the nulls run (Davidson et al. 2009; Tirole et al. 2022).
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._random import _entropy, _name_key, _stream
from neurospatial._results import ResultMixin, reduce_through_constructor
from neurospatial._validation import is_integer_at_least
from neurospatial.replay._inputs import (
    _CORRELATION_WHY,
    _FIT_WHY,
    _posterior_times,
    _probability_rows,
)
from neurospatial.replay.correlation import (
    WeightedCorrelationResult,
    _track_bin_centers,
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
    detect_trajectory_radon,
    regression_line_score,
)
from neurospatial.stats import _TAILS

if TYPE_CHECKING:
    from neurospatial.decoding.likelihood import _PoissonRates
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment
    from neurospatial.stats.shuffle import ShuffleTestResult

_OPERATION = "replay_significance()"


@dataclass(frozen=True)
class _NullSpec:
    """What one null of :func:`replay_significance` needs and changes.

    Attributes
    ----------
    needs : {"posterior", "decoding", "pool"}
        Inputs beyond the posterior: none; the decoding inputs
        (``spike_counts``, ``encoding_models``, ``dt``), as it decodes
        shuffled counts or rate maps again; or ``pseudo_event_pool``.
    pool_use : str
        Why it needs the pool, for the refusal when the pool is missing.
    default : bool
        Run when ``nulls=None``, if its inputs are given (Davidson et al.
        2009).
    moves_rows : bool
        A time bin no longer holds its own spikes after it.
    spreads_spikes : bool
        It moves each cell's spikes to other time bins of the event, so the
        time bins without spikes differ from shuffle to shuffle.
    rolls : bool
        Each shuffle is a row order and a circular shift of each row along
        the track, which :func:`dataset_values` tabulates.
    along_track : bool
        It shifts rows along the track, so it needs a linearized track.
    shareable : bool
        One draw (a permutation of cells, a set of shifted rate maps) can be
        shared by every event of a data set.
    options : Mapping[str, Any]
        The options it takes in ``null_options``, with their defaults.
    """

    needs: Literal["posterior", "decoding", "pool"] = "posterior"
    pool_use: str = ""
    default: bool = False
    moves_rows: bool = False
    spreads_spikes: bool = False
    rolls: bool = False
    along_track: bool = False
    shareable: bool = False
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


# Every null, in the order the random numbers are drawn.
_NULL_SPECS: Mapping[str, _NullSpec] = MappingProxyType(
    {
        "column_cycle": _NullSpec(
            default=True,
            rolls=True,
            along_track=True,
            options=MappingProxyType({"exclude_zero_shift": False}),
        ),
        "unit_identity": _NullSpec(
            needs="decoding",
            default=True,
            shareable=True,
            options=MappingProxyType({"active_only": False}),
        ),
        "pseudo_event": _NullSpec(
            needs="pool",
            pool_use="whose rows its time bins are drawn from",
            default=True,
            moves_rows=True,
        ),
        "spike_train_circular": _NullSpec(
            needs="decoding", moves_rows=True, spreads_spikes=True
        ),
        "place_field_circular": _NullSpec(needs="decoding", shareable=True),
        "time_bin_permutation": _NullSpec(moves_rows=True, rolls=True),
        "spike_time_uniform": _NullSpec(
            needs="decoding", moves_rows=True, spreads_spikes=True
        ),
        "column_cycle_weighted": _NullSpec(
            needs="pool",
            pool_use="whose mean row weights the peaks",
            rolls=True,
            along_track=True,
            options=MappingProxyType({"edge_exclusion": 0.0}),
        ),
    }
)
_NULLS = tuple(_NULL_SPECS)
_DEFAULT_NULLS = tuple(name for name, spec in _NULL_SPECS.items() if spec.default)
_ROW_MOVING_NULLS = tuple(name for name, spec in _NULL_SPECS.items() if spec.moves_rows)
_DECODING_NULLS = tuple(
    name for name, spec in _NULL_SPECS.items() if spec.needs == "decoding"
)
_COUNT_CHANGING_NULLS = tuple(
    name for name, spec in _NULL_SPECS.items() if spec.spreads_spikes
)
_POOL_NULLS = tuple(name for name, spec in _NULL_SPECS.items() if spec.needs == "pool")
_DECODING_INPUTS = "spike_counts, encoding_models and dt"
_ALTERNATIVES = _TAILS


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
    def p_value(self) -> float:
        """The largest p-value among ``shuffles``: every null must reject."""
        return max(test.p_value for test in self.shuffles.values())

    @property
    def n_shuffles_used(self) -> Mapping[str, int]:
        """Shuffles with a defined score per null: those its p-value counts.

        Equal to ``n_shuffles`` unless some shuffles' scores are undefined
        (NaN); those are left out.
        """
        return MappingProxyType(
            {name: test.n_shuffles_used for name, test in self.shuffles.items()}
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
    _check_test_options(n_shuffles, alternative, empty_time_bins)
    decoded_posterior = np.asarray(posterior, dtype=np.float64)
    if empty_time_bins == "zero":
        posterior = _zeroed_event_rows(decoded_posterior, spike_counts)
    score, options = _unwrap_partial(score, dict(score_options or {}))
    spec = _score_specs().get(score)
    if use_absolute is None:
        use_absolute = spec is not None and spec.signed
    generator = np.random.default_rng(rng)
    if _takes_rng(score) and "rng" not in options:
        options["rng"] = int(generator.integers(2**63 - 1))
    track_env = _track_env(env, encoding_models, template)
    # One template's share of a joint decode keeps its joint weights only for
    # a score built for them; any other score sees the conditional posterior.
    conditional = template is not None and not _uses_joint_weights(score, options)
    if conditional:
        posterior = _conditional_rows(posterior)
    scorer = _scorer_for(score, track_env, posterior, times, options)
    run = _nulls_to_run(nulls, spike_counts, encoding_models, dt, pseudo_event_pool)
    options_of = _checked_null_options(null_options, run, _OPERATION)
    _check_nulls_fit_event(run, times, dt, options, empty_time_bins)
    pool = (
        _validated_pool(pseudo_event_pool, track_env)
        if any(name in run for name in _POOL_NULLS)
        else None
    )

    observed_result = scorer.observe()
    observed = _observed_statistic(observed_result, score, spec, use_absolute)
    decoder = None
    if any(name in run for name in _DECODING_NULLS):
        # _nulls_to_run runs these nulls only when all three inputs are given.
        assert encoding_models is not None and dt is not None
        decoder = _EventDecoder(env, encoding_models, dt, prior, template)
    event = _EventShuffles(
        track_env=track_env,
        posterior=scorer.posterior,
        generator=generator,
        n_shuffles=n_shuffles,
        options=options_of,
        pool=pool,
        decoder=decoder,
        counts=None if decoder is None else np.asarray(spike_counts),
        conditional=conditional,
        zero_empty=empty_time_bins == "zero",
    )
    null_scores = _null_scores(run, scorer, event, decoded_posterior)
    shuffles = _shuffle_tests(
        observed,
        {
            name: _statistic(scores, use_absolute)
            for name, scores in null_scores.items()
        },
        score=float(observed_result.score),
        alternative=alternative,
        n_shuffles=n_shuffles,
    )
    return ReplaySignificanceResult(
        observed=observed_result,
        shuffles=shuffles,
        alternative=alternative,
        use_absolute=use_absolute,
    )


def _check_test_options(
    n_shuffles: Any, alternative: Any, empty_time_bins: Any
) -> None:
    """Refuse a shuffle count, alternative or empty-bin rule that is not offered."""
    if not is_integer_at_least(n_shuffles, 1):
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


def _zeroed_event_rows(
    posterior: NDArray[np.float64], spike_counts: NDArray[Any] | None
) -> NDArray[np.float64]:
    """The event's posterior with ``empty_time_bins="zero"`` applied."""
    if spike_counts is None:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: empty_time_bins='zero' needs spike_counts.",
                why="the time bins whose posterior is set to zero are those "
                "without spikes",
                fix="pass the event's spike_counts, one row per posterior row",
            )
        )
    return _zero_empty_rows(posterior, np.asarray(spike_counts))


def _track_env(
    env: Environment | Mapping[str, Environment],
    encoding_models: Any,
    template: str | None,
) -> Environment:
    """The environment the scored posterior is on.

    Refuses a ``template`` without a mapping of ``encoding_models`` (or the
    reverse), and a mapping of environments without a joint decode.
    """
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
    if not isinstance(env, Mapping):
        return env
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
    return env[template]


def _check_nulls_fit_event(
    run: Sequence[str],
    times: NDArray[np.float64],
    dt: float | None,
    score_options: Mapping[str, Any],
    empty_time_bins: str,
) -> None:
    """Refuse nulls that cannot shuffle this event as it is given.

    The nulls that spread spikes need contiguous time bins; spike counts in
    ``score_options`` would not follow the nulls that move rows; and the
    pseudo-event null has no spike counts to zero its empty time bins with.
    """
    spreading = [name for name in run if name in _COUNT_CHANGING_NULLS]
    if spreading:
        assert dt is not None  # _nulls_to_run needs the decoding inputs.
        _require_contiguous_time_bins(
            times, dt, spreading, overlap="spike_time_uniform" in run
        )
    moving = [name for name in run if name in _ROW_MOVING_NULLS]
    if "spike_counts" in score_options and moving:
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


def _observed_statistic(
    observed_result: Any,
    score: Callable[..., Any],
    spec: _ScoreSpec | None,
    use_absolute: bool,
) -> float:
    """The event's tested statistic, refusing an undefined (NaN) score."""
    observed = float(_statistic(np.asarray([observed_result.score]), use_absolute)[0])
    if np.isfinite(observed):
        return observed
    example = (
        spec.undefined
        if spec is not None and spec.undefined
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


@dataclass(frozen=True)
class _EventShuffles:
    """What the nulls of :func:`replay_significance` shuffle for one event.

    Attributes
    ----------
    track_env : Environment
        The environment the scored posterior is on.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        The event's posterior as it is scored: conditioned on the template
        and with empty time bins zeroed, if the event's is.
    generator : np.random.Generator
        The stream every null draws from, one null after the other.
    n_shuffles : int
        Shuffles per null.
    options : Mapping[str, Mapping[str, Any]]
        Each null's options, from :func:`_checked_null_options`.
    pool : NDArray[np.float64] or None, shape (n_pool, n_bins)
        The checked pseudo-event pool; None when no null run uses it.
    decoder : _EventDecoder or None
        Decodes shuffled counts and rate maps; None when no null run decodes
        again.
    counts : NDArray[np.int64] or None, shape (n_time_bins, n_neurons)
        The event's spike counts, for the nulls that decode again.
    conditional : bool
        Whether a shuffle's decoded posterior is conditioned on the template.
    zero_empty : bool
        Whether a shuffle's decoded time bins without spikes are zeroed.
    """

    track_env: Environment
    posterior: NDArray[np.float64]
    generator: np.random.Generator
    n_shuffles: int
    options: Mapping[str, Mapping[str, Any]]
    pool: NDArray[np.float64] | None
    decoder: _EventDecoder | None
    counts: NDArray[np.int64] | None
    conditional: bool
    zero_empty: bool

    def decoding(self) -> tuple[_EventDecoder, NDArray[np.int64]]:
        """The decoder and the event's counts, for a null that decodes again."""
        assert self.decoder is not None and self.counts is not None
        return self.decoder, self.counts

    def finish(self, decoded: Any, event_counts: Any) -> Any:
        """A shuffle's decoded posterior, conditioned and zeroed as the event's."""
        if self.conditional:
            decoded = _conditional_rows(decoded)
        if self.zero_empty:
            return _zero_empty_rows(decoded, np.asarray(event_counts))
        return decoded

    def decode_each(self, shuffled_counts: Iterable[Any]) -> Iterator[Any]:
        """Decode shuffled counts as the posterior was decoded."""
        decoder, _ = self.decoding()
        for decoded, event_counts in decoder.decode_each(shuffled_counts):
            yield self.finish(decoded, event_counts)


def _null_scores(
    run: Sequence[str],
    scorer: _Scorer,
    event: _EventShuffles,
    decoded_posterior: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """The score of every shuffle of each null in ``run``, in its order.

    The nulls draw from ``event.generator`` one after the other, in the
    order of ``run``. The column cycle needs only the posterior and is
    drawn first; the decoding inputs are checked against
    ``decoded_posterior``, the posterior as decoded, after it and before
    any other null.
    """
    first = [name for name in run[:1] if name == "column_cycle"]
    null_scores = {
        name: scorer.scores(_SHUFFLED_POSTERIORS[name](event), event.n_shuffles)
        for name in first
    }
    if event.decoder is not None:
        decoder, counts = event.decoding()
        _require_reproducible_decode(decoder.decode(counts), decoded_posterior)
    for name in run[len(first) :]:
        null_scores[name] = scorer.scores(
            _SHUFFLED_POSTERIORS[name](event), event.n_shuffles
        )
    return null_scores


def _column_cycle_shuffles(event: _EventShuffles) -> Iterable[NDArray[np.float64]]:
    """Each row shifted circularly along the track by a random distance."""
    from neurospatial.replay.nulls import shuffle_posterior_circular

    return shuffle_posterior_circular(
        event.track_env,
        event.posterior,
        n_shuffles=event.n_shuffles,
        rng=event.generator,
        **event.options["column_cycle"],
    )


def _unit_identity_shuffles(event: _EventShuffles) -> Iterable[NDArray[np.float64]]:
    """The event's counts decoded with the units' rate maps permuted."""
    from neurospatial.replay.nulls import shuffle_cell_identity

    decoder, counts = event.decoding()
    permuted = (
        shuffled
        for shuffled, _ in shuffle_cell_identity(
            counts,
            decoder.first_rates,
            **event.options.get("unit_identity", {}),
            n_shuffles=event.n_shuffles,
            rng=event.generator,
        )
    )
    return event.decode_each(permuted)


def _pseudo_event_shuffles(event: _EventShuffles) -> Iterable[NDArray[np.float64]]:
    """Each time bin's row drawn, with replacement, from the pool."""
    pool = event.pool
    assert pool is not None  # _validated_pool runs for every pool null.
    draws = event.generator.integers(
        len(pool), size=(event.n_shuffles, len(event.posterior))
    )
    return (pool[rows] for rows in draws)


def _spike_train_circular_shuffles(
    event: _EventShuffles,
) -> Iterable[NDArray[np.float64]]:
    """The event decoded with each cell's counts shifted circularly in time."""
    from neurospatial.replay.nulls import shuffle_spike_counts_circular

    _, counts = event.decoding()
    return event.decode_each(
        shuffle_spike_counts_circular(
            counts, n_shuffles=event.n_shuffles, rng=event.generator
        )
    )


def _place_field_circular_shuffles(
    event: _EventShuffles,
) -> Iterable[NDArray[np.float64]]:
    """The event decoded with each cell's rate map shifted along the track."""
    decoder, counts = event.decoding()
    return (
        event.finish(decoder.decode(counts, models), counts)
        for models in decoder.shifted_models(event.n_shuffles, event.generator)
    )


def _time_bin_permutation_shuffles(
    event: _EventShuffles,
) -> Iterable[NDArray[np.float64]]:
    """The event's rows in random order."""
    posterior, generator = event.posterior, event.generator
    return (
        posterior[generator.permutation(len(posterior))]
        for _ in range(event.n_shuffles)
    )


def _spike_time_uniform_shuffles(
    event: _EventShuffles,
) -> Iterable[NDArray[np.float64]]:
    """The event decoded with each cell's spikes spread uniformly over it."""
    from neurospatial.replay.nulls import shuffle_spike_counts_uniform

    _, counts = event.decoding()
    return event.decode_each(
        shuffle_spike_counts_uniform(
            counts, n_shuffles=event.n_shuffles, rng=event.generator
        )
    )


def _column_cycle_weighted_shuffles(
    event: _EventShuffles,
) -> Iterable[NDArray[np.float64]]:
    """Each row's peak moved to a position drawn from the pool's mean row."""
    from neurospatial.replay.nulls import shuffle_posterior_weighted_circular

    pool = event.pool
    assert pool is not None  # _validated_pool runs for every pool null.
    return shuffle_posterior_weighted_circular(
        event.track_env,
        event.posterior,
        _pool_mean(pool),
        **event.options["column_cycle_weighted"],
        n_shuffles=event.n_shuffles,
        rng=event.generator,
    )


# Each null's shuffled posteriors, keyed like _NULL_SPECS.
_SHUFFLED_POSTERIORS: Mapping[
    str, Callable[[_EventShuffles], Iterable[NDArray[np.float64]]]
] = MappingProxyType(
    {
        "column_cycle": _column_cycle_shuffles,
        "unit_identity": _unit_identity_shuffles,
        "pseudo_event": _pseudo_event_shuffles,
        "spike_train_circular": _spike_train_circular_shuffles,
        "place_field_circular": _place_field_circular_shuffles,
        "time_bin_permutation": _time_bin_permutation_shuffles,
        "spike_time_uniform": _spike_time_uniform_shuffles,
        "column_cycle_weighted": _column_cycle_weighted_shuffles,
    }
)


def _shuffle_tests(
    observed: float,
    null_statistics: Mapping[str, NDArray[np.float64]],
    *,
    score: float,
    alternative: Literal["greater", "less", "two-sided"],
    n_shuffles: int,
) -> dict[str, ShuffleTestResult]:
    """One shuffle test per null, leaving out shuffles with an undefined score.

    ``observed`` and ``null_statistics`` are the tested statistic (the
    absolute score when it is tested); ``score`` is the event's score,
    reported when a null has no defined shuffle. ``ShuffleTestResult.from_null``
    warns when it leaves out undefined shuffles.
    """
    from neurospatial.stats.shuffle import ShuffleTestResult

    shuffles = {}
    for name, statistic in null_statistics.items():
        if np.isnan(statistic).all():
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
        spec = _score_specs().get(base)
        use_absolute = spec is not None and spec.signed
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
    )
    return ReplaySignificanceResult(
        observed=results[best].observed,
        shuffles=shuffles,
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
    entropy = _entropy(rng)
    return {name: _stream(entropy, _name_key(name)) for name in names}


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
    missing = _missing_decoding_inputs(spike_counts, encoding_models, dt)
    requested = (
        None if nulls is None else ([nulls] if isinstance(nulls, str) else list(nulls))
    )
    wants_decoding = requested is None or any(
        name in _DECODING_NULLS for name in requested
    )
    if 0 < len(missing) < 3 and wants_decoding:
        _refuse_missing_decoding_inputs(
            _OPERATION,
            _DECODING_NULLS,
            missing,
            alternative="or none to run only the nulls that need the posterior alone",
        )
    given = {
        "posterior": True,
        "decoding": not missing,
        "pool": pseudo_event_pool is not None,
    }
    if requested is None:
        return tuple(name for name in _DEFAULT_NULLS if given[_NULL_SPECS[name].needs])
    unknown = [name for name in requested if name not in _NULLS]
    if unknown or not requested:
        raise ValueError(
            f"{_OPERATION}: nulls must name at least one of {_NULLS}, got "
            f"{requested!r}."
        )
    lacking = [name for name in requested if not given[_NULL_SPECS[name].needs]]
    if lacking:
        needs = {
            name: _DECODING_INPUTS
            if _NULL_SPECS[name].needs == "decoding"
            else f"pseudo_event_pool, {_NULL_SPECS[name].pool_use}"
            for name in lacking
        }
        raise ValueError(
            _format_error(
                f"{_OPERATION}: nulls asks for {lacking} without their inputs.",
                why="; ".join(f"{name} needs {needs[name]}" for name in lacking),
                fix="pass those inputs, or leave these nulls out",
            )
        )
    return tuple(name for name in _NULLS if name in requested)


def _missing_decoding_inputs(
    spike_counts: Any, encoding_models: Any, dt: Any
) -> list[str]:
    """The inputs of the nulls that decode again that are not given, by name."""
    return [
        name
        for name, value in (
            ("spike_counts", spike_counts),
            ("encoding_models", encoding_models),
            ("dt", dt),
        )
        if value is None
    ]


def _refuse_missing_decoding_inputs(
    operation: str,
    nulls: Sequence[str],
    missing: Sequence[str],
    *,
    alternative: str = "",
) -> NoReturn:
    """Refuse nulls that decode again without all of their inputs."""
    one = len(nulls) == 1
    raise ValueError(
        _format_error(
            f"{operation}: the {', '.join(nulls)} null{'' if one else 's'} "
            f"decode{'s' if one else ''} the event again and need"
            f"{'s' if one else ''} {_DECODING_INPUTS}, but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} missing.",
            why="these nulls decode shuffled spike counts or shifted rate maps "
            "the way the posterior was decoded",
            fix="pass all three (and prior if the decode used one)"
            + (f", {alternative}" if alternative else ""),
        )
    )


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
        # Nulls of other tests, such as the rank-order cell_order, take none.
        defaults = _NULL_SPECS[name].options if name in _NULL_SPECS else {}
        options = dict(given.get(name, {}))
        unknown = sorted(set(options) - set(defaults))
        if unknown:
            takes = (
                f"it takes {sorted(defaults)}" if defaults else "it takes no options"
            )
            raise ValueError(f"{owner}: null_options gives {name} {unknown}; {takes}.")
        if defaults:
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

    # Posterior entries of the shuffles decoded together by decode_each
    # (about 32 MB).
    _BATCH_VALUES = 4_000_000

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
        # The rate maps' likelihood terms and the prior, for an event length.
        self._prepared: tuple[int, _PoissonRates, NDArray[np.float64] | None] | None
        self._prepared = None

    @property
    def first_rates(self) -> NDArray[np.float64]:
        """Rate maps whose rows name the neurons, for the unit-identity shuffle."""
        models = self.encoding_models
        if self.template is not None:
            models = models[self.template]
        return np.asarray(getattr(models, "firing_rates", models))

    def decode(self, counts: Any, models: Any = None) -> NDArray[np.float64]:
        """The posterior of ``counts``, with ``models`` replacing the rate maps.

        With one set of rate maps this is ``decode_position(env, counts,
        encoding_models, dt=dt, prior=prior).posterior``, with the rate maps
        checked and their likelihood terms built once per event length
        rather than for every shuffle. ``models`` is a place-field shuffle's
        ``(rates, prior)`` from :meth:`shifted_models`: finite, non-negative
        rates whose unsupported bins the prior excludes.
        """
        from neurospatial.decoding import decode_jointly
        from neurospatial.decoding._support import validate_prior
        from neurospatial.decoding.likelihood import _PoissonRates
        from neurospatial.decoding.posterior import _validate_spike_counts

        if self.template is None:
            counts = np.asarray(counts)
            _validate_spike_counts(counts)
            n_time = len(counts)
            if models is None:
                rates, prior = self._rates_and_prior(n_time)
            else:
                maps, support_prior = models
                rates = _PoissonRates.from_rates(np.asarray(maps), float(self.dt))
                prior = validate_prior(
                    support_prior, n_time, self.env.n_bins, context="decode_position"
                )
            return _posterior(counts, rates, prior)
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

    def decode_each(
        self, shuffled: Iterable[NDArray[np.int64]]
    ) -> Iterator[tuple[NDArray[np.float64], NDArray[np.int64]]]:
        """Each shuffle's posterior, with its counts, as :meth:`decode` gives it.

        With one set of rate maps, the shuffles' counts are stacked and
        decoded together, a batch at a time: the posterior of a time bin
        depends only on its own counts. Each posterior is a view of its
        batch.
        """
        from neurospatial.decoding.posterior import _validate_spike_counts

        if self.template is not None:
            for counts in shuffled:
                yield self.decode(counts), counts
            return

        def decoded(
            batch: list[NDArray[np.int64]],
        ) -> Iterator[tuple[NDArray[np.float64], NDArray[np.int64]]]:
            n_time = len(batch[0])
            stacked = np.concatenate(batch)
            _validate_spike_counts(stacked)
            rates, prior = self._rates_and_prior(n_time)
            if prior is not None and prior.ndim == 2:  # one row per time bin
                prior = np.tile(prior, (len(batch), 1))
            posterior = _posterior(stacked, rates, prior)
            for index, counts in enumerate(batch):
                yield posterior[index * n_time : (index + 1) * n_time], counts

        batch: list[NDArray[np.int64]] = []
        size = 1
        for counts in shuffled:
            batch.append(np.asarray(counts))
            if len(batch) == 1:
                posterior_size = max(len(batch[0]), 1) * self.env.n_bins
                size = max(1, self._BATCH_VALUES // posterior_size)
            if len(batch) == size:
                yield from decoded(batch)
                batch = []
        if batch:
            yield from decoded(batch)

    def _rates_and_prior(
        self, n_time: int
    ) -> tuple[_PoissonRates, NDArray[np.float64] | None]:
        """The checked rate maps' likelihood terms and resolved prior.

        As :func:`decode_position` prepares them for ``n_time`` time bins,
        kept for the next call with as many.
        """
        from neurospatial.decoding._time_bins import validate_dt
        from neurospatial.decoding.posterior import _prepare_decode_inputs

        if self._prepared is None or self._prepared[0] != n_time:
            rates, prior = _prepare_decode_inputs(
                self.env,
                self.encoding_models,
                self.prior,
                n_time,
                dt=validate_dt(self.dt),
                validate=True,
                context="decode_position",
            )
            self._prepared = (n_time, rates, prior)
        return self._prepared[1], self._prepared[2]

    def shifted_models(
        self, n_shuffles: int, generator: np.random.Generator
    ) -> Iterator[Any]:
        """Rate maps for the place-field shuffle, shifted over supported bins."""
        from neurospatial.replay.nulls import shuffle_place_fields_circular

        if self.template is None:
            rates, support_prior = _decode_support(
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
                _decode_support(envs[name], self.encoding_models[name], None)[0],
                n_shuffles=n_shuffles,
                rng=generator,
            )
            for name in names
        }
        for _ in range(n_shuffles):
            yield {name: next(streams[name]) for name in names}


def _posterior(
    counts: NDArray[np.int64],
    rates: _PoissonRates,
    prior: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """The posterior of ``counts``, decoded as :func:`decode_position` does."""
    from neurospatial.decoding.posterior import _posterior_blocks

    ((_, _, posterior, _),) = _posterior_blocks(
        lambda start, stop: counts[start:stop],
        len(counts),
        rates,
        prior,
        time_chunk=max(len(counts), 1),
        dtype=np.float64,
    )
    return posterior


def _decode_support(
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


@dataclass
class _CheckedPool:
    """A pseudo-event pool shared by many tests, with its rows once checked."""

    pool: Any
    rows: NDArray[np.float64] | None = None
    mean: NDArray[np.float64] | None = None


# Inside _checking_pool_once(pool): that pool, and its rows once checked.
_CHECKED_POOL: ContextVar[_CheckedPool | None] = ContextVar(
    "_CHECKED_POOL", default=None
)


@contextlib.contextmanager
def _checking_pool_once(pool: Any) -> Iterator[None]:
    """Check ``pool``'s rows, and take its mean row, once inside this block.

    A caller that runs many tests on one pseudo-event pool, such as
    :func:`compare_replay_methods` testing every event with every method,
    runs them inside this block, which the pool must not change in: a test
    given this same pool object reuses the first check that passed instead
    of checking every row again. A pool that fails is checked, and refused,
    by every test, as outside the block.
    """
    token = _CHECKED_POOL.set(_CheckedPool(pool))
    try:
        yield
    finally:
        _CHECKED_POOL.reset(token)


def _validated_pool(pool: Any, env: Environment) -> NDArray[np.float64]:
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
    shared = _CHECKED_POOL.get()
    if shared is None or shared.pool is not pool:
        return _probability_rows(rows, _OPERATION, sums="one")
    if shared.rows is None:
        shared.rows = _probability_rows(rows, _OPERATION, sums="one")
    return shared.rows


def _pool_mean(pool: NDArray[np.float64]) -> NDArray[np.float64]:
    """The mean row of a checked pool, taken once inside :func:`_checking_pool_once`."""
    shared = _CHECKED_POOL.get()
    if shared is None or shared.rows is not pool:
        mean: NDArray[np.float64] = pool.mean(axis=0)
        return mean
    if shared.mean is None:
        shared.mean = pool.mean(axis=0)
    return shared.mean


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
    spec = _score_specs().get(score)
    return (
        spec is not None
        and bool(spec.joint_weights)
        and all(
            options.get(name) == value for name, value in spec.joint_weights.items()
        )
    )


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


@dataclass(frozen=True)
class _ScoreSpec:
    """How the tests treat one of the package's replay scores.

    Attributes
    ----------
    scorer : callable
        The class that scores an event and its shuffles in batches, called
        as ``scorer(env, posterior, times, **score_options)``. It takes the
        options the score's signature takes.
    signed : bool
        Whether the score's sign is the replay direction, so its absolute
        value is tested by default.
    undefined : str
        When the score is undefined, for the refusal of such an event.
    joint_weights : Mapping[str, Any]
        Options under which the score reads a template's share of a joint
        decode as weights, rather than rows that sum to 1.
    moment_tables : Mapping[str, Any] or None
        Options, with their defaults, under which :func:`dataset_values`
        measures the score from moment tables; None when it cannot.
    """

    scorer: Callable[..., _Scorer]
    signed: bool = False
    undefined: str = ""
    joint_weights: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    moment_tables: Mapping[str, Any] | None = None


@functools.cache
def _score_specs() -> Mapping[Callable[..., Any], _ScoreSpec]:
    """The scores the tests know, by function."""
    # circular.py builds its scorer on _Scorer, so it is imported here.
    from neurospatial.replay.circular import (
        _CircularSequenceScorer,
        fit_circular_sequence,
    )

    return MappingProxyType(
        {
            detect_trajectory_radon: _ScoreSpec(_RadonScorer),
            weighted_correlation: _ScoreSpec(
                _WeightedCorrelationScorer,
                signed=True,
                undefined="the weighted correlation is undefined when all of "
                "the posterior mass is at one position or in one time bin",
                joint_weights=MappingProxyType({"row_normalization": "none"}),
                moment_tables=MappingProxyType({"row_normalization": "per_time_bin"}),
            ),
            regression_line_score: _ScoreSpec(_RegressionLineScorer),
            fit_circular_sequence: _ScoreSpec(_CircularSequenceScorer),
        }
    )


def _scorer_for(
    score: Callable[..., Any],
    env: Environment,
    posterior: Any,
    times: Any,
    options: dict[str, Any],
) -> _Scorer:
    """A scorer for ``score``: batched for the scores that support it."""
    spec = _score_specs().get(score)
    if spec is not None:
        accepted = list(inspect.signature(score).parameters)[3:]
        unknown = sorted(set(options) - set(accepted))
        if unknown:
            raise TypeError(
                f"{_OPERATION}: score_options has {unknown}, which "
                f"{score.__name__} does not take; it takes {accepted}."
            )
        return spec.scorer(env, posterior, times, **options)
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
        self.times = _posterior_times(
            times, len(self.weights), _OPERATION, why=_CORRELATION_WHY
        )

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
        self.times = _posterior_times(
            times, len(self.posterior), _OPERATION, why=_FIT_WHY, distinct=True
        )

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
            rows = _probability_rows(shuffled, _OPERATION, sums="one_or_zero")
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
