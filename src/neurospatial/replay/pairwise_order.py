"""Pairwise firing order of nearby spikes in an event (Gupta et al. 2010).

Gupta et al. (2010) scored a candidate event without decoding. Every pair of
spikes less than 80 ms apart, from two cells with place fields, adds +1 when
the cells fire in the order of their fields along the lap and -1 when they
fire in the reverse order. The sum is Kendall's concordant-minus-discordant
count over every spike, restricted to nearby pairs and not normalized. A cell
with several fields takes the field that makes the score largest. The event
is tested against shuffled spike times and shuffled cell identities.
:func:`pairwise_order_score` computes the score,
:func:`pairwise_order_significance` tests it, and :class:`PairwiseOrderMethod`
with :func:`detect_pairwise_order_replay` applies the published test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import (
    ReadOnlyArraysMixin,
    ResultMixin,
    reduce_through_constructor,
)
from neurospatial.replay.rank_order import _supported_rates, _validated_interval

if TYPE_CHECKING:
    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment
    from neurospatial.replay.methods import ReplayCriterion, ReplayDetectionResult
    from neurospatial.stats.shuffle import ShuffleTestResult

__all__ = [
    "PairwiseOrderMethod",
    "PairwiseOrderResult",
    "PairwiseOrderSignificanceResult",
    "detect_pairwise_order_replay",
    "pairwise_order_score",
    "pairwise_order_significance",
]

_DIRECTIONS = ("forward", "backward")
_NULLS = ("spike_time_permutation", "cell_identity")
_FIELD_CENTERS = ("rate_weighted", "peak")
# Most combinations of one field per active cell searched for one score.
_MAX_FIELD_ASSIGNMENTS = 2**16


class _FieldSearchTooLargeError(ValueError):
    """The exact field-assignment search of one score exceeds the cap."""


@dataclass(frozen=True, repr=False, eq=False)
class PairwiseOrderResult(ReadOnlyArraysMixin, ResultMixin):
    """Pairwise-order score of one candidate event.

    Attributes
    ----------
    score : int
        Over every pair of spikes less than ``max_time_lag`` apart from two
        cells with fields, +1 when they fire in the tested direction's
        order of the cells' fields and -1 when they fire against it (0 for
        equal times or equal field positions), summed. Each cell with
        several fields takes the field that makes the score largest. It
        lies in ``[-n_pairs, n_pairs]``.
    direction : {"forward", "backward"}
        The tested order: ``"forward"`` is the order of increasing position
        along the track, ``"backward"`` the reverse.
    n_pairs : int
        Pairs of spikes the score compares: from two different cells with
        fields, less than ``max_time_lag`` apart.
    n_active_cells : int
        Cells that fired in the event, with or without a field.
    scored_units : NDArray[np.intp], shape (n_scored_cells,)
        Index in ``spike_times`` of each cell that fired and has a field.
    assigned_positions : NDArray[np.float64], shape (n_scored_cells,)
        The field center each scored cell takes, along the track. With
        several best assignments, the one listed first when each cell's
        fields are taken in track order.

    See Also
    --------
    pairwise_order_score : Function that produces this result.
    """

    score: int
    direction: str
    n_pairs: int
    n_active_cells: int
    scored_units: NDArray[np.intp]
    assigned_positions: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "score": self.score,
            "direction": self.direction,
            "n_pairs": self.n_pairs,
            "n_active_cells": self.n_active_cells,
            "n_scored_cells": int(self.scored_units.size),
        }


@dataclass(frozen=True, repr=False, eq=False)
class PairwiseOrderSignificanceResult(ResultMixin):
    """Pairwise-order score of one event tested against shuffles.

    Attributes
    ----------
    observed : PairwiseOrderResult
        The event's score.
    shuffles : Mapping[str, ShuffleTestResult]
        One one-sided test per null, keyed by the null's name: the score of
        every shuffle and the Monte Carlo p-value ``(k + 1) / (n + 1)``,
        where ``k`` counts shuffles scoring at least as high as the event.
    p_value : float
        The largest p-value among ``shuffles``: the event must beat every
        null.

    See Also
    --------
    pairwise_order_significance : Function that produces this result.
    """

    observed: PairwiseOrderResult
    shuffles: Mapping[str, ShuffleTestResult]
    p_value: float

    def __post_init__(self) -> None:
        """Freeze the mapping of shuffle tests."""
        object.__setattr__(self, "shuffles", MappingProxyType(dict(self.shuffles)))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mapping cannot pickle."""
        return reduce_through_constructor(self)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            **self.observed.summary(),
            "p_value": self.p_value,
            **{f"{name}_p_value": test.p_value for name, test in self.shuffles.items()},
        }


@dataclass(frozen=True)
class PairwiseOrderMethod:
    """A published pairwise-order replay test (Gupta et al. 2010).

    Each test scores the event with :func:`pairwise_order_score` on one
    template (the rate maps of one lap or running direction) in one
    direction, against the shuffles of
    :func:`pairwise_order_significance`. Every template is tested in every
    direction and the test with the smallest p-value is reported.

    Attributes
    ----------
    name : str
        Short name, for example ``"gupta_2010"``.
    reference : str
        The paper the method follows.
    field_threshold : float
        A place field is a run of bins whose rate exceeds this fraction of
        the highest rate of any cell in any bin of the template (Gupta et
        al. 2010: 5%).
    merge_gap_bins : int
        Runs separated by at most this many bins below the threshold form
        one field (Gupta et al. 2010: 1, in 3.5 cm bins).
    field_center : {"rate_weighted", "peak"}
        Each field's position: its rate-weighted center or its peak bin.
    max_time_lag : float
        Spikes are compared when less than this far apart, in seconds
        (Gupta et al. 2010: 0.08).
    directions : tuple of {"forward", "backward"}
        The orders tested on each template.
    nulls : tuple of {"spike_time_permutation", "cell_identity"}
        The shuffles; the event must beat every one.
    n_shuffles : int
        Shuffles per null.
    alpha : float
        Significance level: the largest p-value over the nulls must be
        below it.
    min_active_cells : int
        Fewest cells that fire in the event, with or without a field.
    notes : tuple of str
        Choices the paper does not specify, and event-selection rules not
        applied.
    """

    name: str
    reference: str
    field_threshold: float = 0.05
    merge_gap_bins: int = 1
    field_center: Literal["rate_weighted", "peak"] = "rate_weighted"
    max_time_lag: float = 0.08
    directions: tuple[Literal["forward", "backward"], ...] = ("forward", "backward")
    nulls: tuple[Literal["spike_time_permutation", "cell_identity"], ...] = (
        "spike_time_permutation",
        "cell_identity",
    )
    n_shuffles: int = 300
    alpha: float = 0.05
    min_active_cells: int = 3
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse settings that cannot run."""
        prefix = f"PairwiseOrderMethod {self.name!r}"
        for name in ("directions", "nulls", "notes"):
            value = getattr(self, name)
            object.__setattr__(
                self, name, (value,) if isinstance(value, str) else tuple(value)
            )
        _check_field_options(
            prefix,
            field_threshold=self.field_threshold,
            merge_gap_bins=self.merge_gap_bins,
            field_center=self.field_center,
            max_time_lag=self.max_time_lag,
        )
        for name, allowed in (("directions", _DIRECTIONS), ("nulls", _NULLS)):
            values = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or any(value not in allowed for value in values)
            ):
                raise ValueError(
                    f"{prefix}: {name} must be one or more distinct values of "
                    f"{list(allowed)}, got {values!r}."
                )
        _check_n_shuffles(prefix, self.n_shuffles)
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
                f"got {value!r}; a score compares spikes of two cells."
            )

    @property
    def criteria(self) -> tuple[ReplayCriterion, ...]:
        """The active-cell threshold, as a criterion."""
        from neurospatial.replay.methods import ReplayCriterion

        return (ReplayCriterion("n_active_cells", ">=", self.min_active_cells),)


def pairwise_order_score(
    env: Environment,
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    direction: Literal["forward", "backward"] = "forward",
    max_time_lag: float = 0.08,
    field_threshold: float = 0.05,
    merge_gap_bins: int = 1,
    field_center: Literal["rate_weighted", "peak"] = "rate_weighted",
) -> PairwiseOrderResult:
    """Pairwise firing order of nearby spikes in an event (Gupta et al. 2010).

    Every pair of spikes less than ``max_time_lag`` apart, from two cells
    with place fields, adds +1 when the earlier spike's field comes first in
    ``direction`` along the track and -1 when it comes second. The score is
    the sum. A cell with several fields takes, for all its spikes, the field
    that makes the score largest.

    Parameters
    ----------
    env : Environment
        The template's environment: a 1-D environment or a linearized track,
        with position increasing in the direction the animal ran (one lap's
        linearization in Gupta et al.).
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the same order as the
        rows of ``encoding_models``. Any input
        :func:`~neurospatial.decoding.bin_spikes_in_time` accepts.
    interval : (start, stop)
        The candidate event, in seconds. Spikes at ``start <= t < stop``
        count.
    encoding_models : array of shape (n_cells, n_bins), or SpatialRatesResult
        Each cell's rate map on ``env``, in Hz. NaN marks bins without
        training data, which hold no field.
    direction : {"forward", "backward"}, default="forward"
        ``"forward"`` scores firing in the order of increasing field
        position, ``"backward"`` the reverse. With one field per cell the
        backward score is minus the forward score; with several fields each
        direction takes its own best fields.
    max_time_lag : float, default=0.08
        Spikes are compared when strictly less than this far apart, in
        seconds (Gupta et al. 2010: 80 ms). ``np.inf`` compares every pair.
    field_threshold : float, default=0.05
        A field is a run of bins, in track order, whose rate exceeds this
        fraction of the highest rate of any cell in any bin with support
        (Gupta et al. 2010: 5% of "the maximum rate observed over the
        session for any cell at any pixel").
    merge_gap_bins : int, default=1
        Runs separated by at most this many bins form one field, which
        spans the gap (Gupta et al. 2010: fields one 3.5 cm pixel apart
        were merged). 0 merges nothing.
    field_center : {"rate_weighted", "peak"}, default="rate_weighted"
        Each field's position: its rate-weighted center (the paper's
        "center", not defined there) or its peak bin.

    Returns
    -------
    PairwiseOrderResult
        The score, the number of pairs compared, the active cells and the
        field each scored cell takes.

    Raises
    ------
    ValueError
        If ``env`` is neither 1-D nor a linearized track, if
        ``encoding_models`` does not have one row per cell of
        ``spike_times`` and one column per bin, has negative rates, or is a
        rate result computed on another environment, if ``interval`` is not
        a finite ``(start, stop)`` with ``start < stop``, if an option is
        out of range, or if the active cells' fields give more than 65,536
        combinations of one field per cell to search.

    See Also
    --------
    pairwise_order_significance : Test the score against shuffles.
    rank_order_correlation : Spearman correlation of field and firing order.

    Notes
    -----
    The score is Kendall's concordant-minus-discordant count of field
    position against spike time, over all spikes, restricted to pairs less
    than ``max_time_lag`` apart and not normalized, so it grows with the
    number of nearby pairs. Two spikes of one cell, equal spike times and
    equal field positions add 0.

    Gupta et al. assigned each spike of a multi-field cell the field that
    maximized the score without saying how. Here all spikes of a cell take
    one field, and the best combination over the active multi-field cells is
    found exactly, by enumeration. With more than 65,536 combinations the
    score is refused; raise ``field_threshold`` or pass rate maps with fewer
    fields.

    Gupta et al. also limited pairs to fields less than 80 cm apart when
    scanning for events, and dropped the limit when testing them; no limit
    is applied here.

    References
    ----------
    Gupta, A. S., van der Meer, M. A. A., Touretzky, D. S., & Redish, A. D.
    (2010). Hippocampal replay is not a simple function of experience.
    *Neuron*, 65(5), 695-705.

    Examples
    --------
    Three cells with fields at 15, 25 and 35 cm fire 10 ms apart, in field
    order. All three pairs are concordant:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import pairwise_order_score
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [50.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> rates = np.zeros((3, env.n_bins))
    >>> rates[[0, 1, 2], [1, 2, 3]] = 10.0
    >>> spike_times = [np.array([1.00]), np.array([1.01]), np.array([1.02])]
    >>> result = pairwise_order_score(env, spike_times, (1.0, 1.1), rates)
    >>> result.score, result.n_pairs
    (3, 3)

    The last spike 100 ms later is too far from the others to be compared:

    >>> late = [np.array([1.00]), np.array([1.01]), np.array([1.11])]
    >>> pairwise_order_score(env, late, (1.0, 1.2), rates).score
    1
    """
    operation = "pairwise_order_score()"
    _check_direction(operation, direction)
    _check_field_options(
        operation,
        field_threshold=field_threshold,
        merge_gap_bins=merge_gap_bins,
        field_center=field_center,
        max_time_lag=max_time_lag,
    )
    fields = _place_fields(
        env,
        encoding_models,
        field_threshold=field_threshold,
        merge_gap_bins=merge_gap_bins,
        field_center=field_center,
        operation=operation,
    )
    event = _event_spikes(spike_times, interval, len(fields), operation)
    return _observed(event, fields, direction, max_time_lag, operation)


def pairwise_order_significance(
    env: Environment,
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    direction: Literal["forward", "backward"] = "forward",
    max_time_lag: float = 0.08,
    field_threshold: float = 0.05,
    merge_gap_bins: int = 1,
    field_center: Literal["rate_weighted", "peak"] = "rate_weighted",
    nulls: Sequence[str] = _NULLS,
    n_shuffles: int = 300,
    rng: np.random.Generator | int | None = None,
) -> PairwiseOrderSignificanceResult:
    """Test an event's pairwise-order score against shuffles (Gupta et al. 2010).

    Each shuffle is scored as the event is, with each multi-field cell again
    taking the field that makes the score largest. The test is one-sided:
    the event must score higher than the shuffles in ``direction``.

    Parameters
    ----------
    env, spike_times, interval, encoding_models
        As for :func:`pairwise_order_score`.
    direction, max_time_lag, field_threshold, merge_gap_bins, field_center
        As for :func:`pairwise_order_score`.
    nulls : sequence of {"spike_time_permutation", "cell_identity"}
        The shuffles, both by default (Gupta et al. 2010):

        - ``"spike_time_permutation"``: permute the times of the scored
          spikes (those of cells with a field) among those spikes, so each
          cell keeps its number of spikes and the event keeps its spike
          times and its number of nearby pairs. Gupta et al. "randomly
          shuffled the spike times for each event"; redrawing times
          uniformly over the event is another reading, which changes the
          number of nearby pairs.
        - ``"cell_identity"``: permute the cells' sets of fields across
          every cell given, active or not, with or without a field, so each
          spike train takes another cell's fields (Davidson et al. 2009's
          cell-identity shuffle, which Gupta et al. cite).
    n_shuffles : int, default=300
        Shuffles per null (Gupta et al. 2010: 300). A p-value is at least
        ``1 / (n_shuffles + 1)``.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed. Each null draws from its own
        stream, so its shuffles do not depend on the other nulls.

    Returns
    -------
    PairwiseOrderSignificanceResult
        The event's score, one shuffle test per null and the largest
        p-value.

    Raises
    ------
    ValueError
        For inputs :func:`pairwise_order_score` refuses, including a
        shuffle whose field combinations exceed 65,536; if ``nulls`` is
        empty, repeats a null or names an unknown one; or if ``n_shuffles``
        is not a positive integer.

    See Also
    --------
    pairwise_order_score : The score alone.
    detect_pairwise_order_replay : The published test on every template
        and direction.

    Notes
    -----
    Each p-value is ``(k + 1) / (n + 1)``, where ``k`` counts shuffles
    scoring at least as high as the event (Phipson & Smyth 2010). Gupta et
    al. called an event significant when its score was "greater than 95% of
    both independent sets of shuffled sequences": with 300 shuffles, at most
    14 shuffles at or above the event, which is exactly
    ``(k + 1) / 301 < 0.05``.

    Examples
    --------
    Twelve cells with fields along a 120 cm track fire 8 ms apart in field
    order. The 63 pairs less than 80 ms apart are all in field order:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import pairwise_order_significance
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=5.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> fields = np.linspace(5.0, 115.0, 12)
    >>> rates = 20.0 * np.exp(-0.5 * ((x - fields[:, None]) / 4.0) ** 2)
    >>> spike_times = [np.array([1.0 + 0.008 * i]) for i in range(12)]
    >>> result = pairwise_order_significance(env, spike_times, (1.0, 1.1), rates, rng=0)
    >>> result.observed.score, result.p_value < 0.05
    (63, True)
    """
    operation = "pairwise_order_significance()"
    _check_direction(operation, direction)
    _check_field_options(
        operation,
        field_threshold=field_threshold,
        merge_gap_bins=merge_gap_bins,
        field_center=field_center,
        max_time_lag=max_time_lag,
    )
    nulls = (nulls,) if isinstance(nulls, str) else tuple(nulls)
    if not nulls or len(set(nulls)) != len(nulls) or set(nulls) - set(_NULLS):
        raise ValueError(
            f"{operation}: nulls must be one or more distinct values of "
            f"{list(_NULLS)}, got {nulls!r}."
        )
    _check_n_shuffles(operation, n_shuffles)
    fields = _place_fields(
        env,
        encoding_models,
        field_threshold=field_threshold,
        merge_gap_bins=merge_gap_bins,
        field_center=field_center,
        operation=operation,
    )
    event = _event_spikes(spike_times, interval, len(fields), operation)
    return _significance(
        event,
        fields,
        direction,
        max_time_lag,
        nulls,
        n_shuffles,
        rng,
        operation,
    )


def detect_pairwise_order_replay(
    env: Environment | Mapping[str, Environment],
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: Any,
    *,
    pairwise_order_method: PairwiseOrderMethod,
    template: str | None = None,
    rng: np.random.Generator | int | None = None,
) -> ReplayDetectionResult:
    """Apply a published pairwise-order method to one candidate event.

    Every template is tested in each of the method's directions against its
    shuffles (:func:`pairwise_order_significance`), and the test with the
    smallest p-value is reported. Gupta et al. (2010) tested left-lap and
    right-lap templates forward and backward, four tests per event, without
    a correction: an event without a sequence is called replay up to
    ``1 - (1 - alpha) ** 4``, about 19%, of the time at alpha = 0.05 when
    the tests are independent, and less when they are not.

    Parameters
    ----------
    env : Environment or Mapping[str, Environment]
        The environment of the rate maps: a 1-D environment or a linearized
        track with position increasing in the running direction; or one
        environment per template, keyed like ``encoding_models``.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the order of the rows
        of ``encoding_models``.
    interval : (start, stop)
        The candidate event, in seconds.
    encoding_models : array of shape (n_cells, n_bins), SpatialRatesResult, or Mapping
        Each cell's rate map on ``env``, in Hz; or a mapping of such rate
        maps, one per template (such as each lap of a maze), with the cells
        in the same order.
    pairwise_order_method : PairwiseOrderMethod
        The method, for example ``neurospatial.replay.methods.GUPTA_2010``.
    template : str or None
        With a mapping of rate maps, test only this template; None tests
        every template.
    rng : np.random.Generator, int, or None
        Random number generator or seed. Each (template, direction) test
        draws from its own stream, derived from ``rng`` and the test's
        name.

    Returns
    -------
    ReplayDetectionResult
        The verdict of the test with the smallest p-value (the first in
        template order, forward before backward, on a tie). ``score`` is
        its pairwise-order score, signed by direction: the forward score,
        or minus the backward score, so a backward sequence scores below
        zero. ``p_value`` is the largest over the method's nulls,
        ``template`` the tested template (None for one set of rate maps),
        and ``criteria`` the active-cell rule, counted over every cell
        given.

    Raises
    ------
    ValueError
        If ``template`` is given without a mapping of rate maps or is not
        one of its keys, if a mapping of environments lacks a template, or
        for inputs :func:`pairwise_order_score` refuses.

    See Also
    --------
    pairwise_order_significance : One template and direction.
    compare_replay_methods : Several methods, pairwise-order, rank-order
        and decoded, on several events.

    Examples
    --------
    Ten cells fire 10 ms apart in reverse field order. The 42 pairs less
    than 80 ms apart all run backward, so the score is -42:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import detect_pairwise_order_replay
    >>> from neurospatial.replay.methods import GUPTA_2010
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [120.0, 0.0]], bin_size=5.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> rates = 20.0 * np.exp(
    ...     -0.5 * ((x - np.linspace(10.0, 110.0, 10)[:, None]) / 4.0) ** 2
    ... )
    >>> spikes = [np.array([1.09 - 0.01 * i]) for i in range(10)]
    >>> result = detect_pairwise_order_replay(
    ...     env, spikes, (1.0, 1.1), rates, pairwise_order_method=GUPTA_2010, rng=0
    ... )
    >>> result.is_replay, result.score
    (True, -42.0)
    """
    from neurospatial.replay.methods import ReplayDetectionResult
    from neurospatial.replay.significance import _template_streams

    method = pairwise_order_method
    operation = "detect_pairwise_order_replay()"
    names: list[str | None]
    if isinstance(encoding_models, Mapping):
        names = list(encoding_models)
        if template is not None:
            if template not in encoding_models:
                raise ValueError(
                    f"{operation}: template {template!r} is not one of "
                    f"{list(encoding_models)}."
                )
            names = [template]
    elif template is not None:
        raise ValueError(
            _format_error(
                f"{operation}: template={template!r} needs a mapping of rate maps.",
                why="a template names one of several rate-map sets",
                fix="pass encoding_models={'left': rates_left, 'right': "
                "rates_right}, or leave out template=",
            )
        )
    else:
        names = [None]
    labels = {
        (name, direction): direction if name is None else f"{name}:{direction}"
        for name in names
        for direction in method.directions
    }
    streams = _template_streams(rng, list(labels.values()))
    event = None
    best: tuple[float, float, str | None] | None = None
    for name in names:
        if isinstance(env, Mapping):
            if name is None or name not in env:
                raise ValueError(
                    _format_error(
                        f"{operation}: env has no environment for template {name!r}.",
                        why="each template's fields are read on its own track",
                        fix="pass a mapping of rate maps and a mapping of "
                        "environments with the same keys, or one environment",
                    )
                )
            template_env = env[name]
        else:
            template_env = env
        fields = _place_fields(
            template_env,
            encoding_models if name is None else encoding_models[name],
            field_threshold=method.field_threshold,
            merge_gap_bins=method.merge_gap_bins,
            field_center=method.field_center,
            operation=operation,
        )
        if event is None:
            event = _event_spikes(spike_times, interval, len(fields), operation)
        elif len(fields) != event.n_cells:
            raise ValueError(
                f"{operation}: template {name!r} has {len(fields)} rate maps "
                f"but spike_times has {event.n_cells} cells."
            )
        for direction in method.directions:
            tested = _significance(
                event,
                fields,
                direction,
                method.max_time_lag,
                method.nulls,
                method.n_shuffles,
                streams[labels[name, direction]],
                operation,
            )
            sign = 1.0 if direction == "forward" else -1.0
            candidate = (tested.p_value, sign * tested.observed.score, name)
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None and event is not None
    p_value, score, chosen = best
    n_active = float(event.active_units.size)
    criterion = method.criteria[0]
    passes = {criterion.label: criterion.evaluate(n_active, np.nan)}
    return ReplayDetectionResult(
        method=method.name,
        is_replay=bool(p_value < method.alpha) and all(passes.values()),
        score=score,
        p_value=p_value,
        criteria=passes,
        values={criterion.label: n_active},
        template=chosen,
        n_shuffles_used=method.n_shuffles,
    )


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _EventSpikes:
    """The spikes of one event, in time order.

    ``cell`` maps each spike to its position among ``active_units``, the
    cells (rows of the rate maps) that fired in the event.
    """

    times: NDArray[np.float64]  # (n_spikes,), sorted
    cell: NDArray[np.intp]  # (n_spikes,)
    active_units: NDArray[np.intp]  # (n_active,)
    n_cells: int  # cells given, active or not


def _check_direction(operation: str, direction: str) -> None:
    if direction not in _DIRECTIONS:
        raise ValueError(
            f"{operation}: direction must be one of {list(_DIRECTIONS)}, got "
            f"{direction!r}."
        )


def _check_n_shuffles(prefix: str, n_shuffles: Any) -> None:
    if (
        isinstance(n_shuffles, bool)
        or not isinstance(n_shuffles, (int, np.integer))
        or n_shuffles < 1
    ):
        raise ValueError(
            f"{prefix}: n_shuffles must be a positive integer, got "
            f"{n_shuffles!r}. Gupta et al. (2010) used 300 per null."
        )


def _check_field_options(
    prefix: str,
    *,
    field_threshold: Any,
    merge_gap_bins: Any,
    field_center: Any,
    max_time_lag: Any,
) -> None:
    """Refuse field and pair options that cannot run."""
    numbers = (int, float, np.integer, np.floating)
    if (
        isinstance(field_threshold, bool)
        or not isinstance(field_threshold, numbers)
        or not 0 < field_threshold < 1
    ):
        raise ValueError(
            f"{prefix}: field_threshold must be a fraction in (0, 1) of the "
            f"highest rate, got {field_threshold!r} (Gupta et al. 2010: 0.05)."
        )
    if (
        isinstance(merge_gap_bins, bool)
        or not isinstance(merge_gap_bins, (int, np.integer))
        or merge_gap_bins < 0
    ):
        raise ValueError(
            f"{prefix}: merge_gap_bins must be an integer >= 0, got {merge_gap_bins!r}."
        )
    if field_center not in _FIELD_CENTERS:
        raise ValueError(
            f"{prefix}: field_center must be one of {list(_FIELD_CENTERS)}, got "
            f"{field_center!r}."
        )
    if (
        isinstance(max_time_lag, bool)
        or not isinstance(max_time_lag, numbers)
        or not max_time_lag > 0
    ):
        raise ValueError(
            f"{prefix}: max_time_lag must be a positive time in seconds (np.inf "
            f"compares every pair), got {max_time_lag!r}."
        )


def _place_fields(
    env: Environment,
    encoding_models: Any,
    *,
    field_threshold: float,
    merge_gap_bins: int,
    field_center: str,
    operation: str,
) -> list[NDArray[np.float64]]:
    """Each cell's field positions along the track, ascending; empty for none."""
    from neurospatial.replay.correlation import _track_bin_centers
    from neurospatial.replay.rank_order import _field_positions

    # Validates the rate maps against env (shape, sign, environment).
    _field_positions(env, encoding_models, "peak", operation)
    centers = _track_bin_centers(env, operation)
    order = np.argsort(centers, kind="stable")
    positions = centers[order]
    rates = _supported_rates(encoding_models)[:, order]
    rates = np.where(np.isfinite(rates), rates, 0.0)
    peak = float(rates.max()) if rates.size else 0.0
    above = rates > field_threshold * peak
    fields = []
    for row, row_above in zip(rates, above, strict=True):
        on = np.flatnonzero(row_above)
        if not on.size:
            fields.append(np.empty(0))
            continue
        breaks = np.flatnonzero(np.diff(on) > merge_gap_bins + 1)
        starts = on[np.r_[0, breaks + 1]]
        stops = on[np.r_[breaks, on.size - 1]] + 1
        row_fields = []
        for start, stop in zip(starts, stops, strict=True):
            weights, where = row[start:stop], positions[start:stop]
            row_fields.append(
                float(where @ weights / weights.sum())
                if field_center == "rate_weighted"
                else float(where[np.argmax(weights)])
            )
        fields.append(np.sort(np.asarray(row_fields, dtype=np.float64)))
    return fields


def _event_spikes(
    spike_times: Any, interval: Any, n_cells: int, operation: str
) -> _EventSpikes:
    from neurospatial._spikes import as_spike_trains

    start, stop = _validated_interval(interval, operation)
    trains = as_spike_trains(spike_times)
    if len(trains) != n_cells:
        raise ValueError(
            _format_error(
                f"{operation}: spike_times has {len(trains)} cells but "
                f"encoding_models has {n_cells} rate maps.",
                why="each cell's spikes are placed at its own fields",
                fix="pass spike times and rate maps for the same cells, in the "
                "same order",
            )
        )
    in_event = [train[(train >= start) & (train < stop)] for train in trains]
    active = np.array(
        [unit for unit, spikes in enumerate(in_event) if spikes.size], dtype=np.intp
    )
    times = (
        np.concatenate([in_event[unit] for unit in active])
        if active.size
        else np.empty(0)
    )
    cell = np.repeat(np.arange(active.size), [in_event[unit].size for unit in active])
    order = np.argsort(times, kind="stable")
    return _EventSpikes(
        times=times[order],
        cell=cell[order].astype(np.intp),
        active_units=active,
        n_cells=len(trains),
    )


def _lag_pairs(
    times: NDArray[np.float64], max_time_lag: float
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Pairs ``i < j`` of sorted spike times less than ``max_time_lag`` apart.

    Returns the two spikes' indices and ``sign(t_j - t_i)`` (0 or 1).
    """
    n = times.size
    # Every t_j above the rounded t_i + lag has t_j - t_i >= lag once rounded,
    # so the search keeps every candidate and the difference decides.
    ends = np.searchsorted(times, times + max_time_lag, side="right")
    counts = ends - np.arange(n) - 1
    first = np.repeat(np.arange(n), counts)
    offsets = np.arange(first.size) - np.repeat(np.cumsum(counts) - counts, counts)
    second = first + 1 + offsets
    gaps = times[second] - times[first]
    keep = gaps < max_time_lag
    return first[keep], second[keep], np.sign(gaps[keep])


def _order_weights(
    first: NDArray[np.intp],
    second: NDArray[np.intp],
    signs: NDArray[np.float64],
    cell: NDArray[np.intp],
    n_cells: int,
) -> NDArray[np.float64]:
    """``W[u, v]``: pairs where u fires before v minus pairs where v fires first.

    Over the given pairs of spikes, for spikes labeled ``cell``. The event's
    score for field positions ``c`` is ``sum over u < v of W[u, v] *
    sign(c[v] - c[u])``.
    """
    before = np.asarray(
        np.bincount(
            cell[first] * n_cells + cell[second],
            weights=signs,
            minlength=n_cells * n_cells,
        ),
        dtype=np.float64,
    ).reshape(n_cells, n_cells)
    weights: NDArray[np.float64] = before - before.T
    return weights


def _best_assignment(
    weights: NDArray[np.float64], fields: Sequence[NDArray[np.float64]], sign: float
) -> tuple[int, NDArray[np.float64]]:
    """Largest score over one field per cell, and the field each cell takes.

    ``weights`` is :func:`_order_weights` over the cells of ``fields``. The
    score splits into a constant over single-field cells, a term per
    multi-field cell and a term per pair of them; every combination of the
    multi-field cells' fields is scored. Cells without a field get NaN.
    """
    sizes = np.array([cell_fields.size for cell_fields in fields])
    n_combinations = math.prod(int(size) for size in sizes if size)
    if n_combinations > _MAX_FIELD_ASSIGNMENTS:
        raise _FieldSearchTooLargeError(n_combinations)
    w = sign * weights
    single = np.flatnonzero(sizes == 1)
    multi = np.flatnonzero(sizes > 1)
    x = np.array([fields[u][0] for u in single], dtype=np.float64)
    order = np.sign(x[None, :] - x[:, None])
    total: NDArray[np.float64] = np.full(
        n_combinations, np.triu(w[np.ix_(single, single)] * order, 1).sum()
    )
    grid = np.indices([fields[m].size for m in multi]).reshape(
        multi.size, n_combinations
    )
    for a, m in enumerate(multi):
        c = fields[m]
        unary = (w[m, single][None, :] * np.sign(x[None, :] - c[:, None])).sum(axis=1)
        total += unary[grid[a]]
        for b in range(a + 1, multi.size):
            n = multi[b]
            between = w[m, n] * np.sign(fields[n][None, :] - c[:, None])
            total += between[grid[a], grid[b]]
    best = int(np.argmax(total))
    positions = np.full(len(fields), np.nan)
    positions[single] = x
    for a, m in enumerate(multi):
        positions[m] = fields[m][grid[a, best]]
    return round(float(total[best])), positions


def _observed(
    event: _EventSpikes,
    fields: Sequence[NDArray[np.float64]],
    direction: str,
    max_time_lag: float,
    operation: str,
) -> PairwiseOrderResult:
    n_active = event.active_units.size
    active_fields = [fields[unit] for unit in event.active_units]
    first, second, signs = _lag_pairs(event.times, max_time_lag)
    weights = _order_weights(first, second, signs, event.cell, n_active)
    score, positions = _searched(
        weights, active_fields, 1.0 if direction == "forward" else -1.0, operation
    )
    has_field = np.array([f.size > 0 for f in active_fields], dtype=bool)
    a, b = event.cell[first], event.cell[second]
    scored = np.flatnonzero(has_field)
    return PairwiseOrderResult(
        score=score,
        direction=direction,
        n_pairs=int(np.sum((a != b) & has_field[a] & has_field[b])),
        n_active_cells=int(n_active),
        scored_units=event.active_units[scored],
        assigned_positions=positions[scored],
    )


def _searched(
    weights: NDArray[np.float64],
    fields: Sequence[NDArray[np.float64]],
    sign: float,
    operation: str,
) -> tuple[int, NDArray[np.float64]]:
    """:func:`_best_assignment`, with an actionable error past the cap."""
    try:
        return _best_assignment(weights, fields, sign)
    except _FieldSearchTooLargeError as error:
        raise _FieldSearchTooLargeError(
            _format_error(
                f"{operation}: the active cells' fields give {error.args[0]:,} "
                "combinations of one field per cell, more than the "
                f"{_MAX_FIELD_ASSIGNMENTS:,} searched exactly.",
                why="each multi-field cell takes the field that makes the score "
                "largest, found by scoring every combination, in the event "
                "and in each shuffle",
                fix="raise field_threshold, or pass rate maps with fewer fields "
                "(set a cell's secondary fields to 0)",
            )
        ) from None


def _significance(
    event: _EventSpikes,
    fields: Sequence[NDArray[np.float64]],
    direction: str,
    max_time_lag: float,
    nulls: Sequence[str],
    n_shuffles: int,
    rng: np.random.Generator | int | None,
    operation: str,
) -> PairwiseOrderSignificanceResult:
    """The event's score and one shuffle test per null."""
    from neurospatial.replay.significance import _template_streams
    from neurospatial.stats.shuffle import (
        ShuffleTestResult,
        compute_shuffle_pvalue,
        compute_shuffle_zscore,
    )

    observed = _observed(event, fields, direction, max_time_lag, operation)
    sign = 1.0 if direction == "forward" else -1.0
    streams = _template_streams(rng, list(nulls))
    n_active = event.active_units.size
    active_fields = [fields[unit] for unit in event.active_units]
    tests = {}
    for null in nulls:
        generator = streams[null]
        null_scores = np.empty(n_shuffles)
        if null == "spike_time_permutation":
            has_field = np.array([f.size > 0 for f in active_fields], dtype=bool)
            scored = has_field[event.cell]
            cell = event.cell[scored]
            first, second, signs = _lag_pairs(event.times[scored], max_time_lag)
            draws = np.argsort(generator.random((n_shuffles, cell.size)), axis=1)
            for k, draw in enumerate(draws):
                weights = _order_weights(first, second, signs, cell[draw], n_active)
                null_scores[k] = _searched(weights, active_fields, sign, operation)[0]
        else:
            first, second, signs = _lag_pairs(event.times, max_time_lag)
            weights = _order_weights(first, second, signs, event.cell, n_active)
            draws = np.argsort(generator.random((n_shuffles, len(fields))), axis=1)
            for k, draw in enumerate(draws):
                shuffled = [fields[draw[unit]] for unit in event.active_units]
                null_scores[k] = _searched(weights, shuffled, sign, operation)[0]
        tests[null] = ShuffleTestResult(
            observed_score=float(observed.score),
            null_scores=null_scores,
            p_value=compute_shuffle_pvalue(
                float(observed.score), null_scores, tail="greater"
            ),
            z_score=compute_shuffle_zscore(float(observed.score), null_scores),
            shuffle_type=null,
            n_shuffles=n_shuffles,
        )
    return PairwiseOrderSignificanceResult(
        observed=observed,
        shuffles=tests,
        p_value=max(test.p_value for test in tests.values()),
    )
