"""Recurring spike sequences within a session (Nádasdy et al. 1999).

Nádasdy et al. (1999) searched a session's spike trains for spike
sequences that repeat: each spike of a reference cell starts a time window,
the other cells' spikes in the window form a template, and the template
recurs wherever the reference cell fires again followed by the same cells
at the same lags, within a tolerance ("sliding sweeps", after Abeles &
Gerstein 1988). Whether a session has more recurring sequences than chance
is tested against surrogate spike trains that keep some of the spike
trains' structure (:func:`surrogate_spike_trains`). The test needs no
behavioral template and gives no per-event verdict.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import (
    ReadOnlyArraysMixin,
    ResultMixin,
    read_only_copy,
    reduce_through_constructor,
)
from neurospatial.replay.sequence_match import (
    _check_seconds,
    _is_count,
    _onsets,
    _sorted_trains,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

__all__ = [
    "NADASDY_1999",
    "RecurringSequenceMethod",
    "RecurringSequencesResult",
    "RecurringSequencesSignificanceResult",
    "recurring_sequences_significance",
    "recurring_spike_sequences",
    "surrogate_spike_trains",
]

_SURROGATES = ("isi_shuffle", "spike_jitter", "across_trains", "theta_phase")


def surrogate_spike_trains(
    spike_times: Any,
    *,
    kind: Literal["isi_shuffle", "spike_jitter", "across_trains", "theta_phase"],
    max_jitter: float = 0.05,
    theta_peak_times: NDArray[np.float64] | None = None,
    rng: np.random.Generator | int | None = None,
) -> list[NDArray[np.float64]]:
    """Spike trains with some structure kept and the rest randomized.

    The four surrogates of Nádasdy et al. (1999):

    - ``"isi_shuffle"``: each train's interspike intervals in random order,
      from its first spike. Keeps each cell's number of spikes, intervals
      (and so bursts and refractoriness), first and last spike.
    - ``"spike_jitter"``: each spike moved later by a random interval from
      0 to ``max_jitter``, uniformly. Keeps each cell's number of spikes
      and its rate at scales longer than ``max_jitter``.
    - ``"across_trains"``: each spike given to a random cell, every cell
      equally likely. Keeps the pooled spike times (the population rate);
      differences in rate between cells are lost.
    - ``"theta_phase"``: each train's theta cycles (from one LFP peak to the
      next) in random order, each cycle's spikes keeping their phase within
      the cycle they move to. Keeps each spike's theta phase and each
      cell's number of spikes. Spikes before the first or after the last
      peak stay in place.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell.
    kind : {"isi_shuffle", "spike_jitter", "across_trains", "theta_phase"}
        The surrogate.
    max_jitter : float, default=0.05
        ``"spike_jitter"``: the largest displacement in seconds (Nádasdy et
        al.: 0 to 50 ms, later only).
    theta_peak_times : NDArray[np.float64] of shape (n_peaks,), optional
        ``"theta_phase"``: the times of the peaks of the theta-filtered LFP,
        in seconds; at least two.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed.

    Returns
    -------
    list of NDArray[np.float64]
        One sorted surrogate train per cell, in seconds.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the options, ``max_jitter`` is not a
        positive number, or ``theta_peak_times`` is missing for
        ``"theta_phase"`` or does not have two finite, increasing peaks.

    See Also
    --------
    recurring_sequences_significance : Test recurring sequences against
        these surrogates.

    Examples
    --------
    The interval shuffle keeps each train's intervals:

    >>> import numpy as np
    >>> from neurospatial.replay import surrogate_spike_trains
    >>> train = np.array([0.0, 0.1, 0.3, 0.6])
    >>> shuffled = surrogate_spike_trains([train], kind="isi_shuffle", rng=0)[0]
    >>> np.sort(np.diff(shuffled)).round(3).tolist(), shuffled[0], shuffled[-1].round(3)
    ([0.1, 0.2, 0.3], np.float64(0.0), np.float64(0.6))
    """
    from neurospatial._spikes import as_spike_trains

    operation = "surrogate_spike_trains()"
    peaks = _checked_surrogate_options(
        operation, (kind,), max_jitter=max_jitter, theta_peak_times=theta_peak_times
    )
    trains = _sorted_trains(as_spike_trains(spike_times))
    return _surrogate(trains, kind, max_jitter, peaks, np.random.default_rng(rng))


def _checked_surrogate_kinds(
    operation: str, kinds: tuple[str, ...], max_jitter: Any
) -> None:
    unknown = [kind for kind in kinds if kind not in _SURROGATES]
    if unknown:
        raise ValueError(
            f"{operation}: surrogates must be among {list(_SURROGATES)}, got {unknown}."
        )
    _check_seconds(max_jitter, "max_jitter", operation, allow_zero=False)


def _checked_surrogate_options(
    operation: str,
    kinds: tuple[str, ...],
    *,
    max_jitter: Any,
    theta_peak_times: Any,
) -> NDArray[np.float64] | None:
    """Check the surrogate kinds, and the theta peaks when a kind needs them."""
    _checked_surrogate_kinds(operation, kinds, max_jitter)
    if "theta_phase" not in kinds:
        return None
    if theta_peak_times is None:
        raise ValueError(
            _format_error(
                f"{operation}: the theta-phase surrogate needs theta_peak_times.",
                why="it moves each theta cycle's spikes to another cycle, keeping "
                "their phase, and the cycles run from one LFP theta peak to the "
                "next",
                fix="pass the peak times of the theta-filtered LFP (e.g. 5-12 Hz) "
                "as theta_peak_times=, or leave out 'theta_phase'",
            )
        )
    peaks = np.asarray(theta_peak_times, dtype=np.float64)
    if peaks.ndim != 1 or peaks.size < 2 or not np.isfinite(peaks).all():
        raise ValueError(
            f"{operation}: theta_peak_times must be a 1-D array of at least two "
            f"finite times, got shape {peaks.shape}."
        )
    if np.any(np.diff(peaks) <= 0):
        raise ValueError(
            f"{operation}: theta_peak_times must be strictly increasing; sort them "
            "and drop repeated peaks."
        )
    return peaks


def _surrogate(
    trains: list[NDArray[np.float64]],
    kind: str,
    max_jitter: float,
    peaks: NDArray[np.float64] | None,
    rng: np.random.Generator,
) -> list[NDArray[np.float64]]:
    if kind == "isi_shuffle":
        return [
            train.copy()
            if train.size < 3
            else train[0] + np.r_[0.0, np.cumsum(rng.permutation(np.diff(train)))]
            for train in trains
        ]
    if kind == "spike_jitter":
        return [
            np.sort(train + rng.uniform(0.0, max_jitter, size=train.size))
            for train in trains
        ]
    if kind == "across_trains":
        pooled = np.concatenate([np.empty(0), *trains])
        owner = rng.integers(len(trains), size=pooled.size)
        return [np.sort(pooled[owner == cell]) for cell in range(len(trains))]
    assert peaks is not None
    durations = np.diff(peaks)
    surrogates = []
    for train in trains:
        cycle = np.searchsorted(peaks, train, side="right") - 1
        inside = (cycle >= 0) & (cycle < durations.size)
        phase = (train[inside] - peaks[cycle[inside]]) / durations[cycle[inside]]
        target = rng.permutation(durations.size)[cycle[inside]]
        moved = peaks[target] + phase * durations[target]
        surrogates.append(np.sort(np.r_[train[~inside], moved]))
    return surrogates


@dataclass(frozen=True, repr=False, eq=False)
class RecurringSequencesResult(ReadOnlyArraysMixin, ResultMixin):
    """Spike sequences that start with a reference cell's spike, and their recurrences.

    Each template is one spike (burst onset) of the reference cell and the
    other cells' spikes in the ``window`` after it. A later or earlier
    window of the reference cell, not overlapping the template's, repeats
    the template when every template spike has a spike of the same cell
    within ``tolerance / 2`` of the same lag, a different spike for each;
    other spikes in the window are allowed.

    Attributes
    ----------
    reference_unit : int
        The reference cell, as an index into ``spike_times``.
    reference_times : NDArray[np.float64], shape (n_templates,)
        Each template's reference spike, in seconds.
    complexity : NDArray[np.intp], shape (n_templates,)
        Spikes in each template, the reference spike included (Nádasdy et
        al.'s ``c``).
    repetitions : NDArray[np.intp], shape (n_templates,)
        Other windows that repeat each template (``r``).
    element_templates : NDArray[np.intp], shape (n_elements,)
        The template of each non-reference spike.
    element_units : NDArray[np.intp], shape (n_elements,)
        Its cell.
    element_lags : NDArray[np.float64], shape (n_elements,)
        Its time after the reference spike, in seconds.
    window : float
        Template window in seconds.
    tolerance : float
        Full width of the match window in seconds (Nádasdy et al.'s ``dT``).
    burst_isi : float
        Spikes closer than this, in seconds, were merged into one at the
        first spike.
    min_complexity : int
        Templates with fewer spikes were not searched and are not listed.

    See Also
    --------
    recurring_spike_sequences : Function that produces this result.
    """

    reference_unit: int
    reference_times: NDArray[np.float64]
    complexity: NDArray[np.intp]
    repetitions: NDArray[np.intp]
    element_templates: NDArray[np.intp]
    element_units: NDArray[np.intp]
    element_lags: NDArray[np.float64]
    window: float
    tolerance: float
    burst_isi: float
    min_complexity: int

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    @property
    def n_recurring(self) -> int:
        """Templates that recur at least once (Nádasdy et al.'s ``m``)."""
        return int(np.sum(self.repetitions >= 1))

    def sequences_table(self) -> pd.DataFrame:
        """One row per template: ``reference_time``, ``complexity``,
        ``repetitions`` and the template's ``units`` and ``lags`` (tuples, in
        time order), indexed by ``template``.
        """
        import pandas as pd

        units: list[tuple[int, ...]] = [() for _ in self.reference_times]
        lags: list[tuple[float, ...]] = [() for _ in self.reference_times]
        for template in np.unique(self.element_templates):
            members = self.element_templates == template
            units[template] = tuple(int(unit) for unit in self.element_units[members])
            lags[template] = tuple(float(lag) for lag in self.element_lags[members])
        table = pd.DataFrame(
            {
                "reference_time": self.reference_times,
                "complexity": self.complexity,
                "repetitions": self.repetitions,
                "units": units,
                "lags": lags,
            },
            index=pd.RangeIndex(self.reference_times.size, name="template"),
        )
        table.attrs.update(
            reference_unit=self.reference_unit,
            window=self.window,
            tolerance=self.tolerance,
        )
        return table

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "reference_unit": self.reference_unit,
            "n_templates": int(self.reference_times.size),
            "n_recurring": self.n_recurring,
            "max_repetitions": int(self.repetitions.max())
            if self.repetitions.size
            else 0,
            "window": self.window,
            "tolerance": self.tolerance,
        }


def recurring_spike_sequences(
    spike_times: Any,
    *,
    reference_unit: int,
    window: float = 0.2,
    tolerance: float = 0.01,
    burst_isi: float = 0.006,
    min_complexity: int = 3,
) -> RecurringSequencesResult:
    """Find spike sequences that recur in a session (Nádasdy et al. 1999).

    Bursts (spikes less than ``burst_isi`` apart) are merged into one spike
    at the first. Each spike of the reference cell starts a template: the
    other cells' spikes in the following ``window``, at their lags. Every
    other spike of the reference cell whose window does not overlap the
    template's is an exemplar, which repeats the template when, for every
    template spike, the same cell fires within ``tolerance / 2`` of the
    same lag after it, with a different spike for each template spike.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell.
    reference_unit : int
        The reference cell, as an index into ``spike_times``.
    window : float, default=0.2
        Template window in seconds (Nádasdy et al.: typically 200 ms).
    tolerance : float, default=0.01
        Full width of the match window in seconds, ``dT`` (Nádasdy et al.:
        10 or 20 ms in most searches, 2.5 to 20 ms explored).
    burst_isi : float, default=0.006
        Spikes closer than this, in seconds, are one spike at the first
        (complex-spike bursts, < 6 ms). 0 keeps every spike.
    min_complexity : int, default=3
        Fewest spikes in a template, the reference spike included; smaller
        templates are not searched. At least 2.

    Returns
    -------
    RecurringSequencesResult
        Every template with at least ``min_complexity`` spikes and its
        number of repetitions.

    Raises
    ------
    ValueError
        If ``reference_unit`` is not a cell of ``spike_times``, or a window,
        tolerance or interval is not a positive number of seconds
        (``burst_isi`` may be 0), or ``min_complexity`` is below 2.

    See Also
    --------
    recurring_sequences_significance : Compare the number of recurring
        sequences with surrogate spike trains.

    Notes
    -----
    Nádasdy et al. do not say whether ``dT`` is the full width or a
    tolerance on each side, whether an exemplar may hold extra spikes, how
    exemplars that overlap the template are treated, or which complexity
    counts. The readings here: ``dT`` is the full width (``+/- dT / 2``),
    extra spikes are allowed (a template recurs within a larger pattern),
    exemplars do not overlap the template's window (so a template does not
    match its own spikes shifted by a few milliseconds), and only templates
    of at least ``min_complexity`` spikes count. A cell may contribute
    several spikes to a template; each needs its own spike of the cell in
    the exemplar, taken in time order.

    References
    ----------
    Nádasdy, Z., Hirase, H., Czurkó, A., Csicsvari, J., & Buzsáki, G.
    (1999). Replay and time compression of recurring spike sequences in the
    hippocampus. *Journal of Neuroscience*, 19(21), 9497-9507.

    Examples
    --------
    Cell 0 fires every second, followed by cells 1 and 2 at 30 and 80 ms:

    >>> import numpy as np
    >>> from neurospatial.replay import recurring_spike_sequences
    >>> starts = np.arange(5.0)
    >>> spikes = [starts, starts + 0.03, starts + 0.08]
    >>> result = recurring_spike_sequences(spikes, reference_unit=0)
    >>> result.complexity.tolist(), result.repetitions.tolist()
    ([3, 3, 3, 3, 3], [4, 4, 4, 4, 4])
    """
    from neurospatial._spikes import as_spike_trains

    operation = "recurring_spike_sequences()"
    trains = _sorted_trains(as_spike_trains(spike_times))
    _check_search_options(
        operation,
        n_cells=len(trains),
        reference_unit=reference_unit,
        window=window,
        tolerance=tolerance,
        burst_isi=burst_isi,
        min_complexity=min_complexity,
    )
    onsets = [_onsets(train, burst_isi) for train in trains]
    return _search(
        onsets,
        int(reference_unit),
        window=window,
        tolerance=tolerance,
        burst_isi=burst_isi,
        min_complexity=min_complexity,
    )


def _check_search_options(
    operation: str,
    *,
    n_cells: int,
    reference_unit: Any,
    window: Any,
    tolerance: Any,
    burst_isi: Any,
    min_complexity: Any,
) -> None:
    if not _is_count(reference_unit, 0, n_cells - 1):
        raise ValueError(
            f"{operation}: reference_unit must be the index of a cell, from 0 to "
            f"{n_cells - 1}, got {reference_unit!r}."
        )
    _check_seconds(window, "window", operation, allow_zero=False)
    _check_seconds(tolerance, "tolerance", operation, allow_zero=False)
    _check_seconds(burst_isi, "burst_isi", operation, allow_zero=True)
    if not _is_count(min_complexity, 2):
        raise ValueError(
            f"{operation}: min_complexity must be an integer of at least 2, got "
            f"{min_complexity!r}; a sequence holds the reference spike and at "
            "least one more."
        )


def _search(
    onsets: list[NDArray[np.float64]],
    reference_unit: int,
    *,
    window: float,
    tolerance: float,
    burst_isi: float,
    min_complexity: int,
) -> RecurringSequencesResult:
    """Sliding sweeps over burst-merged trains."""
    reference = onsets[reference_unit]
    others = [cell for cell in range(len(onsets)) if cell != reference_unit]
    pooled = np.concatenate([np.empty(0), *(onsets[cell] for cell in others)])
    pooled_cells = np.repeat(others, [onsets[cell].size for cell in others]).astype(
        np.intp
    )
    order = np.argsort(pooled, kind="stable")
    pooled, pooled_cells = pooled[order], pooled_cells[order]
    first = np.searchsorted(pooled, reference, side="left")
    last = np.searchsorted(pooled, reference + window, side="left")
    half = tolerance / 2.0
    kept, repetitions, elements = [], [], []
    for index, (lo, hi) in enumerate(zip(first, last, strict=True)):
        if 1 + hi - lo < min_complexity:
            continue
        cells = pooled_cells[lo:hi]
        lags = pooled[lo:hi] - reference[index]
        candidates = np.flatnonzero(np.abs(reference - reference[index]) >= window)
        # Each cell's template spikes, in time order, take distinct exemplar
        # spikes in order: the earliest spike in the match window after the
        # one the cell's previous template spike took.
        next_free: dict[int, NDArray[np.intp]] = {}
        for cell, lag in zip(cells, lags, strict=True):
            if not candidates.size:
                break
            train = onsets[cell]
            shifted = reference[candidates] + lag
            earliest = np.searchsorted(train, shifted - half, side="left")
            if cell in next_free:
                earliest = np.maximum(earliest, next_free[cell])
            found = np.searchsorted(train, shifted + half, side="right") > earliest
            candidates = candidates[found]
            next_free = {key: free[found] for key, free in next_free.items()}
            next_free[int(cell)] = earliest[found] + 1
        kept.append(index)
        repetitions.append(candidates.size)
        elements.append((cells, lags))
    n = len(kept)
    return RecurringSequencesResult(
        reference_unit=reference_unit,
        reference_times=reference[np.asarray(kept, dtype=np.intp)],
        complexity=np.array([1 + cells.size for cells, _ in elements], dtype=np.intp),
        repetitions=np.asarray(repetitions, dtype=np.intp),
        element_templates=np.repeat(
            np.arange(n), [cells.size for cells, _ in elements]
        ).astype(np.intp),
        element_units=np.concatenate(
            [np.empty(0, dtype=np.intp), *(cells for cells, _ in elements)]
        ),
        element_lags=np.concatenate([np.empty(0), *(lags for _, lags in elements)]),
        window=float(window),
        tolerance=float(tolerance),
        burst_isi=float(burst_isi),
        min_complexity=int(min_complexity),
    )


@dataclass(frozen=True)
class RecurringSequenceMethod:
    """A published session-level test of recurring spike sequences.

    Unlike the per-event replay methods, it has no behavioral template and
    no verdict on single events: it asks whether a session's spike trains
    hold more recurring sequences than surrogate trains (see
    :func:`recurring_sequences_significance`), so it is not accepted by
    :func:`~neurospatial.replay.compare_replay_methods`.

    Attributes
    ----------
    name : str
        Short name, for example ``"nadasdy_1999"``.
    reference : str
        The paper the method follows.
    window : float
        Template window in seconds.
    tolerance : float
        Full width of the match window in seconds (``dT``).
    burst_isi : float
        Spikes closer than this, in seconds, are one spike.
    min_complexity : int
        Fewest spikes in a counted template, the reference spike included.
    surrogates : tuple of str
        The surrogates the count is compared with (see
        :func:`surrogate_spike_trains`).
    max_jitter : float
        Largest displacement of the ``"spike_jitter"`` surrogate, in seconds.
    n_surrogates : int
        Surrogate sessions per surrogate kind.
    alpha : float
        Significance level; the session is significant when the p-value of
        every surrogate kind is below it.
    notes : tuple of str
        Choices the paper does not specify, and analyses not implemented.
    """

    name: str
    reference: str
    window: float = 0.2
    tolerance: float = 0.01
    burst_isi: float = 0.006
    min_complexity: int = 3
    surrogates: tuple[str, ...] = _SURROGATES
    max_jitter: float = 0.05
    n_surrogates: int = 100
    alpha: float = 0.01
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse settings that cannot run."""
        object.__setattr__(self, "surrogates", tuple(self.surrogates))
        object.__setattr__(self, "notes", tuple(self.notes))
        prefix = f"RecurringSequenceMethod {self.name!r}"
        if not self.surrogates or len(set(self.surrogates)) != len(self.surrogates):
            raise ValueError(
                f"{prefix}: surrogates must name each surrogate kind once, got "
                f"{self.surrogates!r}."
            )
        _checked_surrogate_kinds(prefix, self.surrogates, self.max_jitter)
        _check_seconds(self.window, "window", prefix, allow_zero=False)
        _check_seconds(self.tolerance, "tolerance", prefix, allow_zero=False)
        _check_seconds(self.burst_isi, "burst_isi", prefix, allow_zero=True)
        for field_name, minimum in (("min_complexity", 2), ("n_surrogates", 1)):
            value = getattr(self, field_name)
            if not _is_count(value, minimum):
                raise ValueError(
                    f"{prefix}: {field_name} must be an integer of at least "
                    f"{minimum}, got {value!r}."
                )
        if not 0 < self.alpha < 1:
            raise ValueError(f"{prefix}: alpha must be in (0, 1), got {self.alpha!r}.")


@dataclass(frozen=True, repr=False, eq=False)
class RecurringSequencesSignificanceResult(ResultMixin):
    """A session's recurring sequences compared with surrogate spike trains.

    Attributes
    ----------
    method : str
        The method's name.
    observed : RecurringSequencesResult
        The session's templates and repetitions.
    surrogate_counts : Mapping[str, NDArray[np.int64]]
        For each surrogate kind, the number of recurring templates in each
        surrogate session, shape (n_surrogates,).
    p_values : Mapping[str, float]
        For each surrogate kind, ``(k + 1) / (n + 1)``, where ``k``
        surrogate sessions have at least as many recurring templates as
        the session. With 100 surrogates, 1/101 < 0.01 means the session
        exceeds every surrogate, Nádasdy et al.'s criterion.
    p_value : float
        The largest p-value over the surrogate kinds.
    alpha : float
        The significance level.
    is_significant : bool
        Whether every surrogate kind's p-value is below ``alpha``.

    See Also
    --------
    recurring_sequences_significance : Function that produces this result.
    """

    method: str
    observed: RecurringSequencesResult
    surrogate_counts: Mapping[str, NDArray[np.int64]]
    p_values: Mapping[str, float]
    p_value: float
    alpha: float
    is_significant: bool

    def __post_init__(self) -> None:
        """Freeze the mappings and own read-only copies of the counts."""
        object.__setattr__(
            self,
            "surrogate_counts",
            MappingProxyType(
                {
                    kind: read_only_copy(counts)
                    for kind, counts in self.surrogate_counts.items()
                }
            ),
        )
        object.__setattr__(self, "p_values", MappingProxyType(dict(self.p_values)))

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Rebuild through the constructor; the read-only mappings cannot pickle."""
        rebuild, (cls, state) = reduce_through_constructor(self)
        state["surrogate_counts"] = dict(self.surrogate_counts)
        state["p_values"] = dict(self.p_values)
        return rebuild, (cls, state)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "method": self.method,
            "n_recurring": self.observed.n_recurring,
            **{f"p_value[{kind}]": value for kind, value in self.p_values.items()},
            "p_value": self.p_value,
            "is_significant": self.is_significant,
        }


def recurring_sequences_significance(
    spike_times: Any,
    *,
    recurring_sequence_method: RecurringSequenceMethod,
    reference_unit: int,
    theta_peak_times: NDArray[np.float64] | None = None,
    rng: np.random.Generator | int | None = None,
) -> RecurringSequencesSignificanceResult:
    """Test whether a session holds more recurring sequences than surrogates.

    The session's spikes are burst-merged and searched for recurring
    sequences (:func:`recurring_spike_sequences`); the number of templates
    that recur at least once is compared with the same number in
    ``n_surrogates`` surrogate sessions of each kind, made from the
    burst-merged trains (:func:`surrogate_spike_trains`).

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, from one state (such as
        one sleep session).
    recurring_sequence_method : RecurringSequenceMethod
        The method, for example ``neurospatial.replay.methods.NADASDY_1999``.
    reference_unit : int
        The reference cell, as an index into ``spike_times``.
    theta_peak_times : NDArray[np.float64] of shape (n_peaks,), optional
        Peak times of the theta-filtered LFP in seconds, for the
        ``"theta_phase"`` surrogate.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed. Each surrogate kind draws from its
        own stream, so leaving one out does not change the others.

    Returns
    -------
    RecurringSequencesSignificanceResult
        The session's sequences, each kind's surrogate counts and p-value.

    Raises
    ------
    ValueError
        If the method uses the ``"theta_phase"`` surrogate and
        ``theta_peak_times`` is missing, or for inputs
        :func:`recurring_spike_sequences` refuses.

    See Also
    --------
    recurring_spike_sequences : The search.
    surrogate_spike_trains : The surrogates.

    Notes
    -----
    Nádasdy et al. (1999) rejected the null when the repetitions in the
    session exceeded those of every one of 100 surrogates (p < 0.01); with
    the ``(k + 1) / (n + 1)`` p-value used here and 100 surrogates, that is
    p = 1/101. They found across-train shuffling gave the most repetitions,
    making it the strictest surrogate. The statistic here is the number of
    templates of at least ``min_complexity`` spikes that recur at least
    once; the paper compared repetitions per sequence and per complexity
    without one summary number.

    Examples
    --------
    A sequence of three cells repeats 30 times among random spikes:

    >>> import numpy as np
    >>> from neurospatial.replay import recurring_sequences_significance
    >>> from neurospatial.replay.methods import NADASDY_1999
    >>> import dataclasses
    >>> rng = np.random.default_rng(0)
    >>> starts = np.sort(rng.uniform(0, 300, 30))
    >>> spikes = [
    ...     np.sort(np.r_[starts + lag, rng.uniform(0, 300, 30)])
    ...     for lag in (0, 0.03, 0.08)
    ... ]
    >>> method = dataclasses.replace(
    ...     NADASDY_1999, surrogates=("across_trains", "spike_jitter"), n_surrogates=19
    ... )
    >>> result = recurring_sequences_significance(
    ...     spikes, recurring_sequence_method=method, reference_unit=0, rng=1
    ... )
    >>> result.observed.n_recurring > 20, result.p_values["across_trains"]
    (True, 0.05)
    """
    from neurospatial._spikes import as_spike_trains

    operation = "recurring_sequences_significance()"
    method = recurring_sequence_method
    trains = _sorted_trains(as_spike_trains(spike_times))
    peaks = _checked_surrogate_options(
        operation,
        method.surrogates,
        max_jitter=method.max_jitter,
        theta_peak_times=theta_peak_times,
    )
    _check_search_options(
        operation,
        n_cells=len(trains),
        reference_unit=reference_unit,
        window=method.window,
        tolerance=method.tolerance,
        burst_isi=method.burst_isi,
        min_complexity=method.min_complexity,
    )
    options: dict[str, Any] = {
        "window": method.window,
        "tolerance": method.tolerance,
        "burst_isi": method.burst_isi,
        "min_complexity": method.min_complexity,
    }
    onsets = [_onsets(train, method.burst_isi) for train in trains]
    observed = _search(onsets, int(reference_unit), **options)
    entropy = (
        int(rng.integers(2**63))
        if isinstance(rng, np.random.Generator)
        else np.random.SeedSequence(rng).entropy
    )
    counts: dict[str, NDArray[np.int64]] = {}
    p_values: dict[str, float] = {}
    for kind in method.surrogates:
        stream = np.random.default_rng(
            np.random.SeedSequence(entropy, spawn_key=(zlib.crc32(kind.encode()),))
        )
        kind_counts = np.array(
            [
                _search(
                    _surrogate(onsets, kind, method.max_jitter, peaks, stream),
                    int(reference_unit),
                    **options,
                ).n_recurring
                for _ in range(method.n_surrogates)
            ],
            dtype=np.int64,
        )
        counts[kind] = kind_counts
        p_values[kind] = float(
            (np.sum(kind_counts >= observed.n_recurring) + 1) / (kind_counts.size + 1)
        )
    p_value = max(p_values.values())
    return RecurringSequencesSignificanceResult(
        method=method.name,
        observed=observed,
        surrogate_counts=counts,
        p_values=p_values,
        p_value=p_value,
        alpha=method.alpha,
        is_significant=bool(p_value < method.alpha),
    )


NADASDY_1999 = RecurringSequenceMethod(
    name="nadasdy_1999",
    reference=(
        "Nádasdy, Z., Hirase, H., Czurkó, A., Csicsvari, J., & Buzsáki, G. "
        "(1999). Replay and time compression of recurring spike sequences in the "
        "hippocampus. Journal of Neuroscience, 19(21), 9497-9507."
    ),
    window=0.2,
    tolerance=0.01,
    burst_isi=0.006,
    min_complexity=3,
    surrogates=_SURROGATES,
    max_jitter=0.05,
    n_surrogates=100,
    alpha=0.01,
    notes=(
        "Sliding sweeps: each spike of the reference cell starts a 200 ms "
        "template of the other cells' spikes; another window of the reference "
        "cell repeats it when each template spike has a spike of the same cell "
        "within the tolerance (dT 10 ms; 20 ms in some searches, 2.5-20 ms "
        "explored). Complex-spike bursts (< 6 ms) are one spike at the first.",
        "Unstated, read here as in Abeles & Gerstein (1988): dT is the full "
        "width (+/- 5 ms); an exemplar may hold extra spikes; exemplars do not "
        "overlap the template's window; templates are whole windows of at least "
        "3 spikes (the reference included); a cell may give several spikes to "
        "a template, each matched to its own spike of the cell in the exemplar.",
        "Surrogates: interspike intervals in random order (the paper swapped "
        "pairs repeatedly, number not stated; a uniform permutation is its "
        "limit), spikes moved later by 0-50 ms, spikes given to cells at random "
        "(every cell equally likely; the paper's most rigorous surrogate), and "
        "theta cycles of each train in random order with spike phases kept "
        "(cycles from one theta peak to the next; spikes outside them stay). "
        "Surrogates are made from the burst-merged trains.",
        "Significant when the number of recurring templates exceeds that of "
        "every one of 100 surrogates of each kind (p = 1/101 < 0.01); the "
        "paper compared repetitions per sequence without one summary number, "
        "and analyzed each state (sleep, run) separately.",
        "Session level only; no per-event verdict and no behavioral template, "
        "so it is not in PUBLISHED_METHODS and compare_replay_methods does not "
        "take it. The RUN-to-sleep comparison used joint-probability maps of "
        "cell triplets with timing discarded, whose expected counts (from three "
        "cross-correlograms) and per-pixel Fisher tests are not specified; it "
        "is not implemented.",
    ),
)
