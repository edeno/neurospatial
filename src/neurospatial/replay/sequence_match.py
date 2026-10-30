"""Template matching of firing order: does an event replay the cells' track order?

Lee & Wilson (2002) and Ji & Wilson (2007) asked whether place cells fire
during sleep in the order of their place fields on the track (the
template), without decoding position. Lee & Wilson cut each cell's spikes
into letters (a burst counts once, at its first spike), merged the letters
of the template cells into words, and scored each word by its best match
to the template; the probability of a match at least that good among the
orderings of the word's letters is the event's p-value
(:func:`best_sequence_match`). Ji & Wilson took one firing time per active
cell, the peak of its smoothed spike train, and scored the order of the
cells by a matching index, the share of cell pairs in template order minus
the share in opposite order (:func:`sequence_matching_index`). Both test
the forward order only; reverse replay is a match to the reversed
template.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._validation import is_integer_at_least, validate_integer_at_least
from neurospatial.replay._compare import _tolerance

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.encoding.spatial import SpatialRatesResult
    from neurospatial.environment import Environment

__all__ = [
    "MatchingIndexResult",
    "SequenceMatchResult",
    "SpikeWordsResult",
    "best_sequence_match",
    "burst_onset_times",
    "sequence_matching_index",
    "spike_words",
]

_MATCH_ORDERS = ("interruptions_first", "length_first")
_FIELD_ORDERS = ("peak", "center")
# Largest number of letters whose orderings are enumerated: 10! = 3.6 M.
_MAX_EXACT_LETTERS = 10
# Rows of orderings scored at once, to bound memory.
_CHUNK_ROWS = 1 << 18
# Kernel peaks are located on this grid (seconds) and then refined.
_PEAK_GRID = 0.001


# --------------------------------------------------------------------------
# Letters and words (Lee & Wilson 2002)
# --------------------------------------------------------------------------


def burst_onset_times(spike_times: Any, *, max_isi: float) -> list[NDArray[np.float64]]:
    """The first spike of each burst, one train per cell.

    A burst is a chain of spikes whose consecutive interspike intervals are
    shorter than ``max_isi``; a spike that follows the previous spike of its
    cell by ``max_isi`` or more (within 1 ns of round-off) starts a new
    burst. A long chain stays one
    burst even when it lasts longer than ``max_isi``. Each burst is
    represented by the time of its first spike.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell. Any input
        :func:`~neurospatial.decoding.bin_spikes_in_time` accepts.
    max_isi : float
        Interspike interval, in seconds, below which a spike joins the
        burst before it. Lee & Wilson (2002): 0.05 s. Nádasdy et al.
        (1999): 0.006 s, the complex-spike bursts of Ranck (1973). 0 keeps
        every spike.

    Returns
    -------
    list of NDArray[np.float64]
        Each cell's burst onsets in seconds, sorted.

    Raises
    ------
    ValueError
        If ``max_isi`` is negative or not finite, or for spike input
        :func:`~neurospatial.decoding.bin_spikes_in_time` refuses.

    Examples
    --------
    A burst of three spikes 10 ms apart, then an isolated spike:

    >>> import numpy as np
    >>> from neurospatial.replay import burst_onset_times
    >>> burst_onset_times([np.array([1.0, 1.01, 1.02, 1.5])], max_isi=0.05)[0]
    array([1. , 1.5])
    """
    from neurospatial._spikes import as_spike_trains

    _check_seconds(max_isi, "max_isi", "burst_onset_times()", allow_zero=True)
    return [
        _onsets(train, max_isi)
        for train in _sorted_trains(as_spike_trains(spike_times))
    ]


@dataclass(frozen=True, repr=False, eq=False)
class SpikeWordsResult(ReadOnlyArraysMixin, ResultMixin):
    """Spike trains parsed into words (Lee & Wilson 2002).

    Attributes
    ----------
    intervals : NDArray[np.float64], shape (n_words, 2)
        Each word's ``(start, stop)`` in seconds: its first letter, and its
        last letter plus ``word_gap / 2`` (cut at the end of its epoch). A
        word's interval holds exactly its letters, and intervals do not
        overlap, so they can be passed as candidate events to
        :func:`~neurospatial.replay.detect_template_match_replay` or
        :func:`~neurospatial.replay.template_match_dataset_test`.
    word_ids : NDArray[np.intp], shape (n_letters,)
        The word of each letter.
    letter_units : NDArray[np.intp], shape (n_letters,)
        The cell of each letter, as an index into ``spike_times``.
    letter_times : NDArray[np.float64], shape (n_letters,)
        Each letter's time in seconds: the first spike of its burst. Letters
        are in time order, ties by cell index.
    max_isi : float
        The burst interval, in seconds.
    word_gap : float
        The gap, in seconds, above which a word ends.

    See Also
    --------
    spike_words : Function that produces this result.
    """

    intervals: NDArray[np.float64]
    word_ids: NDArray[np.intp]
    letter_units: NDArray[np.intp]
    letter_times: NDArray[np.float64]
    max_isi: float
    word_gap: float

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    @property
    def n_words(self) -> int:
        """Number of words."""
        return int(self.intervals.shape[0])

    def words_table(self) -> pd.DataFrame:
        """One row per word: ``start``, ``stop``, ``n_letters`` and
        ``n_distinct_letters`` (the cells that fired), indexed by ``word``.
        """
        import pandas as pd

        n_letters = np.bincount(self.word_ids, minlength=self.n_words)
        pairs = np.unique(np.stack([self.word_ids, self.letter_units]), axis=1)
        n_distinct = np.bincount(pairs[0], minlength=self.n_words)
        table = pd.DataFrame(
            {
                "start": self.intervals[:, 0],
                "stop": self.intervals[:, 1],
                "n_letters": n_letters,
                "n_distinct_letters": n_distinct,
            },
            index=pd.RangeIndex(self.n_words, name="word"),
        )
        table.attrs.update(max_isi=self.max_isi, word_gap=self.word_gap)
        return table

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "n_words": self.n_words,
            "n_letters": int(self.letter_times.size),
            "max_isi": self.max_isi,
            "word_gap": self.word_gap,
        }


def spike_words(
    spike_times: Any,
    *,
    max_isi: float,
    word_gap: float,
    cells: NDArray[np.bool_] | None = None,
    epochs: NDArray[np.float64] | None = None,
) -> SpikeWordsResult:
    """Parse spike trains into words of letters (Lee & Wilson 2002).

    Each cell's burst (spikes less than ``max_isi`` apart) becomes one
    letter at its first spike, and an isolated spike a letter of its own.
    The letters of all ``cells`` are merged in time and the string is split
    between adjacent letters more than ``word_gap`` apart. Lee & Wilson
    parsed each template's cells separately (``cells=``), within slow-wave
    sleep (``epochs=``), with ``max_isi=0.05`` and ``word_gap=0.1``.

    Parameters
    ----------
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell.
    max_isi : float
        Interspike interval, in seconds, below which a spike joins its
        cell's burst (see :func:`burst_onset_times`).
    word_gap : float
        A word ends where the next letter is more than ``word_gap`` seconds
        later (a gap equal to it within 1 ns of round-off stays in the word).
        Must be positive and at least ``max_isi``, as in the paper.
    cells : NDArray[np.bool_] of shape (n_cells,), optional
        The cells whose letters form the words, such as one template's
        cells. None uses every cell.
    epochs : NDArray[np.float64] of shape (n_epochs, 2), optional
        Sorted, non-overlapping ``(start, stop)`` intervals in seconds, such
        as slow-wave sleep; letters at ``start <= t < stop`` are kept and
        words do not cross an epoch boundary. Bursts are found on the whole
        trains, so a burst that began before an epoch adds no letter in it.
        None keeps every letter.

    Returns
    -------
    SpikeWordsResult
        The words' intervals and letters. Words of one letter are kept;
        filter them with ``words_table()``.

    Raises
    ------
    ValueError
        If ``max_isi`` is negative, ``word_gap`` is not positive or is below
        ``max_isi``, ``cells`` is not one boolean per cell, or ``epochs`` is
        not sorted, non-overlapping ``(start, stop)`` rows with
        ``start < stop``.

    See Also
    --------
    best_sequence_match : Score a word against the template.

    Examples
    --------
    Two cells fire a burst and a spike, then, 300 ms later, one spike each:

    >>> import numpy as np
    >>> from neurospatial.replay import spike_words
    >>> spikes = [np.array([1.0, 1.02, 1.4]), np.array([1.06, 1.45])]
    >>> words = spike_words(spikes, max_isi=0.05, word_gap=0.1)
    >>> words.words_table()[["n_letters", "n_distinct_letters"]].to_numpy().tolist()
    [[2, 2], [2, 2]]
    >>> words.intervals.round(3).tolist()
    [[1.0, 1.11], [1.4, 1.5]]
    """
    from neurospatial._spikes import as_spike_trains

    operation = "spike_words()"
    _check_seconds(max_isi, "max_isi", operation, allow_zero=True)
    _check_seconds(word_gap, "word_gap", operation, allow_zero=False)
    if max_isi > word_gap:
        raise ValueError(
            _format_error(
                f"{operation}: max_isi ({max_isi:g} s) is larger than word_gap "
                f"({word_gap:g} s).",
                why="a burst would then span gaps that end words (Lee & Wilson "
                "2002 require max_isi <= word_gap)",
                fix="use max_isi <= word_gap, e.g. max_isi=0.05, word_gap=0.1",
            )
        )
    trains = _sorted_trains(as_spike_trains(spike_times))
    selected = np.ones(len(trains), dtype=bool)
    if cells is not None:
        selected = np.asarray(cells)
        if selected.dtype != bool or selected.shape != (len(trains),):
            raise ValueError(
                f"{operation}: cells must be one boolean per cell, shape "
                f"({len(trains)},), got dtype {selected.dtype} and shape "
                f"{selected.shape}."
            )
    units = np.flatnonzero(selected)
    onsets = [_onsets(trains[unit], max_isi) for unit in units]
    times = np.concatenate([np.empty(0), *onsets])
    letter_units = np.repeat(units, [onset.size for onset in onsets]).astype(np.intp)
    order = np.lexsort((letter_units, times))
    times, letter_units = times[order], letter_units[order]
    epoch_stop = np.full(times.size, np.inf)
    epoch_id = np.zeros(times.size, dtype=np.intp)
    if epochs is not None:
        bounds = _validated_epochs(epochs, operation)
        index = np.searchsorted(bounds[:, 0], times, side="right") - 1
        inside = (index >= 0) & (times < bounds[np.maximum(index, 0), 1])
        times, letter_units, index = times[inside], letter_units[inside], index[inside]
        epoch_id = index
        epoch_stop = bounds[index, 1]
    new_word = np.ones(times.size, dtype=bool)
    new_word[1:] = ~_at_most(np.diff(times), word_gap) | (np.diff(epoch_id) != 0)
    word_ids = np.cumsum(new_word) - 1
    first = np.flatnonzero(new_word)
    last = np.r_[first[1:] - 1, times.size - 1] if first.size else first
    intervals = np.column_stack(
        [times[first], np.minimum(times[last] + word_gap / 2, epoch_stop[last])]
    ).reshape(-1, 2)
    return SpikeWordsResult(
        intervals=intervals,
        word_ids=word_ids.astype(np.intp),
        letter_units=letter_units,
        letter_times=times,
        max_isi=float(max_isi),
        word_gap=float(word_gap),
    )


# --------------------------------------------------------------------------
# Best (x, y) match (Lee & Wilson 2002)
# --------------------------------------------------------------------------


@dataclass(frozen=True, repr=False, eq=False)
class SequenceMatchResult(ReadOnlyArraysMixin, ResultMixin):
    """The best match of one event's word to the template (Lee & Wilson 2002).

    A word contains an ``(x, y)`` match when ``x + y`` consecutive letters
    hold ``x`` letters in strictly increasing template order, not
    necessarily adjacent in the template: ``x`` letters in order with ``y``
    interruptions.

    Attributes
    ----------
    matched_letters : int
        ``x`` of the best match; 0 when the word has no match (no two
        letters in template order with fewer interruptions than ordered
        letters minus one).
    interruptions : int
        ``y`` of the best match; 0 without a match.
    match_probability : float
        The share of the orderings of the word's letters whose best match
        is at least as good: the event's p-value. 1 without a match, NaN
        for fewer than 2 letters. With more than ``exact_max_letters``
        letters, a Monte Carlo estimate ``(k + 1) / (n + 1)`` from ``n``
        random orderings.
    best_possible_probability : float
        The same probability for the best match any ordering of these
        letters gives (a ``(K, 0)`` match over its ``K`` distinct letters):
        the smallest ``match_probability`` the word can have. NaN for fewer
        than 2 letters; a sampled estimate (``k / n``) for long words.
    trial_group : {"pair", "triplet", "low_probability"} or None
        Lee & Wilson's trial type: two or three letters of distinct cells,
        or a word whose ``best_possible_probability`` is at most the
        ``trial_probability`` (low-probability trial); None otherwise.
    n_letters : int
        Letters in the event.
    n_distinct_letters : int
        Template cells with a letter in the event.
    letter_units : NDArray[np.intp], shape (n_letters,)
        The cell of each letter in firing order.
    letter_times : NDArray[np.float64], shape (n_letters,)
        Each letter's time in seconds.
    exact : bool
        Whether ``match_probability`` enumerates every ordering.
    standard_error : float
        Monte Carlo standard error of ``match_probability``; 0 when exact.
    match_order : {"interruptions_first", "length_first"}
        How matches were ranked.

    See Also
    --------
    best_sequence_match : Function that produces this result.
    """

    matched_letters: int
    interruptions: int
    match_probability: float
    best_possible_probability: float
    trial_group: str | None
    n_letters: int
    n_distinct_letters: int
    letter_units: NDArray[np.intp]
    letter_times: NDArray[np.float64]
    exact: bool
    standard_error: float
    match_order: str

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "matched_letters": self.matched_letters,
            "interruptions": self.interruptions,
            "match_probability": self.match_probability,
            "best_possible_probability": self.best_possible_probability,
            "trial_group": self.trial_group,
            "n_letters": self.n_letters,
            "n_distinct_letters": self.n_distinct_letters,
            "exact": self.exact,
        }


def best_sequence_match(
    env: Environment,
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    max_isi: float = 0.05,
    field_order: Literal["peak", "center"] = "peak",
    match_order: Literal["interruptions_first", "length_first"] = "interruptions_first",
    trial_probability: float = 1 / 24,
    exact_max_letters: int = 10,
    n_samples: int = 100_000,
    rng: np.random.Generator | int | None = None,
) -> SequenceMatchResult:
    """Best match of an event's letters to the field order (Lee & Wilson 2002).

    The template is the order of the cells' fields along the track. Each
    template cell's burst (spikes less than ``max_isi`` apart) in the event
    is one letter at its first spike, and the letters in time order form
    the event's word. The word's best ``(x, y)`` match is ranked on Lee &
    Wilson's list ``(K, 0) > (K, 1) > (K - 1, 0) > (K, 2) > ... > (2, 0)``:
    the fewer interruptions beyond the ordered letters (larger ``x - y``)
    the better, then the more ordered letters. The p-value is the share of
    the orderings of the word's letters whose best match is at least as
    good. Only the forward order is tested.

    Parameters
    ----------
    env : Environment
        The environment of the rate maps: a 1-D environment or a linearized
        track.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the order of the rows
        of ``encoding_models``.
    interval : (start, stop)
        The event, in seconds: one word. Letters whose burst starts at
        ``start <= t < stop`` count; bursts are found on the whole trains,
        so a burst that began before ``start`` adds no letter.
    encoding_models : array of shape (n_cells, n_bins), or SpatialRatesResult
        Each cell's rate map on ``env``, in Hz; NaN marks bins without
        data. Cells without a finite positive rate are not in the template.
        Lee & Wilson used one running direction's cells (at least one spike
        per lap, single-peaked, firing through the session); leave the
        others out by setting their rate maps to NaN.
    max_isi : float, default=0.05
        Burst interval in seconds (Lee & Wilson 2002: 50 ms).
    field_order : {"peak", "center"}, default="peak"
        Each cell's place on the track: the peak of its rate map (Lee &
        Wilson) or its rate-weighted center.
    match_order : {"interruptions_first", "length_first"}, default="interruptions_first"
        How matches are ranked: Lee & Wilson's list (larger ``x - y``, then
        larger ``x``), or their alternative, which they report gave similar
        results (larger ``x``, then smaller ``y``). Matches with
        ``x - y < 2`` are not counted in either.
    trial_probability : float, default=1/24
        ``P`` of the low-probability trials: a word whose best possible
        match has probability at most ``P`` (Lee & Wilson: 1/24, a perfect
        four-letter match).
    exact_max_letters : int, default=10
        Words with up to this many letters enumerate every ordering (10! =
        3.6 million: about 2.5 s for ten distinct cells, less when cells
        repeat, computed once per process for each multiset of letters);
        longer words draw ``n_samples`` random orderings. At most 10.
    n_samples : int, default=100000
        Random orderings for longer words. At p = 1/24 the standard error
        is about 0.0006.
    rng : np.random.Generator, int, or None, default=None
        Random number generator or seed for the sampled orderings; unused
        when every ordering is enumerated.

    Returns
    -------
    SequenceMatchResult
        The best match, its probability, the trial type and the letters.

    Raises
    ------
    ValueError
        If two template cells have their field at the same position (the
        template must be a strict order: use finer bins, or set one cell's
        rate map to NaN); if ``max_isi`` is negative, ``trial_probability``
        is not in (0, 1), ``exact_max_letters`` is not an integer from 1
        to 10, ``n_samples`` is not a positive integer, or ``match_order``
        or ``field_order`` is not one of the options; or for inputs
        :func:`~neurospatial.replay.rank_order_correlation` refuses.

    See Also
    --------
    spike_words : Parse spike trains into words.
    sequence_matching_index : Ji & Wilson's pairwise index.
    detect_template_match_replay : The verdict of a published method.

    Notes
    -----
    Lee & Wilson's worked example: against the template ``123456789A``,
    the word ``325789A`` has a best match of ``(6, 0)`` (``25789A``); 12 of
    its 7! = 5040 orderings have a ``(6, 0)`` match and one a better
    match, so p = 13 / 5040 = 0.0026. A perfect four-letter word has
    p = 1/24, which Lee & Wilson count as significant (p <= 1/24).

    Two letters at the same time are put in the order opposite to the
    template, so a tie never helps a match. Repeated letters (a cell that
    fires two bursts) are kept, and the orderings permute all letters.

    References
    ----------
    Lee, A. K., & Wilson, M. A. (2002). Memory of sequential experience in
    the hippocampus during slow wave sleep. *Neuron*, 36(6), 1183-1194.

    Examples
    --------
    Seven cells with fields along a 200 cm track fire 3, 2, 5, 7, 8, 9 and
    10 (in template numbering), as in Lee & Wilson's example:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import best_sequence_match
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> rates = 20.0 * np.exp(
    ...     -0.5 * ((x - (np.arange(10) * 20.0 + 15)[:, None]) / 8.0) ** 2
    ... )
    >>> word = [2, 1, 4, 6, 7, 8, 9]
    >>> spikes = [
    ...     np.array([1.0 + 0.06 * word.index(c)]) if c in word else np.array([])
    ...     for c in range(10)
    ... ]
    >>> match = best_sequence_match(env, spikes, (1.0, 1.5), rates)
    >>> (
    ...     match.matched_letters,
    ...     match.interruptions,
    ...     round(match.match_probability * 5040),
    ... )
    (6, 0, 13)
    """
    from neurospatial._spikes import as_spike_trains
    from neurospatial.replay.rank_order import _validated_interval

    operation = "best_sequence_match()"
    _check_match_options(
        operation,
        max_isi=max_isi,
        match_order=match_order,
        trial_probability=trial_probability,
        exact_max_letters=exact_max_letters,
        n_samples=n_samples,
    )
    start, stop = _validated_interval(interval, operation)
    ranks = _template_ranks(
        env,
        encoding_models,
        field_order=field_order,
        min_peak_rate=None,
        operation=operation,
    )
    trains = _matched_trains(as_spike_trains(spike_times), ranks, operation)
    return _sequence_match(
        ranks,
        trains,
        start,
        stop,
        max_isi=max_isi,
        match_order=match_order,
        trial_probability=trial_probability,
        exact_max_letters=exact_max_letters,
        n_samples=n_samples,
        rng=np.random.default_rng(rng),
    )


# --------------------------------------------------------------------------
# Matching index (Ji & Wilson 2007)
# --------------------------------------------------------------------------


@dataclass(frozen=True, repr=False, eq=False)
class MatchingIndexResult(ReadOnlyArraysMixin, ResultMixin):
    """The matching index of one event (Ji & Wilson 2007).

    Attributes
    ----------
    score : float
        The matching index ``I = (m - n) / (m + n)`` over the ``M (M - 1) /
        2`` pairs of active cells, from -1 (reverse order) to 1 (template
        order): ``m`` pairs fire in template order and ``n`` in the opposite
        order. NaN with fewer than ``min_active_cells`` active cells.
    match_probability : float
        Exact one-sided p-value: the probability of an index at least this
        large when every order of the ``M`` active cells is equally likely
        (the share of the ``M!`` orders with at most ``n`` opposite pairs).
        NaN with ``score``.
    n_active_cells : int
        ``M``: template cells that fired in the event.
    n_same_order_pairs : int
        ``m``: pairs of active cells that fired in template order.
    n_opposite_order_pairs : int
        ``n``: pairs that fired in the opposite order or at the same time.
    active_units : NDArray[np.intp], shape (n_active_cells,)
        The active cells, as indices into ``spike_times``, in increasing
        order.
    firing_times : NDArray[np.float64], shape (n_active_cells,)
        Each active cell's firing time in seconds, in ``active_units``
        order: the peak of the sum of Gaussian kernels on its spikes in the
        event.
    firing_order : NDArray[np.intp], shape (n_active_cells,)
        The active cells in the order they fired.
    bandwidth : float
        Kernel standard deviation, in seconds.

    See Also
    --------
    sequence_matching_index : Function that produces this result.
    """

    score: float
    match_probability: float
    n_active_cells: int
    n_same_order_pairs: int
    n_opposite_order_pairs: int
    active_units: NDArray[np.intp]
    firing_times: NDArray[np.float64]
    firing_order: NDArray[np.intp]
    bandwidth: float

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics for this result."""
        return {
            "score": self.score,
            "match_probability": self.match_probability,
            "n_active_cells": self.n_active_cells,
            "n_same_order_pairs": self.n_same_order_pairs,
            "n_opposite_order_pairs": self.n_opposite_order_pairs,
            "bandwidth": self.bandwidth,
        }


def sequence_matching_index(
    env: Environment,
    spike_times: Any,
    interval: tuple[float, float] | NDArray[np.float64],
    encoding_models: NDArray[np.float64] | SpatialRatesResult,
    *,
    bandwidth: float = 0.18,
    field_order: Literal["peak", "center"] = "peak",
    min_active_cells: int = 4,
) -> MatchingIndexResult:
    """Matching index of firing order and field order (Ji & Wilson 2007).

    Each template cell that fired in the event gets one firing time: the
    peak of the sum of Gaussian kernels (standard deviation ``bandwidth``)
    on its spikes in the event. Of the ``M (M - 1) / 2`` pairs of the
    ``M`` active cells, ``m`` fire in the order of their fields and ``n``
    in the opposite order; the matching index is ``I = (m - n) / (m + n)``
    and its p-value the probability of an index at least as large over the
    ``M!`` equally likely orders of the active cells. Only the forward
    order is tested.

    Parameters
    ----------
    env : Environment
        The environment of the rate maps: a 1-D environment or a linearized
        track.
    spike_times : array, sequence of arrays, or spike group
        Spike times in seconds, one train per cell, in the order of the rows
        of ``encoding_models``.
    interval : (start, stop)
        The event (Ji & Wilson's frame), in seconds; spikes at
        ``start <= t < stop`` count.
    encoding_models : array of shape (n_cells, n_bins), or SpatialRatesResult
        Each cell's rate map on ``env``, in Hz; NaN marks bins without
        data. Cells without a finite positive rate are not in the template.
        Ji & Wilson kept cells with more than 0.8 bit/spike and a stable
        dominant field; leave others out by setting their rate maps to NaN.
    bandwidth : float, default=0.18
        Kernel standard deviation in seconds: 0.18 for hippocampal and 0.4
        for cortical cells (Ji & Wilson 2007, who found similar results
        from 0.05 to 0.25 s and 0.1 to 0.6 s).
    field_order : {"peak", "center"}, default="peak"
        Each cell's place on the track: the peak of its rate map (Ji &
        Wilson) or its rate-weighted center.
    min_active_cells : int, default=4
        Fewest active template cells for an index (Ji & Wilson: frames with
        at least 4; with fewer, p cannot fall below 0.05). At least 2.

    Returns
    -------
    MatchingIndexResult
        The index, its exact p-value and each active cell's firing time.

    Raises
    ------
    ValueError
        If two template cells have their field at the same position (use
        finer bins, or set one cell's rate map to NaN), if ``bandwidth`` is
        not a positive number, if ``min_active_cells`` is not an integer of
        at least 2, or for inputs
        :func:`~neurospatial.replay.rank_order_correlation` refuses.

    See Also
    --------
    best_sequence_match : Lee & Wilson's best-match score.
    detect_template_match_replay : The verdict of a published method.

    Notes
    -----
    With distinct firing times ``I`` is Kendall's tau-a and ``n`` the number
    of inversions, so the p-value is the share of permutations of ``M``
    items with at most ``n`` inversions (the Mahonian numbers), exact for
    every ``M``. Ji & Wilson enumerated orders for ``M < 11`` and sampled
    10,000 above; their cutoff at ``M = 14`` (30 opposite pairs, p =
    0.047) is 0.0505 exactly. Their worked example ``02546`` has ``I =
    0.8`` and ``p = 5 / 120``.

    The kernel peak is located on a 1 ms grid over the cell's spikes and
    then refined to the exact local maximum (mean shift). Two cells with
    the same firing time count as a pair in opposite order, so a tie never
    helps a match.

    References
    ----------
    Ji, D., & Wilson, M. A. (2007). Coordinated memory replay in the visual
    cortex and hippocampus during sleep. *Nature Neuroscience*, 10(1),
    100-107.

    Examples
    --------
    Five cells fire in the order 0, 2, 5, 4, 6 of a seven-cell template:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import sequence_matching_index
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [140.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> x = env.bin_centers[:, 0]
    >>> rates = 20.0 * np.exp(
    ...     -0.5 * ((x - (np.arange(7) * 20.0 + 5)[:, None]) / 8.0) ** 2
    ... )
    >>> order = [0, 2, 5, 4, 6]
    >>> spikes = [
    ...     np.array([1.0 + 0.05 * order.index(c)]) if c in order else np.array([])
    ...     for c in range(7)
    ... ]
    >>> result = sequence_matching_index(env, spikes, (1.0, 1.3), rates)
    >>> result.score, round(result.match_probability * 120)
    (0.8, 5)
    """
    from neurospatial._spikes import as_spike_trains
    from neurospatial.replay.rank_order import _validated_interval

    operation = "sequence_matching_index()"
    _check_index_options(
        operation, bandwidth=bandwidth, min_active_cells=min_active_cells
    )
    start, stop = _validated_interval(interval, operation)
    ranks = _template_ranks(
        env,
        encoding_models,
        field_order=field_order,
        min_peak_rate=None,
        operation=operation,
    )
    trains = _matched_trains(as_spike_trains(spike_times), ranks, operation)
    return _matching_index(
        ranks,
        trains,
        start,
        stop,
        bandwidth=bandwidth,
        min_active_cells=min_active_cells,
    )


# --------------------------------------------------------------------------
# Shared private machinery
# --------------------------------------------------------------------------


def _check_seconds(value: Any, name: str, operation: str, *, allow_zero: bool) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        bound = "non-negative" if allow_zero else "positive"
        raise ValueError(
            f"{operation}: {name} must be a {bound} number of seconds, got {value!r}."
        )


def _check_match_options(
    operation: str,
    *,
    max_isi: Any,
    match_order: Any,
    trial_probability: Any,
    exact_max_letters: Any,
    n_samples: Any,
) -> None:
    _check_seconds(max_isi, "max_isi", operation, allow_zero=True)
    if match_order not in _MATCH_ORDERS:
        raise ValueError(
            f"{operation}: match_order must be one of {_MATCH_ORDERS}, got "
            f"{match_order!r}."
        )
    if (
        isinstance(trial_probability, bool)
        or not isinstance(trial_probability, (int, float, np.integer, np.floating))
        or not 0 < trial_probability < 1
    ):
        raise ValueError(
            f"{operation}: trial_probability must be in (0, 1), got "
            f"{trial_probability!r}; Lee & Wilson used 1/24."
        )
    if not (
        is_integer_at_least(exact_max_letters, 1)
        and exact_max_letters <= _MAX_EXACT_LETTERS
    ):
        raise ValueError(
            f"{operation}: exact_max_letters must be an integer from 1 to "
            f"{_MAX_EXACT_LETTERS}, got {exact_max_letters!r}; enumerating the "
            f"orderings of more letters needs more than {_MAX_EXACT_LETTERS}! rows."
        )
    validate_integer_at_least(n_samples, 1, name="n_samples", owner=operation)


def _check_index_options(
    operation: str, *, bandwidth: Any, min_active_cells: Any
) -> None:
    _check_seconds(bandwidth, "bandwidth", operation, allow_zero=False)
    validate_integer_at_least(
        min_active_cells,
        2,
        name="min_active_cells",
        owner=operation,
        hint="a matching index needs a pair of cells",
    )


def _sorted_trains(trains: list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    return [
        train if train.size < 2 or np.all(np.diff(train) >= 0) else np.sort(train)
        for train in trains
    ]


def _onsets(train: NDArray[np.float64], max_isi: float) -> NDArray[np.float64]:
    """Burst onsets of one sorted train: spikes at least max_isi after the last."""
    keep = np.ones(train.size, dtype=bool)
    # An interval equal to max_isi (within round-off) starts a new burst.
    keep[1:] = ~(np.diff(train) < max_isi - _tolerance(max_isi))
    onsets: NDArray[np.float64] = train[keep]
    return onsets


def _validated_epochs(epochs: Any, operation: str) -> NDArray[np.float64]:
    bounds = np.asarray(epochs, dtype=np.float64)
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(bounds[1:, 0] < bounds[:-1, 1])
    ):
        raise ValueError(
            _format_error(
                f"{operation}: epochs must be sorted, non-overlapping (start, "
                f"stop) rows in seconds with start < stop, got shape "
                f"{bounds.shape}.",
                why="each letter belongs to at most one epoch, and words do not "
                "cross epochs",
                fix="pass epochs of shape (n_epochs, 2), sorted by start, merging "
                "overlapping ones",
            )
        )
    return bounds


def _template_ranks(
    env: Environment,
    encoding_models: Any,
    *,
    field_order: str,
    min_peak_rate: float | None,
    operation: str,
) -> NDArray[np.intp]:
    """Each cell's place in the field order; -1 for a cell not in the template."""
    from neurospatial.replay.rank_order import _template_fields

    if field_order not in _FIELD_ORDERS:
        raise ValueError(
            f"{operation}: field_order must be one of {_FIELD_ORDERS}, got "
            f"{field_order!r}."
        )
    _, positions = _template_fields(
        env, encoding_models, field_order, min_peak_rate, operation
    )
    template = np.flatnonzero(np.isfinite(positions))
    order = template[np.argsort(positions[template], kind="stable")]
    tied = np.flatnonzero(np.diff(positions[order]) == 0)
    if tied.size:
        first, second = int(order[tied[0]]), int(order[tied[0] + 1])
        raise ValueError(
            _format_error(
                f"{operation}: cells {first} and {second} have their fields at "
                f"the same position ({positions[first]:g}), so the template does "
                f"not order them ({tied.size} tied pairs in all).",
                why="the template is the order of the cells' fields, and a "
                "sequence match needs a strict order (Lee & Wilson's template "
                "has no repeated or tied cells)",
                fix="use finer position bins, or leave one of the cells out of "
                "the template by setting its rate map to NaN",
            )
        )
    ranks = np.full(positions.size, -1, dtype=np.intp)
    ranks[order] = np.arange(order.size)
    return ranks


def _matched_trains(
    trains: list[NDArray[np.float64]], ranks: NDArray[np.intp], operation: str
) -> list[NDArray[np.float64]]:
    from neurospatial.replay.rank_order import _require_same_cells

    _require_same_cells(len(trains), ranks.size, operation)
    return _sorted_trains(trains)


def _event_letters(
    trains: list[NDArray[np.float64]],
    ranks: NDArray[np.intp],
    start: float,
    stop: float,
    max_isi: float,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Template cells' burst onsets in [start, stop), ordered against the template on ties.

    A spike is a burst onset when the spike before it in the whole train is
    at least ``max_isi`` earlier, so only the spike before the event is read.
    """
    units, times = [], []
    for unit in np.flatnonzero(ranks >= 0):
        train = trains[unit]
        lo, hi = np.searchsorted(train, [start, stop], side="left")
        if hi == lo:
            continue
        onset = _onsets(train[max(lo - 1, 0) : hi], max_isi)
        if lo > 0:
            onset = onset[onset >= start]
        units.append(np.full(onset.size, unit, dtype=np.intp))
        times.append(onset)
    letter_units = np.concatenate([np.empty(0, dtype=np.intp), *units])
    letter_times = np.concatenate([np.empty(0), *times])
    order = _firing_order(letter_units, letter_times, ranks)
    return letter_units[order], letter_times[order]


def _firing_order(
    units: NDArray[np.intp], times: NDArray[np.float64], ranks: NDArray[np.intp]
) -> NDArray[np.intp]:
    """Letters in time order; letters at the same time against the template.

    Tied letters go in decreasing template rank, so a tie never helps a match.
    """
    order: NDArray[np.intp] = np.lexsort((-ranks[units], times))
    return order


def _all_permutations(n: int) -> NDArray[np.int8]:
    """Every permutation of range(n), one per row, shape (n!, n)."""
    perms = np.zeros((1, 1), dtype=np.int8)
    for k in range(1, n):
        out = np.empty((k + 1, perms.shape[0], k + 1), dtype=np.int8)
        for i in range(k + 1):
            out[i, :, :i] = perms[:, :i]
            out[i, :, i] = k
            out[i, :, i + 1 :] = perms[:, i:]
        perms = out.reshape(-1, k + 1)
    return perms


def _distinct_orderings(pattern: tuple[int, ...]) -> NDArray[np.int8]:
    """Every distinct ordering of the letters ``pattern``, one per row.

    Shape (n! / prod(m!), n), with ``m`` the number of letters of each value.
    Built one value at a time: each row's free places, in every combination,
    take the next value's letters. Distinct letters use the faster
    permutation table.
    """
    values, counts = np.unique(pattern, return_counts=True)
    if counts.max() == 1:
        return values.astype(np.int8)[_all_permutations(values.size)]
    rows = np.full((1, len(pattern)), -1, dtype=np.int8)
    for value, count in zip(values, counts, strict=True):
        free = np.nonzero(rows < 0)[1].reshape(rows.shape[0], -1)
        combinations = np.array(
            list(itertools.combinations(range(free.shape[1]), int(count))),
            dtype=np.intp,
        )
        places = free[:, combinations].reshape(-1, int(count))
        rows = np.repeat(rows, combinations.shape[0], axis=0)
        rows[np.arange(rows.shape[0])[:, None], places] = value
    return rows


def _window_lis(ranks: NDArray[np.int16]) -> NDArray[np.int16]:
    """Longest strictly increasing subsequence in the best window of each length.

    Returns ``f`` of shape (n_rows, n + 1), ``f[:, w]`` the largest LIS over
    the row's windows of ``w`` consecutive letters (patience sorting from
    each start, vectorized over rows).
    """
    n_rows, n = ranks.shape
    f = np.zeros((n_rows, n + 1), dtype=np.int16)
    rows = np.arange(n_rows)
    top = np.iinfo(np.int16).max
    for s in range(n):
        tails = np.full((n_rows, n - s), top, dtype=np.int16)
        length = np.zeros(n_rows, dtype=np.int16)
        for j in range(s, n):
            w = j - s + 1
            value = ranks[:, j]
            position = (tails[:, :w] < value[:, None]).sum(axis=1)
            tails[rows, position] = value
            np.maximum(length, position + 1, out=length)
            np.maximum(f[:, w], length, out=f[:, w])
    return f


def _match_keys(ranks: NDArray[Any], match_order: str) -> NDArray[np.int64]:
    """The best match of each row, as an integer that orders matches.

    Rows are words of ``n`` letters in template ranks. -1 means no match.
    """
    ranks = np.asarray(ranks, dtype=np.int16)
    n_rows, n = ranks.shape
    keys = np.empty(n_rows, dtype=np.int64)
    width = np.arange(n + 1)
    for start in range(0, n_rows, _CHUNK_ROWS):
        x = _window_lis(ranks[start : start + _CHUNK_ROWS]).astype(np.int64)
        y = width - x
        valid = (width >= 2) & (x - y >= 2)
        if match_order == "interruptions_first":
            key = (x - y) * (n + 1) + x
        else:
            key = x * (n + 1) + (n - y)
        keys[start : start + _CHUNK_ROWS] = np.where(valid, key, -1).max(axis=1)
    return keys


def _decode_key(key: int, n: int, match_order: str) -> tuple[int, int]:
    """(x, y) of a match key; (0, 0) for no match."""
    if key < 0:
        return 0, 0
    if match_order == "interruptions_first":
        difference, x = divmod(int(key), n + 1)
        return x, x - difference
    x, rest = divmod(int(key), n + 1)
    return x, n - rest


def _perfect_key(n_letters: int, n_distinct: int, match_order: str) -> int:
    """The key of the best match any ordering gives: (n_distinct, 0)."""
    if n_distinct < 2:
        return -1
    if match_order == "interruptions_first":
        return n_distinct * (n_letters + 1) + n_distinct
    return n_distinct * (n_letters + 1) + n_letters


@dataclass(frozen=True)
class _KeyDistribution:
    """Best-match keys over the orderings of one multiset of letters.

    ``keys`` are the distinct keys in increasing order and ``at_least`` the
    number of orderings with a key at least each of them, out of ``total``
    (every ordering when ``exact``, else random orderings).
    """

    keys: NDArray[np.int64]
    at_least: NDArray[np.int64]
    total: int
    exact: bool

    def count(self, key: int) -> int:
        index = int(np.searchsorted(self.keys, key, side="left"))
        return int(self.at_least[index]) if index < self.keys.size else 0

    def p_value(self, key: int) -> float:
        """Share of orderings at least as good; (k + 1) / (n + 1) when sampled."""
        if self.exact:
            return self.count(key) / self.total
        return (self.count(key) + 1) / (self.total + 1)

    def attainable_size(self, probability: float) -> float:
        """Largest share of orderings at or above a key that is at most ``probability``."""
        shares = self.at_least / self.total
        allowed = shares[_at_most(shares, probability)]
        return float(allowed.max()) if allowed.size else 0.0


def _distribution_from_keys(
    keys: NDArray[np.int64], exact: bool, weight: int = 1
) -> _KeyDistribution:
    """The distribution of ``keys``, each standing for ``weight`` orderings."""
    values, counts = np.unique(keys, return_counts=True)
    at_least = np.cumsum(counts[::-1])[::-1] * weight
    return _KeyDistribution(values, at_least, int(keys.size) * weight, exact)


def _exact_distribution(pattern: tuple[int, ...], match_order: str) -> _KeyDistribution:
    """Keys over all n! orderings of letters with dense template ranks ``pattern``.

    ``pattern`` is sorted. Reversing a word and its template keeps every
    increasing run of letters in every window, so a pattern and its mirror
    (ranks ``max - rank``) have the same distribution and share one
    computation.
    """
    mirror = tuple(sorted(pattern[-1] - rank for rank in pattern))
    return _exact_distribution_of(min(pattern, mirror), match_order)


@functools.cache
def _exact_distribution_of(
    pattern: tuple[int, ...], match_order: str
) -> _KeyDistribution:
    """Keys over all n! orderings, scoring each distinct word once.

    Orderings that only exchange letters of one cell give the same word, so
    each distinct word stands for prod(m!) orderings, ``m`` the number of
    letters of each cell. The counts are exact.
    """
    words = _distinct_orderings(pattern)
    keys = np.concatenate(
        [
            _match_keys(words[start : start + _CHUNK_ROWS], match_order)
            for start in range(0, words.shape[0], _CHUNK_ROWS)
        ]
    )
    weight = math.prod(
        math.factorial(int(count))
        for count in np.unique(pattern, return_counts=True)[1]
    )
    return _distribution_from_keys(keys, exact=True, weight=weight)


def _sampled_distribution(
    pattern: tuple[int, ...], match_order: str, orderings: NDArray[np.intp]
) -> _KeyDistribution:
    """Keys over the given random orderings (rows of indices into ``pattern``)."""
    word = np.asarray(pattern, dtype=np.int16)
    return _distribution_from_keys(
        _match_keys(word[orderings], match_order), exact=False
    )


def _random_orderings(
    n: int, n_samples: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    return np.argsort(rng.random((n_samples, n)), axis=1)


def _at_most(value: Any, threshold: float) -> Any:
    """``value <= threshold``, counting values within round-off of it as equal."""
    return np.asarray(value) <= threshold + _tolerance(threshold)


def _passes(p_value: Any, alpha: float, comparison: str) -> Any:
    """``p_value < alpha`` or ``p_value <= alpha``, equal within round-off.

    A p-value such as 5/120 and an alpha of 1/24 are the same number but can
    differ in the last bit as floats; within the tolerance they are equal.
    NaN fails.
    """
    tolerance = _tolerance(alpha)
    p = np.asarray(p_value, dtype=np.float64)
    if comparison == "<=":
        return p <= alpha + tolerance
    return p < alpha - tolerance


def _dense(ranks: NDArray[np.intp]) -> NDArray[np.int16]:
    """Template ranks relabeled 0, 1, ... in increasing order, ties kept."""
    return np.unique(ranks, return_inverse=True)[1].reshape(-1).astype(np.int16)


def _word_distribution(
    dense: NDArray[np.int16],
    *,
    match_order: str,
    exact_max_letters: int,
    n_samples: int,
    rng: np.random.Generator,
) -> _KeyDistribution:
    pattern = tuple(int(value) for value in np.sort(dense))
    if dense.size <= exact_max_letters:
        return _exact_distribution(pattern, match_order)
    return _sampled_distribution(
        pattern, match_order, _random_orderings(dense.size, n_samples, rng)
    )


def _trial_group(
    n_letters: int, n_distinct: int, best_possible: float, trial_probability: float
) -> str | None:
    """Lee & Wilson's trial type of a word."""
    if n_letters == 2 and n_distinct == 2:
        return "pair"
    if n_letters == 3 and n_distinct == 3:
        return "triplet"
    if np.isfinite(best_possible) and bool(_at_most(best_possible, trial_probability)):
        return "low_probability"
    return None


def _sequence_match(
    ranks: NDArray[np.intp],
    trains: list[NDArray[np.float64]],
    start: float,
    stop: float,
    *,
    max_isi: float,
    match_order: str,
    trial_probability: float,
    exact_max_letters: int,
    n_samples: int,
    rng: np.random.Generator,
) -> SequenceMatchResult:
    units, times = _event_letters(trains, ranks, start, stop, max_isi)
    n = int(units.size)
    n_distinct = int(np.unique(units).size)
    if n < 2:
        return SequenceMatchResult(
            matched_letters=0,
            interruptions=0,
            match_probability=np.nan,
            best_possible_probability=np.nan,
            trial_group=None,
            n_letters=n,
            n_distinct_letters=n_distinct,
            letter_units=units,
            letter_times=times,
            exact=True,
            standard_error=0.0,
            match_order=match_order,
        )
    if n_distinct < 2:
        # Every ordering of one cell's letters has no match.
        return SequenceMatchResult(
            matched_letters=0,
            interruptions=0,
            match_probability=1.0,
            best_possible_probability=1.0,
            trial_group=None,
            n_letters=n,
            n_distinct_letters=n_distinct,
            letter_units=units,
            letter_times=times,
            exact=True,
            standard_error=0.0,
            match_order=match_order,
        )
    dense = _dense(ranks[units])
    key = int(_match_keys(dense[None, :], match_order)[0])
    distribution = _word_distribution(
        dense,
        match_order=match_order,
        exact_max_letters=exact_max_letters,
        n_samples=n_samples,
        rng=rng,
    )
    p_value = distribution.p_value(key)
    best_possible = (
        distribution.count(_perfect_key(n, n_distinct, match_order))
        / distribution.total
    )
    x, y = _decode_key(key, n, match_order)
    return SequenceMatchResult(
        matched_letters=x,
        interruptions=y,
        match_probability=float(p_value),
        best_possible_probability=float(best_possible),
        trial_group=_trial_group(n, n_distinct, best_possible, trial_probability),
        n_letters=n,
        n_distinct_letters=n_distinct,
        letter_units=units,
        letter_times=times,
        exact=distribution.exact,
        standard_error=0.0
        if distribution.exact
        else float(np.sqrt(p_value * (1.0 - p_value) / distribution.total)),
        match_order=match_order,
    )


def _kernel_peak(spikes: NDArray[np.float64], bandwidth: float) -> float:
    """Time of the largest value of the sum of Gaussian kernels on ``spikes``.

    Every maximum lies between the first and last spike. The best of a 1 ms
    grid and the spikes is refined by mean shift, which only climbs, to the
    local maximum; the earliest candidate wins a tie.
    """
    if spikes.size == 1 or spikes[0] == spikes[-1]:
        return float(spikes[0])
    candidates = np.union1d(np.arange(spikes[0], spikes[-1], _PEAK_GRID), spikes)
    density = np.exp(-0.5 * ((candidates[:, None] - spikes[None, :]) / bandwidth) ** 2)
    peak = float(candidates[np.argmax(density.sum(axis=1))])
    for _ in range(1000):
        weights = np.exp(-0.5 * ((peak - spikes) / bandwidth) ** 2)
        shifted = float(weights @ spikes / weights.sum())
        if abs(shifted - peak) <= 1e-12:
            return shifted
        peak = shifted
    return peak


@functools.lru_cache(maxsize=1024)
def _mahonian_cumulative(m: int) -> tuple[int, ...]:
    """Orders of ``m`` items with at most k inversions, for k = 0 ... m(m-1)/2."""
    counts = [1]
    for size in range(2, m + 1):
        prefix = [0]
        for value in counts:
            prefix.append(prefix[-1] + value)
        length = len(counts) + size - 1
        counts = [
            prefix[min(k + 1, len(counts))] - prefix[max(k - size + 1, 0)]
            for k in range(length)
        ]
    cumulative = []
    running = 0
    for value in counts:
        running += value
        cumulative.append(running)
    return tuple(cumulative)


@functools.lru_cache(maxsize=1024)
def _inversion_p_values(n_cells: int) -> NDArray[np.float64]:
    """P(at most k inversions) over the ``n_cells!`` orders, for each k."""
    from fractions import Fraction

    total = math.factorial(n_cells)
    values = np.array(
        [float(Fraction(count, total)) for count in _mahonian_cumulative(n_cells)]
    )
    values.flags.writeable = False
    return values


def _index_test_size(n_cells: int, alpha: float, comparison: str) -> float:
    """Largest p-value at ``n_cells`` that passes ``alpha``: the test's size."""
    if n_cells < 2:
        return 0.0
    p_values = _inversion_p_values(n_cells)
    allowed = p_values[_passes(p_values, alpha, comparison)]
    return float(allowed.max()) if allowed.size else 0.0


def _opposite_pairs(times: NDArray[np.float64], ranks: NDArray[Any]) -> Any:
    """Pairs not in template order (opposite or tied in time).

    ``ranks`` is (M,) or (n_rows, M); the count has its leading shape.
    """
    m = times.size
    upper = np.triu(np.ones((m, m), dtype=bool), k=1)
    time_sign = np.sign(times[None, :] - times[:, None])
    ranks = np.asarray(ranks)
    rank_sign = np.sign(ranks[..., None, :] - ranks[..., :, None])
    same = ((time_sign * rank_sign > 0) & upper).sum(axis=(-2, -1))
    return m * (m - 1) // 2 - same


def _active_firing_times(
    trains: list[NDArray[np.float64]],
    ranks: NDArray[np.intp],
    start: float,
    stop: float,
    bandwidth: float,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    active, times = [], []
    for unit in np.flatnonzero(ranks >= 0):
        train = trains[unit]
        lo, hi = np.searchsorted(train, [start, stop], side="left")
        if hi > lo:
            active.append(unit)
            times.append(_kernel_peak(train[lo:hi], bandwidth))
    return np.asarray(active, dtype=np.intp), np.asarray(times, dtype=np.float64)


def _matching_index(
    ranks: NDArray[np.intp],
    trains: list[NDArray[np.float64]],
    start: float,
    stop: float,
    *,
    bandwidth: float,
    min_active_cells: int,
) -> MatchingIndexResult:
    active, times = _active_firing_times(trains, ranks, start, stop, bandwidth)
    m = int(active.size)
    order = active[np.lexsort((-ranks[active], times))]
    pairs = m * (m - 1) // 2
    if m < min_active_cells:
        score, p_value, opposite = np.nan, np.nan, 0
    else:
        opposite = int(_opposite_pairs(times, ranks[active]))
        score = (pairs - 2 * opposite) / pairs
        p_value = float(_inversion_p_values(m)[opposite])
    return MatchingIndexResult(
        score=float(score),
        match_probability=float(p_value),
        n_active_cells=m,
        n_same_order_pairs=pairs - opposite if m >= min_active_cells else 0,
        n_opposite_order_pairs=opposite,
        active_units=active,
        firing_times=times,
        firing_order=order,
        bandwidth=float(bandwidth),
    )
