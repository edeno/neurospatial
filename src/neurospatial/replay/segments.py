"""Find the part of a decoded candidate event that is a replayed trajectory.

A candidate event (a ripple, a population burst, a stopping period) often
holds a trajectory for only part of its duration: the decoded position may
jump, blur or stall before and after it. Criteria-based replay detection
therefore first cuts the event into segments in which the decoded position
moves in small steps, and then keeps the longest segment (Pfeiffer & Foster
2013; Mallory et al. 2025) or every segment that is long and dispersed
enough (Widloski & Foster 2022). :func:`replay_segments` is one segmenter
whose parameters reproduce each published rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin
from neurospatial._validation import is_integer_at_least, validate_choice
from neurospatial.replay._compare import _compare
from neurospatial.replay.criteria import (
    _bin_distance_lookup,
    _check_metric,
    _increasing_times,
    _trajectory_metrics,
)

if TYPE_CHECKING:
    from neurospatial.environment import Environment

_OPERATION = "replay_segments()"
_METHODS = ("map", "expected")
_JUMP_RULES = ("between", "drop_earlier")


@dataclass(frozen=True, repr=False, eq=False)
class ReplaySegmentsResult(ReadOnlyArraysMixin, ResultMixin):
    """Segments of one decoded candidate event that pass a segmentation rule.

    Segments are in time order and do not overlap. Times are in seconds and
    distances in environment units (cm for an environment in cm).

    Attributes
    ----------
    segments : NDArray[np.int64], shape (n_segments, 2)
        First row and one past the last row of each segment in the
        posterior, so ``posterior[start:stop]`` is the segment. A segment
        spans every row between its first and last bin, including rows
        skipped by ``max_gap_bins`` or between merged pieces.
    start_times, end_times : NDArray[np.float64], shape (n_segments,)
        Center time of each segment's first and last time bin.
    durations : NDArray[np.float64], shape (n_segments,)
        ``end_times - start_times``. Add one time-bin width for the time
        from the start of the first bin to the end of the last.
    time_bin_counts : NDArray[np.int64], shape (n_segments,)
        Rows each segment spans, ``stop - start``.
    dispersions : NDArray[np.float64], shape (n_segments,)
        Root-mean-square distance of the posterior means of the segment's
        trajectory time bins from their average, as
        :attr:`TrajectoryMetrics.dispersion`.
    start_to_end_distances : NDArray[np.float64], shape (n_segments,)
        Distance between the decoded positions of each segment's first and
        last time bins.
    path_lengths : NDArray[np.float64], shape (n_segments,)
        Sum of the distances between consecutive trajectory time bins of
        each segment, across skipped and merged gaps.
    trajectory_time_bins : NDArray[np.bool_], shape (n_time_bins,)
        True for each time bin on a kept segment's trajectory. Rows that a
        segment spans but skips (a gap, or the rows between merged pieces)
        are False, except, with ``jump_rule="drop_earlier"``, a lone
        accepted time bin between merged pieces that lies within
        ``max_jump`` of the next time bin, which the Widloski & Foster
        (2022) code counts in the segment's dispersion. With
        ``"between"``, as in the Mallory et al. (2025) code, a lone bin is
        never on the trajectory.
    longest : int or None
        Index of the segment spanning the most time bins, the earliest on a
        tie; None without segments.
    method : {"map", "expected"}
        Decoded position used: MAP bin center or posterior mean.
    jump_rule : {"between", "drop_earlier"}
        Which time bins a jump removed.
    metric : {"geodesic", "euclidean"}
        How distances between MAP bins were measured. Distances between
        posterior means are Euclidean.
    units : str or None
        Units of distances (the environment's units).
    n_time_bins : int
        Number of time bins in the event.

    See Also
    --------
    replay_segments : Function that produces this result.
    """

    segments: NDArray[np.int64]
    start_times: NDArray[np.float64]
    end_times: NDArray[np.float64]
    durations: NDArray[np.float64]
    time_bin_counts: NDArray[np.int64]
    dispersions: NDArray[np.float64]
    start_to_end_distances: NDArray[np.float64]
    path_lengths: NDArray[np.float64]
    trajectory_time_bins: NDArray[np.bool_]
    longest: int | None
    method: str
    jump_rule: str
    metric: str
    units: str | None
    n_time_bins: int

    def __post_init__(self) -> None:
        """Own read-only copies of the arrays."""
        self._own_arrays()

    @property
    def n_segments(self) -> int:
        """Number of segments."""
        return len(self.segments)

    def rows(self, index: int | None = None) -> slice:
        """Rows of one segment, to index the posterior and times.

        Parameters
        ----------
        index : int or None, default=None
            Segment to select; None selects the longest.

        Returns
        -------
        slice
            ``slice(start, stop)``: ``posterior[result.rows()]`` and
            ``times[result.rows()]`` are the longest segment's rows.

        Raises
        ------
        ValueError
            If there are no segments.
        IndexError
            If ``index`` is not a segment's index.
        """
        if self.longest is None:
            raise ValueError(
                _format_error(
                    "rows(): the event has no segment that passes the rules.",
                    fix="check result.n_segments before selecting a segment, "
                    "or relax the thresholds of replay_segments()",
                )
            )
        if index is None:
            index = self.longest
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise IndexError(
                f"rows(): index must be an integer segment index, got {index!r}."
            )
        if not -self.n_segments <= index < self.n_segments:
            raise IndexError(
                f"rows(): segment {index} does not exist; there are "
                f"{self.n_segments} segments."
            )
        start, stop = self.segments[index]
        return slice(int(start), int(stop))

    def summary(self) -> dict[str, Any]:
        """Flat dict: the number of segments and the longest segment's values.

        The ``longest_*`` values are NaN without segments.
        """
        values: dict[str, Any] = {
            "method": self.method,
            "jump_rule": self.jump_rule,
            "metric": self.metric,
            "units": self.units,
            "n_time_bins": self.n_time_bins,
            "n_segments": self.n_segments,
            "longest": self.longest,
        }
        per_segment = {
            "start_time": self.start_times,
            "duration": self.durations,
            "time_bin_count": self.time_bin_counts,
            "dispersion": self.dispersions,
            "start_to_end_distance": self.start_to_end_distances,
            "path_length": self.path_lengths,
        }
        for name, array in per_segment.items():
            values[f"longest_{name}"] = (
                np.nan if self.longest is None else array[self.longest].item()
            )
        return values


def replay_segments(
    env: Environment,
    posterior: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    max_jump: float,
    method: Literal["map", "expected"] = "map",
    jump_rule: Literal["between", "drop_earlier"] = "between",
    max_gap_bins: int = 0,
    max_spread: float | None = None,
    min_peak_probability: float | None = None,
    merge_max_time_gap: float | None = None,
    merge_max_distance: float | None = None,
    min_duration: float | None = None,
    min_time_bins: int | None = None,
    min_dispersion: float | None = None,
    valid_time_bins: NDArray[np.bool_] | None = None,
    metric: Literal["geodesic", "euclidean"] | None = None,
) -> ReplaySegmentsResult:
    """Cut a decoded event into segments in which the position moves smoothly.

    Criteria-based replay detection keeps the part of a candidate event in
    which the decoded position moves in small steps: the longest such
    segment (Pfeiffer & Foster 2013, 2015; Igata et al. 2021; Mallory et
    al. 2025), or every segment that lasts and travels long enough
    (Widloski & Foster 2022; Carey et al. 2019). The steps are applied in
    this order:

    1. **Reject time bins** marked invalid in ``valid_time_bins``, whose
       posterior spread is not below ``max_spread``, or whose peak
       probability is not above ``min_peak_probability``.
    2. **Split at jumps.** A jump is the distance between the decoded
       positions of consecutive time bins ``t - 1`` and ``t``; a jump not
       below ``max_jump`` separates them. ``jump_rule`` sets whether a bin
       is lost.
    3. **Bridge gaps.** With ``max_gap_bins > 0``, a segment continues
       past up to that many time bins to the next bin within ``max_jump``.
    4. **Merge** neighboring segments separated by less than
       ``merge_max_time_gap`` and ``merge_max_distance``. Pieces of a
       single time bin have no step and are dropped before merging, as the
       Widloski & Foster (2022) and Mallory et al. (2025) code drops them.
    5. **Filter** segments by ``min_duration``, ``min_time_bins`` and
       ``min_dispersion``.

    Works on 1-D environments, linearized tracks and mazes, and 2-D or 3-D
    arenas. For a segment without skipped or merged rows, its own metrics
    follow from
    ``trajectory_metrics(env, posterior[result.rows(i)], times[result.rows(i)])``;
    otherwise ``rows(i)`` also holds the skipped rows, which
    ``trajectory_time_bins`` marks False.

    Parameters
    ----------
    env : Environment
        The environment the posterior was decoded on.
    posterior : NDArray[np.float64], shape (n_time_bins, n_bins)
        Posterior probability mass per bin in each time bin of the event.
        Each row sums to 1, or to less than 1 when it is one template's
        share of a joint decode, ``result.joint(name)``; positions, jumps
        and spread then use each row rescaled to sum to 1, and the peak
        probability uses the mass as given, as in
        :func:`trajectory_metrics`.
    times : NDArray[np.float64], shape (n_time_bins,)
        Center time of each posterior row, in seconds, increasing.
    max_jump : float
        Largest distance, in environment units, the decoded position may
        move between consecutive time bins of a segment; a jump of
        ``max_jump`` or more separates them. Published values range from
        20 cm to 40% of the track (see Examples), so there is no default.
    method : {"map", "expected"}, default="map"
        Decoded position of each time bin. ``"map"`` is the center of the
        MAP bin (Pfeiffer & Foster 2013; Igata et al. 2021; Carey et al.
        2019; Mallory et al. 2025); ``"expected"`` is the posterior mean
        (Pfeiffer & Foster 2015; Widloski & Foster 2022).
    jump_rule : {"between", "drop_earlier"}, default="between"
        Which time bin a jump between ``t - 1`` and ``t`` removes.
        ``"between"`` removes none: one segment ends at ``t - 1`` and the
        next starts at ``t`` (Pfeiffer & Foster 2013, "the longest
        sequence of time frames with peak posterior probability less than
        20 cm from that of the previous frame"; Carey et al. 2019; Mallory
        et al. 2025, see Notes). ``"drop_earlier"`` rejects bin ``t - 1``,
        which is how the Widloski & Foster (2022) code assigns each bin the
        jump to the next bin; a lone outlier bin then costs its neighbor
        too, and merging can bridge the gap.
    max_gap_bins : int, default=0
        Time bins a segment may skip (Igata et al. 2021, "allowing a 1-gap
        time bin"). From a segment's last bin ``i``, the segment continues
        to the earliest bin ``j`` with ``j - i - 1 <= max_gap_bins`` that is
        not rejected and lies less than ``max_jump`` from bin ``i``; the
        bins between are skipped. With 0, a segment holds consecutive bins
        only.
    max_spread : float or None, default=None
        Reject time bins whose posterior spread
        (:attr:`TrajectoryMetrics.spread`) is not below this, in
        environment units (Widloski & Foster 2022: 10 cm). None keeps all.
    min_peak_probability : float or None, default=None
        Reject time bins whose posterior mass in the MAP bin, as given, is
        not above this (Igata et al. 2021 and the Mallory et al. 2025 code:
        5 times chance, ``5 / n_bins`` over all decoded bins, both
        templates of a joint decode). None keeps all.
    merge_max_time_gap, merge_max_distance : float or None, default=None
        Merge consecutive segments when the time between the last bin of
        one and the first bin of the next is below ``merge_max_time_gap``
        seconds and the distance between their decoded positions is below
        ``merge_max_distance`` (Widloski & Foster 2022: 50 ms and 20 cm).
        The merged segment spans both and the rows between them, which are
        not trajectory time bins. Give both or neither.
    min_duration : float or None, default=None
        Drop segments whose duration, last minus first time-bin center, is
        below this, in seconds (Mallory et al. 2025 code: 30 ms; Widloski &
        Foster 2022: 100 ms).
    min_time_bins : int or None, default=None
        Drop segments spanning fewer time bins than this (Carey et al.
        2019: 10; Pfeiffer & Foster 2013: 11, see Notes; Igata et al. 2021:
        5).
    min_dispersion : float or None, default=None
        Drop segments whose dispersion is below this, in environment units
        (Widloski & Foster 2022: 12 cm).
    valid_time_bins : NDArray[np.bool_] or None, shape (n_time_bins,), default=None
        False for time bins that must not be part of a trajectory, such as
        bins with too few active cells (Carey et al. 2019: fewer than 4) or
        while the animal runs (Widloski & Foster 2022: 5 cm/s or faster).
        They are rejected like a blurred bin: they split segments, and
        merging may bridge them. With ``jump_rule="drop_earlier"``, the bin
        before an invalid bin is rejected too, because its jump to the next
        position is undefined. None treats every time bin as valid. Keep the
        rows in ``posterior`` rather than deleting them: deleting rows makes
        their neighbors consecutive and measures a jump across them.
    metric : {"geodesic", "euclidean"} or None, default=None
        Distance between MAP bins, as in :func:`trajectory_metrics`: None
        uses geodesic on a track or 1-D environment and Euclidean in an
        arena. Distances between posterior means are Euclidean.

    Returns
    -------
    ReplaySegmentsResult
        The segments that pass, in time order, with their durations,
        dispersions and distances; ``longest`` indexes the longest and
        ``rows(i)`` selects a segment's rows.

    Raises
    ------
    ValueError
        If ``method``, ``jump_rule`` or ``metric`` is not one of the
        options; if a threshold is out of range; if only one of the merge
        thresholds is given; if ``valid_time_bins`` is not one boolean per
        row; if ``posterior`` does not have one column per
        bin, or a row has NaN, Inf or negative entries, no mass, or more
        than 1; or if ``times`` does not have one finite, increasing value
        per row.
    CartesianOnlyError
        If ``env`` is an egocentric polar environment.

    See Also
    --------
    trajectory_metrics : Per-time-bin positions, jumps, spread and peak.
    template_fractions : Share of a joint decode in each template.

    Notes
    -----
    **Thresholds.** Jumps, spread and merge distances and times must be
    strictly below their maximum, and the peak probability strictly above
    its minimum, as in the papers. A segment passes a minimum duration, time
    bin count or dispersion when it reaches it. The published code compares
    whole bins and time steps; a value within :math:`10^{-9}` (absolute
    plus relative) of a threshold counts as equal to it, so that rounding in
    times such as ``t0 + 0.005 * k`` or in distances that are multiples of
    the bin size does not flip a decision.

    **Longest.** The longest segment spans the most time bins; the
    earliest wins a tie. Filters apply before the choice, as in the Mallory
    et al. (2025) code, which drops segments shorter than 30 ms and then
    keeps the longest. Pfeiffer & Foster (2013, 2015) and Igata et al.
    (2021) choose the longest and then test it; for a count or duration
    threshold the order does not matter, and their distance criteria are
    tested on the chosen segment (``start_to_end_distances``,
    ``path_lengths``).

    **Reading of the published methods.**

    - Pfeiffer & Foster (2013) eliminate events "with fewer than 10 steps
      in the final sequence". Pfeiffer & Foster (2015) define a step as
      the move between two consecutive frames, so 10 steps are 11 time
      bins: ``min_time_bins=11``.
    - The Widloski & Foster (2022) code (``compute_sequenceJumps.m`` with
      ``jumps = [jumps; jumps(end)]``) gives bin ``t - 1`` the jump to bin
      ``t``, so a large jump rejects the earlier bin. The appended last
      value also rejects the event's last bin when the final jump is large;
      that padding is not reproduced, and the last bin, which has no next
      bin, is never rejected by a jump. Segments are merged by the time
      between bin centers and the Euclidean distance between posterior
      means. The code computes dispersion over the merged segment's
      accepted bins only, ignoring the rejected rows between pieces, as
      ``dispersions`` does here; a lone accepted bin between pieces counts.
      Its bins at running speed of 5 cm/s or more have no decoded position;
      pass them as ``valid_time_bins=False``. The code's final tests are
      strict, duration > 100 ms and dispersion > 12 cm, while
      ``min_duration`` and ``min_dispersion`` pass a segment that reaches
      them; on 5 ms steps, ``min_time_bins=22`` gives the strict duration
      test.
    - The Mallory et al. (2025) code (``filter_candidate_events.m``) rejects
      bin ``t`` when the MAP jump from ``t - 1`` is at least the threshold
      (``jumps = [0; abs(diff(x))]``), but its run finder
      (``compute_allSequences_NaNseparated_cm.m``) then moves the start of
      every run back by one bin, "to keep the time bin following the big
      jump as part of the next sequence". A jump therefore splits the
      event without losing a bin, ``jump_rule="between"``. The move also
      applies after a bin rejected for low peak probability, which then
      opens the next run, counts toward its length and is the point the
      merge step measures from, until the code removes low-probability
      bins from the final segment's ends; here, rejected bins never belong
      to a segment. The run finder's helper ``seqle`` is not deposited; its
      run lengths are read as last minus first index, as the merge step
      computes them.
    - The Mallory code merges runs by calling
      ``compute_allSequences_NaNseparated_merge(x, boundaries, jumpThr,
      ..., binSize)`` with ``jumpThr`` already in bins, and the function
      divides it by ``binSize`` again, so runs merge only within half the
      jump threshold (2 cm bins). The intended threshold is used here:
      pass ``merge_max_distance`` equal to ``max_jump``.
    - Igata et al. (2021) do not say whether the bin skipped by "a 1-gap
      time bin" may be a bin below the peak threshold, a bin whose MAP is
      out of range, or both. Both are skipped here, and a segment continues
      to the earliest bin within reach. They also computed the MAP of each
      path of the maze separately and extended sequences into neighboring
      paths; the example below uses one MAP over the whole maze.
    - The Carey et al. (2019) code (``DecSeqDetectZ.m``) keeps jumps of
      ``round(40 / bin_size)`` bins or fewer; on their 3 cm bins this is
      the same as ``max_jump=40.0``. It runs over whole sessions and keeps
      every sequence. Its time bins with fewer than 4 active cells are
      NaN, which break sequences: pass them as ``valid_time_bins=False``.

    References
    ----------
    Carey, A. A., Tanaka, Y., & van der Meer, M. A. A. (2019). Reward
    revaluation biases hippocampal replay content away from the preferred
    outcome. *Nature Neuroscience*, 22(9), 1450-1459.

    Igata, H., Ikegaya, Y., & Sasaki, T. (2021). Prioritized experience
    replays on a hippocampal predictive map for learning. *Proceedings of
    the National Academy of Sciences*, 118(1), e2011266118.

    Mallory, C. S., Widloski, J., & Foster, D. J. (2025). The time course and
    organization of hippocampal replay. *Science*, 387(6733), 541-548.

    Pfeiffer, B. E., & Foster, D. J. (2013). Hippocampal place-cell sequences
    depict future paths to remembered goals. *Nature*, 497(7447), 74-79.

    Pfeiffer, B. E., & Foster, D. J. (2015). Autoassociative dynamics in the
    generation of sequences of hippocampal place cells. *Science*,
    349(6244), 180-183.

    Widloski, J., & Foster, D. J. (2022). Flexible rerouting of hippocampal
    replay sequences around changing barriers in the absence of global
    place field remapping. *Neuron*, 110(9), 1547-1558.

    Examples
    --------
    Published settings, in environment units of cm and times in seconds:

    - Pfeiffer & Foster (2013): ``max_jump=20.0, method="map",
      min_time_bins=11``; keep ``result.longest`` if its
      ``start_to_end_distances`` is at least 40 cm.
    - Pfeiffer & Foster (2015): ``max_jump=50.0, method="expected",
      min_time_bins=11``; start-to-end distance at least 80 cm.
    - Igata et al. (2021): ``max_jump=40.0, method="map", max_gap_bins=1,
      min_peak_probability=5 / n_bins, min_time_bins=5``; path length of
      the longest above 20 cm.
    - Carey et al. (2019): ``max_jump=40.0, method="map",
      min_time_bins=10``; every segment.
    - Widloski & Foster (2022): ``max_jump=20.0, method="expected",
      jump_rule="drop_earlier", max_spread=10.0, merge_max_time_gap=0.05,
      merge_max_distance=20.0, min_duration=0.1, min_dispersion=12.0``;
      every segment.
    - Mallory et al. (2025): on the template share ``result.joint(name)``
      with more posterior, ``max_jump=0.4 * track_length, method="map",
      min_peak_probability=5 / (2 * env.n_bins), merge_max_time_gap=0.05,
      merge_max_distance=0.4 * track_length, min_duration=0.03``; keep
      ``result.longest``. The code compares the templates' posterior with
      10 bins (20 cm) at each end of the track left out, and rejects the
      segment if it holds more posterior in the other template.

    A sweep along a 100 cm track that jumps from bin 5 to bin 9 after five
    time bins and then runs back:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.replay import replay_segments, trajectory_metrics
    >>> env = Environment.linear_track(
    ...     endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    ... )
    >>> posterior = np.eye(env.n_bins)[[1, 2, 3, 4, 5, 9, 8, 7]]
    >>> times = np.arange(8) * 0.02
    >>> result = replay_segments(env, posterior, times, max_jump=20.0)
    >>> result.segments.tolist()
    [[0, 5], [5, 8]]
    >>> result.longest, result.start_to_end_distances.round(6).tolist()
    (0, [40.0, 20.0])

    With ``jump_rule="drop_earlier"`` the jump also removes bin 4:

    >>> replay_segments(
    ...     env, posterior, times, max_jump=20.0, jump_rule="drop_earlier"
    ... ).segments.tolist()
    [[0, 4], [5, 8]]

    Metrics of the longest segment:

    >>> rows = result.rows()
    >>> metrics = trajectory_metrics(env, posterior[rows], times[rows])
    >>> round(metrics.map_path_length, 6), round(metrics.duration, 3)
    (40.0, 0.08)
    """
    checked = _checked_options(
        _OPERATION,
        max_jump=max_jump,
        method=method,
        jump_rule=jump_rule,
        max_gap_bins=max_gap_bins,
        max_spread=max_spread,
        min_peak_probability=min_peak_probability,
        merge_max_time_gap=merge_max_time_gap,
        merge_max_distance=merge_max_distance,
        min_duration=min_duration,
        min_time_bins=min_time_bins,
        min_dispersion=min_dispersion,
        metric=metric,
    )
    max_jump = checked["max_jump"]
    max_gap_bins = checked["max_gap_bins"]
    max_spread = checked["max_spread"]
    min_peak_probability = checked["min_peak_probability"]
    merge_max_time_gap = checked["merge_max_time_gap"]
    merge_max_distance = checked["merge_max_distance"]
    min_duration = checked["min_duration"]
    min_time_bins = checked["min_time_bins"]
    min_dispersion = checked["min_dispersion"]

    metrics = _trajectory_metrics(
        env,
        posterior,
        times,
        spike_counts=None,
        metric=metric,
        operation=_OPERATION,
    )
    times = _increasing_times(times, metrics.n_time_bins, _OPERATION)
    n_time_bins = metrics.n_time_bins
    distance = _distance_function(env, metrics, method)
    jumps = metrics.map_jumps if method == "map" else metrics.mean_jumps

    valid = (
        np.ones(n_time_bins, dtype=bool)
        if valid_time_bins is None
        else _valid_time_bins(valid_time_bins, n_time_bins)
    )
    accepted = valid.copy()
    if max_spread is not None:
        accepted &= _compare(metrics.spread, "<", max_spread)
    if min_peak_probability is not None:
        accepted &= _compare(metrics.peak_probability, ">", min_peak_probability)
    if jump_rule == "drop_earlier":
        accepted[:-1] &= _compare(jumps, "<", max_jump) & valid[1:]

    # The bin each time bin's segment continues to: the earliest accepted
    # bin within reach, or -1. Writing the farthest offset first lets the
    # nearest overwrite it.
    successor = np.full(n_time_bins, -1, dtype=np.intp)
    for offset in range(max_gap_bins + 1, 0, -1):
        earlier = np.arange(n_time_bins - offset)
        later = earlier + offset
        reach = accepted[earlier] & accepted[later]
        reach[reach] = _compare(distance(earlier[reach], later[reach]), "<", max_jump)
        successor[earlier[reach]] = later[reach]

    on_trajectory = np.zeros(n_time_bins, dtype=bool)
    pieces = []
    t = 0
    while t < n_time_bins:
        if not accepted[t]:
            t += 1
            continue
        start = t
        on_trajectory[t] = True
        while successor[t] >= 0:
            t = int(successor[t])
            on_trajectory[t] = True
        pieces.append((start, t + 1))
        t += 1
    bounds = np.array(pieces, dtype=np.int64).reshape(-1, 2)
    lone = bounds[bounds[:, 1] - bounds[:, 0] == 1, 0]
    on_trajectory[lone] = False
    bounds = bounds[bounds[:, 1] - bounds[:, 0] > 1]

    if (
        merge_max_time_gap is not None
        and merge_max_distance is not None
        and len(bounds) > 1
    ):
        last = bounds[:-1, 1] - 1
        first = bounds[1:, 0]
        join = _compare(times[first] - times[last], "<", merge_max_time_gap) & _compare(
            distance(last, first), "<", merge_max_distance
        )
        bounds = np.column_stack(
            [bounds[np.r_[True, ~join], 0], bounds[np.r_[~join, True], 1]]
        )
        if jump_rule == "drop_earlier":
            # A lone accepted bin inside a merged segment leads on to the
            # next bin, so the Widloski & Foster (2022) code keeps it on the
            # trajectory. It never ends a segment: lone + 1 is inside too.
            span = np.searchsorted(bounds[:, 0], lone, side="right") - 1
            inside = (span >= 0) & (lone < bounds[np.maximum(span, 0), 1])
            lone = lone[inside]
            on_trajectory[lone[_compare(distance(lone, lone + 1), "<", max_jump)]] = (
                True
            )

    starts, stops = bounds[:, 0], bounds[:, 1]
    durations = times[stops - 1] - times[starts]
    counts = stops - starts
    dispersions, path_lengths = _per_segment(
        starts, on_trajectory, metrics.mean_positions, distance
    )
    keep = np.ones(len(bounds), dtype=bool)
    if min_duration is not None:
        keep &= _compare(durations, ">=", min_duration)
    if min_time_bins is not None:
        keep &= counts >= min_time_bins
    if min_dispersion is not None:
        keep &= _compare(dispersions, ">=", min_dispersion)
    for start, stop in bounds[~keep]:
        on_trajectory[start:stop] = False
    bounds, starts, stops = bounds[keep], starts[keep], stops[keep]
    counts = counts[keep]

    return ReplaySegmentsResult(
        segments=bounds,
        start_times=times[starts],
        end_times=times[stops - 1],
        durations=durations[keep],
        time_bin_counts=counts,
        dispersions=dispersions[keep],
        start_to_end_distances=distance(starts, stops - 1),
        path_lengths=path_lengths[keep],
        trajectory_time_bins=on_trajectory,
        longest=int(np.argmax(counts)) if len(counts) else None,
        method=method,
        jump_rule=jump_rule,
        metric=metrics.metric,
        units=metrics.units,
        n_time_bins=n_time_bins,
    )


def _valid_time_bins(
    valid_time_bins: Any, n_time_bins: int, operation: str = _OPERATION
) -> NDArray[np.bool_]:
    """One boolean per time bin."""
    valid = np.asarray(valid_time_bins)
    if valid.dtype != np.bool_ or valid.shape != (n_time_bins,):
        raise ValueError(
            _format_error(
                f"{operation}: valid_time_bins must be a boolean array of shape "
                f"({n_time_bins},), got dtype {valid.dtype} and shape "
                f"{valid.shape}.",
                why="it marks each posterior row as usable or not",
                fix="pass one True/False per time bin, e.g. n_active_cells >= 4",
            )
        )
    return valid


def _distance_function(env: Environment, metrics: Any, method: str) -> Any:
    """Distance between the decoded positions of time bins ``first`` and ``second``.

    MAP positions use the event's metric; posterior means are Euclidean.
    """
    if method == "expected":
        means = metrics.mean_positions

        def between_means(
            first: NDArray[np.intp], second: NDArray[np.intp]
        ) -> NDArray[np.float64]:
            return np.asarray(
                np.linalg.norm(means[second] - means[first], axis=-1),
                dtype=np.float64,
            )

        return between_means
    between, rows_of = _bin_distance_lookup(env, metrics.map_bins, metrics.metric)

    def between_map_bins(
        first: NDArray[np.intp], second: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        return np.asarray(between[rows_of[first], rows_of[second]], dtype=np.float64)

    return between_map_bins


def _per_segment(
    starts: NDArray[np.int64],
    on_trajectory: NDArray[np.bool_],
    means: NDArray[np.float64],
    distance: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Dispersion and path length of each segment's trajectory time bins."""
    bins = np.flatnonzero(on_trajectory)
    segment = np.searchsorted(starts, bins, side="right") - 1
    n_segments = len(starts)
    n_bins = np.bincount(segment, minlength=n_segments)
    centers = (
        np.stack(
            [
                np.bincount(segment, weights=means[bins, axis], minlength=n_segments)
                for axis in range(means.shape[1])
            ],
            axis=1,
        )
        / np.maximum(n_bins, 1)[:, None]
    )
    squared = np.sum((means[bins] - centers[segment]) ** 2, axis=1)
    dispersions = np.sqrt(
        np.bincount(segment, weights=squared, minlength=n_segments)
        / np.maximum(n_bins, 1)
    )
    same = segment[:-1] == segment[1:]
    steps = distance(bins[:-1][same], bins[1:][same])
    path_lengths = np.bincount(segment[:-1][same], weights=steps, minlength=n_segments)
    return dispersions, path_lengths.astype(np.float64)


def _checked_options(
    operation: str,
    *,
    max_jump: Any,
    method: Any,
    jump_rule: Any,
    max_gap_bins: Any,
    max_spread: Any,
    min_peak_probability: Any,
    merge_max_time_gap: Any,
    merge_max_distance: Any,
    min_duration: Any,
    min_time_bins: Any,
    min_dispersion: Any,
    metric: Any,
) -> dict[str, Any]:
    """The scalar options of :func:`replay_segments`, checked.

    Thresholds come back as floats and counts as ints; errors name
    ``operation``.
    """
    validate_choice(method, _METHODS, name="method", owner=operation)
    if jump_rule not in _JUMP_RULES:
        raise ValueError(
            _format_error(
                f"{operation}: jump_rule must be one of {_JUMP_RULES}, got "
                f"{jump_rule!r}.",
                why="a jump either just separates two segments or also "
                "rejects the time bin before it",
                fix="use 'between' (Pfeiffer & Foster 2013; the Mallory et al. "
                "2025 code) or 'drop_earlier' (the Widloski & Foster 2022 code)",
            )
        )
    max_jump = _threshold("max_jump", max_jump, minimum="positive", operation=operation)
    max_gap_bins = _count("max_gap_bins", max_gap_bins, minimum=0, operation=operation)
    max_spread = _optional_threshold(
        "max_spread", max_spread, minimum="positive", operation=operation
    )
    min_peak_probability = _optional_threshold(
        "min_peak_probability",
        min_peak_probability,
        minimum="zero",
        operation=operation,
    )
    if min_peak_probability is not None and min_peak_probability >= 1:
        raise ValueError(
            f"{operation}: min_peak_probability must be below 1, got "
            f"{min_peak_probability}; every time bin would be rejected."
        )
    if (merge_max_time_gap is None) != (merge_max_distance is None):
        raise ValueError(
            _format_error(
                f"{operation}: merge_max_time_gap and merge_max_distance "
                "must be given together.",
                why="segments merge when both their time gap and their "
                "distance are small",
                fix="pass both, such as merge_max_time_gap=0.05, "
                "merge_max_distance=20.0 (Widloski & Foster 2022), or neither",
            )
        )
    merge_max_time_gap = _optional_threshold(
        "merge_max_time_gap",
        merge_max_time_gap,
        minimum="positive",
        operation=operation,
    )
    merge_max_distance = _optional_threshold(
        "merge_max_distance",
        merge_max_distance,
        minimum="positive",
        operation=operation,
    )
    min_duration = _optional_threshold(
        "min_duration", min_duration, minimum="zero", operation=operation
    )
    min_dispersion = _optional_threshold(
        "min_dispersion", min_dispersion, minimum="zero", operation=operation
    )
    if min_time_bins is not None:
        min_time_bins = _count(
            "min_time_bins", min_time_bins, minimum=1, operation=operation
        )
    _check_metric(metric, operation)
    return {
        "max_jump": max_jump,
        "method": method,
        "jump_rule": jump_rule,
        "max_gap_bins": max_gap_bins,
        "max_spread": max_spread,
        "min_peak_probability": min_peak_probability,
        "merge_max_time_gap": merge_max_time_gap,
        "merge_max_distance": merge_max_distance,
        "min_duration": min_duration,
        "min_time_bins": min_time_bins,
        "min_dispersion": min_dispersion,
        "metric": metric,
    }


def _threshold(
    name: str,
    value: Any,
    *,
    minimum: Literal["positive", "zero"],
    operation: str,
) -> float:
    """A finite threshold, positive or non-negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(
            f"{operation}: {name} must be a number, got {type(value).__name__}."
        )
    threshold = float(value)
    too_small = threshold <= 0 if minimum == "positive" else threshold < 0
    if not np.isfinite(threshold) or too_small:
        bound = "> 0" if minimum == "positive" else ">= 0"
        raise ValueError(
            f"{operation}: {name} must be a finite number {bound}, got {value!r}."
        )
    return threshold


def _optional_threshold(
    name: str,
    value: Any,
    *,
    minimum: Literal["positive", "zero"],
    operation: str,
) -> float | None:
    """None, or a checked threshold."""
    if value is None:
        return None
    return _threshold(name, value, minimum=minimum, operation=operation)


def _count(name: str, value: Any, *, minimum: int, operation: str) -> int:
    """An integer of at least ``minimum``."""
    if not is_integer_at_least(value, minimum):
        raise ValueError(
            f"{operation}: {name} must be an integer >= {minimum}, got {value!r}."
        )
    return int(value)
