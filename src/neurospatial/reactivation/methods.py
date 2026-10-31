# The recipes' notes quote the papers verbatim, with their symbols.
# ruff: noqa: RUF001
"""Published reactivation methods as explicit, comparable recipes.

Each :class:`ReactivationMethod` records one paper's reactivation analysis:
the time bin, which cell pairs count, the windows of the template, control
and match periods (:class:`ReactivationWindows`), the statistic and the
within-session test. :func:`compare_reactivation_methods` bins one
session's spike times for each method and applies it, so the same
recording can be analyzed the way each lab analyzed it.

Three families are covered:

- Explained variance (EV): the pair-correlation pattern of a template
  period (running) predicts that of a match period (rest after running)
  beyond what a control period (rest before) already predicts (Kudrimoti
  et al. 1999), and the reversed EV (REV) that swaps the rest periods
  (Pennartz et al. 2004).
- Assembly patterns: patterns detected in the template period, and their
  activation strength R(t) in rest after minus rest before (Peyrache et
  al. 2010; van de Ven et al. 2016), or around sharp-wave ripples (SWRs)
  in each (Harvey et al. 2023; Liu et al. 2023; Chang et al. 2025;
  Robinson et al. 2026).
- Member co-firing: the correlation of each pattern's members over the
  SWRs of rest before and after (El-Gaby et al. 2021).

Some recipes count spikes in events instead of fixed bins: one bin per SWR
of rest, or per theta cycle of the task (``rest_bins="events"``,
``template_bins="events"``), with the events passed to
:func:`compare_reactivation_methods`.

Unlike the replay recipes, which record their decoding settings for
reference only, a reactivation recipe applies its ``bin_size``: binning is
part of the statistic (a correlation of 100 ms counts is not one of 250 ms
counts). Choices a paper does not specify are listed in each recipe's
``notes``.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals, times_in_intervals
from neurospatial._spikes import as_labelled_spike_trains
from neurospatial._stacklevel import _warnings_with_context
from neurospatial._time_bins import _count_spikes_in_time_bins
from neurospatial.reactivation.assemblies import (
    AssemblyPattern,
    _activation_strength,
    _z_scores,
    detect_assemblies,
    explained_variance_reactivation,
)
from neurospatial.reactivation.calibration import equalize_firing_rates
from neurospatial.reactivation.coactivity import (
    _in_lag_window,
    _lags,
    _peri_event_values,
    _signed_rank_p,
    assembly_member_cofiring,
    ev_rev_contrast,
)
from neurospatial.reactivation.coactivity import (
    _nanmean as _nanmean_along,
)
from neurospatial.stats._utils import _entropy, _name_key, _stream

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "CHANG_2025_ASSEMBLY",
    "EL_GABY_2021",
    "GIRI_2019",
    "GIRI_2024",
    "HARVEY_2023_ASSEMBLY",
    "KUDRIMOTI_1999",
    "LANSINK_2009",
    "LIU_2023_ASSEMBLY",
    "LIU_2023_EV",
    "LIU_2023_PAIRWISE",
    "OLIVA_2020_ASSEMBLY",
    "OLIVA_2020_EV",
    "PENNARTZ_2004",
    "PEYRACHE_2010",
    "PUBLISHED_REACTIVATION_METHODS",
    "ROBINSON_2026_ASSEMBLY",
    "ROBINSON_2026_PAIRWISE",
    "TATSUNO_2006",
    "VAN_DE_VEN_2016",
    "EventResponse",
    "ReactivationMethod",
    "ReactivationWindows",
    "compare_reactivation_methods",
]

_CALL = "compare_reactivation_methods()"
_MEASURES = ("explained_variance", "assembly", "member_cofiring")
_PAIRS = ("all", "different_tetrodes", "different_shanks", "different_regions")
# The unit_groups key each pair rule reads.
_GROUP_OF_PAIRS = {
    "different_tetrodes": "tetrode",
    "different_shanks": "shank",
    "different_regions": "region",
}
_STATISTICS = {
    "explained_variance": (
        "explained_variance",
        "ev_minus_rev",
        "ev_rev_contrast",
        "partial_correlation",
        "semipartial_correlation",
    ),
    "assembly": (
        "mean_activation_difference",
        "event_response_difference",
        "event_response_ratio",
    ),
    "member_cofiring": ("mean_member_correlation_difference",),
}
# Statistics of one control window: they are not averaged over windows.
_ONE_CONTROL_STATISTICS = ("partial_correlation", "semipartial_correlation")
_EVENT_RESPONSE_STATISTICS = ("event_response_difference", "event_response_ratio")
_TESTS = {
    "explained_variance": (None, "partial_correlation_t", "pair_bootstrap"),
    "assembly": (None, "cell_identity_shuffle", "event_modulation"),
    "member_cofiring": (None,),
}
_DETECTIONS = ("pca", "ica")
_BINS = ("fixed", "events")
_MEMBER_RULES = ("abs_z", "positive_z", "otsu", "inverse_sqrt_n")
_Z_MEMBER_RULES = ("abs_z", "positive_z")
_DETECTION_UNITS = ("all", "each_region", "each_region_and_cross_region")
_ZSCORE_OVER = ("period", "control_and_match", "all_periods")


def _is_count(value: Any) -> bool:
    """Whether ``value`` is a positive integer, not a bool or a float."""
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and bool(value >= 1)
    )


@dataclass(frozen=True)
class ReactivationWindows:
    """How a period's epochs are cut into analysis windows.

    Attributes
    ----------
    length : float or None
        Window length in seconds. None (default): one window holding all the
        period's epochs.
    step : float or None
        Seconds between the starts of consecutive windows; None (default)
        is ``length``, adjacent blocks. A step shorter than ``length`` gives
        sliding windows (Giri et al. 2019: "15 min sliding windows (window
        length 15 min, sliding 5 min steps)").
    anchor : {"start", "end"}
        Cut windows from the start of the period forward or from its end
        backward; ``"end"`` with ``count=1`` is the last ``length`` seconds
        (Kudrimoti et al. 1999: "the last 10 min of baseline SWS").
    count : int or None
        Keep only the first ``count`` windows from the anchor; None
        (default) keeps every whole window. A remainder shorter than
        ``length`` is dropped.
    clock : {"concatenated", "wall"}
        ``"concatenated"`` (default) measures window lengths in time inside
        the epochs, as if they were joined end to end (Lansink et al. 2009:
        "20-min time blocks of concatenated periods of quiet rest and
        sleep"). ``"wall"`` measures them on the clock from the first epoch's
        start to the last epoch's stop, and keeps the epochs' time inside
        each window (Pennartz et al. 2004: "the last 10 min of REST1,
        immediately preceding the behavioral phase"). The two agree when a
        period is one epoch.
    allow_shorter : bool
        False (default): a period shorter than ``length`` gives no window,
        and a method using it reports an error. True: it gives one window
        holding all of it, for "the first hour of sleep, or all of it when
        shorter", as the labs' code truncates sleep (Harvey et al. 2023;
        Liu et al. 2023). A remainder after whole windows is still dropped.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reactivation import ReactivationWindows
    >>> epochs = np.array([[0.0, 400.0], [500.0, 900.0]])  # 800 s of sleep
    >>> blocks = ReactivationWindows(length=300.0)
    >>> [w.tolist() for w in blocks.split(epochs)]
    [[[0.0, 300.0]], [[300.0, 400.0], [500.0, 700.0]]]
    >>> last = ReactivationWindows(length=300.0, anchor="end", count=1, clock="wall")
    >>> [w.tolist() for w in last.split(epochs)]
    [[[600.0, 900.0]]]
    >>> first_hour = ReactivationWindows(length=3600.0, count=1, allow_shorter=True)
    >>> [w.tolist() for w in first_hour.split(epochs)]  # all 800 s
    [[[0.0, 400.0], [500.0, 900.0]]]
    """

    length: float | None = None
    step: float | None = None
    anchor: Literal["start", "end"] = "start"
    count: int | None = None
    clock: Literal["concatenated", "wall"] = "concatenated"
    allow_shorter: bool = False

    def __post_init__(self) -> None:
        """Check the window options."""
        for name in ("length", "step"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"ReactivationWindows: {name} must be a positive number of "
                    f"seconds or None, got {value!r}."
                )
        if not isinstance(self.allow_shorter, (bool, np.bool_)):
            raise ValueError(
                "ReactivationWindows: allow_shorter must be True or False, got "
                f"{self.allow_shorter!r}."
            )
        if self.length is None and (
            self.step is not None or self.count is not None or self.allow_shorter
        ):
            raise ValueError(
                _format_error(
                    "ReactivationWindows: step=, count= and allow_shorter= need "
                    "length=.",
                    why="Why: without a length the period is one window.",
                    fix="pass length= in seconds, or drop step=, count= and "
                    "allow_shorter=",
                )
            )
        if self.anchor not in ("start", "end"):
            raise ValueError(
                f"ReactivationWindows: anchor must be 'start' or 'end', got "
                f"{self.anchor!r}."
            )
        if self.clock not in ("concatenated", "wall"):
            raise ValueError(
                "ReactivationWindows: clock must be 'concatenated' or 'wall', "
                f"got {self.clock!r}."
            )
        if self.count is not None and not _is_count(self.count):
            raise ValueError(
                f"ReactivationWindows: count must be a positive integer or None, "
                f"got {self.count!r}."
            )

    def split(self, epochs: Any) -> list[NDArray[np.float64]]:
        """Cut epochs into this rule's windows.

        Parameters
        ----------
        epochs : array-like, shape (n_epochs, 2), or IntervalSet
            The period's ``[start, stop)`` epochs in seconds; overlapping or
            touching epochs are merged.

        Returns
        -------
        list of ndarray, each shape (n_pieces, 2)
            The windows in time order, each the parts of the epochs it
            covers. A ``"wall"`` window that covers no epoch time is left
            out. When the epochs are shorter than one window: empty, or all
            of them with ``allow_shorter=True``.
        """
        rows = as_intervals(epochs, name="epochs")
        assert rows is not None
        if self.length is None:
            return [rows]
        starts, stops = rows[:, 0], rows[:, 1]
        if self.clock == "concatenated":
            edges = np.concatenate([[0.0], np.cumsum(stops - starts)])
            total = float(edges[-1])
        else:
            total = float(stops[-1] - starts[0])
        step = self.length if self.step is None else self.step
        # Whole windows only; a window ending within rounding of the period's
        # end is whole.
        tolerance = 1e-9 * max(1.0, total)
        n_windows = int(np.floor((total - self.length + tolerance) / step)) + 1
        if total + tolerance < self.length:
            return [rows] if self.allow_shorter else []
        if self.count is not None:
            n_windows = min(n_windows, self.count)
        offsets = np.arange(n_windows) * step
        lows = offsets if self.anchor == "start" else total - self.length - offsets
        windows = []
        for low in np.sort(lows):
            high = min(low + self.length, total)
            if self.clock == "concatenated":
                pieces = _concatenated_pieces(starts, edges, max(low, 0.0), high)
            else:
                pieces = _clipped(rows, starts[0] + max(low, 0.0), starts[0] + high)
            if len(pieces):
                windows.append(pieces)
        return windows


def _concatenated_pieces(
    starts: NDArray[np.float64], edges: NDArray[np.float64], low: float, high: float
) -> NDArray[np.float64]:
    """Epoch pieces covering concatenated time ``[low, high)``.

    Parameters
    ----------
    starts : ndarray, shape (n_epochs,)
        Epoch starts on the clock.
    edges : ndarray, shape (n_epochs + 1,)
        Cumulative epoch durations: epoch ``i`` spans ``edges[i]`` to
        ``edges[i + 1]`` of concatenated time.
    low, high : float
        The window on concatenated time.

    Returns
    -------
    ndarray, shape (n_pieces, 2)
        The window's pieces on the clock.
    """
    begin = np.maximum(edges[:-1], low)
    end = np.minimum(edges[1:], high)
    keep = end > begin
    offsets = starts - edges[:-1]
    return np.column_stack([begin[keep], end[keep]]) + offsets[keep, None]


def _clipped(rows: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
    """The parts of ``rows`` inside ``[low, high)``, shape (n_pieces, 2)."""
    begin = np.maximum(rows[:, 0], low)
    end = np.minimum(rows[:, 1], high)
    keep = end > begin
    return np.column_stack([begin[keep], end[keep]])


@dataclass(frozen=True)
class EventResponse:
    """How an assembly method measures R(t) around events, such as SWR onsets.

    Attributes
    ----------
    window : tuple of float
        ``(start, stop)`` of the peri-event lags, seconds relative to each
        event onset, half-open. Default ``(-0.5, 0.5)``, the ±0.5 s of the
        labs' peri-event histograms.
    response_window : tuple of float
        Lags averaged into each event's response. Default ``(0.0, 0.1)``:
        "the mean reactivation strength within 0–100 ms of SWR onset"
        (Robinson et al. 2026).
    reduce : {"mean", "peak"}
        A period's response: the mean of its events' responses, or the peak
        of its peri-event mean over the response lags (Harvey et al. 2023).
    baseline_window : tuple of float or None
        Lags of each event's baseline for ``test="event_modulation"``
        (Chang et al. 2025: "baseline (−500 to −100 ms relative to SWR
        onset)").
    significant_z : float or None
        Report the share of each period's events whose response, z-scored
        over the events of both periods, exceeds this value (Robinson et al.
        2026: "responses were z-scored within each assembly ... responses
        exceeding z = 1.96 (P < 0.05) were considered significant").

    See Also
    --------
    neurospatial.reactivation.assembly_event_response : The same measure on
        one R(t) trace.
    """

    window: tuple[float, float] = (-0.5, 0.5)
    response_window: tuple[float, float] = (0.0, 0.1)
    reduce: Literal["mean", "peak"] = "mean"
    baseline_window: tuple[float, float] | None = None
    significant_z: float | None = None

    def __post_init__(self) -> None:
        """Check the windows."""
        windows = {
            "window": self.window,
            "response_window": self.response_window,
            "baseline_window": self.baseline_window,
        }
        for name, value in windows.items():
            if value is None:
                continue
            pair = np.asarray(value, dtype=np.float64)
            if (
                pair.shape != (2,)
                or not np.all(np.isfinite(pair))
                or pair[0] >= pair[1]
            ):
                raise ValueError(
                    f"EventResponse: {name} must be (start, stop) in seconds with "
                    f"start < stop, got {value!r}."
                )
            if name != "window" and (
                pair[0] < self.window[0] or pair[1] > self.window[1]
            ):
                raise ValueError(
                    f"EventResponse: {name}={value} must lie inside "
                    f"window={self.window}."
                )
        if self.reduce not in ("mean", "peak"):
            raise ValueError(
                f"EventResponse: reduce must be 'mean' or 'peak', got {self.reduce!r}."
            )
        if self.significant_z is not None and not np.isfinite(self.significant_z):
            raise ValueError(
                "EventResponse: significant_z must be a finite z-score or None, "
                f"got {self.significant_z!r}."
            )


@dataclass(frozen=True)
class ReactivationMethod:
    """A published reactivation analysis: bins, pairs, windows and test.

    Attributes
    ----------
    name : str
        Short name, for example ``"kudrimoti_1999"``.
    reference : str
        The paper the method follows.
    measure : {"explained_variance", "assembly", "member_cofiring"}
        ``"explained_variance"`` compares pair correlations between periods
        (:func:`~neurospatial.reactivation.explained_variance_reactivation`);
        ``"assembly"`` detects patterns in the template period
        (:func:`~neurospatial.reactivation.detect_assemblies`) and compares
        their activation strength R(t) between the control and match
        periods; ``"member_cofiring"`` detects patterns and compares the
        correlation of each pattern's members between the periods
        (:func:`~neurospatial.reactivation.assembly_member_cofiring`).
    bin_size : float or None
        Time bin in seconds. Applied: spike times are binned at this width
        inside each window. None when every period is counted in events.
    pairs : {"all", "different_tetrodes", "different_shanks", "different_regions"}
        Which pairs of cells an EV method uses. A rule other than ``"all"``
        keeps the pairs whose units are in different groups of
        ``unit_groups["tetrode"]``, ``["shank"]`` or ``["region"]`` given to
        :func:`compare_reactivation_methods`.
    control_windows, match_windows : ReactivationWindows
        How the control and match epochs are cut into windows. Each match
        window gives a row of the comparison table. An EV method with
        several control windows averages the single-control EVs and REVs
        over them (Tatsuno et al. 2006); an assembly method needs one
        control window.
    template_bins, rest_bins : {"fixed", "events"}
        ``"fixed"`` (default): counts in ``bin_size`` bins. ``"events"``:
        one bin per event, the spikes from its start to its stop, for the
        template (``template_events``, for example theta cycles) or for the
        control and match periods (``control_events`` and ``match_events``,
        for example SWRs), keeping the events whose start lies in the
        period's window. Events shorter than ``bin_size`` are kept. Spike
        counts of events of different lengths share a duration component
        that raises every pair correlation alike; it cancels between
        periods only when their events have similar durations. ``rest_bins``
        does not apply to ``"assembly"`` methods, which measure R(t) around
        events with ``event_response``.
    min_events : int or None
        A control or match window with fewer events is refused, and the
        method's row reports why in ``error`` (El-Gaby et al. 2021: "only
        sessions where at least 200 SWR events were detected were
        included"). A window without events is refused even when None.
    detection : {"pca", "ica"} or None
        Pattern detection on the template counts, with the Marchenko-Pastur
        threshold; None for EV methods.
    member_rule : {"abs_z", "positive_z", "otsu", "inverse_sqrt_n"}
        How the patterns' members are chosen (``detect_assemblies(member_rule=...)``).
    member_threshold : float or None
        Z-score cutoff of the ``"abs_z"`` and ``"positive_z"`` rules; None
        for the other rules, which set their own.
    detection_units : {"all", "each_region", "each_region_and_cross_region"}
        Units of each detection: all units together; each region's units on
        their own (``unit_groups["region"]``); or each region and also all
        units together, keeping the patterns of the latter whose members
        lie in at least two regions (Robinson et al. 2026's cross-structural
        assemblies). The table's ``region`` column says which. A region
        that cannot be detected in (for example one with fewer than 3
        units) is left out with a warning; the method's row reports an
        error only when every region fails.
    smoothing_bandwidth : float or None
        For assembly methods, the standard deviation in seconds of a
        Gaussian kernel applied to the spike trains before z-scoring and
        computing R(t) in the control and match periods (van de Ven et al.
        2016); None uses counts in ``bin_size`` bins.
    activation_step : float or None
        For assembly methods, seconds between samples of R(t). Without
        smoothing, counts in ``bin_size`` bins that slide by this step (it
        must divide ``bin_size``); with smoothing, the spacing of the
        smoothed samples. None: ``bin_size`` without smoothing,
        ``smoothing_bandwidth / 2`` with it.
    zscore_over : {"period", "control_and_match", "all_periods"}
        For assembly methods, the samples each unit is z-scored over before
        R(t): each period on its own (default), the control and match
        periods together, or the template, control and match periods
        together. Neither removes a rate change: if the members fire more
        in the match period without firing together more, their R(t) in
        its events rises when the periods are z-scored together and falls
        when each is z-scored on its own (its larger variance shrinks the
        z-scores).
    activation_threshold : float or None
        For assembly methods, R(t) peaks above this value are activations;
        their rate in each period is reported. None reports none.
    event_response : EventResponse or None
        For assembly methods, measure R(t) around the onsets of
        ``control_events`` and ``match_events``.
    statistic : str
        The method's headline value, the ``value`` column of the table:

        - EV methods: ``"explained_variance"``, ``"ev_minus_rev"``,
          ``"ev_rev_contrast"`` ((EV - REV) / (EV + REV)),
          ``"partial_correlation"`` (the correlation of the residuals of the
          template and match pair correlations regressed on the control's)
          or ``"semipartial_correlation"`` (the template correlations
          against the residual of the match correlations regressed on the
          control's). The last two need one control window.
        - Assembly methods: ``"mean_activation_difference"``,
          ``"event_response_difference"`` (match response minus control
          response) or ``"event_response_ratio"`` ((match - control) /
          control; NaN unless the control response is positive).
        - Member co-firing: ``"mean_member_correlation_difference"``.

    test : {"partial_correlation_t", "pair_bootstrap", "cell_identity_shuffle", "event_modulation"} or None
        The within-session test:

        - ``"partial_correlation_t"``: the t test of the partial
          correlation (one control window), reactivated when ``p < alpha``
          and the correlation is positive.
        - ``"pair_bootstrap"``: ``n_shuffles`` resamples of the pairs with
          replacement, keeping each pair's three correlations together;
          reactivated when the ``100 * alpha / 2`` percentile of EV - REV is
          above 0 (Lansink et al. 2009). One control window.
        - ``"cell_identity_shuffle"``: in every time bin, the cells'
          identities are permuted ``n_shuffles`` times, and each bin's R(t)
          is compared with the ``percentile``-th percentile of its shuffles
          (Peyrache et al. 2010). Reports the share of control and match
          bins above it; no verdict.
        - ``"event_modulation"``: the two-sided Wilcoxon signed-rank test,
          over the match events, of each event's response against its
          baseline (``event_response.baseline_window``); reactivated when
          ``p < alpha`` and the median difference is positive (Chang et al.
          2025).
        - None: no within-session test; ``is_reactivated`` is missing.

    percentile : float or None
        For assembly methods, the percentile of the control period's R(t)
        that match bins are compared with, and of the cell-identity
        shuffles; None reports neither share.
    n_shuffles : int
        Resamples or shuffles of the test.
    alpha : float
        Significance level of the test.
    n_rate_subsamples : int or None
        For EV methods, the rate control of Giri et al. (2019): for each
        pair of control and match windows, each unit's spikes are subsampled
        so it fires at the same rate in both
        (:func:`~neurospatial.reactivation.equalize_firing_rates`), this
        many times ("median Pearson's r … median p … for 100 different
        random subsamples"). The row reports the subsample whose
        ``statistic`` is the median (the lower of the middle two for an
        even number): ``value`` is the median of the statistic, and EV,
        REV, the partial correlation, its p-value and the verdict all come
        from that one subsample, so they agree with each other. None
        (default) uses every spike. Needs ``rest_bins="fixed"`` and is not
        combined with ``test="pair_bootstrap"``.
    notes : tuple of str
        Each value's source in the paper and the choices it leaves open.

    Examples
    --------
    >>> import dataclasses
    >>> from neurospatial.reactivation import EL_GABY_2021, KUDRIMOTI_1999
    >>> KUDRIMOTI_1999.bin_size, KUDRIMOTI_1999.pairs
    (0.1, 'different_tetrodes')
    >>> wider = dataclasses.replace(
    ...     KUDRIMOTI_1999, name="kudrimoti_200ms", bin_size=0.2
    ... )
    >>> EL_GABY_2021.rest_bins, EL_GABY_2021.min_events
    ('events', 200)
    >>> EL_GABY_2021.needs_events
    ('control_events', 'match_events')
    """

    name: str
    reference: str
    measure: Literal["explained_variance", "assembly", "member_cofiring"]
    bin_size: float | None
    pairs: Literal[
        "all", "different_tetrodes", "different_shanks", "different_regions"
    ] = "all"
    control_windows: ReactivationWindows = ReactivationWindows()
    match_windows: ReactivationWindows = ReactivationWindows()
    template_bins: Literal["fixed", "events"] = "fixed"
    rest_bins: Literal["fixed", "events"] = "fixed"
    min_events: int | None = None
    detection: Literal["pca", "ica"] | None = None
    member_rule: Literal["abs_z", "positive_z", "otsu", "inverse_sqrt_n"] = "abs_z"
    member_threshold: float | None = 2.0
    detection_units: Literal["all", "each_region", "each_region_and_cross_region"] = (
        "all"
    )
    smoothing_bandwidth: float | None = None
    activation_step: float | None = None
    zscore_over: Literal["period", "control_and_match", "all_periods"] = "period"
    activation_threshold: float | None = None
    event_response: EventResponse | None = None
    statistic: Literal[
        "explained_variance",
        "ev_minus_rev",
        "ev_rev_contrast",
        "partial_correlation",
        "semipartial_correlation",
        "mean_activation_difference",
        "event_response_difference",
        "event_response_ratio",
        "mean_member_correlation_difference",
    ] = "explained_variance"
    test: (
        Literal[
            "partial_correlation_t",
            "pair_bootstrap",
            "cell_identity_shuffle",
            "event_modulation",
        ]
        | None
    ) = None
    percentile: float | None = None
    n_shuffles: int = 1000
    alpha: float = 0.05
    n_rate_subsamples: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def needs_events(self) -> tuple[str, ...]:
        """Event arguments of :func:`compare_reactivation_methods` the method needs."""
        needed = []
        if self.template_bins == "events":
            needed.append("template_events")
        if self.rest_bins == "events" or self.event_response is not None:
            needed.extend(["control_events", "match_events"])
        return tuple(needed)

    def __post_init__(self) -> None:
        """Check that the options fit the measure."""
        where = f"ReactivationMethod {self.name!r}"
        if self.measure not in _MEASURES:
            raise ValueError(
                f"{where}: measure must be one of {_MEASURES}, got {self.measure!r}."
            )
        for name, choices in (
            ("template_bins", _BINS),
            ("rest_bins", _BINS),
            ("member_rule", _MEMBER_RULES),
            ("detection_units", _DETECTION_UNITS),
            ("zscore_over", _ZSCORE_OVER),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(
                    f"{where}: {name} must be one of {choices}, got "
                    f"{getattr(self, name)!r}."
                )
        assembly = self.measure == "assembly"
        if assembly and self.rest_bins == "events":
            raise ValueError(
                _format_error(
                    f"{where}: rest_bins='events' does not apply to an assembly "
                    "method.",
                    why="Why: an assembly method follows R(t) over time.",
                    fix="pass event_response=EventResponse() to measure R(t) "
                    "around the events",
                )
            )
        fixed_bins = assembly or "fixed" in (self.template_bins, self.rest_bins)
        if fixed_bins and not (
            self.bin_size is not None
            and np.isfinite(self.bin_size)
            and self.bin_size > 0
        ):
            raise ValueError(
                f"{where}: bin_size must be a positive number of seconds, got "
                f"{self.bin_size!r}."
            )
        if not fixed_bins and self.bin_size is not None:
            raise ValueError(
                f"{where}: every period is counted in events, so bin_size must "
                f"be None, got {self.bin_size!r}."
            )
        if self.pairs not in _PAIRS:
            raise ValueError(
                f"{where}: pairs must be one of {_PAIRS}, got {self.pairs!r}."
            )
        if self.statistic not in _STATISTICS[self.measure]:
            raise ValueError(
                f"{where}: statistic {self.statistic!r} does not fit measure "
                f"{self.measure!r}; use one of {_STATISTICS[self.measure]}."
            )
        if self.test not in _TESTS[self.measure]:
            raise ValueError(
                f"{where}: test {self.test!r} does not fit measure "
                f"{self.measure!r}; use one of {_TESTS[self.measure]}."
            )
        patterns = self.measure in ("assembly", "member_cofiring")
        if patterns and self.detection not in _DETECTIONS:
            raise ValueError(
                f"{where}: a {self.measure} method needs detection='pca' or "
                f"'ica', got {self.detection!r}."
            )
        assembly_only = (
            "smoothing_bandwidth",
            "activation_step",
            "activation_threshold",
            "percentile",
            "event_response",
        )
        given = [name for name in assembly_only if getattr(self, name) is not None]
        if not assembly and self.zscore_over != "period":
            given.append("zscore_over")
        if not patterns:
            given += [name for name in ("detection",) if self.detection is not None]
            if self.member_rule != "abs_z" or self.detection_units != "all":
                given += ["member_rule", "detection_units"]
        if not assembly and given:
            raise ValueError(
                _format_error(
                    f"{where}: {given} do not apply to {self.measure} methods.",
                    why=f"Why: a {self.measure} method would ignore them.",
                    fix="drop them, or use measure='assembly'",
                )
            )
        if self.measure != "explained_variance" and self.pairs != "all":
            raise ValueError(
                f"{where}: pairs={self.pairs!r} applies only to EV methods; "
                f"a {self.measure} method uses every unit."
            )
        z_rule = self.member_rule in _Z_MEMBER_RULES
        if z_rule and not (
            self.member_threshold is not None and np.isfinite(self.member_threshold)
        ):
            raise ValueError(
                f"{where}: member_rule={self.member_rule!r} needs a finite "
                f"member_threshold, got {self.member_threshold!r}."
            )
        if not z_rule and self.member_threshold is not None:
            raise ValueError(
                f"{where}: member_rule={self.member_rule!r} sets its own cutoff; "
                "pass member_threshold=None."
            )
        if self.min_events is not None and (
            not _is_count(self.min_events) or not self.needs_events
        ):
            raise ValueError(
                f"{where}: min_events must be a positive integer for a method "
                f"that counts or follows events, got {self.min_events!r}."
            )
        for name in ("smoothing_bandwidth", "activation_step"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"{where}: {name} must be positive seconds or None, got {value!r}."
                )
        if self.activation_step is not None and self.smoothing_bandwidth is None:
            assert self.bin_size is not None
            ratio = self.bin_size / self.activation_step
            if not (ratio >= 1 - 1e-9 and abs(ratio - round(ratio)) < 1e-6):
                raise ValueError(
                    f"{where}: without smoothing, activation_step "
                    f"({self.activation_step:g} s) must divide bin_size "
                    f"({self.bin_size:g} s) into a whole number of steps."
                )
        if self.statistic in _EVENT_RESPONSE_STATISTICS and self.event_response is None:
            raise ValueError(
                f"{where}: statistic {self.statistic!r} needs "
                "event_response=EventResponse(...)."
            )
        if self.event_response is not None:
            step = _sample_step(self)
            lags = _lags(self.event_response.window, step)
            for name in ("response_window", "baseline_window"):
                part = getattr(self.event_response, name)
                if part is not None and not _in_lag_window(lags, part, step).any():
                    raise ValueError(
                        _format_error(
                            f"{where}: event_response {name}={part} holds no lag "
                            f"of R(t), which is sampled every {step:g} s.",
                            why="Why: its mean over the events would be "
                            "undefined (NaN).",
                            fix=f"widen {name}, or sample R(t) more finely with "
                            "activation_step=",
                        )
                    )
        if self.test == "event_modulation" and (
            self.event_response is None or self.event_response.baseline_window is None
        ):
            raise ValueError(
                f"{where}: test='event_modulation' compares each event's "
                "response with its baseline; pass "
                "event_response=EventResponse(baseline_window=(start, stop))."
            )
        if self.percentile is not None and not 0 < self.percentile < 100:
            raise ValueError(
                f"{where}: percentile must be in (0, 100), got {self.percentile!r}."
            )
        if self.test == "cell_identity_shuffle" and self.percentile is None:
            raise ValueError(
                f"{where}: the cell-identity shuffle compares each bin with a "
                "percentile of its shuffles; pass percentile=."
            )
        if not _is_count(self.n_shuffles):
            raise ValueError(
                f"{where}: n_shuffles must be a positive integer, got "
                f"{self.n_shuffles!r}."
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"{where}: alpha must be in (0, 1), got {self.alpha!r}.")
        subsamples = self.n_rate_subsamples
        if subsamples is not None:
            if not _is_count(subsamples):
                raise ValueError(
                    f"{where}: n_rate_subsamples must be a positive integer or "
                    f"None, got {subsamples!r}."
                )
            if self.measure != "explained_variance":
                raise ValueError(
                    _format_error(
                        f"{where}: n_rate_subsamples applies only to EV methods.",
                        why="Why: Giri et al. (2019) equalized rates for the "
                        "pair correlations of fixed time bins; activation "
                        "strength and member co-firing z-score each period on "
                        "its own.",
                        fix="drop n_rate_subsamples, or use measure="
                        "'explained_variance'",
                    )
                )
            if self.rest_bins == "events":
                raise ValueError(
                    _format_error(
                        f"{where}: n_rate_subsamples needs rest_bins='fixed'.",
                        why="Why: it equalizes each unit's spikes per bin; "
                        "events of different durations make spikes per event "
                        "and spikes per second disagree, and Giri et al. "
                        "(2019) equalized fixed 250 ms bins.",
                        fix="drop n_rate_subsamples, or bin rest periods with "
                        "rest_bins='fixed' and bin_size=",
                    )
                )
            if self.test == "pair_bootstrap":
                raise ValueError(
                    _format_error(
                        f"{where}: n_rate_subsamples cannot be combined with "
                        "the pair bootstrap.",
                        why="Why: the bootstrap resamples one set of pair "
                        "correlations, and each rate subsample has its own.",
                        fix="use test='partial_correlation_t' or None with "
                        "n_rate_subsamples",
                    )
                )


# =============================================================================
# Comparison
# =============================================================================

_COLUMNS = (
    "method",
    "window",
    "window_start",
    "window_stop",
    "window_duration",
    "component",
    "region",
    "statistic",
    "value",
    "explained_variance",
    "reversed_explained_variance",
    "partial_correlation",
    "semipartial_correlation",
    "ev_minus_rev_lower",
    "n_pairs",
    "n_control_windows",
    "n_control_events",
    "n_match_events",
    "n_components",
    "member_unit_ids",
    "control_mean_activation",
    "match_mean_activation",
    "fraction_above_control_threshold",
    "control_fraction_above_shuffle",
    "match_fraction_above_shuffle",
    "control_activation_rate",
    "match_activation_rate",
    "control_event_response",
    "match_event_response",
    "control_significant_event_fraction",
    "match_significant_event_fraction",
    "control_member_correlation",
    "match_member_correlation",
    "n_member_pairs",
    "p_value",
    "n_shuffles_used",
    "is_reactivated",
    "error",
)
_COUNT_COLUMNS = (
    "window",
    "component",
    "n_pairs",
    "n_control_windows",
    "n_control_events",
    "n_match_events",
    "n_components",
    "n_member_pairs",
    "n_shuffles_used",
)
_TEXT_COLUMNS = (
    "method",
    "region",
    "statistic",
    "member_unit_ids",
    "is_reactivated",
    "error",
)


def compare_reactivation_methods(
    spike_times: Any,
    template_epochs: Any,
    match_epochs: Any,
    control_epochs: Any = None,
    *,
    reactivation_methods: Sequence[ReactivationMethod],
    template_events: Any = None,
    control_events: Any = None,
    match_events: Any = None,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    unit_groups: Mapping[str, NDArray[Any] | Sequence[Any]] | None = None,
    pair_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Apply several published reactivation methods to one session.

    Parameters
    ----------
    spike_times : sequence of arrays or spike group
        One spike train per unit, in seconds. A spike group (a dict of unit
        ID to train, :class:`~neurospatial.encoding.SpikeTrains` or a
        pynapple ``TsGroup``) names its units with its own IDs.
    template_epochs : array-like, shape (n, 2), or IntervalSet
        Epochs of the template period, for example running on the track
        (or its theta periods).
    match_epochs : array-like, shape (n, 2), or IntervalSet
        Epochs of the match period, for example slow-wave sleep after the
        task. Cut into windows by each method's ``match_windows``.
    control_epochs : array-like, shape (n, 2), or IntervalSet, optional
        Epochs of the control period, for example slow-wave sleep before
        the task. Every published method needs one.
    reactivation_methods : sequence of ReactivationMethod
        The methods to compare, with distinct names.
    template_events, control_events, match_events : array-like, shape (n, 2), or IntervalSet, optional
        Start and stop of each event, seconds, for the methods whose
        ``needs_events`` names them: for example theta cycles of the task
        and SWRs of rest before and after it. Events may touch but not
        overlap, since a spike would then count in two events. A period
        uses the events whose start lies in its window; a method counting
        spikes in events counts them from each event's start to its stop,
        and one following R(t) around events aligns it to their starts.
    unit_ids : array-like, shape (n_units,), optional
        Label of each unit, for ``member_unit_ids``; default the IDs a
        spike group carries, else ``np.arange(n_units)``. With a spike
        group it must equal the group's IDs.
    unit_groups : mapping of str to array-like, optional
        Each unit's group, by kind: ``"tetrode"``, ``"shank"`` or
        ``"region"``, one entry per unit, as methods with a ``pairs`` rule
        or a ``detection_units`` by region need (for example
        ``{"tetrode": tetrode_of_unit}``).
    pair_mask : array-like of bool, shape (n_units * (n_units - 1) / 2,), optional
        Pairs every EV method may use, in ``np.triu_indices(n_units, k=1)``
        order, for example waveform similarity below 0.8 (Giri et al. 2024).
        Combined with each method's ``pairs`` rule.
    rng : numpy.random.Generator, int, or None
        Seed or generator for ICA, bootstraps, shuffles and rate
        subsamples. Each method draws from its own streams, derived from
        ``rng`` and the method's name: one for pattern detection and one
        per match window for its bootstrap, shuffles or subsamples, so
        adding, removing or reordering methods does not change the other
        methods' results. A method's components share its streams: they
        are detected together, and each bin's cell-identity shuffles are
        shared by the components.

    Returns
    -------
    pandas.DataFrame
        One row per (method, match window, component): EV methods have one
        row per match window with ``component`` missing; assembly and
        member co-firing methods one per detected component (one row with
        ``n_components`` 0 and ``component`` missing when none is detected).
        Columns:

        - ``method``, ``window`` (match window number), ``window_start``,
          ``window_stop`` (the window's first and last epoch time) and
          ``window_duration`` (seconds of epoch time inside it).
        - ``component``, ``n_components``, ``member_unit_ids`` (the
          pattern's core members) and ``region`` (the region whose units
          the pattern was detected in, ``"cross_region"`` for a pattern of
          all units with members in several regions, missing when all units
          were used together) for pattern methods.
        - ``statistic`` (its name) and ``value``.
        - ``n_control_events`` and ``n_match_events``: events in the
          control window (the fewest over windows) and the match window,
          for methods that use events.
        - EV methods: ``explained_variance``, ``reversed_explained_variance``
          (averaged over control windows), ``partial_correlation`` and
          ``semipartial_correlation`` (one control window), ``n_pairs``
          (fewest over control windows), ``n_control_windows`` and
          ``ev_minus_rev_lower`` (the bootstrap's ``100 * alpha / 2``
          percentile).
        - Assembly methods: ``control_mean_activation`` and
          ``match_mean_activation`` (mean R(t)),
          ``fraction_above_control_threshold``,
          ``control_fraction_above_shuffle``,
          ``match_fraction_above_shuffle``, ``control_activation_rate`` and
          ``match_activation_rate`` (activations per second); with
          ``event_response``, ``control_event_response`` and
          ``match_event_response`` and, with ``significant_z``,
          ``control_significant_event_fraction`` and
          ``match_significant_event_fraction``.
        - Member co-firing: ``control_member_correlation`` and
          ``match_member_correlation`` (mean over member pairs defined in
          both periods) and ``n_member_pairs``.
        - ``p_value``, ``n_shuffles_used`` and ``is_reactivated`` (the
          method's within-session verdict; missing for methods without
          one), and ``error``: why a method could not be applied, with
          the other columns missing. A failure of the template or control
          period (for example a period shorter than one window) is one row
          of the method; a match window that fails (for example one
          without events or with fewer than ``min_events``) is a row of
          that window, with its window columns, and the method's other
          windows keep their rows.

        Columns that do not apply to a method are NaN. ``table.attrs``
        holds ``rng_entropy``, the seed the streams derive from: passing it
        as ``rng`` reproduces the table.

    Raises
    ------
    ValueError
        If the methods' names repeat, a method needs ``control_epochs``,
        events or a ``unit_groups`` kind that is not given, events overlap,
        a group or ``pair_mask`` does not have one entry per unit or pair,
        ``unit_ids`` does not hold one distinct label per unit or differs
        from the IDs of a spike group, or a spike time is NaN or infinite.

    Warns
    -----
    UserWarning
        Warnings of the underlying functions (constant units, undefined EV,
        no pattern above the threshold), prefixed with the method's name.

    Notes
    -----
    Each method bins spike times at its own ``bin_size`` inside each of its
    windows (:func:`~neurospatial.decoding.bin_spikes_in_time`), so bins
    never span two epochs, or counts spikes in each event. The template
    period is always one window of all ``template_epochs``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.reactivation import (
    ...     KUDRIMOTI_1999,
    ...     compare_reactivation_methods,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> spike_times = [np.sort(rng.uniform(0, 3600, 3600)) for _ in range(12)]
    >>> table = compare_reactivation_methods(
    ...     spike_times,
    ...     template_epochs=[[1200.0, 2400.0]],
    ...     match_epochs=[[2400.0, 3600.0]],
    ...     control_epochs=[[0.0, 1200.0]],
    ...     reactivation_methods=[KUDRIMOTI_1999],
    ...     unit_groups={"tetrode": np.repeat(np.arange(4), 3)},
    ...     rng=0,
    ... )
    >>> table[["method", "window", "n_pairs"]].to_dict("records")
    [{'method': 'kudrimoti_1999', 'window': 0, 'n_pairs': 54}, {'method': 'kudrimoti_1999', 'window': 1, 'n_pairs': 54}]

    A method with events, one bin per SWR of each rest period (here
    random 50-150 ms intervals):

    >>> from neurospatial.reactivation import LIU_2023_EV
    >>> def events(start, stop):
    ...     onsets = np.sort(rng.uniform(start, stop - 1, 300))
    ...     onsets = onsets[np.r_[True, np.diff(onsets) > 0.2]]
    ...     return np.column_stack(
    ...         [onsets, onsets + rng.uniform(0.05, 0.15, onsets.size)]
    ...     )
    >>> table = compare_reactivation_methods(
    ...     spike_times,
    ...     template_epochs=[[1200.0, 2400.0]],
    ...     match_epochs=[[2400.0, 3600.0]],
    ...     control_epochs=[[0.0, 1200.0]],
    ...     reactivation_methods=[LIU_2023_EV],
    ...     template_events=np.column_stack(
    ...         [np.arange(1200.0, 2400.0, 0.125), np.arange(1200.125, 2400.1, 0.125)]
    ...     ),
    ...     control_events=events(0.0, 1200.0),
    ...     match_events=events(2400.0, 3600.0),
    ... )
    >>> int(table.loc[0, "n_pairs"])
    66
    """
    import pandas as pd

    methods = list(reactivation_methods)
    names = [method.name for method in methods]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            _format_error(
                f"{_CALL}: method names {repeated} repeat.",
                why="Why: each method's random stream and rows are keyed by its name.",
                fix="give each method its own name, e.g. "
                "dataclasses.replace(method, name='...')",
            )
        )
    trains, labels, _ = as_labelled_spike_trains(spike_times, unit_ids, context=_CALL)
    n_units = len(trains)
    template = as_intervals(template_epochs, name="template_epochs")
    match = as_intervals(match_epochs, name="match_epochs")
    control = as_intervals(control_epochs, name="control_epochs")
    assert template is not None and match is not None
    if control is None and methods:
        raise ValueError(
            _format_error(
                f"{_CALL}: methods {names} need control_epochs.",
                why="Why: EV partials out the control period's correlations, "
                "and assembly methods compare the match period with it.",
                fix="pass control_epochs=, for example slow-wave sleep before the task",
            )
        )
    events = {
        "template_events": _checked_events(template_events, "template_events"),
        "control_events": _checked_events(control_events, "control_events"),
        "match_events": _checked_events(match_events, "match_events"),
    }
    missing = {
        name: [method.name for method in methods if name in method.needs_events]
        for name, value in events.items()
        if value is None
    }
    missing = {name: users for name, users in missing.items() if users}
    if missing:
        raise ValueError(
            _format_error(
                f"{_CALL}: {sorted(missing)} not given, but methods "
                f"{sorted({user for users in missing.values() for user in users})} "
                "use them.",
                why="Why: these methods count spikes in events (one bin per "
                "event) or follow R(t) around events.",
                fix="pass the events' (start, stop) times, for example "
                "control_events=pre_sleep_swrs, match_events=post_sleep_swrs "
                "and template_events=theta_cycles",
            )
        )
    groups = _checked_groups(unit_groups, methods, n_units)
    n_pairs = n_units * (n_units - 1) // 2
    selected = np.ones(n_pairs, dtype=bool)
    if pair_mask is not None:
        mask = np.asarray(pair_mask)
        if mask.dtype != np.bool_ or mask.shape != (n_pairs,):
            raise ValueError(
                _format_error(
                    f"{_CALL}: pair_mask must be a boolean array with one entry "
                    f"per pair ({n_pairs} for {n_units} units), got dtype "
                    f"{mask.dtype} and shape {mask.shape}.",
                    fix="pass a boolean array in np.triu_indices(n_units, k=1) order",
                )
            )
        selected = mask.copy()

    entropy = _entropy(rng)

    rows: list[dict[str, Any]] = []
    if methods:
        assert control is not None  # every method needs one, checked above
        session = _Session(trains, template, control, match, events, labels, groups)
    for method in methods:
        key = _name_key(method.name)

        def stream(*indices: int, key: int = key) -> np.random.Generator:
            return _stream(entropy, key, *indices)

        with _warnings_with_context(f"{_CALL}: method {method.name!r}"):
            try:
                if method.measure == "explained_variance":
                    pair_rule = selected.copy()
                    if method.pairs != "all":
                        unit_group = groups[_GROUP_OF_PAIRS[method.pairs]]
                        first, second = np.triu_indices(n_units, k=1)
                        pair_rule &= unit_group[first] != unit_group[second]
                    method_rows = _explained_variance_rows(
                        method, session, pair_rule, stream
                    )
                elif method.measure == "assembly":
                    method_rows = _assembly_rows(method, session, stream)
                else:
                    method_rows = _member_cofiring_rows(method, session, stream)
            except _MethodError as failure:
                method_rows = [{"error": str(failure)}]
        rows.extend({"method": method.name, **row} for row in method_rows)
    table = pd.DataFrame(rows, columns=list(_COLUMNS))
    numeric = [
        name
        for name in _COLUMNS
        if name not in _COUNT_COLUMNS and name not in _TEXT_COLUMNS
    ]
    table[numeric] = table[numeric].astype(np.float64)
    for name in _COUNT_COLUMNS:
        table[name] = table[name].astype("Int64")
    table["is_reactivated"] = table["is_reactivated"].astype("boolean")
    table["error"] = table["error"].fillna("").astype(str)
    table.attrs["rng_entropy"] = entropy
    return table


@dataclass(frozen=True)
class _Session:
    """One session's inputs, checked."""

    trains: list[NDArray[np.float64]]
    template: NDArray[np.float64]
    control: NDArray[np.float64]
    match: NDArray[np.float64]
    events: dict[str, NDArray[np.float64] | None]
    labels: NDArray[Any]
    groups: dict[str, NDArray[Any]]


def _checked_events(value: Any, name: str) -> NDArray[np.float64] | None:
    """Events as ``(n, 2)`` rows sorted by start; overlapping rows refused.

    Unlike epochs, touching or overlapping events are not merged: each event
    is one bin, so merging two would change the number of bins.
    """
    if value is None:
        return None
    if hasattr(value, "start") and hasattr(value, "end"):
        value = np.column_stack([np.asarray(value.start), np.asarray(value.end)])
    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError(
            f"{_CALL}: {name} must be an (n_events, 2) array of start and stop "
            f"times, got shape {rows.shape}."
        )
    bad = ~np.all(np.isfinite(rows), axis=1) | (rows[:, 1] <= rows[:, 0])
    if bad.any():
        raise ValueError(
            f"{_CALL}: {name} rows {np.flatnonzero(bad)[:10].tolist()} are not "
            "finite with stop > start."
        )
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    overlapping = np.flatnonzero(rows[1:, 0] < rows[:-1, 1])
    if overlapping.size:
        first = rows[overlapping[0] : overlapping[0] + 2]
        raise ValueError(
            _format_error(
                f"{_CALL}: {overlapping.size} events of {name} overlap the next "
                f"one, e.g. {first.tolist()}.",
                why="Why: each event is one bin of spike counts, and a spike in "
                "two overlapping events would count twice.",
                fix="merge overlapping events into one before the call, or keep "
                "the first of each overlapping group",
            )
        )
    return rows


def _events_in(
    events: NDArray[np.float64] | None, window: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The events whose start lies in a piece of ``window``."""
    assert events is not None  # checked against needs_events
    inside: NDArray[np.float64] = events[times_in_intervals(window, events[:, 0])]
    return inside


def _event_counts(
    trains: list[NDArray[np.float64]], events: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Spikes of each unit in each event, shape (n_events, n_units)."""
    return _count_spikes_in_time_bins(trains, events[:, 0], events[:, 1]).astype(
        np.float64
    )


def _rest_counts(
    method: ReactivationMethod,
    session: _Session,
    window: NDArray[np.float64],
    period: Literal["control", "match"],
) -> tuple[NDArray[np.float64], int | None]:
    """Counts of a control or match window, and its number of events.

    Returns
    -------
    counts : ndarray, shape (n_bins, n_units)
        One row per event with ``rest_bins="events"``, else per time bin.
    n_events : int or None
        Events in the window; None for fixed bins.
    """
    if method.rest_bins == "fixed":
        assert method.bin_size is not None
        return _counts(session.trains, window, method.bin_size), None
    events = _events_in(session.events[f"{period}_events"], window)
    _check_min_events(method, events.shape[0], period)
    return _event_counts(session.trains, events), events.shape[0]


def _check_min_events(method: ReactivationMethod, n_events: int, period: str) -> None:
    """Refuse a window without events, or with fewer than ``min_events``."""
    if not n_events:
        raise _MethodError(
            f"no {period}_events start inside the {period} window, so its "
            f"counts per event or event responses are undefined; pass "
            f"{period}_events that start inside {period}_epochs"
        )
    if method.min_events is not None and n_events < method.min_events:
        raise _MethodError(
            f"a {period} window has {n_events} events, fewer than "
            f"min_events={method.min_events}"
        )


def _template_counts(
    method: ReactivationMethod, session: _Session
) -> NDArray[np.float64]:
    """Counts of the template period, per event or per time bin."""
    if method.template_bins == "events":
        events = _events_in(session.events["template_events"], session.template)
        if not len(events):
            raise _MethodError("no template event starts inside template_epochs")
        return _event_counts(session.trains, events)
    assert method.bin_size is not None
    return _counts(session.trains, session.template, method.bin_size)


class _MethodError(ValueError):
    """A method cannot be applied to this session; reported in ``error``."""


def _checked_groups(
    unit_groups: Mapping[str, Any] | None,
    methods: Sequence[ReactivationMethod],
    n_units: int,
) -> dict[str, NDArray[Any]]:
    """Each needed group kind as an array with one entry per unit.

    Parameters
    ----------
    unit_groups : mapping or None
        Group arrays by kind.
    methods : sequence of ReactivationMethod
        The methods, whose ``pairs`` rules and ``detection_units`` say which
        kinds are needed.
    n_units : int
        Number of units.

    Returns
    -------
    dict of str to ndarray, each shape (n_units,)
    """
    given = dict(unit_groups or {})
    needed = {
        _GROUP_OF_PAIRS[method.pairs]: method.name
        for method in methods
        if method.pairs != "all"
    }
    for method in methods:
        if method.detection_units != "all":
            needed["region"] = method.name
    missing = {kind: name for kind, name in needed.items() if kind not in given}
    if missing:
        raise ValueError(
            _format_error(
                f"{_CALL}: methods {sorted(missing.values())} use groups of units "
                f"but unit_groups has no {sorted(missing)}.",
                why="Why: the papers used only pairs of cells on different "
                "tetrodes or shanks (cells on one tetrode can share sorting "
                "errors) or in different regions, or detected patterns in each "
                "region.",
                fix="pass unit_groups={'tetrode': tetrode_of_unit, ...} with one "
                "entry per unit",
            )
        )
    checked = {}
    for kind, values in given.items():
        if kind not in _GROUP_OF_PAIRS.values():
            raise ValueError(
                f"{_CALL}: unit_groups keys must be 'tetrode', 'shank' or "
                f"'region', got {kind!r}."
            )
        array = np.asarray(values)
        if array.shape != (n_units,):
            raise ValueError(
                f"{_CALL}: unit_groups[{kind!r}] must hold one group per unit "
                f"({n_units}), got shape {array.shape}."
            )
        checked[kind] = array
    return checked


def _counts(
    trains: list[NDArray[np.float64]], windows: NDArray[np.float64], bin_size: float
) -> NDArray[np.float64]:
    """Spike counts in ``bin_size`` bins inside ``windows``, (n_bins, n_units)."""
    return _binned(trains, windows, bin_size)[0]


def _binned(
    trains: list[NDArray[np.float64]], windows: NDArray[np.float64], bin_size: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Counts in ``bin_size`` bins inside ``windows`` and the bin centers.

    Returns
    -------
    counts : ndarray, shape (n_bins, n_units)
    centers : ndarray, shape (n_bins,)
    """
    from neurospatial.decoding import bin_spikes_in_time

    try:
        counts, centers = bin_spikes_in_time(trains, dt=bin_size, epochs=windows)
    except ValueError as error:
        raise _MethodError(str(error)) from error
    return counts.astype(np.float64), np.asarray(centers, dtype=np.float64)


def _pair_correlations(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pearson correlation of each pair, NaN with a constant unit.

    Parameters
    ----------
    counts : ndarray, shape (n_bins, n_units)
        Spike counts.

    Returns
    -------
    ndarray, shape (n_units * (n_units - 1) / 2,)
        In ``np.triu_indices(n_units, k=1)`` order.
    """
    z, constant = _z_scores(counts)
    matrix = z.T @ z / len(z)
    matrix[constant, :] = np.nan
    matrix[:, constant] = np.nan
    correlations: NDArray[np.float64] = matrix[np.triu_indices(counts.shape[1], k=1)]
    return correlations


def _windows_or_error(
    rule: ReactivationWindows, epochs: NDArray[np.float64], period: str
) -> list[NDArray[np.float64]]:
    """The rule's windows, refusing a period shorter than one window."""
    windows = rule.split(epochs)
    if not windows:
        duration = float(np.sum(epochs[:, 1] - epochs[:, 0]))
        raise _MethodError(
            f"the {period} epochs ({duration:g} s) are shorter than one "
            f"{rule.length:g} s window"
        )
    return windows


def _window_columns(index: int, window: NDArray[np.float64]) -> dict[str, Any]:
    return {
        "window": index,
        "window_start": float(window[0, 0]),
        "window_stop": float(window[-1, 1]),
        "window_duration": float(np.sum(window[:, 1] - window[:, 0])),
    }


def _match_window_rows(
    method: ReactivationMethod,
    session: _Session,
    window_rows: Callable[[int, NDArray[np.float64]], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Rows of each match window; a window that fails is its own error row.

    ``window_rows(index, window)`` gives the rows of one match window. A
    window that cannot be analysed (for example one without events) gives
    one row with its window columns and ``error``, and the other windows'
    rows are kept. The template and control periods are analysed before,
    once for every window, so a failure there fails the whole method.
    """
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(
        _windows_or_error(method.match_windows, session.match, "match")
    ):
        try:
            rows.extend(window_rows(index, window))
        except _MethodError as failure:
            rows.append({**_window_columns(index, window), "error": str(failure)})
    return rows


def _explained_variance_rows(
    method: ReactivationMethod,
    session: _Session,
    pair_rule: NDArray[np.bool_],
    stream: Any,
) -> list[dict[str, Any]]:
    """One row per match window of an EV method."""
    n_units = len(session.trains)
    if pair_rule.size < 3:
        raise _MethodError(
            f"EV needs at least 3 pairs of units; {n_units} units hold "
            f"{pair_rule.size} pair{'s' * (pair_rule.size != 1)}"
        )
    template_correlations = _pair_correlations(_template_counts(method, session))
    control_counts = []
    control_events = []
    for window in _windows_or_error(method.control_windows, session.control, "control"):
        counts, n_events = _rest_counts(method, session, window, "control")
        control_counts.append(counts)
        control_events.append(n_events)
    control_correlations = [_pair_correlations(counts) for counts in control_counts]
    n_controls = len(control_correlations)
    if (
        method.test is not None or method.statistic in _ONE_CONTROL_STATISTICS
    ) and n_controls != 1:
        needs = method.test if method.test is not None else method.statistic
        raise _MethodError(f"{needs} needs one control window, got {n_controls}")

    def window_rows(index: int, window: NDArray[np.float64]) -> list[dict[str, Any]]:
        match_counts, n_match_events = _rest_counts(method, session, window, "match")
        if method.n_rate_subsamples is None:
            match_correlations = _pair_correlations(match_counts)
            summary = _ev_summary(
                template_correlations,
                [(match_correlations, control) for control in control_correlations],
                pair_rule,
            )
        else:
            summary = _rate_equalized_ev_summary(
                template_correlations,
                match_counts,
                control_counts,
                pair_rule,
                method.n_rate_subsamples,
                method.statistic,
                stream(index, 1),
            )
        row: dict[str, Any] = {
            **_window_columns(index, window),
            "statistic": method.statistic,
            "value": _ev_statistics(summary)[method.statistic],
            "explained_variance": summary["explained_variance"],
            "reversed_explained_variance": summary["reversed_explained_variance"],
            "partial_correlation": summary["partial_correlation"],
            "semipartial_correlation": summary["semipartial_correlation"],
            "n_pairs": summary["n_pairs"],
            "n_control_windows": n_controls,
        }
        if method.rest_bins == "events":
            row["n_control_events"] = min(n for n in control_events if n is not None)
            row["n_match_events"] = n_match_events
        if method.test == "partial_correlation_t":
            p_value = summary["p_value"]
            row["p_value"] = p_value
            row["is_reactivated"] = (
                bool(p_value < method.alpha and summary["partial_correlation"] > 0)
                if np.isfinite(p_value)
                else _missing()
            )
        elif method.test == "pair_bootstrap":
            valid = (
                pair_rule
                & np.isfinite(template_correlations)
                & np.isfinite(match_correlations)
                & np.isfinite(control_correlations[0])
            )
            differences = _bootstrap_ev_minus_rev(
                template_correlations[valid],
                match_correlations[valid],
                control_correlations[0][valid],
                method.n_shuffles,
                stream(index, 0),
            )
            finite = differences[np.isfinite(differences)]
            row["n_shuffles_used"] = finite.size
            if finite.size:
                lower = float(np.percentile(finite, 100 * method.alpha / 2))
                row["ev_minus_rev_lower"] = lower
                row["is_reactivated"] = lower > 0
            else:
                row["is_reactivated"] = _missing()
        return [row]

    return _match_window_rows(method, session, window_rows)


def _ev_summary(
    template_correlations: NDArray[np.float64],
    periods: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    pair_rule: NDArray[np.bool_],
) -> dict[str, float]:
    """EV and REV of one match window, averaged over the control windows.

    Parameters
    ----------
    template_correlations : ndarray, shape (n_pairs,)
        Pair correlations of the template.
    periods : list of (ndarray, ndarray), each shape (n_pairs,)
        Pair correlations of the match window and of one control window,
        one tuple per control window.
    pair_rule : ndarray of bool, shape (n_pairs,)
        Pairs the method may use.

    Returns
    -------
    dict
        ``explained_variance`` and ``reversed_explained_variance`` (means
        over control windows), ``partial_correlation``,
        ``semipartial_correlation`` and ``p_value`` (NaN unless there is one
        control window) and ``n_pairs`` (fewest over control windows).
    """
    # One partial correlation per control window, each on the pairs
    # defined in the template, the match and that control window.
    results = [
        explained_variance_reactivation(
            template_correlations,
            match_correlations,
            control_correlations=control_correlations,
            pair_mask=pair_rule,
        )
        for match_correlations, control_correlations in periods
    ]
    single = len(results) == 1
    return {
        "explained_variance": _nanmean(
            [result.explained_variance for result in results]
        ),
        "reversed_explained_variance": _nanmean(
            [result.reversed_explained_variance for result in results]
        ),
        "partial_correlation": results[0].partial_correlation if single else np.nan,
        "semipartial_correlation": results[0].semipartial_correlation
        if single
        else np.nan,
        "p_value": results[0].p_value if single else np.nan,
        "n_pairs": min(result.n_pairs for result in results),
    }


def _ev_statistics(summary: dict[str, float]) -> dict[str, float]:
    """Each EV statistic a method can report, from an :func:`_ev_summary`."""
    explained = summary["explained_variance"]
    reversed_ = summary["reversed_explained_variance"]
    return {
        "explained_variance": explained,
        "ev_minus_rev": explained - reversed_,
        "ev_rev_contrast": ev_rev_contrast(explained, reversed_),
        "partial_correlation": summary["partial_correlation"],
        "semipartial_correlation": summary["semipartial_correlation"],
    }


def _rate_equalized_ev_summary(
    template_correlations: NDArray[np.float64],
    match_counts: NDArray[np.float64],
    control_counts: list[NDArray[np.float64]],
    pair_rule: NDArray[np.bool_],
    n_subsamples: int,
    statistic: str,
    generator: np.random.Generator,
) -> dict[str, float]:
    """:func:`_ev_summary` of the median rate-equalizing subsample.

    In each subsample, every (control window, match window) pair gets its
    own draw of :func:`equalize_firing_rates`, so each unit fires at one
    rate in both windows of the pair; the template is not subsampled (Giri
    et al. 2019). The subsamples are ordered by ``statistic``, and the one
    at the median (the lower of the middle two for an even number) is
    returned whole, so its EV, REV, partial correlation and p-value agree
    with each other and with ``statistic``. Separate medians of each would
    not: the median EV is not the squared median partial correlation, nor
    the median of EV - REV the difference of the medians.

    Parameters
    ----------
    template_correlations : ndarray, shape (n_pairs,)
        Pair correlations of the template.
    match_counts : ndarray, shape (n_match_bins, n_units)
        Spike counts of the match window.
    control_counts : list of ndarray, each shape (n_control_bins, n_units)
        Spike counts of each control window.
    pair_rule : ndarray of bool, shape (n_pairs,)
        Pairs the method may use.
    n_subsamples : int
        Number of subsamples.
    statistic : str
        The method's statistic, which orders the subsamples.
    generator : numpy.random.Generator
        Source of the subsamples.

    Returns
    -------
    dict
        The keys of :func:`_ev_summary`, from the median subsample over
        those that define ``statistic``; the first subsample when none does.
    """
    draws = []
    for _ in range(n_subsamples):
        periods = []
        for counts in control_counts:
            control, match = equalize_firing_rates(counts, match_counts, rng=generator)
            periods.append((_pair_correlations(match), _pair_correlations(control)))
        draws.append(_ev_summary(template_correlations, periods, pair_rule))
    values = np.array([_ev_statistics(draw)[statistic] for draw in draws])
    defined = np.flatnonzero(np.isfinite(values))
    if not defined.size:
        return draws[0]
    ordered = defined[np.argsort(values[defined], kind="stable")]
    return draws[int(ordered[(ordered.size - 1) // 2])]


def _missing() -> Any:
    """pandas' missing value, imported lazily."""
    import pandas as pd

    return pd.NA


def _nanmean(values: Sequence[float]) -> float:
    """Mean over the defined values, NaN when none is (no warning)."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else np.nan


def _bootstrap_ev_minus_rev(
    template: NDArray[np.float64],
    match: NDArray[np.float64],
    control: NDArray[np.float64],
    n_resamples: int,
    generator: np.random.Generator,
) -> NDArray[np.float64]:
    """EV - REV of pair resamples that keep each pair's three correlations.

    Parameters
    ----------
    template, match, control : ndarray, shape (n_pairs,)
        The defined pair correlations of the three periods.
    n_resamples : int
        Number of resamples.
    generator : numpy.random.Generator
        Source of the resamples.

    Returns
    -------
    ndarray, shape (n_resamples,)
        EV - REV of each resample of ``n_pairs`` pairs drawn with
        replacement; NaN for a resample whose EV or REV is undefined.
    """
    n_pairs = template.size
    if n_pairs < 3:
        return np.full(n_resamples, np.nan)
    t, m, c = template, match, control
    products = ["t", "m", "c", "tt", "mm", "cc", "tm", "tc", "mc"]
    columns = np.column_stack([t, m, c, t * t, m * m, c * c, t * m, t * c, m * c])
    # Every resample's weighted mean of each product, shape (n_resamples, 9).
    # The resamples are drawn in blocks of about 4 million counts, in the
    # order of one draw of all of them.
    sums = np.empty((n_resamples, len(products)))
    block = max(1, 4_000_000 // n_pairs)
    uniform = np.full(n_pairs, 1 / n_pairs)
    for first in range(0, n_resamples, block):
        size = min(block, n_resamples - first)
        # How often each pair is drawn in each resample, shape (size, n_pairs).
        sums[first : first + size] = (
            generator.multinomial(n_pairs, uniform, size) @ columns
        )
    moment = dict(zip(products, (sums / n_pairs).T, strict=True))

    def correlation(a: str, b: str) -> NDArray[np.float64]:
        covariance = moment[a + b] - moment[a] * moment[b]
        variance_a = moment[a + a] - moment[a] ** 2
        variance_b = moment[b + b] - moment[b] ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            result: NDArray[np.float64] = covariance / np.sqrt(variance_a * variance_b)
        return result

    r_tm, r_tc, r_mc = (
        correlation("t", "m"),
        correlation("t", "c"),
        correlation("m", "c"),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        explained = (r_tm - r_tc * r_mc) ** 2 / ((1 - r_tc**2) * (1 - r_mc**2))
        reversed_ = (r_tc - r_tm * r_mc) ** 2 / ((1 - r_tm**2) * (1 - r_mc**2))
    difference: NDArray[np.float64] = explained - reversed_
    return difference


# =============================================================================
# Assembly methods
# =============================================================================


def _detected_patterns(
    method: ReactivationMethod,
    counts: NDArray[np.float64],
    session: _Session,
    stream: Any,
) -> list[tuple[AssemblyPattern, str | None]]:
    """The method's patterns of the template counts, with their region.

    Each pattern has a weight for every unit (0 for units outside its
    detection) and the session's labels.

    Returns
    -------
    list of (AssemblyPattern, region)
        ``region`` is the region a pattern was detected in, ``"cross_region"``
        for a pattern of all units with members in several regions, or None
        when all units were detected together.
    """
    n_units = counts.shape[1]
    every = np.arange(n_units)
    sets: list[tuple[NDArray[np.intp], str | None]] = []
    if method.detection_units == "all":
        sets.append((every, None))
    else:
        regions = session.groups["region"]
        for region in np.unique(regions):
            sets.append((np.flatnonzero(regions == region), str(region)))
        if method.detection_units == "each_region_and_cross_region":
            sets.append((every, "cross_region"))
    generator = stream()
    patterns: list[tuple[AssemblyPattern, str | None]] = []
    failures = []
    for units, region in sets:
        try:
            detection = detect_assemblies(
                counts[:, units],
                method=method.detection,  # type: ignore[arg-type]
                member_rule=method.member_rule,
                member_threshold=2.0
                if method.member_threshold is None
                else method.member_threshold,
                rng=int(generator.integers(2**32 - 1)),
                unit_ids=session.labels[units],
            )
        except ValueError as error:
            if region is None:
                raise _MethodError(str(error)) from error
            # One region too small or too sparse to detect in: keep the
            # patterns of the others.
            failures.append(f"{error} (units of {region})")
            if len(failures) == len(sets):
                raise _MethodError(
                    "detection failed in every region: " + "; ".join(failures)
                ) from error
            warnings.warn(
                f"{failures[-1]}; the patterns of these units are left out.",
                UserWarning,
                stacklevel=2,
            )
            continue
        for pattern in detection.patterns:
            members = units[pattern.member_indices]
            if region == "cross_region":
                member_regions = session.groups["region"][members]
                if np.unique(member_regions).size < 2:
                    continue
            weights = np.zeros(n_units)
            weights[units] = pattern.weights
            full = AssemblyPattern(
                weights=weights,
                member_indices=np.sort(members).astype(np.int64),
                explained_variance_ratio=pattern.explained_variance_ratio,
                unit_ids=session.labels,
            )
            patterns.append((full, region))
    return patterns


def _assembly_rows(
    method: ReactivationMethod, session: _Session, stream: Any
) -> list[dict[str, Any]]:
    """One row per (match window, component) of an assembly method."""
    patterns = _detected_patterns(
        method, _template_counts(method, session), session, stream
    )
    control_windows = _windows_or_error(
        method.control_windows, session.control, "control"
    )
    if len(control_windows) != 1:
        raise _MethodError(
            f"an assembly method compares one control window, got {len(control_windows)}"
        )
    control_window = control_windows[0]
    control_samples = _period_samples(method, session.trains, control_window)
    template_samples = (
        _period_samples(method, session.trains, session.template)
        if method.zscore_over == "all_periods"
        else None
    )
    if method.event_response is not None:
        # Every match window is compared with the control window's events.
        _check_min_events(
            method,
            len(_events_in(session.events["control_events"], control_window)),
            "control",
        )
    n_components = len(patterns)
    all_weights = np.array([pattern.weights for pattern, _ in patterns])

    def window_rows(index: int, window: NDArray[np.float64]) -> list[dict[str, Any]]:
        window_columns = {
            **_window_columns(index, window),
            "statistic": method.statistic,
            "n_components": n_components,
        }
        if not n_components:
            return [window_columns]
        match_samples = _period_samples(method, session.trains, window)
        control_z, match_z, included = _z_scored(
            method, control_samples, match_samples, template_samples
        )
        # A unit without a z-score (constant over the samples it is z-scored
        # over) is left out of both periods, as assembly_reactivation() does,
        # and the weights are rescaled.
        weights = all_weights.copy()
        weights[:, ~included] = 0.0
        norms = np.linalg.norm(weights, axis=1)
        defined = norms > 0
        weights[defined] /= norms[defined, None]
        # Shape (n_bins, n_components).
        control_activation = _activation_strength(control_z, weights)
        match_activation = _activation_strength(match_z, weights)
        if method.test == "cell_identity_shuffle":
            assert method.percentile is not None
            # One set of shuffles per period, shared by the components.
            control_shares = _fraction_above_cell_shuffle(
                control_z[:, included],
                weights[:, included],
                method.n_shuffles,
                method.percentile,
                stream(index, 0),
            )
            match_shares = _fraction_above_cell_shuffle(
                match_z[:, included],
                weights[:, included],
                method.n_shuffles,
                method.percentile,
                stream(index, 1),
            )
        if method.event_response is not None:
            event_columns = _event_response_columns(
                method,
                session,
                (control_window, control_samples.times, control_activation),
                (window, match_samples.times, match_activation),
            )
        rows: list[dict[str, Any]] = []
        for component, (pattern, region) in enumerate(patterns):
            row = {
                **window_columns,
                "component": component,
                "region": region,
                "member_unit_ids": tuple(pattern.member_unit_ids.tolist()),
            }
            if method.event_response is not None:
                row.update(event_columns["shared"])
            if not defined[component]:
                rows.append(row)
                continue
            control_strength = control_activation[:, component]
            match_strength = match_activation[:, component]
            control_mean = float(control_strength.mean())
            match_mean = float(match_strength.mean())
            row.update(
                value=match_mean - control_mean,
                control_mean_activation=control_mean,
                match_mean_activation=match_mean,
            )
            if method.event_response is not None:
                row.update(event_columns["components"][component])
            if method.percentile is not None:
                threshold = np.percentile(control_strength, method.percentile)
                row["fraction_above_control_threshold"] = float(
                    np.mean(match_strength > threshold)
                )
            if method.activation_threshold is not None:
                row["control_activation_rate"] = _activation_rate(
                    control_strength,
                    control_samples.segments,
                    method.activation_threshold,
                )
                row["match_activation_rate"] = _activation_rate(
                    match_strength, match_samples.segments, method.activation_threshold
                )
            if method.test == "cell_identity_shuffle":
                row["control_fraction_above_shuffle"] = control_shares[component]
                row["match_fraction_above_shuffle"] = match_shares[component]
                row["n_shuffles_used"] = method.n_shuffles
            rows.append(row)
        return rows

    return _match_window_rows(method, session, window_rows)


def _event_response_columns(
    method: ReactivationMethod,
    session: _Session,
    control: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    match: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> dict[str, Any]:
    """Event-response columns of each component in one match window.

    Parameters
    ----------
    method : ReactivationMethod
        Its ``event_response``, ``statistic``, ``test`` and ``alpha``.
    session : _Session
        The events.
    control, match : tuple of (window, times, activation)
        Each period's window, sample times (n_samples,) and R(t), shape
        (n_samples, n_components).

    Returns
    -------
    dict
        ``"shared"``: the event counts; ``"components"``: one dict of
        columns per component.
    """
    response = method.event_response
    assert response is not None
    step = _sample_step(method)
    lags = _lags(response.window, step)
    in_response = _in_lag_window(lags, response.response_window, step)
    periods = {}
    for name, (window, times, activation) in (("control", control), ("match", match)):
        onsets = _events_in(session.events[f"{name}_events"], window)[:, 0]
        _check_min_events(method, onsets.size, name)
        # Shape (n_events, n_lags, n_components).
        values = _peri_event_values(times, activation, step, onsets, lags)
        responses = _nanmean_along(values[:, in_response], axis=1)
        if response.reduce == "mean":
            summary = _nanmean_along(responses, axis=0)
        else:
            peri_event = _nanmean_along(values, axis=0)[in_response]
            summary = np.where(
                np.isfinite(peri_event).any(axis=0),
                np.nanmax(
                    np.where(np.isfinite(peri_event), peri_event, -np.inf), axis=0
                ),
                np.nan,
            )
        baselines = (
            _nanmean_along(
                values[:, _in_lag_window(lags, response.baseline_window, step)], axis=1
            )
            if response.baseline_window is not None
            else None
        )
        periods[name] = (onsets.size, responses, summary, baselines)
    n_control, control_responses, control_summary, _ = periods["control"]
    n_match, match_responses, match_summary, match_baselines = periods["match"]
    components = []
    for component in range(control_responses.shape[1]):
        before = float(control_summary[component])
        after = float(match_summary[component])
        if method.statistic == "event_response_ratio":
            value = (after - before) / before if before > 0 else np.nan
            if not before > 0:
                warnings.warn(
                    f"component {component}: the control event response is "
                    f"{before:.3g}, not positive, so the ratio to it is undefined "
                    "(NaN).",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            value = after - before
        columns: dict[str, Any] = {
            "control_event_response": before,
            "match_event_response": after,
        }
        if method.statistic in _EVENT_RESPONSE_STATISTICS:
            columns["value"] = value
        if response.significant_z is not None:
            pooled = np.concatenate(
                [control_responses[:, component], match_responses[:, component]]
            )
            pooled = pooled[np.isfinite(pooled)]
            spread = pooled.std() if pooled.size else 0.0
            for name, responses in (
                ("control", control_responses[:, component]),
                ("match", match_responses[:, component]),
            ):
                defined = responses[np.isfinite(responses)]
                columns[f"{name}_significant_event_fraction"] = (
                    float(
                        np.mean(
                            (defined - pooled.mean()) / spread > response.significant_z
                        )
                    )
                    if defined.size and spread > 0
                    else np.nan
                )
        if method.test == "event_modulation":
            assert match_baselines is not None
            p_value, median = _signed_rank_p(
                match_responses[:, component], match_baselines[:, component]
            )
            columns["p_value"] = p_value
            columns["is_reactivated"] = (
                bool(p_value < method.alpha and median > 0)
                if np.isfinite(p_value)
                else _missing()
            )
        components.append(columns)
    return {
        "shared": {"n_control_events": n_control, "n_match_events": n_match},
        "components": components,
    }


@dataclass(frozen=True)
class _Samples:
    """One period's activity for R(t), before z-scoring.

    Attributes
    ----------
    values : ndarray, shape (n_samples, n_units)
        Counts, sliding counts or smoothed counts.
    times : ndarray, shape (n_samples,)
        Center of each sample's bin, seconds.
    segments : list of (first, stop, sample_duration)
        Sample ranges of each epoch piece and the seconds per sample.
    """

    values: NDArray[np.float64]
    times: NDArray[np.float64]
    segments: list[tuple[int, int, float]]


def _period_samples(
    method: ReactivationMethod,
    trains: list[NDArray[np.float64]],
    window: NDArray[np.float64],
) -> _Samples:
    """Activity of one period for R(t): counts, sliding or smoothed.

    Without smoothing, counts in ``bin_size`` bins, sliding by
    ``activation_step`` when it is given. With a smoothing bandwidth, spike
    trains convolved with a Gaussian of that standard deviation and sampled
    every ``activation_step`` (default half the bandwidth). Each epoch piece
    is tiled with bins and smoothed on its own, so nothing crosses a gap; a piece
    shorter than one bin holds no sample.
    """
    from itertools import pairwise

    from scipy.ndimage import gaussian_filter1d

    assert method.bin_size is not None  # assembly methods have one
    smoothing = method.smoothing_bandwidth
    step = _sample_step(method)
    width = method.bin_size if smoothing is None else step
    per_bin = round(width / step)
    pieces = window[window[:, 1] - window[:, 0] >= width]
    if not len(pieces):
        raise _MethodError(f"no epoch of the window holds a whole {width:g} s bin")
    # One pass over the spikes bins every piece; each piece's steps are then
    # summed or smoothed on their own and packed in place, as a piece has no
    # more samples than steps.
    values, times = _binned(trains, pieces, step)
    piece_of_step = np.searchsorted(pieces[:, 0], times, side="right") - 1
    bounds = np.searchsorted(piece_of_step, np.arange(len(pieces) + 1))
    n_samples = 0
    segments = []
    for first, stop in pairwise(bounds):
        counts, centers = values[first:stop], times[first:stop]
        if smoothing is not None:
            counts = gaussian_filter1d(
                counts, sigma=smoothing / step, axis=0, mode="constant"
            )
        elif per_bin > 1:
            # Sliding bins: each sample sums per_bin consecutive steps.
            cumulative = np.vstack(
                [np.zeros((1, counts.shape[1])), np.cumsum(counts, 0)]
            )
            counts = cumulative[per_bin:] - cumulative[:-per_bin]
            centers = centers[: counts.shape[0]] - step / 2 + width / 2
        n = counts.shape[0]
        if n:
            values[n_samples : n_samples + n] = counts
            times[n_samples : n_samples + n] = centers
            segments.append((n_samples, n_samples + n, step))
            n_samples += n
    if not n_samples:
        raise _MethodError(f"no epoch of the window holds a whole {width:g} s bin")
    return _Samples(values[:n_samples], times[:n_samples], segments)


def _sample_step(method: ReactivationMethod) -> float:
    """Seconds between samples of an assembly method's R(t).

    ``activation_step`` when given; otherwise ``bin_size`` without
    smoothing and half the smoothing bandwidth with it.
    """
    if method.activation_step is not None:
        return method.activation_step
    if method.smoothing_bandwidth is not None:
        return method.smoothing_bandwidth / 2
    assert method.bin_size is not None  # assembly methods have one
    return method.bin_size


def _z_scored(
    method: ReactivationMethod,
    control: _Samples,
    match: _Samples,
    template: _Samples | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Z-score the control and match samples as ``method.zscore_over`` says.

    Returns
    -------
    control_z, match_z : ndarray, shape (n_samples, n_units)
        0 for the units left out.
    included : ndarray of bool, shape (n_units,)
        Units with a z-score: not constant over the samples they are
        z-scored over (each period, for ``"period"``).
    """
    if method.zscore_over == "period":
        control_z, control_constant = _z_scores(control.values)
        match_z, match_constant = _z_scores(match.values)
        return control_z, match_z, ~(control_constant | match_constant)
    pooled = [control.values, match.values]
    if method.zscore_over == "all_periods":
        assert template is not None
        pooled.append(template.values)
    stacked = np.vstack(pooled)
    constant = np.ptp(stacked, axis=0) == 0
    mean = stacked.mean(axis=0)
    std = np.where(constant, 1.0, stacked.std(axis=0))
    control_z = (control.values - mean) / std
    match_z = (match.values - mean) / std
    control_z[:, constant] = 0.0
    match_z[:, constant] = 0.0
    return control_z, match_z, ~constant


def _member_cofiring_rows(
    method: ReactivationMethod, session: _Session, stream: Any
) -> list[dict[str, Any]]:
    """One row per (match window, component) of a member co-firing method."""
    patterns = _detected_patterns(
        method, _template_counts(method, session), session, stream
    )
    control_windows = _windows_or_error(
        method.control_windows, session.control, "control"
    )
    if len(control_windows) != 1:
        raise _MethodError(
            "a member co-firing method compares one control window, got "
            f"{len(control_windows)}"
        )
    control_counts, n_control_events = _rest_counts(
        method, session, control_windows[0], "control"
    )

    def window_rows(index: int, window: NDArray[np.float64]) -> list[dict[str, Any]]:
        match_counts, n_match_events = _rest_counts(method, session, window, "match")
        window_columns = {
            **_window_columns(index, window),
            "statistic": method.statistic,
            "n_components": len(patterns),
            "n_control_events": n_control_events,
            "n_match_events": n_match_events,
        }
        if not patterns:
            return [window_columns]
        rows: list[dict[str, Any]] = []
        for component, (pattern, region) in enumerate(patterns):
            cofiring = assembly_member_cofiring(control_counts, match_counts, pattern)
            rows.append(
                {
                    **window_columns,
                    "component": component,
                    "region": region,
                    "member_unit_ids": tuple(pattern.member_unit_ids.tolist()),
                    "value": cofiring.mean_correlation_difference,
                    "control_member_correlation": cofiring.control_mean_correlation,
                    "match_member_correlation": cofiring.match_mean_correlation,
                    "n_member_pairs": cofiring.n_member_pairs,
                }
            )
        return rows

    return _match_window_rows(method, session, window_rows)


def _activation_rate(
    activation: NDArray[np.float64],
    segments: list[tuple[int, int, float]],
    threshold: float,
) -> float:
    """Peaks of R(t) above ``threshold`` per second, within each epoch piece."""
    from scipy.signal import find_peaks

    n_peaks = 0
    duration = 0.0
    for first, stop, step in segments:
        peaks, _ = find_peaks(activation[first:stop], height=threshold)
        n_peaks += peaks.size
        duration += (stop - first) * step
    return n_peaks / duration


def _fraction_above_cell_shuffle(
    z: NDArray[np.float64],
    weights: NDArray[np.float64],
    n_shuffles: int,
    percentile: float,
    generator: np.random.Generator,
) -> NDArray[np.float64]:
    """Share of bins whose R(t) beats the percentile of its own shuffles.

    In each time bin the units' z-scores are permuted ``n_shuffles`` times
    (equivalently, the cells' identities), which keeps the bin's population
    activity and destroys which cells fire together (Peyrache et al. 2010).

    Parameters
    ----------
    z : ndarray, shape (n_bins, n_units)
        Z-scored activity of the units included.
    weights : ndarray, shape (n_components, n_units)
        Unit-length pattern weights.
    n_shuffles : int
        Permutations per bin, shared by the components.
    percentile : float
        Percentile of each bin's shuffled R(t).
    generator : numpy.random.Generator
        Source of the permutations.

    Returns
    -------
    ndarray, shape (n_components,)
        Share of bins with R(t) strictly above their shuffle percentile.
    """
    actual = _activation_strength(z, weights)
    n_bins, n_units = z.shape
    squared = weights**2
    above = np.zeros(len(weights))
    # Bins per block so a block's shuffles hold about 4 million values.
    block = max(1, 4_000_000 // max(1, n_shuffles * n_units))
    for first in range(0, n_bins, block):
        values = np.repeat(z[first : first + block, None, :], n_shuffles, axis=1)
        shuffled = generator.permuted(values, axis=2)
        # Shape (n_block_bins, n_shuffles, n_components).
        strength = (shuffled @ weights.T) ** 2 - shuffled**2 @ squared.T
        bound = np.percentile(strength, percentile, axis=1)
        above += np.sum(actual[first : first + block] > bound, axis=0)
    shares: NDArray[np.float64] = above / n_bins
    return shares


# =============================================================================
# Published recipes
# =============================================================================

KUDRIMOTI_1999 = ReactivationMethod(
    name="kudrimoti_1999",
    reference=(
        "Kudrimoti, H. S., Barnes, C. A., & McNaughton, B. L. (1999). "
        "Reactivation of hippocampal cell assemblies: effects of behavioral "
        "state, experience, and EEG dynamics. The Journal of Neuroscience, "
        "19(10), 4090-4101."
    ),
    measure="explained_variance",
    bin_size=0.1,
    pairs="different_tetrodes",
    control_windows=ReactivationWindows(length=600.0, anchor="end", count=1),
    match_windows=ReactivationWindows(length=600.0, count=3),
    statistic="explained_variance",
    test="partial_correlation_t",
    alpha=0.05,
    notes=(
        "100 ms bins: 'The spike trains of n pyramidal neurons in a data set "
        "were binned into T (100 msec) intervals' (p. 4092); 50 and 200 ms "
        "gave 'comparable results'.",
        "Pairs on different tetrodes: 'only correlations between cells on "
        "different tetrodes were used in the analyses' (p. 4092).",
        "EV = r²(RUN, POST | PRE) with control the last 10 min of PRE and "
        "match three 10-min POST blocks: 'The analysis typically used "
        "correlations during the last 10 min of baseline SWS (PRE) and track "
        "running (RUN) as independent variables and the correlations in "
        "three 10 min epochs in postbehavior SWS (POST) as dependent "
        "variables' (p. 4093). Pass slow-wave sleep as control_epochs and "
        "match_epochs.",
        "Whether the 10-min windows are contiguous clock time or "
        "concatenated SWS time is not stated; here they are concatenated "
        "time inside the epochs given.",
        "Per-session test: the paper reports p-values of the partial "
        "correlation computed with SPSS without naming the test ('There was "
        "a significant relationship (rRUN–POST|PRE, p < 0.0001)', p. 4094); "
        "the standard partial-correlation t test with n_pairs - 3 degrees of "
        "freedom is inferred. Pairs share cells, so it is anticonservative.",
        "Reactivated when p < 0.05 and the partial correlation is positive; "
        "0.05 is the level of the paper's across-session tests ('p < 0.05, t "
        "tests', p. 4096).",
        "Kudrimoti et al. define no reversed EV; REV (Pennartz et al. 2004) "
        "is reported for comparison and not tested.",
        "Pyramidal-cell and stability inclusion criteria are not applied.",
    ),
)

PENNARTZ_2004 = ReactivationMethod(
    name="pennartz_2004",
    reference=(
        "Pennartz, C. M. A., Lee, E., Verheul, J., Lipa, P., Barnes, C. A., & "
        "McNaughton, B. L. (2004). The ventral striatum in off-line "
        "processing: ensemble reactivation during sleep and modulation by "
        "hippocampal ripples. The Journal of Neuroscience, 24(29), 6446-6456."
    ),
    measure="explained_variance",
    bin_size=0.05,
    pairs="all",
    control_windows=ReactivationWindows(
        length=600.0, anchor="end", count=1, clock="wall"
    ),
    match_windows=ReactivationWindows(length=600.0, clock="wall"),
    statistic="ev_minus_rev",
    test=None,
    notes=(
        "50 ms bins: 'A bin size of 50 msec was chosen unless noted "
        "otherwise' (Materials and Methods).",
        "All pairs: 'correlations between all active, stable, and well "
        "isolated neurons are presented; when we removed correlations "
        "between cells recorded on the same tetrode, no significant "
        "deviation was noted.'",
        "Windows on the clock: 'we used the last 10 min of REST1, "
        "immediately preceding the behavioral phase, as well as the first 10 "
        "min of REST2'; the time course used every 10-min block of REST2 "
        "with the same control: 'The EV for the four blocks making up REST2 "
        "was calculated using the last 10 min of REST1'. Window 0 is the "
        "main analysis.",
        "REV is introduced here: 'an REV was computed by switching around R1 "
        "and R2 in Equation 3'; the statistic is EV - REV.",
        "No per-session test: EV and REV were compared across sessions "
        "('Wilcoxon's matched-pairs signed rank test'), so is_reactivated "
        "is missing.",
    ),
)

TATSUNO_2006 = ReactivationMethod(
    name="tatsuno_2006",
    reference=(
        "Tatsuno, M., Lipa, P., & McNaughton, B. L. (2006). Methodological "
        "considerations on the use of template matching to study long-lasting "
        "memory trace replay. The Journal of Neuroscience, 26(42), "
        "10727-10742."
    ),
    measure="explained_variance",
    bin_size=0.25,
    pairs="different_tetrodes",
    control_windows=ReactivationWindows(length=900.0, anchor="end", clock="wall"),
    match_windows=ReactivationWindows(length=900.0, clock="wall"),
    statistic="explained_variance",
    test=None,
    notes=(
        "15-min segments and 250 ms bins: 'The pre-exposure and "
        "postexposure epochs are divided into 15 min segments … with a "
        "typical bin-width of 250 ms and MPRE(1) corresponding to 15 min "
        "just before the novel experience' (Explained variance method). "
        "Pre-exposure segments are cut back from its end; whether a "
        "remainder shorter than 15 min is used is not stated (dropped here).",
        "Pairs on different tetrodes: 'the pair-wise correlation matrix cij "
        "of all cell pairs from different tetrodes is calculated'.",
        "Several control blocks: EV(POST(h)) = (1/K) Σ_k R²(EXP, POST(h) | "
        "PRE(k)), and REV likewise; 'For the error bars, the SDs over the K "
        "pre-exposure blocks are taken' (not reported here).",
        "The printed EV denominator √((1 − R)²) is a typesetting error for "
        "√(1 − R²): the text defines EV as 'The square of the partial "
        "correlation'. Kudrimoti's formula is used.",
        "No per-session test: 'the EV method compares the EV values with "
        "reversed-EV values', in 'a wider neighborhood of the segment'.",
    ),
)

LANSINK_2009 = ReactivationMethod(
    name="lansink_2009",
    reference=(
        "Lansink, C. S., Goltstein, P. M., Lankelma, J. V., McNaughton, B. L., "
        "& Pennartz, C. M. A. (2009). Hippocampus leads ventral striatum in "
        "replay of place-reward information. PLoS Biology, 7(8), e1000173."
    ),
    measure="explained_variance",
    bin_size=0.05,
    pairs="different_regions",
    control_windows=ReactivationWindows(length=1200.0, anchor="end", count=1),
    match_windows=ReactivationWindows(length=1200.0),
    statistic="ev_minus_rev",
    test="pair_bootstrap",
    n_shuffles=10_000,
    alpha=0.05,
    notes=(
        "50 ms bins: 'binned spike trains (50-ms bin size) of each "
        "rest/active episode' (Reactivation analysis).",
        "Cross-region pairs: 'pairs always consisted of one hippocampal and "
        "one ventral striatal cell; i.e., intra-area pairs were not taken "
        "into account.' Pass unit_groups={'region': ...}.",
        "Blocks of concatenated rest: '20-min time blocks of concatenated "
        "periods of quiet rest and sleep, i.e., periods of active behavior "
        "were excluded'. Pass rest periods as control_epochs and "
        "match_epochs.",
        "Which part of pre-behavioral rest is the control is not stated; "
        "here its last 20-min block, as Pennartz et al. (2004) used the "
        "last 10 min of REST1.",
        "Bootstrap: 'random samples were taken (n = 10,000) from the "
        "observed set of Pearson correlation coefficients … the triplets of "
        "correlation coefficients belonging to the three rest/active "
        "episodes were kept together … A subgroup was considered to "
        "reactivate when the 2.5 percentile of this distribution [EV−REV] "
        "exceeded zero' (Reactivation analysis for subgroups). The paper "
        "applied it to subgroups of pairs pooled across sessions; here it "
        "is applied to one session's pairs. Sessions were compared with "
        "'the Wilcoxon matched-pairs signed rank test'.",
        "Inclusion ('at least five well-isolated active neurons from each "
        "area'; interneurons excluded) is not applied.",
    ),
)

GIRI_2019 = ReactivationMethod(
    name="giri_2019",
    reference=(
        "Giri, B., Miyawaki, H., Mizuseki, K., Cheng, S., & Diba, K. (2019). "
        "Hippocampal reactivation extends for several hours following novel "
        "experience. The Journal of Neuroscience, 39(5), 866-875."
    ),
    measure="explained_variance",
    bin_size=0.25,
    pairs="different_shanks",
    control_windows=ReactivationWindows(length=900.0, step=300.0, clock="wall"),
    match_windows=ReactivationWindows(length=900.0, step=300.0, clock="wall"),
    statistic="explained_variance",
    test=None,
    notes=(
        "'Spike times were binned into 250 ms time bins' (Explained variance measure).",
        "'Pearson's correlations, R, were determined for spike counts from "
        "neuronal pairs recorded on different shanks in 15 min sliding "
        "windows (window length 15 min, sliding 5 min steps) in PRE and POST "
        "and in the entire MAZE session'. Pass unit_groups={'shank': ...}.",
        "EV and REV for each POST window 'averaged over all PRE(k) windows "
        "for which WIN ≠ PRE(k)'; a POST window never is a PRE window, so "
        "every PRE window is averaged. REV 'as proposed in a previous study "
        "(Tatsuno et al., 2006)'. 'Error bars … indicate SD over PRE(k)' "
        "(not reported here).",
        "No per-session test.",
        "Rate control, not applied here: 'we repeated this analysis after "
        "subsampling to equalize the firing rate for each neuron in PRE and "
        "POST … (median Pearson's r = 0.454, median p = 6.417 × 10−197 for "
        "100 different random subsamples equalizing PRE and POST spike "
        "counts of each cell; no. of pairs = 3809)' (Results). It checked "
        "the pooled comparison of whole PRE and POST NREM sleep (pairs of "
        "all sessions, Fig. 1c-e), not the 15-min windows. "
        "dataclasses.replace(GIRI_2019, n_rate_subsamples=100) applies it to "
        "every pair of PRE and POST windows, where equal rates and equal "
        "counts are the same, and reports the subsample with the median EV "
        "(the paper took separate medians of r and p, which can come from "
        "different subsamples). "
        "Which spikes were removed is not stated; here a uniform random "
        "subset of each unit's spikes in the window that fires more.",
    ),
)

GIRI_2024 = ReactivationMethod(
    name="giri_2024",
    reference=(
        "Giri, B., Kinsky, N., Kaya, U., Maboudi, K., Abel, T., & Diba, K. "
        "(2024). Sleep loss diminishes hippocampal reactivation and replay. "
        "Nature, 630(8018), 935-942."
    ),
    measure="explained_variance",
    bin_size=0.25,
    pairs="all",
    control_windows=ReactivationWindows(length=900.0, step=300.0, clock="wall"),
    match_windows=ReactivationWindows(length=900.0, step=300.0, clock="wall"),
    statistic="explained_variance",
    test=None,
    notes=(
        "'spike times were binned into 250 ms time bins' and correlated 'in "
        "15 min sliding windows (window length 15 min, sliding 5 min steps)' "
        "(Methods, Explained variance measure for reactivation).",
        "'only pairs with waveform similarity < 0.8 were used': pass that "
        "as pair_mask=.",
        "EV for a POST window 'averaged over all windows in PRE'. The "
        "printed REV formula has EV's denominator; the lab's code (NeuroPy "
        "ExplainedVariance) computes the partial correlation of the MAZE and "
        "PRE(k) correlations given the window's and averages the squared "
        "partial correlations over PRE windows, which is followed here. As "
        "in that code, each PRE window uses the pairs defined in it.",
        "No per-session test; sessions were combined with a hierarchical "
        "bootstrap. Inclusion ('Only sessions with > 15 stable units') is "
        "not applied.",
    ),
)

PEYRACHE_2010 = ReactivationMethod(
    name="peyrache_2010",
    reference=(
        "Peyrache, A., Benchenane, K., Khamassi, M., Wiener, S. I., & "
        "Battaglia, F. P. (2010). Principal component analysis of ensemble "
        "recordings reveals cell assemblies at high temporal resolution. "
        "Journal of Computational Neuroscience, 29(1-2), 309-325."
    ),
    measure="assembly",
    bin_size=0.1,
    detection="pca",
    statistic="mean_activation_difference",
    test="cell_identity_shuffle",
    percentile=99.0,
    n_shuffles=1000,
    notes=(
        "'Throughout the article, the bin width is fixed at 100 ms.'",
        "Signal components: principal components of the template period "
        "with eigenvalues above the Marchenko-Pastur edge; R(t) with the "
        "diagonal of the projector removed, each epoch z-scored on its own.",
        "Statistic: mean R(t) in POST minus PRE ('Note that the average "
        "reactivation strength is equal to γ − 1'; Fig. 6c compares 'the "
        "reactivation measure average for sleep PRE and POST'). Pass "
        "slow-wave sleep as control_epochs and match_epochs.",
        "Share of POST bins above the PRE 99th percentile: '4.5% of the bins "
        "from POST exceeded the 99th percentile of the PRE distribution', "
        "against the expected 1% (Reactivation is a rare event).",
        "Cell-identity shuffle: 'for each time bin, the identity of the "
        "cells was randomly permuted … shuffled 1000 times. The 99th "
        "percentile of the resulting distribution, for each time bin' (1% "
        "of PRE and 3.4% of POST bins above it in the example).",
        "No per-component verdict: shares and differences were tested "
        "across components pooled over sessions (t-tests), so "
        "is_reactivated is missing.",
    ),
)

VAN_DE_VEN_2016 = ReactivationMethod(
    name="van_de_ven_2016",
    reference=(
        "van de Ven, G. M., Trouche, S., McNamara, C. G., Allen, K., & Dupret, "
        "D. (2016). Hippocampal offline reactivation consolidates recently "
        "formed cell assembly patterns during sharp wave-ripples. Neuron, "
        "92(5), 968-974."
    ),
    measure="assembly",
    bin_size=0.025,
    detection="ica",
    smoothing_bandwidth=0.025 / np.sqrt(12.0),
    activation_threshold=5.0,
    statistic="mean_activation_difference",
    test=None,
    notes=(
        "Detection on 25 ms bins: 'Epochs of active exploration … were "
        "divided into 25 ms time bins' (Supplemental Experimental "
        "Procedures, p. 10); z-scored, PCA with the Marchenko-Pastur "
        "threshold, then ICA in the significant subspace; weights 'scaled "
        "to unit length' with the highest absolute weight positive.",
        "Tracking: 'a Gaussian kernel was chosen with standard deviation "
        "w/√12, so that the kernel had the same standard deviation as fixed "
        "time bins of w ms … We set w to 25 ms' (p. 12), so σ ≈ 7.2 ms. "
        "The smooth trace's sampling is not stated; here spike counts in "
        "σ/2 bins are smoothed with a Gaussian of σ.",
        "The smooth trace 'was normalized by a z-score transform' (p. 12); "
        "over which epoch is not stated: here each rest period on its own.",
        "Reactivation strength: 'its average expression strength during the "
        "rest after minus its average expression strength during the rest "
        "before' (p. 12).",
        "Activations: 'peaks in the expression strength exceeding R_THRES = "
        "5' (p. 12); their rate per second in each rest period is reported.",
        "No per-pattern test: the patterns' reactivation strengths were "
        "tested against 0 with a one-sample t-test and the share of "
        "patterns stronger after than before against 50% with a z-test for "
        "proportions (p. 13-14).",
    ),
)

_SWR_EVENTS_NOTE = (
    "One bin per SWR: pass each sleep period's SWR (start, stop) times as "
    "control_events and match_events. SWRs vary in length, so spike counts "
    "per SWR share a duration component that raises every pair correlation "
    "alike; it cancels between the periods only when their SWRs have "
    "similar durations."
)

OLIVA_2020_EV = dataclasses.replace(
    KUDRIMOTI_1999,
    name="oliva_2020_ev",
    reference=(
        "Oliva, A., Fernández-Ruiz, A., Leroy, F., & Siegelbaum, S. A. (2020). "
        "Hippocampal CA2 sharp-wave ripples reactivate and promote social "
        "memory. Nature, 587(7833), 264-269. "
        "https://doi.org/10.1038/s41586-020-2758-y"
    ),
    bin_size=0.1,
    rest_bins="events",
    pairs="all",
    control_windows=ReactivationWindows(),
    match_windows=ReactivationWindows(),
    statistic="ev_minus_rev",
    test=None,
    notes=(
        "Kudrimoti's EV with the social-learning trials as template: "
        "'r_social episode,POST and r_social episode,PRE are the Pearson's "
        "correlation coefficients of pairwise correlation matrices between "
        "the social learning episodes and SWRs times of the post-sleep and "
        "pre-sleep periods' (Methods, Reactivation analysis). The printed "
        "formula is garbled in the manuscript; it is Kudrimoti's.",
        "100 ms bins: 'Pairwise neuronal correlations for EV and REV were "
        "calculated using the Pearson correlation coefficient on 100 ms "
        "binned spike trains'. 100 ms bins cannot tile SWRs, which are "
        "mostly shorter: here the task is binned at 100 ms and each SWR is "
        "one bin ('all epochs/events were detected individually and pooled "
        "together'), as in the lab's code (neurocode ExplainedVariance.m "
        "counts spikes per interval). " + _SWR_EVENTS_NOTE,
        "Whole sleep periods: SWRs of all of pre- and post-sleep are pooled; "
        "no window is stated.",
        "All pairs of interest: 'calculated in each session using the cell "
        "pairs of interest ... for pyramidal cells only'; pass the pairs as "
        "pair_mask=.",
        "Statistic EV - REV: 'The control value (REV) was obtained by "
        "switching the temporal order of the pre- and post-sleep sessions'; "
        "Extended Data Fig. 8b plots '(EV – REV) measure of post-sleep "
        "reactivation'.",
        "No per-session test: 'EV was significantly higher than REV (chance) "
        "... n = 13 mice' (Fig. 3b), so is_reactivated is missing. The "
        "per-pair contributions are explained_variance_contributions().",
    ),
)

OLIVA_2020_ASSEMBLY = dataclasses.replace(
    VAN_DE_VEN_2016,
    name="oliva_2020_assembly",
    reference=OLIVA_2020_EV.reference,
    bin_size=0.02,
    smoothing_bandwidth=0.01,
    zscore_over="all_periods",
    notes=(
        "20 ms bins: 'spike trains of each neuron were binned in 20-ms "
        "intervals for the whole session and z-scored firing rates were "
        "calculated for each bin' (Methods, Cell assembly analysis). "
        "Detection uses the social-learning trials passed as "
        "template_epochs.",
        "Smoothing: 'Spike trains were convolved with a Gaussian kernel (SD = "
        "10 ms)'. Whether detection used the smoothed or the binned trains is "
        "not clear; here detection uses the 20 ms counts and R(t) the "
        "smoothed trains, as in VAN_DE_VEN_2016, sampled every 5 ms.",
        "Detection: 'principal components whose eigenvalues exceeded the "
        "threshold for random firing correlations (Marčenko-Pastur "
        "distribution). Independent component analysis was then used'.",
        "Activations: 'Assembly activity was only considered when its "
        "expression strength exceeded a threshold of 5'.",
        "Statistic: 'Reactivation strength of an assembly was defined as the "
        "difference in its average expression strength during post-sleep "
        "minus that during pre-sleep'. Over all sleep or only SWRs is not "
        "stated; pass the sleep (or NREM) epochs.",
        "Z-scoring: R(t) multiplies 'the convolved z-scored firing rate', "
        "and the rates were z-scored 'for the whole session', read here as "
        "the template, control and match periods together "
        "(zscore_over='all_periods'), as in HARVEY_2023_ASSEMBLY, whose code "
        "(Fernández-Ruiz lab) z-scores over the whole session. "
        "VAN_DE_VEN_2016 z-scores each period on its own.",
        "No per-pattern test in the text.",
    ),
)

EL_GABY_2021 = ReactivationMethod(
    name="el_gaby_2021",
    reference=(
        "El-Gaby, M., Reeve, H. M., Lopes-dos-Santos, V., Campo-Urriza, N., "
        "Perestenko, P. V., Morley, A., Strickland, L. A. M., Lukács, I. P., "
        "Paulsen, O., & Dupret, D. (2021). An emergent neural coactivity code "
        "for dynamic memory. Nature Neuroscience, 24(5), 694-704. "
        "https://doi.org/10.1038/s41593-021-00820-w"
    ),
    measure="member_cofiring",
    bin_size=0.025,
    rest_bins="events",
    min_events=200,
    detection="ica",
    member_rule="positive_z",
    member_threshold=2.0,
    statistic="mean_member_correlation_difference",
    test=None,
    notes=(
        "Detection on 25 ms counts: 'Spikes discharged by each neuron were "
        "counted in 25-ms ... time bins and standardised (z-scored ...)' "
        "(Methods: Neuronal pattern isolation and tracking); 'the number of "
        "principal components of the activity matrix with variances above a "
        "threshold derived from an analytical probability function for "
        "uncorrelated data ... independent component analysis' (ref. 26, "
        "Lopes-dos-Santos et al. 2013). No smoothing is stated.",
        "Template: 'active periods (speed > 2 cm.s-1) separately during the "
        "entire last session of contingency X, contingency Y (i.e. X2 and Y2) "
        "or the exploration session of the control enclosure'. Pass one "
        "session's running epochs as template_epochs and call once per "
        "session.",
        "Members: 'the direction where weights were highest was assigned "
        "positive weights, and principal CA1 neurons whose weight was "
        "positive and exceeded 2 standard deviations from the mean were "
        'defined as pattern "members"\' (signed weights, unit length, the '
        "largest absolute weight positive); 1 and 3 SD were robustness "
        "checks.",
        "Rest bins: 'we used SWR events as activity bins and calculated the "
        "Pearson correlations between pairs of pattern members separately "
        "across each sleep session ... only sessions where at least 200 SWR "
        "events were detected were included' (Methods: LFP analyses). Whether "
        "counts were divided by SWR duration is not stated; counts are used. "
        + _SWR_EVENTS_NOTE,
        "Control and match: Fig. 4c compares pre-learning (post-exploration) "
        "sleep with post-learning sleep; Fig. 4b pre- with post-exploration "
        "sleep. Sleep analyses are not speed-filtered.",
        "Statistic: mean member-pair r in the match period minus the control "
        "period, over the pairs defined in both. The paper reports the "
        "per-session means and forms no difference.",
        "No per-pattern or per-session test: Fig. 4b,c report DABEST "
        "bootstrap confidence intervals (1,000 resamples) without stating "
        "whether pairs, members or patterns are the unit, so is_reactivated "
        "is missing.",
        "The 'contingency-discriminating' and 'contingency-invariant' "
        "classes use best-match cosine similarity with thresholds pooled "
        "over recording days (assembly_similarity()); join your "
        "classification to component and member_unit_ids.",
    ),
)

_HARVEY_2023_REFERENCE = (
    "Harvey, R. E., Robinson, H. L., Liu, C., Oliva, A., & Fernandez-Ruiz, A. "
    "(2023). Hippocampo-cortical circuits for selective memory encoding, "
    "routing, and replay. Neuron, 111(13), 2076-2090. "
    "https://doi.org/10.1016/j.neuron.2023.04.015"
)
_LIU_2023_REFERENCE = (
    "Liu, C., Todorova, R., Tang, W., Oliva, A., & Fernandez-Ruiz, A. (2023). "
    "Associative and predictive hippocampal codes support memory-guided "
    "behaviors. Science, 382(6668), eadi8237. "
    "https://doi.org/10.1126/science.adi8237"
)

HARVEY_2023_ASSEMBLY = ReactivationMethod(
    name="harvey_2023_assembly",
    reference=_HARVEY_2023_REFERENCE,
    measure="assembly",
    bin_size=0.025,
    control_windows=ReactivationWindows(
        length=3600.0, count=1, clock="wall", allow_shorter=True
    ),
    match_windows=ReactivationWindows(
        length=3600.0, count=1, clock="wall", allow_shorter=True
    ),
    detection="ica",
    member_rule="otsu",
    member_threshold=None,
    smoothing_bandwidth=0.025 / np.sqrt(12.0),
    activation_step=0.002,
    zscore_over="all_periods",
    activation_threshold=5.0,
    event_response=EventResponse(response_window=(0.0, 0.05), reduce="peak"),
    statistic="event_response_difference",
    test=None,
    notes=(
        "Detection on 25 ms bins: 'The spike trains for each neuron were "
        "binned into time windows (25-ms for inter-hippocampal in Figure 1 "
        "...) and z-score transformed ... Marcenko-Pastur ... ICA, using the "
        "fast ICA algorithm' (STAR Methods, Cell assembly analyses, p. e5). "
        "The lab's code detects in the task's theta epochs: pass them as "
        "template_epochs.",
        "Members: 'Cell assembly members were identified using Otsu's method "
        "to divide the absolute weights into two groups that maximized "
        "inter-class variance. Neurons in the group with greater absolute "
        "weights were classified as members.'",
        "R(t): 'z(t) was obtained by convolving the spike-train of each "
        "neuron with a Gaussian kernel (s.d. = w/√(dt/2)) and then z-scoring "
        "each trace. w was set to 25 ms'. The code computes the s.d. as "
        "w / √⌊w_ms / 2⌋, 25 / √12 ≈ 7.2 ms (50 / √25 = 10 ms for the 50 ms "
        "bins of Fig. 7), on spikes binned at 2 ms: activation_step=0.002.",
        "Z-scoring: 'z-scoring each trace'; the code z-scores over the whole "
        "session, read here as the template, control and match periods "
        "together (zscore_over='all_periods').",
        "Windows: the code takes the first 3,600 s of the pre- and post-task "
        "sleep sessions restricted to NREM, or all of a shorter session "
        "(ripple_heterogeneity assembly_react_deep_sup_ind_detect.py: "
        "epochs[0][EpochArray([pre_start, pre_start + 3600])][nrem_epochs]); "
        "here the first hour on the clock from the first NREM epoch given, "
        "or all of it when shorter (allow_shorter=True). Pass the first hour "
        "of each sleep session intersected with NREM to match the code.",
        "Response: Fig. 2G, 'SWR-triggered average reactivation strength in "
        "post-task sleep'; the figure's statistic in the code is the peak of "
        "the post-sleep peri-SWR mean over 0-50 ms after SWR start "
        "(reduce='peak', response_window=(0, 0.05)), around ±0.5 s.",
        "Activations: 'Assembly activations were defined as peaks exceeding "
        "R_THRES > 5'.",
        "Statistic: the paper compares post-sleep responses of deep and "
        "superficial assemblies ('p = 0.022, rank-sum test, 197 "
        "assemblies'), so match_event_response is the paper's number; post "
        "minus pre is reported for comparison. No per-pattern test.",
        "Fig. 7 variant: dataclasses.replace(HARVEY_2023_ASSEMBLY, "
        "name=..., bin_size=0.05, smoothing_bandwidth=0.01, "
        "activation_step=0.005, zscore_over='control_and_match'). Its "
        "cross-region selection and per-lag z-scoring of the peri-event "
        "curve are not applied. The code also drops patterns whose Otsu "
        "members have mixed signs (not applied by the Fig. 2 scripts, nor "
        "here).",
    ),
)

LIU_2023_EV = ReactivationMethod(
    name="liu_2023_ev",
    reference=_LIU_2023_REFERENCE,
    measure="explained_variance",
    bin_size=None,
    template_bins="events",
    rest_bins="events",
    statistic="explained_variance",
    test=None,
    notes=(
        "Theta-cycle template: 'Each spike train was binned in individual "
        "theta cycles, and the cofiring of a pair of cells was defined as the "
        "Pearson correlation coefficient of the binned firing rates of the "
        "two cells across all theta cycles in the recorded session' "
        "(Materials and methods, Coactivity and reactivation analysis, p. "
        "13). Pass the task's theta cycles as template_events.",
        "Counts, not rates: the paper says 'binned firing rates', but the "
        "lab's code counts each unit's spikes in each theta cycle and each "
        "SWR (neurocode CofiringCoefficient.m: 'count the spikes emitted by "
        "each unit in each interval'), and so does this recipe. Theta "
        "cycles vary in length, so counts per cycle share a duration "
        "component that raises every pair correlation alike; dividing by "
        "the cycle's duration would remove it and give a different r.",
        "SWR bins in sleep: 'C, CPRE, and CPOST refer to the cofiring matrix "
        "and the correlation matrices across SWRs in pre- and post-task "
        "sleep'; the lab's code counts spikes per SWR (neurocode "
        "ExplainedVariance.m). " + _SWR_EVENTS_NOTE,
        "EV = ((r_C,CPOST − r_C,CPRE r_CPOST,CPRE) / √((1 − r²_C,CPRE)(1 − "
        "r²_CPOST,CPRE)))², 'compared with the control value of reverse "
        "explained variance (REV) obtained by recomputing EV after swapping "
        "CPRE and CPOST'.",
        "Windows: 'EV was computed for each recorded session separately'; no "
        "sleep window is stated, so all SWRs of each sleep period are used.",
        "No per-session test: 'Wilcoxon signed-rank test ... n = 5 sessions' "
        "of EV against REV (Fig. 4E).",
    ),
)

LIU_2023_PAIRWISE = dataclasses.replace(
    LIU_2023_EV,
    name="liu_2023_pairwise",
    statistic="semipartial_correlation",
    notes=(
        "Pairwise reactivation: 'we used linear regression between CPOST and "
        "CPRE to remove the portion of the variance in CPOST that can be "
        "explained by preexisting patterns, and we correlated the resultant "
        "residuals CRESIDUAL with C' (after Giri et al. 2019). Only the "
        "post-task side is residualized, so this is the semipartial "
        "correlation (r_tm − r_tc r_mc) / √(1 − r_mc²), not the partial "
        "correlation; the lab's CorrResiduals.m fits the line with an "
        "intercept over the pairs defined in both periods.",
        "Bins as LIU_2023_EV: one per theta cycle in the task (template_events) "
        "and one per SWR in sleep. " + _SWR_EVENTS_NOTE,
        "For the cheeseboard task the paper compared pairs of cells "
        "preferring the same side; pass them as pair_mask=.",
        "No per-session test.",
    ),
)

LIU_2023_ASSEMBLY = ReactivationMethod(
    name="liu_2023_assembly",
    reference=_LIU_2023_REFERENCE,
    measure="assembly",
    bin_size=0.025,
    control_windows=ReactivationWindows(
        length=3600.0, anchor="end", count=1, allow_shorter=True
    ),
    match_windows=ReactivationWindows(length=3600.0, count=1, allow_shorter=True),
    detection="ica",
    member_rule="positive_z",
    member_threshold=2.0,
    zscore_over="control_and_match",
    activation_threshold=5.0,
    event_response=EventResponse(),
    statistic="event_response_difference",
    test=None,
    notes=(
        "Detection: 'spike trains of each neuron were binned in 25-ms "
        "intervals for the whole session (including only task periods during "
        "movement) ... Marchenko-Pastur ... fast-ICA' (Materials and methods, "
        "Cell assembly analyses, p. 13). Pass the task's movement epochs as "
        "template_epochs.",
        "Members: 'Assembly members were thus considered those cells whose "
        "weight exceeded the mean weight of the assembly by two standard "
        "deviations' (signed weights).",
        "Windows: 'the last hour of cumulative baseline sleep and the first "
        "hour of cumulative post-task sleep, excluding periods outside of "
        "non-REM sleep' (fig. S9B, the reactivation-strength profiles). Pass "
        "NREM epochs; the hours are concatenated NREM time, or all of a "
        "shorter sleep period (allow_shorter=True). The lab's tutorial "
        "(neurocode pipelineReactivation.m, TruncateIntervals(preSleep, "
        "3600)) instead takes the first NREM hour of both periods, or all of "
        "a shorter one; dataclasses.replace(..., control_windows="
        "ReactivationWindows(length=3600.0, count=1, allow_shorter=True)) "
        "follows it.",
        "Response: 'Reactivation strength of task-related assemblies "
        "centered on post-task sleep SWR, increased relative to baseline' "
        "(Fig. 4D). The lab's pipeline (neurocode pipelineReactivation.m) "
        "takes 'the mean response to ripples as the strength between 0 and "
        "100ms after the ripple start', over ±0.5 s.",
        "R(t) sampling is not stated: here the 25 ms detection bins (the "
        "lab tutorial uses 20 ms bins in 1 ms steps; "
        "dataclasses.replace(..., bin_size=0.02, activation_step=0.001) is "
        "close to it, though detection then uses 20 ms bins too).",
        "Z-scoring: the lab's ReactivationStrength.m z-scores the bins it is "
        "given, pre- and post-task NREM together "
        "(zscore_over='control_and_match').",
        "Activations: 'Assembly activation events were detected as peaks in "
        "the assembly activation strength exceeding a value of 5'.",
        "No per-component test: 'Wilcoxon signed-rank test, n = 31 "
        "components' across components (Fig. 4D).",
    ),
)

_CHANG_2025_REFERENCE = (
    "Chang, H., Tang, W., Wulf, A. M., Nyasulu, T., Wolf, M. E., "
    "Fernandez-Ruiz, A., & Oliva, A. (2025). Sleep microstructure organizes "
    "memory replay. Nature, 637(8048), 1161-1169. "
    "https://doi.org/10.1038/s41586-024-08340-w"
)

CHANG_2025_ASSEMBLY = ReactivationMethod(
    name="chang_2025_assembly",
    reference=_CHANG_2025_REFERENCE,
    measure="assembly",
    bin_size=0.05,
    detection="ica",
    member_rule="positive_z",
    member_threshold=3.0,
    activation_step=0.01,
    zscore_over="control_and_match",
    event_response=EventResponse(baseline_window=(-0.5, -0.1)),
    statistic="event_response_ratio",
    test="event_modulation",
    alpha=0.05,
    notes=(
        "Detection: 'spike trains for each neuron were binned (50-ms bin) "
        "for the whole session. During the locomotor periods of the task (> "
        "2 cm/s), the matrix of firing correlation coefficients ... "
        "Marčenko-Pastur law ... ICA (fast-ICA algorithm)' (Methods, "
        "Reactivation strength and cell assembly analyses). Pass running "
        "epochs as template_epochs.",
        "Members: 'A given cell was considered a member of an assembly if "
        "its weight exceeded the mean weight of the assembly by 3 SDs'.",
        "R(t) sampling and z-scoring are not stated; the lab's code "
        "(pupil_project cal_ReactivationStrength.m) uses 50 ms bins in 10 ms "
        "steps over PRE and POST NREM, z-scored together "
        "(activation_step=0.01, zscore_over='control_and_match').",
        "Per-SWR strength: 'Reactivation strength (RS) for a given SWR was "
        "calculated as the mean of S during that event'; the classification "
        "uses 'the mean RS within a 100-ms window after SWR onset', as does "
        "the code, which is used here for every SWR.",
        "Statistic: 'we normalized RS during POST sleep by the mean RS in PRE "
        "sleep'; the code computes (POST − PRE) / PRE, NaN here unless the "
        "PRE response is positive.",
        "Test: 'cell assemblies with a significant increase in RS during POST "
        "SWRs (0 to 100 ms after SWR onset) compared to baseline (−500 to "
        "−100 ms relative to SWR onset; with p < 0.05, rank-sum paired "
        "test)'. Read as the two-sided Wilcoxon signed-rank test over POST "
        "SWRs; is_reactivated when p < 0.05 and the median difference is "
        "positive. The paper uses it to select assemblies, then calls them "
        "'rigid' or 'plastic' by whether POST exceeds PRE.",
        "Windows: whole PRE and POST sleep; pass their NREM epochs.",
    ),
)

_ROBINSON_2026_REFERENCE = (
    "Robinson, H. L., Todorova, R., Nagy, G. A., Gruzdeva, A., Paudel, P., "
    "Oliva, A., & Fernandez-Ruiz, A. (2026). Large sharp-wave ripples "
    "promote hippocampo-cortical memory reactivation and consolidation "
    "during sleep. Neuron, 114(2), 226-236.e6. "
    "https://doi.org/10.1016/j.neuron.2025.10.003"
)
_ROBINSON_2026_SOURCE = (
    "Read from the bioRxiv preprint v2 (10.1101/2025.06.27.662061, STAR "
    "Methods); the published version was not checked."
)

ROBINSON_2026_PAIRWISE = dataclasses.replace(
    KUDRIMOTI_1999,
    name="robinson_2026_pairwise",
    reference=_ROBINSON_2026_REFERENCE,
    bin_size=0.1,
    rest_bins="events",
    pairs="all",
    control_windows=ReactivationWindows(),
    match_windows=ReactivationWindows(),
    statistic="partial_correlation",
    test=None,
    notes=(
        _ROBINSON_2026_SOURCE,
        "'we binned the activity of all units in 100 ms during movement "
        "intervals of training of the object task ... Cbehavior' (STAR "
        "Methods, Pairwise reactivation). Pass the training's movement "
        "epochs as template_epochs.",
        "'we binned the activity of all units during SWRs in baseline sleep "
        "... Cbaseline ... during SWRs in post-task sleep ... Cpost'. The bin "
        "is not stated: one bin per SWR, as in the lab's CorrInIntervals. "
        + _SWR_EVENTS_NOTE,
        "'We used linear regression between Cbaseline and Cbehavior and "
        "between Cbaseline and Cpost to remove the portion of the variance "
        "that can be explained by preexisting patterns. The reactivation of "
        "pairwise behavioral patterns in post-task sleep was quantified as "
        "the correlation between the resultant residuals Rbehavior and "
        "Rpost.' The correlation of two residuals on the same regressor is "
        "the partial correlation, so this is Kudrimoti's partial "
        "correlation without the square (statistic='partial_correlation').",
        "Pairs: CA1, PFC or CA1-PFC pairs; dataclasses.replace(..., "
        "pairs='different_regions') with unit_groups={'region': ...} for the "
        "cross-region pairs of Fig. 4A.",
        "No per-session test: a hierarchical bootstrap (animals, sessions, "
        "pairs; 5,000 resamples) across sessions.",
    ),
)

ROBINSON_2026_ASSEMBLY = ReactivationMethod(
    name="robinson_2026_assembly",
    reference=_ROBINSON_2026_REFERENCE,
    measure="assembly",
    bin_size=0.05,
    detection="ica",
    member_rule="inverse_sqrt_n",
    member_threshold=None,
    detection_units="each_region_and_cross_region",
    activation_step=0.01,
    zscore_over="control_and_match",
    event_response=EventResponse(significant_z=1.96),
    statistic="event_response_difference",
    test=None,
    notes=(
        _ROBINSON_2026_SOURCE,
        "Detection: 'for each region (the hippocampus and the prefrontal "
        "cortex), spike trains of each neuron were binned in 50-ms intervals "
        "for the whole session ... Marčenko-Pastur law ... fast-ICA' (STAR "
        "Methods, Cell assembly analyses). Pass unit_groups={'region': ...} "
        "and the training's exploration epochs as template_epochs.",
        "Cross-structural assemblies: 'we combined the recordings in the "
        "hippocampus and the prefrontal cortex and performed assembly "
        "detection as described above. For a given assembly, the weight "
        "vector was normalized to have unit Euclidean norm, and units whose "
        "weights exceeded 1/√N, where N is the number of recorded units, "
        "were considered assembly members. Cell assemblies with members in "
        "both the hippocampus and the prefrontal cortex were retained'. The "
        "same rule names the members of the per-region patterns, with N the "
        "region's units (no rule is stated for them).",
        "Response: 'Assembly responses to SWRs were quantified as the mean "
        "reactivation strength within 0–100 ms of SWR onset.' Per-SWR "
        "significance: 'responses were z-scored within each assembly ... "
        "responses exceeding z = 1.96 (P < 0.05) were considered "
        "significant', read as z-scored over the assembly's SWRs of both "
        "sleep periods (significant_z=1.96).",
        "R(t) sampling and z-scoring are not stated: 50 ms bins in 10 ms "
        "steps over both sleep periods together, as in the lab's code for "
        "Chang et al. (2025).",
        "Not applied: 'We therefore discarded cell assemblies with "
        "significant reactivation in baseline sleep before object "
        "exploration' (no test is stated; control_significant_event_fraction "
        "and control_event_response describe baseline sleep).",
        "No per-pattern test: hierarchical bootstrap 'of assembly "
        "reactivation in post-task sleep relative to baseline sleep' (Fig. "
        "3J).",
    ),
)

PUBLISHED_REACTIVATION_METHODS: tuple[ReactivationMethod, ...] = (
    KUDRIMOTI_1999,
    PENNARTZ_2004,
    TATSUNO_2006,
    LANSINK_2009,
    PEYRACHE_2010,
    VAN_DE_VEN_2016,
    GIRI_2019,
    OLIVA_2020_EV,
    OLIVA_2020_ASSEMBLY,
    EL_GABY_2021,
    HARVEY_2023_ASSEMBLY,
    LIU_2023_EV,
    LIU_2023_PAIRWISE,
    LIU_2023_ASSEMBLY,
    GIRI_2024,
    CHANG_2025_ASSEMBLY,
    ROBINSON_2026_PAIRWISE,
    ROBINSON_2026_ASSEMBLY,
)
"""Every published reactivation recipe in this module, by year."""
