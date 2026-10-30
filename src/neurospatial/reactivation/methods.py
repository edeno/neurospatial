# The recipes' notes quote the papers verbatim, with their symbols.
# ruff: noqa: RUF001
"""Published reactivation methods as explicit, comparable recipes.

Each :class:`ReactivationMethod` records one paper's reactivation analysis:
the time bin, which cell pairs count, the windows of the template, control
and match periods (:class:`ReactivationWindows`), the statistic and the
within-session test. :func:`compare_reactivation_methods` bins one
session's spike times for each method and applies it, so the same
recording can be analyzed the way each lab analyzed it.

Two families are covered:

- Explained variance (EV): the pair-correlation pattern of a template
  period (running) predicts that of a match period (rest after running)
  beyond what a control period (rest before) already predicts (Kudrimoti
  et al. 1999), and the reversed EV (REV) that swaps the rest periods
  (Pennartz et al. 2004).
- Assembly patterns: patterns detected in the template period, and their
  activation strength R(t) in rest after minus rest before (Peyrache et
  al. 2010; van de Ven et al. 2016).

Unlike the replay recipes, which record their decoding settings for
reference only, a reactivation recipe applies its ``bin_size``: binning is
part of the statistic (a correlation of 100 ms counts is not one of 250 ms
counts). Choices a paper does not specify are listed in each recipe's
``notes``.
"""

from __future__ import annotations

import contextlib
import warnings
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import as_intervals
from neurospatial._results import resolve_unit_ids
from neurospatial._spikes import as_spike_trains
from neurospatial.reactivation.assemblies import (
    _activation_strength,
    _z_scores,
    detect_assemblies,
    explained_variance_reactivation,
)
from neurospatial.reactivation.calibration import equalize_firing_rates

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "GIRI_2019",
    "GIRI_2024",
    "KUDRIMOTI_1999",
    "LANSINK_2009",
    "PENNARTZ_2004",
    "PEYRACHE_2010",
    "PUBLISHED_REACTIVATION_METHODS",
    "TATSUNO_2006",
    "VAN_DE_VEN_2016",
    "ReactivationMethod",
    "ReactivationWindows",
    "compare_reactivation_methods",
]

_CALL = "compare_reactivation_methods()"
_MEASURES = ("explained_variance", "assembly")
_PAIRS = ("all", "different_tetrodes", "different_shanks", "different_regions")
# The unit_groups key each pair rule reads.
_GROUP_OF_PAIRS = {
    "different_tetrodes": "tetrode",
    "different_shanks": "shank",
    "different_regions": "region",
}
_STATISTICS = {
    "explained_variance": ("explained_variance", "ev_minus_rev"),
    "assembly": ("mean_activation_difference",),
}
_TESTS = {
    "explained_variance": (None, "partial_correlation_t", "pair_bootstrap"),
    "assembly": (None, "cell_identity_shuffle"),
}
_DETECTIONS = ("pca", "ica")


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
    """

    length: float | None = None
    step: float | None = None
    anchor: Literal["start", "end"] = "start"
    count: int | None = None
    clock: Literal["concatenated", "wall"] = "concatenated"

    def __post_init__(self) -> None:
        """Check the window options."""
        for name in ("length", "step"):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"ReactivationWindows: {name} must be a positive number of "
                    f"seconds or None, got {value!r}."
                )
        if self.length is None and (self.step is not None or self.count is not None):
            raise ValueError(
                _format_error(
                    "ReactivationWindows: step= and count= need length=.",
                    why="Why: without a length the period is one window.",
                    fix="pass length= in seconds, or drop step= and count=",
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
        if self.count is not None and (
            isinstance(self.count, bool)
            or int(self.count) != self.count
            or self.count < 1
        ):
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
            out. Empty when the epochs are shorter than one window.
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
            n_windows = 0
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
class ReactivationMethod:
    """A published reactivation analysis: bins, pairs, windows and test.

    Attributes
    ----------
    name : str
        Short name, for example ``"kudrimoti_1999"``.
    reference : str
        The paper the method follows.
    measure : {"explained_variance", "assembly"}
        ``"explained_variance"`` compares pair correlations between periods
        (:func:`~neurospatial.reactivation.explained_variance_reactivation`);
        ``"assembly"`` detects patterns in the template period
        (:func:`~neurospatial.reactivation.detect_assemblies`) and compares
        their activation strength R(t) between the control and match
        periods.
    bin_size : float
        Time bin in seconds. Applied: spike times are binned at this width
        inside each window.
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
    detection : {"pca", "ica"} or None
        Assembly detection on the template counts, with the
        Marchenko-Pastur threshold; None for EV methods.
    smoothing_bandwidth : float or None
        For assembly methods, the standard deviation in seconds of a
        Gaussian kernel applied to the spike trains before z-scoring and
        computing R(t) in the control and match periods (van de Ven et al.
        2016); None uses counts in ``bin_size`` bins.
    activation_threshold : float or None
        For assembly methods, R(t) peaks above this value are activations;
        their rate in each period is reported. None reports none.
    statistic : {"explained_variance", "ev_minus_rev", "mean_activation_difference"}
        The method's headline value, the ``value`` column of the table.
    test : {"partial_correlation_t", "pair_bootstrap", "cell_identity_shuffle"} or None
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
        many times; EV, REV, the partial correlation and its p-value are the
        medians over the subsamples ("median Pearson's r … median p … for
        100 different random subsamples"), and ``value`` is computed from
        them. None (default) uses every spike. Not combined with
        ``test="pair_bootstrap"``.
    notes : tuple of str
        Each value's source in the paper and the choices it leaves open.

    Examples
    --------
    >>> import dataclasses
    >>> from neurospatial.reactivation import KUDRIMOTI_1999
    >>> KUDRIMOTI_1999.bin_size, KUDRIMOTI_1999.pairs
    (0.1, 'different_tetrodes')
    >>> wider = dataclasses.replace(
    ...     KUDRIMOTI_1999, name="kudrimoti_200ms", bin_size=0.2
    ... )
    """

    name: str
    reference: str
    measure: Literal["explained_variance", "assembly"]
    bin_size: float
    pairs: Literal[
        "all", "different_tetrodes", "different_shanks", "different_regions"
    ] = "all"
    control_windows: ReactivationWindows = ReactivationWindows()
    match_windows: ReactivationWindows = ReactivationWindows()
    detection: Literal["pca", "ica"] | None = None
    smoothing_bandwidth: float | None = None
    activation_threshold: float | None = None
    statistic: Literal[
        "explained_variance", "ev_minus_rev", "mean_activation_difference"
    ] = "explained_variance"
    test: (
        Literal["partial_correlation_t", "pair_bootstrap", "cell_identity_shuffle"]
        | None
    ) = None
    percentile: float | None = None
    n_shuffles: int = 1000
    alpha: float = 0.05
    n_rate_subsamples: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check that the options fit the measure."""
        where = f"ReactivationMethod {self.name!r}"
        if self.measure not in _MEASURES:
            raise ValueError(
                f"{where}: measure must be one of {_MEASURES}, got {self.measure!r}."
            )
        if not (np.isfinite(self.bin_size) and self.bin_size > 0):
            raise ValueError(
                f"{where}: bin_size must be a positive number of seconds, got "
                f"{self.bin_size!r}."
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
        assembly = self.measure == "assembly"
        if assembly and self.detection not in _DETECTIONS:
            raise ValueError(
                f"{where}: an assembly method needs detection='pca' or 'ica', "
                f"got {self.detection!r}."
            )
        if not assembly:
            given = [
                name
                for name in (
                    "detection",
                    "smoothing_bandwidth",
                    "activation_threshold",
                    "percentile",
                )
                if getattr(self, name) is not None
            ]
            if given:
                raise ValueError(
                    _format_error(
                        f"{where}: {given} apply only to assembly methods.",
                        why="Why: an EV method compares pair correlations and "
                        "would ignore them.",
                        fix="drop them, or use measure='assembly'",
                    )
                )
        elif self.pairs != "all":
            raise ValueError(
                f"{where}: pairs={self.pairs!r} applies only to EV methods; "
                "an assembly method uses every unit."
            )
        if self.smoothing_bandwidth is not None and not self.smoothing_bandwidth > 0:
            raise ValueError(
                f"{where}: smoothing_bandwidth must be positive seconds or None, "
                f"got {self.smoothing_bandwidth!r}."
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
        if (
            isinstance(self.n_shuffles, bool)
            or int(self.n_shuffles) != self.n_shuffles
            or self.n_shuffles < 1
        ):
            raise ValueError(
                f"{where}: n_shuffles must be a positive integer, got "
                f"{self.n_shuffles!r}."
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"{where}: alpha must be in (0, 1), got {self.alpha!r}.")
        subsamples = self.n_rate_subsamples
        if subsamples is not None:
            if (
                isinstance(subsamples, bool)
                or int(subsamples) != subsamples
                or subsamples < 1
            ):
                raise ValueError(
                    f"{where}: n_rate_subsamples must be a positive integer or "
                    f"None, got {subsamples!r}."
                )
            if assembly:
                raise ValueError(
                    _format_error(
                        f"{where}: n_rate_subsamples applies only to EV methods.",
                        why="Why: Giri et al. (2019) equalized rates for the "
                        "pair correlations; R(t) z-scores each period with its "
                        "own mean and standard deviation.",
                        fix="drop n_rate_subsamples, or use measure="
                        "'explained_variance'",
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
    "statistic",
    "value",
    "explained_variance",
    "reversed_explained_variance",
    "partial_correlation",
    "ev_minus_rev_lower",
    "n_pairs",
    "n_control_windows",
    "n_components",
    "member_unit_ids",
    "control_mean_activation",
    "match_mean_activation",
    "fraction_above_control_threshold",
    "control_fraction_above_shuffle",
    "match_fraction_above_shuffle",
    "control_activation_rate",
    "match_activation_rate",
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
    "n_components",
    "n_shuffles_used",
)


def compare_reactivation_methods(
    spike_times: Any,
    template_epochs: Any,
    match_epochs: Any,
    control_epochs: Any = None,
    *,
    reactivation_methods: Sequence[ReactivationMethod],
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    unit_groups: Mapping[str, NDArray[Any] | Sequence[Any]] | None = None,
    pair_mask: NDArray[np.bool_] | Sequence[bool] | None = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Apply several published reactivation methods to one session.

    Parameters
    ----------
    spike_times : sequence of arrays or spike group
        One spike train per unit, in seconds.
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
    unit_ids : array-like, shape (n_units,), optional
        Label of each unit, for ``member_unit_ids``; default
        ``np.arange(n_units)``.
    unit_groups : mapping of str to array-like, optional
        Each unit's group, by kind: ``"tetrode"``, ``"shank"`` or
        ``"region"``, one entry per unit, as methods with a ``pairs`` rule
        need (for example ``{"tetrode": tetrode_of_unit}``).
    pair_mask : array-like of bool, shape (n_units * (n_units - 1) / 2,), optional
        Pairs every EV method may use, in ``np.triu_indices(n_units, k=1)``
        order, for example waveform similarity below 0.8 (Giri et al. 2024).
        Combined with each method's ``pairs`` rule.
    rng : numpy.random.Generator, int, or None
        Seed or generator for ICA, bootstraps and shuffles. Each method,
        match window and component draws from its own stream, derived from
        ``rng`` and the method's name, so adding, removing or reordering
        methods does not change the other methods' results.

    Returns
    -------
    pandas.DataFrame
        One row per (method, match window, component): EV methods have one
        row per match window with ``component`` missing; assembly methods
        one per detected component (one row with ``n_components`` 0 and
        ``component`` missing when none is detected). Columns:

        - ``method``, ``window`` (match window number), ``window_start``,
          ``window_stop`` (the window's first and last epoch time) and
          ``window_duration`` (seconds of epoch time inside it).
        - ``component``, ``n_components`` and ``member_unit_ids`` (the
          pattern's core members) for assembly methods.
        - ``statistic`` (its name) and ``value``.
        - EV methods: ``explained_variance``, ``reversed_explained_variance``
          (averaged over control windows), ``partial_correlation`` (one
          control window), ``n_pairs`` (fewest over control windows),
          ``n_control_windows`` and ``ev_minus_rev_lower`` (the bootstrap's
          ``100 * alpha / 2`` percentile).
        - Assembly methods: ``control_mean_activation`` and
          ``match_mean_activation`` (mean R(t)),
          ``fraction_above_control_threshold``,
          ``control_fraction_above_shuffle``,
          ``match_fraction_above_shuffle``, ``control_activation_rate`` and
          ``match_activation_rate`` (activations per second).
        - ``p_value``, ``n_shuffles_used`` and ``is_reactivated`` (the
          method's within-session verdict; missing for methods without
          one), and ``error``: why a method could not be applied (for
          example a period shorter than one window), with the other
          columns missing.

        Columns that do not apply to a method are NaN. ``table.attrs``
        holds ``rng_entropy``, the seed the streams derive from: passing it
        as ``rng`` reproduces the table.

    Raises
    ------
    ValueError
        If the methods' names repeat, a method needs ``control_epochs`` or
        a ``unit_groups`` kind that is not given, a group or ``pair_mask``
        does not have one entry per unit or pair, or ``unit_ids`` does not
        hold one distinct label per unit.

    Warns
    -----
    UserWarning
        Warnings of the underlying functions (constant units, undefined EV,
        no pattern above the threshold), prefixed with the method's name.

    Notes
    -----
    Each method bins spike times at its own ``bin_size`` inside each of its
    windows (:func:`~neurospatial.decoding.bin_spikes_in_time`), so bins
    never span two epochs. The template period is always one window of all
    ``template_epochs``.

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
    trains = as_spike_trains(spike_times)
    n_units = len(trains)
    labels = resolve_unit_ids(unit_ids, n_units, context=_CALL)
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

    entropy = (
        int(rng.integers(2**63))
        if isinstance(rng, np.random.Generator)
        else np.random.SeedSequence(rng).entropy
    )

    rows: list[dict[str, Any]] = []
    for method in methods:
        key = zlib.crc32(method.name.encode())

        def stream(*indices: int, key: int = key) -> np.random.Generator:
            return np.random.default_rng(
                np.random.SeedSequence(entropy, spawn_key=(key, *indices))
            )

        with _warnings_with_context(f"{_CALL}: method {method.name!r}"):
            assert control is not None  # every method needs one, checked above
            try:
                if method.measure == "explained_variance":
                    pair_rule = selected.copy()
                    if method.pairs != "all":
                        unit_group = groups[_GROUP_OF_PAIRS[method.pairs]]
                        first, second = np.triu_indices(n_units, k=1)
                        pair_rule &= unit_group[first] != unit_group[second]
                    method_rows = _explained_variance_rows(
                        method, trains, template, control, match, pair_rule, stream
                    )
                else:
                    method_rows = _assembly_rows(
                        method, trains, template, control, match, labels, stream
                    )
            except _MethodError as failure:
                method_rows = [{"error": str(failure)}]
        rows.extend({"method": method.name, **row} for row in method_rows)
    table = pd.DataFrame(rows, columns=list(_COLUMNS))
    numeric = [
        name
        for name in _COLUMNS
        if name not in _COUNT_COLUMNS
        and name
        not in ("method", "statistic", "member_unit_ids", "is_reactivated", "error")
    ]
    table[numeric] = table[numeric].astype(np.float64)
    for name in _COUNT_COLUMNS:
        table[name] = table[name].astype("Int64")
    table["is_reactivated"] = table["is_reactivated"].astype("boolean")
    table["error"] = table["error"].fillna("").astype(str)
    table.attrs["rng_entropy"] = entropy
    return table


class _MethodError(ValueError):
    """A method cannot be applied to this session; reported in ``error``."""


@contextlib.contextmanager
def _warnings_with_context(context: str) -> Iterator[None]:
    """Re-emit the warnings raised in the block with ``context`` in front."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for caught_warning in caught:
        warnings.warn(
            f"{context}: {caught_warning.message}",
            caught_warning.category,
            stacklevel=4,
        )


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
        The methods, whose ``pairs`` rules say which kinds are needed.
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
    missing = {kind: name for kind, name in needed.items() if kind not in given}
    if missing:
        raise ValueError(
            _format_error(
                f"{_CALL}: methods {sorted(missing.values())} use pairs of units in "
                f"different groups but unit_groups has no {sorted(missing)}.",
                why="Why: the papers used only pairs of cells on different "
                "tetrodes or shanks (cells on one tetrode can share sorting "
                "errors) or in different regions.",
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
    from neurospatial.decoding import bin_spikes_in_time

    try:
        counts, _centers = bin_spikes_in_time(trains, dt=bin_size, epochs=windows)
    except ValueError as error:
        raise _MethodError(str(error)) from error
    return counts.astype(np.float64)


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


def _explained_variance_rows(
    method: ReactivationMethod,
    trains: list[NDArray[np.float64]],
    template: NDArray[np.float64],
    control: NDArray[np.float64],
    match: NDArray[np.float64],
    pair_rule: NDArray[np.bool_],
    stream: Any,
) -> list[dict[str, Any]]:
    """One row per match window of an EV method."""
    template_correlations = _pair_correlations(
        _counts(trains, template, method.bin_size)
    )
    control_counts = [
        _counts(trains, window, method.bin_size)
        for window in _windows_or_error(method.control_windows, control, "control")
    ]
    control_correlations = [_pair_correlations(counts) for counts in control_counts]
    n_controls = len(control_correlations)
    if method.test is not None and n_controls != 1:
        raise _MethodError(
            f"the {method.test} test needs one control window, got {n_controls}"
        )
    rows = []
    for index, window in enumerate(
        _windows_or_error(method.match_windows, match, "match")
    ):
        match_counts = _counts(trains, window, method.bin_size)
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
                stream(index, 1),
            )
        explained = summary["explained_variance"]
        reversed_ = summary["reversed_explained_variance"]
        row: dict[str, Any] = {
            **_window_columns(index, window),
            "statistic": method.statistic,
            "value": explained
            if method.statistic == "explained_variance"
            else explained - reversed_,
            "explained_variance": explained,
            "reversed_explained_variance": reversed_,
            "partial_correlation": summary["partial_correlation"],
            "n_pairs": summary["n_pairs"],
            "n_control_windows": n_controls,
        }
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
        rows.append(row)
    return rows


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
        over control windows), ``partial_correlation`` and ``p_value`` (NaN
        unless there is one control window) and ``n_pairs`` (fewest over
        control windows).
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
        "p_value": results[0].p_value if single else np.nan,
        "n_pairs": min(result.n_pairs for result in results),
    }


def _rate_equalized_ev_summary(
    template_correlations: NDArray[np.float64],
    match_counts: NDArray[np.float64],
    control_counts: list[NDArray[np.float64]],
    pair_rule: NDArray[np.bool_],
    n_subsamples: int,
    generator: np.random.Generator,
) -> dict[str, float]:
    """:func:`_ev_summary` medians over rate-equalizing subsamples.

    In each subsample, every (control window, match window) pair gets its
    own draw of :func:`equalize_firing_rates`, so each unit fires at one
    rate in both windows of the pair; the template is not subsampled (Giri
    et al. 2019).

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
    generator : numpy.random.Generator
        Source of the subsamples.

    Returns
    -------
    dict
        The keys of :func:`_ev_summary`: medians over subsamples of the
        statistics, NaN where no subsample defines one; ``n_pairs`` is the
        fewest over subsamples.
    """
    draws = []
    for _ in range(n_subsamples):
        periods = []
        for counts in control_counts:
            control, match = equalize_firing_rates(counts, match_counts, rng=generator)
            periods.append((_pair_correlations(match), _pair_correlations(control)))
        draws.append(_ev_summary(template_correlations, periods, pair_rule))
    summary = {
        key: _nanmedian([draw[key] for draw in draws])
        for key in (
            "explained_variance",
            "reversed_explained_variance",
            "partial_correlation",
            "p_value",
        )
    }
    summary["n_pairs"] = min(draw["n_pairs"] for draw in draws)
    return summary


def _nanmedian(values: Sequence[float]) -> float:
    """Median over the defined values, NaN when none is (no warning)."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else np.nan


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
    # How often each pair is drawn in each resample, shape (n_resamples, n_pairs).
    weights = (
        generator.multinomial(n_pairs, np.full(n_pairs, 1 / n_pairs), n_resamples)
        / n_pairs
    )
    vectors = {"t": template, "m": match, "c": control}
    means = {name: weights @ values for name, values in vectors.items()}

    def correlation(a: str, b: str) -> NDArray[np.float64]:
        covariance = weights @ (vectors[a] * vectors[b]) - means[a] * means[b]
        variance_a = weights @ vectors[a] ** 2 - means[a] ** 2
        variance_b = weights @ vectors[b] ** 2 - means[b] ** 2
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


def _assembly_rows(
    method: ReactivationMethod,
    trains: list[NDArray[np.float64]],
    template: NDArray[np.float64],
    control: NDArray[np.float64],
    match: NDArray[np.float64],
    labels: NDArray[Any],
    stream: Any,
) -> list[dict[str, Any]]:
    """One row per (match window, component) of an assembly method."""
    template_counts = _counts(trains, template, method.bin_size)
    try:
        detection = detect_assemblies(
            template_counts,
            method=method.detection,  # type: ignore[arg-type]
            rng=int(stream().integers(2**32 - 1)),
            unit_ids=labels,
        )
    except ValueError as error:
        raise _MethodError(str(error)) from error
    control_windows = _windows_or_error(method.control_windows, control, "control")
    if len(control_windows) != 1:
        raise _MethodError(
            f"an assembly method compares one control window, got {len(control_windows)}"
        )
    control_z, control_constant, control_segments = _period_z(
        method, trains, control_windows[0]
    )
    n_components = len(detection.patterns)
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(
        _windows_or_error(method.match_windows, match, "match")
    ):
        window_columns = {
            **_window_columns(index, window),
            "statistic": method.statistic,
            "n_components": n_components,
        }
        if not n_components:
            rows.append(window_columns)
            continue
        match_z, match_constant, match_segments = _period_z(method, trains, window)
        # A unit constant in either period has no z-score there; leave it out
        # of both, as assembly_reactivation() does, and rescale the weights.
        included = ~(control_constant | match_constant)
        weights = np.array([pattern.weights for pattern in detection.patterns])
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
        for component, pattern in enumerate(detection.patterns):
            row = {
                **window_columns,
                "component": component,
                "member_unit_ids": tuple(pattern.member_unit_ids.tolist()),
            }
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
            if method.percentile is not None:
                threshold = np.percentile(control_strength, method.percentile)
                row["fraction_above_control_threshold"] = float(
                    np.mean(match_strength > threshold)
                )
            if method.activation_threshold is not None:
                row["control_activation_rate"] = _activation_rate(
                    control_strength, control_segments, method.activation_threshold
                )
                row["match_activation_rate"] = _activation_rate(
                    match_strength, match_segments, method.activation_threshold
                )
            if method.test == "cell_identity_shuffle":
                row["control_fraction_above_shuffle"] = control_shares[component]
                row["match_fraction_above_shuffle"] = match_shares[component]
                row["n_shuffles_used"] = method.n_shuffles
            rows.append(row)
    return rows


def _period_z(
    method: ReactivationMethod,
    trains: list[NDArray[np.float64]],
    window: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], list[tuple[int, int, float]]]:
    """Z-scored activity of one period for R(t).

    Parameters
    ----------
    method : ReactivationMethod
        Its ``bin_size`` and ``smoothing_bandwidth``.
    trains : list of ndarray
        Spike trains.
    window : ndarray, shape (n_pieces, 2)
        The period's epochs.

    Returns
    -------
    z : ndarray, shape (n_samples, n_units)
        Counts in ``bin_size`` bins, or, with a smoothing bandwidth, spike
        trains convolved with a Gaussian of that standard deviation and
        sampled every half bandwidth (each epoch piece on its own), z-scored
        over the period; 0 for constant units.
    constant : ndarray of bool, shape (n_units,)
        Units with the same value in every sample.
    segments : list of (first, stop, sample_duration)
        Sample ranges of each epoch piece and the seconds per sample.
    """
    from scipy.ndimage import gaussian_filter1d

    smoothing = method.smoothing_bandwidth
    step = method.bin_size if smoothing is None else smoothing / 2
    # Each epoch piece on its own, so smoothing does not cross a gap; a piece
    # shorter than one sample holds no bin.
    pieces = [
        _counts(trains, piece[None, :], step)
        for piece in window
        if piece[1] - piece[0] >= step
    ]
    if not pieces:
        raise _MethodError(f"no epoch of the window holds a whole {step:g} s bin")
    if smoothing is not None:
        # Kernel standard deviation of 2 samples of smoothing / 2 seconds.
        pieces = [
            gaussian_filter1d(piece, sigma=2.0, axis=0, mode="constant")
            for piece in pieces
        ]
    stops = np.cumsum([len(piece) for piece in pieces])
    segments = [
        (int(stop - len(piece)), int(stop), step)
        for stop, piece in zip(stops, pieces, strict=True)
    ]
    z, constant = _z_scores(np.vstack(pieces))
    return z, constant, segments


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
        "counts are the same, and reports medians over the subsamples. "
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

PUBLISHED_REACTIVATION_METHODS: tuple[ReactivationMethod, ...] = (
    KUDRIMOTI_1999,
    PENNARTZ_2004,
    TATSUNO_2006,
    LANSINK_2009,
    PEYRACHE_2010,
    VAN_DE_VEN_2016,
    GIRI_2019,
    GIRI_2024,
)
"""Every published reactivation recipe in this module, by year."""
