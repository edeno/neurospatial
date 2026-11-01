"""Firing rate as a function of one behavioral variable: tuning curves.

A tuning curve bins a variable sampled with the trajectory, such as progress
along the route to a goal, distance to the goal, running speed or a task
timer, and divides each bin's spike count by the time spent in it. Head
direction, a circular variable, has its own family
(:func:`~neurospatial.encoding.compute_directional_rate`); positions have
rate maps (:func:`~neurospatial.encoding.compute_spatial_rate`).

Result Classes
--------------
TuningCurveResult
    One unit's tuning curve, with occupancy, spike counts and bin edges.
TuningCurvesResult
    Tuning curves of a population sharing occupancy and bin edges.

Compute Functions
-----------------
compute_tuning_curve
    Tuning curve of one unit.
compute_tuning_curves
    Tuning curves of several units.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    _is_pynapple_object,
    interval_gates,
    resolve_time_windows,
    times_in_intervals,
    validate_speed_gate,
    value_frame_bins,
)
from neurospatial._observation import validate_break_before
from neurospatial._pynapple import interval_rows, reads_pynapple_samples
from neurospatial._results import _copy_analysis_settings
from neurospatial._spikes import spike_unit_table
from neurospatial._validation import (
    refuse_infinite,
    validate_finite_scalar,
    validate_instance,
)
from neurospatial.encoding._base import (
    BandwidthRule,
    PopulationRateMixin,
    SingleUnitRateMixin,
    _to_numpy,
)
from neurospatial.encoding._smoothing import kernel_ratio_rates, validate_min_occupancy
from neurospatial.encoding._spatial_counts import (
    _warn_if_population_silent,
    prepare_frame_binning,
    tracked_samples,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial._typing import SpikeTrainsLike

__all__ = [
    "TuningCurveResult",
    "TuningCurvesResult",
    "compute_tuning_curve",
    "compute_tuning_curves",
]

#: Estimators of a tuning curve and what each requires of ``bandwidth``.
TUNING_ESTIMATORS: Mapping[str, BandwidthRule] = MappingProxyType(
    {"binned": "none", "gaussian": "positive"}
)

#: Relative tolerance for ``value_range`` being tiled by ``bin_size`` and for
#: bins sharing one width.
_TILING_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------


def _checked_edges(bin_edges: Any, *, owner: str) -> NDArray[np.float64]:
    """``bin_edges`` as float64 after checking 1-D, >= 2, finite, increasing."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            _format_error(
                f"{owner}: bin_edges must be a 1-D array of at least two finite, "
                f"strictly increasing values; got shape {edges.shape}.",
                why="Why: bin k holds values from bin_edges[k] up to bin_edges[k + 1].",
                fix="pass sorted edges, e.g. np.linspace(0, 1, 21) for 20 bins "
                "of progress.",
            )
        )
    return edges


def tuning_bin_edges(
    values: NDArray[np.float64],
    *,
    bin_size: Any,
    bin_edges: Any,
    value_range: Any,
    owner: str,
) -> tuple[NDArray[np.float64], str]:
    """Bin edges of a tuning curve, and where they came from.

    Parameters
    ----------
    values : ndarray, shape (n_analyzed,)
        The variable at the samples inside the analysis windows; used only
        when neither ``bin_edges`` nor ``value_range`` fixes the range.
    bin_size, bin_edges, value_range
        As documented on :func:`compute_tuning_curves`.
    owner : str
        Calling function, named in error messages.

    Returns
    -------
    edges : ndarray, shape (n_bins + 1,)
    source : {"bin_edges", "value_range", "data"}

    Raises
    ------
    TypeError
        If not exactly one of ``bin_size`` and ``bin_edges`` is given, or if
        ``value_range`` is given with ``bin_edges``.
    ValueError
        If the edges are invalid, ``bin_size`` does not tile ``value_range``,
        or no finite value is analyzed to set the range.
    """
    if (bin_size is None) == (bin_edges is None):
        msg = (
            f"{owner}: give exactly one of bin_size= or bin_edges=. A variable "
            "has no default bin width; pick one in its units, or pass the edges "
            "so several conditions share them."
        )
        raise TypeError(msg)
    if bin_edges is not None:
        if value_range is not None:
            msg = (
                f"{owner}: value_range= is fixed by bin_edges= (from "
                "bin_edges[0] to bin_edges[-1]); pass one or the other."
            )
            raise TypeError(msg)
        return _checked_edges(bin_edges, owner=owner), "bin_edges"
    size = validate_finite_scalar(bin_size, name="bin_size", positive=True)
    assert size is not None
    if value_range is not None:
        lo, hi = _checked_range(value_range, owner=owner)
        ratio = (hi - lo) / size
        n = round(ratio)
        if n < 1 or abs(ratio - n) > _TILING_RTOL * max(1.0, ratio):
            options = sorted(
                {
                    (hi - lo) / k
                    for k in (int(np.floor(ratio)), int(np.ceil(ratio)))
                    if k
                }
            )
            named = " or ".join(f"{option:.10g}" for option in options)
            raise ValueError(
                _format_error(
                    f"{owner}: bin_size={size:g} does not tile value_range=({lo:g}, "
                    f"{hi:g}): it gives {ratio:.6g} bins.",
                    why="Why: value_range is the range analyzed; values outside it "
                    "are excluded, so the bins must end exactly at its limits.",
                    fix=f"pass bin_size={named}, or bin_edges= for bins of your "
                    "choosing.",
                )
            )
        # linspace sets both ends exactly, so a value at hi is in the last bin.
        return np.linspace(lo, hi, n + 1), "value_range"
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(
            _format_error(
                f"{owner}: no finite value is analyzed, so bin_size= cannot set "
                "the range of the bins.",
                why="Why: without value_range, the bins cover the values in the "
                "analyzed time.",
                fix="pass value_range=(lo, hi) or bin_edges=, and use the same "
                "edges for every condition you compare.",
            )
        )
    lo, hi = float(finite.min()), float(finite.max())
    k0, k1 = int(np.floor(lo / size)), int(np.ceil(hi / size))
    # Rounding in k * size can leave the extremes out; widen until covered.
    while k0 * size > lo:
        k0 -= 1
    while k1 * size < hi:
        k1 += 1
    k1 = max(k1, k0 + 1)
    return size * np.arange(k0, k1 + 1, dtype=np.float64), "data"


def _checked_range(value_range: Any, *, owner: str) -> tuple[float, float]:
    """``value_range`` as two finite floats with ``hi > lo``."""
    try:
        lo, hi = (float(v) for v in value_range)
    except (TypeError, ValueError):
        lo = hi = float("nan")
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError(
            _format_error(
                f"{owner}: value_range={value_range!r} is not (lo, hi) with "
                "finite lo < hi.",
                why="Why: it is the range of the variable that is analyzed.",
                fix="pass value_range=(lo, hi) in the variable's units, e.g. "
                "(0, 1) for progress.",
            )
        )
    return lo, hi


def _axis_label(variable: str, units: str | None) -> str:
    return variable if not units else f"{variable} ({units})"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class _TuningCurveMixin:
    """Bins, exports and checks shared by the tuning-curve results.

    Hosts store ``bin_edges`` in the variable's units, its name in
    ``variable`` and its units in ``units``, and inherit
    :class:`~neurospatial.encoding._base.SingleUnitRateMixin` or
    :class:`~neurospatial.encoding._base.PopulationRateMixin` after this class.
    """

    bin_edges: ArrayLike
    bandwidth: float | None
    variable: str
    units: str | None
    spike_counts: ArrayLike | None
    n_samples_outside_range: int | None
    _ESTIMATORS: ClassVar[Mapping[str, BandwidthRule]] = TUNING_ESTIMATORS

    if TYPE_CHECKING:
        # Provided by the rate-result base the host inherits after this class.
        def _get_rates(self) -> Any: ...
        def peak_location(self) -> NDArray[np.float64]: ...
        def peak_firing_rate(self) -> NDArray[np.float64] | float: ...
        def _accounting_columns(self) -> dict[str, Any]: ...
        def _analysis_attrs(self) -> dict[str, Any]: ...
        def _check_recorded_settings(self, fields: Mapping[str, Any]) -> None: ...

    @property
    def method(self) -> str:
        """Estimator: ``"gaussian"`` when smoothed, else ``"binned"``."""
        return "binned" if self.bandwidth is None else "gaussian"

    def _n_bins(self) -> int:
        owner = type(self).__name__
        edges = _checked_edges(self.bin_edges, owner=owner)
        object.__setattr__(self, "bin_edges", edges)
        return len(edges) - 1

    def _check_family(self) -> None:
        owner = type(self).__name__
        validate_instance(self.variable, str, name="variable", owner=owner)
        if self.units is not None:
            validate_instance(self.units, str, name="units", owner=owner)
        if self.spike_counts is not None:
            counts = np.asarray(self.spike_counts, dtype=np.float64)
            if counts.shape != np.shape(self._get_rates()):
                raise ValueError(
                    _format_error(
                        f"{owner}: spike_counts has shape {counts.shape}, not the "
                        f"shape of the rates {np.shape(self._get_rates())}.",
                        why="Why: each count belongs to one bin of one unit.",
                        fix="pass the counts of the same computation, or None.",
                    )
                )
            object.__setattr__(self, "spike_counts", counts)
        n_outside = self.n_samples_outside_range
        if n_outside is not None:
            if isinstance(n_outside, bool) or not isinstance(
                n_outside, int | np.integer
            ):
                msg = (
                    f"{owner}: n_samples_outside_range must be an int >= 0 or "
                    f"None, got {n_outside!r}."
                )
                raise TypeError(msg)
            if n_outside < 0:
                msg = f"{owner}: n_samples_outside_range must be >= 0, got {n_outside}."
                raise ValueError(msg)
            object.__setattr__(self, "n_samples_outside_range", int(n_outside))
        self._check_recorded_settings(
            {
                "variable": self.variable,
                "units": self.units,
                "bin_edges": self.bin_edges,
                "bandwidth": self.bandwidth,
                "method": self.method,
                "n_samples_outside_range": self.n_samples_outside_range,
            }
        )

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        # Override SpatialResultMixin: the bins are values of the variable,
        # not places in an Environment.
        edges = np.asarray(self.bin_edges, dtype=np.float64)
        centers: NDArray[np.float64] = (edges[:-1] + edges[1:]) / 2
        return centers

    @property
    def bin_centers(self) -> NDArray[np.float64]:
        """Center of each bin in the variable's units, shape (n_bins,)."""
        return self._bin_centers

    def _bin_center_columns(self) -> dict[str, NDArray[np.float64]]:
        return {"bin_center": self._bin_centers}

    def _xarray_attrs(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "units": self.units or "",
            "bin_edges": np.asarray(self.bin_edges).tolist(),
        }

    def _tuning_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Name the variable and the bin centers' units on a dense frame."""
        frame.attrs["units"]["bin_center"] = self.units or ""
        frame.attrs["variable"] = self.variable
        return frame

    def _tuning_summary_frame(self, index: Sequence[Any]) -> pd.DataFrame:
        """Per-unit peak and spike accounting, one row per unit."""
        import pandas as pd

        rates = np.atleast_2d(np.asarray(_to_numpy(self._get_rates())))
        peaks = np.asarray(self.peak_location()).reshape(len(rates))
        frame = pd.DataFrame(
            {
                "peak_firing_rate": np.atleast_1d(self.peak_firing_rate()),
                "peak_location": peaks,
                **self._accounting_columns(),
            },
            index=pd.Index(list(index), name="unit_id"),
        )
        frame.attrs["units"] = {
            "peak_firing_rate": "Hz",
            "peak_location": self.units or "",
        }
        frame.attrs["variable"] = self.variable
        frame.attrs.update(self._analysis_attrs())
        return frame

    def _plot_curve(
        self, rates: NDArray[np.float64], ax: Axes | None, **kwargs: Any
    ) -> Axes:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        edges = np.asarray(self.bin_edges, dtype=np.float64)
        # NaN (unsupported) bins leave gaps in the line.
        ax.plot(self._bin_centers, rates, **kwargs)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel(_axis_label(self.variable, self.units))
        ax.set_ylabel("Firing rate (Hz)")
        return ax


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class TuningCurveResult(_TuningCurveMixin, SingleUnitRateMixin):
    """One unit's firing rate as a function of a behavioral variable.

    Parameters
    ----------
    firing_rate : ArrayLike, shape (n_bins,)
        Firing rate in Hz in each bin of the variable; NaN where the bin is
        unsupported (no occupancy, or less than ``min_occupancy``).
    occupancy : ArrayLike, shape (n_bins,)
        Seconds the variable was in each bin, unsmoothed.
    bin_edges : ArrayLike, shape (n_bins + 1,)
        Bin edges in the variable's units. Bin ``k`` holds values from
        ``bin_edges[k]`` up to (not including) ``bin_edges[k + 1]``; the last
        bin also holds ``bin_edges[-1]``.
    bandwidth : float or None, default=None
        Gaussian smoothing standard deviation in the variable's units, or
        None for an unsmoothed (binned) curve.
    spike_counts : ArrayLike or None, shape (n_bins,), optional
        Unsmoothed spike count in each bin.
    unit_id : int or str or None, optional
        The unit's label. Set when indexing a population result; None for a
        standalone computation.
    variable : str, default="value"
        Keyword-only. Name of the variable, used in plot labels and exports.
    units : str or None, default=None
        Keyword-only. Units of the variable (for example ``"cm"``); None for
        a unitless variable such as progress.

    Other Parameters
    ----------------
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to be supported.
    n_spikes_outside_intervals : int or None
        Spikes outside the analyzed time intervals
        (``analysis_settings["observed_intervals"]``), which are in no bin.
    n_samples_outside_range : int or None
        Samples in the analysis windows with a finite value outside
        ``[bin_edges[0], bin_edges[-1]]``; their sampling intervals are
        excluded.
    spike_window : ndarray, shape (n_windows, 2), or None
        When the spikes were recorded; None when assumed.
    analysis_settings : dict or None
        Resolved windows, retained intervals, bin edges and smoothing.
    _unit_ids_generated : bool, default=False
        True when the unit label was generated rather than supplied.

    Notes
    -----
    ``spatial_support`` marks the bins with occupancy of at least
    ``min_occupancy`` and a finite rate; unsupported bins are not evidence of
    silence.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import TuningCurveResult
    >>> result = TuningCurveResult(
    ...     firing_rate=np.array([1.0, 4.0, np.nan]),
    ...     occupancy=np.array([2.0, 1.0, 0.0]),
    ...     bin_edges=np.array([0.0, 0.5, 1.0, 1.5]),
    ...     variable="progress",
    ... )
    >>> result.bin_centers.tolist()
    [0.25, 0.75, 1.25]
    >>> result.peak_location().tolist()
    [0.75]
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    bin_edges: ArrayLike
    bandwidth: float | None = None
    spike_counts: ArrayLike | None = None
    unit_id: int | str | None = None
    variable: str = field(default="value", kw_only=True)
    units: str | None = field(default=None, kw_only=True)

    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: int | None = field(default=None, kw_only=True)
    n_samples_outside_range: int | None = field(default=None, kw_only=True)
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def _headline_metrics(self) -> dict[str, float]:
        return {"peak_location": float(np.asarray(self.peak_location())[0])}

    def to_dataframe(self) -> pd.DataFrame:
        """Dense table: one row per bin, with ``unit_id`` and ``bin_center``.

        Returns
        -------
        pandas.DataFrame
            Columns ``unit_id``, ``bin``, ``bin_center`` (the variable's
            units), ``firing_rate`` (Hz, NaN where unsupported),
            ``occupancy`` (s), ``method`` and ``spatial_support``.
            ``attrs`` hold the units, ``variable`` and analysis settings.
        """
        return self._tuning_dataframe(super().to_dataframe())

    def summary_table(self) -> pd.DataFrame:
        """One row for this unit, indexed by ``unit_id``.

        Returns
        -------
        pandas.DataFrame
            Columns ``peak_firing_rate`` (Hz), ``peak_location`` (the
            variable's units) and ``n_spikes_outside_intervals``.
        """
        return self._tuning_summary_frame(self._row_unit_ids().tolist())

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot firing rate against the variable.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when None.
        **kwargs
            Passed to ``Axes.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            Unsupported (NaN) bins appear as gaps.
        """
        rates = np.asarray(_to_numpy(self.firing_rate), dtype=np.float64)
        return self._plot_curve(rates, ax, **kwargs)


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class TuningCurvesResult(_TuningCurveMixin, PopulationRateMixin[TuningCurveResult]):
    """Firing rates of several units as a function of one behavioral variable.

    The units share the occupancy, bin edges and analyzed time.

    Parameters
    ----------
    firing_rates : ArrayLike, shape (n_units, n_bins)
        Firing rate in Hz of each unit in each bin; NaN where the bin is
        unsupported.
    occupancy : ArrayLike, shape (n_bins,)
        Seconds the variable was in each bin, shared by the units.
    bin_edges : ArrayLike, shape (n_bins + 1,)
        Bin edges in the variable's units; the last bin is closed.
    bandwidth : float or None, default=None
        Gaussian smoothing standard deviation in the variable's units, or
        None.
    spike_counts : ArrayLike or None, shape (n_units, n_bins), optional
        Unsmoothed spike counts.
    unit_ids : array-like, shape (n_units,), optional
        Unit labels; ``np.arange(n_units)`` when not given.
    unit_table : pandas.DataFrame or None, optional
        Per-unit metadata aligned to ``unit_ids``.
    variable : str, default="value"
        Keyword-only. Name of the variable.
    units : str or None, default=None
        Keyword-only. Units of the variable.

    Other Parameters
    ----------------
    min_occupancy, n_spikes_outside_intervals, n_samples_outside_range, spike_window, analysis_settings, _unit_ids_generated
        As on :class:`TuningCurveResult`; ``n_spikes_outside_intervals`` has
        one count per unit.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import TuningCurvesResult
    >>> result = TuningCurvesResult(
    ...     firing_rates=np.array([[1.0, 4.0], [3.0, 0.5]]),
    ...     occupancy=np.array([2.0, 1.0]),
    ...     bin_edges=np.array([0.0, 0.5, 1.0]),
    ...     unit_ids=["a", "b"],
    ...     variable="progress",
    ... )
    >>> result[1].unit_id
    'b'
    >>> result.peak_location().ravel().tolist()
    [0.75, 0.25]
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    bin_edges: ArrayLike
    bandwidth: float | None = None
    spike_counts: ArrayLike | None = None
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)
    variable: str = field(default="value", kw_only=True)
    units: str | None = field(default=None, kw_only=True)

    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: NDArray[np.int64] | None = field(
        default=None, kw_only=True
    )
    n_samples_outside_range: int | None = field(default=None, kw_only=True)
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    def __getitem__(self, idx: int) -> TuningCurveResult:
        """The tuning curve of the unit at position ``idx``.

        Parameters
        ----------
        idx : int
            Unit position (0-based); select by label with ``sel(unit_id=...)``.

        Returns
        -------
        TuningCurveResult
            With the unit's label and the shared occupancy, edges and settings.
        """
        idx = self._unit_position(idx)
        counts = self.spike_counts
        return self._unit_result(
            TuningCurveResult,
            idx,
            bin_edges=self.bin_edges,
            bandwidth=self.bandwidth,
            spike_counts=None if counts is None else np.asarray(counts)[idx],
            variable=self.variable,
            units=self.units,
            n_samples_outside_range=self.n_samples_outside_range,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Dense table: one row per ``(unit_id, bin)``, with ``bin_center``.

        Returns
        -------
        pandas.DataFrame
            Columns as for :meth:`TuningCurveResult.to_dataframe`.
        """
        return self._tuning_dataframe(super().to_dataframe())

    def summary_table(
        self, unit_ids: Sequence[str | int] | None = None
    ) -> pd.DataFrame:
        """One row per unit, indexed by ``unit_id``.

        Parameters
        ----------
        unit_ids : sequence of str or int, optional
            Labels for the index, one per unit; the result's own by default.

        Returns
        -------
        pandas.DataFrame
            Columns ``peak_firing_rate`` (Hz), ``peak_location`` (the
            variable's units) and ``n_spikes_outside_intervals``.

        Raises
        ------
        ValueError
            If ``unit_ids`` has the wrong length or repeats a label.
        """
        from neurospatial._results import resolve_unit_ids

        index = self._row_unit_ids() if unit_ids is None else unit_ids
        labels = resolve_unit_ids(index, len(self), context="summary_table")
        return self._tuning_summary_frame(list(labels))

    def plot(self, idx: int, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot one unit's firing rate against the variable.

        Parameters
        ----------
        idx : int
            Unit position.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when None.
        **kwargs
            Passed to ``Axes.plot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return self[idx].plot(ax=ax, **kwargs)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


@reads_pynapple_samples
def compute_tuning_curve(
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    bin_size: float | None = None,
    bin_edges: ArrayLike | None = None,
    value_range: tuple[float, float] | None = None,
    variable: str = "value",
    units: str | None = None,
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
) -> TuningCurveResult:
    """Firing rate of one unit as a function of a behavioral variable.

    A tuning curve for a variable sampled with the trajectory, such as
    progress along the route to a goal (:func:`~neurospatial.behavior.path_progress`),
    distance to the goal (:func:`~neurospatial.behavior.distance_to_goal`),
    running speed or time in a trial. Head direction has its own circular
    family (:func:`compute_directional_rate`).

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike times of one unit, in seconds, on the clock of ``times``.
    times : ndarray, shape (n_samples,)
        Timestamps of the variable's samples, in seconds.
    values : ndarray, shape (n_samples,)
        The variable at each sample, for example ``path_progress(...)`` or
        ``distance_to_goal(...)``, or a pynapple ``Tsd`` of it. NaN marks a sample where the variable is
        undefined (between trials, a trial without a goal); its sampling
        interval is excluded, never filled from the previous sample. Infinite
        values are refused.
    bin_size : float, optional
        Bin width in the variable's units. Give exactly one of ``bin_size``
        and ``bin_edges``: a variable of unknown units has no default width.
    bin_edges : array-like, shape (n_bins + 1,), optional
        Finite, strictly increasing edges. Pass the same edges for every
        condition you compare. Values outside ``[bin_edges[0],
        bin_edges[-1]]`` are excluded.
    value_range : (lo, hi), optional
        With ``bin_size``: the range analyzed, an exclusion boundary.
        ``(hi - lo) / bin_size`` must be a whole number (within 1e-9
        relative), else ``ValueError`` names the nearest bin sizes that
        tile it. Without it, the edges are the multiples of ``bin_size``
        covering the finite values in the analysis windows, so sessions
        binned with one ``bin_size`` share edges. Not allowed with
        ``bin_edges``.
    variable : str, default="value"
        Name of the variable, recorded on the result and used in labels.
    units : str or None, default=None
        Units of the variable, for example ``"cm"``; None when unitless.
    bandwidth : float or None, default=None
        Standard deviation of a Gaussian, in the variable's units, applied to
        the spike counts and to the occupancy separately before dividing.
        Needs bins of equal width. None leaves the curve unsmoothed.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied for its rate to be reported; others
        are NaN. 0.0 reports every occupied bin, as the directional,
        egocentric and view rate functions do; raise it so a bin crossed
        for a few frames does not report a noisy rate.
    speed : ndarray, shape (n_samples,), optional
        Running speed at each sample, for example
        ``compute_speed(times, positions)``. Needed with ``min_speed``.
    min_speed : float, optional
        Exclude sampling intervals that start below this speed, from the
        spike counts and the occupancy alike. None (default) analyzes all
        time. There are no positions to derive speed from, so ``speed`` is
        required with it.
    max_gap : float or None, default=1.0
        Longest time in seconds between consecutive samples that is
        analyzed; longer sampling intervals are recording gaps. None
        analyzes every interval.
    epochs : Intervals, (start, stop), array-like of shape (n, 2), or None
        Analyze only these half-open ``[start, stop)`` windows, such as the
        trials of one condition (``epochs=trials[trials.outcome ==
        "correct"]``). An :class:`~neurospatial.Intervals` is read through
        its ``windows``, and its ``interval_ids`` are recorded in
        ``analysis_settings["epoch_interval_ids"]``. With ``bin_edges`` or
        ``value_range``, a selection with no rows gives NaN rates, zero
        occupancy and a warning; with ``bin_size`` alone there are no values
        to set the bins from, and it raises. None (default)
        analyzes all time; when ``values`` is a pynapple ``Tsd``, None uses
        its ``time_support``, so time outside it (such as between trials of
        a restricted series) is not analyzed.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        True at sample ``k`` declares a recording break between samples
        ``k - 1`` and ``k``; that interval is excluded.
    spike_window : same forms as ``epochs``, or None
        When the spikes were recorded. Intervals outside it are excluded.
        None assumes spikes were recorded throughout.

    Returns
    -------
    TuningCurveResult
        ``firing_rate`` (Hz, NaN in unsupported bins), ``occupancy`` (s),
        ``spike_counts``, ``bin_edges``, ``variable``, ``units`` and
        ``analysis_settings`` (bin edges, ``value_range`` and whether it was
        given, ``edges_from``, ``min_speed``, ``speed_source``, ``epochs``,
        ``epoch_interval_ids``, ``spike_window``, ``break_times``,
        ``n_samples_outside_range``).

    Raises
    ------
    TypeError
        If not exactly one of ``bin_size`` and ``bin_edges`` is given,
        ``value_range`` is given with ``bin_edges``, or ``variable`` /
        ``units`` is not a string.
    ValueError
        If the edges are invalid, ``bin_size`` does not tile
        ``value_range``, ``bandwidth`` is given with bins of unequal width,
        ``values`` does not have one value per sample or holds an infinite
        value, ``min_speed`` is given without ``speed`` (or the reverse), or
        no finite value is analyzed to set the range from ``bin_size`` (as
        for an empty selection of trials without ``value_range``).

    Warns
    -----
    UserWarning
        When every sampling interval is excluded, for example by an empty
        selection of trials; the rates are then NaN and the occupancy zero.

    See Also
    --------
    compute_tuning_curves : Several units at once.
    compute_directional_rate : Tuning to head direction (circular bins).
    compute_spatial_rate : Rate maps over the environment.
    neurospatial.behavior.path_progress : Progress from start to goal.
    neurospatial.behavior.distance_to_goal : Remaining distance to the goal.

    Notes
    -----
    **What is estimated.** The time-weighted mean firing rate in each bin
    of the variable, pooling spikes and time over all analyzed intervals: a
    bin's rate is its spike count divided by the seconds spent in it. It
    does not show that the cell fires consistently from trial to trial
    (compare curves from odd and even trials, against circularly shifted
    spike trains for chance), nor that the tuning is independent of running
    speed. Where speed changes along the variable, a cell that follows only
    speed gives a curve that rises and falls along the variable as speed
    does, and ``min_speed`` (which removes stops) does not change that;
    compare curves for slower and faster trials with one test across bins,
    or fit a model with both.

    **Bins.** Bin ``k`` holds values ``bin_edges[k] <= v < bin_edges[k + 1]``
    and the last bin also holds ``bin_edges[-1]``, as in ``numpy.histogram``,
    so progress of exactly 1 is counted. Time histograms (PSTHs) stay
    half-open instead: a spike at a window's stop belongs to the next window.

    **Time and spikes.** Each sampling interval ``[t_k, t_{k+1})`` is
    credited to the bin of the value at ``t_k``, and each spike to the
    bin of the sample before it. The interval is analyzed only if it is no
    longer than ``max_gap``, does not cross a declared break, lies inside
    ``epochs`` and ``spike_window``, starts at or above ``min_speed`` and
    starts at a finite value inside the edges. The same intervals are
    removed from the spike counts and the occupancy. An undefined value is
    excluded, not bridged: progress is undefined between trials, so the
    time between trials is in no bin.

    **Smoothing.** With ``bandwidth``, the spike counts and the occupancy
    are each smoothed with the same Gaussian and then divided. Near the
    first and last bins the kernel loses the same mass from both, so the
    ratio stays a kernel-weighted rate. Bins without enough occupancy stay
    NaN even where smoothing reaches them: missing coverage is not zero
    firing.

    **Comparison with pynapple.** ``pynapple.compute_tuning_curves`` gives
    each spike the value of the closest sample and counts occupancy as
    samples divided by the estimated sampling rate. Here a spike takes the
    sample before it and occupancy is the time of each sampling interval,
    so per-bin rates differ by several percent (up to 12% in simulated
    30 Hz recordings).

    Examples
    --------
    Ten 2-s runs from start (progress 0) to goal (progress 1), sampled at
    10 Hz, by a cell that fires midway:

    >>> import numpy as np
    >>> from neurospatial.encoding import compute_tuning_curve
    >>> times = np.arange(0.0, 20.0, 0.1)
    >>> progress = (times % 2.0) / 2.0
    >>> spike_times = times[(progress >= 0.4) & (progress < 0.6)] + 0.05
    >>> curve = compute_tuning_curve(
    ...     spike_times,
    ...     times,
    ...     progress,
    ...     bin_edges=np.linspace(0, 1, 6),
    ...     variable="progress",
    ... )
    >>> curve.firing_rate.round(1).tolist()
    [0.0, 0.0, 10.0, 0.0, 0.0]
    >>> curve.peak_location().tolist()
    [0.5]
    """
    population = _tuning_curves(
        spike_times,
        times,
        values,
        bin_size=bin_size,
        bin_edges=bin_edges,
        value_range=value_range,
        variable=variable,
        units=units,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        n_jobs=1,
        unit_ids=None,
        owner="compute_tuning_curve",
        single_unit=True,
    )
    unit = population[0]
    return TuningCurveResult(
        firing_rate=unit.firing_rate,
        occupancy=unit.occupancy,
        bin_edges=unit.bin_edges,
        bandwidth=unit.bandwidth,
        spike_counts=unit.spike_counts,
        variable=unit.variable,
        units=unit.units,
        min_occupancy=unit.min_occupancy,
        n_spikes_outside_intervals=unit.n_spikes_outside_intervals,
        n_samples_outside_range=unit.n_samples_outside_range,
        spike_window=unit.spike_window,
        analysis_settings=unit.analysis_settings,
    )


@reads_pynapple_samples
def compute_tuning_curves(
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    bin_size: float | None = None,
    bin_edges: ArrayLike | None = None,
    value_range: tuple[float, float] | None = None,
    variable: str = "value",
    units: str | None = None,
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    n_jobs: int = 1,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> TuningCurvesResult:
    """Firing rates of several units as a function of a behavioral variable.

    The population version of :func:`compute_tuning_curve`: the units share
    the bins, the analyzed time and the occupancy, which is computed once.

    Parameters
    ----------
    spike_times : sequence of arrays, 2-D NaN-padded array, or pynapple TsGroup
        Spike times of each unit, in seconds. A labelled group's index
        becomes ``unit_ids``.
    times : ndarray, shape (n_samples,)
        Timestamps of the variable's samples, in seconds.
    values : ndarray, shape (n_samples,)
        The variable at each sample, for example ``path_progress(...)`` or
        ``distance_to_goal(...)``, or a pynapple ``Tsd`` of it. NaN marks a sample where the variable is
        undefined (between trials, a trial without a goal); its sampling
        interval is excluded, never filled from the previous sample. Infinite
        values are refused.
    bin_size : float, optional
        Bin width in the variable's units. Give exactly one of ``bin_size``
        and ``bin_edges``: a variable of unknown units has no default width.
    bin_edges : array-like, shape (n_bins + 1,), optional
        Finite, strictly increasing edges. Pass the same edges for every
        condition you compare. Values outside ``[bin_edges[0],
        bin_edges[-1]]`` are excluded.
    value_range : (lo, hi), optional
        With ``bin_size``: the range analyzed, an exclusion boundary.
        ``(hi - lo) / bin_size`` must be a whole number (within 1e-9
        relative), else ``ValueError`` names the nearest bin sizes that
        tile it. Without it, the edges are the multiples of ``bin_size``
        covering the finite values in the analysis windows, so sessions
        binned with one ``bin_size`` share edges. Not allowed with
        ``bin_edges``.
    variable : str, default="value"
        Name of the variable, recorded on the result and used in labels.
    units : str or None, default=None
        Units of the variable, for example ``"cm"``; None when unitless.
    bandwidth : float or None, default=None
        Standard deviation of a Gaussian, in the variable's units, applied to
        the spike counts and to the occupancy separately before dividing.
        Needs bins of equal width. None leaves the curve unsmoothed.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied for its rate to be reported; others
        are NaN. 0.0 reports every occupied bin, as the directional,
        egocentric and view rate functions do; raise it so a bin crossed
        for a few frames does not report a noisy rate.
    speed : ndarray, shape (n_samples,), optional
        Running speed at each sample, for example
        ``compute_speed(times, positions)``. Needed with ``min_speed``.
    min_speed : float, optional
        Exclude sampling intervals that start below this speed, from the
        spike counts and the occupancy alike. None (default) analyzes all
        time. There are no positions to derive speed from, so ``speed`` is
        required with it.
    max_gap : float or None, default=1.0
        Longest time in seconds between consecutive samples that is
        analyzed; longer sampling intervals are recording gaps. None
        analyzes every interval.
    epochs : Intervals, (start, stop), array-like of shape (n, 2), or None
        Analyze only these half-open ``[start, stop)`` windows, such as the
        trials of one condition (``epochs=trials[trials.outcome ==
        "correct"]``). An :class:`~neurospatial.Intervals` is read through
        its ``windows``, and its ``interval_ids`` are recorded in
        ``analysis_settings["epoch_interval_ids"]``. With ``bin_edges`` or
        ``value_range``, a selection with no rows gives NaN rates, zero
        occupancy and a warning; with ``bin_size`` alone there are no values
        to set the bins from, and it raises. None (default)
        analyzes all time; when ``values`` is a pynapple ``Tsd``, None uses
        its ``time_support``, so time outside it (such as between trials of
        a restricted series) is not analyzed.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        True at sample ``k`` declares a recording break between samples
        ``k - 1`` and ``k``; that interval is excluded.
    spike_window : same forms as ``epochs``, or None
        When the spikes were recorded. Intervals outside it are excluded.
        None assumes spikes were recorded throughout.
    n_jobs : int, default=1
        Parallel jobs for counting spikes; -1 uses every CPU.
    unit_ids : array-like, shape (n_units,), optional
        A label per unit, in the order of ``spike_times``; defaults to the
        group's labels, else ``np.arange(n_units)``. A wrong length or a
        repeated label raises ``ValueError``.

    Returns
    -------
    TuningCurvesResult
        ``firing_rates`` of shape (n_units, n_bins), the shared
        ``occupancy`` and ``bin_edges``, ``spike_counts`` and
        ``analysis_settings`` as for :func:`compute_tuning_curve`. Index
        it (``result[i]``) for one unit's :class:`TuningCurveResult`.

    Raises
    ------
    TypeError, ValueError
        As for :func:`compute_tuning_curve`, and ``ValueError`` for
        ``unit_ids`` of the wrong length or with repeated labels.

    Warns
    -----
    UserWarning
        When every sampling interval is excluded (NaN rates, zero
        occupancy), or when at least five units are all silent for at least
        60 s and ``spike_window`` was not given.

    See Also
    --------
    compute_tuning_curve : One unit.
    compute_directional_rates : Tuning to head direction (circular bins).
    compute_spatial_rates : Rate maps over the environment.
    neurospatial.behavior.path_progress : Progress from start to goal.
    neurospatial.behavior.distance_to_goal : Remaining distance to the goal.

    Notes
    -----
    **What is estimated.** The time-weighted mean firing rate in each bin
    of the variable, pooling spikes and time over all analyzed intervals: a
    bin's rate is its spike count divided by the seconds spent in it. It
    does not show that the cell fires consistently from trial to trial
    (compare curves from odd and even trials, against circularly shifted
    spike trains for chance), nor that the tuning is independent of running
    speed. Where speed changes along the variable, a cell that follows only
    speed gives a curve that rises and falls along the variable as speed
    does, and ``min_speed`` (which removes stops) does not change that;
    compare curves for slower and faster trials with one test across bins,
    or fit a model with both.

    **Bins.** Bin ``k`` holds values ``bin_edges[k] <= v < bin_edges[k + 1]``
    and the last bin also holds ``bin_edges[-1]``, as in ``numpy.histogram``,
    so progress of exactly 1 is counted. Time histograms (PSTHs) stay
    half-open instead: a spike at a window's stop belongs to the next window.

    **Time and spikes.** Each sampling interval ``[t_k, t_{k+1})`` is
    credited to the bin of the value at ``t_k``, and each spike to the
    bin of the sample before it. The interval is analyzed only if it is no
    longer than ``max_gap``, does not cross a declared break, lies inside
    ``epochs`` and ``spike_window``, starts at or above ``min_speed`` and
    starts at a finite value inside the edges. The same intervals are
    removed from the spike counts and the occupancy. An undefined value is
    excluded, not bridged: progress is undefined between trials, so the
    time between trials is in no bin.

    **Smoothing.** With ``bandwidth``, the spike counts and the occupancy
    are each smoothed with the same Gaussian and then divided. Near the
    first and last bins the kernel loses the same mass from both, so the
    ratio stays a kernel-weighted rate. Bins without enough occupancy stay
    NaN even where smoothing reaches them: missing coverage is not zero
    firing.

    **Comparison with pynapple.** ``pynapple.compute_tuning_curves`` gives
    each spike the value of the closest sample and counts occupancy as
    samples divided by the estimated sampling rate. Here a spike takes the
    sample before it and occupancy is the time of each sampling interval,
    so per-bin rates differ by several percent (up to 12% in simulated
    30 Hz recordings).

    Examples
    --------
    Tuning curves over the correct trials; pass the same ``bin_edges`` for
    the error trials so the two can be compared bin by bin:

    >>> import numpy as np
    >>> from neurospatial import Intervals
    >>> from neurospatial.encoding import compute_tuning_curves
    >>> times = np.arange(0.0, 20.0, 0.1)
    >>> progress = (times % 2.0) / 2.0
    >>> trials = Intervals.from_arrays(
    ...     np.arange(0.0, 20.0, 2.0),
    ...     np.arange(2.0, 22.0, 2.0),
    ...     outcome=["correct"] * 8 + ["error"] * 2,
    ... )
    >>> spikes = [times[progress >= 0.8] + 0.05, times[progress < 0.2] + 0.05]
    >>> edges = np.linspace(0, 1, 6)
    >>> correct = compute_tuning_curves(
    ...     spikes,
    ...     times,
    ...     progress,
    ...     bin_edges=edges,
    ...     variable="progress",
    ...     epochs=trials[trials.outcome == "correct"],
    ... )
    >>> correct.firing_rates.round(1).tolist()
    [[0.0, 0.0, 0.0, 0.0, 10.0], [10.0, 0.0, 0.0, 0.0, 0.0]]
    >>> correct.analysis_settings["epoch_interval_ids"]
    [0, 1, 2, 3, 4, 5, 6, 7]
    """
    return _tuning_curves(
        spike_times,
        times,
        values,
        bin_size=bin_size,
        bin_edges=bin_edges,
        value_range=value_range,
        variable=variable,
        units=units,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        n_jobs=n_jobs,
        unit_ids=unit_ids,
        owner="compute_tuning_curves",
        single_unit=False,
    )


def _tuning_curves(
    spike_times: Any,
    times: Any,
    values: Any,
    *,
    bin_size: Any,
    bin_edges: Any,
    value_range: Any,
    variable: Any,
    units: Any,
    bandwidth: Any,
    min_occupancy: Any,
    speed: Any,
    min_speed: float | None,
    max_gap: float | None,
    epochs: Any,
    break_before: Any,
    spike_window: Any,
    n_jobs: int,
    unit_ids: Any,
    owner: str,
    single_unit: bool,
) -> TuningCurvesResult:
    """Validate, bin the variable, count every unit and estimate the rates."""
    from neurospatial._interval_table import Intervals
    from neurospatial._spikes import as_labelled_spike_trains, as_spike_train
    from neurospatial.encoding._provenance import (
        rate_analysis_settings,
        spikes_outside_intervals,
    )
    from neurospatial.encoding._validation import validate_trajectory

    call = f"{owner}()"
    if _is_pynapple_object(values):
        # As for pynapple positions and headings: time outside the series'
        # support (between trials, for a restricted series) is not analyzed.
        support = getattr(values, "time_support", None)
        if epochs is None and support is not None:
            epochs = interval_rows(support)
        values = np.asarray(values.d)
    validate_instance(variable, str, name="variable", owner=call)
    if units is not None:
        validate_instance(units, str, name="units", owner=call)
    epoch_ids = (
        np.asarray(epochs.interval_ids).tolist()
        if isinstance(epochs, Intervals)
        else None
    )
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)

    if single_unit:
        spike_trains = [as_spike_train(spike_times)]
        resolved_unit_ids = None
    else:
        spike_trains, resolved_unit_ids = as_labelled_spike_trains(
            spike_times, unit_ids, context=owner
        )

    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    validate_trajectory(times, context=owner)
    if values.shape != times.shape:
        raise ValueError(
            _format_error(
                f"{call}: values has shape {values.shape}, but times has "
                f"{times.shape}.",
                why="Why: each value is the variable at one sample time.",
                fix="pass one value per sample (NaN where it is undefined).",
            )
        )
    refuse_infinite(values, name="values", call=owner)
    breaks = validate_break_before(break_before, len(times), call=owner)
    speed_values = validate_speed_gate(speed, min_speed, len(times))
    bandwidth = validate_finite_scalar(
        bandwidth, name="bandwidth", positive=True, none_allowed=True
    )
    min_occupancy = validate_min_occupancy(min_occupancy)

    in_windows = np.ones(len(times), dtype=bool)
    for rows in (resolved_epochs, resolved_spike_window):
        if rows is not None:
            in_windows &= times_in_intervals(rows, times)
    edges, edges_source = tuning_bin_edges(
        values[in_windows],
        bin_size=bin_size,
        bin_edges=bin_edges,
        value_range=value_range,
        owner=call,
    )
    widths = np.diff(edges)
    # Edges far from zero carry rounding in proportion to their size.
    rounding = 4 * float(np.spacing(np.abs(edges).max()))
    if bandwidth is not None and not np.allclose(
        widths, widths[0], rtol=_TILING_RTOL, atol=rounding
    ):
        raise ValueError(
            _format_error(
                f"{call}: bandwidth={bandwidth:g} needs bins of equal width; "
                f"bin_edges has widths from {widths.min():g} to {widths.max():g}.",
                why="Why: the Gaussian is applied over bins, so its width in "
                "the variable's units is fixed only when every bin has one width.",
                fix="pass equally spaced bin_edges (np.linspace), or bandwidth=None.",
            )
        )

    frame_bins = value_frame_bins(values, edges, closed_last=True)
    in_range = (values >= edges[0]) & (values <= edges[-1])
    n_outside = int(np.count_nonzero(in_windows & np.isfinite(values) & ~in_range))

    if not single_unit:
        # Recording coverage is about time, whether or not the variable is
        # defined, so every sample counts as tracked.
        _warn_if_population_silent(
            spike_trains,
            times,
            max_gap=max_gap,
            epochs=resolved_epochs,
            spike_window=resolved_spike_window,
            tracked=tracked_samples(len(times)),
            break_before=breaks,
        )

    # Excluded samples are bin -1 and stay valid samples, so their interval
    # is dropped rather than filled from the previous bin.
    gates = interval_gates(
        times,
        start_bin=frame_bins,
        speed=speed_values,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=breaks,
    )
    prepared = prepare_frame_binning(
        gates, frame_bins, len(edges) - 1, samples="values"
    )
    spike_counts, occupancy = prepared.count(spike_trains, n_jobs=n_jobs)
    firing_rates = kernel_ratio_rates(
        occupancy,
        bandwidth=bandwidth,
        bin_size=float(widths.mean()),
        mode="constant",
        min_occupancy=min_occupancy,
    )(spike_counts)

    settings = rate_analysis_settings(
        prepared.support,
        method="binned" if bandwidth is None else "gaussian",
        bandwidth=bandwidth,
        backend="numpy",
        min_occupancy=min_occupancy,
        min_speed=min_speed,
        speed=speed,
        variable=variable,
        units=units,
        bin_edges=edges.tolist(),
        value_range=[float(edges[0]), float(edges[-1])],
        edges_from=edges_source,
        epoch_interval_ids=epoch_ids,
        n_samples_outside_range=n_outside,
    )
    return TuningCurvesResult(
        firing_rates=firing_rates,
        occupancy=occupancy,
        bin_edges=edges,
        bandwidth=bandwidth,
        spike_counts=spike_counts,
        unit_ids=resolved_unit_ids,
        unit_table=None if single_unit else spike_unit_table(spike_times),
        variable=variable,
        units=units,
        min_occupancy=min_occupancy,
        n_spikes_outside_intervals=spikes_outside_intervals(
            spike_trains, prepared.support
        ),
        n_samples_outside_range=n_outside,
        spike_window=resolved_spike_window,
        analysis_settings=settings,
    )
