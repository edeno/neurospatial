"""Align continuous behavioral signals to events.

Running speed, head direction or the occupied zone around each reward is the
behavioral counterpart of a peri-event time histogram. These signals are
sampled at fixed times relative to each event, only where they were observed:
nothing is interpolated across a tracking gap, a declared break or an epoch
edge, and nothing is extrapolated past the recording. Each feature is
summarized by its kind: linear features by their mean, angles by their
circular mean and resultant length, and categorical codes by counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP, as_intervals
from neurospatial._observation import validated_signals
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin, as_label_array
from neurospatial.events._continuous import (
    _EVENT_STATUSES,
    _VALUE_KINDS,
    _check_category_codes,
    _circular_means,
    _event_chunks,
    _overlapping,
    _relative_times,
    _resolve_features,
    _sample_events,
    _Summaries,
    _window_status,
)
from neurospatial.events.alignment import _validate_window

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

__all__ = ["ContinuousEventResult", "align_continuous_to_events"]

_WINDOW_COVERAGES = ("complete", "partial")
_MEAN_STATUSES = ("defined", "no_support", "undefined_direction", "not_numeric")


@dataclass(frozen=True, repr=False, eq=False)
class ContinuousEventResult(ReadOnlyArraysMixin, ResultMixin):
    """Continuous signals sampled around events, from ``align_continuous_to_events``.

    Every input event has a status: ``"retained"`` or the reason it was
    excluded. Retained events are summarized per relative time and feature by
    sufficient statistics (``counts``, ``sums``, ``sin_sums``, ``cos_sums``,
    ``category_counts``), from which ``means``, ``resultant_lengths`` and
    ``mean_status`` are derived. The sampled values themselves are kept in
    ``values`` unless the alignment ran with ``retain_aligned=False``.

    Attributes
    ----------
    relative_times : NDArray[np.float64], shape (n_times,)
        Query offsets from each event, in seconds.
    feature_ids : NDArray, shape (n_features,)
        Feature labels.
    value_kinds : tuple of str, length n_features
        ``"linear"``, ``"circular"`` or ``"categorical"`` per feature.
    interpolation : tuple of str, length n_features
        Resolved interpolation per feature (``"linear"``, ``"circular"`` or
        ``"previous"``).
    feature_units : tuple of str or None, length n_features
        Unit per feature; ``"rad"`` for circular features.
    event_status : NDArray[np.str_], shape (n_input_events,)
        ``"retained"``, ``"outside_recording"``, ``"outside_epochs"``,
        ``"incomplete_support"`` or ``"no_support"`` per input event.
    overlapping : NDArray[np.bool_], shape (n_input_events,)
        True when a retained event's window shares time with another retained
        event's window (overlapping windows reuse samples). Duplicate event
        times overlap each other; excluded events are False. Bounds are
        compared in floating point, so windows meant only to touch can be
        flagged.
    counts : NDArray[np.int64], shape (n_times, n_features)
        Retained events with a supported value in each cell.
    sums : NDArray[np.float64], shape (n_times, n_features)
        Sum of supported values; NaN columns for non-linear features.
    sin_sums, cos_sums : NDArray[np.float64], shape (n_times, n_features)
        Sum of the sine and cosine of supported angles; NaN columns for
        non-circular features.
    category_features : NDArray[np.intp], shape (n_categories,)
        Feature column of each category row.
    category_codes : NDArray[np.int64], shape (n_categories,)
        Integer code of each category row, sorted within each feature.
    category_labels : NDArray[np.str_], shape (n_categories,)
        Label of each category row.
    category_counts : NDArray[np.int64], shape (n_times, n_categories)
        Retained events holding each category row's code, per relative time.
    window : tuple of float
        ``(start, stop)`` in seconds relative to each event, half-open.
    dt : float
        Query spacing in seconds.
    window_coverage : {"complete", "partial"}
        The coverage rule that decided which events were kept.
    resultant_threshold : float
        Circular means are undefined at resultant lengths at or below it.
    max_gap : float or None
        The gap limit that was applied, in seconds.
    epochs : NDArray[np.float64] or None, shape (n_epochs, 2)
        Normalized half-open ``[start, stop)`` windows that were applied.
    values : NDArray[np.float64] or None, shape (n_events, n_times, n_features)
        Sampled values of the retained events, in input order; NaN where
        unsupported. Categorical codes are stored as floats. Circular values
        are radians in ``[-pi, pi)``, while circular ``means`` are in
        ``(-pi, pi]``, so due West reads ``-pi`` here and ``pi`` in ``means``.
        None when the alignment ran with ``retain_aligned=False``.
    support : NDArray[np.bool_] or None, shape (n_events, n_times, n_features)
        True where ``values`` was observed. None with ``values``.
    event_ids : NDArray or None, shape (n_input_events,)
        Caller labels of the input events, when given.
    """

    relative_times: NDArray[np.float64]
    feature_ids: NDArray[Any]
    value_kinds: tuple[str, ...]
    interpolation: tuple[str, ...]
    feature_units: tuple[str | None, ...]
    event_status: NDArray[np.str_]
    overlapping: NDArray[np.bool_]
    counts: NDArray[np.int64]
    sums: NDArray[np.float64]
    sin_sums: NDArray[np.float64]
    cos_sums: NDArray[np.float64]
    category_features: NDArray[np.intp]
    category_codes: NDArray[np.int64]
    category_labels: NDArray[np.str_]
    category_counts: NDArray[np.int64]
    window: tuple[float, float]
    dt: float
    window_coverage: str
    resultant_threshold: float
    max_gap: float | None
    epochs: NDArray[np.float64] | None
    values: NDArray[np.float64] | None
    support: NDArray[np.bool_] | None
    event_ids: NDArray[Any] | None

    def __post_init__(self) -> None:
        """Check that the fields describe one alignment, then own the arrays."""
        problems = self._inconsistencies()
        if problems:
            raise ValueError(
                "ContinuousEventResult fields are inconsistent: "
                + "; ".join(problems)
                + "."
            )
        self._own_arrays()

    def _inconsistencies(self) -> list[str]:
        """List every way the fields disagree with each other."""
        problems: list[str] = []
        relative_times = np.asarray(self.relative_times)
        feature_ids = np.asarray(self.feature_ids)
        status = np.asarray(self.event_status)
        if relative_times.ndim != 1 or feature_ids.ndim != 1 or status.ndim != 1:
            return ["relative_times, feature_ids and event_status must be 1-D"]
        n_times, n_features, n_input = (
            relative_times.size,
            feature_ids.size,
            status.size,
        )

        for name in ("value_kinds", "interpolation", "feature_units"):
            if len(getattr(self, name)) != n_features:
                problems.append(f"{name} needs {n_features} entries")
        if not set(self.value_kinds) <= set(_VALUE_KINDS):
            problems.append(f"value_kinds must be in {list(_VALUE_KINDS)}")
        if not np.isin(status, _EVENT_STATUSES).all():
            problems.append(f"event_status must be in {list(_EVENT_STATUSES)}")
        if np.shape(self.overlapping) != (n_input,):
            problems.append(f"overlapping needs shape ({n_input},)")
        if self.event_ids is not None and np.shape(self.event_ids) != (n_input,):
            problems.append(f"event_ids needs shape ({n_input},)")

        cell = (n_times, n_features)
        for name in ("counts", "sums", "sin_sums", "cos_sums"):
            if np.shape(getattr(self, name)) != cell:
                problems.append(f"{name} needs shape {cell}")

        category_features = np.asarray(self.category_features)
        n_categories = category_features.size
        if category_features.ndim != 1 or any(
            np.shape(getattr(self, name)) != (n_categories,)
            for name in ("category_codes", "category_labels")
        ):
            problems.append(
                "category_features, category_codes and category_labels need one "
                "entry per category row"
            )
        elif np.shape(self.category_counts) != (n_times, n_categories):
            problems.append(f"category_counts needs shape ({n_times}, {n_categories})")
        elif n_categories and len(self.value_kinds) == n_features:
            in_range = (category_features >= 0) & (category_features < n_features)
            kinds = np.asarray(self.value_kinds)
            if not (
                in_range.all() and (kinds[category_features] == "categorical").all()
            ):
                problems.append("category_features must name categorical features")

        aligned = (int(np.count_nonzero(status == "retained")), *cell)
        if self.values is None or self.support is None:
            if (self.values is None) != (self.support is None):
                problems.append("values and support are both given or both None")
        elif np.shape(self.values) != aligned or np.shape(self.support) != aligned:
            problems.append(
                f"values and support need shape {aligned}, one row per retained event"
            )

        if self.window_coverage not in _WINDOW_COVERAGES:
            problems.append(f"window_coverage must be in {list(_WINDOW_COVERAGES)}")
        threshold = self.resultant_threshold
        if not (np.isfinite(threshold) and 0.0 <= threshold <= 1.0):
            problems.append("resultant_threshold must be finite and in [0, 1]")
        return problems

    @property
    def event_indices(self) -> NDArray[np.intp]:
        """Input positions of the retained events, shape ``(n_events,)``.

        Row ``i`` of ``values`` belongs to input event ``event_indices[i]``;
        use ``event_times[result.event_indices]`` for the matching cohort.
        """
        return np.flatnonzero(self.event_status == "retained")

    @property
    def n_events(self) -> int:
        """Number of retained events."""
        return int(np.count_nonzero(self.event_status == "retained"))

    @property
    def n_events_dropped(self) -> int:
        """Number of input events that were dropped (``event_status`` says why)."""
        return int(self.event_status.size - self.n_events)

    @property
    def means(self) -> NDArray[np.float64]:
        """Mean value per relative time and feature, shape ``(n_times, n_features)``.

        Linear features: ``sums / counts``. Circular features: the circular
        mean ``atan2(sin_sums, cos_sums)`` in ``(-pi, pi]``, NaN where the
        resultant length is at or below ``resultant_threshold`` (the angles
        cancel and have no mean direction). NaN for categorical features and
        for cells without support; ``mean_status`` says which.
        """
        means, _, _ = _circular_means(
            self.sin_sums, self.cos_sums, self.counts, self.resultant_threshold
        )
        linear = np.asarray(self.value_kinds) == "linear"
        linear_means = np.full(self.counts.shape, np.nan)
        np.divide(
            self.sums,
            self.counts,
            out=linear_means,
            where=(self.counts > 0) & linear,
        )
        means[:, linear] = linear_means[:, linear]
        return means

    @property
    def resultant_lengths(self) -> NDArray[np.float64]:
        """Mean resultant length, shape ``(n_times, n_features)``.

        ``hypot(sin_sums, cos_sums) / counts`` in ``[0, 1]`` for circular
        features: 1 when every event has the same angle, near 0 when the
        angles cancel. NaN for non-circular features and cells without
        support.
        """
        _, lengths, _ = _circular_means(
            self.sin_sums, self.cos_sums, self.counts, self.resultant_threshold
        )
        return lengths

    @property
    def mean_status(self) -> NDArray[np.str_]:
        """Why each entry of ``means`` is or is not defined, shape ``(n_times, n_features)``.

        ``"defined"``; ``"no_support"`` (no retained event was observed in
        the cell); ``"undefined_direction"`` (circular, resultant length at or
        below ``resultant_threshold``); ``"not_numeric"`` (categorical: see
        ``category_counts``).
        """
        _, _, defined = _circular_means(
            self.sin_sums, self.cos_sums, self.counts, self.resultant_threshold
        )
        kinds = np.asarray(self.value_kinds)
        width = max(len(name) for name in _MEAN_STATUSES)
        status = np.full(self.counts.shape, "defined", dtype=f"<U{width}")
        undefined = (self.counts > 0) & ~defined & (kinds == "circular")
        status[undefined] = "undefined_direction"
        status[self.counts == 0] = "no_support"
        status[:, kinds == "categorical"] = "not_numeric"
        return status

    # ------------------------------------------------------------ exports

    def _table_attrs(self) -> dict[str, Any]:
        """Alignment settings stored in ``df.attrs`` of every table."""
        return {
            "window": (float(self.window[0]), float(self.window[1])),
            "dt": float(self.dt),
            "window_coverage": self.window_coverage,
            "max_gap": self.max_gap,
            "resultant_threshold": float(self.resultant_threshold),
            "feature_units": dict(
                zip(self.feature_ids.tolist(), self.feature_units, strict=True)
            ),
        }

    def _frame(self, data: dict[str, Any]) -> pd.DataFrame:
        """Build a table from fresh column copies and attach ``attrs``."""
        import pandas as pd

        df = pd.DataFrame({name: np.array(col) for name, col in data.items()})
        df.attrs.update(self._table_attrs())
        return df

    def _category_proportions(self) -> NDArray[np.float64]:
        """Category counts over the feature's supported events, NaN at zero."""
        totals = self.counts[:, self.category_features]
        proportion = np.full(self.category_counts.shape, np.nan)
        np.divide(self.category_counts, totals, out=proportion, where=totals > 0)
        return proportion

    def events_table(self) -> pd.DataFrame:
        """One row per input event with its coverage decision.

        Returns
        -------
        pandas.DataFrame
            Columns ``event_index`` (input row), ``event_id`` (only when
            ``event_ids`` was given), ``status`` (``"retained"`` or the
            exclusion reason) and ``overlapping``. ``df.attrs`` holds
            ``window``, ``dt``, ``window_coverage``, ``max_gap``,
            ``resultant_threshold`` and ``feature_units``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events import align_continuous_to_events
        >>> times = np.arange(0, 10.0001, 0.125)
        >>> result = align_continuous_to_events(
        ...     times, times, [2.0, 5.0], (-0.5, 0.5), dt=0.25, event_ids=["a", "b"]
        ... )
        >>> result.events_table()[["event_index", "event_id", "status"]]
           event_index event_id    status
        0            0        a  retained
        1            1        b  retained
        """
        data: dict[str, Any] = {"event_index": np.arange(self.event_status.size)}
        if self.event_ids is not None:
            data["event_id"] = self.event_ids
        data["status"] = self.event_status
        data["overlapping"] = self.overlapping
        return self._frame(data)

    def aligned_table(self) -> pd.DataFrame:
        """One row per retained event, relative time and feature.

        Rows are ordered by ``event_index``, then ``relative_time``, then
        feature, so ``n_events * n_times * n_features`` rows.

        Returns
        -------
        pandas.DataFrame
            Columns ``event_index``, ``event_id`` (only when ``event_ids`` was
            given), ``relative_time``, ``feature_id``, ``value`` (NaN where
            unsupported; categorical codes as floats) and ``supported``.
            ``df.attrs`` as in ``events_table``.

        Raises
        ------
        ValueError
            If the alignment ran with ``retain_aligned=False``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events import align_continuous_to_events
        >>> times = np.arange(0, 10.0001, 0.125)
        >>> result = align_continuous_to_events(
        ...     times, times, [2.0, 5.0], (-0.5, 0.5), dt=0.25, event_ids=["a", "b"]
        ... )
        >>> table = result.aligned_table()
        >>> len(table)
        8
        >>> table[["event_id", "relative_time", "value"]].head(4)
          event_id  relative_time  value
        0        a          -0.50   1.50
        1        a          -0.25   1.75
        2        a           0.00   2.00
        3        a           0.25   2.25
        """
        if self.values is None or self.support is None:
            raise ValueError(
                _format_error(
                    "aligned_table needs the sampled values, but this result "
                    "kept only summary statistics (retain_aligned=False)",
                    why="the per-event values were not stored",
                    fix="rerun align_continuous_to_events with retain_aligned=True",
                )
            )
        n_events, n_times, n_features = self.values.shape
        indices = self.event_indices
        data: dict[str, Any] = {"event_index": np.repeat(indices, n_times * n_features)}
        if self.event_ids is not None:
            data["event_id"] = np.repeat(self.event_ids[indices], n_times * n_features)
        data["relative_time"] = np.tile(
            np.repeat(self.relative_times, n_features), n_events
        )
        data["feature_id"] = np.tile(self.feature_ids, n_events * n_times)
        data["value"] = self.values.reshape(-1)
        data["supported"] = self.support.reshape(-1)
        return self._frame(data)

    def time_course_table(self) -> pd.DataFrame:
        """One row per relative time and feature with the retained-event summary.

        Returns
        -------
        pandas.DataFrame
            Columns ``relative_time``, ``feature_id``, ``value_kind``,
            ``count`` (retained events with support), ``mean`` (NaN for
            categorical features and undefined means), ``resultant_length``
            (circular features only) and ``status`` (``mean_status``). Rows
            are ordered by relative time, then feature. ``df.attrs`` as in
            ``events_table``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events import align_continuous_to_events
        >>> times = np.arange(0, 10.0001, 0.125)
        >>> result = align_continuous_to_events(
        ...     times, times, [2.0, 5.0], (-0.5, 0.5), dt=0.25, event_ids=["a", "b"]
        ... )
        >>> result.time_course_table()[["relative_time", "count", "mean"]]
           relative_time  count  mean
        0          -0.50      2  3.00
        1          -0.25      2  3.25
        2           0.00      2  3.50
        3           0.25      2  3.75
        """
        n_times, n_features = self.counts.shape
        return self._frame(
            {
                "relative_time": np.repeat(self.relative_times, n_features),
                "feature_id": np.tile(self.feature_ids, n_times),
                "value_kind": np.tile(np.asarray(self.value_kinds), n_times),
                "count": self.counts.reshape(-1),
                "mean": self.means.reshape(-1),
                "resultant_length": self.resultant_lengths.reshape(-1),
                "status": self.mean_status.reshape(-1),
            }
        )

    def category_counts_table(self) -> pd.DataFrame:
        """One row per relative time, categorical feature and category.

        Returns
        -------
        pandas.DataFrame
            Columns ``relative_time``, ``feature_id``, ``category_code``,
            ``category_label``, ``count`` and ``proportion`` (``count``
            divided by the feature's supported events at that time; NaN when
            there are none). Empty when no feature is categorical. Rows are
            ordered by relative time, then feature, then code. ``df.attrs``
            as in ``events_table``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events import align_continuous_to_events
        >>> times = np.arange(0, 10.0001, 0.125)
        >>> result = align_continuous_to_events(
        ...     times, times, [2.0, 5.0], (-0.5, 0.5), dt=0.25, event_ids=["a", "b"]
        ... )
        >>> result.category_counts_table().empty
        True
        """
        n_times = self.relative_times.size
        n_categories = self.category_codes.size
        rows = self.category_counts.reshape(-1)
        proportion = self._category_proportions().reshape(-1)
        return self._frame(
            {
                "relative_time": np.repeat(self.relative_times, n_categories),
                "feature_id": np.tile(
                    self.feature_ids[self.category_features], n_times
                ),
                "category_code": np.tile(self.category_codes, n_times),
                "category_label": np.tile(self.category_labels, n_times),
                "count": rows,
                "proportion": proportion,
            }
        )

    def summary(self) -> dict[str, Any]:
        """Scalar description of the alignment and its coverage decisions.

        Returns
        -------
        dict
            ``n_events``, ``n_events_dropped``, ``n_features``,
            ``n_relative_times``, ``window_start``, ``window_stop`` (s),
            ``dt`` (s), ``window_coverage``, ``n_overlapping_events``,
            ``min_count`` (smallest supported-event count over cells) and
            ``n_dropped_<reason>`` for each exclusion reason present. All
            values are Python ``int``, ``float`` or ``str``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.events import align_continuous_to_events
        >>> times = np.arange(0, 10.0001, 0.125)
        >>> result = align_continuous_to_events(
        ...     times, times, [2.0, 5.0], (-0.5, 0.5), dt=0.25, event_ids=["a", "b"]
        ... )
        >>> result.summary()["n_events"], result.summary()["min_count"]
        (2, 2)
        """
        out: dict[str, Any] = {
            "n_events": self.n_events,
            "n_events_dropped": self.n_events_dropped,
            "n_features": int(self.feature_ids.size),
            "n_relative_times": int(self.relative_times.size),
            "window_start": float(self.window[0]),
            "window_stop": float(self.window[1]),
            "dt": float(self.dt),
            "window_coverage": str(self.window_coverage),
            "n_overlapping_events": int(np.count_nonzero(self.overlapping)),
            "min_count": int(self.counts.min()),
        }
        reasons, reason_counts = np.unique(
            self.event_status[self.event_status != "retained"], return_counts=True
        )
        for reason, n in zip(reasons.tolist(), reason_counts.tolist(), strict=True):
            out[f"n_dropped_{reason}"] = int(n)
        return out

    def plot(self, ax: Axes | None = None, *, feature_id: Any = None) -> Axes:
        """Plot one feature's summary against time from the event.

        Linear features show ``means`` as a line (NaN leaves gaps). Circular
        features show ``means`` as markers on ``[-pi, pi]``, so no line crosses
        the wrap. Categorical features show one ``proportion`` line per
        category.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created when omitted.
        feature_id : optional
            Feature to plot. Required when the result has several features.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Raises
        ------
        ValueError
            If ``feature_id`` is missing for several features or unknown.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from neurospatial.events import align_continuous_to_events
        >>> times = np.arange(0, 10.0001, 0.125)
        >>> result = align_continuous_to_events(
        ...     times, times, [2.0, 5.0], (-0.5, 0.5), dt=0.25
        ... )
        >>> ax = result.plot()
        >>> ax.get_xlabel()
        'Time from event (s)'
        """
        import matplotlib.pyplot as plt

        ids = self.feature_ids.tolist()
        if feature_id is None:
            if len(ids) != 1:
                raise ValueError(
                    _format_error(
                        "plot needs feature_id because the result has "
                        f"{len(ids)} features",
                        why="one axes shows one feature",
                        fix=f"pass feature_id= one of {ids!r}",
                    )
                )
            column = 0
        elif feature_id in ids:
            column = ids.index(feature_id)
        else:
            raise ValueError(
                _format_error(
                    f"feature_id={feature_id!r} is not a feature of this result",
                    why=f"the feature ids are {ids!r}",
                    fix="pass one of them",
                )
            )
        if ax is None:
            _, ax = plt.subplots()

        kind = self.value_kinds[column]
        unit = self.feature_units[column]
        t = self.relative_times
        if kind == "linear":
            ax.plot(t, self.means[:, column])
            ax.set_ylabel(unit if unit is not None else "value")
        elif kind == "circular":
            ax.plot(t, self.means[:, column], linestyle="none", marker="o")
            ax.set_ylim(-np.pi, np.pi)
            ax.set_ylabel("radians")
        else:
            rows = np.flatnonzero(self.category_features == column)
            proportions = self._category_proportions()
            for row in rows:
                ax.plot(t, proportions[:, row], label=str(self.category_labels[row]))
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("proportion of events")
            ax.legend()
        ax.axvline(0.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time from event (s)")
        return ax


def align_continuous_to_events(
    times: ArrayLike,
    signals: ArrayLike,
    event_times: ArrayLike,
    window: tuple[float, float],
    *,
    dt: float,
    feature_ids: Sequence[Any] | NDArray[Any] | None = None,
    feature_units: Mapping[Any, str | None] | None = None,
    event_ids: Sequence[Any] | NDArray[Any] | None = None,
    value_kinds: Mapping[Any, str] | None = None,
    category_labels: Mapping[Any, Mapping[int, str]] | None = None,
    interpolation: Mapping[Any, str] | None = None,
    resultant_threshold: float = 1e-12,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    valid: ArrayLike | None = None,
    break_before: ArrayLike | None = None,
    window_coverage: str = "complete",
    retain_aligned: bool = True,
) -> ContinuousEventResult:
    """
    Sample continuous signals at fixed times around each event.

    For each event, every signal column (feature) is sampled at
    ``event + relative_times``, with ``relative_times = start + dt * j`` for
    every ``j >= 0`` before ``stop``. A value is taken only where the feature
    was observed: at a valid sample's exact time, or between two valid
    samples. An invalid sample is treated as a deleted row, so values are
    interpolated across it between the valid samples on either side. Nothing
    is interpolated across a gap longer than ``max_gap``, a declared break or
    an ``epochs`` edge, and nothing is extrapolated before the first or after
    the last sample. This samples a signal; it does not bin or average it.

    With the default ``window_coverage="complete"``, an event is kept only
    when its whole window ``[event + start, event + stop)`` lies inside one
    continuous run of observation for every feature, so every event
    contributes to every cell. ``window_coverage="partial"`` keeps every event
    with at least one supported cell, and each cell is summarized over the
    events observed there (``counts``).

    Parameters
    ----------
    times : ArrayLike, shape (n_samples,)
        Sample times in seconds; finite and strictly increasing.
    signals : ArrayLike, shape (n_samples, n_features) or (n_samples,)
        One column per feature, on the clock of ``times``. NaN or infinity
        marks a missing sample of that feature; a finite zero is an
        observation.
    event_times : ArrayLike, shape (n_input_events,)
        Event times in seconds, on the same clock. Input order is kept and
        duplicate times are separate events.
    window : tuple of float
        ``(start, stop)`` in seconds relative to each event, half-open,
        ``start < stop``. For example ``(-1.0, 2.0)``.
    dt : float
        Spacing of the query times in seconds. No default: choose the time
        resolution of your analysis, usually no finer than the sampling
        interval.
    feature_ids : sequence, shape (n_features,), optional
        Unique feature labels; default ``0 .. n_features - 1``. The mappings
        below are keyed by these ids.
    feature_units : mapping of feature id to str or None, optional
        Unit of each feature, recorded on the result. Circular features are
        radians: their unit must be ``"rad"`` (the default) or omitted.
    event_ids : sequence, shape (n_input_events,), optional
        Your labels for the events (trial numbers, reward ids), kept for every
        input event.
    value_kinds : mapping of feature id to str, optional
        ``"linear"`` (default), ``"circular"`` (angles in radians, such as
        head direction) or ``"categorical"`` (integer codes, such as the
        occupied zone). Units never imply a kind.
    category_labels : mapping of feature id to mapping of int to str, optional
        Required for every categorical feature: ``{code: label}``, for example
        ``{"zone": {0: "center", 1: "arm"}}``. Code 0 is valid. Every finite
        code in the signal must be listed.
    interpolation : mapping of feature id to str, optional
        How to sample between two samples. Linear features: ``"linear"``
        (default) or ``"previous"``. Circular: ``"circular"`` (default,
        shorter arc) or ``"previous"``. Categorical: ``"previous"`` only (the
        last observed code).
    resultant_threshold : float, default=1e-12
        Circular means are undefined (NaN, ``"undefined_direction"``) where
        the mean resultant length is at or below this value, in ``[0, 1]``.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : None, (start, stop) or array-like of shape (n_epochs, 2), optional
        Half-open ``[start, stop)`` windows in seconds, on the clock of
        ``times``. An interval must lie inside one window to be used.
    valid : ArrayLike of bool, shape (n_samples,) or (n_samples, n_features), optional
        Extra sample validity, for all features (1-D) or per feature (2-D),
        for example False where tracking confidence is low.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1``
        and ``k`` (for example where a new recording starts). Entry 0 is
        ignored.
    window_coverage : {"complete", "partial"}, default="complete"
        Which events to keep; see above.
    retain_aligned : bool, default=True
        Keep the sampled ``values`` and ``support`` of every retained event.
        ``False`` keeps only the summaries, computed in bounded-memory chunks,
        for many events or long windows; the summaries are the same.

    Returns
    -------
    ContinuousEventResult
        Per-event status and overlap flags, per-cell counts and summaries
        (``means``, ``resultant_lengths``, ``category_counts``) and, unless
        ``retain_aligned=False``, the sampled values.

    Raises
    ------
    ValueError
        If ``window`` is not ``start < stop`` in seconds or holds no query
        time, ``dt`` is not a positive finite number, ``window_coverage``,
        ``resultant_threshold`` or ``retain_aligned`` is invalid, the signals,
        feature metadata, ``event_times`` or ``event_ids`` are malformed, or
        every event is excluded.

    Notes
    -----
    Exclusion reasons, one per excluded event in ``event_status``:

    - ``"outside_recording"``: complete coverage, the window is not inside
      ``[times[0], times[-1]]``; partial coverage, the window does not
      overlap it at all.
    - ``"outside_epochs"``: complete coverage, ``epochs`` given and the
      window is not inside one epoch.
    - ``"incomplete_support"``: complete coverage, some feature has no single
      run of observation covering the window.
    - ``"no_support"``: partial coverage, no feature was observed at any
      query time of the window.

    Overlapping windows reuse the same samples; ``overlapping`` flags those
    events so you can decide whether to keep them.

    See Also
    --------
    peri_event_histogram : Spike counts around events.

    Examples
    --------
    Running speed that grows as ``2 * t`` cm/s, sampled at 10 Hz, around two
    rewards. Each mean is the average of the two events' speeds at that
    relative time.

    >>> import numpy as np
    >>> from neurospatial.events import align_continuous_to_events
    >>> times = np.arange(100) / 10
    >>> speed = 2 * times
    >>> result = align_continuous_to_events(
    ...     times, speed, [2.0, 5.0], (-1.0, 1.0), dt=0.5, feature_ids=["speed"]
    ... )
    >>> result.relative_times
    array([-1. , -0.5,  0. ,  0.5])
    >>> result.means[:, 0]
    array([5., 6., 7., 8.])
    >>> result.event_status.tolist()
    ['retained', 'retained']

    Head direction near +pi around one reward and near -pi around the other
    points the same way. The circular mean is pi with a resultant length near
    1, where an arithmetic mean would give 0.

    >>> heading = np.where(times < 3.5, 3.1, -3.1)
    >>> result = align_continuous_to_events(
    ...     times,
    ...     np.column_stack([speed, heading]),
    ...     [2.0, 5.0],
    ...     (-1.0, 1.0),
    ...     dt=0.5,
    ...     feature_ids=["speed", "heading"],
    ...     value_kinds={"heading": "circular"},
    ... )
    >>> np.round(result.means[:, 1], 4)
    array([3.1416, 3.1416, 3.1416, 3.1416])
    >>> np.round(result.resultant_lengths[:, 1], 3)
    array([0.999, 0.999, 0.999, 0.999])
    >>> result.feature_units
    (None, 'rad')
    """
    call = "align_continuous_to_events"
    _validate_window(window, context=call)
    start, stop = float(window[0]), float(window[1])
    if not start < stop:
        raise ValueError(
            _format_error(
                f"{call}: window start ({start}) must be less than stop ({stop}).",
                why="The window is the half-open interval [start, stop) of times "
                "relative to each event, so it needs a positive length.",
                fix="pass window=(-1.0, 2.0), in seconds",
            )
        )
    dt = _positive_number(dt, "dt", call=call)
    relative_times = _relative_times((start, stop), dt)
    if relative_times.size == 0:
        raise ValueError(
            _format_error(
                f"{call}: window={window!r} with dt={dt} holds no query time.",
                why=f"The window is {stop - start:g} s long, far shorter than dt.",
                fix="widen window or use a smaller dt",
            )
        )
    if window_coverage not in _WINDOW_COVERAGES:
        raise ValueError(
            _format_error(
                f"{call}: window_coverage={window_coverage!r} is not supported.",
                why="'complete' keeps events whose whole window was observed; "
                "'partial' keeps every event observed at some query time.",
                fix="pass window_coverage='complete' or window_coverage='partial'",
            )
        )
    if not (
        _is_real_number(resultant_threshold)
        and np.isfinite(resultant_threshold)
        and 0.0 <= resultant_threshold <= 1.0
    ):
        raise ValueError(
            _format_error(
                f"{call}: resultant_threshold={resultant_threshold!r} is not in "
                "[0, 1].",
                why="It is compared with mean resultant lengths, which lie in [0, 1].",
                fix="keep the default 1e-12, or pass a value such as 0.1",
            )
        )
    if not isinstance(retain_aligned, bool | np.bool_):
        raise ValueError(
            _format_error(
                f"{call}: retain_aligned={retain_aligned!r} is not a bool.",
                fix="pass retain_aligned=True to keep the sampled values, or "
                "False to keep only the summaries",
            )
        )

    normalized_epochs = as_intervals(epochs, name="epochs")
    t, sig, supports = validated_signals(
        times,
        signals,
        valid=valid,
        call=call,
        max_gap=max_gap,
        epochs=normalized_epochs,
        break_before=break_before,
        result="no event window can be sampled",
    )
    n_features = sig.shape[1]
    spec = _resolve_features(
        n_features,
        feature_ids=feature_ids,
        feature_units=feature_units,
        value_kinds=value_kinds,
        category_labels=category_labels,
        interpolation=interpolation,
        call=call,
    )
    sample_valid = np.column_stack([support.sample_valid for support in supports])
    _check_category_codes(sig, sample_valid, spec, call=call)
    events = _validated_event_times(event_times, call=call)
    ids = _validated_event_ids(event_ids, events.size, call=call)

    status = _window_status(
        events, (start, stop), t, supports, normalized_epochs, window_coverage
    )
    provisional = np.flatnonzero(status == "retained")
    summaries = _Summaries.empty(relative_times.size, spec)
    kept_values: list[NDArray[np.float64]] = []
    kept_support: list[NDArray[np.bool_]] = []
    for chunk in _event_chunks(provisional.size, relative_times.size * n_features):
        rows = provisional[chunk]
        values, support = _sample_events(
            events[rows], relative_times, t, sig, supports, spec, normalized_epochs
        )
        observed = support.any(axis=(1, 2))
        status[rows[~observed]] = "no_support"
        summaries.add(values, support)
        if retain_aligned:
            kept_values.append(values[observed])
            kept_support.append(support[observed])

    retained = status == "retained"
    if not retained.any():
        reasons = ", ".join(
            f"{np.count_nonzero(status == reason)} {reason}"
            for reason in _EVENT_STATUSES[1:]
            if np.any(status == reason)
        )
        raise ValueError(
            _format_error(
                f"{call}: All {events.size} events were excluded: {reasons}.",
                why="No event window was observed well enough to be sampled.",
                fix="check that event_times and times share a clock and units, "
                "that epochs and max_gap keep the event windows, use "
                "window_coverage='partial' to keep partly observed events, or "
                "narrow window",
            )
        )

    return ContinuousEventResult(
        relative_times=relative_times,
        feature_ids=spec.feature_ids,
        value_kinds=spec.value_kinds,
        interpolation=spec.interpolation,
        feature_units=spec.units,
        event_status=status,
        overlapping=_overlapping(events, (start, stop), retained),
        counts=summaries.counts,
        sums=summaries.sums,
        sin_sums=summaries.sin_sums,
        cos_sums=summaries.cos_sums,
        category_features=spec.category_features,
        category_codes=spec.category_codes,
        category_labels=spec.category_labels,
        category_counts=summaries.category_counts,
        window=(start, stop),
        dt=dt,
        window_coverage=window_coverage,
        resultant_threshold=float(resultant_threshold),
        max_gap=None if max_gap is None else float(max_gap),
        epochs=normalized_epochs,
        values=np.concatenate(kept_values) if retain_aligned else None,
        support=np.concatenate(kept_support) if retain_aligned else None,
        event_ids=ids,
    )


def _is_real_number(value: Any) -> bool:
    """True for one real number; False for bools, arrays, strings and None."""
    return isinstance(value, Real) and not isinstance(value, bool | np.bool_)


def _positive_number(value: Any, name: str, *, call: str) -> float:
    """Return ``value`` as a float, or raise unless it is positive and finite."""
    if not (_is_real_number(value) and np.isfinite(value) and value > 0):
        raise ValueError(
            _format_error(
                f"{call}: {name} must be a positive finite number of seconds; "
                f"got {value!r}.",
                why=f"{name} is the spacing of the query times around each event.",
                fix=f"pass {name}=0.02 for 20 ms steps",
            )
        )
    return float(value)


def _validated_event_times(event_times: ArrayLike, *, call: str) -> NDArray[np.float64]:
    """Return event times as a 1-D finite float array, or raise."""
    fix = "pass a 1-D array of event times in seconds, e.g. reward_times"
    try:
        events = np.asarray(event_times, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            _format_error(f"{call}: event_times must be numbers in seconds.", fix=fix)
        ) from None
    if events.ndim != 1 or events.size == 0:
        raise ValueError(
            _format_error(
                f"{call}: event_times must be a non-empty 1-D array; got shape "
                f"{events.shape}.",
                fix=fix,
            )
        )
    bad = np.flatnonzero(~np.isfinite(events))
    if bad.size:
        raise ValueError(
            _format_error(
                f"{call}: event_times has {bad.size} non-finite value(s), first at "
                f"index {bad[0]} ({events[bad[0]]}).",
                why="An event without a time has no window to sample.",
                fix="drop those events, e.g. event_times[np.isfinite(event_times)]",
            )
        )
    return events


def _validated_event_ids(
    event_ids: Sequence[Any] | NDArray[Any] | None, n_events: int, *, call: str
) -> NDArray[Any] | None:
    """Return the event labels as an array of length ``n_events``, or None."""
    if event_ids is None:
        return None
    if isinstance(event_ids, str | bytes) or np.ndim(event_ids) == 0:
        raise ValueError(
            _format_error(
                f"{call}: event_ids must be a sequence of labels, one per event; "
                f"got {event_ids!r}.",
                why="A single string or number would be split or cannot label "
                "each event.",
                fix="pass a list or array, e.g. event_ids=['trial_1', 'trial_2']",
            )
        )
    ids = as_label_array(event_ids)
    if ids.shape != (n_events,):
        raise ValueError(
            _format_error(
                f"{call}: got event_ids of shape {ids.shape} for {n_events} "
                "event_times.",
                why="event_ids label every input event, retained or not.",
                fix="pass one label per entry of event_times",
            )
        )
    return ids
