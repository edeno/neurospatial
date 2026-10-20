"""Identified event selection and annotation on an animation's clock."""

from __future__ import annotations

import copy
import html
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from neurospatial.animation._view import AnimationView
    from neurospatial.animation.scene import Animation


def _copy_event_table(table: pd.DataFrame) -> pd.DataFrame:
    """Own nested metadata without asking pandas to infer column types again."""
    result = table.copy(deep=True).reset_index(drop=True)
    for column, dtype in enumerate(result.dtypes):
        if pd.api.types.is_object_dtype(dtype):
            for row in range(len(result)):
                result.iat[row, column] = copy.deepcopy(result.iat[row, column])
    result.attrs = copy.deepcopy(table.attrs)
    return result


def _id_scalar_family(value: Any) -> str | None:
    """Allow native and NumPy JSON scalars, keeping bool distinct from int."""
    if isinstance(value, (str, np.str_)):
        return "string"
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (int, np.integer)):
        return "integer"
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return "real"
    return None


class EventInspector:
    """Browse event windows without merging intervals or losing event IDs.

    Parameters
    ----------
    animation : Animation
        The full recording, including fields and synchronized overlays.
    events : pandas.DataFrame or sequence of mappings
        One row per identified event, with start/stop recording times in
        seconds. Additional columns (scores, trial type, etc.) are retained.
        Overlapping and adjacent events remain distinct.
    id_column, start_column, stop_column : str
        Column names, defaulting to ``event_id``, ``start_time``, ``stop_time``.
    padding : float, default=0
        Seconds before and after an event to include, clipped to the recording.

    Notes
    -----
    ``events_table()`` returns an independent table. Pass a Boolean mask on
    that table to ``filter_events``. Filtering changes visibility, never IDs
    or annotations. ``annotate`` stores columns with an ``annotation_``
    prefix so a review cannot overwrite an analysis score.
    """

    def __init__(
        self,
        animation: Animation,
        events: pd.DataFrame | Sequence[Mapping[str, Any]],
        *,
        id_column: str = "event_id",
        start_column: str = "start_time",
        stop_column: str = "stop_time",
        padding: float = 0.0,
    ) -> None:
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("padding must be nonnegative finite seconds.")
        columns = (id_column, start_column, stop_column)
        if len(set(columns)) != 3:
            raise ValueError("ID, start and stop columns must have different names.")
        if isinstance(events, pd.DataFrame):
            table = _copy_event_table(events)
        else:
            # Give IDs an explicit object dtype before pandas can coalesce a
            # mixed integer/float column and round a large integer ID.
            records = list(events)
            data: dict[str, Any] = {
                name: [record.get(name) for record in records]
                for name in dict.fromkeys(name for record in records for name in record)
            }
            if id_column in data:
                data[id_column] = pd.Series(data[id_column], dtype=object)
            table = _copy_event_table(pd.DataFrame(data))
        missing = set(columns) - set(table.columns)
        if missing:
            raise ValueError(f"Event records are missing columns: {sorted(missing)}.")
        ids = table[id_column]
        try:
            valid_ids = not ids.isna().any() and ids.is_unique
            for value in ids:
                hash(value)
        except TypeError as exc:
            raise ValueError(
                "Event IDs must be unique, nonmissing, hashable values."
            ) from exc
        if not valid_ids:
            raise ValueError("Event IDs must be unique and nonmissing.")
        try:
            starts = table[start_column].to_numpy(dtype=float)
            stops = table[stop_column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Event start and stop times must be numeric seconds."
            ) from exc
        if not np.isfinite([starts, stops]).all() or np.any(starts >= stops):
            raise ValueError("Each event needs finite start_time < stop_time.")
        bounds = animation.frame_times[[0, -1]]
        if np.any(stops <= bounds[0]) or np.any(starts >= bounds[1]):
            raise ValueError(
                "Each event must overlap the animation's recording clock; "
                "remove events outside the recording before inspection."
            )
        self.animation = animation
        self.id_column, self.start_column, self.stop_column = columns
        self.padding = float(padding)
        self._table = table
        self._rows_by_id = {value: index for index, value in enumerate(ids)}
        self._visible = np.ones(len(table), dtype=bool)
        self._selected: Any = ids.iloc[0] if len(ids) else None
        self._view: AnimationView | None = None
        self._controls: list[Any] = []
        self._observers: list[tuple[Any, Any]] = []
        self._refresh: Any = None

    @property
    def selected_id(self) -> Any:
        """The selected event's original ID, or None for no visible events."""
        return self._selected

    @property
    def view(self) -> AnimationView | None:
        """The current interactive view, if ``show`` has been called."""
        return self._view

    def events_table(self, *, visible_only: bool = False) -> pd.DataFrame:
        """Return independent event records, analysis metadata and annotations."""
        rows = self._table.loc[self._visible] if visible_only else self._table
        return _copy_event_table(rows)

    def filter_events(self, mask: ArrayLike) -> EventInspector:
        """Show rows selected by a Boolean mask on the full ``events_table``."""
        selected = np.asarray(mask)
        if selected.dtype.kind != "b" or selected.shape != (len(self._table),):
            raise ValueError(
                "mask must be one Boolean per row of events_table(), "
                "including currently hidden events."
            )
        self._visible = selected.copy()
        ids = self._table.loc[self._visible, self.id_column].tolist()
        if self._selected not in ids:
            self._selected = ids[0] if ids else None
        if self._selected is not None and self._view is not None:
            self.select(self._selected)
        self._update_controls()
        return self

    def reset_filter(self) -> EventInspector:
        """Make all original event rows visible."""
        return self.filter_events(np.ones(len(self._table), dtype=bool))

    def _row_index(self, event_id: Any) -> int:
        try:
            return self._rows_by_id[event_id]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"Unknown event ID {event_id!r}.") from exc

    def select(self, event_id: Any) -> EventInspector:
        """Select an event by its original ID and seek the view to its window."""
        row = self._row_index(event_id)
        if not self._visible[row]:
            raise ValueError(
                f"Event {event_id!r} is filtered out; call reset_filter first."
            )
        self._selected = event_id
        if self._view is not None:
            start, stop = self.time_range()
            self._view.select_time(start, stop)
            self._view.seek(start)
        self._update_controls()
        return self

    def time_range(self) -> tuple[float, float]:
        """Selected event window, including padding and clipped to the scene."""
        if self._selected is None:
            raise ValueError("No event is selected. Select a visible event first.")
        row = self._table.iloc[self._row_index(self._selected)]
        bounds = self.animation.frame_times[[0, -1]]
        return (
            max(float(bounds[0]), float(row[self.start_column]) - self.padding),
            min(float(bounds[1]), float(row[self.stop_column]) + self.padding),
        )

    def annotate(self, event_id: Any, **annotations: Any) -> None:
        """Attach review metadata by ID, retained even while an event is hidden."""
        row = self._row_index(event_id)
        for name, value in annotations.items():
            if not name:
                raise ValueError("Annotation names must be nonempty.")
            column = f"annotation_{name}"
            if column not in self._table:
                self._table[column] = pd.Series([None] * len(self._table), dtype=object)
            elif self._table[column].dtype != object:
                self._table[column] = self._table[column].astype(object)
            self._table.at[row, column] = copy.deepcopy(value)
        self._update_controls()

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        """Export the selected event with the scene's complete panels."""
        if "time_range" in kwargs:
            raise TypeError(
                "EventInspector.export uses the selected event's time range."
            )
        return self.animation.export(path, time_range=self.time_range(), **kwargs)

    def export_events(
        self,
        path: str | Path,
        *,
        visible_only: bool = False,
        selected_only: bool = False,
        overwrite: bool = False,
    ) -> Path:
        """Save event IDs, intervals, scores and annotations as CSV or table JSON.

        JSON uses pandas' ``orient='table'`` schema, readable with
        ``pd.read_json(path, orient='table')``. IDs must round-trip with their
        exact value and scalar family (string, Boolean, integer or finite
        real; NumPy equivalents are accepted). Composite IDs, mixed numeric
        columns that pandas coerces, and floats that lose precision are
        rejected before opening the file. Encode such IDs explicitly as
        strings for JSON, or retain them in memory with ``events_table()``.
        CSV accepts composite IDs but does not preserve their Python types.
        """
        path = Path(path)
        if path.suffix.lower() not in {".csv", ".json"}:
            raise ValueError("Use a .csv or .json path for event records.")
        table = self.events_table(visible_only=visible_only)
        if selected_only:
            if self._selected is None:
                raise ValueError("No event is selected.")
            table = table.loc[table[self.id_column].isin([self._selected])]
        # Serialize and check identity before opening or truncating the file.
        if path.suffix.lower() == ".csv":
            serialized = table.to_csv(index=False)
        else:
            error = (
                "Event IDs cannot round-trip exactly through pandas table JSON. "
                "Use string, Boolean, integer or finite real IDs whose values "
                "and scalar types survive pd.read_json(..., orient='table'); "
                "encode composite IDs explicitly as strings or retain them "
                "with events_table()."
            )
            ids = table[self.id_column].tolist()
            if any(_id_scalar_family(value) is None for value in ids):
                raise ValueError(error)
            try:
                serialized = table.to_json(
                    orient="table", index=False, double_precision=15
                )
                restored_ids = pd.read_json(StringIO(serialized), orient="table")[
                    self.id_column
                ].tolist()
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(error) from exc
            if len(ids) != len(restored_ids) or any(
                _id_scalar_family(original) != _id_scalar_family(restored)
                or original != restored
                for original, restored in zip(ids, restored_ids, strict=True)
            ):
                raise ValueError(error)
        # Exclusive creation prevents a race from overwriting another review.
        with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
            stream.write(serialized)
        return path

    def show(self, *, backend: str = "widget", **kwargs: Any) -> Any:
        """Return notebook controls for event selection and review annotations.

        Selecting an event updates every panel and its export interval.
        ``view`` exposes the common controls and ``view.native`` the backend.
        """
        import ipywidgets as widgets

        if backend != "widget":
            raise ValueError(
                "EventInspector.show uses notebook controls; choose backend='widget'. "
                "Use Animation.show(backend='fastplotlib') for native GPU controls."
            )
        self.close()
        self._view = self.animation.show(backend=backend, **kwargs)
        selector = widgets.Select(description="Event", rows=8)
        decision = widgets.Dropdown(
            options=["unreviewed", "accepted", "rejected"], description="Review"
        )
        note = widgets.Textarea(description="Note")
        details = widgets.HTML()
        updating = False

        def refresh() -> None:
            nonlocal updating
            updating = True
            try:
                ids = self._table.loc[self._visible, self.id_column].tolist()
                selector.options = [(str(value), value) for value in ids]
                selector.value = self._selected
                decision.disabled = note.disabled = self._selected is None
                if self._selected is not None:
                    row = self._table.iloc[self._row_index(self._selected)]
                    status = row.get("annotation_review")
                    decision.value = (
                        status
                        if isinstance(status, str) and status in decision.options
                        else "unreviewed"
                    )
                    annotation_note = row.get("annotation_note")
                    note.value = "" if annotation_note is None else str(annotation_note)
                    details.value = f"<pre>{html.escape(row.to_string())}</pre>"
                else:
                    details.value = "<p>No events match the filter.</p>"
            finally:
                updating = False

        def selected(change: dict[str, Any]) -> None:
            if not updating and change["new"] is not None:
                self.select(change["new"])

        def annotated(change: dict[str, Any]) -> None:
            if not updating and self._selected is not None:
                self.annotate(self._selected, review=decision.value, note=note.value)

        self._observers = [
            (selector, selected),
            (decision, annotated),
            (note, annotated),
        ]
        for control, callback in self._observers:
            control.observe(callback, names="value")
        native = self._view.native
        controls = widgets.HBox([selector, widgets.VBox([decision, note, details])])
        container = widgets.VBox([controls, native])
        shown_view = self._view
        close_container = container.close

        def close() -> None:
            if self._view is shown_view:
                self.close()
            else:
                close_container()

        container.close = close
        self._controls = [
            selector,
            decision,
            note,
            details,
            controls.children[1],
            controls,
            container,
        ]
        self._refresh = refresh

        def view_closed() -> None:
            if self._view is shown_view:
                self.close()

        shown_view._add_close_callback(view_closed)
        try:
            refresh()
            if self._selected is not None:
                self.select(self._selected)
        except Exception:
            self.close()
            raise
        return container

    def _update_controls(self) -> None:
        if self._refresh is not None:
            self._refresh()

    def close(self) -> None:
        """Close this inspector's view and controls; keep the scene available."""
        from neurospatial.animation._resources import _cleanup_error

        self._refresh = None
        view, self._view = self._view, None
        controls, self._controls = self._controls, []
        observers, self._observers = self._observers, []
        failures: list[BaseException] = []
        for control, callback in observers:
            try:
                control.unobserve(callback, names="value")
            except BaseException as error:
                failures.append(error)
        cleanup = ([] if view is None else [view.close]) + [
            control.close for control in controls
        ]
        for close in cleanup:
            try:
                close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise _cleanup_error("Event inspector cleanup failed.", failures)
