"""Backend-neutral sampling and scientific-panel data for a prepared scene.

The scene owns its streams and support metadata. This interpreter retains those
sources and derives only the requested time window; it owns no canvas or reader.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import chain
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.animation._sampling import sample_observation
from neurospatial.animation._style import _heading_displacement
from neurospatial.animation._tracks import (
    _project,
    _projection,
    track_extents,
    track_position,
)

if TYPE_CHECKING:
    from neurospatial.animation.scene import Animation, AnimationField


class _EventRecords(NamedTuple):
    """One event type's located events in time order, with their frames."""

    times: NDArray[np.float64]
    frames: NDArray[np.int_]
    positions: NDArray[np.float64]


class VisibleEvents(NamedTuple):
    """The events of one type shown at a time.

    Attributes
    ----------
    positions : ndarray of float64, shape (n_shown, n_dims)
        Their locations, in time order.
    opacity : ndarray of float64, shape (n_shown,)
        Each one's opacity: the overlay's, faded with its age.
    key : tuple of int
        Changes exactly when ``positions`` or ``opacity`` change, so a view
        updates its markers only then.
    """

    positions: NDArray[np.float64]
    opacity: NDArray[np.float64]
    key: tuple[int, ...]


class TrackWindow(NamedTuple):
    """Finite supported cells and observed position segments in one time window."""

    polygons: np.ndarray
    values: np.ndarray
    positions: list[tuple[np.ndarray, str]]


def _ordered_clock_chunks(clock: np.ndarray) -> Iterator[np.ndarray]:
    """Search point/interval records without sorting a whole recording copy."""
    for start in range(0, len(clock), 65_536):
        chunk = clock[start : start + 65_536]
        # Event records need not be chronological, including adapter outputs.
        # Already ordered clocks remain views of their authoritative arrays.
        yield chunk if np.all(chunk[1:] >= chunk[:-1]) else np.sort(chunk)


def iter_observation_clocks(scene: Animation) -> Iterator[np.ndarray]:
    """Yield prepared source clocks for export's ULP-only time correction.

    Read converted streams so custom adapters' clocks agree with rendering. No
    renderer, source values, decoder reads or combined recording-sized clock
    are needed. Point and interval records may be unordered; search bounded
    ordered chunks without reordering the records.
    """
    for field in scene.fields.values():
        yield field.times
    data = scene._overlay_data
    for item in chain(data.sampled(), data.timeseries):
        yield item.times
    for events in data.events:
        for clock in events.event_times.values():
            yield from _ordered_clock_chunks(clock)
    for raster in data.spike_rasters:
        for clock in raster.spike_times:
            yield from _ordered_clock_chunks(clock)
    for epochs in data.epochs:
        for column in range(2):
            yield from _ordered_clock_chunks(epochs.intervals[:, column])


@dataclass(frozen=True)
class FieldScale:
    """How one field's values become colors, the same in every view.

    Attributes
    ----------
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float
        Values at the colormap's ends; colors clip values outside them.
    label : str
        Colorbar label.
    extend : tuple of bool
        Whether sampled values fall below ``vmin`` and above ``vmax``.
        Colorbars mark each clipped end.
    opaque_from : float or None
        With a video below the field, low values are see-through: opacity
        rises from zero at ``vmin`` to one at this value, the typical
        frame's peak (``vmax`` at the latest). None draws every value
        opaque.
    """

    cmap: str
    vmin: float
    vmax: float
    label: str
    extend: tuple[bool, bool] = (False, False)
    opaque_from: float | None = None

    @property
    def extend_name(self) -> str:
        """``extend`` as Matplotlib's colorbar option."""
        below, above = self.extend
        return {
            (False, False): "neither",
            (True, False): "min",
            (False, True): "max",
            (True, True): "both",
        }[below, above]

    def colors(self, n_colors: int = 256) -> NDArray[np.float64]:
        """RGBA colors from ``vmin`` to ``vmax``, shape (n_colors, 4).

        The opacity is linear in the displayed (sRGB) color, as Matplotlib
        and napari blend layers.
        """
        from neurospatial.animation._style import field_colors

        return field_colors(
            self.cmap, self.vmin, self.vmax, self.opaque_from, n_colors=n_colors
        )


def field_scale(
    field: AnimationField,
    label: str,
    *,
    see_through: bool = False,
    projection: Any = None,
    limits: tuple[float, float] | None = None,
) -> FieldScale:
    """The color scale of ``field``, or of its projection onto a track.

    Statistics come from at most 100 of the field's supported rows, read once
    and cached with the field (see ``AnimationField.color_limits``).

    Parameters
    ----------
    field : AnimationField
        The field.
    label : str
        Colorbar label.
    see_through : bool, default=False
        Whether low values fade, for a video drawn below the field.
    projection : sparse matrix, optional
        Bins summed into track bins (``_tracks._projection``). The field's
        own ``vmin`` and ``vmax`` are in other units and do not apply.
    limits : tuple of float, optional
        Explicit ``(vmin, vmax)``; None uses the field's limits.
    """
    from neurospatial.animation._field_stats import robust_limits, typical_peak

    lows, highs = field._scale_extremes(projection)
    if limits is None:
        limits = (
            robust_limits(lows, highs)
            if projection is not None
            else field.color_limits()
        )
    vmin, vmax = limits
    peak = typical_peak(highs)
    return FieldScale(
        field.cmap,
        float(vmin),
        float(vmax),
        label,
        (bool(np.any(lows < vmin)), bool(np.any(highs > vmax))),
        (peak if peak > 0 else 1.0) if see_through else None,
    )


def signal_groups(signals: list[Any]) -> dict[tuple[str, str | int], list[Any]]:
    """Signals sharing a ``group`` share a panel; others get their own.

    Panels follow the order in which their first signal was listed.
    """
    groups: dict[tuple[str, str | int], list[Any]] = {}
    for i, signal in enumerate(signals):
        key = ("group", signal.group) if signal.group is not None else ("signal", i)
        groups.setdefault(key, []).append(signal)
    return groups


class PreparedScene:
    """Interpret authoritative scene streams without allocating a renderer."""

    def __init__(self, scene: Animation) -> None:
        from neurospatial.animation._capabilities import require_scene_rendering

        require_scene_rendering(scene)
        self.scene = scene
        self.source_lock = scene._source_lock
        # Bound overlapping-window retention by both row count and bytes. A
        # single source row can exceed the byte budget; it is then not retained.
        self._rows: OrderedDict[tuple[int, int, bool], np.ndarray] = OrderedDict()
        self._row_bytes = 0
        self._row_budget = 32 * 1024 * 1024
        self._held: dict[int, tuple[int, np.ndarray]] = {}
        self._missing: dict[int, np.ndarray] = {}
        self._video_held: dict[int, tuple[int, np.ndarray | None]] = {}
        self._raster_clocks = {
            (id(raster), row): (
                spikes if np.all(spikes[1:] >= spikes[:-1]) else np.sort(spikes)
            )
            for raster in scene._overlay_data.spike_rasters
            for row, spikes in enumerate(raster.spike_times)
        }
        # Events with a location, in time order: those shown at any time are
        # one slice of them.
        self._event_records: dict[tuple[int, str], _EventRecords] = {}
        for events in scene._overlay_data.events:
            for name, positions in events.event_positions.items():
                times = events.event_times.get(name)
                if times is None:
                    times = scene.frame_times[events.event_frame_indices[name]]
                located = np.flatnonzero(np.isfinite(positions).all(axis=1))
                order = located[np.argsort(times[located], kind="stable")]
                self._event_records[id(events), name] = _EventRecords(
                    times[order],
                    np.maximum(
                        np.searchsorted(scene.frame_times, times[order], "right") - 1,
                        0,
                    ),
                    positions[order],
                )
        data = scene._overlay_data
        # Without exactly one position, heading arrows start at the arena's center.
        self._arena_center = np.mean(scene.env.bin_centers, axis=0)
        self.signal_groups = signal_groups(data.timeseries)
        self.track_env = scene.env if scene.track is None else scene.track
        self.track_extents = track_extents(self.track_env)
        self.projection = (
            None if scene.track is None else _projection(scene.env, scene.track)
        )
        self.panel_count = (
            len(self.signal_groups)
            + len(data.spike_rasters)
            + bool(data.epochs)
            + (self.track_extents is not None)
        )
        below = any(video.z_order == "below" for video in data.videos)
        with self.source_lock:
            # One scale per field and per projected track, for every view.
            self.field_scales = {
                name: field_scale(field, field.label or name, see_through=below)
                for name, field in scene.fields.items()
            }
            self.track_scales = {
                name: (
                    replace(scale, opaque_from=None)
                    if self.projection is None
                    else field_scale(
                        scene.fields[name],
                        f"Projected {scale.label} (sum per track bin)",
                        projection=self.projection,
                    )
                )
                for name, scale in self.field_scales.items()
            }

    def clear_cache(self) -> None:
        """Release view-local row and decoder buffers after pending reads drain."""
        self._rows.clear()
        self._row_bytes = 0
        self._held.clear()
        self._missing.clear()
        self._video_held.clear()

    def _field_row(
        self, field: AnimationField, index: int, *, projected: bool = False
    ) -> np.ndarray:
        key = (id(field), index, projected)
        if key in self._rows:
            self._rows.move_to_end(key)
            return self._rows[key]
        values: np.ndarray
        if projected and self.projection is not None:
            values = _project(self._field_row(field, index)[None, :], self.projection)[
                0
            ]
        else:
            with self.source_lock:
                # AnimationField owns the sampled row before another source
                # read can overwrite a lazy adapter's scratch buffer.
                values = cast("np.ndarray", field[index])
        if values.nbytes <= self._row_budget:
            while self._rows and (
                len(self._rows) >= 4096
                or self._row_bytes + values.nbytes > self._row_budget
            ):
                _, expired = self._rows.popitem(last=False)
                self._row_bytes -= expired.nbytes
            self._rows[key] = values
            self._row_bytes += values.nbytes
        return values

    def field_samples(self, time: float) -> dict[str, np.ndarray]:
        """Hold one row per stream without rereading a slow source each tick."""
        return {
            name: self.field_sample(field, time)
            for name, field in self.scene.fields.items()
        }

    def field_sample(self, field: AnimationField, time: float) -> np.ndarray:
        """The row ``field`` shows at ``time``, held until its index changes.

        The returned array is shared between calls; do not modify it.
        """
        index = field._sample_index(time)
        if index is None:
            if id(field) not in self._missing:
                self._missing[id(field)] = np.full(field.n_bins, np.nan)
            return self._missing[id(field)]
        previous = self._held.get(id(field))
        if previous is None or previous[0] != index:
            previous = (index, self._field_row(field, index))
            self._held[id(field)] = previous
        return previous[1]

    def trail_seconds(self, position: Any) -> float:
        """How far back a position's trail reaches: ``trail_length`` scene frames.

        Measured in the scene's typical frame interval, as fastplotlib does,
        so every view draws the same stretch of recent path whatever the
        tracking rate.
        """
        frames = self.scene.frame_times
        interval = float(np.median(np.diff(frames))) if len(frames) > 1 else 0.0
        return float(position.trail_length or 0) * interval

    def trail_rows(self, position: Any, time: float) -> range:
        """The position's own samples in its trail at ``time``."""
        times = position.times
        stop = int(np.searchsorted(times, time, side="right"))
        start = int(
            np.searchsorted(times, time - self.trail_seconds(position), side="left")
        )
        return range(min(start, stop), stop)

    def trail(self, position: Any, time: float) -> tuple[list[np.ndarray], np.ndarray]:
        """The position's trail at ``time``: segments and their opacities.

        Segments join consecutive supported samples, and none is drawn while
        the current position is missing. Each fades with the age of its newer
        sample, from ``_TRAIL_MAX_ALPHA`` now to zero ``trail_seconds`` ago.
        """
        from neurospatial.animation._style import _TRAIL_MAX_ALPHA

        point = self.sample(position, position.data, time)
        if not np.isfinite(point).all():
            return [], np.empty(0)
        # A valid interval joins two valid, finite samples.
        rows = [
            j
            for j in self.trail_rows(position, time)[:-1]
            if position.interval_valid[j]
        ]
        segments = [position.data[j : j + 2] for j in rows]
        seconds = self.trail_seconds(position)
        if seconds <= 0:
            return segments, np.full(len(rows), _TRAIL_MAX_ALPHA)
        age = time - position.times[np.asarray(rows, dtype=int) + 1]
        return segments, _TRAIL_MAX_ALPHA * np.clip(1 - age / seconds, 0, 1)

    def trail_capacity(self, position: Any) -> int:
        """The most samples any trail of ``position`` holds."""
        times = position.times
        starts = np.searchsorted(times, times - self.trail_seconds(position), "left")
        return int(np.max(np.arange(1, len(times) + 1) - starts, initial=0))

    def sample(
        self,
        item: Any,
        values: np.ndarray,
        time: float,
        *,
        circular: bool = False,
        name: str | None = None,
    ) -> np.ndarray:
        """Sample a prepared stream using its existing clock and support masks."""
        return sample_observation(item, values, time, circular=circular, name=name)

    def events(self, events: Any, time: float) -> dict[str, VisibleEvents]:
        """Each event type's events shown at ``time``, with their opacity.

        An event shows from its own time on the frame that holds it; with
        ``decay_frames`` it fades over that many scene frames. Persistence
        counts frames of the complete scene clock, so a selected export does
        not reset an earlier event's age. Events without a location are
        never shown.
        """
        from neurospatial.animation._style import _event_visibility

        frame = int(np.searchsorted(self.scene.frame_times, time, side="right")) - 1
        decay = events.decay_frames
        shown = {}
        for name in events.event_positions:
            records = self._event_records[id(events), name]
            stop = int(np.searchsorted(records.times, time, side="right"))
            start = (
                0
                if decay is None
                else min(
                    int(np.searchsorted(records.frames, frame - decay, side="left")),
                    stop,
                )
            )
            key: tuple[int, ...] = (start, stop)
            if decay:
                # Fading events change opacity with every frame.
                _, alphas = _event_visibility(records.frames[start:stop], frame, decay)
                opacity, key = alphas * events.opacity, (start, stop, frame)
            else:
                opacity = np.full(stop - start, float(events.opacity))
            shown[name] = VisibleEvents(records.positions[start:stop], opacity, key)
        return shown

    def heading_arrow(
        self, heading: Any, time: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """The heading arrow at ``time``: origin and displacement, or None.

        The arrow starts at the position overlay's sample when the scene has
        exactly one, otherwise at the arena's center, and is
        ``heading.length`` long in environment coordinates. None when the
        heading or its origin is missing, or the heading has no direction.
        """
        positions = self.scene._overlay_data.positions
        origin = (
            self.sample(positions[0], positions[0].data, time)
            if len(positions) == 1
            else self._arena_center
        )
        value = self.sample(
            heading, heading.data, time, circular=heading.data.ndim == 1
        )
        direction = _heading_displacement(value, heading.length, n_dims=len(origin))
        if direction is None or not np.isfinite(origin).all():
            return None
        return origin, direction

    def video_frame(self, video: Any, time: float) -> np.ndarray | None:
        """Read the nearest supported source row through its authoritative map."""
        times = video.times
        if not len(times) or time < times[0] or time > times[-1]:
            return None
        right = int(np.searchsorted(times, time, side="left"))
        index = (
            right - 1
            if right > 0 and time - times[right - 1] <= times[right] - time
            else right
        )
        previous = self._video_held.get(id(video))
        if previous is None or previous[0] != index:
            with self.source_lock:
                image = video.get_frame(index)
                image = None if image is None else np.asarray(image).copy()
            # Decoder buffers may be reused by the next read. A displayed
            # frame must remain stable while the worker prepares its successor.
            previous = (index, image)
            self._video_held[id(video)] = previous
        return previous[1]

    def bounds(self, time: float) -> tuple[float, float]:
        half = self.scene.window_seconds / 2
        return time - half, time + half

    def signal_window(self, signal: Any, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Observed signal samples in the current window, with requested scaling."""
        left, right = self.bounds(time)
        clock = signal.times
        keep = slice(
            np.searchsorted(clock, left, side="left"),
            np.searchsorted(clock, right, side="right"),
        )
        # Conversion owns normalization; prepared values are final for every
        # consumer, including custom adapters with separate display metadata.
        return clock[keep], signal.data[keep]

    def signal_limits(self, signals: list[Any]) -> tuple[float, float]:
        """The y-limits of a panel of ``signals``: their whole recording.

        Fixed limits keep the panel still during playback. A near-zero floor
        reads as zero and a constant signal gets half a unit either side
        (:func:`~neurospatial.animation._field_stats.tidy_range`).
        Normalized signals span [0, 1].
        """
        from neurospatial.animation._field_stats import tidy_range

        return tidy_range(
            min(signal.global_vmin for signal in signals),
            max(signal.global_vmax for signal in signals),
        )

    def raster_window(self, raster: Any, row: int, time: float) -> np.ndarray:
        """One unit's spike times in the current window, preserving unit order."""
        left, right = self.bounds(time)
        spikes = self._raster_clocks[id(raster), row]
        return cast(
            "np.ndarray",
            spikes[
                np.searchsorted(spikes, left, side="left") : np.searchsorted(
                    spikes, right, side="right"
                )
            ],
        )

    def track_window(self, field: AnimationField, time: float) -> TrackWindow:
        """Derive cells only for authoritative supported native field intervals."""
        assert self.track_extents is not None
        starts, stops = self.track_extents
        left, right = self.bounds(time)
        first = max(int(np.searchsorted(field.times, left, side="right")) - 1, 0)
        stop = min(
            int(np.searchsorted(field.times, right, side="left")), len(field) - 1
        )
        selected = np.arange(first, max(first, stop))
        selected = selected[
            field.interval_valid[selected] & field.sample_valid[selected]
        ]
        polygons, colors = [], []
        for index in selected:
            values = self._field_row(
                field, int(index), projected=self.projection is not None
            )
            lo, hi = max(left, field.times[index]), min(right, field.times[index + 1])
            finite = np.isfinite(values)
            bottom, top = starts[finite], stops[finite]
            cells = np.empty((len(bottom), 4, 2))
            cells[:, :, 0] = [lo, hi, hi, lo]
            cells[:, :, 1] = np.column_stack([bottom, bottom, top, top])
            polygons.append(cells)
            colors.append(values[finite])
        paths = []
        for position in self.scene._overlay_data.positions:
            times = position.times
            start = int(np.searchsorted(times, left, side="left"))
            stop = int(np.searchsorted(times, right, side="right"))
            distances = track_position(self.track_env, position.data[start:stop])
            for index in range(start, stop - 1):
                if position.interval_valid[index]:
                    segment = np.column_stack(
                        [
                            times[index : index + 2],
                            distances[index - start : index - start + 2],
                        ]
                    )
                    if np.isfinite(segment).all():
                        paths.append((segment, position.color))
        return TrackWindow(
            np.concatenate(polygons) if polygons else np.empty((0, 4, 2)),
            np.concatenate(colors) if colors else np.empty(0),
            paths,
        )
