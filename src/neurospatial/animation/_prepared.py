"""Backend-neutral display rules, sampling and panel data for a prepared scene.

Every view draws a scene through a PreparedScene, which decides what is shown
at a time: color scales, trails, heading arrows, poses, shown events, signal
limits and the windows of the time panels. The scene owns its streams and
support metadata; this interpreter retains those sources and derives only the
requested time window. It owns no canvas or reader, and returns arrays in
environment coordinates for each view to draw.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import chain
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.animation._sampling import sample_observation, sample_parts
from neurospatial.animation._style import _heading_displacement
from neurospatial.animation._tracks import (
    _project,
    _projection,
    track_extents,
    track_position,
    track_segments,
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
    rows : slice
        Their rows of ``PreparedScene.event_positions``, for a view that
        draws every event once and sets each one's opacity.
    """

    positions: NDArray[np.float64]
    opacity: NDArray[np.float64]
    key: tuple[int, ...]
    rows: slice


class Pose(NamedTuple):
    """A pose at one time.

    Attributes
    ----------
    points : ndarray of float64, shape (n_parts, n_dims)
        Each body part, in ``bodyparts`` order; NaN where missing.
    skeleton : ndarray of float64, shape (n_edges, 2, n_dims)
        The skeleton edges whose two parts are both observed.
    """

    points: NDArray[np.float64]
    skeleton: NDArray[np.float64]


class _PoseStreams(NamedTuple):
    """A bodypart overlay's parts with their support stacked by part."""

    parts: list[NDArray[np.float64]]
    sample_valid: NDArray[np.bool_]
    interval_valid: NDArray[np.bool_]
    edges: NDArray[np.intp]


class TrackWindow(NamedTuple):
    """A field along the track in one time window, and the observed path.

    The field is a mesh of cells, one row per track interval and one column
    per time interval.

    Attributes
    ----------
    time_edges : ndarray of float64, shape (n_columns + 1,)
        Column edges, in seconds, clipped to the window.
    observed : ndarray of bool, shape (n_columns,)
        Whether a column holds a sample; False for the empty column between
        two shown intervals.
    track_edges : ndarray of float64, shape (n_rows + 1,)
        Row edges, in position along the track.
    values : ndarray of float64, shape (n_rows, n_columns)
        Each cell's value: the field's held sample across a supported
        interval. NaN where nothing was observed: an unsupported interval,
        a missing bin, or the spacing between track segments.
    paths : ndarray of float64, shape (n_segments, 2, 2)
        ``(time, position along the track)`` segments joining consecutive
        supported position samples on the same segment of the track.
    path_colors : list of str
        Each segment's position-overlay color.
    """

    time_edges: NDArray[np.float64]
    observed: NDArray[np.bool_]
    track_edges: NDArray[np.float64]
    values: NDArray[np.float64]
    paths: NDArray[np.float64]
    path_colors: list[str]


def _track_rows(
    starts: NDArray[np.float64], stops: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Mesh rows for track bins: their edges, and each bin's row.

    Bins are ordered by position along the track; the spacing between two
    segments is an empty row of its own.
    """
    order = np.argsort(starts, kind="stable")
    low, high = starts[order], stops[order]
    gap = low[1:] > high[:-1]
    rows = np.arange(len(low)) + np.r_[0, np.cumsum(gap)]
    edges = np.empty(len(low) + int(gap.sum()) + 1)
    edges[rows] = low
    edges[rows + 1] = high
    bin_rows = np.empty(len(order), dtype=np.intp)
    bin_rows[order] = rows
    return edges, bin_rows


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
    """Interpret authoritative scene streams without allocating a renderer.

    Methods take a recording time in seconds and return what a view draws at
    that time. Streams are sampled with their own clocks and support masks;
    each stream's latest sample is held, so the panels of one time share it.
    """

    def __init__(self, scene: Animation) -> None:
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
        self._sampled: dict[tuple[int, str | None, bool], tuple[float, np.ndarray]] = {}
        frames = scene.frame_times
        # The scene's typical frame interval, which measures trail lengths.
        self._frame_interval = (
            float(np.median(np.diff(frames))) if len(frames) > 1 else 0.0
        )
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
        self._poses: dict[int, _PoseStreams] = {}
        for body in data.bodypart_sets:
            names = list(body.bodyparts)
            edges = () if body.skeleton is None else body.skeleton.edges
            n_samples = len(body.times)
            self._poses[id(body)] = _PoseStreams(
                list(body.bodyparts.values()),
                np.array([body.sample_valid[name] for name in names], bool)
                .reshape(-1, n_samples)
                .T,
                np.array([body.interval_valid[name] for name in names], bool)
                .reshape(-1, max(n_samples - 1, 0))
                .T,
                np.array(
                    [[names.index(a), names.index(b)] for a, b in edges], dtype=np.intp
                ).reshape(-1, 2),
            )
        # Without exactly one position, heading arrows start at the arena's center.
        self._arena_center = np.mean(scene.env.bin_centers, axis=0)
        self.signal_groups = signal_groups(data.timeseries)
        # Seconds every time panel shows; a view may change it, as
        # fastplotlib's [ and ] keys do.
        self.window_seconds = scene.window_seconds
        self.track_env = scene.env if scene.track is None else scene.track
        self.track_extents = track_extents(self.track_env)
        # The track's mesh rows: their edges, and each bin's row.
        self.track_rows = (
            None if self.track_extents is None else _track_rows(*self.track_extents)
        )
        # Where each segment of a linearized track starts along it.
        self._segment_starts = np.array(
            [start for _, start, _ in track_segments(self.track_env)], dtype=float
        )
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
        self._sampled.clear()

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
        return float(position.trail_length or 0) * self._frame_interval

    def trail_rows(self, position: Any, time: float) -> range:
        """The position's own samples in its trail at ``time``."""
        times = position.times
        stop = int(np.searchsorted(times, time, side="right"))
        start = int(
            np.searchsorted(times, time - self.trail_seconds(position), side="left")
        )
        return range(min(start, stop), stop)

    def trail(
        self, position: Any, time: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The position's trail at ``time``: segments and their opacities.

        Segments join consecutive supported samples, and none is drawn while
        the current position is missing. Each fades with the age of its newer
        sample, from ``_TRAIL_MAX_ALPHA`` now to zero ``trail_seconds`` ago.

        Returns
        -------
        segments : ndarray of float64, shape (n_segments, 2, n_dims)
        alphas : ndarray of float64, shape (n_segments,)
        """
        from neurospatial.animation._style import _TRAIL_MAX_ALPHA

        data = position.data
        if not np.isfinite(self.sample(position, data, time)).all():
            return np.empty((0, 2, data.shape[1])), np.empty(0)
        rows = self.trail_rows(position, time)
        # A valid interval joins two valid, finite samples.
        starts = np.arange(rows.start, max(rows.start, rows.stop - 1))
        starts = starts[position.interval_valid[starts]]
        segments = np.stack([data[starts], data[starts + 1]], axis=1)
        seconds = self.trail_seconds(position)
        if seconds <= 0:
            return segments, np.full(len(starts), _TRAIL_MAX_ALPHA)
        age = time - position.times[starts + 1]
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
        """Sample one of ``item``'s prepared streams at ``time``.

        ``values`` is one of ``item``'s arrays and ``name`` selects its
        support (:func:`~neurospatial.animation._sampling.sample_observation`).
        Each stream's latest sample is held, so the field panels, trails and
        heading arrows of one time share a single sample. The returned array
        is shared between calls; do not modify it.
        """
        key = (id(values), name, circular)
        held = self._sampled.get(key)
        if held is None or held[0] != time:
            held = (
                time,
                sample_observation(item, values, time, circular=circular, name=name),
            )
            self._sampled[key] = held
        return held[1]

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
            shown[name] = VisibleEvents(
                records.positions[start:stop], opacity, key, slice(start, stop)
            )
        return shown

    def event_positions(self, events: Any) -> dict[str, NDArray[np.float64]]:
        """Each event type's located events, in time order.

        Events without a location are never shown and are left out.
        """
        return {
            name: self._event_records[id(events), name].positions
            for name in events.event_positions
        }

    def pose(self, body: Any, time: float) -> Pose:
        """The body parts and complete skeleton edges at ``time``.

        One search of the overlay's clock samples every part, each with its
        own support.
        """
        streams = self._poses[id(body)]
        points = sample_parts(
            body.times,
            streams.parts,
            streams.sample_valid,
            streams.interval_valid,
            time,
            interp=body.interp,
        )
        pairs = points[streams.edges]
        return Pose(points, pairs[np.isfinite(pairs).all(axis=(1, 2))])

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
        """Read the nearest supported source row through its authoritative map.

        The nearer sample wins and a tie goes to the earlier one; outside the
        video's samples nothing is shown.
        """
        from neurospatial.animation.overlays import _find_nearest_indices

        index = int(_find_nearest_indices(video.times, np.array([time]))[0])
        if index < 0:
            return None
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
        """The window of the time panels at ``time``: ``window_seconds`` around it."""
        half = self.window_seconds / 2
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

    def raster_ticks(self, raster: Any, time: float) -> NDArray[np.float64]:
        """Every unit's spikes in the current window as tick segments.

        Returns
        -------
        ndarray of float64, shape (n_spikes, 2, 2)
            Each tick from ``(spike time, row - 0.4)`` to
            ``(spike time, row + 0.4)``; row ``i`` is the raster's ``i``-th
            unit, bottom first.
        """
        windows = [
            self.raster_window(raster, row, time)
            for row in range(len(raster.spike_times))
        ]
        spikes = np.concatenate([np.empty(0), *windows])
        rows = np.repeat(
            np.arange(len(windows), dtype=float), [len(w) for w in windows]
        )
        ticks = np.empty((len(spikes), 2, 2))
        ticks[:, :, 0] = spikes[:, None]
        ticks[:, 0, 1], ticks[:, 1, 1] = rows - 0.4, rows + 0.4
        return ticks

    def track_window(self, field: AnimationField, time: float) -> TrackWindow:
        """The field along the track and the observed path, in the window.

        Only the field's supported intervals hold its samples; a projected
        2-D field shows its rows summed into track bins.
        """
        assert self.track_rows is not None
        track_edges, bin_rows = self.track_rows
        left, right = self.bounds(time)
        first = max(int(np.searchsorted(field.times, left, side="right")) - 1, 0)
        stop = min(
            int(np.searchsorted(field.times, right, side="left")), len(field) - 1
        )
        selected = np.arange(first, max(first, stop))
        selected = selected[
            field.interval_valid[selected] & field.sample_valid[selected]
        ]
        # A skipped interval between two shown ones is an empty column.
        skipped = selected[1:] != selected[:-1] + 1
        columns = (
            np.arange(len(selected)) + np.r_[0, np.cumsum(skipped)][: len(selected)]
        )
        time_edges = np.empty(len(selected) + int(skipped.sum()) + 1)
        time_edges[columns] = np.maximum(left, field.times[selected])
        time_edges[columns + 1] = np.minimum(right, field.times[selected + 1])
        observed = np.zeros(len(time_edges) - 1, dtype=bool)
        observed[columns] = True
        values = np.full((len(track_edges) - 1, len(time_edges) - 1), np.nan)
        if len(selected):
            projected = self.projection is not None
            rows = np.stack(
                [self._field_row(field, int(i), projected=projected) for i in selected]
            )
            values[bin_rows[:, None], columns[None, :]] = rows.T
        paths, colors = [np.empty((0, 2, 2))], []
        for position in self.scene._overlay_data.positions:
            times = position.times
            start = int(np.searchsorted(times, left, side="left"))
            stop = int(np.searchsorted(times, right, side="right"))
            along = track_position(self.track_env, position.data[start:stop])
            segment = np.searchsorted(self._segment_starts, along, side="right")
            # A valid interval joins two supported samples on one segment of
            # the track: a line between segments would cross positions along
            # the track that the animal never passed.
            joined = np.flatnonzero(
                position.interval_valid[start : max(start, stop - 1)]
                & np.isfinite(along[:-1])
                & np.isfinite(along[1:])
                & (segment[:-1] == segment[1:])
            )
            segments = np.empty((len(joined), 2, 2))
            segments[:, :, 0] = np.column_stack(
                [times[start + joined], times[start + joined + 1]]
            )
            segments[:, :, 1] = np.column_stack([along[joined], along[joined + 1]])
            paths.append(segments)
            colors += [position.color] * len(joined)
        return TrackWindow(
            time_edges, observed, track_edges, values, np.concatenate(paths), colors
        )
