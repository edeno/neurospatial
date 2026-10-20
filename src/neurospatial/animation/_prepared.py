"""Backend-neutral sampling and scientific-panel data for a prepared scene.

The scene owns its streams and support metadata. This interpreter retains those
sources and derives only the requested time window; it owns no canvas or reader.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from neurospatial.animation._sampling import sample_observation
from neurospatial.animation._tracks import (
    _project,
    _projection,
    track_extents,
    track_position,
)

if TYPE_CHECKING:
    from neurospatial.animation.scene import Animation, AnimationField


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

    Read converted streams so custom adapters' clocks and inherited scene
    clocks agree with rendering. No renderer, source values, decoder reads or
    combined recording-sized clock are needed. Point and interval records may
    be unordered; search bounded ordered chunks without reordering the records.
    """
    for field in scene.fields.values():
        yield field.times
    data = scene._overlay_data
    sampled: Iterator[Any] = chain(
        data.positions,
        data.bodypart_sets,
        data.head_directions,
        data.object_vectors,
        data.videos,
        data.timeseries,
    )
    for item in sampled:
        yield scene.frame_times if item.times is None else item.times
    for events in data.events:
        for clock in events.event_times.values():
            yield from _ordered_clock_chunks(clock)
    for raster in data.spike_rasters:
        for clock in raster.spike_times:
            yield from _ordered_clock_chunks(clock)
    for epochs in data.epochs:
        for column in range(2):
            yield from _ordered_clock_chunks(epochs.intervals[:, column])


def projected_field_limits(
    fields: Sequence[AnimationField], projection: Any
) -> tuple[float, float]:
    """Bounded projected limits from each stream's own supported source rows."""
    from neurospatial.animation._field_stats import animation_colormap_range

    projected = [
        _project(
            np.asarray(field._scale_rows(), dtype=float).reshape(-1, field.n_bins),
            projection,
        )
        for field in fields
    ]
    return animation_colormap_range(np.concatenate(projected), None, None)


class PreparedScene:
    """Interpret authoritative scene streams without allocating a renderer."""

    def __init__(self, scene: Animation) -> None:
        from neurospatial.animation._capabilities import require_scene_rendering

        require_scene_rendering(scene)
        self.scene = scene
        self.signal_groups: dict[tuple[str, str | int], list[Any]] = {}
        data = scene._overlay_data
        for i, signal in enumerate(data.timeseries):
            key = ("group", signal.group) if signal.group is not None else ("signal", i)
            self.signal_groups.setdefault(key, []).append(signal)
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
        self.limits = {
            name: field.color_limits() for name, field in scene.fields.items()
        }
        self.track_limits = dict(self.limits)
        if self.projection is not None:
            for name, field in scene.fields.items():
                self.track_limits[name] = projected_field_limits(
                    (field,), self.projection
                )

    def clock(self, item: Any) -> np.ndarray:
        """The source clock, with the scene clock for explicitly aligned rows."""
        return cast(
            "np.ndarray", self.scene.frame_times if item.times is None else item.times
        )

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
        return sample_observation(
            item, self.clock(item), values, time, circular=circular, name=name
        )

    def video_frame(self, video: Any, time: float) -> np.ndarray | None:
        """Read the nearest supported source row through its authoritative map."""
        times = self.clock(video)
        if not len(times) or time < times[0] or time > times[-1]:
            return None
        right = int(np.searchsorted(times, time, side="left"))
        index = (
            right - 1
            if right > 0 and time - times[right - 1] <= times[right] - time
            else right
        )
        return cast("np.ndarray | None", video.get_frame(index))

    def bounds(self, time: float) -> tuple[float, float]:
        half = self.scene.window_seconds / 2
        return time - half, time + half

    def signal_window(self, signal: Any, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Observed signal samples in the current window, with requested scaling."""
        left, right = self.bounds(time)
        clock = self.clock(signal)
        keep = slice(
            np.searchsorted(clock, left, side="left"),
            np.searchsorted(clock, right, side="right"),
        )
        # Conversion owns normalization; prepared values are final for every
        # consumer, including custom adapters with separate display metadata.
        return clock[keep], signal.data[keep]

    def raster_window(self, raster: Any, row: int, time: float) -> np.ndarray:
        """One unit's spike times in the current window, preserving unit order."""
        left, right = self.bounds(time)
        spikes = raster.spike_times[row]
        return cast("np.ndarray", spikes[(spikes >= left) & (spikes <= right)])

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
            values = field[int(index)]
            if self.projection is not None:
                values = _project(values[None, :], self.projection)[0]
            lo, hi = max(left, field.times[index]), min(right, field.times[index + 1])
            for first, last, value in zip(starts, stops, values, strict=True):
                if np.isfinite(value):
                    polygons.append([(lo, first), (hi, first), (hi, last), (lo, last)])
                    colors.append(value)
        paths = []
        for position in self.scene._overlay_data.positions:
            times = self.clock(position)
            start = int(np.searchsorted(times, left, side="left"))
            stop = int(np.searchsorted(times, right, side="right"))
            distances = track_position(self.track_env, position.data[start:stop])
            for index in range(start, stop - 1):
                if position.interval_valid is None or position.interval_valid[index]:
                    segment = np.column_stack(
                        [
                            times[index : index + 2],
                            distances[index - start : index - start + 2],
                        ]
                    )
                    if np.isfinite(segment).all():
                        paths.append((segment, position.color))
        return TrackWindow(
            np.asarray(polygons, dtype=float).reshape(-1, 4, 2),
            np.asarray(colors),
            paths,
        )
