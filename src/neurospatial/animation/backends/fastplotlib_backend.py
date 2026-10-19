"""Animate fields over time on the GPU with fastplotlib.

``env.animate_fields(fields, frame_times=..., backend="fastplotlib")`` returns
a :class:`FastplotlibAnimation`, a fastplotlib ``NDWidget`` with a time
slider in seconds and playback controls. Each frame recolors the bins' cells
and moves the overlays; nothing is pre-rendered, so long recordings start at
once. Inside Jupyter, frames are rendered where the kernel runs and streamed
to the browser, so this works on a remote GPU server.

Drawn: position (marker and fading trail), body parts with skeleton, head
direction, object vectors, events, video placed through its calibration,
time series and spike rasters beside the field on the same clock (with event
times marked), a track's fields against position and time, regions, walls
and landmarks.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.animation._parallel import (
    _TRAIL_MAX_ALPHA,
    _event_visibility,
    _head_direction_color,
)
from neurospatial.environment._fastplotlib import (
    ColorScale,
    FieldGraphic,
    _fans,
    _rectangles,
    camera_options,
    draw_colorbar,
    draw_layers,
    frame_view,
    import_fastplotlib,
    pair_segments,
    plot_title,
    xyz,
)

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from neurospatial.animation.overlays import (
        BodypartData,
        EpochData,
        EventData,
        HeadDirectionData,
        ObjectVectorData,
        OverlayData,
        PositionData,
        SpikeRasterData,
        TimeSeriesData,
        VideoData,
    )
    from neurospatial.environment import Environment

# Matplotlib marker codes, as the other backends take them, to pygfx markers.
_MARKERS = {
    "o": "circle",
    "s": "square",
    "D": "diamond",
    "d": "diamond",
    "^": "triangle_up",
    "v": "triangle_down",
    "<": "triangle_left",
    ">": "triangle_right",
    "+": "plus",
    "x": "cross",
    "*": "asterisk6",
}


def _marker(code: str) -> str:
    """A Matplotlib marker code as a pygfx marker; circles, with a warning, if none."""
    marker = _MARKERS.get(code)
    if marker is None:
        warnings.warn(
            f"The fastplotlib backend has no marker {code!r}; drawing circles. "
            f"Use one of {sorted(_MARKERS)}.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
        return "circle"
    return marker


def _rgba(color: str) -> NDArray[np.float32]:
    from matplotlib.colors import to_rgba

    return np.asarray(to_rgba(color), dtype=np.float32)


class _Samples:
    """Which row of an overlay the slider's time shows.

    Rows aligned to the animation's frames follow the field. Rows with times
    of their own are each shown from their time until the next, or for one
    typical interval at a gap, and none before the first.
    """

    def __init__(
        self,
        times: NDArray[np.float64] | None,
        start: float,
        frame_of: Callable[[float], int],
        frame_interval: float,
    ) -> None:
        self.frame_of, self.frame_interval = frame_of, frame_interval
        self.times = None if times is None else times - start
        if self.times is not None:
            intervals = np.diff(self.times)
            typical = float(np.median(intervals)) if len(intervals) else np.inf
            self.ends = self.times + np.minimum(
                np.diff(self.times, append=np.inf), typical
            )

    def at(self, time: float) -> int:
        """The row shown at ``time`` on the clock; -1 for none."""
        if self.times is None:
            return self.frame_of(time)
        row = int(np.searchsorted(self.times, time, side="right")) - 1
        return row if row >= 0 and time < self.ends[row] else -1

    def recent(self, row: int, n_frames: int) -> slice:
        """The rows of the last ``n_frames`` animation frames, up to ``row``."""
        if self.times is None:
            return slice(max(0, row - n_frames + 1), row + 1)
        since = self.times[row] - n_frames * self.frame_interval
        return slice(int(np.searchsorted(self.times, since, side="right")), row + 1)

    def most_recent(self, n_frames: int) -> int:
        """The most rows ``recent`` returns."""
        if self.times is None:
            return n_frames
        span = n_frames * self.frame_interval
        reach = np.searchsorted(self.times, self.times - span, side="right")
        return int(np.max(np.arange(len(self.times)) + 1 - reach, initial=1))


def _row(values: NDArray[np.float64], row: int) -> NDArray[np.float64]:
    """Row ``row`` of ``values``, or NaN for -1 (nothing shown)."""
    return values[row] if row >= 0 else np.full(values.shape[1:], np.nan)


class _Position:
    """The animal's position and a fading trail of recent positions.

    The trail spans the last ``trail_length`` animation frames.
    """

    def __init__(self, subplot: Any, data: PositionData, clock: _Samples) -> None:
        self.data, self.clock = data, clock
        self.frames = data.trail_length or 0
        self.length = clock.most_recent(self.frames) if self.frames > 1 else 0
        if self.length > 1:
            colors = np.tile(_rgba(data.color), (self.length, 1))
            colors[:, 3] = np.linspace(0.0, _TRAIL_MAX_ALPHA, self.length)
            self.trail = subplot.add_line(
                xyz(np.full((self.length, 2), np.nan)), colors=colors, thickness=2
            )
        self.marker = subplot.add_scatter(
            xyz([np.nan, np.nan]),
            colors=data.color,
            sizes=data.size,
            edge_colors="white",
            edge_width=1,
        )

    def update(self, time: float) -> None:
        row = self.clock.at(time)
        self.marker.data = xyz(_row(self.data.data, row))
        if self.length > 1:
            window = np.full((self.length, self.data.data.shape[1]), np.nan)
            if row >= 0:
                recent = self.data.data[self.clock.recent(row, self.frames)]
                window[self.length - len(recent) :] = recent
            self.trail.data = xyz(window)


class _Bodyparts:
    """Body parts as points, joined by the skeleton's edges."""

    def __init__(self, subplot: Any, data: BodypartData, clock: _Samples) -> None:
        self.data, self.clock = data, clock
        self.names = list(data.bodyparts)
        colors = [(data.colors or {}).get(name, "cyan") for name in self.names]
        self.points = subplot.add_scatter(
            xyz(np.full((len(self.names), 2), np.nan)),
            colors=np.array([_rgba(color) for color in colors]),
            sizes=8,
            edge_colors="white",
            edge_width=0.5,
        )
        self.edges: list[tuple[str, str]] = []
        skeleton = data.skeleton
        if skeleton is not None:
            self.edges = [
                (start, end)
                for start, end in skeleton.edges
                if start in data.bodyparts and end in data.bodyparts
            ]
        if self.edges:
            assert skeleton is not None
            self.lines = subplot.add_line(
                self._segments(-1),
                colors=skeleton.edge_color,
                thickness=skeleton.edge_width,
            )

    def _segments(self, row: int) -> NDArray[np.float32]:
        parts = self.data.bodyparts
        return pair_segments(
            [_row(parts[start], row) for start, _ in self.edges],
            [_row(parts[end], row) for _, end in self.edges],
        )

    def update(self, time: float) -> None:
        row = self.clock.at(time)
        parts = self.data.bodyparts
        self.points.data = xyz([_row(parts[name], row) for name in self.names])
        if self.edges:
            self.lines.data = self._segments(row)


class _HeadDirection:
    """A heading line from the animal's position, or from the arena's center."""

    def __init__(
        self,
        subplot: Any,
        data: HeadDirectionData,
        clock: _Samples,
        anchor: _Position | None,
        center: NDArray[np.float64],
    ) -> None:
        self.data, self.clock, self.anchor, self.center = data, clock, anchor, center
        self.line = subplot.add_line(
            xyz(np.full((2, 2), np.nan)),
            colors="w",
            thickness=data.width,
        )

    def update(self, time: float) -> None:
        origin = self.center
        if self.anchor is not None:
            animal = _row(self.anchor.data.data, self.anchor.clock.at(time))
            if np.all(np.isfinite(animal)):
                origin = animal
        heading = _row(self.data.data, self.clock.at(time))
        segment = heading_segment(origin, heading, self.data.length)
        self.line.data = xyz(segment)
        if np.all(np.isfinite(segment)):
            direction = segment[1] - segment[0]
            color = _head_direction_color(self.data.color, direction)
            self.line.colors = color


def heading_segment(
    origin: NDArray[np.float64], heading: NDArray[np.float64] | float, length: float
) -> NDArray[np.float64]:
    """The heading line, shape (2, n_dims): from ``origin``, ``length`` long.

    Parameters
    ----------
    origin : ndarray of float64, shape (n_dims,)
        Where the line starts: the animal, or the environment's center.
    heading : float or ndarray of float64, shape (n_dims,)
        An angle in radians, in the x-y plane, or a direction vector with all
        of its coordinates (a 3-D heading keeps its z).
    length : float
        Line length, in the environment's units.

    Returns
    -------
    ndarray of float64, shape (2, n_dims)
        NaN for a missing heading.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.zeros(len(origin))
    if np.ndim(heading) == 0:
        direction[:2] = np.cos(heading), np.sin(heading)
    else:
        vector = np.asarray(heading, dtype=np.float64)[: len(origin)]
        norm = np.linalg.norm(vector)
        direction[: len(vector)] = vector / norm if norm > 0 else vector
    segment = np.array([origin, origin + direction * length])
    return segment if np.all(np.isfinite(segment)) else np.full_like(segment, np.nan)


class _Events:
    """Events at their positions: all so far, only the current, or fading.

    Positions are drawn once; each frame only sets the opacity of each
    event's fill and border.
    """

    def __init__(
        self, subplot: Any, data: EventData, frame_of: Callable[[float], int]
    ) -> None:
        self.data, self.frame_of = data, frame_of
        self.kinds = []
        for name, positions in data.event_positions.items():
            if len(positions) == 0:
                continue
            fill = np.tile(_rgba(data.colors[name]), (len(positions), 1))
            border = np.tile(_rgba(data.border_color), (len(positions), 1))
            fill[:, 3] = border[:, 3] = 0.0
            graphic = subplot.add_scatter(
                xyz(positions),
                colors=fill,
                sizes=data.size,
                markers=_marker(data.markers.get(name, "o")),
                edge_colors=border,
                edge_width=data.border_width * data.size,  # a fraction of the size
                alpha_mode="blend",
            )
            frames = data.event_frame_indices[name]
            self.kinds.append((frames, fill, border, graphic))

    def update(self, time: float) -> None:
        frame = self.frame_of(time)
        for frames, fill, border, graphic in self.kinds:
            shown, alphas = _event_visibility(frames, frame, self.data.decay_frames)
            fill[:, 3] = border[:, 3] = np.where(shown, alphas * self.data.opacity, 0)
            graphic.colors, graphic.edge_colors = fill, border


class _ObjectVectors:
    """Lines from the animal to each object; opacity follows the firing rate.

    With firing rates, a rate of zero draws faint lines and the highest rate
    opaque ones.
    """

    def __init__(self, subplot: Any, data: ObjectVectorData, clock: _Samples) -> None:
        self.data, self.clock = data, clock
        self.color = _rgba(data.color)
        rates = data.firing_rates
        self.top_rate = (
            float(rates.max()) if rates is not None and rates.max() > 0 else 1.0
        )
        n_objects = len(data.object_positions)
        self.lines = subplot.add_line(
            np.full((3 * n_objects, 3), np.nan, dtype=np.float32),
            colors=self.color,
            thickness=data.linewidth,
            alpha_mode="blend",
        )
        if data.show_objects:
            subplot.add_scatter(
                xyz(data.object_positions),
                colors=self.color,
                sizes=data.object_size,
                markers=_marker(data.object_marker),
                edge_colors="white",
                edge_width=1,
            )

    def update(self, time: float) -> None:
        row = self.clock.at(time)
        animal = _row(self.data.animal_positions, row)
        objects = self.data.object_positions
        self.lines.data = pair_segments(np.broadcast_to(animal, objects.shape), objects)
        if self.data.firing_rates is not None and row >= 0:
            color = self.color.copy()
            color[3] *= 0.1 + 0.9 * self.data.firing_rates[row] / self.top_rate
            self.lines.colors = color


class _Video:
    """Video frames placed in the environment through the calibration.

    ``depth`` puts the frames above or below the field; walls, landmarks and
    the moving overlays are drawn above both.
    """

    def __init__(
        self, subplot: Any, data: VideoData, clock: _Samples, depth: float
    ) -> None:
        self.data, self.clock = data, clock
        if isinstance(data.reader, np.ndarray):
            height, width = data.reader.shape[1:3]
        else:
            width, height = data.reader.frame_size_px
        self.image = subplot.add_image(
            np.zeros((height, width, 3), dtype=np.uint8), vmin=0, vmax=255
        )
        self.image.alpha_mode = "blend"
        self.image.alpha = data.alpha
        matrix = np.eye(4)
        matrix[np.ix_([0, 1], [0, 1, 3])] = video_to_env(data, width, height)[:2]
        matrix[2, 3] = depth
        self.image.world_object.local.matrix = matrix

    def show(self, image: NDArray[np.uint8] | None) -> None:
        """Show a frame read by ``data.get_frame``; None hides the video."""
        self.image.visible = image is not None
        if image is not None:
            self.image.data = image


def video_to_env(data: VideoData, width: int, height: int) -> NDArray[np.float64]:
    """The matrix taking a video pixel ``(column, row, 1)`` to ``(x, y, 1)``.

    The first pixel's center is ``(0, 0)``. With a calibration this is its
    complete pixel-to-environment transform, rotation and reflection
    included; without one the frame is stretched upright over the
    environment's extent.

    Parameters
    ----------
    data : VideoData
        The video overlay.
    width, height : int
        Frame size, in pixels.

    Returns
    -------
    ndarray of float64, shape (3, 3)
    """
    if data.transform_to_env is not None:
        return np.asarray(data.transform_to_env.A, dtype=np.float64)
    xmin, xmax, ymin, ymax = data.env_bounds
    sx, sy = (xmax - xmin) / width, (ymax - ymin) / height
    # The first row at the top, and the frame's edges on the extent.
    return np.array(
        [[sx, 0.0, xmin + sx / 2], [0.0, -sy, ymax - sy / 2], [0.0, 0.0, 1.0]]
    )


def track_extents(
    env: Environment,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Where each bin starts and ends along a track; None off a track.

    Linearized tracks (``linear_track``, ``from_graph``) measure distance
    along the track, with the spacing between segments left empty; 1-D
    environments use their own coordinate.
    """
    if env.is_linearized_track:
        linearization = env._binning.linearization
        assert linearization is not None
        edges = np.asarray(linearization.grid_edges[0], dtype=np.float64)
        active = linearization.active_mask
        return edges[:-1][active], edges[1:][active]
    if env.n_dims == 1:
        centers, half = env.bin_centers[:, 0], env.bin_sizes / 2
        return centers - half, centers + half
    return None


def track_position(
    env: Environment, positions: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Positions as distances along the track, shape (n,); NaN stays NaN."""
    if env.is_linearized_track:
        return np.asarray(env.to_linear(positions), dtype=np.float64)
    return np.asarray(positions, dtype=np.float64)[:, 0]


class _TimeWindow:
    """A panel's view of ``window`` seconds around the slider's time.

    A cursor marks the time; dragging it moves every panel.
    """

    def __init__(
        self, subplot: Any, window: float, move_to: Callable[[float], None]
    ) -> None:
        from fastplotlib import LinearSelector

        self.subplot, self.window = subplot, window
        self.cursor = LinearSelector(
            0.0, limits=(-np.inf, np.inf), edge_color="lightgray"
        )
        self.cursor.add_event_handler(
            lambda event: move_to(event.info["value"]), "selection"
        )
        subplot.add_graphic(self.cursor)
        subplot.camera.maintain_aspect = False

    def follow(self, time: float) -> None:
        """Center the view and the cursor on ``time``."""
        from fastplotlib.graphics.utils import pause_events

        with pause_events(self.cursor):
            self.cursor.selection = time
        half = self.window / 2
        self.subplot.x_range = (time - half, time + half)


class _PositionTime:
    """A track's fields over time: a column per frame, a row per bin.

    A column starts at its frame's time and lasts until the next frame, as
    the field panel holds it, or one typical frame interval at a gap, which
    stays empty. Rows span each bin along the track. Positions are drawn as
    lines along the track, every sample of their own.
    """

    def __init__(
        self,
        subplot: Any,
        env: Environment,
        extents: tuple[NDArray[np.float64], NDArray[np.float64]],
        scale: ColorScale,
        clock: NDArray[np.float64],
        window: float,
        positions: Sequence[_Position],
        start: float,
        move_to: Callable[[float], None],
    ) -> None:
        self.scale, self.clock, self.start = scale, clock, start
        self.window = window
        self.lower, self.upper = extents
        typical = float(np.median(np.diff(clock))) if len(clock) > 1 else 1.0
        self.ends = clock + np.minimum(np.diff(clock, append=np.inf), typical)
        # Columns a window can hold: frames starting less than one typical
        # interval before it, up to its end.
        reach = np.searchsorted(clock, clock + window + typical, side="right")
        self.n_columns = int(np.max(reach - np.arange(len(clock))))
        n_bins = len(self.lower)
        quads = _fans(np.zeros((4 * self.n_columns * n_bins, 2)), 4)
        self.mesh = subplot.add_mesh(
            quads.positions,
            quads.triangles,
            mode="basic",
            mapcoords=np.zeros(len(quads.positions), dtype=np.float32),
            cmap=scale.texture,
            clim=(0.0, 1.0),
        )
        self.mesh.tooltip_format = self.describe
        labels = env._plot_labels()
        self.axis = "position along track" + (
            f" ({labels.units})" if labels.units else ""
        )
        self.tracks = [
            (track_position(env, item.data.data), item.clock) for item in positions
        ]
        self.lines = [
            subplot.add_line(
                np.full(
                    (self._most_in_window(item.clock), 3), np.nan, dtype=np.float32
                ),
                colors=item.data.color,
                thickness=2,
            )
            for item in positions
        ]
        self.first, self.rows = 0, np.empty((0, n_bins))
        self.view = _TimeWindow(subplot, window, move_to)
        subplot.title = self.axis
        subplot.y_range = (float(self.lower.min()), float(self.upper.max()))

    def _most_in_window(self, clock: _Samples) -> int:
        """The most samples of ``clock`` a window holds."""
        if clock.times is None:
            return self.n_columns
        times = clock.times
        reach = np.searchsorted(times, times + self.window, side="right")
        return int(np.max(reach - np.arange(len(times)), initial=1))

    def frames_at(self, time: float) -> slice:
        """The frames whose columns show in the window around ``time``."""
        half = self.window / 2
        return slice(
            int(np.searchsorted(self.ends, time - half, side="right")),
            int(np.searchsorted(self.clock, time + half, side="left")),
        )

    def show(self, time: float, first: int, rows: NDArray[np.float64]) -> None:
        """Draw ``rows``, the fields from frame ``first``, around ``time``."""
        self.first, self.rows = first, rows
        n_frames, n_bins = rows.shape
        frames = np.arange(first, first + n_frames)
        corners = _rectangles(
            np.repeat(self.clock[frames], n_bins),
            np.repeat(self.ends[frames], n_bins),
            np.tile(self.lower, n_frames),
            np.tile(self.upper, n_frames),
        )
        positions = np.zeros((len(self.mesh.positions), 3), dtype=np.float32)
        positions[: len(corners), :2] = corners  # unused columns have no area
        self.mesh.positions = positions
        mapcoords = np.zeros(len(positions), dtype=np.float32)
        mapcoords[: len(corners)] = np.repeat(
            self.scale.texture_coordinates(rows.ravel()), 4
        )
        self.mesh.mapcoords = mapcoords
        half = self.window / 2
        for line, (track, clock) in zip(self.lines, self.tracks, strict=True):
            if clock.times is None:
                shown, times = frames, self.clock[frames]
            else:
                shown = np.arange(
                    np.searchsorted(clock.times, time - half, side="left"),
                    np.searchsorted(clock.times, time + half, side="right"),
                )
                times = clock.times[shown]
            points = np.full((len(line.data), 2), np.nan)
            points[: len(shown)] = np.column_stack([times, track[shown]])
            line.data = xyz(points)
        self.view.follow(time)

    def describe(self, pick_info: dict[str, Any]) -> str:
        """The tooltip for the cell under the pointer."""
        column, bin_index = divmod(pick_info["face_index"] // 2, len(self.lower))
        frame = self.first + column
        value = self.rows[column, bin_index]
        middle = (self.lower[bin_index] + self.upper[bin_index]) / 2
        return "\n".join(
            [
                f"t = {self.start + self.clock[frame]:.3f} s",
                f"bin {bin_index}, {self.axis}: {middle:.4g}",
                f"value: {'missing' if np.isnan(value) else f'{value:.4g}'}",
            ]
        )


_SELECTED_COLOR = "orange"


class _UnitRates:
    """The rate map of the unit chosen in a raster.

    Drawn like the field, all missing until a unit is chosen, or on a track
    as rate against position along it.
    """

    def __init__(self, subplot: Any, env: Environment, cmap: str | Colormap) -> None:
        self.subplot, self.cmap = subplot, cmap
        self.raster: _Raster | None = None
        self.extents = track_extents(env)
        if self.extents is not None:
            self.line = subplot.add_line(
                np.full((env.n_bins, 3), np.nan, dtype=np.float32),
                colors=_SELECTED_COLOR,
                thickness=2,
            )
            subplot.camera.maintain_aspect = False
        else:
            self.field = FieldGraphic(
                subplot,
                env,
                np.full(env.n_bins, np.nan),
                ColorScale(cmap, 0.0, 1.0, "lightgray"),
            )
            subplot.camera.maintain_aspect = env._plot_labels().coordinates.equal_aspect
            subplot.auto_scale()
            frame_view(subplot, env)
        subplot.title = "click a spike to show its unit's rate map"

    def show(self, raster: _Raster, row: int) -> None:
        """Show the rate map of ``row`` in ``raster``, and highlight it there."""
        if self.raster is not None and self.raster is not raster:
            self.raster.highlight(None)
        self.raster = raster
        raster.highlight(row)
        assert raster.data.rates is not None
        rates = raster.data.rates[row]
        peak = float(np.nanmax(rates)) if np.isfinite(rates).any() else np.nan
        top = peak if peak > 0 else 1.0
        if self.extents is not None:
            lower, upper = self.extents
            self.line.data = xyz(np.column_stack([(lower + upper) / 2, rates]))
            self.subplot.x_range = (float(lower.min()), float(upper.max()))
            self.subplot.y_range = (0.0, 1.05 * top)
        else:
            self.field.scale = ColorScale(self.cmap, 0.0, top, "lightgray")
            self.field.update(rates)
        self.subplot.title = (
            f"unit {raster.data.unit_ids[row]}: rate map, peak {peak:.3g} Hz"
        )


class _Raster:
    """Spikes as ticks, a row per unit, around the slider's time.

    Every spike is drawn once, as a vertical segment most of its row's
    height; the view follows the time. With rate maps, clicking a unit's
    row shows its map in ``rates`` and highlights the unit's spikes.
    """

    def __init__(
        self,
        subplot: Any,
        data: SpikeRasterData,
        start: float,
        move_to: Callable[[float], None],
        rates: _UnitRates | None,
    ) -> None:
        self.data, self.start, self.rates = data, start, rates
        counts = [len(train) for train in data.spike_times]
        self.row = np.repeat(np.arange(len(counts)), counts)
        self.times = np.concatenate(data.spike_times)
        x = self.times - start
        # Three vertices a spike: its segment's ends and a gap.
        segments = pair_segments(
            np.column_stack([x, self.row - 0.4]), np.column_stack([x, self.row + 0.4])
        )
        if not len(segments):
            segments = np.full((1, 3), np.nan, dtype=np.float32)
        self.ticks = subplot.add_line(
            segments, colors=self._colors(None), thickness=data.linewidth
        )
        self.ticks.tooltip_format = self.describe
        if rates is not None and data.rates is not None:
            # The whole row answers a click, not only its thin ticks.
            subplot.renderer.add_event_handler(self.clicked, "click")
        self.view = _TimeWindow(subplot, data.window_seconds, move_to)
        subplot.title = f"spikes, {len(counts)} units"
        subplot.y_range = (-0.5, len(counts) - 0.5)

    def describe(self, pick_info: dict[str, Any]) -> str:
        """The tooltip for the spike under the pointer."""
        spike = int(pick_info["vertex_index"]) // 3
        unit = self.data.unit_ids[self.row[spike]]
        return f"unit {unit}\nt = {self.times[spike]:.4f} s"

    def clicked(self, event: Any) -> None:
        """Show the unit of the row clicked, if the click is in this panel."""
        assert self.rates is not None
        where = self.view.subplot.map_screen_to_world(event)
        if where is None:
            return
        row = round(float(where[1]))
        if 0 <= row < len(self.data.spike_times):
            self.rates.show(self, row)

    def _colors(self, row: int | None) -> NDArray[np.float32]:
        """Each vertex's color: the spikes of ``row`` highlighted."""
        colors = np.tile(_rgba(self.data.color), (max(3 * len(self.row), 1), 1))
        if row is not None:
            colors[np.repeat(self.row == row, 3)] = _rgba(_SELECTED_COLOR)
        return colors

    def highlight(self, row: int | None) -> None:
        """Color the spikes of ``row``; None for none."""
        self.ticks.colors = self._colors(row)


def playback_step(frame_times: NDArray[np.float64]) -> float:
    """Seconds that play and the step buttons move: short enough to reach
    every frame, unless frames are much closer than the typical interval.

    Warns when the frames are unevenly spaced enough for playback to hold
    one frame through a gap or pass over frames.
    """
    if len(frame_times) < 2:
        return 1.0
    intervals = np.diff(frame_times)
    typical = float(np.median(intervals))
    step = max(float(intervals.min()), typical / 2)
    if intervals.min() < typical / 2 or intervals.max() > 10 * typical:
        warnings.warn(
            f"frame_times are unevenly spaced (intervals from {intervals.min():.3g} "
            f"to {intervals.max():.3g} s). Play and the step buttons move "
            f"{step:.3g} s at a time, so a long gap plays as one held frame and "
            f"frames closer than {step:.3g} s can be passed over. Set "
            "animation.frame to visit every frame, animate each event "
            "separately, or use backend='napari', which steps frame by frame.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return step


class _Frame(NamedTuple):
    """What one time shows, read on a worker thread."""

    frame: int
    values: NDArray[np.float64]  # the field, shape (n_bins,)
    images: list[Any]  # each video's frame, or None
    first: int  # the first frame of ``rows``
    rows: NDArray[np.float64]  # a track's fields in its window, (n, n_bins)


def _frames_graphic(
    nd_subplot: Any,
    indices: Any,
    fields: Any,
    frame_of: Callable[[float], int],
    read: Callable[[float], Any],
    show: Callable[[float, Any], None],
    graphic: Any,
) -> Any:
    """The field and every frame-aligned panel, as one ``NDGraphic``.

    The widget's slider is in seconds; ``frame_of`` maps it to the frame
    shown, which is held until the next frame starts. ``read`` loads the
    data for a time (field values, video images) on a worker thread, so disk
    reads do not stall the controls; ``show`` then draws all of it together,
    so the overlays never show another frame than the field. Built in a
    function because fastplotlib is optional.
    """
    from concurrent.futures import ThreadPoolExecutor

    from fastplotlib.widgets.nd_widget import NDGraphic, NDSlicer
    from fastplotlib.widgets.nd_widget._async import run_in_thread_pool

    class Frames(NDGraphic):  # type: ignore[misc,unused-ignore]
        def __init__(self) -> None:
            super().__init__(nd_subplot, "field")
            # Names the dims for the widget; the frames are read by ``read``.
            self._slicer = NDSlicer(
                fields, ("time", "bin"), ("bin",), slider_maps={"time": frame_of}
            )
            self._graphic = graphic
            self._reader = ThreadPoolExecutor(max_workers=1)

        @property
        def slicer(self) -> Any:
            return self._slicer

        @property
        def graphic(self) -> Any:
            return self._graphic

        @property
        def display_dims(self) -> tuple[str, ...]:
            return self._slicer.display_dims  # type: ignore[no-any-return]

        @property
        def indices(self) -> dict[str, float]:
            return {"time": indices["time"]}

        async def _set_indices_(self, at: dict[str, float] | None = None) -> None:
            at = self.indices if at is None else at
            time = at["time"]
            show(time, await run_in_thread_pool(self._reader, read, time))
            self._last_indices = at

    return Frames()


class FastplotlibAnimation:
    """Fields over time on the GPU, with a time slider and playback controls.

    Returned by ``env.animate_fields(..., backend="fastplotlib")``. It is a
    fastplotlib ``NDWidget``: a slider in seconds, play, step and loop
    buttons, and a frame rate box. Time-series overlays are drawn beside the
    field on the same clock, each showing its window around the current
    time, with event times marked, and so are spike rasters. On a track (a linearized track or a 1-D
    environment) the fields are also drawn over time, position along the
    track against time, with the positions as lines over them: a decoded
    posterior beside the animal's position, say. Hovering over a bin shows
    its value. In Jupyter it displays itself; elsewhere call :meth:`show`
    (in a script, then ``fastplotlib.loop.run()``).

    The slider and the time-series axes count seconds from the first frame,
    so session timestamps (such as 1.7e9 s) keep their precision on the GPU;
    the title shows each frame's own time.

    Keyboard, with the figure focused (click it first):

    - N and Shift+N: the start of the next or previous epoch
      (``EpochOverlay``) or event (``EventOverlay``).

    Parameters
    ----------
    env : Environment
        The bins the fields are defined on.
    fields : sequence of ndarray of shape (n_bins,), or ndarray (n_frames, n_bins)
        One field per frame. An array, including a memory map, is read one
        frame at a time.
    frame_times : ndarray of shape (n_frames,)
        Time of each frame, in seconds. A frame is shown from its time until
        the next frame's.
    fps : float
        Steps per second while playing; changeable in the controls.
    cmap, vmin, vmax
        Colormap and its limits; limits default to the 1st percentile of the
        frames' minima and the 99th of their maxima, estimated from a sample
        of a very long recording.
    frame_labels : sequence of str, optional
        A label per frame; the frame times by default.
    title : str, optional
        Plot title; the environment's name and units by default.
    overlay_data : OverlayData, optional
        Overlays aligned to the frames.
    show_regions : bool or sequence of str, default=False
        Draw all regions, or the named ones.
    region_alpha : float, default=0.3
        Opacity of region areas.
    window_seconds : float, default=2.0
        Seconds of fields shown in a track's position-by-time panel, centered
        on the current time.

    Attributes
    ----------
    widget : fastplotlib.NDWidget
        The widget; ``widget.indices["time"]`` is the slider's time, in
        seconds from the first frame.
    figure : fastplotlib.Figure
        The widget's figure; add graphics to ``figure["field"]``. The other
        panels are ``"track"`` (a track's fields against position and time),
        ``"raster 0"``, ``"raster 1"``, ... (spike rasters), ``"unit"`` (the
        rate map of the unit chosen in a raster) and ``"series 0"``, ...
    n_frames : int
        Number of frames.
    """

    def __init__(
        self,
        env: Environment,
        fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
        *,
        frame_times: NDArray[np.float64],
        fps: float,
        cmap: str | Colormap = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        frame_labels: Sequence[str] | None = None,
        title: str | None = None,
        overlay_data: OverlayData | None = None,
        show_regions: bool | Sequence[str] = False,
        region_alpha: float = 0.3,
        window_seconds: float = 2.0,
    ) -> None:
        from neurospatial.animation._timeseries import _group_timeseries
        from neurospatial.animation.rendering import (
            _validate_frame_labels,
            animation_colormap_range,
        )

        fastplotlib = import_fastplotlib()
        self._fields = fields if hasattr(fields, "shape") else np.asarray(fields)
        self.n_frames = len(fields)
        self._frame_times = np.asarray(frame_times, dtype=np.float64)
        # The widget's clock: seconds from the first frame.
        self._start = float(self._frame_times[0])
        self._clock = self._frame_times - self._start
        self._labels = _validate_frame_labels(
            None if frame_labels is None else list(frame_labels),
            self.n_frames,
            "fastplotlib",
        )
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be positive, got {window_seconds}. It is "
                "the seconds of fields shown in a track's position-by-time panel."
            )
        groups = _group_timeseries(overlay_data.timeseries if overlay_data else [])
        step = playback_step(self._frame_times)
        extents_along_track = track_extents(env)
        # Panels on the clock fill the right, stacked top to bottom: the
        # track's fields over time, the rasters, then the time series; the
        # field narrows when the right holds more than time series.
        rasters = overlay_data.spike_rasters if overlay_data else []
        panels = [("track", 2.0)] if extents_along_track is not None else []
        panels += [(f"raster {i}", 2.0) for i in range(len(rasters))]
        panels += [(f"series {i}", 1.0) for i in range(len(groups))]
        split = (0.4 if len(panels) > len(groups) else 0.6) if panels else 1.0
        # A raster with rate maps adds the chosen unit's map under the field.
        with_rates = any(item.rates is not None for item in rasters)
        extents = {"field": (0.0, split, 0.0, 0.65 if with_rates else 1.0)}
        if with_rates:
            extents["unit"] = (0.0, split, 0.65, 1.0)
        total, top = sum(weight for _, weight in panels), 0.0
        for name, weight in panels:
            extents[name] = (split, 1.0, top / total, (top + weight) / total)
            top += weight
        camera, controller = camera_options(env)
        # The slider stops at the last frame.
        self.widget = fastplotlib.NDWidget(
            ranges={"time": (0.0, self._clock[-1] + step, step)},
            extents=extents,
            cameras=[camera] * (1 + with_rates) + ["2d"] * len(panels),
            controller_types=[controller] * (1 + with_rates)
            + ["panzoom"] * len(panels),
            size=(1100 if panels else 800, 760),
        )
        # Playing moves one step per 1 / fps s (the box allows 1-100); a step
        # shorter than the typical interval plays faster to keep the speed.
        typical = float(np.median(np.diff(self._clock))) if self.n_frames > 1 else step
        steps_per_second = min(max(round(fps * typical / step), 1), 100)
        self.widget.ui_sliders._fps["time"] = steps_per_second
        self.widget.ui_sliders._frame_time["time"] = 1 / steps_per_second
        self.figure = self.widget.figure
        self._plot = self.figure["field"]
        self._plot.camera.maintain_aspect = env._plot_labels().coordinates.equal_aspect

        low, high = animation_colormap_range(fields, vmin, vmax)
        scale = ColorScale(cmap, low, high, "lightgray")
        first = np.asarray(fields[0], dtype=np.float64)
        self._field = FieldGraphic(self._plot, env, first, scale)

        def clock(times: NDArray[np.float64] | None) -> _Samples:
            return _Samples(times, self._start, self._frame_of, typical)

        self._videos: list[_Video] = []
        if overlay_data is not None and overlay_data.videos:
            # Depth orders the field, the videos and everything else (at 0).
            # Fractional, so that fastplotlib does not restack the videos.
            self._field.graphic.offset = (0.0, 0.0, -1.5)
            self._videos = [
                _Video(
                    self._plot,
                    item,
                    clock(item.times),
                    (-0.5 if item.z_order == "above" else -2.5) - i / 100,
                )
                for i, item in enumerate(overlay_data.videos)
            ]
        draw_layers(
            self._plot,
            env,
            show_walls=True,
            show_landmarks=True,
            show_regions=show_regions,
            region_alpha=region_alpha,
        )
        draw_colorbar(self._plot, scale, "")
        self._title = title or plot_title(env)
        events = overlay_data.events if overlay_data else []
        epochs = overlay_data.epochs if overlay_data else []
        self._epochs = epochs
        # Where N and Shift+N jump: epoch starts and event times, on the clock.
        marks = [item.intervals[:, 0] for item in epochs] + [
            times for item in events for times in item.event_times.values()
        ]
        self._marks = np.unique(np.concatenate([[], *marks])) - self._start
        self._layers: list[Any] = []
        positions: list[_Position] = []
        if overlay_data is not None:
            self._layers += [
                _ObjectVectors(self._plot, item, clock(item.times))
                for item in overlay_data.object_vectors
            ]
            positions = [
                _Position(self._plot, item, clock(item.times))
                for item in overlay_data.positions
            ]
            self._layers += positions
            self._layers += [
                _Bodyparts(self._plot, item, clock(item.times))
                for item in overlay_data.bodypart_sets
            ]
            anchor = positions[0] if len(positions) == 1 else None
            center = env.bin_centers.mean(axis=0)
            self._layers += [
                _HeadDirection(self._plot, item, clock(item.times), anchor, center)
                for item in overlay_data.head_directions
            ]
            self._layers += [
                _Events(self._plot, item, self._frame_of)
                for item in overlay_data.events
            ]
        self._track: _PositionTime | None = None
        if extents_along_track is not None:
            self._track = _PositionTime(
                self.figure["track"],
                env,
                extents_along_track,
                scale,
                self._clock,
                window_seconds,
                positions,
                self._start,
                self._move_to,
            )
            _mark_events(self.figure["track"], events, self._start)
            _shade_epochs(self.figure["track"], epochs, self._start, lines=True)
        unit = _UnitRates(self.figure["unit"], env, cmap) if with_rates else None
        self._rasters = [
            _Raster(self.figure[f"raster {i}"], item, self._start, self._move_to, unit)
            for i, item in enumerate(rasters)
        ]
        for i in range(len(rasters)):
            _mark_events(self.figure[f"raster {i}"], events, self._start)
            _shade_epochs(self.figure[f"raster {i}"], epochs, self._start, lines=False)
        frames = _frames_graphic(
            self.widget[self._plot],
            self.widget.indices,
            self._fields,
            self._frame_of,
            self._read,
            self._show,
            self._field.graphic,
        )
        # fastplotlib has no public way to add a custom NDGraphic yet.
        self.widget[self._plot]._nd_graphics.append(frames)
        for i, group in enumerate(groups):
            _add_series(self.widget, f"series {i}", group, events, epochs, self._start)
        self.widget.indices = {"time": 0.0}  # adding a series can move the time
        # Only the field is fitted: fitting a series subplot's camera moves
        # the time, which follows the center of its view.
        self._plot.auto_scale()
        frame_view(self._plot, env)
        self._canvas: Any = None
        self.figure.renderer.add_event_handler(self._key, "key_down")
        self._show(0.0, self._read(0.0))

    def _key(self, event: Any) -> None:
        """The keyboard shortcuts listed in the class docstring."""
        key = event.key
        if key.lower() == "n":
            self._jump(-1 if "Shift" in event.modifiers or key == "N" else 1)

    def _jump(self, direction: int) -> None:
        """Move to the next (1) or previous (-1) epoch start or event."""
        now = self.widget.indices["time"]
        if direction > 0:
            later = self._marks[self._marks > now + 1e-9]
            target = later[0] if len(later) else None
        else:
            earlier = self._marks[self._marks < now - 1e-9]
            target = earlier[-1] if len(earlier) else None
        if target is not None:
            self._move_to(float(target))

    def _move_to(self, time: float) -> None:
        """Move the slider, and so every panel, to ``time`` on the clock."""
        self.widget.indices = {"time": time}

    def _frame_of(self, time: float) -> int:
        """The frame shown at ``time`` on the clock: the last one that started."""
        return max(int(np.searchsorted(self._clock, time, side="right")) - 1, 0)

    def _read(self, time: float) -> _Frame:
        """What ``time`` shows, read from the fields and videos.

        Runs on a worker thread.
        """
        frame = self._frame_of(time)
        values = np.asarray(self._fields[frame], dtype=np.float64)
        images = [video.data.get_frame(video.clock.at(time)) for video in self._videos]
        if self._track is None:
            return _Frame(frame, values, images, 0, np.empty((0, len(values))))
        window = self._track.frames_at(time)
        rows = np.asarray(self._fields[window], dtype=np.float64)
        return _Frame(
            frame, values, images, window.start, rows.reshape(-1, len(values))
        )

    def _show(self, time: float, data: _Frame) -> None:
        frame, values, images = data.frame, data.values, data.images
        if self._track is not None:
            self._track.show(time, data.first, data.rows)
        for raster in self._rasters:
            raster.view.follow(time)
        self._field.update(values)
        for video, image in zip(self._videos, images, strict=True):
            video.show(image)
        for layer in self._layers:
            layer.update(time)
        label = (
            self._labels[frame]
            if self._labels is not None
            else f"t = {self._frame_times[frame]:.2f} s"
        )
        parts = [self._title, label, *self._epochs_at(self._start + time)]
        self._plot.title = "  |  ".join(parts)

    def _epochs_at(self, time: float) -> list[str]:
        """Each epoch ``time`` (absolute) is in, as "ripple 3 of 120"."""
        names = []
        for data in self._epochs:
            index = int(np.searchsorted(data.intervals[:, 0], time, side="right")) - 1
            if index >= 0 and time < data.intervals[index, 1]:
                names.append(f"{data.label} {index + 1} of {len(data.intervals)}")
        return names

    @property
    def frame(self) -> int:
        """The frame at the slider's time; set it to move the slider there."""
        return self._frame_of(self.widget.indices["time"])

    @frame.setter
    def frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.widget.indices = {"time": self._clock[frame]}

    def show(self, **kwargs: Any) -> Any:
        """Display the animation with its controls.

        Parameters
        ----------
        **kwargs
            Passed to ``fastplotlib.Figure.show``, e.g. ``sidecar=True`` to
            open it beside the notebook in JupyterLab.

        Returns
        -------
        object
            The canvas: a widget in Jupyter, a window elsewhere.
        """
        if self._canvas is None:
            self._canvas = self.widget.show(autoscale=False, **kwargs)
            # fastplotlib turns y downward for images; environments keep y up.
            self._plot.camera.local.scale_y = abs(self._plot.camera.local.scale_y)
        return self._canvas

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.show())


# The most datapoints a series draws in its window before it thins them.
_MAX_SERIES_POINTS = 100_000


def _add_series(
    widget: Any,
    name: str,
    group: list[TimeSeriesData],
    events: Sequence[EventData],
    epochs: Sequence[EpochData],
    start: float,
) -> None:
    """Draw a group of time series in one subplot, on the widget's clock.

    Times are shifted by ``start``, the first frame's time. Event times are
    marked as vertical lines in the events' colors, and epochs shaded.
    """
    from fastplotlib import LinearSelector

    subplot = widget.figure[name]
    end = widget.ranges["time"].stop
    for item in group:
        times = item.times - start
        finite = np.isfinite(item.data)
        half = item.window_seconds / 2
        if not finite.any() or not (
            times[finite][0] - half <= end and times[finite][-1] + half >= 0.0
        ):
            warnings.warn(
                f"Time series {item.label!r} has no values while the animation "
                "runs, so it is not drawn. Check that its times are on the "
                "same clock as frame_times.",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
            continue
        now = widget.indices["time"]
        around = finite & (np.abs(times - now) <= half)
        if not around.any():
            # fastplotlib cannot add a series with nothing to show: add it
            # where its values start.
            widget.indices = {"time": max(times[finite][0], 0.0)}
        color = _rgba(item.color)
        color[3] *= item.alpha
        rate = 1 / np.median(np.diff(times)) if len(times) > 1 else 1.0
        widget[name].add_nd_timeseries(
            # The GPU draws float32: about 1 us at 10 s, 0.06 ms at 1000 s.
            np.column_stack([times, item.data])[None].astype(np.float32),
            ("series", "time", "xy"),
            ("series", "time", "xy"),
            slider_maps={"time": times},
            display_window=item.window_seconds,
            # Every sample in the window, so fast signals keep their shape.
            max_display_datapoints=min(
                int(np.ceil(item.window_seconds * rate)) + 2, _MAX_SERIES_POINTS
            ),
            x_range_mode="auto",
            colors=color,
            thickness=item.linewidth,
            name=item.label,
        )
    _mark_events(subplot, events, start)
    _shade_epochs(subplot, epochs, start, lines=False)
    subplot.title = "  |  ".join(item.label for item in group)
    subplot.camera.maintain_aspect = False
    low = min(item.global_vmin for item in group)
    high = max(item.global_vmax for item in group)
    margin = 0.05 * (high - low) or 0.5
    subplot.y_range = (low - margin, high + margin)
    for cursor in (g for g in subplot.selectors if isinstance(g, LinearSelector)):
        # The cursor marks the current time; dragging it moves every panel.
        cursor.visible = group[0].show_cursor
        cursor.edge_color = group[0].cursor_color


def _mark_events(subplot: Any, events: Sequence[EventData], start: float) -> None:
    """Event times as vertical lines in the events' colors, on the clock."""
    for data in events:
        for event, event_times in data.event_times.items():
            if len(event_times):
                subplot.add_inf_line(
                    event_times - start,
                    axis="x",
                    colors=data.colors[event],
                    thickness=1.0,
                    name=f"{event} times",
                )


def _shade_epochs(
    subplot: Any, epochs: Sequence[EpochData], start: float, *, lines: bool
) -> None:
    """Epochs on the clock: shaded behind the data, or, over a heatmap that
    shading would tint, as lines at their starts and stops."""
    for data in epochs:
        starts, stops = data.intervals[:, 0] - start, data.intervals[:, 1] - start
        if lines:
            subplot.add_inf_line(
                np.concatenate([starts, stops]),
                axis="x",
                colors=data.color,
                thickness=1.5,
                name=f"{data.label} bounds",
            )
            continue
        # Far past any data, so a band spans the panel at any zoom.
        corners = _rectangles(
            starts, stops, np.full(len(starts), -1e6), np.full(len(starts), 1e6)
        )
        bands = _fans(corners, 4)
        positions = bands.positions.copy()
        positions[:, 2] = -1.0  # behind the data
        subplot.add_mesh(
            positions,
            bands.triangles,
            mode="basic",
            colors=data.color,
            alpha=data.alpha,
            alpha_mode="blend",
            name=f"{data.label} epochs",
        )


def render_fastplotlib(
    env: Environment,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    dry_run: bool = False,
    **options: Any,
) -> FastplotlibAnimation | None:
    """Animate ``fields`` with fastplotlib; see :class:`FastplotlibAnimation`.

    Parameters
    ----------
    env : Environment
        The bins the fields are defined on.
    fields : sequence of ndarray, or ndarray (n_frames, n_bins)
        One field per frame.
    dry_run : bool, default=False
        Check that fastplotlib is installed and report the size of the
        animation, without drawing.
    **options
        The options of :class:`FastplotlibAnimation`.

    Returns
    -------
    FastplotlibAnimation or None
        None for a dry run.
    """
    if dry_run:
        from neurospatial.animation._utils import _emit_status
        from neurospatial.environment._fastplotlib import import_fastplotlib

        import_fastplotlib()
        _emit_status(
            f"Fastplotlib dry run: {len(fields):,} frames of {env.n_bins:,} bins, "
            "each drawn when shown, so nothing is pre-rendered. Nothing was "
            "displayed. To proceed, call again with dry_run=False."
        )
        return None
    return FastplotlibAnimation(env, fields, **options)
