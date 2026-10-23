"""Animate fields over time on the GPU with fastplotlib.

``env.animate_fields(fields, frame_times=..., backend="fastplotlib")`` returns
a view whose ``native`` viewer is a :class:`FastplotlibAnimation`, a
fastplotlib ``NDWidget`` with a time
slider in seconds and playback controls. Each frame recolors the bins' cells
and moves the overlays; nothing is pre-rendered, so long recordings start at
once. Inside Jupyter, frames are rendered where the kernel runs and streamed
to the browser, so this works on a remote GPU server.

Drawn: position (marker and fading trail), body parts with skeleton, head
direction, object vectors, events, video placed through its calibration,
time series and spike rasters beside the field on the same clock (with event
times marked), a track's fields against position and time, regions, walls
and landmarks. What is shown at a time (color scales, samples, trails,
arrows, poses, events and the time panels' windows) comes from the scene's
``PreparedScene``, as in every view; this module only draws it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from itertools import pairwise
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.animation._backend_view import _CloseNotification
from neurospatial.animation._prepared import PreparedScene
from neurospatial.animation._style import _head_direction_color, _rate_opacity
from neurospatial.animation._tracks import track_extents, track_segments
from neurospatial.environment._fastplotlib import (
    COLORBAR_WIDTH,
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
    perceived_alpha,
    plot_title,
    xyz,
)
from neurospatial.environment._fastplotlib_compat import (
    _frames_graphic,
    figure_output,
    invalidate_pending,
    mark_displayed,
    playback_state,
    prepare_figure_close,
    register_frames,
    require_animation_support,
    require_volume_support,
    set_playback_fps,
    update_graphic,
)
from neurospatial.environment._fastplotlib_volume import resolve_field_rendering

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from neurospatial.animation._prepared import TrackWindow
    from neurospatial.animation.overlays import (
        BodypartData,
        EpochData,
        EventData,
        HeadDirectionData,
        ObjectVectorData,
        PositionData,
        SpikeRasterData,
        TimeSeriesData,
        VideoData,
    )
    from neurospatial.animation.scene import Animation, AnimationField
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


def _most_within(times: NDArray[np.float64], span: float) -> int:
    """The most of the sorted ``times`` within ``span`` seconds of the first
    of them; at least 1. Sizes the buffers of the panels drawn against time."""
    reach = np.searchsorted(times, times + span, side="right")
    return int(np.max(reach - np.arange(len(times)), initial=1))


def _segments_buffer(
    segments: NDArray[np.float64], capacity: int
) -> NDArray[np.float32]:
    """``segments``, shape (n, 2, n_dims), as a line of ``capacity`` segments.

    Each segment is two vertices and a NaN break; unused room is NaN.
    """
    vertices = np.full((3 * capacity, 3), np.nan, dtype=np.float32)
    vertices[: 3 * len(segments)] = pair_segments(segments[:, 0], segments[:, 1])
    return vertices


class _Field:
    """One field panel's bins, recolored when the field's held row changes."""

    def __init__(
        self, graphic: FieldGraphic, prepared: PreparedScene, field: AnimationField
    ) -> None:
        self.graphic, self.prepared, self.field = graphic, prepared, field

    def prepare(self, time: float) -> NDArray[np.float64]:
        # The held row: the same array until the field's sample changes.
        return self.prepared.field_sample(self.field, time)

    def apply(self, time: float, values: NDArray[np.float64]) -> None:
        if values is not self.graphic.values:
            self.graphic.update(values)


class _Position:
    """The animal's position and its trail, each segment fading with its age."""

    def __init__(
        self, plots: Sequence[Any], prepared: PreparedScene, data: PositionData
    ) -> None:
        self.prepared, self.data = prepared, data
        self.color = _rgba(data.color)
        # Room for the most segments any trail holds.
        self.capacity = (
            max(prepared.trail_capacity(data) - 1, 0) if data.trail_length else 0
        )
        self.trails = [
            plot.add_line(
                np.full((3 * self.capacity, 3), np.nan, dtype=np.float32),
                colors=np.tile(self.color, (3 * self.capacity, 1)),
                thickness=2,
                # Faded segments blend into what is below, not into black.
                alpha_mode="blend",
            )
            for plot in (plots if self.capacity else [])
        ]
        for line in self.trails:
            # Blended after the field, so a 3-D field's translucent points
            # do not hide it.
            line.world_object.render_order = 1
        self.markers = [
            plot.add_scatter(
                xyz([np.nan, np.nan]),
                colors=data.color,
                sizes=data.size,
                edge_colors="white",
                edge_width=1,
            )
            for plot in plots
        ]

    def prepare(self, time: float) -> Any:
        data = self.data
        point = np.array(self.prepared.sample(data, data.data, time))
        return point, self.prepared.trail(data, time) if self.capacity else None

    def apply(self, time: float, prepared: Any) -> None:
        point, trail = prepared
        for marker in self.markers:
            marker.data = xyz(point)
        if trail is not None:
            segments, alphas = trail
            vertices = _segments_buffer(segments, self.capacity)
            colors = np.tile(self.color, (len(vertices), 1))
            colors[:, 3] = 0.0
            colors[: 3 * len(alphas), 3] = np.repeat(perceived_alpha(alphas), 3)
            for line in self.trails:
                line.data = vertices
                line.colors = colors


class _Bodyparts:
    """Body parts as points, joined by the skeleton's observed edges."""

    def __init__(
        self, plots: Sequence[Any], prepared: PreparedScene, data: BodypartData
    ) -> None:
        self.prepared, self.data = prepared, data
        colors = np.array([_rgba(data.colors[name]) for name in data.bodyparts])
        self.points = [
            plot.add_scatter(
                xyz(np.full((len(data.bodyparts), 2), np.nan)),
                colors=colors,
                sizes=8,
                edge_colors="white",
                edge_width=0.5,
            )
            for plot in plots
        ]
        skeleton = data.skeleton
        self.n_edges = 0 if skeleton is None else len(skeleton.edges)
        self.lines = []
        if skeleton is not None and self.n_edges:
            # The skeleton's own style, as in every view.
            self.lines = [
                plot.add_line(
                    np.full((3 * self.n_edges, 3), np.nan, dtype=np.float32),
                    colors=skeleton.edge_color,
                    thickness=skeleton.edge_width,
                )
                for plot in plots
            ]

    def prepare(self, time: float) -> Any:
        return self.prepared.pose(self.data, time)

    def apply(self, time: float, pose: Any) -> None:
        for points in self.points:
            points.data = xyz(pose.points)
        if self.lines:
            edges = _segments_buffer(pose.skeleton, self.n_edges)
            for line in self.lines:
                line.data = edges


class _HeadDirection:
    """The heading line, from the animal's position or the arena's center."""

    def __init__(
        self, plots: Sequence[Any], prepared: PreparedScene, data: HeadDirectionData
    ) -> None:
        self.prepared, self.data = prepared, data
        self.lines = [
            plot.add_line(
                xyz(np.full((2, 2), np.nan)), colors="w", thickness=data.width
            )
            for plot in plots
        ]

    def prepare(self, time: float) -> Any:
        return self.prepared.heading_arrow(self.data, time)

    def apply(self, time: float, arrow: Any) -> None:
        if arrow is None:
            segment = np.full((2, self.prepared.scene.env.n_dims), np.nan)
        else:
            origin, direction = arrow
            segment = np.array([origin, origin + direction])
        for line in self.lines:
            line.data = xyz(segment)
            if arrow is not None:
                line.colors = _head_direction_color(self.data.color, arrow[1])


class _Events:
    """Events at their positions; each update sets only their opacity.

    Every located event is drawn once, in time order; the prepared scene
    says which are shown at a time and how faded.
    """

    def __init__(
        self, plots: Sequence[Any], prepared: PreparedScene, data: EventData
    ) -> None:
        self.prepared, self.data = prepared, data
        self.kinds: dict[str, tuple[Any, Any, list[Any]]] = {}
        self.drawn: dict[str, tuple[int, ...]] = {}
        for name, positions in prepared.event_positions(data).items():
            if len(positions) == 0:
                continue
            fill = np.tile(_rgba(data.colors[name]), (len(positions), 1))
            border = np.tile(_rgba(data.border_color), (len(positions), 1))
            fill[:, 3] = border[:, 3] = 0.0
            marker = _marker(data.markers.get(name, "o"))
            graphics = [
                plot.add_scatter(
                    xyz(positions),
                    colors=fill,
                    sizes=data.size,
                    markers=marker,
                    edge_colors=border,
                    edge_width=data.border_width * data.size,  # a fraction of it
                    alpha_mode="blend",
                )
                for plot in plots
            ]
            self.kinds[name] = (fill, border, graphics)

    def prepare(self, time: float) -> Any:
        return self.prepared.events(self.data, time)

    def apply(self, time: float, shown: Any) -> None:
        for name, (fill, border, graphics) in self.kinds.items():
            visible = shown[name]
            if self.drawn.get(name) == visible.key:
                continue
            self.drawn[name] = visible.key
            fill[:, 3] = border[:, 3] = 0.0
            fill[visible.rows, 3] = border[visible.rows, 3] = perceived_alpha(
                visible.opacity
            )
            for graphic in graphics:
                graphic.colors, graphic.edge_colors = fill, border


class _ObjectVectors:
    """Lines from the animal to each object; opacity follows the firing rate.

    With firing rates, a rate of zero draws faint lines and the highest rate
    opaque ones.
    """

    def __init__(
        self, plots: Sequence[Any], prepared: PreparedScene, data: ObjectVectorData
    ) -> None:
        self.prepared, self.data = prepared, data
        self.color = _rgba(data.color)
        n_objects = len(data.object_positions)
        self.lines = [
            plot.add_line(
                np.full((3 * n_objects, 3), np.nan, dtype=np.float32),
                colors=self.color,
                thickness=data.linewidth,
                alpha_mode="blend",
            )
            for plot in plots
        ]
        if data.show_objects:
            for plot in plots:
                plot.add_scatter(
                    xyz(data.object_positions),
                    colors=self.color,
                    sizes=data.object_size,
                    markers=_marker(data.object_marker),
                    edge_colors="white",
                    edge_width=1,
                )

    def prepare(self, time: float) -> Any:
        data = self.data
        animal = np.array(self.prepared.sample(data, data.animal_positions, time))
        if data.firing_rates is None:
            return animal, None
        rate = self.prepared.sample(data, data.firing_rates, time, name="firing_rates")
        return animal, _rate_opacity(float(rate), data._rate_peak)

    def apply(self, time: float, prepared: Any) -> None:
        animal, opacity = prepared
        objects = self.data.object_positions
        segments = pair_segments(np.broadcast_to(animal, objects.shape), objects)
        color = self.color.copy()
        if opacity is not None:
            color[3] *= opacity
        for line in self.lines:
            line.data = segments
            if opacity is not None:
                line.colors = color


class _Video:
    """Video frames placed in the environment through the calibration.

    ``depth`` puts the frames above or below the field; walls, landmarks and
    the moving overlays are drawn above both.
    """

    def __init__(
        self,
        plots: Sequence[Any],
        prepared: PreparedScene,
        data: VideoData,
        depth: float,
    ) -> None:
        self.prepared, self.data = prepared, data
        if isinstance(data.reader, np.ndarray):
            height, width = data.reader.shape[1:3]
        else:
            width, height = data.reader.frame_size_px
        matrix = np.eye(4)
        matrix[np.ix_([0, 1], [0, 1, 3])] = data.pixel_to_env(width, height)[:2]
        matrix[2, 3] = depth
        self.images = []
        for plot in plots:
            image = plot.add_image(
                np.zeros((height, width, 3), dtype=np.uint8), vmin=0, vmax=255
            )
            image.alpha_mode = "blend"
            image.alpha = float(perceived_alpha(data.alpha))
            image.world_object.local.matrix = matrix
            self.images.append(image)
        self.shown: Any = self  # no frame yet

    def prepare(self, time: float) -> NDArray[np.uint8] | None:
        # The held frame: the same array until the nearest sample changes.
        return self.prepared.video_frame(self.data, time)

    def apply(self, time: float, frame: NDArray[np.uint8] | None) -> None:
        if frame is self.shown:
            return
        self.shown = frame
        for image in self.images:
            image.visible = frame is not None
            if frame is not None:
                image.data = frame


class _TimeWindow:
    """A panel's view of the prepared scene's window around the slider's time.

    A cursor marks the time; dragging it moves every panel. Times on the
    GPU count from the first frame (``start``), so session timestamps keep
    their precision.
    """

    def __init__(
        self,
        subplot: Any,
        prepared: PreparedScene,
        start: float,
        move_to: Callable[[float], None],
        *,
        color: str = "lightgray",
        visible: bool = True,
    ) -> None:
        from fastplotlib import LinearSelector

        self.subplot, self.prepared, self.start = subplot, prepared, start
        # A dark edge, so the cursor shows on light colormap colors too.
        self.halo = subplot.add_inf_line([0.0], axis="x", colors="black", thickness=4)
        self.cursor = LinearSelector(0.0, limits=(-np.inf, np.inf), edge_color=color)
        self.cursor.add_event_handler(
            lambda event: move_to(event.info["value"]), "selection"
        )
        subplot.add_graphic(self.cursor)
        self.halo.visible = self.cursor.visible = visible
        subplot.camera.maintain_aspect = False

    def follow(self, time: float) -> None:
        """Center the view and the cursor on ``time``, in recording seconds."""
        from fastplotlib.graphics.utils import pause_events

        shown = time - self.start
        with pause_events(self.cursor):
            self.cursor.selection = shown
        self.halo.data = [shown]
        left, right = self.prepared.bounds(time)
        self.subplot.x_range = (left - self.start, right - self.start)


def range_ticks(low: float, high: float) -> dict[float, str]:
    """Y ticks at a panel's lowest and highest values, and 0 between them.

    A short panel has room for few automatic ticks, often only one near 0;
    these name its range ("0" and "85.3" for speed in cm/s).
    """
    values = {low, high} | ({0.0} if low < 0.0 < high else set())
    return {value: f"{value:.3g}" for value in sorted(values)}


class _Room:
    """Keeps a panel's data, and the labels of its y range, in view.

    fastplotlib draws the x axis inside the panel, its tick labels' height
    (plus 4 pixels) above the bottom edge, so in a short panel it runs
    through the data; and a y tick label at the very top or bottom of the
    range is centered on it, half outside the panel. The y range is padded
    by as many pixels, refit whenever the panel's height or the data's range
    changes.
    """

    def __init__(self, subplot: Any, low: float, high: float) -> None:
        self.subplot, self.range = subplot, (low, high)
        self._fitted: tuple[Any, ...] | None = None
        subplot.add_animations(self.fit)
        self.fit()

    def set(self, low: float, high: float) -> None:
        """Keep ``low`` to ``high`` in view."""
        self.range = (low, high)
        self.fit()

    def fit(self) -> None:
        axes = self.subplot.axes
        height = float(self.subplot.viewport.rect[3])
        label = axes.y.text.font_size / 2 + 2 if axes.y.visible else 0.0
        below = axes._get_label_margins()[0] + 4 if axes.x.visible else label
        key = (height, below, label, self.range)
        if key == self._fitted or height <= below + label:
            return
        self._fitted = key
        low, high = self.range
        per_pixel = (high - low) / (height - below - label)
        self.subplot.y_range = (low - below * per_pixel, high + label * per_pixel)


class _PositionTime:
    """A track's field over time: a column per sample, a row per bin.

    The prepared scene's track window gives the field's supported samples
    in the window and the positions' paths along the track; this panel
    draws them, missing bins in the missing color and unobserved time and
    the spacing between segments empty. Each segment of a linearized track
    is named at the right of the panel.
    """

    def __init__(
        self,
        subplot: Any,
        prepared: PreparedScene,
        field: AnimationField,
        scale: ColorScale,
        epochs: Sequence[EpochData],
        start: float,
        move_to: Callable[[float], None],
    ) -> None:
        assert prepared.track_extents is not None and prepared.track_rows is not None
        self.subplot, self.prepared, self.field = subplot, prepared, field
        self.scale, self.start = scale, start
        self.lower, self.upper = prepared.track_extents
        self.bin_rows = prepared.track_rows[1]
        env = prepared.track_env
        labels = env._plot_labels()
        self.axis = "position along track" + (
            f" ({labels.units})" if labels.units else ""
        )
        self.mesh: Any = None
        self.paths: Any = None
        self.view = _TimeWindow(subplot, prepared, start, move_to)
        self.set_window()
        segments = track_segments(env) if env.is_linearized_track else []
        self.labels = []
        if len(segments) > 1:
            _segment_joints(subplot, segments, axis="y")
            self.labels = [
                (_edge_label(subplot, name, _SEGMENT_COLOR), (low + high) / 2)
                for name, low, high in segments
            ]
        subplot.title = self.axis
        # A labeled strip above the fields for each kind of epoch marks them
        # without tinting the fields.
        bottom, top = float(self.lower.min()), float(self.upper.max())
        span = top - bottom
        self.strips = [
            (top + (0.015 + 0.045 * k) * span, top + (0.045 + 0.045 * k) * span)
            for k in range(len(epochs))
        ]
        self.labels += [
            (_edge_label(subplot, data.label, data.color), (low + high) / 2)
            for data, (low, high) in zip(epochs, self.strips, strict=True)
        ]
        subplot.y_range = (bottom, self.strips[-1][1] if self.strips else top)

    def set_window(self) -> None:
        """Make room for the cells and paths the prepared window holds."""
        window = self.prepared.window_seconds
        for graphic in (self.mesh, self.paths):
            if graphic is not None:
                self.subplot.delete_graphic(graphic)
        # Columns a window can hold: the samples within it and the one held
        # into it from before its start.
        times = self.field.times
        self.n_columns = min(len(times), _most_within(times, window) + 1)
        # Each cell's corners, counterclockwise from its start at its bin's
        # lower edge: the bins are fixed, and each update moves the times.
        n_bins = len(self.lower)
        self.corners = np.zeros((self.n_columns, n_bins, 4, 3), dtype=np.float32)
        self.corners[:, :, [0, 1], 1] = self.lower[None, :, None]
        self.corners[:, :, [2, 3], 1] = self.upper[None, :, None]
        self.edges = np.zeros((0, 2))  # each drawn column's start and end
        self.values = np.zeros((0, n_bins))  # and its values
        quads = _fans(np.zeros((4 * self.n_columns * n_bins, 2)), 4)
        self.mesh = self.subplot.add_mesh(
            self.corners.reshape(-1, 3),
            quads.triangles,
            mode="basic",
            mapcoords=np.zeros(len(quads.positions), dtype=np.float32),
            cmap=self.scale.texture,
            clim=(0.0, 1.0),
        )
        self.mesh.tooltip_format = self.describe
        # Room for every segment a window holds of every position.
        self.n_paths = max(
            sum(
                _most_within(position.times, window)
                for position in self.prepared.scene._overlay_data.positions
            ),
            1,
        )
        self.paths = self.subplot.add_line(
            np.full((3 * self.n_paths, 3), np.nan, dtype=np.float32), thickness=2
        )

    def prepare(self, time: float) -> TrackWindow:
        return self.prepared.track_window(self.field, time)

    def apply(self, time: float, window: TrackWindow) -> None:
        columns = np.flatnonzero(window.observed)
        n_bins = len(self.lower)
        self.edges = np.column_stack(
            [window.time_edges[columns], window.time_edges[columns + 1]]
        )
        self.values = window.values[self.bin_rows][:, columns].T
        # Each column's start and end; unused columns have no width.
        x = np.zeros((self.n_columns, 1, 2), dtype=np.float32)
        x[: len(columns), 0] = self.edges - self.start
        self.corners[:, :, [0, 3], 0] = x[..., :1]
        self.corners[:, :, [1, 2], 0] = x[..., 1:]
        self.mesh.positions = self.corners.reshape(-1, 3)
        mapcoords = np.zeros(len(self.mesh.positions), dtype=np.float32)
        mapcoords[: 4 * len(columns) * n_bins] = np.repeat(
            self.scale.texture_coordinates(self.values.ravel()), 4
        )
        self.mesh.mapcoords = mapcoords
        paths = window.paths.copy()
        paths[:, :, 0] -= self.start
        self.paths.data = _segments_buffer(paths, self.n_paths)
        colors = np.zeros((3 * self.n_paths, 4), dtype=np.float32)
        palette = {name: _rgba(name) for name in set(window.path_colors)}
        if window.path_colors:
            colors[: 3 * len(paths)] = np.repeat(
                [palette[name] for name in window.path_colors], 3, axis=0
            )
        self.paths.colors = colors
        right = self.prepared.bounds(time)[1] - self.start
        for label, middle in self.labels:  # segments' and epochs' names
            label.offset = (right, middle, 1.0)  # at the right edge
        self.view.follow(time)

    def describe(self, pick_info: dict[str, Any]) -> str:
        """The tooltip for the cell under the pointer."""
        column, bin_index = divmod(pick_info["face_index"] // 2, len(self.lower))
        if column >= len(self.values):
            return ""
        value = self.values[column, bin_index]
        middle = (self.lower[bin_index] + self.upper[bin_index]) / 2
        return "\n".join(
            [
                f"t = {self.edges[column, 0] - self.start:.3f} s",
                f"bin {bin_index}, {self.axis}: {middle:.4g}",
                f"value: {'missing' if np.isnan(value) else f'{value:.4g}'}",
            ]
        )


_SELECTED_COLOR = "orange"
_SEGMENT_COLOR = "lightgray"


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
            segments = track_segments(env) if env.is_linearized_track else []
            if len(segments) > 1:
                _segment_joints(subplot, segments, axis="x")
            subplot.camera.maintain_aspect = False
            self.room = _Room(subplot, 0.0, 1.0)
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
        self.clear()

    def clear(self) -> None:
        """Show no unit."""
        if self.raster is not None:
            self.raster.highlight(None)
        self.raster = None
        if self.extents is not None:
            self.line.data = np.full(self.line.data[:].shape, np.nan, np.float32)
        else:
            self.field.update(np.full(len(self.field.values), np.nan))
        self.subplot.title = "click a unit's row, or press Up or Down, for its rate map"

    def step(self, rasters: Sequence[_Raster], direction: int) -> None:
        """Show the unit one row up (1) or down (-1), from the bottom row."""
        raster = self.raster or next(r for r in rasters if r.data.rates is not None)
        row = -1 if self.raster is None else raster.selected
        n_rows = len(raster.data.spike_times)
        self.show(raster, int(np.clip(row + direction, 0, n_rows - 1)))

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
            self.room.set(0.0, top)
            self.subplot.axes.y.ticks = range_ticks(0.0, top)
        else:
            self.field.scale = ColorScale(self.cmap, 0.0, top, "lightgray")
            self.field.update(rates)
        self.subplot.title = (
            f"unit {raster.data.unit_ids[row]}: rate map, 0 to {peak:.3g} Hz"
        )


class _Raster:
    """Spikes as ticks, a row per unit, in the window around the slider's time.

    Each spike in the window is a vertical segment most of its row's
    height. Hovering anywhere in a row names its unit. With rate maps,
    clicking a unit's row shows its map in ``rates`` and marks the unit: its
    spikes highlighted and its row striped.
    """

    def __init__(
        self,
        subplot: Any,
        prepared: PreparedScene,
        data: SpikeRasterData,
        start: float,
        move_to: Callable[[float], None],
        rates: _UnitRates | None,
    ) -> None:
        self.subplot, self.prepared, self.data = subplot, prepared, data
        self.start, self.rates = start, rates
        self.selected: int = -1
        self.spikes = np.sort(np.concatenate([np.empty(0), *data.spike_times]))
        self.ticks: Any = None
        self.shown: NDArray[np.float64] = np.empty((0, 2, 2))  # ticks in the window
        self.set_window()
        # A band behind each row, far past any time: hovering it names the
        # unit, and the chosen unit's is striped.
        n_rows = len(data.spike_times)
        reach = np.full(n_rows, 1e6)
        rows = np.arange(n_rows, dtype=np.float64)
        bands = _fans(_rectangles(-reach, reach, rows - 0.5, rows + 0.5), 4)
        positions = bands.positions.copy()
        positions[:, 2] = -0.5  # behind the ticks, before epochs
        self.bands = subplot.add_mesh(
            positions,
            bands.triangles,
            mode="basic",
            colors=self._band_colors(),
            alpha_mode="blend",
        )
        self.bands.tooltip_format = self.describe_row
        if rates is not None and data.rates is not None:
            # The whole row answers a click, not only its thin ticks.
            subplot.renderer.add_event_handler(self.clicked, "click")
        self.view = _TimeWindow(subplot, prepared, start, move_to)
        subplot.title = f"spikes, {n_rows} units"
        subplot.y_range = (-0.5, n_rows - 0.5)
        # Its ticks would number the rows, not name the units: hover names them.
        subplot.axes.y.visible = False

    def set_window(self) -> None:
        """Make room for the most spikes a window holds."""
        if self.ticks is not None:
            self.subplot.delete_graphic(self.ticks)
        self.capacity = _most_within(self.spikes, self.prepared.window_seconds)
        self.ticks = self.subplot.add_line(
            np.full((3 * self.capacity, 3), np.nan, dtype=np.float32),
            colors=_rgba(self.data.color),
            thickness=self.data.linewidth,
        )
        self.ticks.tooltip_format = self.describe

    def prepare(self, time: float) -> NDArray[np.float64]:
        return self.prepared.raster_ticks(self.data, time)

    def apply(self, time: float, ticks: NDArray[np.float64]) -> None:
        self.shown = ticks
        drawn = ticks.copy()
        drawn[:, :, 0] -= self.start
        self.ticks.data = _segments_buffer(drawn, self.capacity)
        self.ticks.colors = self._colors()
        self.view.follow(time)

    def _rows(self) -> NDArray[np.intp]:
        """The unit row of each tick in the window."""
        rows: NDArray[np.intp] = np.rint(self.shown[:, :, 1].mean(axis=1)).astype(
            np.intp
        )
        return rows

    def describe(self, pick_info: dict[str, Any]) -> str:
        """The tooltip for the spike under the pointer."""
        spike = int(pick_info["vertex_index"]) // 3
        if spike >= len(self.shown):
            return ""
        unit = self.data.unit_ids[self._rows()[spike]]
        return f"unit {unit}\nt = {self.shown[spike, 0, 0] - self.start:.4f} s"

    def describe_row(self, pick_info: dict[str, Any]) -> str:
        """The tooltip for a row, away from its spikes."""
        row = int(pick_info["face_index"]) // 2
        return f"unit {self.data.unit_ids[row]}"

    def clicked(self, event: Any) -> None:
        """Show the unit of the row clicked, if the click is in this panel."""
        assert self.rates is not None
        where = self.view.subplot.map_screen_to_world(event)
        if where is None:
            return
        row = round(float(where[1]))
        if 0 <= row < len(self.data.spike_times):
            self.rates.show(self, row)

    def _colors(self) -> NDArray[np.float32]:
        """Each vertex's color: the chosen unit's spikes highlighted."""
        colors = np.tile(_rgba(self.data.color), (3 * self.capacity, 1))
        if self.selected >= 0:
            chosen = np.flatnonzero(self._rows() == self.selected)
            colors[(3 * chosen[:, None] + np.arange(3)).ravel()] = _rgba(
                _SELECTED_COLOR
            )
        return colors

    def _band_colors(self) -> NDArray[np.float32]:
        """Each row's band: barely there, and the chosen row's striped."""
        colors = np.zeros((4 * len(self.data.spike_times), 4), dtype=np.float32)
        colors[:, :3] = 1.0
        colors[:, 3] = 1e-4  # unseen, but picked
        if self.selected >= 0:
            colors[4 * self.selected : 4 * self.selected + 4, 3] = perceived_alpha(0.2)
        return colors

    def highlight(self, row: int | None) -> None:
        """Mark the unit of ``row``; None for none."""
        self.selected = -1 if row is None else row
        self.ticks.colors = self._colors()
        self.bands.colors = self._band_colors()


class _Series:
    """Time series sharing a panel, each a line through its samples in the window.

    Every sample in the window is drawn, so a ripple-band LFP keeps its
    shape. The y range spans the whole recording, so the panel stays still.
    """

    def __init__(
        self,
        subplot: Any,
        prepared: PreparedScene,
        group: list[TimeSeriesData],
        start: float,
        move_to: Callable[[float], None],
    ) -> None:
        self.subplot, self.prepared, self.group = subplot, prepared, group
        self.start = start
        self.lines: list[Any] = []
        self.set_window()
        subplot.title = "  |  ".join(item.label for item in group)
        low, high = prepared.signal_limits(group)
        subplot.axes.y.ticks = range_ticks(low, high)
        # A panel's signals share one cursor, styled by its first signal.
        self.view = _TimeWindow(
            subplot,
            prepared,
            start,
            move_to,
            color=group[0].cursor_color,
            visible=group[0].show_cursor,
        )
        _Room(subplot, low, high)

    def set_window(self) -> None:
        """Make room for the most samples a window holds of each signal."""
        for line in self.lines:
            self.subplot.delete_graphic(line)
        window = self.prepared.window_seconds
        self.capacities = [_most_within(item.times, window) for item in self.group]
        self.lines = []
        for item, capacity in zip(self.group, self.capacities, strict=True):
            color = _rgba(item.color)
            color[3] = perceived_alpha(color[3] * item.alpha)
            self.lines.append(
                self.subplot.add_line(
                    np.full((capacity, 3), np.nan, dtype=np.float32),
                    colors=color,
                    thickness=item.linewidth,
                )
            )

    def prepare(self, time: float) -> list[tuple[NDArray[np.float64], Any]]:
        return [self.prepared.signal_window(item, time) for item in self.group]

    def apply(
        self, time: float, windows: list[tuple[NDArray[np.float64], Any]]
    ) -> None:
        for line, capacity, (times, values) in zip(
            self.lines, self.capacities, windows, strict=True
        ):
            points = np.full((capacity, 3), np.nan, dtype=np.float32)
            # The GPU draws float32: about 1 us at 10 s, 0.06 ms at 1000 s.
            points[: len(times)] = xyz(np.column_stack([times - self.start, values]))
            line.data = points
        self.view.follow(time)


def _edge_label(subplot: Any, text: str, color: str) -> Any:
    """An outlined label, placed by its right edge's middle."""
    return subplot.add_text(
        text,
        font_size=13,
        face_color=color,
        outline_color="black",
        outline_thickness=0.25,
        anchor="middle-right",
    )


def _segment_joints(
    subplot: Any, segments: Sequence[tuple[str, float, float]], axis: str
) -> None:
    """Lines where one segment of a maze meets the next along the track.

    Segments with spacing between them are apart already, so get none. A
    dark edge keeps a line visible on light colormap colors too.
    """
    joints = [
        stop
        for (_, _, stop), (_, start, _) in pairwise(segments)
        if np.isclose(stop, start)
    ]
    if joints:
        subplot.add_inf_line(joints, axis=axis, colors="black", thickness=3.0)
        subplot.add_inf_line(joints, axis=axis, colors=_SEGMENT_COLOR, thickness=1.0)


def _panel_names(kind: str, n: int) -> list[str]:
    """Names of the ``n`` panels of a kind: "field", or "field 0", "field 1"."""
    return [f"{kind} {i}" for i in range(n)] if n > 1 else [kind]


def _arrange(
    n: int, layout: str | None, right: float, bottom: float
) -> list[tuple[float, float, float, float]]:
    """Extents of ``n`` fields in the box from (0, 0) to (right, bottom).

    Side by side for up to three, or as ``layout`` says: "horizontal",
    "vertical" or "grid".
    """
    if n == 0:
        return []
    layout = layout or ("horizontal" if n <= 3 else "grid")
    if layout == "horizontal":
        rows, columns = 1, n
    elif layout == "vertical":
        rows, columns = n, 1
    elif layout == "grid":
        columns = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / columns))
    else:
        raise ValueError(
            f"layout must be 'horizontal', 'vertical' or 'grid', got {layout!r}."
        )
    width, height = right / columns, bottom / rows
    return [
        (
            (i % columns) * width,
            (i % columns + 1) * width,
            (i // columns) * height,
            (i // columns + 1) * height,
        )
        for i in range(n)
    ]


def _spatial_overlays(plots: Sequence[Any], prepared: PreparedScene) -> list[Any]:
    """The overlays drawn over every field panel."""
    data = prepared.scene._overlay_data
    layers: list[Any] = [
        _ObjectVectors(plots, prepared, item) for item in data.object_vectors
    ]
    layers += [_Position(plots, prepared, item) for item in data.positions]
    layers += [_Bodyparts(plots, prepared, item) for item in data.bodypart_sets]
    layers += [_HeadDirection(plots, prepared, item) for item in data.head_directions]
    layers += [_Events(plots, prepared, item) for item in data.events]
    return layers


# The most steps a second fastplotlib's frame rate box allows.
_MAX_STEPS = 100


def _playback(
    frame_step: float, overlay_interval: float, speed: float, max_steps: int
) -> tuple[float, int]:
    """The time each step moves, and the steps a second, to play at ``speed``.

    Every field frame is visited: if that needs more than ``max_steps`` a
    second, playback is slower, with a warning. Overlays with their own
    samples are stepped through as finely as the speed allows. The step is
    then shortened a little so the speed comes out exactly.

    Parameters
    ----------
    frame_step : float
        Seconds a step may move without passing over a frame.
    overlay_interval : float
        Typical sampling interval of the fastest overlay with its own times,
        in seconds; infinite without one.
    speed : float
        Seconds of recording a second.
    max_steps : int
        The most steps a second.

    Returns
    -------
    step : float
    steps_per_second : int
    """
    played = min(speed, max_steps * frame_step)
    if played < speed:
        warnings.warn(
            f"Playing every frame at speed {speed:g} needs "
            f"{speed / frame_step:.0f} steps a second; the controls take at most "
            f"{max_steps}, so it plays at speed {played:.2g}. Raise "
            "max_playback_fps (up to 100), lower speed, or subsample the frames.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    step = min(frame_step, max(overlay_interval, played / max_steps))
    steps_per_second = min(max(int(np.ceil(played / step - 1e-9)), 1), max_steps)
    return played / steps_per_second, steps_per_second


def _finest_interval(scene: Animation) -> float:
    """The typical sampling interval of the fastest overlay with its own
    samples, in seconds; infinite without one."""
    return min(
        (
            float(np.median(np.diff(item.times)))
            for item in scene._overlay_data.sampled()
            if len(item.times) > 1
        ),
        default=np.inf,
    )


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
            f"frames closer than {step:.3g} s can be passed over. Step "
            "frame by frame with the Left and Right arrow keys, jump between "
            "events with N given an EpochOverlay, or set animation.frame.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return step


class _Frame(NamedTuple):
    """What one recording time shows, prepared on a worker thread."""

    time: float  # recording seconds
    layers: list[Any]  # each layer's prepared data


class FastplotlibAnimation:
    """Fields over time on the GPU, with a time slider and playback controls.

    The ``native`` viewer of the view that
    ``env.animate_fields(..., backend="fastplotlib")`` and
    ``Animation.show(backend="fastplotlib")`` return; create it through
    them, not directly. It is a
    fastplotlib ``NDWidget``: a slider in seconds, play, step and loop
    buttons, and a frame rate box. Time series and spike rasters are drawn
    beside the field on the same clock, each showing its window around the
    current time, with event times marked. On a track (a linearized track or
    a 1-D environment) the fields are also drawn over time, position along
    the track against time, with the positions as lines over them: a decoded
    posterior beside the animal's position, say. Several fields are shown
    side by side, each with its own color scale and colorbar. Hovering over
    a bin shows its value. In Jupyter it displays itself; elsewhere call
    :meth:`show` (in a script, then ``fastplotlib.loop.run()``).

    The scene decides what is drawn: its fields' colormaps, limits and
    labels, its title, regions, ``window_seconds`` and ``track``. What is
    shown at a time (each field's held sample, trails, arrows, poses,
    events and the windows of the time panels) follows the same rules as in
    the other views.

    A line above the panels gives the time and the epochs it is in (see
    ``status``). Every time shown (the slider, the time axes, that line and
    the tooltips) counts seconds from the first frame, so session timestamps
    (such as 1.7e9 s) keep their precision on the GPU; the line also gives
    the first frame's own time ("t = 4.50 s from 1700000000.00 s").

    Keyboard, with the figure focused (click it first):

    - Space: play or pause.
    - Left and Right arrows: the previous or next frame, one at a time
      however unevenly the frames are spaced.
    - N and Shift+N: the start of the next or previous epoch
      (``EpochOverlay``) or event (``EventOverlay``).
    - ``[`` and ``]``: halve or double the window of every panel drawn
      against time.
    - Up and Down arrows: the unit one row up or down in a spike raster
      with rate maps; Escape: no unit.

    The options of ``Animation.show(backend="fastplotlib")`` and
    ``env.animate_fields(..., backend="fastplotlib")`` that set up this
    viewer, besides the scene, follow.

    Options
    -------
    speed : float, default=1.0
        Playback speed, in seconds of recording a second. Play visits every
        frame; overlays with their own times are sampled as finely as the
        speed allows.
    max_playback_fps : int, default=60
        The most steps a second Play takes (fastplotlib allows 100). When
        visiting every frame at ``speed`` needs more, it plays slower and
        warns.
    frame_labels : sequence of str, optional
        A label per frame; the frame times by default.
    layout : {"horizontal", "vertical", "grid"}, optional
        How several fields are arranged; side by side for up to three, a
        grid for more.
    layer_names : sequence of str, optional
        A name for each field; the scene's field names by default.
    field_rendering : {"auto", "points", "mip", "iso", "slice"}, default="auto"
        Uniform 3D grids use a native maximum intensity volume in auto mode;
        other environments use bin cells or points. Affine transformed grids
        retain their world coordinates. Explicit volume modes require uniform
        3D source cells. Missing and inactive voxels are transparent.
    volume_threshold : float or None, default=None
        Isosurface threshold in original field units, within fixed vmin/vmax.
        None uses their midpoint. Only valid with field_rendering="iso".
    slice_plane : tuple of float or None, default=None
        WORLD XYZ plane (a, b, c, d): a*x + b*y + c*z + d = 0. None uses
        a world-horizontal plane through the volume center. Only valid with
        field_rendering="slice". Values sample nearest voxels.

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
        Several fields are ``"field 0"``, ``"field 1"``, ... and, on a track,
        ``"track 0"``, ``"track 1"``, ...
    n_frames : int
        Number of frames.
    status : str
        What the line above the panels shows: the time ("t = 4.50 s from
        1700000000.00 s", or the frame's label) and the epochs it is in.

    Raises
    ------
    ValueError
        If ``layer_names`` or ``frame_labels`` do not fit the fields and
        frames.
    """

    def __init__(
        self,
        scene: Animation,
        *,
        speed: float = 1.0,
        max_playback_fps: int = 60,
        frame_labels: Sequence[str] | None = None,
        layout: str | None = None,
        layer_names: Sequence[str] | None = None,
        field_rendering: str = "auto",
        volume_threshold: float | None = None,
        slice_plane: tuple[float, float, float, float] | None = None,
    ) -> None:
        env = scene.env
        rendering = resolve_field_rendering(
            env,
            field_rendering,
            volume_threshold=volume_threshold,
            slice_plane=slice_plane,
        )
        fastplotlib = import_fastplotlib()
        require_animation_support(fastplotlib)
        require_volume_support(rendering)
        self._closed = False
        self._close_notification = _CloseNotification()
        self._key_registered = False
        self._source_lock = scene._source_lock
        try:
            # Scales and other source statistics are read under the scene's
            # source lock, before native GUI code that may wait for workers.
            self._prepared = prepared = PreparedScene(scene)
            streams = list(scene.fields.values())
            names = list(scene.fields) if layer_names is None else list(layer_names)
            if len(names) != len(streams):
                raise ValueError(
                    f"layer_names has {len(names)} names for {len(streams)} "
                    "fields; pass one name per field."
                )
            frame_times = scene.frame_times
            self.n_frames = len(frame_times)
            # The widget's clock: seconds from the first frame, so session
            # timestamps keep their precision on the GPU. Only what is drawn
            # counts from it; every stream keeps its own recording clock.
            self._start = float(frame_times[0])
            self._clock = frame_times - self._start
            self._labels = None if frame_labels is None else list(frame_labels)
            if self._labels is not None and len(self._labels) != self.n_frames:
                raise ValueError(
                    f"frame_labels has {len(self._labels)} labels for "
                    f"{self.n_frames} frames. Pass one label per frame, or omit "
                    "frame_labels to label frames by their time."
                )
            data = scene._overlay_data
            groups = list(prepared.signal_groups.values())
            step, steps_per_second = _playback(
                playback_step(frame_times),
                _finest_interval(scene),
                speed,
                min(max_playback_fps, _MAX_STEPS),
            )
            extents_along_track = prepared.track_extents
            several = len(streams) > 1
            # On a 1-D environment the track panel shows all a field panel would,
            # at a size to read: no field panel.
            with_fields = not (env.n_dims == 1 and extents_along_track is not None)
            field_names = _panel_names("field", len(streams)) if with_fields else []
            track_names = (
                _panel_names("track", len(streams))
                if extents_along_track is not None
                else []
            )
            # Panels on the clock fill the right, stacked top to bottom: each
            # field over time on a track, the rasters, then the time series; the
            # fields narrow when the right holds more than time series.
            rasters = data.spike_rasters
            # Heights: a track's fields over time most, then rasters, then series.
            panels = [(name, 3.0) for name in track_names]
            panels += [(f"raster {i}", 2.0) for i in range(len(rasters))]
            panels += [(f"series {i}", 1.0) for i in range(len(groups))]
            time_panels = [name for name, _ in panels]
            # A raster with rate maps adds the chosen unit's map, under the
            # fields or, without them, at the bottom of the right.
            with_rates = any(item.rates is not None for item in rasters)
            rates_under_fields = with_rates and with_fields
            if with_rates and not with_fields:
                panels.append(("unit", 1.5))
            if not with_fields:
                split = 0.0
            else:
                split = (
                    (0.4 if len(time_panels) > len(groups) else 0.6) if panels else 1.0
                )
            bottom = 0.65 if rates_under_fields else 1.0
            extents = dict(
                zip(
                    field_names,
                    _arrange(len(field_names), layout, split, bottom),
                    strict=True,
                )
            )
            if rates_under_fields:
                extents["unit"] = (0.0, split, bottom, 1.0)
            total, top = sum(weight for _, weight in panels), 0.0
            for name, weight in panels:
                extents[name] = (split, 1.0, top / total, (top + weight) / total)
                top += weight
            camera, controller = camera_options(env)
            n_spatial = len(field_names) + rates_under_fields
            # Each field carries its own units, so it gets its own scale and
            # colorbar. With a video below the field, the scene's scale
            # makes low values see-through.
            field_scales = [prepared.field_scales[name] for name in scene.fields]
            track_scales = [prepared.track_scales[name] for name in scene.fields]
            scales = [
                ColorScale(
                    scale.cmap,
                    scale.vmin,
                    scale.vmax,
                    None if env.n_dims == 3 else "lightgray",
                    opaque_from=scale.opaque_from,
                )
                for scale in field_scales
            ]
            self._field_scales = scales
            self._track_scales = [
                ColorScale(scale.cmap, scale.vmin, scale.vmax, "lightgray")
                for scale in track_scales
            ]

            if rendering == "iso" and volume_threshold is not None:
                for scale in scales:
                    if not scale.vmin <= volume_threshold <= scale.vmax:
                        raise ValueError(
                            "volume_threshold must lie between fixed vmin and vmax "
                            f"({scale.vmin}, {scale.vmax}), in field data units."
                        )

            # The slider stops at the last frame.
            self.widget = fastplotlib.NDWidget(
                ranges={"time": (0.0, self._clock[-1] + step, step)},
                extents=extents,
                cameras=[camera] * n_spatial + ["2d"] * len(panels),
                controller_types=[controller] * n_spatial + ["panzoom"] * len(panels),
                size=(1100 if panels or several else 800, 760),
            )
            self.figure = self.widget.figure
            set_playback_fps(self.widget, steps_per_second)
            self._plots = [self.figure[name] for name in field_names]
            # The panel that holds the frames' NDGraphic, and without field
            # panels the colorbar: the first field, or the first track.
            self._plot = self.figure[(field_names or track_names)[0]]
            for name in time_panels:
                # One time axis, at the bottom, and no toolbar: the panels read as
                # one plot against time.
                self.figure[name].toolbar = False
                self.figure[name].axes.x.visible = name == time_panels[-1]

            # An unnamed environment is not titled by its layout's class name.
            self._title = scene.title or (
                plot_title(env) if env._plot_labels().name else ""
            )
            events = data.events
            epochs = data.epochs
            self._epochs = epochs
            # Where N and Shift+N jump: epoch starts and event times, on the
            # widget's clock.
            marks = [item.intervals[:, 0] for item in epochs] + [
                times for item in events for times in item.event_times.values()
            ]
            self._marks = np.unique(np.concatenate([[], *marks])) - self._start
            self._field_graphics: list[FieldGraphic] = []
            with self._source_lock:
                first_values = prepared.field_samples(float(frame_times[0]))
            # No field panels on a 1-D environment.
            fields = []
            for i, (plot, field) in enumerate(zip(self._plots, streams, strict=False)):
                plot.camera.maintain_aspect = (
                    env._plot_labels().coordinates.equal_aspect
                )
                graphic = FieldGraphic(
                    plot,
                    env,
                    first_values[list(scene.fields)[i]],
                    scales[i],
                    field_rendering=rendering,
                    volume_threshold=volume_threshold,
                    slice_plane=slice_plane,
                )
                if data.videos:
                    # Depth orders the field, the videos and everything else (at
                    # 0). Fractional, so that fastplotlib does not restack them.
                    graphic.graphic.offset = (0.0, 0.0, -1.5)
                self._field_graphics.append(graphic)
                fields.append(_Field(graphic, prepared, field))
            self._videos = [
                _Video(
                    self._plots,
                    prepared,
                    item,
                    (-0.5 if item.z_order == "above" else -2.5) - k / 100,
                )
                for k, item in enumerate(data.videos)
            ]
            for plot, shared, graphic in zip(
                self._plots, field_scales, self._field_graphics, strict=False
            ):
                draw_layers(
                    plot,
                    env,
                    show_walls=True,
                    show_landmarks=True,
                    show_regions=scene.show_regions,
                    region_alpha=scene.region_alpha,
                )
                draw_colorbar(plot, graphic.scale, shared.label, beyond=shared.extend)
            self._layers = _spatial_overlays(self._plots, prepared)
            # Each field its name; the time and epochs go above the panels, which
            # a field's title is too narrow for.
            for plot, name in zip(
                self._plots, names if several else [self._title], strict=False
            ):
                plot.title = name
            self.status = ""
            self._tracks: list[_PositionTime] = []
            if extents_along_track is not None:
                for i, (name, label) in enumerate(zip(track_names, names, strict=True)):
                    panel = _PositionTime(
                        self.figure[name],
                        prepared,
                        streams[i],
                        self._track_scales[i],
                        epochs,
                        self._start,
                        self._move_to,
                    )
                    if several:
                        self.figure[name].title = f"{label}: {panel.axis}"
                    self._tracks.append(panel)
            if prepared.projection is not None or not with_fields:
                # Summed along the track, a posterior is on its own scale; on
                # a 1-D environment, the track panels carry the fields'.
                for name, shared, drawn in zip(
                    track_names, track_scales, self._track_scales, strict=True
                ):
                    draw_colorbar(
                        self.figure[name], drawn, shared.label, beyond=shared.extend
                    )
                # Every time panel keeps a colorbar's room, so their axes line up.
                for name in time_panels:
                    self.figure[name].docks["right"].size = COLORBAR_WIDTH
            unit = (
                _UnitRates(self.figure["unit"], env, field_scales[0].cmap)
                if with_rates
                else None
            )
            self._unit = unit
            self._rasters = [
                _Raster(
                    self.figure[f"raster {i}"],
                    prepared,
                    item,
                    self._start,
                    self._move_to,
                    unit,
                )
                for i, item in enumerate(rasters)
            ]
            self._series = [
                _Series(
                    self.figure[f"series {i}"],
                    prepared,
                    group,
                    self._start,
                    self._move_to,
                )
                for i, group in enumerate(groups)
            ]
            self._updates: list[Any] = [
                *self._tracks,
                *self._rasters,
                *self._series,
                *fields,
                *self._videos,
                *self._layers,
            ]
            frames = _frames_graphic(
                self.widget[self._plot],
                self.widget.indices,
                (self.n_frames, env.n_bins),
                self._frame_at,
                self._read,
                self._show,
                # What the right-click menu shows the settings of.
                self._field_graphics[0].graphic
                if self._field_graphics
                else self._tracks[0].view.cursor,
            )
            self._frames = frames
            # fastplotlib has no public way to add a custom NDGraphic yet.
            register_frames(self.widget, self._plot, frames)
            # Events marked, and epochs shaded, in every panel drawn against time;
            # over a track's fields, which shading would tint, as labeled strips.
            strips = {
                name: panel.strips
                for name, panel in zip(track_names, self._tracks, strict=True)
            }
            for name in time_panels:
                _mark_events(self.figure[name], events, self._start)
                _shade_epochs(
                    self.figure[name], epochs, self._start, strips=strips.get(name)
                )
            for plot in self._plots:
                plot.auto_scale()
                frame_view(plot, env)
            self._canvas: Any = None
            self._step = step
            # The panels drawn against time, each titled with the window it shows.
            self._time_titles = {
                self.figure[name]: self.figure[name].title.text for name in time_panels
            }
            self._title_windows()
            if time_panels and not groups:  # a series keeps its own room
                # The one time axis, at the bottom, clear of the data.
                bottom_panel = self.figure[time_panels[-1]]
                _Room(bottom_panel, *bottom_panel.y_range)
            self.figure.renderer.add_event_handler(self._key, "key_down")
            self._key_registered = True
            self._add_key_hint()
            self._show(0.0, self._prepare(float(frame_times[0])))
            self.figure.canvas.add_event_handler(self._canvas_closed, "close")
        except BaseException as exc:
            from neurospatial.animation._resources import _add_cleanup_note

            try:
                self.close()
            except BaseException as cleanup:
                _add_cleanup_note(
                    exc, f"Native animation cleanup also failed: {cleanup!r}"
                )
            raise

    @property
    def window_seconds(self) -> float:
        """Seconds shown by every panel drawn against time."""
        return self._prepared.window_seconds

    def _add_key_hint(self) -> None:
        """Two lines above the panels, as wide as the figure: the status (the
        time and the epochs it is in) and the keyboard shortcuts."""
        from imgui_bundle import imgui

        keys = ["Space play", "Left/Right frame", "[ ] window"]
        if len(self._marks):
            keys.insert(2, "N/Shift+N next/previous event")
        if self._unit is not None:
            keys.append("Up/Down unit, Esc none")
        self.key_hint = "Keys, after a click on the figure:  " + "  |  ".join(keys)

        @self.figure.add_imgui_window(location="top", size=48, title=None)
        def hint() -> None:
            imgui.text(self.status)
            imgui.text_disabled(self.key_hint)

    def _key(self, event: Any) -> None:
        """The keyboard shortcuts listed in the class docstring."""
        key = event.key
        if key in (" ", "Space"):
            playing = playback_state(self.widget)
            playing["time"] = not playing["time"]
        elif key in ("ArrowRight", "ArrowLeft"):
            self.frame = self.frame + (1 if key == "ArrowRight" else -1)
        elif key.lower() == "n":
            self._jump(-1 if "Shift" in event.modifiers or key == "N" else 1)
        elif key in ("[", "]"):
            self._scale_windows(0.5 if key == "[" else 2.0)
        elif key in ("ArrowUp", "ArrowDown") and self._unit is not None:
            self._unit.step(self._rasters, 1 if key == "ArrowUp" else -1)
        elif key == "Escape" and self._unit is not None:
            self._unit.clear()

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

    def _scale_windows(self, factor: float) -> None:
        """Widen or narrow the window every panel drawn against time shares."""
        longest = float(self._clock[-1]) + self._step
        window = float(
            np.clip(self.window_seconds * factor, self._step, max(longest, self._step))
        )
        # A read already on the worker was sized for the old window.
        self._frames.synchronize()
        with self._source_lock:
            self._prepared.window_seconds = window
        self._title_windows()
        panels: list[_PositionTime | _Raster | _Series] = [
            *self._tracks,
            *self._rasters,
            *self._series,
        ]
        for panel in panels:
            panel.set_window()
        self._show(0.0, self._prepare(self.time))

    def _title_windows(self) -> None:
        """Name the shared window in the title of each panel drawn against time."""
        for subplot, title in self._time_titles.items():
            subplot.title = self._windowed(title)

    def _windowed(self, title: str) -> str:
        return f"{title}  ·  {self.window_seconds:.3g} s window"

    def _move_to(self, time: float) -> None:
        """Move the slider, and so every panel, to ``time`` on the clock."""
        self.widget.indices = {"time": time}

    def _frame_at(self, time: float) -> int:
        """The frame shown at ``time`` on the widget's clock: the last one
        that started (the first, before any)."""
        return max(int(np.searchsorted(self._clock, time, side="right")) - 1, 0)

    def _read(self, time: float) -> _Frame:
        """What the slider's ``time`` shows; runs on a worker thread."""
        return self._prepare(time + self._start)

    def _prepare(self, time: float) -> _Frame:
        """Read and sample every layer at recording ``time``.

        The scene's source lock serializes these reads with other views,
        exports and this view's own GUI-thread reads.
        """
        with self._source_lock:
            return _Frame(time, [layer.prepare(time) for layer in self._updates])

    def _show(self, time: float, frame: _Frame) -> None:
        """Draw ``frame``; ``time`` is the slider's, which ``frame`` was read for."""
        for layer, prepared in zip(self._updates, frame.layers, strict=True):
            layer.apply(frame.time, prepared)
        shown = frame.time - self._start
        # The slider's time, which the overlays and epochs show, not the time
        # of the field frame held; a frame's own label names the frame.
        label = (
            self._labels[self._frame_at(shown)]
            if self._labels is not None
            else f"t = {shown:.2f} s"
        )
        if self._start != 0.0 and self._labels is None:
            # Every time shown counts from the first frame; its own, once.
            label += f" from {self._start:.2f} s"
        # Without a field panel to carry it, the title heads the status.
        parts = [label, *self._epochs_at(frame.time)]
        if not self._plots:
            parts.insert(0, self._title)
        self.status = "  |  ".join(part for part in parts if part)

    def _epochs_at(self, time: float) -> list[str]:
        """Each epoch recording ``time`` is in, as "ripple 3 of 120"."""
        names = []
        for data in self._epochs:
            index = int(np.searchsorted(data.intervals[:, 0], time, side="right")) - 1
            if index >= 0 and time < data.intervals[index, 1]:
                names.append(f"{data.label} {index + 1} of {len(data.intervals)}")
        return names

    @property
    def time(self) -> float:
        """Current absolute recording time, in seconds."""
        return float(self.widget.indices["time"]) + self._start

    @property
    def frame(self) -> int:
        """The frame at the slider's time; set it to move the slider there."""
        return self._frame_at(self.widget.indices["time"])

    @frame.setter
    def frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.widget.indices = {"time": self._clock[frame]}

    def seek(self, time: float) -> None:
        """Move every panel to an absolute recording time, in seconds."""
        if self._closed:
            raise RuntimeError(
                "This animation is closed. Create a new view before seeking."
            )
        if not np.isfinite(time):
            raise ValueError("time must be a finite recording time in seconds.")
        self.widget.indices = {
            "time": float(np.clip(time - self._start, 0, self._clock[-1]))
        }

    @contextmanager
    def capture_session(self) -> Iterator[Callable[[float], NDArray[np.uint8]]]:
        """Capture the native figure synchronously, then restore viewer state.

        The yielded callable takes absolute recording times and returns RGB
        pixels at the existing canvas size. Camera edits, added graphics and
        the selected unit's rate map remain part of the rendered figure.
        Playback and asynchronous field reads stay suspended until exit.
        """
        if self._closed:
            raise RuntimeError(
                "This animation is closed. Create a new view before exporting."
            )
        self.show()
        original_time = self.time
        playing = playback_state(self.widget)
        original_playing = dict(playing)
        graphics = list(self.widget.ndgraphics)
        paused = [graphic.pause for graphic in graphics]
        cameras = [(plot.camera, plot.camera.get_state()) for plot in self.figure]
        first, last = (float(t) for t in self._prepared.scene.frame_times[[0, -1]])

        def update(time: float) -> None:
            if self._closed:
                raise RuntimeError("This animation was closed during capture.")
            # Drawn at the recording time itself, not one rounded through
            # the widget's clock.
            time = float(np.clip(time, first, last))
            shown = time - self._start
            self.widget.indices = {"time": shown}
            self._show(shown, self._prepare(time))
            mark_displayed(self._frames, shown)
            for graphic, was_paused in zip(graphics, paused, strict=True):
                if graphic is not self._frames and not was_paused:
                    update_graphic(graphic)

        def capture(time: float) -> NDArray[np.uint8]:
            if not np.isfinite(time):
                raise ValueError("time must be a finite recording time in seconds.")
            update(time)
            # force_draw works on GUI, notebook and offscreen canvases. It
            # submits all updated GPU buffers before the snapshot reads back.
            self.figure.canvas.force_draw()
            return np.ascontiguousarray(
                self.figure.export_numpy(rgb=True), dtype=np.uint8
            )

        synchronized = False
        try:
            for dimension in playing:
                playing[dimension] = False
            for graphic in graphics:
                graphic.pause = True
            # Reads queued before export must never overwrite captured frames
            # or the restored viewer when their callbacks resume afterward.
            invalidate_pending(self.widget, graphics)
            self._frames.synchronize()
            synchronized = True
            yield capture
        finally:
            try:
                if synchronized and not self._closed:
                    try:
                        self._frames.synchronize()
                    finally:
                        if not self._closed:
                            update(original_time)
            finally:
                try:
                    if not self._closed:
                        for camera, state in cameras:
                            camera.set_state(state)
                        if synchronized:
                            self.figure.canvas.force_draw()
                finally:
                    if not self._closed:
                        for graphic, pause in zip(graphics, paused, strict=True):
                            graphic.pause = pause
                        playing.update(original_playing)

    def capture_frame(self, time: float) -> NDArray[np.uint8]:
        """Return the native figure's RGB pixels at an absolute recording time."""
        with self.capture_session() as capture:
            return capture(time)

    def _canvas_closed(self, event: Any) -> None:
        """Closing the native canvas also stops the workers and the view."""
        self.close()

    def _add_close_callback(self, callback: Callable[[], None]) -> None:
        self._close_notification.add(callback)

    def close(self) -> None:
        """Stop worker reads and release complete or partly constructed viewers."""
        if self._closed:
            return
        self._closed = True
        widget = getattr(self, "widget", None)
        figure = getattr(self, "figure", None)
        frames = getattr(self, "_frames", None)
        playing = playback_state(widget)
        for dimension in playing:
            playing[dimension] = False
        # Continue cleanup after native-window errors so all workers drain
        # before the scene may release its readers; report every failure
        # afterward.
        from neurospatial.animation._resources import _raise_cleanup_errors

        failures: list[BaseException] = []

        def release(action: Callable[[], Any]) -> None:
            try:
                action()
            except Exception as exc:
                failures.append(exc)

        if frames is not None:
            release(frames.close)
        for graphic in getattr(widget, "ndgraphics", []):
            graphic.pause = True
            if graphic is not frames:
                release(graphic.slicer.close)
        if figure is not None:
            if self._key_registered:
                release(
                    lambda: figure.renderer.remove_event_handler(self._key, "key_down")
                )
                self._key_registered = False
            prepare_figure_close(figure)
        if widget is not None:
            release(widget.close)
        canvas = getattr(self, "_canvas", None)
        if canvas is not None:
            if canvas is not figure_output(figure):
                release(canvas.close)
            self._canvas = None
        prepared = getattr(self, "_prepared", None)
        if prepared is not None:
            # The workers have drained: release the view's held rows.
            release(prepared.clear_cache)
        release(self._close_notification.notify)
        _raise_cleanup_errors("Failed to close native animation resources.", failures)

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
            if self._canvas is None:
                # fastplotlib's offscreen show initializes layout but returns
                # None; retain its actual canvas for capture and cleanup.
                self._canvas = self.figure.canvas
                prepare_figure_close(self.figure)
            # fastplotlib turns y downward in every subplot with an image (a
            # video under each field); environments keep y up.
            for plot in self._plots:
                plot.camera.local.scale_y = abs(plot.camera.local.scale_y)
        return self._canvas

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.show())


def _mark_events(subplot: Any, events: Sequence[EventData], start: float) -> None:
    """Event times as vertical lines in the events' colors, on the widget's clock."""
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
    subplot: Any,
    epochs: Sequence[EpochData],
    start: float,
    *,
    strips: Sequence[tuple[float, float]] | None = None,
) -> None:
    """Epochs on the widget's clock, shaded behind the data; over a heatmap,
    which shading would tint, each kind as a solid strip between its heights
    in ``strips``."""
    for k, data in enumerate(epochs):
        starts, stops = data.intervals[:, 0] - start, data.intervals[:, 1] - start
        strip = strips[k] if strips is not None else None
        # Far past any data, so a band spans the panel at any zoom.
        low, high = strip or (-1e6, 1e6)
        corners = _rectangles(
            starts, stops, np.full(len(starts), low), np.full(len(starts), high)
        )
        bands = _fans(corners, 4)
        positions = bands.positions.copy()
        positions[:, 2] = 0.0 if strip else -1.0  # behind the data
        subplot.add_mesh(
            positions,
            bands.triangles,
            mode="basic",
            colors=data.color,
            alpha=1.0 if strip else float(perceived_alpha(data.alpha)),
            alpha_mode="blend",
            name=f"{data.label} epochs",
        )
