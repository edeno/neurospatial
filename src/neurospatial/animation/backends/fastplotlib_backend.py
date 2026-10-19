"""Animate fields over time on the GPU with fastplotlib.

``env.animate_fields(fields, frame_times=..., backend="fastplotlib")`` returns
a :class:`FastplotlibAnimation`, a fastplotlib ``NDWidget`` with a time
slider in seconds and playback controls. Each frame recolors the bins' cells
and moves the overlays; nothing is pre-rendered, so long recordings start at
once. Inside Jupyter, frames are rendered where the kernel runs and streamed
to the browser, so this works on a remote GPU server.

Drawn: position (marker and fading trail), body parts with skeleton, head
direction, events, video placed through its calibration, time series beside
the field on the same clock, regions, walls and landmarks. Object-vector
overlays are skipped with a warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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
        EventData,
        HeadDirectionData,
        OverlayData,
        PositionData,
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


class _Position:
    """The animal's position and a fading trail of recent positions."""

    def __init__(self, subplot: Any, data: PositionData) -> None:
        self.data = data
        self.length = data.trail_length or 0
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

    def update(self, frame: int) -> None:
        self.marker.data = xyz(self.data.data[frame])
        if self.length > 1:
            window = np.full((self.length, self.data.data.shape[1]), np.nan)
            recent = self.data.data[max(0, frame - self.length + 1) : frame + 1]
            window[self.length - len(recent) :] = recent
            self.trail.data = xyz(window)


class _Bodyparts:
    """Body parts as points, joined by the skeleton's edges."""

    def __init__(self, subplot: Any, data: BodypartData) -> None:
        self.data = data
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
                self._segments(0),
                colors=skeleton.edge_color,
                thickness=skeleton.edge_width,
            )

    def _segments(self, frame: int) -> NDArray[np.float32]:
        parts = self.data.bodyparts
        return pair_segments(
            [parts[start][frame] for start, _ in self.edges],
            [parts[end][frame] for _, end in self.edges],
        )

    def update(self, frame: int) -> None:
        parts = self.data.bodyparts
        self.points.data = xyz([parts[name][frame] for name in self.names])
        if self.edges:
            self.lines.data = self._segments(frame)


class _HeadDirection:
    """A heading line from the animal's position, or from the arena's center."""

    def __init__(
        self,
        subplot: Any,
        data: HeadDirectionData,
        anchor: PositionData | None,
        center: NDArray[np.float64],
    ) -> None:
        self.data, self.anchor, self.center = data, anchor, center
        self.line = subplot.add_line(
            xyz(np.full((2, 2), np.nan)),
            colors="w",
            thickness=data.width,
        )

    def update(self, frame: int) -> None:
        origin = self.center
        if self.anchor is not None and np.all(np.isfinite(self.anchor.data[frame])):
            origin = self.anchor.data[frame]
        segment = heading_segment(origin, self.data.data[frame], self.data.length)
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

    def __init__(self, subplot: Any, data: EventData) -> None:
        self.data = data
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

    def update(self, frame: int) -> None:
        for frames, fill, border, graphic in self.kinds:
            shown, alphas = _event_visibility(frames, frame, self.data.decay_frames)
            fill[:, 3] = border[:, 3] = np.where(shown, alphas * self.data.opacity, 0)
            graphic.colors, graphic.edge_colors = fill, border


class _Video:
    """Video frames placed in the environment through the calibration.

    ``depth`` puts the frames above or below the field; walls, landmarks and
    the moving overlays are drawn above both.
    """

    def __init__(self, subplot: Any, data: VideoData, depth: float) -> None:
        self.data = data
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

    def update(self, frame: int) -> None:
        image = self.data.get_frame(frame)
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


def _skip_object_vectors(overlay_data: OverlayData) -> OverlayData:
    """Warn that this backend does not draw object vectors, and drop them."""
    if not overlay_data.object_vectors:
        return overlay_data
    warnings.warn(
        "The fastplotlib backend does not draw object-vector overlays; they "
        "are skipped and the other overlays are drawn. Use backend='html' "
        "to include them.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )
    return replace(overlay_data, object_vectors=[])


def _frames_graphic(
    nd_subplot: Any,
    indices: Any,
    fields: Any,
    frame_of: Callable[[float], int],
    show: Callable[[int, NDArray[np.float64]], None],
) -> Any:
    """The field and every frame-aligned overlay, as one ``NDGraphic``.

    The widget's slider is in seconds; ``frame_of`` maps it to the frame
    shown, which is held until the next frame starts. Everything drawn for a
    frame is updated together, so the overlays never show another frame
    than the field. Built in a function because fastplotlib is optional.
    """
    from fastplotlib.widgets.nd_widget import NDGraphic, NDSlicer

    class FieldSlicer(NDSlicer):  # type: ignore[misc,unused-ignore]
        async def get(self, indices: dict[str, float]) -> NDArray[np.float64]:
            values = await self.get_window_output(indices)
            return np.asarray(values, dtype=np.float64)

    class Frames(NDGraphic):  # type: ignore[misc,unused-ignore]
        def __init__(self) -> None:
            super().__init__(nd_subplot, "field")
            self._slicer = FieldSlicer(
                fields, ("time", "bin"), ("bin",), slider_maps={"time": frame_of}
            )

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
            show(frame_of(at["time"]), await self._slicer.get(at))
            self._last_indices = at

    return Frames()


class FastplotlibAnimation:
    """Fields over time on the GPU, with a time slider and playback controls.

    Returned by ``env.animate_fields(..., backend="fastplotlib")``. It is a
    fastplotlib ``NDWidget``: a slider in seconds, play, step and loop
    buttons, and a frame rate box. Time-series overlays are drawn beside the
    field on the same clock, each showing its window around the current
    time. In Jupyter it displays itself; elsewhere call :meth:`show` (in a
    script, then ``fastplotlib.loop.run()``).

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
        Frames per second while playing; changeable in the controls.
    cmap, vmin, vmax
        Colormap and its limits; limits default to the range over all frames,
        estimated from a sample of a very long recording.
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

    Attributes
    ----------
    widget : fastplotlib.NDWidget
        The widget; ``widget.indices["time"]`` is the time shown.
    figure : fastplotlib.Figure
        The widget's figure; add graphics to ``figure["field"]``.
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
        self._labels = _validate_frame_labels(
            None if frame_labels is None else list(frame_labels),
            self.n_frames,
            "fastplotlib",
        )
        if overlay_data is not None:
            overlay_data = _skip_object_vectors(overlay_data)
        groups = _group_timeseries(overlay_data.timeseries if overlay_data else [])
        times = self._frame_times
        step = float(np.median(np.diff(times))) if self.n_frames > 1 else 1.0
        # The slider stops at the last frame; series subplots fill the right.
        extents = {"field": (0.0, 0.6 if groups else 1.0, 0.0, 1.0)}
        for i in range(len(groups)):
            extents[f"series {i}"] = (0.6, 1.0, i / len(groups), (i + 1) / len(groups))
        camera, controller = camera_options(env)
        self.widget = fastplotlib.NDWidget(
            ranges={"time": (times[0], times[-1] + step, step)},
            extents=extents,
            cameras=[camera] + ["2d"] * len(groups),
            controller_types=[controller] + ["panzoom"] * len(groups),
            size=(1100 if groups else 800, 760),
        )
        # Playback steps one frame per 1 / fps seconds; the box allows 1-100.
        self.widget.ui_sliders._fps["time"] = min(max(round(fps), 1), 100)
        self.widget.ui_sliders._frame_time["time"] = (
            1 / self.widget.ui_sliders._fps["time"]
        )
        self.figure = self.widget.figure
        self._plot = self.figure["field"]
        self._plot.camera.maintain_aspect = env._plot_labels().coordinates.equal_aspect

        low, high = animation_colormap_range(fields, vmin, vmax)  # type: ignore[arg-type]
        scale = ColorScale(cmap, low, high, "lightgray")
        self._field = FieldGraphic(
            self._plot, env, np.asarray(fields[0], dtype=np.float64), scale
        )
        self._layers: list[Any] = []
        if overlay_data is not None and overlay_data.videos:
            # Depth orders the field, the videos and everything else (at 0).
            # Fractional, so that fastplotlib does not restack the videos.
            self._field.graphic.offset = (0.0, 0.0, -1.5)
            self._layers += [
                _Video(
                    self._plot,
                    item,
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
        if overlay_data is not None:
            positions = overlay_data.positions
            self._layers += [_Position(self._plot, item) for item in positions]
            self._layers += [
                _Bodyparts(self._plot, item) for item in overlay_data.bodypart_sets
            ]
            anchor = positions[0] if len(positions) == 1 else None
            center = env.bin_centers.mean(axis=0)
            self._layers += [
                _HeadDirection(self._plot, item, anchor, center)
                for item in overlay_data.head_directions
            ]
            self._layers += [_Events(self._plot, item) for item in overlay_data.events]
        frames = _frames_graphic(
            self.widget[self._plot],
            self.widget.indices,
            self._fields,
            self._frame_of,
            self._show_frame,
        )
        # fastplotlib has no public way to add a custom NDGraphic yet.
        self.widget[self._plot]._nd_graphics.append(frames)
        for i, group in enumerate(groups):
            _add_series(self.widget, f"series {i}", group)
        # Only the field is fitted: fitting a series subplot's camera moves
        # the time, which follows the center of its view.
        self._plot.auto_scale()
        frame_view(self._plot, env)
        self._canvas: Any = None
        self._shown = -1
        self._show_frame(0, np.asarray(fields[0], dtype=np.float64))

    def _frame_of(self, time: float) -> int:
        """The frame shown at ``time``: the last one that started."""
        return max(int(np.searchsorted(self._frame_times, time, side="right")) - 1, 0)

    def _show_frame(self, frame: int, values: NDArray[np.float64]) -> None:
        self._shown = frame
        self._field.update(values)
        for layer in self._layers:
            layer.update(frame)
        label = (
            self._labels[frame]
            if self._labels is not None
            else f"t = {self._frame_times[frame]:.2f} s"
        )
        self._plot.title = f"{self._title}  |  {label}"

    @property
    def frame(self) -> int:
        """The frame at the slider's time; set it to move the slider there."""
        return self._frame_of(self.widget.indices["time"])

    @frame.setter
    def frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        self.widget.indices = {"time": self._frame_times[frame]}

    def show(self) -> Any:
        """Display the animation with its controls.

        Returns
        -------
        object
            The canvas: a widget in Jupyter, a window elsewhere.
        """
        if self._canvas is None:
            self._canvas = self.widget.show(autoscale=False)
            # fastplotlib turns y downward for images; environments keep y up.
            self._plot.camera.local.scale_y = abs(self._plot.camera.local.scale_y)
        return self._canvas

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.show())


def _add_series(widget: Any, name: str, group: list[TimeSeriesData]) -> None:
    """Draw a group of time series in one subplot, on the widget's clock."""
    subplot = widget.figure[name]
    for item in group:
        color = _rgba(item.color)
        color[3] *= item.alpha
        widget[name].add_nd_timeseries(
            # The GPU draws float32: times resolve to 1 us at 10 s, 0.06 ms at 1000 s.
            np.column_stack([item.times, item.data])[None].astype(np.float32),
            ("series", "time", "xy"),
            ("series", "time", "xy"),
            slider_maps={"time": item.times},
            display_window=item.window_seconds,
            x_range_mode="auto",
            colors=color,
            thickness=item.linewidth,
            name=item.label,
        )
    subplot.title = "  |  ".join(item.label for item in group)
    subplot.camera.maintain_aspect = False
    low = min(item.global_vmin for item in group)
    high = max(item.global_vmax for item in group)
    margin = 0.05 * (high - low) or 0.5
    subplot.y_range = (low - margin, high + margin)


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
