"""Animate fields over time on the GPU with fastplotlib.

``env.animate_fields(fields, frame_times=..., backend="fastplotlib")`` returns
a :class:`FastplotlibAnimation`. Each frame recolors the bins' cells and moves
the overlays; nothing is pre-rendered, so long recordings start at once.

Inside Jupyter, frames are rendered where the kernel runs and streamed to the
browser, with a play button and a slider; this works on a remote GPU server.
In a desktop window, space plays and pauses, the arrow keys step one frame,
and Home and End jump to the ends.

Drawn: position (marker and fading trail), body parts with skeleton, head
direction, events, regions, walls and landmarks. Video, time-series and
object-vector overlays are skipped with a warning.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
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
    draw_colorbar,
    draw_layers,
    frame_view,
    new_figure,
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
            colors=np.ones((2, 4), dtype=np.float32),
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
            self.line.colors = np.tile(np.asarray(color, dtype=np.float32), (2, 1))


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
                markers=_MARKERS.get(data.markers.get(name, "o"), "circle"),
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


def _skip_unsupported(overlay_data: OverlayData) -> OverlayData:
    """Warn about overlays this backend does not draw, and drop them."""
    skipped = [
        name
        for name, items in (
            ("video", overlay_data.videos),
            ("time-series", overlay_data.timeseries),
            ("object-vector", overlay_data.object_vectors),
        )
        if items
    ]
    if not skipped:
        return overlay_data
    warnings.warn(
        f"The fastplotlib backend does not draw {', '.join(skipped)} overlays; "
        "they are skipped and the other overlays are drawn. Use "
        "backend='napari' or backend='video' to include them.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )
    return replace(overlay_data, videos=[], timeseries=[], object_vectors=[])


class FastplotlibAnimation:
    """Fields over time on the GPU, with playback controls.

    Returned by ``env.animate_fields(..., backend="fastplotlib")``. In Jupyter
    it displays itself with a play button and a slider; elsewhere call
    :meth:`show` (in a script, then ``fastplotlib.loop.run()``).

    Parameters
    ----------
    env : Environment
        The bins the fields are defined on.
    fields : sequence of ndarray of shape (n_bins,), or ndarray (n_frames, n_bins)
        One field per frame. An array, including a memory map, is read one
        frame at a time.
    frame_times : ndarray of shape (n_frames,)
        Time of each frame, in seconds; labels the frames.
    fps : float
        Frames per second while playing.
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
    figure : fastplotlib.Figure
        The figure; add graphics to ``figure[0, 0]``.
    n_frames : int
        Number of frames.
    fps : float
        Frames per second while playing.
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
        from neurospatial.animation.rendering import (
            _validate_frame_labels,
            animation_colormap_range,
        )

        self._fields = fields
        self.n_frames = len(fields)
        self.fps = float(fps)
        self._frame_times = np.asarray(frame_times, dtype=np.float64)
        self._labels = _validate_frame_labels(
            None if frame_labels is None else list(frame_labels),
            self.n_frames,
            "fastplotlib",
        )
        low, high = animation_colormap_range(fields, vmin, vmax)  # type: ignore[arg-type]
        scale = ColorScale(cmap, low, high, "lightgray")
        self.figure, self._plot = new_figure(env)
        self._field = FieldGraphic(
            self._plot, env, np.asarray(fields[0], dtype=np.float64), scale
        )
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
        self._layers: list[Any] = []
        if overlay_data is not None:
            overlay_data = _skip_unsupported(overlay_data)
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
        frame_view(self._plot, env)
        self._frame = -1
        self._playing = False
        self._clock = (0.0, 0)
        self._slider: Any = None
        self._controls: Any = None
        self.frame = 0

    @property
    def frame(self) -> int:
        """The frame shown; set it to show another."""
        return self._frame

    @frame.setter
    def frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        if frame == self._frame:
            return
        self._frame = frame
        self._field.update(np.asarray(self._fields[frame], dtype=np.float64))
        for layer in self._layers:
            layer.update(frame)
        label = (
            self._labels[frame]
            if self._labels is not None
            else f"t = {self._frame_times[frame]:.2f} s"
        )
        self._plot.title = f"{self._title}  |  {label}"
        if self._slider is not None and self._slider.value != frame:
            self._slider.value = frame

    def play(self) -> None:
        """Play from the current frame; at the last frame, from the start."""
        if self._frame >= self.n_frames - 1:
            self.frame = 0
        self._clock = (time.perf_counter(), self._frame)
        self._playing = True

    def pause(self) -> None:
        """Stop playing."""
        self._playing = False

    def _advance(self) -> None:
        """Move to the frame the playback clock has reached."""
        if not self._playing:
            return
        started, first = self._clock
        frame = first + int((time.perf_counter() - started) * self.fps)
        if frame >= self.n_frames - 1:
            frame = self.n_frames - 1
            self._playing = False
        self.frame = frame

    def _on_key(self, event: Any) -> None:
        key = getattr(event, "key", None)
        if key == " ":
            self.pause() if self._playing else self.play()
        elif key in ("ArrowRight", "ArrowLeft"):
            self.pause()
            self.frame = self._frame + (1 if key == "ArrowRight" else -1)
        elif key in ("Home", "End"):
            self.pause()
            self.frame = 0 if key == "Home" else self.n_frames - 1

    def show(self) -> Any:
        """Display the animation with its controls.

        Returns
        -------
        object
            In Jupyter, a widget holding the figure, a play button and a
            slider; elsewhere the figure's canvas.
        """
        if self._controls is not None:  # shown already: the controls exist
            return self._controls
        output = self.figure.show()
        if _in_notebook(self.figure.canvas):
            import ipywidgets

            play = ipywidgets.Play(
                min=0,
                max=self.n_frames - 1,
                value=self._frame,
                interval=max(1, int(1000 / self.fps)),
            )
            self._slider = ipywidgets.IntSlider(
                min=0, max=self.n_frames - 1, value=self._frame
            )
            ipywidgets.jslink((play, "value"), (self._slider, "value"))
            self._slider.observe(self._on_slider, names="value")
            self._controls = ipywidgets.VBox(
                [output, ipywidgets.HBox([play, self._slider])]
            )
        else:
            self.figure.renderer.add_event_handler(self._on_key, "key_down")
            self.figure.add_animations(self._advance)
            self._controls = output
        return self._controls

    def _on_slider(self, change: dict[str, Any]) -> None:
        self.frame = change["new"]

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.show())


def _in_notebook(canvas: Any) -> bool:
    """Whether ``canvas`` is a Jupyter widget, which streams frames to a browser."""
    try:
        import ipywidgets
    except ImportError:
        return False
    return isinstance(canvas, ipywidgets.DOMWidget)


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
