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
from neurospatial.environment._fastplotlib import (
    ColorScale,
    FieldGraphic,
    color_limits,
    draw_colorbar,
    draw_layers,
    frame_view,
    new_figure,
    title,
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
_TRAIL_MAX_ALPHA = 0.7


def _rgba(color: str) -> NDArray[np.float32]:
    from matplotlib.colors import to_rgba

    return np.asarray(to_rgba(color), dtype=np.float32)


def _xyz(points: NDArray[np.float64]) -> NDArray[np.float32]:
    """(n, n_dims) points as (n, 3), with z = 0 in two dimensions."""
    points = np.atleast_2d(points)
    out = np.zeros((len(points), 3), dtype=np.float32)
    out[:, : min(points.shape[1], 3)] = points[:, :3]
    return out


class _Position:
    """The animal's position and a fading trail of recent positions."""

    def __init__(self, subplot: Any, data: PositionData) -> None:
        self.data = data
        color = _rgba(data.color)
        self.length = data.trail_length or 0
        if self.length > 1:
            alphas = np.linspace(0.0, _TRAIL_MAX_ALPHA, self.length)
            colors = np.tile(color, (self.length, 1))
            colors[:, 3] = alphas
            self.trail = subplot.add_line(
                np.full((self.length, 3), np.nan, dtype=np.float32),
                colors=colors,
                thickness=2,
            )
        self.marker = subplot.add_scatter(
            np.full((1, 3), np.nan, dtype=np.float32),
            colors=data.color,
            sizes=data.size,
            edge_colors="white",
            edge_width=1,
        )

    def update(self, frame: int) -> None:
        self.marker.data = _xyz(self.data.data[frame])
        if self.length > 1:
            window = np.full((self.length, self.data.data.shape[1]), np.nan)
            recent = self.data.data[max(0, frame - self.length + 1) : frame + 1]
            window[self.length - len(recent) :] = recent
            self.trail.data = _xyz(window)


class _Bodyparts:
    """Body parts as points, joined by the skeleton's edges."""

    def __init__(self, subplot: Any, data: BodypartData) -> None:
        self.data = data
        self.names = list(data.bodyparts)
        colors = [(data.colors or {}).get(name, "cyan") for name in self.names]
        self.points = subplot.add_scatter(
            np.full((len(self.names), 3), np.nan, dtype=np.float32),
            colors=np.array([_rgba(color) for color in colors]),
            sizes=8,
            edge_colors="white",
            edge_width=0.5,
        )
        skeleton = data.skeleton
        self.edges = (
            [
                (start, end)
                for start, end in skeleton.edges
                if start in data.bodyparts and end in data.bodyparts
            ]
            if skeleton is not None
            else []
        )
        if self.edges:
            assert skeleton is not None
            self.lines = subplot.add_line(
                np.full((3 * len(self.edges), 3), np.nan, dtype=np.float32),
                colors=skeleton.edge_color,
                thickness=skeleton.edge_width,
            )

    def update(self, frame: int) -> None:
        parts = self.data.bodyparts
        self.points.data = _xyz(np.array([parts[name][frame] for name in self.names]))
        if self.edges:
            n_dims = next(iter(parts.values())).shape[1]
            segments = np.full((3 * len(self.edges), n_dims), np.nan)
            segments[0::3] = [parts[start][frame] for start, _ in self.edges]
            segments[1::3] = [parts[end][frame] for _, end in self.edges]
            self.lines.data = _xyz(segments)


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
            np.full((2, 3), np.nan, dtype=np.float32),
            colors=np.tile(np.array([1, 1, 1, 1], dtype=np.float32), (2, 1)),
            thickness=data.width,
        )

    def update(self, frame: int) -> None:
        from neurospatial.animation._parallel import _head_direction_color

        origin = self.center
        if self.anchor is not None and np.all(np.isfinite(self.anchor.data[frame])):
            origin = self.anchor.data[frame][:2]
        heading = self.data.data[frame]
        if self.data.data.ndim == 1:
            direction = np.array([np.cos(heading), np.sin(heading)])
        else:
            direction = np.asarray(heading[:2], dtype=np.float64)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 0 else direction
        direction = direction * self.data.length
        if not np.all(np.isfinite(direction)):
            self.line.data = np.full((2, 3), np.nan, dtype=np.float32)
            return
        self.line.data = _xyz(np.array([origin, origin + direction]))
        color = np.asarray(
            _head_direction_color(self.data.color, direction), dtype=np.float32
        )
        self.line.colors = np.tile(color, (2, 1))


class _Events:
    """Events at their positions: all so far, only the current, or fading."""

    def __init__(self, subplot: Any, data: EventData) -> None:
        self.data = data
        self.kinds = []
        for name, positions in data.event_positions.items():
            if len(positions) == 0:
                continue
            color = _rgba(data.colors[name])
            graphic = subplot.add_scatter(
                np.full((len(positions), 3), np.nan, dtype=np.float32),
                colors=np.tile(color, (len(positions), 1)),
                sizes=data.size,
                markers=_MARKERS.get(data.markers.get(name, "o"), "circle"),
                edge_colors=data.border_color,
                edge_width=data.border_width * data.size,  # a fraction of the size
                alpha_mode="blend",
            )
            self.kinds.append((name, positions, color, graphic))

    def update(self, frame: int) -> None:
        decay = self.data.decay_frames
        for name, positions, color, graphic in self.kinds:
            frames = self.data.event_frame_indices[name]
            if decay is None:
                shown = frames <= frame
            elif decay == 0:
                shown = frames == frame
            else:
                shown = (frames >= frame - decay) & (frames <= frame)
            points = np.where(shown[:, None], positions, np.nan)
            graphic.data = _xyz(points)
            colors = np.tile(color, (len(positions), 1))
            alphas = np.ones(len(positions))
            if decay:
                alphas = 1.0 - (frame - frames) / (decay + 1)
            colors[:, 3] = np.where(shown, alphas * self.data.opacity, 0.0)
            graphic.colors = colors


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

    Attributes
    ----------
    figure : fastplotlib.Figure
        The figure; add graphics to ``figure[0]`` or ``figure[0, 0]``.
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
        title_text: str | None = None,
        overlay_data: OverlayData | None = None,
        show_regions: bool | Sequence[str] = False,
        region_alpha: float = 0.3,
    ) -> None:
        self._fields = fields
        self.n_frames = len(fields)
        self.fps = float(fps)
        self._frame_times = np.asarray(frame_times, dtype=np.float64)
        self._labels = frame_labels
        low, high = _limits(fields, vmin, vmax)
        scale = ColorScale(cmap, low, high, "lightgray")
        self.figure, self._plot, scale_plot = new_figure(env, colorbar=True)
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
        draw_colorbar(scale_plot, scale, "")
        self._title = title_text or title(env, "")
        self._layers: list[Any] = []
        if overlay_data is not None:
            overlay_data = _skip_unsupported(overlay_data)
            positions = overlay_data.positions
            self._layers += [_Position(self._plot, item) for item in positions]
            self._layers += [
                _Bodyparts(self._plot, item) for item in overlay_data.bodypart_sets
            ]
            anchor = positions[0] if len(positions) == 1 else None
            center = env.bin_centers.mean(axis=0)[:2]
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
            self._slider.observe(
                lambda change: setattr(self, "frame", change["new"]), names="value"
            )
            return ipywidgets.VBox([output, ipywidgets.HBox([play, self._slider])])
        self.figure.renderer.add_event_handler(self._on_key, "key_down")
        self.figure.add_animations(self._advance)
        return output

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


def _limits(
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Color limits over every frame, ignoring missing values."""
    if vmin is not None and vmax is not None:
        return color_limits(np.array([vmin, vmax]), vmin, vmax)
    low, high = np.inf, -np.inf
    for field in fields:
        finite = np.asarray(field, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low, high = min(low, finite.min()), max(high, finite.max())
    values = np.array([low, high]) if np.isfinite(low) else np.array([np.nan])
    return color_limits(values, vmin, vmax)


def render_fastplotlib(
    env: Environment,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    *,
    frame_times: NDArray[np.float64],
    fps: float,
    overlay_data: OverlayData | None = None,
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    frame_labels: Sequence[str] | None = None,
    title: str | None = None,
    show_regions: bool | Sequence[str] = False,
    region_alpha: float = 0.3,
    dry_run: bool = False,
) -> FastplotlibAnimation | None:
    """Animate ``fields`` with fastplotlib; see :class:`FastplotlibAnimation`.

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
    overlay_data : OverlayData, optional
        Overlays aligned to the frames.
    cmap, vmin, vmax
        Colormap and its limits; limits default to the range over all frames.
    frame_labels : sequence of str, optional
        A label per frame; the frame times by default.
    title : str, optional
        Plot title; the environment's name and units by default.
    show_regions : bool or sequence of str, default=False
        Draw all regions, or the named ones.
    region_alpha : float, default=0.3
        Opacity of region areas.
    dry_run : bool, default=False
        Check that fastplotlib is installed and report the size of the
        animation, without drawing.

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
    return FastplotlibAnimation(
        env,
        fields,
        frame_times=frame_times,
        fps=fps,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        frame_labels=frame_labels,
        title_text=title,
        overlay_data=overlay_data,
        show_regions=show_regions,
        region_alpha=region_alpha,
    )
