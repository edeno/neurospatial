"""Views over a reusable animation, with recording-time controls."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.animation._backend_view import (
    CaptureSession as _CaptureSession,
)
from neurospatial.animation._backend_view import (
    _BackendView,
    _CallbackBackendView,
    _NativeBackendView,
)
from neurospatial.animation._resources import _close_keeping_error, _run_all

if TYPE_CHECKING:
    from neurospatial.animation.scene import Animation


class AnimationView:
    """A scene's viewer and its selected recording interval.

    Use :meth:`Animation.show` to create a view. ``native`` exposes the
    backend's viewer. Times are absolute recording times in seconds.
    Exports capture the current view, including native camera edits, layer
    styling and custom graphics. Pass ``render="scene"`` to export the scene's
    standard layout instead. Export restores the recording time and playback.
    Closing a view leaves its scene available for other views and exports.
    """

    def __init__(self) -> None:
        raise TypeError("Create a view with Animation.show().")

    @classmethod
    def _from_callbacks(
        cls,
        animation: Animation,
        native: Any,
        *,
        seek: Callable[[float], None],
        close: Callable[[], None],
        read_time: Callable[[], float] | None = None,
        capture_session: _CaptureSession | None = None,
    ) -> AnimationView:
        """A view whose viewer is controlled through plain callables."""
        return cls._from_backend(
            animation,
            _CallbackBackendView(
                native,
                initial_time=float(animation.frame_times[0]),
                seek=seek,
                close=close,
                read_time=read_time,
                capture_session=capture_session,
            ),
        )

    @classmethod
    def _from_backend(
        cls, animation: Animation, backend: _BackendView
    ) -> AnimationView:
        view = cls.__new__(cls)
        view._initialize(animation, backend)
        return view

    def _initialize(self, animation: Animation, backend: _BackendView) -> None:
        self.animation = animation
        self._backend = backend
        self.native = backend.native
        self._closed = False
        self._close_callbacks: list[Callable[[], None]] = []
        self._time = float(animation.frame_times[0])
        # None until select_time: then export covers the whole recording,
        # through its last time, rather than a half-open selection.
        self._selection: tuple[float, float] | None = None
        try:
            animation._register_view(self)
            backend._add_close_callback(self._native_closed)
        except BaseException as error:
            self._closed = True
            animation._unregister_view(self)
            _close_keeping_error(backend.close, error, what="View")
            raise

    @property
    def time(self) -> float:
        """The recording time currently shown by the native controls, in seconds."""
        if not self._closed:
            self._time = float(self._backend.time)
        return self._time

    @property
    def time_range(self) -> tuple[float, float]:
        """The half-open interval selected for export, in recording seconds.

        Before :meth:`select_time`, the whole recording, which export covers
        through its last time.
        """
        if self._selection is None:
            start, stop = self.animation.frame_times[[0, -1]]
            return float(start), float(stop)
        return self._selection

    def seek(self, time: float) -> None:
        """Show a recording time; times outside the scene raise."""
        self._require_open()
        time = float(time)
        start, stop = self.animation.frame_times[[0, -1]]
        if not np.isfinite(time) or not start <= time <= stop:
            raise ValueError(
                f"time must be within the recording [{start:g}, {stop:g}] seconds."
            )
        self._backend.seek(time)
        self._time = time

    def select_time(self, start: float, stop: float) -> AnimationView:
        """Select ``[start, stop)`` in recording seconds for export."""
        self._require_open()
        bounds = self.animation.frame_times[[0, -1]]
        if (
            not np.isfinite([start, stop]).all()
            or not bounds[0] <= start < stop <= bounds[1]
        ):
            raise ValueError(
                "Select a nonempty interval within the animation's recording "
                f"[{bounds[0]:g}, {bounds[1]:g}] seconds."
            )
        self._selection = (float(start), float(stop))
        return self

    def export(
        self,
        path: str | Path,
        *,
        render: Literal["view", "scene"] = "view",
        **kwargs: Any,
    ) -> Path:
        """Export the selected interval with the current view or scene layout.

        ``render="view"`` captures the visible scientific panels at their current
        size and preserves native camera, styling and custom graphics. Export
        pauses playback and restores the time/playback state even if encoding
        fails. ``render="scene"`` uses the standard scene renderer and accepts
        ``dpi``, ``figsize`` and ``n_workers``. Other options are those of
        :meth:`Animation.export`, for every container.
        """
        self._require_open()
        if render not in {"view", "scene"}:
            raise ValueError("render must be 'view' or 'scene'.")
        kwargs.setdefault("time_range", self._selection)
        if render == "scene" or self._backend.capture_session is None:
            return self.animation.export(path, **kwargs)

        from neurospatial.animation._export import prepare_export
        from neurospatial.animation._prepared import iter_observation_clocks

        if "dpi" in kwargs or "figsize" in kwargs:
            raise TypeError(
                "View capture uses the current viewer size. Resize the viewer "
                "first, or use render='scene' with dpi/figsize."
            )
        if kwargs.get("n_workers", 1) != 1:
            raise ValueError(
                "View capture reads each frame from this viewer, so it renders in "
                "this process. Pass n_workers=1, or render='scene' to render the "
                "standard scene layout in worker processes."
            )
        plan = prepare_export(
            path,
            self.animation.frame_times,
            title=self.animation.title,
            sample_times=iter_observation_clocks(self.animation),
            **kwargs,
        )
        with self._backend.capture_session() as capture:
            return plan.write(capture)

    def _add_close_callback(self, callback: Callable[[], None]) -> None:
        if self._closed:
            callback()
        else:
            self._close_callbacks.append(callback)

    def close(self) -> None:
        """Release this view's controls and rendering resources, once."""
        self._finish_close(close_backend=True)

    def _native_closed(self) -> None:
        """Finish a native close without closing the native window again."""
        self._finish_close(close_backend=False)

    def _finish_close(self, *, close_backend: bool) -> None:
        if self._closed:
            return
        self._closed = True
        callbacks, self._close_callbacks = self._close_callbacks, []
        closes = (self._backend.close, *callbacks) if close_backend else callbacks
        try:
            _run_all((self._read_final_time, *closes), "Animation view cleanup failed.")
        finally:
            self.animation._unregister_view(self)

    def _read_final_time(self) -> None:
        # A native window may already have disposed its time controls.
        with suppress(Exception):
            self._time = float(self._backend.time)

    def _require_open(self) -> None:
        self.animation._ensure_view_allowed()
        if self._closed:
            raise RuntimeError(
                "This view is closed. Call Animation.show() for a new view."
            )

    def __enter__(self) -> AnimationView:
        self._require_open()
        return self

    def __exit__(self, kind: Any, error: BaseException | None, traceback: Any) -> None:
        _close_keeping_error(self.close, error, what="View")

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.native)


def _png(frame: NDArray[np.uint8]) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def _notebook_view(animation: Animation, **kwargs: Any) -> AnimationView:
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError(
            "Notebook animation requires ipywidgets. Install it or choose "
            "backend='fastplotlib'."
        ) from exc

    # require_scene_view checked these options.
    dpi = kwargs.get("dpi", 100)
    figsize = kwargs.get("figsize")
    speed = float(kwargs.get("speed", 1.0))
    max_playback_fps = int(kwargs.get("max_playback_fps", 30))
    times = animation.frame_times
    start, stop = float(times[0]), float(times[-1])
    # Notebook ticks advance physical seconds even when source samples are irregular.
    interval_ms = max(1, round(1000 / max_playback_fps))
    step = speed * interval_ms / 1000
    n_ticks = int(np.ceil((stop - start) / step))
    image = widgets.Image(format="png")
    slider = widgets.FloatSlider(
        value=start,
        min=start,
        max=stop,
        step=step,
        description="Time (s)",
        readout_format=".3f",
        continuous_update=False,
        layout=widgets.Layout(width="95%"),
    )
    play = widgets.Play(min=0, max=n_ticks, interval=interval_ms)
    container = widgets.VBox([image, widgets.HBox([play, slider])])
    updating = False
    close_container = container.close
    capture_frames: list[NDArray[np.uint8]] | None = None

    def seek(time: float) -> None:
        nonlocal updating
        frame = animation.render_frame(time, dpi=dpi, figsize=figsize)
        image.value = _png(frame)
        if capture_frames is not None:
            capture_frames[:] = [frame]
        updating = True
        try:
            slider.value = time
            play.value = (
                n_ticks
                if time >= stop
                else int(np.clip(round((time - start) / step), 0, n_ticks))
            )
            if time >= stop:
                play.playing = False
        finally:
            updating = False

    def changed(change: dict[str, Any]) -> None:
        if not updating:
            view.seek(change["new"])

    def played(change: dict[str, Any]) -> None:
        if not updating:
            view.seek(min(stop, start + int(change["new"]) * step))

    def close() -> None:
        play.playing = False
        slider.unobserve(changed, names="value")
        play.unobserve(played, names="value")
        for widget in (image, slider, play, container.children[1]):
            widget.close()
        close_container()

    @contextmanager
    def capture_session() -> Iterator[Callable[[float], NDArray[np.uint8]]]:
        # Controls are outside the scientific image; the exported frames use
        # precisely the resolution and panels currently displayed.
        nonlocal capture_frames
        time, playing = view.time, play.playing
        play.playing = False
        previous_frames, capture_frames = capture_frames, []

        def capture(time: float) -> NDArray[np.uint8]:
            view.seek(time)
            # Hand off the owned RGB frame used by the lossless display PNG;
            # retain no extra image between captures or outside this session.
            assert capture_frames is not None
            return capture_frames.pop()

        try:
            yield capture
        finally:
            capture_frames = previous_frames
            if view._closed or animation._closed or animation._closing:
                play.playing = False
            else:
                try:
                    view.seek(time)
                finally:
                    play.playing = playing

    view = AnimationView._from_callbacks(
        animation,
        container,
        seek=seek,
        close=close,
        read_time=lambda: float(slider.value),
        capture_session=capture_session,
    )
    container.close = view.close
    slider.observe(changed, names="value")
    play.observe(played, names="value")
    try:
        view.seek(start)
    except BaseException as error:
        _close_keeping_error(view.close, error, what="View")
        raise
    return view


def show_animation(
    animation: Animation, *, backend: str = "widget", **kwargs: Any
) -> AnimationView:
    """Create a view after checking that all requested data can be drawn."""
    from neurospatial.animation._capabilities import require_scene_view

    animation._ensure_view_allowed()
    require_scene_view(animation, backend, kwargs)
    if backend == "widget":
        return _notebook_view(animation, **kwargs)
    if backend == "napari":
        from neurospatial.animation.backends._napari_scene import NapariScene

        handle = NapariScene(animation, **kwargs)
        return AnimationView._from_backend(
            animation, _NativeBackendView(handle.viewer, handle)
        )
    from neurospatial.animation.backends.fastplotlib_backend import (
        FastplotlibAnimation,
    )

    native = FastplotlibAnimation(animation, **kwargs)
    try:
        native.show()
    except BaseException as error:
        _close_keeping_error(native.close, error, what="View")
        raise
    return AnimationView._from_backend(animation, _NativeBackendView(native, native))
