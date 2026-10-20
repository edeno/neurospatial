"""Views over a reusable animation, with recording-time controls."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial.animation._resources import _add_cleanup_note

if TYPE_CHECKING:
    from neurospatial.animation.scene import Animation, AnimationField

from neurospatial.animation._backend_view import (
    CaptureSession as _CaptureSession,
)
from neurospatial.animation._backend_view import (
    _BackendView,
    _CallbackBackendView,
    _close_after_error,
    _NativeBackendView,
)


class AnimationView:
    """A scene's viewer and its selected recording interval.

    Use :meth:`Animation.show` to create a view. ``native`` exposes the
    backend's viewer. Times are absolute recording times in seconds.
    Exports capture the current view, including native camera edits, layer
    styling and custom graphics. Pass ``render="scene"`` to export the scene's
    standard layout instead. Export restores the recording time and playback.
    Closing a view leaves its scene available for other views and exports.
    """

    def __init__(
        self,
        animation: Animation,
        native: Any,
        *,
        seek: Callable[[float], None],
        close: Callable[[], None],
        read_time: Callable[[], float] | None = None,
        capture_session: _CaptureSession | None = None,
    ) -> None:
        self._initialize(
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
        self._selection = (
            float(animation.frame_times[0]),
            float(animation.frame_times[-1]),
        )
        try:
            animation._register_view(self)
        except BaseException as error:
            self._closed = True
            _close_after_error(backend.close, error)
            raise

    @property
    def time(self) -> float:
        """The recording time currently shown by the native controls, in seconds."""
        if not self._closed:
            self._time = float(self._backend.time)
        return self._time

    @property
    def time_range(self) -> tuple[float, float]:
        """The half-open recording interval selected for export."""
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
        ``dpi`` and ``figsize``.
        """
        self._require_open()
        if render not in {"view", "scene"}:
            raise ValueError("render must be 'view' or 'scene'.")
        kwargs.setdefault(
            "time_range",
            self._selection if self._selection[0] < self._selection[1] else None,
        )
        if render == "scene" or self._backend.capture_session is None:
            return self.animation.export(path, **kwargs)

        from neurospatial.animation._export import prepare_export

        if "dpi" in kwargs or "figsize" in kwargs:
            raise TypeError(
                "View capture uses the current viewer size. Resize the viewer "
                "first, or use render='scene' with dpi/figsize."
            )
        plan = prepare_export(
            path,
            self.animation.frame_times,
            sample_times=(field.times for field in self.animation.fields.values()),
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
        if self._closed:
            return
        from neurospatial.animation._resources import _cleanup_error

        failures: list[BaseException] = []
        try:
            # A native window may already have disposed its time controls.
            with suppress(Exception):
                self._time = float(self._backend.time)
        except BaseException as error:
            failures.append(error)
        self._closed = True
        callbacks, self._close_callbacks = self._close_callbacks, []
        try:
            for close in (self._backend.close, *callbacks):
                try:
                    close()
                except BaseException as error:
                    failures.append(error)
        finally:
            self.animation._unregister_view(self)
        if failures:
            raise _cleanup_error("Animation view cleanup failed.", failures)

    def _require_open(self) -> None:
        self.animation._ensure_view_allowed()
        if self._closed:
            raise RuntimeError(
                "This view is closed. Call Animation.show() for a new view."
            )

    def __enter__(self) -> AnimationView:
        self._require_open()
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if len(args) > 1 and isinstance(args[1], BaseException):
                _add_cleanup_note(args[1], f"View cleanup also failed: {cleanup!r}")
            else:
                raise

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.native)


class _FieldFrames:
    """Lazy sampling of one field on a viewer's shared clock."""

    def __init__(self, field: AnimationField, times: NDArray[np.float64]) -> None:
        self.field, self.times = field, times
        self.shape = (len(times), field.n_bins)
        self.ndim = 2
        self.dtype = np.dtype(np.float64)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, key: Any) -> NDArray[np.float64]:
        times = self.times[key]
        if np.ndim(times) == 0:
            return self.field.sample(float(times))
        if not len(times):
            return np.empty((0, self.shape[1]))
        return np.stack([self.field.sample(float(time)) for time in times])


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

    dpi = kwargs.pop("dpi", 100)
    figsize = kwargs.pop("figsize", None)
    try:
        speed = float(kwargs.pop("speed", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "speed must be a positive finite playback multiplier."
        ) from exc
    max_playback_fps = kwargs.pop("max_playback_fps", 30)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be a positive finite playback multiplier.")
    if (
        isinstance(max_playback_fps, bool)
        or not isinstance(max_playback_fps, (int, np.integer))
        or max_playback_fps <= 0
    ):
        raise ValueError("max_playback_fps must be a positive integer.")
    if kwargs:
        raise TypeError(f"Unknown notebook view options: {', '.join(sorted(kwargs))}.")
    times = animation.frame_times
    start, stop = float(times[0]), float(times[-1])
    # Notebook ticks advance physical seconds even when source samples are irregular.
    interval_ms = max(1, round(1000 / max_playback_fps))
    step = float(speed) * interval_ms / 1000
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

    def seek(time: float) -> None:
        nonlocal updating
        image.value = _png(animation.render_frame(time, dpi=dpi, figsize=figsize))
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
        from PIL import Image

        time, playing = view.time, play.playing
        play.playing = False

        def capture(time: float) -> NDArray[np.uint8]:
            view.seek(time)
            frame: NDArray[np.uint8] = np.asarray(
                Image.open(io.BytesIO(image.value)).convert("RGB"), dtype=np.uint8
            ).copy()
            return frame

        try:
            yield capture
        finally:
            if view._closed or animation._closed or animation._closing:
                play.playing = False
            else:
                try:
                    view.seek(time)
                finally:
                    play.playing = playing

    view = AnimationView._from_backend(
        animation,
        _CallbackBackendView(
            container,
            initial_time=start,
            seek=seek,
            close=close,
            read_time=lambda: float(slider.value),
            capture_session=capture_session,
        ),
    )
    container.close = view.close
    slider.observe(changed, names="value")
    play.observe(played, names="value")
    try:
        view.seek(start)
    except BaseException as error:
        _close_after_error(view.close, error)
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
        from neurospatial.animation.backends._napari_scene import build_napari_scene

        handle = build_napari_scene(animation, **kwargs)
        return AnimationView._from_backend(
            animation, _NativeBackendView(handle.viewer, handle)
        )
    data = animation.overlay_data(own_samples=True)
    fields = list(animation.fields.values())
    first = fields[0]
    sources = {
        name: _FieldFrames(field, animation.frame_times)
        for name, field in animation.fields.items()
    }
    options = {
        "cmap": first.cmap,
        "vmin": first.color_limits()[0],
        "vmax": first.color_limits()[1],
        "title": animation.title,
        "show_regions": animation.show_regions,
        "region_alpha": animation.region_alpha,
        **kwargs,
    }
    from neurospatial.animation.backends.fastplotlib_backend import render_fastplotlib

    options.setdefault("window_seconds", animation.window_seconds)
    options.setdefault("track", animation.track)
    options.setdefault("colorbar_label", first.label or "")
    native = render_fastplotlib(
        animation.env,
        sources,
        frame_times=animation.frame_times,
        overlay_data=data,
        _field_streams=tuple(animation.fields.values()),
        _field_at_time=lambda time: list(animation.sample_fields(time).values()),
        _field_styles=[
            {
                "cmap": kwargs.get("cmap", field.cmap),
                "vmin": kwargs.get("vmin", field.color_limits()[0]),
                "vmax": kwargs.get("vmax", field.color_limits()[1]),
                "label": kwargs.get("colorbar_label", field.label or name),
                "opaque_from": field._typical_peak(),
            }
            for name, field in animation.fields.items()
        ],
        **options,
    )
    if native is None:
        raise ValueError("Animation.show() creates a view; dry_run is not supported.")
    try:
        native.show()
    except BaseException as error:
        _close_after_error(native.close, error)
        raise
    return AnimationView._from_backend(animation, _NativeBackendView(native, native))
