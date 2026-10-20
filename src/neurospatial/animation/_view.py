"""Views over a reusable animation, with recording-time controls."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.animation.scene import Animation, AnimationField


class AnimationView:
    """A scene's viewer and its selected recording interval.

    Use :meth:`Animation.show` to create a view. ``native`` exposes the
    backend's viewer. Times are absolute recording times in seconds.
    Exports retain the scene's fields, overlays and scales; native camera
    edits and graphics added directly to ``native`` are not exported.
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
    ) -> None:
        self.animation = animation
        self.native = native
        self._seek_native = seek
        self._close_native = close
        self._read_native_time = read_time
        self._closed = False
        self._time = float(animation.frame_times[0])
        self._selection = (
            float(animation.frame_times[0]),
            float(animation.frame_times[-1]),
        )

    @property
    def time(self) -> float:
        """The recording time currently shown by the native controls, in seconds."""
        if not self._closed and self._read_native_time is not None:
            self._time = float(self._read_native_time())
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
        self._seek_native(time)
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

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        """Export the selected interval with the scene's panels and scales."""
        self._require_open()
        kwargs.setdefault(
            "time_range",
            self._selection if self._selection[0] < self._selection[1] else None,
        )
        return self.animation.export(path, **kwargs)

    def close(self) -> None:
        """Release this view's controls and rendering resources, once."""
        if not self._closed:
            self._time = self.time
            self._closed = True
            self._close_native()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "This view is closed. Call Animation.show() for a new view."
            )

    def __enter__(self) -> AnimationView:
        self._require_open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.native)


class _FieldFrames:
    """Lazy sampling of one field on a viewer's shared clock."""

    def __init__(self, field: AnimationField, times: NDArray[np.float64]) -> None:
        self.field, self.times = field, times
        self.shape = (len(times), len(np.asarray(field.values[0])))
        self.ndim = 2

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

    view = AnimationView(
        animation,
        container,
        seek=seek,
        close=close,
        read_time=lambda: float(slider.value),
    )
    container.close = view.close
    slider.observe(changed, names="value")
    play.observe(played, names="value")
    try:
        view.seek(start)
    except Exception:
        view.close()
        raise
    return view


def show_animation(
    animation: Animation, *, backend: str = "widget", **kwargs: Any
) -> AnimationView:
    """Create a view after checking that all requested data can be drawn."""
    if backend == "widget":
        return _notebook_view(animation, **kwargs)
    if backend == "napari":
        raise NotImplementedError(
            "The napari scene adapter cannot yet preserve independent source clocks "
            "and recording-time playback. Use backend='widget' or 'fastplotlib' "
            "for Animation.show(); frame-based napari animation remains available "
            "through Environment.animate_fields()."
        )
    if backend != "fastplotlib":
        raise ValueError("Use backend='widget' or 'fastplotlib'.")
    if kwargs.get("dry_run", False):
        raise ValueError("Animation.show() creates a view; dry_run is not supported.")

    from neurospatial.animation._capabilities import require_overlays

    data = animation.overlay_data(own_samples=backend == "fastplotlib")
    require_overlays(backend, data)
    # The scene owns readers it opened; a viewer only borrows them.
    data = replace(
        data, videos=[replace(video, _owns_reader=False) for video in data.videos]
    )
    fields = list(animation.fields.values())
    first = fields[0]
    scales = {(str(field.cmap), *field.color_limits()) for field in fields}
    if len(scales) > 1:
        raise ValueError(
            f"The {backend} scene adapter uses one shared color scale. "
            "Give the fields identical cmap/vmin/vmax, or use backend='widget' "
            "to retain independent field scales."
        )
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
        _field_at_time=lambda time: list(animation.sample_fields(time).values()),
        **options,
    )
    if native is None:
        raise ValueError("Animation.show() creates a view; dry_run is not supported.")
    return AnimationView(
        animation,
        native,
        seek=native.seek,
        close=native.close,
        read_time=lambda: float(animation.frame_times[0])
        + float(native.widget.indices["time"]),
    )
