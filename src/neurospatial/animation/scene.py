"""A physical-clock animation scene shared by inspection and export."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from neurospatial._observation import (
    _validated_signal_times,
    resolve_sample_support,
    validate_break_before,
    validate_max_gap,
)


def _readonly(value: Any, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


class AnimationField:
    """One spatial field stream on its own clock, in seconds.

    ``values`` is a two-dimensional array or a lazy, indexable sequence of
    spatial rows. The source is retained without copying or materializing it.
    Samples are held until the next sample only across supported intervals;
    recording gaps, declared breaks and times outside the stream show NaN.
    Partial NaNs in a spatial row preserve missing spatial observations.

    Parameters
    ----------
    values : array or indexable sequence, shape (n_samples, n_bins)
        Spatial values. Caller-owned sources are never closed by the scene.
    times : array-like, shape (n_samples,)
        Finite, strictly increasing timestamps in seconds.
    cmap : str, default="viridis"
        Matplotlib colormap.
    vmin, vmax : float or None
        Fixed display limits; omitted limits use a bounded sample of rows.
    max_gap : float or None, default=0.5
        Largest supported interval in seconds; None disables this gate.
    break_before : array-like of bool or None
        True at a sample starts a new recording segment.
    label : str or None
        Colorbar label.
    """

    def __init__(
        self,
        values: Any,
        times: Any,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        max_gap: float | None = 0.5,
        break_before: Any = None,
        label: str | None = None,
    ) -> None:
        self.times = _readonly(_validated_signal_times(times, call="AnimationField"))
        if not hasattr(values, "__getitem__") or not hasattr(values, "__len__"):
            raise TypeError("values must be an indexable sequence of spatial rows.")
        if len(values) != len(self.times):
            raise ValueError("values and times must have the same number of samples.")
        shape = getattr(values, "shape", None)
        if shape is not None:
            if len(shape) != 2 or shape[1] == 0:
                raise ValueError("values must have shape (n_samples, n_bins).")
            self.n_bins = int(shape[1])
        else:
            first = np.asarray(values[0])
            if first.ndim != 1 or first.size == 0:
                raise ValueError(
                    "Each values row must be a nonempty one-dimensional spatial field."
                )
            self.n_bins = first.size
        for name, limit in (("vmin", vmin), ("vmax", vmax)):
            if limit is not None and not np.isfinite(limit):
                raise ValueError(f"{name} must be finite or None.")
        if vmin is not None and vmax is not None and vmin > vmax:
            raise ValueError("vmin must be less than or equal to vmax.")
        self.values = values
        self.cmap, self.vmin, self.vmax, self.label = cmap, vmin, vmax, label
        self.max_gap = validate_max_gap(max_gap)
        flags = validate_break_before(
            break_before, len(self.times), call="AnimationField"
        )
        self.break_before = None if flags is None else _readonly(flags)
        support = resolve_sample_support(
            self.times,
            np.ones(len(self.times), dtype=bool),
            max_gap=max_gap,
            epochs=None,
            break_before=self.break_before,
            result="the animation shows exact samples only",
        )
        self.sample_valid = support.sample_valid
        self.interval_valid = support.interval_mask
        self._limits: tuple[float, float] | None = None
        self._peak: float | None = None

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index: Any) -> Any:
        """Read spatial rows from the original source."""
        if isinstance(index, slice):
            return self.values[index]
        row = np.asarray(self.values[index], dtype=float)
        if row.shape != (self.n_bins,):
            raise ValueError(
                f"Field row {index} has shape {row.shape}; expected ({self.n_bins},)."
            )
        return row

    def sample(self, time: float) -> np.ndarray:
        """Return the previous sample within support, or an all-NaN row."""
        time = float(time)
        if not np.isfinite(time):
            raise ValueError("time must be finite seconds.")
        index = int(np.searchsorted(self.times, time, side="right")) - 1
        if index >= 0 and (
            self.times[index] == time
            or (index < len(self.times) - 1 and self.interval_valid[index])
        ):
            return cast("np.ndarray", self[index])
        return np.full(self.n_bins, np.nan)

    def color_limits(self) -> tuple[float, float]:
        """Return stable display limits, reading at most 100 source rows."""
        if self._limits is None and self.vmin is not None and self.vmax is not None:
            self._limits = (self.vmin, self.vmax)
        if self._limits is None:
            from neurospatial.animation.rendering import animation_colormap_range

            rows = [
                self[int(i)]
                for i in np.unique(
                    np.linspace(0, len(self) - 1, min(len(self), 100), dtype=int)
                )
            ]
            self._limits = animation_colormap_range(rows, self.vmin, self.vmax)
            from neurospatial.animation.rendering import frame_extremes, typical_peak

            self._peak = typical_peak(frame_extremes(rows)[1])
        return self._limits

    def _typical_peak(self) -> float:
        if self._peak is None:
            from neurospatial.animation.rendering import frame_extremes, typical_peak

            rows = [
                self[int(i)]
                for i in np.unique(
                    np.linspace(0, len(self) - 1, min(len(self), 100), dtype=int)
                )
            ]
            self._peak = typical_peak(frame_extremes(rows)[1])
        return self._peak if self._peak > 0 else 1.0


class Animation:
    """Named spatial fields and overlays synchronized in physical time.

    Plain arrays require ``frame_times``. A mapping may contain
    :class:`AnimationField` streams with independent clocks. When no output
    clock is supplied, their sorted union becomes ``frame_times``. Metadata
    is copied; large spatial sources stay lazy and caller-owned.
    """

    def __init__(
        self,
        env: Any,
        fields: Any,
        *,
        frame_times: Any = None,
        overlays: Any = None,
        title: str | None = None,
        window_seconds: float = 2.0,
        show_regions: bool | list[str] = False,
        region_alpha: float = 0.3,
        scale_bar: Any = None,
        track: Any = None,
    ) -> None:
        self._closed = False
        if track is not None:
            from neurospatial.animation.backends.fastplotlib_backend import (
                _checked_track,
            )

            track = _checked_track(env, track)
        if env.n_dims not in (1, 2):
            raise ValueError("Animation supports one- or two-dimensional environments.")
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive finite seconds.")
        if not np.isfinite(region_alpha) or not 0 <= region_alpha <= 1:
            raise ValueError("region_alpha must be between zero and one.")
        clock = (
            None
            if frame_times is None
            else _readonly(_validated_signal_times(frame_times, call="Animation"))
        )
        streams = dict(fields) if isinstance(fields, Mapping) else {"field": fields}
        if not streams or any(
            not isinstance(name, str) or not name for name in streams
        ):
            raise ValueError(
                "fields must contain at least one stream with a nonempty string name."
            )
        for name, field in streams.items():
            if not isinstance(field, AnimationField):
                if clock is None:
                    raise ValueError(
                        "Plain field arrays require frame_times in seconds; use AnimationField for independent clocks."
                    )
                field = AnimationField(field, clock)
                streams[name] = field
            if field.n_bins != env.n_bins:
                raise ValueError(
                    f"Field {name!r} has {field.n_bins} bins; the environment has {env.n_bins}."
                )
        self.env = env
        self.fields = MappingProxyType(streams)
        self.frame_times = (
            clock
            if clock is not None
            else _readonly(
                np.unique(np.concatenate([field.times for field in streams.values()]))
            )
        )
        self.title, self.window_seconds = title, float(window_seconds)
        self.show_regions = copy.deepcopy(show_regions)
        self.region_alpha, self.scale_bar, self.track = (
            region_alpha,
            copy.deepcopy(scale_bar),
            track,
        )
        self.overlays = tuple(
            self._snapshot_overlay(overlay)
            for overlay in (() if overlays is None else overlays)
        )
        self._renderer: Any = None
        # Validate overlays once on the scene clock. Keep converted video readers
        # for repeated frame inspection rather than reopening the source.
        self._overlay_data = self.overlay_data(own_samples=True)

    def _snapshot_overlay(self, overlay: Any) -> Any:
        result = copy.copy(overlay)
        for name in (
            "times",
            "break_before",
            "position_times",
            "unit_ids",
            "order",
            "_order",
            "_intervals",
        ):
            value = getattr(result, name, None)
            if value is not None:
                setattr(result, name, _readonly(value))
        for name in ("colors", "markers", "skeleton", "calibration", "epochs"):
            if hasattr(result, name):
                setattr(result, name, copy.deepcopy(getattr(result, name)))
        for name in ("data", "event_positions"):
            if isinstance(getattr(result, name, None), dict):
                setattr(result, name, dict(getattr(result, name)))
        if isinstance(getattr(result, "event_times", None), dict):
            result.event_times = {
                name: _readonly(times) for name, times in result.event_times.items()
            }
        if hasattr(result, "_rows"):
            rows, ids = result._rows
            result._rows = (list(rows), _readonly(ids))
        from neurospatial.animation.overlays import (
            BodypartOverlay,
            HeadDirectionOverlay,
            ObjectVectorOverlay,
            PositionOverlay,
            VideoOverlay,
        )

        if (
            isinstance(
                result,
                (
                    PositionOverlay,
                    BodypartOverlay,
                    HeadDirectionOverlay,
                    ObjectVectorOverlay,
                    VideoOverlay,
                ),
            )
            and result.times is None
        ):
            result.times = self.frame_times
        return result

    def overlay_data(
        self, frame_times: Any = None, *, own_samples: bool = False
    ) -> Any:
        """Convert overlays on a requested physical clock."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        self._ensure_open()
        if frame_times is None and own_samples and hasattr(self, "_overlay_data"):
            return self._overlay_data
        clock = (
            self.frame_times
            if frame_times is None
            else np.asarray(frame_times, dtype=float)
        )
        return _convert_overlays_to_data(
            list(self.overlays),
            clock,
            len(clock),
            self.env,
            self.show_regions,
            own_samples=own_samples,
        )

    @classmethod
    def from_decoding(cls, result: Any, **kwargs: Any) -> Animation:
        """Build a probability animation from a decoding result's own clock."""
        if getattr(result, "times", None) is None:
            raise ValueError(
                "Decoding result has no times. Decode with times to preserve its physical clock."
            )
        field_options = {
            name: kwargs.pop(name)
            for name in ("cmap", "vmin", "vmax", "label", "max_gap", "break_before")
            if name in kwargs
        }
        field_options.setdefault("label", "Probability")
        field_options.setdefault("max_gap", getattr(result, "max_gap", 0.5))
        field_options.setdefault("break_before", getattr(result, "break_before", None))
        field = AnimationField(result.posterior, result.times, **field_options)
        return cls(result.env, {"posterior": field}, **kwargs)

    def sample_fields(self, time: float) -> dict[str, np.ndarray]:
        """Sample every named stream at the same physical time."""
        self._ensure_open()
        return {name: field.sample(time) for name, field in self.fields.items()}

    def render_frame(
        self, time: float, *, dpi: int = 100, figsize: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Render a complete scene as an RGB array, without a display."""
        from neurospatial.animation._scene_rendering import SceneRenderer

        self._ensure_open()
        settings = (dpi, figsize)
        if self._renderer is None or self._renderer.settings != settings:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = SceneRenderer(self, dpi=dpi, figsize=figsize)
        return cast("np.ndarray", self._renderer.render(float(time)))

    def export(
        self,
        path: str | Path,
        *,
        time_range: Any = None,
        fps: float = 30,
        speed: float = 1,
        overwrite: bool = False,
        dpi: int = 100,
        figsize: tuple[float, float] | None = None,
        codec: str | None = None,
    ) -> Path:
        """Write MP4 or GIF at uniform physical-clock intervals.

        Adjacent output frames advance by ``speed / fps`` seconds. A selected
        half-open range ``[start, stop)`` changes only the output clock; scales and overlay clocks remain
        those of the complete scene. Returns the destination ``Path``.
        """
        from neurospatial.animation._scene_rendering import export_scene

        self._ensure_open()
        return export_scene(
            self,
            Path(path),
            time_range=time_range,
            fps=fps,
            speed=speed,
            overwrite=overwrite,
            dpi=dpi,
            figsize=figsize,
            codec=codec,
        )

    def show(self, *, backend: str = "widget", **kwargs: Any) -> Any:
        """Open an interactive view of this scene."""
        from neurospatial.animation._view import show_animation

        self._ensure_open()
        return show_animation(self, backend=backend, **kwargs)

    def inspect_events(self, events: Any, **kwargs: Any) -> Any:
        """Inspect event-centered windows on this scene's physical clock."""
        from neurospatial.animation.events import EventInspector

        self._ensure_open()
        return EventInspector(self, events, **kwargs)

    def close(self) -> None:
        """Close figures and video readers owned by the scene."""
        if self._closed:
            return
        self._closed = True
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        for video in self._overlay_data.videos:
            close = getattr(video.reader, "close", None)
            if close is not None and getattr(video, "_owns_reader", False):
                close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Animation is closed. Create a new scene to inspect or export it."
            )

    def __enter__(self) -> Animation:
        self._ensure_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
