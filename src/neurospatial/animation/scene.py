"""A physical-clock animation scene shared by inspection and export."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from neurospatial._intervals import as_intervals, intersect_intervals, intervals_contain
from neurospatial._observation import (
    _validated_signal_times,
    resolve_sample_support,
    validate_break_before,
    validate_max_gap,
)


def _readonly(value: Any, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.dtype.hasobject:
        result.flags.writeable = False
        return result
    # Immutable backing prevents callers from re-enabling writes on metadata.
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


@dataclass(frozen=True, init=False, eq=False)
class AnimationField:
    """One spatial field stream on its own clock, in seconds.

    ``values`` is a two-dimensional array or a lazy, indexable sequence of
    spatial rows. The source is retained without copying or materializing it.
    Samples are held until the next sample only across supported intervals;
    recording gaps, declared breaks and times outside the stream show NaN.
    A held interval must lie wholly inside one epoch; a pair of samples on
    opposite sides of an epoch boundary is shown only at its supported exact
    sample times. Partial NaNs preserve missing spatial observations.

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
    epochs : interval-like or None
        Half-open ``[start, stop)`` recording windows in seconds. Exact samples
        at an excluded stop or outside the windows show NaN. Disabling
        ``max_gap`` does not disable these known recording boundaries.
    label : str or None
        Colorbar label.
    """

    values: Any
    times: np.ndarray
    n_bins: int = dataclass_field(init=False)
    cmap: str
    vmin: float | None
    vmax: float | None
    label: str | None
    max_gap: float | None
    break_before: np.ndarray | None
    epochs: np.ndarray | None
    sample_valid: np.ndarray = dataclass_field(init=False)
    interval_valid: np.ndarray = dataclass_field(init=False)
    _limits: tuple[float, float] | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )
    _peak: float | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )

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
        epochs: Any = None,
        label: str | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "times",
            _readonly(_validated_signal_times(times, call="AnimationField")),
        )
        if not hasattr(values, "__getitem__") or not hasattr(values, "__len__"):
            raise TypeError("values must be an indexable sequence of spatial rows.")
        if len(values) != len(self.times):
            raise ValueError("values and times must have the same number of samples.")
        shape = getattr(values, "shape", None)
        if shape is not None:
            if len(shape) != 2 or shape[1] == 0:
                raise ValueError("values must have shape (n_samples, n_bins).")
            object.__setattr__(self, "n_bins", int(shape[1]))
        else:
            first = np.asarray(values[0])
            if first.ndim != 1 or first.size == 0:
                raise ValueError(
                    "Each values row must be a nonempty one-dimensional spatial field."
                )
            object.__setattr__(self, "n_bins", first.size)
        for name, limit in (("vmin", vmin), ("vmax", vmax)):
            if limit is not None and not np.isfinite(limit):
                raise ValueError(f"{name} must be finite or None.")
        if vmin is not None and vmax is not None and vmin > vmax:
            raise ValueError("vmin must be less than or equal to vmax.")
        object.__setattr__(self, "values", values)
        for name, value in (
            ("cmap", cmap),
            ("vmin", vmin),
            ("vmax", vmax),
            ("label", label),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "max_gap", validate_max_gap(max_gap))
        flags = validate_break_before(
            break_before, len(self.times), call="AnimationField"
        )
        object.__setattr__(
            self, "break_before", None if flags is None else _readonly(flags)
        )
        windows = as_intervals(epochs, name="epochs")
        object.__setattr__(
            self, "epochs", None if windows is None else _readonly(windows)
        )
        sample_valid = (
            np.ones(len(self.times), dtype=bool)
            if self.epochs is None
            else intervals_contain(
                self.epochs, self.times, np.nextafter(self.times, np.inf)
            )
        )
        support = resolve_sample_support(
            self.times,
            sample_valid,
            max_gap=max_gap,
            epochs=self.epochs,
            break_before=self.break_before,
            result="the animation shows exact samples only",
        )
        object.__setattr__(self, "sample_valid", _readonly(support.sample_valid))
        # A held [start, stop) row needs no observation at its excluded stop.
        # Keep canonical interval gates separate from exact-sample membership.
        object.__setattr__(self, "interval_valid", _readonly(support.interval_gates))
        object.__setattr__(self, "_limits", None)
        object.__setattr__(self, "_peak", None)

    def __copy__(self) -> AnimationField:
        result = object.__new__(type(self))
        result.__setstate__(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict[int, Any]) -> AnimationField:
        result = self.__copy__()
        memo[id(self)] = result
        return result

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            if (
                name
                in {"times", "break_before", "epochs", "sample_valid", "interval_valid"}
                and value is not None
            ):
                value = _readonly(value)
            object.__setattr__(self, name, value)

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
        if (
            index >= 0
            and self.sample_valid[index]
            and (
                self.times[index] == time
                or (index < len(self.times) - 1 and self.interval_valid[index])
            )
        ):
            return cast("np.ndarray", self[index])
        return np.full(self.n_bins, np.nan)

    def color_limits(self) -> tuple[float, float]:
        """Return stable display limits, reading at most 100 source rows."""
        if self._limits is None and self.vmin is not None and self.vmax is not None:
            object.__setattr__(self, "_limits", (self.vmin, self.vmax))
        if self._limits is None:
            from neurospatial.animation._field_stats import animation_colormap_range

            rows = self._scale_rows()
            object.__setattr__(
                self, "_limits", animation_colormap_range(rows, self.vmin, self.vmax)
            )
            from neurospatial.animation._field_stats import frame_extremes, typical_peak

            object.__setattr__(self, "_peak", typical_peak(frame_extremes(rows)[1]))
        assert self._limits is not None
        return self._limits

    def _scale_rows(self) -> list[np.ndarray]:
        indices = np.flatnonzero(self.sample_valid)
        if len(indices) > 100:
            indices = indices[np.linspace(0, len(indices) - 1, 100, dtype=int)]
        return [self[int(index)] for index in indices]

    def _typical_peak(self) -> float:
        if self._peak is None:
            from neurospatial.animation._field_stats import frame_extremes, typical_peak

            rows = self._scale_rows()
            object.__setattr__(self, "_peak", typical_peak(frame_extremes(rows)[1]))
        assert self._peak is not None
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
            from neurospatial.animation._tracks import (
                _checked_track,
            )

            track = _checked_track(env, track)
        if env.n_dims not in (1, 2, 3):
            raise ValueError(
                "Animation supports one-, two- or three-dimensional environments."
            )
        if env.n_dims == 1 and show_regions:
            from neurospatial.animation._region_names import shown_region_names

            if shown_region_names(env, show_regions):
                raise ValueError(
                    "Animation cannot draw regions in a 1D environment. "
                    "Use a 2D environment for region display, or omit show_regions."
                )
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
            streams[name] = copy.copy(field)
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
        from neurospatial.animation.overlays import _convert_overlays_to_data

        self._overlay_data = _convert_overlays_to_data(
            list(self.overlays),
            self.frame_times,
            len(self.frame_times),
            self.env,
            self.show_regions,
            own_samples=True,
        )
        if env.n_dims == 3:
            if any(item.data.ndim == 1 for item in self._overlay_data.head_directions):
                self.close()
                raise ValueError(
                    "3D head directions require (n_samples, 3) unit vectors; "
                    "angles alone do not specify a direction in 3D."
                )
            from neurospatial.animation._region_names import shown_region_names

            planar_regions = any(
                env.regions[name].kind != "point"
                or np.asarray(env.regions[name].data).shape != (3,)
                for name in (
                    shown_region_names(env, show_regions) if show_regions else []
                )
            )
            if planar_regions or scale_bar:
                self.close()
                raise ValueError(
                    "3D Animation cannot place planar regions or a 2D scale bar "
                    "unambiguously. Use 3D point regions and omit scale_bar; the 3D axes "
                    "show physical coordinates and units."
                )

    def _snapshot_overlay(self, overlay: Any) -> Any:
        from neurospatial.animation.overlays import (
            BodypartOverlay,
            EpochOverlay,
            EventOverlay,
            HeadDirectionOverlay,
            ObjectVectorOverlay,
            PositionOverlay,
            SpikeRasterOverlay,
            TimeSeriesOverlay,
            VideoOverlay,
        )

        if type(overlay) not in {
            PositionOverlay,
            BodypartOverlay,
            HeadDirectionOverlay,
            VideoOverlay,
            ObjectVectorOverlay,
            EventOverlay,
            SpikeRasterOverlay,
            EpochOverlay,
            TimeSeriesOverlay,
        }:
            # An adapter may be frozen, expose read-only properties, or own
            # native resources that cannot be copied. Only its converted data
            # is snapshotted; renderers never consult the adapter afterward.
            return overlay
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
                if name == "unit_ids":
                    from neurospatial._results import as_label_array

                    value = as_label_array(value)
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
        from neurospatial.animation.overlays import _realign_prepared_overlays

        self._ensure_open()
        clock = (
            self.frame_times
            if frame_times is None
            else _readonly(
                _validated_signal_times(frame_times, call="Animation.overlay_data")
            )
        )
        if clock[0] < self.frame_times[0] or clock[-1] > self.frame_times[-1]:
            raise ValueError(
                "overlay_data frame_times must stay within the scene recording "
                f"[{self.frame_times[0]:g}, {self.frame_times[-1]:g}] seconds. "
                "Create a scene with a wider frame_times range to include those times."
            )
        return _realign_prepared_overlays(
            self._overlay_data,
            clock,
            own_samples=own_samples,
        )

    @classmethod
    def from_decoding(cls, result: Any, **kwargs: Any) -> Animation:
        """Build a probability animation on its decoding clock and recording support.

        Known ``result.spike_window`` always intersects ``epochs`` supplied
        here. Neither wider epochs nor ``max_gap=None`` can fill known
        recording gaps. Held intervals must fit wholly inside one resulting
        half-open window; supported isolated samples remain visible exactly.
        """
        if getattr(result, "times", None) is None:
            raise ValueError(
                "Decoding result has no times. Decode with times to preserve its physical clock."
            )
        field_options = {
            name: kwargs.pop(name)
            for name in (
                "cmap",
                "vmin",
                "vmax",
                "label",
                "max_gap",
                "break_before",
                "epochs",
            )
            if name in kwargs
        }
        known = as_intervals(getattr(result, "spike_window", None), name="spike_window")
        selected = as_intervals(field_options.get("epochs"), name="epochs")
        if known is not None:
            selected = (
                known if selected is None else intersect_intervals(known, selected)
            )
            if not len(selected):
                raise ValueError(
                    "epochs does not overlap the decoding result's known recording "
                    "support (spike_window). Select epochs inside that support."
                )
        field_options["epochs"] = selected
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
        closed_readers: set[int] = set()
        for video in self._overlay_data.videos:
            close = getattr(video.reader, "close", None)
            if (
                close is not None
                and getattr(video, "_owns_reader", False)
                and id(video.reader) not in closed_readers
            ):
                closed_readers.add(id(video.reader))
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
