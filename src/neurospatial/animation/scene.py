"""A physical-clock animation scene shared by inspection and export."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from neurospatial._intervals import as_intervals, intersect_intervals, intervals_contain
from neurospatial._observation import (
    _validated_signal_times,
    resolve_sample_support,
    validate_break_before,
    validate_max_gap,
)

if TYPE_CHECKING:
    from neurospatial.animation._view import AnimationView
    from neurospatial.decoding import DecodingResult


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
    _extremes: tuple[np.ndarray, np.ndarray] | None = dataclass_field(
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
        object.__setattr__(self, "_extremes", None)

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
        """Read spatial rows; a single-row result owns its sampled values."""
        if isinstance(index, slice):
            return self.values[index]
        # Lazy readers may reuse their scratch storage on the next access.
        # A sampled observation must remain stable across independent streams
        # and the worker-to-GUI handoff without materializing the whole source.
        row = np.array(self.values[index], dtype=float, copy=True)
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
        index = self._sample_index(time)
        if index is not None:
            return cast("np.ndarray", self[index])
        return np.full(self.n_bins, np.nan)

    def _sample_index(self, time: float) -> int | None:
        """The held source row within support, or None for a missing sample."""
        index = int(np.searchsorted(self.times, time, side="right")) - 1
        if (
            index >= 0
            and self.sample_valid[index]
            and (
                self.times[index] == time
                or (index < len(self.times) - 1 and self.interval_valid[index])
            )
        ):
            return index
        return None

    def color_limits(self) -> tuple[float, float]:
        """Return stable display limits, reading at most 100 source rows."""
        if self.vmin is not None and self.vmax is not None:
            return self.vmin, self.vmax
        from neurospatial.animation._field_stats import robust_limits

        low, high = robust_limits(*self._scale_extremes())
        return (
            low if self.vmin is None else self.vmin,
            high if self.vmax is None else self.vmax,
        )

    def _scale_rows(self) -> list[np.ndarray]:
        indices = np.flatnonzero(self.sample_valid)
        if len(indices) > 100:
            indices = indices[np.linspace(0, len(indices) - 1, 100, dtype=int)]
        return [self[int(index)] for index in indices]

    def _scale_extremes(self) -> tuple[np.ndarray, np.ndarray]:
        """Each sampled row's smallest and largest finite value, read once."""
        if self._extremes is None:
            from neurospatial.animation._field_stats import frame_extremes

            object.__setattr__(self, "_extremes", frame_extremes(self._scale_rows()))
        assert self._extremes is not None
        return self._extremes

    def _beyond(self, vmin: float, vmax: float) -> tuple[bool, bool]:
        """Whether sampled values lie below ``vmin`` and above ``vmax``.

        Colors clip values outside the scale, so a colorbar marks each end
        that sampled values pass, whether the limits were given or computed.
        """
        lows, highs = self._scale_extremes()
        return bool(np.any(lows < vmin)), bool(np.any(highs > vmax))

    def _typical_peak(self) -> float:
        from neurospatial.animation._field_stats import typical_peak

        peak = typical_peak(self._scale_extremes()[1])
        return peak if peak > 0 else 1.0


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
        self._closing = False
        self._view_lock = RLock()
        # Native viewers prepare frames on workers. All views and exports of
        # this scene share readers, which need not support concurrent seeks.
        self._source_lock = RLock()
        self._views: set[AnimationView] = set()
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
        if env.n_dims == 3:
            from neurospatial.animation._region_names import shown_region_names

            planar_regions = any(
                env.regions[name].kind != "point"
                or np.asarray(env.regions[name].data).shape != (3,)
                for name in (
                    shown_region_names(env, show_regions) if show_regions else []
                )
            )
            if planar_regions or scale_bar:
                raise ValueError(
                    "3D Animation cannot place planar regions or a 2D scale bar "
                    "unambiguously. Use 3D point regions and omit scale_bar; the 3D axes "
                    "show physical coordinates and units."
                )
        from neurospatial.animation.overlays import (
            _convert_overlays_to_data,
            _freeze_overlay_arrays,
            _Overlay,
        )

        # Built-in overlays are snapshotted. An adapter may be frozen, expose
        # read-only properties, or own native resources that cannot be
        # copied; only its converted data is owned, and renderers never
        # consult the adapter afterward.
        self.overlays: tuple[Any, ...] = tuple(
            overlay._snapshot() if isinstance(overlay, _Overlay) else overlay
            for overlay in (() if overlays is None else overlays)
        )
        self._renderer: Any = None
        # Validate overlays once on the scene clock. Keep converted video readers
        # for repeated frame inspection rather than reopening the source.
        self._overlay_data, self._owned_readers = _convert_overlays_to_data(
            list(self.overlays), self.frame_times, len(self.frame_times), self.env
        )
        _freeze_overlay_arrays(self._overlay_data)

    def overlay_data(self, frame_times: Any = None) -> Any:
        """The overlays sampled on a physical clock, by default the scene's.

        Sampled overlays (positions, pose, headings, object vectors and
        video) get one row per requested time, with ``times`` set to it and
        the support each overlay was prepared with. Arrays shared with the
        scene are read-only; copy one to modify it.
        """
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
        return _realign_prepared_overlays(self._overlay_data, clock)

    @classmethod
    def from_decoding(cls, result: DecodingResult, **kwargs: Any) -> Animation:
        """Build a probability animation on its decoding clock and recording support.

        Known ``result.spike_window`` always intersects ``epochs`` supplied
        here. Neither wider epochs nor ``max_gap=None`` can fill known
        recording gaps. Held intervals must fit wholly inside one resulting
        half-open window; supported isolated samples remain visible exactly.
        """
        if result.times is None:
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
        known = as_intervals(result.spike_window, name="spike_window")
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
        field = AnimationField(result.posterior, result.times, **field_options)
        return cls(result.env, {"posterior": field}, **kwargs)

    def sample_fields(self, time: float) -> dict[str, np.ndarray]:
        """Sample every named stream at the same physical time."""
        with self._source_lock:
            self._ensure_open()
            return {name: field.sample(time) for name, field in self.fields.items()}

    def render_frame(
        self, time: float, *, dpi: int = 100, figsize: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Render a complete scene as an RGB array, without a display."""
        from neurospatial.animation._scene_rendering import SceneRenderer

        with self._source_lock:
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
        crf: int | None = None,
        preset: str | None = None,
        bitrate: float | None = None,
        n_workers: int | None = 1,
        image_format: str = "png",
        max_html_frames: int = 500,
        embed: bool = True,
        frames_dir: str | Path | None = None,
    ) -> Path:
        """Write a video, GIF or HTML player at uniform physical-clock intervals.

        Adjacent output frames advance by ``speed / fps`` seconds. A selected
        half-open range ``[start, stop)`` changes only the output clock; scales
        and overlay clocks remain those of the complete scene. The file appears
        only when complete, and an existing file is replaced only with
        ``overwrite=True``. Options that do not apply to the container raise
        ``TypeError``.

        Parameters
        ----------
        path : str or Path
            Destination. The extension selects the container: ``.mp4``,
            ``.webm``, ``.mov``, ``.avi``, ``.gif`` or ``.html``.
        time_range : (float, float), optional
            Half-open ``[start, stop)`` recording interval in seconds. By
            default the whole scene, through its last time.
        fps : float, default=30
            Output frames per second.
        speed : float, default=1
            Recording seconds per playback second.
        overwrite : bool, default=False
            Replace an existing file, and the frames in a nonempty
            ``frames_dir``.
        dpi : int, default=100
            Resolution of the rendered frames.
        figsize : (float, float), optional
            Figure size in inches; sized to the panels by default.
        codec : str, optional
            Video encoder: ``"h264"``, ``"h265"``, ``"vp9"``, ``"mpeg4"`` or
            another ffmpeg encoder name. Defaults to H.264, or VP9 for
            ``.webm``. Video containers only.
        crf : int, optional
            Constant quality from 0 (best) to 51 (63 for VP9); lower values
            give larger files. H.264, H.265 and VP9 use 18, visually
            lossless, unless ``bitrate`` is given. Video containers only.
        preset : str, optional
            H.264/H.265 encoding speed, ``"ultrafast"`` to ``"veryslow"``;
            slower presets compress better. Defaults to ``"medium"``.
        bitrate : float, optional
            Constant bitrate in kilobits per second; overrides ``crf``.
            Video containers only.
        n_workers : int or None, default=1
            Processes that render frames; None uses every CPU. Workers rebuild
            this scene, so its environment, fields and overlays must be
            picklable. Memory-mapped fields are reopened from their files
            rather than copied. Frames are identical to ``n_workers=1``.
            Scripts that use workers need an ``if __name__ == "__main__":``
            guard on macOS and Windows.
        image_format : {"png", "jpeg"}, default="png"
            HTML frame encoding. PNG is lossless; JPEG is smaller. ``.html``
            only.
        max_html_frames : int, default=500
            Most frames embedded in one page, which a browser holds in
            memory. ``.html`` with ``embed=True`` only.
        embed : bool, default=True
            Embed frames in one self-contained page. False writes image files
            to ``frames_dir``, loaded by relative URL. ``.html`` only.
        frames_dir : str or Path, optional
            Directory for ``embed=False`` frames. Defaults to a directory
            beside the page named after it (``player.html`` -> ``player/``).

        Returns
        -------
        Path
            The destination path.
        """
        from neurospatial.animation._scene_rendering import export_scene

        # Views' workers share these sources, and close() waits for a running
        # export to finish before releasing readers.
        with self._source_lock:
            self._ensure_open()
            return export_scene(
                self,
                Path(path),
                dpi=dpi,
                figsize=figsize,
                time_range=time_range,
                fps=fps,
                speed=speed,
                overwrite=overwrite,
                codec=codec,
                crf=crf,
                preset=preset,
                bitrate=bitrate,
                n_workers=n_workers,
                image_format=image_format,
                max_html_frames=max_html_frames,
                embed=embed,
                frames_dir=frames_dir,
            )

    def _construction_arguments(self) -> dict[str, Any]:
        """Arguments that rebuild an equivalent scene, e.g. in a worker process."""
        return {
            "env": self.env,
            "fields": dict(self.fields),
            "frame_times": self.frame_times,
            "overlays": self.overlays,
            "title": self.title,
            "window_seconds": self.window_seconds,
            "show_regions": self.show_regions,
            "region_alpha": self.region_alpha,
            "scale_bar": self.scale_bar,
            "track": self.track,
        }

    def show(self, *, backend: str = "widget", **kwargs: Any) -> AnimationView:
        """Open an interactive view of this scene."""
        from neurospatial.animation._view import show_animation

        self._ensure_view_allowed()
        return show_animation(self, backend=backend, **kwargs)

    def inspect_events(self, events: Any, **kwargs: Any) -> Any:
        """Inspect event-centered windows on this scene's physical clock."""
        from neurospatial.animation.events import EventInspector

        self._ensure_open()
        return EventInspector(self, events, **kwargs)

    def _register_view(self, view: AnimationView) -> None:
        with self._view_lock:
            self._ensure_view_allowed()
            self._views.add(view)

    def _unregister_view(self, view: AnimationView) -> None:
        with self._view_lock:
            self._views.discard(view)

    def _ensure_view_allowed(self) -> None:
        with self._view_lock:
            self._ensure_open()
            if self._closing:
                raise RuntimeError(
                    "Animation is closing. Create a new scene to open a view."
                )

    def close(self) -> None:
        """Stop and drain all views, then release owned figures and readers."""
        with self._view_lock:
            if self._closed or self._closing:
                return
            self._closing = True
            views = tuple(self._views)
        from neurospatial.animation._resources import _cleanup_error

        failures: list[BaseException] = []
        try:
            # Keep sources open while native controls stop and pending reads drain.
            for view in views:
                try:
                    view.close()
                except BaseException as error:
                    failures.append(error)
            with self._view_lock:
                self._views.clear()
            # Native workers have drained. Also let any already-running public
            # read/export finish before closing shared sources, and publish the
            # closed state before a waiting reader can acquire this lock.
            with self._source_lock:
                try:
                    if self._renderer is not None:
                        renderer, self._renderer = self._renderer, None
                        try:
                            renderer.close()
                        except BaseException as error:
                            failures.append(error)
                    try:
                        self._owned_readers.close()
                    except BaseException as error:
                        failures.append(error)
                finally:
                    self._closed = True
        finally:
            with self._view_lock:
                self._closed = True
                self._closing = False
        if failures:
            raise _cleanup_error("Animation cleanup failed.", failures)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Animation is closed. Create a new scene to inspect or export it."
            )

    def __enter__(self) -> Animation:
        self._ensure_open()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if len(exc) > 1 and isinstance(exc[1], BaseException):
                from neurospatial.animation._resources import _add_cleanup_note

                _add_cleanup_note(exc[1], f"Animation cleanup also failed: {cleanup!r}")
            else:
                raise
