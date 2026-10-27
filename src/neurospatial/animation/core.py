"""Core animation orchestration.

This module routes ``Environment.animate_fields`` onto an :class:`Animation`
scene and provides utilities for working with large-scale time series data.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.animation._region_names import shown_region_names

if TYPE_CHECKING:
    from neurospatial.animation._view import AnimationView
    from neurospatial.animation.overlays import OverlayProtocol
    from neurospatial.animation.scene import AnimationField
    from neurospatial.environment import Environment

# Playback speed limits
MAX_PLAYBACK_FPS: int = 60  # Display refresh rate limit
DEFAULT_SPEED: float = 1.0  # Real-time by default

# Options that build the scene's fields and panels; every backend uses them.
_SCENE_OPTIONS = frozenset(
    {"cmap", "vmin", "vmax", "colorbar_label", "layer_names", "title", "track"}
    | {"window_seconds"}
)

# The further caller options each backend uses. A name in no set is refused;
# one that only other backends use is ignored with a warning. None means the
# caller did not pass the option.
_BACKEND_OPTIONS: dict[str, frozenset[str]] = {
    "napari": frozenset({"viewer", "show", "ndisplay", "layout"}),
    "video": frozenset(
        {
            "save_path",
            "overwrite",
            "dpi",
            "figsize",
            "codec",
            "crf",
            "preset",
            "bitrate",
            "n_workers",
        }
    ),
    "html": frozenset(
        {
            "save_path",
            "overwrite",
            "dpi",
            "figsize",
            "n_workers",
            "image_format",
            "max_html_frames",
            "embed",
            "frames_dir",
        }
    ),
    "widget": frozenset({"dpi", "figsize"}),
    "fastplotlib": frozenset(
        {"layout", "frame_labels", "field_rendering", "volume_threshold"}
        | {"slice_plane"}
    ),
}

# animate_fields parameters every backend uses: they set the frame timing,
# the overlays and what the scene shows rather than how one backend draws.
_SHARED_OPTIONS = frozenset(
    {"backend", "overlays", "frame_times", "speed", "max_playback_fps", "max_gap"}
    | {"show_regions", "region_alpha", "scale_bar"}
    | _SCENE_OPTIONS
)

_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".avi", ".gif")

_BACKEND_CALLS: dict[str, str] = {
    "napari": "backend='napari'",
    "video": "backend='video', save_path='animation.mp4'",
    "html": "backend='html', save_path='animation.html'",
    "widget": "backend='widget'",
    "fastplotlib": "backend='fastplotlib'",
}


def _auto_reason(backend: str, n_frames: int, save_path: str | Path | None) -> str:
    """Why ``backend="auto"`` picked ``backend`` for these inputs."""
    if backend in ("video", "html"):
        return f"save_path ends in {Path(str(save_path)).suffix!r}"
    if backend == "widget":
        return "this is a Jupyter session and save_path is not a video or .html file"
    if n_frames > 10_000:
        return f"there are {n_frames:,} frames, more than 10,000"
    return (
        "save_path is not a video or .html file, this is not a Jupyter "
        f"session, and there are {n_frames:,} frames (10,000 or fewer)"
    )


def _owners(name: str) -> list[str]:
    """The backends that use option ``name``."""
    return [kind for kind, names in _BACKEND_OPTIONS.items() if name in names]


def _chosen(backend: str, auto_reason: str | None) -> str:
    """Say which backend ``auto`` chose and why; nothing when it was named."""
    if auto_reason is None:
        return ""
    return (
        f"backend='auto' chose {backend!r} because {auto_reason}; name a "
        "backend to choose another. "
    )


def _backend_options(
    backend: str, options: dict[str, Any], *, auto_reason: str | None = None
) -> dict[str, Any]:
    """Return the options ``backend`` uses, and warn about the others.

    Parameters
    ----------
    backend : str
        The backend that renders.
    options : dict
        The options the caller passed.
    auto_reason : str, optional
        Why ``backend="auto"`` chose ``backend``; None when it was named.

    Returns
    -------
    dict
        The options ``backend`` uses.

    Raises
    ------
    TypeError
        If an option is not an option of any backend; the message names the
        closest valid option.

    Warns
    -----
    UserWarning
        Once, naming each option that another backend uses and this one
        ignores.
    """
    known = frozenset().union(*_BACKEND_OPTIONS.values())
    for name in options:
        if name not in known:
            raise _unknown_option_error(backend, name, auto_reason)
    uses = _BACKEND_OPTIONS[backend]
    ignored = [name for name in options if name not in uses]
    if ignored:
        warnings.warn(
            _ignored_options_message(backend, ignored, auto_reason),
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return {name: value for name, value in options.items() if name in uses}


def _ignored_options_message(
    backend: str, ignored: list[str], auto_reason: str | None
) -> str:
    """Name each ignored option, the backend that uses it, and how to use it."""
    lines = []
    for name in ignored:
        owners = _owners(name)
        plural = "s" if len(owners) > 1 else ""
        lines.append(
            f"{name} is used by the {' and '.join(owners)} backend{plural}; "
            f"the {backend} backend ignores it."
        )
    owners = list(dict.fromkeys(owner for name in ignored for owner in _owners(name)))
    choices = " or ".join(_BACKEND_CALLS[owner] for owner in owners)
    return (
        _chosen(backend, auto_reason)
        + "\n".join(lines)
        + f"\nTo use {'them' if len(ignored) > 1 else 'it'}, pass {choices}; "
        "otherwise remove " + ", ".join(f"{name}=" for name in ignored) + "."
    )


def _unknown_option_error(
    backend: str, name: str, auto_reason: str | None
) -> TypeError:
    """Explain an option no backend has, with the closest valid name."""
    from difflib import get_close_matches

    valid = sorted(frozenset().union(*_BACKEND_OPTIONS.values()) | _SHARED_OPTIONS)
    matches = get_close_matches(name, valid, n=1)
    if name == "fps":
        # The playback rate is derived, so the nearest name would mislead.
        fix = (
            "remove fps=; the playback rate is the sampling rate of "
            "frame_times times speed, so pass speed=2.0 to play twice as fast "
            "as real time"
        )
    elif matches:
        fix = f"pass {matches[0]}= instead of {name}="
    else:
        fix = f"remove {name}=...; the options are {', '.join(valid)}"
    return TypeError(
        _format_error(
            f"animate_fields() got an unknown option {name!r}. "
            + _chosen(backend, auto_reason),
            why="Why: no backend has this option, and ignoring it would hide "
            "a misspelled option.",
            fix=fix,
        )
    )


def _field_streams(
    fields: Any,
    frame_times: NDArray[np.float64],
    *,
    layer_names: Sequence[str] | None,
    max_gap: float | None,
    field_options: dict[str, Any],
) -> dict[str, AnimationField]:
    """Name each field stream and put it on the frame clock.

    A 2-D array or a list of 1-D frames is one field. A mapping, or a list
    whose items are themselves sequences of frames, holds several fields.
    """
    from neurospatial.animation.scene import AnimationField

    if isinstance(fields, Mapping):
        streams: dict[str, Any] = dict(fields)
    elif _is_field_list(fields):
        names = (
            list(layer_names)
            if layer_names is not None
            else [f"field {i + 1}" for i in range(len(fields))]
        )
        if len(names) != len(fields):
            raise ValueError(
                f"layer_names has {len(names)} names for {len(fields)} fields. "
                "Pass one name per field."
            )
        streams = dict(zip(names, fields, strict=True))
    else:
        # Arrays, lazy readers (memmap, h5py) and lists of frames are one
        # field, read by row.
        streams = {"field": fields}
    if not streams or any(len(values) == 0 for values in streams.values()):
        raise _empty_fields_error()
    for name, values in streams.items():
        if not isinstance(values, AnimationField) and len(values) != len(frame_times):
            raise ValueError(
                _format_error(
                    f"Field {name!r} has {len(values)} frames, but frame_times "
                    f"has {len(frame_times)} times",
                    fix="pass one row per entry of frame_times for every field, "
                    "or an AnimationField with its own times",
                    why="Why: each field row is shown from its time in frame_times.",
                )
            )
    label = field_options.pop("colorbar_label", None)
    built = {
        name: values
        if isinstance(values, AnimationField)
        else AnimationField(
            values, frame_times, max_gap=max_gap, label=label, **field_options
        )
        for name, values in streams.items()
    }
    plain = [name for name, values in streams.items() if values is not built[name]]
    vmin, vmax = field_options.get("vmin"), field_options.get("vmax")
    if len(plain) > 1 and (vmin is None or vmax is None):
        # Fields shown side by side share one cmap and colorbar label, so they
        # share one color scale: the same color is the same value in each.
        # Each field caches the rows it read, so its scale reads them once.
        from neurospatial.animation._field_stats import robust_limits

        extremes = [built[name]._scale_extremes() for name in plain]
        low, high = robust_limits(
            np.concatenate([lows for lows, _ in extremes]),
            np.concatenate([highs for _, highs in extremes]),
        )
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
        if vmin > vmax:
            raise ValueError("vmin must be less than or equal to vmax.")
        for name in plain:
            shared = copy.copy(built[name])
            object.__setattr__(shared, "vmin", vmin)
            object.__setattr__(shared, "vmax", vmax)
            built[name] = shared
    return built


def _is_field_list(fields: Any) -> bool:
    """Whether a list holds several fields rather than one field's frames.

    Each item of a field list is itself a sequence of frames: a list, a
    tuple or a 2-D array.
    """
    return (
        isinstance(fields, (list, tuple))
        and bool(fields)
        and (
            isinstance(fields[0], (list, tuple))
            or (isinstance(fields[0], np.ndarray) and fields[0].ndim == 2)
        )
    )


def _empty_fields_error() -> ValueError:
    return ValueError(
        _format_error(
            "fields cannot be empty (got 0 frames)",
            fix="pass at least one field and a matching frame_times array",
            why="Why: an animation cannot render zero frames.",
        )
    )


def _n_frames(fields: Any) -> int:
    """Frames in the first field, before its streams are built.

    Raises
    ------
    ValueError
        If an array of fields is not 2-D, or there are no frames.
    """
    if not isinstance(fields, Mapping) and not (
        hasattr(fields, "__getitem__") and hasattr(fields, "__len__")
    ):
        raise TypeError(
            "fields must be an array or an indexable sequence of rows "
            "(n_frames, n_bins), a list of such fields, or a mapping from name "
            f"to field; got {type(fields).__name__}. Fix: pass np.asarray(rows) "
            "or a memory-mapped array instead of an iterator."
        )
    if isinstance(fields, Mapping):
        fields = next(iter(fields.values()), [])
    elif isinstance(fields, np.ndarray):
        if fields.ndim != 2:
            raise ValueError(
                _format_error(
                    f"fields must have shape (n_frames, n_bins), got {fields.shape}",
                    fix="pass fields with shape (n_frames, n_bins), e.g. field[None, :]",
                    why="Why: each animation frame needs one value per environment bin.",
                )
            )
    elif _is_field_list(fields):
        fields = fields[0]
    if len(fields) == 0:
        raise _empty_fields_error()
    return len(fields)


def animate_fields(
    env: Environment,
    fields: Sequence[NDArray[np.float64]] | NDArray[np.float64] | Mapping[str, Any],
    *,
    backend: Literal[
        "auto", "napari", "video", "html", "widget", "fastplotlib"
    ] = "auto",
    save_path: str | Path | None = None,
    overwrite: bool | None = None,
    overlays: list[OverlayProtocol] | None = None,
    frame_times: NDArray[np.float64],
    speed: float = DEFAULT_SPEED,
    max_playback_fps: int = MAX_PLAYBACK_FPS,
    max_gap: float | None = DEFAULT_MAX_GAP,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
    scale_bar: bool | Any = None,  # bool | ScaleBarConfig
    **kwargs: Any,
) -> AnimationView | Path:
    """Animate spatial fields on their recording clock in one call.

    Builds an :class:`~neurospatial.animation.Animation` from ``fields`` and
    ``overlays``, then opens a view of it or exports it to ``save_path``. Use
    ``Animation`` directly to inspect events, open several views of one
    scene, or export selected time windows.

    Parameters
    ----------
    env : Environment
        Environment defining spatial structure.
    fields : ndarray, shape (n_frames, n_bins), or sequence or mapping
        Spatial values, one row per entry of ``frame_times``. A list of
        ``(n_bins,)`` arrays is one field. Several fields shown side by side
        are a mapping from name to field, or a list of fields named by
        ``layer_names``. Rows are read lazily, so memory-mapped arrays stay
        on disk.
    backend : {"auto", "napari", "video", "html", "widget", "fastplotlib"}
        ``"video"`` and ``"html"`` write ``save_path``; the others open an
        interactive view. ``"auto"`` exports when ``save_path`` is a video
        or ``.html`` file, uses the notebook widget in Jupyter, and napari
        otherwise.
    save_path : str or Path, optional
        Output file: ``.mp4``, ``.webm``, ``.mov``, ``.avi``, ``.gif`` or
        ``.html``.
    overwrite : bool, default=False
        Allow replacing an existing output file.
    overlays : list of overlays, optional
        Positions, pose, head direction, video, events, signals, spike
        rasters, epochs and object vectors drawn on each field.
    frame_times : ndarray, shape (n_frames,)
        Strictly increasing time of each field row, in seconds.
    speed : float, default=1.0
        Playback speed relative to real time.
    max_playback_fps : int, default=60
        Highest frame rate shown or written. Output frames advance
        ``speed / fps`` recording seconds, where ``fps`` is the sampling rate
        of ``frame_times`` times ``speed``, capped at this value. A capped
        export keeps ``speed`` and skips the rows between output frames.
    max_gap : float or None, default=1.0
        Longest interval, in seconds, over which a field row is held until
        the next one; across longer gaps the field shows as missing. Pass
        None to hold every row until the next, for example for one field per
        trial. The default is shared with the trajectory analyses, calibrated
        for typical video and DeepLabCut tracking; choose a value longer than
        the spacing of your rows and shorter than real pauses in the recording.
    show_regions : bool or list of str, default=False
        Draw all environment regions, or the named ones.
    region_alpha : float, default=0.3
        Region opacity, between 0 and 1.
    scale_bar : bool or ScaleBarConfig, optional
        Draw a scale bar in the environment's units.
    **kwargs
        Scene options for every backend: ``cmap``, ``vmin``, ``vmax``,
        ``colorbar_label``, ``layer_names``, ``title``, ``window_seconds``,
        ``track``. File options: ``dpi``, ``figsize``, ``n_workers`` and,
        for video, ``codec``, ``crf``, ``preset``, ``bitrate``; for HTML,
        ``image_format``, ``max_html_frames``, ``embed``, ``frames_dir``.
        View options: ``dpi`` and ``figsize`` (widget); ``viewer``, ``show``,
        ``ndisplay``, ``layout`` (napari); ``layout``, ``frame_labels``,
        ``field_rendering``, ``volume_threshold``, ``slice_plane``
        (fastplotlib). An option no backend uses raises ``TypeError``; one
        that only another backend uses warns.

    Returns
    -------
    AnimationView or Path
        The open view, whose ``native`` attribute is the viewer, or the
        written file. Closing the view releases the readers it acquired.

    Raises
    ------
    FileExistsError
        If the output exists and ``overwrite`` is False.
    TypeError
        If an option is not an option of any backend.
    ValueError
        If the fields, times or backend are invalid.

    See Also
    --------
    neurospatial.animation.Animation : Reusable scenes for views and exports.

    Examples
    --------
    >>> env.animate_fields(
    ...     fields, frame_times=times, backend="napari"
    ... )  # doctest: +SKIP
    >>> env.animate_fields(
    ...     fields, frame_times=times, save_path="replay.mp4"
    ... )  # doctest: +SKIP
    """
    from neurospatial.animation.scene import Animation

    # A misspelled region name fails here, before any frame is rendered.
    shown_region_names(env, show_regions)
    from neurospatial.animation._capabilities import (
        checked_max_playback_fps,
        checked_speed,
    )

    checked_speed(speed)
    checked_max_playback_fps(max_playback_fps)
    from neurospatial.animation.overlays import _validate_frame_times

    frame_times = _validate_frame_times(frame_times, _n_frames(fields))

    auto_reason = None
    if backend == "auto":
        backend = _select_backend(len(frame_times), save_path)
        auto_reason = _auto_reason(backend, len(frame_times), save_path)
    if backend not in _BACKEND_OPTIONS:
        raise ValueError(
            _format_error(
                f"Unknown backend: {backend}",
                fix="pass backend='napari', 'video', 'html', 'widget', "
                "'fastplotlib', or 'auto'",
                why="Why: only the advertised rendering backends are supported.",
            )
        )
    scene_options = {
        name: kwargs.pop(name) for name in _SCENE_OPTIONS if name in kwargs
    }
    # overwrite=False is the default, not a request a view should warn about.
    passed = {"save_path": save_path, "overwrite": overwrite or None, **kwargs}
    options = _backend_options(
        backend,
        {name: value for name, value in passed.items() if value is not None},
        auto_reason=auto_reason,
    )
    if backend in ("video", "html"):
        _check_save_path(backend, options.get("save_path"))

    # Output frames advance speed / fps recording seconds: one per row at the
    # sampling rate of frame_times, unless that exceeds max_playback_fps.
    duration = float(frame_times[-1] - frame_times[0])
    rate = (len(frame_times) - 1) / duration if duration > 0 else None
    fps = (
        float(max_playback_fps)
        if rate is None
        else min(rate * speed, float(max_playback_fps))
    )

    field_options = {
        name: scene_options.pop(name)
        for name in ("cmap", "vmin", "vmax", "colorbar_label")
        if name in scene_options
    }
    streams = _field_streams(
        fields,
        frame_times,
        layer_names=scene_options.pop("layer_names", None),
        max_gap=max_gap,
        field_options=field_options,
    )
    scene = Animation(
        env,
        streams,
        frame_times=frame_times,
        overlays=overlays,
        show_regions=show_regions,
        region_alpha=region_alpha,
        scale_bar=scale_bar,
        **scene_options,
    )
    if backend in ("video", "html"):
        if rate is not None and rate * speed > max_playback_fps:
            warnings.warn(
                f"speed={speed:g} would need {rate * speed:.0f} fps; the file is "
                f"written at max_playback_fps={max_playback_fps}, so it skips "
                "rows between output frames. Raise max_playback_fps or lower "
                "speed to keep every row.",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
        with scene:
            return scene.export(
                options.pop("save_path"),
                fps=fps,
                speed=speed,
                overwrite=bool(options.pop("overwrite", False)),
                **options,
            )
    options.pop("overwrite", None)
    try:
        if backend == "napari":
            view = scene.show(backend="napari", fps=fps, speed=speed, **options)
        else:
            view = scene.show(
                backend=backend,
                speed=speed,
                max_playback_fps=max_playback_fps,
                **options,
            )
        # The scene was made for this view; closing the view releases it.
        view._add_close_callback(scene.close)
    except BaseException as error:
        # Report the failure itself; a cleanup failure is noted on it.
        scene.__exit__(type(error), error, error.__traceback__)
        raise
    return view


def _check_save_path(backend: str, save_path: Any) -> None:
    """Refuse a missing output or one whose suffix the backend cannot write."""
    suffixes = (".html",) if backend == "html" else _VIDEO_SUFFIXES
    if save_path is None or Path(save_path).suffix.lower() not in suffixes:
        example = "animation.html" if backend == "html" else "animation.mp4"
        raise ValueError(
            _format_error(
                f"The {backend} backend writes a {' or '.join(suffixes)} file; "
                f"got save_path={save_path!r}",
                fix=f"pass save_path='{example}'",
                why=f"Why: the {backend} backend writes to an output file.",
            )
        )


def _select_backend(
    n_frames: int, save_path: str | Path | None
) -> Literal["napari", "video", "html", "widget"]:
    """Choose a backend for ``backend="auto"``, logging why at INFO level.

    Raises
    ------
    RuntimeError
        If the chosen interactive viewer is not installed.
    """
    logger = logging.getLogger(__name__)
    if save_path:
        suffix = Path(save_path).suffix.lower()
        if suffix in _VIDEO_SUFFIXES:
            logger.info("Auto-selected 'video' backend (save_path suffix %s)", suffix)
            return "video"
        if suffix == ".html":
            logger.info("Auto-selected 'html' backend (save_path suffix .html)")
            return "html"
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]

        in_jupyter = get_ipython() is not None
    except ImportError:
        in_jupyter = False
    if in_jupyter and n_frames <= 10_000:
        logger.info("Auto-selected 'widget' backend (running in Jupyter)")
        return "widget"
    if importlib.util.find_spec("napari") is None:
        raise RuntimeError(
            _format_error(
                "backend='auto' chose napari for an interactive view, but napari "
                "is not installed",
                fix="install napari (pip install 'napari[all]'), pass "
                "backend='widget' or 'fastplotlib', or pass save_path='out.mp4' "
                "or 'out.html' to export",
                why="Why: outside Jupyter, and for more than 10,000 frames, "
                "napari is the interactive viewer.",
            )
        )
    logger.info("Auto-selected 'napari' backend for %s frames", f"{n_frames:,}")
    return "napari"


def _subsample_indices(n: int, source_fps: int, target_fps: int) -> NDArray[np.int_]:
    """Compute indices for frame subsampling with proper rounding.

    Avoids duplicate or skipped frames by using rounding instead of truncation,
    and ensures strictly increasing indices.

    Parameters
    ----------
    n : int
        Total number of frames
    source_fps : int
        Original sampling rate
    target_fps : int
        Desired output frame rate

    Returns
    -------
    indices : ndarray
        Frame indices to keep (strictly increasing, no duplicates)
    """
    rate = source_fps / target_fps
    # Use rounding to avoid drift from floating-point truncation
    idx = np.rint(np.arange(0, n, rate)).astype(np.int64)
    # Clamp to valid range
    idx = idx[idx < n]
    # Ensure strictly increasing by removing duplicates (preserves order)
    result: NDArray[np.int_] = np.unique(idx)
    # Ensure first frame is always included
    if result.size > 0 and result[0] != 0:
        result = np.concatenate([[0], result[result != 0]])
    return result


def subsample_frames(
    fields: NDArray | list,
    *,
    target_fps: int,
    source_fps: int,
) -> NDArray | list:
    """Subsample frames to target frame rate.

    Essential utility for large-scale sessions. Allows users to reduce
    900K frames at 250 Hz to a manageable video at 30 fps.

    Parameters
    ----------
    fields : ndarray of shape (n_frames, n_bins) or list of ndarray of shape (n_bins,)
        Full field data, dtype float64. If ndarray, first dimension is time.
        If list, each element is a single frame's field values.
    target_fps : int
        Desired output frame rate (e.g., 30 for video). Must be positive and
        less than or equal to source_fps.
    source_fps : int
        Original sampling rate (e.g., 250 Hz for neural recording). Must be positive.

    Returns
    -------
    subsampled : ndarray of shape (n_subsampled, n_bins) or list of ndarray of shape (n_bins,)
        Subsampled fields at target_fps, same type and dtype as input.
        n_subsampled is approximately n_frames * target_fps / source_fps.

    Raises
    ------
    ValueError
        If target_fps exceeds source_fps

    Examples
    --------
    >>> # 250 Hz recording → 30 fps video
    >>> fields_video = subsample_frames(
    ...     fields_full, target_fps=30, source_fps=250
    ... )  # doctest: +SKIP
    >>> frame_times_video = subsample_frames(
    ...     frame_times, target_fps=30, source_fps=250
    ... )  # doctest: +SKIP
    >>> env.animate_fields(
    ...     fields_video, frame_times=frame_times_video, save_path="output.mp4"
    ... )  # doctest: +SKIP

    >>> # 1000 Hz → 60 fps
    >>> fields_video = subsample_frames(
    ...     fields_full, target_fps=60, source_fps=1000
    ... )  # doctest: +SKIP

    Notes
    -----
    Subsampling uses proper rounding to avoid frame drift over long sequences.
    The algorithm ensures:
    - No duplicate frames (strictly increasing indices)
    - No skipped frames due to floating-point truncation
    - First frame (index 0) is always included
    - Works efficiently with memory-mapped arrays

    For example:
    - 250 Hz → 30 fps: takes frames at indices [0, 8, 17, 25, ...]
    - 1000 Hz → 60 fps: takes frames at indices [0, 17, 33, 50, ...]
    """
    if target_fps > source_fps:
        raise ValueError(
            f"target_fps ({target_fps}) cannot exceed source_fps ({source_fps})"
        )

    indices = _subsample_indices(len(fields), source_fps, target_fps)

    if isinstance(fields, np.ndarray):
        result: NDArray | list = fields[indices]
        return result
    else:
        result_list: NDArray | list = [fields[i] for i in indices]
        return result_list


def estimate_colormap_range_from_subset(
    fields: NDArray | list,
    *,
    n_samples: int = 10_000,
    percentile: tuple[float, float] = (1.0, 99.0),
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate colormap range from random subset of frames.

    The same rule `animate_fields` uses by default, on a random sample of
    frames: the bottom is a low percentile of the sampled frames' minima and
    the top a high percentile of their maxima, so a sharp posterior's peak
    reaches the top color rather than passing it in every frame. Useful
    for large sessions, where reading every frame would be slow.

    Parameters
    ----------
    fields : ndarray of shape (n_frames, n_bins) or list of ndarray of shape (n_bins,)
        Full field data, dtype float64. If ndarray, first dimension is time.
        If list, each element is a single frame's field values.
    n_samples : int, default=10_000
        Maximum number of frames to sample. If dataset has fewer frames,
        all frames are used.
    percentile : tuple of float, default=(1.0, 99.0)
        The percentile of the frames' minima for the bottom and of their
        maxima for the top: about this share of frames reach past each end,
        so a few outlier frames cannot wash out the rest.
    seed : int, optional
        Random seed for reproducibility. If None, results may vary between calls.

    Returns
    -------
    vmin, vmax : tuple of float
        Estimated colormap range (lower and upper bounds).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> fields = rng.random((100_000, 500))  # 100K frames, 500 bins
    >>> vmin, vmax = estimate_colormap_range_from_subset(fields)
    >>> env.animate_fields(
    ...     fields, vmin=vmin, vmax=vmax, backend="napari"
    ... )  # doctest: +SKIP

    >>> # Tighter range with 5th/95th percentiles
    >>> vmin, vmax = estimate_colormap_range_from_subset(
    ...     fields, percentile=(5.0, 95.0)
    ... )  # doctest: +SKIP

    Notes
    -----
    A bottom between zero and a thousandth of the top is zero, and a
    constant field gets limits half a unit around it.

    This function is designed to complete quickly (<0.5s) even for
    datasets with 1M+ frames by sampling a subset rather than computing
    over the entire dataset.

    Works efficiently with memory-mapped arrays - only the sampled
    indices are loaded into memory.
    """
    # Get number of frames
    n_frames = fields.shape[0] if isinstance(fields, np.ndarray) else len(fields)

    # Determine indices to sample
    rng = np.random.default_rng(seed)
    if n_frames <= n_samples:
        # Use all frames
        indices = np.arange(n_frames)
    else:
        # Sample without replacement
        indices = rng.choice(n_frames, size=n_samples, replace=False)

    from neurospatial.animation._field_stats import frame_extremes, robust_limits

    # Sorted, so a memory-mapped array is read forward.
    indices = np.sort(indices)
    sampled = (
        fields[indices]
        if isinstance(fields, np.ndarray)
        else [fields[i] for i in indices]
    )
    return robust_limits(*frame_extremes(sampled), percentile)
