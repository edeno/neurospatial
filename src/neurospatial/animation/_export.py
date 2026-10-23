"""Physical-clock sampling and atomic file writing shared by scene and views."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial._validation import check_writable

_VIDEO_CODECS = {
    ".mp4": "libx264",
    ".mov": "libx264",
    ".avi": "libx264",
    ".webm": "libvpx-vp9",
}
_CONTAINERS = (*_VIDEO_CODECS, ".gif", ".html")
_CODEC_ALIASES = {
    "h264": "libx264",
    "h265": "libx265",
    "vp9": "libvpx-vp9",
    "mpeg4": "mpeg4",
}
_WEBM_CODECS = {"libvpx", "libvpx-vp9", "libaom-av1", "libsvtav1"}
# Encoders whose quality options are known. Other ffmpeg encoders receive the
# options as given, without defaults.
_CRF_RANGE = {"libx264": 51, "libx265": 51, "libvpx-vp9": 63}
_PRESET_CODECS = {"libx264", "libx265"}
_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)
_HTML_DEFAULTS = {
    "image_format": "png",
    "max_html_frames": 500,
    "embed": True,
    "frames_dir": None,
}


@dataclass(frozen=True)
class _ExportPlan:
    path: Path
    times: NDArray[np.float64]
    fps: float
    overwrite: bool
    n_workers: int = 1
    title: str = "Animation"
    # Video encoding.
    codec: str | None = None
    output_params: tuple[str, ...] = ()
    quality: float | None = None
    # HTML player.
    image_format: str = "png"
    frames_dir: Path | None = None

    @property
    def container(self) -> str:
        return self.path.suffix.lower()

    def write(self, render_frame: Callable[[float], NDArray[np.uint8]]) -> Path:
        """Render each output time in order, then publish the finished file."""
        return self.write_frames(render_frame(float(time)) for time in self.times)

    def write_frames(self, frames: Iterable[NDArray[np.uint8]]) -> Path:
        """Write frames for ``times`` without publishing a partial file.

        An existing destination is replaced only with ``overwrite=True``, and
        never a file another writer created after ``prepare_export`` checked.
        """
        frames = _checked_frames(frames)
        if self.container == ".html":
            return self._write_html(frames)
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "Animation export requires imageio. Install video support with: "
                "pip install 'neurospatial[video]'."
            ) from exc

        options: dict[str, Any] = (
            {"mode": "I", "duration": 1000 / self.fps, "loop": 0}
            if self.container == ".gif"
            else {
                "fps": self.fps,
                "codec": self.codec,
                "pixelformat": "yuv420p",
                "quality": self.quality,
                "output_params": list(self.output_params),
                "macro_block_size": 2,
            }
        )
        with _temporary_sibling(self.path) as temporary:
            with cast("Any", imageio.get_writer(temporary, **options)) as writer:
                for frame in frames:
                    writer.append_data(frame)
            self._publish(temporary, self.path)
        return self.path

    def _write_html(self, frames: Iterator[NDArray[np.uint8]]) -> Path:
        from neurospatial.animation._html_player import (
            _generate_html_player,
            _generate_non_embedded_html_player,
            encode_frame,
            encode_frame_b64,
            frame_filename,
        )

        labels = [f"t = {time:.3f} s" for time in self.times]
        if self.frames_dir is None:
            page = _generate_html_player(
                [encode_frame_b64(frame, self.image_format) for frame in frames],
                labels,
                self.fps,
                self.title,
                self.image_format,
            )
            self._publish_text(page)
            return self.path

        frames_dir = self.frames_dir
        digits = max(5, len(str(len(self.times) - 1)))
        frames_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".animation-frames-", dir=frames_dir.parent)
        )
        try:
            names = []
            for index, frame in enumerate(frames):
                names.append(frame_filename(index, self.image_format, digits=digits))
                (staging / names[-1]).write_bytes(
                    encode_frame(frame, self.image_format)
                )
            page = _generate_non_embedded_html_player(
                frames_dir,
                len(names),
                labels,
                self.fps,
                self.title,
                self.image_format,
                output_path=self.path,
                digits=digits,
            )
            frames_dir.mkdir(exist_ok=True)
            for name in names:
                self._publish(staging / name, frames_dir / name)
            if self.overwrite:
                # Frames of an earlier, longer export would otherwise remain.
                current = set(names)
                for stale in frames_dir.iterdir():
                    if _FRAME_NAME.fullmatch(stale.name) and stale.name not in current:
                        stale.unlink()
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        self._publish_text(page)
        return self.path

    def _publish_text(self, text: str) -> None:
        with _temporary_sibling(self.path) as temporary:
            temporary.write_text(text, encoding="utf-8")
            self._publish(temporary, self.path)

    def _publish(self, temporary: Path, path: Path) -> None:
        if self.overwrite:
            temporary.replace(path)
        else:
            _publish_exclusive(temporary, path)


_FRAME_NAME = re.compile(r"frame_\d+\.(png|jpeg)")


@contextmanager
def _temporary_sibling(path: Path) -> Iterator[Path]:
    """A temporary file beside ``path``, removed unless moved into place.

    Unlike ``tempfile.mkstemp`` (owner-only), the file gets the permissions
    the user's umask gives any new file, which the published file keeps.
    """
    while True:
        name = path.parent / f".animation-{secrets.token_hex(8)}{path.suffix}"
        try:
            os.close(os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
            break
        except FileExistsError:
            continue
    try:
        yield name
    finally:
        name.unlink(missing_ok=True)


def _checked_frames(
    frames: Iterable[NDArray[np.uint8]],
) -> Iterator[NDArray[np.uint8]]:
    """Yield RGB frames, refusing other images or a size change."""
    shape = None
    for frame in frames:
        frame = np.asarray(frame)
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[-1] not in (3, 4)
            or min(frame.shape[:2]) == 0
        ):
            raise ValueError(
                "The renderer must return a nonempty uint8 RGB or RGBA image."
            )
        frame = frame[..., :3]
        if shape is None:
            shape = frame.shape
        elif frame.shape != shape:
            raise ValueError(
                "The rendered image size changed during export. "
                "Keep the viewer window size fixed until export completes."
            )
        yield frame


def _publish_exclusive(temporary: Path, path: Path) -> None:
    """Publish ``temporary`` at ``path`` without replacing a file that appeared.

    A different writer may create the destination after prepare_export checked
    it. Hard links publish complete files exclusively; filesystems without them
    (exFAT, FAT32, some network mounts) use an exclusive create and copy.
    """
    try:
        os.link(temporary, path)
        return
    except FileExistsError:
        raise
    except OSError:
        pass
    with path.open("xb") as destination:
        try:
            with temporary.open("rb") as source:
                shutil.copyfileobj(source, destination)
        except BaseException:
            destination.close()
            path.unlink(missing_ok=True)
            raise


def _number(value: object, *, integer: bool = True) -> float:
    """``value`` as a float if it is a finite (integer) number, else NaN."""
    if isinstance(value, bool) or not isinstance(value, Integral if integer else Real):
        return np.nan
    return float(value)


def _inapplicable(option: str, container: str, applies: str) -> TypeError:
    return TypeError(
        f"{option} does not apply to {container} exports; it applies to "
        f"{applies}. Remove {option}= or choose a {applies} path."
    )


def _video_output_params(
    container: str,
    codec: str | None,
    crf: int | None,
    preset: str | None,
    bitrate: float | None,
) -> tuple[str, tuple[str, ...], float | None]:
    """Resolve the encoder, its ffmpeg options and imageio's quality fallback.

    Encoders without a known constant-quality option keep imageio's default
    variable-quality setting unless a bitrate is given.
    """
    codec = _CODEC_ALIASES.get(codec, codec) if codec else _VIDEO_CODECS[container]
    if container == ".webm" and codec not in _WEBM_CODECS:
        raise ValueError(
            f"WebM holds VP8, VP9 or AV1 video, not {codec}. Use codec='vp9' "
            "(the default) for .webm, or an .mp4 path for H.264."
        )
    if bitrate is not None and not _number(bitrate, integer=False) > 0:
        raise ValueError("bitrate must be a positive number of kilobits per second.")
    if crf is not None:
        if codec == "mpeg4":
            raise TypeError(
                "crf does not apply to the mpeg4 codec. Use bitrate= for mpeg4, "
                "or codec='h264' for constant-quality encoding."
            )
        limit = _CRF_RANGE.get(codec)
        if not 0 <= _number(crf) <= (limit or np.inf):
            raise ValueError(
                f"crf must be an integer from 0 (best) to {limit or 51} (worst) "
                f"for {codec}; 18 is visually lossless for H.264."
            )
    if preset is not None:
        if codec in _CRF_RANGE and codec not in _PRESET_CODECS:
            raise TypeError(
                f"preset does not apply to the {codec} codec; it applies to "
                "h264 and h265. Remove preset=."
            )
        if codec in _PRESET_CODECS and preset not in _PRESETS:
            raise ValueError(f"preset must be one of {', '.join(_PRESETS)}.")
    if crf is not None and bitrate is not None:
        warnings.warn(
            f"bitrate={bitrate!r} and crf={crf!r} were both passed; the video is "
            "encoded at the constant bitrate, so crf is ignored. Pass only crf= "
            "for constant quality, or only bitrate= for a fixed bitrate.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    params: list[str] = []
    if codec in _PRESET_CODECS or preset is not None:
        params += ["-preset", preset or "medium"]
    if bitrate is not None:
        params += ["-b:v", f"{bitrate:g}k"]
    elif crf is not None or codec in _CRF_RANGE:
        params += ["-crf", str(18 if crf is None else int(crf))]
        if codec == "libvpx-vp9":
            # VP9 is constant quality only without a target bitrate.
            params += ["-b:v", "0"]
    if codec == "libx265":
        # x265 logs encoder statistics to stderr unless told otherwise.
        params += ["-x265-params", "log-level=error"]
    if container in (".mp4", ".mov"):
        # Web players can start before the whole file downloads.
        params += ["-movflags", "+faststart"]
        if codec == "libx265":
            # Apple players open H.265 only with the hvc1 sample entry.
            params += ["-tag:v", "hvc1"]
    rate_control = bitrate is not None or crf is not None or codec in _CRF_RANGE
    return codec, tuple(params), None if rate_control else 5


def prepare_export(
    path: str | Path,
    source_times: NDArray[np.float64],
    *,
    time_range: Any = None,
    fps: float = 30,
    speed: float = 1,
    overwrite: bool = False,
    codec: str | None = None,
    crf: int | None = None,
    preset: str | None = None,
    bitrate: float | None = None,
    n_workers: int | None = 1,
    image_format: str = "png",
    max_html_frames: int = 500,
    embed: bool = True,
    frames_dir: str | Path | None = None,
    title: str | None = None,
    sample_times: Iterable[NDArray[np.float64]] = (),
) -> _ExportPlan:
    """Validate export before touching a viewer and select its frame times.

    A ``time_range`` selects ``[start, stop)``; without one, the whole
    recording is exported, including its last sample when a frame lands on it.

    Correct arithmetic roundoff to the nearest observation within four ULPs.
    Exact queries stay exact; ties prefer the scene clock, then earlier clocks
    in ``sample_times``. Distinct nearby source floats cannot all be exact at
    one shared export time. Options that do not apply to the container raise
    ``TypeError``.
    """
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(speed) or speed <= 0:
        raise ValueError("fps and speed must be positive finite numbers.")
    path = Path(path)
    container = path.suffix.lower()
    if container not in _CONTAINERS:
        raise ValueError(
            f"Animation export writes {', '.join(_CONTAINERS)} files, not "
            f"{container or 'a path without an extension'}. Add one of these "
            "extensions to the path."
        )
    video = {"codec": codec, "crf": crf, "preset": preset, "bitrate": bitrate}
    html = {
        "image_format": image_format,
        "max_html_frames": max_html_frames,
        "embed": embed,
        "frames_dir": frames_dir,
    }
    if container != ".html":
        for option, value in html.items():
            if value != _HTML_DEFAULTS[option]:
                raise _inapplicable(option, container, ".html")
    if container in (".html", ".gif"):
        for option, value in video.items():
            if value is not None:
                raise _inapplicable(option, container, ".mp4, .webm, .mov and .avi")
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    elif not _number(n_workers) >= 1:
        raise ValueError(
            "n_workers must be a positive integer, or None to use every CPU."
        )
    resolved_codec, output_params, quality = (
        _video_output_params(container, codec, crf, preset, bitrate)
        if container in _VIDEO_CODECS
        else (None, (), None)
    )
    resolved_frames = None
    if container == ".html":
        image_format = str(image_format).lower()
        if image_format not in ("png", "jpeg"):
            raise ValueError(
                f"image_format must be 'png' or 'jpeg', got {image_format!r}."
            )
        if not isinstance(embed, bool):
            raise TypeError("embed must be True or False.")
        if embed and frames_dir is not None:
            raise TypeError(
                "frames_dir applies only with embed=False, which writes frames "
                "beside the page. Pass embed=False or remove frames_dir=."
            )
        if not embed and max_html_frames != _HTML_DEFAULTS["max_html_frames"]:
            raise TypeError(
                "max_html_frames limits only embedded players (embed=True); a "
                "page with external frames loads them on demand. Remove "
                "max_html_frames=."
            )
        if not _number(max_html_frames) >= 1:
            raise ValueError("max_html_frames must be a positive integer.")
        if not embed:
            resolved_frames = (
                Path(frames_dir) if frames_dir is not None else path.with_suffix("")
            )
            if resolved_frames.exists() and not resolved_frames.is_dir():
                raise FileExistsError(
                    f"frames_dir {resolved_frames} is a file. Choose a directory "
                    "path for the frames."
                )
            check_writable(
                [resolved_frames],
                overwrite=overwrite,
                what="animation frames directory",
                argument="frames_dir",
            )
    check_writable([path], overwrite=overwrite, what="animation file", argument="path")
    bounds = np.asarray(
        (source_times[0], source_times[-1]) if time_range is None else time_range,
        dtype=float,
    )
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[1] < bounds[0]
        or (time_range is not None and bounds[1] == bounds[0])
    ):
        raise ValueError(
            "time_range must be a finite half-open (start, stop) pair with start < stop."
        )
    if bounds[0] < source_times[0] or bounds[1] > source_times[-1]:
        raise ValueError("time_range must lie inside the scene's physical clock.")
    steps = float((bounds[1] - bounds[0]) * fps / speed)
    nearest = round(steps)
    if np.isclose(steps, nearest, rtol=1e-12, atol=1e-12):
        steps = float(nearest)
    # A selected [start, stop) excludes stop, so adjacent clips never repeat a
    # frame; the whole recording includes its last sample when a frame lands on it.
    whole = time_range is None
    count = max(1, int(np.floor(steps)) + 1 if whole else int(np.ceil(steps)))
    times = bounds[0] + np.arange(count) * speed / fps
    # Adding a decimal frame interval to an absolute recording time can land
    # one ULP before an observation (100.1 + 0.1 < 100.2). That must not hold
    # the preceding field or hide an exact sample beside a recording break.
    # Keep genuinely intermediate times unchanged, and consider each stream's
    # clock even when the caller chose a separate scene frame clock.
    snapped = times.copy()
    best_distance = np.full(times.shape, np.inf)
    tolerance = 4 * np.abs(np.spacing(times))
    for clock in chain((source_times,), sample_times):
        if not len(clock):
            continue
        right = np.searchsorted(clock, times)
        for index in (np.maximum(right - 1, 0), np.minimum(right, len(clock) - 1)):
            candidate = clock[index]
            distance = np.abs(candidate - times)
            valid = (candidate >= bounds[0]) & (candidate <= bounds[1])
            nearer = valid & (distance <= tolerance) & (distance < best_distance)
            snapped[nearer] = candidate[nearer]
            best_distance[nearer] = distance[nearer]
    times = snapped
    if whole:
        times = times[times <= bounds[1]]
    elif bounds[1] > bounds[0]:
        times = times[times < bounds[1]]
    times.flags.writeable = False
    if container == ".html" and embed and len(times) > max_html_frames:
        raise ValueError(
            f"The embedded HTML player would hold {len(times)} frames, more than "
            f"max_html_frames={max_html_frames}; large embedded pages load slowly "
            "or crash browsers. Export fewer frames with a shorter time_range= "
            "or a lower fps= (frames are speed/fps seconds apart), pass "
            "embed=False to write frames beside the page, or raise "
            "max_html_frames."
        )
    return _ExportPlan(
        path,
        times,
        float(fps),
        overwrite,
        n_workers=int(n_workers),
        title=title or path.stem,
        codec=resolved_codec,
        output_params=output_params,
        quality=quality,
        image_format=image_format,
        frames_dir=resolved_frames,
    )
