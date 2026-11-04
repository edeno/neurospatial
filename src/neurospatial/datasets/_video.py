"""A session's video: where it is, its frame times, and how its pixels map to space.

A dataset that publishes the camera's video gives each session a
:class:`VideoSource`: the file in the archive (asset id or URL, SHA-256 and
size, from the registry), one time per frame, and a calibration from video
pixels to the environment's centimetres. Each of the last two is recorded as
a :class:`~neurospatial.datasets.Conversion` saying where it comes from.
Nothing is downloaded until :meth:`VideoSource.fetch`.

Frame times (:func:`video_frame_times`) are the file's own video timestamps,
unless the video and the position series hold the same frames on two clocks;
the position clock is then used, since it is the one the spikes are on.

The calibration (:func:`video_calibration`) comes from the first source that
gives one, in this order: a pixel-to-cm transform stated in the file, a
least-squares fit between pixel tracking stored in the file and the
converted positions, landmarks placed by hand on one frame and kept in the
dataset's registry, or a least-squares fit of the tracked LED's position
detected in the video to the positions, run once by
``scripts/fit_video_led_calibration.py`` and kept in the registry. A fit or
landmarks whose residual exceeds one bin is refused. Without a calibration the video is not drawn: stretching the frame
over the environment's extent would place it wrong.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import InputError, _format_error
from neurospatial._results import read_only_copy
from neurospatial.datasets._recording import Conversion

if TYPE_CHECKING:
    from neurospatial.ops.transforms import AffineND, VideoCalibration

__all__ = [
    "CALIBRATION_SOURCES",
    "Landmarks",
    "LedFit",
    "VideoSource",
    "fit_pixel_affine",
    "led_fit_watch_out",
    "video_calibration",
    "video_frame_times",
]

#: The calibration sources, in the order they are tried.
CALIBRATION_SOURCES = (
    "a pixel-to-cm transform stated in the file",
    "a least-squares fit between pixel tracking in the file and its positions",
    "landmarks placed by hand on one frame, in the dataset's registry",
    "a least-squares fit of LED positions detected in the video to the "
    "positions, in the dataset's registry",
)

#: Two clocks hold the same frames when the position times are an offset and
#: rate of the video times to within this share of a frame interval.
SAME_FRAMES_TOLERANCE = 0.5


@dataclass(frozen=True)
class Landmarks:
    """Points placed by hand on one video frame, with their environment positions.

    Parameters
    ----------
    px : ndarray, shape (n_points, 2)
        Each point in video pixels, ``(column, row)`` from the top-left
        pixel's centre; at least three, not on one line.
    env : ndarray, shape (n_points, 2)
        The same points in the environment's coordinates, cm.
    frame_size_px : (int, int)
        The frame's ``(width, height)`` in pixels.
    frame : int
        The frame they were placed on.
    placed_by : str
        Who placed them, and when.
    """

    px: NDArray[np.float64]
    env: NDArray[np.float64]
    frame_size_px: tuple[int, int]
    frame: int
    placed_by: str

    def __post_init__(self) -> None:
        px = read_only_copy(self.px, dtype=np.float64)
        env = read_only_copy(self.env, dtype=np.float64)
        if px.ndim != 2 or px.shape[1] != 2 or px.shape != env.shape or len(px) < 3:
            raise InputError(
                _format_error(
                    f"Landmarks: px has shape {px.shape} and env {env.shape}.",
                    why="a calibration needs at least three points, each in "
                    "pixels and in the environment.",
                    fix="give both as (n_points, 2) arrays, n_points >= 3.",
                )
            )
        if not self.placed_by.strip():
            raise InputError(
                _format_error(
                    "Landmarks: placed_by is empty.",
                    why="a hand-placed calibration is checked by who placed it.",
                    fix="name the person and the date.",
                )
            )
        object.__setattr__(self, "px", px)
        object.__setattr__(self, "env", env)


@dataclass(frozen=True)
class LedFit:
    """A pixel-to-cm affine fitted from the LED detected in the video, stored.

    ``scripts/fit_video_led_calibration.py`` detects the tracked LED in
    frames of the video, fits a least-squares affine from its pixel position
    to the file's position of the same sample, and writes this record into
    the dataset's registry, so loading needs no video.

    Parameters
    ----------
    matrix : ndarray, shape (3, 3)
        The affine from video pixels ``(column, row, 1)`` to cm.
    frame_size_px : (int, int)
        The video's ``(width, height)``.
    method : str
        How the LED was detected and the affine fitted.
    frames : str
        Which frames were used, for example ``"every 30th frame from 0"``.
    n_frames : int
        The frames used: detected, with a finite position.
    median_cm, p90_cm : float
        Median and 90th percentile distance of the fitted positions from the
        file's, cm.
    median_px, p90_px : float
        The same in pixels: the file's positions mapped back into the frame
        against the detected LED.
    fitted_by : str
        The script, its version or commit, and the date.
    """

    matrix: NDArray[np.float64]
    frame_size_px: tuple[int, int]
    method: str
    frames: str
    n_frames: int
    median_cm: float
    p90_cm: float
    median_px: float
    p90_px: float
    fitted_by: str

    def __post_init__(self) -> None:
        matrix = read_only_copy(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not self.method.strip() or self.n_frames < 3:
            raise InputError(
                _format_error(
                    f"LedFit: matrix shape {matrix.shape}, method "
                    f"{self.method!r}, {self.n_frames} frames.",
                    why="a stored fit is used only with its 3 x 3 affine, its "
                    "method and the frames it rests on.",
                    fix="rerun scripts/fit_video_led_calibration.py.",
                )
            )
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class VideoSource:
    """A session's camera video: where it is, its frame times and its calibration.

    ``rec.video`` holds it for a session whose dataset publishes the video,
    and is None otherwise. The video is downloaded only by :meth:`fetch`, or
    by ``rec.animation(video=True)``, into the cache the session was loaded
    into, at the path the session's NWB file names relative to itself.

    Parameters
    ----------
    dataset : str
        The dataset id, which names the cache folder.
    external_file : str
        The video's path as the session's NWB file states it, relative to
        that file.
    sha256 : str
        The video's SHA-256, which every download is verified against.
    size_bytes : int
        The video's size.
    frame_times : ndarray, shape (n_frames,)
        One time per video frame, seconds on the recording's clock.
    frame_times_basis : Conversion
        Where ``frame_times`` come from.
    calibration : VideoCalibration or None
        Video pixels to the environment's cm; None when no source gives one.
    calibration_basis : Conversion or None
        Where ``calibration`` comes from, its residual in px and cm in the
        evidence; None without a calibration.
    calibration_tried : tuple of str
        Each calibration source tried before the one used (all of them,
        without a calibration), with why it gave none.
    asset_id, url : str or None
        The DANDI asset id, or the URL of a file elsewhere; exactly one.
    cache : path or None
        The cache the session was loaded into; None for the default.

    Raises
    ------
    InputError
        If the location, checksum, frame times or calibration fields are
        inconsistent; the message names each.
    """

    dataset: str
    external_file: str
    sha256: str
    size_bytes: int
    frame_times: NDArray[np.float64]
    frame_times_basis: Conversion
    calibration: VideoCalibration | None
    calibration_basis: Conversion | None
    calibration_tried: tuple[str, ...] = ()
    asset_id: str | None = None
    url: str | None = None
    cache: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        problems = []
        if (self.asset_id is None) == (self.url is None):
            problems.append("give exactly one of asset_id and url")
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            problems.append(f"sha256 {self.sha256!r} is not 64 hexadecimal digits")
        if self.size_bytes < 0:
            problems.append(f"size_bytes {self.size_bytes} is negative")
        times = read_only_copy(self.frame_times, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0):
            problems.append(
                "frame_times: not a 1-D, strictly increasing array of at least "
                "two times"
            )
        if (self.calibration is None) != (self.calibration_basis is None):
            problems.append("give calibration and calibration_basis together")
        if self.calibration is None and not self.calibration_tried:
            problems.append("without a calibration, say which sources were tried")
        if problems:
            raise InputError(
                _format_error(
                    f"VideoSource {self.external_file!r} cannot be built:\n- "
                    + "\n- ".join(problems),
                    why="the video is fetched, timed and placed from these fields.",
                    fix="correct the loader so each field listed meets the "
                    "contract in the VideoSource docstring.",
                )
            )
        object.__setattr__(self, "frame_times", times)

    @property
    def download_url(self) -> str:
        """The URL the video is fetched from."""
        if self.asset_id is not None:
            from neurospatial.datasets._fetch import dandi_asset_url

            return dandi_asset_url(self.asset_id)
        assert self.url is not None  # checked at construction
        return self.url

    def fetch(self, *, cache: str | os.PathLike[str] | None = None) -> Path:
        """Download the video once, verified against its SHA-256.

        Parameters
        ----------
        cache : path, optional
            The cache directory. Default: the one the session was loaded
            into (``$NEUROSPATIAL_DATA_DIR``, else
            ``~/.cache/neurospatial/datasets``).

        Returns
        -------
        Path
            ``<cache>/<dataset>/<external_file>``, beside the session's NWB
            file, so the file's own relative path finds it.

        Raises
        ------
        ValueError
            If the downloaded file's SHA-256 differs from the pinned one.
        """
        from neurospatial.datasets._fetch import fetch

        relative = Path(self.external_file)
        return fetch(
            self.download_url,
            relative.name,
            sha256=self.sha256,
            subdir=str(Path(self.dataset) / relative.parent),
            cache=Path(cache) if cache is not None else self.cache,
        )


def video_frame_times(
    video_times: NDArray[np.float64], position_times: NDArray[np.float64]
) -> tuple[NDArray[np.float64], Conversion, str | None]:
    """Each video frame's time, and where it comes from.

    When the video holds as many frames as the position series has samples,
    and the position times are an offset and rate of the video times to
    within half a frame interval, the two are the same frames on two clocks.
    The position clock is then used, since the spikes are on it, as an
    inferred conversion; where the clocks differ by more than half a frame,
    a watch-out gives the offset. Otherwise the video's own times are used.

    Parameters
    ----------
    video_times : ndarray, shape (n_frames,)
        The video's timestamps as the file stores them, seconds.
    position_times : ndarray, shape (n_time,)
        The position samples' times, seconds.

    Returns
    -------
    frame_times : ndarray, shape (n_frames,)
    basis : Conversion
    watch_out : str or None
        The clock offset, where the video's own times are off by more than
        half a frame; None otherwise.
    """
    video = np.asarray(video_times, dtype=np.float64)
    position = np.asarray(position_times, dtype=np.float64)
    interval = float(np.median(np.diff(video)))
    own = Conversion(
        "video frame times: the file's video timestamps, as stored",
        "stated",
        f"the file's video series: {len(video)} frames from {video[0]:.3f} to "
        f"{video[-1]:.3f} s; the {len(position)} position samples are not the "
        "same frames"
        + (
            " (another count)"
            if len(video) != len(position)
            else " (their times are no offset and rate of the video's)"
        ),
    )
    if len(video) != len(position):
        return video, own, None
    design = np.column_stack([np.ones_like(video), video - video[0]])
    (start, rate), *_ = np.linalg.lstsq(design, position, rcond=None)
    residual = float(np.max(np.abs(position - design @ np.array([start, rate]))))
    if residual > SAME_FRAMES_TOLERANCE * interval:
        return video, own, None
    offset = position - video
    if float(np.max(np.abs(offset))) <= SAME_FRAMES_TOLERANCE * interval:
        same = Conversion(
            "video frame times: the file's video timestamps, which are the "
            "position timestamps",
            "stated",
            f"the file's video series: {len(video)} frames, each within half a "
            "frame of a position sample",
        )
        return video, same, None
    low, high = float(offset.min()), float(offset.max())
    spread = f"{low:+.2f} s" if high - low < 0.005 else f"{low:+.2f} to {high:+.2f} s"
    basis = Conversion(
        "video frame times: the position timestamps, not the file's video timestamps",
        "inferred",
        f"the video and the position series both hold {len(video)} frames, and "
        f"the position times are the video times plus {start - video[0]:+.4f} s "
        f"at {rate:.6f} s per video second, to within {residual * 1e3:.3f} ms "
        f"(half a frame is {interval * 500:.1f} ms): the same frames on two "
        f"clocks, position minus video time {spread}. The recording's times, "
        "which spikes and events are checked against, are the position times",
    )
    watch_out = (
        f"The file's video timestamps are off the position clock: the video and "
        f"the position series hold the same {len(video)} frames, and position "
        f"minus video time is {spread}. The overlay shows each frame at its "
        "position sample's time; reading the video by its own timestamps puts "
        "the camera that far out of step with the positions and spikes."
    )
    return position, basis, watch_out


def video_calibration(
    *,
    bin_size: float,
    stated: tuple[AffineND, tuple[int, int], str] | None = None,
    tracking: tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, int], str]
    | None = None,
    landmarks: Landmarks | None = None,
    led_fit: LedFit | None = None,
    absent: Sequence[str] = ("", "", "", ""),
) -> tuple[VideoCalibration | None, Conversion | None, tuple[str, ...]]:
    """The video's pixel-to-cm calibration from the first source that gives one.

    The sources, in order (``CALIBRATION_SOURCES``):

    1. ``stated``: a pixel-to-cm transform the file states, used as stated.
    2. ``tracking``: pixel tracking the file stores beside its converted
       positions; a least-squares affine from one to the other.
    3. ``landmarks``: points placed by hand on one frame, from the registry;
       a similarity transform (scale, rotation, translation) through them.
    4. ``led_fit``: an affine fitted once from the LED detected in the video
       to the positions, from the registry (:class:`LedFit`).

    A fit from the file's pixel tracking or landmarks whose root-mean-square
    residual exceeds ``bin_size`` is refused, as is an LED fit whose median
    residual does (detection outliers, such as reflections, would dominate
    a mean), and the next source is tried.

    Parameters
    ----------
    bin_size : float
        The environment's bin, cm: the largest residual accepted.
    stated : (AffineND, (width, height), str), optional
        The file's own pixel-to-cm transform, its frame size and where it is
        stated.
    tracking : (pixels, positions, (width, height), str), optional
        ``pixels`` ``(n_time, 2)`` in video pixels (column, row) and
        ``positions`` ``(n_time, 2)`` in cm, the same samples; the frame size
        and which series they are.
    landmarks : Landmarks, optional
    led_fit : LedFit, optional
    absent : (str, str, str, str)
        Why each source the dataset has none of is absent, for example
        ``"the file's video series states no pixel size"``; empty strings
        use "none given".

    Returns
    -------
    calibration : VideoCalibration or None
    basis : Conversion or None
        Where the calibration comes from, with its residual in px and cm.
    tried : tuple of str
        Each source tried before the one used, with why it gave none: every
        source, when there is no calibration.
    """
    from neurospatial.ops.transforms import (
        AffineND,
        VideoCalibration,
        calibrate_from_landmarks,
    )

    tried: list[str] = []

    def gave_none(index: int, reason: str) -> None:
        tried.append(f"{CALIBRATION_SOURCES[index]}: {reason or 'none given'}")

    if stated is not None:
        transform, size, source = stated
        basis = Conversion(
            "video calibration: the file's pixel-to-cm transform",
            "stated",
            f"{source}; {_scale_text(transform)}",
        )
        return VideoCalibration(transform, frame_size_px=size), basis, ()
    gave_none(0, absent[0])

    if tracking is not None:
        pixels, positions, size, source = tracking
        transform, in_cm, in_px = fit_pixel_affine(pixels, positions)
        cm = float(np.sqrt(np.mean(in_cm**2)))
        px = float(np.sqrt(np.mean(in_px**2)))
        evidence = (
            f"{source}: a least-squares affine over the {in_cm.size} samples "
            f"tracked in both, residual {cm:.3g} cm ({px:.3g} px) root mean "
            f"square; {_scale_text(transform)}"
        )
        if cm <= bin_size:
            basis = Conversion(
                "video calibration: fitted from the file's pixel tracking to "
                "its positions",
                "inferred",
                evidence,
            )
            return VideoCalibration(transform, frame_size_px=size), basis, tuple(tried)
        gave_none(1, f"{evidence}, more than one {bin_size:g} cm bin")
    else:
        gave_none(1, absent[1])

    if landmarks is not None:
        transform = calibrate_from_landmarks(
            landmarks.px,
            landmarks.env,
            landmarks.frame_size_px,
            kind="similarity",
        )
        cm, px = _residuals(transform, landmarks.px, landmarks.env)
        evidence = (
            f"{len(landmarks.px)} landmarks placed by {landmarks.placed_by} on "
            f"frame {landmarks.frame}, residual {cm:.3g} cm ({px:.3g} px) root "
            f"mean square; {_scale_text(transform)}"
        )
        if cm <= bin_size:
            basis = Conversion(
                "video calibration: a similarity transform through landmarks "
                "placed by hand",
                "inferred",
                evidence,
            )
            calibration = VideoCalibration(
                transform, frame_size_px=landmarks.frame_size_px
            )
            return calibration, basis, tuple(tried)
        gave_none(2, f"{evidence}, more than one {bin_size:g} cm bin")
    else:
        gave_none(2, absent[2])

    if led_fit is not None:
        transform = AffineND(led_fit.matrix)
        evidence = (
            f"{led_fit.method}, over {led_fit.n_frames} frames ({led_fit.frames}); "
            f"residual median {led_fit.median_cm:.3g} cm ({led_fit.median_px:.3g} "
            f"px), 90th percentile {led_fit.p90_cm:.3g} cm ({led_fit.p90_px:.3g} "
            f"px); the median is the statistic compared with the {bin_size:g} cm "
            f"bin; {_scale_text(transform)}; fitted by {led_fit.fitted_by}"
        )
        if led_fit.median_cm <= bin_size:
            basis = Conversion(
                "video calibration: a least-squares affine from the LED detected "
                "in the video to the tracked positions",
                "inferred",
                evidence,
            )
            calibration = VideoCalibration(
                transform, frame_size_px=led_fit.frame_size_px
            )
            return calibration, basis, tuple(tried)
        gave_none(3, f"{evidence}: the median is more than one {bin_size:g} cm bin")
    else:
        gave_none(3, absent[3])
    return None, None, tuple(tried)


def fit_pixel_affine(
    pixels: NDArray[np.float64], positions: NDArray[np.float64]
) -> tuple[AffineND, NDArray[np.float64], NDArray[np.float64]]:
    """A least-squares affine from pixels to cm over the samples finite in both.

    Parameters
    ----------
    pixels : ndarray, shape (n_samples, 2)
        Video pixels, ``(column, row)``.
    positions : ndarray, shape (n_samples, 2)
        The same samples' positions, cm.

    Returns
    -------
    transform : AffineND
        Pixels to cm.
    residual_cm : ndarray, shape (n_used,)
        Each used sample's distance from its fitted position, cm.
    residual_px : ndarray, shape (n_used,)
        Each used sample's position mapped back into the frame, its distance
        from the pixel, px.
    """
    from neurospatial.ops.transforms import AffineND

    pixels = np.asarray(pixels, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    both = np.all(np.isfinite(pixels), axis=1) & np.all(np.isfinite(positions), axis=1)
    design = np.column_stack([pixels[both], np.ones(both.sum())])
    coef, *_ = np.linalg.lstsq(design, positions[both], rcond=None)
    transform = AffineND(np.vstack([coef.T, [0.0, 0.0, 1.0]]))
    residual_cm = np.linalg.norm(transform(pixels[both]) - positions[both], axis=1)
    back = transform.inverse()(positions[both])
    residual_px = np.linalg.norm(back - pixels[both], axis=1)
    return transform, residual_cm, residual_px


def led_fit_watch_out(fit: LedFit) -> str:
    """What a video calibrated by :class:`LedFit` can and cannot show."""
    return (
        "The video's calibration was fitted to the tracked positions (the LED "
        "detected in the video against the file's positions), so the overlay "
        "shows the camera and the data aligned by construction: it does not "
        "check that the tracking or the declared track is placed correctly, "
        "and the track itself is not visible in the video. 90th-percentile "
        f"residual {fit.p90_cm:.3g} cm ({fit.p90_px:.3g} px)."
    )


def _residuals(
    transform: AffineND, px: NDArray[np.float64], cm: NDArray[np.float64]
) -> tuple[float, float]:
    """Root-mean-square residual of ``transform`` in cm and, inverted, in px."""
    in_cm = float(np.sqrt(np.mean(np.sum((transform(px) - cm) ** 2, axis=1))))
    back = transform.inverse()(cm)
    in_px = float(np.sqrt(np.mean(np.sum((back - px) ** 2, axis=1))))
    return in_cm, in_px


def _scale_text(transform: Any) -> str:
    """The cm one step of a pixel column and of a pixel row moves."""
    linear = np.asarray(transform.A, dtype=np.float64)[:2, :2]
    column, row = np.linalg.norm(linear, axis=0)
    return f"{column:.4g} cm per pixel column and {row:.4g} cm per pixel row"
