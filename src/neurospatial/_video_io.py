"""Shared video I/O utilities for annotation and animation.

This module provides the _VideoReader class for lazy-loading video frames
with LRU caching, designed for efficient memory usage during animation
rendering.
"""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._validation import is_integer_at_least

# A forward jump of at most this many frames is decoded through instead of
# seeking: a seek restarts decoding at the previous keyframe.
_MAX_FRAMES_GRABBED = 16


@runtime_checkable
class VideoReaderProtocol(Protocol):
    """Protocol defining the interface for lazy video frame readers.

    :class:`_VideoReader` conforms. A pre-loaded frame array does not (it
    has no ``n_frames`` or ``frame_size_px``), so ``VideoData.reader`` is
    typed as an ndarray or a reader. Classes implementing this protocol
    must provide frame count, frame dimensions, and indexed access.

    Attributes
    ----------
    n_frames : int
        Total number of frames in the video.
    frame_size_px : tuple[int, int]
        Frame dimensions as (width, height) in pixels.

    Methods
    -------
    __getitem__(idx: int) -> NDArray[np.uint8]
        Get a frame by index, returning RGB uint8 array of shape (H, W, 3).

    Examples
    --------
    >>> def render_video(reader: VideoReaderProtocol) -> None:  # doctest: +SKIP
    ...     for i in range(reader.n_frames):
    ...         frame = reader[i]
    ...         # Process frame...

    Notes
    -----
    The protocol is runtime-checkable, allowing isinstance() checks:

    >>> isinstance(
    ...     my_reader, VideoReaderProtocol
    ... )  # True if implements protocol  # doctest: +SKIP
    """

    @property
    def n_frames(self) -> int:
        """Total number of frames in the video."""
        ...

    @property
    def frame_size_px(self) -> tuple[int, int]:
        """Frame dimensions as (width, height) in pixels."""
        ...

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        """Get frame by index.

        Parameters
        ----------
        idx : int
            Frame index (0-based).

        Returns
        -------
        NDArray[np.uint8]
            RGB frame array of shape (height, width, 3).
        """
        ...


def _validate_frame_processing(
    frame_size_px: tuple[int, int],
    crop: tuple[int, int, int, int] | None,
    downsample: int,
) -> tuple[int, int]:
    """Refuse a crop or downsample factor that does not fit the video frame.

    Parameters
    ----------
    frame_size_px : tuple of (width, height)
        Size of the decoded video frame in pixels.
    crop : tuple of (x, y, width, height) or None
        Region cut from each frame before downsampling, in pixels.
    downsample : int
        Factor by which the (cropped) frame is shrunk.

    Returns
    -------
    tuple of (width, height)
        Size of the processed frame in pixels, after crop and downsample.

    Raises
    ------
    ValueError
        If ``downsample`` is not a positive integer, ``crop`` is not four
        whole-pixel values with a non-negative corner and positive size
        inside the frame, or the processed frame has no pixels.
    """
    width, height = frame_size_px
    if not is_integer_at_least(downsample, 1):
        raise ValueError(
            _format_error(
                f"downsample must be a positive integer; got {downsample!r}.",
                why="the frame is shrunk by this whole-number factor in each direction.",
                fix="pass downsample=1 to keep full resolution, or 2 to halve it.",
            )
        )
    if crop is not None:
        values = tuple(crop) if np.iterable(crop) else ()
        if (
            len(values) != 4
            or not all(is_integer_at_least(value, 0) for value in values[:2])
            or not all(is_integer_at_least(value, 1) for value in values[2:])
            or values[0] + values[2] > width
            or values[1] + values[3] > height
        ):
            raise ValueError(
                _format_error(
                    "crop must be (x, y, width, height) in whole pixels, with a "
                    "non-negative corner and positive size inside the "
                    f"{width}x{height} video frame; got {crop!r}.",
                    why="the crop is cut from each decoded frame before "
                    "downsampling, measured from the top-left pixel.",
                    fix=f"pass four integers that fit the frame, e.g. "
                    f"crop=(0, 0, {width}, {height}) for the whole frame.",
                )
            )
        width, height = values[2], values[3]
    if width // downsample < 1 or height // downsample < 1:
        raise ValueError(
            _format_error(
                "downsample leaves no video pixels.",
                fix="use a smaller downsample factor or a larger crop.",
            )
        )
    return width // downsample, height // downsample


class CacheInfo(NamedTuple):
    """Cache statistics for _VideoReader.

    Attributes
    ----------
    hits : int
        Number of cache hits.
    misses : int
        Number of cache misses.
    maxsize : int
        Maximum cache size.
    currsize : int
        Current number of cached items.
    """

    hits: int
    misses: int
    maxsize: int
    currsize: int


class _VideoReader:
    """Lazy-loading video reader with LRU caching.

    Provides efficient, memory-conscious access to video frames through
    lazy loading and LRU caching. Frames are only read from disk when
    accessed and recently-used frames are cached to avoid repeated I/O.

    Parameters
    ----------
    path : str or Path
        Path to video file. Must exist and be readable by OpenCV.
    cache_size : int, default=100
        Maximum number of frames to keep in LRU cache.
    downsample : int, default=1
        Downsample factor for frames. Must be positive integer.
        A value of 2 halves the frame dimensions.
    crop : tuple of (x, y, width, height), optional
        Region to crop from each frame in pixel coordinates.
        Applied before downsampling.
    prefetch_ahead : int, default=0
        Number of frames to prefetch in background when a frame is accessed.
        Set to 0 (default) to disable prefetching. A value of 5 will load
        frames [current+1, current+5] in a background thread after each access.
        This can improve playback smoothness by hiding disk I/O latency.

    Attributes
    ----------
    n_frames : int
        Total number of frames in video.
    frame_size_px : tuple of (width, height)
        Size of returned frames after crop and downsample.
    original_size_px : tuple of (width, height)
        Original video frame size before any processing.
    prefetch_ahead : int
        Number of frames to prefetch ahead.

    Notes
    -----
    - Frames are returned in RGB format (OpenCV reads as BGR internally).
    - One OpenCV capture serves all reads: frames read in order need no
      seek. Pickling keeps the settings and drops the capture, cache and
      prefetch thread, so a reader can be sent to parallel rendering workers.

    Examples
    --------
    >>> reader = _VideoReader("video.mp4", cache_size=50)  # doctest: +SKIP
    >>> frame = reader[0]  # Load first frame  # doctest: +SKIP
    >>> print(frame.shape)  # (height, width, 3)  # doctest: +SKIP

    >>> # With crop and downsample
    >>> reader = _VideoReader(  # doctest: +SKIP
    ...     "video.mp4",
    ...     crop=(100, 100, 200, 200),  # x, y, width, height
    ...     downsample=2,
    ... )
    >>> frame = reader[0]  # doctest: +SKIP
    >>> print(frame.shape)  # (100, 100, 3)  # doctest: +SKIP
    """

    def __init__(
        self,
        path: str | Path,
        cache_size: int = 100,
        downsample: int = 1,
        crop: tuple[int, int, int, int] | None = None,
        prefetch_ahead: int = 0,
    ) -> None:
        import cv2

        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._downsample = downsample
        self._cache_size = cache_size

        if prefetch_ahead < 0:
            raise ValueError(
                f"prefetch_ahead must be non-negative, got {prefetch_ahead}"
            )
        self._prefetch_ahead = prefetch_ahead

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {self._path}")
        self._n_frames: int = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._original_size_px: tuple[int, int] = (width, height)
        # One capture serves every read; ``_next_frame`` is the frame its
        # next ``read()`` returns, so in-order reads need no seek.
        self._capture: cv2.VideoCapture | None = capture
        self._next_frame = 0
        try:
            self._frame_size_px = _validate_frame_processing(
                (width, height), crop, downsample
            )
        except ValueError:
            self._release_capture()
            raise
        self._crop = crop

        # The lock guards the capture and the cache, so a frame wanted by
        # the caller and by the prefetch thread is decoded only once.
        self._lock = threading.Lock()
        self._cache: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._hits = 0
        self._misses = 0

        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_task: Future[None] | None = None
        if self._prefetch_ahead > 0:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _read_frame_uncached(self, frame_idx: int) -> NDArray[np.uint8]:
        """Decode one frame from disk; the caller holds ``self._lock``.

        Parameters
        ----------
        frame_idx : int
            Frame index to read.

        Returns
        -------
        NDArray[np.uint8]
            Read-only RGB frame array of shape (height, width, 3).
        """
        import cv2

        if self._capture is None:
            self._capture = cv2.VideoCapture(str(self._path))
            self._next_frame = 0
        capture = self._capture
        skip = frame_idx - self._next_frame
        if 0 <= skip <= _MAX_FRAMES_GRABBED:
            for _ in range(skip):
                capture.grab()
        else:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret:
            # The capture's position is unknown after a failed read.
            self._release_capture()
            raise OSError(
                _format_error(
                    f"Could not decode frame {frame_idx} of video '{self._path}'.",
                    why=(
                        "OpenCV read no image, although the video "
                        f"reports {self._n_frames} frames. "
                        "The frame may be corrupt, this OpenCV build may "
                        "lack the codec, or the container may overstate "
                        "its frame count (common for variable-frame-rate "
                        "MP4 files)."
                    ),
                    fix=(
                        "check that the video plays to this frame in a "
                        "media player, or re-encode it at a constant frame "
                        "rate, e.g. ffmpeg -i in.mp4 -c:v libx264 "
                        "-fps_mode cfr out.mp4, then reopen it."
                    ),
                )
            )
        self._next_frame = frame_idx + 1

        if self._crop is not None:
            x, y, w, h = self._crop
            frame = frame[y : y + h, x : x + w]
        if self._downsample > 1:
            new_h = frame.shape[0] // self._downsample
            new_w = frame.shape[1] // self._downsample
            frame = cv2.resize(frame, (new_w, new_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Cast for mypy (cv2 already returns uint8)
        result: NDArray[np.uint8] = frame.astype(np.uint8, copy=False)
        # The cache hands this array to every later reader of the frame.
        result.setflags(write=False)
        return result

    def _cached_read_frame(self, frame_idx: int) -> NDArray[np.uint8]:
        """Return a frame from the LRU cache, decoding it on a miss."""
        with self._lock:
            frame = self._cache.get(frame_idx)
            if frame is not None:
                self._hits += 1
                self._cache.move_to_end(frame_idx)
                return frame
            self._misses += 1
            frame = self._read_frame_uncached(frame_idx)
            if self._cache_size > 0:
                self._cache[frame_idx] = frame
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return frame

    def __getitem__(self, frame_idx: int) -> NDArray[np.uint8]:
        """Get a video frame by index.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based). Negative indices not supported.

        Returns
        -------
        NDArray[np.uint8]
            Read-only RGB frame array of shape (height, width, 3), shared
            with the cache; copy it before editing.

        Raises
        ------
        IndexError
            If frame_idx is out of range or negative.
        OSError
            If OpenCV cannot decode a frame inside the reported range.
        """
        if frame_idx < 0 or frame_idx >= self._n_frames:
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {self._n_frames})"
            )
        executor = self._prefetch_executor
        if executor is not None:
            self._trigger_prefetch(executor, frame_idx)
        return self._cached_read_frame(frame_idx)

    def _trigger_prefetch(
        self, executor: ThreadPoolExecutor, current_frame: int
    ) -> None:
        """Load the frames after ``current_frame`` in the background.

        At most one prefetch task is pending: while one runs, later
        accesses do not queue more, so the thread does not fall behind
        playback.

        Parameters
        ----------
        executor : ThreadPoolExecutor
            The single-thread prefetch executor.
        current_frame : int
            The frame that was just accessed.
        """
        if self._prefetch_task is not None and not self._prefetch_task.done():
            return
        start_frame = current_frame + 1
        end_frame = min(current_frame + self._prefetch_ahead + 1, self._n_frames)
        if start_frame >= end_frame:
            return

        def prefetch_frames() -> None:
            for idx in range(start_frame, end_frame):
                # A frame that fails here raises again when it is accessed.
                with contextlib.suppress(Exception):
                    self._cached_read_frame(idx)

        # The executor may already be shut down.
        with contextlib.suppress(RuntimeError):
            self._prefetch_task = executor.submit(prefetch_frames)

    def cache_info(self) -> CacheInfo:
        """Return cache statistics.

        Returns
        -------
        CacheInfo
            Named tuple with hits, misses, maxsize, currsize.
        """
        with self._lock:
            return CacheInfo(
                hits=self._hits,
                misses=self._misses,
                maxsize=self._cache_size,
                currsize=len(self._cache),
            )

    @property
    def n_frames(self) -> int:
        """Total number of frames in video."""
        return self._n_frames

    @property
    def frame_size_px(self) -> tuple[int, int]:
        """Frame size (width, height) after crop and downsample."""
        return self._frame_size_px

    @property
    def original_size_px(self) -> tuple[int, int]:
        """Original frame size (width, height) before processing."""
        return self._original_size_px

    @property
    def prefetch_ahead(self) -> int:
        """Number of frames to prefetch ahead during sequential access."""
        return self._prefetch_ahead

    def __reduce__(self) -> tuple:
        """Support pickle serialization.

        The capture, cache and prefetch executor are dropped; the restored
        reader opens its own.

        Returns
        -------
        tuple
            Pickle-compatible representation.
        """
        return (
            self.__class__,
            (
                self._path,
                self._cache_size,
                self._downsample,
                self._crop,
                self._prefetch_ahead,
            ),
        )

    def close(self) -> None:
        """Release resources held by the _VideoReader.

        Shuts down the prefetch thread and releases the video capture. Safe
        to call multiple times. After closing, the reader can still be used:
        it reopens the video on the next read, without prefetching.

        Examples
        --------
        >>> reader = _VideoReader("video.mp4")  # doctest: +SKIP
        >>> frame = reader[0]  # doctest: +SKIP
        >>> reader.close()  # Release resources  # doctest: +SKIP
        """
        executor = getattr(self, "_prefetch_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            self._prefetch_executor = None
        if hasattr(self, "_lock"):
            with self._lock:
                self._release_capture()

    def __enter__(self) -> _VideoReader:
        """Enter context manager.

        Returns
        -------
        _VideoReader
            Self for use in with statement.

        Examples
        --------
        >>> with _VideoReader("video.mp4") as reader:  # doctest: +SKIP
        ...     frame = reader[0]
        ...     # Resources automatically released on exit
        """
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object
    ) -> None:
        """Exit context manager, releasing resources.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception was raised.
        exc_val : Exception or None
            Exception instance if an exception was raised.
        exc_tb : object
            Traceback if an exception was raised.
        """
        self.close()

    def __del__(self) -> None:
        """Release the prefetch executor and capture on deletion."""
        self.close()
