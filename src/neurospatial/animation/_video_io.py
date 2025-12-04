"""Video I/O utilities for neurospatial animations.

This module provides the VideoReader class for lazy-loading video frames
with LRU caching, designed for efficient memory usage during animation
rendering.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class VideoReaderProtocol(Protocol):
    """Protocol defining the interface for video frame readers.

    This protocol enables type-safe handling of both VideoReader instances
    and pre-loaded numpy arrays in VideoData. Classes implementing this
    protocol must provide frame count, frame dimensions, and indexed access.

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
    >>> def render_video(reader: VideoReaderProtocol) -> None:
    ...     for i in range(reader.n_frames):
    ...         frame = reader[i]
    ...         # Process frame...

    Notes
    -----
    The protocol is runtime-checkable, allowing isinstance() checks:

    >>> isinstance(my_reader, VideoReaderProtocol)  # True if implements protocol
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


class CacheInfo(NamedTuple):
    """Cache statistics for VideoReader.

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


class VideoReader:
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
    fps : float
        Frames per second of video.
    frame_size_px : tuple of (width, height)
        Size of returned frames after crop and downsample.
    original_size_px : tuple of (width, height)
        Original video frame size before any processing.
    duration : float
        Total video duration in seconds.
    crop_offset_px : tuple of (x, y)
        Offset of crop region from top-left corner.
    prefetch_ahead : int
        Number of frames to prefetch ahead.

    Notes
    -----
    - Frames are returned in RGB format (OpenCV reads as BGR internally).
    - Cache is dropped during pickling to ensure pickle safety for
      parallel rendering workflows.
    - The reader is pickle-safe: settings are preserved but cache is reset.

    Examples
    --------
    >>> reader = VideoReader("video.mp4", cache_size=50)
    >>> frame = reader[0]  # Load first frame
    >>> print(frame.shape)  # (height, width, 3)

    >>> # With crop and downsample
    >>> reader = VideoReader(
    ...     "video.mp4",
    ...     crop=(100, 100, 200, 200),  # x, y, width, height
    ...     downsample=2,
    ... )
    >>> frame = reader[0]
    >>> print(frame.shape)  # (100, 100, 3)
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

        # Validate path
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")

        # Validate downsample
        if downsample < 1:
            raise ValueError(f"downsample must be a positive integer, got {downsample}")
        self._downsample = downsample
        self._cache_size = cache_size

        # Validate prefetch_ahead
        if prefetch_ahead < 0:
            raise ValueError(
                f"prefetch_ahead must be non-negative, got {prefetch_ahead}"
            )
        self._prefetch_ahead = prefetch_ahead

        # Open video to read metadata
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self._path}")

        try:
            self._n_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps: float = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._original_size_px: tuple[int, int] = (width, height)
        finally:
            cap.release()

        # Validate and store crop
        self._crop = crop
        if crop is not None:
            crop_x, crop_y, crop_w, crop_h = crop
            if (crop_x + crop_w > width) or (crop_y + crop_h > height):
                raise ValueError(
                    f"Crop region {crop} extends beyond frame bounds ({width}x{height})"
                )
            self._crop_offset_px = (crop_x, crop_y)
            effective_width = crop_w
            effective_height = crop_h
        else:
            self._crop_offset_px = (0, 0)
            effective_width = width
            effective_height = height

        # Calculate final frame size after downsample
        final_width = effective_width // downsample
        final_height = effective_height // downsample
        self._frame_size_px = (final_width, final_height)

        # Create the cached frame reader
        self._setup_cache()

        # Set up prefetching infrastructure
        self._prefetch_executor: ThreadPoolExecutor | None = None
        if self._prefetch_ahead > 0:
            # Create a single-thread executor for prefetching
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1)

    def _setup_cache(self) -> None:
        """Set up the LRU-cached frame reader."""

        # Create a new cached function bound to this instance
        @lru_cache(maxsize=self._cache_size)
        def _read_frame(frame_idx: int) -> NDArray[np.uint8]:
            return self._read_frame_uncached(frame_idx)

        self._cached_read_frame = _read_frame

    def _read_frame_uncached(self, frame_idx: int) -> NDArray[np.uint8]:
        """Read a single frame from disk without caching.

        Parameters
        ----------
        frame_idx : int
            Frame index to read.

        Returns
        -------
        NDArray[np.uint8]
            RGB frame array of shape (height, width, 3).
        """
        import cv2

        cap = cv2.VideoCapture(str(self._path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise IndexError(f"Could not read frame {frame_idx}")

            # Apply crop if specified
            if self._crop is not None:
                x, y, w, h = self._crop
                frame = frame[y : y + h, x : x + w]

            # Apply downsample if specified
            if self._downsample > 1:
                new_h = frame.shape[0] // self._downsample
                new_w = frame.shape[1] // self._downsample
                frame = cv2.resize(frame, (new_w, new_h))

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Cast for mypy (cv2 already returns uint8)
            result: NDArray[np.uint8] = frame.astype(np.uint8, copy=False)
            return result
        finally:
            cap.release()

    def __getitem__(self, frame_idx: int) -> NDArray[np.uint8]:
        """Get a video frame by index.

        Parameters
        ----------
        frame_idx : int
            Frame index (0-based). Negative indices not supported.

        Returns
        -------
        NDArray[np.uint8]
            RGB frame array of shape (height, width, 3).

        Raises
        ------
        IndexError
            If frame_idx is out of range or negative.
        """
        if frame_idx < 0 or frame_idx >= self._n_frames:
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {self._n_frames})"
            )

        # Trigger prefetching for upcoming frames
        if self._prefetch_ahead > 0 and self._prefetch_executor is not None:
            self._trigger_prefetch(frame_idx)

        return self._cached_read_frame(frame_idx)

    def _trigger_prefetch(self, current_frame: int) -> None:
        """Trigger prefetching of upcoming frames in background thread.

        Parameters
        ----------
        current_frame : int
            The frame that was just accessed.
        """
        if self._prefetch_executor is None:
            return

        # Calculate frames to prefetch (current+1 to current+prefetch_ahead)
        start_frame = current_frame + 1
        end_frame = min(current_frame + self._prefetch_ahead + 1, self._n_frames)

        if start_frame >= end_frame:
            return

        # Submit prefetch task (non-blocking)
        def prefetch_frames() -> None:
            """
            Prefetch upcoming frames into cache in background thread.

            Iterates through the specified frame range and loads each frame
            into the LRU cache. Errors are suppressed since they will be
            caught on actual frame access.
            """
            for idx in range(start_frame, end_frame):
                # Ignore errors in prefetching - they'll be caught on actual access
                with contextlib.suppress(Exception):
                    # This will cache the frame via lru_cache
                    self._cached_read_frame(idx)

        # Executor might be shut down
        with contextlib.suppress(RuntimeError):
            self._prefetch_executor.submit(prefetch_frames)

    def cache_info(self) -> CacheInfo:
        """Return cache statistics.

        Returns
        -------
        CacheInfo
            Named tuple with hits, misses, maxsize, currsize.
        """
        info = self._cached_read_frame.cache_info()
        return CacheInfo(
            hits=info.hits,
            misses=info.misses,
            maxsize=info.maxsize or 0,
            currsize=info.currsize,
        )

    def get_timestamps(self) -> NDArray[np.float64]:
        """Get timestamps for all frames.

        Returns
        -------
        NDArray[np.float64]
            Array of timestamps in seconds, shape (n_frames,).
        """
        timestamps: NDArray[np.float64] = (
            np.arange(self._n_frames, dtype=np.float64) / self._fps
        )
        return timestamps

    @property
    def n_frames(self) -> int:
        """Total number of frames in video."""
        return self._n_frames

    @property
    def fps(self) -> float:
        """Frames per second."""
        return self._fps

    @property
    def frame_size_px(self) -> tuple[int, int]:
        """Frame size (width, height) after crop and downsample."""
        return self._frame_size_px

    @property
    def original_size_px(self) -> tuple[int, int]:
        """Original frame size (width, height) before processing."""
        return self._original_size_px

    @property
    def crop_offset_px(self) -> tuple[int, int]:
        """Crop offset (x, y) from top-left corner."""
        return self._crop_offset_px

    @property
    def duration(self) -> float:
        """Total video duration in seconds."""
        return float(self._n_frames) / self._fps

    @property
    def prefetch_ahead(self) -> int:
        """Number of frames to prefetch ahead during sequential access."""
        return self._prefetch_ahead

    def __reduce__(self) -> tuple:
        """Support pickle serialization.

        Cache and prefetch executor are dropped during pickling for safety
        in parallel workflows.

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

    def __del__(self) -> None:
        """Clean up prefetch executor on deletion."""
        if hasattr(self, "_prefetch_executor") and self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=False)
            self._prefetch_executor = None
