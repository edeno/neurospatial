"""Video I/O utilities for neurospatial animations.

This module provides the VideoReader class for lazy-loading video frames
with LRU caching, designed for efficient memory usage during animation
rendering.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        return self._cached_read_frame(frame_idx)

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

    def __reduce__(self) -> tuple:
        """Support pickle serialization.

        Cache is dropped during pickling for safety in parallel workflows.

        Returns
        -------
        tuple
            Pickle-compatible representation.
        """
        return (
            self.__class__,
            (self._path, self._cache_size, self._downsample, self._crop),
        )
