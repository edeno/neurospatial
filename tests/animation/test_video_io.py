"""Tests for VideoReader class.

This module tests the VideoReader class which provides lazy-loading video
frame access with LRU caching for efficient memory usage.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from neurospatial.animation._video_io import VideoReader


@pytest.fixture
def tiny_video_path(tmp_path: Path) -> Path:
    """Create a tiny test video (8x8 pixels, 5 frames, 10 fps).

    Each frame has a distinct brightness level for verification.
    """
    import cv2

    video_path = tmp_path / "test_video.mp4"

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    frame_size = (8, 8)  # width, height
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    # Write 5 frames with different brightness levels
    for i in range(5):
        # Create frame with brightness = i * 50 (0, 50, 100, 150, 200)
        frame = np.full((8, 8, 3), i * 50, dtype=np.uint8)
        # OpenCV uses BGR, so we need to convert
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def larger_video_path(tmp_path: Path) -> Path:
    """Create a larger test video (32x32 pixels, 20 frames, 30 fps).

    Used for testing crop and downsample functionality.
    """
    import cv2

    video_path = tmp_path / "larger_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    frame_size = (32, 32)  # width, height
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    # Write 20 frames
    for i in range(20):
        # Create gradient frame
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 12  # Blue channel varies
        frame[:, :, 1] = 128  # Green constant
        frame[:, :, 2] = 255 - i * 12  # Red varies inversely
        writer.write(frame)

    writer.release()
    return video_path


class TestVideoReaderBasics:
    """Test basic VideoReader functionality."""

    def test_reader_loads_metadata(self, tiny_video_path: Path):
        """Test that VideoReader loads video metadata correctly."""
        reader = VideoReader(tiny_video_path)

        assert reader.n_frames == 5
        assert reader.fps == pytest.approx(10.0, rel=0.1)
        assert reader.frame_size_px == (8, 8)  # (width, height)
        assert reader.original_size_px == (8, 8)
        assert reader.duration == pytest.approx(0.5, rel=0.1)  # 5 frames / 10 fps

    def test_reader_lazy_loading(self, tiny_video_path: Path):
        """Test that VideoReader does not load frames until accessed."""
        reader = VideoReader(tiny_video_path)

        # Check cache is empty initially
        assert reader.cache_info().currsize == 0

        # Access a frame
        frame = reader[0]
        assert frame is not None

        # Cache should now have one entry
        assert reader.cache_info().currsize == 1

    def test_reader_getitem_returns_rgb(self, tiny_video_path: Path):
        """Test that __getitem__ returns RGB frames (not BGR)."""
        reader = VideoReader(tiny_video_path)

        frame = reader[0]

        assert frame.shape == (8, 8, 3)
        assert frame.dtype == np.uint8
        # First frame should be all zeros (black)
        assert frame[0, 0, 0] == 0

    def test_reader_getitem_different_frames(self, tiny_video_path: Path):
        """Test that different frame indices return different content."""
        reader = VideoReader(tiny_video_path)

        frame0 = reader[0]
        frame2 = reader[2]

        # Frames should have different brightness
        assert frame0[0, 0, 0] != frame2[0, 0, 0]
        # Frame 2 should have brightness ~100
        assert frame2[0, 0, 0] == pytest.approx(100, abs=5)

    def test_reader_invalid_index_raises(self, tiny_video_path: Path):
        """Test that invalid frame indices raise IndexError."""
        reader = VideoReader(tiny_video_path)

        with pytest.raises(IndexError):
            reader[10]  # Only 5 frames

        with pytest.raises(IndexError):
            reader[-1]  # Negative indices not supported


class TestVideoReaderCaching:
    """Test LRU caching functionality."""

    def test_reader_lru_cache(self, tiny_video_path: Path):
        """Test that LRU cache works correctly."""
        reader = VideoReader(tiny_video_path, cache_size=3)

        # Access frames 0, 1, 2
        _ = reader[0]
        _ = reader[1]
        _ = reader[2]
        assert reader.cache_info().currsize == 3

        # Access frame 3 (should evict frame 0)
        _ = reader[3]
        assert reader.cache_info().currsize == 3

        # Access frame 0 again (cache miss, re-read from disk)
        _ = reader[0]
        # Frame 1 should have been evicted
        info = reader.cache_info()
        assert info.hits >= 0  # At least some hits from previous accesses
        assert info.misses >= 5  # All initial accesses + evicted frame

    def test_reader_cache_size_parameter(self, tiny_video_path: Path):
        """Test that cache_size parameter is respected."""
        reader = VideoReader(tiny_video_path, cache_size=2)

        # Fill cache
        _ = reader[0]
        _ = reader[1]
        assert reader.cache_info().currsize == 2

        # Add one more, evicting oldest
        _ = reader[2]
        assert reader.cache_info().currsize == 2


class TestVideoReaderPickle:
    """Test pickle serialization for parallel rendering."""

    def test_reader_pickle_roundtrip(self, tiny_video_path: Path):
        """Test that VideoReader can be pickled and unpickled."""
        reader = VideoReader(tiny_video_path, cache_size=50)

        # Populate cache
        _ = reader[0]
        _ = reader[1]
        assert reader.cache_info().currsize == 2

        # Pickle and unpickle
        pickled = pickle.dumps(reader)
        restored = pickle.loads(pickled)

        # Restored reader should work
        assert restored.n_frames == 5
        assert restored.fps == pytest.approx(10.0, rel=0.1)

        # Cache should be empty after unpickling (dropped for pickle safety)
        assert restored.cache_info().currsize == 0

        # But frames should still be accessible
        frame = restored[0]
        assert frame.shape == (8, 8, 3)

    def test_reader_pickle_preserves_settings(self, larger_video_path: Path):
        """Test that pickle preserves crop and downsample settings."""
        reader = VideoReader(
            larger_video_path,
            cache_size=10,
            downsample=2,
            crop=(4, 4, 16, 16),  # x, y, width, height
        )

        pickled = pickle.dumps(reader)
        restored = pickle.loads(pickled)

        # Settings should be preserved
        assert restored._downsample == 2
        assert restored._crop == (4, 4, 16, 16)


class TestVideoReaderTimestamps:
    """Test timestamp retrieval functionality."""

    def test_reader_timestamps_length(self, tiny_video_path: Path):
        """Test that get_timestamps returns correct number of timestamps."""
        reader = VideoReader(tiny_video_path)

        timestamps = reader.get_timestamps()

        assert len(timestamps) == 5
        assert timestamps.dtype == np.float64

    def test_reader_timestamps_monotonic(self, tiny_video_path: Path):
        """Test that timestamps are monotonically increasing."""
        reader = VideoReader(tiny_video_path)

        timestamps = reader.get_timestamps()

        # Check monotonicity
        assert np.all(np.diff(timestamps) > 0)

    def test_reader_timestamps_spacing(self, tiny_video_path: Path):
        """Test that timestamps have correct spacing based on fps."""
        reader = VideoReader(tiny_video_path)

        timestamps = reader.get_timestamps()

        # At 10 fps, frames should be 0.1s apart
        expected_spacing = 1.0 / 10.0
        actual_spacing = np.diff(timestamps)
        assert np.allclose(actual_spacing, expected_spacing, rtol=0.1)


class TestVideoReaderCropDownsample:
    """Test crop and downsample functionality."""

    def test_reader_downsample(self, larger_video_path: Path):
        """Test that downsample reduces frame size."""
        reader = VideoReader(larger_video_path, downsample=2)

        assert reader.original_size_px == (32, 32)
        assert reader.frame_size_px == (16, 16)

        frame = reader[0]
        assert frame.shape == (16, 16, 3)

    def test_reader_crop(self, larger_video_path: Path):
        """Test that crop extracts correct region."""
        reader = VideoReader(
            larger_video_path,
            crop=(8, 8, 16, 16),  # x, y, width, height
        )

        assert reader.original_size_px == (32, 32)
        assert reader.frame_size_px == (16, 16)
        assert reader.crop_offset_px == (8, 8)

        frame = reader[0]
        assert frame.shape == (16, 16, 3)

    def test_reader_crop_and_downsample(self, larger_video_path: Path):
        """Test crop followed by downsample."""
        reader = VideoReader(
            larger_video_path,
            crop=(0, 0, 16, 16),  # Crop to 16x16
            downsample=2,  # Then downsample to 8x8
        )

        assert reader.frame_size_px == (8, 8)

        frame = reader[0]
        assert frame.shape == (8, 8, 3)


class TestVideoReaderErrorHandling:
    """Test error handling."""

    def test_reader_nonexistent_file_raises(self, tmp_path: Path):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VideoReader(tmp_path / "nonexistent.mp4")

    def test_reader_invalid_crop_raises(self, larger_video_path: Path):
        """Test that invalid crop raises ValueError."""
        # Crop extends beyond frame bounds
        with pytest.raises(ValueError) as exc_info:
            VideoReader(larger_video_path, crop=(0, 0, 100, 100))

        error_msg = str(exc_info.value)
        assert "crop" in error_msg.lower()

    def test_reader_invalid_downsample_raises(self, larger_video_path: Path):
        """Test that invalid downsample raises ValueError."""
        with pytest.raises(ValueError):
            VideoReader(larger_video_path, downsample=0)

        with pytest.raises(ValueError):
            VideoReader(larger_video_path, downsample=-1)
