"""Tests for _VideoReader class.

This module tests the _VideoReader class which provides lazy-loading video
frame access with LRU caching for efficient memory usage.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurospatial._video_io import _VideoReader

# Check for optional dependencies
try:
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Skip entire module if cv2 is not installed
pytestmark = pytest.mark.skipif(not HAS_CV2, reason="opencv-python not installed")


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
    """Test basic _VideoReader functionality."""

    def test_reader_loads_metadata(self, tiny_video_path: Path):
        """Test that _VideoReader loads video metadata correctly."""
        reader = _VideoReader(tiny_video_path)

        assert reader.n_frames == 5
        assert reader.fps == pytest.approx(10.0, rel=0.1)
        assert reader.frame_size_px == (8, 8)  # (width, height)
        assert reader.original_size_px == (8, 8)
        assert reader.duration == pytest.approx(0.5, rel=0.1)  # 5 frames / 10 fps

    def test_reader_lazy_loading(self, tiny_video_path: Path):
        """Test that _VideoReader does not load frames until accessed."""
        reader = _VideoReader(tiny_video_path)

        # Check cache is empty initially
        assert reader.cache_info().currsize == 0

        # Access a frame
        frame = reader[0]
        assert frame is not None

        # Cache should now have one entry
        assert reader.cache_info().currsize == 1

    def test_reader_getitem_returns_rgb(self, tiny_video_path: Path):
        """Test that __getitem__ returns RGB frames (not BGR)."""
        reader = _VideoReader(tiny_video_path)

        frame = reader[0]

        assert frame.shape == (8, 8, 3)
        assert frame.dtype == np.uint8
        # First frame should be all zeros (black)
        assert frame[0, 0, 0] == 0

    def test_reader_getitem_different_frames(self, tiny_video_path: Path):
        """Test that different frame indices return different content."""
        reader = _VideoReader(tiny_video_path)

        frame0 = reader[0]
        frame2 = reader[2]

        # Frames should have different brightness
        assert frame0[0, 0, 0] != frame2[0, 0, 0]
        # Frame 2 should have brightness ~100
        assert frame2[0, 0, 0] == pytest.approx(100, abs=5)

    def test_reader_invalid_index_raises(self, tiny_video_path: Path):
        """Test that invalid frame indices raise IndexError."""
        reader = _VideoReader(tiny_video_path)

        with pytest.raises(IndexError):
            reader[10]  # Only 5 frames

        with pytest.raises(IndexError):
            reader[-1]  # Negative indices not supported


class TestVideoReaderCaching:
    """Test LRU caching functionality."""

    def test_reader_lru_cache(self, tiny_video_path: Path):
        """Test that LRU cache works correctly."""
        reader = _VideoReader(tiny_video_path, cache_size=3)

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
        reader = _VideoReader(tiny_video_path, cache_size=2)

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
        """Test that _VideoReader can be pickled and unpickled."""
        reader = _VideoReader(tiny_video_path, cache_size=50)

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
        reader = _VideoReader(
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
        reader = _VideoReader(tiny_video_path)

        timestamps = reader.get_timestamps()

        assert len(timestamps) == 5
        assert timestamps.dtype == np.float64

    def test_reader_timestamps_monotonic(self, tiny_video_path: Path):
        """Test that timestamps are monotonically increasing."""
        reader = _VideoReader(tiny_video_path)

        timestamps = reader.get_timestamps()

        # Check monotonicity
        assert np.all(np.diff(timestamps) > 0)

    def test_reader_timestamps_spacing(self, tiny_video_path: Path):
        """Test that timestamps have correct spacing based on fps."""
        reader = _VideoReader(tiny_video_path)

        timestamps = reader.get_timestamps()

        # At 10 fps, frames should be 0.1s apart
        expected_spacing = 1.0 / 10.0
        actual_spacing = np.diff(timestamps)
        assert np.allclose(actual_spacing, expected_spacing, rtol=0.1)


class TestVideoReaderCropDownsample:
    """Test crop and downsample functionality."""

    def test_reader_downsample(self, larger_video_path: Path):
        """Test that downsample reduces frame size."""
        reader = _VideoReader(larger_video_path, downsample=2)

        assert reader.original_size_px == (32, 32)
        assert reader.frame_size_px == (16, 16)

        frame = reader[0]
        assert frame.shape == (16, 16, 3)

    def test_reader_crop(self, larger_video_path: Path):
        """Test that crop extracts correct region."""
        reader = _VideoReader(
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
        reader = _VideoReader(
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
            _VideoReader(tmp_path / "nonexistent.mp4")

    def test_reader_invalid_crop_raises(self, larger_video_path: Path):
        """Test that invalid crop raises ValueError."""
        # Crop extends beyond frame bounds
        with pytest.raises(ValueError) as exc_info:
            _VideoReader(larger_video_path, crop=(0, 0, 100, 100))

        error_msg = str(exc_info.value)
        assert "crop" in error_msg.lower()

    def test_reader_invalid_downsample_raises(self, larger_video_path: Path):
        """Test that invalid downsample raises ValueError."""
        with pytest.raises(ValueError):
            _VideoReader(larger_video_path, downsample=0)

        with pytest.raises(ValueError):
            _VideoReader(larger_video_path, downsample=-1)


def test_reader_pickle_starts_fresh_process_without_animation(
    larger_video_path, tmp_path
):
    """A worker restores settings/cache and reads frames without importing animation."""
    with _VideoReader(
        larger_video_path,
        cache_size=7,
        downsample=2,
        crop=(2, 4, 20, 16),
        prefetch_ahead=2,
    ) as reader:
        expected = reader[3].copy()
        assert reader.cache_info().currsize > 0
        payload = tmp_path / "reader.pkl"
        payload.write_bytes(pickle.dumps(reader))
    expected_path = tmp_path / "frame.npy"
    np.save(expected_path, expected)
    script = """
import importlib.abc
import pickle
import sys
import numpy as np

class BlockAnimation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "neurospatial.animation" or fullname.startswith("neurospatial.animation."):
            raise AssertionError("A video reader worker imported animation")

sys.meta_path.insert(0, BlockAnimation())
with open(sys.argv[1], "rb") as stream:
    reader = pickle.load(stream)
with reader:
    assert reader.frame_size_px == (10, 8)
    assert reader.crop_offset_px == (2, 4)
    assert reader.cache_info().maxsize == 7
    assert reader.cache_info().currsize == 0
    assert reader.prefetch_ahead == 2
    assert reader._prefetch_executor is not None
    np.testing.assert_array_equal(reader[3], np.load(sys.argv[2]))
assert reader._prefetch_executor is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(payload), str(expected_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
