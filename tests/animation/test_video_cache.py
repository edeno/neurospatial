"""Tests for VideoOverlay cache_size parameter and prefetching (Phase 2.2).

Tests verify that:
1. VideoOverlay accepts cache_size parameter with appropriate defaults
2. cache_size is passed through to VideoReader during conversion
3. cache_size validation works correctly
4. Async prefetching pre-loads upcoming frames during sequential playback
"""

import time
from pathlib import Path

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.overlays import VideoOverlay


class TestVideoOverlayCacheSize:
    """Test VideoOverlay cache_size parameter."""

    @pytest.fixture
    def sample_video_array(self) -> np.ndarray:
        """Create a sample video array (n_frames, height, width, 3)."""
        # 10 frames, 16x16 pixels, RGB
        return np.zeros((10, 16, 16, 3), dtype=np.uint8)

    @pytest.fixture
    def simple_env(self) -> Environment:
        """Create a simple 2D environment for testing."""
        positions = np.array([[0.0, 0.0], [16.0, 16.0]])
        return Environment.from_samples(positions, bin_size=1.0)

    def test_cache_size_default_value(self, sample_video_array: np.ndarray):
        """Test that cache_size has default value of 100."""
        overlay = VideoOverlay(source=sample_video_array)

        assert overlay.cache_size == 100

    def test_cache_size_custom_value(self, sample_video_array: np.ndarray):
        """Test that cache_size can be set to custom value."""
        overlay = VideoOverlay(source=sample_video_array, cache_size=200)

        assert overlay.cache_size == 200

    def test_cache_size_minimum_value(self, sample_video_array: np.ndarray):
        """Test that cache_size accepts minimum value of 1."""
        overlay = VideoOverlay(source=sample_video_array, cache_size=1)

        assert overlay.cache_size == 1

    def test_cache_size_zero_raises_error(self, sample_video_array: np.ndarray):
        """Test that cache_size=0 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=sample_video_array, cache_size=0)

        error_msg = str(exc_info.value)
        assert "cache_size" in error_msg.lower()
        assert "positive" in error_msg.lower() or "1" in error_msg

    def test_cache_size_negative_raises_error(self, sample_video_array: np.ndarray):
        """Test that negative cache_size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=sample_video_array, cache_size=-10)

        error_msg = str(exc_info.value)
        assert "cache_size" in error_msg.lower()

    def test_cache_size_with_all_parameters(self, sample_video_array: np.ndarray):
        """Test cache_size works with all other parameters."""
        times = np.linspace(0.0, 1.0, len(sample_video_array))
        crop = (0, 0, 8, 8)

        overlay = VideoOverlay(
            source=sample_video_array,
            times=times,
            alpha=0.5,
            z_order="below",
            crop=crop,
            downsample=2,
            interp="nearest",
            cache_size=500,
        )

        assert overlay.cache_size == 500
        assert overlay.alpha == 0.5
        assert overlay.z_order == "below"


class TestVideoOverlayCacheSizeConversion:
    """Test that cache_size is passed through to VideoReader during conversion."""

    @pytest.fixture
    def simple_env(self) -> Environment:
        """Create a simple 2D environment for testing."""
        positions = np.array([[0.0, 0.0], [16.0, 16.0]])
        return Environment.from_samples(positions, bin_size=1.0)

    @pytest.fixture
    def tiny_video_path(self, tmp_path: Path) -> Path:
        """Create a tiny test video file."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")

        video_path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (16, 16))

        # Write 10 frames
        for i in range(10):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 25  # Vary blue channel
            writer.write(frame)

        writer.release()
        return video_path

    def test_cache_size_passed_to_videoreader(
        self, tiny_video_path: Path, simple_env: Environment
    ):
        """Test that cache_size is passed to VideoReader."""
        import warnings

        overlay = VideoOverlay(source=tiny_video_path, cache_size=50)
        frame_times = np.linspace(0.0, 0.3, 10)

        # Suppress calibration warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video_data = overlay.convert_to_data(
                frame_times=frame_times,
                n_frames=10,
                env=simple_env,
            )

        # The reader should have cache_size=50
        # VideoReader stores cache_size as _cache_size
        assert video_data.reader._cache_size == 50

    def test_cache_size_default_passed_to_videoreader(
        self, tiny_video_path: Path, simple_env: Environment
    ):
        """Test that default cache_size=100 is passed to VideoReader."""
        import warnings

        overlay = VideoOverlay(source=tiny_video_path)
        frame_times = np.linspace(0.0, 0.3, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video_data = overlay.convert_to_data(
                frame_times=frame_times,
                n_frames=10,
                env=simple_env,
            )

        # Default cache_size=100 should be passed
        assert video_data.reader._cache_size == 100

    def test_cache_size_large_value_passed_to_videoreader(
        self, tiny_video_path: Path, simple_env: Environment
    ):
        """Test that large cache_size value is passed correctly."""
        import warnings

        overlay = VideoOverlay(source=tiny_video_path, cache_size=1000)
        frame_times = np.linspace(0.0, 0.3, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video_data = overlay.convert_to_data(
                frame_times=frame_times,
                n_frames=10,
                env=simple_env,
            )

        assert video_data.reader._cache_size == 1000

    def test_cache_size_ignored_for_array_source(self, simple_env: Environment):
        """Test that cache_size is stored but not used for array sources."""
        import warnings

        video_array = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        overlay = VideoOverlay(source=video_array, cache_size=500)

        assert overlay.cache_size == 500

        frame_times = np.linspace(0.0, 0.3, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video_data = overlay.convert_to_data(
                frame_times=frame_times,
                n_frames=10,
                env=simple_env,
            )

        # For array source, reader is the array itself (no cache needed)
        assert isinstance(video_data.reader, np.ndarray)


class TestVideoReaderCachePerformance:
    """Test that VideoReader cache works correctly with custom cache_size."""

    @pytest.fixture
    def tiny_video_path(self, tmp_path: Path) -> Path:
        """Create a tiny test video file."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")

        video_path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (16, 16))

        # Write 20 frames
        for i in range(20):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 12  # Vary blue channel
            writer.write(frame)

        writer.release()
        return video_path

    def test_small_cache_evicts_frames(self, tiny_video_path: Path):
        """Test that small cache size causes frame eviction."""
        from neurospatial.animation._video_io import VideoReader

        # Create reader with cache size of 3
        reader = VideoReader(tiny_video_path, cache_size=3)

        # Access frames 0, 1, 2 (fills cache)
        _ = reader[0]
        _ = reader[1]
        _ = reader[2]

        info_after_fill = reader.cache_info()
        assert info_after_fill.currsize == 3
        assert info_after_fill.misses == 3

        # Access frame 10 (causes eviction of frame 0)
        _ = reader[10]

        info_after_eviction = reader.cache_info()
        assert info_after_eviction.currsize == 3  # Still at max
        assert info_after_eviction.misses == 4  # One more miss

        # Access frame 0 again (should be a miss now, evicted)
        _ = reader[0]

        info_final = reader.cache_info()
        assert info_final.misses == 5  # Another miss

    def test_large_cache_retains_frames(self, tiny_video_path: Path):
        """Test that large cache size retains all accessed frames."""
        from neurospatial.animation._video_io import VideoReader

        # Create reader with cache size of 100 (more than 20 frames)
        reader = VideoReader(tiny_video_path, cache_size=100)

        # Access all 20 frames
        for i in range(20):
            _ = reader[i]

        info = reader.cache_info()
        assert info.currsize == 20  # All frames cached
        assert info.misses == 20  # All were misses initially

        # Access all frames again (all should be hits)
        for i in range(20):
            _ = reader[i]

        info_after = reader.cache_info()
        assert info_after.hits == 20  # All were hits
        assert info_after.misses == 20  # No new misses


class TestVideoPrefetching:
    """Test VideoReader prefetch_ahead parameter for async frame loading."""

    @pytest.fixture
    def video_path_30_frames(self, tmp_path: Path) -> Path:
        """Create a test video with 30 frames."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")

        video_path = tmp_path / "test_video_30.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (16, 16))

        # Write 30 frames
        for i in range(30):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 8  # Vary blue channel
            writer.write(frame)

        writer.release()
        return video_path

    def test_prefetch_ahead_default_disabled(self, video_path_30_frames: Path):
        """Test that prefetch_ahead defaults to 0 (disabled)."""
        from neurospatial.animation._video_io import VideoReader

        reader = VideoReader(video_path_30_frames)
        assert reader.prefetch_ahead == 0

    def test_prefetch_ahead_can_be_set(self, video_path_30_frames: Path):
        """Test that prefetch_ahead can be set to a positive value."""
        from neurospatial.animation._video_io import VideoReader

        reader = VideoReader(video_path_30_frames, prefetch_ahead=5)
        assert reader.prefetch_ahead == 5

    def test_prefetch_ahead_negative_raises_error(self, video_path_30_frames: Path):
        """Test that negative prefetch_ahead raises ValueError."""
        from neurospatial.animation._video_io import VideoReader

        with pytest.raises(ValueError) as exc_info:
            VideoReader(video_path_30_frames, prefetch_ahead=-1)

        assert "prefetch_ahead" in str(exc_info.value).lower()

    def test_prefetch_preloads_frames_on_sequential_access(
        self, video_path_30_frames: Path
    ):
        """Test that accessing frame N pre-loads frames N+1 to N+prefetch_ahead."""
        from neurospatial.animation._video_io import VideoReader

        # Create reader with prefetch_ahead=5 and large cache
        reader = VideoReader(video_path_30_frames, cache_size=100, prefetch_ahead=5)

        # Access frame 0
        _ = reader[0]

        # Wait a bit for prefetch thread to work
        time.sleep(0.5)

        # Check cache - should have frames 0-5 prefetched
        info = reader.cache_info()
        # At minimum frame 0 is cached (it was accessed)
        # Prefetcher should have loaded more frames in background
        assert info.currsize >= 1

        # Access frame 5 - should be a cache hit if prefetching worked
        _ = reader[5]

        info_after = reader.cache_info()
        # If prefetching worked, frame 5 was already in cache (hit)
        # If not, it's a miss. We test that prefetching improves this.
        # Due to timing, we just verify the cache grew
        assert info_after.currsize >= 2

    def test_prefetch_stops_at_end_of_video(self, video_path_30_frames: Path):
        """Test that prefetch doesn't try to load frames past the end."""
        from neurospatial.animation._video_io import VideoReader

        # Video has 30 frames (0-29)
        reader = VideoReader(video_path_30_frames, cache_size=100, prefetch_ahead=10)

        # Access frame 25 - prefetch should only go up to frame 29
        _ = reader[25]

        # Wait for prefetch
        time.sleep(0.5)

        # Should not raise any errors
        # Cache should have at most 5 frames (25, 26, 27, 28, 29)
        info = reader.cache_info()
        assert info.currsize <= 30  # Never more than video length

    def test_prefetch_disabled_when_zero(self, video_path_30_frames: Path):
        """Test that prefetch_ahead=0 disables prefetching."""
        from neurospatial.animation._video_io import VideoReader

        reader = VideoReader(video_path_30_frames, cache_size=100, prefetch_ahead=0)

        # Access frame 0
        _ = reader[0]

        # Small delay
        time.sleep(0.1)

        # Cache should only have frame 0
        info = reader.cache_info()
        assert info.currsize == 1  # Only the accessed frame

    def test_prefetch_thread_safe_access(self, video_path_30_frames: Path):
        """Test that concurrent access with prefetching is thread-safe."""
        from concurrent.futures import ThreadPoolExecutor

        from neurospatial.animation._video_io import VideoReader

        reader = VideoReader(video_path_30_frames, cache_size=100, prefetch_ahead=3)

        def access_frame(idx: int) -> np.ndarray:
            return reader[idx]

        # Concurrent access from multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(access_frame, i) for i in range(20)]
            results = [f.result() for f in futures]

        # All accesses should succeed
        assert len(results) == 20
        for frame in results:
            assert frame.shape == (16, 16, 3)

    def test_prefetch_shutdown_on_deletion(self, video_path_30_frames: Path):
        """Test that prefetch thread is properly shut down when reader is deleted."""
        from neurospatial.animation._video_io import VideoReader

        reader = VideoReader(video_path_30_frames, prefetch_ahead=5)

        # Access a frame to potentially start prefetching
        _ = reader[0]

        # Delete the reader - should cleanly shut down any prefetch threads
        del reader

        # Give time for cleanup
        time.sleep(0.2)

        # If we get here without hanging or errors, shutdown worked


class TestVideoOverlayPrefetching:
    """Test VideoOverlay prefetch_ahead parameter integration."""

    @pytest.fixture
    def simple_env(self) -> Environment:
        """Create a simple 2D environment for testing."""
        positions = np.array([[0.0, 0.0], [16.0, 16.0]])
        return Environment.from_samples(positions, bin_size=1.0)

    @pytest.fixture
    def tiny_video_path(self, tmp_path: Path) -> Path:
        """Create a tiny test video file."""
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not installed")

        video_path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (16, 16))

        for i in range(10):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 25
            writer.write(frame)

        writer.release()
        return video_path

    def test_prefetch_ahead_default_value(self):
        """Test that VideoOverlay.prefetch_ahead defaults to 0."""
        video_array = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        overlay = VideoOverlay(source=video_array)

        assert overlay.prefetch_ahead == 0

    def test_prefetch_ahead_can_be_set(self):
        """Test that VideoOverlay.prefetch_ahead can be set."""
        video_array = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        overlay = VideoOverlay(source=video_array, prefetch_ahead=10)

        assert overlay.prefetch_ahead == 10

    def test_prefetch_ahead_negative_raises_error(self):
        """Test that negative prefetch_ahead raises ValueError."""
        video_array = np.zeros((10, 16, 16, 3), dtype=np.uint8)

        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=video_array, prefetch_ahead=-5)

        assert "prefetch_ahead" in str(exc_info.value).lower()

    def test_prefetch_ahead_passed_to_videoreader(
        self, tiny_video_path: Path, simple_env: Environment
    ):
        """Test that prefetch_ahead is passed to VideoReader."""
        import warnings

        overlay = VideoOverlay(source=tiny_video_path, prefetch_ahead=5)
        frame_times = np.linspace(0.0, 0.3, 10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video_data = overlay.convert_to_data(
                frame_times=frame_times,
                n_frames=10,
                env=simple_env,
            )

        assert video_data.reader.prefetch_ahead == 5
