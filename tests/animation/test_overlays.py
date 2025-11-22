"""Tests for animation overlay dataclasses and validation functions.

This module tests the public API dataclasses (PositionOverlay, BodypartOverlay,
HeadDirectionOverlay) and their validation/conversion pipeline.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neurospatial.animation.overlays import (
    BodypartOverlay,
    HeadDirectionOverlay,
    PositionOverlay,
    VideoOverlay,
    _validate_bounds,
    _validate_finite_values,
    _validate_monotonic_time,
    _validate_pickle_ability,
    _validate_shape,
    _validate_skeleton_consistency,
    _validate_temporal_alignment,
)
from neurospatial.animation.skeleton import Skeleton
from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar


class TestPositionOverlay:
    """Test PositionOverlay dataclass."""

    def test_basic_creation(self):
        """Test creating a PositionOverlay with required fields."""
        data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        overlay = PositionOverlay(data=data)

        assert_array_equal(overlay.data, data)
        assert overlay.times is None
        assert overlay.color == "red"
        assert overlay.size == 10.0
        assert overlay.trail_length is None

    def test_with_times(self):
        """Test PositionOverlay with timestamps."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        times = np.array([0.0, 1.0])
        overlay = PositionOverlay(data=data, times=times)

        assert_array_equal(overlay.data, data)
        assert_array_equal(overlay.times, times)

    def test_custom_styling(self):
        """Test PositionOverlay with custom color, size, trail."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        overlay = PositionOverlay(data=data, color="blue", size=20.0, trail_length=10)

        assert overlay.color == "blue"
        assert overlay.size == 20.0
        assert overlay.trail_length == 10

    def test_3d_data(self):
        """Test PositionOverlay with 3D coordinates."""
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        overlay = PositionOverlay(data=data)

        assert overlay.data.shape == (2, 3)
        assert_array_equal(overlay.data, data)


class TestBodypartOverlay:
    """Test BodypartOverlay dataclass."""

    def test_basic_creation(self):
        """Test creating a BodypartOverlay with required fields."""
        data = {
            "head": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "body": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        overlay = BodypartOverlay(data=data)

        assert "head" in overlay.data
        assert "body" in overlay.data
        assert_array_equal(overlay.data["head"], data["head"])
        assert_array_equal(overlay.data["body"], data["body"])
        assert overlay.times is None
        assert overlay.skeleton is None
        assert overlay.colors is None

    def test_with_skeleton(self):
        """Test BodypartOverlay with Skeleton object."""
        data = {
            "head": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "body": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "tail": np.array([[2.0, 3.0], [4.0, 5.0]]),
        }
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body", "tail"),
            edges=(("head", "body"), ("body", "tail")),
        )
        overlay = BodypartOverlay(data=data, skeleton=skeleton)

        assert overlay.skeleton is skeleton
        assert len(overlay.skeleton.edges) == 2
        assert overlay.skeleton.name == "test"

    def test_with_custom_colors(self):
        """Test BodypartOverlay with per-part colors."""
        data = {
            "head": np.array([[0.0, 1.0]]),
            "body": np.array([[1.0, 2.0]]),
        }
        colors = {"head": "red", "body": "blue"}
        overlay = BodypartOverlay(data=data, colors=colors)

        assert overlay.colors == colors
        assert overlay.colors["head"] == "red"
        assert overlay.colors["body"] == "blue"

    def test_custom_skeleton_styling(self):
        """Test BodypartOverlay with custom skeleton appearance via Skeleton."""
        data = {"head": np.array([[0.0, 1.0]]), "body": np.array([[1.0, 2.0]])}
        skeleton = Skeleton(
            name="styled",
            nodes=("head", "body"),
            edges=(("head", "body"),),
            edge_color="yellow",
            edge_width=3.0,
        )
        overlay = BodypartOverlay(data=data, skeleton=skeleton)

        assert overlay.skeleton.edge_color == "yellow"
        assert overlay.skeleton.edge_width == 3.0

    def test_with_times(self):
        """Test BodypartOverlay with timestamps."""
        data = {
            "head": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "body": np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        times = np.array([0.0, 1.0])
        overlay = BodypartOverlay(data=data, times=times)

        assert_array_equal(overlay.times, times)


class TestHeadDirectionOverlay:
    """Test HeadDirectionOverlay dataclass."""

    def test_basic_creation_with_angles(self):
        """Test creating HeadDirectionOverlay with angle data."""
        data = np.array([0.0, np.pi / 2, np.pi])
        overlay = HeadDirectionOverlay(data=data)

        assert_array_equal(overlay.data, data)
        assert overlay.times is None
        assert overlay.color == "yellow"
        assert overlay.length == 0.25

    def test_with_unit_vectors(self):
        """Test HeadDirectionOverlay with unit vector data."""
        data = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        overlay = HeadDirectionOverlay(data=data)

        assert overlay.data.shape == (3, 2)
        assert_array_equal(overlay.data, data)

    def test_custom_styling(self):
        """Test HeadDirectionOverlay with custom appearance."""
        data = np.array([0.0, np.pi])
        overlay = HeadDirectionOverlay(data=data, color="red", length=30.0)

        assert overlay.color == "red"
        assert overlay.length == 30.0

    def test_with_times(self):
        """Test HeadDirectionOverlay with timestamps."""
        data = np.array([0.0, np.pi / 2])
        times = np.array([0.0, 1.0])
        overlay = HeadDirectionOverlay(data=data, times=times)

        assert_array_equal(overlay.times, times)

    def test_3d_vectors(self):
        """Test HeadDirectionOverlay with 3D unit vectors."""
        data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        overlay = HeadDirectionOverlay(data=data)

        assert overlay.data.shape == (2, 3)
        assert_array_equal(overlay.data, data)


class TestVideoOverlay:
    """Test VideoOverlay dataclass."""

    @pytest.fixture
    def sample_video_array(self) -> np.ndarray:
        """Create a sample video array (n_frames, height, width, 3)."""
        # 10 frames, 16x16 pixels, RGB
        return np.zeros((10, 16, 16, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_calibration(self) -> VideoCalibration:
        """Create a sample VideoCalibration for testing."""
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )
        return VideoCalibration(transform, frame_size_px=(640, 480))

    def test_basic_creation_with_array(self, sample_video_array: np.ndarray):
        """Test creating VideoOverlay with pre-loaded array."""
        overlay = VideoOverlay(source=sample_video_array)

        assert_array_equal(overlay.source, sample_video_array)
        assert overlay.calibration is None
        assert overlay.times is None
        assert overlay.alpha == 0.7
        assert overlay.z_order == "below"
        assert overlay.crop is None
        assert overlay.downsample == 1
        assert overlay.interp == "nearest"

    def test_with_calibration(
        self, sample_video_array: np.ndarray, sample_calibration: VideoCalibration
    ):
        """Test VideoOverlay with VideoCalibration."""
        overlay = VideoOverlay(
            source=sample_video_array,
            calibration=sample_calibration,
        )

        assert overlay.calibration is sample_calibration
        assert overlay.calibration.frame_size_px == (640, 480)

    def test_with_timestamps(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with timestamps."""
        times = np.linspace(0.0, 1.0, len(sample_video_array))
        overlay = VideoOverlay(
            source=sample_video_array,
            times=times,
        )

        assert_array_equal(overlay.times, times)

    def test_custom_alpha(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with custom alpha."""
        overlay = VideoOverlay(source=sample_video_array, alpha=0.5)

        assert overlay.alpha == 0.5

    def test_z_order_below(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with z_order='below' (default)."""
        overlay = VideoOverlay(source=sample_video_array, z_order="below")

        assert overlay.z_order == "below"

    def test_z_order_above(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with z_order='above'."""
        overlay = VideoOverlay(source=sample_video_array, z_order="above")

        assert overlay.z_order == "above"

    def test_with_crop(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with crop region."""
        crop = (10, 20, 100, 80)  # x, y, width, height
        overlay = VideoOverlay(source=sample_video_array, crop=crop)

        assert overlay.crop == crop

    def test_with_downsample(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with downsampling factor."""
        overlay = VideoOverlay(source=sample_video_array, downsample=2)

        assert overlay.downsample == 2

    def test_interp_nearest(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with nearest interpolation (default)."""
        overlay = VideoOverlay(source=sample_video_array, interp="nearest")

        assert overlay.interp == "nearest"

    def test_interp_linear(self, sample_video_array: np.ndarray):
        """Test VideoOverlay with linear interpolation."""
        overlay = VideoOverlay(source=sample_video_array, interp="linear")

        assert overlay.interp == "linear"

    def test_all_parameters(
        self, sample_video_array: np.ndarray, sample_calibration: VideoCalibration
    ):
        """Test VideoOverlay with all parameters specified."""
        times = np.linspace(0.0, 1.0, len(sample_video_array))
        crop = (0, 0, 8, 8)

        overlay = VideoOverlay(
            source=sample_video_array,
            calibration=sample_calibration,
            times=times,
            alpha=0.3,
            z_order="above",
            crop=crop,
            downsample=2,
            interp="linear",
        )

        assert overlay.calibration is sample_calibration
        assert_array_equal(overlay.times, times)
        assert overlay.alpha == 0.3
        assert overlay.z_order == "above"
        assert overlay.crop == crop
        assert overlay.downsample == 2
        assert overlay.interp == "linear"


class TestVideoOverlayValidation:
    """Test VideoOverlay __post_init__ validation."""

    def test_alpha_lower_bound(self):
        """Test that alpha < 0.0 raises ValueError."""
        data = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data, alpha=-0.1)

        error_msg = str(exc_info.value)
        assert "alpha" in error_msg.lower()
        assert "0.0" in error_msg or "0" in error_msg

    def test_alpha_upper_bound(self):
        """Test that alpha > 1.0 raises ValueError."""
        data = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data, alpha=1.5)

        error_msg = str(exc_info.value)
        assert "alpha" in error_msg.lower()
        assert "1.0" in error_msg or "1" in error_msg

    def test_alpha_at_boundaries(self):
        """Test that alpha at boundaries (0.0 and 1.0) is valid."""
        data = np.zeros((10, 16, 16, 3), dtype=np.uint8)

        overlay_min = VideoOverlay(source=data, alpha=0.0)
        overlay_max = VideoOverlay(source=data, alpha=1.0)

        assert overlay_min.alpha == 0.0
        assert overlay_max.alpha == 1.0

    def test_array_wrong_ndim(self):
        """Test that 3D array (not 4D) raises ValueError."""
        data = np.zeros((16, 16, 3), dtype=np.uint8)  # Missing frames dimension
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data)

        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower() or "dimension" in error_msg.lower()

    def test_array_wrong_channels(self):
        """Test that non-RGB array (not 3 channels) raises ValueError."""
        data = np.zeros((10, 16, 16, 4), dtype=np.uint8)  # RGBA, not RGB
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data)

        error_msg = str(exc_info.value)
        assert "channel" in error_msg.lower() or "rgb" in error_msg.lower()

    def test_array_wrong_dtype(self):
        """Test that non-uint8 array raises ValueError."""
        data = np.zeros((10, 16, 16, 3), dtype=np.float32)  # float, not uint8
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data)

        error_msg = str(exc_info.value)
        assert "uint8" in error_msg.lower() or "dtype" in error_msg.lower()

    def test_downsample_must_be_positive(self):
        """Test that downsample <= 0 raises ValueError."""
        data = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data, downsample=0)

        error_msg = str(exc_info.value)
        assert "downsample" in error_msg.lower()

    def test_downsample_negative(self):
        """Test that negative downsample raises ValueError."""
        data = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data, downsample=-1)

        error_msg = str(exc_info.value)
        assert "downsample" in error_msg.lower()

    def test_downsample_float_not_integer(self):
        """Test that float downsample raises ValueError (must be integer)."""
        data = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc_info:
            VideoOverlay(source=data, downsample=1.5)  # type: ignore[arg-type]

        error_msg = str(exc_info.value)
        assert "downsample" in error_msg.lower()


class TestVideoOverlayWithFilePath:
    """Test VideoOverlay with file path source."""

    def test_with_string_path(self, tmp_path):
        """Test VideoOverlay with string file path (deferred existence check)."""
        # Note: For file paths, existence check happens during VideoReader creation,
        # not during VideoOverlay construction (lazy loading)
        video_path = str(tmp_path / "nonexistent.mp4")

        # Construction should succeed (lazy loading)
        overlay = VideoOverlay(source=video_path)

        assert overlay.source == video_path

    def test_with_path_object(self, tmp_path):
        """Test VideoOverlay with Path object."""

        video_path = tmp_path / "test.mp4"

        overlay = VideoOverlay(source=video_path)

        assert overlay.source == video_path


class TestVideoData:
    """Test VideoData internal container."""

    @pytest.fixture
    def sample_video_frames(self) -> np.ndarray:
        """Create sample video frames (n_frames, height, width, 3)."""
        # 5 video frames, 8x8 pixels, RGB
        frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        # Add distinct patterns to each frame for testing
        for i in range(5):
            frames[i, :, :, :] = i * 50  # Different brightness per frame
        return frames

    @pytest.fixture
    def sample_frame_indices(self) -> np.ndarray:
        """Create sample frame indices mapping animation frames to video frames."""
        # 10 animation frames mapping to 5 video frames
        # -1 indicates out-of-range (no video frame available)
        return np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, -1], dtype=np.int_)

    @pytest.fixture
    def sample_env_bounds(self) -> tuple[float, float, float, float]:
        """Sample environment bounds (xmin, xmax, ymin, ymax)."""
        return (0.0, 100.0, 0.0, 80.0)

    def test_basic_creation(
        self,
        sample_video_frames: np.ndarray,
        sample_frame_indices: np.ndarray,
        sample_env_bounds: tuple[float, float, float, float],
    ):
        """Test creating VideoData with array source."""
        from neurospatial.animation.overlays import VideoData

        video_data = VideoData(
            frame_indices=sample_frame_indices,
            reader=sample_video_frames,
            transform_to_env=None,
            env_bounds=sample_env_bounds,
            alpha=0.7,
            z_order="below",
        )

        assert_array_equal(video_data.frame_indices, sample_frame_indices)
        assert video_data.transform_to_env is None
        assert video_data.env_bounds == sample_env_bounds
        assert video_data.alpha == 0.7
        assert video_data.z_order == "below"

    def test_with_transform(
        self,
        sample_video_frames: np.ndarray,
        sample_frame_indices: np.ndarray,
        sample_env_bounds: tuple[float, float, float, float],
    ):
        """Test VideoData with Affine2D transform."""
        from neurospatial.animation.overlays import VideoData
        from neurospatial.transforms import identity

        transform = identity()
        video_data = VideoData(
            frame_indices=sample_frame_indices,
            reader=sample_video_frames,
            transform_to_env=transform,
            env_bounds=sample_env_bounds,
            alpha=0.5,
            z_order="above",
        )

        assert video_data.transform_to_env is transform
        assert video_data.z_order == "above"

    def test_get_frame_valid_index(
        self,
        sample_video_frames: np.ndarray,
        sample_frame_indices: np.ndarray,
        sample_env_bounds: tuple[float, float, float, float],
    ):
        """Test get_frame returns correct frame for valid animation index."""
        from neurospatial.animation.overlays import VideoData

        video_data = VideoData(
            frame_indices=sample_frame_indices,
            reader=sample_video_frames,
            transform_to_env=None,
            env_bounds=sample_env_bounds,
            alpha=0.7,
            z_order="below",
        )

        # Animation frame 0 maps to video frame 0
        frame = video_data.get_frame(0)
        assert frame is not None
        assert frame.shape == (8, 8, 3)
        assert frame.dtype == np.uint8
        # First frame has brightness 0
        assert frame[0, 0, 0] == 0

        # Animation frame 4 maps to video frame 2
        frame = video_data.get_frame(4)
        assert frame is not None
        # Video frame 2 has brightness 100
        assert frame[0, 0, 0] == 100

    def test_get_frame_out_of_range_returns_none(
        self,
        sample_video_frames: np.ndarray,
        sample_frame_indices: np.ndarray,
        sample_env_bounds: tuple[float, float, float, float],
    ):
        """Test get_frame returns None for index -1 (out of range)."""
        from neurospatial.animation.overlays import VideoData

        video_data = VideoData(
            frame_indices=sample_frame_indices,
            reader=sample_video_frames,
            transform_to_env=None,
            env_bounds=sample_env_bounds,
            alpha=0.7,
            z_order="below",
        )

        # Animation frame 9 maps to -1 (out of range)
        frame = video_data.get_frame(9)
        assert frame is None

    def test_get_frame_negative_index_returns_none(
        self,
        sample_video_frames: np.ndarray,
        sample_frame_indices: np.ndarray,
        sample_env_bounds: tuple[float, float, float, float],
    ):
        """Test get_frame returns None for animation index out of bounds."""
        from neurospatial.animation.overlays import VideoData

        video_data = VideoData(
            frame_indices=sample_frame_indices,
            reader=sample_video_frames,
            transform_to_env=None,
            env_bounds=sample_env_bounds,
            alpha=0.7,
            z_order="below",
        )

        # Animation index beyond frame_indices length
        frame = video_data.get_frame(100)
        assert frame is None

    def test_pickle_safety(
        self,
        sample_video_frames: np.ndarray,
        sample_frame_indices: np.ndarray,
        sample_env_bounds: tuple[float, float, float, float],
    ):
        """Test VideoData can be pickled for parallel rendering."""
        import pickle

        from neurospatial.animation.overlays import VideoData

        video_data = VideoData(
            frame_indices=sample_frame_indices,
            reader=sample_video_frames,
            transform_to_env=None,
            env_bounds=sample_env_bounds,
            alpha=0.7,
            z_order="below",
        )

        # Should be pickle-able
        pickled = pickle.dumps(video_data)
        restored = pickle.loads(pickled)

        assert_array_equal(restored.frame_indices, video_data.frame_indices)
        assert restored.alpha == video_data.alpha
        assert restored.z_order == video_data.z_order

        # Restored object should work
        frame = restored.get_frame(0)
        assert frame is not None


class TestOverlayDataclassDefaults:
    """Test that overlay dataclasses have correct default values."""

    def test_position_overlay_defaults(self):
        """Test PositionOverlay default values."""
        data = np.array([[0.0, 1.0]])
        overlay = PositionOverlay(data=data)

        assert overlay.color == "red"
        assert overlay.size == 10.0
        assert overlay.trail_length is None

    def test_bodypart_overlay_defaults(self):
        """Test BodypartOverlay default values."""
        data = {"head": np.array([[0.0, 1.0]])}
        overlay = BodypartOverlay(data=data)

        assert overlay.skeleton is None
        assert overlay.colors is None

    def test_head_direction_overlay_defaults(self):
        """Test HeadDirectionOverlay default values."""
        data = np.array([0.0])
        overlay = HeadDirectionOverlay(data=data)

        assert overlay.color == "yellow"
        assert overlay.length == 0.25

    def test_video_overlay_defaults(self):
        """Test VideoOverlay default values."""
        data = np.zeros((10, 16, 16, 3), dtype=np.uint8)
        overlay = VideoOverlay(source=data)

        assert overlay.calibration is None
        assert overlay.times is None
        assert overlay.alpha == 0.7
        assert overlay.z_order == "below"
        assert overlay.crop is None
        assert overlay.downsample == 1
        assert overlay.interp == "nearest"


class TestMultiAnimalSupport:
    """Test that multiple overlays can be created for multi-animal scenarios."""

    def test_multiple_position_overlays(self):
        """Test creating multiple PositionOverlay instances for multiple animals."""
        animal1_data = np.array([[0.0, 1.0], [2.0, 3.0]])
        animal2_data = np.array([[5.0, 6.0], [7.0, 8.0]])

        overlay1 = PositionOverlay(data=animal1_data, color="red")
        overlay2 = PositionOverlay(data=animal2_data, color="blue")

        assert overlay1.color == "red"
        assert overlay2.color == "blue"
        assert not np.array_equal(overlay1.data, overlay2.data)

    def test_multiple_bodypart_overlays(self):
        """Test creating multiple BodypartOverlay instances."""
        animal1_data = {"head": np.array([[0.0, 1.0]])}
        animal2_data = {"head": np.array([[5.0, 6.0]])}

        overlay1 = BodypartOverlay(data=animal1_data)
        overlay2 = BodypartOverlay(data=animal2_data)

        assert not np.array_equal(overlay1.data["head"], overlay2.data["head"])


# =============================================================================
# Validation Functions Tests (Milestone 1.4)
# =============================================================================


class TestValidateMonotonicTime:
    """Test _validate_monotonic_time() function."""

    def test_valid_monotonic_increasing(self):
        """Test that monotonically increasing times pass validation."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # Should not raise any exception
        _validate_monotonic_time(times, name="test_overlay")

    def test_strictly_increasing_times(self):
        """Test that strictly increasing times pass validation."""
        times = np.array([0.5, 1.5, 3.0, 10.0])
        _validate_monotonic_time(times, name="test_overlay")

    def test_non_monotonic_decreasing(self):
        """Test that decreasing times raise ValueError with actionable message."""
        times = np.array([0.0, 2.0, 1.0, 3.0])  # Decreases at index 2
        with pytest.raises(ValueError) as exc_info:
            _validate_monotonic_time(times, name="test_overlay")

        error_msg = str(exc_info.value)
        # WHAT: Non-monotonic times detected
        assert "non-monotonic" in error_msg.lower() or "monotonic" in error_msg.lower()
        # WHY: Interpolation requires increasing timestamps
        assert "interpolation" in error_msg.lower()
        # HOW: Sort or call fix_monotonic_timestamps()
        assert "sort" in error_msg.lower() or "fix" in error_msg.lower()

    def test_duplicate_times(self):
        """Test that duplicate consecutive times raise ValueError."""
        times = np.array([0.0, 1.0, 1.0, 2.0])  # Duplicate at index 2
        with pytest.raises(ValueError) as exc_info:
            _validate_monotonic_time(times, name="test_overlay")

        error_msg = str(exc_info.value)
        assert "monotonic" in error_msg.lower()

    def test_all_identical_times(self):
        """Test that all identical times raise ValueError."""
        times = np.array([1.0, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError) as exc_info:
            _validate_monotonic_time(times, name="test_overlay")

        error_msg = str(exc_info.value)
        assert "monotonic" in error_msg.lower()

    def test_empty_times_array(self):
        """Test validation with empty times array."""
        times = np.array([])
        # Empty array should pass (no validation needed)
        _validate_monotonic_time(times, name="test_overlay")

    def test_single_time_value(self):
        """Test validation with single time value (trivially monotonic)."""
        times = np.array([5.0])
        _validate_monotonic_time(times, name="test_overlay")


class TestValidateFiniteValues:
    """Test _validate_finite_values() function."""

    def test_all_finite_values(self):
        """Test that arrays with all finite values pass validation."""
        data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        # Should not raise any exception
        _validate_finite_values(data, name="test_data")

    def test_contains_nan(self):
        """Test that NaN values raise ValueError with count and index."""
        data = np.array([[0.0, 1.0], [np.nan, 3.0], [4.0, np.nan]])
        with pytest.raises(ValueError) as exc_info:
            _validate_finite_values(data, name="test_data")

        error_msg = str(exc_info.value)
        # WHAT: Found NaN in arrays
        assert "nan" in error_msg.lower()
        # Count of NaN values
        assert "2" in error_msg  # 2 NaN values
        # WHY: Rendering cannot place invalid coordinates
        assert "render" in error_msg.lower() or "invalid" in error_msg.lower()
        # HOW: Clean or mask; suggest interpolation
        assert "clean" in error_msg.lower() or "mask" in error_msg.lower()

    def test_contains_inf(self):
        """Test that Inf values raise ValueError."""
        data = np.array([[0.0, 1.0], [np.inf, 3.0], [4.0, 5.0]])
        with pytest.raises(ValueError) as exc_info:
            _validate_finite_values(data, name="test_data")

        error_msg = str(exc_info.value)
        assert "inf" in error_msg.lower()
        assert "1" in error_msg  # 1 Inf value

    def test_contains_neg_inf(self):
        """Test that -Inf values raise ValueError."""
        data = np.array([[0.0, 1.0], [-np.inf, 3.0]])
        with pytest.raises(ValueError) as exc_info:
            _validate_finite_values(data, name="test_data")

        error_msg = str(exc_info.value)
        assert "inf" in error_msg.lower()

    def test_mixed_nan_and_inf(self):
        """Test array with both NaN and Inf values."""
        data = np.array([[np.nan, 1.0], [np.inf, 3.0], [4.0, -np.inf]])
        with pytest.raises(ValueError) as exc_info:
            _validate_finite_values(data, name="test_data")

        error_msg = str(exc_info.value)
        # Should report total count of non-finite values
        assert "3" in error_msg  # 3 non-finite values

    def test_1d_array_with_nan(self):
        """Test validation with 1D array containing NaN."""
        data = np.array([0.0, 1.0, np.nan, 3.0])
        with pytest.raises(ValueError) as exc_info:
            _validate_finite_values(data, name="test_data")

        error_msg = str(exc_info.value)
        assert "nan" in error_msg.lower()


class TestValidateShape:
    """Test _validate_shape() function."""

    def test_correct_2d_shape(self):
        """Test that correct shape passes validation."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        expected_ndims = 2
        _validate_shape(data, expected_ndims, name="test_data")

    def test_correct_3d_shape(self):
        """Test validation with 3D data."""
        data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        expected_ndims = 3
        _validate_shape(data, expected_ndims, name="test_data")

    def test_wrong_dimensions(self):
        """Test that dimension mismatch raises ValueError."""
        data = np.array([[0.0, 1.0], [2.0, 3.0]])  # 2D
        expected_ndims = 3  # Expecting 3D
        with pytest.raises(ValueError) as exc_info:
            _validate_shape(data, expected_ndims, name="test_data")

        error_msg = str(exc_info.value)
        # WHAT: Shape mismatch
        assert "shape" in error_msg.lower() or "dimension" in error_msg.lower()
        # Expected vs actual
        assert "2" in error_msg  # actual
        assert "3" in error_msg  # expected
        # WHY: Coordinate dimensionality must match environment
        assert "environment" in error_msg.lower() or "match" in error_msg.lower()
        # HOW: Project/reformat
        assert "reformat" in error_msg.lower() or "project" in error_msg.lower()

    def test_1d_array_wrong_shape(self):
        """Test that 1D array raises error when 2D expected."""
        data = np.array([0.0, 1.0, 2.0])  # 1D
        expected_ndims = 2
        with pytest.raises(ValueError) as exc_info:
            _validate_shape(data, expected_ndims, name="test_data")

        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower() or "dimension" in error_msg.lower()

    def test_empty_array(self):
        """Test validation with empty array."""
        data = np.array([]).reshape(0, 2)
        expected_ndims = 2
        # Empty array with correct shape should pass
        _validate_shape(data, expected_ndims, name="test_data")

    def test_3d_array_raises_clear_error(self):
        """Test that 3D+ arrays raise clear error about invalid shape."""
        data = np.array([[[0.0, 1.0], [2.0, 3.0]]])  # 3D array
        expected_ndims = 2
        with pytest.raises(ValueError) as exc_info:
            _validate_shape(data, expected_ndims, name="test_data")

        error_msg = str(exc_info.value)
        # Should mention invalid shape
        assert "invalid" in error_msg.lower() or "shape" in error_msg.lower()
        # Should show actual dimensions
        assert "3" in error_msg  # 3 dimensions
        # Should mention it must be 1D or 2D
        assert "1d" in error_msg.lower() or "2d" in error_msg.lower()


class TestValidateTemporalAlignment:
    """Test _validate_temporal_alignment() function."""

    def test_full_overlap(self):
        """Test that full temporal overlap passes validation."""
        overlay_times = np.array([0.0, 1.0, 2.0, 3.0])
        frame_times = np.array([0.0, 1.0, 2.0, 3.0])
        # Should not raise any exception or warning
        _validate_temporal_alignment(overlay_times, frame_times, name="test_overlay")

    def test_overlay_subset_of_frames(self):
        """Test when overlay times are subset of frame times."""
        overlay_times = np.array([1.0, 2.0, 3.0])
        frame_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        _validate_temporal_alignment(overlay_times, frame_times, name="test_overlay")

    def test_frames_subset_of_overlay(self):
        """Test when frame times are subset of overlay times."""
        overlay_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        frame_times = np.array([1.0, 2.0, 3.0])
        _validate_temporal_alignment(overlay_times, frame_times, name="test_overlay")

    def test_no_overlap_error(self):
        """Test that no temporal overlap raises ValueError."""
        overlay_times = np.array([0.0, 1.0, 2.0])
        frame_times = np.array([5.0, 6.0, 7.0])
        with pytest.raises(ValueError) as exc_info:
            _validate_temporal_alignment(
                overlay_times, frame_times, name="test_overlay"
            )

        error_msg = str(exc_info.value)
        # WHAT: No overlap
        assert "overlap" in error_msg.lower()
        # WHY: Interpolation domain is disjoint
        assert "disjoint" in error_msg.lower() or "interpolation" in error_msg.lower()
        # HOW: Provide overlapping time ranges
        assert "overlapping" in error_msg.lower() or "resample" in error_msg.lower()

    def test_partial_overlap_warning(self):
        """Test that partial overlap <50% raises warning."""
        overlay_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 5 seconds
        frame_times = np.array(
            [3.5, 4.0, 4.5, 5.0, 5.5]
        )  # Overlap: 3.5-4.0 = 0.5s (10%)

        # Should emit UserWarning
        with pytest.warns(UserWarning) as warn_info:
            _validate_temporal_alignment(
                overlay_times, frame_times, name="test_overlay"
            )

        warning_msg = str(warn_info[0].message)
        # Should report overlap percentage
        assert "overlap" in warning_msg.lower()
        assert "%" in warning_msg  # Percentage reported

    def test_good_overlap_no_warning(self):
        """Test that >50% overlap does not warn."""
        import warnings

        overlay_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        frame_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # >50% overlap

        # Should not warn
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            _validate_temporal_alignment(
                overlay_times, frame_times, name="test_overlay"
            )

        # Check no UserWarnings were issued
        user_warnings = [w for w in warn_list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


class TestValidateBounds:
    """Test _validate_bounds() function."""

    def test_all_points_in_bounds(self):
        """Test that all points within bounds pass without warning."""
        import warnings

        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        dim_ranges = [(0.0, 10.0), (0.0, 10.0)]

        # Should not warn
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            _validate_bounds(data, dim_ranges, name="test_data", threshold=0.1)

        # Check no UserWarnings were issued
        user_warnings = [w for w in warn_list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_some_points_out_of_bounds_below_threshold(self):
        """Test that few out-of-bounds points below threshold don't warn."""
        import warnings

        data = np.array([[1.0, 2.0], [3.0, 4.0], [15.0, 16.0]])  # 1/3 = 33% out
        dim_ranges = [(0.0, 10.0), (0.0, 10.0)]

        # With threshold=0.5 (50%), should not warn
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            _validate_bounds(data, dim_ranges, name="test_data", threshold=0.5)

        # Check no UserWarnings were issued
        user_warnings = [w for w in warn_list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_many_points_out_of_bounds_warning(self):
        """Test that many out-of-bounds points raise warning."""
        data = np.array([[15.0, 16.0], [20.0, 25.0], [3.0, 4.0]])  # 2/3 = 66% out
        dim_ranges = [(0.0, 10.0), (0.0, 10.0)]

        with pytest.warns(UserWarning) as warn_info:
            _validate_bounds(data, dim_ranges, name="test_data", threshold=0.5)

        warning_msg = str(warn_info[0].message)
        # WARN: >X% points outside dimension_ranges
        assert "%" in warning_msg
        assert "outside" in warning_msg.lower() or "out" in warning_msg.lower()
        # HOW: Confirm coordinate system and units
        assert "coordinate" in warning_msg.lower() or "unit" in warning_msg.lower()

    def test_show_min_max_values(self):
        """Test that warning shows min/max vs environment ranges."""
        data = np.array([[15.0, 16.0], [20.0, 25.0]])
        dim_ranges = [(0.0, 10.0), (0.0, 10.0)]

        with pytest.warns(UserWarning) as warn_info:
            _validate_bounds(data, dim_ranges, name="test_data", threshold=0.0)

        warning_msg = str(warn_info[0].message)
        # Should show actual min/max
        assert "15" in warning_msg or "20" in warning_msg or "25" in warning_msg
        # Should show env ranges
        assert "0" in warning_msg or "10" in warning_msg

    def test_skip_validation_for_1d_angles(self):
        """Test that 1D angle arrays skip bounds validation."""
        import warnings

        data = np.array([0.0, np.pi, 2 * np.pi, 10 * np.pi])  # Angles, no bounds
        dim_ranges = [(0.0, 10.0), (0.0, 10.0)]

        # Should not warn (1D data skips bounds checking)
        with warnings.catch_warnings(record=True) as warn_list:
            warnings.simplefilter("always")
            _validate_bounds(data, dim_ranges, name="angles", threshold=0.0)

        # Check no UserWarnings were issued
        user_warnings = [w for w in warn_list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


class TestValidateSkeletonConsistency:
    """Test _validate_skeleton_consistency() function."""

    def test_valid_skeleton(self):
        """Test that skeleton with all valid part names passes."""
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body", "tail"),
            edges=(("head", "body"), ("body", "tail")),
        )
        bodypart_names = ["head", "body", "tail"]
        # Should not raise
        _validate_skeleton_consistency(skeleton, bodypart_names, name="test_skeleton")

    def test_skeleton_with_missing_parts(self):
        """Test that skeleton referencing missing parts raises ValueError."""
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body", "tail"),
            edges=(("head", "body"), ("body", "tail")),
        )
        bodypart_names = ["head", "body"]  # Missing 'tail' in data

        with pytest.raises(ValueError) as exc_info:
            _validate_skeleton_consistency(
                skeleton, bodypart_names, name="test_skeleton"
            )

        error_msg = str(exc_info.value)
        # WHAT: Skeleton references missing part(s)
        assert "skeleton" in error_msg.lower()
        assert "missing" in error_msg.lower()
        # WHY: Cannot draw edges without endpoints
        assert "edge" in error_msg.lower() or "endpoint" in error_msg.lower()
        # Should mention 'tail'
        assert "tail" in error_msg.lower()

    def test_skeleton_with_suggestions(self):
        """Test that error includes nearest match suggestions."""
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body", "tale"),  # Typo: 'tale' vs 'tail'
            edges=(("head", "body"), ("body", "tale")),
        )
        bodypart_names = ["head", "body", "tail"]

        with pytest.raises(ValueError) as exc_info:
            _validate_skeleton_consistency(
                skeleton, bodypart_names, name="test_skeleton"
            )

        error_msg = str(exc_info.value)
        # HOW: Should suggest nearest matches
        assert "tail" in error_msg.lower()  # Suggestion for 'tale'

    def test_empty_skeleton(self):
        """Test that empty skeleton passes validation."""
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body", "tail"),
            edges=(),  # No edges
        )
        bodypart_names = ["head", "body", "tail"]
        _validate_skeleton_consistency(skeleton, bodypart_names, name="test_skeleton")

    def test_none_skeleton(self):
        """Test that None skeleton passes validation."""
        skeleton = None
        bodypart_names = ["head", "body", "tail"]
        _validate_skeleton_consistency(skeleton, bodypart_names, name="test_skeleton")


class TestValidatePickleAbility:
    """Test _validate_pickle_ability() function."""

    def test_pickleable_overlay_data(self):
        """Test that pickle-able OverlayData passes validation."""
        # Import the internal data classes (will implement later)
        from neurospatial.animation.overlays import OverlayData, PositionData

        position_data = PositionData(
            data=np.array([[0.0, 1.0], [2.0, 3.0]]),
            color="red",
            size=10.0,
            trail_length=None,
        )
        overlay_data = OverlayData(positions=[position_data])

        # Should not raise
        _validate_pickle_ability(overlay_data, n_workers=4)

    def test_unpickleable_with_lambda(self):
        """Test that OverlayData with lambda raises ValueError."""
        from neurospatial.animation.overlays import OverlayData

        # Create OverlayData with unpickleable attribute
        overlay_data = OverlayData()
        overlay_data._unpickleable_func = lambda x: x * 2  # Add unpickleable attribute

        with pytest.raises(ValueError) as exc_info:
            _validate_pickle_ability(overlay_data, n_workers=4)

        error_msg = str(exc_info.value)
        # WHAT: OverlayData not pickle-able
        assert "pickle" in error_msg.lower()
        # WHY: Parallel video rendering requires pickling
        assert "parallel" in error_msg.lower()
        # HOW: Remove unpickleable obj or n_workers=1
        assert "n_workers=1" in error_msg or "remove" in error_msg.lower()

    def test_skip_validation_with_single_worker(self):
        """Test that pickle validation is skipped when n_workers=1."""
        from neurospatial.animation.overlays import OverlayData

        overlay_data = OverlayData()
        overlay_data._unpickleable_func = lambda x: x * 2

        # Should not raise when n_workers=1 or None
        _validate_pickle_ability(overlay_data, n_workers=1)
        _validate_pickle_ability(overlay_data, n_workers=None)


# =============================================================================
# Conversion Funnel Tests (Milestone 1.5)
# =============================================================================


class TestConvertOverlaysToData:
    """Test _convert_overlays_to_data() conversion funnel."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""

        class MockEnv:
            n_dims: ClassVar[int] = 2
            dimension_ranges: ClassVar[list[tuple[float, float]]] = [
                (0.0, 100.0),
                (0.0, 100.0),
            ]

        return MockEnv()

    def test_convert_position_overlay_with_times(self, mock_env):
        """Test converting PositionOverlay with timestamps to OverlayData."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create position overlay with timestamps
        positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        times = np.array([0.0, 1.0, 2.0])
        overlay = PositionOverlay(
            data=positions, times=times, color="red", size=15.0, trail_length=5
        )

        # Define frame times (matching overlay times)
        frame_times = np.array([0.0, 1.0, 2.0])
        n_frames = 3

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Verify OverlayData structure
        assert len(overlay_data.positions) == 1
        assert len(overlay_data.bodypart_sets) == 0
        assert len(overlay_data.head_directions) == 0

        # Verify PositionData
        pos_data = overlay_data.positions[0]
        assert pos_data.data.shape == (3, 2)
        assert pos_data.color == "red"
        assert pos_data.size == 15.0
        assert pos_data.trail_length == 5
        assert_array_equal(pos_data.data, positions)

    def test_convert_position_overlay_without_times(self, mock_env):
        """Test converting PositionOverlay without timestamps (uniform spacing)."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create position overlay without timestamps
        positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        overlay = PositionOverlay(data=positions)

        # Frame times from fps
        frame_times = np.array([0.0, 1.0 / 30, 2.0 / 30])  # 30 fps
        n_frames = 3

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Should assume uniform spacing and copy data directly
        assert len(overlay_data.positions) == 1
        pos_data = overlay_data.positions[0]
        assert pos_data.data.shape == (3, 2)

    def test_convert_position_overlay_with_interpolation(self, mock_env):
        """Test PositionOverlay conversion with temporal interpolation."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Overlay at 2 Hz (0.0, 0.5, 1.0)
        positions = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        times = np.array([0.0, 0.5, 1.0])
        overlay = PositionOverlay(data=positions, times=times)

        # Animation at 4 Hz (0.0, 0.25, 0.5, 0.75, 1.0)
        frame_times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        n_frames = 5

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        pos_data = overlay_data.positions[0]
        assert pos_data.data.shape == (5, 2)

        # Check interpolated values
        # At t=0.25, should be between (0,0) and (5,5) -> (2.5, 2.5)
        assert np.allclose(pos_data.data[1], [2.5, 2.5])
        # At t=0.75, should be between (5,5) and (10,10) -> (7.5, 7.5)
        assert np.allclose(pos_data.data[3], [7.5, 7.5])

    def test_convert_bodypart_overlay_with_times(self, mock_env):
        """Test converting BodypartOverlay with timestamps to OverlayData."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create bodypart overlay
        data = {
            "head": np.array([[10.0, 20.0], [30.0, 40.0]]),
            "body": np.array([[15.0, 25.0], [35.0, 45.0]]),
        }
        times = np.array([0.0, 1.0])
        skeleton = Skeleton(
            name="test",
            nodes=("head", "body"),
            edges=(("head", "body"),),
        )
        colors = {"head": "red", "body": "blue"}
        overlay = BodypartOverlay(
            data=data, times=times, skeleton=skeleton, colors=colors
        )

        frame_times = np.array([0.0, 1.0])
        n_frames = 2

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Verify OverlayData structure
        assert len(overlay_data.positions) == 0
        assert len(overlay_data.bodypart_sets) == 1
        assert len(overlay_data.head_directions) == 0

        # Verify BodypartData
        bodypart_data = overlay_data.bodypart_sets[0]
        assert "head" in bodypart_data.bodyparts
        assert "body" in bodypart_data.bodyparts
        assert bodypart_data.bodyparts["head"].shape == (2, 2)
        assert bodypart_data.bodyparts["body"].shape == (2, 2)
        assert bodypart_data.skeleton is skeleton
        assert bodypart_data.colors == colors

    def test_convert_bodypart_overlay_with_interpolation(self, mock_env):
        """Test BodypartOverlay conversion with per-keypoint interpolation."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Bodypart overlay at 2 Hz
        data = {
            "head": np.array([[0.0, 0.0], [10.0, 10.0]]),
            "body": np.array([[0.0, 5.0], [10.0, 15.0]]),
        }
        times = np.array([0.0, 1.0])
        overlay = BodypartOverlay(data=data, times=times)

        # Animation at 4 Hz
        frame_times = np.array([0.0, 0.5, 1.0])
        n_frames = 3

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        bodypart_data = overlay_data.bodypart_sets[0]
        assert bodypart_data.bodyparts["head"].shape == (3, 2)
        assert bodypart_data.bodyparts["body"].shape == (3, 2)

        # Check interpolated values at t=0.5
        assert np.allclose(bodypart_data.bodyparts["head"][1], [5.0, 5.0])
        assert np.allclose(bodypart_data.bodyparts["body"][1], [5.0, 10.0])

    def test_convert_head_direction_overlay_angles(self, mock_env):
        """Test converting HeadDirectionOverlay with angles."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Head direction as angles
        angles = np.array([0.0, np.pi / 2, np.pi])
        times = np.array([0.0, 1.0, 2.0])
        overlay = HeadDirectionOverlay(
            data=angles, times=times, color="yellow", length=25.0
        )

        frame_times = np.array([0.0, 1.0, 2.0])
        n_frames = 3

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Verify
        assert len(overlay_data.head_directions) == 1
        hd_data = overlay_data.head_directions[0]
        assert hd_data.data.shape == (3,)
        assert hd_data.color == "yellow"
        assert hd_data.length == 25.0
        assert_array_equal(hd_data.data, angles)

    def test_convert_head_direction_overlay_vectors(self, mock_env):
        """Test converting HeadDirectionOverlay with unit vectors."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Head direction as unit vectors
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        times = np.array([0.0, 1.0, 2.0])
        overlay = HeadDirectionOverlay(data=vectors, times=times)

        frame_times = np.array([0.0, 1.0, 2.0])
        n_frames = 3

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        hd_data = overlay_data.head_directions[0]
        assert hd_data.data.shape == (3, 2)
        assert_array_equal(hd_data.data, vectors)

    def test_convert_multiple_overlays(self, mock_env):
        """Test converting multiple overlays (multi-animal scenario)."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Multiple position overlays (2 animals)
        animal1_pos = np.array([[10.0, 20.0], [30.0, 40.0]])
        animal2_pos = np.array([[50.0, 60.0], [70.0, 80.0]])
        times = np.array([0.0, 1.0])

        overlay1 = PositionOverlay(data=animal1_pos, times=times, color="red")
        overlay2 = PositionOverlay(data=animal2_pos, times=times, color="blue")

        frame_times = np.array([0.0, 1.0])
        n_frames = 2

        # Convert
        overlay_data = _convert_overlays_to_data(
            overlays=[overlay1, overlay2],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Verify both overlays converted
        assert len(overlay_data.positions) == 2
        assert overlay_data.positions[0].color == "red"
        assert overlay_data.positions[1].color == "blue"
        assert overlay_data.positions[0].data.shape == (2, 2)
        assert overlay_data.positions[1].data.shape == (2, 2)

    def test_convert_mixed_overlay_types(self, mock_env):
        """Test converting mixed overlay types in single call."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create one of each type
        pos_data = np.array([[10.0, 20.0], [30.0, 40.0]])
        bodypart_data = {"head": np.array([[15.0, 25.0], [35.0, 45.0]])}
        hd_data = np.array([0.0, np.pi])
        times = np.array([0.0, 1.0])

        pos_overlay = PositionOverlay(data=pos_data, times=times)
        bodypart_overlay = BodypartOverlay(data=bodypart_data, times=times)
        hd_overlay = HeadDirectionOverlay(data=hd_data, times=times)

        frame_times = np.array([0.0, 1.0])
        n_frames = 2

        # Convert all at once
        overlay_data = _convert_overlays_to_data(
            overlays=[pos_overlay, bodypart_overlay, hd_overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Verify all types present
        assert len(overlay_data.positions) == 1
        assert len(overlay_data.bodypart_sets) == 1
        assert len(overlay_data.head_directions) == 1

    def test_validation_called_during_conversion(self, mock_env):
        """Test that validation functions are called during conversion."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create overlay with non-monotonic times (should trigger validation error)
        positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        times = np.array([0.0, 2.0, 1.0])  # Non-monotonic!
        overlay = PositionOverlay(data=positions, times=times)

        frame_times = np.array([0.0, 1.0, 2.0])
        n_frames = 3

        # Should raise ValueError from _validate_monotonic_time
        with pytest.raises(ValueError) as exc_info:
            _convert_overlays_to_data(
                overlays=[overlay],
                frame_times=frame_times,
                n_frames=n_frames,
                env=mock_env,
            )

        error_msg = str(exc_info.value)
        assert "monotonic" in error_msg.lower()

    def test_validation_finite_values_during_conversion(self, mock_env):
        """Test that finite value validation is called."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create overlay with NaN values
        positions = np.array([[10.0, 20.0], [np.nan, 40.0], [50.0, 60.0]])
        times = np.array([0.0, 1.0, 2.0])
        overlay = PositionOverlay(data=positions, times=times)

        frame_times = np.array([0.0, 1.0, 2.0])
        n_frames = 3

        # Should raise ValueError from _validate_finite_values
        with pytest.raises(ValueError) as exc_info:
            _convert_overlays_to_data(
                overlays=[overlay],
                frame_times=frame_times,
                n_frames=n_frames,
                env=mock_env,
            )

        error_msg = str(exc_info.value)
        assert "nan" in error_msg.lower()

    def test_validation_shape_mismatch(self, mock_env):
        """Test that shape validation catches dimension mismatches."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create 3D positions but env is 2D
        positions = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])  # 3D
        times = np.array([0.0, 1.0])
        overlay = PositionOverlay(data=positions, times=times)

        frame_times = np.array([0.0, 1.0])
        n_frames = 2

        # Should raise ValueError from _validate_shape
        with pytest.raises(ValueError) as exc_info:
            _convert_overlays_to_data(
                overlays=[overlay],
                frame_times=frame_times,
                n_frames=n_frames,
                env=mock_env,
            )

        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower() or "dimension" in error_msg.lower()

    def test_validation_skeleton_consistency(self, mock_env):
        """Test that skeleton validation is called."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Create bodypart with skeleton referencing missing part
        data = {"head": np.array([[10.0, 20.0]])}
        skeleton = Skeleton(
            name="test",
            nodes=("head", "tail"),
            edges=(("head", "tail"),),
        )  # 'tail' doesn't exist in data!
        times = np.array([0.0])
        overlay = BodypartOverlay(data=data, skeleton=skeleton, times=times)

        frame_times = np.array([0.0])
        n_frames = 1

        # Should raise ValueError from _validate_skeleton_consistency
        with pytest.raises(ValueError) as exc_info:
            _convert_overlays_to_data(
                overlays=[overlay],
                frame_times=frame_times,
                n_frames=n_frames,
                env=mock_env,
            )

        error_msg = str(exc_info.value)
        assert "skeleton" in error_msg.lower()
        assert "missing" in error_msg.lower()

    def test_empty_overlays_list(self, mock_env):
        """Test conversion with empty overlays list."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        frame_times = np.array([0.0, 1.0, 2.0])
        n_frames = 3

        # Convert with no overlays
        overlay_data = _convert_overlays_to_data(
            overlays=[],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Should return empty OverlayData
        assert len(overlay_data.positions) == 0
        assert len(overlay_data.bodypart_sets) == 0
        assert len(overlay_data.head_directions) == 0

    def test_result_is_pickle_safe(self, mock_env):
        """Test that returned OverlayData is pickle-safe."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        times = np.array([0.0, 1.0])
        overlay = PositionOverlay(data=positions, times=times)

        frame_times = np.array([0.0, 1.0])
        n_frames = 2

        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        # Should be pickle-able
        import pickle

        pickled = pickle.dumps(overlay_data)
        unpickled = pickle.loads(pickled)

        # Verify unpickled data matches
        assert len(unpickled.positions) == 1
        assert_array_equal(unpickled.positions[0].data, positions)

    def test_extrapolation_produces_nan(self, mock_env):
        """Test that extrapolation outside overlay time range produces NaN."""
        from neurospatial.animation.overlays import _convert_overlays_to_data

        # Overlay covers [1.0, 2.0]
        positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        times = np.array([1.0, 2.0])
        overlay = PositionOverlay(data=positions, times=times)

        # Frames extend beyond overlay range [0.0, 3.0]
        frame_times = np.array([0.0, 1.0, 2.0, 3.0])
        n_frames = 4

        overlay_data = _convert_overlays_to_data(
            overlays=[overlay],
            frame_times=frame_times,
            n_frames=n_frames,
            env=mock_env,
        )

        pos_data = overlay_data.positions[0]

        # First frame (t=0.0) and last frame (t=3.0) should be NaN
        assert np.isnan(pos_data.data[0]).all()  # t=0.0 < 1.0
        assert not np.isnan(pos_data.data[1]).all()  # t=1.0 OK
        assert not np.isnan(pos_data.data[2]).all()  # t=2.0 OK
        assert np.isnan(pos_data.data[3]).all()  # t=3.0 > 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
