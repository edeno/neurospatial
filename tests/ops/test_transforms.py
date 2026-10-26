"""Tests for transforms.py calibration helpers.

These tests cover the video calibration functions added in v0.5.0
for pixel↔cm coordinate conversion.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neurospatial.ops.transforms import (
    AffineND,
    calibrate_from_landmarks,
    calibrate_from_scale_bar,
    flip_y,
    scale_2d,
)


class TestCalibrateFromScaleBar:
    """Tests for calibrate_from_scale_bar()."""

    def test_horizontal_scale_bar(self):
        """Horizontal scale bar produces correct scale."""
        # 200 pixel scale bar represents 50 cm
        # px distance = 200, so cm_per_px = 50/200 = 0.25
        transform = calibrate_from_scale_bar(
            p1_px=(100.0, 200.0),
            p2_px=(300.0, 200.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        # Check scale: 100 pixels should map to 25 cm
        point_px = np.array([[100.0, 0.0]])
        point_cm = transform(point_px)
        # x: 100 * 0.25 = 25 cm
        assert_allclose(point_cm[0, 0], 25.0, atol=1e-10)

    def test_vertical_scale_bar(self):
        """Vertical scale bar produces correct scale."""
        # 100 pixel vertical scale bar represents 25 cm
        # cm_per_px = 25/100 = 0.25
        transform = calibrate_from_scale_bar(
            p1_px=(50.0, 100.0),
            p2_px=(50.0, 200.0),
            known_length_cm=25.0,
            frame_size_px=(640, 480),
        )

        # Check that transform is correctly computed
        assert isinstance(transform, AffineND)

    def test_diagonal_scale_bar(self):
        """Diagonal scale bar computes Euclidean distance correctly."""
        # 3-4-5 triangle: endpoints (0, 0) and (300, 400) -> distance = 500 pixels
        # If 500 pixels = 100 cm, cm_per_px = 0.2
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(300.0, 400.0),
            known_length_cm=100.0,
            frame_size_px=(640, 480),
        )

        # 500 pixels should map to 100 cm
        point_px = np.array([[500.0, 0.0]])
        point_cm = transform(point_px)
        # x: 500 * 0.2 = 100 cm
        assert_allclose(point_cm[0, 0], 100.0, atol=1e-10)

    def test_y_flip_applied(self):
        """Transform includes Y-flip (video origin top-left to env bottom-left)."""
        # Frame height 480, scale bar at y=100 in video
        # After flip: y_cm = (480 - y_px) * scale
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 200.0),
            p2_px=(100.0, 200.0),  # 100 px = 50 cm -> cm_per_px = 0.5
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        # Point at top of video (y_px=0) should map to y_cm = 480 * 0.5 = 240
        top_point_px = np.array([[0.0, 0.0]])
        top_point_cm = transform(top_point_px)
        assert_allclose(top_point_cm[0, 1], 240.0, atol=1e-10)

        # Point at bottom of video (y_px=480) should map to y_cm = 0
        bottom_point_px = np.array([[0.0, 480.0]])
        bottom_point_cm = transform(bottom_point_px)
        assert_allclose(bottom_point_cm[0, 1], 0.0, atol=1e-10)

    def test_returns_affine_nd(self):
        """Returns an AffineND instance."""
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=10.0,
            frame_size_px=(640, 480),
        )
        assert isinstance(transform, AffineND)

    def test_zero_length_raises(self):
        """Zero-length scale bar raises ValueError."""
        with pytest.raises(ValueError, match="length"):
            calibrate_from_scale_bar(
                p1_px=(100.0, 100.0),
                p2_px=(100.0, 100.0),  # Same point = zero length
                known_length_cm=50.0,
                frame_size_px=(640, 480),
            )

    def test_negative_length_raises(self):
        """Negative known_length_cm raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calibrate_from_scale_bar(
                p1_px=(0.0, 0.0),
                p2_px=(100.0, 0.0),
                known_length_cm=-10.0,
                frame_size_px=(640, 480),
            )


class TestCalibrateFromLandmarks:
    """Tests for calibrate_from_landmarks()."""

    @pytest.mark.parametrize("kind", ["rigid", "similarity", "affine"])
    def test_maps_raw_pixels_to_world_landmarks_and_unseen_points(self, kind):
        """A calibrated transform accepts raw image pixels, including their Y-down axis."""
        landmarks_px = np.array(
            [[30, 20], [220, 35], [200, 160], [40, 170]], dtype=float
        )
        angle = np.deg2rad(27.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        if kind == "rigid":
            linear = rotation
        elif kind == "similarity":
            linear = 0.35 * rotation
        else:
            linear = np.array([[0.3, 0.07], [-0.04, 0.4]])

        def world_coordinates(raw_pixels):
            cartesian = np.column_stack([raw_pixels[:, 0], 240.0 - raw_pixels[:, 1]])
            return cartesian @ linear.T + [10.0, -30.0]

        landmarks_cm = world_coordinates(landmarks_px)
        original_pixels = landmarks_px.copy()
        transform = calibrate_from_landmarks(
            landmarks_px, landmarks_cm, frame_size_px=(320, 240), kind=kind
        )

        assert_allclose(transform(landmarks_px), landmarks_cm, atol=1e-10)
        assert_allclose(transform.inverse()(landmarks_cm), landmarks_px, atol=1e-10)
        unseen_px = np.array([[15.0, 215.0], [147.0, 73.0]])
        assert_allclose(transform(unseen_px), world_coordinates(unseen_px), atol=1e-10)
        assert_allclose(landmarks_px, original_pixels)

    def test_landmark_and_scale_bar_calibrations_share_raw_pixel_convention(self):
        """Both calibration methods map a raw pixel height to the same world height."""
        landmarks_px = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=float)
        landmarks_cm = np.array([[0, 120], [160, 120], [160, 0], [0, 0]], dtype=float)
        landmark_transform = calibrate_from_landmarks(
            landmarks_px, landmarks_cm, frame_size_px=(640, 480)
        )
        scale_bar_transform = calibrate_from_scale_bar(
            (100.0, 200.0), (300.0, 200.0), 50.0, frame_size_px=(640, 480)
        )
        raw_pixels = np.array([[7.0, 31.0], [503.0, 419.0]])
        expected_cm = np.array([[1.75, 112.25], [125.75, 15.25]])
        assert_allclose(landmark_transform(raw_pixels), expected_cm, atol=1e-10)
        assert_allclose(scale_bar_transform(raw_pixels), expected_cm, atol=1e-10)

    def test_rigid_transform(self):
        """Rigid (rotation + translation) transform from landmarks."""
        # Simple case: 4 corners, no rotation, pure scaling + Y-flip
        landmarks_px = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        landmarks_cm = np.array(
            [
                [0.0, 10.0],  # Y-flipped: px(0,0) -> cm(0, 100*0.1 = 10)
                [10.0, 10.0],
                [10.0, 0.0],
                [0.0, 0.0],
            ]
        )

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(640, 480),
            kind="rigid",
        )

        assert isinstance(transform, AffineND)

    def test_similarity_transform(self):
        """Similarity (scale + rotation + translation) transform."""
        # Create landmarks with uniform scale
        landmarks_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        # 2x scale; image row 0 is the top of the arena, so y points up in cm.
        landmarks_cm = np.array([[0, 200], [200, 200], [200, 0], [0, 0]], dtype=float)

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(100, 100),
            kind="similarity",
        )

        assert isinstance(transform, AffineND)
        assert_allclose(transform(landmarks_px), landmarks_cm, atol=1e-10)

    def test_mirrored_landmarks_are_refused(self):
        """Landmarks listed in image orientation are a mirror image after the y flip."""
        landmarks_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        landmarks_cm = 2 * landmarks_px
        with pytest.raises(ValueError, match="mirror image"):
            calibrate_from_landmarks(landmarks_px, landmarks_cm, (100, 100))

    def test_affine_transform(self):
        """Full affine transform from landmarks."""
        # Simple 1:1 mapping for test
        landmarks_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        landmarks_cm = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(100, 100),
            kind="affine",
        )

        assert isinstance(transform, AffineND)

    def test_roundtrip_accuracy(self):
        """Pixel→cm→pixel roundtrip within tolerance."""
        # Define landmarks
        landmarks_px = np.array(
            [
                [50, 50],
                [590, 50],
                [590, 430],
                [50, 430],
            ],
            dtype=float,
        )
        landmarks_cm = np.array(
            [
                [0, 80],
                [100, 80],
                [100, 0],
                [0, 0],
            ],
            dtype=float,
        )

        transform = calibrate_from_landmarks(
            landmarks_px=landmarks_px,
            landmarks_cm=landmarks_cm,
            frame_size_px=(640, 480),
            kind="similarity",
        )

        # Transform px -> cm
        test_px = landmarks_px.copy()
        result_cm = transform(test_px)

        # Transform cm -> px (inverse)
        inverse = transform.inverse()
        roundtrip_px = inverse(result_cm)

        # Should get back to original (within tolerance)
        # Use scale-dependent tolerance: max(1e-4, 1e-6 * extent)
        extent = max(640, 480)
        atol = max(1e-4, 1e-6 * extent)
        assert_allclose(roundtrip_px, test_px, atol=atol)

    def test_insufficient_landmarks_raises(self):
        """Less than 3 landmarks raises ValueError."""
        landmarks_px = np.array([[0, 0], [100, 0]], dtype=float)  # Only 2 points
        landmarks_cm = np.array([[0, 0], [10, 0]], dtype=float)

        with pytest.raises(ValueError, match="landmark"):
            calibrate_from_landmarks(
                landmarks_px=landmarks_px,
                landmarks_cm=landmarks_cm,
                frame_size_px=(640, 480),
                kind="similarity",
            )

    def test_non_finite_landmark_is_named(self):
        """A landmark that was not annotated is reported by its row."""
        corners_px = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        corners_cm = np.array([[0, 0], [50, 0], [50, 50], [np.nan, 50]])
        with pytest.raises(ValueError, match=r"landmarks_cm has NaN .* row\(s\) \[3\]"):
            calibrate_from_landmarks(corners_px, corners_cm, (100, 100))

    def test_mismatched_landmark_count_raises(self):
        """Different number of px and cm landmarks raises ValueError."""
        landmarks_px = np.array([[0, 0], [100, 0], [100, 100]], dtype=float)
        landmarks_cm = np.array([[0, 0], [10, 0]], dtype=float)  # Only 2!

        with pytest.raises(ValueError, match="same number"):
            calibrate_from_landmarks(
                landmarks_px=landmarks_px,
                landmarks_cm=landmarks_cm,
                frame_size_px=(640, 480),
                kind="similarity",
            )


class TestFlipYIntegration:
    """Test flip_y() integration with calibration."""

    def test_flip_y_compose_with_scale(self):
        """scale_2d @ flip_y produces correct video->env transform.

        Composition order: scale_2d @ flip_y means "apply flip_y first, then scale_2d".
        """
        frame_height = 480
        cm_per_px = 0.25

        # Compose: flip_y then scale (right-to-left application)
        transform = scale_2d(cm_per_px, cm_per_px) @ flip_y(frame_height)

        # Point at top of video (y=0) should map to y = 480 * 0.25 = 120 cm
        top_px = np.array([[0.0, 0.0]])
        top_cm = transform(top_px)
        assert_allclose(top_cm[0, 1], 120.0, atol=1e-10)

        # Point at bottom of video (y=480) should map to y = 0 cm
        bottom_px = np.array([[0.0, 480.0]])
        bottom_cm = transform(bottom_px)
        assert_allclose(bottom_cm[0, 1], 0.0, atol=1e-10)


class TestVideoCalibration:
    """Tests for VideoCalibration dataclass."""

    def test_basic_creation(self):
        """VideoCalibration can be created with transform and frame size."""
        from neurospatial.ops.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        assert calib.frame_size_px == (640, 480)
        assert isinstance(calib.transform_px_to_cm, AffineND)

    def test_cm_per_px_property(self):
        """cm_per_px returns approximate scale factor."""
        from neurospatial.ops.transforms import VideoCalibration

        # 100 pixels = 50 cm -> cm_per_px = 0.5
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        assert_allclose(calib.cm_per_px, 0.5, atol=1e-10)

    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2])
    @pytest.mark.parametrize("scales", [(0.5, 0.5), (0.25, -0.75)])
    def test_cm_per_px_invariant_under_rotation(self, angle, scales):
        """Rotating or reflecting calibrated axes does not change their scale."""
        from neurospatial.ops.transforms import VideoCalibration

        rotation = AffineND(
            np.array(
                [
                    [np.cos(angle), -np.sin(angle), 10],
                    [np.sin(angle), np.cos(angle), 20],
                    [0, 0, 1],
                ]
            )
        )
        calib = VideoCalibration(rotation @ scale_2d(*scales), (640, 480))
        assert calib.cm_per_px == pytest.approx(np.mean(np.abs(scales)))

    def test_transform_cm_to_px_inverse(self):
        """transform_cm_to_px is the inverse of transform_px_to_cm."""
        from neurospatial.ops.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(200.0, 0.0),
            known_length_cm=100.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        # Test roundtrip px -> cm -> px
        test_px = np.array([[100.0, 200.0]])
        result_cm = calib.transform_px_to_cm(test_px)
        roundtrip_px = calib.transform_cm_to_px(result_cm)

        assert_allclose(roundtrip_px, test_px, atol=1e-10)

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization roundtrip preserves calibration."""
        from neurospatial.ops.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        original = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        # Roundtrip through dict
        d = original.to_dict()
        restored = VideoCalibration.from_dict(d)

        # Check restored calibration works identically
        test_px = np.array([[320.0, 240.0]])
        original_cm = original.transform_px_to_cm(test_px)
        restored_cm = restored.transform_px_to_cm(test_px)

        assert_allclose(restored_cm, original_cm, atol=1e-10)
        assert restored.frame_size_px == original.frame_size_px

    def test_to_dict_contains_expected_keys(self):
        """to_dict produces expected structure."""
        from neurospatial.ops.transforms import VideoCalibration

        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),
            known_length_cm=50.0,
            frame_size_px=(640, 480),
        )

        calib = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )

        d = calib.to_dict()

        assert "transform_px_to_cm" in d
        assert "frame_size_px" in d
        assert d["frame_size_px"] == [640, 480]

    def test_is_immutable_so_the_inverse_cannot_go_stale(self):
        """The inverse is fixed with the forward transform at construction."""
        from dataclasses import FrozenInstanceError

        from neurospatial.ops.transforms import VideoCalibration

        calib = VideoCalibration(scale_2d(0.5), (640, 480))
        with pytest.raises(FrozenInstanceError):
            calib.transform_px_to_cm = scale_2d(2.0)
        with pytest.raises(TypeError):
            VideoCalibration(scale_2d(0.5), (640, 480), scale_2d(2.0))
        assert_allclose(calib.transform_cm_to_px.A, scale_2d(2.0).A)

    def test_equality_compares_transform_and_frame_size(self):
        from neurospatial.ops.transforms import VideoCalibration

        calib = VideoCalibration(scale_2d(0.5), (640, 480))
        assert calib == VideoCalibration(scale_2d(0.5), (640, 480))
        assert calib != VideoCalibration(scale_2d(0.5), (320, 240))
        assert calib != VideoCalibration(scale_2d(0.25), (640, 480))
        assert calib == VideoCalibration.from_dict(calib.to_dict())

    @pytest.mark.parametrize(
        "frame_size", [(-5, 0), (640,), (640.5, 480), (True, 480), (0, 480)]
    )
    def test_refuses_invalid_frame_size(self, frame_size):
        from neurospatial.ops.transforms import VideoCalibration

        with pytest.raises(ValueError, match="two positive integers"):
            VideoCalibration(scale_2d(0.5), frame_size)

    def test_refuses_a_transform_that_is_not_2d(self):
        from neurospatial.ops.transforms import VideoCalibration, scale_3d

        with pytest.raises(ValueError, match="2D transform"):
            VideoCalibration(scale_3d(0.5), (640, 480))
        with pytest.raises(TypeError, match="AffineND"):
            VideoCalibration(np.eye(3), (640, 480))


class TestCalibrationEdgeCases:
    """Edge case tests for calibration validation."""

    def test_rejects_ill_conditioned_landmarks(self):
        """Collinear landmarks raise ValueError due to ill-conditioned transform.

        Landmarks that are collinear (all on a line) produce an ill-conditioned
        transform that cannot reliably map points. This should be rejected with
        a clear error message.
        """
        # All points on a horizontal line (y=200 constant)
        collinear_px = np.array(
            [
                [100.0, 200.0],
                [200.0, 200.0],
                [300.0, 200.0],
                [400.0, 200.0],
            ]
        )
        # Corresponding environment points (also collinear)
        collinear_cm = np.array(
            [
                [0.0, 50.0],
                [25.0, 50.0],
                [50.0, 50.0],
                [75.0, 50.0],
            ]
        )

        with pytest.raises(ValueError, match=r"[Ii]ll.?condition"):
            calibrate_from_landmarks(
                landmarks_px=collinear_px,
                landmarks_cm=collinear_cm,
                frame_size_px=(640, 480),
                kind="similarity",
            )

    def test_rejects_nearly_collinear_landmarks(self):
        """Nearly collinear landmarks also raise ValueError.

        Points that are almost on a line (very small deviation) should also
        be rejected as the resulting transform will be numerically unstable.
        """
        # Points almost on a horizontal line (tiny y variation)
        nearly_collinear_px = np.array(
            [
                [100.0, 200.0],
                [200.0, 200.0001],  # Tiny deviation
                [300.0, 200.0],
                [400.0, 200.0001],
            ]
        )
        nearly_collinear_cm = np.array(
            [
                [0.0, 50.0],
                [25.0, 50.0001],
                [50.0, 50.0],
                [75.0, 50.0001],
            ]
        )

        with pytest.raises(ValueError, match=r"[Ii]ll.?condition"):
            calibrate_from_landmarks(
                landmarks_px=nearly_collinear_px,
                landmarks_cm=nearly_collinear_cm,
                frame_size_px=(640, 480),
                kind="similarity",
            )


class TestBoundsMismatchWarning:
    """Tests for bounds coverage validation."""

    def test_warns_bounds_mismatch(self, tmp_path):
        """Warns when environment bounds exceed calibrated video coverage.

        When the environment extends beyond what the calibrated video covers,
        a warning should be emitted to alert the user that some regions will
        appear blank.
        """
        # calibrate_video reads frames via OpenCV; both cv2 and a working
        # imageio video plugin live in the optional `animation` extra.
        pytest.importorskip("cv2")
        pytest.importorskip("imageio_ffmpeg")

        # Create a tiny test video file
        import imageio

        video_path = tmp_path / "test_video.mp4"
        # 64x48 video (small for fast test)
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
        imageio.mimwrite(str(video_path), frames, fps=10)

        # Create environment with bounds LARGER than video will cover
        # Video is 64x48 pixels with scale 1.0 cm/px = 64x48 cm coverage
        # But environment is 100x100 cm
        from neurospatial import Environment

        positions = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ]
        )
        env = Environment.from_samples(positions, bin_size=10.0)

        # Import calibrate_video
        from neurospatial.animation.calibration import calibrate_video

        # Calibration with scale 1.0 cm/px means video covers 64x48 cm
        # Environment needs 100x100 cm - should warn
        with pytest.warns(UserWarning, match="[Bb]ounds.*extend|exceed"):
            calibrate_video(
                env,
                video_path,
                cm_per_px=1.0,
            )

    def test_no_warning_when_video_covers_env(self):
        """No warning when video coverage fully contains environment bounds.

        Test the validation function directly with a calibration that clearly
        covers the environment bounds.
        """
        import warnings

        from neurospatial import Environment
        from neurospatial.animation.calibration import _validate_calibration_coverage
        from neurospatial.ops.transforms import VideoCalibration

        # Create a small environment
        positions = np.array(
            [
                [10.0, 10.0],
                [20.0, 10.0],
                [20.0, 20.0],
                [10.0, 20.0],
            ]
        )
        # With bin_size=5.0, bounds will be ~[7.5, 22.5] x [7.5, 22.5]
        env = Environment.from_samples(positions, bin_size=5.0)

        # Create calibration that maps 640x480 video to 0-100 cm
        # Using simple 1:1 pixel-to-cm with Y-flip: video covers 0-640 x 0-480 cm
        # which is much larger than env bounds [7.5, 22.5]
        transform = calibrate_from_scale_bar(
            p1_px=(0.0, 0.0),
            p2_px=(100.0, 0.0),  # 100 px = 100 cm -> 1 cm/px
            known_length_cm=100.0,
            frame_size_px=(640, 480),
        )
        calibration = VideoCalibration(
            transform_px_to_cm=transform,
            frame_size_px=(640, 480),
        )
        # Video covers x: 0 to 640 cm, y: 0 to 480 cm
        # Env bounds: [7.5, 22.5] x [7.5, 22.5] - fully inside

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should NOT raise any warnings
            _validate_calibration_coverage(calibration, env)
