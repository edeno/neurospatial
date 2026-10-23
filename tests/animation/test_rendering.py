"""Field color limits and frame-label checks shared by renderers."""

import numpy as np
import pytest


class TestValidateFrameLabels:
    """Test _validate_frame_labels helper function."""

    def test_valid_labels_returned_unchanged(self):
        """Test that valid frame_labels are returned unchanged."""
        from neurospatial.animation.rendering import _validate_frame_labels

        labels = ["Frame 1", "Frame 2", "Frame 3"]
        result = _validate_frame_labels(labels, n_frames=3, backend_name="test")

        assert result == labels
        assert result is labels  # Same object reference

    def test_none_returns_none(self):
        """Test that None frame_labels returns None."""
        from neurospatial.animation.rendering import _validate_frame_labels

        result = _validate_frame_labels(None, n_frames=10, backend_name="test")

        assert result is None

    def test_mismatched_length_raises_value_error(self):
        """Test that mismatched length raises ValueError with helpful message."""
        from neurospatial.animation.rendering import _validate_frame_labels

        labels = ["Frame 1", "Frame 2"]  # 2 labels

        with pytest.raises(ValueError) as exc_info:
            _validate_frame_labels(labels, n_frames=5, backend_name="html")

        error_msg = str(exc_info.value)
        # Check WHAT/WHY/HOW format
        assert "WHAT:" in error_msg
        assert "WHY:" in error_msg
        assert "Fix:" in error_msg
        # Check specific details
        assert "2" in error_msg  # frame_labels length
        assert "5" in error_msg  # n_frames
        assert "html" in error_msg  # backend_name

    def test_error_message_includes_backend_name(self):
        """Test error message includes the backend name."""
        from neurospatial.animation.rendering import _validate_frame_labels

        labels = ["Label"]

        for backend_name in ["html", "video", "widget"]:
            with pytest.raises(ValueError) as exc_info:
                _validate_frame_labels(labels, n_frames=10, backend_name=backend_name)
            assert backend_name in str(exc_info.value)


def test_animation_color_limits_ignore_a_few_extreme_frames():
    """The limits are percentiles of the frames' extremes, not all values."""
    from neurospatial.animation._field_stats import animation_colormap_range

    rng = np.random.default_rng(0)
    # A sharp bump in a few of 1000 bins each frame, as a posterior.
    fields = np.zeros((200, 1000))
    peaks = rng.uniform(0.2, 0.3, 200)
    fields[np.arange(200), rng.integers(0, 1000, 200)] = peaks
    fields[7, 3] = 50.0  # one outlier frame
    fields[:, 500] = np.nan
    low, high = animation_colormap_range(fields, None, None)
    assert low == 0.0
    # Close to the bumps' heights, not the outlier, and not the 99th
    # percentile of all values (0), which every bump would pass.
    assert np.percentile(peaks, 95) < high < 0.31
    assert np.mean(np.nanmax(fields, axis=1) > high) <= 0.01 + 1 / 200
    assert animation_colormap_range(fields, -1.0, None)[0] == -1.0
    # A posterior's floor, 1e-56, is zero; a floor near the top is kept.
    assert animation_colormap_range(fields + 1e-56, None, None)[0] == 0.0
    assert animation_colormap_range(fields + 0.1, None, None)[0] == pytest.approx(0.1)
    assert animation_colormap_range(np.full((3, 4), np.nan), None, None) == (0.0, 1.0)
    assert animation_colormap_range([np.full(4, 2.0)] * 3, None, None) == (1.5, 2.5)


def test_typical_peak_is_the_median_frame_maximum():
    from neurospatial.animation._field_stats import frame_extremes, typical_peak

    fields = np.zeros((5, 4))
    fields[:, 1] = [0.1, 0.2, 0.3, 0.4, 9.0]  # one extreme frame
    fields[2, 3] = np.nan
    assert typical_peak(frame_extremes(fields)[1]) == pytest.approx(0.3)
    assert typical_peak(np.full(3, np.nan)) == 1.0


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # Non-finite values are ignored, frame by frame.
        ([np.array([np.nan, 0.0, 5.0]), np.array([0.0, np.inf, 5.0])], (0.0, 5.0)),
        ([np.array([-np.inf, 1.0, 3.0]), np.array([1.0, np.nan, 3.0])], (1.0, 3.0)),
        # One finite value: half a unit either side.
        ([np.array([np.nan, 5.0]), np.array([np.inf, np.nan])], (4.5, 5.5)),
        # Nothing finite, or no frames at all.
        ([np.array([np.nan, np.inf]), np.array([-np.inf, np.nan])], (0.0, 1.0)),
        ([], (0.0, 1.0)),
    ],
)
def test_animation_color_limits_use_finite_values_only(fields, expected):
    from neurospatial.animation._field_stats import animation_colormap_range

    assert animation_colormap_range(fields, None, None) == expected


def test_long_recordings_use_every_nth_frame_for_color_limits():
    from neurospatial.animation._field_stats import frame_extremes

    fields = np.zeros((250_001, 2), dtype=np.float32)
    fields[1, 0] = 99.0  # not a sampled frame
    with pytest.warns(UserWarning, match="sample_stride=5"):
        _, highs = frame_extremes(fields)
    assert len(highs) == 50_001 and highs.max() == 0.0
