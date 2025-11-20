"""Error message quality tests for animation module.

Verifies that all error messages:
- Are clear and descriptive
- Include diagnostic information (actual values)
- Suggest actionable solutions
- Provide installation instructions for missing dependencies
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation import subsample_frames
from neurospatial.animation.core import animate_fields


@pytest.fixture
def test_env():
    """Create test environment."""
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)
    return env


@pytest.fixture
def test_fields(test_env):
    """Create test fields matching environment."""
    return [np.random.rand(test_env.n_bins) for _ in range(10)]


# ============================================================================
# Missing Dependency Errors
# ============================================================================


def test_napari_missing_error_message(test_env, test_fields):
    """Verify napari ImportError has installation instructions."""
    # Mock napari as unavailable
    with patch(
        "neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", False
    ):
        with pytest.raises(ImportError) as exc_info:
            animate_fields(test_env, test_fields, backend="napari")

        error_msg = str(exc_info.value)
        # Check for key elements
        assert "napari" in error_msg.lower()
        assert "pip install napari[all]" in error_msg
        assert "uv add napari[all]" in error_msg


def test_ipywidgets_missing_error_message(test_env, test_fields):
    """Verify ipywidgets ImportError has installation instructions."""
    # Mock ipywidgets as unavailable
    with patch(
        "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", False
    ):
        with pytest.raises(ImportError) as exc_info:
            animate_fields(test_env, test_fields, backend="widget")

        error_msg = str(exc_info.value)
        # Check for key elements
        assert "ipywidgets" in error_msg.lower()
        assert "pip install ipywidgets" in error_msg
        assert "uv add ipywidgets" in error_msg


def test_ffmpeg_missing_error_message(test_env, test_fields, tmp_path):
    """Verify ffmpeg RuntimeError has platform-specific install instructions."""
    # Mock ffmpeg as unavailable
    with patch(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        return_value=False,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            animate_fields(
                test_env, test_fields, backend="video", save_path=tmp_path / "out.mp4"
            )

        error_msg = str(exc_info.value)
        # Check for platform-specific instructions
        assert "ffmpeg" in error_msg.lower()
        assert any(
            platform in error_msg
            for platform in ["macOS", "Ubuntu", "Windows", "brew", "apt"]
        )


def test_pillow_missing_error_message(test_env, test_fields):
    """Verify Pillow ImportError has installation instructions."""
    from neurospatial.animation.rendering import render_field_to_image_bytes

    # Mock PIL as unavailable
    with patch.dict(sys.modules, {"PIL": None}):
        with pytest.raises(ImportError) as exc_info:
            render_field_to_image_bytes(
                test_env, test_fields[0], "viridis", 0, 1, image_format="jpeg"
            )

        error_msg = str(exc_info.value)
        # Check for key elements
        assert "Pillow" in error_msg or "pillow" in error_msg.lower()
        assert "pip install pillow" in error_msg
        assert "uv add pillow" in error_msg
        # Check for workaround suggestion
        assert "image_format='png'" in error_msg


# ============================================================================
# Validation Errors
# ============================================================================


def test_environment_not_fitted_error_message():
    """Verify unfitted environment error suggests factory methods."""
    # Create environment and manually mark as unfitted
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)
    # Manually break the fitted state
    env._is_fitted = False

    with pytest.raises(RuntimeError) as exc_info:
        animate_fields(env, [np.array([1, 2, 3])])

    error_msg = str(exc_info.value)
    # Check for key elements
    assert "fitted" in error_msg.lower()
    assert "Environment.from_samples()" in error_msg


def test_field_shape_mismatch_error_message(test_env):
    """Verify field shape mismatch shows expected vs actual."""
    wrong_fields = [np.random.rand(test_env.n_bins + 10)]

    with pytest.raises(ValueError) as exc_info:
        animate_fields(test_env, wrong_fields)

    error_msg = str(exc_info.value)
    # Check for diagnostic info
    assert str(test_env.n_bins) in error_msg
    assert str(test_env.n_bins + 10) in error_msg
    assert "Expected shape" in error_msg


def test_empty_fields_error_message(test_env):
    """Verify empty fields error is clear."""
    with pytest.raises(ValueError) as exc_info:
        animate_fields(test_env, [])

    error_msg = str(exc_info.value)
    assert "empty" in error_msg.lower()


def test_fields_dimension_error_message(test_env):
    """Verify fields dimension error is clear."""
    # 1D array (missing time dimension)
    fields_1d = np.random.rand(test_env.n_bins)

    with pytest.raises(ValueError) as exc_info:
        animate_fields(test_env, fields_1d)

    error_msg = str(exc_info.value)
    assert "2D" in error_msg
    assert "n_frames" in error_msg


def test_pickle_failure_error_message(test_env, test_fields, tmp_path, monkeypatch):
    """Verify pickle failure provides actionable solutions."""
    # Create environment with unpicklable cache
    test_env._kdtree_cache = {"unpicklable": lambda: None}  # Functions can't pickle

    # Skip ffmpeg check for this test
    monkeypatch.setattr(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        lambda: True,
    )

    with pytest.raises(ValueError) as exc_info:
        animate_fields(
            test_env,
            test_fields,
            backend="video",
            save_path=tmp_path / "out.mp4",
            n_workers=2,
        )

    error_msg = str(exc_info.value)
    # Check for key solutions
    assert "pickle" in error_msg.lower()
    assert "clear_cache()" in error_msg
    assert "n_workers=1" in error_msg
    assert "backend='html'" in error_msg


def test_save_path_required_error_message(test_env, test_fields):
    """Verify video backend save_path requirement is clear."""
    with pytest.raises(ValueError) as exc_info:
        animate_fields(test_env, test_fields, backend="video")

    error_msg = str(exc_info.value)
    assert "save_path" in error_msg.lower()
    assert "required" in error_msg.lower()


def test_unknown_backend_error_message(test_env, test_fields):
    """Verify unknown backend error is clear."""
    with pytest.raises(ValueError) as exc_info:
        animate_fields(test_env, test_fields, backend="invalid_backend")  # type: ignore[arg-type]

    error_msg = str(exc_info.value)
    assert "backend" in error_msg.lower()
    assert "invalid_backend" in error_msg


def test_html_max_frames_error_message(test_env):
    """Verify HTML max frames error provides subsample examples."""
    # Create large dataset exceeding limit
    large_fields = [np.random.rand(test_env.n_bins) for _ in range(2000)]

    with pytest.raises(ValueError) as exc_info:
        animate_fields(test_env, large_fields, backend="html")

    error_msg = str(exc_info.value)
    # Check for diagnostic info
    assert "2000" in error_msg or "2,000" in error_msg
    # Check for solutions with code examples
    assert "fields[::10]" in error_msg or "subsample" in error_msg.lower()
    assert "backend='video'" in error_msg or "backend='napari'" in error_msg


def test_target_fps_exceeds_source_error_message():
    """Verify subsample_frames fps validation is clear."""
    fields = [np.array([1, 2, 3]) for _ in range(100)]

    with pytest.raises(ValueError) as exc_info:
        subsample_frames(fields, target_fps=100, source_fps=50)

    error_msg = str(exc_info.value)
    # Check for diagnostic info
    assert "100" in error_msg
    assert "50" in error_msg
    assert "exceed" in error_msg.lower()


def test_n_workers_validation_error_message(test_env, test_fields, tmp_path):
    """Verify n_workers validation shows actual value."""
    from neurospatial.animation.backends.video_backend import render_video

    # Mock ffmpeg availability to test n_workers validation
    with patch(
        "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
        return_value=True,
    ):
        with pytest.raises(ValueError) as exc_info:
            render_video(test_env, test_fields, tmp_path / "out.mp4", n_workers=-1)

        error_msg = str(exc_info.value)
        assert "n_workers" in error_msg
        assert "positive" in error_msg.lower()
        assert "-1" in error_msg


def test_image_format_validation_error_message(test_env, test_fields):
    """Verify image format validation shows valid options."""
    from neurospatial.animation.rendering import render_field_to_image_bytes

    with pytest.raises(ValueError) as exc_info:
        render_field_to_image_bytes(
            test_env, test_fields[0], "viridis", 0, 1, image_format="gif"
        )

    error_msg = str(exc_info.value)
    assert "image_format" in error_msg
    assert "png" in error_msg.lower()
    assert "jpeg" in error_msg.lower()
    assert "gif" in error_msg


# ============================================================================
# Large Dataset Auto-Selection Errors
# ============================================================================


def test_large_dataset_no_napari_error_message(test_env):
    """Verify large dataset error provides comprehensive options."""
    # Create very large dataset
    large_fields = [np.random.rand(test_env.n_bins) for _ in range(50_000)]

    # Mock napari as unavailable
    with patch(
        "neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", False
    ):
        with pytest.raises(RuntimeError) as exc_info:
            animate_fields(test_env, large_fields, backend="auto")

        error_msg = str(exc_info.value)
        # Check for diagnostic info
        assert "50,000" in error_msg or "50000" in error_msg
        assert "10,000" in error_msg  # Threshold
        # Check for installation instructions
        assert "pip install napari[all]" in error_msg
        # Check for workaround options
        assert "subsample" in error_msg.lower()
        assert "backend='video'" in error_msg or "backend='html'" in error_msg


def test_no_backend_available_error_message(test_env, test_fields):
    """Verify no suitable backend error lists all options."""
    # Mock all interactive backends as unavailable
    # get_ipython is imported from IPython inside _select_backend
    mock_ipython = MagicMock()
    mock_ipython.get_ipython.side_effect = ImportError

    with (
        patch("neurospatial.animation.backends.napari_backend.NAPARI_AVAILABLE", False),
        patch(
            "neurospatial.animation.backends.widget_backend.IPYWIDGETS_AVAILABLE", False
        ),
        patch.dict(sys.modules, {"IPython": mock_ipython}),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            animate_fields(test_env, test_fields, backend="auto")

        error_msg = str(exc_info.value)
        # Check for installation options
        assert "napari" in error_msg.lower()
        assert "ipywidgets" in error_msg.lower()
        # Check for export options
        assert "save_path='output.mp4'" in error_msg
        assert "save_path='output.html'" in error_msg
        assert "ffmpeg" in error_msg.lower()


# ============================================================================
# Napari-Specific Validation Errors
# ============================================================================


def test_trajectory_shape_error_message(test_env, test_fields):
    """Verify trajectory shape validation shows actual shape."""
    pytest.importorskip("napari")

    # 1D trajectory (invalid)
    invalid_trajectory = np.random.rand(100)

    with pytest.raises(ValueError) as exc_info:
        animate_fields(
            test_env,
            test_fields,
            backend="napari",
            overlay_trajectory=invalid_trajectory,
        )

    error_msg = str(exc_info.value)
    assert "overlay_trajectory" in error_msg
    assert "2D" in error_msg
    assert str(invalid_trajectory.shape) in error_msg


def test_multi_field_layout_required_error_message():
    """Verify multi-field layout requirement is clear."""
    pytest.importorskip("napari")
    from neurospatial.animation.backends.napari_backend import render_napari

    # Create environment with known bin count
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Multi-field input without layout parameter
    # Must be list of lists of arrays (not ndarray) to trigger multi-field detection
    field_sequences = [
        [np.random.rand(env.n_bins) for _ in range(10)],
        [np.random.rand(env.n_bins) for _ in range(10)],
    ]

    # Call napari backend directly to bypass core validation
    with pytest.raises(ValueError) as exc_info:
        render_napari(env, field_sequences)

    error_msg = str(exc_info.value)
    assert "layout" in error_msg.lower()
    assert "horizontal" in error_msg.lower()
    assert "vertical" in error_msg.lower()
    assert "grid" in error_msg.lower()


def test_multi_field_length_mismatch_error_message():
    """Verify multi-field length mismatch shows actual lengths."""
    pytest.importorskip("napari")
    from neurospatial.animation.backends.napari_backend import render_napari

    # Create environment with known bin count
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Different length sequences
    field_sequences = [
        [np.random.rand(env.n_bins) for _ in range(10)],
        [np.random.rand(env.n_bins) for _ in range(15)],
    ]

    # Call napari backend directly to bypass core validation
    with pytest.raises(ValueError) as exc_info:
        render_napari(env, field_sequences, layout="horizontal")

    error_msg = str(exc_info.value)
    assert "length" in error_msg.lower()
    assert "10" in error_msg
    assert "15" in error_msg


def test_multi_field_layer_names_count_error_message():
    """Verify layer_names count mismatch shows counts."""
    pytest.importorskip("napari")
    from neurospatial.animation.backends.napari_backend import render_napari

    # Create environment with known bin count
    positions = np.random.randn(100, 2) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    field_sequences = [
        [np.random.rand(env.n_bins) for _ in range(10)],
        [np.random.rand(env.n_bins) for _ in range(10)],
    ]

    # Call napari backend directly to bypass core validation
    with pytest.raises(ValueError) as exc_info:
        render_napari(
            env,
            field_sequences,
            layout="horizontal",
            layer_names=["Field A"],  # Only 1 name for 2 sequences
        )

    error_msg = str(exc_info.value)
    assert "layer_names" in error_msg
    assert "1" in error_msg  # Provided count
    assert "2" in error_msg  # Expected count
