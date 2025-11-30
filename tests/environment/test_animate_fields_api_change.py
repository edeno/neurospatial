"""Tests for Environment.animate_fields() API change (Task 4.1).

This module tests the new API where:
- `frame_times` is required (no default)
- `speed` parameter replaces `fps`
- `fps` is no longer accepted as a parameter

These tests verify the signature change from visualization.py.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from neurospatial import Environment

# Mark all tests as integration tests
pytestmark = pytest.mark.integration


class TestEnvironmentAnimateFieldsApiChange:
    """Tests for the new Environment.animate_fields() API (Task 4.1)."""

    @pytest.fixture
    def env_and_fields(self):
        """Create a simple environment and fields for testing."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (100, 2))
        env = Environment.from_samples(positions, bin_size=10.0)
        n_frames = 5
        fields = [rng.random(env.n_bins) for _ in range(n_frames)]
        # Create frame_times spanning 1 second at 5 Hz (n_frames=5)
        frame_times = np.linspace(0, 1.0, n_frames)
        return env, fields, frame_times

    @patch("neurospatial.animation.core.animate_fields")
    def test_frame_times_is_required(self, mock_animate, env_and_fields):
        """Test that frame_times is a required parameter (no default).

        After Task 4.1, calling animate_fields() without frame_times should
        raise a TypeError because it's a required keyword-only argument.
        """
        env, fields, _ = env_and_fields

        # Should raise TypeError because frame_times is required
        with pytest.raises(TypeError, match="frame_times"):
            env.animate_fields(fields, backend="html")

    @patch("neurospatial.animation.core.animate_fields")
    def test_speed_parameter_exists(self, mock_animate, env_and_fields):
        """Test that speed parameter is accepted."""
        env, fields, frame_times = env_and_fields

        # Should accept speed parameter without error
        env.animate_fields(fields, frame_times=frame_times, speed=0.5, backend="html")

        # Verify speed was passed to core
        call_kwargs = mock_animate.call_args.kwargs
        assert "speed" in call_kwargs
        assert call_kwargs["speed"] == 0.5

    @patch("neurospatial.animation.core.animate_fields")
    def test_speed_default_is_one(self, mock_animate, env_and_fields):
        """Test that speed defaults to 1.0 (real-time)."""
        env, fields, frame_times = env_and_fields

        # Call without explicit speed
        env.animate_fields(fields, frame_times=frame_times, backend="html")

        # Verify default speed=1.0 was passed to core
        call_kwargs = mock_animate.call_args.kwargs
        assert call_kwargs["speed"] == 1.0

    @patch("neurospatial.animation.core.animate_fields")
    def test_fps_parameter_not_in_signature(self, mock_animate, env_and_fields):
        """Test that fps is NOT a direct parameter in the method signature.

        After Task 4.1, fps should not be a direct keyword argument.
        It may still be passed via **kwargs but should not be explicitly
        handled in the Environment method.
        """
        env, _fields, _frame_times = env_and_fields

        # The method should NOT have fps as a named parameter
        import inspect

        sig = inspect.signature(env.animate_fields)
        param_names = list(sig.parameters.keys())

        # fps should not be in the parameter list
        assert "fps" not in param_names, (
            "fps should not be a direct parameter after Task 4.1"
        )

        # speed SHOULD be in the parameter list
        assert "speed" in param_names, "speed should be a direct parameter"

        # frame_times SHOULD be in the parameter list
        assert "frame_times" in param_names, "frame_times should be a direct parameter"

    @patch("neurospatial.animation.core.animate_fields")
    def test_frame_times_passed_to_core(self, mock_animate, env_and_fields):
        """Test that frame_times is passed through to core dispatcher."""
        env, fields, frame_times = env_and_fields

        env.animate_fields(fields, frame_times=frame_times, backend="html")

        call_kwargs = mock_animate.call_args.kwargs
        assert "frame_times" in call_kwargs
        np.testing.assert_array_equal(call_kwargs["frame_times"], frame_times)

    @patch("neurospatial.animation.core.animate_fields")
    def test_speed_with_slow_motion(self, mock_animate, env_and_fields):
        """Test slow motion playback with speed < 1.0."""
        env, fields, frame_times = env_and_fields

        env.animate_fields(fields, frame_times=frame_times, speed=0.1, backend="html")

        call_kwargs = mock_animate.call_args.kwargs
        assert call_kwargs["speed"] == 0.1

    @patch("neurospatial.animation.core.animate_fields")
    def test_speed_with_fast_forward(self, mock_animate, env_and_fields):
        """Test fast forward playback with speed > 1.0."""
        env, fields, frame_times = env_and_fields

        env.animate_fields(fields, frame_times=frame_times, speed=2.0, backend="html")

        call_kwargs = mock_animate.call_args.kwargs
        assert call_kwargs["speed"] == 2.0

    @patch("neurospatial.animation.core.animate_fields")
    def test_all_core_parameters_still_forwarded(self, mock_animate, env_and_fields):
        """Test that all other core parameters are still forwarded correctly."""
        env, fields, frame_times = env_and_fields

        env.animate_fields(
            fields,
            frame_times=frame_times,
            speed=1.5,
            backend="video",
            save_path="test.mp4",
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            frame_labels=["A", "B", "C", "D", "E"],
            dpi=150,
            n_workers=2,
        )

        call_kwargs = mock_animate.call_args.kwargs
        assert call_kwargs["backend"] == "video"
        assert call_kwargs["save_path"] == "test.mp4"
        assert call_kwargs["cmap"] == "hot"
        assert call_kwargs["vmin"] == 0.0
        assert call_kwargs["vmax"] == 1.0
        assert call_kwargs["dpi"] == 150
        assert call_kwargs["n_workers"] == 2
        assert call_kwargs["speed"] == 1.5

    @patch("neurospatial.animation.core.animate_fields")
    def test_no_frame_times_synthesis_from_fps(self, mock_animate, env_and_fields):
        """Test that frame_times is NOT synthesized from fps.

        The old behavior (synthesizing frame_times from fps) should be removed.
        """
        env, fields, _ = env_and_fields

        # Trying to use fps without frame_times should fail
        # because frame_times is required
        with pytest.raises(TypeError):
            env.animate_fields(fields, backend="html")

    @patch("neurospatial.animation.core.animate_fields")
    def test_works_with_2d_ndarray_fields(self, mock_animate, env_and_fields):
        """Test that method works with 2D ndarray fields input."""
        env, _, frame_times = env_and_fields
        rng = np.random.default_rng(42)

        # Pass ndarray instead of list
        fields_array = rng.random((5, env.n_bins))

        env.animate_fields(fields_array, frame_times=frame_times, backend="html")

        mock_animate.assert_called_once()
        passed_fields = mock_animate.call_args.kwargs["fields"]
        assert isinstance(passed_fields, np.ndarray)
        assert passed_fields.shape == (5, env.n_bins)
