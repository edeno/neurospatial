"""Environment.animate_fields() takes frame times and a playback speed.

- ``frame_times`` is required (no default), so every frame has a time.
- ``speed`` sets the playback rate relative to real time; there is no
  ``fps`` parameter.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np
import pytest

from neurospatial import Environment

pytestmark = pytest.mark.integration


@pytest.fixture
def env_and_fields():
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, (100, 2))
    env = Environment.from_samples(positions, bin_size=10.0)
    n_frames = 5
    fields = [rng.random(env.n_bins) for _ in range(n_frames)]
    frame_times = np.linspace(0, 1.0, n_frames)
    return env, fields, frame_times


def test_frame_times_is_required(env_and_fields):
    """``frame_times`` is keyword-only and required: omitting it raises."""
    env, fields, _ = env_and_fields
    with pytest.raises(TypeError, match="frame_times"):
        env.animate_fields(fields, backend="html")


def test_fps_is_gone_and_speed_is_present():
    """Playback rate is set by ``speed``; there is no ``fps`` parameter."""
    sig = inspect.signature(Environment.animate_fields)
    param_names = list(sig.parameters.keys())
    assert "fps" not in param_names
    assert "speed" in param_names
    assert "frame_times" in param_names


@patch("neurospatial.animation.core.animate_fields")
def test_speed_is_forwarded_with_correct_default(mock_animate, env_and_fields):
    """``speed`` defaults to 1.0 and a non-default value is forwarded as given."""
    env, fields, frame_times = env_and_fields

    env.animate_fields(fields, frame_times=frame_times, backend="html")
    assert mock_animate.call_args.kwargs["speed"] == 1.0

    env.animate_fields(fields, frame_times=frame_times, speed=2.0, backend="html")
    assert mock_animate.call_args.kwargs["speed"] == 2.0
