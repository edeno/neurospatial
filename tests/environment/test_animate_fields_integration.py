"""Integration test for Environment.animate_fields() delegation.

Environment.animate_fields() is a thin wrapper around
``neurospatial.animation.core.animate_fields``. We have exactly two
properties worth testing at the Environment-method level:

- It forwards every documented kwarg through to the core dispatcher.
- It accepts both a ``list[NDArray]`` and a ``(n_frames, n_bins)`` ndarray
  as the ``fields`` argument.

The previous version of this file exercised the same forwarding contract
once per layout type (RegularGrid, Hexagonal, 1D graph, MaskedGrid) — the
mock makes those tests indistinguishable. Real layout coverage lives in
``tests/animation/``.
"""

from __future__ import annotations

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
    n_frames = 3
    fields = [rng.random(env.n_bins) for _ in range(n_frames)]
    frame_times = np.linspace(0, 1.0, n_frames)
    return env, fields, frame_times


@patch("neurospatial.animation.core.animate_fields")
def test_forwards_all_parameters(mock_animate, env_and_fields):
    """Every documented kwarg lands on the core dispatcher unchanged."""
    env, fields, frame_times = env_and_fields
    overlays = [object()]

    env.animate_fields(
        fields,
        frame_times=frame_times,
        backend="video",
        save_path="test.mp4",
        overwrite=True,
        overlays=overlays,
        speed=0.5,
        max_playback_fps=30,
        max_gap=None,
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
        title="Session 1",
        show_regions=["goal"],
        region_alpha=0.5,
        scale_bar=True,
        dpi=150,
        codec="h265",
        n_workers=4,
    )

    (passed_env, passed_fields), call = mock_animate.call_args
    assert passed_env is env
    assert passed_fields is fields
    np.testing.assert_array_equal(call["frame_times"], frame_times)
    assert call["overlays"] is overlays
    assert call == {
        "frame_times": call["frame_times"],
        "backend": "video",
        "save_path": "test.mp4",
        "overwrite": True,
        "overlays": overlays,
        "speed": 0.5,
        "max_playback_fps": 30,
        "max_gap": None,
        "cmap": "hot",
        "vmin": 0.0,
        "vmax": 1.0,
        "title": "Session 1",
        "show_regions": ["goal"],
        "region_alpha": 0.5,
        "scale_bar": True,
        "dpi": 150,
        "codec": "h265",
        "n_workers": 4,
    }


@patch("neurospatial.animation.core.animate_fields")
def test_accepts_ndarray_input(mock_animate):
    """``fields`` accepts a 2-D ndarray as well as a list of 1-D arrays."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, (100, 2))
    env = Environment.from_samples(positions, bin_size=10.0)
    n_frames = 5
    fields = rng.random((n_frames, env.n_bins))
    frame_times = np.linspace(0, 1.0, n_frames)

    env.animate_fields(fields, frame_times=frame_times, backend="html")

    # The array itself is passed on, so a memory-mapped array stays lazy.
    assert mock_animate.call_args.args[1] is fields
