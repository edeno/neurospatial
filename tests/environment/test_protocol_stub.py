"""Verify the EnvironmentProtocol stubs agree with the real Environment methods."""

from __future__ import annotations

import inspect

import numpy as np

from neurospatial import Environment
from neurospatial.environment._protocols import EnvironmentProtocol


def _make_env() -> Environment:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 50, (200, 2))
    return Environment.from_samples(positions, bin_size=5.0)


def test_protocol_stub_matches_impl():
    """animate_fields and occupancy stubs match the real Environment methods."""
    env = _make_env()

    # --- animate_fields ---
    proto_animate = inspect.signature(EnvironmentProtocol.animate_fields).parameters
    real_animate = inspect.signature(env.animate_fields).parameters

    # frame_times is a required keyword-only parameter (no default) in both.
    assert proto_animate["frame_times"].kind is inspect.Parameter.KEYWORD_ONLY
    assert proto_animate["frame_times"].default is inspect.Parameter.empty
    assert real_animate["frame_times"].default is inspect.Parameter.empty

    # No fps; speed is present with default 1.0 in both.
    assert "fps" not in proto_animate
    assert "fps" not in real_animate
    assert "speed" in proto_animate
    assert proto_animate["speed"].default == real_animate["speed"].default == 1.0

    # --- occupancy ---
    proto_occ = inspect.signature(EnvironmentProtocol.occupancy).parameters
    real_occ = inspect.signature(env.occupancy).parameters
    assert proto_occ["max_gap"].default == 0.5
    assert proto_occ["max_gap"].default == real_occ["max_gap"].default
