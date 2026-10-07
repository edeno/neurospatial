"""Public first-run errors explain the inputs and corrected calls."""

from dataclasses import dataclass

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rate
from neurospatial.encoding._validation import validate_spike_times, validate_trajectory


@dataclass(frozen=True)
class Recording:
    env: Environment
    times: np.ndarray
    positions: np.ndarray
    spike_times: np.ndarray


@pytest.fixture(scope="module")
def recording():
    times = np.arange(1800) / 30.0
    positions = (
        50 + 40 * np.c_[np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    )
    env = Environment.from_samples(positions, bin_size=4.0, units="cm")
    spikes = np.sort(np.random.default_rng(0).uniform(0, times[-1], 300))
    return Recording(env, times, positions, spikes)


def test_missing_env_names_the_call(recording):
    r = recording
    with pytest.raises(TypeError, match="expects an Environment") as caught:
        compute_spatial_rate(r.spike_times, r.times, r.positions)
    message = str(caught.value)
    assert "compute_spatial_rate" in message
    assert "ndarray" in message
    assert message.splitlines()[-1].startswith("Fix: ")
    assert "compute_spatial_rate(env, spike_times, times, positions)" in message


def test_1d_positions_on_2d_env(recording):
    r = recording
    with pytest.raises(ValueError, match=r"shape \(1800,\).*2-D") as caught:
        compute_spatial_rate(r.env, r.spike_times, r.times, r.positions[:, 0])
    assert "Fix:" in str(caught.value)
    assert "grid_edges" not in str(caught.value)


def test_swapped_times_positions(recording):
    r = recording
    with pytest.raises(ValueError, match="did you pass positions before times"):
        compute_spatial_rate(r.env, r.spike_times, r.positions, r.times)


def test_trajectory_reports_length_and_dimensions_together(recording):
    r = recording
    positions = np.zeros((len(r.times) - 1, 3))
    with pytest.raises(ValueError) as caught:
        compute_spatial_rate(r.env, r.spike_times, r.times, positions)
    message = str(caught.value)
    assert "times length (1800)" in message
    assert "positions length (1799)" in message
    assert "2-D" in message
    assert len([line for line in message.splitlines() if line.startswith("- ")]) == 2
    assert message.splitlines()[-1].startswith("Fix: ")


def test_trajectory_reports_time_and_heading_problems_together():
    with pytest.raises(ValueError) as caught:
        validate_trajectory(
            np.array([1.0, 0.0, np.nan]),
            headings=np.ones((2, 2)),
            context="compute_directional_rate",
        )
    message = str(caught.value)
    assert "compute_directional_rate" in message
    for problem in [
        "finite",
        "monotonically non-decreasing",
        "headings must be 1D",
        "headings length (2)",
    ]:
        assert problem in message
    assert "Fix:" in message


def test_spike_validation_reports_all_problems():
    with pytest.raises(ValueError) as caught:
        validate_spike_times(np.array([1.0, -1.0, np.nan]), context="decode_position")
    message = str(caught.value)
    assert "decode_position" in message
    assert "finite" in message
    assert "non-negative" in message
    assert "monotonically non-decreasing" in message
    assert "Fix:" in message


def test_valid_linear_trajectory_is_accepted():
    validate_trajectory(
        np.arange(10.0),
        positions=np.arange(10.0),
        n_dims=1,
        context="compute_spatial_rate",
    )


@pytest.mark.parametrize("bad_shape", [(10, 3), (10, 2, 1)])
def test_invalid_coordinate_dimensions_name_the_call(bad_shape):
    with pytest.raises(ValueError) as caught:
        validate_trajectory(
            np.arange(10.0),
            positions=np.ones(bad_shape),
            n_dims=2,
            context="compute_view_rate",
        )
    assert "compute_view_rate" in str(caught.value)
    assert "positions" in str(caught.value)
    assert "Fix:" in str(caught.value)
