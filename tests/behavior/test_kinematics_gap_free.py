"""Gap-free kinematic outputs captured from baseline commit a51ff5eb.

Frozen inputs from the continuous 50 Hz fixture supply the inventory functions
and shared composites. Home-range calls use distinct visit counts at 50 Hz to
avoid depending on the order of tied bins. Structure and labels are exact;
floating values use rtol=1e-12.
The baseline heading API uses a scalar dt. The speed- and heading-based entries
were captured again when velocity became the Gaussian-smoothed central
difference of ``compute_speed`` (``smooth_window`` in seconds).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ._kinematics_snapshots import (
    capture_kinematics_outputs,
    load_kinematic_recording,
)
from ._segmentation_snapshots import structural_snapshot
from .test_segmentation_gap_free import assert_snapshot_equal


@pytest.fixture(scope="session")
def kinematic_outputs(continuous_recording):
    recording = load_kinematic_recording(continuous_recording)
    return capture_kinematics_outputs(recording)


@pytest.fixture(scope="session")
def kinematic_goldens():
    file = Path(__file__).parent / "data" / "kinematics_gap_free.npz"
    with np.load(file, allow_pickle=False) as archive:
        return {name: json.loads(archive[name].item()) for name in archive.files}


@pytest.mark.parametrize(
    "name",
    [
        "heading_from_velocity",
        "heading_from_velocity_smoothed",
        "pre_decision_heading_stats",
        "pre_decision_speed_stats",
        "head_sweep_from_positions",
        "heading_direction_labels",
        "heading_direction_precomputed",
        "compute_path_efficiency",
        "compute_path_efficiency_geodesic",
        "instantaneous_goal_alignment",
        "goal_bias",
        "approach_rate",
        "approach_rate_geodesic",
        "compute_goal_directed_metrics",
        "compute_trajectory_curvature",
        "compute_trajectory_curvature_no_times",
        "compute_home_range",
        "compute_home_range_no_times",
        "add_positions",
        "compute_pre_decision_metrics",
        "compute_vte_trial",
        "compute_vte_session",
    ],
)
def test_gap_free_outputs_unchanged(
    name, kinematic_outputs, kinematic_goldens, continuous_recording
):
    actual = structural_snapshot(kinematic_outputs[name])
    expected = kinematic_goldens[name]
    if name == "heading_from_velocity_smoothed":
        from neurospatial.behavior import compute_speed

        assert actual.keys() == expected.keys()
        assert actual["array"] == expected["array"]
        assert actual["shape"] == expected["shape"]
        # The golden filled samples slower than min_speed=5.0 by interpolation;
        # they now have no heading. Every other sample keeps its golden value.
        r = load_kinematic_recording(continuous_recording)
        slow = compute_speed(r.times, r.positions, smooth_window=0.04) < 5.0
        assert slow.sum() == 753
        values = np.asarray(actual["values"], dtype=float)
        assert np.isnan(values[slow]).all()
        # Actual floating-point intervals replace the old scalar median.
        # The maintainer approved this radian rounding floor near zero.
        np.testing.assert_allclose(
            values[~slow],
            np.asarray(expected["values"], dtype=float)[~slow],
            rtol=1e-12,
            atol=1e-15,
            equal_nan=True,
        )
    else:
        assert_snapshot_equal(actual, expected)
