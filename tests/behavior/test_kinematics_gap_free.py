"""Gap-free kinematic outputs captured from baseline commit a51ff5eb.

The same continuous 50 Hz fixture supplies all inventory functions and shared
composites. Structure and labels are exact; floating values use rtol=1e-12.
The baseline heading API uses a scalar dt and smoothing sigma in samples.
"""

import json
from pathlib import Path

import pytest
import numpy as np

from ._kinematics_snapshots import capture_kinematics_outputs
from ._segmentation_snapshots import structural_snapshot
from .test_segmentation_gap_free import assert_snapshot_equal


@pytest.fixture(scope="session")
def kinematic_outputs(continuous_recording):
    return capture_kinematics_outputs(continuous_recording)


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
def test_gap_free_outputs_unchanged(name, kinematic_outputs, kinematic_goldens):
    actual = structural_snapshot(kinematic_outputs[name])
    expected = kinematic_goldens[name]
    if name == "heading_from_velocity_smoothed":
        assert actual.keys() == expected.keys()
        assert actual["array"] == expected["array"]
        assert actual["shape"] == expected["shape"]
        # Actual floating-point intervals replace the old scalar median.
        # The maintainer approved this radian rounding floor near zero.
        np.testing.assert_allclose(
            actual["values"],
            expected["values"],
            rtol=1e-12,
            atol=1e-15,
            equal_nan=True,
        )
    else:
        assert_snapshot_equal(actual, expected)
