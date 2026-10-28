"""Gap-free outputs captured from baseline commit 35313646.

Every segmentation/window/sequence entry point is compared to a detached
baseline capture using identical fixture builders, with exact segment/index
structure and rtol=1e-12 for floating metrics. The ``compute_decision_analysis``
entry was captured again when each goal region began to be represented by its
centroid bin (the bin ``trials_to_region_arrays`` uses), which moves the
goal-region boundary. The ``segment_trials`` entry was captured again when
trials began at departure from the start region instead of entry, and the
``segment_by_velocity``, ``compute_pre_decision_metrics`` and
``compute_vte_trial`` entries when speed became the Gaussian-smoothed velocity
of ``compute_speed``. ``running_direction_labels`` marks samples outside
every run with ``None`` instead of ``"other"``; its golden is compared with
that one substitution and is otherwise unchanged. The decision and VTE
entries (``detect_boundary_crossings``, ``compute_pre_decision_metrics``,
``compute_decision_analysis``, ``compute_vte_trial``, ``compute_vte_session``)
were captured again when boundary crossings became arrays, the boundary
distance became ``(d2 - d1) / 2``, the chosen goal became the first goal
entered after the decision entry, result fields were renamed
(``min_observed_speed``, ``distance_to_decision_boundary``) and VTE results
gained a per-trial ``status`` with derived session statistics. The lap entries
(``detect_laps_region``, ``detect_laps_auto``, ``detect_laps_reference``) were
captured again when a lap had to go round the loop and a turn back started the
next lap afresh: the joined lap fixture jumps back to the lap start at 100 s,
which now ends the region lap begun at 88.6 s on the other side of the start
region and starts the template laps at the jump. ``detect_goal_directed_runs``
was captured again when each approach to the goal became its own row.
``compute_decision_analysis`` was captured again when decision boundaries
began to use the geodesic distance of the path module (the shortest path
inside the environment between bin centers) instead of the bin graph.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ._segmentation_snapshots import capture_segmentation_outputs, structural_snapshot


@pytest.fixture(scope="session")
def gap_free_outputs(continuous_pause_track, continuous_lap_track):
    return capture_segmentation_outputs(continuous_pause_track, continuous_lap_track)


@pytest.fixture(scope="session")
def gap_free_goldens():
    path = Path(__file__).parent / "data" / "segmentation_gap_free.npz"
    with np.load(path, allow_pickle=False) as archive:
        goldens = {name: json.loads(archive[name].item()) for name in archive.files}
    # Voronoi goal labels use the existing native np.int_ API (32-bit on
    # Windows); their values and shape remain exact, and other dtypes stay strict.
    labels = goldens["compute_decision_analysis"]["fields"]["boundary"]["fields"][
        "goal_labels"
    ]
    labels["array"] = str(np.dtype(np.int_))
    running = goldens["running_direction_labels"]
    running["values"] = [
        None if label == "other" else label for label in running["values"]
    ]
    return goldens


def assert_snapshot_equal(actual, expected):
    """Check structure exactly; allow only numerical rounding in float metrics."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        if "array" in expected:
            assert actual["array"] == expected["array"]
            assert actual["shape"] == expected["shape"]
            if np.issubdtype(np.dtype(expected["array"]), np.floating):
                np.testing.assert_allclose(
                    actual["values"],
                    expected["values"],
                    rtol=1e-12,
                    atol=0,
                    equal_nan=True,
                )
            else:
                np.testing.assert_array_equal(actual["values"], expected["values"])
        else:
            for name in expected:
                assert_snapshot_equal(actual[name], expected[name])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_snapshot_equal(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0, equal_nan=True)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "name",
    [
        "detect_region_crossings",
        "detect_runs_between_regions",
        "segment_by_velocity",
        "detect_laps_region",
        "detect_laps_auto",
        "detect_laps_reference",
        "segment_trials",
        "detect_goal_directed_runs",
        "running_direction_labels",
        "detect_boundary_crossings",
        "extract_pre_decision_window",
        "compute_pre_decision_metrics",
        "compute_decision_analysis",
        "compute_vte_trial",
        "compute_vte_session",
        "bin_sequence",
        "bin_sequence_per_sample",
        "bin_sequence_with_runs",
        "transitions",
        "transitions_normalized",
    ],
)
def test_gap_free_outputs_unchanged(name, gap_free_outputs, gap_free_goldens):
    assert_snapshot_equal(
        structural_snapshot(gap_free_outputs[name]), gap_free_goldens[name]
    )
