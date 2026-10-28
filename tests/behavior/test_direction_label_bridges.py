"""Tests for the lap/run -> direction-label bridge helpers.

Covers ``laps_to_direction_labels``, ``runs_to_direction_labels``, and
``running_direction_labels`` and confirms their output is drop-in for
``compute_directional_place_fields``.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.behavior.segmentation import (
    Lap,
    Run,
    laps_to_direction_labels,
    running_direction_labels,
    runs_to_direction_labels,
)
from neurospatial.encoding.spatial import compute_directional_place_fields


def test_laps_to_direction_labels_shape():
    """Output length matches times; unlabeled samples are None."""
    times = np.linspace(0.0, 10.0, 11)
    laps = [
        Lap(start_time=0.0, end_time=3.0, direction="clockwise", overlap_score=1.0),
        Lap(
            start_time=7.0,
            end_time=10.0,
            direction="counter-clockwise",
            overlap_score=1.0,
        ),
    ]

    labels = laps_to_direction_labels(laps, times)

    # Shape / dtype contract for compute_directional_place_fields.
    assert labels.shape == (len(times),)
    assert labels.dtype == object

    # Labeled samples match the lap directions.
    assert labels[0] == "clockwise"  # t=0.0
    assert labels[2] == "clockwise"  # t=2.0
    # The lap's last sample starts the interval that leaves the lap.
    assert labels[3] is None  # t=3.0
    assert labels[9] == "counter-clockwise"  # t=9.0
    assert labels[-1] is None  # t=10.0

    # The unlabeled gap between laps is None.
    assert labels[5] is None  # t=5.0


def test_laps_between_and_after_are_none():
    """Samples between laps and the last sample are None, not a string."""
    times = np.linspace(0.0, 10.0, 11)
    laps = [
        Lap(0.0, 4.0, "clockwise", 1.0),
        Lap(6.0, 10.0, "counter-clockwise", 1.0),
    ]

    labels = laps_to_direction_labels(laps, times)

    assert labels[0] == "clockwise"
    assert labels[5] is None
    assert labels[-1] is None


def test_laps_unknown_direction_are_none():
    """Laps whose direction is unknown leave their samples None."""
    times = np.linspace(0.0, 10.0, 11)
    laps = [Lap(start_time=0.0, end_time=4.0, direction="unknown", overlap_score=1.0)]

    labels = laps_to_direction_labels(laps, times)
    assert all(label is None for label in labels)

    # There is no producer-side marker for unknown laps.
    with pytest.raises(TypeError):
        laps_to_direction_labels(laps, times, unknown_label="x")  # type: ignore[call-arg]


def test_runs_to_direction_labels_scalar_and_sequence():
    """Scalar label fills all runs; sequence labels per run; None elsewhere."""
    times = np.linspace(0.0, 10.0, 11)
    bins = np.array([0, 1], dtype=np.int64)
    runs = [
        Run(0.0, 3.0, "home", "reward", "reached", bins),
        Run(7.0, 10.0, "reward", "home", "timeout", bins),
    ]

    scalar = runs_to_direction_labels(runs, times)
    assert scalar[0] == "run"
    assert scalar[9] == "run"
    assert scalar[-1] is None  # a run's last sample leaves the run
    assert scalar[5] is None

    per_run = runs_to_direction_labels(runs, times, labels=["outbound", "inbound"])
    assert per_run[0] == "outbound"
    assert per_run[9] == "inbound"
    assert per_run[5] is None

    # success_only drops the failed run.
    filtered = runs_to_direction_labels(runs, times, success_only=True)
    assert filtered[0] == "run"
    assert filtered[9] is None


def test_runs_to_direction_labels_ndarray_labels():
    """ndarray labels are treated per-run (not silently broadcast as a scalar)."""
    times = np.linspace(0.0, 10.0, 11)
    bins = np.array([0, 1], dtype=np.int64)
    runs = [
        Run(0.0, 3.0, "home", "reward", "reached", bins),
        Run(7.0, 10.0, "reward", "home", "reached", bins),
    ]

    # Correct-length ndarray must match the list-input result element-by-element.
    list_labels = runs_to_direction_labels(runs, times, labels=["outbound", "inbound"])
    array_labels = runs_to_direction_labels(
        runs, times, labels=np.array(["outbound", "inbound"], dtype=object)
    )
    assert np.array_equal(array_labels, list_labels)
    assert array_labels[0] == "outbound"
    assert array_labels[9] == "inbound"
    assert array_labels[5] is None

    # Wrong-length ndarray must raise the length-mismatch ValueError, not
    # silently broadcast the whole array into every run's label.
    with pytest.raises(ValueError, match="one entry per run"):
        runs_to_direction_labels(
            runs, times, labels=np.array(["outbound", "inbound", "extra"], dtype=object)
        )


def test_running_direction_labels_inbound_outbound(out_and_back_track):
    """Out-and-back trajectory yields contiguous outbound then inbound runs."""
    fx = out_and_back_track

    labels = running_direction_labels(
        fx["position_bins"],
        fx["times"],
        fx["env"],
        start_region=fx["start_region"],
        end_regions=fx["end_region"],
    )

    assert labels.shape == (len(fx["times"]),)
    assert labels.dtype == object
    assert set(labels) <= {"inbound", "outbound", None}
    assert "outbound" in labels
    assert "inbound" in labels

    # Outbound must come before inbound, with None at the turnaround.
    out_idx = np.where(labels == "outbound")[0]
    in_idx = np.where(labels == "inbound")[0]
    assert out_idx.max() < in_idx.min()

    # Each label group is a single contiguous block (no interleaving).
    assert np.all(np.diff(out_idx) == 1)
    assert np.all(np.diff(in_idx) == 1)

    # The turnaround samples between the two blocks are None.
    assert any(label is None for label in labels[out_idx.max() + 1 : in_idx.min()])


def test_labels_feed_directional_place_fields(out_and_back_track):
    """Produced labels are accepted by compute_directional_place_fields."""
    fx = out_and_back_track
    env = fx["env"]
    times = fx["times"]

    labels = running_direction_labels(
        fx["position_bins"],
        times,
        env,
        start_region=fx["start_region"],
        end_regions=fx["end_region"],
    )

    rng = np.random.default_rng(0)
    spike_times = np.sort(rng.uniform(times[0], times[-1], 30))

    # No reshaping: pass labels straight through.
    result = compute_directional_place_fields(
        env,
        spike_times,
        times,
        fx["positions"],
        labels,
        bandwidth=10.0,
    )

    # None is excluded; inbound/outbound become directional fields.
    assert result.labels == ("inbound", "outbound")
    for rate in result.firing_rates.values():
        assert rate.shape == (env.n_bins,)
