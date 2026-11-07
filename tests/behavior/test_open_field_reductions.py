"""Geodesic goal measures reduce to their Euclidean forms in an open arena.

With no wall between a position and a goal, the shortest path inside the
arena is the straight line, so the geodesic versions must give the published
Euclidean answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.behavior import geodesic_voronoi_labels, instantaneous_goal_alignment


@pytest.fixture(scope="module")
def open_box():
    """A 100 x 100 cm square arena of 5 cm bins, with no interior wall."""
    x, y = np.meshgrid(np.arange(0.5, 100.0), np.arange(0.5, 100.0))
    return Environment.from_samples(
        np.column_stack([x.ravel(), y.ravel()]), bin_size=5.0, units="cm"
    )


def test_voronoi_labels_are_the_euclidean_voronoi_partition(open_box):
    goals = [np.array([13.0, 71.0]), np.array([62.0, 24.0]), np.array([88.0, 83.0])]
    labels = geodesic_voronoi_labels(open_box, goals=goals)

    # Goal points are measured from the center of the bin that holds them.
    anchors = open_box.bin_centers[[open_box.bin_at(g[None])[0] for g in goals]]
    distances = np.linalg.norm(open_box.bin_centers[:, None, :] - anchors[None], axis=2)
    nearest = np.sort(distances, axis=1)
    tied = np.isclose(nearest[:, 0], nearest[:, 1], rtol=1e-9)
    expected = np.where(tied, -1, np.argmin(distances, axis=1))

    np.testing.assert_array_equal(labels, expected)
    assert np.sum(expected >= 0) > 0.9 * open_box.n_bins


def test_goal_alignment_geodesic_equals_euclidean(open_box):
    rng = np.random.default_rng(2)
    n_samples = 200
    times = np.arange(n_samples) * 0.05
    positions = rng.uniform(10.0, 90.0, (n_samples, 2))
    headings = rng.uniform(-np.pi, np.pi, n_samples)
    goal = np.array([47.5, 52.5])

    kwargs = {"goal": goal, "headings": headings, "max_gap": None}
    geodesic = instantaneous_goal_alignment(
        open_box, times, positions, metric="geodesic", **kwargs
    )
    euclidean = instantaneous_goal_alignment(
        open_box, times, positions, metric="euclidean", **kwargs
    )
    bearing = np.arctan2(goal[1] - positions[:, 1], goal[0] - positions[:, 0])

    finite = np.isfinite(euclidean)
    assert finite.sum() > 0.9 * n_samples
    np.testing.assert_allclose(geodesic, euclidean, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(
        euclidean[finite], np.cos(headings - bearing)[finite], atol=1e-9
    )
