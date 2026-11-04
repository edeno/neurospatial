"""The nearest-segment helper against straightforward per-segment loops."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.datasets._tracks import nearest_segment


def _reference(points, starts, ends):
    """Distance, segment, along and lateral by a loop kept apart from the helper."""
    distance = np.full(len(points), np.inf)
    segment = np.zeros(len(points), dtype=np.intp)
    along = np.zeros(len(points))
    lateral = np.zeros(len(points))
    for k, (start, end) in enumerate(zip(starts, ends, strict=True)):
        step = end - start
        relative = points - start
        t = np.clip(relative @ step / (step @ step), 0.0, 1.0)
        here = np.linalg.norm(points - start - t[:, None] * step, axis=1)
        cross = step[0] * relative[:, 1] - step[1] * relative[:, 0]
        closer = here < distance
        distance[closer] = here[closer]
        segment[closer] = k
        along[closer] = t[closer]
        lateral[closer] = cross[closer] / np.linalg.norm(step)
    return distance, segment, along, lateral


@pytest.mark.parametrize("n_segments", [1, 3, 9, 158])
def test_nearest_segment_matches_a_per_segment_loop(n_segments):
    rng = np.random.default_rng(n_segments)
    nodes = rng.uniform(-100, 100, (n_segments + 1, 2))
    points = rng.uniform(-150, 150, (5000, 2))
    found = nearest_segment(points, nodes[:-1], nodes[1:])
    for got, expected in zip(
        found, _reference(points, nodes[:-1], nodes[1:]), strict=True
    ):
        np.testing.assert_array_equal(got, expected)


def test_one_segment_matches_projection_on_its_unit_axis():
    # The single-edge form: clip the projection on the unit axis to the length.
    rng = np.random.default_rng(0)
    length = 110.0
    axis = np.array([np.cos(0.3), np.sin(0.3)])
    ends = np.array([[5.0, -2.0], [5.0, -2.0]]) + np.outer([0.0, length], axis)
    points = rng.uniform(-50, 150, (5000, 2))
    relative = points - ends[0]
    t = np.clip(relative @ axis / length, 0.0, 1.0)
    expected = np.linalg.norm(relative - np.outer(t * length, axis), axis=1)
    found = nearest_segment(points, ends[:1], ends[1:])
    np.testing.assert_allclose(found.distance, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(found.along, t, rtol=0, atol=1e-12)


def test_lateral_is_positive_left_of_the_segment():
    found = nearest_segment(
        np.array([[5.0, 2.0], [5.0, -3.0], [12.0, 0.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[10.0, 0.0]]),
    )
    np.testing.assert_array_equal(found.lateral, [2.0, -3.0, 0.0])
    np.testing.assert_array_equal(found.distance, [2.0, 3.0, 2.0])
    np.testing.assert_array_equal(found.along, [0.5, 0.5, 1.0])
