"""Polar environments locate ``(distance, angle)`` rows in the cell that holds them.

An angle names a direction, so an angle a whole number of turns outside the
angle range finds the same bin as the angle inside it. Rows beyond the
distance range, in an excluded sector of a partial window, or with a NaN
coordinate have no bin (``-1``). Every per-sample method (``bin_at``,
``contains``, ``occupancy``, the bin sequences and ``transitions``) and every
``points=`` query uses this one lookup.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment, Regions

TURN = 2 * np.pi


def _circle() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
        units="cm",
    )


def _frontal() -> Environment:
    """Only the half-plane ahead of the animal: -π/2 to π/2."""
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 40.0),
        angle_range=(-np.pi / 2, np.pi / 2),
        distance_bin_size=10.0,
        angle_bin_size=np.pi / 4,
        circular_angle=False,
        units="cm",
    )


@pytest.mark.parametrize("build", [_circle, _frontal], ids=["circle", "frontal"])
@pytest.mark.parametrize("turns", [-2, -1, 1, 3])
def test_points_queries_wrap_whole_turns(build, turns):
    env = build()
    for index, center in enumerate(env.bin_centers):
        wrapped = center + np.array([0.0, turns * TURN])
        np.testing.assert_array_equal(
            env.distance_to(points=[wrapped]), env.distance_to(bins=[index])
        )


def test_the_seam_angle_is_one_direction_on_a_full_circle():
    env = _circle()
    np.testing.assert_array_equal(
        env.distance_to(points=[[15.0, np.pi]]),
        env.distance_to(points=[[15.0, -np.pi]]),
    )


def test_point_regions_wrap_angles():
    regions = Regions()
    regions.add("ahead", point=(25.0, 0.4))
    regions.add("ahead_turned", point=(25.0, 0.4 + TURN))
    env = _circle().with_regions(regions)
    (bin_index,) = env.bins_in_region("ahead")
    np.testing.assert_array_equal(env.bins_in_region("ahead_turned"), [bin_index])


def test_an_excluded_sector_stays_outside_a_partial_window():
    env = _frontal()
    for angle in (np.pi, -3 * np.pi / 4, np.pi + TURN):
        with pytest.raises(ValueError, match="outside the environment"):
            env.distance_to(points=[[15.0, angle]])


def _containing_bin(env: Environment, distance: float, angle: float) -> int:
    """The bin whose (distance, angle) cell holds the row, found from grid edges."""
    distance_edges, angle_edges = env.grid_edges
    low = angle_edges[0]
    angle = low + np.remainder(angle - low, TURN)
    i = np.searchsorted(distance_edges, distance, side="right") - 1
    j = np.searchsorted(angle_edges, angle, side="right") - 1
    if not (0 <= i < len(distance_edges) - 1 and 0 <= j < len(angle_edges) - 1):
        return -1
    center = [
        (distance_edges[i] + distance_edges[i + 1]) / 2,
        (angle_edges[j] + angle_edges[j + 1]) / 2,
    ]
    (index,) = np.flatnonzero(np.all(np.isclose(env.bin_centers, center), axis=1))
    return int(index)


@pytest.mark.parametrize("build", [_circle, _frontal], ids=["circle", "frontal"])
def test_bin_at_finds_the_cell_that_holds_each_row(build):
    env = build()
    rng = np.random.default_rng(0)
    rows = np.column_stack(
        (rng.uniform(-5.0, 45.0, 500), rng.uniform(-3 * np.pi, 3 * np.pi, 500))
    )
    expected = [_containing_bin(env, d, a) for d, a in rows]
    np.testing.assert_array_equal(env.bin_at(rows), expected)
    np.testing.assert_array_equal(env.contains(rows), np.asarray(expected) >= 0)
    # Rows inside and outside both occur.
    assert 0 < np.mean(np.asarray(expected) >= 0) < 1


@pytest.mark.parametrize("build", [_circle, _frontal], ids=["circle", "frontal"])
def test_whole_turns_name_the_same_direction(build):
    env = build()
    centers = env.bin_centers
    for turns in (-2, -1, 1, 3):
        shifted = centers + np.array([0.0, turns * TURN])
        np.testing.assert_array_equal(env.bin_at(shifted), np.arange(env.n_bins))


def test_missing_and_infinite_rows_have_no_bin():
    env = _circle()
    rows = np.array([[np.nan, 0.0], [15.0, np.nan], [np.inf, 0.0], [15.0, np.inf]])
    np.testing.assert_array_equal(env.bin_at(rows), [-1, -1, -1, -1])


def _samples() -> tuple[np.ndarray, np.ndarray]:
    """An object circling the animal: angles keep growing past π."""
    times = np.arange(0.0, 4.0, 0.1)
    angles = np.linspace(0.0, 3 * TURN, times.size)
    distances = np.where(np.arange(times.size) % 7 == 3, 50.0, 15.0)  # some beyond
    return times, np.column_stack((distances, angles))


def test_occupancy_gives_each_interval_to_its_first_sample_bin():
    env = _circle()
    times, positions = _samples()
    bins = env.bin_at(positions)
    expected = np.zeros(env.n_bins)
    np.add.at(expected, bins[:-1][bins[:-1] >= 0], np.diff(times)[bins[:-1] >= 0])
    np.testing.assert_allclose(env.occupancy(times, positions), expected)
    # Time beyond the 40 cm surround is not counted anywhere.
    assert env.occupancy(times, positions).sum() < times[-1] - times[0]


def test_bin_sequences_and_transitions_follow_bin_at():
    env = _circle()
    times, positions = _samples()
    bins = env.bin_at(positions)
    np.testing.assert_array_equal(env.bin_sequence(times, positions, dedup=False), bins)
    runs = env.bin_sequence_with_runs(times, positions)
    np.testing.assert_array_equal(np.repeat(runs.bins, runs.run_lengths), bins)
    counts = env.transitions(
        times=times, positions=positions, normalize=False, allow_teleports=True
    ).toarray()
    keep = (bins[:-1] >= 0) & (bins[1:] >= 0)
    expected = np.zeros((env.n_bins, env.n_bins))
    np.add.at(expected, (bins[:-1][keep], bins[1:][keep]), 1)
    np.testing.assert_array_equal(counts, expected)
