"""Boundary-vector rate maps recover simulated wall tuning and keep their counts."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.encoding import (
    BoundaryVectorRateResult,
    compute_boundary_vector_rate,
    compute_boundary_vector_rates,
)
from neurospatial.encoding._spatial_counts import (
    count_spikes_by_frame,
    start_allocated_occupancy,
)
from neurospatial.encoding.boundary_vector import _boundary_frame_bins
from neurospatial.simulation import generate_poisson_spikes

SIDE = 100.0  # cm
DT = 1 / 30  # 30 Hz tracking


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment.from_polygon(box(0, 0, SIDE, SIDE), bin_size=5.0, units="cm")


@pytest.fixture(scope="module")
def session():
    """Ten minutes of positions and headings that cover the box evenly."""
    rng = np.random.default_rng(7)
    times = np.arange(0, 600, DT)
    positions = rng.uniform(2, SIDE - 2, (len(times), 2))
    headings = rng.uniform(-np.pi, np.pi, len(times))
    return times, positions, headings


def _wall_distance(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Distance from each position to the box side along its own angle."""
    cos, sin = np.cos(angles), np.sin(angles)
    x, y = positions.T
    with np.errstate(divide="ignore", invalid="ignore"):
        to_x = np.where(cos > 0, (SIDE - x) / cos, np.where(cos < 0, -x / cos, np.inf))
        to_y = np.where(sin > 0, (SIDE - y) / sin, np.where(sin < 0, -y / sin, np.inf))
    return np.minimum(to_x, to_y)


def _tuned_spikes(times, distance, preferred=15.0, width=4.0, seed=100):
    rate = 0.5 + 20.0 * np.exp(-0.5 * ((distance - preferred) / width) ** 2)
    return generate_poisson_spikes(rate, times, seed=seed)


@pytest.mark.parametrize(
    ("wall", "angle"), [("East", 0.0), ("North", np.pi / 2), ("South", -np.pi / 2)]
)
def test_a_boundary_vector_cell_is_recovered(env, session, wall, angle):
    """A cell firing for a wall 15 cm away in one room direction peaks there.

    North and South pin the sign of the y axis; East alone would not.
    """
    times, positions, _ = session
    distance = _wall_distance(positions, np.full(len(positions), angle))
    spikes = _tuned_spikes(times, distance)

    result = compute_boundary_vector_rate(
        env,
        spikes,
        times,
        positions,
        direction_frame="allocentric",
        direction_bin_size=np.pi / 12,
    )

    sector = np.pi / 24  # half a direction bin
    error = np.angle(np.exp(1j * (result.preferred_direction() - angle)))
    assert abs(error) <= sector + 1e-12, wall
    assert 10.0 <= result.preferred_distance() <= 20.0


def test_an_egocentric_boundary_cell_is_recovered(env, session):
    """A cell firing for a wall 15 cm to the left peaks at +pi/2 (left)."""
    times, positions, headings = session
    left = _wall_distance(positions, headings + np.pi / 2)
    spikes = _tuned_spikes(times, left)

    result = compute_boundary_vector_rate(
        env,
        spikes,
        times,
        positions,
        headings,
        direction_frame="egocentric",
        direction_bin_size=np.pi / 12,
    )

    assert abs(result.preferred_direction() - np.pi / 2) <= np.pi / 24 + 1e-12
    assert 10.0 <= result.preferred_distance() <= 20.0
    # The same cell has no allocentric direction preference to speak of.
    allocentric = compute_boundary_vector_rate(
        env,
        spikes,
        times,
        positions,
        direction_frame="allocentric",
        direction_bin_size=np.pi / 12,
    )
    assert allocentric.distance_information() < result.distance_information()


def test_every_ray_reaching_a_wall_adds_the_sample_time(env, session):
    """With every wall in range, each ray counts every interval and spike."""
    times, positions, headings = session
    spikes = np.sort(np.random.default_rng(1).uniform(times[0], times[-1], 500))
    n_directions = 8

    result = compute_boundary_vector_rate(
        env,
        spikes,
        times,
        positions,
        headings,
        direction_frame="egocentric",
        distance_range=(0.0, 150.0),
        direction_bin_size=2 * np.pi / n_directions,
    )
    counts = np.asarray(result.firing_rate) * np.asarray(result.occupancy)

    duration = times[-1] - times[0]
    assert np.asarray(result.occupancy).sum() == pytest.approx(n_directions * duration)
    assert np.nansum(counts) == pytest.approx(n_directions * len(spikes))


def test_vectorized_counts_equal_the_ray_by_ray_counts(env, session):
    times, positions, headings = session
    positions = positions.copy()
    positions[100:130] = np.nan  # a tracking dropout
    spikes = np.sort(np.random.default_rng(2).uniform(times[0], times[-1], 800))
    result = compute_boundary_vector_rate(
        env,
        spikes,
        times,
        positions,
        headings,
        direction_frame="egocentric",
        distance_range=(0.0, 40.0),
    )

    frame_bins, observed = _boundary_frame_bins(
        env,
        positions,
        headings,
        direction_frame="egocentric",
        boundary_source="physical",
        distance_range=(0.0, 40.0),
        n_distance_bins=8,
        n_direction_bins=12,
    )
    mask = observed[:-1] & (np.diff(times) <= 0.5)
    n_bins = result.env.n_bins
    rays = range(frame_bins.shape[1])
    occupancy = sum(
        start_allocated_occupancy(
            frame_bins[:, k], np.diff(times), mask & (frame_bins[:-1, k] >= 0), n_bins
        )
        for k in rays
    )
    counts = sum(
        count_spikes_by_frame(spikes, times, frame_bins[:, k], mask, n_bins)
        for k in rays
    )
    np.testing.assert_allclose(result.occupancy, occupancy)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.where(occupancy > 0, counts / occupancy, np.nan)
    np.testing.assert_allclose(result.firing_rate, expected)


def test_single_unit_equals_unit_zero_of_the_population(env, session):
    times, positions, headings = session
    spikes = np.sort(np.random.default_rng(3).uniform(times[0], times[-1], 300))

    single = compute_boundary_vector_rate(
        env, spikes, times, positions, headings, direction_frame="egocentric"
    )
    population = compute_boundary_vector_rates(
        env,
        [spikes, spikes[::2]],
        times,
        positions,
        headings,
        direction_frame="egocentric",
        unit_ids=["a", "b"],
    )

    assert isinstance(population[0], BoundaryVectorRateResult)
    np.testing.assert_array_equal(single.firing_rate, population[0].firing_rate)
    np.testing.assert_array_equal(single.occupancy, population.occupancy)
    assert list(single.summary_table().columns) == list(
        population.summary_table().columns
    )
    assert population[1].unit_id == "b"
    assert population.summary_table().attrs["boundary_source"] == "physical"


def test_an_unvisited_corner_is_not_a_wall_unless_asked():
    """Physical walls ignore what the animal never visited; analysis edges do not."""
    rng = np.random.default_rng(4)
    times = np.arange(0, 300, DT)
    # The west half only, away from the north and south walls, so no eastward
    # ray from a sample meets a physical wall within 10 cm.
    positions = rng.uniform([0, 20], [50, 80], (len(times), 2))
    from neurospatial.arena import OpenArena

    env = Environment.from_samples(
        positions, bin_size=5.0, arena=OpenArena(box(0, 0, SIDE, SIDE))
    )
    spikes = np.sort(rng.uniform(0, 300, 300))
    kwargs = {
        "direction_frame": "allocentric",
        "distance_range": (0.0, 100.0),
        "distance_bin_size": 5.0,
    }

    physical = compute_boundary_vector_rate(env, spikes, times, positions, **kwargs)
    analysis = compute_boundary_vector_rate(
        env, spikes, times, positions, boundary_source="analysis", **kwargs
    )

    # Near the visited edge at x = 50, only the analysis map has eastward
    # walls closer than 10 cm from samples at x > 40.
    def east_near(result):
        centers = result.env.bin_centers
        near = (centers[:, 0] < 10) & (np.abs(centers[:, 1]) < np.pi / 6)
        return np.asarray(result.occupancy)[near].sum()

    assert east_near(analysis) > 0
    assert east_near(physical) == 0


def test_the_direction_frame_must_match_the_headings(env, session):
    times, positions, headings = session

    with pytest.raises(ValueError, match="needs headings"):
        compute_boundary_vector_rate(
            env, np.array([1.0]), times, positions, direction_frame="egocentric"
        )
    with pytest.raises(ValueError, match="does not use headings"):
        compute_boundary_vector_rate(
            env,
            np.array([1.0]),
            times,
            positions,
            headings,
            direction_frame="allocentric",
        )


def test_an_environment_without_known_walls_is_refused(session):
    times, positions, _ = session
    env = Environment.from_samples(positions, bin_size=5.0, units="cm")

    with pytest.raises(NotImplementedError, match="Physical boundaries are unknown"):
        compute_boundary_vector_rate(
            env, np.array([1.0]), times, positions, direction_frame="allocentric"
        )


@pytest.mark.parametrize(("units", "scale"), [("cm", 1.0), ("m", 0.01)])
def test_length_defaults_are_converted_from_cm(units, scale):
    """The 5 cm rings and 5 cm kernel follow the environment's units."""
    env = Environment.from_polygon(
        box(0, 0, SIDE * scale, SIDE * scale), bin_size=5.0 * scale, units=units
    )
    rng = np.random.default_rng(0)
    times = np.arange(0, 60, DT)
    positions = rng.uniform(2, SIDE - 2, (len(times), 2)) * scale
    result = compute_boundary_vector_rate(
        env,
        [],
        times,
        positions,
        direction_frame="allocentric",
        method="diffusion_kde",
    )
    assert result.distance_range[1] / result.n_distance_bins == pytest.approx(
        5.0 * scale
    )
    assert result.analysis_settings["bandwidth"] == pytest.approx(5.0 * scale)


def test_unknown_units_need_explicit_lengths(session):
    times, positions, _ = session
    env = Environment.from_polygon(box(0, 0, SIDE, SIDE), bin_size=5.0)
    with pytest.raises(ValueError, match="distance_range is required"):
        compute_boundary_vector_rate(
            env, [], times, positions, direction_frame="allocentric"
        )
    with pytest.raises(ValueError, match="distance_bin_size is required"):
        compute_boundary_vector_rate(
            env,
            [],
            times,
            positions,
            direction_frame="allocentric",
            distance_range=(0.0, 50.0),
        )
    with pytest.raises(ValueError, match="needs a bandwidth"):
        compute_boundary_vector_rate(
            env,
            [],
            times,
            positions,
            direction_frame="allocentric",
            distance_range=(0.0, 50.0),
            distance_bin_size=5.0,
            method="gaussian_kde",
        )


def test_distance_information_is_skaggs_information_along_the_preferred_ray(
    env, session
):
    """Each ray partitions time by wall distance; the whole map does not."""
    from neurospatial.encoding import spatial_information

    times, positions, _ = session
    east = _wall_distance(positions, np.zeros(len(positions)))
    result = compute_boundary_vector_rate(
        env, _tuned_spikes(times, east), times, positions, direction_frame="allocentric"
    )
    duration = times[-1] - times[0]
    occupancy = np.asarray(result.occupancy).reshape(
        result.n_distance_bins, result.n_direction_bins
    )
    assert np.all(occupancy.sum(axis=0) <= duration + 1e-9)
    assert occupancy.sum() > 2 * duration  # summed over rays: not a partition

    rates = np.asarray(result.firing_rate).reshape(occupancy.shape)
    edges = result.env.grid_edges[1]
    centers = (edges[:-1] + edges[1:]) / 2
    column = int(np.argmin(np.abs(centers - result.preferred_direction())))
    assert abs(centers[column]) <= np.pi / 12 + 1e-12  # one of the two East rays
    expected = spatial_information(rates[:, column], occupancy[:, column])
    assert result.distance_information() == pytest.approx(expected)
    assert result.summary_table()["distance_information"].iloc[0] == pytest.approx(
        expected
    )


def test_a_near_wall_cell_is_not_read_as_tuned_to_the_opposite_wall(env, session):
    """The default range keeps the peak on the near wall.

    In a 100 cm box, firing 15 cm from the East wall is firing 85 cm from the
    West wall. A grid out to the far walls lets the peak jump there; the
    default 0-50 cm range of near walls does not.
    """
    times, positions, _ = session
    east = _wall_distance(positions, np.zeros(len(positions)))
    spikes = _tuned_spikes(times, east)

    result = compute_boundary_vector_rate(
        env, spikes, times, positions, direction_frame="allocentric"
    )

    assert result.distance_range == pytest.approx((0.0, 50.0))
    assert (result.n_distance_bins, result.n_direction_bins) == (10, 12)
    assert abs(result.preferred_direction()) <= np.pi / 12 + 1e-12
    assert 10.0 <= result.preferred_distance() <= 20.0


def test_the_range_records_the_ray_time_beyond_it(env, session):
    """A range that ends inside a field reports how much ray time it left out."""
    times, positions, _ = session
    east = _wall_distance(positions, np.zeros(len(positions)))
    spikes = _tuned_spikes(times, east, preferred=30.0, width=6.0)

    result = compute_boundary_vector_rate(
        env,
        spikes,
        times,
        positions,
        direction_frame="allocentric",
        distance_range=(0.0, 25.0),
    )

    # Every ray from inside the box meets a wall, within the range or beyond.
    duration = times[-1] - times[0]
    in_range = np.asarray(result.occupancy).sum()
    beyond = result.analysis_settings["time_beyond_distance_range"]
    assert beyond > 0.0
    assert in_range + beyond == pytest.approx(result.n_direction_bins * duration)
    assert result.analysis_settings["fraction_of_time_in_distance_range"] == (
        pytest.approx(in_range / (in_range + beyond))
    )


def test_direction_bins_have_a_physical_width(env, session):
    times, positions, _ = session
    result = compute_boundary_vector_rate(
        env,
        [],
        times[:300],
        positions[:300],
        direction_frame="allocentric",
        direction_bin_size=np.deg2rad(15),
    )
    assert result.n_direction_bins == 24
