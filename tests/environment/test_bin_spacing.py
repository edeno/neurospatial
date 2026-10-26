"""``Environment.bin_spacing`` is one physical length scale on every layout."""

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("layout", ["RegularGrid", "Hexagonal"])
def test_grid_spacing_is_the_bin_size(rng, layout):
    env = Environment.from_samples(
        rng.uniform(0, 100, (20_000, 2)), bin_size=4.0, layout=layout
    )
    assert env.bin_spacing == pytest.approx(4.0)


def test_polygon_and_3d_grids_use_the_bin_size(rng):
    polygon = Environment.from_polygon(Point(50, 50).buffer(40), bin_size=4.0)
    cube = Environment.from_samples(rng.uniform(0, 40, (30_000, 3)), bin_size=4.0)
    assert polygon.bin_spacing == pytest.approx(4.0)
    assert cube.bin_spacing == pytest.approx(4.0)


def test_track_spacing_follows_the_track_not_the_nearest_arm():
    """Arms 2 apart with 4-long bins: the nearest center is on the other arm."""
    track = Environment.linear_track(
        node_positions=[[0, 0], [0, 100], [2, 100], [2, 0]], bin_size=4.0
    )
    nearest = track.bin_center_tree().query(track.bin_centers, k=2)[0][:, 1]
    assert np.median(nearest) == pytest.approx(2.0)
    assert track.bin_spacing == pytest.approx(4.0, rel=0.05)


def test_polar_spacing_is_a_length_not_an_angle():
    """(distance, angle) rows mix cm and radians; spacing uses places."""
    polar = Environment.from_polar_egocentric(
        distance_range=(0, 40),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=4.0,
        angle_bin_size=np.pi / 8,
    )
    places = polar.coordinate_system.cartesian_embedding(polar.bin_centers)
    gaps = np.linalg.norm(places[:, None] - places[None], axis=-1)
    np.fill_diagonal(gaps, np.inf)
    assert polar.bin_spacing == pytest.approx(np.median(gaps.min(axis=1)))
    assert polar.bin_spacing > 1.0  # not the 0.39 rad angle bin


def test_single_bin_has_no_spacing():
    env = Environment.from_samples(np.array([[0.0, 0.0]]), bin_size=1.0)
    assert env.bin_spacing == np.inf


def test_spacing_is_cached_and_cleared(rng):
    env = Environment.from_samples(rng.uniform(0, 20, (2000, 2)), bin_size=2.0)
    first = env.bin_spacing
    env.clear_cache()
    assert env.bin_spacing == first
