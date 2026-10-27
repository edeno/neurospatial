"""Graph-connected component extraction and place-field segmentation."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from neurospatial import Environment
from neurospatial.encoding._field_metrics import _seed_component


def _disk_mask(env, radius):
    center = env.bin_centers.mean(axis=0)
    distances = np.linalg.norm(env.bin_centers - center, axis=1)
    return distances < radius, int(np.argmin(distances))


def test_seed_component_is_the_complete_connected_component():
    """The component holds the seed, only masked bins, and every connected one."""
    rng = np.random.default_rng(42)
    env = Environment.from_samples(rng.random((1000, 2)) * 100, bin_size=5.0)
    mask, seed_idx = _disk_mask(env, 20.0)

    result = _seed_component(seed_idx, mask, env)

    assert result.dtype == np.int64
    assert seed_idx in result
    assert mask[result].all()
    assert_array_equal(result, np.sort(result))
    members = set(result.tolist())
    for i in result:
        for neighbor in env.connectivity.neighbors(int(i)):
            if mask[neighbor]:
                assert neighbor in members, f"missed connected bin {neighbor}"


def test_seed_component_isolated_seed():
    """A seed with no masked neighbors is its own component."""
    rng = np.random.default_rng(42)
    env = Environment.from_samples(rng.random((500, 2)) * 50, bin_size=5.0)
    mask = np.zeros(env.n_bins, dtype=bool)
    mask[10] = True

    assert_array_equal(_seed_component(10, mask, env), [10])


def test_seed_component_includes_a_seed_outside_the_mask():
    """The seed starts its component even when the mask excludes it."""
    env = Environment.from_grid_mask(
        np.ones((1, 4), dtype=bool), grid_edges=(np.arange(2.0), np.arange(5.0))
    )
    mask = np.array([False, True, True, False])

    assert_array_equal(_seed_component(0, mask, env), [0, 1, 2])
    assert_array_equal(_seed_component(3, mask, env), [1, 2, 3])


def test_seed_component_separates_disconnected_parts():
    """Masked bins with no path through the mask to the seed are left out."""
    env = Environment.from_grid_mask(
        np.ones((1, 5), dtype=bool), grid_edges=(np.arange(2.0), np.arange(6.0))
    )
    mask = np.array([True, True, False, True, True])

    assert_array_equal(_seed_component(0, mask, env), [0, 1])
    assert_array_equal(_seed_component(4, mask, env), [3, 4])


def test_detect_place_fields_grid():
    """Test field detection on a synthetic grid rate map."""
    # Create 2D grid environment with synthetic place field
    rng = np.random.default_rng(42)
    positions = rng.random((5000, 2)) * 100
    env = Environment.from_samples(positions, bin_size=5.0)

    # Create synthetic firing rate with one place field
    firing_rate = np.zeros(env.n_bins)
    center = env.bin_centers.mean(axis=0)
    for i in range(env.n_bins):
        dist = np.linalg.norm(env.bin_centers[i] - center)
        firing_rate[i] = 8.0 * np.exp(-(dist**2) / (2 * 10.0**2))

    from neurospatial.encoding.spatial import detect_place_fields

    # Detect the Gaussian field using graph edges
    fields = detect_place_fields(env, firing_rate, bandwidth=0.0)

    # Should detect exactly one field
    assert len(fields) == 1
    assert len(fields[0]) > 0
