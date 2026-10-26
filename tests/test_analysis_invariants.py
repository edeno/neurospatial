"""Scientific invariants across geometry, encoding and decoding operations.

These tests compare equivalent analyses through public APIs. They deliberately
use nonuniform rate maps, nondefault unit labels, holes and recording gaps so
that a vacuous uniform-posterior or all-zero result cannot pass.
"""

from itertools import permutations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from neurospatial import Environment, compute_spatial_rate, compute_spatial_rates
from neurospatial.decoding import BayesianDecoder
from neurospatial.encoding import SpikeTrains
from neurospatial.ops.distance import distance_field
from neurospatial.ops.transforms import AffineND, translate


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_label_permutation_preserves_encoding_and_decoding(
    analysis_invariant_recording, analysis_invariant_method, order
):
    """Refitting permuted units and predicting in either order gives one result."""
    env, spikes, times, positions, coverage = analysis_invariant_recording
    order = np.asarray(order)
    reordered = SpikeTrains([spikes[i] for i in order], unit_ids=spikes.unit_ids[order])
    options = {**analysis_invariant_method, "spike_window": coverage}
    original = compute_spatial_rates(env, spikes, times, positions, **options)
    permuted = compute_spatial_rates(env, reordered, times, positions, **options)
    np.testing.assert_array_equal(permuted.unit_ids, original.unit_ids[order])
    np.testing.assert_array_equal(permuted.occupancy, original.occupancy)
    np.testing.assert_allclose(
        permuted.firing_rates, original.firing_rates[order], atol=1e-10
    )
    assert np.ptp(original.firing_rates) > 0.1
    assert original.occupancy.sum() == pytest.approx(np.diff(coverage).sum())

    expected = BayesianDecoder.from_rates(original, dt=0.1).predict(
        spikes, times, spike_window=coverage
    )
    assert np.isfinite(expected.posterior).all()
    assert np.ptp(expected.posterior) > 0.01
    for inputs in (spikes, reordered):
        actual = BayesianDecoder.from_rates(permuted, dt=0.1).predict(
            inputs, times, spike_window=coverage
        )
        np.testing.assert_array_equal(actual.times, expected.times)
        np.testing.assert_allclose(actual.posterior, expected.posterior, atol=1e-10)
        np.testing.assert_allclose(actual.mean_position, expected.mean_position)
        assert not np.any(
            (actual.times >= coverage[0, 1]) & (actual.times < coverage[1, 0])
        )


def test_single_and_population_estimators_agree(
    analysis_invariant_recording, analysis_invariant_method
):
    """Batching independent fits preserves rates, occupancy and recording windows."""
    env, spikes, times, positions, coverage = analysis_invariant_recording
    options = {**analysis_invariant_method, "spike_window": coverage}
    batch = compute_spatial_rates(env, spikes, times, positions, **options)
    for index, label in enumerate(spikes.unit_ids):
        single = compute_spatial_rate(
            env, spikes.sel(unit_id=label), times, positions, **options
        )
        np.testing.assert_allclose(
            single.firing_rate, batch[index].firing_rate, atol=1e-10
        )
        np.testing.assert_array_equal(single.occupancy, batch.occupancy)
        np.testing.assert_array_equal(batch[index].spike_window, coverage)
        assert batch[index].unit_id == label


# Each example creates new transformed environments; the source fixture is
# intentionally reused read-only across generated coordinate changes.
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    angle=st.floats(min_value=-np.pi, max_value=np.pi),
    scale=st.floats(min_value=0.25, max_value=4.0),
    reflect=st.booleans(),
)
# A reflected quarter-scale map whose inverse used to fail hexagonal-cell
# validation on a cell that its locator assigns correctly.
@example(angle=-2.0965322377907283, scale=0.25, reflect=True)
# A scaled grid whose stored centers drifted from its re-mapped cell edges
# over four transforms, so the inverse failed geometry validation.
@example(angle=0.0, scale=1.263995540083514, reflect=False)
# A subnormal rotation angle: its sine has fewer significant bits than the
# relative roundoff model assumed, so the recovered grid frame was refused.
@example(angle=2.2250738585e-313, scale=4.0, reflect=False)
def test_chained_coordinate_changes_preserve_queries_and_occupancy(
    analysis_invariant_env, angle, scale, reflect
):
    """A similarity and its inverse retain bin identity on every layout."""
    env = analysis_invariant_env
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    if reflect:
        matrix[:2, 0] *= -1
    matrix[:2, 2] = [10.0, -7.0]
    transform = AffineND(matrix)
    points = np.vstack([env.bin_centers, [[1.43, 1.57], [-20.0, -20.0]]])
    # Probe interiors, holes and outside points; exact boundaries are excluded
    # because roundoff there can legitimately change the owning adjacent cell.
    original_bins = env.bin_at(points)
    shifted = env.apply_transform(transform, units=env.units).apply_transform(
        translate(3.0, 2.0)
    )
    moved = translate(3.0, 2.0)(transform(points))
    np.testing.assert_array_equal(shifted.bin_at(moved), original_bins)
    np.testing.assert_allclose(
        distance_field(shifted.connectivity, [0]),
        distance_field(env.connectivity, [0]) * scale,
        atol=1e-12,
    )
    times = np.arange(len(points), dtype=float) * 0.1
    np.testing.assert_allclose(
        shifted.occupancy(times, moved), env.occupancy(times, points)
    )
    restored = shifted.apply_transform(translate(-3.0, -2.0)).apply_transform(
        transform.inverse(), units=env.units
    )
    np.testing.assert_allclose(restored.bin_centers, env.bin_centers, atol=1e-12)
    np.testing.assert_array_equal(restored.bin_at(points), original_bins)


@pytest.mark.parametrize("angle", [2.2250738585e-313, 5e-324, -1e-310])
def test_grid_rotated_by_a_subnormal_angle_is_accepted(angle):
    """A rotation by an angle below the normal range keeps a valid grid frame."""
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    env = Environment.from_grid_mask(mask, grid_edges=(np.arange(4.0),) * 2)
    matrix = np.eye(3)
    matrix[:2, :2] = 4.0 * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    matrix[:2, 2] = [10.0, -7.0]
    transform = AffineND(matrix)
    rotated = env.apply_transform(transform)
    np.testing.assert_allclose(
        rotated.bin_centers, transform(env.bin_centers), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        rotated.bin_at(transform(env.bin_centers)), np.arange(env.n_bins)
    )
    restored = rotated.apply_transform(transform.inverse())
    np.testing.assert_allclose(restored.bin_centers, env.bin_centers, atol=1e-12)


def test_chained_grid_transforms_accept_their_inverse():
    """Scaling a masked grid four times and back keeps its cells and centers.

    The cell edges are mapped at each step; the stored centers must stay the
    midpoints of those edges, or the inverse no longer validates.
    """
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    env = Environment.from_grid_mask(mask, grid_edges=(np.arange(4.0),) * 2)
    matrix = np.eye(3)
    matrix[:2, :2] = 1.263995540083514 * np.eye(2)
    matrix[:2, 2] = [10.0, -7.0]
    transform = AffineND(matrix)
    shifted = env.apply_transform(transform).apply_transform(translate(3.0, 2.0))
    restored = shifted.apply_transform(translate(-3.0, -2.0)).apply_transform(
        transform.inverse()
    )
    np.testing.assert_allclose(restored.bin_centers, env.bin_centers, atol=1e-12)
    np.testing.assert_array_equal(restored.bin_at(env.bin_centers), np.arange(8))
    for edges in restored.grid_edges:
        assert np.allclose(edges, np.arange(4.0), atol=1e-12)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
@pytest.mark.parametrize("scale", [0.01, 10.0])
@pytest.mark.parametrize("method", ["binned", "gaussian_kde", "diffusion_kde"])
def test_coordinate_units_preserve_rates_and_posterior(method, n_dims, scale):
    """Changing cm to m/mm scales geometry and bandwidth, never Hz or seconds."""
    mask = np.ones((3,) * n_dims, dtype=bool)
    mask[(1,) * n_dims] = False
    edges = (np.arange(4.0),) * n_dims
    env = Environment.from_grid_mask(mask, grid_edges=edges)
    converted = Environment.from_grid_mask(
        mask, grid_edges=tuple(edge * scale + 10.0 for edge in edges)
    )
    bins = np.tile(np.arange(env.n_bins), 12)
    times = np.arange(len(bins), dtype=float) * 0.1
    positions = env.bin_centers[bins]
    spikes = [times[:-1:3], times[1:-1:4]]
    options = {
        "method": method,
        "bandwidth": 0.8,
        "min_occupancy": 0.5,
        "fill_value": 0.0,
    }
    original = compute_spatial_rates(env, spikes, times, positions, **options)
    changed = compute_spatial_rates(
        converted,
        spikes,
        times,
        positions * scale + 10.0,
        **{**options, "bandwidth": options["bandwidth"] * scale},
    )
    np.testing.assert_allclose(changed.occupancy, original.occupancy, atol=1e-12)
    np.testing.assert_allclose(changed.firing_rates, original.firing_rates, atol=1e-8)
    first = BayesianDecoder.from_rates(original, dt=0.2).predict(spikes, times)
    second = BayesianDecoder.from_rates(changed, dt=0.2).predict(spikes, times)
    assert np.isfinite(first.posterior).all()
    np.testing.assert_array_equal(second.times, first.times)
    np.testing.assert_allclose(second.posterior, first.posterior, atol=1e-8)
    np.testing.assert_allclose(
        second.mean_position, first.mean_position * scale + 10.0, atol=1e-8
    )
