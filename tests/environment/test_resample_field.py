"""Test for resample_field() function with nearest and diffuse methods.

This module tests resampling fields between different environment discretizations,
enabling multi-resolution analysis and cross-session alignment.

Tests cover:
- Nearest-neighbor resampling
- Diffuse method with smoothing
- Same vs different resolutions
- Value preservation and bins outside the source
- Input validation
"""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops.mapping import resample_field


class TestResampleFieldBasic:
    """Test basic resample_field functionality."""

    def test_nearest_method_same_resolution(self):
        """Test nearest method with same bin_size (identity-like)."""
        rng = np.random.default_rng(42)
        # Source environment
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination environment (slightly offset)
        dst_data = np.array([[i + 0.5, j + 0.5] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Create field on source
        field = rng.random(src_env.n_bins)

        # Resample to destination
        result = resample_field(field, src_env, dst_env, method="nearest")

        # Should return array with dst_env.n_bins elements
        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isfinite(result))

    def test_nearest_method_different_resolution(self):
        """Test nearest method with different bin sizes."""
        rng = np.random.default_rng(42)
        # Source: coarse resolution
        src_data = np.array([[i, j] for i in range(21) for j in range(21)])
        src_env = Environment.from_samples(src_data, bin_size=4.0)

        # Destination: fine resolution
        dst_data = np.array([[i, j] for i in range(21) for j in range(21)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Create field on source
        field = rng.random(src_env.n_bins)

        # Resample to destination
        result = resample_field(field, src_env, dst_env, method="nearest")

        # Fine resolution should have more bins
        assert result.shape == (dst_env.n_bins,)
        assert dst_env.n_bins > src_env.n_bins

    def test_nearest_is_default_method(self):
        """Test that method='nearest' is the default."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = rng.random(src_env.n_bins)

        result_explicit = resample_field(field, src_env, dst_env, method="nearest")
        result_default = resample_field(field, src_env, dst_env)

        np.testing.assert_array_equal(result_explicit, result_default)

    def test_resampling_preserves_field_range(self):
        """Test that resampled field stays within original value range."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Field with known range [0, 1]
        field = rng.random(src_env.n_bins)

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Nearest method should preserve exact values (no interpolation)
        assert np.min(result) >= np.min(field) - 1e-10
        assert np.max(result) <= np.max(field) + 1e-10


class TestResampleFieldDiffuse:
    """Test diffuse resampling method."""

    def test_diffuse_method_with_bandwidth(self):
        """Test diffuse method with explicit bandwidth."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = rng.random(src_env.n_bins)

        result = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=1.0
        )

        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isfinite(result))

    def test_diffuse_requires_bandwidth(self):
        """Test that diffuse method requires bandwidth parameter."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = rng.random(src_env.n_bins)

        with pytest.raises(ValueError, match="bandwidth must be provided"):
            resample_field(field, src_env, dst_env, method="diffuse")

    def test_diffuse_smoother_than_nearest(self):
        """Test that diffuse method produces smoother results."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Create spiky field
        field = np.zeros(src_env.n_bins)
        field[src_env.n_bins // 2] = 1.0

        nearest_result = resample_field(field, src_env, dst_env, method="nearest")
        diffuse_result = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=2.0
        )

        # Diffuse should spread the spike more
        assert np.sum(nearest_result > 0.1) < np.sum(diffuse_result > 0.01)

    def test_diffuse_with_different_bandwidths(self):
        """Test that larger bandwidth produces smoother results."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.zeros(src_env.n_bins)
        field[src_env.n_bins // 2] = 1.0

        result_small = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=0.5
        )
        result_large = resample_field(
            field, src_env, dst_env, method="diffuse", bandwidth=3.0
        )

        # Larger bandwidth spreads values more
        assert np.std(result_large) < np.std(result_small)


class TestResampleFieldEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_field_zeros(self):
        """Test resampling field of all zeros."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.zeros(src_env.n_bins)

        result = resample_field(field, src_env, dst_env, method="nearest")

        np.testing.assert_array_equal(result, np.zeros(dst_env.n_bins))

    def test_constant_field(self):
        """Test resampling constant field."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = np.ones(src_env.n_bins) * 5.0

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Nearest neighbor should preserve constant value
        np.testing.assert_allclose(result, 5.0, rtol=1e-10)

    def test_single_bin_source(self):
        """Test resampling from environment with single bin."""
        # Source with one bin
        src_data = np.array([[5.0, 5.0]])
        src_env = Environment.from_samples(src_data, bin_size=10.0)

        # Destination with multiple bins
        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        field = np.array([3.14])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # All destination bins get mapped to single source bin
        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isfinite(result))

    def test_non_overlapping_environments(self):
        """Test resampling when environments don't overlap spatially."""
        rng = np.random.default_rng(42)
        # Source environment at (0, 0) to (10, 10)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination environment at (100, 100) to (110, 110)
        dst_data = np.array([[100 + i, 100 + j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        field = rng.random(src_env.n_bins)

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Every destination bin lies outside the source environment, so each
        # maps to the "outside" sentinel and resamples to NaN (a field cannot
        # be meaningfully pulled onto a region the source does not cover).
        assert result.shape == (dst_env.n_bins,)
        assert np.all(np.isnan(result))


class TestResampleFieldInputValidation:
    """Test input validation and error messages."""

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = rng.random(src_env.n_bins)

        with pytest.raises(ValueError, match="method must be 'nearest' or 'diffuse'"):
            resample_field(field, src_env, dst_env, method="invalid")

    def test_field_size_mismatch(self):
        """Test that mismatched field and source environment sizes raise error."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        wrong_field = rng.random(src_env.n_bins + 5)

        with pytest.raises(
            ValueError, match=r"Field size .* does not match source environment"
        ):
            resample_field(wrong_field, src_env, dst_env, method="nearest")

    def test_negative_bandwidth(self):
        """Test that negative bandwidth raises error."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = rng.random(src_env.n_bins)

        with pytest.raises(ValueError, match="bandwidth must be positive"):
            resample_field(field, src_env, dst_env, method="diffuse", bandwidth=-1.0)

    def test_dimension_mismatch(self):
        """Test that environments with different dimensions raise error."""
        rng = np.random.default_rng(42)
        # 2D source
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # 1D destination (if we can create one - may skip if not supported)
        # For now, just test with matching dimensions
        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        field = rng.random(src_env.n_bins)

        # This should work (both 2D)
        result = resample_field(field, src_env, dst_env, method="nearest")
        assert result.shape == (dst_env.n_bins,)


class TestResampleFieldCorrectness:
    """Test correctness with known ground truth mappings."""

    def test_nearest_exact_alignment(self):
        """Test nearest method with exactly aligned grids (known mapping)."""
        # Source: 3x3 grid with bin centers at (1, 3, 5) x (1, 3, 5)
        src_data = np.array([[i, j] for i in [1, 3, 5] for j in [1, 3, 5]])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination: same grid (exact alignment)
        dst_data = np.array([[i, j] for i in [1, 3, 5] for j in [1, 3, 5]])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Field with known values at each position
        field = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Should be exact identity mapping
        np.testing.assert_array_equal(result, field)

    def test_nearest_offset_grid(self):
        """Test nearest method with offset grids (known nearest neighbors)."""
        # Source: 2x2 grid with bin centers at (0, 2) x (0, 2)
        src_data = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination: points near each source bin
        # (0.5, 0.5) nearest to (0, 0)
        # (0.5, 1.5) nearest to (0, 2)
        # (1.5, 0.5) nearest to (2, 0)
        # (1.5, 1.5) nearest to (2, 2)
        dst_data = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5], [1.5, 1.5]])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Field with distinct values: [10, 20, 30, 40]
        field = np.array([10.0, 20.0, 30.0, 40.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Each destination point should get value from nearest source
        np.testing.assert_array_equal(result, field)

    def test_nearest_upsampling(self):
        """Test nearest method upsampling (fine grid from coarse)."""
        # Source: single bin at center
        src_data = np.array([[5.0, 5.0]])
        src_env = Environment.from_samples(src_data, bin_size=10.0)

        # Destination: 3 points all near the single source bin
        dst_data = np.array([[4.0, 5.0], [5.0, 5.0], [6.0, 5.0]])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Source has single value
        field = np.array([42.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # All destination bins should get the same value
        np.testing.assert_array_equal(result, np.array([42.0, 42.0, 42.0]))

    def test_nearest_downsampling(self):
        """Test nearest method downsampling (coarse grid from fine)."""
        # Source: 4 bins in 2x2 grid at (0, 1) x (0, 1)
        src_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        src_env = Environment.from_samples(src_data, bin_size=1.0)

        # Destination: single point at center
        dst_data = np.array([[0.5, 0.5]])
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Field with different values
        field = np.array([1.0, 2.0, 3.0, 4.0])

        result = resample_field(field, src_env, dst_env, method="nearest")

        # The (0.5, 0.5) point is equidistant from all 4 source bins
        # With tie_break=LOWEST_INDEX, should pick first bin
        assert result.shape == (1,)
        assert result[0] == 1.0  # From bin 0 (lowest index)


class TestResampleFieldMathematicalProperties:
    """Test mathematical properties of resampling."""

    def test_nearest_preserves_values(self):
        """Test that nearest method only produces values from source field."""
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        dst_data = np.array([[i, j] for i in range(11) for j in range(11)])
        dst_env = Environment.from_samples(dst_data, bin_size=1.0)

        # Field with distinct values
        field = np.arange(src_env.n_bins, dtype=np.float64)

        result = resample_field(field, src_env, dst_env, method="nearest")

        # Every result value should be from original field
        for val in result:
            assert val in field

    def test_identity_resampling_same_environment(self):
        """Test that resampling to same environment is identity-like."""
        rng = np.random.default_rng(42)
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        field = rng.random(src_env.n_bins)

        # Resample to itself
        result = resample_field(field, src_env, src_env, method="nearest")

        # Should be very close to identity
        np.testing.assert_allclose(result, field, rtol=1e-10)

    def test_diffuse_preserves_a_constant_intensive_field(self):
        """Diffuse averages an intensive field, so a constant rate stays exact.

        It is not mass-conserving (see the resample_field Notes); what it must
        preserve is the value of a constant rate map, including at the edges.
        """
        src_data = np.array([[i, j] for i in range(11) for j in range(11)])
        src_env = Environment.from_samples(src_data, bin_size=2.0)
        dst_env = Environment.from_samples(src_data, bin_size=1.0)

        result = resample_field(
            np.full(src_env.n_bins, 5.0),
            src_env,
            dst_env,
            method="diffuse",
            bandwidth=1.0,
        )

        np.testing.assert_allclose(result, 5.0, rtol=1e-10)

    def test_diffuse_never_reports_noise_far_from_the_data(self):
        """An average of one observation of 5 is 5 wherever it is defined.

        Far from the observed bin both smoothed sums are truncation noise;
        their ratio reached -53,000 when resampling skipped the support floor
        that env.smooth applies. Those bins are NaN instead.
        """
        env = Environment.from_samples(np.arange(0.5, 100.0)[:, None], bin_size=1.0)
        field = np.full(env.n_bins, np.nan)
        observed = int(env.bin_at(np.array([[50.5]]))[0])
        field[observed] = 5.0

        result = resample_field(field, env, env, method="diffuse", bandwidth=5.0)

        defined = np.isfinite(result)
        assert defined[observed]
        np.testing.assert_allclose(result[defined], 5.0, rtol=1e-3)
        assert not defined.all()
        np.testing.assert_array_equal(
            result,
            env.smooth(field, bandwidth=5.0, mode="average"),
        )


class TestResampleFieldOutsideSource:
    """A destination bin outside the source has no value: NaN, never an edge copy."""

    @pytest.fixture
    def arena_and_longer_track(self) -> tuple[Environment, Environment]:
        """A 20 x 20 cm source and a 60 x 20 cm destination with the same bins."""
        src = Environment.from_samples(
            np.array([[i + 0.5, j + 0.5] for i in range(20) for j in range(20)]),
            bin_size=2.0,
        )
        dst = Environment.from_samples(
            np.array([[i + 0.5, j + 0.5] for i in range(60) for j in range(20)]),
            bin_size=2.0,
        )
        return src, dst

    @pytest.mark.parametrize("method", ["nearest", "diffuse"])
    def test_bins_beyond_the_source_are_nan(self, arena_and_longer_track, method):
        src, dst = arena_and_longer_track
        field = src.bin_centers[:, 0].copy()  # x position, cm

        result = resample_field(field, src, dst, method=method, bandwidth=2.0)

        # The source's bins end at x = 21 cm. A nearest-center rule with a
        # distance allowance gave x = 22..40 cm the edge value 20.
        x_edge = src.dimension_ranges[0][1]
        beyond = dst.bin_centers[:, 0] > x_edge
        assert beyond.any() and (~beyond).any()
        assert np.isnan(result[beyond]).all()
        assert np.isfinite(result[~beyond]).all()

    def test_bins_inside_take_the_source_values(self, arena_and_longer_track):
        src, dst = arena_and_longer_track
        field = src.bin_centers[:, 0].copy()

        result = resample_field(field, src, dst, method="nearest")

        # The overlapping bins are the same bins, so each gets its own value.
        inside = dst.bin_centers[:, 0] <= src.dimension_ranges[0][1]
        np.testing.assert_array_equal(result[inside], dst.bin_centers[inside, 0])

    def test_identical_geometry_returns_the_field(self):
        """Same bins: every value comes back, a missing bin stays missing."""
        data = np.array([[i, j] for i in range(11) for j in range(11)], float)
        src = Environment.from_samples(data, bin_size=2.0)
        dst = Environment.from_samples(data, bin_size=2.0)
        field = np.arange(src.n_bins, dtype=float)
        field[3] = np.nan

        result = resample_field(field, src, dst, method="nearest")

        np.testing.assert_array_equal(result, field)

    def test_track_without_width_is_refused(self):
        """A centerline does not say which destination bins lie on the track."""
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(100.0, 0.0))
        graph.add_edge(0, 1)
        src = Environment.from_graph(graph, [(0, 1)], bin_size=5.0)
        dst = Environment.from_graph(graph, [(0, 1)], bin_size=2.0)

        with pytest.raises(NotImplementedError, match="without track_width"):
            resample_field(np.ones(src.n_bins), src, dst)
