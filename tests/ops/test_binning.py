"""Tests for neurospatial.ops.binning module."""

import numpy as np
import pytest


class TestBinningImports:
    """Test that binning module exports are available from new location."""

    def test_import_map_points_to_bins(self):
        """Test map_points_to_bins can be imported from ops.binning."""
        from neurospatial.ops.binning import map_points_to_bins

        assert callable(map_points_to_bins)

    def test_import_tie_break_strategy(self):
        """Test TieBreakStrategy can be imported from ops.binning."""
        from neurospatial.ops.binning import TieBreakStrategy

        assert hasattr(TieBreakStrategy, "LOWEST_INDEX")
        assert hasattr(TieBreakStrategy, "CLOSEST_CENTER")

    def test_import_regions_to_mask(self):
        """Test regions_to_mask can be imported from ops.binning."""
        from neurospatial.ops.binning import regions_to_mask

        assert callable(regions_to_mask)

    def test_import_resample_field(self):
        """Test resample_field can be imported from ops.binning."""
        from neurospatial.ops.binning import resample_field

        assert callable(resample_field)

    def test_import_clear_kdtree_cache(self):
        """Test clear_kdtree_cache can be imported from ops.binning."""
        from neurospatial.ops.binning import clear_kdtree_cache

        assert callable(clear_kdtree_cache)


class TestMapPointsToBins:
    """Test map_points_to_bins function from new location."""

    @pytest.fixture
    def grid_env(self):
        """Create a simple grid environment."""
        from neurospatial import Environment

        # Create data on a regular grid (deterministic, no RNG needed)
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        env = Environment.from_samples(data, bin_size=2.0, name="grid")
        return env

    def test_map_points_basic(self, grid_env):
        """Test basic point mapping from new import location."""
        from neurospatial.ops.binning import map_points_to_bins

        points = np.array([[5.0, 5.0], [0.0, 0.0], [10.0, 10.0]])
        bins = map_points_to_bins(points, grid_env)

        assert bins.shape == (3,)
        assert bins.dtype == np.int64
        # All points should map to valid bins (>= 0)
        assert np.all(bins >= 0)

    def test_tie_break_strategy_enum(self, grid_env):
        """Test using TieBreakStrategy enum from new location."""
        from neurospatial.ops.binning import TieBreakStrategy, map_points_to_bins

        points = np.array([[5.0, 5.0]])
        bins = map_points_to_bins(
            points, grid_env, tie_break=TieBreakStrategy.LOWEST_INDEX
        )

        assert bins.shape == (1,)
        assert bins[0] >= 0

    def test_lowest_index_tie_break_handles_more_than_ten_ties(self):
        """lowest_index should consider all equidistant bins, not only KDTree k."""
        from neurospatial.ops.binning import map_points_to_bins

        class MinimalEnvironment:
            def __init__(self, bin_centers):
                self.bin_centers = bin_centers
                self._kdtree_cache = None

        n_bins = 20
        angles = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
        bin_centers = np.column_stack([np.cos(angles), np.sin(angles)])
        env = MinimalEnvironment(bin_centers)
        points = np.array([[0.0, 0.0]])

        bins = map_points_to_bins(
            points,
            env,  # type: ignore[arg-type]
            tie_break="lowest_index",
            max_distance=2.0,
        )

        assert bins[0] == 0


# Construction bin_size for the regular-grid fixture below. There is no scalar
# ``env.bin_size`` attribute (``env.bin_sizes`` is per-bin), so the construction
# value is tracked as a local constant for distance tolerances.
_GRID_BIN_SIZE = 2.0


class TestMapPointsToBinsBehavior:
    """Behavioral tests for map_points_to_bins distance/threshold/tie semantics."""

    @pytest.fixture
    def regular_grid_env(self):
        """Regular grid env with bin_size=2.0 covering [0, 20] x [0, 20].

        Bin centers fall on even multiples of bin_size (0, 2, 4, ..., 20),
        so the typical bin spacing equals bin_size = 2.0.
        """
        from neurospatial import Environment

        x = np.arange(0.0, 21.0, 1.0)
        y = np.arange(0.0, 21.0, 1.0)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])
        return Environment.from_samples(data, bin_size=_GRID_BIN_SIZE, name="grid")

    def test_returns_dist_finite_for_inside_points(self, regular_grid_env):
        """Inside points get finite distance < bin_size; one point pinned exactly."""
        from neurospatial.ops.binning import map_points_to_bins

        point = np.array([[7.3, 11.7]])
        bins, dists = map_points_to_bins(point, regular_grid_env, return_dist=True)

        assert bins[0] >= 0
        assert np.isfinite(dists[0])
        assert dists[0] < _GRID_BIN_SIZE

        # Pin the distance to the assigned bin center.
        expected = np.linalg.norm(point[0] - regular_grid_env.bin_centers[bins[0]])
        assert dists[0] == pytest.approx(expected, abs=1e-10)

    def test_returns_dist_inf_for_outside_points(self, regular_grid_env):
        """A point far outside the env returns inf distance."""
        from neurospatial.ops.binning import map_points_to_bins

        far_point = np.array([[1000.0, 1000.0]])
        bins, dists = map_points_to_bins(far_point, regular_grid_env, return_dist=True)

        assert bins[0] == -1
        assert np.isinf(dists[0])

    def test_max_distance_marks_outside(self, regular_grid_env):
        """max_distance threshold rejects points beyond it (bin index -1)."""
        from neurospatial.ops.binning import map_points_to_bins

        # Point at distance 1.5 * bin_size = 3.0 from nearest center (20, 0).
        point = np.array([[23.0, 0.0]])

        bins_reject = map_points_to_bins(
            point, regular_grid_env, max_distance=_GRID_BIN_SIZE
        )
        assert bins_reject[0] == -1

        # Re-create env so the previous query's threshold does not leak via cache.
        from neurospatial import Environment

        x = np.arange(0.0, 21.0, 1.0)
        y = np.arange(0.0, 21.0, 1.0)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])
        env2 = Environment.from_samples(data, bin_size=_GRID_BIN_SIZE)

        bins_accept = map_points_to_bins(point, env2, max_distance=2.0 * _GRID_BIN_SIZE)
        assert bins_accept[0] >= 0

    def test_max_distance_factor_relative_to_bin_spacing(self, regular_grid_env):
        """max_distance_factor=0.5 equals max_distance = 0.5 * typical spacing."""
        from scipy.spatial import cKDTree

        from neurospatial import Environment
        from neurospatial.ops.binning import (
            _estimate_typical_bin_spacing,
            map_points_to_bins,
        )

        kdtree = cKDTree(regular_grid_env.bin_centers)
        spacing = _estimate_typical_bin_spacing(kdtree, regular_grid_env.bin_centers)

        x = np.arange(0.0, 21.0, 1.0)
        y = np.arange(0.0, 21.0, 1.0)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        # Probe just inside (0.9 * 0.5 * spacing) and just outside (1.1 *) the
        # 0.5-factor threshold; compute both ways and assert equivalence.
        for offset in (0.9, 1.1):
            point = np.array([[20.0 + offset, 0.0]])  # dist = offset from (20, 0)

            env_factor = Environment.from_samples(data, bin_size=_GRID_BIN_SIZE)
            bins_factor = map_points_to_bins(point, env_factor, max_distance_factor=0.5)

            env_abs = Environment.from_samples(data, bin_size=_GRID_BIN_SIZE)
            bins_abs = map_points_to_bins(point, env_abs, max_distance=0.5 * spacing)

            assert bins_factor[0] == bins_abs[0]

    def test_max_distance_and_factor_together_raises(self, regular_grid_env):
        """Specifying both max_distance and max_distance_factor raises."""
        from neurospatial.ops.binning import map_points_to_bins

        point = np.array([[5.0, 5.0]])
        with pytest.raises(ValueError, match="Cannot specify both"):
            map_points_to_bins(
                point, regular_grid_env, max_distance=1.0, max_distance_factor=1.0
            )

    def test_max_distance_negative_raises(self, regular_grid_env):
        """A negative max_distance raises."""
        from neurospatial.ops.binning import map_points_to_bins

        point = np.array([[5.0, 5.0]])
        with pytest.raises(ValueError, match=r"positive|non-negative"):
            map_points_to_bins(point, regular_grid_env, max_distance=-1.0)

    def test_default_threshold_is_10x_typical_spacing(self, monkeypatch):
        """The implicit default threshold is 10 x typical bin spacing.

        Monkeypatch the spacing estimator to a known K; a point at 9.5*K is
        inside while a point at 10.5*K is rejected. Pins the 10x heuristic.
        """
        import neurospatial.ops.binning as binning
        from neurospatial import Environment

        x = np.arange(0.0, 21.0, 1.0)
        y = np.arange(0.0, 21.0, 1.0)
        xx, yy = np.meshgrid(x, y)
        data = np.column_stack([xx.ravel(), yy.ravel()])

        K = 1.0
        monkeypatch.setattr(
            binning, "_estimate_typical_bin_spacing", lambda kdtree, bc: K
        )

        env_in = Environment.from_samples(data, bin_size=_GRID_BIN_SIZE)
        point_in = np.array([[20.0 + 9.5 * K, 0.0]])  # dist 9.5*K from (20, 0)
        bins_in = binning.map_points_to_bins(point_in, env_in)
        assert bins_in[0] >= 0

        env_out = Environment.from_samples(data, bin_size=_GRID_BIN_SIZE)
        point_out = np.array([[20.0 + 10.5 * K, 0.0]])  # dist 10.5*K
        bins_out = binning.map_points_to_bins(point_out, env_out)
        assert bins_out[0] == -1

    @pytest.mark.parametrize("use_enum", [True, False])
    def test_closest_center_vs_lowest_index_on_tie(self, regular_grid_env, use_enum):
        """On an exact tie, lowest_index returns the lower bin; both call paths work.

        A point at (1.0, 0.0) is exactly equidistant from centers (0, 0) and
        (2, 0). lowest_index must return the lower of the two bin indices;
        closest_center returns one of the two equidistant bins.
        """
        from neurospatial.ops.binning import TieBreakStrategy, map_points_to_bins

        tie_point = np.array([[1.0, 0.0]])

        centers = regular_grid_env.bin_centers
        idx_00 = int(np.flatnonzero((centers[:, 0] == 0.0) & (centers[:, 1] == 0.0))[0])
        idx_20 = int(np.flatnonzero((centers[:, 0] == 2.0) & (centers[:, 1] == 0.0))[0])
        lower_idx = min(idx_00, idx_20)

        lowest = TieBreakStrategy.LOWEST_INDEX if use_enum else "lowest_index"
        closest = TieBreakStrategy.CLOSEST_CENTER if use_enum else "closest_center"

        bins_lowest = map_points_to_bins(tie_point, regular_grid_env, tie_break=lowest)
        assert bins_lowest[0] == lower_idx

        bins_closest = map_points_to_bins(
            tie_point, regular_grid_env, tie_break=closest
        )
        # closest_center returns one of the two equidistant bins.
        assert bins_closest[0] in (idx_00, idx_20)

    def test_nan_in_points_raises(self, regular_grid_env):
        """NaN coordinates raise a ValueError from the KDTree query.

        Pinned to current behavior: the underlying scipy cKDTree query rejects
        non-finite inputs rather than returning the -1 sentinel.
        """
        from neurospatial.ops.binning import map_points_to_bins

        points = np.array([[np.nan, np.nan], [50.0, 50.0]])
        with pytest.raises(ValueError, match="finite"):
            map_points_to_bins(points, regular_grid_env)

    def test_returns_correct_bin_on_regular_grid(self, regular_grid_env):
        """A point maps to the bin whose center is nearest (round-to-center)."""
        from neurospatial.ops.binning import map_points_to_bins

        point = np.array([[7.3, 11.7]])
        bins = map_points_to_bins(point, regular_grid_env)

        # Centers lie on even multiples of bin_size starting at 0, so the nearest
        # center is (round(7.3 / 2) * 2, round(11.7 / 2) * 2) = (8, 12).
        expected_center = np.array(
            [
                round(7.3 / _GRID_BIN_SIZE) * _GRID_BIN_SIZE,
                round(11.7 / _GRID_BIN_SIZE) * _GRID_BIN_SIZE,
            ]
        )
        assert np.allclose(regular_grid_env.bin_centers[bins[0]], expected_center)


class TestResampleFieldOutOfBounds:
    """Destination bins outside the source environment must become NaN."""

    def test_out_of_source_bins_become_nan(self):
        """Destination bins with no source coverage map to NaN, not field[-1].

        ``map_points_to_bins`` returns -1 for destination bin centers that fall
        outside the source environment. Negative indexing would silently
        resolve -1 to ``field[-1]`` (the last source bin's value); those
        positions must instead be NaN.
        """
        from neurospatial import Environment
        from neurospatial.ops.binning import resample_field

        # Source: a small region covering only x,y in roughly [0, 20].
        src_data = np.array(
            [[i, j] for i in range(0, 21, 2) for j in range(0, 21, 2)],
            dtype=np.float64,
        )
        src_env = Environment.from_samples(src_data, bin_size=2.0)

        # Destination: a much larger region extending well beyond the source.
        dst_data = np.array(
            [[i, j] for i in range(0, 61, 2) for j in range(0, 61, 2)],
            dtype=np.float64,
        )
        dst_env = Environment.from_samples(dst_data, bin_size=2.0)

        # Distinctive field with a large value in the LAST source bin so that
        # any accidental ``field[-1]`` fallback would be obvious.
        field = np.arange(src_env.n_bins, dtype=np.float64)
        field[-1] = 999.0

        resampled = resample_field(field, src_env, dst_env, method="nearest")

        assert resampled.shape == (dst_env.n_bins,)

        # Identify which destination bins fall outside the source coverage.
        from neurospatial.ops.binning import TieBreakStrategy, map_points_to_bins

        dst_to_src = map_points_to_bins(
            dst_env.bin_centers, src_env, tie_break=TieBreakStrategy.LOWEST_INDEX
        )
        outside = dst_to_src < 0

        # There must be some out-of-source destination bins for a valid test.
        assert np.any(outside)

        # Out-of-source bins must be NaN (not field[-1] == 999.0).
        assert np.all(np.isnan(resampled[outside]))
        # In-source bins must be finite and exactly equal to the gathered value.
        assert np.all(np.isfinite(resampled[~outside]))
        assert np.array_equal(resampled[~outside], field[dst_to_src[~outside]])

    def test_identity_resample_unchanged(self):
        """Resampling onto the same environment preserves all values (no NaN)."""
        from neurospatial import Environment
        from neurospatial.ops.binning import resample_field

        data = np.array(
            [[i, j] for i in range(0, 21, 2) for j in range(0, 21, 2)],
            dtype=np.float64,
        )
        env = Environment.from_samples(data, bin_size=2.0)
        field = np.arange(env.n_bins, dtype=np.float64)

        resampled = resample_field(field, env, env, method="nearest")
        assert np.allclose(resampled, field)
