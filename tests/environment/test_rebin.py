"""Tests for Environment.rebin() - grid coarsening operations."""

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

from neurospatial import Environment
from neurospatial.arena import encode_arena


def _arena_record(arena):
    """Compare domain values independently of detached public export identity."""
    metadata, arrays = encode_arena(arena)
    return metadata, {name: array.tolist() for name, array in arrays.items()}


def _coarse_oracle(env, factor):
    """Enumerate candidate cells and test each center against arena geometry."""
    factors = (factor,) * env.n_dims if isinstance(factor, int) else factor
    edges = tuple(
        edge[: (size // step) * step + 1 : step]
        for edge, size, step in zip(
            env.grid_edges, env.grid_shape, factors, strict=True
        )
    )
    axes = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    candidates = np.asarray(list(product(*axes)))
    if not env.arena.domain_known:
        keep = np.ones(len(candidates), dtype=bool)
    elif (domain := env.arena.polygonal_domain()) is not None:
        keep = np.array([domain.contains(Point(center)) for center in candidates])
    else:
        keep = env.arena.contains(candidates)
    return edges, candidates[keep], keep.reshape(tuple(map(len, axes)))


def _assert_coarse_oracle(env, coarse, factor):
    edges, centers, active = _coarse_oracle(env, factor)
    np.testing.assert_array_equal(coarse.bin_centers, centers)
    np.testing.assert_array_equal(coarse.active_mask, active)
    for actual, expected in zip(coarse.grid_edges, edges, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert _arena_record(coarse.arena) == _arena_record(env.arena)


def _assert_masked_connectivity(env, coarse, *, diagonal):
    """Compare every edge with adjacency of retained full-grid indices."""
    _assert_coarse_oracle(env, coarse, 2)
    _, _, active = _coarse_oracle(env, 2)
    indices = np.argwhere(active)
    expected = {
        (u, v)
        for u, v in combinations(range(len(indices)), 2)
        if (
            np.max(np.abs(indices[u] - indices[v])) == 1
            if diagonal
            else np.sum(np.abs(indices[u] - indices[v])) == 1
        )
    }
    assert set(coarse.connectivity.edges) == expected
    return indices


@pytest.mark.parametrize("factor", [2, 3])
def test_sparse_observations_do_not_invent_unsupported_coarse_cells(factor):
    env = Environment.from_samples(
        positions=np.array([[0.0, 0.0], [100.0, 100.0]]), bin_size=10.0
    )
    _, centers, active = _coarse_oracle(env, factor)
    assert centers.size == 0 and not active.any()
    with (
        pytest.warns(UserWarning, match="truncated"),
        pytest.raises(ValueError, match="no coarse bin center"),
    ):
        env.rebin(factor)


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("factor", [1, 2, (2, 1)])
def test_rebinned_environment_round_trips(
    tmp_path: Path, diagonal: bool, factor: int | tuple[int, ...]
) -> None:
    """A saved coarse grid restores its cells, queries, graph and smoothing."""
    env = Environment.from_layout(
        "RegularGrid",
        {
            "bin_size": 1.0,
            "dimension_ranges": [(0, 4), (0, 4)],
            "connect_diagonal_neighbors": diagonal,
        },
    ).rebin(factor)
    path = tmp_path / "coarse"
    env.to_file(path)
    for restored in (Environment.from_file(path), Environment.from_dict(env.to_dict())):
        np.testing.assert_array_equal(restored.bin_centers, env.bin_centers)
        np.testing.assert_array_equal(restored.bin_sizes, env.bin_sizes)
        np.testing.assert_array_equal(
            restored.bin_at(env.bin_centers), np.arange(env.n_bins)
        )
        assert set(restored.connectivity.edges()) == set(env.connectivity.edges())
        field = np.arange(env.n_bins, dtype=float)
        np.testing.assert_allclose(restored.smooth(field, 1.0), env.smooth(field, 1.0))


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2), (3, 3)])
def test_rebin_identity_keeps_small_grid_edges(
    shape: tuple[int, ...], diagonal: bool
) -> None:
    """Rebinning a full grid by one preserves edges even without interior nodes."""
    env = Environment.from_layout(
        "RegularGrid",
        {
            "bin_size": 1.0,
            "dimension_ranges": [(0, side) for side in shape],
            "connect_diagonal_neighbors": diagonal,
        },
    )
    assert set(env.rebin(1).connectivity.edges()) == set(env.connectivity.edges())


class TestRebinBasic:
    """Basic rebin functionality tests."""

    def test_rebin_factor_2_shape(self):
        """Rebinning by factor 2 reduces grid dimensions by half."""
        # Create grid with explicit range to control shape
        env = Environment.from_samples(
            positions=np.array([[0, 0], [100, 100]]),
            bin_size=10.0,
            infer_active_bins=False,
        )

        # Get original shape
        original_shape = env.grid_shape

        # Rebin by factor 2 (may warn if not evenly divisible)
        coarse = env.rebin(factor=2)

        # Check new shape is original // 2 (truncated if not evenly divisible)
        expected_shape = (original_shape[0] // 2, original_shape[1] // 2)
        assert coarse.grid_shape == expected_shape
        assert coarse.n_bins == expected_shape[0] * expected_shape[1]
        _assert_coarse_oracle(env, coarse, 2)

    def test_rebin_factor_2_bin_positions(self):
        """Rebinned bin centers are at correct positions."""
        # Create simple 4x4 grid
        data = np.array(
            [
                [0, 0],
                [10, 0],
                [20, 0],
                [30, 0],
                [0, 10],
                [10, 10],
                [20, 10],
                [30, 10],
                [0, 20],
                [10, 20],
                [20, 20],
                [30, 20],
                [0, 30],
                [10, 30],
                [20, 30],
                [30, 30],
            ]
        )
        env = Environment.from_samples(data, bin_size=10.0)
        assert env.grid_shape == (4, 4)

        # Rebin by factor 2 → 2x2 grid
        coarse = env.rebin(factor=2)

        # Should have 2x2 grid with 4 bins
        assert coarse.grid_shape == (2, 2)
        assert coarse.n_bins == 4

        # Bin centers should be within the original data range
        assert np.all(coarse.bin_centers[:, 0] >= 0)
        assert np.all(coarse.bin_centers[:, 0] <= 40)
        assert np.all(coarse.bin_centers[:, 1] >= 0)
        assert np.all(coarse.bin_centers[:, 1] <= 40)

    def test_rebin_coarse_grid_geometry(self):
        """Rebinning creates correct coarse grid geometry."""
        # Create grid with known shape (use dimension_ranges to control)
        env = Environment.from_samples(
            positions=np.array([[0, 0], [100, 100]]),
            bin_size=10.0,
            infer_active_bins=False,
        )

        # Get grid shape
        grid_shape = env.grid_shape

        # Rebin (may warn if not evenly divisible)
        coarse = env.rebin(factor=2)

        # Verify geometry is correct (truncated to nearest divisible size)
        expected_shape = (grid_shape[0] // 2, grid_shape[1] // 2)
        assert coarse.n_bins == expected_shape[0] * expected_shape[1]
        assert coarse.grid_shape == expected_shape
        _assert_coarse_oracle(env, coarse, 2)

    def test_rebin_creates_valid_environment(self):
        """Rebinning creates a valid Environment instance."""
        # Create 4x4 grid with known values
        data = np.array(
            [
                [0, 0],
                [10, 0],
                [20, 0],
                [30, 0],
                [0, 10],
                [10, 10],
                [20, 10],
                [30, 10],
                [0, 20],
                [10, 20],
                [20, 20],
                [30, 20],
                [0, 30],
                [10, 30],
                [20, 30],
                [30, 30],
            ]
        )
        env = Environment.from_samples(data, bin_size=10.0)

        # Rebin
        coarse = env.rebin(factor=2)

        # Verify shape is correct
        assert coarse.grid_shape == (2, 2)
        assert coarse.n_bins == 4

        # Verify it's a valid environment
        assert isinstance(coarse, Environment)
        assert coarse._is_fitted

    def test_rebin_runs_setup_through_constructor(self):
        """rebin builds the coarse env via the normal constructor path.

        Previously rebin constructed an intermediate unfitted Environment and
        manually called ``_setup_from_layout()`` afterward. The constructor now
        runs setup itself (``layout_type_used="RegularGrid"``). The resulting
        env must be fully fitted, carry the RegularGrid metadata, and have a
        ``_state_version`` reflecting exactly one setup bump -- not a stale or
        double-counted value.
        """
        env = Environment.from_samples(
            positions=np.array([[0, 0], [100, 100]]),
            bin_size=10.0,
            infer_active_bins=False,
        )

        coarse = env.rebin(factor=2)

        assert coarse._is_fitted
        assert coarse._layout_type_used == "RegularGrid"
        # A factory-built grid env bumps the version once in _setup_from_layout.
        assert coarse._state_version == 1
        # Versioned caches must work against the rebinned graph. The
        # differential operator has shape (n_bins, n_edges).
        D = coarse.get_differential_operator()
        assert D.shape == (coarse.n_bins, coarse.connectivity.number_of_edges())

    def test_rebin_output_matches_manual_setup_reference(self):
        """Rebinned geometry/connectivity is unchanged by the constructor refactor.

        Recomputes the expected coarse grid independently and checks the
        rebinned env reproduces it: same bin centers, same edge count, same
        grid shape.
        """
        data = np.array([[i * 10.0, j * 10.0] for i in range(4) for j in range(4)])
        env = Environment.from_samples(data, bin_size=10.0)
        coarse = env.rebin(factor=2)

        # Independent expectation: 2x2 coarse grid covering the same extent.
        assert coarse.grid_shape == (2, 2)
        assert coarse.n_bins == 4
        # Bin centers sorted should be the midpoints of each 2x2 block.
        sorted_centers = coarse.bin_centers[np.lexsort(coarse.bin_centers.T[::-1])]
        # All centers lie within the original [0, 30] extent.
        assert np.all(sorted_centers >= 0.0)
        assert np.all(sorted_centers <= 30.0)
        # A 2x2 RegularGrid (4-/8-connected) has at least the 4 axis edges.
        assert coarse.connectivity.number_of_edges() >= 4


class TestRebinFactorVariations:
    """Test different factor specifications."""

    def test_rebin_factor_tuple(self):
        """Factor can be specified as tuple for anisotropic coarsening."""
        rng = np.random.default_rng(42)
        # Create 10x6 grid
        x = rng.random(1000) * 100
        y = rng.random(1000) * 60
        data = np.column_stack([x, y])
        env = Environment.from_samples(data, bin_size=10.0)

        original_shape = env.grid_shape

        # Coarsen by (2, 3)
        coarse = env.rebin(factor=(2, 3))

        # Expected shape: (original[0]//2, original[1]//3)
        expected_shape = (original_shape[0] // 2, original_shape[1] // 3)
        assert coarse.grid_shape == expected_shape

    def test_rebin_factor_3(self):
        """Factor 3 works correctly."""
        # Create grid with explicit range to get 9x9
        env = Environment.from_samples(
            positions=np.array([[0, 0], [90, 90]]),
            bin_size=10.0,
            infer_active_bins=False,
        )

        grid_shape = env.grid_shape

        # Rebin by 3 (may warn if not evenly divisible)
        coarse = env.rebin(factor=3)

        # Should be shape // 3 (truncated if not evenly divisible)
        expected_shape = (grid_shape[0] // 3, grid_shape[1] // 3)
        assert coarse.grid_shape == expected_shape
        assert coarse.n_bins == expected_shape[0] * expected_shape[1]
        _assert_coarse_oracle(env, coarse, 3)


class TestRebinConnectivity:
    """Test that connectivity is correctly rebuilt."""

    def test_rebin_preserves_connectivity_structure(self):
        """Rebinned grid has correct connectivity."""
        rng = np.random.default_rng(42)
        # Create 4x4 grid
        data = rng.random((200, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        # Rebin to 2x2
        coarse = env.rebin(factor=2)

        # 2x2 grid should have 4 nodes
        assert coarse.connectivity.number_of_nodes() == 4

        # In a 2x2 grid with diagonal connectivity (default for from_samples):
        # Each corner has 3 neighbors (2 edges + 1 diagonal)
        degrees = [coarse.connectivity.degree(n) for n in coarse.connectivity.nodes()]

        # 2x2 grid with diagonal: all 4 corners have 3 neighbors each
        assert sorted(degrees) == [3, 3, 3, 3]

    def test_rebin_edge_attributes(self):
        """Rebinned connectivity has required edge attributes."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        coarse = env.rebin(factor=2)

        # Check edges have required attributes
        for _u, _v, attrs in coarse.connectivity.edges(data=True):
            assert "distance" in attrs
            assert "vector" in attrs
            assert "edge_id" in attrs
            assert isinstance(attrs["distance"], (int, float))
            assert isinstance(attrs["vector"], tuple)


class TestRebinValidation:
    """Input validation tests."""

    def test_rebin_non_grid_layout_raises(self):
        """Rebinning non-grid layout raises NotImplementedError."""
        # Create a polygon layout (not a regular grid)
        from shapely.geometry import box

        polygon = box(0, 0, 50, 50)
        env = Environment.from_polygon(polygon, bin_size=10.0)

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match=r"only.*RegularGridLayout"):
            env.rebin(factor=2)

    def test_rebin_invalid_factor_raises(self):
        """Invalid factor values raise ValueError."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)

        # Factor must be positive
        with pytest.raises(ValueError, match=r"positive"):
            env.rebin(factor=0)

        with pytest.raises(ValueError, match=r"positive"):
            env.rebin(factor=-2)

    def test_rebin_factor_too_large_raises(self):
        """Factor larger than grid dimension raises ValueError."""
        rng = np.random.default_rng(42)
        # Create 4x4 grid
        data = rng.random((100, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        # Factor 10 is too large for 4x4 grid
        with pytest.raises(ValueError, match=r"too large.*grid shape"):
            env.rebin(factor=10)

    def test_rebin_non_divisible_warns(self):
        """Non-divisible factor warns and truncates."""
        # Create grid with known non-divisible shape
        env = Environment.from_samples(
            positions=np.array([[0, 0], [100, 100]]),
            bin_size=10.0,
            infer_active_bins=False,
        )

        grid_shape = env.grid_shape

        # Only run if shape is not divisible by 3
        if grid_shape[0] % 3 == 0 and grid_shape[1] % 3 == 0:
            pytest.skip(f"Grid shape {grid_shape} is divisible by 3")

        # Factor 3 doesn't divide evenly
        with pytest.warns(UserWarning, match=r"not evenly divisible") as record:
            coarse = env.rebin(factor=3)

        assert record[0].filename == __file__

        # Should truncate
        expected_shape = (grid_shape[0] // 3, grid_shape[1] // 3)
        assert coarse.grid_shape == expected_shape
        _assert_coarse_oracle(env, coarse, 3)


class TestRebinEdgeCases:
    """Edge cases and boundary conditions."""

    def test_rebin_factor_1_returns_copy(self):
        """Factor 1 returns a new environment with full grid."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)

        coarse = env.rebin(factor=1)

        # Should have same grid shape
        assert coarse.grid_shape == env.grid_shape

        # May have more bins if original had inactive bins
        # (rebin creates a full grid with all bins active)
        assert coarse.n_bins >= env.n_bins

        # But should be different object
        assert coarse is not env

    def test_rebin_2d_vs_3d(self):
        """Rebinning works for 2D and 3D grids."""
        rng = np.random.default_rng(42)
        # 2D grid
        data_2d = rng.random((200, 2)) * 40
        env_2d = Environment.from_samples(data_2d, bin_size=10.0)
        coarse_2d = env_2d.rebin(factor=2)
        assert coarse_2d.n_dims == 2

        # 3D grid
        data_3d = rng.random((500, 3)) * 40
        env_3d = Environment.from_samples(data_3d, bin_size=10.0)
        coarse_3d = env_3d.rebin(factor=2)
        assert coarse_3d.n_dims == 3

    def test_rebin_preserves_units_and_frame(self):
        """Rebinning preserves units and frame attributes."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)
        env = env.with_metadata(units="cm")
        env = env.with_metadata(frame="session1")

        coarse = env.rebin(factor=2)

        assert coarse.units == "cm"
        assert coarse.frame == "session1"


class TestRebinIntegration:
    """Integration tests with other operations."""

    def test_rebin_then_smooth(self):
        """Rebinning then smoothing works correctly."""
        rng = np.random.default_rng(42)
        data = rng.random((500, 2)) * 100
        env = Environment.from_samples(data, bin_size=10.0)

        # Rebin first
        coarse = env.rebin(factor=2)

        # Then smooth a field
        field = rng.random(coarse.n_bins)
        smoothed = coarse.smooth(field, bandwidth=5.0)

        assert smoothed.shape == (coarse.n_bins,)
        assert not np.any(np.isnan(smoothed))

    def test_rebin_bin_centers_mappable(self):
        """Can map original bin centers to rebinned environment."""
        rng = np.random.default_rng(42)
        data = rng.random((200, 2)) * 40
        env = Environment.from_samples(data, bin_size=10.0)

        coarse = env.rebin(factor=2)

        # All original bin centers should map to some bin in coarse
        from neurospatial.ops import map_points_to_bins

        indices = map_points_to_bins(coarse, env.bin_centers)

        # Should all be valid (not -1)
        assert np.all(indices >= 0)
        assert np.all(indices < coarse.n_bins)


class TestRebinDiagonalConnectivityMaskedGrid:
    """Regression: rebin infers diagonal connectivity from active-bin degrees."""

    def test_rebin_preserves_diagonal_connectivity_on_masked_grid(
        self, holed_grid_env_8conn
    ):
        """An 8-connected holed source rebins to an 8-connected grid."""
        env = holed_grid_env_8conn

        # Source should have a degree-8 interior node.
        src_max_degree = max(deg for _, deg in env.connectivity.degree())
        assert src_max_degree == 8

        coarse = env.rebin(factor=2)
        indices = _assert_masked_connectivity(env, coarse, diagonal=True)
        assert any(
            np.count_nonzero(indices[u] != indices[v]) > 1
            for u, v in coarse.connectivity.edges
        )

    def test_rebin_preserves_orthogonal_connectivity_on_masked_grid(
        self, holed_grid_env_4conn
    ):
        """A 4-connected holed source rebins to a 4-connected grid."""
        env = holed_grid_env_4conn

        src_max_degree = max(deg for _, deg in env.connectivity.degree())
        assert src_max_degree == 4

        coarse = env.rebin(factor=2)
        _assert_masked_connectivity(env, coarse, diagonal=False)
