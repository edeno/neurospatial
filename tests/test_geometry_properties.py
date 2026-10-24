"""
Property-based tests for geometry operations using Hypothesis.

These tests verify mathematical invariants for spatial operations on regions
and environments. Property-based testing helps uncover edge cases in geometry
and binning logic that example-based tests might miss.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from shapely import Polygon

from neurospatial import Environment
from neurospatial.regions import Region, Regions

# =============================================================================
# Hypothesis Strategies for Geometry
# =============================================================================


@st.composite
def environment_and_samples_2d_strategy(
    draw: st.DrawFn,
    min_samples: int = 50,
    max_samples: int = 500,
    min_extent: float = 10.0,
    max_extent: float = 100.0,
    min_bin_size: float = 1.0,
    max_bin_size: float = 10.0,
) -> tuple[Environment, np.ndarray]:
    """Generate valid 2D environments with the samples that built them.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_samples : int, default=50
        Minimum number of position samples.
    max_samples : int, default=500
        Maximum number of position samples.
    min_extent : float, default=10.0
        Minimum spatial extent in each dimension.
    max_extent : float, default=100.0
        Maximum spatial extent in each dimension.
    min_bin_size : float, default=1.0
        Minimum bin size.
    max_bin_size : float, default=10.0
        Maximum bin size.

    Returns
    -------
    env : Environment
        A valid fitted Environment.
    positions : ndarray, shape (n_samples, 2)
        The samples it was built from, with ``bin_count_threshold=1``.
    """
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    extent = draw(st.floats(min_value=min_extent, max_value=max_extent))
    bin_size = draw(st.floats(min_value=min_bin_size, max_value=max_bin_size))

    # Ensure bin_size is reasonable relative to extent
    assume(extent / bin_size >= 3)  # At least 3 bins in each dimension

    # Generate random positions
    seed = draw(st.integers(min_value=0, max_value=10000))
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, extent, size=(n_samples, 2))

    try:
        env = Environment.from_samples(
            positions, bin_size=bin_size, bin_count_threshold=1
        )
        assume(env.n_bins >= 5)  # Need enough bins for meaningful tests
        return env, positions
    except (ValueError, RuntimeError):
        assume(False)  # Skip this example


def environment_2d_strategy(**kwargs: float) -> st.SearchStrategy[Environment]:
    """Generate valid 2D environments; see environment_and_samples_2d_strategy."""
    return environment_and_samples_2d_strategy(**kwargs).map(lambda pair: pair[0])


# =============================================================================
# Property Tests for Regions
# =============================================================================


class TestRegionsContainerProperties:
    """Property-based tests for Regions container operations."""

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=30, deadline=5000)
    def test_regions_container_length(self, n_regions: int):
        """Property: Regions container length matches number of added regions."""
        regions_list = []
        for i in range(n_regions):
            region = Region(
                name=f"region_{i}",
                kind="polygon",
                data=Polygon(
                    [(i * 20, 0), ((i + 1) * 20, 0), ((i + 1) * 20, 10), (i * 20, 10)]
                ),
            )
            regions_list.append(region)

        container = Regions(regions_list)
        assert len(container) == n_regions

    @given(
        st.lists(
            st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
            min_size=2,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=30, deadline=5000)
    def test_regions_dict_like_access(self, names: list[str]):
        """Property: Regions container supports dict-like access by name."""
        regions_list = []
        for i, name in enumerate(names):
            region = Region(
                name=name,
                kind="polygon",
                data=Polygon(
                    [(i * 20, 0), ((i + 1) * 20, 0), ((i + 1) * 20, 10), (i * 20, 10)]
                ),
            )
            regions_list.append(region)

        container = Regions(regions_list)

        # All names should be accessible
        for name in names:
            assert name in container
            assert container[name].name == name


# =============================================================================
# Property Tests for Environment Binning
# =============================================================================


class TestEnvironmentBinningProperties:
    """Property-based tests for Environment bin operations."""

    @given(environment_2d_strategy())
    @settings(max_examples=30, deadline=10000)
    def test_bin_centers_within_bounds(self, env: Environment):
        """Property: All bin centers are within the dimension ranges."""
        bin_sizes = env.bin_sizes  # It's a property, not a method
        for bin_center in env.bin_centers:
            for dim, (lo, hi) in enumerate(env.dimension_ranges):
                # Allow half bin_size tolerance at edges
                tolerance = bin_sizes[dim] / 2.0
                assert bin_center[dim] >= lo - tolerance, (
                    f"Bin center {bin_center[dim]} below range {lo}"
                )
                assert bin_center[dim] <= hi + tolerance, (
                    f"Bin center {bin_center[dim]} above range {hi}"
                )

    @given(environment_and_samples_2d_strategy())
    @settings(max_examples=30, deadline=10000)
    def test_bin_at_matches_brute_force_cells(
        self, env_and_samples: tuple[Environment, np.ndarray]
    ):
        """Property: bin_at agrees with a brute-force search of the grid cells.

        With ``bin_count_threshold=1`` every sample used to build the
        environment lies in an active bin, so each maps to a bin. Every other
        point in range maps to the active cell whose edges contain it, or -1
        when that cell is inactive.
        """
        env, samples = env_and_samples
        edges = env.grid_edges
        active = env.active_mask
        flat_to_bin = np.full(active.size, -1, dtype=np.intp)
        flat_to_bin[np.flatnonzero(active)] = np.arange(env.n_bins)

        def brute_force(points: np.ndarray) -> np.ndarray:
            cells = [
                np.clip(
                    np.searchsorted(edge, points[:, axis], side="right") - 1,
                    0,
                    len(edge) - 2,
                )
                for axis, edge in enumerate(edges)
            ]
            return flat_to_bin[np.ravel_multi_index(cells, active.shape)]

        sample_bins = env.bin_at(samples)
        assert np.all(sample_bins >= 0), "a building sample has no bin"
        np.testing.assert_array_equal(sample_bins, brute_force(samples))

        rng = np.random.default_rng(42)
        lo = np.array([edge[0] for edge in edges])
        hi = np.array([edge[-1] for edge in edges])
        points = rng.uniform(lo, hi, size=(200, 2))
        np.testing.assert_array_equal(env.bin_at(points), brute_force(points))

    @given(environment_2d_strategy())
    @settings(max_examples=30, deadline=10000)
    def test_bin_at_batch_matches_single_points(self, env: Environment):
        """Property: a point's bin does not depend on the other query points."""
        rng = np.random.default_rng(7)
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])
        span = hi - lo
        points = rng.uniform(lo - 0.1 * span, hi + 0.1 * span, size=(30, 2))
        batch = env.bin_at(points)
        single = np.array([env.bin_at(point[None, :])[0] for point in points])
        np.testing.assert_array_equal(batch, single)
        order = rng.permutation(len(points))
        np.testing.assert_array_equal(env.bin_at(points[order]), batch[order])

    @given(environment_2d_strategy(min_samples=100, max_samples=300))
    @settings(max_examples=20, deadline=15000)
    def test_bin_centers_are_unique(self, env: Environment):
        """Property: All bin centers are unique."""
        centers = env.bin_centers
        unique_centers = np.unique(centers, axis=0)
        assert len(unique_centers) == len(centers), "Bin centers should all be unique"

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_connectivity_graph_nodes_match_bins(self, env: Environment):
        """Property: Connectivity graph has one node per bin."""
        n_graph_nodes = env.connectivity.number_of_nodes()
        assert n_graph_nodes == env.n_bins, (
            f"Graph has {n_graph_nodes} nodes but environment has {env.n_bins} bins"
        )


class TestEnvironmentOccupancyProperties:
    """Property-based tests for occupancy computation."""

    @given(
        st.integers(min_value=2, max_value=8),
        st.integers(min_value=2, max_value=8),
        st.floats(min_value=0.5, max_value=20.0),
        st.integers(min_value=2, max_value=400),
        st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=30, deadline=15000)
    def test_occupancy_sums_to_total_time(
        self, n_x: int, n_y: int, bin_size: float, n_samples: int, seed: int
    ):
        """Property: on a fully active grid, occupancy sums to the recording time.

        Every position is inside the grid (its outer edges included) and
        every sampling interval is shorter than ``max_gap``, so no interval
        is excluded and the per-bin times add up to ``times[-1] - times[0]``.
        """
        edges = (
            np.arange(n_x + 1) * bin_size - 3.0,
            np.arange(n_y + 1) * bin_size + 11.0,
        )
        env = Environment.from_grid_mask(np.ones((n_x, n_y), dtype=bool), edges)
        rng = np.random.default_rng(seed)
        lo = np.array([edge[0] for edge in edges])
        hi = np.array([edge[-1] for edge in edges])
        positions = rng.uniform(lo, hi, size=(n_samples, 2))
        # Put some samples exactly on the outer edges, which belong to a bin.
        positions[::7] = np.where(rng.random((len(positions[::7]), 2)) < 0.5, lo, hi)
        max_gap = 0.5
        times = 10.0 + np.cumsum(rng.uniform(0.001, 0.9 * max_gap, n_samples))

        occupancy = env.occupancy(times, positions, max_gap=max_gap)

        assert occupancy.shape == (env.n_bins,)
        assert np.all(occupancy >= 0)
        np.testing.assert_allclose(
            occupancy.sum(), times[-1] - times[0], rtol=1e-12, atol=0
        )

    @given(environment_2d_strategy(min_samples=100, max_samples=300))
    @settings(max_examples=20, deadline=15000)
    def test_occupancy_non_negative(self, env: Environment):
        """Property: Occupancy is always non-negative."""
        # Generate trajectory
        rng = np.random.default_rng(42)
        n_samples = 200
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        positions = rng.uniform(lo, hi, size=(n_samples, 2))
        # times must be monotonically increasing
        times = np.sort(rng.uniform(0, n_samples / 30.0, n_samples))

        # Signature: occupancy(times, positions, ...)
        occupancy = env.occupancy(times, positions)

        assert np.all(occupancy >= 0), "Occupancy should never be negative"


class TestEnvironmentNeighborProperties:
    """Property-based tests for neighbor relationships."""

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_neighbor_relationship_symmetric(self, env: Environment):
        """Property: If A is neighbor of B, then B is neighbor of A."""
        for node in env.connectivity.nodes():
            for neighbor in env.connectivity.neighbors(node):
                assert env.connectivity.has_edge(neighbor, node), (
                    f"Neighbor relationship not symmetric: "
                    f"{node} has neighbor {neighbor} but not vice versa"
                )

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_edge_distances_positive(self, env: Environment):
        """Property: All edge distances are positive."""
        for u, v, data in env.connectivity.edges(data=True):
            assert data["distance"] > 0, (
                f"Edge ({u}, {v}) has non-positive distance {data['distance']}"
            )

    @given(environment_2d_strategy())
    @settings(max_examples=20, deadline=10000)
    def test_edge_distances_match_centers(self, env: Environment):
        """Property: Edge distance matches Euclidean distance between centers."""
        for u, v, data in env.connectivity.edges(data=True):
            center_u = env.bin_centers[u]
            center_v = env.bin_centers[v]
            euclidean_dist = np.linalg.norm(center_u - center_v)

            np.testing.assert_allclose(
                data["distance"],
                euclidean_dist,
                rtol=1e-10,
                err_msg=f"Edge ({u}, {v}) distance doesn't match Euclidean distance",
            )


# =============================================================================
# Property Tests for Trajectories
# =============================================================================


class TestTrajectoryProperties:
    """Property-based tests for trajectory analysis."""

    @given(
        environment_2d_strategy(min_samples=100, max_samples=300),
        st.integers(min_value=50, max_value=200),
    )
    @settings(max_examples=15, deadline=20000)
    def test_bin_sequence_length_bounded(self, env: Environment, n_positions: int):
        """Property: bin_sequence length is at most n_positions."""
        rng = np.random.default_rng(42)
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        positions = rng.uniform(lo, hi, size=(n_positions, 2))
        # bin_sequence requires times - use sorted times
        times = np.sort(rng.uniform(0, n_positions / 30.0, n_positions))

        try:
            # Signature: bin_sequence(times, positions, ...)
            bin_seq = env.bin_sequence(times, positions, dedup=False)
            # Bin sequence length should be at most n_positions
            # (may be less due to filtering invalid bins)
            assert len(bin_seq) <= n_positions
        except (ValueError, RuntimeError):
            pass  # Some positions may be outside environment

    @given(environment_2d_strategy(min_samples=100, max_samples=300))
    @settings(max_examples=15, deadline=15000)
    def test_bin_sequence_indices_valid(self, env: Environment):
        """Property: All bin sequence indices are valid bin indices or -1."""
        rng = np.random.default_rng(42)
        n_positions = 100
        lo = np.array([r[0] for r in env.dimension_ranges])
        hi = np.array([r[1] for r in env.dimension_ranges])

        positions = rng.uniform(lo, hi, size=(n_positions, 2))
        times = np.sort(rng.uniform(0, n_positions / 30.0, n_positions))

        try:
            # Signature: bin_sequence(times, positions, ...)
            bin_seq = env.bin_sequence(times, positions, dedup=False)
            for idx in bin_seq:
                # Valid indices are 0 to n_bins-1, or -1 for outside
                assert idx == -1 or 0 <= idx < env.n_bins, (
                    f"Invalid bin index {idx} in sequence (n_bins={env.n_bins})"
                )
        except (ValueError, RuntimeError):
            pass
