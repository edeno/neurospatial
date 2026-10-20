"""Tests for distance_to_reward and distance_to_boundary."""

import numpy as np
import pytest

from neurospatial import Regions

# =============================================================================
# Test distance_to_reward
# =============================================================================


@pytest.fixture
def simple_grid_env():
    """Create a simple 10x10 grid environment for testing."""
    from neurospatial import Environment

    # Create positions covering 10x10 grid
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    env = Environment.from_samples(positions, bin_size=1.0)
    return env


class TestDistanceToReward:
    """Tests for distance_to_reward function."""

    def test_basic_functionality(self, simple_grid_env):
        """Test basic distance to reward calculation."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        # Trajectory moving from (0,0) to (9,9)
        n_samples = 50
        times = np.linspace(0, 10, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=5 (midpoint)
        reward_times = np.array([5.0])

        result = distance_to_reward(
            env, times, positions, reward_times, metric="euclidean"
        )

        assert result.shape == (n_samples,)
        assert not np.all(np.isnan(result))
        # At t=5, animal is at midpoint, reward is there, distance ~0
        mid_idx = n_samples // 2
        assert result[mid_idx] < result[0]  # Closer at reward time

    def test_mode_nearest(self, simple_grid_env):
        """Test mode='nearest' selects closest reward in time."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.column_stack(
            [
                np.full(n_samples, 5.0),  # Stay at center
                np.full(n_samples, 5.0),
            ]
        )

        # Two rewards at different locations and times
        reward_times = np.array([2.0, 8.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            mode="nearest",
            metric="euclidean",
        )

        # At t=2, should be distance to (0,5) which is 5 units
        idx_at_2 = 20  # t=2.0
        # At t=8, should be distance to (9,5) which is 4 units
        idx_at_8 = 80  # t=8.0

        assert result[idx_at_2] == pytest.approx(5.0, abs=0.5)
        assert result[idx_at_8] == pytest.approx(4.0, abs=0.5)

    def test_mode_last(self, simple_grid_env):
        """Test mode='last' uses most recent reward."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.full((n_samples, 2), 5.0)

        reward_times = np.array([3.0, 7.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            mode="last",
            metric="euclidean",
        )

        # Before first reward (t<3): NaN
        assert np.isnan(result[0])
        assert np.isnan(result[10])

        # After first reward (3<=t<7): distance to (0,5)
        assert not np.isnan(result[30])  # t=3.0
        assert not np.isnan(result[50])  # t=5.0

        # After second reward (t>=7): distance to (9,5)
        assert not np.isnan(result[70])  # t=7.0
        assert not np.isnan(result[90])  # t=9.0

    def test_mode_next(self, simple_grid_env):
        """Test mode='next' uses upcoming reward."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 100
        times = np.linspace(0, 10, n_samples)
        positions = np.full((n_samples, 2), 5.0)

        reward_times = np.array([3.0, 7.0])
        reward_positions = np.array([[0.0, 5.0], [9.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            mode="next",
            metric="euclidean",
        )

        # Before first reward: distance to (0,5)
        assert not np.isnan(result[0])
        assert not np.isnan(result[20])

        # Between rewards (3<=t<7): distance to (9,5)
        assert not np.isnan(result[50])

        # After last reward: NaN
        assert np.isnan(result[80])
        assert np.isnan(result[99])

    def test_empty_rewards_returns_nan(self, simple_grid_env):
        """Test empty reward_times returns all NaN."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        reward_times = np.array([])

        result = distance_to_reward(env, times, positions, reward_times)

        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_empty_positions(self, simple_grid_env):
        """Test empty positions returns empty array."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        times = np.array([])
        positions = np.empty((0, 2))
        reward_times = np.array([1.0, 2.0])

        result = distance_to_reward(env, times, positions, reward_times)

        assert len(result) == 0

    def test_reward_positions_interpolated(self, simple_grid_env):
        """Test reward positions are interpolated when not provided."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 50
        times = np.linspace(0, 10, n_samples)
        # Trajectory from (0,0) to (9,9)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=5 (position should be ~(4.5, 4.5))
        reward_times = np.array([5.0])

        # Without explicit reward_positions
        result = distance_to_reward(
            env, times, positions, reward_times, metric="euclidean"
        )

        # At t=5, animal is at ~(4.5, 4.5), reward interpolated to same
        mid_idx = n_samples // 2
        assert result[mid_idx] < 1.0  # Close to zero at reward time

    def test_reward_time_outside_session_warns_on_clip(self, simple_grid_env):
        """A reward time outside the session is clipped and warns the user."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 50
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )

        # Reward at t=20.0 is past the end of the session (times max = 10.0).
        reward_times = np.array([20.0])

        with pytest.warns(UserWarning, match=r"clip"):
            result = distance_to_reward(
                env, times, positions, reward_times, metric="euclidean"
            )
        assert result.shape == (n_samples,)

    def test_reward_time_within_session_does_not_warn(self, simple_grid_env):
        """Reward times inside the session must not trigger the clip warning."""
        import warnings

        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        n_samples = 50
        times = np.linspace(0.0, 10.0, n_samples)
        positions = np.column_stack(
            [
                np.linspace(0, 9, n_samples),
                np.linspace(0, 9, n_samples),
            ]
        )
        reward_times = np.array([3.0, 7.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            distance_to_reward(env, times, positions, reward_times, metric="euclidean")

    def test_position_times_mismatch_raises(self, simple_grid_env):
        """Test mismatched positions and times raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        positions = np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]])
        times = np.array([0.0, 1.0])  # Wrong length
        reward_times = np.array([0.5])

        with pytest.raises(ValueError, match=r"positions and times.*same length"):
            distance_to_reward(env, times, positions, reward_times)

    def test_reward_positions_times_mismatch_raises(self, simple_grid_env):
        """Test mismatched reward_positions and reward_times raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        positions = np.array([[0.0, 0.0], [5.0, 5.0]])
        times = np.array([0.0, 1.0])
        reward_times = np.array([0.5, 0.8])
        reward_positions = np.array([[1.0, 1.0]])  # Wrong length

        with pytest.raises(ValueError, match=r"reward_positions and reward_times"):
            distance_to_reward(
                env, times, positions, reward_times, reward_positions=reward_positions
            )

    def test_geodesic_respects_graph(self, simple_grid_env):
        """Test geodesic metric uses graph distances."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env

        # Position at (0, 0), reward at (1, 1)
        positions = np.array([[0.0, 0.0]])
        times = np.array([0.0])
        reward_times = np.array([0.0])
        reward_positions = np.array([[1.0, 1.0]])

        # Geodesic distance (Manhattan-like on grid)
        result_geodesic = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            metric="geodesic",
        )

        # Euclidean distance (sqrt(2) ~= 1.41)
        result_euclidean = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            metric="euclidean",
        )

        # Geodesic should be >= Euclidean (diagonal not directly connected)
        assert result_geodesic[0] >= result_euclidean[0]


class TestDistanceToRewardValidation:
    """Tests for non-finite and unsorted-time guards in distance_to_reward."""

    def test_distance_to_reward_nan_reward_time_raises(self, simple_grid_env):
        """A NaN in reward_times must raise, naming the argument."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        n_samples = 10
        times = np.linspace(0, 9, n_samples)
        positions = np.column_stack(
            [np.linspace(0, 9, n_samples), np.linspace(0, 9, n_samples)]
        )
        reward_times = np.array([2.0, np.nan])

        with pytest.raises(ValueError, match="reward_times"):
            distance_to_reward(env, times, positions, reward_times)

    def test_distance_to_reward_nan_position_raises(self, simple_grid_env):
        """A NaN in positions must raise, naming the argument."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        n_samples = 10
        times = np.linspace(0, 9, n_samples)
        positions = np.column_stack(
            [np.linspace(0, 9, n_samples), np.linspace(0, 9, n_samples)]
        )
        positions[3, 0] = np.nan
        reward_times = np.array([2.0])

        with pytest.raises(ValueError, match="positions"):
            distance_to_reward(env, times, positions, reward_times)

    def test_distance_to_reward_unsorted_times_raises(self, simple_grid_env):
        """Descending times with inferred reward positions must raise."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        # Descending step at index 2 -> 3.
        times = np.array([0.0, 1.0, 5.0, 2.0, 6.0])
        positions = np.column_stack([np.linspace(0, 9, 5), np.linspace(0, 9, 5)])
        reward_times = np.array([3.0])

        with pytest.raises(ValueError, match=r"ascending|sorted"):
            distance_to_reward(env, times, positions, reward_times)

    def test_distance_to_reward_unsorted_times_ok_with_explicit_positions(
        self, simple_grid_env
    ):
        """Unsorted times with explicit reward_positions must not raise."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        times = np.array([0.0, 1.0, 5.0, 2.0, 6.0])
        positions = np.full((5, 2), 5.0)
        reward_times = np.array([3.0])
        reward_positions = np.array([[5.0, 5.0]])

        result = distance_to_reward(
            env,
            times,
            positions,
            reward_times,
            reward_positions=reward_positions,
            metric="euclidean",
        )

        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_distance_to_reward_finite_inputs_unchanged(self, simple_grid_env):
        """A finite, sorted example returns correct, unchanged distances."""
        from neurospatial.behavior.navigation import distance_to_reward

        env = simple_grid_env
        n_samples = 10
        times = np.linspace(0, 9, n_samples)
        # Animal stays on the y=5 line, moving x from 0 to 9.
        positions = np.column_stack(
            [np.linspace(0, 9, n_samples), np.full(n_samples, 5.0)]
        )
        # Reward inferred at t=4.5 -> x=4.5, y=5.
        reward_times = np.array([4.5])

        result = distance_to_reward(
            env, times, positions, reward_times, metric="euclidean"
        )

        assert result.shape == (n_samples,)
        assert np.all(np.isfinite(result))
        # Distance decreases monotonically toward the reward, then increases.
        reward_idx = int(np.argmin(result))
        assert np.all(np.diff(result[: reward_idx + 1]) <= 1e-9)
        assert np.all(np.diff(result[reward_idx:]) >= -1e-9)
        # First sample is at x=0; reward at x=4.5 -> distance 4.5.
        assert result[0] == pytest.approx(4.5, abs=0.6)


# =============================================================================
# Test distance_to_boundary
# =============================================================================


class TestDistanceToBoundary:
    """Tests for distance_to_boundary function."""

    def test_basic_edge_distance(self, simple_grid_env):
        """Test basic distance to edge calculation."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        # Positions: center and corner
        positions = np.array([[5.0, 5.0], [0.0, 0.0]])

        result = distance_to_boundary(env, positions, boundary_type="edge")

        assert result.shape == (2,)
        # Center is farther from edge than corner
        assert result[0] > result[1]
        # Corner is at edge, distance should be 0 or very small
        assert result[1] < 1.0

    def test_region_boundary_distance(self, simple_grid_env):
        """Test distance to region boundary."""
        from shapely.geometry import box as shapely_box

        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        # Add a region (simple polygon using shapely)
        goal_polygon = shapely_box(7.0, 7.0, 9.0, 9.0)  # minx, miny, maxx, maxy
        region_builder = Regions(env.regions.values())
        region_builder.add("goal", polygon=goal_polygon)
        env = env.with_regions(region_builder)

        # Positions: one far from region, one near region
        positions = np.array([[1.0, 1.0], [6.0, 6.0]])

        result = distance_to_boundary(
            env, positions, boundary_type="region", region_name="goal"
        )

        assert result.shape == (2,)
        # Position at (1,1) is farther from goal region than (6,6)
        assert result[0] > result[1]

    def test_missing_region_raises(self, simple_grid_env):
        """Test missing region name raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"Region.*not found"):
            distance_to_boundary(
                env, positions, boundary_type="region", region_name="nonexistent"
            )

    def test_region_name_required_for_region_type(self, simple_grid_env):
        """Test region_name is required when boundary_type='region'."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"region_name.*required"):
            distance_to_boundary(env, positions, boundary_type="region")

    def test_empty_positions(self, simple_grid_env):
        """Test empty positions returns empty array."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.empty((0, 2))

        result = distance_to_boundary(env, positions, boundary_type="edge")

        assert len(result) == 0

    def test_geodesic_vs_euclidean(self, simple_grid_env):
        """Test geodesic and euclidean metrics give different results."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        result_geodesic = distance_to_boundary(
            env, positions, boundary_type="edge", metric="geodesic"
        )

        result_euclidean = distance_to_boundary(
            env, positions, boundary_type="edge", metric="euclidean"
        )

        # Both should return valid distances
        assert not np.isnan(result_geodesic[0])
        assert not np.isnan(result_euclidean[0])

    def test_invalid_boundary_type_raises(self, simple_grid_env):
        """Test invalid boundary_type raises ValueError."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        positions = np.array([[5.0, 5.0]])

        with pytest.raises(ValueError, match=r"boundary_type"):
            distance_to_boundary(env, positions, boundary_type="invalid")

    def test_position_outside_env_returns_nan(self, simple_grid_env):
        """Test positions outside environment return NaN."""
        from neurospatial.behavior.navigation import distance_to_boundary

        env = simple_grid_env

        # Position far outside the 10x10 grid
        positions = np.array([[100.0, 100.0]])

        result = distance_to_boundary(env, positions, boundary_type="edge")

        # Position outside env → invalid bin → NaN
        assert np.isnan(result[0])

    def test_edge_bins_1d_track_are_the_two_ends(self):
        """On a 1D linear track the boundary is the two end bins.

        A path-graph track has interior degree 2 and endpoint degree 1, so
        the edge bins are exactly the two ends of the track. Distance to the
        boundary must be largest in the middle of the track.
        """
        import networkx as nx

        from neurospatial import Environment
        from neurospatial.behavior.navigation import _find_edge_bins

        graph = nx.path_graph(8)
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]["pos"] = (float(i) * 5.0, 0.0)
        for u, v in graph.edges:
            graph.edges[u, v]["distance"] = 5.0
        env = Environment.from_graph(
            graph,
            edge_order=list(graph.edges),
            edge_spacing=0.0,
            bin_size=5.0,
            track_width=5.0,
        )

        edge_bins = _find_edge_bins(env)

        # Exactly the two ends of the linearized track.
        assert len(edge_bins) == 2
        assert set(edge_bins) == {0, env.n_bins - 1}

        # The middle of the track is the farthest point from either end.
        from neurospatial.behavior.navigation import distance_to_boundary

        # bin_centers are along the linearized axis; pick the geometric
        # midpoint and the two ends.
        mid_pos = env.bin_centers[env.n_bins // 2]
        end_pos = env.bin_centers[0]
        positions = np.vstack([mid_pos, end_pos])
        dist = distance_to_boundary(env, positions, boundary_type="edge")
        assert dist[0] > dist[1]

    def test_edge_bins_masked_2d_includes_inner_hole_boundary(self):
        """On a masked 2D env edge bins include the boundary of an inner hole.

        Bins flanking a masked-out hole have fewer neighbors than a fully
        surrounded interior bin and so must be flagged as edge bins, not
        only the outer rim.
        """
        from neurospatial import Environment
        from neurospatial.behavior.navigation import _find_edge_bins

        # 7x7 occupied grid with a single hole punched in the center.
        x = np.arange(7, dtype=float)
        y = np.arange(7, dtype=float)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()])
        # Drop the exact center (3, 3) so it becomes an unoccupied hole.
        keep = ~((positions[:, 0] == 3.0) & (positions[:, 1] == 3.0))
        positions = positions[keep]

        env = Environment.from_samples(positions, bin_size=1.0)

        edge_bins = set(_find_edge_bins(env))

        # The four bins orthogonally adjacent to the hole at (3, 3) must be
        # edge bins because they border unoccupied space.
        for hole_neighbor in [(3.0, 2.0), (3.0, 4.0), (2.0, 3.0), (4.0, 3.0)]:
            bin_idx = int(env.bin_at(np.asarray([hole_neighbor]))[0])
            assert bin_idx in edge_bins, (
                f"bin at {hole_neighbor} (index {bin_idx}) borders the hole "
                "and should be an edge bin"
            )
