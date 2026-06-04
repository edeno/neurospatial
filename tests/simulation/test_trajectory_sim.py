"""Tests for trajectory simulation functions."""

import numpy as np
import pytest

from neurospatial.simulation.trajectory import (
    simulate_trajectory_laps,
    simulate_trajectory_ou,
    simulate_trajectory_sinusoidal,
)


@pytest.fixture
def linear_track_env():
    """A real 1D linearized track (GraphLayout) for sinusoidal motion tests."""
    import networkx as nx

    from neurospatial import Environment

    g = nx.Graph()
    g.add_node(0, pos=(0.0,))
    g.add_node(1, pos=(50.0,))
    g.add_node(2, pos=(100.0,))
    g.add_edge(0, 1, distance=50.0)
    g.add_edge(1, 2, distance=50.0)
    env = Environment.from_graph(
        graph=g, edge_order=[(0, 1), (1, 2)], edge_spacing=0.0, bin_size=2.0
    )
    env.units = "cm"
    return env


class TestSimulateTrajectoryOU:
    """Tests for Ornstein-Uhlenbeck trajectory simulation."""

    def test_basic_trajectory_generation(self, simple_2d_env):
        """Test basic trajectory generation works."""
        positions, times = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            dt=0.01,
            seed=42,
            speed_units="cm",
        )

        # Check shapes
        assert positions.shape[1] == 2  # 2D environment
        assert len(times) == len(positions)
        assert len(times) == 100  # 1.0s / 0.01s dt

    def test_trajectory_stays_in_bounds(self, simple_2d_env):
        """Test that trajectory stays within environment bounds."""
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=10.0,
            boundary_mode="reflect",
            seed=42,
            speed_units="cm",
        )

        # Check all positions are within environment
        for pos in positions:
            assert simple_2d_env.contains(pos), f"Position {pos} outside environment"

    def test_velocity_autocorrelation_is_correlated_and_decays(self, simple_2d_env):
        """OU velocity is temporally correlated: 0 < acf(coherence_time) < acf(0)."""
        coherence_time = 0.5
        positions, times = simulate_trajectory_ou(
            simple_2d_env,
            duration=200.0,
            dt=0.01,
            coherence_time=coherence_time,
            seed=42,
            speed_units="cm",
        )
        dt = times[1] - times[0]
        vx = np.diff(positions, axis=0)[:, 0]
        vx = vx - vx.mean()
        acf0 = np.mean(vx**2)
        lag = int(coherence_time / dt)
        acf_lag = np.mean(vx[:-lag] * vx[lag:])
        # White noise -> ~0; a constant -> ~acf0. A real OU process sits strictly
        # between, confirming coherence_time produces decaying-but-nonzero memory.
        assert 0.0 < acf_lag < acf0

    def test_boundary_mode_reflect(self, simple_2d_env):
        """Test reflect boundary mode."""
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=5.0,
            boundary_mode="reflect",
            seed=42,
            speed_units="cm",
        )

        # All positions should be in bounds
        for pos in positions:
            assert simple_2d_env.contains(pos)

    def test_boundary_mode_periodic(self, simple_2d_env):
        """Test periodic boundary mode."""
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=5.0,
            boundary_mode="periodic",
            seed=42,
            speed_units="cm",
        )

        # Positions should wrap around
        # Check that positions are within dimension ranges
        for dim in range(simple_2d_env.n_dims):
            range_min, range_max = simple_2d_env.dimension_ranges[dim]
            assert np.all(positions[:, dim] >= range_min)
            assert np.all(positions[:, dim] <= range_max)

    def test_boundary_mode_stop(self, simple_2d_env):
        """Test stop boundary mode."""
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=5.0,
            boundary_mode="stop",
            seed=42,
            speed_units="cm",
        )

        # All positions should be in bounds
        for pos in positions:
            assert simple_2d_env.contains(pos)

    def test_reproducibility_with_seed(self, simple_2d_env):
        """Test that same seed produces same trajectory."""
        pos1, times1 = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            seed=42,
            speed_units="cm",
        )
        pos2, times2 = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            seed=42,
            speed_units="cm",
        )

        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(times1, times2)

    def test_different_seeds_produce_different_trajectories(self, simple_2d_env):
        """Test that different seeds produce different trajectories."""
        pos1, _ = simulate_trajectory_ou(
            simple_2d_env, duration=1.0, seed=42, speed_units="cm"
        )
        pos2, _ = simulate_trajectory_ou(
            simple_2d_env, duration=1.0, seed=43, speed_units="cm"
        )

        assert not np.array_equal(pos1, pos2)

    def test_requires_env_units(self, simple_2d_env):
        """Test that ValueError is raised if env.units is not set."""
        # Remove units
        simple_2d_env.units = None

        with pytest.raises(ValueError, match="units must be set"):
            simulate_trajectory_ou(simple_2d_env, duration=1.0, speed_units="cm")

    def test_speed_units_must_match_env_units(self, simple_2d_env):
        """speed_units must match env.units exactly.

        Auto-conversion between unit families was removed in v0.4
        because silent rescaling was easy to miss. Callers must
        pre-convert if their speeds are in a different unit.
        """
        assert simple_2d_env.units == "cm"

        # Mismatched units must raise.
        with pytest.raises(ValueError, match=r"speed_units .* must match env\.units"):
            simulate_trajectory_ou(
                simple_2d_env,
                duration=1.0,
                speed_mean=0.10,
                speed_units="m",
                seed=42,
            )

        # Matching units works.
        positions, _times = simulate_trajectory_ou(
            simple_2d_env,
            duration=1.0,
            speed_mean=10.0,  # cm/s
            speed_units="cm",
            seed=42,
        )
        assert len(positions) > 0


class TestSimulateTrajectorysinusoidal:
    """Tests for sinusoidal trajectory simulation."""

    def test_requires_1d_environment(self, simple_2d_env):
        """Test that ValueError is raised for non-1D environment."""
        with pytest.raises(ValueError, match="1D environments"):
            simulate_trajectory_sinusoidal(simple_2d_env, duration=10.0)

    def test_basic_sinusoidal_generation(self, linear_track_env):
        """Produces (n_time, 1) positions sampled at the requested rate."""
        positions, times = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=10.0,
            sampling_frequency=100.0,
            speed=20.0,
            period=4.0,
            seed=1,
        )
        assert positions.shape == (1000, 1)  # 10s * 100Hz
        assert times.shape == (1000,)
        assert times[1] - times[0] == pytest.approx(0.01)

    def test_stays_within_track_bounds(self, linear_track_env):
        """Sinusoidal positions never leave the linearized track range."""
        range_min, range_max = linear_track_env.dimension_ranges[0]
        positions, _ = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=20.0,
            sampling_frequency=100.0,
            speed=20.0,
            period=4.0,
        )
        assert positions.min() >= range_min - 1e-9
        assert positions.max() <= range_max + 1e-9

    def test_is_periodic_with_given_period(self, linear_track_env):
        """With pauses off, x(t) repeats every `period` seconds."""
        period = 4.0
        positions, _times = simulate_trajectory_sinusoidal(
            linear_track_env,
            duration=12.0,
            sampling_frequency=100.0,
            speed=20.0,
            period=period,
            pause_at_peaks=False,
        )
        n_per_period = int(period * 100.0)
        # Compare the first period to the second period sample-for-sample.
        first = positions[:n_per_period, 0]
        second = positions[n_per_period : 2 * n_per_period, 0]
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_seed_has_no_effect(self, linear_track_env):
        """`seed` is a documented no-op: output is identical regardless of seed."""
        kw = {
            "duration": 5.0,
            "sampling_frequency": 100.0,
            "speed": 20.0,
            "period": 4.0,
        }
        a, _ = simulate_trajectory_sinusoidal(linear_track_env, seed=1, **kw)
        b, _ = simulate_trajectory_sinusoidal(linear_track_env, seed=999, **kw)
        np.testing.assert_array_equal(a, b)


class TestSimulateTrajectoryLaps:
    """Tests for lap-based trajectory simulation."""

    def test_n_laps_produces_correct_number_of_laps(self, simple_2d_env):
        """Test that n_laps parameter produces correct number of laps."""
        n_laps = 5
        positions, _times, metadata = simulate_trajectory_laps(
            simple_2d_env,
            n_laps=n_laps,
            speed_mean=10.0,
            pause_duration=0.0,
            return_metadata=True,
            seed=42,
        )

        # Check that metadata contains lap_ids
        assert "lap_ids" in metadata, "metadata should contain 'lap_ids'"

        # Check that lap_ids has the same length as positions
        assert len(metadata["lap_ids"]) == len(positions), (
            "lap_ids should have same length as positions"
        )

        # Count unique laps (0-indexed, so n_laps should give lap_ids 0 to n_laps-1)
        unique_laps = np.unique(metadata["lap_ids"])
        assert len(unique_laps) == n_laps, (
            f"Expected {n_laps} laps, got {len(unique_laps)}"
        )
        assert np.all(unique_laps == np.arange(n_laps)), (
            "lap_ids should be 0 to n_laps-1"
        )

    def test_metadata_contains_lap_ids_and_directions(self, simple_2d_env):
        """Test that metadata contains lap_ids and direction arrays."""
        positions, _times, metadata = simulate_trajectory_laps(
            simple_2d_env,
            n_laps=3,
            return_metadata=True,
            seed=42,
        )

        # Check required metadata keys
        assert "lap_ids" in metadata, "metadata should contain 'lap_ids'"
        assert "direction" in metadata, "metadata should contain 'direction'"
        assert "lap_boundaries" in metadata, "metadata should contain 'lap_boundaries'"

        # Check shapes
        assert len(metadata["lap_ids"]) == len(positions), (
            "lap_ids should have same length as positions"
        )
        assert len(metadata["direction"]) == len(positions), (
            "direction should have same length as positions"
        )

        # Check direction values are valid
        unique_directions = np.unique(metadata["direction"])
        valid_directions = {"outbound", "inbound"}
        for direction in unique_directions:
            assert direction in valid_directions, (
                f"Invalid direction '{direction}', expected one of {valid_directions}"
            )

    def test_pauses_at_lap_ends(self, simple_2d_env):
        """Test that trajectory includes pauses at lap ends."""
        pause_duration = 0.5  # seconds
        sampling_frequency = 100.0  # Hz
        expected_pause_samples = int(pause_duration * sampling_frequency)

        positions, _times, metadata = simulate_trajectory_laps(
            simple_2d_env,
            n_laps=3,
            pause_duration=pause_duration,
            sampling_frequency=sampling_frequency,
            return_metadata=True,
            seed=42,
        )

        # Check lap boundaries
        lap_boundaries = metadata["lap_boundaries"]

        # At each lap boundary (except first), check for pauses
        # Pause is indicated by zero velocity (position not changing)
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        for i in range(1, len(lap_boundaries)):
            boundary_idx = lap_boundaries[i]
            if boundary_idx > 0 and boundary_idx < len(velocities):
                # Check samples around boundary for low velocity (paused)
                # Allow some tolerance for numerical issues
                start_idx = max(0, boundary_idx - expected_pause_samples // 2)
                end_idx = min(
                    len(velocities), boundary_idx + expected_pause_samples // 2
                )

                # At least some samples should have very low velocity (paused)
                paused_samples = velocities[start_idx:end_idx] < 0.1
                assert np.sum(paused_samples) > 0, (
                    f"Expected pause at lap boundary {i}, but found no low-velocity samples"
                )

    def test_return_metadata_false_returns_tuple_of_two(self, simple_2d_env):
        """Test that return_metadata=False returns only (positions, times)."""
        result = simulate_trajectory_laps(
            simple_2d_env,
            n_laps=3,
            return_metadata=False,
            seed=42,
        )

        # Should return tuple of 2 elements
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, (
            "Should return (positions, times) when return_metadata=False"
        )

        positions, times = result
        assert len(positions) == len(times), (
            "positions and times should have same length"
        )

    def test_custom_paths(self, simple_2d_env):
        """Test that custom outbound and inbound paths are used."""
        # Create custom paths (just use some bin indices)
        n_bins = simple_2d_env.n_bins
        outbound_path = list(range(0, min(10, n_bins), 2))  # [0, 2, 4, 6, 8]
        inbound_path = list(range(min(9, n_bins - 1), 0, -2))  # [9, 7, 5, 3, 1]

        positions, times, metadata = simulate_trajectory_laps(
            simple_2d_env,
            n_laps=2,
            outbound_path=outbound_path,
            inbound_path=inbound_path,
            return_metadata=True,
            seed=42,
        )

        # Check that trajectory was generated
        assert len(positions) > 0, "Should generate non-empty trajectory"
        assert len(times) > 0, "Should generate non-empty times"

        # Check metadata structure
        assert "lap_ids" in metadata
        assert "direction" in metadata

    def test_reproducibility_with_seed(self, simple_2d_env):
        """Test that same seed produces same trajectory."""
        result1 = simulate_trajectory_laps(
            simple_2d_env, n_laps=3, seed=42, return_metadata=True
        )
        result2 = simulate_trajectory_laps(
            simple_2d_env, n_laps=3, seed=42, return_metadata=True
        )

        positions1, times1, metadata1 = result1
        positions2, times2, metadata2 = result2

        np.testing.assert_array_equal(positions1, positions2)
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(metadata1["lap_ids"], metadata2["lap_ids"])

    def test_trajectory_stays_in_environment(self, simple_2d_env):
        """Test that all positions are within environment bounds."""
        positions, _times = simulate_trajectory_laps(
            simple_2d_env,
            n_laps=5,
            return_metadata=False,
            seed=42,
        )

        # Use map_points_to_bins to check - this is more efficient than calling
        # contains() in a loop, and it validates that positions can be mapped
        from neurospatial.ops import map_points_to_bins

        bin_indices = map_points_to_bins(positions, simple_2d_env)

        # All bin indices should be valid (>= 0)
        assert np.all(bin_indices >= 0), "Some positions could not be mapped to bins"

        # Spot check a few positions with contains()
        sample_indices = np.linspace(
            0, len(positions) - 1, min(10, len(positions)), dtype=int
        )
        for idx in sample_indices:
            pos = positions[idx]
            assert simple_2d_env.contains(pos), (
                f"Position {pos} at index {idx} is outside environment bounds"
            )
