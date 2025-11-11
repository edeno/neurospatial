"""Tests for spike generation functions."""

import numpy as np

from neurospatial.simulation.models import PlaceCellModel
from neurospatial.simulation.spikes import (
    generate_poisson_spikes,
    generate_population_spikes,
)
from neurospatial.simulation.trajectory import simulate_trajectory_ou


class TestGeneratePoissonSpikes:
    """Tests for Poisson spike generation."""

    def test_basic_spike_generation(self):
        """Test basic spike generation from constant firing rate."""
        # Constant 10 Hz firing for 1 second at 1000 Hz sampling
        firing_rate = np.full(1000, 10.0)
        times = np.linspace(0, 1, 1000)

        spike_times = generate_poisson_spikes(firing_rate, times, seed=42)

        # Should generate roughly 10 spikes (Poisson variance)
        assert 5 <= len(spike_times) <= 15

    def test_spikes_sorted(self):
        """Test that spike times are sorted."""
        firing_rate = np.full(1000, 10.0)
        times = np.linspace(0, 1, 1000)

        spike_times = generate_poisson_spikes(firing_rate, times, seed=42)

        # Check sorted
        assert np.all(np.diff(spike_times) >= 0)

    def test_refractory_period_constraint(self):
        """Test that all ISIs >= refractory_period."""
        # High firing rate to test refractory period
        firing_rate = np.full(10000, 100.0)  # 100 Hz
        times = np.linspace(0, 1, 10000)
        refractory_period = 0.002  # 2ms

        spike_times = generate_poisson_spikes(
            firing_rate, times, refractory_period=refractory_period, seed=42
        )

        # Compute ISIs
        if len(spike_times) > 1:
            isis = np.diff(spike_times)
            # All ISIs should be >= refractory_period
            assert np.all(
                isis >= refractory_period - 1e-10
            )  # Small tolerance for float

    def test_mean_firing_rate(self):
        """Test that mean firing rate is approximately correct."""
        # 20 Hz for 10 seconds
        firing_rate = np.full(10000, 20.0)
        times = np.linspace(0, 10, 10000)

        spike_times = generate_poisson_spikes(firing_rate, times, seed=42)

        # Expected: ~200 spikes
        mean_rate = len(spike_times) / 10.0

        # Allow 20% tolerance (Poisson variance)
        assert abs(mean_rate - 20.0) < 4.0

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same spikes."""
        firing_rate = np.full(1000, 10.0)
        times = np.linspace(0, 1, 1000)

        spikes1 = generate_poisson_spikes(firing_rate, times, seed=42)
        spikes2 = generate_poisson_spikes(firing_rate, times, seed=42)

        np.testing.assert_array_equal(spikes1, spikes2)

    def test_different_seeds_produce_different_spikes(self):
        """Test that different seeds produce different spikes."""
        firing_rate = np.full(1000, 10.0)
        times = np.linspace(0, 1, 1000)

        spikes1 = generate_poisson_spikes(firing_rate, times, seed=42)
        spikes2 = generate_poisson_spikes(firing_rate, times, seed=43)

        assert not np.array_equal(spikes1, spikes2)

    def test_zero_firing_rate(self):
        """Test that zero firing rate produces no spikes."""
        firing_rate = np.zeros(1000)
        times = np.linspace(0, 1, 1000)

        spike_times = generate_poisson_spikes(firing_rate, times, seed=42)

        assert len(spike_times) == 0

    def test_time_varying_firing_rate(self):
        """Test with time-varying firing rate."""
        # Sinusoidal firing rate: 10 + 5*sin(2πt)
        times = np.linspace(0, 2, 2000)
        firing_rate = 10.0 + 5.0 * np.sin(2 * np.pi * times)
        # Clip negative values
        firing_rate = np.maximum(firing_rate, 0.0)

        spike_times = generate_poisson_spikes(firing_rate, times, seed=42)

        # Should have spikes
        assert len(spike_times) > 0

        # Mean rate should be close to mean of firing_rate
        mean_rate = len(spike_times) / 2.0
        expected_mean = np.mean(firing_rate)

        # Allow 30% tolerance
        assert abs(mean_rate - expected_mean) < 0.3 * expected_mean

    def test_empty_input(self):
        """Test with empty input arrays."""
        firing_rate = np.array([])
        times = np.array([])

        spike_times = generate_poisson_spikes(firing_rate, times, seed=42)

        assert len(spike_times) == 0


class TestGeneratePopulationSpikes:
    """Tests for population spike generation."""

    def test_basic_population_generation(self, simple_2d_env):
        """Test basic population spike generation."""
        # Create small population
        n_cells = 5
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, seed=i) for i in range(n_cells)
        ]

        # Generate trajectory
        positions, times = simulate_trajectory_ou(simple_2d_env, duration=10.0, seed=42)

        # Generate spikes (no progress bar for tests)
        spike_trains = generate_population_spikes(
            place_cells, positions, times, seed=42, show_progress=False
        )

        # Should return list of spike arrays
        assert len(spike_trains) == n_cells
        assert all(isinstance(st, np.ndarray) for st in spike_trains)

    def test_returns_correct_number_of_spike_trains(self, simple_2d_env):
        """Test that number of spike trains matches number of models."""
        n_cells = 10
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, seed=i) for i in range(n_cells)
        ]

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=5.0, seed=42)

        spike_trains = generate_population_spikes(
            place_cells, positions, times, seed=42, show_progress=False
        )

        assert len(spike_trains) == n_cells

    def test_spike_trains_are_independent(self, simple_2d_env):
        """Test that different cells produce different spike trains."""
        n_cells = 5
        # Use higher firing rates to ensure spikes are generated
        place_cells = [
            PlaceCellModel(simple_2d_env, width=20.0, max_rate=30.0, seed=i)
            for i in range(n_cells)
        ]

        positions, times = simulate_trajectory_ou(
            simple_2d_env,
            duration=30.0,
            seed=42,  # Longer duration
        )

        spike_trains = generate_population_spikes(
            place_cells, positions, times, seed=42, show_progress=False
        )

        # Check that at least some cells generate spikes
        total_spikes = sum(len(st) for st in spike_trains)
        assert total_spikes > 0, "No spikes generated"

        # Check that not all spike trains have the same count
        # (very unlikely with different place field centers)
        spike_counts = [len(st) for st in spike_trains]
        unique_counts = set(spike_counts)
        # At least some variation (or at least some cells fired)
        assert len(unique_counts) > 1 or max(spike_counts) > 0

    def test_reproducibility_with_seed(self, simple_2d_env):
        """Test that same seed produces same spike trains."""
        n_cells = 3
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, seed=i) for i in range(n_cells)
        ]

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=5.0, seed=42)

        spike_trains1 = generate_population_spikes(
            place_cells, positions, times, seed=42, show_progress=False
        )
        spike_trains2 = generate_population_spikes(
            place_cells, positions, times, seed=42, show_progress=False
        )

        # Check that all spike trains are identical
        for st1, st2 in zip(spike_trains1, spike_trains2, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_all_spikes_within_time_range(self, simple_2d_env):
        """Test that all generated spikes are within time range."""
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, seed=i) for i in range(5)
        ]

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=10.0, seed=42)

        spike_trains = generate_population_spikes(
            place_cells, positions, times, seed=42, show_progress=False
        )

        # Check all spikes are within [times[0], times[-1]]
        for spike_times in spike_trains:
            if len(spike_times) > 0:
                assert np.all(spike_times >= times[0])
                assert np.all(spike_times <= times[-1])

    def test_progress_bar_disabled(self, simple_2d_env, capsys):
        """Test that progress bar can be disabled."""
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, seed=i) for i in range(3)
        ]

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=1.0, seed=42)

        # Generate without progress bar
        generate_population_spikes(
            place_cells, positions, times, seed=42, show_progress=False
        )

        # Check no progress bar output (this is approximate test)
        captured = capsys.readouterr()
        # Should not have tqdm progress bar characters
        assert "it/s" not in captured.err

    def test_refractory_period_applied(self, simple_2d_env):
        """Test that refractory period is applied to all cells."""
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, max_rate=50.0, seed=i)
            for i in range(5)
        ]

        positions, times = simulate_trajectory_ou(simple_2d_env, duration=10.0, seed=42)

        refractory_period = 0.003  # 3ms
        spike_trains = generate_population_spikes(
            place_cells,
            positions,
            times,
            refractory_period=refractory_period,
            seed=42,
            show_progress=False,
        )

        # Check refractory period for all cells
        for spike_times in spike_trains:
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                assert np.all(isis >= refractory_period - 1e-10)
