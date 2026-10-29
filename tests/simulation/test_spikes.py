"""Tests for spike generation functions."""

import numpy as np
import pytest

from neurospatial.simulation.models import HeadDirectionCellModel, PlaceCellModel
from neurospatial.simulation.spikes import (
    add_modulation,
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

    def test_per_bin_probability_uses_exponential_not_linear(self):
        """Per-bin spike probability must be 1 - exp(-rate*dt), not rate*dt.

        Poisson counts have mean and variance one when rate*dt=1, while only
        about 63% of intervals contain spikes. Checking occupancy separately
        from counts detects both Bernoulli undercounting and saturated rates.
        """
        n_bins = 20_000
        dt = 0.1
        rate = 10.0
        times = np.arange(n_bins + 1) * dt
        firing_rate = np.full(len(times), rate)

        spike_times = generate_poisson_spikes(
            firing_rate, times, refractory_period=0.0, seed=0
        )

        counts = np.histogram(spike_times, bins=times)[0]
        empirical_p = np.mean(counts > 0)
        expected_p = 1.0 - np.exp(-rate * dt)  # ~0.6321

        assert empirical_p == pytest.approx(expected_p, abs=0.015)
        assert counts.mean() == pytest.approx(rate * dt, abs=0.035)
        assert counts.var() == pytest.approx(rate * dt, abs=0.08)
        assert np.any(counts > 1)

    def test_coarse_nonuniform_intervals_preserve_time_varying_rates(self):
        """Coarse intervals retain multiple events and their own rate integrals."""
        times = np.r_[np.arange(1001) * 0.01, 10 + np.arange(1, 1001) * 0.1]
        firing_rate = np.r_[np.full(1000, 20.0), np.full(1001, 30.0)]
        spikes = generate_poisson_spikes(
            firing_rate, times, refractory_period=0.0, seed=17
        )
        assert np.sum(spikes < 10) == pytest.approx(20 * 10, rel=0.2)
        assert np.sum(spikes >= 10) == pytest.approx(30 * 100, rel=0.08)
        assert np.all((spikes >= times[0]) & (spikes < times[-1]))
        assert np.any(~np.isin(spikes, times))

    def test_final_sample_does_not_extrapolate_recording(self):
        """The endpoint rate must not invent a final unobserved interval."""
        spikes = generate_poisson_spikes(
            np.array([0.0, 0.0, 1e6]),
            np.array([10.0, 11.0, 12.0]),
            refractory_period=0.0,
            seed=0,
        )
        assert spikes.size == 0

    def test_coarse_candidates_still_obey_refractory_period(self):
        """Multiple candidates in a coarse interval respect continuous-time ISIs."""
        times = np.array([10.0, 11.0, 12.0])
        spikes = generate_poisson_spikes(
            np.full(3, 1000.0), times, refractory_period=0.002, seed=0
        )
        assert spikes.size > len(times)
        assert np.all(np.diff(spikes) >= 0.002)
        assert np.all((spikes >= times[0]) & (spikes < times[-1]))

    def test_nonincreasing_times_raises(self):
        """Non-increasing times raise ValueError mentioning 'strictly increasing'."""
        firing_rate = np.full(4, 10.0)
        # Decreasing step between index 1 and 2.
        times = np.array([0.0, 0.1, 0.05, 0.2])

        with pytest.raises(ValueError, match="strictly increasing"):
            generate_poisson_spikes(firing_rate, times, seed=42)

        # Duplicate (zero-width) step also rejected.
        times_dup = np.array([0.0, 0.1, 0.1, 0.2])
        with pytest.raises(ValueError, match="strictly increasing"):
            generate_poisson_spikes(firing_rate, times_dup, seed=42)

    def test_nonuniform_times_uses_per_bin_dt(self):
        """Spike probability tracks each bin's width, not the first interval.

        Build strictly increasing but non-uniform times where the first
        many bins are narrow and the rest are 10x wider, with a constant
        rate. With per-bin dt the wide bins fire ~10x more often than the
        narrow bins. The old code used the first (narrow) interval as a
        uniform dt for every bin, underestimating the wide bins by ~10x.
        """
        rate = 1.0  # Hz; small rate*dt so probability ~ rate*dt is ~linear
        n_each = 50_000
        narrow_dt = 0.001
        wide_dt = 0.01  # 10x wider

        # First half narrow, second half wide.
        deltas = np.concatenate([np.full(n_each, narrow_dt), np.full(n_each, wide_dt)])
        times = np.concatenate([[0.0], np.cumsum(deltas)])
        firing_rate = np.full(len(times), rate)

        spikes = generate_poisson_spikes(
            firing_rate, times, refractory_period=0.0, seed=0
        )

        # Split spike counts by region using the time boundary.
        boundary = times[n_each]
        n_narrow = int(np.sum(spikes < boundary))
        n_wide = int(np.sum(spikes >= boundary))

        # Per-bin dt: narrow bins average n_each * rate * narrow_dt spikes,
        # wide bins average n_each * rate * wide_dt spikes. The wide region
        # should fire about wide_dt/narrow_dt = 10x more.
        ratio = n_wide / n_narrow
        assert 8.0 < ratio < 12.0, (
            f"wide/narrow spike ratio {ratio:.2f} should be ~10 with per-bin dt; "
            "a uniform first-interval dt would give ~1.0"
        )

    def test_length_mismatch_raises(self):
        """firing_rate and times length mismatch raises ValueError naming both lengths."""
        firing_rate = np.full(5, 10.0)
        times = np.linspace(0, 1, 6)

        with pytest.raises(ValueError, match="firing_rate=5, times=6"):
            generate_poisson_spikes(firing_rate, times, seed=42)

    def test_nan_firing_rate_raises(self):
        """A NaN in firing_rate raises ValueError naming 'firing_rate'."""
        firing_rate = np.full(100, 10.0)
        firing_rate[50] = np.nan
        times = np.linspace(0, 1, 100)

        with pytest.raises(ValueError, match="firing_rate"):
            generate_poisson_spikes(firing_rate, times, seed=42)

    def test_inf_firing_rate_raises(self):
        """A +inf in firing_rate raises ValueError."""
        firing_rate = np.full(100, 10.0)
        firing_rate[10] = np.inf
        times = np.linspace(0, 1, 100)

        with pytest.raises(ValueError, match="firing_rate"):
            generate_poisson_spikes(firing_rate, times, seed=42)

    def test_negative_rate_raises(self):
        """A negative rate is a bug in the rate, not zero activity."""
        firing_rate = np.full(1000, 2.0)
        firing_rate[300] = -5.0
        times = np.linspace(0, 1, 1000)

        with pytest.raises(ValueError, match="non-negative") as caught:
            generate_poisson_spikes(firing_rate, times, seed=42)
        assert "index 300" in str(caught.value)

    def test_interspike_intervals_are_exponential(self):
        """Without a refractory period, a constant rate is a Poisson process."""
        from scipy import stats

        rate = 20.0
        times = np.linspace(0.0, 500.0, 50_001)
        spikes = generate_poisson_spikes(
            np.full(times.shape, rate), times, refractory_period=0.0, seed=1
        )
        assert len(spikes) / 500.0 == pytest.approx(rate, rel=0.03)
        isis = np.diff(spikes)
        assert stats.kstest(isis, "expon", args=(0.0, 1.0 / rate)).pvalue > 0.01

    @pytest.mark.parametrize("rate", [50.0, 200.0])
    def test_refractory_rate_is_dead_time_corrected(self, rate):
        """Each kept spike is followed by a dead time τ: rate r / (1 + r τ)."""
        tau = 0.002
        times = np.linspace(0.0, 200.0, 200_001)
        spikes = generate_poisson_spikes(
            np.full(times.shape, rate), times, refractory_period=tau, seed=2
        )
        realized = len(spikes) / 200.0
        assert realized == pytest.approx(rate / (1 + rate * tau), rel=0.03)
        assert realized < 0.97 * rate or rate * tau < 0.03


class _HeadingRecorder:
    """A model whose rate is constant and which keeps the headings it is given."""

    requires_headings = True

    def __init__(self):
        self.headings = None

    def firing_rate(self, times, positions, *, headings=None):
        self.headings = headings
        return np.full(len(times), 5.0)

    @property
    def ground_truth(self):
        return {"cell_type": "recorder"}


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
        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=10.0, seed=42, speed_units="cm"
        )

        # Generate spikes (no progress bar for tests)
        spike_trains = generate_population_spikes(
            place_cells, times, positions, seed=7, show_progress=False
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

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=5.0, seed=42, speed_units="cm"
        )

        spike_trains = generate_population_spikes(
            place_cells, times, positions, seed=7, show_progress=False
        )

        assert len(spike_trains) == n_cells

    def test_identical_cells_fire_independently(self):
        """Identical rates give uncorrelated trains: each unit has its own stream."""
        times = np.linspace(0.0, 200.0, 20_001)
        positions = np.zeros((len(times), 2))
        models = [_ConstantModel(20.0) for _ in range(4)]

        trains = generate_population_spikes(
            models, times, positions, seed=0, show_progress=False
        )

        edges = np.arange(0.0, 200.0 + 0.5, 0.5)
        counts = np.array([np.histogram(t, edges)[0] for t in trains])
        n_bins = counts.shape[1]
        correlation = np.corrcoef(counts)[np.triu_indices(len(models), k=1)]
        # Null standard deviation of a correlation is ~1/sqrt(n_bins).
        assert np.all(np.abs(correlation) < 4 / np.sqrt(n_bins))

    def test_reproducibility_with_seed(self, simple_2d_env):
        """Test that same seed produces same spike trains."""
        n_cells = 3
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, seed=i) for i in range(n_cells)
        ]

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=5.0, seed=42, speed_units="cm"
        )

        spike_trains1 = generate_population_spikes(
            place_cells, times, positions, seed=7, show_progress=False
        )
        spike_trains2 = generate_population_spikes(
            place_cells, times, positions, seed=7, show_progress=False
        )

        # Check that all spike trains are identical
        for st1, st2 in zip(spike_trains1, spike_trains2, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_all_spikes_within_time_range(self, simple_2d_env):
        """Test that all generated spikes are within time range."""
        place_cells = [
            PlaceCellModel(simple_2d_env, width=8.0, seed=i) for i in range(5)
        ]

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=10.0, seed=42, speed_units="cm"
        )

        spike_trains = generate_population_spikes(
            place_cells, times, positions, seed=7, show_progress=False
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

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=1.0, seed=42, speed_units="cm"
        )

        # Generate without progress bar
        generate_population_spikes(
            place_cells, times, positions, seed=7, show_progress=False
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

        positions, times = simulate_trajectory_ou(
            simple_2d_env, duration=10.0, seed=42, speed_units="cm"
        )

        refractory_period = 0.003  # 3ms
        spike_trains = generate_population_spikes(
            place_cells,
            times,
            positions,
            refractory_period=refractory_period,
            seed=7,
            show_progress=False,
        )

        # Check refractory period for all cells
        for spike_times in spike_trains:
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                assert np.all(isis >= refractory_period - 1e-10)

    def test_derived_heading_of_a_northward_path_is_north(self):
        """Without headings, a model that needs them sees the movement direction."""
        recorder = _HeadingRecorder()
        times = np.linspace(0.0, 1.0, 101)
        positions = np.column_stack([np.zeros_like(times), 10.0 * times])

        generate_population_spikes(
            [recorder], times, positions, seed=7, show_progress=False
        )

        np.testing.assert_allclose(recorder.headings, np.pi / 2, atol=1e-9)

    def test_head_direction_cell_fires_for_the_movement_direction(self):
        """A North cell fires on a northward path, a South cell does not."""
        times = np.linspace(0.0, 2.0, 201)
        positions = np.column_stack([np.zeros_like(times), 10.0 * times])
        north, south = (
            HeadDirectionCellModel(preferred_direction=d, max_rate=50.0)
            for d in (np.pi / 2, -np.pi / 2)
        )

        trains = generate_population_spikes(
            [north, south], times, positions, seed=7, show_progress=False
        )

        assert len(trains[0]) > 60  # ~100 expected at 50 Hz for 2 s
        assert len(trains[1]) < 3

    def test_head_direction_model_holds_heading_through_stops(self):
        """Derived headings carry the last movement direction through a stop.

        The path stands still for 1 s, runs East for 1 s, stands still for 1 s
        and runs North for 1 s (100 Hz). The smoothed speed is exactly zero in
        the middle of each stop, where heading_from_velocity has no heading.
        The simulated cell instead sees the last direction of movement (East),
        and the leading stop takes the first one (East).
        """
        recorder = _HeadingRecorder()
        times = np.arange(400) / 100  # s
        x = 10.0 * np.clip(times - 1.0, 0.0, 1.0)  # cm
        y = 10.0 * np.clip(times - 3.0, 0.0, 1.0)  # cm
        positions = np.column_stack([x, y])

        generate_population_spikes(
            [recorder], times, positions, seed=7, show_progress=False
        )

        assert np.isfinite(recorder.headings).all()
        np.testing.assert_allclose(recorder.headings[:50], 0.0, atol=1e-12)
        np.testing.assert_allclose(recorder.headings[245:255], 0.0, atol=1e-12)
        np.testing.assert_allclose(recorder.headings[-10:], np.pi / 2, atol=1e-9)

    def test_head_direction_model_still_path_needs_headings(self):
        """A path that never moves has no movement direction to derive."""
        model = HeadDirectionCellModel(preferred_direction=0.0)
        times = np.arange(100) / 100
        positions = np.zeros((100, 2))

        with pytest.raises(ValueError, match="never moves") as caught:
            generate_population_spikes(
                [model], times, positions, seed=42, show_progress=False
            )
        assert "headings=" in str(caught.value)

    def test_head_direction_model_coarse_sampling_needs_headings(self):
        """A moving path sampled coarser than max_gap has no usable interval."""
        model = HeadDirectionCellModel(preferred_direction=0.0)
        times = 2 * np.arange(20, dtype=float)  # 0.5 Hz
        positions = np.column_stack([times * 10.0, np.zeros(20)])

        with pytest.raises(ValueError, match="max_gap") as caught:
            generate_population_spikes(
                [model], times, positions, seed=42, show_progress=False
            )
        assert "never moves" in str(caught.value)
        assert "headings=" in str(caught.value)

    def test_explicit_headings_override_the_movement_direction(self):
        """The path runs East but the head points North: the North cell fires."""
        times = np.linspace(0.0, 2.0, 201)
        positions = np.column_stack([10.0 * times, np.zeros_like(times)])
        headings = np.full(len(times), np.pi / 2)
        north, east = (
            HeadDirectionCellModel(preferred_direction=d, max_rate=50.0)
            for d in (np.pi / 2, 0.0)
        )

        trains = generate_population_spikes(
            [north, east],
            times,
            positions,
            headings=headings,
            seed=7,
            show_progress=False,
        )

        assert len(trains[0]) > 60
        assert len(trains[1]) < 3

    def test_spatial_view_cell_without_headings_uses_movement(self):
        """A spatial view cell needs headings; the population supplies them."""
        from shapely.geometry import box

        from neurospatial import Environment
        from neurospatial.simulation.models import SpatialViewCellModel

        env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")
        times = np.linspace(0.0, 2.0, 201)
        positions = np.column_stack([20.0 + 10.0 * times, np.full_like(times, 50.0)])
        east_wall = SpatialViewCellModel(
            env, preferred_view_location=[100.0, 50.0], max_rate=40.0
        )

        derived = generate_population_spikes(
            [east_wall], times, positions, seed=7, show_progress=False
        )
        given = generate_population_spikes(
            [east_wall],
            times,
            positions,
            headings=np.zeros(len(times)),
            seed=7,
            show_progress=False,
        )

        np.testing.assert_array_equal(derived[0], given[0])
        assert len(derived[0]) > 40

    def test_missing_position_names_the_model_and_input(self, simple_2d_env):
        """A NaN rate names the unit and the missing input, not firing_rate."""
        times = np.linspace(0.0, 1.0, 11)
        positions = np.full((11, 2), 50.0)
        positions[4] = np.nan
        models = [
            _ConstantModel(1.0),
            PlaceCellModel(simple_2d_env, center=[50.0, 50.0], width=10.0),
        ]

        with pytest.raises(ValueError, match=r"Model 1 \(PlaceCellModel\)") as caught:
            generate_population_spikes(
                models, times, positions, seed=7, show_progress=False
            )
        assert "1 have a missing (NaN) position" in str(caught.value)
        assert "t = 0.4 s" in str(caught.value)

    def test_missing_heading_names_the_input(self):
        times = np.linspace(0.0, 1.0, 11)
        positions = np.zeros((11, 2))
        headings = np.zeros(11)
        headings[[2, 3]] = np.nan

        with pytest.raises(ValueError, match="2 have a missing heading"):
            generate_population_spikes(
                [HeadDirectionCellModel(preferred_direction=0.0)],
                times,
                positions,
                headings=headings,
                seed=7,
                show_progress=False,
            )


class _ConstantModel:
    """A cell firing at the same rate everywhere."""

    requires_headings = False

    def __init__(self, rate):
        self.rate = rate

    def firing_rate(self, times, positions, *, headings=None):
        return np.full(len(times), self.rate)

    @property
    def ground_truth(self):
        return {"cell_type": "constant", "max_rate": self.rate}


def _circular_mean(angles):
    return float(np.angle(np.mean(np.exp(1j * angles))))


class TestAddModulation:
    """Tests for rhythmic spike modulation."""

    @pytest.fixture(scope="class")
    def dense_train(self):
        """200,000 spikes spread uniformly over 100 s (800 theta cycles)."""
        return np.sort(np.random.default_rng(11).uniform(0.0, 100.0, 200_000))

    @pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi, -np.pi / 2])
    def test_kept_spikes_concentrate_at_the_preferred_phase(self, dense_train, phase):
        """φ = 0 keeps cosine peaks, φ = π/2 sine peaks, and so on."""
        freq = 8.0
        kept = add_modulation(
            dense_train,
            modulation_freq=freq,
            modulation_depth=0.8,
            modulation_phase=phase,
            seed=3,
        )
        oscillation_phase = 2 * np.pi * freq * kept
        error = np.angle(np.exp(1j * (_circular_mean(oscillation_phase) - phase)))
        assert abs(error) < 0.02

    @pytest.mark.parametrize("depth", [1e-6, 0.25, 0.5, 1.0])
    def test_kept_fraction_is_one_over_one_plus_depth(self, dense_train, depth):
        """Thinning keeps 1 / (1 + d) of the spikes; all of them as d -> 0."""
        kept = add_modulation(
            dense_train, modulation_freq=8.0, modulation_depth=depth, seed=4
        )
        assert len(kept) / len(dense_train) == pytest.approx(1 / (1 + depth), abs=0.005)

    def test_preferred_phase_keeps_the_input_rate(self, dense_train):
        """Every spike at the preferred phase is kept, so its rate is unchanged."""
        freq, phase = 8.0, np.pi
        kept = add_modulation(
            dense_train,
            modulation_freq=freq,
            modulation_depth=0.5,
            modulation_phase=phase,
            seed=5,
        )

        def near_preferred(spikes):
            offset = np.angle(np.exp(1j * (2 * np.pi * freq * spikes - phase)))
            return np.abs(offset) < 0.05

        assert near_preferred(kept).sum() == pytest.approx(
            near_preferred(dense_train).sum(), rel=0.01
        )

    def test_zero_modulation_depth_returns_all_spikes(self):
        """Test that zero modulation depth keeps all spikes."""
        spike_times = np.linspace(0, 5, 200)

        modulated = add_modulation(
            spike_times,
            modulation_freq=8.0,
            modulation_depth=0.0,
            seed=42,
        )

        np.testing.assert_array_equal(modulated, spike_times)

    def test_modulated_spikes_are_a_sorted_subset(self):
        spike_times = np.linspace(0, 5, 300)

        modulated = add_modulation(
            spike_times, modulation_freq=8.0, modulation_depth=0.5, seed=42
        )

        assert np.all(np.diff(modulated) >= 0)
        assert np.isin(modulated, spike_times).all()

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same modulated spikes."""
        spike_times = np.linspace(0, 5, 300)

        modulated1 = add_modulation(
            spike_times, modulation_freq=8.0, modulation_depth=0.5, seed=42
        )
        modulated2 = add_modulation(
            spike_times, modulation_freq=8.0, modulation_depth=0.5, seed=42
        )

        np.testing.assert_array_equal(modulated1, modulated2)

    def test_empty_spike_train(self):
        """Test with empty spike train."""
        modulated = add_modulation(
            np.array([]), modulation_freq=8.0, modulation_depth=0.5, seed=42
        )

        assert len(modulated) == 0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"modulation_depth": 1.5}, "modulation_depth must be in"),
            ({"modulation_depth": -0.1}, "modulation_depth must be in"),
            ({"modulation_depth": np.nan}, "modulation_depth must be in"),
            ({"modulation_freq": -8.0}, "modulation_freq must be a finite positive"),
            ({"modulation_freq": 0.0}, "modulation_freq must be a finite positive"),
            ({"modulation_freq": np.nan}, "modulation_freq must be a finite positive"),
            ({"modulation_phase": np.nan}, "modulation_phase must be a finite"),
        ],
    )
    @pytest.mark.parametrize("n_spikes", [0, 100])
    def test_invalid_parameters_raise(self, kwargs, match, n_spikes):
        """Refused even for an empty train; NaN would otherwise delete every spike."""
        params = {"modulation_freq": 8.0, **kwargs}
        with pytest.raises(ValueError, match=match):
            add_modulation(np.linspace(0, 5, n_spikes), **params)

    @pytest.mark.parametrize(
        ("spike_times", "match"),
        [
            (np.array([0.1, np.nan, 0.3]), "spike_times"),
            (np.array([0.3, 0.1, 0.2]), "sorted"),
            (np.zeros((2, 2)), "1-D"),
        ],
    )
    def test_invalid_spike_times_raise(self, spike_times, match):
        with pytest.raises(ValueError, match=match):
            add_modulation(spike_times, modulation_freq=8.0)
