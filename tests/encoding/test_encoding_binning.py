"""Tests for neurospatial.encoding._binning module.

This module tests the binning layer that converts spike trains to
spike counts and occupancy arrays.

TDD approach: Tests written first, implementation follows.

Task 2.7: Implement binning layer for spatial encoding
- Create helper to convert (env, spike_times, times, positions) -> (spike_counts, occupancy)
- Spike counts shape: (n_neurons, n_bins)
- Occupancy shape: (n_bins,)
- Parallelize over neurons with joblib
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment

# ==============================================================================
# Test fixtures
# ==============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    # Create a 10x10 grid covering 0-100 in each dimension
    x = np.linspace(0, 100, 11)
    y = np.linspace(0, 100, 11)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def trajectory_data(simple_env: Environment) -> dict:
    """Create trajectory data for testing.

    Returns dict with:
        - times: array of timestamps (1 second total, 100 samples at 100 Hz)
        - positions: array of positions (random walk within environment)
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    times = np.linspace(0, 1.0, n_samples)  # 1 second, 100 samples

    # Random walk within environment bounds
    positions = np.zeros((n_samples, 2))
    positions[0] = [50, 50]  # Start in center
    for i in range(1, n_samples):
        step = rng.normal(size=2) * 5  # Random step
        positions[i] = positions[i - 1] + step
        # Clip to environment bounds
        positions[i] = np.clip(positions[i], 0, 100)

    return {"times": times, "positions": positions}


@pytest.fixture
def single_neuron_spikes() -> NDArray[np.float64]:
    """Create spike times for a single neuron."""
    # Spikes at 0.1, 0.3, 0.5, 0.7, 0.9 seconds
    return np.array([0.1, 0.3, 0.5, 0.7, 0.9])


@pytest.fixture
def multiple_neuron_spikes() -> list[NDArray[np.float64]]:
    """Create spike times for multiple neurons."""
    return [
        np.array([0.1, 0.3, 0.5]),  # Neuron 0: 3 spikes
        np.array([0.2, 0.4, 0.6, 0.8]),  # Neuron 1: 4 spikes
        np.array([0.15]),  # Neuron 2: 1 spike
        np.array([]),  # Neuron 3: 0 spikes (silent)
    ]


# ==============================================================================
# Test bin_spike_train (single neuron)
# ==============================================================================


class TestBinSpikeTrain:
    """Tests for bin_spike_train function (single neuron binning)."""

    def test_spike_counts_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """bin_spike_train should return array with shape (n_bins,)."""
        from neurospatial.encoding._binning import bin_spike_train

        spike_counts = bin_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (simple_env.n_bins,)

    def test_spike_counts_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Spike counts should all be non-negative."""
        from neurospatial.encoding._binning import bin_spike_train

        spike_counts = bin_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert np.all(spike_counts >= 0)

    def test_total_spike_count(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Total spike count should equal number of spikes (within time range)."""
        from neurospatial.encoding._binning import bin_spike_train

        spike_counts = bin_spike_train(
            simple_env,
            single_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # Filter spikes to valid time range
        valid_spikes = single_neuron_spikes[
            (single_neuron_spikes >= trajectory_data["times"].min())
            & (single_neuron_spikes <= trajectory_data["times"].max())
        ]

        # Some spikes may fall outside active bins, so sum should be <= n_valid_spikes
        # but for a well-covered environment, should be close
        assert np.sum(spike_counts) <= len(valid_spikes)

    def test_empty_spike_train(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Empty spike train should produce all-zero spike counts."""
        from neurospatial.encoding._binning import bin_spike_train

        empty_spikes = np.array([])

        spike_counts = bin_spike_train(
            simple_env,
            empty_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (simple_env.n_bins,)
        assert np.all(spike_counts == 0)

    def test_spikes_outside_time_range(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Spikes outside time range should be excluded."""
        from neurospatial.encoding._binning import bin_spike_train

        # Spikes at -1.0 and 5.0, both outside the 0-1 second range.
        # warn_on_drop=False: this test only checks zero-count behavior,
        # not the warning itself (see TestWarnOnDrop for warning tests).
        spikes_outside = np.array([-1.0, 5.0])

        spike_counts = bin_spike_train(
            simple_env,
            spikes_outside,
            trajectory_data["times"],
            trajectory_data["positions"],
            warn_on_drop=False,
        )

        assert np.sum(spike_counts) == 0


# ==============================================================================
# Test compute_occupancy
# ==============================================================================


class TestComputeOccupancy:
    """Tests for compute_occupancy function."""

    def test_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """compute_occupancy should return array with shape (n_bins,)."""
        from neurospatial.encoding._binning import compute_occupancy

        occupancy = compute_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert occupancy.shape == (simple_env.n_bins,)

    def test_occupancy_non_negative(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Occupancy should all be non-negative."""
        from neurospatial.encoding._binning import compute_occupancy

        occupancy = compute_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert np.all(occupancy >= 0)

    def test_total_occupancy_approximately_equals_duration(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Total occupancy should approximately equal recording duration."""
        from neurospatial.encoding._binning import compute_occupancy

        occupancy = compute_occupancy(
            simple_env,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        total_duration = trajectory_data["times"][-1] - trajectory_data["times"][0]
        total_occupancy = np.sum(occupancy)

        # Should be close to total duration (may differ slightly due to
        # invalid bin positions or max_gap handling)
        assert total_occupancy <= total_duration + 0.01  # Allow small tolerance


# ==============================================================================
# Test bin_spike_trains (batch)
# ==============================================================================


class TestBinSpikeTrains:
    """Tests for bin_spike_trains function (multiple neurons)."""

    def test_spike_counts_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Spike counts should have shape (n_neurons, n_bins)."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_counts, _ = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        n_neurons = len(multiple_neuron_spikes)
        assert spike_counts.shape == (n_neurons, simple_env.n_bins)

    def test_occupancy_shape(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Occupancy should have shape (n_bins,)."""
        from neurospatial.encoding._binning import bin_spike_trains

        _, occupancy = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert occupancy.shape == (simple_env.n_bins,)

    def test_empty_neuron_has_zero_counts(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Neuron with no spikes should have all-zero counts."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_counts, _ = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # Neuron 3 has no spikes (empty array)
        assert np.all(spike_counts[3] == 0)

    def test_consistent_with_single_neuron(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """Batch result should be consistent with single-neuron results."""
        from neurospatial.encoding._binning import (
            bin_spike_train,
            bin_spike_trains,
        )

        spike_counts_batch, _occupancy_batch = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # Check each neuron matches single-neuron result
        for i, spikes in enumerate(multiple_neuron_spikes):
            spike_counts_single = bin_spike_train(
                simple_env,
                spikes,
                trajectory_data["times"],
                trajectory_data["positions"],
            )
            np.testing.assert_array_equal(spike_counts_batch[i], spike_counts_single)

    def test_n_jobs_parameter(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        multiple_neuron_spikes: list[NDArray[np.float64]],
    ) -> None:
        """n_jobs parameter should not change results."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_counts_serial, occupancy_serial = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            n_jobs=1,
        )

        spike_counts_parallel, occupancy_parallel = bin_spike_trains(
            simple_env,
            multiple_neuron_spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
            n_jobs=2,
        )

        np.testing.assert_array_equal(spike_counts_serial, spike_counts_parallel)
        np.testing.assert_array_equal(occupancy_serial, occupancy_parallel)

    def test_single_neuron_list(
        self,
        simple_env: Environment,
        trajectory_data: dict,
        single_neuron_spikes: NDArray[np.float64],
    ) -> None:
        """Should handle list with single neuron."""
        from neurospatial.encoding._binning import bin_spike_trains

        spike_times_list = [single_neuron_spikes]

        spike_counts, occupancy = bin_spike_trains(
            simple_env,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (1, simple_env.n_bins)
        assert occupancy.shape == (simple_env.n_bins,)

    def test_normalizes_input_via_normalize_spike_times(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Should accept normalized input from normalize_spike_times formats."""
        from neurospatial.encoding._binning import bin_spike_trains

        # Test with 2D NaN-padded array (common format)
        spikes_2d = np.array(
            [
                [0.1, 0.3, 0.5, np.nan],
                [0.2, 0.4, np.nan, np.nan],
                [0.15, np.nan, np.nan, np.nan],
            ]
        )

        spike_counts, _occupancy = bin_spike_trains(
            simple_env,
            spikes_2d,  # type: ignore[arg-type]
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        assert spike_counts.shape == (3, simple_env.n_bins)


# ==============================================================================
# Test edge cases and error handling
# ==============================================================================


class TestBinningEdgeCases:
    """Tests for edge cases in binning layer."""

    def test_all_spikes_in_one_bin(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """All spikes in one location should be binned together."""
        from neurospatial.encoding._binning import bin_spike_train

        # Create spikes at times when we know the position
        # Position at t=0.5 (midpoint)
        t_mid = 0.5
        mid_idx = np.argmin(np.abs(trajectory_data["times"] - t_mid))
        pos_at_mid = trajectory_data["positions"][mid_idx]

        # Multiple spikes all at t=0.5
        spikes = np.array([0.5, 0.5001, 0.5002])

        spike_counts = bin_spike_train(
            simple_env,
            spikes,
            trajectory_data["times"],
            trajectory_data["positions"],
        )

        # Should have counts in just one bin
        target_bin = simple_env.bin_at(pos_at_mid.reshape(1, -1))[0]
        if target_bin >= 0:  # Valid bin
            assert spike_counts[target_bin] == 3
            # All other bins should be 0
            assert np.sum(spike_counts) == 3

    def test_handles_narrow_2d_environment(self) -> None:
        """Should handle narrow 2D environments correctly."""
        from neurospatial import Environment
        from neurospatial.encoding._binning import bin_spike_trains

        # Create a narrow 2D environment (effectively 1D-like behavior)
        # This is a linear track approximation
        x = np.linspace(0, 100, 11)
        y = np.ones(11) * 5  # Narrow in y-direction
        positions_env = np.column_stack([x, y])
        env = Environment.from_samples(positions_env, bin_size=10.0)

        # Create trajectory along the track
        times = np.linspace(0, 1.0, 100)
        positions = np.column_stack(
            [np.linspace(5, 95, 100), np.ones(100) * 5]  # Move along x
        )

        spikes = [np.array([0.1, 0.5]), np.array([0.3])]

        spike_counts, occupancy = bin_spike_trains(
            env,
            spikes,
            times,
            positions,
        )

        assert spike_counts.shape == (2, env.n_bins)
        assert occupancy.shape == (env.n_bins,)


class TestBinningInputValidation:
    """Tests for input validation in binning functions."""

    def test_mismatched_times_positions_length(
        self,
        simple_env: Environment,
    ) -> None:
        """Should raise error if times and positions have different lengths."""
        from neurospatial.encoding._binning import compute_occupancy

        times = np.linspace(0, 1, 100)
        rng = np.random.default_rng(42)
        positions = rng.random((50, 2)) * 100  # Different length

        with pytest.raises(ValueError, match=r"length|shape|mismatch"):
            compute_occupancy(simple_env, times, positions)

    def test_position_dims_mismatch(
        self,
        simple_env: Environment,
        trajectory_data: dict,
    ) -> None:
        """Should raise error if position dimensions don't match env."""
        from neurospatial.encoding._binning import compute_occupancy

        # simple_env is 2D, but provide 3D positions
        positions_3d = np.column_stack(
            [trajectory_data["positions"], np.zeros(len(trajectory_data["times"]))]
        )

        with pytest.raises(ValueError, match=r"dimension|shape"):
            compute_occupancy(simple_env, trajectory_data["times"], positions_3d)


class TestSpikeInterpolationRegression:
    """Regression tests guarding against snapshot-instead-of-interpolated binning.

    A previous refactor mapped each spike to the bin of its most recent
    trajectory frame instead of its interpolated position. With sparse
    sampling or fast movement this shifts spikes to the previous bin —
    a silent rate-map error.
    """

    def test_spike_between_samples_uses_interpolated_position(self) -> None:
        """A spike at t=0.75 with samples at 0/10/20 maps to the 7.5 bin, not 0."""
        from neurospatial import Environment
        from neurospatial.encoding._binning import bin_spike_train

        # 1D environment with bins of width 5 spanning [0, 20].
        sample_positions = np.linspace(0.0, 20.0, 100).reshape(-1, 1)
        env = Environment.from_samples(sample_positions, bin_size=5.0)

        # Sparse trajectory: three samples at t=0, 1, 2 with positions 0, 10, 20.
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0], [10.0], [20.0]])

        # Single spike at t=0.75; linear interpolation gives position 7.5
        # which falls in the same bin as 7.5 (a different bin from position 0).
        spike_times = np.array([0.75])

        counts = bin_spike_train(env, spike_times, times, positions)

        # Identify the bin that interpolated position 7.5 lives in, vs the
        # bin that the snapshot position 0.0 lives in. The two must differ
        # for this regression test to be meaningful, and the spike count
        # must land in the interpolated bin.
        interp_bin = env.bin_at(np.array([[7.5]]))[0]
        snapshot_bin = env.bin_at(np.array([[0.0]]))[0]
        assert interp_bin != snapshot_bin, "test setup not exercising the bug"
        assert counts[interp_bin] == 1.0
        assert counts[snapshot_bin] == 0.0

    def test_batch_matches_single_under_interpolation(self) -> None:
        """bin_spike_trains must match bin_spike_train per neuron exactly."""
        from neurospatial import Environment
        from neurospatial.encoding._binning import bin_spike_train, bin_spike_trains

        sample_positions = np.linspace(0.0, 20.0, 100).reshape(-1, 1)
        env = Environment.from_samples(sample_positions, bin_size=5.0)
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0], [10.0], [20.0]])
        spike_lists = [
            np.array([0.75]),
            np.array([0.25, 1.5]),
            np.array([]),
        ]

        batch_counts, _ = bin_spike_trains(env, spike_lists, times, positions)
        for i, spikes in enumerate(spike_lists):
            single = bin_spike_train(env, spikes, times, positions)
            np.testing.assert_array_equal(batch_counts[i], single)


# ==============================================================================
# Test warn_on_drop: spike-drop warnings (Task 0.1)
# ==============================================================================


class TestWarnOnDrop:
    """Tests for the warn_on_drop warning mechanism in bin_spike_train/bin_spike_trains.

    These tests verify that silent spike dropping is replaced with UserWarning
    messages when the dropped fraction exceeds the threshold or all spikes are
    dropped.  They also verify that warn_on_drop=False truly silences all
    warnings, and that in-window spikes produce no spurious warnings.
    """

    # ------------------------------------------------------------------
    # Shared fixtures (inline, not module-level, to keep them scoped)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_env_and_traj() -> tuple:
        """Return (env, times, positions) for a simple 1D environment."""
        sample_pos = np.linspace(0.0, 100.0, 200).reshape(-1, 1)
        env = Environment.from_samples(sample_pos, bin_size=5.0)
        times = np.linspace(0.0, 10.0, 1000)  # 0–10 s
        positions = np.linspace(0.0, 100.0, 1000).reshape(-1, 1)
        return env, times, positions

    # ------------------------------------------------------------------
    # 1. ms-vs-s mismatch → all spikes outside window → warn, near-zero
    # ------------------------------------------------------------------

    def test_out_of_window_single_neuron_warns(self) -> None:
        """Spikes in milliseconds while times in seconds → UserWarning."""

        from neurospatial.encoding._binning import bin_spike_train

        env, times, positions = self._make_env_and_traj()
        # spike_times in ms (1000–9000 ms) while times in s (0–10)
        # These map to 1–9 s inside the position window, but expressed in ms
        # they are WAY outside the 0–10 s window, so all spikes are dropped.
        spike_times_ms = np.array([1000.0, 2000.0, 5000.0, 8000.0, 9000.0])

        with pytest.warns(UserWarning, match=r"spike_times"):
            counts = bin_spike_train(env, spike_times_ms, times, positions)

        # Field should be near-zero (all spikes dropped)
        assert np.sum(counts) == 0

    def test_out_of_window_batch_warns_once(self) -> None:
        """All 3 neurons have out-of-window spikes → exactly one UserWarning."""
        import warnings

        from neurospatial.encoding._binning import bin_spike_trains

        env, times, positions = self._make_env_and_traj()
        spike_times_ms = [
            np.array([1000.0, 5000.0]),  # all out-of-window
            np.array([2000.0, 8000.0]),  # all out-of-window
            np.array([3000.0, 9000.0]),  # all out-of-window
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_spike_trains(env, spike_times_ms, times, positions)

        time_window_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "spike_times" in str(x.message)
        ]
        # Must warn exactly ONCE (not 3× for 3 neurons)
        assert len(time_window_warnings) == 1, (
            f"Expected exactly 1 time-window warning, got {len(time_window_warnings)}"
        )

    def test_out_of_window_compute_spatial_rate_warns(self) -> None:
        """compute_spatial_rate: out-of-window spikes → UserWarning."""
        from neurospatial.encoding.spatial import compute_spatial_rate

        env, times, positions = self._make_env_and_traj()
        spike_times_ms = np.array([1000.0, 5000.0, 9000.0])

        with pytest.warns(UserWarning, match=r"spike_times"):
            compute_spatial_rate(env, spike_times_ms, times, positions.squeeze())

    def test_out_of_window_compute_spatial_rates_warns_once(self) -> None:
        """compute_spatial_rates: batch with all out-of-window → exactly one warning."""
        import warnings

        from neurospatial.encoding.spatial import compute_spatial_rates

        env, times, positions = self._make_env_and_traj()
        spike_times_ms = [
            np.array([1000.0, 5000.0]),
            np.array([2000.0, 8000.0]),
            np.array([3000.0, 9000.0]),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_spatial_rates(env, spike_times_ms, times, positions.squeeze())

        time_window_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "spike_times" in str(x.message)
        ]
        assert len(time_window_warnings) == 1, (
            f"Expected exactly 1 warning, got {len(time_window_warnings)}"
        )

    # ------------------------------------------------------------------
    # 2. In-window spikes → no spurious warning
    # ------------------------------------------------------------------

    def test_in_window_no_warning(self) -> None:
        """Fully in-window spikes should produce no UserWarning."""
        import warnings

        from neurospatial.encoding._binning import bin_spike_train

        env, times, positions = self._make_env_and_traj()
        # All spikes well within [0, 10] s
        spike_times = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_spike_train(env, spike_times, times, positions)

        drop_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and (
                "spike_times" in str(x.message)
                or "inactive" in str(x.message).lower()
                or "interpolated to positions" in str(x.message).lower()
            )
        ]
        assert len(drop_warnings) == 0, (
            f"Unexpected drop warning(s): {[str(x.message) for x in drop_warnings]}"
        )

    # ------------------------------------------------------------------
    # 3. warn_on_drop=False silences all warnings
    # ------------------------------------------------------------------

    def test_warn_on_drop_false_single_neuron(self) -> None:
        """warn_on_drop=False: no warning even when all spikes are out-of-window."""
        import warnings

        from neurospatial.encoding._binning import bin_spike_train

        env, times, positions = self._make_env_and_traj()
        spike_times_ms = np.array([1000.0, 5000.0, 9000.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_spike_train(env, spike_times_ms, times, positions, warn_on_drop=False)

        drop_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and (
                "spike_times" in str(x.message)
                or "inactive" in str(x.message).lower()
                or "interpolated to positions" in str(x.message).lower()
            )
        ]
        assert len(drop_warnings) == 0

    def test_warn_on_drop_false_batch(self) -> None:
        """warn_on_drop=False: no warning from bin_spike_trains even with drops."""
        import warnings

        from neurospatial.encoding._binning import bin_spike_trains

        env, times, positions = self._make_env_and_traj()
        spike_times_ms = [
            np.array([1000.0, 5000.0]),
            np.array([2000.0, 8000.0]),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_spike_trains(env, spike_times_ms, times, positions, warn_on_drop=False)

        drop_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and (
                "spike_times" in str(x.message)
                or "inactive" in str(x.message).lower()
                or "interpolated to positions" in str(x.message).lower()
            )
        ]
        assert len(drop_warnings) == 0

    def test_warn_on_drop_false_compute_spatial_rate(self) -> None:
        """compute_spatial_rate warn_on_drop=False: no warning even with drops."""
        import warnings

        from neurospatial.encoding.spatial import compute_spatial_rate

        env, times, positions = self._make_env_and_traj()
        spike_times_ms = np.array([1000.0, 5000.0, 9000.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_spatial_rate(
                env,
                spike_times_ms,
                times,
                positions.squeeze(),
                warn_on_drop=False,
            )

        drop_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and (
                "spike_times" in str(x.message)
                or "inactive" in str(x.message).lower()
                or "interpolated to positions" in str(x.message).lower()
            )
        ]
        assert len(drop_warnings) == 0

    def test_warn_on_drop_false_compute_spatial_rates(self) -> None:
        """compute_spatial_rates warn_on_drop=False: no warning even with drops."""
        import warnings

        from neurospatial.encoding.spatial import compute_spatial_rates

        env, times, positions = self._make_env_and_traj()
        spike_times_ms = [
            np.array([1000.0, 5000.0]),
            np.array([2000.0, 8000.0]),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_spatial_rates(
                env,
                spike_times_ms,
                times,
                positions.squeeze(),
                warn_on_drop=False,
            )

        drop_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and (
                "spike_times" in str(x.message)
                or "inactive" in str(x.message).lower()
                or "interpolated to positions" in str(x.message).lower()
            )
        ]
        assert len(drop_warnings) == 0

    # ------------------------------------------------------------------
    # 4. Warning message content: counts, ranges, units hint
    # ------------------------------------------------------------------

    def test_out_of_window_message_content(self) -> None:
        """Warning message must name dropped count, total, and both time ranges."""
        import warnings

        from neurospatial.encoding._binning import bin_spike_train

        env, times, positions = self._make_env_and_traj()
        # 5 spikes, all outside the 0–10 s window (times in ms)
        spike_times_ms = np.array([1000.0, 2000.0, 5000.0, 8000.0, 9000.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_spike_train(env, spike_times_ms, times, positions)

        time_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "spike_times" in str(x.message)
        ]
        assert len(time_warnings) >= 1
        msg = str(time_warnings[0].message)

        # Must contain n/N (dropped/total)
        assert "5/5" in msg
        # Must contain the spike_times range fields
        assert "spike_times.min()=" in msg
        assert "spike_times.max()=" in msg
        # Must contain the position time window (format: "[t_min, t_max]")
        assert "[" in msg and "]" in msg
        # Must mention units (seconds)
        assert "second" in msg.lower() or "units" in msg.lower()
        # Must include the escape-hatch hint
        assert "warn_on_drop=False" in msg

    # ------------------------------------------------------------------
    # 5. Inactive-bin drop warning
    # ------------------------------------------------------------------

    def test_inactive_bin_warns(self) -> None:
        """Spikes mapping to bins outside the environment → UserWarning."""
        import warnings

        from neurospatial.encoding._binning import bin_spike_train

        # Create a very small environment (only covers [0, 10] x [0, 10])
        sample_pos = np.column_stack(
            [
                np.linspace(0, 10, 50),
                np.linspace(0, 10, 50),
            ]
        )
        env = Environment.from_samples(sample_pos, bin_size=2.0)

        # All spikes are at times within the window, but at positions OUTSIDE
        # the environment bounds (far from sample_pos).
        # Use sparse times/positions so interpolated spike positions land outside.
        times_narrow = np.array([0.0, 5.0, 10.0])
        positions_outside = np.array([[500.0, 500.0], [500.0, 500.0], [500.0, 500.0]])
        spike_times = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # 5 spikes, all in-window

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_spike_train(env, spike_times, times_narrow, positions_outside)

        inactive_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning)
            and (
                "inactive" in str(x.message).lower()
                or "outside" in str(x.message).lower()
                or "environment" in str(x.message).lower()
            )
        ]
        assert len(inactive_warnings) >= 1

    # ------------------------------------------------------------------
    # 6. Below-threshold drop: no warning when fraction is small
    # ------------------------------------------------------------------

    def test_small_drop_fraction_no_warning(self) -> None:
        """A small fraction of dropped spikes (<= threshold) should not warn."""
        import warnings

        from neurospatial.encoding._binning import bin_spike_train

        env, times, positions = self._make_env_and_traj()
        # 100 spikes in-window + 1 out-of-window → 1% dropped → below default 50% threshold
        rng = np.random.default_rng(0)
        in_window = rng.uniform(0.5, 9.5, 99)  # 99 spikes well inside [0, 10]
        out_of_window = np.array([1000.0])  # 1 spike outside
        spike_times = np.concatenate([in_window, out_of_window])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bin_spike_train(env, spike_times, times, positions)

        time_warnings = [
            x
            for x in w
            if issubclass(x.category, UserWarning) and "spike_times" in str(x.message)
        ]
        assert len(time_warnings) == 0, (
            "Should not warn when only 1% of spikes are out-of-window"
        )
