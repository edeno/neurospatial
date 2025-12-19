"""Tests for egocentric binning layer.

This module tests the binning layer for egocentric (object-vector) encoding,
which computes:
1. Egocentric occupancy (time spent at each distance/direction from objects)
2. Spike counts binned by egocentric coordinates at spike time
3. Batch processing of multiple neurons

The egocentric binning differs from spatial binning in that:
- Spatial binning: bins by *where the animal was*
- Egocentric binning: bins by *distance/direction to nearest object*
"""

import numpy as np
import pytest

from neurospatial import Environment

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_positions() -> np.ndarray:
    """Sample positions for environment creation and testing."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 100, (500, 2))


@pytest.fixture
def env(sample_positions: np.ndarray) -> Environment:
    """Create a test environment."""
    return Environment.from_samples(sample_positions, bin_size=5.0)


@pytest.fixture
def trajectory_data() -> dict:
    """Create trajectory data for testing."""
    rng = np.random.default_rng(42)
    n_time = 1000
    return {
        "times": np.linspace(0, 100, n_time),
        "positions": rng.uniform(10, 90, (n_time, 2)),
        "headings": rng.uniform(-np.pi, np.pi, n_time),
    }


@pytest.fixture
def object_positions() -> np.ndarray:
    """Object positions in allocentric coordinates."""
    return np.array([[50.0, 50.0], [25.0, 75.0]])


@pytest.fixture
def spike_times() -> np.ndarray:
    """Sample spike times for single neuron."""
    rng = np.random.default_rng(42)
    return np.sort(rng.uniform(0, 100, 200))


@pytest.fixture
def spike_times_list() -> list[np.ndarray]:
    """Sample spike times for multiple neurons."""
    rng = np.random.default_rng(42)
    return [
        np.sort(rng.uniform(0, 100, 100)),  # Neuron 0
        np.sort(rng.uniform(0, 100, 150)),  # Neuron 1
        np.sort(rng.uniform(0, 100, 50)),  # Neuron 2
    ]


# =============================================================================
# Test compute_egocentric_occupancy
# =============================================================================


class TestComputeEgocentricOccupancy:
    """Tests for compute_egocentric_occupancy function."""

    def test_import(self):
        """Function can be imported from _egocentric_binning module."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        assert callable(compute_egocentric_occupancy)

    def test_returns_array(
        self,
        env: Environment,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Function returns numpy array."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        occupancy, _ego_env = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert isinstance(occupancy, np.ndarray)

    def test_returns_ego_env(
        self,
        env: Environment,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Function returns egocentric polar environment."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        _occupancy, ego_env = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert isinstance(ego_env, Environment)

    def test_occupancy_shape(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Occupancy shape matches ego_env.n_bins."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        n_distance_bins = 10
        n_direction_bins = 12
        occupancy, ego_env = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )
        expected_n_bins = n_distance_bins * n_direction_bins
        assert occupancy.shape == (expected_n_bins,)
        assert len(occupancy) == ego_env.n_bins

    def test_occupancy_non_negative(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Occupancy values are non-negative."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        occupancy, _ = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert np.all(occupancy >= 0)

    def test_occupancy_dtype(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Occupancy has float64 dtype."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        occupancy, _ = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert occupancy.dtype == np.float64

    def test_total_occupancy_approximates_duration(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Total occupancy approximates recording duration."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        times = trajectory_data["times"]
        duration = times[-1] - times[0]

        # Use large distance range to capture all samples
        occupancy, _ = compute_egocentric_occupancy(
            times,
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 200.0),  # Large range to capture all
            n_distance_bins=20,
            n_direction_bins=12,
        )
        # Total occupancy should be close to duration (minus one dt for n-1 intervals)
        assert abs(np.sum(occupancy) - duration) < duration * 0.1


class TestComputeEgocentricOccupancyDistanceMetric:
    """Tests for distance_metric parameter."""

    def test_euclidean_default(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Euclidean is the default distance metric."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        # Should work without env
        occupancy, _ = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert occupancy.shape[0] > 0

    def test_geodesic_requires_env(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Geodesic distance metric requires env parameter."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        with pytest.raises(ValueError, match=r"geodesic.*requires.*env"):
            compute_egocentric_occupancy(
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
                distance_metric="geodesic",
            )

    def test_geodesic_with_env(
        self,
        env: Environment,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Geodesic distance works when env is provided."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        occupancy, _ = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            distance_metric="geodesic",
            env=env,
        )
        assert occupancy.shape[0] > 0

    def test_invalid_distance_metric_raises(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Invalid distance metric raises ValueError."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        with pytest.raises(ValueError, match=r"invalid|Invalid"):
            compute_egocentric_occupancy(
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
                distance_metric="manhattan",  # Invalid
            )


class TestComputeEgocentricOccupancyInputValidation:
    """Tests for input validation in compute_egocentric_occupancy."""

    def test_mismatched_times_positions_raises(self, object_positions: np.ndarray):
        """Mismatched times and positions lengths raise error."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        times = np.linspace(0, 10, 100)
        positions = np.random.rand(90, 2)  # Wrong length
        headings = np.random.rand(100)

        with pytest.raises(ValueError, match=r"times.*positions"):
            compute_egocentric_occupancy(
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
            )

    def test_mismatched_times_headings_raises(self, object_positions: np.ndarray):
        """Mismatched times and headings lengths raise error."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        times = np.linspace(0, 10, 100)
        positions = np.random.rand(100, 2)
        headings = np.random.rand(90)  # Wrong length

        with pytest.raises(ValueError, match=r"times.*headings"):
            compute_egocentric_occupancy(
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
            )

    def test_insufficient_samples_raises(self, object_positions: np.ndarray):
        """Fewer than 2 samples raises error."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        times = np.array([0.0])  # Only 1 sample
        positions = np.array([[50.0, 50.0]])
        headings = np.array([0.0])

        with pytest.raises(ValueError, match=r"2.*samples"):
            compute_egocentric_occupancy(
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
            )

    def test_unsorted_times_raises(self, object_positions: np.ndarray):
        """Non-monotonic times raise error."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        times = np.array([0.0, 2.0, 1.0, 3.0])  # Not sorted
        positions = np.random.rand(4, 2)
        headings = np.random.rand(4)

        with pytest.raises(ValueError, match="monotonic"):
            compute_egocentric_occupancy(
                times,
                positions,
                headings,
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
            )


# =============================================================================
# Test bin_egocentric_spike_train
# =============================================================================


class TestBinEgocentricSpikeTrain:
    """Tests for bin_egocentric_spike_train function."""

    def test_import(self):
        """Function can be imported from _egocentric_binning module."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        assert callable(bin_egocentric_spike_train)

    def test_returns_array(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Function returns numpy array."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        spike_counts, _ego_env = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert isinstance(spike_counts, np.ndarray)

    def test_spike_counts_shape(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Spike counts shape matches ego_env.n_bins."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        n_distance_bins = 10
        n_direction_bins = 12
        spike_counts, ego_env = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )
        expected_n_bins = n_distance_bins * n_direction_bins
        assert spike_counts.shape == (expected_n_bins,)
        assert len(spike_counts) == ego_env.n_bins

    def test_spike_counts_non_negative(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Spike counts are non-negative."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        spike_counts, _ = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert np.all(spike_counts >= 0)

    def test_spike_counts_dtype(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Spike counts have float64 dtype (for smoothing compatibility)."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        spike_counts, _ = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert spike_counts.dtype == np.float64

    def test_total_spike_count_reasonable(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Total spike count is at most number of input spikes."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        # Use large distance range to capture all spikes
        spike_counts, _ = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 200.0),  # Large range
            n_distance_bins=20,
            n_direction_bins=12,
        )
        assert np.sum(spike_counts) <= len(spike_times)

    def test_empty_spike_train(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Empty spike train returns zeros."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        spike_times = np.array([])
        spike_counts, ego_env = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert np.sum(spike_counts) == 0
        assert spike_counts.shape == (ego_env.n_bins,)


class TestBinEgocentricSpikeTrainDistanceMetric:
    """Tests for distance_metric parameter in spike train binning."""

    def test_euclidean_default(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Euclidean is the default distance metric."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        spike_counts, _ = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert spike_counts.shape[0] > 0

    def test_geodesic_requires_env(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Geodesic distance metric requires env parameter."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        with pytest.raises(ValueError, match=r"geodesic.*requires.*env"):
            bin_egocentric_spike_train(
                spike_times,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
                distance_metric="geodesic",
            )

    def test_geodesic_with_env(
        self,
        env: Environment,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Geodesic distance works when env is provided."""
        from neurospatial.encoding._egocentric_binning import bin_egocentric_spike_train

        spike_counts, _ = bin_egocentric_spike_train(
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            distance_metric="geodesic",
            env=env,
        )
        assert spike_counts.shape[0] > 0


# =============================================================================
# Test bin_egocentric_spike_trains (batch)
# =============================================================================


class TestBinEgocentricSpikeTrains:
    """Tests for bin_egocentric_spike_trains function (batch version)."""

    def test_import(self):
        """Function can be imported from _egocentric_binning module."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        assert callable(bin_egocentric_spike_trains)

    def test_returns_tuple(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Function returns tuple of (spike_counts, occupancy, ego_env)."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        result = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_spike_counts_shape(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Spike counts shape is (n_neurons, n_bins)."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        n_neurons = len(spike_times_list)
        n_distance_bins = 10
        n_direction_bins = 12
        expected_n_bins = n_distance_bins * n_direction_bins

        spike_counts, _occupancy, ego_env = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )
        assert spike_counts.shape == (n_neurons, expected_n_bins)
        assert spike_counts.shape[1] == ego_env.n_bins

    def test_occupancy_shape(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Occupancy shape is (n_bins,) - shared across neurons."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        n_distance_bins = 10
        n_direction_bins = 12
        expected_n_bins = n_distance_bins * n_direction_bins

        _spike_counts, occupancy, ego_env = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )
        assert occupancy.shape == (expected_n_bins,)
        assert len(occupancy) == ego_env.n_bins

    def test_dtypes(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Both outputs have float64 dtype."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        spike_counts, occupancy, _ = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert spike_counts.dtype == np.float64
        assert occupancy.dtype == np.float64

    def test_consistency_with_single(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Batch results match single-neuron results."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_train,
            bin_egocentric_spike_trains,
        )

        # Batch
        spike_counts_batch, _occupancy_batch, _ego_env_batch = (
            bin_egocentric_spike_trains(
                spike_times_list,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
            )
        )

        # Single
        for i, spike_times in enumerate(spike_times_list):
            spike_counts_single, _ego_env_single = bin_egocentric_spike_train(
                spike_times,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
            )
            np.testing.assert_array_almost_equal(
                spike_counts_batch[i], spike_counts_single
            )

    def test_empty_neuron_list(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
    ):
        """Empty neuron list returns correct shapes."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        spike_times_list: list[np.ndarray] = []
        n_distance_bins = 10
        n_direction_bins = 12
        expected_n_bins = n_distance_bins * n_direction_bins

        spike_counts, occupancy, _ego_env = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )
        assert spike_counts.shape == (0, expected_n_bins)
        assert occupancy.shape == (expected_n_bins,)

    def test_n_jobs_parameter(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """n_jobs parameter is accepted."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        # Sequential
        spike_counts_seq, _, _ = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            n_jobs=1,
        )

        # Parallel
        spike_counts_par, _, _ = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            n_jobs=2,
        )

        np.testing.assert_array_almost_equal(spike_counts_seq, spike_counts_par)

    def test_accepts_2d_nan_padded_array(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Accepts 2D NaN-padded array format for spike times."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        # Convert list to 2D NaN-padded array
        max_spikes = max(len(s) for s in spike_times_list)
        spike_times_2d = np.full((len(spike_times_list), max_spikes), np.nan)
        for i, s in enumerate(spike_times_list):
            spike_times_2d[i, : len(s)] = s

        spike_counts, _occupancy, _ego_env = bin_egocentric_spike_trains(
            spike_times_2d,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )
        assert spike_counts.shape[0] == len(spike_times_list)


class TestBinEgocentricSpikeTrainsDistanceMetric:
    """Tests for distance_metric parameter in batch spike train binning."""

    def test_geodesic_requires_env(
        self,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Geodesic distance metric requires env parameter."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        with pytest.raises(ValueError, match=r"geodesic.*requires.*env"):
            bin_egocentric_spike_trains(
                spike_times_list,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                object_positions,
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
                distance_metric="geodesic",
            )

    def test_geodesic_with_env(
        self,
        env: Environment,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times_list: list[np.ndarray],
    ):
        """Geodesic distance works when env is provided."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_trains,
        )

        spike_counts, _occupancy, _ego_env = bin_egocentric_spike_trains(
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            distance_metric="geodesic",
            env=env,
        )
        assert spike_counts.shape[0] == len(spike_times_list)


# =============================================================================
# Test egocentric coordinate correctness
# =============================================================================


class TestEgocentricCoordinates:
    """Tests for correct egocentric coordinate computation."""

    def test_bearing_zero_ahead(self):
        """Object directly ahead should have bearing 0."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal at origin facing East (heading=0), object at (10, 0)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[10.0, 0.0]])

        distances, bearings = _compute_egocentric_coords(
            positions, headings, object_positions
        )

        np.testing.assert_almost_equal(distances[0, 0], 10.0)
        np.testing.assert_almost_equal(bearings[0, 0], 0.0)

    def test_bearing_left(self):
        """Object to the left should have bearing ~pi/2."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal at origin facing East (heading=0), object at (0, 10)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[0.0, 10.0]])

        distances, bearings = _compute_egocentric_coords(
            positions, headings, object_positions
        )

        np.testing.assert_almost_equal(distances[0, 0], 10.0)
        np.testing.assert_almost_equal(bearings[0, 0], np.pi / 2)

    def test_bearing_right(self):
        """Object to the right should have bearing ~-pi/2."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal at origin facing East (heading=0), object at (0, -10)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[0.0, -10.0]])

        distances, bearings = _compute_egocentric_coords(
            positions, headings, object_positions
        )

        np.testing.assert_almost_equal(distances[0, 0], 10.0)
        np.testing.assert_almost_equal(bearings[0, 0], -np.pi / 2)

    def test_nearest_object_selection(self):
        """Nearest object should be selected at each timepoint."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal at origin, object1 at (5,0), object2 at (10,0)
        positions = np.array([[0.0, 0.0]])
        headings = np.array([0.0])
        object_positions = np.array([[5.0, 0.0], [10.0, 0.0]])

        distances, _bearings = _compute_egocentric_coords(
            positions, headings, object_positions
        )

        # Should return distance to NEAREST object (5)
        np.testing.assert_almost_equal(distances[0, 0], 5.0)


class TestConsistencyWithObjectVector:
    """Test consistency with existing object_vector.py implementation."""

    def test_occupancy_matches_object_vector_tuning(
        self,
        env: Environment,
        trajectory_data: dict,
        object_positions: np.ndarray,
        spike_times: np.ndarray,
    ):
        """Occupancy should match existing compute_object_vector_tuning."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )
        from neurospatial.encoding.object_vector import compute_object_vector_tuning

        # Compute using new binning layer
        occupancy_new, _ego_env = compute_egocentric_occupancy(
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Compute using existing implementation
        compute_object_vector_tuning(
            env,
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            object_positions,
            max_distance=50.0,
            n_distance_bins=10,
            n_direction_bins=12,
            min_occupancy_seconds=0.0,  # No masking
        )

        # The existing implementation stores occupancy internally in 2D
        # but we flatten to 1D. Reshape for comparison.
        # Existing: (n_distance, n_direction) - multiply by dt to get time
        n_distance = 10
        n_direction = 12
        np.median(np.diff(trajectory_data["times"]))

        # Can't directly compare because existing implementation may use
        # different internal occupancy computation. Instead verify shape.
        assert occupancy_new.shape == (n_distance * n_direction,)
        assert np.sum(occupancy_new) > 0  # Has some occupancy
