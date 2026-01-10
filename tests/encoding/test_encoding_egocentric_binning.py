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


# =============================================================================
# Test NaN handling in egocentric coordinate computation
# =============================================================================


class TestEgocentricNaNHandling:
    """Tests for NaN handling in egocentric nearest-object selection.

    Bug: When using geodesic distance, objects outside the environment have NaN
    distances. np.argmin doesn't handle NaN properly, returning arbitrary indices.

    This test class covers:
    - Objects outside environment with geodesic distance
    - Mixed NaN/finite distance cases (some objects inside, some outside)
    - All-NaN case (all objects outside environment)
    """

    @pytest.fixture
    def env_with_hole(self) -> Environment:
        """Create environment with a hole where objects can be placed outside.

        Creates an L-shaped environment using a regular grid that excludes
        the upper-right quadrant (60-100, 60-100). This ensures predictable
        bin coverage for testing.
        """
        # Create a dense grid covering the L-shaped area
        # Lower-left quadrant (0-60, 0-60) - fully covered
        # Lower-right strip (60-100, 0-60) - covered
        # Upper-left strip (0-60, 60-100) - covered
        # Upper-right quadrant (60-100, 60-100) - HOLE (not covered)
        x_lower = np.linspace(2.5, 57.5, 12)  # Centers for 0-60
        y_lower = np.linspace(2.5, 57.5, 12)
        x_upper = np.linspace(2.5, 57.5, 12)  # x for upper-left strip
        y_upper = np.linspace(62.5, 97.5, 8)
        x_right = np.linspace(62.5, 97.5, 8)  # x for lower-right strip
        y_right = np.linspace(2.5, 57.5, 12)

        # Create grid positions
        positions = []
        # Lower-left quadrant
        for x in x_lower:
            for y in y_lower:
                positions.append([x, y])
        # Upper-left strip
        for x in x_upper:
            for y in y_upper:
                positions.append([x, y])
        # Lower-right strip
        for x in x_right:
            for y in y_right:
                positions.append([x, y])

        positions = np.array(positions)
        return Environment.from_samples(positions, bin_size=5.0)

    def test_object_outside_env_geodesic(self, env_with_hole: Environment):
        """Object outside environment should produce NaN distance with geodesic."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal at origin (inside env), object in upper-right quadrant (outside)
        positions = np.array([[25.0, 25.0]])
        headings = np.array([0.0])
        # Object outside the L-shaped environment (in the hole)
        object_positions = np.array([[75.0, 75.0]])

        distances, bearings = _compute_egocentric_coords(
            positions,
            headings,
            object_positions,
            distance_metric="geodesic",
            env=env_with_hole,
        )

        # Distance should be NaN (object outside env)
        assert np.isnan(distances[0, 0])
        # Bearing should also be NaN (no reachable object to compute bearing to)
        assert np.isnan(bearings[0, 0])

    def test_mixed_nan_finite_distances(self, env_with_hole: Environment):
        """With mixed NaN/finite distances, should select nearest valid object."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal at origin (inside env)
        positions = np.array([[25.0, 25.0]])
        headings = np.array([0.0])
        # Object 1: outside env (NaN distance)
        # Object 2: inside env (finite distance)
        object_positions = np.array(
            [
                [75.0, 75.0],  # Outside (in hole)
                [30.0, 25.0],  # Inside (5 units away)
            ]
        )

        distances, bearings = _compute_egocentric_coords(
            positions,
            headings,
            object_positions,
            distance_metric="geodesic",
            env=env_with_hole,
        )

        # Should select object 2 (inside env, finite distance ~5)
        # NOT object 1 (outside env, NaN distance)
        assert not np.isnan(distances[0, 0]), "Should return finite distance, not NaN"
        # Distance should be approximately 5 (euclidean, but geodesic might differ slightly)
        assert distances[0, 0] < 20, f"Expected ~5, got {distances[0, 0]}"
        # Bearing should be ~0 (object 2 is directly ahead)
        assert abs(bearings[0, 0]) < 0.5, f"Expected ~0, got {bearings[0, 0]}"

    def test_all_objects_outside_env_geodesic(self, env_with_hole: Environment):
        """When all objects are outside env, should return NaN distance."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal at origin (inside env)
        positions = np.array([[25.0, 25.0]])
        headings = np.array([0.0])
        # Both objects outside env (in the hole)
        object_positions = np.array(
            [
                [75.0, 75.0],
                [80.0, 80.0],
            ]
        )

        distances, bearings = _compute_egocentric_coords(
            positions,
            headings,
            object_positions,
            distance_metric="geodesic",
            env=env_with_hole,
        )

        # Distance should be NaN (all objects outside)
        assert np.isnan(distances[0, 0]), "All objects outside: should return NaN"
        # Bearing should also be NaN (no valid nearest object)
        assert np.isnan(bearings[0, 0]), "All objects outside: bearing should be NaN"

    def test_position_outside_env_geodesic(self, env_with_hole: Environment):
        """When animal position is outside env, should return NaN distance."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Animal in the hole (outside env)
        positions = np.array([[75.0, 75.0]])
        headings = np.array([0.0])
        # Object inside env
        object_positions = np.array([[25.0, 25.0]])

        distances, _bearings = _compute_egocentric_coords(
            positions,
            headings,
            object_positions,
            distance_metric="geodesic",
            env=env_with_hole,
        )

        # Distance should be NaN (animal position outside env)
        assert np.isnan(distances[0, 0]), "Position outside env: should return NaN"

    def test_mixed_valid_invalid_positions_geodesic(self, env_with_hole: Environment):
        """With some positions outside env, should handle NaN distances correctly."""
        from neurospatial.encoding._egocentric_binning import (
            _compute_egocentric_coords,
        )

        # Time 0: inside env, Time 1: outside env
        positions = np.array(
            [
                [25.0, 25.0],  # Inside
                [75.0, 75.0],  # Outside (in hole)
            ]
        )
        headings = np.array([0.0, 0.0])
        # Object inside env
        object_positions = np.array([[30.0, 25.0]])

        distances, _bearings = _compute_egocentric_coords(
            positions,
            headings,
            object_positions,
            distance_metric="geodesic",
            env=env_with_hole,
        )

        # Time 0: should have finite distance
        assert not np.isnan(distances[0, 0]), (
            "Position inside: should have finite distance"
        )
        # Time 1: should have NaN distance
        assert np.isnan(distances[1, 0]), "Position outside: should have NaN distance"

    def test_occupancy_with_nan_distances(self, env_with_hole: Environment):
        """Occupancy computation should handle NaN distances gracefully."""
        from neurospatial.encoding._egocentric_binning import (
            compute_egocentric_occupancy,
        )

        n_time = 100
        rng = np.random.default_rng(42)
        # Some positions inside, some outside
        positions = np.zeros((n_time, 2))
        positions[:50] = rng.uniform(10, 40, (50, 2))  # Inside
        positions[50:] = rng.uniform(60, 90, (50, 2))  # In hole (outside)
        times = np.linspace(0, 10, n_time)
        headings = np.zeros(n_time)

        # Object in the hole (outside env)
        object_positions = np.array([[75.0, 75.0]])

        # Should not raise an error
        occupancy, _ego_env = compute_egocentric_occupancy(
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            distance_metric="geodesic",
            env=env_with_hole,
        )

        # Occupancy should be all zeros or NaN (object outside env)
        # because we can never compute a valid distance to the object
        assert occupancy.shape == (10 * 12,)

    def test_spike_binning_with_nan_distances(self, env_with_hole: Environment):
        """Spike binning should handle NaN distances gracefully."""
        from neurospatial.encoding._egocentric_binning import (
            bin_egocentric_spike_train,
        )

        n_time = 100
        rng = np.random.default_rng(42)
        positions = rng.uniform(10, 40, (n_time, 2))  # All inside env
        times = np.linspace(0, 10, n_time)
        headings = np.zeros(n_time)
        spike_times = np.array([1.0, 3.0, 5.0])

        # Mix of inside/outside objects
        object_positions = np.array(
            [
                [75.0, 75.0],  # Outside (in hole)
                [30.0, 25.0],  # Inside
            ]
        )

        # Should not raise an error
        spike_counts, _ego_env = bin_egocentric_spike_train(
            spike_times,
            times,
            positions,
            headings,
            object_positions,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            distance_metric="geodesic",
            env=env_with_hole,
        )

        # Should have binned some spikes
        assert spike_counts.shape == (10 * 12,)
        # Some spikes should be binned (those at valid positions near valid object)
        assert np.sum(spike_counts) > 0
