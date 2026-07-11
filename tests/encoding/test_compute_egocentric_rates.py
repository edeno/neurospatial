"""Tests for compute_egocentric_rates(None) function.

This module tests the batch version of egocentric (object-vector) rate computation.
Tests cover:
- Import and API surface
- Return type and shapes
- Spike time formats (list of arrays, 2D NaN-padded)
- Parameter handling (distance_range, n_bins, metric, smoothing)
- Edge cases (empty list, single neuron, empty spike trains)
- Input validation
- Consistency with single-neuron compute_egocentric_rate(None)
- n_jobs parallelization
- Signature (canonical argument order, keyword-only params)

See Also
--------
test_compute_egocentric_rate.py : Tests for single-neuron compute_egocentric_rate(None)
test_encoding_egocentric.py : Tests for result classes
test_encoding_egocentric_binning.py : Tests for binning layer
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def trajectory_data():
    """Create trajectory data for testing."""
    rng = np.random.default_rng(42)
    n_samples = 500
    duration = 50.0

    times = np.linspace(0, duration, n_samples)
    positions = rng.uniform(10, 90, (n_samples, 2))
    headings = rng.uniform(-np.pi, np.pi, n_samples)
    object_positions = np.array([[50.0, 50.0], [25.0, 75.0]])

    return {
        "times": times,
        "positions": positions,
        "headings": headings,
        "object_positions": object_positions,
        "duration": duration,
    }


@pytest.fixture
def spike_times_list(trajectory_data):
    """Create spike times for 3 neurons."""
    rng = np.random.default_rng(123)
    duration = trajectory_data["duration"]

    return [
        np.sort(rng.uniform(0, duration, 50)),  # Neuron 0
        np.sort(rng.uniform(0, duration, 100)),  # Neuron 1
        np.sort(rng.uniform(0, duration, 25)),  # Neuron 2
    ]


@pytest.fixture
def spike_times_2d(spike_times_list):
    """Convert spike times list to 2D NaN-padded array."""
    max_spikes = max(len(s) for s in spike_times_list)
    n_neurons = len(spike_times_list)
    arr = np.full((n_neurons, max_spikes), np.nan)
    for i, spikes in enumerate(spike_times_list):
        arr[i, : len(spikes)] = spikes
    return arr


# =============================================================================
# Test: Import Tests
# =============================================================================


class TestComputeEgocentricRatesImport:
    """Tests for compute_egocentric_rates import."""

    def test_import_from_egocentric_module(self):
        """Test import from egocentric module."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        assert callable(compute_egocentric_rates)

    def test_import_from_encoding_package(self):
        """Test import from encoding package."""
        from neurospatial.encoding import compute_egocentric_rates

        assert callable(compute_egocentric_rates)

    def test_in_all(self):
        """Test that compute_egocentric_rates is in __all__."""
        from neurospatial.encoding import egocentric

        assert "compute_egocentric_rates" in egocentric.__all__


# =============================================================================
# Test: Return Type and Shapes
# =============================================================================


class TestComputeEgocentricRatesReturnsResult:
    """Tests for return type and shapes."""

    def test_returns_egocentric_rates_result(self, trajectory_data, spike_times_list):
        """Test that function returns EgocentricRatesResult."""
        from neurospatial.encoding.egocentric import (
            EgocentricRatesResult,
            compute_egocentric_rates,
        )

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert isinstance(result, EgocentricRatesResult)

    def test_firing_rates_shape(self, trajectory_data, spike_times_list):
        """Test firing_rates has shape (n_neurons, n_bins)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        n_distance_bins = 8
        n_direction_bins = 10

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )

        n_neurons = len(spike_times_list)
        n_bins = n_distance_bins * n_direction_bins

        assert result.firing_rates.shape == (n_neurons, n_bins)

    def test_occupancy_shape(self, trajectory_data, spike_times_list):
        """Test occupancy has shape (n_bins,)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        n_distance_bins = 8
        n_direction_bins = 10

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            n_distance_bins=n_distance_bins,
            n_direction_bins=n_direction_bins,
        )

        n_bins = n_distance_bins * n_direction_bins
        assert result.occupancy.shape == (n_bins,)

    def test_env_is_environment(self, trajectory_data, spike_times_list):
        """Test that env is an Environment."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates
        from neurospatial.environment import Environment

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        from neurospatial.environment.polar import EgocentricPolarEnvironment

        # Egocentric rates run on a distinct polar environment type.
        assert isinstance(result.env, EgocentricPolarEnvironment)
        assert not isinstance(result.env, Environment)

    def test_env_n_bins_matches_occupancy(self, trajectory_data, spike_times_list):
        """Test env.n_bins matches occupancy length."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert result.env.n_bins == len(result.occupancy)


# =============================================================================
# Test: Spike Time Formats
# =============================================================================


class TestComputeEgocentricRatesSpikeTimeFormats:
    """Tests for different spike time formats."""

    def test_accepts_list_of_arrays(self, trajectory_data, spike_times_list):
        """Test that list of arrays works."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert result.firing_rates.shape[0] == len(spike_times_list)

    def test_accepts_2d_nan_padded(self, trajectory_data, spike_times_2d):
        """Test that 2D NaN-padded array works."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_2d,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        n_neurons = spike_times_2d.shape[0]
        assert result.firing_rates.shape[0] == n_neurons

    def test_accepts_single_1d_array(self, trajectory_data):
        """Test that single 1D array is wrapped in list."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        rng = np.random.default_rng(42)
        spike_times_1d = np.sort(rng.uniform(0, trajectory_data["duration"], 50))

        result = compute_egocentric_rates(
            None,
            spike_times_1d,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        # Should have 1 neuron
        assert result.firing_rates.shape[0] == 1


# =============================================================================
# Test: Parameters
# =============================================================================


class TestComputeEgocentricRatesParameters:
    """Tests for parameter handling."""

    def test_distance_range_stored(self, trajectory_data, spike_times_list):
        """Test that distance_range parameter is stored."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        distance_range = (5.0, 40.0)

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            distance_range=distance_range,
        )

        assert result.distance_range == distance_range

    def test_n_distance_bins_stored(self, trajectory_data, spike_times_list):
        """Test that n_distance_bins parameter is stored."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        n_distance_bins = 15

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            n_distance_bins=n_distance_bins,
        )

        assert result.n_distance_bins == n_distance_bins

    def test_n_direction_bins_stored(self, trajectory_data, spike_times_list):
        """Test that n_direction_bins parameter is stored."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        n_direction_bins = 24

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            n_direction_bins=n_direction_bins,
        )

        assert result.n_direction_bins == n_direction_bins

    def test_metric_euclidean_default(self, trajectory_data, spike_times_list):
        """Test that metric defaults to euclidean."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # Should work without env parameter
        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert result is not None

    def test_method_options(self, trajectory_data, spike_times_list):
        """Test that all smoothing methods work."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        for method in ["binned", "gaussian_kde", "diffusion_kde"]:
            result = compute_egocentric_rates(
                None,
                spike_times_list,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                trajectory_data["object_positions"],
                method=method,
            )
            assert result is not None

    def test_n_jobs_parameter(self, trajectory_data, spike_times_list):
        """Test that n_jobs parameter works."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # Test with different n_jobs values
        for n_jobs in [1, 2, -1]:
            result = compute_egocentric_rates(
                None,
                spike_times_list,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                trajectory_data["object_positions"],
                n_jobs=n_jobs,
            )
            assert result.firing_rates.shape[0] == len(spike_times_list)


# =============================================================================
# Test: Neuron Iteration
# =============================================================================


class TestComputeEgocentricRatesNeuronIteration:
    """Tests for neuron iteration support."""

    def test_len_returns_n_neurons(self, trajectory_data, spike_times_list):
        """Test that len(result) returns n_neurons."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert len(result) == len(spike_times_list)

    def test_getitem_returns_single_result(self, trajectory_data, spike_times_list):
        """Test that result[i] returns EgocentricRateResult."""
        from neurospatial.encoding.egocentric import (
            EgocentricRateResult,
            compute_egocentric_rates,
        )

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        single = result[0]
        assert isinstance(single, EgocentricRateResult)

    def test_iteration_yields_all_neurons(self, trajectory_data, spike_times_list):
        """Test that iteration yields all neurons."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        singles = list(result)
        assert len(singles) == len(spike_times_list)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestComputeEgocentricRatesEdgeCases:
    """Tests for edge cases."""

    def test_empty_spike_times_list(self, trajectory_data):
        """Test with empty list of neurons."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            [],  # Empty list
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert len(result) == 0
        assert result.firing_rates.shape[0] == 0
        # But occupancy should still be computed
        assert len(result.occupancy) > 0

    def test_single_neuron(self, trajectory_data):
        """Test with single neuron."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        rng = np.random.default_rng(42)
        spike_times = [np.sort(rng.uniform(0, trajectory_data["duration"], 50))]

        result = compute_egocentric_rates(
            None,
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert len(result) == 1

    def test_neuron_with_no_spikes(self, trajectory_data, spike_times_list):
        """Test neuron with no spikes."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # Add empty spike train
        spike_times_with_empty = [*spike_times_list, np.array([])]

        result = compute_egocentric_rates(
            None,
            spike_times_with_empty,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert len(result) == len(spike_times_with_empty)
        # Empty neuron fires nowhere: visited bins are 0; unvisited bins (zero
        # occupancy) are NaN under the raw "binned" rate (undefined, not 0).
        empty_rate = np.asarray(result.firing_rates[-1])
        assert np.all(empty_rate[~np.isnan(empty_rate)] == 0.0)
        assert np.any(~np.isnan(empty_rate))  # some bins were visited


# =============================================================================
# Test: Input Validation
# =============================================================================


class TestComputeEgocentricRatesInputValidation:
    """Tests for input validation."""

    def test_empty_spike_times_list_validates_method(self, trajectory_data):
        """Empty list should still reject invalid smoothing methods."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        with pytest.raises(ValueError, match="method must be one of"):
            compute_egocentric_rates(
                None,
                [],
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                trajectory_data["object_positions"],
                method="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_metric_raises(self, trajectory_data, spike_times_list):
        """Test that invalid metric raises ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        with pytest.raises(ValueError, match="Invalid metric"):
            compute_egocentric_rates(
                None,
                spike_times_list,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                trajectory_data["object_positions"],
                metric="invalid",
            )

    def test_geodesic_without_env_raises(self, trajectory_data, spike_times_list):
        """Test that geodesic without env raises ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        with pytest.raises(ValueError, match="requires env parameter"):
            compute_egocentric_rates(
                None,
                spike_times_list,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                trajectory_data["object_positions"],
                metric="geodesic",
            )

    def test_mismatched_times_positions_raises(self, trajectory_data, spike_times_list):
        """Test that mismatched times/positions raises ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # Create wrong-sized positions
        wrong_positions = trajectory_data["positions"][:-10]

        with pytest.raises(ValueError, match=r"times length.*positions length"):
            compute_egocentric_rates(
                None,
                spike_times_list,
                trajectory_data["times"],
                wrong_positions,
                trajectory_data["headings"],
                trajectory_data["object_positions"],
            )

    def test_mismatched_times_headings_raises(self, trajectory_data, spike_times_list):
        """Test that mismatched times/headings raises ValueError."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # Create wrong-sized headings
        wrong_headings = trajectory_data["headings"][:-10]

        with pytest.raises(ValueError, match=r"times length.*headings length"):
            compute_egocentric_rates(
                None,
                spike_times_list,
                trajectory_data["times"],
                trajectory_data["positions"],
                wrong_headings,
                trajectory_data["object_positions"],
            )


# =============================================================================
# Test: Consistency with Single-Neuron Function
# =============================================================================


class TestComputeEgocentricRatesConsistencyWithSingle:
    """Tests for consistency with single-neuron compute_egocentric_rate."""

    def test_single_neuron_matches_single_function(self, trajectory_data):
        """Test that single neuron result matches compute_egocentric_rate."""
        from neurospatial.encoding.egocentric import (
            compute_egocentric_rate,
            compute_egocentric_rates,
        )

        rng = np.random.default_rng(42)
        spike_times = np.sort(rng.uniform(0, trajectory_data["duration"], 50))

        # Compute with single function
        single_result = compute_egocentric_rate(
            None,
            spike_times,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            method="binned",
        )

        # Compute with batch function
        batch_result = compute_egocentric_rates(
            None,
            [spike_times],
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            method="binned",
        )

        # Compare results
        assert_allclose(
            batch_result.firing_rates[0],
            single_result.firing_rate,
            rtol=1e-10,
        )
        assert_allclose(
            batch_result.occupancy,
            single_result.occupancy,
            rtol=1e-10,
        )

    def test_multiple_neurons_consistent(self, trajectory_data, spike_times_list):
        """Test that batch result is consistent with single computations."""
        from neurospatial.encoding.egocentric import (
            compute_egocentric_rate,
            compute_egocentric_rates,
        )

        # Compute batch
        batch_result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
            method="binned",
        )

        # Compute each neuron individually
        for i, spike_times in enumerate(spike_times_list):
            single_result = compute_egocentric_rate(
                None,
                spike_times,
                trajectory_data["times"],
                trajectory_data["positions"],
                trajectory_data["headings"],
                trajectory_data["object_positions"],
                distance_range=(0.0, 50.0),
                n_distance_bins=10,
                n_direction_bins=12,
                method="binned",
            )

            assert_allclose(
                batch_result.firing_rates[i],
                single_result.firing_rate,
                rtol=1e-10,
                err_msg=f"Neuron {i} firing rates don't match",
            )


# =============================================================================
# Test: Signature (Argument Order and Keyword-Only)
# =============================================================================


class TestComputeEgocentricRatesSignature:
    """Tests for function signature."""

    def test_canonical_argument_order(self, trajectory_data, spike_times_list):
        """Test canonical argument order (env first per CLAUDE.md)."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        # env, spike_times, times, positions, headings, object_positions
        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert result is not None

    def test_keyword_only_parameters(self, trajectory_data, spike_times_list):
        """Test that certain parameters are keyword-only."""
        import inspect

        from neurospatial.encoding.egocentric import compute_egocentric_rates

        sig = inspect.signature(compute_egocentric_rates)

        # `env` is first positional (canonical order); excluded here.
        keyword_only_params = [
            "distance_range",
            "n_distance_bins",
            "n_direction_bins",
            "metric",
            "method",
            "bandwidth",
            "min_occupancy",
            "n_jobs",
        ]

        for param_name in keyword_only_params:
            param = sig.parameters.get(param_name)
            assert param is not None, f"Missing parameter: {param_name}"
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{param_name} should be keyword-only"
            )


# =============================================================================
# Test: Geodesic Distance
# =============================================================================


class TestComputeEgocentricRatesGeodesic:
    """Tests for geodesic distance metric."""

    def test_geodesic_with_env_works(self, trajectory_data, spike_times_list):
        """Test that geodesic with env parameter works."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates
        from neurospatial.environment import Environment

        # Create environment
        env = Environment.from_samples(trajectory_data["positions"], bin_size=5.0)

        result = compute_egocentric_rates(
            env,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            metric="geodesic",
        )

        assert result is not None


# =============================================================================
# Test: Correctness
# =============================================================================


class TestComputeEgocentricRatesCorrectness:
    """Tests for correctness of computed values."""

    def test_firing_rates_non_negative(self, trajectory_data, spike_times_list):
        """Test that firing rates are non-negative."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        # All firing rates should be >= 0 (or NaN)
        non_nan_rates = result.firing_rates[~np.isnan(result.firing_rates)]
        assert np.all(non_nan_rates >= 0)

    def test_occupancy_non_negative(self, trajectory_data, spike_times_list):
        """Test that occupancy is non-negative."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        assert np.all(result.occupancy >= 0)

    def test_total_occupancy_approximates_duration(
        self, trajectory_data, spike_times_list
    ):
        """Test that total occupancy approximately equals duration."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
            # Use large distance range to include all samples
            distance_range=(0.0, 200.0),
        )

        total_occupancy = result.occupancy.sum()
        expected_duration = trajectory_data["duration"]

        # Should be close (within 10% - some bins may be out of range)
        assert total_occupancy > 0
        assert total_occupancy <= expected_duration * 1.1


# =============================================================================
# Test: Result Methods Integration
# =============================================================================


class TestComputeEgocentricRatesResultMethods:
    """Tests for result class method integration."""

    def test_plot_method_works(self, trajectory_data, spike_times_list):
        """Test that plot method works on batch result."""
        import matplotlib

        from neurospatial.encoding.egocentric import compute_egocentric_rates

        matplotlib.use("Agg")

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        ax = result.plot(0)
        assert ax is not None

    def test_preferred_distances_method_works(self, trajectory_data, spike_times_list):
        """Test that preferred_distances method works."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        pref_dists = result.preferred_distances()
        assert len(pref_dists) == len(spike_times_list)

    def test_preferred_directions_method_works(self, trajectory_data, spike_times_list):
        """Test that preferred_directions method works."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        pref_dirs = result.preferred_directions()
        assert len(pref_dirs) == len(spike_times_list)

    def test_preferred_distances_directions_match_loop(
        self, trajectory_data, spike_times_list
    ):
        """Vectorized peak lookup matches an explicit per-neuron loop.

        Equivalence guard for the vectorized ``np.nanargmax(..., axis=1)``
        implementation of ``preferred_distances`` / ``preferred_directions``:
        results must be identical (bit-for-bit) to indexing ``bin_centers``
        with a per-neuron ``int(np.nanargmax(firing_rates[i]))``.
        """
        from neurospatial.encoding._base import _to_numpy
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        firing_rates = _to_numpy(result.firing_rates)
        bin_centers = result.env.bin_centers
        n_neurons = firing_rates.shape[0]

        expected_dist = np.array(
            [
                bin_centers[int(np.nanargmax(firing_rates[i])), 0]
                for i in range(n_neurons)
            ]
        )
        expected_dir = np.array(
            [
                bin_centers[int(np.nanargmax(firing_rates[i])), 1]
                for i in range(n_neurons)
            ]
        )

        np.testing.assert_array_equal(result.preferred_distances(), expected_dist)
        np.testing.assert_array_equal(result.preferred_directions(), expected_dir)

    def test_classify_method_works(self, trajectory_data, spike_times_list):
        """Test that classify method works."""
        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        is_object_vector_cell = result.classify()
        assert len(is_object_vector_cell) == len(spike_times_list)
        assert is_object_vector_cell.dtype == bool

    def test_summary_table_method_works(self, trajectory_data, spike_times_list):
        """Test that summary_table (per-unit) and to_dataframe (dense) work."""
        import pandas as pd

        from neurospatial.encoding.egocentric import compute_egocentric_rates

        result = compute_egocentric_rates(
            None,
            spike_times_list,
            trajectory_data["times"],
            trajectory_data["positions"],
            trajectory_data["headings"],
            trajectory_data["object_positions"],
        )

        summary = result.summary_table()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(spike_times_list)
        assert summary.index.name == "unit_id"

        dense = result.to_dataframe()
        assert {"unit_id", "bin", "firing_rate"} <= set(dense.columns)
        assert len(dense) == len(spike_times_list) * result.env.n_bins
