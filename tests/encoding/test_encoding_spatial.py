"""Tests for neurospatial.encoding.spatial module.

This module tests the spatial encoding result classes:
- SpatialRateResult: Single-neuron spatial rate map result
- SpatialRatesResult: Multi-neuron spatial rate map result

TDD approach: Tests written first, implementation follows.

Task 2.1: Result class definitions
- SpatialRateResult dataclass (frozen=True)
- SpatialRatesResult dataclass (frozen=True)
- Inherit from SpatialResultMixin for shared methods
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

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
    # Create a 5x5 grid of positions
    x = np.linspace(0, 40, 5)
    y = np.linspace(0, 40, 5)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def firing_rate_single(simple_env: Environment) -> NDArray[np.float64]:
    """Create a single-neuron firing rate map with a clear peak."""
    n_bins = simple_env.n_bins
    firing_rate = np.zeros(n_bins, dtype=np.float64)
    # Create a peak near the center
    peak_bin = n_bins // 2
    firing_rate[peak_bin] = 20.0
    firing_rate[peak_bin - 1] = 10.0
    firing_rate[peak_bin + 1] = 10.0
    return firing_rate


@pytest.fixture
def firing_rates_batch(simple_env: Environment) -> NDArray[np.float64]:
    """Create a batch of firing rate maps for multiple neurons."""
    n_bins = simple_env.n_bins
    n_neurons = 5
    firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)
    # Create distinct peaks for each neuron
    for i in range(n_neurons):
        peak_bin = min(i * 4 + 2, n_bins - 1)
        firing_rates[i, peak_bin] = 15.0 + i * 5.0  # Different peak rates
    return firing_rates


@pytest.fixture
def occupancy(simple_env: Environment) -> NDArray[np.float64]:
    """Create uniform occupancy."""
    return np.ones(simple_env.n_bins, dtype=np.float64)


# ==============================================================================
# Test SpatialRateResult class definition
# ==============================================================================


class TestSpatialRateResultDefinition:
    """Tests for SpatialRateResult class definition (Task 2.1)."""

    def test_class_is_importable(self) -> None:
        """SpatialRateResult should be importable from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRateResult

        assert SpatialRateResult is not None

    def test_class_is_dataclass(self) -> None:
        """SpatialRateResult should be a dataclass."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRateResult

        # dataclasses have __dataclass_fields__ attribute
        assert hasattr(SpatialRateResult, "__dataclass_fields__")
        # Check that fields() works
        assert len(fields(SpatialRateResult)) > 0

    def test_class_is_frozen(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should be immutable (frozen=True)."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        # Attempting to modify should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.firing_rate = np.zeros_like(firing_rate_single)  # type: ignore[misc]

    def test_required_fields(self) -> None:
        """SpatialRateResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRateResult

        field_names = {f.name for f in fields(SpatialRateResult)}
        expected_fields = {
            "firing_rate",
            "occupancy",
            "env",
            "smoothing_method",
            "bandwidth",
        }
        assert expected_fields.issubset(field_names)

    def test_instantiation(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should be instantiable with correct fields."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert result.firing_rate is firing_rate_single
        assert result.occupancy is occupancy
        assert result.env is simple_env
        assert result.smoothing_method == "diffusion_kde"
        assert result.bandwidth == 5.0

    def test_inherits_from_spatial_result_mixin(self) -> None:
        """SpatialRateResult should inherit from SpatialResultMixin."""
        from neurospatial.encoding._base import SpatialResultMixin
        from neurospatial.encoding.spatial import SpatialRateResult

        assert issubclass(SpatialRateResult, SpatialResultMixin)

    def test_has_peak_locations_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have peak_locations() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_locations")
        assert callable(result.peak_locations)

    def test_has_peak_firing_rates_method(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRateResult should have peak_firing_rates() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_firing_rates")
        assert callable(result.peak_firing_rates)


class TestSpatialRateResultMixinMethods:
    """Tests for SpatialResultMixin methods on SpatialRateResult."""

    def test_peak_locations_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_locations() should return (n_dims,) for single neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak = result.peak_locations()
        assert isinstance(peak, np.ndarray)
        assert peak.shape == (simple_env.n_dims,)

    def test_peak_locations_finds_maximum(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """peak_locations() should find the bin with maximum firing rate."""
        from neurospatial.encoding.spatial import SpatialRateResult

        n_bins = simple_env.n_bins
        firing_rate = np.zeros(n_bins, dtype=np.float64)
        peak_bin = 5
        firing_rate[peak_bin] = 30.0

        result = SpatialRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak = result.peak_locations()
        expected = simple_env.bin_centers[peak_bin]
        np.testing.assert_array_almost_equal(peak, expected)

    def test_peak_firing_rates_returns_scalar(
        self,
        simple_env: Environment,
        firing_rate_single: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rates() should return a scalar for single neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult

        result = SpatialRateResult(
            firing_rate=firing_rate_single,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rate = result.peak_firing_rates()
        assert isinstance(peak_rate, float)
        assert peak_rate == np.max(firing_rate_single)


# ==============================================================================
# Test SpatialRatesResult class definition
# ==============================================================================


class TestSpatialRatesResultDefinition:
    """Tests for SpatialRatesResult class definition (Task 2.1)."""

    def test_class_is_importable(self) -> None:
        """SpatialRatesResult should be importable from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        assert SpatialRatesResult is not None

    def test_class_is_dataclass(self) -> None:
        """SpatialRatesResult should be a dataclass."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRatesResult

        # dataclasses have __dataclass_fields__ attribute
        assert hasattr(SpatialRatesResult, "__dataclass_fields__")
        assert len(fields(SpatialRatesResult)) > 0

    def test_class_is_frozen(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should be immutable (frozen=True)."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        with pytest.raises(FrozenInstanceError):
            result.firing_rates = np.zeros_like(firing_rates_batch)  # type: ignore[misc]

    def test_required_fields(self) -> None:
        """SpatialRatesResult should have all required fields."""
        from dataclasses import fields

        from neurospatial.encoding.spatial import SpatialRatesResult

        field_names = {f.name for f in fields(SpatialRatesResult)}
        # Note: uses firing_rates (plural) for batch
        expected_fields = {
            "firing_rates",
            "occupancy",
            "env",
            "smoothing_method",
            "bandwidth",
        }
        assert expected_fields.issubset(field_names)

    def test_instantiation(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should be instantiable with correct fields."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="gaussian_kde",
            bandwidth=10.0,
        )
        assert result.firing_rates is firing_rates_batch
        assert result.occupancy is occupancy
        assert result.env is simple_env
        assert result.smoothing_method == "gaussian_kde"
        assert result.bandwidth == 10.0

    def test_inherits_from_spatial_result_mixin(self) -> None:
        """SpatialRatesResult should inherit from SpatialResultMixin."""
        from neurospatial.encoding._base import SpatialResultMixin
        from neurospatial.encoding.spatial import SpatialRatesResult

        assert issubclass(SpatialRatesResult, SpatialResultMixin)

    def test_has_peak_locations_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have peak_locations() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_locations")
        assert callable(result.peak_locations)

    def test_has_peak_firing_rates_method(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """SpatialRatesResult should have peak_firing_rates() method from mixin."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert hasattr(result, "peak_firing_rates")
        assert callable(result.peak_firing_rates)


class TestSpatialRatesResultMixinMethods:
    """Tests for SpatialResultMixin methods on SpatialRatesResult."""

    def test_peak_locations_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_locations() should return (n_neurons, n_dims) for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peaks = result.peak_locations()
        n_neurons = firing_rates_batch.shape[0]
        assert isinstance(peaks, np.ndarray)
        assert peaks.shape == (n_neurons, simple_env.n_dims)

    def test_peak_locations_finds_maximums(
        self, simple_env: Environment, occupancy: NDArray[np.float64]
    ) -> None:
        """peak_locations() should find the bin with maximum firing rate for each neuron."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        n_bins = simple_env.n_bins
        n_neurons = 3
        firing_rates = np.zeros((n_neurons, n_bins), dtype=np.float64)
        peak_bins = [2, 5, 10]
        for i, peak_bin in enumerate(peak_bins):
            firing_rates[i, peak_bin] = 20.0 + i * 10.0

        result = SpatialRatesResult(
            firing_rates=firing_rates,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peaks = result.peak_locations()
        for i, peak_bin in enumerate(peak_bins):
            expected = simple_env.bin_centers[peak_bin]
            np.testing.assert_array_almost_equal(peaks[i], expected)

    def test_peak_firing_rates_returns_correct_shape(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rates() should return (n_neurons,) for batch."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rates = result.peak_firing_rates()
        n_neurons = firing_rates_batch.shape[0]
        assert isinstance(peak_rates, np.ndarray)
        assert peak_rates.shape == (n_neurons,)

    def test_peak_firing_rates_correct_values(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """peak_firing_rates() should return the maximum firing rate for each neuron."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        peak_rates = result.peak_firing_rates()
        expected = np.max(firing_rates_batch, axis=1)
        np.testing.assert_array_almost_equal(peak_rates, expected)


# ==============================================================================
# Test SpatialRatesResult iteration interface
# ==============================================================================


class TestSpatialRatesResultIteration:
    """Tests for SpatialRatesResult iteration interface (__len__, __getitem__, __iter__)."""

    def test_len(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__len__ should return number of neurons."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        assert len(result) == firing_rates_batch.shape[0]

    def test_getitem_returns_single_result(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__getitem__ should return a SpatialRateResult (single neuron)."""
        from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        single = result[0]
        assert isinstance(single, SpatialRateResult)

    def test_getitem_preserves_fields(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__getitem__ should preserve env, smoothing_method, bandwidth."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="gaussian_kde",
            bandwidth=7.5,
        )
        single = result[2]
        assert single.env is simple_env
        assert single.smoothing_method == "gaussian_kde"
        assert single.bandwidth == 7.5
        np.testing.assert_array_equal(single.occupancy, occupancy)

    def test_getitem_extracts_correct_firing_rate(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__getitem__ should extract the correct neuron's firing rate."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        for idx in range(len(result)):
            single = result[idx]
            np.testing.assert_array_equal(single.firing_rate, firing_rates_batch[idx])

    def test_iter(
        self,
        simple_env: Environment,
        firing_rates_batch: NDArray[np.float64],
        occupancy: NDArray[np.float64],
    ) -> None:
        """__iter__ should yield SpatialRateResult for each neuron."""
        from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult

        result = SpatialRatesResult(
            firing_rates=firing_rates_batch,
            occupancy=occupancy,
            env=simple_env,
            smoothing_method="diffusion_kde",
            bandwidth=5.0,
        )
        items = list(result)
        assert len(items) == len(result)
        for i, item in enumerate(items):
            assert isinstance(item, SpatialRateResult)
            np.testing.assert_array_equal(item.firing_rate, firing_rates_batch[i])


# ==============================================================================
# Test module exports
# ==============================================================================


class TestSpatialModuleExports:
    """Tests that the spatial module exports all expected items."""

    def test_spatial_rate_result_exported(self) -> None:
        """SpatialRateResult should be exported from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRateResult

        assert SpatialRateResult is not None

    def test_spatial_rates_result_exported(self) -> None:
        """SpatialRatesResult should be exported from encoding.spatial."""
        from neurospatial.encoding.spatial import SpatialRatesResult

        assert SpatialRatesResult is not None
