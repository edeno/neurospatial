"""Tests for egocentric rate computation result classes.

This test module covers EgocentricRateResult and EgocentricRatesResult
dataclass definitions, following TDD for Task 5.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from neurospatial import Environment


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_ego_env() -> Environment:
    """Create a sample egocentric polar environment for testing.

    The egocentric environment represents distance/direction space centered
    on the animal. For simplicity, we create a regular 2D environment that
    represents the polar grid (distance on one axis, direction on another).
    """
    from neurospatial import Environment

    # Create a 2D grid environment to represent polar coordinates
    # In practice, ego_env would be created by from_polar_egocentric()
    # but for testing the result class definitions, any fitted Environment works
    n_distance = 10
    n_direction = 12
    n_bins = n_distance * n_direction

    # Create positions representing a flattened polar grid
    positions = np.zeros((n_bins, 2))
    for i in range(n_distance):
        for j in range(n_direction):
            idx = i * n_direction + j
            # Distance from 0 to 50 cm, direction from -pi to pi
            positions[idx, 0] = i * 5.0  # distance
            positions[idx, 1] = -np.pi + j * (2 * np.pi / n_direction)  # direction

    # Create environment from samples
    env = Environment.from_samples(positions, bin_size=5.0)
    return env


@pytest.fixture
def single_neuron_firing_rate(sample_ego_env: Environment) -> np.ndarray:
    """Create sample firing rate for a single neuron."""
    n_bins = sample_ego_env.n_bins
    # Create a peaked firing rate map
    rates = np.random.rand(n_bins) * 10.0
    # Add a clear peak
    peak_idx = n_bins // 2
    rates[peak_idx] = 25.0
    return rates


@pytest.fixture
def single_neuron_occupancy(sample_ego_env: Environment) -> np.ndarray:
    """Create sample occupancy for a single neuron."""
    n_bins = sample_ego_env.n_bins
    return np.ones(n_bins) * 0.5  # 0.5 seconds per bin


@pytest.fixture
def batch_firing_rates(sample_ego_env: Environment) -> np.ndarray:
    """Create sample firing rates for multiple neurons."""
    n_neurons = 5
    n_bins = sample_ego_env.n_bins
    rates = np.random.rand(n_neurons, n_bins) * 10.0
    # Add clear peaks for each neuron at different locations
    for i in range(n_neurons):
        peak_idx = (i + 1) * n_bins // (n_neurons + 1)
        rates[i, peak_idx] = 20.0 + i * 5.0
    return rates


# =============================================================================
# EgocentricRateResult Import Tests
# =============================================================================


class TestEgocentricRateResultImport:
    """Test that EgocentricRateResult can be imported correctly."""

    def test_import_from_module(self) -> None:
        """Test importing EgocentricRateResult from encoding.egocentric."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        assert EgocentricRateResult is not None

    def test_import_from_package(self) -> None:
        """Test importing EgocentricRateResult from encoding package."""
        from neurospatial.encoding import EgocentricRateResult

        assert EgocentricRateResult is not None

    def test_in_module_all(self) -> None:
        """Test EgocentricRateResult is in __all__."""
        from neurospatial.encoding import egocentric

        assert "EgocentricRateResult" in egocentric.__all__


# =============================================================================
# EgocentricRateResult Definition Tests
# =============================================================================


class TestEgocentricRateResultDefinition:
    """Test EgocentricRateResult dataclass definition."""

    def test_is_dataclass(self) -> None:
        """Test that EgocentricRateResult is a dataclass."""
        from dataclasses import is_dataclass

        from neurospatial.encoding.egocentric import EgocentricRateResult

        assert is_dataclass(EgocentricRateResult)

    def test_is_frozen(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that EgocentricRateResult is immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        with pytest.raises(FrozenInstanceError):
            result.firing_rate = np.zeros(10)  # type: ignore[misc]

    def test_has_required_fields(self) -> None:
        """Test that EgocentricRateResult has all required fields."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Check field names
        assert hasattr(EgocentricRateResult, "__dataclass_fields__")
        fields = EgocentricRateResult.__dataclass_fields__
        assert "firing_rate" in fields
        assert "occupancy" in fields
        assert "ego_env" in fields
        assert "distance_range" in fields
        assert "n_distance_bins" in fields
        assert "n_direction_bins" in fields


class TestEgocentricRateResultCreation:
    """Test creating EgocentricRateResult instances."""

    def test_create_instance(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test creating an EgocentricRateResult instance."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert result is not None
        assert isinstance(result, EgocentricRateResult)

    def test_firing_rate_accessible(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that firing_rate field is accessible."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        np.testing.assert_array_equal(result.firing_rate, single_neuron_firing_rate)

    def test_occupancy_accessible(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that occupancy field is accessible."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        np.testing.assert_array_equal(result.occupancy, single_neuron_occupancy)

    def test_ego_env_accessible(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that ego_env field is accessible."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert result.ego_env is sample_ego_env

    def test_distance_range_accessible(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that distance_range field is accessible."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert result.distance_range == (0.0, 50.0)

    def test_n_distance_bins_accessible(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that n_distance_bins field is accessible."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert result.n_distance_bins == 10

    def test_n_direction_bins_accessible(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that n_direction_bins field is accessible."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert result.n_direction_bins == 12


# =============================================================================
# EgocentricRatesResult Import Tests
# =============================================================================


class TestEgocentricRatesResultImport:
    """Test that EgocentricRatesResult can be imported correctly."""

    def test_import_from_module(self) -> None:
        """Test importing EgocentricRatesResult from encoding.egocentric."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        assert EgocentricRatesResult is not None

    def test_import_from_package(self) -> None:
        """Test importing EgocentricRatesResult from encoding package."""
        from neurospatial.encoding import EgocentricRatesResult

        assert EgocentricRatesResult is not None

    def test_in_module_all(self) -> None:
        """Test EgocentricRatesResult is in __all__."""
        from neurospatial.encoding import egocentric

        assert "EgocentricRatesResult" in egocentric.__all__


# =============================================================================
# EgocentricRatesResult Definition Tests
# =============================================================================


class TestEgocentricRatesResultDefinition:
    """Test EgocentricRatesResult dataclass definition."""

    def test_is_dataclass(self) -> None:
        """Test that EgocentricRatesResult is a dataclass."""
        from dataclasses import is_dataclass

        from neurospatial.encoding.egocentric import EgocentricRatesResult

        assert is_dataclass(EgocentricRatesResult)

    def test_is_frozen(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that EgocentricRatesResult is immutable (frozen)."""
        from dataclasses import FrozenInstanceError

        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        with pytest.raises(FrozenInstanceError):
            result.firing_rates = np.zeros((5, 10))  # type: ignore[misc]

    def test_has_required_fields(self) -> None:
        """Test that EgocentricRatesResult has all required fields."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        # Check field names
        assert hasattr(EgocentricRatesResult, "__dataclass_fields__")
        fields = EgocentricRatesResult.__dataclass_fields__
        assert "firing_rates" in fields
        assert "occupancy" in fields
        assert "ego_env" in fields
        assert "distance_range" in fields
        assert "n_distance_bins" in fields
        assert "n_direction_bins" in fields


class TestEgocentricRatesResultCreation:
    """Test creating EgocentricRatesResult instances."""

    def test_create_instance(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test creating an EgocentricRatesResult instance."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert result is not None
        assert isinstance(result, EgocentricRatesResult)

    def test_firing_rates_accessible(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that firing_rates field is accessible."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        np.testing.assert_array_equal(result.firing_rates, batch_firing_rates)


# =============================================================================
# EgocentricRatesResult Iteration Interface Tests
# =============================================================================


class TestEgocentricRatesResultLen:
    """Test __len__ method of EgocentricRatesResult."""

    def test_len_returns_n_neurons(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that len() returns number of neurons."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert len(result) == 5  # batch_firing_rates has 5 neurons

    def test_len_returns_int(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that len() returns an int."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        assert isinstance(len(result), int)


class TestEgocentricRatesResultGetitem:
    """Test __getitem__ method of EgocentricRatesResult."""

    def test_getitem_returns_single_result(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that indexing returns EgocentricRateResult."""
        from neurospatial.encoding.egocentric import (
            EgocentricRateResult,
            EgocentricRatesResult,
        )

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        single = result[0]
        assert isinstance(single, EgocentricRateResult)

    def test_getitem_has_correct_firing_rate(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that indexed result has correct firing_rate."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        single = result[2]
        np.testing.assert_array_equal(single.firing_rate, batch_firing_rates[2])

    def test_getitem_shares_occupancy(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that indexed result shares occupancy."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        single = result[0]
        np.testing.assert_array_equal(single.occupancy, single_neuron_occupancy)

    def test_getitem_shares_ego_env(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that indexed result shares ego_env."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        single = result[0]
        assert single.ego_env is sample_ego_env

    def test_getitem_preserves_metadata(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that indexed result preserves distance/direction metadata."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        single = result[0]
        assert single.distance_range == (0.0, 50.0)
        assert single.n_distance_bins == 10
        assert single.n_direction_bins == 12


class TestEgocentricRatesResultIter:
    """Test __iter__ method of EgocentricRatesResult."""

    def test_iter_yields_all_neurons(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that iteration yields all neurons."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        count = 0
        for _ in result:
            count += 1

        assert count == 5

    def test_iter_yields_correct_type(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that iteration yields EgocentricRateResult instances."""
        from neurospatial.encoding.egocentric import (
            EgocentricRateResult,
            EgocentricRatesResult,
        )

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        for single in result:
            assert isinstance(single, EgocentricRateResult)

    def test_iter_yields_correct_order(
        self,
        sample_ego_env: Environment,
        batch_firing_rates: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that iteration yields neurons in correct order."""
        from neurospatial.encoding.egocentric import EgocentricRatesResult

        result = EgocentricRatesResult(
            firing_rates=batch_firing_rates,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        for i, single in enumerate(result):
            np.testing.assert_array_equal(single.firing_rate, batch_firing_rates[i])
