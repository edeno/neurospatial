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


# =============================================================================
# EgocentricRateResult Convenience Methods Tests (Task 5.2)
# =============================================================================


class TestEgocentricRateResultPlot:
    """Test plot() method of EgocentricRateResult."""

    def test_plot_returns_axes(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that plot() returns a matplotlib Axes object."""
        from matplotlib.axes import Axes

        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        ax = result.plot()
        assert isinstance(ax, Axes)

    def test_plot_accepts_ax_parameter(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that plot() accepts an ax parameter."""
        import matplotlib.pyplot as plt

        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        fig, ax = plt.subplots()
        returned_ax = result.plot(ax=ax)
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_accepts_kwargs(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that plot() accepts keyword arguments passed to env.plot_field."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Should not raise - kwargs are passed through
        ax = result.plot(cmap="hot", vmax=30.0)
        assert ax is not None


class TestEgocentricRateResultPreferredDistance:
    """Test preferred_distance() method of EgocentricRateResult."""

    def test_preferred_distance_returns_float(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that preferred_distance() returns a float."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        dist = result.preferred_distance()
        assert isinstance(dist, float)

    def test_preferred_distance_is_nonnegative(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that preferred_distance() returns a non-negative value."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        dist = result.preferred_distance()
        assert dist >= 0.0

    def test_preferred_distance_within_range(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that preferred_distance() is within distance_range."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        distance_range = (0.0, 50.0)
        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=distance_range,
            n_distance_bins=10,
            n_direction_bins=12,
        )

        dist = result.preferred_distance()
        # Should be approximately within distance range (accounting for bin centers)
        assert dist >= distance_range[0] - 5.0  # Allow some margin for bin centers
        assert dist <= distance_range[1] + 5.0

    def test_preferred_distance_corresponds_to_peak_bin(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that preferred_distance() corresponds to peak firing bin."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create firing rate with known peak
        n_bins = sample_ego_env.n_bins
        firing_rate = np.zeros(n_bins)
        peak_idx = n_bins // 2
        firing_rate[peak_idx] = 20.0

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        dist = result.preferred_distance()
        # Distance is first component (index 0) of bin_centers
        expected_dist = float(sample_ego_env.bin_centers[peak_idx, 0])
        assert dist == expected_dist


class TestEgocentricRateResultPreferredDirection:
    """Test preferred_direction() method of EgocentricRateResult."""

    def test_preferred_direction_returns_float(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that preferred_direction() returns a float."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        direction = result.preferred_direction()
        assert isinstance(direction, float)

    def test_preferred_direction_in_valid_range(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that preferred_direction() is in [-pi, pi] range.

        Note: The sample_ego_env fixture creates positions from -pi to slightly
        past +pi to cover 12 direction bins. After binning, bin_centers may
        slightly exceed [-pi, pi] range. We allow a 1.0 radian margin to
        account for this.
        """
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        direction = result.preferred_direction()
        # Should be in valid angle range (allowing margin for bin centers from fixture)
        # Fixture creates directions from -pi to +pi with 12 bins, bin_centers may be offset
        assert direction >= -np.pi - 1.0
        assert direction <= np.pi + 1.0

    def test_preferred_direction_corresponds_to_peak_bin(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that preferred_direction() corresponds to peak firing bin."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create firing rate with known peak
        n_bins = sample_ego_env.n_bins
        firing_rate = np.zeros(n_bins)
        peak_idx = n_bins // 3  # Different from preferred_distance test
        firing_rate[peak_idx] = 25.0

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        direction = result.preferred_direction()
        # Direction is second component (index 1) of bin_centers
        expected_direction = float(sample_ego_env.bin_centers[peak_idx, 1])
        assert direction == expected_direction

    def test_preferred_direction_0_means_ahead(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test convention: 0 radians means object is ahead of animal.

        This verifies the egocentric coordinate convention documented in
        CLAUDE.md and the module docstring.
        """
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Find a bin with direction close to 0 (ahead)
        bin_centers = sample_ego_env.bin_centers
        # Find the bin closest to direction=0
        directions = bin_centers[:, 1]
        ahead_bin = np.argmin(np.abs(directions))

        # Create firing rate peaked at the "ahead" bin
        n_bins = sample_ego_env.n_bins
        firing_rate = np.zeros(n_bins)
        firing_rate[ahead_bin] = 30.0

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        direction = result.preferred_direction()
        # Should be close to 0 (ahead)
        assert abs(direction) < np.pi / 6  # Within 30 degrees of ahead


class TestEgocentricRateResultConvenienceMethodsWithNaN:
    """Test convenience methods handle NaN values correctly."""

    def test_preferred_distance_with_some_nan(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test preferred_distance() works with NaN values in firing rate."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        n_bins = sample_ego_env.n_bins
        firing_rate = np.full(n_bins, np.nan)
        firing_rate[10] = 15.0  # One valid value

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        dist = result.preferred_distance()
        # Should return distance of the only non-NaN bin
        expected_dist = float(sample_ego_env.bin_centers[10, 0])
        assert dist == expected_dist

    def test_preferred_direction_with_some_nan(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test preferred_direction() works with NaN values in firing rate."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        n_bins = sample_ego_env.n_bins
        firing_rate = np.full(n_bins, np.nan)
        firing_rate[15] = 20.0  # One valid value

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        direction = result.preferred_direction()
        # Should return direction of the only non-NaN bin
        expected_direction = float(sample_ego_env.bin_centers[15, 1])
        assert direction == expected_direction


# =============================================================================
# EgocentricRateResult Classification Tests (Task 5.3)
# =============================================================================


class TestEgocentricRateResultIsOVC:
    """Test is_ovc() method of EgocentricRateResult."""

    def test_is_ovc_returns_bool(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that is_ovc() returns a bool."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        is_ovc = result.is_ovc()
        assert isinstance(is_ovc, bool)

    def test_is_ovc_accepts_min_info_parameter(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that is_ovc() accepts min_info parameter."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Should not raise - accepts min_info parameter
        _ = result.is_ovc(min_info=0.3)
        _ = result.is_ovc(min_info=0.7)

    def test_is_ovc_default_threshold_is_0_3(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that is_ovc() has default min_info=0.3.

        This lower threshold (compared to view cells at 0.5) reflects that
        egocentric fields can be sparser and the information calculation
        is affected by the polar coordinate binning.
        """
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Should return the same as explicitly passing 0.3
        default_result = result.is_ovc()
        explicit_result = result.is_ovc(min_info=0.3)
        assert default_result == explicit_result

    def test_is_ovc_true_for_high_info(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that is_ovc() returns True for high spatial information.

        A highly selective neuron (fires only in one bin) should have
        high egocentric spatial information and be classified as OVC.
        """
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create a highly selective firing rate (fires only in one bin)
        n_bins = sample_ego_env.n_bins
        firing_rate = np.zeros(n_bins)
        firing_rate[n_bins // 2] = 30.0  # Single peak

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Should be classified as OVC (high spatial info)
        assert result.is_ovc(min_info=0.3) is True

    def test_is_ovc_false_for_uniform_firing(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that is_ovc() returns False for uniform firing.

        A neuron with uniform firing rate has zero spatial information
        and should not be classified as OVC.
        """
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create uniform firing rate (no spatial selectivity)
        n_bins = sample_ego_env.n_bins
        firing_rate = np.ones(n_bins) * 5.0

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Should NOT be classified as OVC (zero spatial info)
        assert result.is_ovc(min_info=0.3) is False

    def test_is_ovc_respects_threshold(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that is_ovc() classification depends on threshold.

        A neuron should be classified differently depending on
        the min_info threshold used.
        """
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create moderately selective firing (some spatial info)
        n_bins = sample_ego_env.n_bins
        firing_rate = np.random.rand(n_bins) * 5.0
        # Add a mild peak
        firing_rate[n_bins // 2] = 15.0

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Compute actual info to set appropriate thresholds
        info = result.egocentric_spatial_information()

        # Should be True for lower threshold
        assert result.is_ovc(min_info=info - 0.1) is True
        # Should be False for higher threshold
        assert result.is_ovc(min_info=info + 0.1) is False


class TestEgocentricRateResultEgocentricSpatialInformation:
    """Test egocentric_spatial_information() method of EgocentricRateResult.

    This method computes spatial information using egocentric occupancy.
    """

    def test_egocentric_spatial_information_returns_float(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that egocentric_spatial_information() returns a float."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        info = result.egocentric_spatial_information()
        assert isinstance(info, float)

    def test_egocentric_spatial_information_is_nonnegative(
        self,
        sample_ego_env: Environment,
        single_neuron_firing_rate: np.ndarray,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that egocentric_spatial_information() returns non-negative value."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        result = EgocentricRateResult(
            firing_rate=single_neuron_firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        info = result.egocentric_spatial_information()
        assert info >= 0.0

    def test_egocentric_spatial_information_zero_for_uniform(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that uniform firing gives zero spatial information."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create uniform firing rate
        n_bins = sample_ego_env.n_bins
        firing_rate = np.ones(n_bins) * 5.0

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        info = result.egocentric_spatial_information()
        assert abs(info) < 1e-6  # Should be approximately zero

    def test_egocentric_spatial_information_high_for_selective(
        self,
        sample_ego_env: Environment,
        single_neuron_occupancy: np.ndarray,
    ) -> None:
        """Test that selective firing gives high spatial information."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        # Create selective firing (single bin active)
        n_bins = sample_ego_env.n_bins
        firing_rate = np.zeros(n_bins)
        firing_rate[n_bins // 2] = 30.0

        result = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=single_neuron_occupancy,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        info = result.egocentric_spatial_information()
        # Highly selective firing should have high info
        assert info > 1.0  # bits/spike

    def test_egocentric_spatial_information_uses_occupancy(
        self,
        sample_ego_env: Environment,
    ) -> None:
        """Test that egocentric_spatial_information uses occupancy field."""
        from neurospatial.encoding.egocentric import EgocentricRateResult

        n_bins = sample_ego_env.n_bins
        firing_rate = np.zeros(n_bins)
        firing_rate[n_bins // 2] = 30.0

        # Non-uniform occupancy
        occupancy1 = np.ones(n_bins)
        occupancy2 = np.ones(n_bins)
        occupancy2[n_bins // 2] = 10.0  # More time at peak

        result1 = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy1,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        result2 = EgocentricRateResult(
            firing_rate=firing_rate,
            occupancy=occupancy2,
            ego_env=sample_ego_env,
            distance_range=(0.0, 50.0),
            n_distance_bins=10,
            n_direction_bins=12,
        )

        # Different occupancy should give different information
        info1 = result1.egocentric_spatial_information()
        info2 = result2.egocentric_spatial_information()
        assert info1 != info2
