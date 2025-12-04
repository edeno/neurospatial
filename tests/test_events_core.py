"""Tests for events module core functionality."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from neurospatial.events._core import (
    PeriEventResult,
    PopulationPeriEventResult,
    plot_peri_event_histogram,
    validate_events_dataframe,
    validate_spatial_columns,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_events_df():
    """Valid events DataFrame with timestamp column."""
    return pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label": ["a", "b", "a", "b", "a"],
        }
    )


@pytest.fixture
def valid_spatial_events_df():
    """Valid events DataFrame with spatial columns."""
    return pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0],
            "x": [10.0, 20.0, 30.0],
            "y": [15.0, 25.0, 35.0],
        }
    )


@pytest.fixture
def sample_peri_event_result():
    """Sample PeriEventResult for testing."""
    n_bins = 100
    bin_centers = np.linspace(-0.5, 1.0, n_bins)
    histogram = np.random.poisson(5, n_bins).astype(float)
    sem = np.sqrt(histogram) / 10
    return PeriEventResult(
        bin_centers=bin_centers,
        histogram=histogram,
        sem=sem,
        n_events=50,
        window=(-0.5, 1.0),
        bin_size=0.015,
    )


@pytest.fixture
def sample_population_result():
    """Sample PopulationPeriEventResult for testing."""
    n_units = 5
    n_bins = 100
    bin_centers = np.linspace(-0.5, 1.0, n_bins)
    histograms = np.random.poisson(5, (n_units, n_bins)).astype(float)
    sem = np.sqrt(histograms) / 10
    mean_histogram = histograms.mean(axis=0)
    return PopulationPeriEventResult(
        bin_centers=bin_centers,
        histograms=histograms,
        sem=sem,
        mean_histogram=mean_histogram,
        n_events=50,
        n_units=n_units,
        window=(-0.5, 1.0),
        bin_size=0.015,
    )


# =============================================================================
# Test PeriEventResult
# =============================================================================


class TestPeriEventResult:
    """Tests for PeriEventResult dataclass."""

    def test_creation(self, sample_peri_event_result):
        """Test basic creation of PeriEventResult."""
        result = sample_peri_event_result
        assert result.n_events == 50
        assert result.bin_size == 0.015
        assert result.window == (-0.5, 1.0)
        assert len(result.bin_centers) == 100
        assert len(result.histogram) == 100
        assert len(result.sem) == 100

    def test_firing_rate_conversion(self, sample_peri_event_result):
        """Test firing_rate() method converts counts to Hz."""
        result = sample_peri_event_result
        rate = result.firing_rate()
        expected = result.histogram / result.bin_size
        assert_allclose(rate, expected)

    def test_frozen_dataclass(self, sample_peri_event_result):
        """Test that dataclass is immutable."""
        result = sample_peri_event_result
        with pytest.raises(AttributeError):
            result.n_events = 100  # type: ignore[misc]

    def test_shapes_must_match(self):
        """Test that mismatched shapes still create object (no validation)."""
        # Note: dataclass doesn't validate shapes - this is by design
        # Validation happens in the function that creates the result
        result = PeriEventResult(
            bin_centers=np.array([1.0, 2.0]),
            histogram=np.array([1.0, 2.0, 3.0]),  # Different length
            sem=np.array([0.1]),  # Different length
            n_events=10,
            window=(-1.0, 1.0),
            bin_size=0.1,
        )
        # The object is created - validation is caller's responsibility
        assert result.n_events == 10


# =============================================================================
# Test PopulationPeriEventResult
# =============================================================================


class TestPopulationPeriEventResult:
    """Tests for PopulationPeriEventResult dataclass."""

    def test_creation(self, sample_population_result):
        """Test basic creation of PopulationPeriEventResult."""
        result = sample_population_result
        assert result.n_events == 50
        assert result.n_units == 5
        assert result.bin_size == 0.015
        assert result.window == (-0.5, 1.0)
        assert result.histograms.shape == (5, 100)
        assert result.sem.shape == (5, 100)
        assert len(result.mean_histogram) == 100

    def test_firing_rates_conversion(self, sample_population_result):
        """Test firing_rates() method converts counts to Hz for all units."""
        result = sample_population_result
        rates = result.firing_rates()
        expected = result.histograms / result.bin_size
        assert_allclose(rates, expected)
        assert rates.shape == (5, 100)

    def test_frozen_dataclass(self, sample_population_result):
        """Test that dataclass is immutable."""
        result = sample_population_result
        with pytest.raises(AttributeError):
            result.n_units = 10  # type: ignore[misc]


# =============================================================================
# Test validate_events_dataframe
# =============================================================================


class TestValidateEventsDataframe:
    """Tests for validate_events_dataframe function."""

    def test_valid_dataframe_passes(self, valid_events_df):
        """Valid DataFrame should not raise."""
        # Should not raise
        validate_events_dataframe(valid_events_df)

    def test_non_dataframe_raises_typeerror(self):
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError, match=r"Expected pd\.DataFrame"):
            validate_events_dataframe([1, 2, 3])  # type: ignore[arg-type]

        with pytest.raises(TypeError, match=r"Expected pd\.DataFrame"):
            validate_events_dataframe(np.array([1, 2, 3]))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match=r"Expected pd\.DataFrame"):
            validate_events_dataframe({"timestamp": [1, 2]})  # type: ignore[arg-type]

    def test_missing_timestamp_raises_valueerror(self):
        """Missing timestamp column should raise ValueError."""
        df = pd.DataFrame({"time": [1.0, 2.0]})  # Wrong column name
        with pytest.raises(ValueError, match=r"Missing required columns.*timestamp"):
            validate_events_dataframe(df)

    def test_custom_timestamp_column(self):
        """Should accept custom timestamp column name."""
        df = pd.DataFrame({"event_time": [1.0, 2.0]})
        # Should not raise
        validate_events_dataframe(df, timestamp_column="event_time")

    def test_missing_custom_timestamp_raises(self):
        """Missing custom timestamp column should raise."""
        df = pd.DataFrame({"timestamp": [1.0, 2.0]})
        with pytest.raises(ValueError, match=r"Missing required columns.*event_time"):
            validate_events_dataframe(df, timestamp_column="event_time")

    def test_required_columns_check(self, valid_events_df):
        """Should check for additional required columns."""
        # This should pass - has timestamp and label
        validate_events_dataframe(valid_events_df, required_columns=["label"])

        # This should fail - missing 'x' column
        with pytest.raises(ValueError, match=r"Missing required columns.*'x'"):
            validate_events_dataframe(valid_events_df, required_columns=["x"])

    def test_non_numeric_timestamp_raises(self):
        """Non-numeric timestamp column should raise ValueError."""
        df = pd.DataFrame({"timestamp": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="non-numeric values"):
            validate_events_dataframe(df)

    def test_context_in_error_message(self):
        """Context should appear in error message."""
        df = pd.DataFrame({"time": [1.0, 2.0]})
        with pytest.raises(ValueError, match="for peri_event_histogram"):
            validate_events_dataframe(df, context="peri_event_histogram")

    def test_error_message_shows_available_columns(self):
        """Error message should list available columns."""
        df = pd.DataFrame({"time": [1.0], "value": [10]})
        with pytest.raises(ValueError, match=r"Available columns:.*time.*value"):
            validate_events_dataframe(df)

    def test_empty_dataframe_with_timestamp_column(self):
        """Empty DataFrame with timestamp column should pass."""
        df = pd.DataFrame({"timestamp": pd.Series([], dtype=float)})
        # Should not raise
        validate_events_dataframe(df)


# =============================================================================
# Test validate_spatial_columns
# =============================================================================


class TestValidateSpatialColumns:
    """Tests for validate_spatial_columns function."""

    def test_returns_true_when_present(self, valid_spatial_events_df):
        """Should return True when x, y columns present."""
        assert validate_spatial_columns(valid_spatial_events_df) is True

    def test_returns_false_when_missing(self, valid_events_df):
        """Should return False when x, y columns missing."""
        assert validate_spatial_columns(valid_events_df) is False

    def test_returns_false_with_only_x(self):
        """Should return False when only x column present."""
        df = pd.DataFrame({"timestamp": [1.0], "x": [10.0]})
        assert validate_spatial_columns(df) is False

    def test_returns_false_with_only_y(self):
        """Should return False when only y column present."""
        df = pd.DataFrame({"timestamp": [1.0], "y": [10.0]})
        assert validate_spatial_columns(df) is False

    def test_require_positions_raises_when_missing(self, valid_events_df):
        """Should raise ValueError when require_positions=True and missing."""
        with pytest.raises(ValueError, match="missing spatial columns"):
            validate_spatial_columns(valid_events_df, require_positions=True)

    def test_require_positions_passes_when_present(self, valid_spatial_events_df):
        """Should not raise when require_positions=True and columns present."""
        # Should not raise
        result = validate_spatial_columns(
            valid_spatial_events_df, require_positions=True
        )
        assert result is True

    def test_context_in_error_message(self, valid_events_df):
        """Context should appear in error message."""
        with pytest.raises(ValueError, match="spatial_event_rate"):
            validate_spatial_columns(
                valid_events_df,
                require_positions=True,
                context="spatial_event_rate",
            )

    def test_error_message_suggests_add_positions(self, valid_events_df):
        """Error message should suggest using add_positions."""
        with pytest.raises(ValueError, match="add_positions"):
            validate_spatial_columns(valid_events_df, require_positions=True)


# =============================================================================
# Test plot_peri_event_histogram
# =============================================================================


class TestPlotPeriEventHistogram:
    """Tests for plot_peri_event_histogram function."""

    def test_returns_axes(self, sample_peri_event_result):
        """Should return matplotlib Axes."""
        import matplotlib.pyplot as plt

        ax = plot_peri_event_histogram(sample_peri_event_result)
        assert ax is not None
        plt.close("all")

    def test_uses_provided_axes(self, sample_peri_event_result):
        """Should use provided axes if given."""
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots()
        returned_ax = plot_peri_event_histogram(sample_peri_event_result, ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_default_title_shows_n_events(self, sample_peri_event_result):
        """Default title should show number of events."""
        import matplotlib.pyplot as plt

        ax = plot_peri_event_histogram(sample_peri_event_result)
        assert "50 events" in ax.get_title()
        plt.close("all")

    def test_custom_title(self, sample_peri_event_result):
        """Should use custom title if provided."""
        import matplotlib.pyplot as plt

        ax = plot_peri_event_histogram(sample_peri_event_result, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close("all")

    def test_as_rate_false_changes_ylabel(self, sample_peri_event_result):
        """Should change ylabel when as_rate=False."""
        import matplotlib.pyplot as plt

        ax = plot_peri_event_histogram(sample_peri_event_result, as_rate=False)
        assert "count" in ax.get_ylabel().lower()
        plt.close("all")

    def test_as_rate_true_ylabel(self, sample_peri_event_result):
        """Should show firing rate ylabel when as_rate=True."""
        import matplotlib.pyplot as plt

        ax = plot_peri_event_histogram(sample_peri_event_result, as_rate=True)
        assert "Hz" in ax.get_ylabel() or "rate" in ax.get_ylabel().lower()
        plt.close("all")

    def test_custom_xlabel_ylabel(self, sample_peri_event_result):
        """Should use custom axis labels."""
        import matplotlib.pyplot as plt

        ax = plot_peri_event_histogram(
            sample_peri_event_result,
            xlabel="Custom X",
            ylabel="Custom Y",
        )
        assert ax.get_xlabel() == "Custom X"
        assert ax.get_ylabel() == "Custom Y"
        plt.close("all")


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_empty_events_dataframe_passes_validation(self):
        """Empty DataFrame with correct columns should pass validation."""
        df = pd.DataFrame(
            {
                "timestamp": pd.Series([], dtype=float),
                "x": pd.Series([], dtype=float),
                "y": pd.Series([], dtype=float),
            }
        )
        validate_events_dataframe(df)
        assert validate_spatial_columns(df) is True

    def test_single_event_dataframe(self):
        """Single event DataFrame should pass validation."""
        df = pd.DataFrame({"timestamp": [1.0], "x": [10.0], "y": [20.0]})
        validate_events_dataframe(df)
        assert validate_spatial_columns(df) is True

    def test_peri_event_result_with_single_bin(self):
        """PeriEventResult with single bin should work."""
        result = PeriEventResult(
            bin_centers=np.array([0.0]),
            histogram=np.array([5.0]),
            sem=np.array([0.5]),
            n_events=10,
            window=(-0.1, 0.1),
            bin_size=0.2,
        )
        rate = result.firing_rate()
        assert_allclose(rate, np.array([25.0]))  # 5.0 / 0.2 = 25.0

    def test_dataframe_with_nan_timestamp_passes_type_check(self):
        """DataFrame with NaN timestamps passes type check (numeric)."""
        df = pd.DataFrame({"timestamp": [1.0, np.nan, 3.0]})
        # NaN is still numeric type, so validation passes
        # (specific functions may handle NaN differently)
        validate_events_dataframe(df)
