"""
Tests for NWB spatial fields writing functions.

Tests the write_place_field() and write_occupancy() functions for writing
spatial analysis results to NWB files.
"""

from __future__ import annotations

import numpy as np
import pytest

# pynwb is required for all tests
pynwb = pytest.importorskip("pynwb")


class TestWritePlaceField:
    """Tests for write_place_field() function."""

    def test_basic_field_writing(self, empty_nwb, sample_environment):
        """Test basic place field writing to analysis/ module."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        # Create a place field (one value per bin)
        rng = np.random.default_rng(42)
        field = rng.uniform(0, 10, env.n_bins)

        # Write to NWB
        write_place_field(nwbfile, env, field, name="cell_001")

        # Verify analysis module was created
        assert "analysis" in nwbfile.processing

        # Verify place field exists in analysis module
        assert "cell_001" in nwbfile.processing["analysis"].data_interfaces

    def test_field_metadata(self, empty_nwb, sample_environment):
        """Test that field metadata (description, units) is stored correctly."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        field = np.random.default_rng(42).uniform(0, 10, env.n_bins)

        write_place_field(
            nwbfile,
            env,
            field,
            name="cell_001",
            description="Place field for pyramidal cell 001",
        )

        # Get the stored TimeSeries
        place_field_ts = nwbfile.processing["analysis"]["cell_001"]

        # Check description
        assert place_field_ts.description == "Place field for pyramidal cell 001"

        # Check unit (should indicate firing rate or similar)
        assert place_field_ts.unit is not None

    def test_field_data_matches_input(self, empty_nwb, sample_environment):
        """Test that stored field data matches the input array."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        rng = np.random.default_rng(42)
        field = rng.uniform(0, 10, env.n_bins)

        write_place_field(nwbfile, env, field, name="cell_001")

        # Get the stored data
        place_field_ts = nwbfile.processing["analysis"]["cell_001"]
        stored_data = place_field_ts.data[:]

        # Static 1D fields are stored as (1, n_bins) for NWB compatibility
        # Squeeze to compare with original 1D input
        np.testing.assert_array_almost_equal(stored_data.squeeze(), field)

    def test_shape_validation_1d_field(self, empty_nwb, sample_environment):
        """Test shape validation for 1D field (n_bins,)."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        # Wrong shape - too few bins
        wrong_field = np.ones(env.n_bins - 5)

        with pytest.raises(ValueError, match="shape"):
            write_place_field(nwbfile, env, wrong_field, name="bad_field")

    def test_shape_validation_2d_field(self, empty_nwb, sample_environment):
        """Test shape validation for 2D field (n_time, n_bins)."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        # Valid 2D shape: (n_time, n_bins)
        n_time = 100
        field_2d = np.random.default_rng(42).uniform(0, 10, (n_time, env.n_bins))

        # Should succeed
        write_place_field(nwbfile, env, field_2d, name="time_varying_field")

        # Verify data shape
        stored = nwbfile.processing["analysis"]["time_varying_field"]
        assert stored.data.shape == (n_time, env.n_bins)

    def test_shape_validation_wrong_n_bins(self, empty_nwb, sample_environment):
        """Test error when field n_bins doesn't match environment."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        # Wrong number of bins in 2D field
        wrong_field = np.ones((10, env.n_bins + 10))

        with pytest.raises(ValueError, match="n_bins"):
            write_place_field(nwbfile, env, wrong_field, name="bad_field")

    def test_error_duplicate_name_without_overwrite(
        self, empty_nwb, sample_environment
    ):
        """Test ValueError when field with same name exists and overwrite=False."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.ones(env.n_bins)

        # Write first time
        write_place_field(nwbfile, env, field, name="cell_001")

        # Attempt to write again without overwrite
        with pytest.raises(ValueError, match="already exists"):
            write_place_field(nwbfile, env, field, name="cell_001")

    def test_overwrite_replaces_existing(self, empty_nwb, sample_environment):
        """Test overwrite=True replaces existing field."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        # Write first field with value 1.0
        field_v1 = np.ones(env.n_bins)
        write_place_field(nwbfile, env, field_v1, name="cell_001")

        # Overwrite with different values
        field_v2 = np.ones(env.n_bins) * 2.0
        write_place_field(nwbfile, env, field_v2, name="cell_001", overwrite=True)

        # Verify new data (squeeze for static 1D field)
        stored = nwbfile.processing["analysis"]["cell_001"]
        np.testing.assert_array_almost_equal(stored.data[:].squeeze(), field_v2)

    def test_bin_centers_reference_stored(self, empty_nwb, sample_environment):
        """Test that bin_centers reference is stored with the field."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.random.default_rng(42).uniform(0, 10, env.n_bins)

        write_place_field(nwbfile, env, field, name="cell_001")

        # The field should have a reference to bin_centers
        # This could be stored as a linked dataset or as a control/control_description
        place_field_ts = nwbfile.processing["analysis"]["cell_001"]

        # Check that bin_centers are stored somewhere accessible
        # Option 1: Stored in control (metadata array)
        # Option 2: Stored as separate dataset with link
        # Option 3: Stored as JSON string in description

        # Verify bin_centers are recoverable
        # The implementation should store them in a way that allows reconstruction
        # Check if there's a 'bin_centers' dataset in the analysis module
        assert (
            "bin_centers" in nwbfile.processing["analysis"].data_interfaces
            or place_field_ts.control is not None
            or "bin_centers" in str(place_field_ts.comments).lower()
        )

    def test_multiple_fields_same_environment(self, empty_nwb, sample_environment):
        """Test writing multiple place fields for the same environment."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        rng = np.random.default_rng(42)

        # Write multiple cells
        for i in range(5):
            field = rng.uniform(0, 10, env.n_bins)
            write_place_field(nwbfile, env, field, name=f"cell_{i:03d}")

        # Verify all are present
        analysis = nwbfile.processing["analysis"]
        for i in range(5):
            assert f"cell_{i:03d}" in analysis.data_interfaces

    def test_default_name(self, empty_nwb, sample_environment):
        """Test default name 'place_field' when no name provided."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.ones(env.n_bins)

        write_place_field(nwbfile, env, field)

        assert "place_field" in nwbfile.processing["analysis"].data_interfaces

    def test_empty_description_allowed(self, empty_nwb, sample_environment):
        """Test that empty description is allowed."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.ones(env.n_bins)

        # Should not raise
        write_place_field(nwbfile, env, field, name="cell_001", description="")

        stored = nwbfile.processing["analysis"]["cell_001"]
        assert stored.description == ""

    def test_analysis_module_reused(self, empty_nwb, sample_environment):
        """Test that existing analysis module is reused, not recreated."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.ones(env.n_bins)

        # Create analysis module manually with custom description
        nwbfile.create_processing_module(
            name="analysis", description="Custom analysis module"
        )

        write_place_field(nwbfile, env, field, name="cell_001")

        # Module should be reused, not replaced
        assert nwbfile.processing["analysis"].description == "Custom analysis module"
        assert "cell_001" in nwbfile.processing["analysis"].data_interfaces

    def test_field_with_nan_values(self, empty_nwb, sample_environment):
        """Test writing field containing NaN values (unvisited bins)."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        # Field with some NaN values
        field = np.random.default_rng(42).uniform(0, 10, env.n_bins)
        field[::10] = np.nan  # Every 10th bin is NaN

        write_place_field(nwbfile, env, field, name="sparse_field")

        stored = nwbfile.processing["analysis"]["sparse_field"]
        stored_data = stored.data[:].squeeze()  # Squeeze for static 1D field

        # NaNs should be preserved
        assert np.sum(np.isnan(stored_data)) == np.sum(np.isnan(field))
        np.testing.assert_array_equal(np.isnan(stored_data), np.isnan(field))

    def test_field_units_stored(self, empty_nwb, sample_environment):
        """Test that field units default to reasonable value."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.ones(env.n_bins)

        write_place_field(nwbfile, env, field, name="cell_001")

        stored = nwbfile.processing["analysis"]["cell_001"]

        # Default unit should be "Hz" (SI-compliant for firing rates)
        assert stored.unit == "Hz"

    def test_custom_unit_parameter(self, empty_nwb, sample_environment):
        """Test that custom unit parameter is stored correctly."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.ones(env.n_bins)

        write_place_field(
            nwbfile, env, field, name="probability_map", unit="probability"
        )

        stored = nwbfile.processing["analysis"]["probability_map"]
        assert stored.unit == "probability"

    def test_shape_validation_3d_field_rejected(self, empty_nwb, sample_environment):
        """Test that 3D fields are rejected with ValueError."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment

        # 3D field should be rejected
        field_3d = np.ones((5, 10, env.n_bins))

        with pytest.raises(ValueError, match=r"1D.*2D"):
            write_place_field(nwbfile, env, field_3d, name="bad_3d_field")

    def test_bin_centers_deduplicated_multiple_fields(
        self, empty_nwb, sample_environment
    ):
        """Test that bin_centers are stored only once when writing multiple fields."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        rng = np.random.default_rng(42)

        # Write several fields
        for i in range(3):
            field = rng.uniform(0, 10, env.n_bins)
            write_place_field(nwbfile, env, field, name=f"cell_{i:03d}")

        # Bin centers should be stored exactly once, not duplicated
        analysis = nwbfile.processing["analysis"]

        # Count how many 'bin_centers' datasets exist
        bin_centers_count = sum(
            1 for name in analysis.data_interfaces if "bin_centers" in name.lower()
        )

        # Should only have one bin_centers dataset
        assert bin_centers_count <= 1, (
            f"Expected at most 1 bin_centers dataset, found {bin_centers_count}"
        )

    def test_bin_centers_data_matches_environment(self, empty_nwb, sample_environment):
        """Test that stored bin_centers match the environment's bin_centers."""
        from neurospatial.nwb import write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        field = np.ones(env.n_bins)

        write_place_field(nwbfile, env, field, name="cell_001")

        # Find and verify bin_centers data
        analysis = nwbfile.processing["analysis"]

        # The bin_centers should be stored and match the environment
        if "bin_centers" in analysis.data_interfaces:
            # bin_centers stored as (1, n_bins, n_dims) for NWB TimeSeries compatibility
            stored_centers = analysis["bin_centers"].data[:].squeeze()
            np.testing.assert_array_almost_equal(stored_centers, env.bin_centers)


class TestWriteOccupancy:
    """Tests for write_occupancy() function."""

    def test_basic_occupancy_writing(self, empty_nwb, sample_environment):
        """Test basic occupancy writing to analysis/ module."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Create occupancy map (one value per bin)
        rng = np.random.default_rng(42)
        occupancy = rng.uniform(0, 10, env.n_bins)

        # Write to NWB
        write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

        # Verify analysis module was created
        assert "analysis" in nwbfile.processing

        # Verify occupancy exists in analysis module
        assert "occupancy_map" in nwbfile.processing["analysis"].data_interfaces

    def test_occupancy_metadata(self, empty_nwb, sample_environment):
        """Test that occupancy metadata (description, units) is stored correctly."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        occupancy = np.random.default_rng(42).uniform(0, 10, env.n_bins)

        write_occupancy(
            nwbfile,
            env,
            occupancy,
            name="occupancy_map",
            description="Time spent in each bin",
        )

        # Get the stored TimeSeries
        occupancy_ts = nwbfile.processing["analysis"]["occupancy_map"]

        # Check description
        assert occupancy_ts.description == "Time spent in each bin"

        # Check unit (should have units attribute)
        assert occupancy_ts.unit is not None

    def test_occupancy_data_matches_input(self, empty_nwb, sample_environment):
        """Test that stored occupancy data matches the input array."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        rng = np.random.default_rng(42)
        occupancy = rng.uniform(0, 10, env.n_bins)

        write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

        # Get the stored data
        occupancy_ts = nwbfile.processing["analysis"]["occupancy_map"]
        stored_data = occupancy_ts.data[:]

        # Static 1D fields are stored as (1, n_bins) for NWB compatibility
        # Squeeze to compare with original 1D input
        np.testing.assert_array_almost_equal(stored_data.squeeze(), occupancy)

    def test_unit_seconds(self, empty_nwb, sample_environment):
        """Test occupancy with unit='seconds' (default)."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment
        occupancy = np.ones(env.n_bins) * 5.0  # 5 seconds per bin

        write_occupancy(nwbfile, env, occupancy, name="time_occupancy")

        stored = nwbfile.processing["analysis"]["time_occupancy"]
        assert stored.unit == "seconds"

    def test_unit_probability(self, empty_nwb, sample_environment):
        """Test occupancy with unit='probability' for normalized maps."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Normalized occupancy that sums to 1
        rng = np.random.default_rng(42)
        occupancy = rng.uniform(0, 1, env.n_bins)
        occupancy = occupancy / occupancy.sum()  # Normalize to probability

        write_occupancy(
            nwbfile, env, occupancy, name="prob_occupancy", unit="probability"
        )

        stored = nwbfile.processing["analysis"]["prob_occupancy"]
        assert stored.unit == "probability"

    def test_unit_counts(self, empty_nwb, sample_environment):
        """Test occupancy with unit='counts' for raw bin visit counts."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Raw counts - integer-like values
        rng = np.random.default_rng(42)
        occupancy = rng.integers(0, 100, env.n_bins).astype(np.float64)

        write_occupancy(nwbfile, env, occupancy, name="count_occupancy", unit="counts")

        stored = nwbfile.processing["analysis"]["count_occupancy"]
        assert stored.unit == "counts"

    def test_shape_validation_correct_shape(self, empty_nwb, sample_environment):
        """Test shape validation passes for correct 1D shape (n_bins,)."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Correct shape
        occupancy = np.ones(env.n_bins)

        # Should succeed without error
        write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

        assert "occupancy_map" in nwbfile.processing["analysis"].data_interfaces

    def test_shape_validation_wrong_size(self, empty_nwb, sample_environment):
        """Test shape validation fails for wrong number of bins."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Wrong shape - too few bins
        wrong_occupancy = np.ones(env.n_bins - 5)

        with pytest.raises(ValueError, match="shape"):
            write_occupancy(nwbfile, env, wrong_occupancy, name="bad_occupancy")

    def test_shape_validation_2d_rejected(self, empty_nwb, sample_environment):
        """Test that 2D arrays are rejected (occupancy should be 1D only)."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # 2D array should be rejected - occupancy is static, not time-varying
        occupancy_2d = np.ones((10, env.n_bins))

        with pytest.raises(ValueError, match=r"1D|shape"):
            write_occupancy(nwbfile, env, occupancy_2d, name="bad_occupancy")

    def test_error_duplicate_name_without_overwrite(
        self, empty_nwb, sample_environment
    ):
        """Test ValueError when occupancy with same name exists and overwrite=False."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment
        occupancy = np.ones(env.n_bins)

        # Write first time
        write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

        # Attempt to write again without overwrite
        with pytest.raises(ValueError, match="already exists"):
            write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

    def test_overwrite_replaces_existing(self, empty_nwb, sample_environment):
        """Test overwrite=True replaces existing occupancy."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Write first occupancy with value 1.0
        occupancy_v1 = np.ones(env.n_bins)
        write_occupancy(nwbfile, env, occupancy_v1, name="occupancy_map")

        # Overwrite with different values
        occupancy_v2 = np.ones(env.n_bins) * 2.0
        write_occupancy(
            nwbfile, env, occupancy_v2, name="occupancy_map", overwrite=True
        )

        # Verify new data (squeeze for static 1D field)
        stored = nwbfile.processing["analysis"]["occupancy_map"]
        np.testing.assert_array_almost_equal(stored.data[:].squeeze(), occupancy_v2)

    def test_bin_centers_reference_stored(self, empty_nwb, sample_environment):
        """Test that bin_centers reference is stored with the occupancy."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment
        occupancy = np.random.default_rng(42).uniform(0, 10, env.n_bins)

        write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

        # The occupancy should have a reference to bin_centers
        occupancy_ts = nwbfile.processing["analysis"]["occupancy_map"]

        # Verify bin_centers are recoverable
        assert (
            "bin_centers" in nwbfile.processing["analysis"].data_interfaces
            or occupancy_ts.control is not None
            or "bin_centers" in str(occupancy_ts.comments).lower()
        )

    def test_default_name(self, empty_nwb, sample_environment):
        """Test default name 'occupancy' when no name provided."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment
        occupancy = np.ones(env.n_bins)

        write_occupancy(nwbfile, env, occupancy)

        assert "occupancy" in nwbfile.processing["analysis"].data_interfaces

    def test_empty_description_allowed(self, empty_nwb, sample_environment):
        """Test that empty description is allowed."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment
        occupancy = np.ones(env.n_bins)

        # Should not raise
        write_occupancy(nwbfile, env, occupancy, name="occupancy_map", description="")

        stored = nwbfile.processing["analysis"]["occupancy_map"]
        assert stored.description == ""

    def test_analysis_module_reused(self, empty_nwb, sample_environment):
        """Test that existing analysis module is reused, not recreated."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment
        occupancy = np.ones(env.n_bins)

        # Create analysis module manually with custom description
        nwbfile.create_processing_module(
            name="analysis", description="Custom analysis module"
        )

        write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

        # Module should be reused, not replaced
        assert nwbfile.processing["analysis"].description == "Custom analysis module"
        assert "occupancy_map" in nwbfile.processing["analysis"].data_interfaces

    def test_occupancy_with_nan_values(self, empty_nwb, sample_environment):
        """Test writing occupancy containing NaN values (unvisited bins)."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Occupancy with some NaN values (unvisited bins)
        occupancy = np.random.default_rng(42).uniform(0, 10, env.n_bins)
        occupancy[::10] = np.nan  # Every 10th bin is unvisited

        write_occupancy(nwbfile, env, occupancy, name="sparse_occupancy")

        stored = nwbfile.processing["analysis"]["sparse_occupancy"]
        stored_data = stored.data[:].squeeze()  # Squeeze for static 1D field

        # NaNs should be preserved
        assert np.sum(np.isnan(stored_data)) == np.sum(np.isnan(occupancy))
        np.testing.assert_array_equal(np.isnan(stored_data), np.isnan(occupancy))

    def test_occupancy_with_zero_values(self, empty_nwb, sample_environment):
        """Test writing occupancy with zero values (briefly visited bins)."""
        from neurospatial.nwb import write_occupancy

        nwbfile = empty_nwb
        env = sample_environment

        # Occupancy with some zero values
        occupancy = np.random.default_rng(42).uniform(0, 10, env.n_bins)
        occupancy[::5] = 0.0  # Every 5th bin has zero occupancy

        write_occupancy(nwbfile, env, occupancy, name="occupancy_with_zeros")

        stored = nwbfile.processing["analysis"]["occupancy_with_zeros"]
        stored_data = stored.data[:].squeeze()

        # Zeros should be preserved
        assert np.sum(stored_data == 0) == np.sum(occupancy == 0)
        np.testing.assert_array_almost_equal(stored_data, occupancy)

    def test_bin_centers_shared_with_place_field(self, empty_nwb, sample_environment):
        """Test that bin_centers are shared between occupancy and place fields."""
        from neurospatial.nwb import write_occupancy, write_place_field

        nwbfile = empty_nwb
        env = sample_environment
        rng = np.random.default_rng(42)

        # Write occupancy first
        occupancy = rng.uniform(0, 10, env.n_bins)
        write_occupancy(nwbfile, env, occupancy, name="occupancy_map")

        # Write place field
        field = rng.uniform(0, 10, env.n_bins)
        write_place_field(nwbfile, env, field, name="cell_001")

        # Count bin_centers datasets - should only be one
        analysis = nwbfile.processing["analysis"]
        bin_centers_count = sum(
            1 for name in analysis.data_interfaces if "bin_centers" in name.lower()
        )

        assert bin_centers_count == 1, (
            f"Expected exactly 1 bin_centers dataset, found {bin_centers_count}"
        )
