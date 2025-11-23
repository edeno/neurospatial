"""
Tests for NWB environment factory functions.

Tests for environment_from_position() which creates an Environment from
NWB Position data.
"""

from __future__ import annotations

import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


class TestEnvironmentFromPosition:
    """Tests for environment_from_position() function."""

    def test_basic_environment_creation(self, sample_nwb_with_position):
        """Test basic Environment creation from Position data."""
        from neurospatial.nwb import environment_from_position

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should create a valid environment
        assert env is not None
        assert env.n_bins > 0
        assert env.bin_centers.shape[1] == 2  # 2D environment

    def test_environment_matches_position_data_bounds(self, sample_nwb_with_position):
        """Test Environment extent matches Position data bounds."""
        from neurospatial.nwb import environment_from_position, read_position

        # Get position data for comparison
        positions, _ = read_position(sample_nwb_with_position)

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Environment should cover the position data extent
        # bin_centers should fall within the position data range
        pos_min = positions.min(axis=0)
        pos_max = positions.max(axis=0)

        # All bin centers should be near the position data range
        for dim in range(2):
            assert env.bin_centers[:, dim].min() >= pos_min[dim] - 5.0
            assert env.bin_centers[:, dim].max() <= pos_max[dim] + 5.0

    def test_units_parameter_propagation(self, sample_nwb_with_position):
        """Test units parameter is set on Environment."""
        from neurospatial.nwb import environment_from_position

        env = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, units="cm"
        )

        assert env.units == "cm"

    def test_units_defaults_from_spatial_series(self, sample_nwb_with_position):
        """Test units are auto-detected from SpatialSeries when not specified."""
        from neurospatial.nwb import environment_from_position

        # The fixture has unit="cm" in the SpatialSeries
        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should auto-detect units from SpatialSeries
        assert env.units == "cm"

    def test_frame_parameter_propagation(self, sample_nwb_with_position):
        """Test frame parameter is set on Environment."""
        from neurospatial.nwb import environment_from_position

        env = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, frame="session_001"
        )

        assert env.frame == "session_001"

    def test_infer_active_bins_parameter(self, sample_nwb_with_position):
        """Test infer_active_bins parameter is forwarded."""
        from neurospatial.nwb import environment_from_position

        # With infer_active_bins=True, should only include visited bins
        env_active = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, infer_active_bins=True
        )

        # With infer_active_bins=False (default), should include all bins in extent
        env_all = environment_from_position(
            sample_nwb_with_position, bin_size=5.0, infer_active_bins=False
        )

        # Active bins environment should have same or fewer bins
        assert env_active.n_bins <= env_all.n_bins

    def test_kwargs_forwarded_to_from_samples(self, sample_nwb_with_position):
        """Test additional kwargs are forwarded to Environment.from_samples()."""
        from neurospatial.nwb import environment_from_position

        # Test bin_count_threshold parameter
        env = environment_from_position(
            sample_nwb_with_position,
            bin_size=5.0,
            infer_active_bins=True,
            bin_count_threshold=5,  # Require at least 5 samples per bin
        )

        # Should create environment with fewer bins due to threshold
        assert env is not None
        assert env.n_bins > 0

    def test_processing_module_parameter(self, sample_nwb_with_position):
        """Test processing_module parameter is forwarded to read_position."""
        from neurospatial.nwb import environment_from_position

        # Should work when specifying the correct module
        env = environment_from_position(
            sample_nwb_with_position,
            bin_size=5.0,
            processing_module="behavior",
        )

        assert env is not None
        assert env.n_bins > 0

    def test_position_name_parameter(self, sample_nwb_with_position_multiple_series):
        """Test position_name parameter selects specific SpatialSeries."""
        from neurospatial.nwb import environment_from_position

        # Select 'head' position (one of the two available)
        env = environment_from_position(
            sample_nwb_with_position_multiple_series,
            bin_size=5.0,
            position_name="head",
        )

        assert env is not None
        assert env.n_bins > 0

    def test_error_when_position_not_found(self, empty_nwb):
        """Test KeyError when Position not found in NWB file."""
        from neurospatial.nwb import environment_from_position

        with pytest.raises(KeyError, match="No Position data found"):
            environment_from_position(empty_nwb, bin_size=5.0)

    def test_error_when_processing_module_not_found(self, sample_nwb_with_position):
        """Test KeyError when specified processing module not found."""
        from neurospatial.nwb import environment_from_position

        with pytest.raises(KeyError, match="Processing module 'nonexistent' not found"):
            environment_from_position(
                sample_nwb_with_position,
                bin_size=5.0,
                processing_module="nonexistent",
            )

    def test_bin_size_required(self, sample_nwb_with_position):
        """Test that bin_size parameter is required."""
        from neurospatial.nwb import environment_from_position

        with pytest.raises(TypeError):
            environment_from_position(sample_nwb_with_position)  # Missing bin_size

    def test_different_bin_sizes(self, sample_nwb_with_position):
        """Test Environment creation with different bin sizes."""
        from neurospatial.nwb import environment_from_position

        env_small = environment_from_position(sample_nwb_with_position, bin_size=2.0)
        env_large = environment_from_position(sample_nwb_with_position, bin_size=10.0)

        # Smaller bins should result in more bins
        assert env_small.n_bins > env_large.n_bins

    def test_environment_is_fitted(self, sample_nwb_with_position):
        """Test that returned Environment is fitted and ready to use."""
        from neurospatial.nwb import environment_from_position

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should be able to call methods that require fitted state
        assert hasattr(env, "_is_fitted")
        assert env._is_fitted

        # Should be able to use spatial query methods
        point = env.bin_centers[0]
        bin_idx = env.bin_at(point)
        assert bin_idx >= 0

    def test_connectivity_graph_created(self, sample_nwb_with_position):
        """Test that connectivity graph is properly created."""
        from neurospatial.nwb import environment_from_position

        env = environment_from_position(sample_nwb_with_position, bin_size=5.0)

        # Should have connectivity graph
        assert env.connectivity is not None
        assert env.connectivity.number_of_nodes() == env.n_bins
        assert env.connectivity.number_of_edges() > 0
