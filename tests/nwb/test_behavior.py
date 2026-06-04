"""
Tests for NWB behavior reading functions.

Tests the read_position() and read_head_direction() functions for reading
Position and CompassDirection data from NWB files.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


class TestReadPosition:
    """Tests for read_position() function."""

    def test_basic_position_reading(self, sample_nwb_with_position):
        """Test reading position data from NWB file."""
        from neurospatial.io.nwb import read_position

        positions, timestamps = read_position(sample_nwb_with_position)

        # Check shapes
        assert positions.shape == (1000, 2)
        assert timestamps.shape == (1000,)

        # Check dtypes
        assert positions.dtype == np.float64
        assert timestamps.dtype == np.float64

        # Check data is valid (not all zeros, reasonable range)
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(timestamps))
        assert timestamps[0] == 0.0
        assert timestamps[-1] > 0.0

    def test_position_data_matches_original(self, sample_nwb_with_position):
        """Test that read data matches the original data in the NWB file."""
        from neurospatial.io.nwb import read_position

        positions, timestamps = read_position(sample_nwb_with_position)

        # Get original data directly from NWB
        original_position = sample_nwb_with_position.processing["behavior"]["Position"]
        original_series = original_position.spatial_series["position"]

        np.testing.assert_array_almost_equal(positions, original_series.data[:])
        np.testing.assert_array_almost_equal(timestamps, original_series.timestamps[:])

    def test_with_explicit_processing_module(self, empty_nwb):
        """Test reading with explicit processing_module parameter."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        nwbfile = empty_nwb

        # Create Position in a custom module
        custom_module = nwbfile.create_processing_module(
            name="tracking", description="Tracking data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((50, 2)) * 42.0,
                timestamps=np.arange(50) / 10.0,
                reference_frame="test",
                unit="cm",
            )
        )
        custom_module.add(position)

        # Read with explicit module
        positions, _timestamps = read_position(nwbfile, processing_module="tracking")

        assert positions.shape == (50, 2)
        np.testing.assert_array_almost_equal(positions, np.ones((50, 2)) * 42.0)

    def test_with_explicit_position_name(
        self, sample_nwb_with_position_multiple_series
    ):
        """Test reading with explicit position_name parameter."""
        from neurospatial.io.nwb import read_position

        # Request specific series by name
        positions, _timestamps = read_position(
            sample_nwb_with_position_multiple_series,
            position_name="body",
        )

        # Check we got the 'body' series (500 samples)
        assert positions.shape == (500, 2)

        # Verify it's the body data (not head)
        original_position = sample_nwb_with_position_multiple_series.processing[
            "behavior"
        ]["Position"]
        original_body = original_position.spatial_series["body"]
        np.testing.assert_array_almost_equal(positions, original_body.data[:])

    def test_error_when_no_position_found(self, empty_nwb):
        """Test KeyError when no Position container found."""
        from neurospatial.io.nwb import read_position

        with pytest.raises(KeyError, match=r"No Position.*found"):
            read_position(empty_nwb)

    def test_error_when_named_position_not_found(
        self, sample_nwb_with_position_multiple_series
    ):
        """Test KeyError with available list when specific position_name not found."""
        from neurospatial.io.nwb import read_position

        with pytest.raises(KeyError, match=r"nonexistent.*not found.*Available"):
            read_position(
                sample_nwb_with_position_multiple_series,
                position_name="nonexistent",
            )

    def test_error_when_processing_module_not_found(self, sample_nwb_with_position):
        """Test KeyError when specified processing_module not found."""
        from neurospatial.io.nwb import read_position

        with pytest.raises(KeyError, match=r"Processing module.*not found"):
            read_position(sample_nwb_with_position, processing_module="nonexistent")

    def test_multiple_spatial_series_uses_first_alphabetically(
        self, sample_nwb_with_position_multiple_series, caplog
    ):
        """Test that multiple SpatialSeries uses first alphabetically with INFO log."""
        from neurospatial.io.nwb import read_position

        # Enable logging capture
        with caplog.at_level(logging.INFO, logger="neurospatial.nwb"):
            positions, _timestamps = read_position(
                sample_nwb_with_position_multiple_series
            )

        # Should use 'body' (first alphabetically)
        assert positions.shape == (500, 2)

        # Verify it's the body data
        original_position = sample_nwb_with_position_multiple_series.processing[
            "behavior"
        ]["Position"]
        original_body = original_position.spatial_series["body"]
        np.testing.assert_array_almost_equal(positions, original_body.data[:])

        # Check INFO log message
        assert any(
            "Multiple SpatialSeries found" in record.message
            or ("Using" in record.message and "body" in record.message)
            for record in caplog.records
        )

    def test_position_in_acquisition(self, empty_nwb):
        """Test reading Position from acquisition when not in processing."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        rng = np.random.default_rng(42)
        nwbfile = empty_nwb

        # Add Position to acquisition (not processing)
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=rng.random((100, 2)),
                timestamps=np.arange(100) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        nwbfile.add_acquisition(position)

        positions, timestamps = read_position(nwbfile)

        assert positions.shape == (100, 2)
        assert timestamps.shape == (100,)

    def test_prioritizes_behavior_module_over_others(self, empty_nwb):
        """Test that processing/behavior is prioritized over other modules."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        nwbfile = empty_nwb

        # Add Position to analysis module first
        analysis_module = nwbfile.create_processing_module(
            name="analysis", description="Analysis data"
        )
        position_analysis = Position(name="Position")
        position_analysis.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((50, 2)) * 10.0,  # Distinctive value
                timestamps=np.arange(50) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        analysis_module.add(position_analysis)

        # Add Position to behavior module second
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        position_behavior = Position(name="Position")
        position_behavior.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((50, 2)) * 99.0,  # Different distinctive value
                timestamps=np.arange(50) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        behavior_module.add(position_behavior)

        positions, _timestamps = read_position(nwbfile)

        # Should get behavior module Position (value 99.0), not analysis (value 10.0)
        np.testing.assert_array_almost_equal(positions, np.ones((50, 2)) * 99.0)

    def test_1d_position_data(self, empty_nwb):
        """Test reading 1D position data (single dimension)."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.arange(100).reshape(-1, 1).astype(float),  # 1D data
                timestamps=np.arange(100) / 30.0,
                reference_frame="linear track",
                unit="cm",
            )
        )
        behavior_module.add(position)

        positions, timestamps = read_position(nwbfile)

        assert positions.shape == (100, 1)
        assert timestamps.shape == (100,)

    def test_3d_position_data(self, empty_nwb):
        """Test reading 3D position data."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        rng = np.random.default_rng(42)
        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=rng.random((100, 3)),  # 3D data
                timestamps=np.arange(100) / 30.0,
                reference_frame="room coordinates",
                unit="cm",
            )
        )
        behavior_module.add(position)

        positions, timestamps = read_position(nwbfile)

        assert positions.shape == (100, 3)
        assert timestamps.shape == (100,)

    def test_uses_rate_when_timestamps_not_available(self, empty_nwb):
        """Test that timestamps are computed from rate when explicit timestamps not provided."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        rng = np.random.default_rng(42)
        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=rng.random((100, 2)),
                rate=30.0,  # 30 Hz, no explicit timestamps
                starting_time=0.0,
                reference_frame="test",
                unit="cm",
            )
        )
        behavior_module.add(position)

        positions, timestamps = read_position(nwbfile)

        assert positions.shape == (100, 2)
        assert timestamps.shape == (100,)
        # Check computed timestamps are correct
        expected_timestamps = np.arange(100) / 30.0
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps)

    def test_uses_rate_with_nonzero_starting_time(self, empty_nwb):
        """Test that starting_time offset is applied correctly."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        rng = np.random.default_rng(42)
        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        position = Position(name="Position")
        position.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=rng.random((100, 2)),
                rate=30.0,  # 30 Hz
                starting_time=10.5,  # Start at 10.5 seconds
                reference_frame="test",
                unit="cm",
            )
        )
        behavior_module.add(position)

        positions, timestamps = read_position(nwbfile)

        assert positions.shape == (100, 2)
        assert timestamps.shape == (100,)
        # Check computed timestamps include offset
        expected_timestamps = np.arange(100) / 30.0 + 10.5
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps)

    def test_error_when_position_container_empty(self, empty_nwb):
        """Test KeyError when Position container exists but has no SpatialSeries."""
        from pynwb.behavior import Position

        from neurospatial.io.nwb import read_position

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        # Create empty Position (no spatial series added)
        position = Position(name="Position")
        behavior_module.add(position)

        with pytest.raises(KeyError, match="has no SpatialSeries"):
            read_position(nwbfile)


class TestReadHeadDirection:
    """Tests for read_head_direction() function."""

    def test_basic_head_direction_reading(self, sample_nwb_with_head_direction):
        """Test reading head direction data from NWB file."""
        from neurospatial.io.nwb import read_head_direction

        angles, timestamps = read_head_direction(sample_nwb_with_head_direction)

        # Check shapes
        assert angles.shape == (1000,)
        assert timestamps.shape == (1000,)

        # Check dtypes
        assert angles.dtype == np.float64
        assert timestamps.dtype == np.float64

        # Check data is valid (angles in [0, 2*pi])
        assert np.all(np.isfinite(angles))
        assert np.all(np.isfinite(timestamps))
        assert np.all(angles >= 0)
        assert np.all(angles <= 2 * np.pi)
        assert timestamps[0] == 0.0
        assert timestamps[-1] > 0.0

    def test_head_direction_data_matches_original(self, sample_nwb_with_head_direction):
        """Test that read data matches the original data in the NWB file."""
        from neurospatial.io.nwb import read_head_direction

        angles, timestamps = read_head_direction(sample_nwb_with_head_direction)

        # Get original data directly from NWB
        original_compass = sample_nwb_with_head_direction.processing["behavior"][
            "CompassDirection"
        ]
        original_series = original_compass.spatial_series["head_direction"]

        np.testing.assert_array_almost_equal(angles, original_series.data[:])
        np.testing.assert_array_almost_equal(timestamps, original_series.timestamps[:])

    def test_with_explicit_processing_module(self, empty_nwb):
        """Test reading with explicit processing_module parameter."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        nwbfile = empty_nwb

        # Create CompassDirection in a custom module
        custom_module = nwbfile.create_processing_module(
            name="tracking", description="Tracking data"
        )
        compass = CompassDirection(name="CompassDirection")
        compass.add_spatial_series(
            SpatialSeries(
                name="heading",
                data=np.ones(50) * 1.5,  # Constant angle
                timestamps=np.arange(50) / 10.0,
                reference_frame="test",
                unit="radians",
            )
        )
        custom_module.add(compass)

        # Read with explicit module
        angles, _timestamps = read_head_direction(nwbfile, processing_module="tracking")

        assert angles.shape == (50,)
        np.testing.assert_array_almost_equal(angles, np.ones(50) * 1.5)

    def test_with_explicit_compass_name(self, empty_nwb):
        """Test reading with explicit compass_name parameter."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass = CompassDirection(name="CompassDirection")

        # Add two series with different names
        compass.add_spatial_series(
            SpatialSeries(
                name="alpha",
                data=np.ones(100) * 0.5,
                timestamps=np.arange(100) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        compass.add_spatial_series(
            SpatialSeries(
                name="beta",
                data=np.ones(200) * 2.0,
                timestamps=np.arange(200) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass)

        # Request specific series by name
        angles, _timestamps = read_head_direction(nwbfile, compass_name="beta")

        # Check we got the 'beta' series (200 samples, value 2.0)
        assert angles.shape == (200,)
        np.testing.assert_array_almost_equal(angles, np.ones(200) * 2.0)

    def test_error_when_no_compass_direction_found(self, empty_nwb):
        """Test KeyError when no CompassDirection container found."""
        from neurospatial.io.nwb import read_head_direction

        with pytest.raises(KeyError, match=r"No CompassDirection.*found"):
            read_head_direction(empty_nwb)

    def test_error_when_named_compass_not_found(self, empty_nwb):
        """Test KeyError with available list when specific compass_name not found."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass = CompassDirection(name="CompassDirection")
        compass.add_spatial_series(
            SpatialSeries(
                name="existing",
                data=np.ones(50),
                timestamps=np.arange(50) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass)

        with pytest.raises(KeyError, match=r"nonexistent.*not found.*Available"):
            read_head_direction(nwbfile, compass_name="nonexistent")

    def test_error_when_processing_module_not_found(
        self, sample_nwb_with_head_direction
    ):
        """Test KeyError when specified processing_module not found."""
        from neurospatial.io.nwb import read_head_direction

        with pytest.raises(KeyError, match=r"Processing module.*not found"):
            read_head_direction(
                sample_nwb_with_head_direction, processing_module="nonexistent"
            )

    def test_multiple_spatial_series_uses_first_alphabetically(self, empty_nwb, caplog):
        """Test that multiple SpatialSeries uses first alphabetically with INFO log."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass = CompassDirection(name="CompassDirection")

        # Add 'zebra' first (alphabetically last)
        compass.add_spatial_series(
            SpatialSeries(
                name="zebra",
                data=np.ones(100) * 3.0,
                timestamps=np.arange(100) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        # Add 'alpha' second (alphabetically first)
        compass.add_spatial_series(
            SpatialSeries(
                name="alpha",
                data=np.ones(50) * 1.0,
                timestamps=np.arange(50) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass)

        # Enable logging capture
        with caplog.at_level(logging.INFO, logger="neurospatial.nwb"):
            angles, _timestamps = read_head_direction(nwbfile)

        # Should use 'alpha' (first alphabetically)
        assert angles.shape == (50,)
        np.testing.assert_array_almost_equal(angles, np.ones(50) * 1.0)

        # Check INFO log message
        assert any(
            "Multiple SpatialSeries found" in record.message
            or ("Using" in record.message and "alpha" in record.message)
            for record in caplog.records
        )

    def test_compass_direction_in_acquisition(self, empty_nwb):
        """Test reading CompassDirection from acquisition when not in processing."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        rng = np.random.default_rng(42)
        nwbfile = empty_nwb

        # Add CompassDirection to acquisition (not processing)
        compass = CompassDirection(name="CompassDirection")
        compass.add_spatial_series(
            SpatialSeries(
                name="heading",
                data=rng.random(100) * 2 * np.pi,
                timestamps=np.arange(100) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        nwbfile.add_acquisition(compass)

        angles, timestamps = read_head_direction(nwbfile)

        assert angles.shape == (100,)
        assert timestamps.shape == (100,)

    def test_prioritizes_behavior_module_over_others(self, empty_nwb):
        """Test that processing/behavior is prioritized over other modules."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        nwbfile = empty_nwb

        # Add CompassDirection to analysis module first
        analysis_module = nwbfile.create_processing_module(
            name="analysis", description="Analysis data"
        )
        compass_analysis = CompassDirection(name="CompassDirection")
        compass_analysis.add_spatial_series(
            SpatialSeries(
                name="heading",
                data=np.ones(50) * 0.5,  # Distinctive value
                timestamps=np.arange(50) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        analysis_module.add(compass_analysis)

        # Add CompassDirection to behavior module second
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass_behavior = CompassDirection(name="CompassDirection")
        compass_behavior.add_spatial_series(
            SpatialSeries(
                name="heading",
                data=np.ones(50) * 2.5,  # Different distinctive value
                timestamps=np.arange(50) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass_behavior)

        angles, _timestamps = read_head_direction(nwbfile)

        # Should get behavior module (value 2.5), not analysis (value 0.5)
        np.testing.assert_array_almost_equal(angles, np.ones(50) * 2.5)

    def test_uses_rate_when_timestamps_not_available(self, empty_nwb):
        """Test that timestamps are computed from rate when explicit timestamps not provided."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        rng = np.random.default_rng(42)
        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass = CompassDirection(name="CompassDirection")
        compass.add_spatial_series(
            SpatialSeries(
                name="heading",
                data=rng.random(100) * 2 * np.pi,
                rate=30.0,  # 30 Hz, no explicit timestamps
                starting_time=0.0,
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass)

        angles, timestamps = read_head_direction(nwbfile)

        assert angles.shape == (100,)
        assert timestamps.shape == (100,)
        # Check computed timestamps are correct
        expected_timestamps = np.arange(100) / 30.0
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps)

    def test_uses_rate_with_nonzero_starting_time(self, empty_nwb):
        """Test that starting_time offset is applied correctly."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        rng = np.random.default_rng(42)
        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass = CompassDirection(name="CompassDirection")
        compass.add_spatial_series(
            SpatialSeries(
                name="heading",
                data=rng.random(100) * 2 * np.pi,
                rate=30.0,  # 30 Hz
                starting_time=10.5,  # Start at 10.5 seconds
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass)

        angles, timestamps = read_head_direction(nwbfile)

        assert angles.shape == (100,)
        assert timestamps.shape == (100,)
        # Check computed timestamps include offset
        expected_timestamps = np.arange(100) / 30.0 + 10.5
        np.testing.assert_array_almost_equal(timestamps, expected_timestamps)

    def test_error_when_compass_container_empty(self, empty_nwb):
        """Test KeyError when CompassDirection container exists but has no SpatialSeries."""
        from pynwb.behavior import CompassDirection

        from neurospatial.io.nwb import read_head_direction

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        # Create empty CompassDirection (no spatial series added)
        compass = CompassDirection(name="CompassDirection")
        behavior_module.add(compass)

        with pytest.raises(KeyError, match="has no SpatialSeries"):
            read_head_direction(nwbfile)

    def test_head_direction_from_column_vector(self, empty_nwb):
        """Test reading head direction stored as 2D column vector (n, 1)."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass = CompassDirection(name="CompassDirection")
        # Store as column vector (100, 1) instead of 1D (100,)
        compass.add_spatial_series(
            SpatialSeries(
                name="heading",
                data=np.arange(100).reshape(-1, 1).astype(float) * 0.01,
                timestamps=np.arange(100) / 30.0,
                reference_frame="test",
                unit="radians",
            )
        )
        behavior_module.add(compass)

        angles, timestamps = read_head_direction(nwbfile)

        # Should be flattened to 1D
        assert angles.shape == (100,)
        assert angles.ndim == 1
        assert timestamps.shape == (100,)


class TestReadHeadDirectionUnitsAndVectors:
    """Unit handling, (n, 2) vector form, convention, and length checks."""

    def test_read_head_direction_degrees_to_radians(
        self, sample_nwb_with_head_direction_degrees
    ):
        """A degree-unit series is converted to radians on read."""
        from neurospatial.io.nwb import read_head_direction

        angles, timestamps = read_head_direction(sample_nwb_with_head_direction_degrees)

        np.testing.assert_allclose(angles, [0.0, np.pi / 2, np.pi], atol=1e-9)
        assert angles.shape == timestamps.shape

    def test_read_head_direction_vector_form(
        self, sample_nwb_with_head_direction_vectors
    ):
        """An (n, 2) unit-vector series yields n angles via arctan2, not 2n."""
        from neurospatial.io.nwb import read_head_direction

        angles, timestamps = read_head_direction(sample_nwb_with_head_direction_vectors)

        n_samples = 100
        assert angles.shape == (n_samples,)
        assert timestamps.shape == (n_samples,)

        expected = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        # arctan2 returns (-pi, pi]; compare via complex phase to avoid wrap.
        np.testing.assert_allclose(
            np.exp(1j * angles), np.exp(1j * expected), atol=1e-9
        )

    def test_read_head_direction_radians_passthrough(
        self, sample_nwb_with_head_direction
    ):
        """A radians 1-D series is returned unchanged (no regression)."""
        from neurospatial.io.nwb import read_head_direction

        angles, _ = read_head_direction(sample_nwb_with_head_direction)

        original = sample_nwb_with_head_direction.processing["behavior"][
            "CompassDirection"
        ].spatial_series["head_direction"]
        np.testing.assert_array_almost_equal(angles, original.data[:])

    def test_read_head_direction_length_check(self, empty_nwb):
        """A data/timestamps length mismatch raises ValueError."""
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        behavior_module = empty_nwb.create_processing_module(
            name="behavior", description="Behavior data"
        )
        compass = CompassDirection(name="CompassDirection")
        series = SpatialSeries(
            name="heading",
            data=np.zeros(10),
            timestamps=np.arange(10) / 30.0,
            reference_frame="test",
            unit="radians",
        )
        compass.add_spatial_series(series)
        behavior_module.add(compass)

        # pynwb validates data/timestamps agreement at construction time, so
        # simulate a corrupt/externally-written series by shrinking data after
        # the fact. The reader's own length guard must still catch this.
        series.fields["data"] = np.zeros(8)

        with pytest.raises(ValueError, match="Length mismatch"):
            read_head_direction(empty_nwb)


class TestReadPositionLengthCheck:
    """read_position must reject mismatched data/timestamp lengths."""

    def test_read_position_length_check(self, empty_nwb):
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.io.nwb import read_position

        behavior_module = empty_nwb.create_processing_module(
            name="behavior", description="Behavior data"
        )
        position = Position(name="Position")
        series = SpatialSeries(
            name="position",
            data=np.zeros((10, 2)),
            timestamps=np.arange(10) / 30.0,
            reference_frame="test",
            unit="cm",
        )
        position.add_spatial_series(series)
        behavior_module.add(position)

        # pynwb validates data/timestamps agreement at construction time, so
        # simulate a corrupt/externally-written series by shrinking data after
        # the fact. The reader's own length guard must still catch this.
        series.fields["data"] = np.zeros((8, 2))

        with pytest.raises(ValueError, match="Length mismatch"):
            read_position(empty_nwb)
