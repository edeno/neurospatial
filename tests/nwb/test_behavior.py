"""
Tests for NWB behavior reading functions.

Tests the read_position() and read_head_direction() functions for reading
Position and CompassDirection data from NWB files.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


class TestReadPosition:
    """Tests for read_position() function."""

    def test_basic_position_reading(self, sample_nwb_with_position):
        """Test reading position data from NWB file."""
        from neurospatial.io.nwb import read_position

        position_data = read_position(sample_nwb_with_position)
        positions = position_data.positions
        timestamps = position_data.times

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

        position_data = read_position(sample_nwb_with_position)
        positions = position_data.positions
        timestamps = position_data.times

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
        position_data = read_position(nwbfile, processing_module="tracking")
        positions = position_data.positions

        assert positions.shape == (50, 2)
        np.testing.assert_array_almost_equal(positions, np.ones((50, 2)) * 42.0)

    def test_with_explicit_position_name(
        self, sample_nwb_with_position_multiple_series
    ):
        """Test reading with explicit position_name parameter."""
        from neurospatial.io.nwb import read_position

        # Request specific series by name
        position_data = read_position(
            sample_nwb_with_position_multiple_series,
            position_name="body",
        )
        positions = position_data.positions

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

    def test_multiple_spatial_series_require_a_name(
        self, sample_nwb_with_position_multiple_series
    ):
        """Several series in one Position are refused unless one is named."""
        from neurospatial.io.nwb import read_position

        file = sample_nwb_with_position_multiple_series
        with pytest.raises(ValueError) as caught:
            read_position(file)
        message = str(caught.value)
        assert (
            "Found 2 Position series candidates: ['processing/behavior/Position/body', "
            "'processing/behavior/Position/head']" in message
        )
        assert "position_name='processing/behavior/Position/body'" in message

        head = file.processing["behavior"]["Position"].spatial_series["head"]
        np.testing.assert_array_equal(
            read_position(file, position_name="head").positions, head.data[:]
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

        position_data = read_position(nwbfile)
        positions = position_data.positions
        timestamps = position_data.times

        assert positions.shape == (100, 2)
        assert timestamps.shape == (100,)

    def test_behavior_module_is_not_silently_preferred(self, empty_nwb):
        """A series in another module makes an unqualified read ambiguous."""
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

        with pytest.raises(ValueError, match="Found 2 Position series candidates"):
            read_position(nwbfile)
        positions = read_position(nwbfile, processing_module="behavior").positions
        np.testing.assert_array_equal(positions, np.full((50, 2), 99.0))
        positions = read_position(
            nwbfile, position_name="processing/analysis/Position/pos"
        ).positions
        np.testing.assert_array_equal(positions, np.full((50, 2), 10.0))

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

        position_data = read_position(nwbfile)
        positions = position_data.positions
        timestamps = position_data.times

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

        position_data = read_position(nwbfile)
        positions = position_data.positions
        timestamps = position_data.times

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

        position_data = read_position(nwbfile)
        positions = position_data.positions
        timestamps = position_data.times

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

        position_data = read_position(nwbfile)
        positions = position_data.positions
        timestamps = position_data.times

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

        head_direction_data = read_head_direction(sample_nwb_with_head_direction)
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

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

        head_direction_data = read_head_direction(sample_nwb_with_head_direction)
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

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
        head_direction_data = read_head_direction(nwbfile, processing_module="tracking")
        angles = head_direction_data.headings

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
        head_direction_data = read_head_direction(nwbfile, compass_name="beta")
        angles = head_direction_data.headings

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

    def test_multiple_spatial_series_require_a_name(self, empty_nwb):
        """Several compass series are refused unless one is named."""
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

        with pytest.raises(
            ValueError, match="Found 2 CompassDirection series candidates"
        ):
            read_head_direction(nwbfile)
        angles = read_head_direction(nwbfile, compass_name="alpha").headings
        np.testing.assert_array_equal(angles, np.ones(50))

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

        head_direction_data = read_head_direction(nwbfile)
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

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

        with pytest.raises(
            ValueError, match="Found 2 CompassDirection series candidates"
        ):
            read_head_direction(nwbfile)
        angles = read_head_direction(nwbfile, processing_module="behavior").headings
        np.testing.assert_array_equal(angles, np.full(50, 2.5))

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

        head_direction_data = read_head_direction(nwbfile)
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

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

        head_direction_data = read_head_direction(nwbfile)
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

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

        head_direction_data = read_head_direction(nwbfile)
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

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

        head_direction_data = read_head_direction(
            sample_nwb_with_head_direction_degrees
        )
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

        np.testing.assert_allclose(angles, [0.0, np.pi / 2, np.pi], atol=1e-9)
        assert angles.shape == timestamps.shape

    def test_read_head_direction_vector_form(
        self, sample_nwb_with_head_direction_vectors
    ):
        """An (n, 2) unit-vector series yields n angles via arctan2, not 2n."""
        from neurospatial.io.nwb import read_head_direction

        head_direction_data = read_head_direction(
            sample_nwb_with_head_direction_vectors
        )
        angles = head_direction_data.headings
        timestamps = head_direction_data.times

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

        head_direction_data = read_head_direction(sample_nwb_with_head_direction)
        angles = head_direction_data.headings

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


class TestReadScaledSeries:
    """Readers return values in the series' unit: stored * conversion + offset."""

    def test_read_position_applies_conversion_and_offset(
        self, make_scaled_position_nwb
    ):
        from neurospatial.io.nwb import read_position

        position_data = read_position(make_scaled_position_nwb())
        positions = position_data.positions

        np.testing.assert_allclose(positions.min(axis=0), [0.1, 0.1], rtol=1e-12)
        np.testing.assert_allclose(positions.max(axis=0), [1.1, 1.1], rtol=1e-12)

    def test_read_position_lazy_refuses_scaled_series(self, make_scaled_position_nwb):
        from neurospatial.io.nwb import read_position

        with pytest.raises(ValueError, match=r"conversion=0\.002") as excinfo:
            read_position(make_scaled_position_nwb(), lazy=True)
        assert "Fix:" in str(excinfo.value)

        nwbfile = make_scaled_position_nwb(conversion=1.0, offset=0.0)
        position_data = read_position(nwbfile, lazy=True)
        positions = position_data.positions
        series = nwbfile.processing["behavior"]["Position"].spatial_series["position"]
        np.testing.assert_array_equal(positions, series.data)

    def test_read_head_direction_applies_conversion(self, empty_nwb):
        from pynwb.behavior import CompassDirection, SpatialSeries

        from neurospatial.io.nwb import read_head_direction

        stored = np.array([0.0, 22.5, 45.0, 90.0])  # degrees / 2
        compass = CompassDirection(name="CompassDirection")
        compass.add_spatial_series(
            SpatialSeries(
                name="head_direction",
                data=stored,
                timestamps=np.arange(4) / 30.0,
                reference_frame="0 = East, increasing counterclockwise",
                unit="degrees",
                conversion=2.0,
            )
        )
        empty_nwb.create_processing_module(
            name="behavior", description="Behavior data"
        ).add(compass)

        head_direction_data = read_head_direction(empty_nwb)
        angles = head_direction_data.headings

        np.testing.assert_allclose(angles, np.deg2rad(2 * stored), rtol=1e-12)


def _add_compass(nwbfile, data, **series_kwargs):
    from pynwb.behavior import CompassDirection, SpatialSeries

    compass = CompassDirection(name="CompassDirection")
    compass.add_spatial_series(
        SpatialSeries(
            name="head_direction",
            data=data,
            timestamps=np.arange(len(data)) / 30.0,
            **series_kwargs,
        )
    )
    nwbfile.create_processing_module(name="behavior", description="Behavior").add(
        compass
    )


class TestReadHeadDirectionUnits:
    """Angles are converted only from recognized radian or degree units."""

    @pytest.mark.parametrize("unit", [None, "meters", "", "arbitrary"])
    def test_unrecognized_angle_unit_is_refused(self, empty_nwb, unit):
        """Degrees saved with pynwb's default unit are not read as radians."""
        from neurospatial.io.nwb import read_head_direction

        kwargs = {"reference_frame": "0 = East"}
        if unit is not None:
            kwargs["unit"] = unit
        _add_compass(empty_nwb, np.array([0.0, 90.0, 180.0]), **kwargs)
        with pytest.raises(ValueError) as caught:
            read_head_direction(empty_nwb)
        message = str(caught.value)
        assert "is not an angle unit" in message
        assert "unit='radians' or unit='degrees'" in message

    @pytest.mark.parametrize(
        ("unit", "scale"),
        [
            ("radians", 1.0),
            (" Rad ", 1.0),
            ("radian", 1.0),
            ("degrees", np.pi / 180),
            ("Deg", np.pi / 180),
            ("degree", np.pi / 180),
        ],
    )
    def test_recognized_angle_units(self, empty_nwb, unit, scale):
        from neurospatial.io.nwb import read_head_direction

        stored = np.array([0.0, 0.5, 1.0, 1.5])
        _add_compass(empty_nwb, stored, unit=unit, reference_frame="0 = East")
        np.testing.assert_allclose(
            read_head_direction(empty_nwb).headings, stored * scale, rtol=1e-12
        )

    def test_unit_vectors_do_not_need_an_angle_unit(self, empty_nwb):
        from neurospatial.io.nwb import read_head_direction

        angles = np.array([0.0, np.pi / 2, -np.pi / 2])
        vectors = 3.0 * np.column_stack([np.cos(angles), np.sin(angles)])
        _add_compass(empty_nwb, vectors, reference_frame="0 = East")
        np.testing.assert_allclose(
            read_head_direction(empty_nwb).headings, angles, atol=1e-12
        )

    def test_three_column_heading_is_refused(self, empty_nwb):
        from neurospatial.io.nwb import read_head_direction

        _add_compass(empty_nwb, np.zeros((4, 3)), unit="radians", reference_frame="x")
        with pytest.raises(ValueError, match=r"shape \(4, 3\)"):
            read_head_direction(empty_nwb)

    def test_reference_frame_is_carried_not_interpreted(self, empty_nwb, tmp_path):
        """A compass frame is returned verbatim with an eager file-backed read."""
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_head_direction

        frame = "0 = North, increasing clockwise"
        stored = np.array([0.0, 90.0, np.nan, 270.0])
        _add_compass(empty_nwb, stored, unit="degrees", reference_frame=frame)
        path = tmp_path / "heading.nwb"
        with NWBHDF5IO(str(path), "w") as io:
            io.write(empty_nwb)
        with NWBHDF5IO(str(path), "r") as io:
            hd = read_head_direction(io.read())
        assert hd.frame == frame
        np.testing.assert_allclose(hd.headings, np.deg2rad(stored), rtol=1e-12)
        assert np.isnan(hd.headings[2])


def test_zero_heading_vector_has_no_direction(empty_nwb):
    """A (0, 0) direction vector is missing, not East."""
    from neurospatial.io.nwb import read_head_direction

    _add_compass(
        empty_nwb, np.array([[0.0, 1.0], [0.0, 0.0]]), reference_frame="0 = East"
    )
    headings = read_head_direction(empty_nwb).headings
    assert headings[0] == pytest.approx(np.pi / 2)
    assert np.isnan(headings[1])


def test_two_position_containers_in_one_module_are_refused(empty_nwb):
    """Raw pixels and calibrated cm side by side: neither is picked silently."""
    from pynwb.behavior import Position, SpatialSeries

    from neurospatial.io.nwb import read_position

    module = empty_nwb.create_processing_module(name="behavior", description="b")
    for container, unit, value in (
        ("PositionCm", "cm", 2.0),
        ("PositionPx", "px", 50.0),
    ):
        position = Position(name=container)
        position.add_spatial_series(
            SpatialSeries(
                name="position",
                data=np.full((5, 2), value),
                timestamps=np.arange(5) / 30.0,
                reference_frame="arena corner",
                unit=unit,
            )
        )
        module.add(position)
    for kwargs in (
        {},
        {"processing_module": "behavior"},
        {"position_name": "position"},
    ):
        with pytest.raises(ValueError) as caught:
            read_position(empty_nwb, **kwargs)
        assert (
            "Found 2 Position series candidates: ['processing/behavior/PositionCm/position', "
            "'processing/behavior/PositionPx/position']" in str(caught.value)
        )
    pos = read_position(
        empty_nwb, position_name="processing/behavior/PositionCm/position"
    )
    assert pos.units == "cm"
    np.testing.assert_array_equal(pos.positions, np.full((5, 2), 2.0))


def test_missing_position_error_names_every_searched_location(empty_nwb):
    from neurospatial.io.nwb import read_position

    with pytest.raises(KeyError, match="Searched: processing/\\* and acquisition"):
        read_position(empty_nwb)
