"""
Tests for NWB core discovery utilities.

Tests the _find_containers_by_type() function and related discovery utilities.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

# Skip all tests if pynwb is not installed
pynwb = pytest.importorskip("pynwb")


class TestFindContainersByType:
    """Tests for _find_containers_by_type() function."""

    def test_find_position_in_behavior_module(self, sample_nwb_with_position):
        """Test finding Position container in processing/behavior/."""
        from pynwb.behavior import Position

        from neurospatial.nwb._core import _find_containers_by_type

        found = _find_containers_by_type(sample_nwb_with_position, Position)

        assert len(found) == 1
        path, container = found[0]
        assert path == "processing/behavior/Position"
        assert isinstance(container, Position)

    def test_find_compass_direction_in_behavior_module(
        self, sample_nwb_with_head_direction
    ):
        """Test finding CompassDirection container in processing/behavior/."""
        from pynwb.behavior import CompassDirection

        from neurospatial.nwb._core import _find_containers_by_type

        found = _find_containers_by_type(
            sample_nwb_with_head_direction, CompassDirection
        )

        assert len(found) == 1
        path, container = found[0]
        assert path == "processing/behavior/CompassDirection"
        assert isinstance(container, CompassDirection)

    def test_empty_nwb_returns_empty_list(self, empty_nwb):
        """Test that empty NWB file returns empty list."""
        from pynwb.behavior import Position

        from neurospatial.nwb._core import _find_containers_by_type

        found = _find_containers_by_type(empty_nwb, Position)

        assert found == []

    def test_type_not_present_returns_empty_list(self, sample_nwb_with_position):
        """Test that requesting a type not in the file returns empty list."""
        from pynwb.behavior import CompassDirection

        from neurospatial.nwb._core import _find_containers_by_type

        # File has Position but not CompassDirection
        found = _find_containers_by_type(sample_nwb_with_position, CompassDirection)

        assert found == []

    def test_search_order_behavior_first(self, empty_nwb):
        """Test that processing/behavior/ has priority over other modules."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.nwb._core import _find_containers_by_type

        nwbfile = empty_nwb

        # Create Position in a non-behavior module (added first)
        other_module = nwbfile.create_processing_module(
            name="analysis", description="Analysis module"
        )
        position_analysis = Position(name="PositionAnalysis")
        position_analysis.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.zeros((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        other_module.add(position_analysis)

        # Create Position in behavior module (added second)
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior module"
        )
        position_behavior = Position(name="PositionBehavior")
        position_behavior.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        behavior_module.add(position_behavior)

        found = _find_containers_by_type(nwbfile, Position)

        assert len(found) == 2
        # First should be from behavior module
        assert found[0][0] == "processing/behavior/PositionBehavior"
        # Second should be from analysis module
        assert found[1][0] == "processing/analysis/PositionAnalysis"

    def test_search_order_processing_before_acquisition(self, empty_nwb):
        """Test that processing/ has priority over acquisition/."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.nwb._core import _find_containers_by_type

        nwbfile = empty_nwb

        # Add Position to acquisition (raw data location)
        position_acq = Position(name="PositionAcq")
        position_acq.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.zeros((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        nwbfile.add_acquisition(position_acq)

        # Add Position to processing/tracking (non-behavior module)
        tracking_module = nwbfile.create_processing_module(
            name="tracking", description="Tracking module"
        )
        position_tracking = Position(name="PositionTracking")
        position_tracking.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.ones((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        tracking_module.add(position_tracking)

        found = _find_containers_by_type(nwbfile, Position)

        assert len(found) == 2
        # Processing should come first
        assert found[0][0] == "processing/tracking/PositionTracking"
        # Acquisition should come second
        assert found[1][0] == "acquisition/PositionAcq"

    def test_search_order_full_priority(self, empty_nwb):
        """Test full priority order: behavior > other processing > acquisition."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.nwb._core import _find_containers_by_type

        nwbfile = empty_nwb

        # Add in reverse priority order to verify sorting

        # 1. Acquisition (lowest priority)
        position_acq = Position(name="PositionAcq")
        position_acq.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.zeros((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        nwbfile.add_acquisition(position_acq)

        # 2. Processing/zzz (alphabetically last in processing, but before acquisition)
        zzz_module = nwbfile.create_processing_module(
            name="zzz", description="ZZZ module"
        )
        position_zzz = Position(name="PositionZzz")
        position_zzz.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.zeros((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        zzz_module.add(position_zzz)

        # 3. Processing/aaa (alphabetically first in non-behavior processing)
        aaa_module = nwbfile.create_processing_module(
            name="aaa", description="AAA module"
        )
        position_aaa = Position(name="PositionAaa")
        position_aaa.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.zeros((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        aaa_module.add(position_aaa)

        # 4. Processing/behavior (highest priority)
        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior module"
        )
        position_behavior = Position(name="PositionBehavior")
        position_behavior.add_spatial_series(
            SpatialSeries(
                name="pos",
                data=np.zeros((10, 2)),
                timestamps=np.arange(10) / 30.0,
                reference_frame="test",
                unit="cm",
            )
        )
        behavior_module.add(position_behavior)

        found = _find_containers_by_type(nwbfile, Position)

        assert len(found) == 4
        # Verify priority order
        paths = [path for path, _ in found]
        assert paths == [
            "processing/behavior/PositionBehavior",
            "processing/aaa/PositionAaa",
            "processing/zzz/PositionZzz",
            "acquisition/PositionAcq",
        ]

    def test_multiple_containers_same_module(self, empty_nwb):
        """Test finding multiple containers of same type in same module."""
        from pynwb.behavior import Position, SpatialSeries

        from neurospatial.nwb._core import _find_containers_by_type

        nwbfile = empty_nwb

        behavior_module = nwbfile.create_processing_module(
            name="behavior", description="Behavior module"
        )

        # Add multiple Position containers
        for name in ["PositionHead", "PositionBody", "PositionTail"]:
            position = Position(name=name)
            position.add_spatial_series(
                SpatialSeries(
                    name="pos",
                    data=np.zeros((10, 2)),
                    timestamps=np.arange(10) / 30.0,
                    reference_frame="test",
                    unit="cm",
                )
            )
            behavior_module.add(position)

        found = _find_containers_by_type(nwbfile, Position)

        assert len(found) == 3
        # Should be sorted alphabetically within behavior module
        paths = [path for path, _ in found]
        assert paths == [
            "processing/behavior/PositionBody",
            "processing/behavior/PositionHead",
            "processing/behavior/PositionTail",
        ]

    def test_returns_actual_container_objects(self, sample_nwb_with_position):
        """Test that returned containers are the actual NWB objects."""
        from pynwb.behavior import Position

        from neurospatial.nwb._core import _find_containers_by_type

        found = _find_containers_by_type(sample_nwb_with_position, Position)

        _path, container = found[0]
        # Should be the same object, not a copy
        assert container is sample_nwb_with_position.processing["behavior"]["Position"]


class TestRequirePynwb:
    """Tests for _require_pynwb() helper."""

    def test_returns_pynwb_module(self):
        """Test that _require_pynwb returns the pynwb module."""
        from neurospatial.nwb._core import _require_pynwb

        module = _require_pynwb()
        assert module.__name__ == "pynwb"


class TestRequireNdxPose:
    """Tests for _require_ndx_pose() helper."""

    def test_returns_ndx_pose_module(self):
        """Test that _require_ndx_pose returns the ndx_pose module."""
        pytest.importorskip("ndx_pose")

        from neurospatial.nwb._core import _require_ndx_pose

        module = _require_ndx_pose()
        assert module.__name__ == "ndx_pose"


class TestRequireNdxEvents:
    """Tests for _require_ndx_events() helper."""

    def test_returns_ndx_events_module(self):
        """Test that _require_ndx_events returns the ndx_events module."""
        pytest.importorskip("ndx_events")

        from neurospatial.nwb._core import _require_ndx_events

        module = _require_ndx_events()
        assert module.__name__ == "ndx_events"


class TestGetOrCreateProcessingModule:
    """Tests for _get_or_create_processing_module() helper."""

    def test_creates_new_module(self, empty_nwb):
        """Test creating a new processing module."""
        from neurospatial.nwb._core import _get_or_create_processing_module

        module = _get_or_create_processing_module(
            empty_nwb, "test_module", "Test description"
        )

        assert module.name == "test_module"
        assert "test_module" in empty_nwb.processing

    def test_returns_existing_module(self, empty_nwb):
        """Test returning existing processing module."""
        from neurospatial.nwb._core import _get_or_create_processing_module

        # Create module first
        original = empty_nwb.create_processing_module(
            name="existing", description="Original"
        )

        # Should return the same module
        module = _get_or_create_processing_module(
            empty_nwb, "existing", "Different description"
        )

        assert module is original


class TestNwbLogger:
    """Tests for NWB module logger setup."""

    def test_logger_is_properly_named(self):
        """Test that the NWB module logger has the correct name."""
        from neurospatial.nwb._core import logger

        assert logger.name == "neurospatial.nwb"

    def test_logger_is_logging_logger(self):
        """Test that logger is a proper logging.Logger instance."""
        from neurospatial.nwb._core import logger

        assert isinstance(logger, logging.Logger)
