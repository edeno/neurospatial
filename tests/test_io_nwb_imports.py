"""
Tests for neurospatial.io.nwb module.

Tests verify that the nwb module is importable from the new io.nwb location.
"""

from __future__ import annotations


class TestNwbModuleImports:
    """Test that nwb module can be imported from io.nwb."""

    def test_import_io_nwb_module(self):
        """Test that io.nwb module is importable."""
        from neurospatial.io import nwb

        assert nwb is not None

    def test_import_read_functions(self):
        """Test that reading functions are importable from io.nwb."""
        from neurospatial.io.nwb import (
            read_environment,
            read_events,
            read_head_direction,
            read_intervals,
            read_pose,
            read_position,
            read_trials,
        )

        # Verify they're callable (lazy imports resolve)
        assert callable(read_position)
        assert callable(read_head_direction)
        assert callable(read_pose)
        assert callable(read_events)
        assert callable(read_intervals)
        assert callable(read_environment)
        assert callable(read_trials)

    def test_import_write_functions(self):
        """Test that writing functions are importable from io.nwb."""
        from neurospatial.io.nwb import (
            dataframe_to_events_table,
            write_environment,
            write_events,
            write_laps,
            write_occupancy,
            write_place_field,
            write_region_crossings,
            write_trials,
        )

        assert callable(write_place_field)
        assert callable(write_occupancy)
        assert callable(write_events)
        assert callable(write_laps)
        assert callable(write_region_crossings)
        assert callable(dataframe_to_events_table)
        assert callable(write_environment)
        assert callable(write_trials)

    def test_import_factory_functions(self):
        """Test that factory functions are importable from io.nwb."""
        from neurospatial.io.nwb import (
            bodypart_overlay_from_nwb,
            environment_from_position,
            head_direction_overlay_from_nwb,
            position_overlay_from_nwb,
        )

        assert callable(environment_from_position)
        assert callable(position_overlay_from_nwb)
        assert callable(bodypart_overlay_from_nwb)
        assert callable(head_direction_overlay_from_nwb)

    def test_nwb_all_exports(self):
        """Test that __all__ is defined and contains expected exports."""
        from neurospatial.io import nwb

        assert hasattr(nwb, "__all__")
        expected_exports = {
            "read_position",
            "read_head_direction",
            "read_pose",
            "read_events",
            "read_intervals",
            "read_environment",
            "read_trials",
            "write_place_field",
            "write_occupancy",
            "write_events",
            "write_laps",
            "write_region_crossings",
            "dataframe_to_events_table",
            "write_environment",
            "write_trials",
            "environment_from_position",
            "position_overlay_from_nwb",
            "bodypart_overlay_from_nwb",
            "head_direction_overlay_from_nwb",
        }
        assert expected_exports.issubset(set(nwb.__all__))


class TestNwbInternalModules:
    """Test that internal nwb modules are importable from new location."""

    def test_import_core_module(self):
        """Test that _core module is importable."""
        from neurospatial.io.nwb import _core

        assert hasattr(_core, "_require_pynwb")

    def test_import_behavior_module(self):
        """Test that _behavior module is importable."""
        from neurospatial.io.nwb import _behavior

        assert hasattr(_behavior, "read_position")

    def test_import_events_module(self):
        """Test that _events module is importable."""
        from neurospatial.io.nwb import _events

        assert hasattr(_events, "read_events")

    def test_import_environment_module(self):
        """Test that _environment module is importable."""
        from neurospatial.io.nwb import _environment

        assert hasattr(_environment, "read_environment")

    def test_import_fields_module(self):
        """Test that _fields module is importable."""
        from neurospatial.io.nwb import _fields

        assert hasattr(_fields, "write_place_field")

    def test_import_overlays_module(self):
        """Test that _overlays module is importable."""
        from neurospatial.io.nwb import _overlays

        assert hasattr(_overlays, "position_overlay_from_nwb")

    def test_import_pose_module(self):
        """Test that _pose module is importable."""
        from neurospatial.io.nwb import _pose

        assert hasattr(_pose, "read_pose")

    def test_import_adapters_module(self):
        """Test that _adapters module is importable."""
        from neurospatial.io.nwb import _adapters

        assert hasattr(_adapters, "timestamps_from_series")
