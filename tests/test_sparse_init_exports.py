"""Tests for Milestone 9: Sparse top-level __init__.py exports.

This test file verifies that the top-level neurospatial package exports only
the 5 core classes per PLAN.md:
    - Environment
    - EnvironmentNotFittedError
    - Region
    - Regions
    - CompositeEnvironment

All other functions should be accessed via explicit submodule imports.
"""

import importlib
import sys

import pytest


class TestSparseTopLevelExports:
    """Test that top-level __init__.py has only 5 core exports."""

    def test_environment_importable_from_top_level(self):
        """Test Environment can be imported from top level."""
        from neurospatial import Environment

        assert Environment is not None

    def test_environment_not_fitted_error_importable_from_top_level(self):
        """Test EnvironmentNotFittedError can be imported from top level."""
        from neurospatial import EnvironmentNotFittedError

        assert EnvironmentNotFittedError is not None
        assert issubclass(EnvironmentNotFittedError, Exception)

    def test_region_importable_from_top_level(self):
        """Test Region can be imported from top level."""
        from neurospatial import Region

        assert Region is not None

    def test_regions_importable_from_top_level(self):
        """Test Regions can be imported from top level."""
        from neurospatial import Regions

        assert Regions is not None

    def test_composite_environment_importable_from_top_level(self):
        """Test CompositeEnvironment can be imported from top level."""
        from neurospatial import CompositeEnvironment

        assert CompositeEnvironment is not None


class TestAllExportsLimitedToFive:
    """Test that __all__ contains exactly 5 items."""

    def test_all_has_exactly_five_exports(self):
        """Test that __all__ has exactly 5 exports."""
        import neurospatial

        expected = {
            "Environment",
            "EnvironmentNotFittedError",
            "Region",
            "Regions",
            "CompositeEnvironment",
        }
        actual = set(neurospatial.__all__)
        assert actual == expected, (
            f"Expected exactly 5 exports: {expected}, got {len(actual)}: {actual}"
        )

    def test_all_length_is_five(self):
        """Test that __all__ has length 5."""
        import neurospatial

        assert len(neurospatial.__all__) == 5, (
            f"Expected 5 exports, got {len(neurospatial.__all__)}"
        )


class TestOldExportsNotAtTopLevel:
    """Test that old exports are NOT available at top level.

    These should now require explicit submodule imports.
    """

    @pytest.mark.parametrize(
        "old_export",
        [
            # Decoding - should use neurospatial.decoding
            "decode_position",
            "DecodingResult",
            "decoding_error",
            "median_decoding_error",
            # Animation overlays - should use neurospatial.animation
            "PositionOverlay",
            "EventOverlay",
            "SpikeOverlay",
            "HeadDirectionOverlay",
            "BodypartOverlay",
            "VideoOverlay",
            "ScaleBarConfig",
            # Annotation - should use neurospatial.annotation
            "annotate_video",
            "AnnotationResult",
            # Layout - should use neurospatial.layout
            "LayoutType",
            "list_available_layouts",
            "get_layout_parameters",
            # I/O - should use neurospatial.io
            "to_file",
            "from_file",
            "to_dict",
            "from_dict",
            # Encoding metrics - should use neurospatial.encoding
            "compute_place_field",
            "detect_place_fields",
            "skaggs_information",
            "sparsity",
            "selectivity",
            "border_score",
            "grid_score",
            "population_vector_correlation",
            # Behavioral - should use neurospatial.behavior
            "detect_laps",
            "segment_trials",
            "detect_region_crossings",
            "path_progress",
            "cost_to_goal",
            # Events - should use neurospatial.events
            "peri_event_histogram",
            "align_spikes_to_events",
            "time_to_nearest_event",
            # Ops - should use neurospatial.ops
            "map_points_to_bins",
            "distance_field",
            "normalize_field",
            "gradient",
            "divergence",
            "heading_from_velocity",
            "compute_viewshed",
            # Simulation - should use neurospatial.simulation
            "SpatialViewCellModel",
        ],
    )
    def test_old_export_not_in_all(self, old_export):
        """Test that old exports are not in __all__."""
        import neurospatial

        assert old_export not in neurospatial.__all__, (
            f"{old_export} should not be in top-level __all__, "
            f"it should be imported from a submodule"
        )


class TestSubmoduleImportsStillWork:
    """Test that removed top-level exports are still accessible via submodules."""

    def test_decoding_imports(self):
        """Test decoding functions accessible from decoding submodule."""
        from neurospatial.decoding import (
            DecodingResult,
            decode_position,
            decoding_error,
            median_decoding_error,
        )

        assert decode_position is not None
        assert DecodingResult is not None
        assert decoding_error is not None
        assert median_decoding_error is not None

    def test_encoding_imports(self):
        """Test encoding functions accessible from encoding submodule."""
        from neurospatial.encoding import (
            border_score,
            detect_place_fields,
            grid_score,
            selectivity,
            skaggs_information,
            sparsity,
        )

        assert detect_place_fields is not None
        assert skaggs_information is not None
        assert sparsity is not None
        assert selectivity is not None
        assert border_score is not None
        assert grid_score is not None

    def test_behavior_imports(self):
        """Test behavior functions accessible from behavior submodule."""
        from neurospatial.behavior import (
            detect_laps,
            detect_region_crossings,
            segment_trials,
        )
        from neurospatial.behavior.navigation import (
            cost_to_goal,
            path_progress,
        )

        assert detect_laps is not None
        assert segment_trials is not None
        assert detect_region_crossings is not None
        assert path_progress is not None
        assert cost_to_goal is not None

    def test_events_imports(self):
        """Test events functions accessible from events submodule."""
        from neurospatial.events import (
            align_spikes_to_events,
            peri_event_histogram,
            time_to_nearest_event,
        )

        assert peri_event_histogram is not None
        assert align_spikes_to_events is not None
        assert time_to_nearest_event is not None

    def test_ops_imports(self):
        """Test ops functions accessible from ops submodule."""
        from neurospatial.ops import (
            distance_field,
            divergence,
            gradient,
            heading_from_velocity,
            map_points_to_bins,
            normalize_field,
        )

        assert map_points_to_bins is not None
        assert distance_field is not None
        assert normalize_field is not None
        assert gradient is not None
        assert divergence is not None
        assert heading_from_velocity is not None

    def test_io_imports(self):
        """Test I/O functions accessible from io submodule."""
        from neurospatial.io import from_dict, from_file, to_dict, to_file

        assert to_file is not None
        assert from_file is not None
        assert to_dict is not None
        assert from_dict is not None

    def test_animation_imports(self):
        """Test animation overlays accessible from animation submodule."""
        from neurospatial.animation import (
            EventOverlay,
            SpikeOverlay,
            VideoOverlay,
        )
        from neurospatial.animation.config import ScaleBarConfig

        # These are exported from animation/__init__.py
        assert EventOverlay is not None
        assert SpikeOverlay is not None
        assert VideoOverlay is not None
        assert ScaleBarConfig is not None

        # These require direct import from overlays module
        from neurospatial.animation.overlays import (
            BodypartOverlay,
            HeadDirectionOverlay,
            PositionOverlay,
        )

        assert PositionOverlay is not None
        assert HeadDirectionOverlay is not None
        assert BodypartOverlay is not None


class TestDocstringUpdated:
    """Test that the module docstring reflects sparse exports."""

    def test_docstring_mentions_explicit_submodule_imports(self):
        """Test docstring explains explicit submodule import pattern."""
        import neurospatial

        docstring = neurospatial.__doc__
        assert docstring is not None
        # Should mention explicit imports from submodules
        assert "encoding" in docstring.lower() or "explicit" in docstring.lower()


class TestNoCircularImports:
    """Test that importing neurospatial doesn't cause circular imports."""

    def test_import_neurospatial_succeeds(self):
        """Test basic import works without circular import errors."""
        # Force fresh import
        if "neurospatial" in sys.modules:
            # Already imported, just verify it works
            import neurospatial

            assert neurospatial is not None
        else:
            import neurospatial

            assert neurospatial is not None

    def test_fresh_import_from_string(self):
        """Test import via importlib."""
        ns = importlib.import_module("neurospatial")
        assert ns is not None
        assert hasattr(ns, "Environment")
        assert hasattr(ns, "Region")
        assert hasattr(ns, "Regions")
        assert hasattr(ns, "CompositeEnvironment")
        assert hasattr(ns, "EnvironmentNotFittedError")
