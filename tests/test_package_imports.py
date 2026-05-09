"""Comprehensive import tests for the neurospatial package reorganization.

This test file verifies:
1. All new domain module import paths work
2. Core exports from top-level work
3. No circular imports
4. Re-exports work (stats functions from encoding modules)
5. Each domain module's __all__ exports are importable

Created as part of Milestone 13 per TASKS.md.
"""

import importlib
import sys

import pytest


class TestNoCircularImports:
    """Test that importing submodules doesn't cause circular import errors."""

    def test_import_neurospatial_fresh(self):
        """Test that neurospatial can be imported without circular import errors."""
        # Clear cached imports for a fresh test
        cached_modules = {
            name: module
            for name, module in sys.modules.items()
            if name.startswith("neurospatial")
        }
        modules_to_clear = [k for k in sys.modules if k.startswith("neurospatial")]
        for mod in modules_to_clear:
            sys.modules.pop(mod, None)

        try:
            # Import fresh
            ns = importlib.import_module("neurospatial")
            assert ns is not None
            assert hasattr(ns, "Environment")
        finally:
            for mod in list(sys.modules):
                if mod.startswith("neurospatial"):
                    sys.modules.pop(mod, None)
            sys.modules.update(cached_modules)

    def test_import_all_submodules_without_errors(self):
        """Test that all submodules can be imported without circular import errors."""
        submodules = [
            "neurospatial.encoding",
            "neurospatial.decoding",
            "neurospatial.behavior",
            "neurospatial.events",
            "neurospatial.ops",
            "neurospatial.stats",
            "neurospatial.io",
            "neurospatial.animation",
            "neurospatial.simulation",
            "neurospatial.layout",
            "neurospatial.regions",
            "neurospatial.environment",
            "neurospatial.composite",
            "neurospatial.annotation",
        ]

        for submodule in submodules:
            mod = importlib.import_module(submodule)
            assert mod is not None, f"Failed to import {submodule}"

    def test_cross_module_imports_no_circular(self):
        """Test that common cross-module import patterns don't cause circular imports."""
        # These are patterns users might use
        from neurospatial import Environment  # noqa: F401
        from neurospatial.behavior.segmentation import detect_laps  # noqa: F401
        from neurospatial.decoding import decode_position  # noqa: F401
        from neurospatial.encoding.spatial import compute_spatial_rate  # noqa: F401
        from neurospatial.events import peri_event_histogram  # noqa: F401
        from neurospatial.ops.egocentric import heading_from_velocity  # noqa: F401
        from neurospatial.stats.circular import rayleigh_test  # noqa: F401


class TestEncodingModuleAllExports:
    """Test that all encoding module exports are importable and in __all__."""

    def test_encoding_all_exports_importable(self):
        """Test that all items in encoding.__all__ are importable."""
        import neurospatial.encoding

        for name in neurospatial.encoding.__all__:
            assert hasattr(neurospatial.encoding, name), (
                f"'{name}' in __all__ but not importable from encoding"
            )

    def test_encoding_spatial_all_exports(self):
        """Test that all items in encoding.spatial.__all__ are importable."""
        import neurospatial.encoding.spatial as place

        for name in place.__all__:
            assert hasattr(place, name), (
                f"'{name}' in __all__ but not importable from encoding.spatial"
            )

    def test_encoding_grid_all_exports(self):
        """Test that all items in encoding.grid.__all__ are importable."""
        import neurospatial.encoding.grid as grid

        for name in grid.__all__:
            assert hasattr(grid, name), (
                f"'{name}' in __all__ but not importable from encoding.grid"
            )

    def test_encoding_directional_all_exports(self):
        """Test that all items in encoding.directional.__all__ are importable."""
        import neurospatial.encoding.directional as hd

        for name in hd.__all__:
            assert hasattr(hd, name), (
                f"'{name}' in __all__ but not importable from encoding.directional"
            )

    def test_encoding_border_all_exports(self):
        """Test that all items in encoding.border.__all__ are importable."""
        import neurospatial.encoding.border as border

        for name in border.__all__:
            assert hasattr(border, name), (
                f"'{name}' in __all__ but not importable from encoding.border"
            )

    def test_encoding_egocentric_all_exports(self):
        """Test that all items in encoding.egocentric.__all__ are importable."""
        import neurospatial.encoding.egocentric as ov

        for name in ov.__all__:
            assert hasattr(ov, name), (
                f"'{name}' in __all__ but not importable from encoding.egocentric"
            )

    def test_encoding_view_all_exports(self):
        """Test that all items in encoding.view.__all__ are importable."""
        import neurospatial.encoding.view as sv

        for name in sv.__all__:
            assert hasattr(sv, name), (
                f"'{name}' in __all__ but not importable from encoding.view"
            )

    def test_encoding_phase_precession_all_exports(self):
        """Test that all items in encoding.phase_precession.__all__ are importable."""
        # Use importlib to avoid shadowing by function with same name
        pp = importlib.import_module("neurospatial.encoding.phase_precession")

        for name in pp.__all__:
            assert hasattr(pp, name), (
                f"'{name}' in __all__ but not importable from encoding.phase_precession"
            )

    def test_encoding_population_all_exports(self):
        """Test that all items in encoding.population.__all__ are importable."""
        import neurospatial.encoding.population as pop

        for name in pop.__all__:
            assert hasattr(pop, name), (
                f"'{name}' in __all__ but not importable from encoding.population"
            )


class TestDecodingModuleAllExports:
    """Test that all decoding module exports are importable and in __all__."""

    def test_decoding_all_exports_importable(self):
        """Test that all items in decoding.__all__ are importable."""
        import neurospatial.decoding

        for name in neurospatial.decoding.__all__:
            assert hasattr(neurospatial.decoding, name), (
                f"'{name}' in __all__ but not importable from decoding"
            )


class TestBehaviorModuleAllExports:
    """Test that all behavior module exports are importable and in __all__."""

    def test_behavior_all_exports_importable(self):
        """Test that all items in behavior.__all__ are importable."""
        import neurospatial.behavior

        for name in neurospatial.behavior.__all__:
            assert hasattr(neurospatial.behavior, name), (
                f"'{name}' in __all__ but not importable from behavior"
            )

    def test_behavior_trajectory_all_exports(self):
        """Test that all items in behavior.trajectory.__all__ are importable."""
        import neurospatial.behavior.trajectory as traj

        if not hasattr(traj, "__all__"):
            pytest.skip("behavior.trajectory does not have __all__ defined")

        for name in traj.__all__:
            assert hasattr(traj, name), (
                f"'{name}' in __all__ but not importable from behavior.trajectory"
            )

    def test_behavior_segmentation_all_exports(self):
        """Test that all items in behavior.segmentation.__all__ are importable."""
        import neurospatial.behavior.segmentation as seg

        for name in seg.__all__:
            assert hasattr(seg, name), (
                f"'{name}' in __all__ but not importable from behavior.segmentation"
            )

    def test_behavior_navigation_all_exports(self):
        """Test that all items in behavior.navigation.__all__ are importable."""
        import neurospatial.behavior.navigation as nav

        for name in nav.__all__:
            assert hasattr(nav, name), (
                f"'{name}' in __all__ but not importable from behavior.navigation"
            )

    def test_behavior_decisions_all_exports(self):
        """Test that all items in behavior.decisions.__all__ are importable."""
        import neurospatial.behavior.decisions as dec

        for name in dec.__all__:
            assert hasattr(dec, name), (
                f"'{name}' in __all__ but not importable from behavior.decisions"
            )

    def test_behavior_reward_all_exports(self):
        """Test that all items in behavior.reward.__all__ are importable."""
        import neurospatial.behavior.reward as reward

        for name in reward.__all__:
            assert hasattr(reward, name), (
                f"'{name}' in __all__ but not importable from behavior.reward"
            )


class TestOpsModuleAllExports:
    """Test that all ops module exports are importable and in __all__."""

    def test_ops_all_exports_importable(self):
        """Test that all items in ops.__all__ are importable."""
        import neurospatial.ops

        for name in neurospatial.ops.__all__:
            assert hasattr(neurospatial.ops, name), (
                f"'{name}' in __all__ but not importable from ops"
            )

    @pytest.mark.parametrize(
        "submodule",
        [
            "binning",
            "distance",
            "normalize",
            "smoothing",
            "graph",
            "calculus",
            "transforms",
            "alignment",
            "egocentric",
            "visibility",
            "basis",
        ],
    )
    def test_ops_submodule_all_exports(self, submodule):
        """Test that all items in ops submodule __all__ are importable."""
        mod = importlib.import_module(f"neurospatial.ops.{submodule}")
        if hasattr(mod, "__all__"):
            for name in mod.__all__:
                assert hasattr(mod, name), (
                    f"'{name}' in __all__ but not importable from ops.{submodule}"
                )


class TestStatsModuleAllExports:
    """Test that all stats module exports are importable and in __all__."""

    def test_stats_all_exports_importable(self):
        """Test that all items in stats.__all__ are importable."""
        import neurospatial.stats

        for name in neurospatial.stats.__all__:
            assert hasattr(neurospatial.stats, name), (
                f"'{name}' in __all__ but not importable from stats"
            )

    def test_stats_circular_all_exports(self):
        """Test that all items in stats.circular.__all__ are importable."""
        import neurospatial.stats.circular as circ

        for name in circ.__all__:
            assert hasattr(circ, name), (
                f"'{name}' in __all__ but not importable from stats.circular"
            )

    def test_stats_shuffle_all_exports(self):
        """Test that all items in stats.shuffle.__all__ are importable."""
        import neurospatial.stats.shuffle as shuf

        if not hasattr(shuf, "__all__"):
            pytest.skip("stats.shuffle does not have __all__ defined")

        for name in shuf.__all__:
            assert hasattr(shuf, name), (
                f"'{name}' in __all__ but not importable from stats.shuffle"
            )

    def test_stats_surrogates_all_exports(self):
        """Test that all items in stats.surrogates.__all__ are importable."""
        import neurospatial.stats.surrogates as surr

        for name in surr.__all__:
            assert hasattr(surr, name), (
                f"'{name}' in __all__ but not importable from stats.surrogates"
            )


class TestEventsModuleAllExports:
    """Test that all events module exports are importable and in __all__."""

    def test_events_all_exports_importable(self):
        """Test that all items in events.__all__ are importable."""
        import neurospatial.events

        for name in neurospatial.events.__all__:
            assert hasattr(neurospatial.events, name), (
                f"'{name}' in __all__ but not importable from events"
            )


class TestIOModuleAllExports:
    """Test that all io module exports are importable and in __all__."""

    def test_io_all_exports_importable(self):
        """Test that all items in io.__all__ are importable."""
        import neurospatial.io

        for name in neurospatial.io.__all__:
            assert hasattr(neurospatial.io, name), (
                f"'{name}' in __all__ but not importable from io"
            )


class TestAnimationModuleAllExports:
    """Test that all animation module exports are importable and in __all__."""

    def test_animation_all_exports_importable(self):
        """Test that all items in animation.__all__ are importable."""
        import neurospatial.animation

        for name in neurospatial.animation.__all__:
            assert hasattr(neurospatial.animation, name), (
                f"'{name}' in __all__ but not importable from animation"
            )


class TestSimulationModuleAllExports:
    """Test that all simulation module exports are importable and in __all__."""

    def test_simulation_all_exports_importable(self):
        """Test that all items in simulation.__all__ are importable."""
        import neurospatial.simulation

        for name in neurospatial.simulation.__all__:
            assert hasattr(neurospatial.simulation, name), (
                f"'{name}' in __all__ but not importable from simulation"
            )


class TestLayoutModuleAllExports:
    """Test that all layout module exports are importable and in __all__."""

    def test_layout_all_exports_importable(self):
        """Test that all items in layout.__all__ are importable."""
        import neurospatial.layout

        for name in neurospatial.layout.__all__:
            assert hasattr(neurospatial.layout, name), (
                f"'{name}' in __all__ but not importable from layout"
            )


class TestRegionsModuleAllExports:
    """Test that all regions module exports are importable and in __all__."""

    def test_regions_all_exports_importable(self):
        """Test that all items in regions.__all__ are importable."""
        import neurospatial.regions

        for name in neurospatial.regions.__all__:
            assert hasattr(neurospatial.regions, name), (
                f"'{name}' in __all__ but not importable from regions"
            )


class TestPLANMDExampleUsage:
    """Test the example usage patterns from PLAN.md work correctly."""

    def test_plan_md_neural_encoding_pattern(self):
        """Test the neural encoding import pattern from PLAN.md."""
        from neurospatial.encoding import (
            directional,
            grid,
            spatial,
            spatial_information,
        )

        assert hasattr(spatial, "compute_spatial_rate")
        assert callable(spatial_information)
        assert hasattr(spatial, "detect_place_fields")
        assert hasattr(grid, "grid_score")
        assert hasattr(directional, "compute_directional_rate")

    def test_plan_md_neural_decoding_pattern(self):
        """Test the neural decoding import pattern from PLAN.md."""
        from neurospatial.decoding import decode_position
        from neurospatial.stats.shuffle import (
            compute_shuffle_pvalue,
            shuffle_time_bins,
        )

        assert callable(decode_position)
        assert callable(shuffle_time_bins)
        assert callable(compute_shuffle_pvalue)

    def test_plan_md_behavioral_analysis_pattern(self):
        """Test the behavioral analysis import pattern from PLAN.md."""
        from neurospatial.behavior import navigation, segmentation, trajectory

        assert hasattr(segmentation, "detect_laps")
        assert hasattr(navigation, "compute_path_efficiency")
        assert hasattr(trajectory, "mean_square_displacement")

    def test_plan_md_events_pattern(self):
        """Test the events import pattern from PLAN.md."""
        from neurospatial.events import peri_event_histogram

        assert callable(peri_event_histogram)

    def test_plan_md_ops_pattern(self):
        """Test the low-level ops import pattern from PLAN.md."""
        from neurospatial.ops import distance, normalize

        assert hasattr(distance, "distance_field")
        assert hasattr(normalize, "normalize_field")

    def test_plan_md_visualization_pattern(self):
        """Test the visualization import pattern from PLAN.md."""
        from neurospatial.animation import overlays

        assert hasattr(overlays, "PositionOverlay")


class TestEnvironmentReimportConsistency:
    """Test that Environment imported from different paths is the same class."""

    def test_environment_top_level_vs_submodule(self):
        """Test Environment from top level equals Environment from environment submodule."""
        from neurospatial import Environment as TopEnv
        from neurospatial.environment import Environment as SubEnv

        assert TopEnv is SubEnv

    def test_region_top_level_vs_submodule(self):
        """Test Region from top level equals Region from regions submodule."""
        from neurospatial import Region as TopReg
        from neurospatial.regions import Region as SubReg

        assert TopReg is SubReg

    def test_regions_top_level_vs_submodule(self):
        """Test Regions from top level equals Regions from regions submodule."""
        from neurospatial import Regions as TopRegs
        from neurospatial.regions import Regions as SubRegs

        assert TopRegs is SubRegs

    def test_composite_environment_top_level_vs_submodule(self):
        """Test CompositeEnvironment from top level equals from composite submodule."""
        from neurospatial import CompositeEnvironment as TopComp
        from neurospatial.composite import CompositeEnvironment as SubComp

        assert TopComp is SubComp
