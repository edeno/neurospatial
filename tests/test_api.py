"""Tests for top-level API imports.

This test module verifies that all imports documented in CLAUDE.md work correctly
at the top level of the neurospatial package.
"""


class TestTopLevelImports:
    """Test that all documented top-level imports work."""

    def test_core_classes_import(self):
        """Test that core classes can be imported from top level."""
        from neurospatial import CompositeEnvironment, Environment

        assert Environment is not None
        assert CompositeEnvironment is not None

    def test_serialization_functions_import(self):
        """Test that I/O functions can be imported from io submodule.

        These are documented in CLAUDE.md §13 (Import Patterns):
        from neurospatial.io import to_file, from_file, to_dict, from_dict
        """
        from neurospatial.io import from_dict, from_file, to_dict, to_file

        assert to_file is not None
        assert from_file is not None
        assert to_dict is not None
        assert from_dict is not None

    def test_region_classes_import(self):
        """Test that Region classes can be imported from top level.

        These are documented in CLAUDE.md §13 (Import Patterns):
        from neurospatial import Region, Regions
        """
        from neurospatial import Region, Regions

        assert Region is not None
        assert Regions is not None

    def test_public_api_functions_import(self):
        """Test that public API functions can be imported from submodules.

        These are documented in CLAUDE.md §13 (Import Patterns):
        from neurospatial.ops import map_points_to_bins, distance_field, pairwise_distances
        from neurospatial.ops.transforms import estimate_transform, apply_transform_to_environment
        from neurospatial.layout.validation import validate_environment
        """
        from neurospatial.layout.validation import validate_environment
        from neurospatial.ops import (
            distance_field,
            map_points_to_bins,
            pairwise_distances,
        )
        from neurospatial.ops.transforms import (
            apply_transform_to_environment,
            estimate_transform,
        )

        assert validate_environment is not None
        assert map_points_to_bins is not None
        assert estimate_transform is not None
        assert apply_transform_to_environment is not None
        assert distance_field is not None
        assert pairwise_distances is not None

    def test_all_claude_md_imports_work(self):
        """Comprehensive test that all imports in CLAUDE.md Import Patterns work.

        This test imports everything documented in CLAUDE.md §13 in a single
        statement to verify the complete documented API is available.
        """
        # Test the exact import pattern from CLAUDE.md
        from neurospatial import CompositeEnvironment, Environment, Region, Regions
        from neurospatial.io import from_dict, from_file, to_dict, to_file
        from neurospatial.layout.validation import validate_environment
        from neurospatial.ops import (
            distance_field,
            map_points_to_bins,
            pairwise_distances,
        )
        from neurospatial.ops.transforms import (
            apply_transform_to_environment,
            estimate_transform,
        )

        # Verify all are not None
        imports = [
            CompositeEnvironment,
            Environment,
            Region,
            Regions,
            to_file,
            from_file,
            to_dict,
            from_dict,
            map_points_to_bins,
            estimate_transform,
            apply_transform_to_environment,
            distance_field,
            pairwise_distances,
            validate_environment,
        ]

        for imported in imports:
            assert imported is not None


class TestDunderAll:
    """Test that __all__ is properly maintained."""

    def test_all_list_exists(self):
        """Test that __all__ list exists in package."""
        import neurospatial

        assert hasattr(neurospatial, "__all__")
        assert isinstance(neurospatial.__all__, list)

    def test_all_exported_symbols_importable(self):
        """Test that all symbols in __all__ can be imported."""
        import neurospatial

        for symbol in neurospatial.__all__:
            # Verify each symbol in __all__ exists in the module
            assert hasattr(neurospatial, symbol), (
                f"Symbol '{symbol}' in __all__ but not in module"
            )

    def test_documented_imports_in_all(self):
        """Test that all documented imports are in __all__."""
        import neurospatial

        # Core documented imports from CLAUDE.md - only the 5 core classes
        # are exported from top-level neurospatial
        documented_imports = [
            "Environment",
            "CompositeEnvironment",
            "Region",
            "Regions",
            "EnvironmentNotFittedError",
        ]

        for symbol in documented_imports:
            assert symbol in neurospatial.__all__, (
                f"Documented symbol '{symbol}' not in __all__"
            )
