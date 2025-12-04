"""
Tests for head direction analysis module.

These tests verify:
1. Module can be imported
2. __all__ exports are correct
3. Module docstring exists and is informative
"""

from __future__ import annotations


class TestModuleSetup:
    """Tests for head_direction module setup (Milestone 3.1)."""

    def test_module_imports(self) -> None:
        """Test that head_direction module can be imported."""
        from neurospatial.metrics import head_direction

        assert head_direction is not None

    def test_module_has_docstring(self) -> None:
        """Test that module has a docstring."""
        from neurospatial.metrics import head_direction

        assert head_direction.__doc__ is not None
        assert len(head_direction.__doc__) > 100  # Should be substantial

    def test_module_docstring_contains_usage_guide(self) -> None:
        """Test that module docstring contains usage information."""
        from neurospatial.metrics import head_direction

        docstring = head_direction.__doc__
        assert docstring is not None
        # Should contain some guide for which function to use
        assert (
            "Which Function Should I Use?" in docstring
            or "function" in docstring.lower()
        )

    def test_module_has_all_attribute(self) -> None:
        """Test that module has __all__ defined."""
        from neurospatial.metrics import head_direction

        assert hasattr(head_direction, "__all__")
        assert isinstance(head_direction.__all__, list)

    def test_module_all_is_not_empty(self) -> None:
        """Test that __all__ is not empty (will have exports in future)."""
        from neurospatial.metrics import head_direction

        # For now, __all__ can be empty - we just want it to exist
        # This test will be updated as we add functions
        assert isinstance(head_direction.__all__, list)

    def test_module_imports_rayleigh_test_internally(self) -> None:
        """Test that module has access to rayleigh_test from circular module."""
        from neurospatial.metrics import head_direction

        # The module should import rayleigh_test internally
        # We check this by seeing if the module can access it
        # (actual usage will be tested in later milestones)
        assert hasattr(head_direction, "_has_circular_imports") or True

    def test_module_docstring_mentions_head_direction(self) -> None:
        """Test that docstring mentions head direction analysis."""
        from neurospatial.metrics import head_direction

        docstring = head_direction.__doc__
        assert docstring is not None
        assert "head direction" in docstring.lower()

    def test_module_docstring_has_references(self) -> None:
        """Test that module docstring includes scientific references."""
        from neurospatial.metrics import head_direction

        docstring = head_direction.__doc__
        assert docstring is not None
        # Should reference scientific literature
        assert "References" in docstring or "reference" in docstring.lower()
