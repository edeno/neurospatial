"""Tests for animation utility functions."""

from neurospatial.animation._utils import _pickling_guidance


class TestPicklingGuidance:
    """Tests for _pickling_guidance() helper function."""

    def test_returns_string(self) -> None:
        """Helper returns a non-empty string."""
        result = _pickling_guidance()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_clear_cache_option(self) -> None:
        """Guidance includes env.clear_cache() as an option."""
        result = _pickling_guidance()
        assert "clear_cache" in result.lower()

    def test_includes_n_workers_option(self) -> None:
        """Guidance includes n_workers=1 as an option."""
        result = _pickling_guidance()
        assert "n_workers" in result.lower() or "n_workers=1" in result

    def test_includes_serial_rendering_mention(self) -> None:
        """Guidance mentions serial/single-threaded rendering."""
        result = _pickling_guidance()
        # Should mention that this is slower but doesn't require pickling
        result_lower = result.lower()
        assert "serial" in result_lower or "single" in result_lower

    def test_consistent_format(self) -> None:
        """Guidance has consistent format for multiple calls."""
        result1 = _pickling_guidance()
        result2 = _pickling_guidance()
        assert result1 == result2

    def test_includes_numbered_options(self) -> None:
        """Guidance includes numbered options for clarity."""
        result = _pickling_guidance()
        # Should have at least option 1 and option 2
        assert "1." in result or "Option 1" in result
        assert "2." in result or "Option 2" in result

    def test_with_n_workers_parameter(self) -> None:
        """Guidance can include specific n_workers value."""
        result = _pickling_guidance(n_workers=4)
        # Should show the actual n_workers value in examples if provided
        assert "4" in result

    def test_without_n_workers_parameter(self) -> None:
        """Guidance works without n_workers parameter."""
        result = _pickling_guidance()
        # Should still be valid guidance
        assert "clear_cache" in result.lower()
