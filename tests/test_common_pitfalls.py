"""Tests for Common Pitfalls sections in factory method docstrings.

This test suite verifies that comprehensive "Common Pitfalls" sections exist in:
1. Environment.from_samples() docstring
2. CompositeEnvironment.__init__() docstring

These sections help users avoid common mistakes and debug issues faster.
"""

import inspect

import pytest

from neurospatial import Environment
from neurospatial.composite import CompositeEnvironment


class TestFromSamplesCommonPitfalls:
    """Test that from_samples() docstring has a comprehensive Common Pitfalls section."""

    @pytest.fixture
    def from_samples_docstring(self):
        """Get the from_samples() docstring."""
        return inspect.getdoc(Environment.from_samples)

    def test_has_common_pitfalls_section(self, from_samples_docstring):
        """Test that from_samples() has a 'Common Pitfalls' section."""
        assert from_samples_docstring is not None
        assert "Common Pitfalls" in from_samples_docstring, (
            "from_samples() docstring should have a 'Common Pitfalls' section"
        )

    def test_mentions_bin_size_too_large(self, from_samples_docstring):
        """Test that docstring warns about bin_size being too large."""
        # Should mention the pitfall of bin_size being too large relative to data
        assert "bin_size" in from_samples_docstring.lower()
        assert any(
            phrase in from_samples_docstring.lower()
            for phrase in ["too large", "too big", "larger than"]
        ), "Should warn about bin_size being too large"

    def test_mentions_bin_count_threshold_too_high(self, from_samples_docstring):
        """Test that docstring warns about bin_count_threshold being too high."""
        assert "bin_count_threshold" in from_samples_docstring.lower()
        assert any(
            phrase in from_samples_docstring.lower()
            for phrase in ["too high", "too large", "no bins", "no active bins"]
        ), "Should warn about bin_count_threshold being too high"

    def test_mentions_mismatched_units(self, from_samples_docstring):
        """Test that docstring warns about mismatched units."""
        assert any(
            phrase in from_samples_docstring.lower()
            for phrase in [
                "units",
                "unit",
                "centimeter",
                "meter",
                "mismatch",
                "inconsistent",
            ]
        ), "Should warn about unit mismatches"

    def test_mentions_missing_morphological_operations(self, from_samples_docstring):
        """Test that docstring warns about missing morphological operations."""
        # Should mention dilate, fill_holes, close_gaps, or morphological operations
        assert any(
            phrase in from_samples_docstring.lower()
            for phrase in [
                "dilate",
                "fill_holes",
                "close_gaps",
                "morphological",
                "sparse data",
                "holes",
                "gaps",
            ]
        ), "Should warn about missing morphological operations"

    def test_common_pitfalls_section_is_detailed(self, from_samples_docstring):
        """Test that Common Pitfalls section contains substantial guidance."""
        # Find the Common Pitfalls section
        lines = from_samples_docstring.split("\n")
        pitfalls_start_idx = None
        for i, line in enumerate(lines):
            if "Common Pitfalls" in line:
                pitfalls_start_idx = i
                break

        assert pitfalls_start_idx is not None, "Common Pitfalls section not found"

        # Count non-empty lines in the pitfalls section (until next section or end)
        # Skip the header and underline (first 2 lines after header)
        pitfalls_lines = []
        in_pitfalls = False
        for i, line in enumerate(lines[pitfalls_start_idx + 1 :]):
            # Skip the underline
            if i == 0 and line.strip().startswith("-"):
                in_pitfalls = True
                continue

            if in_pitfalls:
                # Stop at next section (capitalized section header at start of line)
                if (
                    line
                    and not line.startswith(" ")
                    and line[0].isupper()
                    and ":" not in line
                ):
                    # Likely a new section header
                    break
                elif line.strip():
                    pitfalls_lines.append(line)

        # Should have at least 10 lines of content (4 pitfalls with descriptions)
        assert len(pitfalls_lines) >= 10, (
            f"Common Pitfalls section should be detailed (got {len(pitfalls_lines)} lines)"
        )

    def test_common_pitfalls_provides_actionable_guidance(self, from_samples_docstring):
        """Test that Common Pitfalls section provides actionable guidance."""
        # Should contain actionable words like "try", "use", "ensure", "check", "set"
        assert any(
            word in from_samples_docstring.lower()
            for word in ["try", "use", "ensure", "check", "set", "reduce", "enable"]
        ), "Common Pitfalls should provide actionable guidance"

    def test_common_pitfalls_positioned_appropriately(self, from_samples_docstring):
        """Test that Common Pitfalls section is positioned after Examples."""
        # Common Pitfalls should come after Examples section (if present)
        # or after Returns/Raises/See Also sections
        lines = from_samples_docstring.split("\n")
        examples_idx = None
        pitfalls_idx = None

        for i, line in enumerate(lines):
            if "Examples" in line and line.strip().startswith("Examples"):
                examples_idx = i
            if "Common Pitfalls" in line:
                pitfalls_idx = i

        assert pitfalls_idx is not None, "Common Pitfalls section not found"

        # If Examples section exists, Common Pitfalls should come after it
        if examples_idx is not None:
            assert pitfalls_idx > examples_idx, (
                "Common Pitfalls should come after Examples section"
            )


class TestCompositeEnvironmentCommonPitfalls:
    """Test that CompositeEnvironment.__init__() has a comprehensive Common Pitfalls section."""

    @pytest.fixture
    def composite_init_docstring(self):
        """Get the CompositeEnvironment.__init__() docstring."""
        return inspect.getdoc(CompositeEnvironment.__init__)

    def test_has_common_pitfalls_section(self, composite_init_docstring):
        """Test that CompositeEnvironment.__init__() has a 'Common Pitfalls' section."""
        assert composite_init_docstring is not None
        assert "Common Pitfalls" in composite_init_docstring, (
            "CompositeEnvironment.__init__() should have a 'Common Pitfalls' section"
        )

    def test_mentions_dimension_mismatch(self, composite_init_docstring):
        """Test that docstring warns about dimension mismatches."""
        assert any(
            phrase in composite_init_docstring.lower()
            for phrase in [
                "dimension",
                "n_dims",
                "2d",
                "3d",
                "mismatch",
                "same dimensionality",
            ]
        ), "Should warn about dimension mismatches"

    def test_mentions_no_bridge_edges(self, composite_init_docstring):
        """Test that docstring warns about no bridge edges."""
        assert any(
            phrase in composite_init_docstring.lower()
            for phrase in [
                "bridge",
                "no bridge",
                "disconnected",
                "not connected",
                "isolated",
                "auto_bridge",
            ]
        ), "Should warn about missing bridge edges"

    def test_mentions_overlapping_bins(self, composite_init_docstring):
        """Test that docstring warns about overlapping bins."""
        assert any(
            phrase in composite_init_docstring.lower()
            for phrase in [
                "overlap",
                "overlapping",
                "same location",
                "duplicate",
                "too close",
            ]
        ), "Should warn about overlapping bins"

    def test_common_pitfalls_section_is_detailed(self, composite_init_docstring):
        """Test that Common Pitfalls section contains substantial guidance."""
        # Find the Common Pitfalls section
        lines = composite_init_docstring.split("\n")
        pitfalls_start_idx = None
        for i, line in enumerate(lines):
            if "Common Pitfalls" in line:
                pitfalls_start_idx = i
                break

        assert pitfalls_start_idx is not None, "Common Pitfalls section not found"

        # Count non-empty lines in the pitfalls section
        # Skip the header and underline (first 2 lines after header)
        pitfalls_lines = []
        in_pitfalls = False
        for i, line in enumerate(lines[pitfalls_start_idx + 1 :]):
            # Skip the underline
            if i == 0 and line.strip().startswith("-"):
                in_pitfalls = True
                continue

            if in_pitfalls:
                # Stop at next section (capitalized section header at start of line)
                if (
                    line
                    and not line.startswith(" ")
                    and line[0].isupper()
                    and ":" not in line
                ):
                    break
                elif line.strip():
                    pitfalls_lines.append(line)

        # Should have at least 8 lines of content (3 pitfalls with descriptions)
        assert len(pitfalls_lines) >= 8, (
            f"Common Pitfalls section should be detailed (got {len(pitfalls_lines)} lines)"
        )

    def test_common_pitfalls_provides_actionable_guidance(
        self, composite_init_docstring
    ):
        """Test that Common Pitfalls section provides actionable guidance."""
        assert any(
            word in composite_init_docstring.lower()
            for word in [
                "ensure",
                "check",
                "verify",
                "use",
                "try",
                "set",
                "increase",
                "adjust",
            ]
        ), "Common Pitfalls should provide actionable guidance"

    def test_common_pitfalls_positioned_appropriately(self, composite_init_docstring):
        """Test that Common Pitfalls section is positioned after main sections."""
        # Common Pitfalls should come after Parameters/Returns/Raises sections
        lines = composite_init_docstring.split("\n")
        parameters_idx = None
        pitfalls_idx = None

        for i, line in enumerate(lines):
            if "Parameters" in line and line.strip().startswith("Parameters"):
                parameters_idx = i
            if "Common Pitfalls" in line:
                pitfalls_idx = i

        assert pitfalls_idx is not None, "Common Pitfalls section not found"

        # Common Pitfalls should come after Parameters section
        if parameters_idx is not None:
            assert pitfalls_idx > parameters_idx, (
                "Common Pitfalls should come after Parameters section"
            )


class TestCommonPitfallsCoverage:
    """Test that all required common pitfalls are documented."""

    def test_from_samples_has_all_four_pitfalls(self):
        """Test that from_samples() documents all 4 required pitfalls."""
        docstring = inspect.getdoc(Environment.from_samples)
        assert docstring is not None

        # Check for all 4 pitfalls
        pitfalls_covered = {
            "bin_size_too_large": any(
                phrase in docstring.lower()
                for phrase in ["bin_size", "too large", "too big"]
            ),
            "threshold_too_high": any(
                phrase in docstring.lower()
                for phrase in ["bin_count_threshold", "threshold", "too high"]
            ),
            "mismatched_units": any(
                phrase in docstring.lower() for phrase in ["unit", "mismatch"]
            ),
            "missing_morphology": any(
                phrase in docstring.lower()
                for phrase in ["dilate", "fill_holes", "close_gaps", "morphological"]
            ),
        }

        uncovered = [k for k, v in pitfalls_covered.items() if not v]
        assert not uncovered, (
            f"from_samples() is missing pitfalls: {', '.join(uncovered)}"
        )

    def test_composite_init_has_all_three_pitfalls(self):
        """Test that CompositeEnvironment.__init__() documents all 3 required pitfalls."""
        docstring = inspect.getdoc(CompositeEnvironment.__init__)
        assert docstring is not None

        # Check for all 3 pitfalls
        pitfalls_covered = {
            "dimension_mismatch": any(
                phrase in docstring.lower()
                for phrase in ["dimension", "n_dims", "mismatch"]
            ),
            "no_bridge_edges": any(
                phrase in docstring.lower() for phrase in ["bridge", "disconnected"]
            ),
            "overlapping_bins": any(
                phrase in docstring.lower() for phrase in ["overlap", "overlapping"]
            ),
        }

        uncovered = [k for k, v in pitfalls_covered.items() if not v]
        assert not uncovered, (
            f"CompositeEnvironment.__init__() is missing pitfalls: {', '.join(uncovered)}"
        )


class TestCommonPitfallsFormat:
    """Test that Common Pitfalls sections follow NumPy docstring format."""

    def test_from_samples_follows_numpy_format(self):
        """Test that from_samples() Common Pitfalls section follows NumPy format."""
        docstring = inspect.getdoc(Environment.from_samples)
        assert docstring is not None

        # Should have section header with dashes
        lines = docstring.split("\n")
        header_found = False
        underline_found = False

        for i, line in enumerate(lines):
            if "Common Pitfalls" in line:
                header_found = True
                # Next line should be dashes
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("-"):
                    underline_found = True
                break

        assert header_found, "Common Pitfalls header not found"
        assert underline_found, (
            "Common Pitfalls section should have dashed underline (NumPy format)"
        )

    def test_composite_init_follows_numpy_format(self):
        """Test that CompositeEnvironment.__init__() Common Pitfalls follows NumPy format."""
        docstring = inspect.getdoc(CompositeEnvironment.__init__)
        assert docstring is not None

        # Should have section header with dashes
        lines = docstring.split("\n")
        header_found = False
        underline_found = False

        for i, line in enumerate(lines):
            if "Common Pitfalls" in line:
                header_found = True
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("-"):
                    underline_found = True
                break

        assert header_found, "Common Pitfalls header not found"
        assert underline_found, (
            "Common Pitfalls section should have dashed underline (NumPy format)"
        )
