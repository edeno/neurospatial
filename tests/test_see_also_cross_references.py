"""Tests for 'See Also' cross-references in Environment factory methods.

This module tests that all factory methods have appropriate cross-references
to related methods, enabling users to discover alternatives when one method
doesn't fit their use case.
"""

import inspect

from neurospatial import Environment


class TestSeeAlsoCrossReferences:
    """Test suite for 'See Also' sections in factory methods."""

    def test_from_samples_has_see_also_section(self):
        """Test that from_samples() has a 'See Also' section."""
        docstring = inspect.getdoc(Environment.from_samples)
        assert docstring is not None, "from_samples() has no docstring"
        assert "See Also" in docstring, "from_samples() missing 'See Also' section"

    def test_from_samples_references_polygon_mask_layout(self):
        """Test that from_samples() references polygon, mask, and layout methods."""
        docstring = inspect.getdoc(Environment.from_samples)
        assert docstring is not None

        # Should reference from_polygon, from_mask, and from_layout
        assert "from_polygon" in docstring, (
            "from_samples() should reference from_polygon()"
        )
        assert "from_mask" in docstring, "from_samples() should reference from_mask()"
        assert "from_layout" in docstring, (
            "from_samples() should reference from_layout()"
        )

    def test_from_polygon_has_see_also_section(self):
        """Test that from_polygon() has a 'See Also' section."""
        docstring = inspect.getdoc(Environment.from_polygon)
        assert docstring is not None, "from_polygon() has no docstring"
        assert "See Also" in docstring, "from_polygon() missing 'See Also' section"

    def test_from_polygon_references_samples_mask_image(self):
        """Test that from_polygon() references samples, mask, and image methods."""
        docstring = inspect.getdoc(Environment.from_polygon)
        assert docstring is not None

        # Should reference from_samples, from_mask, and from_image
        assert "from_samples" in docstring, (
            "from_polygon() should reference from_samples()"
        )
        assert "from_mask" in docstring, "from_polygon() should reference from_mask()"
        assert "from_image" in docstring, "from_polygon() should reference from_image()"

    def test_from_mask_has_see_also_section(self):
        """Test that from_mask() has a 'See Also' section."""
        docstring = inspect.getdoc(Environment.from_mask)
        assert docstring is not None, "from_mask() has no docstring"
        assert "See Also" in docstring, "from_mask() missing 'See Also' section"

    def test_from_mask_references_samples_polygon_image(self):
        """Test that from_mask() references samples, polygon, and image methods."""
        docstring = inspect.getdoc(Environment.from_mask)
        assert docstring is not None

        # Should reference from_samples, from_polygon, and from_image
        assert "from_samples" in docstring, (
            "from_mask() should reference from_samples()"
        )
        assert "from_polygon" in docstring, (
            "from_mask() should reference from_polygon()"
        )
        assert "from_image" in docstring, "from_mask() should reference from_image()"

    def test_from_image_has_see_also_section(self):
        """Test that from_image() has a 'See Also' section."""
        docstring = inspect.getdoc(Environment.from_image)
        assert docstring is not None, "from_image() has no docstring"
        assert "See Also" in docstring, "from_image() missing 'See Also' section"

    def test_from_image_references_mask_polygon_samples(self):
        """Test that from_image() references mask, polygon, and samples methods."""
        docstring = inspect.getdoc(Environment.from_image)
        assert docstring is not None

        # Should reference from_mask, from_polygon, and from_samples
        assert "from_mask" in docstring, "from_image() should reference from_mask()"
        assert "from_polygon" in docstring, (
            "from_image() should reference from_polygon()"
        )
        assert "from_samples" in docstring, (
            "from_image() should reference from_samples()"
        )

    def test_from_graph_has_see_also_section(self):
        """Test that from_graph() has a 'See Also' section."""
        docstring = inspect.getdoc(Environment.from_graph)
        assert docstring is not None, "from_graph() has no docstring"
        assert "See Also" in docstring, "from_graph() missing 'See Also' section"

    def test_from_graph_references_samples_layout(self):
        """Test that from_graph() references samples and layout methods."""
        docstring = inspect.getdoc(Environment.from_graph)
        assert docstring is not None

        # Should reference from_samples and from_layout
        assert "from_samples" in docstring, (
            "from_graph() should reference from_samples()"
        )
        assert "from_layout" in docstring, "from_graph() should reference from_layout()"

    def test_from_layout_has_see_also_section(self):
        """Test that from_layout() has a 'See Also' section."""
        docstring = inspect.getdoc(Environment.from_layout)
        assert docstring is not None, "from_layout() has no docstring"
        assert "See Also" in docstring, "from_layout() missing 'See Also' section"

    def test_from_layout_references_all_specialized_methods(self):
        """Test that from_layout() references all specialized factory methods."""
        docstring = inspect.getdoc(Environment.from_layout)
        assert docstring is not None

        # Should reference all 5 specialized methods
        assert "from_samples" in docstring, (
            "from_layout() should reference from_samples()"
        )
        assert "from_polygon" in docstring, (
            "from_layout() should reference from_polygon()"
        )
        assert "from_mask" in docstring, "from_layout() should reference from_mask()"
        assert "from_image" in docstring, "from_layout() should reference from_image()"
        assert "from_graph" in docstring, "from_layout() should reference from_graph()"

    def test_see_also_sections_positioned_correctly(self):
        """Test that 'See Also' sections appear after Returns and before Examples.

        NumPy docstring convention is: Parameters -> Returns -> See Also -> Examples.
        """
        methods = [
            Environment.from_samples,
            Environment.from_polygon,
            Environment.from_mask,
            Environment.from_image,
            Environment.from_graph,
            Environment.from_layout,
        ]

        for method in methods:
            docstring = inspect.getdoc(method)
            assert docstring is not None, f"{method.__name__} has no docstring"

            # Find positions of sections
            returns_pos = docstring.find("Returns")
            see_also_pos = docstring.find("See Also")
            examples_pos = docstring.find("Examples")

            # Verify positions (if Examples exists, See Also should be between Returns and Examples)
            if examples_pos != -1 and see_also_pos != -1:
                assert returns_pos < see_also_pos < examples_pos, (
                    f"{method.__name__}: 'See Also' should be between 'Returns' and 'Examples'"
                )
            elif see_also_pos != -1:
                # If no Examples, See Also should still come after Returns
                assert returns_pos < see_also_pos, (
                    f"{method.__name__}: 'See Also' should come after 'Returns'"
                )

    def test_bidirectional_cross_references(self):
        """Test that cross-references are bidirectional.

        If method A references method B, then method B should reference method A.
        """
        # Define expected bidirectional relationships
        relationships = [
            ("from_samples", "from_polygon"),
            ("from_samples", "from_mask"),
            ("from_samples", "from_layout"),
            ("from_samples", "from_graph"),
            ("from_polygon", "from_samples"),
            ("from_polygon", "from_mask"),
            ("from_polygon", "from_image"),
            ("from_mask", "from_samples"),
            ("from_mask", "from_polygon"),
            ("from_mask", "from_image"),
            ("from_image", "from_mask"),
            ("from_image", "from_polygon"),
            ("from_image", "from_samples"),
            ("from_graph", "from_samples"),
            ("from_graph", "from_layout"),
        ]

        for method_a_name, method_b_name in relationships:
            method_a = getattr(Environment, method_a_name)
            method_b = getattr(Environment, method_b_name)

            docstring_a = inspect.getdoc(method_a)
            docstring_b = inspect.getdoc(method_b)

            # If A references B, check that B references A
            if method_b_name in docstring_a:
                assert method_a_name in docstring_b, (
                    f"{method_a_name} references {method_b_name}, but not vice versa"
                )

    def test_see_also_format_follows_numpy_style(self):
        """Test that 'See Also' sections follow NumPy docstring format.

        NumPy style uses:
        See Also
        --------
        function_name : Description of relationship.
        """
        methods = [
            Environment.from_samples,
            Environment.from_polygon,
            Environment.from_mask,
            Environment.from_image,
            Environment.from_graph,
            Environment.from_layout,
        ]

        for method in methods:
            docstring = inspect.getdoc(method)
            assert docstring is not None

            # Find the See Also section
            see_also_pos = docstring.find("See Also")
            if see_also_pos == -1:
                continue  # Skip if no See Also section (will fail in other tests)

            # Extract the See Also section (find next section or end of docstring)
            next_section_keywords = ["Notes", "Examples", "References"]
            see_also_section_end = len(docstring)
            for keyword in next_section_keywords:
                pos = docstring.find(keyword, see_also_pos + len("See Also"))
                if pos != -1 and pos < see_also_section_end:
                    see_also_section_end = pos

            see_also_section = docstring[see_also_pos:see_also_section_end]

            # Check for proper underline (should be dashes)
            assert "--------" in see_also_section, (
                f"{method.__name__}: 'See Also' should have dash underline"
            )
