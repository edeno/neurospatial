"""Tests for bin_size parameter requirements across factory methods.

This test suite verifies that bin_size is a required parameter for all
factory methods that use it, ensuring consistent API and preventing
silent bugs from arbitrary defaults.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from neurospatial import Environment


class TestBinSizeRequired:
    """Test that bin_size is required (no defaults) for factory methods."""

    def test_from_samples_requires_bin_size(self):
        """from_samples() should raise TypeError if bin_size not provided."""
        data = np.random.rand(100, 2) * 10

        # Should raise TypeError when bin_size is missing
        with pytest.raises(TypeError, match=r"missing.*required.*bin_size"):
            Environment.from_samples(data)

    def test_from_samples_accepts_explicit_bin_size(self):
        """from_samples() should work when bin_size is explicitly provided."""
        data = np.random.rand(100, 2) * 10

        # Should work with explicit bin_size
        env = Environment.from_samples(data, bin_size=2.0)
        assert env.n_bins > 0

    def test_from_polygon_requires_bin_size(self):
        """from_polygon() should raise TypeError if bin_size not provided."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Should raise TypeError when bin_size is missing
        with pytest.raises(TypeError, match=r"missing.*required.*bin_size"):
            Environment.from_polygon(polygon)

    def test_from_polygon_accepts_explicit_bin_size(self):
        """from_polygon() should work when bin_size is explicitly provided."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Should work with explicit bin_size
        env = Environment.from_polygon(polygon, bin_size=2.0)
        assert env.n_bins > 0

    def test_from_image_requires_bin_size(self):
        """from_image() should raise TypeError if bin_size not provided."""
        image_mask = np.ones((10, 10), dtype=bool)

        # Should raise TypeError when bin_size is missing
        with pytest.raises(TypeError, match=r"missing.*required.*bin_size"):
            Environment.from_image(image_mask)

    def test_from_image_accepts_explicit_bin_size(self):
        """from_image() should work when bin_size is explicitly provided."""
        image_mask = np.ones((10, 10), dtype=bool)

        # Should work with explicit bin_size
        env = Environment.from_image(image_mask, bin_size=1.0)
        assert env.n_bins > 0


class TestBinSizeAlreadyRequired:
    """Test that bin_size is already required for from_graph()."""

    def test_from_graph_requires_bin_size(self):
        """from_graph() already requires bin_size - verify it still does."""
        import networkx as nx

        # Create simple graph
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(10.0, 0.0))
        graph.add_edge(0, 1)
        edge_order = [(0, 1)]

        # Should raise TypeError when bin_size is missing
        with pytest.raises(
            TypeError, match=r"missing.*required.*(bin_size|edge_spacing)"
        ):
            Environment.from_graph(graph, edge_order)


class TestNoDefaults:
    """Test that no factory methods have defaults for bin_size."""

    def test_from_samples_signature_has_no_default(self):
        """Verify from_samples() signature shows bin_size as required."""
        import inspect

        sig = inspect.signature(Environment.from_samples)
        bin_size_param = sig.parameters["bin_size"]

        # Should have no default value
        assert bin_size_param.default is inspect.Parameter.empty, (
            f"from_samples() should have no default for bin_size, "
            f"but found: {bin_size_param.default}"
        )

    def test_from_polygon_signature_has_no_default(self):
        """Verify from_polygon() signature shows bin_size as required."""
        import inspect

        sig = inspect.signature(Environment.from_polygon)
        bin_size_param = sig.parameters["bin_size"]

        # Should have no default value
        assert bin_size_param.default is inspect.Parameter.empty, (
            f"from_polygon() should have no default for bin_size, "
            f"but found: {bin_size_param.default}"
        )

    def test_from_image_signature_has_no_default(self):
        """Verify from_image() signature shows bin_size as required."""
        import inspect

        sig = inspect.signature(Environment.from_image)
        bin_size_param = sig.parameters["bin_size"]

        # Should have no default value
        assert bin_size_param.default is inspect.Parameter.empty, (
            f"from_image() should have no default for bin_size, "
            f"but found: {bin_size_param.default}"
        )

    def test_from_graph_signature_has_no_default(self):
        """Verify from_graph() signature shows bin_size as required."""
        import inspect

        sig = inspect.signature(Environment.from_graph)
        bin_size_param = sig.parameters["bin_size"]

        # Should have no default value (already the case)
        assert bin_size_param.default is inspect.Parameter.empty, (
            f"from_graph() should have no default for bin_size, "
            f"but found: {bin_size_param.default}"
        )


class TestDocstringConsistency:
    """Test that docstrings correctly document bin_size as required."""

    def test_from_samples_docstring_shows_required(self):
        """from_samples() docstring should not say 'optional' or show 'default'."""
        docstring = Environment.from_samples.__doc__

        # Check for bin_size parameter documentation
        assert "bin_size" in docstring

        # Should not indicate it's optional or has a default
        # (We check the parameter section specifically)
        lines = docstring.split("\n")
        in_bin_size_section = False
        bin_size_section = []

        for line in lines:
            if "bin_size :" in line:
                in_bin_size_section = True
            elif in_bin_size_section:
                if line.strip() and not line.startswith(" "):
                    break  # End of bin_size section
                bin_size_section.append(line)

        bin_size_text = " ".join(bin_size_section).lower()

        # Should not mention "optional", "default", "defaults to"
        assert "optional" not in bin_size_text, (
            "from_samples() docstring should not describe bin_size as optional"
        )
        assert "default" not in bin_size_text or "by default" in bin_size_text, (
            "from_samples() docstring should not mention a default value for bin_size"
        )

    def test_from_polygon_docstring_shows_required(self):
        """from_polygon() docstring should not say 'optional' or show 'default'."""
        docstring = Environment.from_polygon.__doc__

        # Check for bin_size parameter documentation
        assert "bin_size" in docstring

        # Should not indicate it's optional
        lines = docstring.split("\n")
        in_bin_size_section = False
        bin_size_section = []

        for line in lines:
            if "bin_size :" in line:
                in_bin_size_section = True
            elif in_bin_size_section:
                if line.strip() and not line.startswith(" "):
                    break
                bin_size_section.append(line)

        bin_size_text = " ".join(bin_size_section).lower()

        # Should not mention "optional", "default", "defaults to"
        assert "optional" not in bin_size_text, (
            "from_polygon() docstring should not describe bin_size as optional"
        )
        assert "default" not in bin_size_text or "by default" in bin_size_text, (
            "from_polygon() docstring should not mention a default value for bin_size"
        )

    def test_from_image_docstring_shows_required(self):
        """from_image() docstring should not say 'optional' or show 'default'."""
        docstring = Environment.from_image.__doc__

        # Check for bin_size parameter documentation
        assert "bin_size" in docstring

        # Should not indicate it's optional
        lines = docstring.split("\n")
        in_bin_size_section = False
        bin_size_section = []

        for line in lines:
            if "bin_size :" in line:
                in_bin_size_section = True
            elif in_bin_size_section:
                if line.strip() and not line.startswith(" "):
                    break
                bin_size_section.append(line)

        bin_size_text = " ".join(bin_size_section).lower()

        # Should not mention "optional", "default", "defaults to"
        assert "optional" not in bin_size_text, (
            "from_image() docstring should not describe bin_size as optional"
        )
        # Note: "Defaults to 1.0" should be removed
        assert "defaults to" not in bin_size_text, (
            "from_image() docstring should not mention a default value for bin_size"
        )
