"""Tests for layout_type property on layout engines.

Phase 1.2: Normalize Layout Metadata

The layout_type property provides a standardized way to categorize layouts
for rendering optimization decisions:
- "grid": Regular rectangular grid (RegularGridLayout)
- "mask": Grid with active/inactive regions (MaskedGridLayout, ImageMaskLayout)
- "polygon": Polygon-bounded grid (ShapelyPolygonLayout)
- "hexagonal": Hexagonal tessellation
- "mesh": Point-based mesh (TriangularMeshLayout)
- "graph": 1D linearized path (GraphLayout)
- "other": Unknown or custom layouts
"""

import numpy as np
import pytest


class TestLayoutTypeProperty:
    """Tests for layout_type property on different layout engines."""

    def test_regular_grid_layout_type(self):
        """RegularGridLayout should have layout_type 'grid'."""
        from neurospatial.layout.engines.regular_grid import RegularGridLayout

        rng = np.random.default_rng(42)
        layout = RegularGridLayout()
        positions = rng.standard_normal((100, 2)) * 50
        layout.build(positions=positions, bin_size=10.0)

        assert hasattr(layout, "layout_type")
        assert layout.layout_type == "grid"

    def test_masked_grid_layout_type(self):
        """MaskedGridLayout should have layout_type 'mask'."""
        from neurospatial.layout.engines.masked_grid import MaskedGridLayout

        layout = MaskedGridLayout()
        # Create a simple mask
        mask = np.ones((5, 5), dtype=bool)
        mask[0, 0] = False  # One inactive bin
        layout.build(
            active_mask=mask,
            grid_edges=(np.arange(6), np.arange(6)),
        )

        assert hasattr(layout, "layout_type")
        assert layout.layout_type == "mask"

    def test_image_mask_layout_type(self):
        """ImageMaskLayout should have layout_type 'mask'."""
        from neurospatial.layout.engines.image_mask import ImageMaskLayout

        layout = ImageMaskLayout()
        # Create a simple boolean mask image
        mask_image = np.ones((10, 10), dtype=bool)
        mask_image[0:2, 0:2] = False  # Some inactive region
        layout.build(
            image_mask=mask_image,
            bin_size=1.0,
        )

        assert hasattr(layout, "layout_type")
        assert layout.layout_type == "mask"

    def test_shapely_polygon_layout_type(self):
        """ShapelyPolygonLayout should have layout_type 'polygon'."""
        pytest.importorskip("shapely")
        from shapely.geometry import Polygon

        from neurospatial.layout.engines.shapely_polygon import ShapelyPolygonLayout

        layout = ShapelyPolygonLayout()
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(polygon=polygon, bin_size=2.0)

        assert hasattr(layout, "layout_type")
        assert layout.layout_type == "polygon"

    def test_hexagonal_layout_type(self):
        """HexagonalLayout should have layout_type 'hexagonal'."""
        from neurospatial.layout.engines.hexagonal import HexagonalLayout

        rng = np.random.default_rng(42)
        layout = HexagonalLayout()
        positions = rng.standard_normal((50, 2)) * 20
        layout.build(positions=positions, hexagon_width=5.0)

        assert hasattr(layout, "layout_type")
        assert layout.layout_type == "hexagonal"

    def test_triangular_mesh_layout_type(self):
        """TriangularMeshLayout should have layout_type 'mesh'."""
        from shapely.geometry import Polygon

        from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout

        layout = TriangularMeshLayout()
        # Create simple boundary polygon for a square
        boundary_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        layout.build(boundary_polygon=boundary_polygon, point_spacing=3.0)

        assert hasattr(layout, "layout_type")
        assert layout.layout_type == "mesh"

    def test_graph_layout_type(self):
        """GraphLayout should have layout_type 'graph'."""
        pytest.importorskip("track_linearization")
        import networkx as nx

        from neurospatial.layout.engines.graph import GraphLayout

        layout = GraphLayout()
        # Simple 1D track graph
        graph = nx.Graph()
        graph.add_node(0, pos=(0.0, 0.0))
        graph.add_node(1, pos=(10.0, 0.0))
        graph.add_node(2, pos=(20.0, 0.0))
        graph.add_edge(0, 1, distance=10.0)
        graph.add_edge(1, 2, distance=10.0)
        edge_order = [(0, 1), (1, 2)]
        layout.build(
            graph_definition=graph,
            edge_order=edge_order,
            edge_spacing=1.0,
            bin_size=2.0,
        )

        assert hasattr(layout, "layout_type")
        assert layout.layout_type == "graph"


class TestLayoutTypeProtocol:
    """Tests for layout_type in the LayoutEngine protocol."""

    def test_layout_type_in_protocol(self):
        """layout_type should be part of the LayoutEngine protocol."""
        from neurospatial.layout.base import LayoutEngine

        # Check that layout_type is mentioned in the protocol
        # (either as property or attribute)
        assert hasattr(LayoutEngine, "layout_type")


def _get_layout_class(layout_kind: str):
    """Get layout class without building it."""
    from neurospatial.layout.engines.hexagonal import HexagonalLayout
    from neurospatial.layout.engines.image_mask import ImageMaskLayout
    from neurospatial.layout.engines.masked_grid import MaskedGridLayout
    from neurospatial.layout.engines.regular_grid import RegularGridLayout
    from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout

    layout_map = {
        "RegularGrid": RegularGridLayout,
        "MaskedGrid": MaskedGridLayout,
        "ImageMask": ImageMaskLayout,
        "Hexagonal": HexagonalLayout,
        "TriangularMesh": TriangularMeshLayout,
    }

    if layout_kind == "ShapelyPolygon":
        from neurospatial.layout.engines.shapely_polygon import ShapelyPolygonLayout

        return ShapelyPolygonLayout
    if layout_kind == "Graph":
        from neurospatial.layout.engines.graph import GraphLayout

        return GraphLayout

    return layout_map[layout_kind]


class TestLayoutTypeConsistency:
    """Tests for layout_type consistency across all registered layouts."""

    @pytest.mark.parametrize(
        "layout_kind",
        [
            "RegularGrid",
            "MaskedGrid",
            "ImageMask",
            "ShapelyPolygon",
            "Hexagonal",
            "TriangularMesh",
            "Graph",
        ],
    )
    def test_all_layouts_have_layout_type(self, layout_kind):
        """All layout engines should have a layout_type property."""
        # Skip if dependencies not available
        if layout_kind == "Graph":
            pytest.importorskip("track_linearization")
        if layout_kind == "ShapelyPolygon":
            pytest.importorskip("shapely")

        layout_class = _get_layout_class(layout_kind)
        layout = layout_class()  # Don't call build()
        assert hasattr(layout, "layout_type"), (
            f"{layout_kind} missing layout_type property"
        )

    @pytest.mark.parametrize(
        "layout_kind,expected_type",
        [
            ("RegularGrid", "grid"),
            ("MaskedGrid", "mask"),
            ("ImageMask", "mask"),
            ("ShapelyPolygon", "polygon"),
            ("Hexagonal", "hexagonal"),
            ("TriangularMesh", "mesh"),
            ("Graph", "graph"),
        ],
    )
    def test_layout_type_values(self, layout_kind, expected_type):
        """Each layout should return the expected layout_type value."""
        # Skip if dependencies not available
        if layout_kind == "Graph":
            pytest.importorskip("track_linearization")
        if layout_kind == "ShapelyPolygon":
            pytest.importorskip("shapely")

        layout_class = _get_layout_class(layout_kind)
        layout = layout_class()  # Don't call build()
        assert layout.layout_type == expected_type, (
            f"{layout_kind}.layout_type should be '{expected_type}', "
            f"got '{layout.layout_type}'"
        )


class TestLayoutTypeIsGridCompatible:
    """Tests for is_grid_compatible helper property.

    Some layouts can be rendered as 2D images (grid, mask, polygon).
    This property helps rendering code identify grid-compatible layouts.
    """

    @pytest.mark.parametrize(
        "layout_kind,expected_compatible",
        [
            ("RegularGrid", True),
            ("MaskedGrid", True),
            ("ImageMask", True),
            ("ShapelyPolygon", True),
            ("Hexagonal", False),
            ("TriangularMesh", False),
            ("Graph", False),
        ],
    )
    def test_is_grid_compatible(self, layout_kind, expected_compatible):
        """is_grid_compatible should reflect if layout can render as 2D image."""
        # Skip if dependencies not available
        if layout_kind == "Graph":
            pytest.importorskip("track_linearization")
        if layout_kind == "ShapelyPolygon":
            pytest.importorskip("shapely")

        layout_class = _get_layout_class(layout_kind)
        layout = layout_class()  # Don't call build()
        assert hasattr(layout, "is_grid_compatible"), (
            f"{layout_kind} missing is_grid_compatible property"
        )
        assert layout.is_grid_compatible == expected_compatible, (
            f"{layout_kind}.is_grid_compatible should be {expected_compatible}"
        )


class TestLayoutTypeInWidgetBackend:
    """Tests for layout_type usage in widget backend."""

    def test_field_to_image_data_uses_layout_type(self):
        """_field_to_image_data should use layout_type or is_grid_compatible."""
        import numpy as np

        from neurospatial import Environment

        rng = np.random.default_rng(42)
        # Create a regular grid environment
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Check that the layout has the required properties
        assert hasattr(env.layout, "layout_type")
        assert hasattr(env.layout, "is_grid_compatible")
        assert env.layout.is_grid_compatible is True

    def test_non_grid_layout_not_compatible(self):
        """Non-grid layouts should have is_grid_compatible=False."""
        from neurospatial.layout.engines.hexagonal import HexagonalLayout

        # is_grid_compatible is a static property, works before build()
        layout = HexagonalLayout()
        assert layout.is_grid_compatible is False


class TestLayoutTypeInRendering:
    """Tests for layout_type usage in rendering functions."""

    def test_field_to_rgb_for_napari_grid_layout(self):
        """field_to_rgb_for_napari should work with grid-compatible layouts."""
        import matplotlib.pyplot as plt
        import numpy as np

        from neurospatial import Environment
        from neurospatial.animation.rendering import field_to_rgb_for_napari

        rng = np.random.default_rng(42)
        # Create grid environment
        positions = rng.standard_normal((100, 2)) * 50
        env = Environment.from_samples(positions, bin_size=10.0)

        # Verify layout is grid-compatible
        assert env.layout.is_grid_compatible is True

        # Create colormap lookup
        cmap_obj = plt.get_cmap("viridis")
        cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        # Create field
        field = rng.random(env.n_bins)

        # Should return 2D RGB image
        rgb = field_to_rgb_for_napari(env, field, cmap_lookup, 0, 1)
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3
