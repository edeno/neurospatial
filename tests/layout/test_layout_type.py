"""Tests for the ``layout_type`` / ``is_grid_compatible`` metadata on layout engines.

Each layout engine exposes a stable ``layout_type`` tag used by rendering
code to decide whether the field can be rasterized as a 2-D image. We pin
the tag and the ``is_grid_compatible`` flag for every registered layout
plus a smoke test that the napari rendering path actually consumes the
flag on a grid env.

The previous version of this file had a per-layout TestLayoutTypeProperty
class that duplicated the parametrized TestLayoutTypeConsistency coverage,
plus a ``hasattr(LayoutEngine, "layout_type")`` test that's tautological
once the parametrized check runs.
"""

import numpy as np
import pytest


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
def test_layout_type_values(layout_kind, expected_type):
    """Each layout reports its canonical ``layout_type`` tag."""
    if layout_kind == "Graph":
        pytest.importorskip("track_linearization")
    if layout_kind == "ShapelyPolygon":
        pytest.importorskip("shapely")

    layout_class = _get_layout_class(layout_kind)
    layout = layout_class()
    assert layout.layout_type == expected_type


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
def test_is_grid_compatible(layout_kind, expected_compatible):
    """``is_grid_compatible`` reflects whether the layout can rasterize."""
    if layout_kind == "Graph":
        pytest.importorskip("track_linearization")
    if layout_kind == "ShapelyPolygon":
        pytest.importorskip("shapely")

    layout_class = _get_layout_class(layout_kind)
    layout = layout_class()
    assert layout.is_grid_compatible == expected_compatible


def test_field_to_rgb_for_napari_grid_layout():
    """End-to-end: napari rendering path consumes ``is_grid_compatible``."""
    import matplotlib.pyplot as plt

    from neurospatial import Environment
    from neurospatial.animation.rendering import field_to_rgb_for_napari

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)
    assert env.layout.is_grid_compatible is True

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    field = rng.random(env.n_bins)

    rgb = field_to_rgb_for_napari(env, field, cmap_lookup, 0, 1)
    assert rgb.ndim == 3
    assert rgb.shape[2] == 3
