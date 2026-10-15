"""Construction adapters extracting owned geometry from built layouts."""

from __future__ import annotations

from typing import cast

import networkx as nx
import numpy as np

from neurospatial.layout._binning import (
    Binning,
    FacesSource,
    LayoutFaces,
    LayoutLocator,
    Locator,
    SpatialBins,
)
from neurospatial.layout._capabilities import (
    RectilinearGridProvider,
    TrackLinearizationProvider,
)
from neurospatial.layout._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.layout._face_normals import FaceNormals, geometric_normals
from neurospatial.layout._rebin_spec import GridBinningSpec
from neurospatial.layout.base import LayoutEngine


def binning_from_layout(layout: LayoutEngine, connectivity: nx.Graph) -> Binning:
    """Snapshot a built layout's bins and validated connectivity capabilities.

    Parameters
    ----------
    layout : LayoutEngine
        Built layout supplying bin coordinates, measures and point lookup.
    connectivity : networkx.Graph
        Validated environment connectivity in its published edge order.

    Returns
    -------
    Binning
        Owned bin arrays and adapters retaining the layout's capabilities.
    """
    factory = getattr(layout, "binning", None)
    if factory is not None:
        return cast("Binning", factory())
    edges = list(connectivity.edges(data=True))
    pairs = np.array([(u, v) for u, v, _ in edges], dtype=np.intp).reshape(-1, 2)
    distances = np.array([data["distance"] for _, _, data in edges], dtype=np.float64)
    centers = np.asarray(layout.bin_centers, dtype=np.float64)
    bins = SpatialBins.from_arrays(
        centers=centers,
        sizes=layout.bin_sizes(),
        measure_dim=1 if layout.is_linearized_track else centers.shape[1],
        pairs=pairs,
        pair_distances=distances,
    )
    grid = (
        layout.rectilinear_grid()
        if isinstance(layout, RectilinearGridProvider)
        else None
    )
    track = _linearization_of(layout)
    from neurospatial.layout.engines._polar import _PolarGridLayout

    spec = None
    if (
        not isinstance(cast("object", layout), _PolarGridLayout)
        and grid is not None
        and len(grid.edges) == bins.n_dims
        and track is None
        and np.allclose(
            bins.sizes,
            np.prod([edge[1] - edge[0] for edge in grid.edges]),
            rtol=64 * np.finfo(float).eps,
            atol=0,
        )
    ):
        spec = getattr(layout, "_grid_spec", None)
        if spec is None:
            # Manually constructed legacy layouts only establish their current
            # edge pattern; native builders supply the actual choice above.
            spec = GridBinningSpec(
                np.array([edge[0] for edge in grid.edges]),
                np.eye(bins.n_dims + 1),
                any(np.count_nonzero(centers[u] != centers[v]) > 1 for u, v in pairs),
            )
    locator = _locator_of(layout)
    normals = (
        geometric_normals(locator, bins.centers, bins.pairs)
        if not isinstance(cast("object", layout), _PolarGridLayout)
        and track is None
        and bins.n_dims in (2, 3)
        else None
    )
    return Binning(
        kind=getattr(layout, "_layout_type_tag", type(layout).__name__),
        bins=bins,
        locator=locator,
        faces=_native_faces_of(layout, bins, connectivity),
        rectilinear=grid,
        linearization=track,
        grid_spec=spec,
        face_normals=FaceNormals(normals) if normals is not None else None,
        shapes=_shapes_of(layout),
        outline=_outline_of(layout),
    )


def _locator_of(layout: LayoutEngine) -> Locator:
    """Use owned lookup data when a layout supplies it."""
    factory = getattr(layout, "locator", None)
    return cast("Locator", factory()) if factory is not None else LayoutLocator(layout)


def _linearization_of(layout: LayoutEngine) -> TrackLinearizationProvider | None:
    """Use owned track coordinates, retaining compatibility with custom layouts."""
    factory = getattr(layout, "linearization", None)
    if factory is not None:
        return cast("TrackLinearizationProvider", factory())
    return layout if isinstance(layout, TrackLinearizationProvider) else None


def _shapes_of(layout: LayoutEngine) -> HexagonShapes | TriangleShapes | None:
    """Read optional drawing shapes without knowing the layout class."""
    factory = getattr(layout, "bin_shapes", None)
    return cast("HexagonShapes | TriangleShapes", factory()) if factory else None


def _outline_of(layout: LayoutEngine) -> Outline | None:
    """Read an optional immutable boundary for drawing."""
    factory = getattr(layout, "outline", None)
    return cast("Outline | None", factory()) if factory else None


def _native_faces_of(layout: object, bins: SpatialBins, graph: nx.Graph) -> FacesSource:
    """Extract minimal native numerical data without materializing faces."""
    from neurospatial.layout._face_builders import track_junctions
    from neurospatial.layout._native_faces import NativeFaces
    from neurospatial.layout.engines._polar import _PolarGridLayout
    from neurospatial.layout.engines.graph import GraphLayout
    from neurospatial.layout.engines.hexagonal import HexagonalLayout
    from neurospatial.layout.engines.triangular_mesh import TriangularMeshLayout

    if isinstance(layout, _PolarGridLayout):
        assert layout.grid_shape is not None
        return NativeFaces(
            bins,
            "polar",
            (layout._axis_widths(),),
            layout.grid_shape[1],
            bool(layout._build_params_used["circular_angle"]),
        )
    if isinstance(layout, GraphLayout):
        junctions = track_junctions(
            graph, layout._build_params_used["graph_definition"], bins.n_dims
        )
        return NativeFaces(bins, "track", (junctions,))
    if isinstance(layout, HexagonalLayout):
        return NativeFaces(bins, "hex")
    if isinstance(layout, TriangularMeshLayout):
        tri = layout._full_delaunay_tri
        assert tri is not None
        return NativeFaces(
            bins,
            "mesh",
            (
                tri.points,
                tri.simplices,
                np.asarray(layout._active_original_simplex_indices),
            ),
        )
    if isinstance(layout, RectilinearGridProvider):
        grid = layout.rectilinear_grid()
        return NativeFaces(
            bins, "grid", (np.array([float(np.diff(edge)[0]) for edge in grid.edges]),)
        )
    return LayoutFaces(cast("LayoutEngine", layout), graph)
