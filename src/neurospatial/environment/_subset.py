"""Select owned bins while preserving their existing navigation geometry."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import shapely
import shapely.geometry.base
from numpy.typing import NDArray
from shapely import contains_xy

from neurospatial.environment._publication import _Derivation, _Provenance
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._components import HexagonShapes, TriangleShapes
from neurospatial.geometry._derive import subset_binning
from neurospatial.geometry._polygon import PolygonCells
from neurospatial.regions import Regions


def _subset_mask(
    bin_centers: NDArray[np.float64],
    regions: Regions,
    *,
    bins: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
    invert: bool = False,
) -> NDArray[np.bool_]:
    """Validate a spatial selection and build its active-bin mask."""
    n_bins = len(bin_centers)
    # --- Input Validation ---

    # Exactly one selection parameter must be provided
    n_params = sum([bins is not None, region_names is not None, polygon is not None])
    if n_params == 0:
        raise ValueError(
            "Exactly one of {bins, region_names, polygon} must be provided."
        )
    if n_params > 1:
        raise ValueError(
            "Exactly one of {bins, region_names, polygon} must be provided. "
            f"Got {n_params} parameters."
        )

    # --- Build Selection Mask ---

    if bins is not None:
        # Validate bins parameter
        bins = np.asarray(bins)

        # Check dtype
        if bins.dtype != bool:
            raise ValueError(
                f"bins must be boolean array (dtype=bool), got dtype={bins.dtype}"
            )

        # Check shape
        if bins.shape != (n_bins,):
            raise ValueError(
                f"bins must have shape (n_bins,) = ({n_bins},), got shape {bins.shape}"
            )

        mask = bins

    elif region_names is not None:
        # Validate region_names parameter
        if not isinstance(region_names, (list, tuple)):
            raise ValueError(
                f"region_names must be a list or tuple, got {type(region_names)}"
            )

        if len(region_names) == 0:
            raise ValueError("region_names cannot be empty")

        # Check all regions exist
        for name in region_names:
            if name not in regions:
                available = list(regions.keys())
                raise ValueError(
                    f"Region '{name}' not found in environment. "
                    f"Available regions: {available}"
                )

        # Build mask from regions
        mask = np.zeros(n_bins, dtype=bool)
        for name in region_names:
            region = regions[name]
            if region.kind == "point":
                raise ValueError(
                    f"Region '{name}' is a point-type region. "
                    "subset() only supports polygon-type regions. "
                    "Use a boolean mask (bins parameter) to select bins containing specific points."
                )
            elif region.kind == "polygon":
                if bin_centers.shape[1] != 2:
                    raise ValueError(
                        f"Polygon regions only work for 2D environments. "
                        f"This environment has {bin_centers.shape[1]} dimensions. "
                        "Use bins parameter for N-dimensional selection."
                    )

                # Vectorized containment check (much faster than loop)
                in_region = contains_xy(
                    region.data, bin_centers[:, 0], bin_centers[:, 1]
                )
                mask |= in_region

    elif polygon is not None:
        # Validate polygon parameter
        try:
            # Type check
            if not isinstance(polygon, shapely.geometry.base.BaseGeometry):
                raise TypeError(
                    f"polygon must be a Shapely geometry object, got {type(polygon)}"
                )

            # Dimension check
            if bin_centers.shape[1] != 2:
                raise ValueError(
                    f"Polygon selection only works for 2D environments. "
                    f"This environment has {bin_centers.shape[1]} dimensions."
                )

            # Vectorized containment check (150x faster than Python loop)
            mask = contains_xy(polygon, bin_centers[:, 0], bin_centers[:, 1])

        except (AttributeError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid polygon: {e}") from e

    else:
        # Should never reach here due to earlier validation
        raise RuntimeError("No selection method specified (should be unreachable)")

    # Apply invert if requested
    if invert:
        mask = ~mask

    # Check that selection is not empty
    if not np.any(mask):
        raise ValueError(
            f"No bins selected. Selection resulted in empty mask. (invert={invert})"
        )

    return mask


def select_bins(
    binning: Binning, connectivity: nx.Graph, mask: NDArray[np.bool_]
) -> tuple[Binning, nx.Graph, list[tuple[float, float]]]:
    """Keep the selected bins of a binning, with their graph and bounds.

    Parameters
    ----------
    binning : Binning
        Bins to select from.
    connectivity : networkx.Graph
        Navigation graph over the bins of ``binning``; it is not modified.
    mask : ndarray of bool, shape (n_bins,)
        Bins to keep.

    Returns
    -------
    binning : Binning
        The kept bins, renumbered ``0..n_kept-1`` in their original order.
    graph : networkx.Graph
        Induced graph, in the parent's node and edge order. Grid selections
        number ``edge_id`` from 0.
    dimension_ranges : list of (float, float)
        Grid extent for grids; otherwise the kept cells' bounds.
    """
    binning = subset_binning(binning, mask)
    selected = np.flatnonzero(mask)
    child_of = {int(old): new for new, old in enumerate(selected)}
    graph = nx.Graph()
    graph.graph.update(connectivity.graph)
    # Retain the parent's node iteration order as well as edge order and
    # orientation; edge fields index that order even on loaded graphs.
    for old, data in connectivity.nodes(data=True):
        if old in child_of:
            graph.add_node(child_of[old], **data)
    for index, (u, v) in enumerate(binning.bins.pairs):
        old_u, old_v = selected[u], selected[v]
        data = dict(connectivity.edges[old_u, old_v])
        if binning.rectilinear is not None:
            data["edge_id"] = index
        graph.add_edge(int(u), int(v), **data)
    if binning.rectilinear is not None:
        ranges = [
            (float(edge[0]), float(edge[-1])) for edge in binning.rectilinear.edges
        ]
    else:
        shapes = binning.shapes
        vertices = binning.bins.centers
        if binning.grid_frame is not None:
            vertices = binning.grid_frame.corners()
        elif isinstance(shapes, HexagonShapes):
            angles = shapes.orientation + np.pi / 2 + np.arange(6) * np.pi / 3
            offsets = shapes.radius * np.column_stack((np.cos(angles), np.sin(angles)))
            vertices = (shapes.centers[:, None, :] + offsets).reshape(-1, 2)
        elif isinstance(shapes, TriangleShapes):
            vertices = shapes.points[shapes.triangles].reshape(-1, 2)
        elif isinstance(shapes, PolygonCells):
            vertices = np.concatenate(
                [shapely.get_coordinates(cell) for cell in shapes.polygons]
            )
        ranges = []
        for axis in vertices.T:
            low, high = float(axis.min()), float(axis.max())
            margin = 0.05 * (high - low) if shapes is not None else 0.0
            ranges.append((low - margin, high + margin))
    return binning, graph, ranges


def _subset(
    binning: Binning,
    connectivity: nx.Graph,
    regions: Regions,
    *,
    bins: NDArray[np.bool_] | None = None,
    region_names: Sequence[str] | None = None,
    polygon: shapely.Polygon | None = None,
    invert: bool = False,
) -> _Derivation:
    """Derive selected bins and their graph and bounds."""
    mask = _subset_mask(
        binning.bins.centers,
        regions,
        bins=bins,
        region_names=region_names,
        polygon=polygon,
        invert=invert,
    )
    selected, graph, ranges = select_bins(binning, connectivity, mask)
    params = {"source": "subset", "original_n_bins": binning.bins.n_bins}
    grid = selected.rectilinear is not None
    return _Derivation(
        selected,
        graph,
        ranges,
        _Provenance(
            "MaskedGrid" if grid else "subset", "mask" if grid else "subset", params
        ),
    )
