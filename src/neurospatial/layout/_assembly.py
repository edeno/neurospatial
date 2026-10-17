"""Assemble a built layout engine's owned ``Binning`` from its own state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry import (
    Binning,
    FaceNormals,
    GridBinningSpec,
    HexagonShapes,
    Locator,
    NativeFaces,
    Outline,
    RectilinearGrid,
    SpatialBins,
    TrackLinearization,
    TriangleShapes,
    geometric_normals,
)


def assemble_binning(
    connectivity: nx.Graph,
    *,
    kind: str,
    centers: NDArray[np.float64],
    sizes: NDArray[np.float64],
    measure_dim: int,
    locator: Locator,
    faces: Literal["grid", "polar", "hex", "mesh", "track"],
    face_arrays: tuple[NDArray[Any], ...] = (),
    n_angle: int = 0,
    circular: bool = False,
    face_normals: bool = True,
    rectilinear: RectilinearGrid | None = None,
    linearization: TrackLinearization | None = None,
    grid_spec: GridBinningSpec | None = None,
    shapes: HexagonShapes | TriangleShapes | None = None,
    outline: Outline | None = None,
    dimension_ranges: Sequence[tuple[float, float]] | None = None,
) -> Binning:
    """Snapshot bins in graph node order and pairs in edge order, faces lazily.

    Every layout engine's ``binning()`` passes its own state here, so all
    engines share one rule for neighbor-pair order, bin measures and face
    directions.

    Parameters
    ----------
    connectivity : networkx.Graph
        Built navigation graph; its edge iteration order becomes the
        authoritative pair order and each edge's ``"distance"`` its
        navigation distance.
    kind : str
        Descriptive layout name stored as ``Binning.kind``.
    centers : ndarray of float64, shape (n_bins, n_dims)
        Active bin centers, row ``i`` for graph node ``i``.
    sizes : ndarray of float64, shape (n_bins,)
        Bin measures in ``units ** measure_dim``.
    measure_dim : int
        One for track lengths; otherwise the coordinate dimension.
    locator : Locator
        Owned point-to-bin lookup data.
    faces : {"grid", "polar", "hex", "mesh", "track"}
        Native finite-volume formula evaluated lazily from ``face_arrays``.
    face_arrays : tuple of ndarray, optional
        Per-axis grid widths; mesh points, triangles and active triangle IDs;
        or per-pair track junction positions. Hexagons need none.
    n_angle : int, default=0
        Number of angular bins of a polar grid.
    circular : bool, default=False
        Whether a polar grid joins its first and last angular bins.
    face_normals : bool, default=True
        Derive unit face directions from the locator's cells for 2-D and 3-D
        bins. Pass False when cell coordinates are not Cartesian lengths,
        such as polar sectors.
    rectilinear, linearization, grid_spec, shapes, outline : optional
        Structural components the engine supplies from its own state.
    dimension_ranges : sequence of (float, float), optional
        Range of each coordinate axis the engine built its bins over.

    Returns
    -------
    Binning
        Validated bins with read-only arrays and the supplied components.
    """
    edges = list(connectivity.edges(data=True))
    pairs = np.array([(u, v) for u, v, _ in edges], dtype=np.intp).reshape(-1, 2)
    distances = np.array([data["distance"] for _, _, data in edges], dtype=np.float64)
    bins = SpatialBins.from_arrays(
        centers=np.asarray(centers, dtype=np.float64),
        sizes=sizes,
        measure_dim=measure_dim,
        pairs=pairs,
        pair_distances=distances,
    )
    normals = (
        geometric_normals(locator, bins.centers, bins.pairs)
        if face_normals and bins.n_dims in (2, 3)
        else None
    )
    return Binning(
        kind=kind,
        bins=bins,
        locator=locator,
        faces=NativeFaces(bins, faces, face_arrays, n_angle, circular),
        rectilinear=rectilinear,
        linearization=linearization,
        grid_spec=grid_spec,
        face_normals=FaceNormals(normals) if normals is not None else None,
        shapes=shapes,
        outline=outline,
        dimension_ranges=None if dimension_ranges is None else tuple(dimension_ranges),
    )
