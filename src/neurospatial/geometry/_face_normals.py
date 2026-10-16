"""Owned physical face directions, separate from the four-field face tuple."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from neurospatial.geometry._binning import Binning, Locator


@dataclass(frozen=True, eq=False)
class FaceNormals:
    """Physical face normals in ``Binning.bins.pairs`` order.

    Parameters
    ----------
    values : ndarray, shape (n_edges, n_dims)
        Unit normals directed from each pair's first bin toward its second.
        Zero-area navigation edges have zero rows. Normal direction does not
        assert orthogonality of a center link or certify a diffusion method.
    """

    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        raw = np.asarray(self.values)
        if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] == 0:
            raise ValueError("face normals must be a real (n_edges, n_dims) array.")
        values = np.array(raw, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("face normals must be finite.")
        # A unit vector cannot have a component larger than one. Reject first
        # so malformed finite inputs cannot overflow the norm calculation.
        if np.any(np.abs(values) > 1 + 1e-12):
            raise ValueError("face normals must have unit length or be zero.")
        lengths = np.hypot.reduce(values, axis=1)
        if np.any((lengths != 0) & (np.abs(lengths - 1) > 1e-12)):
            raise ValueError("face normals must have unit length or be zero.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore write protection after pickle or deepcopy."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.values.setflags(write=False)


def _unit_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    lengths = np.hypot.reduce(values, axis=1)
    result = np.zeros_like(values)
    np.divide(values, lengths[:, None], out=result, where=lengths[:, None] > 0)
    return result


def geometric_normals(
    locator: Locator,
    centers: NDArray[np.float64],
    pairs: NDArray[np.intp],
) -> NDArray[np.float64] | None:
    """Read directions from supported cell geometry, without building faces.

    Returns zero for a geometrically zero-area contact, or None when no exact
    face directions are available. Never infer normals for nearest-only bins.
    """
    from shapely.ops import linemerge

    from neurospatial.geometry._locators import (
        HexLocator,
        MeshLocator,
        RectilinearLocator,
        SubsetLocator,
        TransformedLocator,
    )
    from neurospatial.geometry._polygon import ClippedLocator

    delta = centers[pairs[:, 1]] - centers[pairs[:, 0]]
    if isinstance(locator, SubsetLocator):
        return geometric_normals(locator.base, centers, pairs)
    if isinstance(locator, TransformedLocator):
        source = centers @ locator.inverse[:-1, :-1].T + locator.inverse[:-1, -1]
        normals = geometric_normals(locator.base, source, pairs)
        return (
            None if normals is None else _unit_rows(normals @ locator.inverse[:-1, :-1])
        )
    if isinstance(locator, RectilinearLocator):
        indices = np.column_stack(
            [
                np.searchsorted(edge, centers[:, d], side="right") - 1
                for d, edge in enumerate(locator.edges)
            ]
        )
        steps = indices[pairs[:, 1]] - indices[pairs[:, 0]]
        normals = np.zeros_like(delta)
        neighbors = np.sum(np.abs(steps), axis=1) == 1
        normals[neighbors] = steps[neighbors]
        return normals
    if isinstance(locator, HexLocator):
        # Recover lattice IDs from interior representatives, then use exact
        # relative lattice offsets. Inverse-mapped world centers can lose low
        # bits at large translations; their subtraction is not face geometry.
        active = np.flatnonzero(locator.flat_to_active >= 0)
        active = active[np.argsort(locator.flat_to_active[active])]
        ids = locator.locate(centers)
        if np.any(ids < 0):
            raise ValueError("hex centers must lie in their stored cells.")
        rows, columns = np.unravel_index(active[ids], locator.centers_shape)
        lattice = np.column_stack((2 * columns + rows % 2, 3 * rows))
        steps = lattice[pairs[:, 1]] - lattice[pairs[:, 0]]
        normals = _unit_rows(steps * np.array([np.sqrt(3), 1.0]))
        adjacent = ((np.abs(steps[:, 0]) == 2) & (steps[:, 1] == 0)) | (
            (np.abs(steps[:, 0]) == 1) & (np.abs(steps[:, 1]) == 3)
        )
        normals[~adjacent] = 0
        return normals
    if isinstance(locator, MeshLocator) and centers.shape[1] == 2:
        active = np.flatnonzero(locator.simplex_to_active >= 0)
        active = active[np.argsort(locator.simplex_to_active[active])]
        stored = locator.simplices[active]
        # Identify stored centroid rows without building the triangle lookup or
        # depending on Qhull's platform-specific simplex ordering. Publication
        # separately validates each representative against locator geometry.
        source_centers = locator.points[stored].mean(axis=1)
        ids = cKDTree(source_centers).query(centers)[1]
        triangles = stored[ids]
        normals = np.zeros_like(delta)
        for index, (u, v) in enumerate(pairs):
            shared = np.intersect1d(triangles[u], triangles[v])
            if len(shared) == 2:
                tangent = locator.points[shared[1]] - locator.points[shared[0]]
                normals[index] = [-tangent[1], tangent[0]]
    elif isinstance(locator, ClippedLocator):
        ids = locator.locate(centers)
        if np.any(ids < 0):
            raise ValueError("clipped centers must lie in their stored cells.")
        normals = np.zeros_like(delta)
        for index, (u, v) in enumerate(pairs):
            opening = locator.cells.polygons[ids[u]].boundary.intersection(
                locator.cells.polygons[ids[v]].boundary
            )
            if opening.geom_type == "MultiLineString":
                opening = linemerge(opening)
            if opening.geom_type == "LineString" and opening.length > 0:
                vertices = np.asarray(opening.coords)
                tangent = vertices[-1] - vertices[0]
                normals[index] = [-tangent[1], tangent[0]]
    else:
        return None
    normals = _unit_rows(normals)
    normals[np.sum(normals * delta, axis=1) < 0] *= -1
    return normals


def validate_normals(binning: Binning) -> None:
    """Check shape, known face activity, orientation and geometric direction.

    MissingFaces does not retain areas: a zero row may intentionally suppress
    a geometric contact as a navigation-only edge. In that case only nonzero
    directions can be cross-checked. Such data never enables numerical faces.
    """
    from neurospatial.geometry._binning import KnownFaces

    component = binning.face_normals
    if component is None:
        return
    FaceNormals(component.values)  # Revalidate mutable ndarray headers at ingress.
    bins = binning.bins
    values = component.values
    if bins.measure_dim != bins.n_dims or values.shape != (
        len(bins.pairs),
        bins.n_dims,
    ):
        raise ValueError(
            "face normals must match full-dimensional bins and pair order."
        )
    active = np.any(values != 0, axis=1)
    # A native face recipe is deliberately lazy, including its mesh
    # warning. Explicit/saved records always supply KnownFaces or MissingFaces.
    if isinstance(binning.faces, KnownFaces):
        areas = binning.faces.faces.A
        if not np.array_equal(active, areas > 0):
            raise ValueError(
                "face normals must be unit on positive faces and zero on zero-area edges."
            )
    delta = bins.centers[bins.pairs[:, 1]] - bins.centers[bins.pairs[:, 0]]
    if np.any(np.sum(values[active] * _unit_rows(delta[active]), axis=1) <= 0):
        raise ValueError(
            "face normals must point from each pair's first bin toward its second."
        )
    expected = geometric_normals(binning.locator, bins.centers, bins.pairs)
    if expected is not None and not np.allclose(
        values[active], expected[active], rtol=0, atol=1e-12
    ):
        raise ValueError(
            "face normals disagree with the locator's physical face directions."
        )
