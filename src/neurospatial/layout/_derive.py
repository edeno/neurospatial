"""Derive owned binning data while preserving valid geometric operations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.affinity import affine_transform

from neurospatial._exceptions import _format_error
from neurospatial.layout._binning import (
    Binning,
    FacesSource,
    KnownFaces,
    LayoutLocator,
    Locator,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.layout._capabilities import (
    RectilinearGrid,
    TrackLinearizationProvider,
)
from neurospatial.layout._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.layout._diffusion import FaceGeometry
from neurospatial.layout._locators import (
    AffineMatrix,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
    _check_homogeneous,
)


def _refusal(what: str, why: str, fix: str) -> ValueError:
    return ValueError(_format_error(what, why=f"Why: {why}", fix=fix))


def _restricted_mask(
    mask: NDArray[np.bool_], keep: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    selected: NDArray[np.bool_] = np.array(mask, dtype=np.bool_, copy=True)
    selected.ravel()[np.flatnonzero(mask)] = keep
    return selected


@dataclass(frozen=True, eq=False)
class _LegacyTrackLinearization:
    """Retain a custom provider's conversions through an in-memory derivation.

    Parameters
    ----------
    base : TrackLinearizationProvider
        Custom conversions in the parent coordinate frame.
    grid_edges : tuple of ndarray
        Published full linear edges, or an empty tuple when unavailable.
    active_mask : ndarray of bool
        Published active linear cells, or all current bins without grid metadata.
    forward : ndarray, shape (n_dims + 1, n_dims + 1)
        Homogeneous map from the parent's Cartesian frame to this frame.
    scale : float
        Positive similarity scale for the parent's absolute linear coordinates.
    """

    base: TrackLinearizationProvider
    grid_edges: tuple[NDArray[np.float64], ...]
    active_mask: NDArray[np.bool_]
    forward: NDArray[np.float64]
    scale: float
    inverse: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        edges = tuple(
            np.array(edge, dtype=np.float64, copy=True) for edge in self.grid_edges
        )
        mask = np.array(self.active_mask, dtype=np.bool_, copy=True)
        forward = np.array(self.forward, dtype=np.float64, copy=True)
        inverse = np.asarray(np.linalg.inv(forward), dtype=np.float64)
        for array in (*edges, mask, forward, inverse):
            array.setflags(write=False)
        object.__setattr__(self, "grid_edges", edges)
        object.__setattr__(self, "active_mask", mask)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "inverse", inverse)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore array protection after pickle or deepcopy."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        for array in (*self.grid_edges, self.active_mask, self.forward, self.inverse):
            array.setflags(write=False)

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(n_points, n_dims)`` points through the parent provider."""
        points = _mapped(np.asarray(data_points, dtype=np.float64), self.inverse)
        return np.asarray(self.base.to_linear(points), dtype=np.float64) * self.scale

    def linear_to_nd(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map ``(n_points,)`` linear coordinates through the parent provider."""
        coordinates = np.asarray(linear_coordinates, dtype=np.float64) / self.scale
        points = np.asarray(self.base.linear_to_nd(coordinates), dtype=np.float64)
        return _mapped(points, self.forward)


def _legacy_track_view(
    track: TrackLinearizationProvider, locator: Locator, n_bins: int, n_dims: int
) -> _LegacyTrackLinearization:
    """Capture optional custom grid metadata without requiring that capability."""
    if isinstance(track, _LegacyTrackLinearization):
        return track
    leaf = locator
    while isinstance(leaf, (SubsetLocator, TransformedLocator)):
        leaf = leaf.base
    sources: list[object] = [track]
    if isinstance(leaf, LayoutLocator):
        sources.append(leaf.layout)
    edges: tuple[NDArray[np.float64], ...] = ()
    mask = np.ones(n_bins, dtype=np.bool_)
    for source in sources:
        supplied = getattr(source, "grid_edges", ())
        supplied_mask = np.asarray(getattr(source, "active_mask", None))
        if (
            not isinstance(supplied, (tuple, list))
            or len(supplied) != 1
            or supplied_mask.dtype.kind != "b"
        ):
            continue
        edge = np.asarray(supplied[0])
        if (
            edge.dtype.kind not in "iuf"
            or edge.ndim != 1
            or len(edge) < 2
            or not np.all(np.isfinite(edge))
            or np.any(np.diff(edge) <= 0)
            or supplied_mask.shape != (len(edge) - 1,)
            or np.count_nonzero(supplied_mask) != n_bins
        ):
            continue
        edges, mask = (np.asarray(edge, dtype=np.float64),), supplied_mask
        break
    return _LegacyTrackLinearization(track, edges, mask, np.eye(n_dims + 1), 1.0)


def subset_binning(binning: Binning, keep: NDArray[np.bool_]) -> Binning:
    """Select whole bins, retaining their measures and shared faces.

    Parameters
    ----------
    binning : Binning
        Parent geometry and point lookup.
    keep : ndarray of bool, shape (n_parent_bins,)
        Selected bins, retained in parent index order.

    Returns
    -------
    Binning
        Owned child geometry. Points in dropped bins are outside the child.
    """
    from neurospatial.layout._polygon import (
        ClipMapping,
        PolygonCells,
        subset_polygon_cells,
    )

    keep = np.asarray(keep)
    if (
        keep.dtype.kind != "b"
        or keep.shape != (binning.bins.n_bins,)
        or not np.any(keep)
    ):
        raise _refusal(
            "A bin subset needs a nonempty boolean selection of all parent bins.",
            "whole-bin selection preserves geometry only when each parent bin has one selection flag",
            f"pass a boolean mask of shape ({binning.bins.n_bins},) with at least one True entry",
        )
    parent_ids = np.flatnonzero(keep)
    child_of = np.full(len(keep), -1, dtype=np.intp)
    child_of[parent_ids] = np.arange(len(parent_ids))
    pairs = binning.bins.pairs
    pair_kept = keep[pairs[:, 0]] & keep[pairs[:, 1]]
    bins = SpatialBins.from_arrays(
        binning.bins.centers[parent_ids],
        binning.bins.sizes[parent_ids],
        binning.bins.measure_dim,
        child_of[pairs[pair_kept]],
        binning.bins.pair_distances[pair_kept],
    )
    faces: FacesSource = binning.faces
    if faces.available:
        old_faces = faces.build()
        faces = KnownFaces(
            FaceGeometry(
                bins.pairs,
                old_faces.A[pair_kept],
                old_faces.distance[pair_kept],
                bins.sizes,
            )
        )
    locator = (
        binning.locator.subset(keep)
        if isinstance(binning.locator, RectilinearLocator)
        else SubsetLocator(binning.locator, child_of)
    )
    grid = binning.rectilinear
    if grid is not None:
        grid = RectilinearGrid(
            grid.edges, grid.shape, _restricted_mask(grid.active_mask, keep)
        )
    shapes = binning.shapes
    if isinstance(shapes, HexagonShapes):
        shapes = HexagonShapes(bins.centers, shapes.radius, shapes.orientation)
    elif isinstance(shapes, TriangleShapes):
        shapes = TriangleShapes(shapes.points, shapes.triangles[parent_ids])
    elif isinstance(shapes, PolygonCells):
        shapes = subset_polygon_cells(shapes, keep)
    track = binning.linearization
    if track is not None:
        if isinstance(track, TrackLinearization):
            track = TrackLinearization(
                track.graph,
                track.edge_order,
                track.edge_spacing,
                track.grid_edges,
                _restricted_mask(track.active_mask, keep),
            )
        else:
            legacy = _legacy_track_view(
                track, binning.locator, binning.bins.n_bins, binning.bins.n_dims
            )
            track = _LegacyTrackLinearization(
                legacy,
                legacy.grid_edges,
                _restricted_mask(legacy.active_mask, keep)
                if legacy.grid_edges
                else np.ones(len(parent_ids), dtype=np.bool_),
                np.eye(binning.bins.n_dims + 1),
                1.0,
            )
    return Binning(
        "MaskedGrid" if grid is not None else "subset",
        bins,
        locator,
        faces,
        rectilinear=grid,
        linearization=track,
        shapes=shapes,
        outline=None,
        clip_mapping=(
            ClipMapping(
                binning.clip_mapping.parent_bin_indices[keep],
                binning.clip_mapping.n_parent_bins,
            )
            if binning.clip_mapping is not None
            else None
        ),
    )


def _matrix(transform: AffineMatrix, n_dims: int) -> NDArray[np.float64]:
    try:
        matrix = np.array(transform.A, dtype=np.float64, copy=True)
        _check_homogeneous(matrix)
        if matrix.shape != (n_dims + 1, n_dims + 1):
            raise ValueError(
                f"expected shape ({n_dims + 1}, {n_dims + 1}), got {matrix.shape}"
            )
    except (TypeError, ValueError) as exc:
        raise _refusal(
            f"Cannot derive {n_dims}-D binning from this affine matrix: {exc}.",
            "geometry scaling requires a finite invertible affine map with last row [0, ..., 0, 1]",
            "pass an affine transform with matching coordinate dimensions",
        ) from exc
    return matrix


def _classification(linear: NDArray[np.float64]) -> tuple[str, float, float]:
    """Classify with exact axes and a scale-independent Gram residual."""
    n_dims = len(linear)
    if np.array_equal(linear, np.eye(n_dims)):
        return "translation", 1.0, 0.0
    _, logdet = np.linalg.slogdet(linear)
    scale = math.exp(float(logdet) / n_dims)
    normalized = linear / scale
    # Strong anisotropy can overflow the Gram product without being singular.
    with np.errstate(over="ignore", invalid="ignore"):
        residual = np.max(np.abs(normalized @ normalized.T - np.eye(n_dims)))
    if residual <= 1e-12:
        return "similarity", scale, float(logdet)
    if np.array_equal(linear, np.diag(np.diag(linear))):
        return "axis-aligned scaling", scale, float(logdet)
    return "general affine transform", scale, float(logdet)


def _mapped(
    points: NDArray[np.float64], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.asarray(points @ matrix[:-1, :-1].T + matrix[:-1, -1], dtype=np.float64)


def _track_transform(
    track: TrackLinearization, matrix: NDArray[np.float64], scale: float
) -> TrackLinearization:
    graph = track.graph.copy()
    for _node, attrs in graph.nodes(data=True):
        attrs["pos"] = tuple(_mapped(np.asarray(attrs["pos"])[None, :], matrix)[0])
    for _, _, attrs in graph.edges(data=True):
        attrs["distance"] *= scale
    spacing = track.edge_spacing
    spacing = (
        spacing * scale
        if isinstance(spacing, (int, float))
        else tuple(value * scale for value in spacing)
    )
    return TrackLinearization(
        graph,
        track.edge_order,
        spacing,
        tuple(edge * scale for edge in track.grid_edges),
        track.active_mask,
    )


def transform_binning(binning: Binning, transform: AffineMatrix) -> Binning:
    """Apply an affine map while retaining only supported geometric components.

    Parameters
    ----------
    binning : Binning
        Parent bins, point lookup, cell faces and structural components.
    transform : object with ndarray attribute A
        Homogeneous affine matrix of shape ``(n_dims + 1, n_dims + 1)``.

    Returns
    -------
    Binning
        Derived geometry with inverse point lookup. Unsupported finite-volume
        faces carry an explicit reason, and unsupported shapes are omitted.

    Raises
    ------
    ValueError
        If the map is invalid, measures are not representable, or a track is
        transformed by a map other than a translation or similarity.
    """
    try:
        # Subnormal rotation entries can underflow in otherwise valid Gram or
        # determinant arithmetic; stored measures are checked explicitly below.
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _transform_binning(binning, transform)
    except (FloatingPointError, OverflowError) as exc:
        raise _refusal(
            "Transformed geometry exceeds floating-point range.",
            "bin centers, measures and face geometry must remain finite and representable",
            "rescale the coordinate units before applying the transform",
        ) from exc


def _transform_binning(binning: Binning, transform: AffineMatrix) -> Binning:
    from neurospatial.layout._polygon import PolygonCells, transform_polygon_cells

    parent = binning.bins
    matrix = _matrix(transform, parent.n_dims)
    linear = matrix[:-1, :-1]
    kind, scale, logdet = _classification(linear)
    similarity = kind in {"translation", "similarity"}
    if parent.measure_dim < parent.n_dims and not similarity:
        raise _refusal(
            f"Cannot apply {kind} to embedded track bins.",
            "non-similarity maps change bin lengths according to each segment's direction",
            "rebuild the track from transformed node positions with Environment.from_graph",
        )
    try:
        factor = (
            math.exp(math.log(scale) * parent.measure_dim)
            if similarity
            else math.exp(logdet)
        )
    except OverflowError as exc:
        raise _refusal(
            "Transformed bin measures exceed floating-point range.",
            "bin measures must remain finite and positive",
            "rescale the coordinate units before applying the transform",
        ) from exc
    if not np.isfinite(factor) or factor <= 0:
        raise _refusal(
            "Transformed bin measures are outside floating-point range.",
            "bin measures must remain finite and positive",
            "rescale the coordinate units before applying the transform",
        )
    centers = _mapped(parent.centers, matrix)
    sizes = parent.sizes * factor
    if (
        not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(sizes))
        or np.any(sizes <= 0)
    ):
        raise _refusal(
            "Transformed bins are outside floating-point range.",
            "bin centers and strictly positive measures must remain finite and representable",
            "rescale the coordinate units before applying the transform",
        )
    distances = np.linalg.norm(
        centers[parent.pairs[:, 1]] - centers[parent.pairs[:, 0]], axis=1
    )
    bins = SpatialBins.from_arrays(
        centers, sizes, parent.measure_dim, parent.pairs, distances
    )
    locator: Locator | None = (
        binning.locator.transformed(transform)
        if isinstance(binning.locator, RectilinearLocator)
        else None
    )
    if locator is None:
        locator = TransformedLocator(
            binning.locator, np.asarray(np.linalg.inv(matrix), dtype=np.float64)
        )
    grid = binning.rectilinear
    if grid is not None:
        diagonal = np.diag(linear)
        if np.array_equal(linear, np.diag(diagonal)) and np.all(diagonal > 0):
            grid = RectilinearGrid(
                tuple(
                    diagonal[axis] * edge + matrix[axis, -1]
                    for axis, edge in enumerate(grid.edges)
                ),
                grid.shape,
                grid.active_mask,
            )
        else:
            grid = None
    faces: FacesSource = binning.faces
    if faces.available:
        old = faces.build()
        if similarity:
            faces = KnownFaces(
                FaceGeometry(
                    bins.pairs,
                    old.A * scale ** (parent.measure_dim - 1),
                    old.distance * scale,
                    bins.sizes,
                )
            )
        elif kind == "axis-aligned scaling" and binning.rectilinear is not None:
            vectors = (
                parent.centers[parent.pairs[:, 1]] - parent.centers[parent.pairs[:, 0]]
            )
            axis = np.argmax(np.abs(vectors), axis=1)
            stretch = np.abs(np.diag(linear))[axis]
            face_stretch = stretch.copy()
            diagonal_links = old.A == 0
            face_stretch[diagonal_links] = distances[diagonal_links] / np.linalg.norm(
                vectors[diagonal_links], axis=1
            )
            faces = KnownFaces(
                FaceGeometry(
                    bins.pairs,
                    old.A * factor / stretch,
                    old.distance * face_stretch,
                    bins.sizes,
                )
            )
        else:
            faces = MissingFaces(
                f"The {kind} (including shear for a general affine map) loses the right angle "
                "between center links and shared cell faces; the two-point flux geometry is no longer valid."
            )
    shapes = binning.shapes
    if isinstance(shapes, HexagonShapes):
        shapes = (
            HexagonShapes(
                bins.centers,
                shapes.radius * scale,
                shapes.orientation
                + math.atan2(linear[1, 0] / scale, linear[0, 0] / scale),
            )
            if similarity and np.linalg.slogdet(linear)[0] > 0
            else None
        )
    elif isinstance(shapes, TriangleShapes):
        shapes = TriangleShapes(_mapped(shapes.points, matrix), shapes.triangles)
    elif isinstance(shapes, PolygonCells):
        shapes = transform_polygon_cells(shapes, matrix)
    outline = binning.outline
    if outline is not None:
        if parent.n_dims != 2:
            raise _refusal(
                "Polygon outlines need two-dimensional affine coordinates.",
                "the stored outline is a planar polygon",
                "use a two-dimensional transform for this environment",
            )
        outline = Outline(
            affine_transform(
                outline.geometry,
                [
                    linear[0, 0],
                    linear[0, 1],
                    linear[1, 0],
                    linear[1, 1],
                    matrix[0, -1],
                    matrix[1, -1],
                ],
            )
        )
    track = binning.linearization
    if track is not None:
        if isinstance(track, TrackLinearization):
            track = _track_transform(track, matrix, scale)
        else:
            legacy = _legacy_track_view(
                track, binning.locator, parent.n_bins, parent.n_dims
            )
            track = _LegacyTrackLinearization(
                legacy,
                tuple(edge * scale for edge in legacy.grid_edges),
                legacy.active_mask,
                matrix,
                scale,
            )
    return Binning(
        f"{binning.kind}_transformed",
        bins,
        locator,
        faces,
        rectilinear=grid,
        linearization=track,
        shapes=shapes,
        outline=outline,
        clip_mapping=binning.clip_mapping,
    )
