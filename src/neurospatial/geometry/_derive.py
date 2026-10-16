"""Derive owned binning data while preserving valid geometric operations."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from shapely.affinity import affine_transform

from neurospatial._exceptions import _format_error
from neurospatial.geometry._binning import (
    Binning,
    FacesSource,
    KnownFaces,
    Locator,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.geometry._capabilities import (
    RectilinearGrid,
)
from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._explicit import cell_extent
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._face_transform import (
    certify_transformed_faces,
    transform_normals,
)
from neurospatial.geometry._grid_frame import FramedGrid
from neurospatial.geometry._locators import (
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
    from neurospatial.geometry._polygon import (
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
            raise TypeError(
                "Track derivation requires an owned TrackLinearization snapshot."
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
        grid_spec=binning.grid_spec,
        grid_frame=(
            FramedGrid(
                RectilinearGrid(
                    binning.grid_frame.source.edges,
                    binning.grid_frame.source.shape,
                    _restricted_mask(binning.grid_frame.source.active_mask, keep),
                ),
                binning.grid_frame.forward,
            )
            if binning.grid_frame is not None
            else None
        ),
        face_normals=(
            FaceNormals(binning.face_normals.values[pair_kept])
            if binning.face_normals is not None
            else None
        ),
        clip_mapping=(
            ClipMapping(
                binning.clip_mapping.parent_bin_indices[keep],
                binning.clip_mapping.n_parent_bins,
            )
            if binning.clip_mapping is not None
            else None
        ),
    )


def select_bins(binning: Binning, mask: NDArray[np.bool_]) -> Binning:
    """Keep the selected bins of a binning, with the bounds of what is kept.

    Parameters
    ----------
    binning : Binning
        Bins to select from.
    mask : ndarray of bool, shape (n_bins,)
        Bins to keep.

    Returns
    -------
    Binning
        The kept bins, renumbered ``0..n_kept-1`` in their original order.
        Its ``dimension_ranges`` is the extent of the kept cells, with no
        margin (see :func:`neurospatial.geometry._explicit.cell_extent`): the
        grid extent for grids, and the kept bin centers' bounds for bins
        without shapes.
    """
    binning = subset_binning(binning, mask)
    return replace(binning, dimension_ranges=cell_extent(binning, shapeless="centers"))


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
        transformed by a map other than a translation or similarity. Declared
        non-Euclidean navigation distances also require a similarity: their
        paths cannot be reconstructed from scalar lengths.
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
    from neurospatial.geometry._polygon import (
        PolygonCells,
        materialize_cells,
        transform_polygon_cells,
    )

    parent = binning.bins
    matrix = _matrix(transform, parent.n_dims)
    linear = matrix[:-1, :-1]
    kind, scale, logdet = _classification(linear)
    similarity = kind in {"translation", "similarity"}
    from neurospatial.geometry._composition import bridge_values

    couplings = bridge_values(binning)
    if any(value > 0 for value in couplings.values()):
        if not similarity:
            raise _refusal(
                "Cannot transform declared bridge conductance by a general affine map.",
                "conductance scaling requires a similarity with a known length scale",
                "use a translation, rotation, reflection or uniform scaling",
            )
        try:
            factor_c = scale ** (parent.measure_dim - 2)
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                transformed_c = np.array(list(couplings.values())) * factor_c
            if np.any(
                (np.array(list(couplings.values())) > 0) & (transformed_c == 0)
            ) or not np.all(np.isfinite(transformed_c)):
                raise FloatingPointError
        except (OverflowError, FloatingPointError) as error:
            raise _refusal(
                "Transformed bridge conductance exceeds floating-point range.",
                "positive conductances must remain finite and positive",
                "rescale coordinate units before transforming",
            ) from error
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
    original_distances = np.linalg.norm(
        parent.centers[parent.pairs[:, 1]] - parent.centers[parent.pairs[:, 0]],
        axis=1,
    )
    declared_paths = not np.allclose(
        parent.pair_distances,
        original_distances,
        rtol=64 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    if declared_paths:
        if not similarity:
            raise _refusal(
                f"Cannot apply {kind} to declared navigation path distances.",
                "scalar path lengths do not specify the paths needed for this transform",
                "use a translation, rotation, reflection or uniform scaling, or rebuild "
                "the bins with transformed path distances",
            )
        distances = parent.pair_distances * scale
        if np.any((parent.pair_distances > 0) & (distances == 0)):
            raise _refusal(
                "Transformed navigation path distances are outside floating-point range.",
                "positive path lengths must remain positive after scaling",
                "rescale the coordinate units before applying the transform",
            )
    else:
        # Keep the existing center-based calculation, including its rounding,
        # for native Euclidean connectivity.
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
    frame = binning.grid_frame
    if frame is not None:
        frame = FramedGrid(frame.source, matrix @ frame.forward)
    elif grid is None and binning.rectilinear is not None and parent.n_dims in (2, 3):
        frame = FramedGrid(binning.rectilinear, matrix)
    normal_data = (
        transform_normals(binning.face_normals, linear)
        if binning.face_normals is not None
        else None
    )
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
        elif normal_data is not None:
            faces = certify_transformed_faces(
                binning,
                old,
                linear,
                logdet,
                normal_data[0],
                normal_data[1],
                bins.sizes,
                kind=kind,
            )
        else:
            faces = MissingFaces(
                f"The {kind} (including shear for a general affine map) loses the right angle "
                "between center links and shared cell faces; the two-point flux geometry is no longer valid."
            )
        if isinstance(faces, KnownFaces) and np.any((old.A > 0) & (faces.faces.A == 0)):
            raise ValueError(
                "Transformed positive face areas underflow; rescale coordinate units."
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
            else transform_polygon_cells(
                PolygonCells(materialize_cells(binning)), matrix
            )
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
            raise TypeError(
                "Track derivation requires an owned TrackLinearization snapshot."
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
        grid_frame=frame,
        face_normals=normal_data[0] if normal_data is not None else None,
        grid_spec=(
            type(binning.grid_spec)(
                binning.grid_spec.origin,
                matrix @ binning.grid_spec.forward,
                binning.grid_spec.connect_diagonal,
            )
            if binning.grid_spec is not None
            else None
        ),
    )
