"""Affine face measures and certificates for additional two-point fluxes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import get_centers
from neurospatial.geometry._binning import KnownFaces, MissingFaces
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._face_normals import FaceNormals, _unit_rows

if TYPE_CHECKING:
    from neurospatial.geometry._binning import Binning, Locator

# A dimensionless tangential residual. This does not relax native mesh skew.
ORTHOGONALITY_TOLERANCE = 1e-12


def transform_normals(
    normals: FaceNormals, linear: NDArray[np.float64]
) -> tuple[FaceNormals, NDArray[np.float64]]:
    """Map directions and return log face-normal stretches, without area products."""
    vectors = normals.values @ np.linalg.inv(linear)
    scales = np.max(np.abs(vectors), axis=1)
    active = np.any(normals.values != 0, axis=1)
    if np.any(active & (scales == 0)) or not np.all(np.isfinite(vectors)):
        raise ValueError(
            "Transformed face directions are not representable; rescale coordinates."
        )
    scaled = np.divide(
        vectors, scales[:, None], out=np.zeros_like(vectors), where=scales[:, None] > 0
    )
    lengths = np.hypot.reduce(scaled, axis=1)
    unit = np.divide(
        scaled, lengths[:, None], out=np.zeros_like(scaled), where=lengths[:, None] > 0
    )
    logs = np.zeros(len(vectors))
    logs[active] = np.log(scales[active]) + np.log(lengths[active])
    return FaceNormals(unit), logs


def _center_vectors(binning: Binning) -> NDArray[np.float64]:
    """Read cell representatives without subtracting rounded world offsets."""
    bins = binning.bins
    frame = binning.grid_frame
    if frame is None:
        retained = _combined_vectors(binning.locator)
        if retained is not None:
            pairs, vectors = retained
            if not np.array_equal(pairs, bins.pairs):
                raise ValueError(
                    "Combined physical vectors must follow retained pair order."
                )
            return vectors
        return bins.centers[bins.pairs[:, 1]] - bins.centers[bins.pairs[:, 0]]
    source = frame.source
    indices = np.unravel_index(np.flatnonzero(source.active_mask), source.shape)
    centers = np.column_stack(
        [
            get_centers(edge)[index]
            for edge, index in zip(source.edges, indices, strict=True)
        ]
    )
    return np.asarray(
        (centers[bins.pairs[:, 1]] - centers[bins.pairs[:, 0]])
        @ frame.forward[:-1, :-1].T,
        dtype=np.float64,
    )


def _combined_vectors(
    locator: Locator,
) -> tuple[NDArray[np.intp], NDArray[np.float64]] | None:
    """Carry child vectors through wrappers without subtracting world offsets."""
    from neurospatial.geometry._composition import CombinedLocator
    from neurospatial.geometry._locators import SubsetLocator, TransformedLocator

    if isinstance(locator, CombinedLocator):
        bins = locator.spatial_bins()
        bridge_vectors = (
            bins.centers[locator.bridge_pairs[:, 1]]
            - bins.centers[locator.bridge_pairs[:, 0]]
        )
        return bins.pairs, np.concatenate(
            [*[_center_vectors(child) for child in locator.children], bridge_vectors]
        )
    if isinstance(locator, SubsetLocator):
        base = _combined_vectors(locator.base)
        if base is None:
            return None
        pairs, vectors = base
        mapped = locator.parent_to_child[pairs]
        keep = np.all(mapped >= 0, axis=1)
        return mapped[keep], vectors[keep]
    if isinstance(locator, TransformedLocator):
        base = _combined_vectors(locator.base)
        if base is None:
            return None
        pairs, vectors = base
        return pairs, vectors @ np.linalg.inv(locator.inverse)[:-1, :-1].T
    return None


def certify_transformed_faces(
    binning: Binning,
    old: FaceGeometry,
    linear: NDArray[np.float64],
    logdet: float,
    normals: FaceNormals,
    log_normal_stretch: NDArray[np.float64],
    volumes: NDArray[np.float64],
    *,
    kind: str,
) -> KnownFaces | MissingFaces:
    """Enable new transforms only with an orthogonal, located source metric.

    Known similarities and native diagonal scaling keep their existing path.
    A prior MissingFaces record can never be recertified by normals alone.
    """
    source_normals = binning.face_normals
    assert source_normals is not None
    positive = old.A > 0
    before = _center_vectors(binning)
    after = before @ linear.T
    lengths_before = np.hypot.reduce(before, axis=1)
    lengths_after = np.hypot.reduce(after, axis=1)
    unit_before, unit_after = _unit_rows(before), _unit_rows(after)
    old_projection = np.sum(unit_before * source_normals.values, axis=1)
    new_projection = np.sum(unit_after * normals.values, axis=1)
    old_residual = np.hypot.reduce(
        unit_before - old_projection[:, None] * source_normals.values, axis=1
    )
    new_residual = np.hypot.reduce(
        unit_after - new_projection[:, None] * normals.values, axis=1
    )
    valid = (~positive) | (
        (old_projection > 0)
        & (new_projection > 0)
        & (old_residual <= ORTHOGONALITY_TOLERANCE)
        & (new_residual <= ORTHOGONALITY_TOLERANCE)
        & np.isclose(old.distance, lengths_before, rtol=ORTHOGONALITY_TOLERANCE, atol=0)
    )
    valid &= (lengths_before > 0) & (lengths_after > 0)
    if not np.all(valid):
        index = int(np.flatnonzero(~valid)[0])
        u, v = old.edges[index]
        return MissingFaces(
            f"The {kind} fails the right angle (orthogonality) or source flux-distance certificate at face {index} ({u}, {v}) "
            f"(tangential tolerance {ORTHOGONALITY_TOLERANCE:g}). "
            "Use a similarity affine transform, recenter unresolved coordinates, or rebuild certified cells."
        )
    areas = np.zeros_like(old.A)
    areas[positive] = np.exp(
        np.log(old.A[positive]) + logdet + log_normal_stretch[positive]
    )
    distances = old.distance * (lengths_after / lengths_before)
    distances[positive] = lengths_after[positive]
    if np.any(positive & ((areas <= 0) | ~np.isfinite(areas))):
        raise ValueError(
            "Transformed positive face areas are not representable; rescale coordinates."
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        conductance = areas / distances
        rates = conductance / np.minimum(
            volumes[old.edges[:, 0]], volumes[old.edges[:, 1]]
        )
    if (
        np.any(~np.isfinite(conductance))
        or np.any(~np.isfinite(rates))
        or np.any(positive & ((conductance == 0) | (rates == 0)))
    ):
        return MissingFaces(
            "Transformed face conductances are not representable; rescale coordinates before numerical operations."
        )
    return KnownFaces(FaceGeometry(old.edges, areas, distances, volumes))
