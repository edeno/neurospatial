"""Environment integration for affine transforms; primitives remain in ops."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import product

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon

from neurospatial._exceptions import _format_error
from neurospatial.environment._cache import AnalysisCache
from neurospatial.environment._fields import with_cached_faces
from neurospatial.environment._publication import _Derivation
from neurospatial.environment._regions import _relabeled_region
from neurospatial.geometry import (
    Arena,
    Binning,
    cell_extent,
    mapped_geometry,
    transform_binning,
)
from neurospatial.geometry._derive import _classification
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.ops.transforms import AffineND
from neurospatial.regions import Region, Regions


def _apply_transform(
    transform: object,
    *,
    binning: Binning,
    regions: Mapping[str, Region],
    landmarks: Landmarks,
    arena: Arena,
    cache: AnalysisCache,
    units: str | None,
    old_units: str | None,
) -> _Derivation:
    """Transform owned cells, their bounds, arenas, regions and landmarks together.

    ``units`` is the unit label of the transformed coordinates and
    ``old_units`` the label before; see :func:`_transformed_region` for how
    regions that record their units in ``metadata["coord_system"]`` follow.
    """
    n_dims = binning.bins.n_dims
    if not isinstance(transform, AffineND):
        shape = np.shape(transform) if isinstance(transform, np.ndarray) else None
        given = (
            f"a NumPy array of shape {shape}"
            if shape is not None
            else type(transform).__name__
        )
        raise TypeError(
            _format_error(
                f"apply_transform() takes an AffineND transform, got {given}.",
                why="Why: the transform object says the matrix is homogeneous "
                f"(shape ({n_dims + 1}, {n_dims + 1}) for {n_dims}-D bins, with "
                "the translation in the last column) and supplies the inverse "
                "used to keep bins, arena and regions consistent.",
                fix="wrap the homogeneous matrix: from neurospatial.ops import "
                "AffineND; env.apply_transform(AffineND(matrix)), or build one "
                "with neurospatial.ops.transforms helpers such as translate(...)",
            )
        )

    # Validate dimensionality match
    transform_dims = transform.n_dims
    if n_dims != transform_dims:
        raise ValueError(
            _format_error(
                f"apply_transform() got a {transform_dims}-D transform for "
                f"{n_dims}-D bins.",
                why="Why: a transform maps points of its own dimension, so it "
                "cannot move bins with another number of coordinates.",
                fix=f"build a {n_dims}-D transform from a ({n_dims + 1}, "
                f"{n_dims + 1}) homogeneous matrix, such as the identity: from "
                "neurospatial.ops import AffineND; "
                f"env.apply_transform(AffineND(np.eye({n_dims + 1})))",
            )
        )

    transformed = transform_binning(with_cached_faces(cache, binning), transform)

    # Publication records the transformed cells' extent as the ranges; bins
    # without cells map the corners of their declared box instead.
    ranges = binning.dimension_ranges
    if ranges is not None and cell_extent(transformed, shapeless="none") is None:
        corners = transform(np.array(list(product(*ranges)), dtype=np.float64))
        transformed = replace(
            transformed,
            dimension_ranges=tuple(
                zip(
                    corners.min(axis=0).tolist(),
                    corners.max(axis=0).tolist(),
                    strict=True,
                )
            ),
        )

    # The build choices keep their lengths; record how much longer those
    # lengths are now, when the transform scales every direction alike.
    kind, scale, _ = _classification(np.asarray(transform.A)[:-1, :-1])
    keeps_lengths = kind in ("translation", "similarity") and abs(scale - 1.0) <= 1e-9
    return _Derivation(
        transformed,
        "apply_transform",
        length_scale=scale if kind in ("translation", "similarity") else None,
        regions=Regions(
            _transformed_region(
                region, transform, units, old_units, keeps_lengths=keeps_lengths
            )
            for region in regions.values()
        ),
        landmarks=Landmarks(
            _transformed_landmark(landmark, transform)
            for landmark in landmarks.values()
        ),
        arena=arena.transformed(transform.A),
    )


def _transformed_landmark(landmark: Landmark, transform: AffineND) -> Landmark:
    """Map a landmark's geometry through ``transform``, keeping its name."""
    data: NDArray[np.float64] | LineString | Polygon
    if isinstance(landmark.data, np.ndarray):
        data = transform(landmark.data.reshape(1, -1))[0]
    else:
        data = mapped_geometry(landmark.data, transform.A)
    return Landmark(landmark.name, landmark.kind, data, landmark.metadata)


def _transformed_region(
    region: Region,
    transform: AffineND,
    units: str | None,
    old_units: str | None,
    *,
    keeps_lengths: bool,
) -> Region:
    """Map a point or polygon region through ``transform``, marking it transformed.

    A transform that keeps lengths converts no units, so a recorded
    ``metadata["coord_system"]`` is relabelled as by ``with_metadata`` (see
    :func:`~neurospatial.environment._regions._relabeled_region`). One that
    changes lengths converts the region with the environment: the label
    becomes ``units``, or is dropped when there is no new label.
    """
    if keeps_lengths:
        region = _relabeled_region(region, old_units, units)
    if region.kind == "point":
        data: NDArray[np.float64] | Polygon = transform(
            np.asarray(region.data).reshape(1, -1)
        )[0]
    else:
        # Region construction guarantees a Polygon here. The arena and the
        # cells map their vertices the same way, so shared boundaries agree.
        data = mapped_geometry(region.data, transform.A)
    metadata = {**region.metadata, "transformed": True}
    if "coord_system" in metadata and not keeps_lengths:
        if units is not None:
            metadata["coord_system"] = units
        else:
            del metadata["coord_system"]
    return Region(name=region.name, kind=region.kind, data=data, metadata=metadata)
