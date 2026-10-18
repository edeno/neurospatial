"""Environment integration for affine transforms; primitives remain in ops."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial._exceptions import _format_error
from neurospatial.environment._cache import AnalysisCache
from neurospatial.environment._fields import with_cached_faces
from neurospatial.environment._publication import _Derivation
from neurospatial.geometry import (
    Arena,
    Binning,
    cell_extent,
    mapped_geometry,
    transform_binning,
)
from neurospatial.geometry._derive import _classification
from neurospatial.ops.transforms import Affine2D, AffineND
from neurospatial.regions import Region, Regions


def _apply_transform(
    transform: object,
    *,
    binning: Binning,
    regions: Regions,
    arena: Arena,
    cache: AnalysisCache,
) -> _Derivation:
    """Transform owned cells, their bounds, arenas and regions together."""
    n_dims = binning.bins.n_dims
    if not isinstance(transform, (AffineND, Affine2D)):
        shape = np.shape(transform) if isinstance(transform, np.ndarray) else None
        given = (
            f"a NumPy array of shape {shape}"
            if shape is not None
            else type(transform).__name__
        )
        raise TypeError(
            _format_error(
                f"apply_transform() takes an AffineND or Affine2D transform, got "
                f"{given}.",
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
    transform_dims = transform.n_dims if isinstance(transform, AffineND) else 2
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

    # Containment is evaluated in the original layout's coordinate frame.
    # A singular transform collapses cells, so there is no unambiguous inverse.
    try:
        transform.inverse()
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Environment transform must be invertible.\n"
            "Why: a singular transform collapses bins and cannot preserve containment.\n"
            "Fix: use nonzero scale factors and an invertible affine matrix."
        ) from exc

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
    return _Derivation(
        transformed,
        "apply_transform",
        length_scale=scale if kind in ("translation", "similarity") else None,
        regions=Regions(
            _transformed_region(region, transform) for region in regions.values()
        ),
        arena=arena.transformed(transform.A),
    )


def _transformed_region(region: Region, transform: AffineND | Affine2D) -> Region:
    """Map a point or polygon region through ``transform``, marking it transformed."""
    if region.kind == "point":
        data: NDArray[np.float64] | Polygon = transform(
            np.asarray(region.data).reshape(1, -1)
        )[0]
    else:
        # Region construction guarantees a Polygon here. The arena and the
        # cells map their vertices the same way, so shared boundaries agree.
        data = mapped_geometry(region.data, transform.A)
    return Region(
        name=region.name,
        kind=region.kind,
        data=data,
        metadata={**region.metadata, "transformed": True},
    )
