"""Environment integration for affine transforms; primitives remain in ops."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial._exceptions import _format_error
from neurospatial.environment._publication import _Derivation
from neurospatial.geometry import Arena, Binning, cell_extent, transform_binning
from neurospatial.ops.transforms import Affine2D, AffineND
from neurospatial.regions import Region, Regions


def _apply_transform(
    transform: object,
    *,
    binning: Binning,
    regions: Regions,
    arena: Arena,
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

    transformed = transform_binning(binning, transform)

    # The transformed cells' own extent; bins without cells map their
    # declared box instead.
    transformed_dim_ranges = cell_extent(transformed, shapeless="none")
    dimension_ranges = binning.dimension_ranges
    if transformed_dim_ranges is None and dimension_ranges is not None:
        # Generate all corner points of the N-dimensional bounding box
        # For N dims, we have 2^N corners (all combinations of low/high per dimension)
        ranges = dimension_ranges
        corner_indices = list(product([0, 1], repeat=n_dims))  # All combinations of 0/1
        corners = np.array(
            [
                [ranges[dim][idx] for dim, idx in enumerate(corner)]
                for corner in corner_indices
            ]
        )

        # Transform all corners
        transformed_corners = transform(corners)

        # New bounding box: min/max along each dimension
        transformed_dim_ranges = tuple(
            (
                float(transformed_corners[:, dim].min()),
                float(transformed_corners[:, dim].max()),
            )
            for dim in range(n_dims)
        )

    return _Derivation(
        replace(transformed, dimension_ranges=transformed_dim_ranges),
        "apply_transform",
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
        # Type narrowing: region.data is a Polygon when kind == "polygon"
        if not isinstance(region.data, Polygon):
            raise TypeError(
                f"Region '{region.name}' has kind='polygon' but data is not a Polygon"
            )
        data = Polygon(
            transform(np.asarray(region.data.exterior.coords)),
            holes=[
                transform(np.asarray(ring.coords)) for ring in region.data.interiors
            ],
        )
    return Region(
        name=region.name,
        kind=region.kind,
        data=data,
        metadata={**region.metadata, "transformed": True},
    )
