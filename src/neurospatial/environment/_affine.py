"""Environment integration for affine transforms; primitives remain in ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial.ops.transforms import Affine2D, AffineND
from neurospatial.regions import Region, Regions

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def _apply_transform_to_environment(
    env: Environment,
    transform: AffineND | Affine2D,
    *,
    name: str | None = None,
    frame: str | None = None,
) -> Environment:
    """Transform owned cells, graphs, arenas and regions together."""

    # Validate dimensionality match
    transform_dims = transform.n_dims if isinstance(transform, AffineND) else 2
    if env.n_dims != transform_dims:
        raise ValueError(
            f"Transform dimensionality ({transform_dims}D) does not match "
            f"environment dimensionality ({env.n_dims}D)."
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

    from neurospatial.geometry._derive import transform_binning

    binning = transform_binning(env._binning, transform)
    transformed_centers = binning.bins.centers

    # Create new connectivity graph with updated node positions

    new_graph = env.connectivity.copy()
    for node_id in new_graph.nodes:
        new_graph.nodes[node_id]["pos"] = tuple(transformed_centers[node_id])

    # Recompute edge attributes (distance, vector, and angle_2d for 2D)
    for u, v in new_graph.edges:
        pos_u = np.array(new_graph.nodes[u]["pos"])
        pos_v = np.array(new_graph.nodes[v]["pos"])
        vec = pos_v - pos_u

        new_graph.edges[u, v]["vector"] = tuple(vec)

        # Recompute angle_2d for 2D environments only
        if env.n_dims == 2 and "angle_2d" in new_graph.edges[u, v]:
            angle = float(np.arctan2(vec[1], vec[0]))
            new_graph.edges[u, v]["angle_2d"] = angle

    for (u, v), distance in zip(
        binning.bins.pairs, binning.bins.pair_distances, strict=True
    ):
        new_graph.edges[u, v]["distance"] = float(distance)

    # Transform dimension_ranges (N-dimensional)
    transformed_dim_ranges = None
    if env.dimension_ranges is not None:
        # Generate all corner points of the N-dimensional bounding box
        # For N dims, we have 2^N corners (all combinations of low/high per dimension)
        n_dims = env.n_dims
        ranges = env.dimension_ranges

        # Generate corner points using itertools.product
        from itertools import product

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
        transformed_dim_ranges = [
            (transformed_corners[:, dim].min(), transformed_corners[:, dim].max())
            for dim in range(n_dims)
        ]

    tag = f"{env.layout_type}_transformed"
    params = {**env.layout_parameters, "transformed": True}
    from neurospatial.environment._publication import _Provenance, publish_derived

    return publish_derived(
        env,
        binning,
        connectivity=new_graph,
        dimension_ranges=transformed_dim_ranges,
        provenance=_Provenance(tag, env._provenance.category, params),
        name=name,
        frame=frame,
        regions=Regions(
            _transformed_region(region, transform) for region in env.regions.values()
        ),
        arena=env.arena.transformed(transform.A),
        clip_mapping=env.clip_mapping,
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
