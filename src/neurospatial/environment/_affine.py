"""Environment integration for affine transforms; primitives remain in ops."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neurospatial.ops.transforms import Affine2D, AffineND

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def _apply_transform_to_environment(
    env: Environment,
    transform: AffineND | Affine2D,
    *,
    name: str | None = None,
) -> Environment:
    """Transform owned cells, graphs, arenas and regions together."""
    from neurospatial.environment import Environment
    from neurospatial.regions import Region, Regions

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

    from neurospatial.layout._derive import transform_binning

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
    new_name = name if name is not None else f"{env.name}_transformed"
    new_env = Environment._from_binning(
        binning,
        connectivity=new_graph,
        dimension_ranges=transformed_dim_ranges,
        layout_type_tag=tag,
        layout_type=env._layout_category,
        build_params=params,
        is_linearized_track=env.is_linearized_track,
        name=new_name,
        arena=env.arena.transformed(transform.A),
        clip_mapping=env.clip_mapping,
    )
    # Transform and copy regions
    if env.regions and len(env.regions) > 0:
        transformed_regions = []
        for region in env.regions.values():
            if region.kind == "point":
                # Transform point
                old_point = np.array(region.data)
                new_point = transform(old_point.reshape(1, -1))[0]
                new_region = Region(
                    name=region.name,
                    kind="point",
                    data=new_point,
                    metadata={**region.metadata, "transformed": True},
                )
            elif region.kind == "polygon":
                # Transform polygon
                import shapely.geometry as shp
                from shapely.geometry import Polygon

                # Type narrowing: region.data is a Polygon when kind == "polygon"
                if not isinstance(region.data, Polygon):
                    raise TypeError(
                        f"Region '{region.name}' has kind='polygon' but data is not a Polygon"
                    )
                old_coords = np.array(region.data.exterior.coords)
                new_coords = transform(old_coords)
                new_holes = [
                    transform(np.asarray(ring.coords)) for ring in region.data.interiors
                ]
                new_poly = shp.Polygon(new_coords, holes=new_holes)
                new_region = Region(
                    name=region.name,
                    kind="polygon",
                    data=new_poly,
                    metadata={**region.metadata, "transformed": True},
                )

            transformed_regions.append(new_region)

        new_env = new_env.with_regions(Regions(transformed_regions))

    # Copy units and frame
    if hasattr(env, "units"):
        new_env = new_env.with_metadata(units=env.units)
    if hasattr(env, "frame"):
        new_env = new_env.with_metadata(
            frame=f"{env.frame}_transformed" if env.frame else "transformed"
        )

    return new_env
