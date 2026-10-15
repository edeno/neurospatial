"""Publish exact polygon clips through the normal environment constructor."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from neurospatial.environment._protocols import ClipConsumer
from neurospatial.layout._polygon import clip_binning

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment


def _clip(
    env: ClipConsumer,
    polygon: Polygon | MultiPolygon,
    *,
    name: str | None = None,
) -> Environment:
    """Intersect represented cells with an analysis polygon and known domain."""
    if env.n_dims != 2:
        raise NotImplementedError(
            "Exact polygon clipping requires Cartesian 2-D cells."
        )
    from neurospatial.layout._composition import bridge_values

    if bridge_values(env._binning):
        raise NotImplementedError(
            "Clipping navigation bridge endpoints needs an explicit reconnection rule. "
            "Combine with auto_bridge=False before clipping, or clip the children first."
        )
    domain = env.arena.polygonal_domain()
    if env.arena.domain_known and domain is None:
        raise NotImplementedError(
            "Exact clipping requires a polygonal representation of the known "
            "arena domain. Provide polygon geometry or use whole-bin subset()."
        )
    binning, mapping = clip_binning(
        env._binning,
        polygon,
        domain=domain,
        walls=env.arena.walls,
    )
    graph = nx.Graph()
    for index, center in enumerate(binning.bins.centers):
        graph.add_node(
            index,
            pos=tuple(center),
            source_grid_flat_index=index,
            original_grid_nd_index=(index,),
        )
    for index, (u, v) in enumerate(binning.bins.pairs):
        vector = binning.bins.centers[v] - binning.bins.centers[u]
        graph.add_edge(
            int(u),
            int(v),
            distance=float(binning.bins.pair_distances[index]),
            vector=tuple(vector),
            edge_id=index,
            angle_2d=float(np.arctan2(vector[1], vector[0])),
        )
    shapes = binning.shapes
    assert shapes is not None
    from neurospatial.layout._polygon import PolygonCells

    assert isinstance(shapes, PolygonCells)
    bounds = np.array([cell.bounds for cell in shapes.polygons])
    ranges = [
        (float(bounds[:, axis].min()), float(bounds[:, axis + 2].max()))
        for axis in range(2)
    ]
    env_cls = cast("type[Environment]", type(env))
    result = env_cls._from_binning(
        binning,
        connectivity=graph,
        dimension_ranges=ranges,
        layout_type_tag="PolygonCells",
        layout_type="polygon",
        build_params={"source": "clip", "original_n_bins": env.n_bins},
        is_linearized_track=False,
        name=name if name is not None else env.name,
        arena=env.arena,
        clip_mapping=mapping,
        units=env.units,
        frame=env.frame,
    )
    return result
