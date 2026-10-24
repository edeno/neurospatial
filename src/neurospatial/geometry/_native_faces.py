"""Owned lazy native faces, with no retained layout or provider callback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._arrays import ReadOnlyState
from neurospatial.geometry._binning import BinConnections, SpatialBins
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._face_builders import (
    grid_geometry,
    hex_geometry,
    mesh_geometry,
    polar_geometry,
    track_geometry,
)


@dataclass(frozen=True, eq=False)
class NativeFaces(ReadOnlyState):
    """Closed native recipe over owned numerical data; materialized on demand.

    The recipe holds no bins or pairs of its own: :meth:`build` evaluates it
    on the bins and connections of the binning that carries it, so its
    faces always follow that binning's open pairs.

    Parameters
    ----------
    family : {'grid', 'polar', 'hex', 'mesh', 'track'}
        Native formula to evaluate. No construction parameters are replayed.
    arrays : tuple of ndarray
        Grid widths; mesh points/triangles/active IDs; or per-edge track junction
        positions (NaN marks no junction). Hexagons need no additional arrays.
    n_angle : int, default=0
        Polar angular bin count.
    circular : bool, default=False
        Whether polar seam interfaces are joined.
    """

    family: Literal["grid", "polar", "hex", "mesh", "track"]
    arrays: tuple[NDArray[Any], ...] = ()
    n_angle: int = 0
    circular: bool = False

    def __post_init__(self) -> None:
        arrays = tuple(np.array(array, copy=True) for array in self.arrays)
        for array in arrays:
            array.setflags(write=False)
        object.__setattr__(self, "arrays", arrays)

    @property
    def available(self) -> bool:
        """Native numerical geometry is available without running a builder."""
        return True

    def build(self, bins: SpatialBins, connections: BinConnections) -> FaceGeometry:
        """Evaluate the native formula on ``bins``, in open-pair order.

        Parameters
        ----------
        bins : SpatialBins
            Centers and physical measures of the binning's bins.
        connections : BinConnections
            The binning's neighbor pairs; faces follow the open pairs.

        Returns
        -------
        FaceGeometry
            Faces aligned with ``connections.pairs``.
        """
        pairs = connections.pairs
        graph = nx.Graph()
        graph.add_nodes_from(range(bins.n_bins))
        for (u, v), distance in zip(pairs, connections.pair_distances, strict=True):
            graph.add_edge(int(u), int(v), distance=float(distance))
        if self.family == "grid":
            graph, volumes = grid_geometry(
                graph, bins.centers, bins.sizes, self.arrays[0]
            )
        elif self.family == "polar":
            graph, volumes = polar_geometry(
                graph,
                bins.centers,
                bins.sizes,
                self.arrays[0],
                self.n_angle,
                self.circular,
            )
        elif self.family == "hex":
            graph, volumes = hex_geometry(graph, bins.sizes)
        elif self.family == "mesh":
            graph, volumes = mesh_geometry(graph, bins.sizes, *self.arrays)
        else:
            input_order = {
                frozenset((int(u), int(v))): i for i, (u, v) in enumerate(pairs)
            }
            junctions = self.arrays[0][
                [input_order[frozenset((u, v))] for u, v in graph.edges()]
            ]
            graph, volumes = track_geometry(graph, bins.centers, bins.sizes, junctions)
        faces = FaceGeometry.from_graph(graph, volumes)
        indices = {
            frozenset((int(u), int(v))): i for i, (u, v) in enumerate(faces.edges)
        }
        order = np.array(
            [indices[frozenset((int(u), int(v)))] for u, v in pairs], dtype=np.intp
        )
        areas, distances = faces.A[order], faces.distance[order]
        areas.setflags(write=False)
        distances.setflags(write=False)
        return FaceGeometry(pairs, areas, distances, faces.volumes)
