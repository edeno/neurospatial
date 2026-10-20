"""Hamlet Maze environment for spatial navigation research.

The Hamlet Maze is a star-shaped "village" for mice: a ring of alleys in the
middle, five alleys radiating from the ring, and at the end of each radial
alley a split into two arms that lead to two different goal boxes, the
"houses". Each house sits at a point of the star and is reached from the two
neighbouring radial alleys. Solid blocks fill the space between alleys.

Wijnen et al. (2024) describe the layout: "Both mazes consist of five central
alleys forming a pentagonal ring in the middle of the maze and five
peripheral alleys radiating from this central ring. In the Hamlet maze,
however, the peripheral alleys split up in two arms, each leading to a
different goal box". Studies using the apparatus describe "a 1.6 m diameter
apparatus with an agora in the center and five corridors expanding toward
different compartments, called houses" (Hua et al. 2023, Methods). The
alley positions, the corridor width and the house size are not published in
open text; the defaults are estimates measured from an overhead photograph
of the apparatus scaled to the 1.6 m diameter.

References
----------
Crouzier, L., Gilabert, D., Rossel, M., Trousse, F., & Maurice, T. (2018).
Topographical memory analyzed in mice using the Hamlet test, a novel complex
maze. Neurobiology of Learning and Memory, 149, 118-134.
https://doi.org/10.1016/j.nlm.2018.02.014

Hua, J., Garcia de Paco, E., Linck, N., Maurice, T., Desrumaux, C.,
Manoury, B., Rassendren, F., & Ulmann, L. (2023). Microglial P2X4 receptors
promote ApoE degradation and contribute to memory deficits in Alzheimer's
disease. Cellular and Molecular Life Sciences, 80(5), 138.
https://doi.org/10.1007/s00018-023-04784-x

Wijnen, K., Genzel, L., & van der Meij, J. (2024). Rodent maze studies: from
following simple rules to complex map learning. Brain Structure and Function,
229(4), 823-841. https://doi.org/10.1007/s00429-024-02771-x

Examples
--------
>>> from neurospatial.simulation.mazes.hamlet import make_hamlet_maze, HamletDims
>>> maze = make_hamlet_maze()
>>> maze.env_2d.units
'cm'
>>> [name for name in maze.env_2d.regions if name.startswith("house_")]
['house_0', 'house_1', 'house_2', 'house_3', 'house_4']
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import union_polygons


@dataclass(frozen=True)
class HamletDims(MazeDims):
    """Dimension specifications for Hamlet Maze.

    All radii are measured from the maze center to an alley centerline
    point; each is an estimate from an overhead photograph of the apparatus
    scaled to its published 1.6 m diameter.

    Attributes
    ----------
    ring_radius : float
        Distance from the center to each corner of the central pentagonal
        ring, where a radial alley leaves the ring, in cm. Default is 22.0.
    split_radius : float
        Distance from the center to the end of each radial alley, where it
        splits into two arms, in cm. Default is 43.0.
    house_radius : float
        Distance from the center to the center of each house, in cm.
        Default is 70.0.
    corridor_width : float
        Width of every alley in cm. Default is 9.0.
    house_size : float
        Side of each square house in cm. Default is 16.0.
    n_arms : int
        Number of houses and of radial alleys (default: 5).

    Examples
    --------
    >>> dims = HamletDims()
    >>> dims.ring_radius, dims.split_radius, dims.house_radius
    (22.0, 43.0, 70.0)
    >>> dims.corridor_width
    9.0
    >>> dims.n_arms
    5

    >>> custom = HamletDims(house_radius=80.0, corridor_width=10.0)
    >>> custom.house_radius
    80.0
    """

    ring_radius: float = 22.0
    split_radius: float = 43.0
    house_radius: float = 70.0
    corridor_width: float = 9.0
    house_size: float = 16.0
    n_arms: int = 5


def make_hamlet_maze(
    dims: HamletDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Hamlet Maze environment.

    Creates the central pentagonal ring of alleys, five radial alleys that
    each split into two arms, and a house where the arms of neighbouring
    alleys meet at each point of the star.

    Parameters
    ----------
    dims : HamletDims, optional
        Maze dimensions. If None, uses default dimensions (estimated from
        the apparatus: ring corners at 22 cm, splits at 43 cm, houses at
        70 cm, 9 cm alleys, 16 cm houses).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    include_track : bool, optional
        Whether to create linearized track graph (default: True).
        The track graph provides a 1D representation for trajectory analysis.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment
        - env_track: 1D linearized track environment (if include_track=True)

    Notes
    -----
    The maze is centered at the origin. House ``k`` lies at
    ``90 + 360 k / n_arms`` degrees; radial alley ``k`` lies halfway between
    houses ``k`` and ``k + 1``, leaving the ring at ring corner ``k`` and
    splitting at split point ``k`` into one arm to each of those houses.

    The floor is the alleys, each ``corridor_width`` wide, plus the houses.
    The center of the ring and the region between each house and the ring
    are solid blocks, not floor.

    Regions:
    - house_0 to house_4: Square polygon regions, the goal boxes
    - ring_0 to ring_4: Points where each radial alley leaves the ring
    - split_0 to split_4: Points where each radial alley splits into two
      arms

    Track Graph Topology:
    - 15 nodes: 5 ring corners, 5 split points and 5 houses
    - 20 edges: 5 ring alleys, 5 radial alleys and 10 arms
    - Ring corners and split points are 3-way junctions; each house joins
      two arms

    Examples
    --------
    Create a default Hamlet maze:

    >>> maze = make_hamlet_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = HamletDims(split_radius=50.0, house_radius=80.0)
    >>> maze = make_hamlet_maze(dims=dims, bin_size=4.0)
    >>> "split_4" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_hamlet_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    HamletDims : Dimension specifications for Hamlet Maze.
    """
    if dims is None:
        dims = HamletDims()

    ring, split, house = _node_positions(dims)
    houses = _house_boxes(dims, house)

    # The alleys are the band within half a corridor width of the star's
    # edges; the blocks are each enclosed face of the alley network shrunk
    # by half a corridor width, so every alley keeps its full width and the
    # junctions keep sharp corners.
    half_width = dims.corridor_width / 2
    star = Polygon(np.stack([house, split], axis=1).reshape(-1, 2))
    blocks = [Polygon(ring).buffer(-half_width, join_style="mitre")]
    for k in range(dims.n_arms):
        face = Polygon([ring[k - 1], split[k - 1], house[k], split[k], ring[k]])
        blocks.append(face.buffer(-half_width, join_style="mitre"))
    floor = union_polygons(
        [star.buffer(half_width, join_style="mitre"), *houses]
    ).difference(union_polygons(blocks))

    env_2d = Environment.from_polygon(
        polygon=floor,
        bin_size=bin_size,
        name="hamlet_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    region_builder = Regions()
    for k, box in enumerate(houses):
        region_builder.add(f"house_{k}", polygon=box)
    for k, pos in enumerate(ring):
        region_builder.add(f"ring_{k}", point=pos)
    for k, pos in enumerate(split):
        region_builder.add(f"split_{k}", point=pos)
    env_2d = env_2d.with_regions(region_builder)

    env_track = None
    if include_track:
        env_track = _create_hamlet_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _node_positions(
    dims: HamletDims,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Ring corners, split points and house centers, each ``(n_arms, 2)`` cm."""
    house_angles = np.pi / 2 + 2 * np.pi * np.arange(dims.n_arms) / dims.n_arms
    alley_angles = house_angles + np.pi / dims.n_arms

    def polar(radius: float, angles: NDArray[np.float64]) -> NDArray[np.float64]:
        return radius * np.column_stack([np.cos(angles), np.sin(angles)])

    return (
        polar(dims.ring_radius, alley_angles),
        polar(dims.split_radius, alley_angles),
        polar(dims.house_radius, house_angles),
    )


def _house_boxes(dims: HamletDims, house: NDArray[np.float64]) -> list[Polygon]:
    """Square houses centered on ``house``, sides facing the maze center."""
    half = dims.house_size / 2
    boxes = []
    for x, y in house:
        square = shapely_box(-half, -half, half, half)
        square = rotate(square, np.arctan2(y, x), origin=(0, 0), use_radians=True)
        boxes.append(translate(square, x, y))
    return boxes


def _create_hamlet_track_graph(
    dims: HamletDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Hamlet Maze.

    Nodes sit at the ring corners, the split points and the house centers;
    edges follow the ring alleys, the radial alleys and the arms.

    Parameters
    ----------
    dims : HamletDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the Hamlet maze track.
    """
    n = dims.n_arms
    ring, split, house = _node_positions(dims)

    graph = nx.Graph()
    for k in range(n):
        graph.add_node(f"ring_{k}", pos=tuple(ring[k]))
        graph.add_node(f"split_{k}", pos=tuple(split[k]))
        graph.add_node(f"house_{k}", pos=tuple(house[k]))

    edge_order = [(f"ring_{k}", f"ring_{(k + 1) % n}") for k in range(n)]
    for k in range(n):
        edge_order += [
            (f"ring_{k}", f"split_{k}"),
            (f"split_{k}", f"house_{k}"),
            (f"split_{k}", f"house_{(k + 1) % n}"),
        ]
    for u, v in edge_order:
        distance = np.linalg.norm(
            np.subtract(graph.nodes[u]["pos"], graph.nodes[v]["pos"])
        )
        graph.add_edge(u, v, distance=float(distance))

    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=dims.corridor_width,
        name="hamlet_maze_1d",
    )
    return env_track.with_metadata(units="cm")
