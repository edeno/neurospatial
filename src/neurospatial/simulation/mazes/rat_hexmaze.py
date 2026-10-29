"""Rat HexMaze: four mouse HexMaze modules joined by five bridges.

The mouse HexMaze (Alonso et al. 2021) is one module of narrow corridors
"arranged as six regular densely packed hexagons, forming 12 two-way and 12
three-way choice points (nodes) 36.3 cm apart, in total spanning 2 × 1.9 m",
"assembled from 30 10-cm-wide opaque white acrylic gangways connected by 24
equilateral triangular intersection segments". The six hexagons form a ring
around a central hexagonal hole, so every corridor is one node spacing long
and corridors meet at 120°.

The rat HexMaze (Wijnen et al. 2024) "consist of four times the original
HexMaze, connected through two large and three small bridges, spanning 9 × 5 m
in size (unpublished)… each node (96 in total) can serve both as a start as
well as a goal location". The rat apparatus's node spacing, bridge lengths and
corridor width are not published; the defaults here are estimates (see
:class:`RatHexmazeDims`).

The corridors are walled (``arena.outline == "wall"``): in the mouse HexMaze
"Gangways were enclosed by either 7.5- or 15-cm-tall white acrylic walls" and
"Maze floor and walls were white and opaque" (Alonso et al., 2021).

References
----------
Alonso, A., Bokeria, L., van der Meij, J., Samanta, A., Eichler, R., Lotfi, A.,
Spooner, P., Navarro Lobato, I., & Genzel, L. (2021). The HexMaze: A previous
knowledge task on map learning for mice. eNeuro, 8(4).
https://doi.org/10.1523/ENEURO.0554-20.2021

Wijnen, K., Genzel, L., & van der Meij, J. (2024). Rodent maze studies: from
following simple rules to complex map learning. Brain Structure and Function,
229, 823-841. https://doi.org/10.1007/s00429-024-02771-x

Examples
--------
>>> from neurospatial.simulation.mazes.rat_hexmaze import (
...     make_rat_hexmaze,
...     RatHexmazeDims,
... )
>>> maze = make_rat_hexmaze(bin_size=4.0)
>>> maze.env_2d.units
'cm'
>>> [name for name in maze.env_2d.regions if name.startswith("module_")]
['module_A', 'module_B', 'module_C', 'module_D']
>>> sum(name.startswith("node_") for name in maze.env_2d.regions)
96
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from neurospatial.environment import Environment
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_lengths,
    check_longer_than_wide,
    refuse,
    warn_if_bins_too_coarse,
)
from neurospatial.simulation.mazes._geometry import (
    make_buffered_line,
    single_floor,
    union_polygons,
)

_MODULE_LETTERS = ("A", "B", "C", "D")

# Bridges as (name, module, node, module, node), using the node numbering of
# ``_module_layout``. Each bridge leaves a two-way corner node of the module
# perimeter along its free 60° lattice direction, so it meets the module at
# 120° like every other junction. Small bridges climb or descend at ±60° and
# are ``bridge_length`` long; large bridges are horizontal and
# ``3 * node_spacing + bridge_length`` long.
_BRIDGES = (
    ("AB", 0, 14, 1, 20),  # A's upper-right corner to B's lower-left corner
    ("BC", 1, 23, 2, 17),  # B's lower-right corner to C's upper-left corner
    ("CD", 2, 14, 3, 20),  # C's upper-right corner to D's lower-left corner
    ("AC", 0, 12, 2, 19),  # A's right corner to C's left corner (large)
    ("BD", 1, 13, 3, 18),  # B's right corner to D's left corner (large)
)


@dataclass(frozen=True)
class RatHexmazeDims(MazeDims):
    """Dimension specifications for the Rat HexMaze.

    Attributes
    ----------
    node_spacing : float
        Distance between neighboring nodes (choice points) inside a module,
        which is the length of every module corridor, in cm. Default 48.0.
        This is an estimate: the rat spacing is unpublished, and 48 cm makes
        the whole maze span about 9 × 5 m as reported. The mouse HexMaze uses
        36.3 cm.
    corridor_width : float
        Width of all corridors and bridges in cm. Default 10.0, the published
        mouse HexMaze width; the rat width is unpublished.
    bridge_length : float
        Length of the three small bridges in cm, from one module's corner
        node to the next module's corner node. Default 62.4 (1.3 ×
        ``node_spacing``), estimated from the published drawing. The two
        large bridges are then ``3 * node_spacing + bridge_length`` long,
        which keeps every module on one shared hexagonal lattice.
    n_modules : int
        Number of modules, 1 to 4. Default 4, the full rat HexMaze. 1 builds
        a single mouse HexMaze module; 2 and 3 keep the first modules of the
        rat layout (A-B, or A-B-C) and the bridges between them.

    Examples
    --------
    >>> dims = RatHexmazeDims()
    >>> dims.node_spacing, dims.corridor_width, dims.bridge_length
    (48.0, 10.0, 62.4)
    >>> dims.n_modules
    4

    The published mouse HexMaze is one module with 36.3 cm node spacing:

    >>> mouse = RatHexmazeDims(n_modules=1, node_spacing=36.3, corridor_width=10.0)
    >>> mouse.n_modules
    1
    """

    node_spacing: float = 48.0
    corridor_width: float = 10.0
    bridge_length: float = 62.4
    n_modules: int = 4

    def __post_init__(self) -> None:
        check_lengths(self, "node_spacing", "corridor_width", "bridge_length")
        if self.n_modules not in (1, 2, 3, 4):
            refuse(
                self,
                ("n_modules",),
                "is not a HexMaze layout",
                "Use n_modules 1, 2, 3 or 4: the rat HexMaze has four modules "
                "(n_modules=4), and n_modules=1 builds a single mouse HexMaze "
                "module.",
            )
        check_longer_than_wide(self, "node_spacing", "module corridor")
        check_longer_than_wide(self, "bridge_length", "small bridge")


def make_rat_hexmaze(
    dims: RatHexmazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    r"""Create a Rat HexMaze environment.

    Builds four HexMaze modules on one hexagonal lattice and joins them with
    three small and two large bridges. Every module corridor is one node
    spacing long, and corridors meet at 120° at all three-way nodes,
    including where a bridge joins a module.

    Parameters
    ----------
    dims : RatHexmazeDims, optional
        Maze dimensions. If None, uses the defaults (48 cm node spacing,
        10 cm corridors, 62.4 cm small bridges, 4 modules), which are
        estimates for the rat apparatus.
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    include_track : bool, optional
        Whether to create the linearized track graph (default: True).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D environment covering the corridors and bridges, each
          ``corridor_width`` wide
        - env_track: 1D linearized track whose graph nodes are the maze's
          choice points (if include_track=True)

    Notes
    -----
    Layout (flat-top hexagons, ``s = node_spacing``, ``L = bridge_length``)::

            B=======D
           / \     /
          /   \   /
         /     \ /
        A=======C

    ``/`` and ``\`` are small bridges, ``=`` large bridges. Module origins
    (the center of each module's central hexagonal hole) are A = (0, 0),
    B = (4s + L/2, √3(2s + L/2)), C = (8s + L, 0) and D = (12s + 1.5L,
    √3(2s + L/2)).

    Each module has 24 nodes and 30 corridors. Nodes are numbered from the
    module origin outward:

    - 0-5: around the central hole, at distance s and angles 0°, 60°, … 300°
      (three-way nodes)
    - 6-11: at distance 2s along the same angles, node 6 + k joined to node k
      by a spoke corridor (three-way nodes)
    - 12-23: perimeter corners at distance √7·s; nodes 12 + 2k and 13 + 2k
      flank node 6 + k (two-way nodes unless a bridge leaves them)

    The full four-module maze has 96 nodes, 125 corridors (120 module
    corridors and 5 bridges), 58 three-way and 38 two-way nodes. Including
    the corridor width it spans (17s + 1.5L + w) × (5√3·s + (√3/2)L + w),
    9.2 × 4.8 m with the defaults.

    Regions:

    - module_A ... module_D: polygons covering each module's corridors
    - bridge_AB, bridge_BC, bridge_CD (small) and bridge_AC, bridge_BD
      (large): polygons covering each bridge outside the modules
    - node_A0 ... node_D23: points at the choice points, the start and goal
      locations of the task

    The maze has no landmarks. It has no fixed reward hardware: in Alonso et
    al. (2021) "a heap of food crumbles (Coco Pops, Kellogg's) was placed at
    a previously determined GL [goal location], which varied for each
    animal", and the mouse was placed by hand at start locations that changed
    from trial to trial, so goals and starts are node regions chosen per
    session. The paper's "local and global cues applied in and around the
    maze" have no published positions.

    Examples
    --------
    Create the default four-module rat HexMaze:

    >>> maze = make_rat_hexmaze(bin_size=4.0)
    >>> graph = maze.env_track.binning.linearization.graph
    >>> graph.number_of_nodes(), graph.number_of_edges()
    (96, 125)
    >>> sorted(name for name in maze.env_2d.regions if name.startswith("bridge_"))
    ['bridge_AB', 'bridge_AC', 'bridge_BC', 'bridge_BD', 'bridge_CD']

    Create the published mouse HexMaze (one module, 36.3 cm node spacing):

    >>> mouse = make_rat_hexmaze(
    ...     RatHexmazeDims(n_modules=1, node_spacing=36.3, corridor_width=10.0),
    ...     bin_size=2.0,
    ... )
    >>> graph = mouse.env_track.binning.linearization.graph
    >>> graph.number_of_nodes(), graph.number_of_edges()
    (24, 30)

    Skip track graph creation:

    >>> maze = make_rat_hexmaze(bin_size=4.0, include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    RatHexmazeDims : Dimension specifications for Rat HexMaze.
    """
    if dims is None:
        dims = RatHexmazeDims()

    graph = _hexmaze_graph(dims)
    letters = _MODULE_LETTERS[: dims.n_modules]

    def corridor(u: str, v: str) -> Polygon:
        return make_buffered_line(
            graph.nodes[u]["pos"], graph.nodes[v]["pos"], dims.corridor_width
        )

    modules = {
        letter: union_polygons(
            [
                corridor(u, v)
                for u, v in graph.edges
                if u.startswith(f"node_{letter}") and v.startswith(f"node_{letter}")
            ]
        )
        for letter in letters
    }
    all_modules = union_polygons(list(modules.values()))
    bridges = {
        name: corridor(u, v).difference(all_modules)
        for name, u, v in _bridge_edges(dims.n_modules)
    }

    env_2d = Environment.from_polygon(
        polygon=single_floor(union_polygons([all_modules, *bridges.values()]), dims),
        bin_size=bin_size,
        name="rat_hexmaze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    region_builder = Regions()
    for letter, module in modules.items():
        region_builder.add(f"module_{letter}", polygon=module)
    for name, bridge in bridges.items():
        region_builder.add(f"bridge_{name}", polygon=bridge)
    for node, pos in graph.nodes(data="pos"):
        region_builder.add(node, point=pos)
    env_2d = env_2d.with_regions(region_builder)

    env_track = None
    if include_track:
        env_track = Environment.from_graph(
            graph=graph,
            edge_order=list(graph.edges),
            edge_spacing=0.0,
            bin_size=bin_size,
            track_width=dims.corridor_width,
            name="rat_hexmaze_1d",
        )
        env_track = env_track.with_metadata(units="cm")

    warn_if_bins_too_coarse(
        env_2d,
        maze="rat HexMaze",
        bin_size=bin_size,
        narrowest_corridor=dims.corridor_width,
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _module_layout(
    node_spacing: float,
) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    """Node positions and corridors of one HexMaze module.

    Parameters
    ----------
    node_spacing : float
        Distance between neighboring nodes, in cm.

    Returns
    -------
    positions : NDArray[np.float64], shape (24, 2)
        Node positions relative to the module origin, numbered as described
        in :func:`make_rat_hexmaze`.
    edges : list[tuple[int, int]]
        The 30 corridors as node index pairs, each ``node_spacing`` long.
    """
    s = node_spacing
    angles = np.deg2rad(60.0 * np.arange(6))
    unit = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    inner = s * unit
    middle = 2 * s * unit
    outer = np.empty((12, 2))
    outer[0::2] = middle + s * np.roll(unit, 1, axis=0)  # toward angle 60k - 60°
    outer[1::2] = middle + s * np.roll(unit, -1, axis=0)  # toward angle 60k + 60°
    positions = np.vstack([inner, middle, outer])

    edges = []
    for k in range(6):
        edges += [
            (k, (k + 1) % 6),  # around the central hole
            (k, 6 + k),  # spoke
            (6 + k, 12 + 2 * k),  # perimeter
            (6 + k, 13 + 2 * k),
            (13 + 2 * k, 12 + 2 * ((k + 1) % 6)),
        ]
    return positions, edges


def _module_origins(dims: RatHexmazeDims) -> NDArray[np.float64]:
    """Origins of modules A-D, shape (4, 2), in cm."""
    s, half_bridge = dims.node_spacing, dims.bridge_length / 2
    dx, dy = 4 * s + half_bridge, np.sqrt(3.0) * (2 * s + half_bridge)
    return np.array([[0.0, 0.0], [dx, dy], [2 * dx, 0.0], [3 * dx, dy]])


def _bridge_edges(n_modules: int) -> list[tuple[str, str, str]]:
    """Bridges whose two modules are both built, as (name, node, node)."""
    return [
        (name, f"node_{_MODULE_LETTERS[m1]}{i1}", f"node_{_MODULE_LETTERS[m2]}{i2}")
        for name, m1, i1, m2, i2 in _BRIDGES
        if max(m1, m2) < n_modules
    ]


def _hexmaze_graph(dims: RatHexmazeDims) -> nx.Graph:
    """Choice-point graph of the maze: module corridors, then bridges.

    Nodes are named ``node_<module><index>`` and carry ``pos`` in cm; edges
    carry their Euclidean ``distance``.
    """
    positions, edges = _module_layout(dims.node_spacing)
    graph = nx.Graph()
    n = dims.n_modules
    for letter, origin in zip(
        _MODULE_LETTERS[:n], _module_origins(dims)[:n], strict=True
    ):
        for i, pos in enumerate(positions + origin):
            graph.add_node(f"node_{letter}{i}", pos=(float(pos[0]), float(pos[1])))
        for i, j in edges:
            graph.add_edge(
                f"node_{letter}{i}", f"node_{letter}{j}", distance=dims.node_spacing
            )
    for _, u, v in _bridge_edges(dims.n_modules):
        distance = np.hypot(*np.subtract(graph.nodes[u]["pos"], graph.nodes[v]["pos"]))
        graph.add_edge(u, v, distance=float(distance))
    return graph
