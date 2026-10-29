"""Small Hex Maze environment for spatial navigation research.

The hex maze of Krausz et al. (2023) is "a 1.30m-per-side equilateral
triangular platform with liquid reward ports at each vertex". "Equally spaced
columnar barriers divide the maze into 49 hexagonal units", and movable
barriers (8 or 9 in a session) placed in some of those units make a maze
configuration. The animal runs from hex to hex between the three reward ports.

The geometry follows the Berke lab's ``hexmaze`` package
(https://github.com/calderast/Hex-maze, commit ``8684adfb``), which numbers
the hexes, places their centres and defines which hexes are adjacent and which
barrier sets are allowed:

- 49 open hexes with flat tops, in rows of 1, 1, 2, 2, ..., 7 hexes
  (``plotting.get_hex_centroids``), numbered as in that package, with the
  reward ports in hexes 1 (port A, top), 2 (port B, bottom left) and
  3 (port C, bottom right);
- 15 permanent interior barriers and 12 half barriers along the walls fill
  the remaining hex positions inside the triangle, so each open hex has at most
  three open neighbours and the empty maze has 63 hex-to-hex passages
  (``utils.create_empty_hex_maze``);
- the hex side is 7 cm (corner to corner 14 cm), the package's measurement
  of its ``"new_maze"`` (``plotting.get_pixels_per_cm``).
  Adjacent hex centres are then ``sqrt(3) * 7 = 12.1`` cm apart, the port
  hexes are 126 cm apart, and the triangle's long sides measure about 133 cm,
  close to the published 1.30 m;
- each reward port is in the short wall that cuts off a corner of the
  triangle, the outer side of its port hex, which is the vertex where the
  paper places it.

The floor's outline is modeled as opaque walls (``arena.outline ==
"wall"``). Krausz et al. (2023) describe the barriers as columns and do not
say what bounds the platform's edge; the ``hexmaze`` package places half
barriers "along the walls".

Examples
--------
>>> from neurospatial.simulation.mazes.hex_small import (
...     make_small_hex_maze,
...     SmallHexDims,
... )
>>> maze = make_small_hex_maze()
>>> maze.env_2d.units
'cm'
>>> sorted(maze.env_2d.regions)
['port_A', 'port_B', 'port_C']

References
----------
Krausz, T. A., Comrie, A. E., Kahn, A. E., Frank, L. M., Daw, N. D., &
Berke, J. D. (2023). Dual credit assignment processes underlie dopamine
signals in a complex spatial environment. Neuron, 111(21), 3465-3478.e7.
https://doi.org/10.1016/j.neuron.2023.07.017
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.ops import unary_union

from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    warn_if_bins_too_coarse,
)
from neurospatial.simulation.mazes._geometry import widest_corridor

# Hexes per row from the top port down, and the hex ids in each row from left
# to right (hexmaze ``plotting.get_hex_centroids``, view_angle=1).
_HEXES_PER_ROW = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7)
_HEX_ORDER = (
    1, 4, 6, 5, 8, 7, 11, 10, 9, 14, 13, 12, 18, 17, 16, 15,
    22, 21, 20, 19, 27, 26, 25, 24, 23, 32, 31, 30, 29, 28,
    38, 37, 36, 35, 34, 33, 49, 42, 41, 40, 39, 48, 2, 47, 46, 45, 44, 43, 3,
)  # fmt: skip

# Open neighbours of each hex in the empty maze (hexmaze
# ``utils.create_empty_hex_maze``): 63 passages, at most 3 per hex.
_EMPTY_MAZE_NEIGHBORS: dict[int, tuple[int, ...]] = {
    1: (4,), 4: (1, 5, 6), 6: (4, 8), 5: (4, 7), 8: (6, 11, 10),
    7: (5, 10, 9), 11: (8, 14), 10: (8, 7, 13), 9: (7, 12),
    14: (11, 18, 17), 13: (10, 17, 16), 12: (9, 16, 15), 18: (14, 22),
    17: (14, 13, 21), 16: (13, 12, 20), 15: (12, 19), 22: (18, 27, 26),
    21: (17, 26, 25), 20: (16, 25, 24), 19: (15, 24, 23), 27: (22, 32),
    26: (22, 21, 31), 25: (21, 20, 30), 24: (20, 19, 29), 23: (19, 28),
    32: (27, 38, 37), 31: (26, 37, 36), 30: (25, 36, 35), 29: (24, 35, 34),
    28: (23, 34, 33), 38: (32, 49), 37: (31, 32, 42), 36: (30, 31, 41),
    35: (29, 30, 40), 34: (28, 29, 39), 33: (28, 48), 49: (2, 38, 47),
    42: (37, 46, 47), 41: (36, 45, 46), 40: (35, 44, 45), 39: (34, 43, 44),
    48: (3, 33, 43), 2: (49,), 47: (49, 42), 46: (42, 41), 45: (41, 40),
    44: (40, 39), 43: (39, 48), 3: (48,),
}  # fmt: skip

# Each wall hex is the fourth corner of the parallelogram spanned by three open
# hexes (top, middle, bottom): top + bottom - middle. Hexmaze
# ``plotting.get_permanent_barrier_centroids`` (15 interior barriers) and
# ``plotting.get_edge_hex_centroids`` (12 half barriers along the walls).
_PERMANENT_BARRIER_PARALLELOGRAMS = (
    (4, 6, 8), (8, 11, 14), (7, 10, 13), (14, 18, 22), (13, 17, 21),
    (12, 16, 20), (22, 27, 32), (21, 26, 31), (20, 25, 30), (19, 24, 29),
    (32, 38, 49), (31, 37, 42), (30, 36, 41), (29, 35, 40), (28, 34, 39),
    (6, 8, 11), (11, 14, 18), (18, 22, 27), (27, 32, 38), (5, 7, 9),
    (9, 12, 15), (15, 19, 23), (23, 28, 33), (47, 42, 46), (46, 41, 45),
    (45, 40, 44), (44, 39, 43),
)  # fmt: skip

_PORTS = {"port_A": 1, "port_B": 2, "port_C": 3}

# Hexes a movable barrier may occupy: hexmaze ``core.POSSIBLE_BARRIERS`` keeps
# the ports (1-3), the hex in front of each port (4, 48, 49) and the two hexes
# after it (5, 6, 33, 38, 43, 47) open.
_ALLOWED_BARRIERS = frozenset(range(7, 47)) - {33, 38, 43}

# A configuration from the hexmaze ``maze_configuration_database.csv`` (all
# open hexes reachable, port-to-port paths of 15, 17 and 19 hexes, one
# critical choice point at hex 17). It shares 6 of its 9 barriers with the
# barriers annotated in a video frame of the maze.
_DEFAULT_BARRIERS = frozenset({7, 9, 10, 19, 22, 25, 36, 39, 42})


@dataclass(frozen=True)
class SmallHexDims(MazeDims):
    """Size and barrier configuration of the Small Hex Maze.

    Attributes
    ----------
    hex_side : float
        Side length of one hexagonal unit in cm, equal to the distance from
        its centre to a corner. Default 7.0, the ``"new_maze"`` measurement
        in the Berke lab's ``hexmaze`` package: adjacent hex centres are ``sqrt(3) * hex_side`` (12.1 cm) apart and
        the reward ports ``18 * hex_side`` (126 cm) apart.
    barriers : frozenset of int
        Hexes holding a movable barrier, numbered as in the Berke lab's
        ``hexmaze`` package (open hexes 1-49, ports at 1, 2 and 3). Default
        is a 9-barrier configuration from that package's database of valid
        mazes. Barriers may occupy hexes 7-46 except 33, 38 and 43, which
        keeps the three hexes leading to each port open, and must leave
        every open hex reachable.

    Raises
    ------
    ValueError
        If ``hex_side`` is not positive, if a barrier is outside the allowed
        hexes, or if the barriers cut off any open hex.

    Examples
    --------
    >>> dims = SmallHexDims()
    >>> dims.hex_side
    7.0
    >>> sorted(dims.barriers)
    [7, 9, 10, 19, 22, 25, 36, 39, 42]

    Barriers that wall off a hex are refused:

    >>> SmallHexDims(barriers={39, 40})  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: barriers {39, 40} cut off open hex 44 ...
    """

    hex_side: float = 7.0
    barriers: frozenset[int] = field(default=_DEFAULT_BARRIERS)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.hex_side) and self.hex_side > 0):
            raise ValueError(
                f"hex_side must be a positive length in cm, got {self.hex_side!r}."
            )
        barriers = frozenset(operator.index(b) for b in self.barriers)
        object.__setattr__(self, "barriers", barriers)
        refused = sorted(barriers - _ALLOWED_BARRIERS)
        if refused:
            raise ValueError(
                f"barriers {refused} are not hexes a movable barrier can occupy "
                "in the hexmaze numbering. "
                "Use hexes 7-46 except 33, 38 and 43: hexes 1-3 hold the reward "
                "ports, and the three hexes leading to each port (4-6, 33, 38, "
                "43, 47-49) stay open."
            )
        graph = _maze_graph(barriers)
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if len(components) > 1:
            cut_off = sorted(h for c in components[1:] for h in c)
            raise ValueError(
                f"barriers {{{', '.join(map(str, sorted(barriers)))}}} cut off "
                "open hex"
                f"{'es' if len(cut_off) > 1 else ''} "
                f"{', '.join(map(str, cut_off))} from the rest of the maze. "
                "Every open hex must be reachable; move a barrier next to "
                "the cut-off hexes."
            )


def _maze_graph(barriers: frozenset[int]) -> nx.Graph:
    """Hex adjacency of the maze: the empty maze without the barrier hexes."""
    graph = nx.Graph()
    for hex_id, neighbors in _EMPTY_MAZE_NEIGHBORS.items():
        graph.add_edges_from((hex_id, n) for n in neighbors)
    graph.remove_nodes_from(barriers)
    return graph


def _hex_centroids(corner_to_corner: float) -> dict[int, np.ndarray]:
    """Centres of the 49 open hexes, port A at the top.

    The formula is hexmaze ``plotting.get_hex_centroids`` with
    ``scale=corner_to_corner``, which puts hex 1 at ``(0, 0)`` with the rows
    below it.
    """
    row_spacing = math.sqrt(3) / 2 * corner_to_corner
    centroids: dict[int, np.ndarray] = {}
    y_shift = 0.0
    hex_ids = iter(_HEX_ORDER)
    for row, n_in_row in enumerate(_HEXES_PER_ROW):
        if row % 2 == 0 and row != 0:
            y_shift += row_spacing / 2
        for i in range(n_in_row):
            x = (i * 1.5 - (n_in_row - 1) * 0.75) * corner_to_corner
            centroids[next(hex_ids)] = np.array([x, -row * row_spacing + y_shift])
    return centroids


# Corners of a flat-topped hexagon around its centre, counterclockwise from
# the right, in lattice units of half a side across and one apothem up.
_HEXAGON_CORNERS = np.array([(2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)])


def _hex_lattice(side: float) -> tuple[dict[int, np.ndarray], list[float]]:
    """Hex centres in lattice units, and the map from lattice units to cm.

    Hex centres and corners lie on a lattice half a hex side across and one
    apothem up, so in those units every corner is a pair of integers.
    Geometry built there shares corners exactly between neighbouring hexes.

    Returns
    -------
    centroids : dict of int to ndarray, shape (2,)
        Each hex's centre in lattice units.
    to_cm : list of float
        ``shapely.affinity.affine_transform`` coefficients that scale lattice
        units to cm and put the centre of the three port hexes at the origin.
    """
    unit = np.array([side / 2.0, math.sqrt(3) / 2.0 * side])
    lattice = {h: np.rint(c / unit) for h, c in _hex_centroids(2.0 * side).items()}
    middle = np.mean([lattice[h] for h in _PORTS.values()], axis=0) * unit
    return lattice, [unit[0], 0.0, 0.0, unit[1], -middle[0], -middle[1]]


def _lattice_hexagon(center: np.ndarray) -> Polygon:
    """Flat-topped regular hexagon around a lattice-unit centre."""
    return Polygon(center + _HEXAGON_CORNERS)


def _platform(centroids: dict[int, np.ndarray], to_cm: list[float]) -> Polygon:
    """Triangular platform with each corner cut at its port hex's outer edge.

    Hexmaze ``plotting.get_maze_bounding_box``: the two corners of each port
    hex farthest from the maze centre, measured in cm, are the platform's
    corners. ``centroids`` and the platform are in lattice units; ``to_cm``
    is the map from :func:`_hex_lattice`.
    """
    scale = np.array([to_cm[0], to_cm[3]])
    middle = np.mean([centroids[h] for h in _PORTS.values()], axis=0)
    corners: list[np.ndarray] = []
    for hex_id in _PORTS.values():
        outward = (centroids[hex_id] - middle) * scale
        vertices = centroids[hex_id] + _HEXAGON_CORNERS
        corners.extend(vertices[np.argsort((vertices - middle) * scale @ outward)[-2:]])
    corners_arr = np.asarray(corners, dtype=np.float64)
    angle = np.arctan2(*(corners_arr - corners_arr.mean(axis=0)).T[::-1])
    return Polygon(corners_arr[np.argsort(angle)])


def make_small_hex_maze(
    dims: SmallHexDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create the Small Hex Maze (Krausz et al. 2023; Berke lab).

    A triangular platform holds 49 hexagonal units. Wall hexes between them
    and the movable barriers in ``dims.barriers`` are removed from the floor,
    so the floor is the open hexes and the animal passes between two open
    hexes only across the side they share.

    Parameters
    ----------
    dims : SmallHexDims, optional
        Hex size and barrier configuration. If None, uses
        :class:`SmallHexDims` defaults (7 cm hex side, a 9-barrier
        configuration from the Berke lab's database).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0). The narrowest passage between
        hexes and the thinnest wall are both one hex side (7 cm) wide, so
        bins approaching that size can close a passage or join hexes across
        a wall.
    include_track : bool, optional
        Whether to create the linearized track graph (default: True).

    Returns
    -------
    MazeEnvironments
        Contains:

        - env_2d: the open floor of the maze, centred on the platform's
          centre, port A at the top
        - env_track: (if include_track=True) one node per open hex at its
          centre (``"hex_<id>"``) and one edge per passage between
          adjacent open hexes, each ``sqrt(3) * hex_side`` long. Its
          ``track_width`` is the widest passage between barriers, measured
          where the track crosses from one hex to the next.

    Notes
    -----
    Hex numbering, centres, adjacency and the allowed barrier positions come
    from the Berke lab's ``hexmaze`` package
    (https://github.com/calderast/Hex-maze, commit
    ``8684adfb05143d7b005821d80683731f191747eb``). The real barriers are
    round columns; here each barrier fills its whole hexagonal unit. That
    leaves out small pockets of floor between the columns, but keeps the
    floor's connections exactly those of the hex graph.

    Hexmaze's ``is_valid_maze`` adds rules for designing a session (port-to-
    port path lengths of 15-25 hexes, at most 3 critical choice points, no
    long straight runs). They are not enforced here, so training mazes with
    fewer barriers and the empty maze (``barriers=frozenset()``) can be built.

    The paper has "liquid reward ports at each vertex" of the triangle, and
    hexmaze's maze footprint (``plotting.get_maze_bounding_box``) cuts each
    vertex off at the outer side of its port hex (hexes 1, 2 and 3). The
    ports are landmarks with ``metadata["role"] == "reward_port"``, each a
    point at the middle of that cut-off wall, ``sqrt(3) / 2 * hex_side``
    (6.1 cm) beyond the port hex's centre. Neither source gives the port's
    exact mounting or size, so its region is the port hex, the unit of the
    maze from which the rat reaches the port and which hexmaze numbers as
    the port:

    - port_A: hex 1, top corner
    - port_B: hex 2, bottom left corner
    - port_C: hex 3, bottom right corner

    Each landmark's metadata also gives its ``port_type``,
    ``"liquid_port"``, and names its ``port`` letter and ``hex``.

    Examples
    --------
    >>> maze = make_small_hex_maze()
    >>> maze.env_2d.landmarks["port_A"].data.round(1)  # top corner
    array([ 0. , 78.8])
    >>> dict(maze.env_2d.landmarks["port_A"].metadata)
    {'role': 'reward_port', 'port_type': 'liquid_port', 'port': 'A', 'hex': 1}
    >>> maze.env_2d.regions["port_A"].kind  # the port hex
    'polygon'

    A different barrier configuration, e.g. a training maze with 6 barriers:

    >>> dims = SmallHexDims(barriers={9, 11, 18, 31, 34, 45})
    >>> maze = make_small_hex_maze(dims)
    >>> len(maze.env_2d.components())  # one connected floor
    1

    Skip track graph creation:

    >>> maze = make_small_hex_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    SmallHexDims : Hex size and barrier configuration.

    References
    ----------
    Krausz, T. A., et al. (2023). Dual credit assignment processes underlie
    dopamine signals in a complex spatial environment. Neuron, 111(21),
    3465-3478.e7. https://doi.org/10.1016/j.neuron.2023.07.017
    """
    if dims is None:
        dims = SmallHexDims()

    side = dims.hex_side
    # Built in lattice units, neighbouring hexes share their corners exactly,
    # so cutting out wall hexes leaves no zero-width slits of floor between
    # them; the floor is then mapped to cm.
    lattice, to_cm = _hex_lattice(side)
    centroids = {
        h: np.asarray(affine_transform(shapely.Point(c), to_cm).coords[0])
        for h, c in lattice.items()
    }

    walls = [
        lattice[top] + lattice[bottom] - lattice[mid]
        for top, mid, bottom in _PERMANENT_BARRIER_PARALLELOGRAMS
    ]
    walls.extend(lattice[h] for h in dims.barriers)
    floor_lattice = _platform(lattice, to_cm).difference(
        unary_union([_lattice_hexagon(c) for c in walls])
    )
    if not isinstance(floor_lattice, Polygon):
        # SmallHexDims refuses barriers that cut off hexes.
        raise RuntimeError("the hex maze floor is not one connected piece")
    floor = affine_transform(floor_lattice, to_cm)

    env_2d = Environment.from_polygon(
        polygon=floor,
        bin_size=bin_size,
        name="small_hex_maze",
        connect_diagonal_neighbors=True,
    ).with_metadata(units="cm")
    # Each port sits in the wall that cuts off its corner of the triangle, the
    # outer side of its port hex; the maze centre is the origin.
    apothem = math.sqrt(3) / 2 * side
    regions = Regions()
    landmarks = []
    for name, hex_id in _PORTS.items():
        center = centroids[hex_id]
        regions.add(
            name, polygon=affine_transform(_lattice_hexagon(lattice[hex_id]), to_cm)
        )
        landmarks.append(
            Landmark(
                name,
                "point",
                center + apothem * center / np.linalg.norm(center),
                {
                    "role": "reward_port",
                    "port_type": "liquid_port",
                    "port": name[-1],
                    "hex": hex_id,
                },
            )
        )
    env_2d = env_2d.with_regions(regions).with_landmarks(Landmarks(landmarks))

    env_track = None
    if include_track:
        env_track = _create_hex_track_graph(dims, centroids, floor, bin_size)

    warn_if_bins_too_coarse(
        env_2d,
        maze="small hex maze",
        bin_size=bin_size,
        narrowest_corridor=dims.hex_side,
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_hex_track_graph(
    dims: SmallHexDims,
    centroids: dict[int, np.ndarray],
    floor: Polygon,
    bin_size: float,
) -> Environment:
    """Track graph with one node per open hex and one edge per passage.

    Parameters
    ----------
    dims : SmallHexDims
        Hex size and barrier configuration.
    centroids : dict[int, np.ndarray]
        Centre of each of the 49 hexes in cm, maze-centred.
    floor : Polygon
        The open floor; it sets the track's width.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment of the hex maze.
    """
    hexes = _maze_graph(dims.barriers)
    graph = nx.Graph()
    for hex_id in sorted(hexes.nodes):
        graph.add_node(f"hex_{hex_id}", pos=tuple(centroids[hex_id]))
    for u, v in hexes.edges:
        graph.add_edge(
            f"hex_{u}",
            f"hex_{v}",
            distance=float(np.linalg.norm(centroids[u] - centroids[v])),
        )
    # Linearize outward from port A.
    edge_order = list(nx.edge_bfs(graph, f"hex_{_PORTS['port_A']}"))
    segments = [(graph.nodes[u]["pos"], graph.nodes[v]["pos"]) for u, v in edge_order]
    return Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=widest_corridor(floor, segments),
        name="small_hex_maze_1d",
    ).with_metadata(units="cm")
