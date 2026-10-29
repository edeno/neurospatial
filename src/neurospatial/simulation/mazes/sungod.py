"""Sungod Maze environment for spatial navigation research.

The Sungod Maze (Gillespie et al. 2021) is a central box with 8 radiating
arms. "The maze was roughly 1 m x 1 m, surrounded by 16” external walls
with transparent internal walls of the same height between arms", and "The
maze environment contained 11 ports (home port, 2 center ports, and 8 arm
ports)". The two center ports "were approximately 7.5 cm apart".

This implementation traces the box, arms and ports from CVAT annotations of
a video frame of the maze, scaled so the maze spans about 1 m.

The geometry has:

- a central box whose outline spans about 39 x 58 cm
- 8 arms, 4.5-7.7 cm wide, fanning out from one side of the box
- 11 reward ports, named as in the paper: the home port at the narrow end
  of the box, the left and right center ports (6.9 cm apart) between it and
  the arms, and arm ports 1-8 at the arm ends

The external walls are opaque (``arena.outline == "wall"``). The transparent
walls between arms are the stretches of the floor's outline along each arm
side that faces a neighbouring arm, named ``arm_k_wall_facing_arm_j`` with
kind ``"transparent_wall"``: they keep the rat in its arm but not its view.
The outer sides of arms 1 and 8 face no arm and are modeled as opaque.

Examples
--------
>>> from neurospatial.simulation.mazes.sungod import make_sungod_maze, SungodDims
>>> maze = make_sungod_maze()
>>> maze.env_2d.units
'cm'
>>> list(maze.env_2d.landmarks)[:3]
['home_port', 'center_port_left', 'center_port_right']

References
----------
Gillespie, A. K., Astudillo Maya, D. A., Denovellis, E. L., Liu, D. F.,
Kastner, D. B., Coulter, M. E., Roumis, D. K., Eden, U. T., & Frank, L. M.
(2021). Hippocampal replay reflects specific past experiences rather than a
plan for subsequent choice. Neuron, 109(19), 3149-3163.e6.
https://doi.org/10.1016/j.neuron.2021.07.029
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from neurospatial.arena import OpenArena
from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import corridor_width

# Calibration: pixels to cm conversion for Sungod maze
# Maze is ~1m x 1m, image extent is ~600 px across
_CM_PER_PIXEL = 100.0 / 600.0  # ~0.167 cm/pixel

# ============================================================================
# Geometry extracted from CVAT annotation (annotations 2.xml - sungod)
# All coordinates in pixels - will be converted to cm
# ============================================================================

# Central box polygon
_BOX_POINTS_PX = [
    (617.19, 535.68),
    (600.29, 521.03),
    (590.52, 507.89),
    (576.25, 484.98),
    (565.73, 463.94),
    (558.22, 436.52),
    (556.34, 420.75),
    (554.09, 385.06),
    (557.84, 365.53),
    (554.09, 332.48),
    (565.73, 312.20),
    (568.36, 281.77),
    (574.00, 268.25),
    (581.88, 240.08),
    (596.91, 217.54),
    (615.69, 190.50),
    (628.46, 198.01),
    (698.70, 259.99),
    (784.71, 347.50),
    (789.60, 403.84),
    (762.18, 447.79),
]

# Arm polygons, arm 1 to arm 8: left to right as seen from the home port
# facing the arms, the numbering of Gillespie et al. (2021), Figure 2A.
_ARMS_PX = [
    [(344.33, 109.14), (360.84, 73.03), (615.22, 191.19), (599.22, 216.98)],
    [(306.15, 183.96), (312.85, 158.16), (581.16, 242.27), (573.94, 268.07)],
    [(280.81, 240.05), (273.59, 268.43), (565.11, 311.25), (567.18, 284.42)],
    [(558.84, 364.83), (553.43, 332.92), (253.24, 316.69), (249.46, 353.47)],
    [(259.73, 407.56), (255.41, 440.01), (556.14, 419.46), (552.89, 384.84)],
    [(282.45, 524.93), (274.34, 491.93), (560.46, 437.31), (565.87, 463.27)],
    [(316.53, 608.76), (303.55, 567.12), (575.61, 484.36), (589.67, 506.54)],
    [(350.98, 645.14), (379.96, 682.29), (615.49, 535.17), (599.89, 520.31)],
]

# Port positions: centroids of the port masks in the CVAT annotation.
# The home port sits at the narrow end of the box, farthest from the arms; the
# two center ports sit side by side between it and the arms, left and right
# as seen from the home port facing the arms; each arm port is at the far end
# of its arm, in the order of ``_ARMS_PX``.
_HOME_PORT_PX = (770.35, 385.70)
_CENTER_PORTS_PX = {
    "center_port_left": (680.33, 359.65),
    "center_port_right": (680.12, 401.07),
}
_ARM_PORTS_PX = [
    (363.02, 93.74),
    (319.26, 171.72),
    (288.40, 253.88),
    (262.12, 333.45),
    (265.22, 418.03),
    (285.72, 502.18),
    (319.37, 578.84),
    (377.29, 644.20),
]


@dataclass(frozen=True)
class SungodDims(MazeDims):
    """Dimension specifications for Sungod Maze.

    The Sungod Maze consists of a central box platform with 8 radiating arms.
    Default dimensions are based on the actual Sungod maze geometry
    (~1m x 1m total extent).

    Attributes
    ----------
    scale : float
        Scale factor applied to the maze. Default 1.0 uses the calibrated
        dimensions (~100 cm extent). Use values > 1.0 to enlarge.

    Examples
    --------
    >>> dims = SungodDims()
    >>> dims.scale
    1.0

    >>> custom = SungodDims(scale=1.2)
    >>> custom.scale
    1.2
    """

    scale: float = 1.0


def make_sungod_maze(
    dims: SungodDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Sungod Maze environment.

    Creates a radial arm maze with a central box platform and 8 radiating
    arms. The maze geometry is based on the actual Sungod maze, with arms
    connecting to a central platform.

    Parameters
    ----------
    dims : SungodDims, optional
        Maze dimensions. If None, uses default dimensions
        (scale=1.0, ~100 cm extent).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    include_track : bool, optional
        Whether to create linearized track graph (default: True).
        The track runs from the home port to the box center, joins the box
        center to each arm where it meets the box, then runs along the middle
        of the arm to its far end.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment, with the 11 ports as
          landmarks and regions
        - env_track: 1D linearized track environment (if include_track=True);
          its ``track_width`` is the widest arm's width, so a sample within half
          that width of the centerline is on the track, and a narrower arm can
          admit a few samples just outside its walls

    Notes
    -----
    The maze geometry is derived from CVAT annotations of the Sungod maze.
    The original pixel coordinates are converted to centimeters using an
    estimated calibration factor (maze is ~1m x 1m).

    The ports follow Gillespie et al. (2021). "The maze environment contained
    11 ports (home port, 2 center ports, and 8 arm ports)", the center ports
    "approximately 7.5 cm apart" (6.9 cm here). Their Figure 1A puts the home
    port at the narrow end of the box, the two center ports side by side
    between it and the arms, and a port at the end of each arm; their Figure
    2A numbers the arms 1-8 from left to right. Here left and right are as
    seen from the home port facing the arms, which in these coordinates
    (arms toward -x) puts arm 1 and the left center port toward -y. The
    paper gives no port size, so each port is a point. Every trial runs home
    port, then the lit center port, then one arm port; the home and center
    ports dispense 50 uL of milk and only the goal arm's port rewards
    (150 uL).

    The published 16-inch external walls and the transparent internal walls
    between arms are the edges of the floor. The floor's outline is an opaque
    wall, and each arm side facing a neighbouring arm is a named
    ``"transparent_wall"`` on it (``arm_k_wall_facing_arm_j``), so gaze
    analyses see through the arm dividers while movement analyses stop at
    them. The small plastic wall that kept the rat from walking over the
    center ports is not declared, because the paper does not place it.

    Landmarks, each a point with ``metadata["role"] == "reward_port"``,
    ``metadata["port_type"] == "liquid_port"`` (the ports dispense milk) and
    ``metadata["location"]`` ``"home"``, ``"center"`` or ``"arm"``:

    - home_port
    - center_port_left, center_port_right (``metadata["side"]``)
    - arm_port_1 ... arm_port_8 (``metadata["arm"]``, 1-8)

    Regions:

    - one point region per port, named and placed as its landmark
    - box: polygon of the central box, the paper's "box" maze segment that
      holds the home and center ports

    Track Graph Topology, in linearization order:

    - home_port -> box_center: from the home port, between the two center
      ports, to the centroid of the box
    - box_center -> arm_k_start: across the box to the middle of arm k's end
      at the box
    - arm_k_start -> arm_port_k: along the middle of arm k to its far end,
      for arms 1-8; the end node is named like the port there, as the
      linear, W and radial-arm mazes name theirs

    Examples
    --------
    Create a default Sungod maze:

    >>> maze = make_sungod_maze()
    >>> maze.env_2d.landmarks["home_port"].metadata["location"]
    'home'
    >>> dict(maze.env_2d.landmarks["arm_port_1"].metadata)
    {'role': 'reward_port', 'port_type': 'liquid_port', 'location': 'arm', 'arm': 1}

    Create with custom scale:

    >>> dims = SungodDims(scale=1.2)
    >>> maze = make_sungod_maze(dims=dims, bin_size=3.0)
    >>> "arm_port_1" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_sungod_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    SungodDims : Dimension specifications for Sungod Maze.
    """
    if dims is None:
        dims = SungodDims()

    scale = dims.scale * _CM_PER_PIXEL

    # Convert box polygon to cm
    box_polygon = Polygon([(x * scale, y * scale) for x, y in _BOX_POINTS_PX])

    # Convert arm polygons to cm
    arm_polygons = [
        Polygon([(x * scale, y * scale) for x, y in arm]) for arm in _ARMS_PX
    ]

    # Buffer each polygon slightly to ensure overlap (CVAT annotations have small gaps)
    # Then union all pieces together
    buffer_distance = 0.5 * scale  # Small buffer to bridge gaps
    buffered_box = box_polygon.buffer(buffer_distance)
    buffered_arms = [arm.buffer(buffer_distance) for arm in arm_polygons]

    # Union all buffered polygons
    all_buffered = [buffered_box, *buffered_arms]
    maze_geometry = unary_union(all_buffered)

    # Simplify back to remove buffer artifacts (negative buffer to shrink slightly)
    # Use a smaller buffer for shrinking to preserve the shape
    maze_geometry = maze_geometry.buffer(-buffer_distance * 0.5)

    # Handle MultiPolygon result (should now be single connected polygon)
    if isinstance(maze_geometry, MultiPolygon):
        # Take the largest polygon by area (should be the main maze)
        maze_polygon = max(maze_geometry.geoms, key=lambda p: p.area)
    elif isinstance(maze_geometry, Polygon):
        maze_polygon = maze_geometry
    else:
        # GeometryCollection or other - extract polygons
        polygons = [g for g in maze_geometry.geoms if isinstance(g, Polygon)]
        maze_polygon = max(polygons, key=lambda p: p.area) if polygons else box_polygon

    # Ports in pixels, in the order of the landmarks and regions.
    ports = {
        "home_port": _HOME_PORT_PX,
        **_CENTER_PORTS_PX,
        **{f"arm_port_{k}": xy for k, xy in enumerate(_ARM_PORTS_PX, start=1)},
    }
    metadata: dict[str, dict[str, str | int]] = {
        "home_port": {"location": "home"},
        "center_port_left": {"location": "center", "side": "left"},
        "center_port_right": {"location": "center", "side": "right"},
        **{
            f"arm_port_{k}": {"location": "arm", "arm": k}
            for k in range(1, len(_ARM_PORTS_PX) + 1)
        },
    }

    # Center the maze at origin
    centroid = maze_polygon.centroid
    cx, cy = centroid.x, centroid.y

    # Translate all geometries to center at origin
    from shapely.affinity import translate

    maze_polygon = translate(maze_polygon, -cx, -cy)
    box_polygon = translate(box_polygon, -cx, -cy)
    arm_polygons = [translate(p, -cx, -cy) for p in arm_polygons]
    port_positions = {
        name: (x * scale - cx, y * scale - cy) for name, (x, y) in ports.items()
    }

    # Opaque external walls, with transparent walls between neighbouring arms.
    dividers, divider_names = _arm_dividers(
        maze_polygon, arm_polygons, tolerance=buffer_distance
    )
    arena = OpenArena(
        maze_polygon,
        outline="wall",
        walls=dividers,
        wall_names=divider_names,
        wall_kinds="transparent_wall",
    )

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=maze_polygon,
        bin_size=bin_size,
        arena=arena,
        name="sungod_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # The paper gives no port size, so each port region is a point.
    region_builder = Regions()
    for name, pos in port_positions.items():
        region_builder.add(name, point=pos)
    region_builder.add("box", polygon=box_polygon)
    env_2d = env_2d.with_regions(region_builder).with_landmarks(
        Landmarks(
            Landmark(
                name,
                "point",
                pos,
                {"role": "reward_port", "port_type": "liquid_port", **metadata[name]},
            )
            for name, pos in port_positions.items()
        )
    )

    # Create track graph if requested
    env_track = None
    if include_track:
        box_center = (box_polygon.centroid.x, box_polygon.centroid.y)
        env_track = _create_sungod_track_graph(
            port_positions["home_port"], box_center, arm_polygons, bin_size
        )

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _arm_dividers(
    floor: Polygon, arms: list[Polygon], *, tolerance: float
) -> tuple[list[LineString], list[str]]:
    """The stretches of the floor's outline along the walls between arms.

    Each arm's long side that faces a neighbouring arm is a transparent
    divider. The floor is the union of the traced box and arms, grown by
    ``tolerance`` and shrunk by half of it, so its outline runs a quarter of
    ``tolerance`` from each traced side and an outline edge whose midpoint
    is within ``tolerance`` of a side lies along it; the arms, 4.5-7.7 cm
    wide, are far wider. Each divider is built from the outline's own
    vertices, so it coincides with the outline exactly.

    Parameters
    ----------
    floor : Polygon
        The maze floor, in cm.
    arms : list of Polygon
        The traced arms in order 1-8, four-sided, in the floor's frame.
    tolerance : float
        The buffer the floor was smoothed with, in cm.

    Returns
    -------
    walls : list of LineString
        The divider stretches of the outline.
    names : list of str
        The wall each stretch belongs to: ``"arm_k_wall_facing_arm_j"`` for
        arm ``k``'s side facing its neighbour ``j``. Arms 1 and 8 have one
        such side each.
    """
    sides: dict[str, LineString] = {}
    centroids = [np.array(arm.centroid.coords[0]) for arm in arms]
    for k, arm in enumerate(arms):
        corners = np.asarray(arm.exterior.coords)[:-1]
        edges = [(corners[i], corners[(i + 1) % len(corners)]) for i in range(4)]
        long_sides = sorted(edges, key=lambda e: -np.linalg.norm(e[1] - e[0]))[:2]
        for j in (k - 1, k + 1):
            if 0 <= j < len(arms):
                side = min(
                    long_sides,
                    key=lambda e: np.linalg.norm((e[0] + e[1]) / 2 - centroids[j]),
                )
                sides[f"arm_{k + 1}_wall_facing_arm_{j + 1}"] = LineString(side)
    ring = np.asarray(floor.exterior.coords)
    midpoints = shapely.points((ring[:-1] + ring[1:]) / 2)
    names = list(sides)
    distance = np.array([shapely.distance(midpoints, sides[n]) for n in names])
    owner = np.where(distance.min(axis=0) <= tolerance, distance.argmin(axis=0), -1)
    # Start the walk at an edge that is no divider's, so no stretch wraps.
    start = int(np.flatnonzero(owner < 0)[0])
    order = np.roll(np.arange(len(owner)), -start)
    walls: list[LineString] = []
    wall_names: list[str] = []
    run: list[int] = []
    for edge in [*order, start]:
        if run and owner[edge] != owner[run[-1]]:
            if owner[run[-1]] >= 0:
                walls.append(LineString(ring[[*run, run[-1] + 1]]))
                wall_names.append(names[owner[run[-1]]])
            run = []
        run.append(edge)
    return walls, wall_names


def _arm_centerline(
    arm: Polygon, box_center: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Midpoints of an arm's two ends, the one at the box first.

    Parameters
    ----------
    arm : Polygon
        Four-sided arm outline in cm: two long walls joined by two short ends.
    box_center : np.ndarray, shape (2,)
        Center of the box in cm.

    Returns
    -------
    start, end : tuple[float, float]
        Midpoint of the end at the box, and of the far end.
    """
    corners = np.asarray(arm.exterior.coords)[:-1]
    sides = [(corners[k], corners[(k + 1) % len(corners)]) for k in range(len(corners))]
    ends = sorted(sides, key=lambda side: np.linalg.norm(side[1] - side[0]))[:2]
    midpoints = sorted(
        ((a + b) / 2 for a, b in ends),
        key=lambda p: np.linalg.norm(p - box_center),
    )
    start, end = midpoints
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def _create_sungod_track_graph(
    home_port: tuple[float, float],
    box_center: tuple[float, float],
    arm_polygons: list[Polygon],
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Sungod Maze.

    The track starts at the home port and crosses the box, between the two
    center ports, to the box center. Each arm is then a path from the box
    center to the middle of the arm's end at the box, then along the middle
    of the arm to its far end, so the whole track lies on the maze floor.

    Parameters
    ----------
    home_port : tuple[float, float]
        Position of the home port (x, y) in cm.
    box_center : tuple[float, float]
        Center position of the box (x, y) in cm.
    arm_polygons : list[Polygon]
        Arm polygons in cm, arm 1 to arm 8 (already centered).
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the maze track.
    """
    graph = nx.Graph()
    graph.add_node("home_port", pos=home_port)
    graph.add_node("box_center", pos=box_center)
    edges = [("home_port", "box_center")]

    # Each arm runs from the midpoint of its end at the box to the midpoint of
    # its far end; the box center joins each arm where it meets the box.
    center = np.asarray(box_center)
    for k, arm in enumerate(arm_polygons, start=1):
        start, end = _arm_centerline(arm, center)
        graph.add_node(f"arm_{k}_start", pos=start)
        graph.add_node(f"arm_port_{k}", pos=end)
        edges += [
            ("box_center", f"arm_{k}_start"),
            (f"arm_{k}_start", f"arm_port_{k}"),
        ]
    for u, v in edges:
        graph.add_edge(
            u,
            v,
            distance=float(
                np.linalg.norm(
                    np.subtract(graph.nodes[u]["pos"], graph.nodes[v]["pos"])
                )
            ),
        )

    # Create the 1D environment. The arms differ in width, so the track takes
    # the widest one: a sample within half that width of the centerline is
    # on the track, which lets a narrower arm admit a few samples just outside
    # its walls.
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edges,
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=max(corridor_width(arm) for arm in arm_polygons),
        name="sungod_maze_1d",
    )
    env_track = env_track.with_metadata(units="cm")

    return env_track
