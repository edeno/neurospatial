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
- 11 reward wells: the home port, 2 center ports (6.9 cm apart) and 8 arm
  ports

Examples
--------
>>> from neurospatial.simulation.mazes.sungod import make_sungod_maze, SungodDims
>>> maze = make_sungod_maze()
>>> maze.env_2d.units
'cm'
>>> maze.env_2d.n_bins > 0
True

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
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from neurospatial.environment import Environment
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

# Arm polygons (8 arms radiating from center)
_ARMS_PX = [
    # arm_0 (bottom, pointing down-left)
    [(350.98, 645.14), (379.96, 682.29), (615.49, 535.17), (599.89, 520.31)],
    # arm_1
    [(316.53, 608.76), (303.55, 567.12), (575.61, 484.36), (589.67, 506.54)],
    # arm_2
    [(282.45, 524.93), (274.34, 491.93), (560.46, 437.31), (565.87, 463.27)],
    # arm_3
    [(259.73, 407.56), (255.41, 440.01), (556.14, 419.46), (552.89, 384.84)],
    # arm_4
    [(558.84, 364.83), (553.43, 332.92), (253.24, 316.69), (249.46, 353.47)],
    # arm_5
    [(280.81, 240.05), (273.59, 268.43), (565.11, 311.25), (567.18, 284.42)],
    # arm_6
    [(306.15, 183.96), (312.85, 158.16), (581.16, 242.27), (573.94, 268.07)],
    # arm_7 (top, pointing up-left)
    [(344.33, 109.14), (360.84, 73.03), (615.22, 191.19), (599.22, 216.98)],
]

# Reward well positions (centroids of mask polygons from CVAT)
# 11 wells: home port, 2 center ports, 8 arm ports
_REWARD_WELLS_PX = [
    (680.12, 401.07),  # reward_well_0 (center area)
    (680.33, 359.65),  # reward_well_1 (center area)
    (262.12, 333.45),  # reward_well_2 (arm 4 end)
    (288.40, 253.88),  # reward_well_3 (arm 5 end)
    (265.22, 418.03),  # reward_well_4 (arm 3 end)
    (285.72, 502.18),  # reward_well_5 (arm 2 end)
    (319.26, 171.72),  # reward_well_6 (arm 6 end)
    (363.02, 93.74),  # reward_well_7 (arm 7 end)
    (770.35, 385.70),  # reward_well_8 (box right side)
    (319.37, 578.84),  # reward_well_9 (arm 1 end)
    (377.29, 644.20),  # reward_well_10 (arm 0 end - large, likely home)
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
        The track joins the box center to each arm where it meets the box,
        then runs along the middle of the arm to its far end.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment
        - env_track: 1D linearized track environment (if include_track=True);
          its ``track_width`` is the widest arm's width, so a sample within half
          that width of the centerline is on the track, and a narrower arm can
          admit a few samples just outside its walls

    Notes
    -----
    The maze geometry is derived from CVAT annotations of the Sungod maze.
    The original pixel coordinates are converted to centimeters using an
    estimated calibration factor (maze is ~1m x 1m).

    Maze structure:
    - Central box platform connecting all arms
    - 8 radiating arms at various angles
    - 11 reward wells (home, center, and arm ports)

    Regions:
    - reward_well_0 through reward_well_10: Point regions at well locations
    - center: Point region at box centroid

    Track Graph Topology:
    - center -> arm_i_start: across the box to the middle of arm i's end at
      the box
    - arm_i_start -> arm_i_end: along the middle of arm i to its far end

    Examples
    --------
    Create a default Sungod maze:

    >>> maze = make_sungod_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom scale:

    >>> dims = SungodDims(scale=1.2)
    >>> maze = make_sungod_maze(dims=dims, bin_size=3.0)
    >>> "reward_well_0" in maze.env_2d.regions
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

    # Convert reward wells to cm
    reward_wells = [(x * scale, y * scale) for x, y in _REWARD_WELLS_PX]

    # Center the maze at origin
    centroid = maze_polygon.centroid
    cx, cy = centroid.x, centroid.y

    # Translate all geometries to center at origin
    from shapely.affinity import translate

    maze_polygon = translate(maze_polygon, -cx, -cy)
    box_polygon = translate(box_polygon, -cx, -cy)
    arm_polygons = [translate(p, -cx, -cy) for p in arm_polygons]
    reward_wells = [(x - cx, y - cy) for x, y in reward_wells]

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=maze_polygon,
        bin_size=bin_size,
        name="sungod_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # Add center region at box centroid
    box_center = (box_polygon.centroid.x, box_polygon.centroid.y)
    region_builder = Regions()
    region_builder.add("center", point=box_center)

    # Add reward well regions
    for i, pos in enumerate(reward_wells):
        region_builder.add(f"reward_well_{i}", point=pos)
    env_2d = env_2d.with_regions(region_builder)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_sungod_track_graph(
            box_center, arm_polygons, reward_wells, bin_size, dims
        )

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


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
    box_center: tuple[float, float],
    arm_polygons: list[Polygon],
    reward_wells: list[tuple[float, float]],
    bin_size: float,
    dims: SungodDims,
) -> Environment:
    """Create the 1D linearized track graph for Sungod Maze.

    Each arm is a path from the box center to the middle of the arm's end at
    the box, then along the middle of the arm to its far end, so the whole
    track lies on the maze floor.

    Parameters
    ----------
    box_center : tuple[float, float]
        Center position of the box (x, y) in cm.
    arm_polygons : list[Polygon]
        List of arm polygons (already centered).
    reward_wells : list[tuple[float, float]]
        List of reward well positions (already centered).
    bin_size : float
        Spatial bin size in cm.
    dims : SungodDims
        Maze dimensions.

    Returns
    -------
    Environment
        1D linearized environment representing the maze track.
    """
    graph = nx.Graph()
    graph.add_node("center", pos=box_center)

    # Each arm runs from the midpoint of its end at the box to the midpoint of
    # its far end; the center joins each arm where it meets the box.
    center = np.asarray(box_center)
    edge_order = []
    for i, arm in enumerate(arm_polygons):
        start, end = _arm_centerline(arm, center)
        graph.add_node(f"arm_{i}_start", pos=start)
        graph.add_node(f"arm_{i}_end", pos=end)
        for u, v in [("center", f"arm_{i}_start"), (f"arm_{i}_start", f"arm_{i}_end")]:
            graph.add_edge(
                u,
                v,
                distance=float(
                    np.linalg.norm(
                        np.subtract(graph.nodes[u]["pos"], graph.nodes[v]["pos"])
                    )
                ),
            )
            edge_order.append((u, v))

    # Create the 1D environment. The arms differ in width, so the track takes
    # the widest one: a sample within half that width of the centerline is
    # on the track, which lets a narrower arm admit a few samples just outside
    # its walls.
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=max(corridor_width(arm) for arm in arm_polygons),
        name="sungod_maze_1d",
    )
    env_track = env_track.with_metadata(units="cm")

    return env_track
