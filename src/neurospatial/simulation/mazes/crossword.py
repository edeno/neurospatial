"""Crossword Maze environment for spatial navigation research.

The Crossword Maze is an incomplete grid pattern with corridors connecting
specific intersections, creating a sparse lattice structure. Unlike a complete
grid, not all intersections are connected, making the maze more challenging
for allocentric (map-based) navigation and route planning.

Reference: McNamara et al. (2014)

The maze consists of horizontal and vertical corridors that intersect at
specific points. Four square boxes at the corners serve as start/goal locations,
extending outward from the main corridor structure.

Examples
--------
>>> from neurospatial.simulation.mazes.crossword import (
...     make_crossword_maze,
...     CrosswordDims,
... )
>>> maze = make_crossword_maze()
>>> maze.env_2d.units
'cm'
>>> "box_top_left" in maze.env_2d.regions
True
>>> "box_bottom_right" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from shapely.geometry import box as shapely_box

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_corridor_polygon,
    union_polygons,
)


@dataclass(frozen=True)
class CrosswordDims(MazeDims):
    """Dimension specifications for Crossword Maze.

    The Crossword Maze consists of an incomplete grid pattern with horizontal
    and vertical corridors connecting specific intersections, plus four square
    box rooms at the corners.

    Attributes
    ----------
    grid_spacing : float
        Distance between adjacent grid intersections in cm. Default is 30.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.
    box_size : float
        Size (width and height) of corner box rooms in cm. Default is 15.0.

    Examples
    --------
    >>> dims = CrosswordDims()
    >>> dims.grid_spacing
    30.0
    >>> dims.corridor_width
    10.0
    >>> dims.box_size
    15.0

    >>> custom = CrosswordDims(grid_spacing=40.0, corridor_width=12.0, box_size=20.0)
    >>> custom.grid_spacing
    40.0
    """

    grid_spacing: float = 30.0
    corridor_width: float = 10.0
    box_size: float = 15.0


def make_crossword_maze(
    dims: CrosswordDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Crossword Maze environment.

    Creates an incomplete grid maze with horizontal and vertical corridors
    connecting specific intersections. Four square box rooms at the corners
    serve as start/goal locations, extending outward from the corridor grid.

    Parameters
    ----------
    dims : CrosswordDims, optional
        Maze dimensions. If None, uses default dimensions
        (30 cm grid spacing, 10 cm corridor width, 15 cm box size).
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
    The maze creates an incomplete grid pattern resembling a crossword puzzle:
    - Horizontal and vertical corridors intersect at specific points
    - 4 square box rooms at corners extend outward from the grid
    - Not all grid intersections are connected

    Regions:
    - box_top_left, box_top_right, box_bottom_left, box_bottom_right:
      Square polygon regions at corners
    - junction_X_Y: Point regions at corridor intersections

    Track Graph Topology:
    - Sparse crossword-style grid with selective connectivity
    - Corner boxes connected via corridor network

    Examples
    --------
    Create a default Crossword maze:

    >>> maze = make_crossword_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = CrosswordDims(grid_spacing=40.0, corridor_width=12.0, box_size=20.0)
    >>> maze = make_crossword_maze(dims=dims, bin_size=3.0)
    >>> "box_top_left" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_crossword_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    CrosswordDims : Dimension specifications for Crossword Maze.
    """
    if dims is None:
        dims = CrosswordDims()

    # Grid layout: 3x3 internal intersections
    # Boxes at the 4 outer corners extending beyond the grid
    s = dims.grid_spacing  # shorthand
    w = dims.corridor_width
    b = dims.box_size

    # Define the crossword corridor structure - a sparse grid with selective connectivity
    # The pattern creates distinct corridor segments between junction points
    #
    # Layout (viewed from above, boxes extend outward at corners):
    #
    #   [TL]─j_2_0─j_2_1─j_2_2─j_2_3─[TR]
    #         │     │     │     │
    #         │   j_1_1─j_1_2   │
    #         │     │     │     │
    #   [BL]─j_0_0─j_0_1─j_0_2─j_0_3─[BR]
    #
    # Note: Left edge (j_0_0-j_2_0) and right edge (j_0_3-j_2_3) are direct
    # connections without middle junctions. Interior columns have 3 junctions each.

    corridors = []

    # === HORIZONTAL CORRIDORS (as individual segments) ===
    # Top row segments (at y = 2*s)
    corridors.append(
        make_corridor_polygon((0.0, 2 * s), (s, 2 * s), w)
    )  # j_2_0 to j_2_1
    corridors.append(
        make_corridor_polygon((s, 2 * s), (2 * s, 2 * s), w)
    )  # j_2_1 to j_2_2
    corridors.append(
        make_corridor_polygon((2 * s, 2 * s), (3 * s, 2 * s), w)
    )  # j_2_2 to j_2_3

    # Middle row segment (at y = s) - only connects interior columns
    corridors.append(make_corridor_polygon((s, s), (2 * s, s), w))  # j_1_1 to j_1_2

    # Bottom row segments (at y = 0)
    corridors.append(make_corridor_polygon((0.0, 0.0), (s, 0.0), w))  # j_0_0 to j_0_1
    corridors.append(make_corridor_polygon((s, 0.0), (2 * s, 0.0), w))  # j_0_1 to j_0_2
    corridors.append(
        make_corridor_polygon((2 * s, 0.0), (3 * s, 0.0), w)
    )  # j_0_2 to j_0_3

    # === VERTICAL CORRIDORS ===
    # Left edge: direct connection from bottom to top (no middle junction)
    corridors.append(
        make_corridor_polygon((0.0, 0.0), (0.0, 2 * s), w)
    )  # j_0_0 to j_2_0

    # Interior column 1 (at x = s) - connects through middle junction
    corridors.append(make_corridor_polygon((s, 0.0), (s, s), w))  # j_0_1 to j_1_1
    corridors.append(make_corridor_polygon((s, s), (s, 2 * s), w))  # j_1_1 to j_2_1

    # Interior column 2 (at x = 2*s) - connects through middle junction
    corridors.append(
        make_corridor_polygon((2 * s, 0.0), (2 * s, s), w)
    )  # j_0_2 to j_1_2
    corridors.append(
        make_corridor_polygon((2 * s, s), (2 * s, 2 * s), w)
    )  # j_1_2 to j_2_2

    # Right edge: direct connection from bottom to top (no middle junction)
    corridors.append(
        make_corridor_polygon((3 * s, 0.0), (3 * s, 2 * s), w)
    )  # j_0_3 to j_2_3

    # === CORNER BOX ROOMS ===
    # Boxes extend outward from the grid corners
    half_w = w / 2

    # Top-left box: extends up and left from (0, 2*s)
    box_tl = shapely_box(-b, 2 * s - half_w, half_w, 2 * s + b)
    corridors.append(box_tl)

    # Top-right box: extends up and right from (3*s, 2*s)
    box_tr = shapely_box(3 * s - half_w, 2 * s - half_w, 3 * s + b, 2 * s + b)
    corridors.append(box_tr)

    # Bottom-left box: extends down and left from (0, 0)
    box_bl = shapely_box(-b, -b, half_w, half_w)
    corridors.append(box_bl)

    # Bottom-right box: extends down and right from (3*s, 0)
    box_br = shapely_box(3 * s - half_w, -b, 3 * s + b, half_w)
    corridors.append(box_br)

    # Union all corridors and boxes into single polygon
    maze_polygon = union_polygons(corridors)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=maze_polygon,
        bin_size=bin_size,
        name="crossword_maze",
        connect_diagonal_neighbors=False,  # Manhattan grid only
    )
    env_2d.units = "cm"

    # Add box regions as polygons (the actual square rooms)
    env_2d.regions.add("box_top_left", polygon=box_tl)
    env_2d.regions.add("box_top_right", polygon=box_tr)
    env_2d.regions.add("box_bottom_left", polygon=box_bl)
    env_2d.regions.add("box_bottom_right", polygon=box_br)

    # Add junction regions at key intersections
    junctions = [
        ("junction_0_0", (0.0, 0.0)),
        ("junction_0_1", (s, 0.0)),
        ("junction_0_2", (2 * s, 0.0)),
        ("junction_0_3", (3 * s, 0.0)),
        ("junction_1_1", (s, s)),
        ("junction_1_2", (2 * s, s)),
        ("junction_2_0", (0.0, 2 * s)),
        ("junction_2_1", (s, 2 * s)),
        ("junction_2_2", (2 * s, 2 * s)),
        ("junction_2_3", (3 * s, 2 * s)),
    ]
    for name, pos in junctions:
        env_2d.regions.add(name, point=pos)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_crossword_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_crossword_track_graph(
    dims: CrosswordDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Crossword Maze.

    The track graph represents the sparse crossword topology matching the
    2D corridor layout. The graph includes 4 corner boxes plus internal
    corridor intersections.

    Parameters
    ----------
    dims : CrosswordDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the Crossword maze track.
    """
    graph = nx.Graph()

    s = dims.grid_spacing
    b = dims.box_size

    # Define nodes matching the 2D layout:
    # - 4 corner box centers
    # - Junction points at corridor intersections
    #
    # Layout:
    #   box_tl ─── j_2_0 ─── j_2_1 ─── j_2_2 ─── j_2_3 ─── box_tr
    #              │         │         │         │
    #              │       j_1_1 ─── j_1_2       │
    #              │         │         │         │
    #   box_bl ─── j_0_0 ─── j_0_1 ─── j_0_2 ─── j_0_3 ─── box_br

    nodes = {
        # Corner box centers (extending outward)
        "box_tl": (-b / 2, 2 * s + b / 2),
        "box_tr": (3 * s + b / 2, 2 * s + b / 2),
        "box_bl": (-b / 2, -b / 2),
        "box_br": (3 * s + b / 2, -b / 2),
        # Bottom row junctions (y=0)
        "j_0_0": (0.0, 0.0),
        "j_0_1": (s, 0.0),
        "j_0_2": (2 * s, 0.0),
        "j_0_3": (3 * s, 0.0),
        # Middle row junctions (y=s) - only center ones
        "j_1_1": (s, s),
        "j_1_2": (2 * s, s),
        # Top row junctions (y=2*s)
        "j_2_0": (0.0, 2 * s),
        "j_2_1": (s, 2 * s),
        "j_2_2": (2 * s, 2 * s),
        "j_2_3": (3 * s, 2 * s),
    }

    for node_id, pos in nodes.items():
        graph.add_node(node_id, pos=pos)

    # Define edges based on the crossword corridor layout
    edges = [
        # Corner boxes to adjacent junctions
        ("box_tl", "j_2_0"),
        ("box_tr", "j_2_3"),
        ("box_bl", "j_0_0"),
        ("box_br", "j_0_3"),
        # Top row horizontal (y=2*s)
        ("j_2_0", "j_2_1"),
        ("j_2_1", "j_2_2"),
        ("j_2_2", "j_2_3"),
        # Middle row horizontal (y=s) - partial
        ("j_1_1", "j_1_2"),
        # Bottom row horizontal (y=0)
        ("j_0_0", "j_0_1"),
        ("j_0_1", "j_0_2"),
        ("j_0_2", "j_0_3"),
        # Left column vertical (x=0)
        ("j_0_0", "j_2_0"),
        # Second column vertical (x=s)
        ("j_0_1", "j_1_1"),
        ("j_1_1", "j_2_1"),
        # Third column vertical (x=2*s)
        ("j_0_2", "j_1_2"),
        ("j_1_2", "j_2_2"),
        # Right column vertical (x=3*s)
        ("j_0_3", "j_2_3"),
    ]

    for u, v in edges:
        # Calculate Euclidean distance from positions
        pos_u = graph.nodes[u]["pos"]
        pos_v = graph.nodes[v]["pos"]
        distance = ((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2) ** 0.5
        graph.add_edge(u, v, distance=distance)

    # Edge order for linearization (defines how track is laid out)
    edge_order = edges.copy()

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        name="crossword_maze_1d",
    )
    env_track.units = "cm"

    return env_track
