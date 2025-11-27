"""Crossword Maze environment for spatial navigation research.

The Crossword Maze is an incomplete grid pattern with corridors connecting
specific intersections, creating a sparse lattice structure. Unlike a complete
grid, not all intersections are connected, making the maze more challenging
for allocentric (map-based) navigation and route planning.

Reference: McNamara et al. (2014)

The maze consists of horizontal and vertical corridors that intersect at
specific points. Six boxes/rooms serve as start/goal locations: four at the
corners and two at intermediate positions.

Examples
--------
>>> from neurospatial.simulation.mazes.crossword import (
...     make_crossword_maze,
...     CrosswordDims,
... )
>>> maze = make_crossword_maze()
>>> maze.env_2d.units
'cm'
>>> "box_0" in maze.env_2d.regions
True
>>> "node_1_1" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

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
    and vertical corridors connecting specific intersections.

    Attributes
    ----------
    grid_spacing : float
        Distance between adjacent grid nodes in cm. Default is 30.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.
    n_rows : int
        Number of rows in the base grid. Default is 4.
    n_cols : int
        Number of columns in the base grid. Default is 4.

    Examples
    --------
    >>> dims = CrosswordDims()
    >>> dims.grid_spacing
    30.0
    >>> dims.corridor_width
    10.0
    >>> dims.n_rows
    4
    >>> dims.n_cols
    4

    >>> custom = CrosswordDims(
    ...     grid_spacing=40.0, corridor_width=12.0, n_rows=5, n_cols=5
    ... )
    >>> custom.grid_spacing
    40.0
    """

    grid_spacing: float = 30.0
    corridor_width: float = 10.0
    n_rows: int = 4
    n_cols: int = 4


def make_crossword_maze(
    dims: CrosswordDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Crossword Maze environment.

    Creates an incomplete grid maze with horizontal and vertical corridors
    connecting specific intersections. Unlike a complete Manhattan grid,
    this creates a sparse lattice pattern with 6 boxes as endpoints.

    Parameters
    ----------
    dims : CrosswordDims, optional
        Maze dimensions. If None, uses default dimensions
        (30 cm grid spacing, 10 cm corridor width, 4×4 base grid).
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
    The maze creates an incomplete grid pattern:
    - Not all intersections are connected
    - 6 boxes serve as start/goal locations (4 corners + 2 intermediate)
    - Creates interesting route-planning challenges

    Regions:
    - box_0 to box_5: Six goal/start locations around the perimeter
    - node_i_j: Grid nodes at specific intersections (not all are present)

    Track Graph Topology:
    - Sparse Manhattan-style grid
    - Selected nodes connected by horizontal and vertical corridors

    Examples
    --------
    Create a default Crossword maze:

    >>> maze = make_crossword_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = CrosswordDims(grid_spacing=40.0, corridor_width=12.0)
    >>> maze = make_crossword_maze(dims=dims, bin_size=3.0)
    >>> "box_0" in maze.env_2d.regions
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

    max_y = (dims.n_rows - 1) * dims.grid_spacing
    max_x = (dims.n_cols - 1) * dims.grid_spacing

    # Create an incomplete grid pattern (sparse connectivity)
    # Based on the sketch: horizontal and vertical corridors that don't
    # form a complete grid

    corridors = []

    # Horizontal corridors (not all rows are complete)
    # Row 0 (bottom): full
    corridors.append(
        make_corridor_polygon((0.0, 0.0), (max_x, 0.0), dims.corridor_width)
    )
    # Row 1: partial (left and right sections with gap)
    corridors.append(
        make_corridor_polygon(
            (0.0, dims.grid_spacing),
            (dims.grid_spacing * 1.5, dims.grid_spacing),
            dims.corridor_width,
        )
    )
    corridors.append(
        make_corridor_polygon(
            (dims.grid_spacing * 2.5, dims.grid_spacing),
            (max_x, dims.grid_spacing),
            dims.corridor_width,
        )
    )
    # Row 2: full
    corridors.append(
        make_corridor_polygon(
            (0.0, dims.grid_spacing * 2),
            (max_x, dims.grid_spacing * 2),
            dims.corridor_width,
        )
    )
    # Row 3 (top): full
    corridors.append(
        make_corridor_polygon((0.0, max_y), (max_x, max_y), dims.corridor_width)
    )

    # Vertical corridors (not all columns are complete)
    # Column 0 (left): full
    corridors.append(
        make_corridor_polygon((0.0, 0.0), (0.0, max_y), dims.corridor_width)
    )
    # Column 1: partial (bottom section)
    corridors.append(
        make_corridor_polygon(
            (dims.grid_spacing, 0.0),
            (dims.grid_spacing, dims.grid_spacing * 2),
            dims.corridor_width,
        )
    )
    # Column 2: partial (top section)
    corridors.append(
        make_corridor_polygon(
            (dims.grid_spacing * 2, dims.grid_spacing),
            (dims.grid_spacing * 2, max_y),
            dims.corridor_width,
        )
    )
    # Column 3 (right): full
    corridors.append(
        make_corridor_polygon((max_x, 0.0), (max_x, max_y), dims.corridor_width)
    )

    # Union all corridors into single polygon
    grid_polygon = union_polygons(corridors)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=grid_polygon,
        bin_size=bin_size,
        name="crossword_maze",
        connect_diagonal_neighbors=False,  # Manhattan grid only
    )
    env_2d.units = "cm"

    # Add 6 box regions at strategic locations
    # 4 corners + 2 intermediate positions
    env_2d.regions.add("box_0", point=(0.0, max_y))  # Top-left
    env_2d.regions.add("box_1", point=(max_x, max_y))  # Top-right
    env_2d.regions.add("box_2", point=(max_x, 0.0))  # Bottom-right
    env_2d.regions.add("box_3", point=(0.0, 0.0))  # Bottom-left
    env_2d.regions.add("box_4", point=(0.0, dims.grid_spacing))  # Left middle-bottom
    env_2d.regions.add(
        "box_5", point=(max_x, dims.grid_spacing * 2)
    )  # Right middle-top

    # Add node regions at key intersections
    key_nodes = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),  # Left column
        (1, 0),
        (1, 1),
        (1, 2),  # Second column (partial)
        (2, 1),
        (2, 2),
        (2, 3),  # Third column (partial)
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),  # Right column
    ]
    for row, col in key_nodes:
        x = col * dims.grid_spacing
        y = row * dims.grid_spacing
        region_name = f"node_{row}_{col}"
        env_2d.regions.add(region_name, point=(x, y))

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

    The track graph represents the sparse grid topology where only selected
    nodes are connected based on the incomplete corridor layout.

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

    # Define nodes that exist in the sparse grid
    # (row, col) format matching the 2D layout
    key_nodes = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),  # Bottom row (full)
        (1, 0),
        (1, 1),  # Row 1 left section
        (1, 3),  # Row 1 right section
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),  # Row 2 (full)
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),  # Top row (full)
    ]

    # Add nodes with positions
    for row, col in key_nodes:
        node_id = f"node_{row}_{col}"
        x = col * dims.grid_spacing
        y = row * dims.grid_spacing
        graph.add_node(node_id, pos=(x, y))

    # Define edges based on the sparse corridor layout
    edges = [
        # Bottom row (horizontal)
        ("node_0_0", "node_0_1"),
        ("node_0_1", "node_0_2"),
        ("node_0_2", "node_0_3"),
        # Row 1 partial horizontal
        ("node_1_0", "node_1_1"),
        # Row 2 (horizontal)
        ("node_2_0", "node_2_1"),
        ("node_2_1", "node_2_2"),
        ("node_2_2", "node_2_3"),
        # Top row (horizontal)
        ("node_3_0", "node_3_1"),
        ("node_3_1", "node_3_2"),
        ("node_3_2", "node_3_3"),
        # Left column (vertical, full)
        ("node_0_0", "node_1_0"),
        ("node_1_0", "node_2_0"),
        ("node_2_0", "node_3_0"),
        # Column 1 (partial)
        ("node_0_1", "node_1_1"),
        ("node_1_1", "node_2_1"),
        ("node_2_1", "node_3_1"),
        # Column 2 (partial - top section)
        ("node_2_2", "node_3_2"),
        # Right column (vertical, full)
        ("node_0_3", "node_1_3"),
        ("node_1_3", "node_2_3"),
        ("node_2_3", "node_3_3"),
    ]

    for u, v in edges:
        # Calculate distance from positions
        pos_u = graph.nodes[u]["pos"]
        pos_v = graph.nodes[v]["pos"]
        distance = abs(pos_u[0] - pos_v[0]) + abs(pos_u[1] - pos_v[1])
        graph.add_edge(u, v, distance=distance)

    # Edge order for linearization
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
