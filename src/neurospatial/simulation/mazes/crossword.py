"""Crossword Maze environment for spatial navigation research.

The Crossword Maze is a 4×4 Manhattan-style grid with corridors connecting
intersections. It provides a complex spatial environment with multiple routes
between locations, making it ideal for studying allocentric (map-based)
navigation and route planning.

Reference: McNamara et al. (2014)

The maze consists of horizontal and vertical corridors forming a grid pattern
with 90-degree angles. Four corner boxes serve as start/goal locations.

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

    The Crossword Maze consists of a Manhattan-style grid with horizontal
    and vertical corridors connecting intersections at regular intervals.

    Attributes
    ----------
    grid_spacing : float
        Distance between adjacent grid nodes in cm. Default is 30.0.
    corridor_width : float
        Width of all corridors in cm. Default is 10.0.
    n_rows : int
        Number of rows in the grid. Default is 4.
    n_cols : int
        Number of columns in the grid. Default is 4.

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

    Creates a Manhattan-style grid maze with horizontal and vertical corridors.
    The maze has 4 corner boxes as start/goal locations and regular nodes at
    each intersection. All turns are 90-degree angles.

    Parameters
    ----------
    dims : CrosswordDims, optional
        Maze dimensions. If None, uses default dimensions
        (30 cm grid spacing, 10 cm corridor width, 4×4 grid).
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
    The maze grid starts at the origin (0, 0):
    - X coordinates: 0, grid_spacing, 2*grid_spacing, ..., (n_cols-1)*grid_spacing
    - Y coordinates: 0, grid_spacing, 2*grid_spacing, ..., (n_rows-1)*grid_spacing

    Regions:
    - box_0: Top-left corner (0, (n_rows-1)*grid_spacing)
    - box_1: Top-right corner ((n_cols-1)*grid_spacing, (n_rows-1)*grid_spacing)
    - box_2: Bottom-right corner ((n_cols-1)*grid_spacing, 0)
    - box_3: Bottom-left corner (0, 0)
    - node_i_j: Grid node at row i, column j (i=0 to n_rows-1, j=0 to n_cols-1)

    Track Graph Topology:
    - Manhattan grid with 4-connectivity (up, down, left, right)
    - n_rows × n_cols nodes
    - (n_rows * (n_cols-1)) + (n_cols * (n_rows-1)) edges

    Examples
    --------
    Create a default Crossword maze:

    >>> maze = make_crossword_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = CrosswordDims(grid_spacing=40.0, corridor_width=12.0, n_rows=5, n_cols=5)
    >>> maze = make_crossword_maze(dims=dims, bin_size=3.0)
    >>> "node_2_2" in maze.env_2d.regions
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

    # Create horizontal corridors (one for each row)
    h_corridors = []
    for row in range(dims.n_rows):
        y = row * dims.grid_spacing
        start = (0.0, y)
        end = ((dims.n_cols - 1) * dims.grid_spacing, y)
        h_corridors.append(make_corridor_polygon(start, end, dims.corridor_width))

    # Create vertical corridors (one for each column)
    v_corridors = []
    for col in range(dims.n_cols):
        x = col * dims.grid_spacing
        start = (x, 0.0)
        end = (x, (dims.n_rows - 1) * dims.grid_spacing)
        v_corridors.append(make_corridor_polygon(start, end, dims.corridor_width))

    # Union all corridors into single polygon
    grid_polygon = union_polygons(h_corridors + v_corridors)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=grid_polygon,
        bin_size=bin_size,
        name="crossword_maze",
        connect_diagonal_neighbors=False,  # Manhattan grid only
    )
    env_2d.units = "cm"

    # Add corner box regions
    # box_0: top-left, box_1: top-right, box_2: bottom-right, box_3: bottom-left
    max_y = (dims.n_rows - 1) * dims.grid_spacing
    max_x = (dims.n_cols - 1) * dims.grid_spacing

    env_2d.regions.add("box_0", point=(0.0, max_y))  # Top-left
    env_2d.regions.add("box_1", point=(max_x, max_y))  # Top-right
    env_2d.regions.add("box_2", point=(max_x, 0.0))  # Bottom-right
    env_2d.regions.add("box_3", point=(0.0, 0.0))  # Bottom-left

    # Add node regions at each intersection
    for row in range(dims.n_rows):
        for col in range(dims.n_cols):
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

    The track graph represents the Manhattan grid topology with 4-connectivity.
    Interior nodes have 4 neighbors (up, down, left, right), edge nodes have 3,
    and corner nodes have 2.

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
    # Create a new graph with string node IDs
    # (Environment.from_graph expects string or simple node IDs, not tuples)
    graph = nx.Graph()

    # Add nodes with position attributes
    for row in range(dims.n_rows):
        for col in range(dims.n_cols):
            node_id = f"node_{row}_{col}"
            x = col * dims.grid_spacing
            y = row * dims.grid_spacing
            graph.add_node(node_id, pos=(x, y))

    # Add horizontal edges (within each row)
    for row in range(dims.n_rows):
        for col in range(dims.n_cols - 1):
            u = f"node_{row}_{col}"
            v = f"node_{row}_{col + 1}"
            distance = dims.grid_spacing
            graph.add_edge(u, v, distance=distance)

    # Add vertical edges (within each column)
    for col in range(dims.n_cols):
        for row in range(dims.n_rows - 1):
            u = f"node_{row}_{col}"
            v = f"node_{row + 1}_{col}"
            distance = dims.grid_spacing
            graph.add_edge(u, v, distance=distance)

    # Create edge order for linearization
    # We'll traverse row-by-row, then column-by-column
    edge_order = []

    # Add horizontal edges (within each row)
    for row in range(dims.n_rows):
        for col in range(dims.n_cols - 1):
            u = f"node_{row}_{col}"
            v = f"node_{row}_{col + 1}"
            edge_order.append((u, v))

    # Add vertical edges (within each column)
    for col in range(dims.n_cols):
        for row in range(dims.n_rows - 1):
            u = f"node_{row}_{col}"
            v = f"node_{row + 1}_{col}"
            edge_order.append((u, v))

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges
        bin_size=bin_size,
        name="crossword_maze_1d",
    )
    env_track.units = "cm"

    return env_track
