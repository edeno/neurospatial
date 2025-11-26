"""Rat HexMaze environment for large-scale spatial navigation research.

The Rat HexMaze is a large-scale navigation task consisting of multiple hexagonal
modules connected by corridor bridges. Each module contains junction nodes arranged
in a hexagonal lattice with 120° angles between corridors. The maze design forces
cognitive map formation because all nodes look identical from inside (Warner-Warden trick).

Reference: Alonso et al. 2021 - large-scale navigation for cognitive map learning

From paper: "The HexMaze is a large-scale navigation task for mice, consisting of
30 corridors and 24 crossings... the corridors of this maze are separated by 120° angles"
(p. 830). "The rat HexMaze consist of four times the original HexMaze, connected through
two large and three small bridges, spanning 9×5 m in size... each node (96 in total)
can serve both as a start as well as a goal location" (p. 831).

Examples
--------
>>> from neurospatial.simulation.mazes.rat_hexmaze import (
...     make_rat_hexmaze,
...     RatHexmazeDims,
... )
>>> maze = make_rat_hexmaze()
>>> maze.env_2d.units
'cm'
>>> "module_A" in maze.env_2d.regions
True
>>> "corridor_AB" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import Point

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_buffered_line,
    union_polygons,
)


@dataclass(frozen=True)
class RatHexmazeDims(MazeDims):
    """Dimension specifications for Rat HexMaze.

    The Rat HexMaze consists of multiple hexagonal modules connected by corridor
    bridges. Each module contains a hexagonal lattice of junction nodes with 120°
    angles between corridors.

    Attributes
    ----------
    module_width : float
        Width of each hexagonal module in cm. Default is 90.0.
    corridor_width : float
        Width of all corridors in cm. Default is 11.0.
    n_modules : int
        Number of hexagonal modules (clusters). Default is 3.
    nodes_per_module : int
        Number of junction nodes per module. Default is 24 (mouse HexMaze size).

    Examples
    --------
    >>> dims = RatHexmazeDims()
    >>> dims.module_width
    90.0
    >>> dims.corridor_width
    11.0
    >>> dims.n_modules
    3
    >>> dims.nodes_per_module
    24

    >>> custom = RatHexmazeDims(module_width=120.0, n_modules=2)
    >>> custom.module_width
    120.0
    >>> custom.n_modules
    2
    """

    module_width: float = 90.0
    corridor_width: float = 11.0
    n_modules: int = 3
    nodes_per_module: int = 24


def make_rat_hexmaze(
    dims: RatHexmazeDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Rat HexMaze environment.

    Creates a large-scale hexagonal maze with multiple modules connected by
    corridor bridges. Each module contains junction nodes arranged in a hexagonal
    lattice with 120° angles, forcing cognitive map formation.

    Parameters
    ----------
    dims : RatHexmazeDims, optional
        Maze dimensions. If None, uses default dimensions
        (90 cm module width, 11 cm corridor width, 3 modules, 24 nodes per module).
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
    The maze consists of multiple hexagonal modules arranged linearly:
    - Module A at x=0
    - Module B at x=module_width * 1.5
    - Module C at x=module_width * 3.0
    - Corridor bridges connect adjacent modules

    Regions:
    - module_A: Polygon region covering first hexagonal module
    - module_B: Polygon region covering second hexagonal module
    - module_C: Polygon region covering third hexagonal module
    - corridor_AB: Polygon region for bridge between modules A and B
    - corridor_BC: Polygon region for bridge between modules B and C

    Track Graph Topology:
    - Junction nodes arranged in hexagonal lattice within each module
    - 120° angles between corridors at junctions
    - Bridge edges connecting modules

    Examples
    --------
    Create a default Rat HexMaze:

    >>> maze = make_rat_hexmaze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = RatHexmazeDims(module_width=120.0, n_modules=2)
    >>> maze = make_rat_hexmaze(dims=dims, bin_size=4.0)
    >>> "module_A" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_rat_hexmaze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    RatHexmazeDims : Dimension specifications for Rat HexMaze.
    """
    if dims is None:
        dims = RatHexmazeDims()

    # Create hexagonal modules and bridges
    module_polygons, module_centers = _create_hexagonal_modules(dims)
    bridge_polygons, bridge_centers = _create_corridor_bridges(dims, module_centers)

    # Union all polygons into single geometry
    all_polygons = module_polygons + bridge_polygons
    hexmaze_polygon = union_polygons(all_polygons)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=hexmaze_polygon,
        bin_size=bin_size,
        name="rat_hexmaze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add module regions (as point regions at module centers)
    module_names = ["module_A", "module_B", "module_C"]
    for name, center in zip(
        module_names[: dims.n_modules], module_centers, strict=True
    ):
        # Create circular region for module
        module_region = Point(center).buffer(dims.module_width / 3)
        env_2d.regions.add(name, polygon=module_region)

    # Add corridor bridge regions (as point regions at bridge centers)
    bridge_names = ["corridor_AB", "corridor_BC"]
    for name, center in zip(
        bridge_names[: dims.n_modules - 1], bridge_centers, strict=True
    ):
        # Create rectangular region for bridge
        bridge_length = dims.module_width * 0.5
        bridge_region = Point(center).buffer(bridge_length / 2)
        env_2d.regions.add(name, polygon=bridge_region)

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_hexmaze_track_graph(dims, module_centers, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_hexagonal_modules(
    dims: RatHexmazeDims,
) -> tuple[list, list[tuple[float, float]]]:
    """Create hexagonal module polygons.

    Creates simplified hexagonal modules as circular approximations.
    Each module is positioned along the x-axis.

    Parameters
    ----------
    dims : RatHexmazeDims
        Maze dimensions.

    Returns
    -------
    tuple[list, list[tuple[float, float]]]
        - List of Shapely Polygon objects for each module
        - List of module center positions (x, y)
    """
    module_polygons = []
    module_centers = []

    # Position modules along x-axis with spacing
    module_spacing = dims.module_width * 1.5  # 1.5x width spacing

    for i in range(dims.n_modules):
        # Module center position
        center_x = i * module_spacing
        center_y = 0.0
        center = (center_x, center_y)
        module_centers.append(center)

        # Create hexagonal module as circle approximation
        # A hexagon inscribed in a circle of radius R has width 2R
        radius = dims.module_width / 2
        module_poly = Point(center).buffer(radius, quad_segs=6)  # 6 segments = hexagon
        module_polygons.append(module_poly)

    return module_polygons, module_centers


def _create_corridor_bridges(
    dims: RatHexmazeDims,
    module_centers: list[tuple[float, float]],
) -> tuple[list, list[tuple[float, float]]]:
    """Create corridor bridge polygons connecting modules.

    Parameters
    ----------
    dims : RatHexmazeDims
        Maze dimensions.
    module_centers : list[tuple[float, float]]
        List of module center positions.

    Returns
    -------
    tuple[list, list[tuple[float, float]]]
        - List of Shapely Polygon objects for corridor bridges
        - List of bridge center positions (x, y)
    """
    bridge_polygons = []
    bridge_centers = []

    # Create bridges between adjacent modules
    for i in range(len(module_centers) - 1):
        start_center = module_centers[i]
        end_center = module_centers[i + 1]

        # Bridge center is midpoint
        bridge_center = (
            (start_center[0] + end_center[0]) / 2,
            (start_center[1] + end_center[1]) / 2,
        )
        bridge_centers.append(bridge_center)

        # Create corridor connecting modules (with rounded ends)
        # Start at edge of first module, end at edge of second module
        start_radius = dims.module_width / 2
        end_radius = dims.module_width / 2

        # Direction vector from start to end
        dx = end_center[0] - start_center[0]
        dy = end_center[1] - start_center[1]
        dist = np.sqrt(dx**2 + dy**2)
        unit_x = dx / dist
        unit_y = dy / dist

        # Bridge start and end points (at module edges)
        bridge_start = (
            start_center[0] + unit_x * start_radius,
            start_center[1] + unit_y * start_radius,
        )
        bridge_end = (
            end_center[0] - unit_x * end_radius,
            end_center[1] - unit_y * end_radius,
        )

        # Create bridge corridor with rounded ends
        bridge_poly = make_buffered_line(
            start=bridge_start,
            end=bridge_end,
            width=dims.corridor_width,
        )
        bridge_polygons.append(bridge_poly)

    return bridge_polygons, bridge_centers


def _create_hexmaze_track_graph(
    dims: RatHexmazeDims,
    module_centers: list[tuple[float, float]],
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Rat HexMaze.

    Creates a simplified track graph with junction nodes in each module
    connected by corridor bridges. Each module contains a hexagonal lattice
    of nodes with 120° angles.

    Parameters
    ----------
    dims : RatHexmazeDims
        Maze dimensions.
    module_centers : list[tuple[float, float]]
        List of module center positions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the hexmaze track.
    """
    graph = nx.Graph()

    # Create simplified hexagonal lattice within each module
    # For simplicity, create a small hexagonal cluster around each module center
    all_nodes = []
    module_node_names = []

    for module_idx, center in enumerate(module_centers):
        # Create hexagonal lattice nodes in this module
        # Simple pattern: center + 6 surrounding nodes in hexagonal arrangement
        module_nodes = _create_hexagonal_lattice_nodes(
            center=center,
            radius=dims.module_width / 3,
            n_rings=2,  # 1 center + 6 inner + 12 outer = 19 nodes
        )

        # Add nodes to graph with unique names
        module_names = []
        for i, pos in enumerate(module_nodes):
            node_name = f"module_{module_idx}_node_{i}"
            graph.add_node(node_name, pos=pos)
            all_nodes.append((node_name, pos))
            module_names.append(node_name)

        module_node_names.append(module_names)

        # Connect nodes within module (hexagonal lattice with 120° angles)
        _connect_hexagonal_lattice(graph, module_names, module_nodes)

    # Add bridge connections between modules
    for i in range(len(module_centers) - 1):
        # Connect closest nodes between adjacent modules
        module_a_nodes = module_node_names[i]
        module_b_nodes = module_node_names[i + 1]

        # Find closest pair of nodes between modules
        min_dist: float = float("inf")
        closest_pair = None

        for node_a in module_a_nodes:
            pos_a = np.array(graph.nodes[node_a]["pos"])
            for node_b in module_b_nodes:
                pos_b = np.array(graph.nodes[node_b]["pos"])
                dist: float = float(np.linalg.norm(pos_b - pos_a))
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (node_a, node_b)

        # Add bridge edge
        if closest_pair is not None:
            graph.add_edge(closest_pair[0], closest_pair[1], distance=min_dist)

    # Create edge order for linearization (traverse modules sequentially)
    edge_order = list(graph.edges())

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        name="rat_hexmaze_1d",
    )
    env_track.units = "cm"

    return env_track


def _create_hexagonal_lattice_nodes(
    center: tuple[float, float],
    radius: float,
    n_rings: int = 2,
) -> list[tuple[float, float]]:
    """Create nodes arranged in hexagonal lattice rings.

    Parameters
    ----------
    center : tuple[float, float]
        Center position (x, y).
    radius : float
        Radius of each ring.
    n_rings : int, optional
        Number of rings (default: 2).
        n_rings=1 gives 1+6=7 nodes
        n_rings=2 gives 1+6+12=19 nodes

    Returns
    -------
    list[tuple[float, float]]
        List of node positions (x, y).
    """
    nodes = [center]  # Center node

    cx, cy = center

    for ring in range(1, n_rings + 1):
        # Nodes in this ring at 60° intervals
        n_nodes_in_ring = 6 * ring
        ring_radius = radius * ring / n_rings

        for i in range(n_nodes_in_ring):
            # Angle in radians (60° = π/3)
            angle = 2 * np.pi * i / n_nodes_in_ring
            x = cx + ring_radius * np.cos(angle)
            y = cy + ring_radius * np.sin(angle)
            nodes.append((x, y))

    return nodes


def _connect_hexagonal_lattice(
    graph: nx.Graph,
    node_names: list[str],
    node_positions: list[tuple[float, float]],
) -> None:
    """Connect nodes in hexagonal lattice with 120° angles.

    Connects each node to its nearest neighbors to form a hexagonal lattice.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to add edges to.
    node_names : list[str]
        List of node names.
    node_positions : list[tuple[float, float]]
        List of node positions corresponding to node_names.
    """
    # For each node, connect to nearest neighbors (typically 3 for hexagonal lattice)
    for i, (name_i, pos_i) in enumerate(zip(node_names, node_positions, strict=True)):
        pos_i_arr = np.array(pos_i)

        # Find nearest neighbors
        distances = []
        for j, (name_j, pos_j) in enumerate(
            zip(node_names, node_positions, strict=True)
        ):
            if i == j:
                continue
            pos_j_arr = np.array(pos_j)
            dist = np.linalg.norm(pos_j_arr - pos_i_arr)
            distances.append((dist, name_j))

        # Sort by distance and connect to 3 nearest (hexagonal lattice)
        distances.sort()
        for dist, neighbor_name in distances[:3]:
            # Add edge if not already exists
            if not graph.has_edge(name_i, neighbor_name):
                graph.add_edge(name_i, neighbor_name, distance=dist)
