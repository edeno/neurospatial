"""Standard maze environments for spatial navigation research.

Based on Wijnen et al. 2024 (Brain Structure & Function), Figure 1.
All dimensions in centimeters. Coordinate origin at maze center.

This module provides factory functions to create common maze environments:

Simple Corridor Mazes (Panel a):
- Linear Track: Basic straight corridor for place cell studies
- T-Maze: Single binary decision with perpendicular arms
- Y-Maze: Three symmetric arms at 120 degree angles
- W-Maze: Three parallel wells connected at base

Open-Field Mazes (Panel c):
- Radial Arm Maze: Central platform with radiating arms
- Barnes Maze: Circular platform with perimeter holes
- Cheeseboard Maze: Circular platform with surface wells
- Watermaze: Circular pool with hidden platform

Repeated Alleyway Mazes (Panel b):
- Repeated Y-Maze: Sequential Y-junctions with Warner-Warden design
- Repeated T-Maze: Spine with perpendicular T-arms (comb shape)
- Hampton Court Maze: Complex labyrinth with dead ends

Structured Lattices (Panel d):
- Crossword Maze: 4x4 Manhattan-style grid
- Honeycomb Maze: Hexagonal platform array
- Hamlet Maze: Pentagon ring with branching arms

Complex (Panel e):
- Rat HexMaze: Large-scale 120 degree corridor network

Examples
--------
>>> from neurospatial.simulation.mazes import MazeEnvironments, MazeDims
>>> dims = MazeDims()
>>> dims
MazeDims()

Notes
-----
All factory functions return MazeEnvironments containing:
- env_2d: 2D polygon-based Environment
- env_track: Optional 1D linearized track graph
"""

from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    make_buffered_line,
    make_circular_arena,
    make_corridor_polygon,
    make_star_graph,
    union_polygons,
)
from neurospatial.simulation.mazes.linear_track import (
    LinearTrackDims,
    make_linear_track,
)

__all__ = [
    "LinearTrackDims",
    "MazeDims",
    "MazeEnvironments",
    "make_buffered_line",
    "make_circular_arena",
    "make_corridor_polygon",
    "make_linear_track",
    "make_star_graph",
    "union_polygons",
]
