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
from neurospatial.simulation.mazes.barnes import (
    BarnesDims,
    make_barnes_maze,
)
from neurospatial.simulation.mazes.cheeseboard import (
    CheeseboardDims,
    make_cheeseboard_maze,
)

# Structured Lattices
from neurospatial.simulation.mazes.crossword import (
    CrosswordDims,
    make_crossword_maze,
)
from neurospatial.simulation.mazes.hamlet import (
    HamletDims,
    make_hamlet_maze,
)
from neurospatial.simulation.mazes.hampton_court import (
    HamptonCourtDims,
    make_hampton_court_maze,
)
from neurospatial.simulation.mazes.hex_small import (
    SmallHexDims,
    make_small_hex_maze,
)
from neurospatial.simulation.mazes.honeycomb import (
    HoneycombDims,
    make_honeycomb_maze,
)

# Simple Corridor Mazes
from neurospatial.simulation.mazes.linear_track import (
    LinearTrackDims,
    make_linear_track,
)
from neurospatial.simulation.mazes.radial_arm import (
    RadialArmDims,
    make_radial_arm_maze,
)

# Complex Mazes
from neurospatial.simulation.mazes.rat_hexmaze import (
    RatHexmazeDims,
    make_rat_hexmaze,
)
from neurospatial.simulation.mazes.repeated_t import (
    RepeatedTDims,
    make_repeated_t_maze,
)

# Repeated Alleyway Mazes
from neurospatial.simulation.mazes.repeated_y import (
    RepeatedYDims,
    make_repeated_y_maze,
)
from neurospatial.simulation.mazes.sungod import (
    SungodDims,
    make_sungod_maze,
)
from neurospatial.simulation.mazes.t_maze import (
    TMazeDims,
    make_t_maze,
)
from neurospatial.simulation.mazes.w_maze import (
    WMazeDims,
    make_w_maze,
)

# Open-Field Mazes
from neurospatial.simulation.mazes.watermaze import (
    WatermazeDims,
    make_watermaze,
)
from neurospatial.simulation.mazes.y_maze import (
    YMazeDims,
    make_y_maze,
)

__all__ = [
    "BarnesDims",
    "CheeseboardDims",
    # Structured Lattices
    "CrosswordDims",
    "HamletDims",
    "HamptonCourtDims",
    "HoneycombDims",
    # Simple Corridor Mazes
    "LinearTrackDims",
    # Base classes
    "MazeDims",
    "MazeEnvironments",
    "RadialArmDims",
    # Complex
    "RatHexmazeDims",
    "RepeatedTDims",
    # Repeated Alleyway Mazes
    "RepeatedYDims",
    "SmallHexDims",
    "TMazeDims",
    "WMazeDims",
    # Open-Field Mazes
    "WatermazeDims",
    "YMazeDims",
    "make_barnes_maze",
    # Geometry helpers
    "make_buffered_line",
    "make_cheeseboard_maze",
    "make_circular_arena",
    "make_corridor_polygon",
    "make_crossword_maze",
    "make_hamlet_maze",
    "make_hampton_court_maze",
    "make_honeycomb_maze",
    "make_linear_track",
    "make_radial_arm_maze",
    "make_rat_hexmaze",
    "make_repeated_t_maze",
    "make_repeated_y_maze",
    "make_small_hex_maze",
    "make_star_graph",
    "make_t_maze",
    "make_w_maze",
    "make_watermaze",
    "make_y_maze",
    "union_polygons",
]
