"""Standard maze environments for spatial navigation research.

The layouts follow Figure 1 of Wijnen et al. 2024 (Brain Structure &
Function, doi:10.1007/s00429-024-02771-x). The dimensions come from the
papers that describe each apparatus, cited in each factory's docstring, and
estimates are marked there. All dimensions in centimeters.

This module provides factory functions to create common maze environments:

Simple Corridor Mazes (Panel a):
- T-Maze: Single binary decision with perpendicular arms
- Y-Maze: Three symmetric arms at 120 degree angles
- W-Maze: Frank-lab W-track, three parallel arms joined at the base
- Small Hex Maze: Berke-lab triangular hex maze with movable barriers

Repeated Alleyway Mazes (Panel b):
- Repeated Y-Maze: Sequential Y-junctions with Warner-Warden design
- Repeated T-Maze: Tolman & Nyswander six-unit T maze
- Hampton Court Maze: Small's (1901) rat version of the hedge maze

Open-Field Mazes (Panel c):
- Radial Arm Maze: Central platform with eight radiating arms
- Barnes Maze: Circular platform with perimeter holes
- Cheeseboard Maze: Circular board with 177 food wells
- Watermaze: Circular pool with hidden platform

Structured Lattices (Panel d):
- Crossword Maze: Eight intersecting tracks with four start boxes
- Honeycomb Maze: 37 hexagonal platforms
- Hamlet Maze: Pentagonal ring with alleys that split toward five houses

Complex (Panel e):
- Rat HexMaze: Four HexMaze modules joined by five bridges

Not in the figure:
- Linear Track, Plus Maze, Square Maze (rest box), Sungod Maze

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

from neurospatial.simulation.mazes._base import (
    MAZE_LANDMARK_ROLES,
    MazeDims,
    MazeEnvironments,
)
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
from neurospatial.simulation.mazes.plus_maze import (
    PlusMazeDims,
    make_plus_maze,
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
from neurospatial.simulation.mazes.square_maze import (
    SquareMazeDims,
    make_square_maze,
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
    "MAZE_LANDMARK_ROLES",
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
    # Plus Maze (Cruciate)
    "PlusMazeDims",
    "RadialArmDims",
    # Complex
    "RatHexmazeDims",
    "RepeatedTDims",
    # Repeated Alleyway Mazes
    "RepeatedYDims",
    "SmallHexDims",
    # Open-Field Mazes
    "SquareMazeDims",
    "SungodDims",
    "TMazeDims",
    "WMazeDims",
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
    "make_plus_maze",
    "make_radial_arm_maze",
    "make_rat_hexmaze",
    "make_repeated_t_maze",
    "make_repeated_y_maze",
    "make_small_hex_maze",
    "make_square_maze",
    "make_star_graph",
    "make_sungod_maze",
    "make_t_maze",
    "make_w_maze",
    "make_watermaze",
    "make_y_maze",
    "union_polygons",
]
