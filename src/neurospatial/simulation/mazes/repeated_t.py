"""Repeated T-Maze environment for spatial navigation research.

The Repeated T-Maze consists of three vertical stems connected by two horizontal
corridors at different heights. The structure creates multiple T-junction decision
points for navigation tasks.

Structure (ASCII representation)::

                |
                |
    ------------|  ------+-----
         |      |        |
         |      |        |
         |      |        |
    -----|  ----|--------|
         |      |        |
         |      |        |

Key features:
- Three vertical stems (left, center, right)
- Center stem extends above the upper horizontal (top spur)
- Upper horizontal: left section connects to center stem, right section has T-junction
- Lower horizontal: left arm from left stem, right section connects center and right stems
- Gaps between horizontal sections create the T-junction decision points

The maze structure facilitates:
- Multi-location navigation with repeated T-choice points
- Sequential decision paradigms
- Spatial working memory tasks

Examples
--------
>>> from neurospatial.simulation.mazes.repeated_t import (
...     make_repeated_t_maze,
...     RepeatedTDims,
... )
>>> maze = make_repeated_t_maze()
>>> maze.env_2d.units
'cm'
>>> "start" in maze.env_2d.regions
True
>>> maze.env_2d.n_bins > 0
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
class RepeatedTDims(MazeDims):
    """Dimension specifications for Repeated T-Maze.

    The Repeated T-Maze consists of three vertical stems connected by two
    horizontal corridors at different heights, creating multiple T-junction
    decision points.

    Attributes
    ----------
    stem_spacing : float
        Horizontal distance between adjacent stems in cm. Default is 40.0.
    stem_length : float
        Length of vertical stems below the upper horizontal in cm. Default is 60.0.
    top_spur_length : float
        Length of center stem extension above upper horizontal in cm. Default is 20.0.
    arm_length : float
        Length of horizontal arms extending beyond outer stems in cm. Default is 15.0.
    upper_lower_gap : float
        Vertical distance between upper and lower horizontal corridors in cm.
        Default is 30.0.
    width : float
        Width of all corridors in cm. Default is 10.0.

    Examples
    --------
    >>> dims = RepeatedTDims()
    >>> dims.stem_spacing
    40.0
    >>> dims.stem_length
    60.0
    >>> dims.top_spur_length
    20.0
    >>> dims.width
    10.0

    >>> custom = RepeatedTDims(stem_spacing=50.0, stem_length=80.0)
    >>> custom.stem_spacing
    50.0
    """

    stem_spacing: float = 40.0
    stem_length: float = 60.0
    top_spur_length: float = 20.0
    arm_length: float = 15.0
    upper_lower_gap: float = 30.0
    width: float = 10.0


def make_repeated_t_maze(
    dims: RepeatedTDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Repeated T-Maze environment.

    Creates a maze with three vertical stems connected by two horizontal
    corridors at different heights, forming multiple T-junction decision points.

    Parameters
    ----------
    dims : RepeatedTDims, optional
        Maze dimensions. If None, uses default dimensions.
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    include_track : bool, optional
        Whether to create linearized track graph (default: True).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment
        - env_track: 1D linearized track environment (if include_track=True)

    Notes
    -----
    Maze Structure::

                    |
                    |
        ------------|  ------+-----
             |      |        |
             |      |        |
             |      |        |
        -----|  ----|--------|
             |      |        |
             |      |        |

    Components:
    - Three vertical stems (left, center, right)
    - Center stem extends above upper horizontal (top_spur)
    - Upper horizontal: left section to center, right section with T-junction
    - Lower horizontal: left arm, right section connecting center to right

    Regions:
    - start: End of upper-left horizontal arm
    - goal: End of upper-right horizontal arm
    - top_spur: Top of center stem (above upper horizontal)
    - left_stem_bottom, center_stem_bottom, right_stem_bottom: Bottom of stems
    - lower_left_arm: End of lower-left horizontal arm (from left stem)
    - lower_center_left_arm: End of lower-center-left horizontal arm (from center stem)
    - upper_junction, lower_left_junction, lower_center_junction: T-junctions
    - upper_right_junction: T-junction at right stem (upper level)

    Examples
    --------
    Create a default Repeated T-maze:

    >>> maze = make_repeated_t_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Skip track graph creation:

    >>> maze = make_repeated_t_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    RepeatedTDims : Dimension specifications for Repeated T-Maze.
    """
    if dims is None:
        dims = RepeatedTDims()

    polygons = []

    # Define key y-coordinates
    # Upper horizontal is at y = upper_y
    # Lower horizontal is at y = lower_y
    # Stems extend from upper_y down to bottom_y
    # Center stem extends above upper_y to top_y

    upper_y = dims.stem_length  # Upper horizontal level
    lower_y = upper_y - dims.upper_lower_gap  # Lower horizontal level
    bottom_y = 0.0  # Bottom of stems
    top_y = upper_y + dims.top_spur_length  # Top of center stem

    # Define x-coordinates for three stems
    left_x = 0.0
    center_x = dims.stem_spacing
    right_x = 2 * dims.stem_spacing

    # === Create vertical stems ===

    # Left stem: from bottom to upper horizontal
    polygons.append(
        make_corridor_polygon(
            start=(left_x, bottom_y), end=(left_x, upper_y), width=dims.width
        )
    )

    # Center stem: from bottom through upper horizontal to top spur
    polygons.append(
        make_corridor_polygon(
            start=(center_x, bottom_y), end=(center_x, top_y), width=dims.width
        )
    )

    # Right stem: from bottom to upper horizontal
    polygons.append(
        make_corridor_polygon(
            start=(right_x, bottom_y), end=(right_x, upper_y), width=dims.width
        )
    )

    # === Create upper horizontal corridor ===

    # Upper-left section: from left arm end to center stem
    upper_left_start = (left_x - dims.arm_length, upper_y)
    upper_left_end = (center_x, upper_y)
    polygons.append(
        make_corridor_polygon(
            start=upper_left_start, end=upper_left_end, width=dims.width
        )
    )

    # Upper-right section: from center+gap to right arm end (T-junction at right stem)
    upper_right_start = (center_x + dims.arm_length, upper_y)
    upper_right_end = (right_x + dims.arm_length, upper_y)
    polygons.append(
        make_corridor_polygon(
            start=upper_right_start, end=upper_right_end, width=dims.width
        )
    )

    # === Create lower horizontal corridor ===

    # Lower-left section: left arm extending from left stem
    lower_left_start = (left_x - dims.arm_length, lower_y)
    lower_left_end = (left_x, lower_y)
    polygons.append(
        make_corridor_polygon(
            start=lower_left_start, end=lower_left_end, width=dims.width
        )
    )

    # Lower-center-left section: left arm extending from center stem
    lower_center_left_start = (center_x - dims.arm_length, lower_y)
    lower_center_left_end = (center_x, lower_y)
    polygons.append(
        make_corridor_polygon(
            start=lower_center_left_start, end=lower_center_left_end, width=dims.width
        )
    )

    # Lower-right section: from center stem to right stem
    lower_right_start = (center_x, lower_y)
    lower_right_end = (right_x, lower_y)
    polygons.append(
        make_corridor_polygon(
            start=lower_right_start, end=lower_right_end, width=dims.width
        )
    )

    # Union all corridors
    maze_polygon = union_polygons(polygons)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=maze_polygon,
        bin_size=bin_size,
        name="repeated_t_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # === Add regions ===

    # Start and goal at ends of upper horizontal arms
    env_2d.regions.add("start", point=(left_x - dims.arm_length, upper_y))
    env_2d.regions.add("goal", point=(right_x + dims.arm_length, upper_y))

    # Top spur (center stem above upper horizontal)
    env_2d.regions.add("top_spur", point=(center_x, top_y))

    # Bottom of stems
    env_2d.regions.add("left_stem_bottom", point=(left_x, bottom_y))
    env_2d.regions.add("center_stem_bottom", point=(center_x, bottom_y))
    env_2d.regions.add("right_stem_bottom", point=(right_x, bottom_y))

    # Lower arm ends
    env_2d.regions.add("lower_left_arm", point=(left_x - dims.arm_length, lower_y))
    env_2d.regions.add(
        "lower_center_left_arm", point=(center_x - dims.arm_length, lower_y)
    )

    # T-junction points (where decisions are made)
    env_2d.regions.add("upper_junction", point=(center_x, upper_y))
    env_2d.regions.add("lower_left_junction", point=(left_x, lower_y))
    env_2d.regions.add("lower_center_junction", point=(center_x, lower_y))
    env_2d.regions.add("upper_right_junction", point=(right_x, upper_y))

    # Store stem positions for track graph creation
    stem_positions = {
        "left_x": left_x,
        "center_x": center_x,
        "right_x": right_x,
        "upper_y": upper_y,
        "lower_y": lower_y,
        "bottom_y": bottom_y,
        "top_y": top_y,
    }

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_repeated_t_maze_track_graph(dims, bin_size, stem_positions)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_repeated_t_maze_track_graph(
    dims: RepeatedTDims,
    bin_size: float,
    stem_positions: dict[str, float],
) -> Environment:
    """Create the 1D linearized track graph for Repeated T-Maze.

    Parameters
    ----------
    dims : RepeatedTDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.
    stem_positions : dict
        Dictionary with stem x/y coordinates.

    Returns
    -------
    Environment
        1D linearized environment representing the Repeated T-maze track.
    """
    graph = nx.Graph()

    # Extract positions
    left_x = stem_positions["left_x"]
    center_x = stem_positions["center_x"]
    right_x = stem_positions["right_x"]
    upper_y = stem_positions["upper_y"]
    lower_y = stem_positions["lower_y"]
    bottom_y = stem_positions["bottom_y"]
    top_y = stem_positions["top_y"]

    # === Add nodes ===

    # Start and goal (ends of upper horizontal arms)
    graph.add_node("start", pos=(left_x - dims.arm_length, upper_y))
    graph.add_node("goal", pos=(right_x + dims.arm_length, upper_y))

    # Top spur
    graph.add_node("top_spur", pos=(center_x, top_y))

    # Upper junction (center stem at upper level)
    graph.add_node("upper_junction", pos=(center_x, upper_y))

    # Upper-right junction (right stem at upper level)
    graph.add_node("upper_right_junction", pos=(right_x, upper_y))

    # Left stem nodes
    graph.add_node("left_upper", pos=(left_x, upper_y))
    graph.add_node("left_lower", pos=(left_x, lower_y))
    graph.add_node("left_stem_bottom", pos=(left_x, bottom_y))

    # Center stem nodes (at lower level)
    graph.add_node("center_lower", pos=(center_x, lower_y))
    graph.add_node("center_stem_bottom", pos=(center_x, bottom_y))

    # Right stem nodes
    graph.add_node("right_lower", pos=(right_x, lower_y))
    graph.add_node("right_stem_bottom", pos=(right_x, bottom_y))

    # Lower arm ends
    graph.add_node("lower_left_arm", pos=(left_x - dims.arm_length, lower_y))
    graph.add_node("lower_center_left_arm", pos=(center_x - dims.arm_length, lower_y))

    # === Add edges ===

    # Upper-left horizontal: start -> left_upper -> upper_junction
    graph.add_edge("start", "left_upper", distance=dims.arm_length)
    graph.add_edge("left_upper", "upper_junction", distance=dims.stem_spacing)

    # Top spur: upper_junction -> top_spur
    graph.add_edge("upper_junction", "top_spur", distance=dims.top_spur_length)

    # Upper-right horizontal: upper_right_junction -> goal
    graph.add_edge("upper_right_junction", "goal", distance=dims.arm_length)

    # Left stem: left_upper -> left_lower -> left_stem_bottom
    graph.add_edge("left_upper", "left_lower", distance=dims.upper_lower_gap)
    graph.add_edge("left_lower", "left_stem_bottom", distance=lower_y - bottom_y)

    # Lower-left arm: left_lower -> lower_left_arm
    graph.add_edge("left_lower", "lower_left_arm", distance=dims.arm_length)

    # Center stem: upper_junction -> center_lower -> center_stem_bottom
    graph.add_edge("upper_junction", "center_lower", distance=dims.upper_lower_gap)
    graph.add_edge("center_lower", "center_stem_bottom", distance=lower_y - bottom_y)

    # Lower-center-left arm: center_lower -> lower_center_left_arm
    graph.add_edge("center_lower", "lower_center_left_arm", distance=dims.arm_length)

    # Right stem: upper_right_junction -> right_lower -> right_stem_bottom
    graph.add_edge("upper_right_junction", "right_lower", distance=dims.upper_lower_gap)
    graph.add_edge("right_lower", "right_stem_bottom", distance=lower_y - bottom_y)

    # Lower-right horizontal: center_lower -> right_lower
    graph.add_edge("center_lower", "right_lower", distance=dims.stem_spacing)

    # Define edge order for linearization
    # Main path: start -> through maze -> goal, with branches
    edge_order = [
        # Upper-left section
        ("start", "left_upper"),
        ("left_upper", "upper_junction"),
        # Top spur (branch)
        ("upper_junction", "top_spur"),
        # Left stem down (branch from left_upper)
        ("left_upper", "left_lower"),
        ("left_lower", "lower_left_arm"),  # Lower-left arm branch
        ("left_lower", "left_stem_bottom"),
        # Center stem down
        ("upper_junction", "center_lower"),
        ("center_lower", "lower_center_left_arm"),  # Lower-center-left arm branch
        ("center_lower", "center_stem_bottom"),
        # Lower horizontal to right
        ("center_lower", "right_lower"),
        ("right_lower", "right_stem_bottom"),
        # Right stem up to upper-right junction
        ("right_lower", "upper_right_junction"),
        ("upper_right_junction", "goal"),
    ]

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        name="repeated_t_maze_1d",
    )
    env_track.units = "cm"

    return env_track
