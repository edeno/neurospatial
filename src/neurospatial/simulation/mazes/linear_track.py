"""Linear Track maze environment.

The Linear Track is the simplest spatial navigation paradigm - a straight
corridor with reward ports at both ends. Animals run back and forth (laps)
between rewards. This paradigm was foundational for place cell research
(O'Keefe & Dostrovsky 1971, Wilson & McNaughton 1993).

Examples
--------
>>> from neurospatial.simulation.mazes import make_linear_track, LinearTrackDims
>>> maze = make_linear_track()
>>> maze.env_2d.units
'cm'
>>> "reward_left" in maze.env_2d.regions
True
>>> "reward_right" in maze.env_2d.regions
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import make_corridor_polygon


@dataclass(frozen=True)
class LinearTrackDims(MazeDims):
    """Dimension specifications for Linear Track.

    The Linear Track is a simple straight corridor. Standard dimensions
    are based on common experimental setups (Wilson & McNaughton 1993).

    Attributes
    ----------
    length : float
        Length of the track in cm. Default is 150.0 (typical range: 150-200 cm).
    width : float
        Width of the corridor in cm. Default is 10.0.

    Examples
    --------
    >>> dims = LinearTrackDims()
    >>> dims.length
    150.0
    >>> dims.width
    10.0

    >>> custom = LinearTrackDims(length=200.0, width=15.0)
    >>> custom.length
    200.0
    """

    length: float = 150.0
    width: float = 10.0


def make_linear_track(
    dims: LinearTrackDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Linear Track environment.

    Creates a straight corridor with reward ports at both ends. This is the
    simplest maze design, ideal for studying place cells, theta sequences,
    and sharp-wave ripple replay.

    Parameters
    ----------
    dims : LinearTrackDims, optional
        Track dimensions. If None, uses default dimensions (150 cm × 10 cm).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    include_track : bool, optional
        Whether to create linearized track graph (default: True).
        The track graph provides a 1D representation for lap-based analyses.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment
        - env_track: 1D linearized track environment (if include_track=True)

    Notes
    -----
    The track is centered at the origin:
    - X-axis: from -length/2 to +length/2
    - Y-axis: from -width/2 to +width/2

    Regions:
    - reward_left: Point at left end (-length/2, 0)
    - reward_right: Point at right end (+length/2, 0)

    Examples
    --------
    Create a default linear track:

    >>> maze = make_linear_track()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = LinearTrackDims(length=200.0, width=12.0)
    >>> maze = make_linear_track(dims=dims, bin_size=4.0)
    >>> "reward_left" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_linear_track(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    LinearTrackDims : Dimension specifications for Linear Track.
    """
    if dims is None:
        dims = LinearTrackDims()

    # Track is centered at origin
    half_length = dims.length / 2.0

    # Create polygon for 2D environment (horizontal corridor)
    start = (-half_length, 0.0)
    end = (half_length, 0.0)
    polygon = make_corridor_polygon(start=start, end=end, width=dims.width)

    # Create 2D environment
    env_2d = Environment.from_polygon(
        polygon=polygon,
        bin_size=bin_size,
        name="linear_track",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add regions for reward locations
    env_2d.regions.add("reward_left", point=(-half_length, 0.0))
    env_2d.regions.add("reward_right", point=(half_length, 0.0))

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_linear_track_graph(dims, bin_size)

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_linear_track_graph(
    dims: LinearTrackDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph.

    Parameters
    ----------
    dims : LinearTrackDims
        Track dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the track.
    """
    half_length = dims.length / 2.0

    # Create a simple linear graph from start to end
    graph = nx.Graph()

    # Add start and end nodes with positions
    graph.add_node("start", pos=(-half_length, 0.0))
    graph.add_node("end", pos=(half_length, 0.0))

    # Connect with edge that has distance attribute
    distance = dims.length
    graph.add_edge("start", "end", distance=distance)

    # Edge order is just the single edge
    edge_order = [("start", "end")]

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges (just one edge)
        bin_size=bin_size,
        name="linear_track_1d",
    )
    env_track.units = "cm"

    return env_track
