"""Linear Track maze environment.

The Linear Track is the simplest spatial navigation paradigm - a straight
corridor with a reward well at each end. Animals run back and forth (laps)
between the wells, which makes the track a standard apparatus for place-cell,
theta-sequence and replay studies.

The default length and the wells follow Kim & Frank (2009), PLoS One 4:e5494,
doi:10.1371/journal.pone.0005494, who trained rats on "an elevated linear
track (150 cm long, 6 cm wide)": "Rats were motivated with droplets of
sweetened milk, which were automatically dispensed in food wells located at
the two ends of the track."

The track is elevated, so its edges are drops (``arena.outline == "drop"``).

Examples
--------
>>> from neurospatial.simulation.mazes import make_linear_track, LinearTrackDims
>>> maze = make_linear_track()
>>> maze.env_2d.units
'cm'
>>> list(maze.env_2d.landmarks)
['well_left', 'well_right']
>>> maze.env_2d.landmarks["well_left"].metadata["role"]
'reward_port'
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from neurospatial.arena import OpenArena
from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_corridor_polygon,
)


@dataclass(frozen=True)
class LinearTrackDims(MazeDims):
    """Dimension specifications for Linear Track.

    The Linear Track is a simple straight corridor. The default length is the
    150 cm track of Kim & Frank (2009), doi:10.1371/journal.pone.0005494.

    Attributes
    ----------
    length : float
        Length of the track in cm, end wall to end wall. Default is 150.0.
    width : float
        Width of the corridor in cm. Default is 10.0, wider than the 6 cm
        track of Kim & Frank (2009); pass ``width=6.0`` to match it.

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

    Creates a straight corridor with a reward well at each end. This is the
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

    Each food well is a landmark in ``env_2d.landmarks`` with role
    ``"reward_port"``, and a point region of the same name. Both sit in the
    end square of the track, half a track width back from the end wall:

    - well_left: Point at left end (-length/2 + width/2, 0)
    - well_right: Point at right end (+length/2 - width/2, 0)

    The regions are points because Kim & Frank (2009) do not give the wells'
    size. They scored well visits with "a circle with a 15-cm radius" around
    each well; that circle is an analysis choice, not the well, so add a
    region of that size to reproduce their visits. The track's end nodes, at
    the end walls, share the wells' names.

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
    >>> "well_left" in maze.env_2d.regions
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
        arena=OpenArena(polygon, outline="drop"),
        name="linear_track",
        connect_diagonal_neighbors=True,
    )
    env_2d = env_2d.with_metadata(units="cm")

    # One food well at each end; Kim & Frank (2009) do not give its size
    wells = {
        "well_left": corridor_end_point(end, start, dims.width),
        "well_right": corridor_end_point(start, end, dims.width),
    }
    region_builder = Regions()
    for name, position in wells.items():
        region_builder.add(name, point=position)
    landmarks = Landmarks(
        [
            Landmark(
                name,
                "point",
                position,
                metadata={"role": "reward_port", "port_type": "food_well"},
            )
            for name, position in wells.items()
        ]
    )
    env_2d = env_2d.with_regions(region_builder).with_landmarks(landmarks)

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

    # End nodes at the end walls, named like the wells there
    graph.add_node("well_left", pos=(-half_length, 0.0))
    graph.add_node("well_right", pos=(half_length, 0.0))

    # Connect with edge that has distance attribute
    distance = dims.length
    graph.add_edge("well_left", "well_right", distance=distance)

    # Edge order is just the single edge
    edge_order = [("well_left", "well_right")]

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,  # No spacing between edges (just one edge)
        bin_size=bin_size,
        track_width=dims.width,
        name="linear_track_1d",
    )
    env_track = env_track.with_metadata(units="cm")

    return env_track
