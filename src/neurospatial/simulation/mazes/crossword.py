"""Crossword Maze environment for spatial navigation research.

The Crossword Maze is a lattice of narrow open tracks with four start boxes.
Barriers placed at different intersections each day leave one path from each
start box to the reward, so the animal must plan routes through an
allocentric map rather than repeat a body-turn sequence.

The geometry follows McNamara et al. (2014), Online Methods ("Crossword maze"):
the maze "consisted of four start boxes and eight intersecting open tracks
forming fourteen intersections"; "The width of each track was 5 cm with a
1.5 cm high rim along the edges"; "The entire maze measured 95 cm square
excluding start boxes". The track layout is read from Wijnen et al. (2024),
Fig. 1d.

Each day "two start boxes and one food reward location (at the end of one of
the five tracks protruding from the maze) were selected as in use for that
day". The reward "was 4 μl of condensed milk diluted 30% in water and was
placed on a plastic cap at the goal location. A similar plastic cap (without
reward) was placed in each of the other 4 tracks protruding from the maze."
The maze names the four start boxes and the five caps as landmarks. The start
box and cap sizes are not published: the boxes are squares estimated from the
figure, and each cap is a point region. "Distal cue cards were placed on the
curtain surrounding the maze", with no published placement, so they are not
landmarks.

The tracks are open, with a 1.5 cm rim the animal sees over, so their edges
are drops (``arena.outline == "drop"``).

References
----------
McNamara, C. G., Tejero-Cantero, Á., Trouche, S., Campo-Urriza, N., &
Dupret, D. (2014). Dopaminergic neurons promote hippocampal reactivation and
spatial memory persistence. Nature Neuroscience, 17(12), 1658-1660.
https://doi.org/10.1038/nn.3843

Wijnen, K., Genzel, L., & van der Meij, J. (2024). Rodent maze studies: from
following simple rules to complex map learning. Brain Structure and Function,
229(4), 823–841. https://doi.org/10.1007/s00429-024-02771-x

Examples
--------
>>> from neurospatial.simulation.mazes.crossword import (
...     make_crossword_maze,
...     CrosswordDims,
... )
>>> maze = make_crossword_maze()
>>> maze.env_2d.units
'cm'
>>> "start_box_top_left" in maze.env_2d.regions
True
>>> "cap_north" in maze.env_2d.regions
True
>>> maze.env_2d.landmarks["cap_north"].metadata["role"]
'reward_port'
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import networkx as nx
from shapely.geometry import box as shapely_box

from neurospatial.arena import OpenArena
from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_lengths,
    refuse,
    warn_if_bins_too_coarse,
)
from neurospatial.simulation.mazes._geometry import (
    corridor_end_point,
    make_corridor_polygon,
    single_floor,
    union_polygons,
)

# The layout in grid units, with the origin at the lower-left corner of the
# 5 x 5 unit square (read from Wijnen et al. 2024, Fig. 1d). Each track is
# (fixed coordinate, start, end) along the other axis.
_ROWS = ((4, 1, 5), (3, 0, 4), (2, 1, 5), (1, 2, 5))  # (y, x_start, x_end)
_COLUMNS = ((1, 1, 3), (2, 0, 4), (3, 1, 5), (4, 0, 4))  # (x, y_start, y_end)
_BOXES = {
    "start_box_top_left": (1, 4),
    "start_box_top_right": (5, 4),
    "start_box_bottom_left": (1, 1),
    "start_box_bottom_right": (5, 1),
}
# Free ends of the five tracks that protrude from the maze, where a plastic
# cap sits; the day's reward is on one of them.
_CAP_SITES = {
    "cap_west": (0, 3),
    "cap_east": (5, 2),
    "cap_north": (3, 5),
    "cap_south_left": (2, 0),
    "cap_south_right": (4, 0),
}
_HALF_EXTENT = 2.5  # the square spans 5 grid units; recentre on its middle


@dataclass(frozen=True)
class CrosswordDims(MazeDims):
    """Dimension specifications for Crossword Maze.

    Attributes
    ----------
    grid_spacing : float
        Distance between adjacent parallel tracks in cm. Default is 19.0,
        an estimate: the published 95 cm square divided into five equal
        intervals, consistent with Wijnen et al. (2024), Fig. 1d.
    corridor_width : float
        Width of every track in cm. Default is 5.0 (McNamara et al. 2014).
    box_size : float
        Side of each square start box in cm. Default is 16.0, an estimate
        from Wijnen et al. (2024), Fig. 1d; the box size is not published.

    Examples
    --------
    >>> dims = CrosswordDims()
    >>> dims.grid_spacing
    19.0
    >>> dims.corridor_width
    5.0
    >>> dims.box_size
    16.0

    >>> custom = CrosswordDims(grid_spacing=40.0, corridor_width=12.0, box_size=20.0)
    >>> custom.grid_spacing
    40.0
    """

    grid_spacing: float = 19.0
    corridor_width: float = 5.0
    box_size: float = 16.0

    def __post_init__(self) -> None:
        check_lengths(self, "grid_spacing", "corridor_width", "box_size")
        if self.corridor_width >= self.grid_spacing:
            refuse(
                self,
                ("grid_spacing", "corridor_width"),
                "makes parallel tracks touch, so they merge into one floor",
                "Use a corridor_width below grid_spacing.",
            )
        # A box half box_size from its track's end must stay clear of the
        # parallel track one grid_spacing away.
        if self.box_size / 2.0 + self.corridor_width / 2.0 >= self.grid_spacing:
            refuse(
                self,
                ("box_size", "corridor_width", "grid_spacing"),
                "makes each start box reach the parallel track beside it",
                "Use a box_size below 2 * grid_spacing - corridor_width.",
            )


def make_crossword_maze(
    dims: CrosswordDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Crossword Maze environment.

    Builds the published lattice: four horizontal and four vertical tracks
    that cross at fourteen intersections, four start boxes at track ends and
    five free track ends that protrude from the maze, each with a plastic
    cap where the reward can be placed.

    Parameters
    ----------
    dims : CrosswordDims, optional
        Maze dimensions. If None, uses default dimensions
        (19 cm grid spacing, 5 cm tracks, 16 cm start boxes).
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
    The maze is centered at the origin. Without the start boxes it spans
    ``5 * grid_spacing`` (95 cm by default) in x and y; the right-hand boxes
    extend half a box beyond that square.

    Layout (viewed from above; ``B`` start box, ``C`` cap, ``+`` an
    intersection)::

                          C
                          |
              B-----+-----+-----+-----B
                    |     |     |
        C-----+-----+-----+-----+
              |     |     |     |
              +-----+-----+-----+-----C
              |     |     |     |
              B     +-----+-----+-----B
                    |           |
                    C           C

    Landmarks and regions of the same name:

    - start_box_top_left, start_box_top_right, start_box_bottom_left,
      start_box_bottom_right: the start boxes (role ``"start_box"``). Each
      landmark is the box's centre; each region is the square box, its size
      an estimate.
    - cap_west, cap_east, cap_north, cap_south_left, cap_south_right: the
      plastic caps at the ends of the five protruding tracks, one of which
      holds the day's reward (role ``"reward_port"``, port_type
      ``"plastic_cap"``). Each region is a point
      half a track width back from the track's end; the cap size is not
      published.

    Track Graph Topology:
    - 23 nodes: 14 intersections, 4 start boxes and 5 caps
    - 29 edges: each track split at the intersections along it

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
    >>> "start_box_top_left" in maze.env_2d.regions
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

    w = dims.corridor_width
    half_box = dims.box_size / 2
    tracks = [make_corridor_polygon(start, end, w) for start, end in _tracks(dims)]
    boxes = {}
    for name, grid_xy in _BOXES.items():
        x, y = _to_cm(grid_xy, dims)
        boxes[name] = shapely_box(
            x - half_box, y - half_box, x + half_box, y + half_box
        )

    floor = single_floor(union_polygons([*tracks, *boxes.values()]), dims)
    env_2d = Environment.from_polygon(
        polygon=floor,
        bin_size=bin_size,
        arena=OpenArena(floor, outline="drop"),
        name="crossword_maze",
        connect_diagonal_neighbors=False,  # Manhattan grid only
        units="cm",
    )

    region_builder = Regions()
    landmarks = []
    for name, box in boxes.items():
        region_builder.add(name, polygon=box)
        center = box.centroid.coords[0]
        landmarks.append(Landmark(name, "point", center, {"role": "start_box"}))
    for name, site in _cap_sites(dims).items():
        region_builder.add(name, point=site)
        landmarks.append(
            Landmark(
                name, "point", site, {"role": "reward_port", "port_type": "plastic_cap"}
            )
        )
    env_2d = env_2d.with_regions(region_builder).with_landmarks(Landmarks(landmarks))

    env_track = None
    if include_track:
        env_track = _create_crossword_track_graph(dims, bin_size)

    warn_if_bins_too_coarse(
        env_2d,
        maze="crossword maze",
        bin_size=bin_size,
        narrowest_corridor=dims.corridor_width,
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _to_cm(grid_xy: tuple[float, float], dims: CrosswordDims) -> tuple[float, float]:
    """Grid units (origin at the square's lower-left corner) to centred cm."""
    s = dims.grid_spacing
    return ((grid_xy[0] - _HALF_EXTENT) * s, (grid_xy[1] - _HALF_EXTENT) * s)


def _tracks(
    dims: CrosswordDims,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Centerline ``(start, end)`` of each of the eight tracks, in cm."""
    rows = [((x0, y), (x1, y)) for y, x0, x1 in _ROWS]
    columns = [((x, y0), (x, y1)) for x, y0, y1 in _COLUMNS]
    return [(_to_cm(a, dims), _to_cm(b, dims)) for a, b in rows + columns]


def _cap_sites(dims: CrosswordDims) -> dict[str, tuple[float, float]]:
    """Center of each protruding track's end square, where its cap sits."""
    sites = {}
    for name, grid_end in _CAP_SITES.items():
        end = _to_cm(grid_end, dims)
        track = next(t for t in _tracks(dims) if end in t)
        start = track[0] if track[1] == end else track[1]
        sites[name] = corridor_end_point(start, end, dims.corridor_width)
    return sites


def _create_crossword_track_graph(
    dims: CrosswordDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Crossword Maze.

    Nodes sit at every intersection, start box center and protruding track
    end; each track is split into one edge per pair of consecutive nodes
    along it.

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
    names = {_to_cm(xy, dims): name for name, xy in _BOXES.items()}
    names |= {_to_cm(xy, dims): name for name, xy in _CAP_SITES.items()}
    for y, x0, x1 in _ROWS:
        for x, y0, y1 in _COLUMNS:
            if x0 <= x <= x1 and y0 <= y <= y1:
                names[_to_cm((x, y), dims)] = f"junction_{x}_{y}"

    graph = nx.Graph()
    for pos, name in names.items():
        graph.add_node(name, pos=pos)

    edge_order = []
    for start, end in _tracks(dims):
        axis = 0 if start[1] == end[1] else 1
        on_track = sorted(
            (pos for pos in names if _on_segment(pos, start, end)),
            key=lambda pos: pos[axis],
        )
        for a, b in pairwise(on_track):
            graph.add_edge(names[a], names[b], distance=abs(b[axis] - a[axis]))
            edge_order.append((names[a], names[b]))

    return Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=dims.corridor_width,
        name="crossword_maze_1d",
        units="cm",
    )


def _on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    """Whether ``point`` lies on the axis-aligned segment ``start``-``end``."""
    (x0, x1), (y0, y1) = sorted((start[0], end[0])), sorted((start[1], end[1]))
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1
