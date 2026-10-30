"""Hampton Court Maze: Small's (1901) rat maze.

Small (1901) built one of the first rat mazes, modelled on the hedge maze at
Hampton Court Palace, and studied how rats learn to find food in its central
open space (p. 208):

    "The Hampton Court Maze served as model for the apparatus. [...] The
    dimensions were 6 by 8 feet. [...] The height of the sides was 4 inches;
    the width of galleries, the same. In the center was a large open space.
    [...] The entrance is marked by the figure O."

There is no start box. The rats' cage stood "flush with the maze at O", and a
sliding glass door "closed the passage into the maze". The goal is the centre,
which Small labels C: "Food was placed in the central enclosure. To obtain
this, the rats had to find their way through the tangled maze" (p. 209). The
food was bread and milk (p. 217); Small gives no dish or its place within the
enclosure, so the enclosure's landmark is its centre.

The wall layout is Small's ground plan (p. 207), turned so the entrance is at
the bottom: an 18 x 14 grid of galleries inside the 8 x 6 ft frame, a central
open space entered from below by a short stem, seven blind alleys (Small's
numerals 1-7) and one loop round the centre. Small gives the frame and the
gallery width but not the wall positions, and his plan is not drawn to scale
against them: thin partitions between 4-inch galleries would fill only 6 of
the 8 feet. Here the plan's walls stand on the grid, every gallery is 4
inches wide, and partitions take the remaining space (3.6 cm between columns,
3.1 cm between rows).

The galleries are walled (``arena.outline == "wall"``): Small gives "the
height of the sides" as 4 inches but not their material, so the walls are
modeled as opaque.

Reference: Small, W. S. (1901). Experimental study of the mental processes of
the rat. II. American Journal of Psychology, 12(2), 206-239.
https://doi.org/10.2307/1412534. Drawn as the Hampton Court maze in Wijnen et
al. (2024), Brain Structure and Function 229(4), 823–841, Fig. 1b.
https://doi.org/10.1007/s00429-024-02771-x

Examples
--------
>>> from neurospatial.simulation.mazes.hampton_court import (
...     make_hampton_court_maze,
...     HamptonCourtDims,
... )
>>> maze = make_hampton_court_maze()
>>> maze.env_2d.units
'cm'
>>> sorted(name for name in maze.env_2d.regions if name.startswith("blind_"))
['blind_1', 'blind_2', 'blind_3', 'blind_4', 'blind_5', 'blind_6', 'blind_7']
>>> maze.env_2d.landmarks["central_enclosure"].metadata["role"]
'goal_box'
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import networkx as nx
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

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

_Cell = tuple[int, int]

# Small's ground plan (1901, p. 207), entrance at the bottom, traced and placed
# on its grid of galleries. A cell (column, row) is one gallery square, columns
# 0-17 west to east and rows 0-13 south to north. Walls are polylines through
# grid-line crossings (column line 0-18, row line 0-14): column line c runs
# between cell columns c - 1 and c. The outer frame is the grid's edge, open
# only at the entrance below cell (8, 0).
_N_COLUMNS = 18
_N_ROWS = 14
_ENTRANCE_COLUMN = 8
_WALLS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 7), (1, 7)),
    ((18, 7), (17, 7)),
    ((5, 13), (1, 13), (1, 8), (2, 8), (2, 6), (1, 6), (1, 1), (18, 1)),
    ((6, 14), (6, 12), (2, 12), (2, 9)),
    ((8, 11), (3, 11), (3, 3)),
    ((5, 11), (5, 3), (7, 3), (7, 2), (2, 2), (2, 5)),
    ((4, 10), (4, 2)),
    ((9, 14), (9, 13), (7, 13), (7, 11)),
    # The T north of the centre. Its bar's west end traces at column line 7.4;
    # 8 keeps the passage at its west end that the plan shows.
    ((8, 12), (10, 12)),
    ((9, 12), (9, 10)),
    ((8, 1), (8, 4), (6, 4), (6, 10), (11, 10), (11, 4), (9, 4), (9, 2), (17, 2),
     (17, 6)),
    ((10, 13), (11, 13), (11, 11)),
    ((10, 11), (12, 11), (12, 3), (10, 3)),
    ((12, 14), (12, 12), (16, 12), (16, 9)),
    ((13, 13), (17, 13), (17, 8), (16, 8), (16, 3), (14, 3)),
    ((14, 11), (14, 2)),
    ((13, 3), (13, 11), (15, 11), (15, 4)),
)  # fmt: skip
# The central open space: cell columns 6-10, rows 4-9.
_ROOM_COLUMNS = range(6, 11)
_ROOM_ROWS = range(4, 10)
# Small's numerals 1-7: each blind alley leaves a choice point (first cell)
# through its first gallery cell (second cell), where the numeral stands on
# the plan. Alley 4 is not a dead end: it leads round the centre to the choice
# point of alley 5.
_BLIND_ALLEY_ENTRANCES: dict[int, tuple[_Cell, _Cell]] = {
    1: ((8, 0), (9, 0)),
    2: ((2, 8), (2, 9)),
    3: ((2, 5), (2, 4)),
    4: ((12, 11), (12, 10)),
    5: ((12, 2), (13, 2)),
    6: ((15, 8), (15, 7)),
    7: ((16, 6), (16, 5)),
}


@dataclass(frozen=True)
class HamptonCourtDims(MazeDims):
    """Dimensions of Small's (1901) Hampton Court maze.

    Attributes
    ----------
    frame_width : float
        East-west extent of the frame in cm. Default is 243.84 (8 ft, Small
        1901).
    frame_depth : float
        North-south extent of the frame in cm, entrance side to far side.
        Default is 182.88 (6 ft, Small 1901). Small's "height of the sides"
        is the walls' height, 4 in, which the floor plan does not use.
    corridor_width : float
        Width of every gallery in cm. Default is 10.16 (4 in, Small 1901).

    Raises
    ------
    ValueError
        If a length is not positive and finite, or the frame is too small to
        hold 18 x 14 galleries of ``corridor_width``.

    Notes
    -----
    The walls sit on a grid of 18 x 14 galleries, so the frame must be wider
    than 18 and deeper than 14 galleries; partitions fill what is left.

    Examples
    --------
    >>> dims = HamptonCourtDims()
    >>> dims.frame_width, dims.frame_depth, dims.corridor_width
    (243.84, 182.88, 10.16)

    >>> HamptonCourtDims(corridor_width=8.0).corridor_width
    8.0
    """

    frame_width: float = 243.84
    frame_depth: float = 182.88
    corridor_width: float = 10.16

    def __post_init__(self) -> None:
        check_lengths(self, "frame_width", "frame_depth", "corridor_width")
        if (
            self.frame_width <= _N_COLUMNS * self.corridor_width
            or self.frame_depth <= _N_ROWS * self.corridor_width
        ):
            refuse(
                self,
                ("frame_width", "frame_depth", "corridor_width"),
                f"cannot hold {_N_COLUMNS} x {_N_ROWS} galleries with walls "
                "between them",
                f"Make the frame larger than {_N_COLUMNS} x {_N_ROWS} "
                "corridor widths, or the galleries narrower.",
            )


def make_hampton_court_maze(
    dims: HamptonCourtDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create Small's (1901) Hampton Court maze.

    Parameters
    ----------
    dims : HamptonCourtDims, optional
        Maze dimensions. If None, uses Small's 8 x 6 ft frame and 4-in
        galleries.
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0). Bins on either side of a
        partition thinner than a bin (3.1 cm at the default dimensions) are
        not joined through it.
    include_track : bool, optional
        Whether to create the linearized track graph (default: True).

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: the walkable floor (galleries and central open space)
        - env_track: gallery centrelines as a track graph (if include_track)

    Notes
    -----
    The origin is the centre of the frame; x runs east and y runs north, with
    the entrance in the middle of the south (bottom) side.

    Landmark: central_enclosure, the centre of the central open space,
    where Small put the food (role ``"goal_box"``).

    Regions:

    - entrance: the entrance gap in the south wall, Small's O, where the
      rats' cage opened into the maze
    - central_enclosure: the central open space, Small's C (polygon)
    - blind_1 ... blind_7: Small's seven blind alleys, each the run of
      gallery from the choice point where the numeral stands to the next dead
      end or choice point (polygons)

    Track graph: gallery centrelines with nodes at the entrance, the centre
    of the central enclosure, choice points, corners and dead ends. It has 8 choice points (Small's
    seven plus the fork he marks x, where both ways lead on), 6 dead ends
    and one loop round the centre (Small's "circuit immediately around the
    center"). The track runs up the middle of the stem into the open space,
    so positions near the sides of the open space lie off the track.

    Examples
    --------
    >>> maze = make_hampton_court_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.regions["blind_3"].kind
    'polygon'

    A smaller copy with the same layout:

    >>> dims = HamptonCourtDims(
    ...     frame_width=200.0, frame_depth=150.0, corridor_width=8.0
    ... )
    >>> maze = make_hampton_court_maze(dims=dims, include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    HamptonCourtDims : Dimension specifications for the Hampton Court maze.
    """
    if dims is None:
        dims = HamptonCourtDims()
    galleries = _gallery_graph()
    floor = _floor_polygon(galleries, dims)
    env_2d = Environment.from_polygon(
        polygon=floor,
        bin_size=bin_size,
        name="hampton_court_maze",
        connect_diagonal_neighbors=True,
        units="cm",
    )

    regions = Regions()
    regions.add("entrance", point=_entrance(dims))
    regions.add("central_enclosure", polygon=_room_polygon(dims))
    for number, entrance in _BLIND_ALLEY_ENTRANCES.items():
        cells = _blind_alley_cells(galleries, entrance)
        regions.add(f"blind_{number}", polygon=_cells_polygon(cells, galleries, dims))
    goal = Landmark(
        "central_enclosure",
        "point",
        _room_center(dims),
        {"role": "goal_box", "food": "bread and milk"},
    )
    env_2d = env_2d.with_regions(regions).with_landmarks(Landmarks([goal]))

    env_track = None
    if include_track:
        env_track = _create_hampton_court_track_graph(galleries, dims, bin_size)

    warn_if_bins_too_coarse(
        env_2d,
        maze="Hampton Court maze",
        bin_size=bin_size,
        narrowest_corridor=dims.corridor_width,
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _pitch(dims: HamptonCourtDims) -> tuple[float, float]:
    """Centre-to-centre spacing of neighbouring galleries (x, y), in cm."""
    return (
        (dims.frame_width - dims.corridor_width) / (_N_COLUMNS - 1),
        (dims.frame_depth - dims.corridor_width) / (_N_ROWS - 1),
    )


def _cell_center(cell: _Cell, dims: HamptonCourtDims) -> tuple[float, float]:
    """Centre of a gallery cell, in cm from the centre of the frame."""
    half = dims.corridor_width / 2.0
    pitch_x, pitch_y = _pitch(dims)
    return (
        -dims.frame_width / 2.0 + half + cell[0] * pitch_x,
        -dims.frame_depth / 2.0 + half + cell[1] * pitch_y,
    )


def _entrance(dims: HamptonCourtDims) -> tuple[float, float]:
    """The entrance gap in the middle of the south wall."""
    x, _ = _cell_center((_ENTRANCE_COLUMN, 0), dims)
    return (x, -dims.frame_depth / 2.0)


def _room_center(dims: HamptonCourtDims) -> tuple[float, float]:
    """Centre of the central open space."""
    x0, y0 = _cell_center((_ROOM_COLUMNS[0], _ROOM_ROWS[0]), dims)
    x1, y1 = _cell_center((_ROOM_COLUMNS[-1], _ROOM_ROWS[-1]), dims)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _room_polygon(dims: HamptonCourtDims) -> Polygon:
    """The central open space: its galleries and the open floor between them."""
    half = dims.corridor_width / 2.0
    x0, y0 = _cell_center((_ROOM_COLUMNS[0], _ROOM_ROWS[0]), dims)
    x1, y1 = _cell_center((_ROOM_COLUMNS[-1], _ROOM_ROWS[-1]), dims)
    return box(x0 - half, y0 - half, x1 + half, y1 + half)


def _wall_segments() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Every wall as unit-length segments between neighbouring grid crossings."""
    segments = []
    for wall in _WALLS:
        for (x0, y0), (x1, y1) in pairwise(wall):
            step_x, step_y = int(np.sign(x1 - x0)), int(np.sign(y1 - y0))
            for k in range(max(abs(x1 - x0), abs(y1 - y0))):
                start = (x0 + k * step_x, y0 + k * step_y)
                segments.append((start, (start[0] + step_x, start[1] + step_y)))
    return segments


def _gallery_graph() -> nx.Graph:
    """Gallery cells, joined where no wall separates neighbouring cells."""
    graph = nx.grid_2d_graph(_N_COLUMNS, _N_ROWS)
    for (x0, y0), (x1, y1) in _wall_segments():
        if x0 == x1:  # on column line x0, between rows min(y0, y1) and +1
            pair = ((x0 - 1, min(y0, y1)), (x0, min(y0, y1)))
        else:  # on row line y0, between columns min(x0, x1) and +1
            pair = ((min(x0, x1), y0 - 1), (min(x0, x1), y0))
        if graph.has_edge(*pair):
            graph.remove_edge(*pair)
    return graph


def _cell_box(cell: _Cell, dims: HamptonCourtDims) -> Polygon:
    half = dims.corridor_width / 2.0
    x, y = _cell_center(cell, dims)
    return box(x - half, y - half, x + half, y + half)


def _cells_polygon(
    cells: list[_Cell], galleries: nx.Graph, dims: HamptonCourtDims
) -> Polygon:
    """Floor of ``cells``: their squares and the open gaps between them."""
    members = set(cells)
    pieces = [_cell_box(cell, dims) for cell in cells]
    for a, b in galleries.edges(cells):
        if a in members and b in members:
            pieces.append(_cell_box(a, dims).union(_cell_box(b, dims)).envelope)
    return unary_union(pieces)


def _floor_polygon(galleries: nx.Graph, dims: HamptonCourtDims) -> Polygon:
    """The walkable floor: every gallery, the gaps between joined cells and the
    wall-free grid crossings (inside the central open space)."""
    floor = _cells_polygon(list(galleries.nodes), galleries, dims)
    on_wall = {point for segment in _wall_segments() for point in segment}
    half = dims.corridor_width / 2.0
    crossings = []
    for column_line in range(1, _N_COLUMNS):
        for row_line in range(1, _N_ROWS):
            if (column_line, row_line) in on_wall:
                continue
            x0, y0 = _cell_center((column_line - 1, row_line - 1), dims)
            x1, y1 = _cell_center((column_line, row_line), dims)
            crossings.append(box(x0 + half, y0 + half, x1 - half, y1 - half))
    return unary_union([floor, *crossings])


def _blind_alley_cells(
    galleries: nx.Graph, entrance: tuple[_Cell, _Cell]
) -> list[_Cell]:
    """Cells from a blind alley's first cell to its dead end or next fork."""
    previous, cell = entrance
    cells = [cell]
    while galleries.degree(cell) == 2:
        previous, cell = cell, next(n for n in galleries[cell] if n != previous)
        cells.append(cell)
    if galleries.degree(cell) > 2:  # stop before the next choice point
        cells.pop()
    return cells


def _create_hampton_court_track_graph(
    galleries: nx.Graph,
    dims: HamptonCourtDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph along the gallery centrelines.

    The central open space becomes one ``central_enclosure`` node at its
    centre, reached up the stem; the entrance becomes an ``entrance`` node in
    the south wall.
    Straight runs of cells collapse into single edges, so nodes remain only
    at the entrance, the enclosure, choice points, corners and dead ends.
    """
    room = {(c, r) for c in _ROOM_COLUMNS for r in _ROOM_ROWS}
    cells = galleries.copy()
    room_exits = {n for cell in room for n in cells[cell] if n not in room}
    cells.remove_nodes_from(room)
    cells.add_edges_from(("central_enclosure", cell) for cell in room_exits)
    cells.add_edge("entrance", (_ENTRANCE_COLUMN, 0))

    def position(node: _Cell | str) -> np.ndarray:
        if isinstance(node, str):
            return np.asarray(
                _entrance(dims) if node == "entrance" else _room_center(dims)
            )
        return np.asarray(_cell_center(node, dims))

    def is_vertex(node: _Cell | str) -> bool:
        if cells.degree(node) != 2:
            return True
        a, b = (position(n) - position(node) for n in cells[node])
        return bool(abs(a[0] * b[1] - a[1] * b[0]) > 1e-9)  # corner

    def name(node: _Cell | str) -> str:
        return node if isinstance(node, str) else f"cell_{node[0]}_{node[1]}"

    vertices = {node for node in cells if is_vertex(node)}
    graph = nx.Graph()
    for node in vertices:
        graph.add_node(name(node), pos=tuple(float(v) for v in position(node)))
    for node in vertices:
        for neighbor in cells[node]:
            previous, current = node, neighbor
            while current not in vertices:
                previous, current = (
                    current,
                    next(n for n in cells[current] if n != previous),
                )
            distance = float(np.linalg.norm(position(current) - position(node)))
            graph.add_edge(name(node), name(current), distance=distance)

    return Environment.from_graph(
        graph=graph,
        edge_order=list(nx.edge_dfs(graph, "entrance")),
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=dims.corridor_width,
        name="hampton_court_1d",
        units="cm",
    )
