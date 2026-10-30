"""Hamlet Maze environment for spatial navigation research.

The Hamlet Maze is a star-shaped "village" for mice: a central walled
pentagon, the agora, inside a ring of alleys, five alleys radiating from the ring, and at
the end of each radial alley a split into two arms that lead to two
different goal boxes, the "houses". Each house sits at a point of the star
and is reached from the two neighbouring radial alleys. Solid blocks fill the
space between the outer alleys.

Wijnen et al. (2024) describe the layout: "Both mazes consist of five central
alleys forming a pentagonal ring in the middle of the maze and five
peripheral alleys radiating from this central ring. In the Hamlet maze,
however, the peripheral alleys split up in two arms, each leading to a
different goal box (each containing a different reward)". Crouzier et al.
(2020, Methods) describe the houses and the start:

    "The functionalized houses contained a food dispenser (physiological
    function encoded: Eat), two water dispensers (Drink), a novomaze®
    (Viewpoint) (Hide), a running wheel (Run) or a compartment for a
    stranger mouse (Interact). A wire mesh screen separated the stranger
    mouse, whereas the other houses could be entered freely."

    "The agora served as a gathering area and as a start box for training
    and test trials."

Memory is tested by depriving mice of water and measuring the "latency to
reach the goal" (the Drink house) from the agora (Crouzier et al. 2020; Hua
et al. 2023). The houses keep their places: Crouzier et al. (2020, Fig. 1b)
map Hide at the top and, clockwise, Drink, Eat, Interact and Run, and an
overhead photograph labelled with the houses (Couly 2018, Fig. II-42) shows
the same order. The same map draws the agora as a walled pentagon with two
exits onto the ring, one through the side facing the Hide house and one
through the opposite corner; Couly (2018) writes that "L'Agora présente deux
sorties différentes" (the agora has two different exits).

Couly (2018, p. 160) gives the apparatus "un diamètre de 1,6 m", streets
"30 x 8 cm" and houses of "taille similaire : 28 × 16 cm". The houses here
are 28 x 16 cm, their long side facing the centre as in the map. The alley
positions, the alley width, the agora's wall and its exits are not published;
the defaults are estimates measured from an overhead photograph of the
apparatus scaled to the 1.6 m diameter.

The alleys and the agora are walled (``arena.outline == "wall"``). The
sources do not give the wall material; the walls are modeled as opaque.

References
----------
Couly, S. (2018). Analyse d'un peptide P42 protecteur de la maladie de
Huntington. PhD thesis, Université de Montpellier.
https://theses.hal.science/tel-01925318

Crouzier, L., Gilabert, D., Rossel, M., Trousse, F., & Maurice, T. (2018).
Topographical memory analyzed in mice using the Hamlet test, a novel complex
maze. Neurobiology of Learning and Memory, 149, 118-134.
https://doi.org/10.1016/j.nlm.2018.02.014

Crouzier, L., Couly, S., Roques, C., Peter, C., Belkhiter, R., Arguel Jacquemin,
M., Bonetto, A., Delprat, B., & Maurice, T. (2020). Sigma-1 (σ1) receptor
activity is necessary for physiological brain plasticity in mice. European
Neuropsychopharmacology, 39, 29.
https://doi.org/10.1016/j.euroneuro.2020.08.010

Hua, J., Garcia de Paco, E., Linck, N., Maurice, T., Desrumaux, C.,
Manoury, B., Rassendren, F., & Ulmann, L. (2023). Microglial P2X4 receptors
promote ApoE degradation and contribute to memory deficits in Alzheimer's
disease. Cellular and Molecular Life Sciences, 80(5), 138.
https://doi.org/10.1007/s00018-023-04784-x

Wijnen, K., Genzel, L., & van der Meij, J. (2024). Rodent maze studies: from
following simple rules to complex map learning. Brain Structure and Function,
229(4), 823–841. https://doi.org/10.1007/s00429-024-02771-x

Examples
--------
>>> from neurospatial.simulation.mazes.hamlet import make_hamlet_maze, HamletDims
>>> maze = make_hamlet_maze()
>>> maze.env_2d.units
'cm'
>>> [name for name in maze.env_2d.regions if name.endswith("_house")]
['hide_house', 'run_house', 'interact_house', 'eat_house', 'drink_house']
>>> maze.env_2d.landmarks["drink_house"].metadata["contents"]
'two water dispensers'
>>> maze.env_2d.landmarks["agora"].metadata["role"]
'start_box'
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from neurospatial.environment import Environment
from neurospatial.landmarks import Landmark, Landmarks
from neurospatial.regions import Regions
from neurospatial.simulation.mazes._base import (
    MazeDims,
    MazeEnvironments,
    check_count,
    check_lengths,
    refuse,
    warn_if_bins_too_coarse,
)
from neurospatial.simulation.mazes._geometry import (
    make_corridor_polygon,
    single_floor,
    union_polygons,
)

# The five houses counterclockwise from the top (Crouzier et al. 2020,
# Fig. 1b), each with what it holds (Crouzier et al. 2020, Methods).
_HOUSES: tuple[tuple[str, str], ...] = (
    ("hide", "a novomaze (Viewpoint)"),
    ("run", "a running wheel"),
    ("interact", "a stranger mouse behind a wire mesh screen"),
    ("eat", "a food dispenser"),
    ("drink", "two water dispensers"),
)


@dataclass(frozen=True)
class HamletDims(MazeDims):
    """Dimension specifications for Hamlet Maze.

    All radii are measured from the maze center to an alley centerline
    point; each is an estimate from an overhead photograph of the apparatus
    scaled to its published 1.6 m diameter.

    Attributes
    ----------
    ring_radius : float
        Distance from the center to each corner of the central pentagonal
        ring, where a radial alley leaves the ring, in cm. Default is 22.0.
    split_radius : float
        Distance from the center to the end of each radial alley, where it
        splits into two arms, in cm. Default is 43.0.
    house_radius : float
        Distance from the center to the center of each house, in cm.
        Default is 70.0.
    corridor_width : float
        Width of every alley in cm. Default is 8.0, the "30 x 8 cm" streets
        of Couly (2018).
    house_width : float
        Side of each house that faces the maze center, in cm. Default is
        28.0 (Couly 2018).
    house_depth : float
        Side of each house along the line to the maze center, in cm. Default
        is 16.0 (Couly 2018).
    agora_wall_thickness : float
        Wall between the agora and the ring of alleys around it, in cm.
        Default is 3.0, an estimate: the map draws a thin wall, but bins
        whose diagonal reaches across the slanted wall join the agora to
        the ring.
    n_arms : int
        Number of houses and of radial alleys (default: 5).

    Examples
    --------
    >>> dims = HamletDims()
    >>> dims.ring_radius, dims.split_radius, dims.house_radius
    (22.0, 43.0, 70.0)
    >>> dims.house_width, dims.house_depth
    (28.0, 16.0)
    >>> dims.n_arms
    5

    >>> custom = HamletDims(house_radius=80.0, corridor_width=10.0)
    >>> custom.house_radius
    80.0
    """

    ring_radius: float = 22.0
    split_radius: float = 43.0
    house_radius: float = 70.0
    corridor_width: float = 8.0
    house_width: float = 28.0
    house_depth: float = 16.0
    agora_wall_thickness: float = 3.0
    n_arms: int = 5

    def __post_init__(self) -> None:
        check_lengths(
            self,
            "ring_radius",
            "split_radius",
            "house_radius",
            "corridor_width",
            "house_width",
            "house_depth",
            "agora_wall_thickness",
        )
        check_count(self, "n_arms", 3, "the central ring needs at least three sides")
        if not self.ring_radius < self.split_radius < self.house_radius:
            refuse(
                self,
                ("ring_radius", "split_radius", "house_radius"),
                "does not place the ring, the splits and the houses in order "
                "out from the centre",
                "Use ring_radius < split_radius < house_radius.",
            )
        # The radial alley runs from the ring to the split.
        if self.split_radius - self.ring_radius <= self.corridor_width:
            refuse(
                self,
                ("ring_radius", "split_radius", "corridor_width"),
                "makes each radial alley no longer than it is wide, so its "
                "split overlaps the ring",
                "Use split_radius - ring_radius greater than corridor_width.",
            )
        _agora_polygon(self, _node_positions(self)[0])


def make_hamlet_maze(
    dims: HamletDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Hamlet Maze environment.

    Creates the agora, the ring of alleys around it, five radial alleys that
    each split into two arms, and a house where the arms of neighbouring
    alleys meet at each point of the star.

    Parameters
    ----------
    dims : HamletDims, optional
        Maze dimensions. If None, uses default dimensions (estimated from
        the apparatus: ring corners at 22 cm, splits at 43 cm, houses at
        70 cm; 8 cm alleys and 28 x 16 cm houses, as published).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0). Keep its diagonal below the
        agora's wall (bins under 2.1 cm at the default dimensions); larger
        bins can bridge the wall and trigger a warning.
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
    The maze is centered at the origin. House ``k`` lies at
    ``90 + 360 k / n_arms`` degrees; radial alley ``k`` lies halfway between
    houses ``k`` and ``k + 1``, leaving the ring at ring corner ``k`` and
    splitting at split point ``k`` into one arm to each of those houses.

    The floor is the agora, the alleys, each ``corridor_width`` wide, and the
    houses. The agora fills the ring, within a wall ``agora_wall_thickness``
    thick, and opens onto the ring through two exits one alley wide: one
    through the side facing house 0 (the Hide house, at the top) and one
    through the opposite corner, at ring corner ``(n_arms - 1) // 2``. The
    region between each house and the ring is a solid block, not floor.

    Landmarks, each with a polygon region of the same name and its extent:

    - agora: the start box (role ``"start_box"``), at the maze center
    - hide_house, run_house, interact_house, eat_house, drink_house: the
      houses, counterclockwise from the top (role ``"goal_box"``), each at
      its center, with ``metadata["contents"]`` saying what it holds. With
      ``n_arms`` other than 5, no source describes the houses, and they are
      named house_0 ... house_{n_arms - 1} with no contents.

    Other regions:

    - ring_0 to ring_4: Points where each radial alley leaves the ring
    - split_0 to split_4: Points where each radial alley splits into two
      arms

    Track Graph Topology:
    - 17 nodes: the agora, 5 ring corners, the ring point at the agora's
      top exit (agora_north_exit), 5 split points and 5 houses
    - 23 edges: 6 ring alleys, 2 agora exits, 5 radial alleys and 10 arms

    Examples
    --------
    Create a default Hamlet maze:

    >>> maze = make_hamlet_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom dimensions:

    >>> dims = HamletDims(split_radius=50.0, house_radius=80.0)
    >>> maze = make_hamlet_maze(dims=dims, bin_size=2.0)
    >>> "split_4" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_hamlet_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    HamletDims : Dimension specifications for Hamlet Maze.
    """
    if dims is None:
        dims = HamletDims()

    ring, split, house = _node_positions(dims)
    half_width = dims.corridor_width / 2
    agora = _agora_polygon(dims, ring)
    if bin_size * np.sqrt(2) >= dims.agora_wall_thickness:
        limit = dims.agora_wall_thickness / np.sqrt(2)
        warnings.warn(
            f"bin_size={bin_size} cm puts diagonal neighbours "
            f"{bin_size * np.sqrt(2):.1f} cm apart, across the "
            f"{dims.agora_wall_thickness:.1f} cm slanted wall of the Hamlet maze's "
            "agora, so paths can pass through it. Use bin_size below "
            f"{limit:.1f} cm to keep the wall.",
            UserWarning,
            stacklevel=2,
        )
    houses = _house_boxes(dims, house)
    names = _house_names(dims)

    # The alleys are the band within half a corridor width of the star's
    # edges; the blocks are each enclosed face of the alley network shrunk
    # by half a corridor width, so every alley keeps its full width and the
    # junctions keep sharp corners. The agora, its exits and the houses are
    # floor on top of that.
    star = Polygon(np.stack([house, split], axis=1).reshape(-1, 2))
    blocks = [Polygon(ring).buffer(-half_width, join_style="mitre")]
    for k in range(dims.n_arms):
        face = Polygon([ring[k - 1], split[k - 1], house[k], split[k], ring[k]])
        blocks.append(face.buffer(-half_width, join_style="mitre"))
    alleys = star.buffer(half_width, join_style="mitre").difference(
        union_polygons(blocks)
    )
    exits = [
        make_corridor_polygon((0.0, 0.0), tuple(end), dims.corridor_width)
        for end in _agora_exits(dims, ring)
    ]
    floor = single_floor(union_polygons([alleys, agora, *exits, *houses]), dims)

    env_2d = Environment.from_polygon(
        polygon=floor,
        bin_size=bin_size,
        name="hamlet_maze",
        connect_diagonal_neighbors=True,
        units="cm",
    )

    region_builder = Regions()
    region_builder.add("agora", polygon=agora)
    landmarks = [Landmark("agora", "point", (0.0, 0.0), {"role": "start_box"})]
    for k, (name, box) in enumerate(zip(names, houses, strict=True)):
        region_builder.add(name, polygon=box)
        metadata = {"role": "goal_box"}
        if dims.n_arms == len(_HOUSES):
            metadata["contents"] = _HOUSES[k][1]
        landmarks.append(Landmark(name, "point", house[k], metadata))
    for k, pos in enumerate(ring):
        region_builder.add(f"ring_{k}", point=pos)
    for k, pos in enumerate(split):
        region_builder.add(f"split_{k}", point=pos)
    env_2d = env_2d.with_regions(region_builder).with_landmarks(Landmarks(landmarks))

    env_track = None
    if include_track:
        env_track = _create_hamlet_track_graph(dims, bin_size)

    warn_if_bins_too_coarse(
        env_2d,
        maze="Hamlet maze",
        bin_size=bin_size,
        narrowest_corridor=dims.corridor_width,
    )
    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _house_names(dims: HamletDims) -> list[str]:
    """The published houses' names, or numbered houses for another count."""
    if dims.n_arms == len(_HOUSES):
        return [f"{function}_house" for function, _ in _HOUSES]
    return [f"house_{k}" for k in range(dims.n_arms)]


def _node_positions(
    dims: HamletDims,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Ring corners, split points and house centers, each ``(n_arms, 2)`` cm."""
    house_angles = np.pi / 2 + 2 * np.pi * np.arange(dims.n_arms) / dims.n_arms
    alley_angles = house_angles + np.pi / dims.n_arms

    def polar(radius: float, angles: NDArray[np.float64]) -> NDArray[np.float64]:
        return radius * np.column_stack([np.cos(angles), np.sin(angles)])

    return (
        polar(dims.ring_radius, alley_angles),
        polar(dims.split_radius, alley_angles),
        polar(dims.house_radius, house_angles),
    )


def _agora_polygon(dims: HamletDims, ring: NDArray[np.float64]) -> Polygon:
    """The agora: the ring's inside, within its wall."""
    inset = dims.corridor_width / 2 + dims.agora_wall_thickness
    agora = Polygon(ring).buffer(-inset, join_style="mitre")
    if agora.is_empty:
        refuse(
            dims,
            ("ring_radius", "corridor_width", "agora_wall_thickness"),
            "leaves no walled agora inside the ring of alleys",
            "Use a thinner agora_wall_thickness, narrower alleys or a larger "
            "ring_radius.",
        )
    return agora


def _agora_exits(
    dims: HamletDims, ring: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Where the agora's two exits meet the ring's centerline.

    The top exit leaves through the middle of the side facing house 0; the
    other through the opposite corner, at ring corner ``(n_arms - 1) // 2``.
    """
    return (ring[-1] + ring[0]) / 2, ring[(dims.n_arms - 1) // 2]


def _house_boxes(dims: HamletDims, house: NDArray[np.float64]) -> list[Polygon]:
    """Rectangular houses centered on ``house``, long side facing the center."""
    depth, width = dims.house_depth / 2, dims.house_width / 2
    boxes = []
    for x, y in house:
        rectangle = shapely_box(-depth, -width, depth, width)
        rectangle = rotate(rectangle, np.arctan2(y, x), origin=(0, 0), use_radians=True)
        boxes.append(translate(rectangle, x, y))
    return boxes


def _create_hamlet_track_graph(
    dims: HamletDims,
    bin_size: float,
) -> Environment:
    """Create the 1D linearized track graph for Hamlet Maze.

    Nodes sit at the agora's center, the ring corners, the ring point at the
    agora's top exit, the split points and the house centers; edges follow
    the agora's exits, the ring alleys, the radial alleys and the arms.

    Parameters
    ----------
    dims : HamletDims
        Maze dimensions.
    bin_size : float
        Spatial bin size in cm.

    Returns
    -------
    Environment
        1D linearized environment representing the Hamlet maze track.
    """
    n = dims.n_arms
    ring, split, house = _node_positions(dims)
    names = _house_names(dims)
    north_exit, _ = _agora_exits(dims, ring)

    graph = nx.Graph()
    graph.add_node("agora", pos=(0.0, 0.0))
    graph.add_node("agora_north_exit", pos=tuple(north_exit))
    for k in range(n):
        graph.add_node(f"ring_{k}", pos=tuple(ring[k]))
        graph.add_node(f"split_{k}", pos=tuple(split[k]))
        graph.add_node(names[k], pos=tuple(house[k]))

    # The ring alley from the last corner back to corner 0 passes the
    # agora's top exit.
    edge_order = [
        ("agora", "agora_north_exit"),
        ("agora_north_exit", "ring_0"),
        *((f"ring_{k}", f"ring_{k + 1}") for k in range(n - 1)),
        (f"ring_{n - 1}", "agora_north_exit"),
        ("agora", f"ring_{(n - 1) // 2}"),
    ]
    for k in range(n):
        edge_order += [
            (f"ring_{k}", f"split_{k}"),
            (f"split_{k}", names[k]),
            (f"split_{k}", names[(k + 1) % n]),
        ]
    for u, v in edge_order:
        distance = np.linalg.norm(
            np.subtract(graph.nodes[u]["pos"], graph.nodes[v]["pos"])
        )
        graph.add_edge(u, v, distance=float(distance))

    return Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        track_width=dims.corridor_width,
        name="hamlet_maze_1d",
        units="cm",
    )
