"""Small Hex Maze environment for spatial navigation research.

The Small Hex Maze (Frank Lab Hex Maze) consists of a triangular arena with
hexagonal barriers creating a lattice navigation structure. Animals navigate
between hex cell centers, with barriers blocking direct paths and forcing
route planning through the maze.

This implementation is based on CVAT annotations of the actual Frank Lab
hex maze, providing accurate geometry for simulation and analysis.

The maze has:
- Triangular arena boundary (~100 cm height)
- 37 hexagonal barriers creating the maze structure
- 34 navigable hex cell centers
- 3 reward wells at the triangle corners

Reference: Frank Lab hex maze design

Examples
--------
>>> from neurospatial.simulation.mazes.hex_small import (
...     make_small_hex_maze,
...     SmallHexDims,
... )
>>> maze = make_small_hex_maze()
>>> maze.env_2d.units
'cm'
>>> maze.env_2d.n_bins > 0
True
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from neurospatial import Environment
from neurospatial.simulation.mazes._base import MazeDims, MazeEnvironments

# Calibration: pixels to cm conversion for Frank Lab hex maze
_CM_PER_PIXEL = 1 / 3.14

# ============================================================================
# Geometry extracted from CVAT annotation (hex_maze_annotation.xml)
# All coordinates in pixels - will be converted to cm
# ============================================================================

# Arena boundary (triangular)
_ARENA_POINTS_PX = [
    (150.10, 304.50),
    (168.47, 324.55),
    (476.52, 326.78),
    (490.42, 302.31),
    (338.60, 33.70),
    (309.71, 33.19),
]

# Reward well polygons (3 corners)
_REWARD_WELLS_PX = [
    # reward_well_0 (bottom right)
    [
        (482.0, 298.0),
        (475.0, 299.0),
        (466.0, 311.0),
        (466.0, 316.0),
        (475.0, 323.0),
        (481.0, 322.0),
        (488.0, 312.0),
        (488.0, 304.0),
    ],
    # reward_well_1 (top)
    [
        (316.0, 29.0),
        (313.0, 32.0),
        (313.0, 41.0),
        (315.0, 45.0),
        (330.0, 46.0),
        (334.0, 39.0),
        (334.0, 33.0),
        (330.0, 29.0),
    ],
    # reward_well_2 (bottom left)
    [
        (161.0, 295.0),
        (151.0, 303.0),
        (151.0, 306.0),
        (159.0, 318.0),
        (171.0, 320.0),
        (176.0, 315.0),
        (176.0, 301.0),
        (169.0, 295.0),
    ],
]

# Barrier polygons (37 hexagonal obstacles)
_BARRIERS_PX = [
    [
        (318.0, 203.0),
        (313.0, 205.0),
        (308.0, 211.0),
        (309.0, 226.0),
        (314.0, 231.0),
        (325.0, 232.0),
        (333.0, 225.0),
        (335.0, 211.0),
        (331.0, 206.0),
        (327.0, 204.0),
    ],
    [
        (287.0, 125.0),
        (284.0, 128.0),
        (281.0, 135.0),
        (282.0, 143.0),
        (286.0, 149.0),
        (294.0, 155.0),
        (301.0, 155.0),
        (310.0, 151.0),
        (313.0, 146.0),
        (313.0, 137.0),
        (308.0, 128.0),
        (300.0, 123.0),
        (293.0, 123.0),
    ],
    [
        (264.0, 80.0),
        (260.0, 88.0),
        (261.0, 97.0),
        (264.0, 103.0),
        (268.0, 107.0),
        (273.0, 107.0),
        (276.0, 109.0),
        (286.0, 108.0),
        (290.0, 104.0),
        (291.0, 99.0),
        (287.0, 84.0),
        (279.0, 78.0),
        (268.0, 78.0),
    ],
    [
        (346.0, 126.0),
        (341.0, 128.0),
        (336.0, 134.0),
        (335.0, 144.0),
        (338.0, 150.0),
        (343.0, 154.0),
        (352.0, 155.0),
        (361.0, 149.0),
        (364.0, 142.0),
        (363.0, 136.0),
        (357.0, 129.0),
    ],
    [
        (205.0, 250.0),
        (202.0, 250.0),
        (192.0, 255.0),
        (179.0, 254.0),
        (173.0, 257.0),
        (167.0, 268.0),
        (168.0, 279.0),
        (175.0, 283.0),
        (186.0, 283.0),
        (197.0, 274.0),
        (206.0, 269.0),
        (208.0, 266.0),
        (208.0, 255.0),
    ],
    [
        (234.0, 179.0),
        (229.0, 176.0),
        (219.0, 176.0),
        (206.0, 180.0),
        (203.0, 183.0),
        (201.0, 191.0),
        (204.0, 200.0),
        (211.0, 205.0),
        (219.0, 205.0),
        (224.0, 203.0),
        (237.0, 191.0),
        (237.0, 183.0),
    ],
    [
        (302.0, 216.0),
        (296.0, 216.0),
        (286.0, 221.0),
        (281.0, 228.0),
        (281.0, 241.0),
        (284.0, 245.0),
        (289.0, 248.0),
        (301.0, 247.0),
        (306.0, 243.0),
        (310.0, 235.0),
        (310.0, 231.0),
        (306.0, 220.0),
    ],
    [
        (256.0, 177.0),
        (254.0, 181.0),
        (254.0, 191.0),
        (260.0, 198.0),
        (276.0, 198.0),
        (284.0, 194.0),
        (287.0, 188.0),
        (287.0, 182.0),
        (282.0, 173.0),
        (265.0, 171.0),
        (260.0, 173.0),
    ],
    [
        (319.0, 171.0),
        (315.0, 173.0),
        (310.0, 180.0),
        (311.0, 189.0),
        (317.0, 198.0),
        (327.0, 200.0),
        (332.0, 198.0),
        (337.0, 193.0),
        (339.0, 183.0),
        (334.0, 174.0),
        (327.0, 171.0),
    ],
    [
        (281.0, 260.0),
        (268.0, 259.0),
        (265.0, 260.0),
        (259.0, 266.0),
        (251.0, 281.0),
        (252.0, 291.0),
        (257.0, 297.0),
        (267.0, 299.0),
        (273.0, 297.0),
        (278.0, 292.0),
        (286.0, 276.0),
        (286.0, 268.0),
    ],
    [
        (287.0, 306.0),
        (282.0, 311.0),
        (278.0, 320.0),
        (276.0, 338.0),
        (278.0, 342.0),
        (284.0, 347.0),
        (307.0, 347.0),
        (309.0, 345.0),
        (309.0, 327.0),
        (305.0, 311.0),
        (298.0, 305.0),
    ],
    [
        (340.0, 216.0),
        (337.0, 219.0),
        (335.0, 225.0),
        (335.0, 241.0),
        (339.0, 247.0),
        (345.0, 250.0),
        (354.0, 249.0),
        (360.0, 244.0),
        (362.0, 239.0),
        (362.0, 230.0),
        (360.0, 222.0),
        (352.0, 216.0),
    ],
    [
        (398.0, 193.0),
        (392.0, 196.0),
        (389.0, 200.0),
        (388.0, 207.0),
        (394.0, 220.0),
        (404.0, 222.0),
        (414.0, 215.0),
        (415.0, 202.0),
        (406.0, 193.0),
    ],
    [
        (412.0, 132.0),
        (404.0, 129.0),
        (392.0, 132.0),
        (383.0, 144.0),
        (385.0, 155.0),
        (390.0, 159.0),
        (406.0, 158.0),
        (415.0, 151.0),
        (417.0, 145.0),
        (416.0, 136.0),
    ],
    [
        (319.0, 77.0),
        (315.0, 79.0),
        (310.0, 86.0),
        (310.0, 102.0),
        (316.0, 110.0),
        (320.0, 112.0),
        (328.0, 112.0),
        (332.0, 109.0),
        (337.0, 100.0),
        (338.0, 87.0),
        (333.0, 79.0),
        (328.0, 77.0),
    ],
    [
        (331.0, 310.0),
        (331.0, 329.0),
        (335.0, 340.0),
        (344.0, 346.0),
        (350.0, 346.0),
        (354.0, 344.0),
        (357.0, 338.0),
        (356.0, 322.0),
        (350.0, 317.0),
        (340.0, 316.0),
        (334.0, 309.0),
    ],
    [
        (347.0, 94.0),
        (342.0, 97.0),
        (337.0, 104.0),
        (337.0, 114.0),
        (343.0, 124.0),
        (359.0, 123.0),
        (366.0, 113.0),
        (367.0, 106.0),
        (361.0, 96.0),
        (355.0, 94.0),
    ],
    [
        (393.0, 279.0),
        (393.0, 290.0),
        (391.0, 295.0),
        (391.0, 305.0),
        (393.0, 312.0),
        (406.0, 317.0),
        (414.0, 316.0),
        (418.0, 308.0),
        (417.0, 295.0),
        (415.0, 291.0),
        (408.0, 284.0),
        (407.0, 281.0),
        (402.0, 278.0),
    ],
    [
        (412.0, 270.0),
        (411.0, 280.0),
        (413.0, 285.0),
        (429.0, 299.0),
        (434.0, 301.0),
        (440.0, 301.0),
        (448.0, 298.0),
        (454.0, 293.0),
        (455.0, 288.0),
        (446.0, 274.0),
        (428.0, 265.0),
        (418.0, 265.0),
    ],
    [
        (388.0, 222.0),
        (384.0, 228.0),
        (384.0, 236.0),
        (394.0, 249.0),
        (402.0, 253.0),
        (407.0, 253.0),
        (413.0, 251.0),
        (418.0, 246.0),
        (419.0, 234.0),
        (413.0, 226.0),
        (407.0, 223.0),
        (396.0, 223.0),
        (393.0, 221.0),
    ],
    [
        (370.0, 85.0),
        (365.0, 91.0),
        (365.0, 99.0),
        (369.0, 105.0),
        (374.0, 107.0),
        (384.0, 107.0),
        (391.0, 101.0),
        (392.0, 93.0),
        (386.0, 85.0),
        (379.0, 83.0),
    ],
    [
        (207.0, 222.0),
        (191.0, 223.0),
        (181.0, 228.0),
        (177.0, 235.0),
        (176.0, 245.0),
        (182.0, 253.0),
        (198.0, 253.0),
        (208.0, 244.0),
        (212.0, 237.0),
        (212.0, 229.0),
    ],
    [
        (256.0, 220.0),
        (237.0, 221.0),
        (227.0, 226.0),
        (223.0, 234.0),
        (225.0, 244.0),
        (231.0, 249.0),
        (240.0, 250.0),
        (247.0, 247.0),
        (255.0, 241.0),
        (260.0, 235.0),
        (260.0, 225.0),
    ],
    [
        (324.0, 259.0),
        (313.0, 262.0),
        (309.0, 272.0),
        (309.0, 288.0),
        (311.0, 292.0),
        (318.0, 297.0),
        (326.0, 297.0),
        (332.0, 294.0),
        (335.0, 291.0),
        (337.0, 285.0),
        (337.0, 270.0),
        (330.0, 261.0),
    ],
    [
        (408.0, 183.0),
        (408.0, 190.0),
        (418.0, 199.0),
        (427.0, 201.0),
        (438.0, 201.0),
        (445.0, 194.0),
        (444.0, 183.0),
        (440.0, 178.0),
        (434.0, 175.0),
        (416.0, 176.0),
    ],
    [
        (345.0, 273.0),
        (339.0, 275.0),
        (338.0, 277.0),
        (338.0, 286.0),
        (334.0, 295.0),
        (333.0, 304.0),
        (338.0, 314.0),
        (342.0, 316.0),
        (353.0, 316.0),
        (358.0, 312.0),
        (360.0, 303.0),
        (358.0, 279.0),
        (356.0, 276.0),
        (352.0, 274.0),
    ],
    [
        (393.0, 317.0),
        (387.0, 323.0),
        (386.0, 332.0),
        (390.0, 342.0),
        (392.0, 344.0),
        (399.0, 345.0),
        (405.0, 343.0),
        (412.0, 336.0),
        (413.0, 324.0),
        (408.0, 319.0),
        (402.0, 317.0),
    ],
    [
        (365.0, 176.0),
        (361.0, 181.0),
        (361.0, 187.0),
        (366.0, 199.0),
        (372.0, 201.0),
        (387.0, 200.0),
        (392.0, 190.0),
        (392.0, 184.0),
        (390.0, 179.0),
        (387.0, 176.0),
        (380.0, 173.0),
        (370.0, 173.0),
    ],
    [
        (368.0, 261.0),
        (360.0, 268.0),
        (360.0, 275.0),
        (363.0, 280.0),
        (363.0, 292.0),
        (368.0, 297.0),
        (384.0, 298.0),
        (388.0, 295.0),
        (390.0, 289.0),
        (389.0, 276.0),
        (381.0, 263.0),
        (377.0, 261.0),
    ],
    [
        (438.0, 225.0),
        (433.0, 231.0),
        (433.0, 241.0),
        (443.0, 250.0),
        (450.0, 254.0),
        (463.0, 254.0),
        (469.0, 246.0),
        (469.0, 240.0),
        (464.0, 233.0),
        (450.0, 228.0),
        (447.0, 225.0),
    ],
    [
        (226.0, 133.0),
        (226.0, 147.0),
        (232.0, 153.0),
        (238.0, 155.0),
        (256.0, 150.0),
        (261.0, 142.0),
        (261.0, 135.0),
        (255.0, 127.0),
        (235.0, 126.0),
        (230.0, 128.0),
    ],
    [
        (247.0, 190.0),
        (239.0, 191.0),
        (233.0, 194.0),
        (230.0, 198.0),
        (231.0, 211.0),
        (235.0, 216.0),
        (242.0, 219.0),
        (253.0, 217.0),
        (262.0, 206.0),
        (262.0, 201.0),
        (253.0, 192.0),
    ],
    [
        (250.0, 301.0),
        (242.0, 301.0),
        (237.0, 303.0),
        (233.0, 308.0),
        (232.0, 314.0),
        (228.0, 318.0),
        (225.0, 332.0),
        (228.0, 339.0),
        (231.0, 342.0),
        (241.0, 343.0),
        (245.0, 341.0),
        (249.0, 335.0),
        (249.0, 330.0),
        (257.0, 322.0),
        (259.0, 317.0),
        (259.0, 310.0),
        (257.0, 306.0),
    ],
    [
        (372.0, 109.0),
        (368.0, 112.0),
        (365.0, 117.0),
        (364.0, 126.0),
        (366.0, 133.0),
        (370.0, 137.0),
        (383.0, 138.0),
        (388.0, 135.0),
        (393.0, 128.0),
        (392.0, 115.0),
        (390.0, 112.0),
        (384.0, 109.0),
    ],
    [
        (229.0, 261.0),
        (216.0, 260.0),
        (198.0, 276.0),
        (195.0, 283.0),
        (195.0, 287.0),
        (200.0, 295.0),
        (214.0, 299.0),
        (234.0, 279.0),
        (235.0, 269.0),
    ],
    [
        (292.0, 157.0),
        (288.0, 159.0),
        (284.0, 164.0),
        (283.0, 173.0),
        (290.0, 183.0),
        (302.0, 184.0),
        (310.0, 177.0),
        (310.0, 163.0),
        (305.0, 158.0),
    ],
    [
        (369.0, 174.0),
        (363.0, 180.0),
        (362.0, 184.0),
        (365.0, 196.0),
        (373.0, 201.0),
        (386.0, 200.0),
        (392.0, 188.0),
        (391.0, 181.0),
        (385.0, 175.0),
        (380.0, 173.0),
    ],
]

# Hex center points (34 navigable nodes)
_HEX_CENTERS_PX = [
    (221.52, 246.23),
    (225.41, 216.73),
    (251.57, 172.21),
    (274.94, 158.85),
    (274.39, 128.80),
    (300.54, 113.77),
    (300.54, 83.16),
    (323.36, 59.23),
    (350.63, 85.95),
    (325.59, 129.91),
    (324.47, 158.85),
    (347.12, 171.43),
    (373.00, 157.96),
    (397.60, 174.21),
    (345.56, 201.26),
    (296.26, 199.42),
    (272.38, 214.50),
    (248.67, 259.97),
    (321.36, 243.78),
    (298.04, 258.30),
    (298.20, 286.96),
    (347.90, 258.86),
    (372.11, 245.78),
    (370.66, 215.12),
    (399.32, 263.37),
    (425.82, 247.73),
    (423.37, 219.35),
    (447.69, 261.92),
    (424.20, 304.66),
    (274.39, 301.60),
    (222.52, 302.43),
    (245.84, 285.63),
    (185.79, 299.60),
    (461.27, 297.37),
]


@dataclass(frozen=True)
class SmallHexDims(MazeDims):
    """Dimension specifications for Small Hex Maze (Frank Lab).

    The Small Hex Maze consists of a triangular arena with hexagonal barriers
    creating a lattice navigation structure. Default dimensions are based on
    the actual Frank Lab hex maze geometry.

    Attributes
    ----------
    scale : float
        Scale factor applied to the maze. Default 1.0 uses the calibrated
        dimensions (~108 cm triangle height). Use values > 1.0 to enlarge.

    Examples
    --------
    >>> dims = SmallHexDims()
    >>> dims.scale
    1.0

    >>> custom = SmallHexDims(scale=1.5)
    >>> custom.scale
    1.5
    """

    scale: float = 1.0


def make_small_hex_maze(
    dims: SmallHexDims | None = None,
    bin_size: float = 2.0,
    include_track: bool = True,
) -> MazeEnvironments:
    """Create a Small Hex Maze (Frank Lab) environment.

    Creates a triangular arena with hexagonal barriers forming a lattice
    navigation structure. The maze geometry is based on the actual Frank Lab
    hex maze, with barriers defining paths between navigable hex cell centers.

    Parameters
    ----------
    dims : SmallHexDims, optional
        Maze dimensions. If None, uses default dimensions
        (scale=1.0, ~108 cm triangle height).
    bin_size : float, optional
        Spatial bin size in cm (default: 2.0).
    include_track : bool, optional
        Whether to create linearized track graph (default: True).
        The track graph uses hex cell centers as nodes with edges
        connecting centers that are not separated by barriers.

    Returns
    -------
    MazeEnvironments
        Contains:
        - env_2d: 2D polygon-based environment (barriers subtracted)
        - env_track: 1D linearized track environment (if include_track=True)

    Notes
    -----
    The maze geometry is derived from CVAT annotations of the Frank Lab
    hex maze. The original pixel coordinates are converted to centimeters
    using the calibration factor CM_PER_PIXEL = 1/3.14.

    Maze structure:
    - Triangular arena boundary (~108 cm height at scale=1.0)
    - 37 hexagonal barriers creating the maze structure
    - 34 navigable hex cell centers
    - 3 reward wells at triangle corners

    Regions:
    - reward_well_0: Bottom right corner (goal location)
    - reward_well_1: Top corner (goal location)
    - reward_well_2: Bottom left corner (goal location)

    Track Graph Topology:
    - 34 nodes at hex cell centers
    - Edges connect adjacent centers not separated by barriers
    - Barrier-aware connectivity computed via line-of-sight

    Examples
    --------
    Create a default hex maze:

    >>> maze = make_small_hex_maze()
    >>> maze.env_2d.units
    'cm'
    >>> maze.env_2d.n_bins > 0
    True

    Create with custom scale:

    >>> dims = SmallHexDims(scale=1.5)
    >>> maze = make_small_hex_maze(dims=dims, bin_size=3.0)
    >>> "reward_well_0" in maze.env_2d.regions
    True

    Skip track graph creation:

    >>> maze = make_small_hex_maze(include_track=False)
    >>> maze.env_track is None
    True

    See Also
    --------
    SmallHexDims : Dimension specifications for Small Hex Maze.
    """
    if dims is None:
        dims = SmallHexDims()

    scale = dims.scale * _CM_PER_PIXEL

    # Convert arena polygon to cm and apply scale
    arena_polygon = Polygon([(x * scale, y * scale) for x, y in _ARENA_POINTS_PX])

    # Convert barrier polygons to cm
    barrier_polygons = [
        Polygon([(x * scale, y * scale) for x, y in barrier])
        for barrier in _BARRIERS_PX
    ]

    # Subtract barriers from arena to get navigable polygon
    # Union all barriers first, then subtract from arena
    all_barriers = unary_union(barrier_polygons)
    navigable_geometry = arena_polygon.difference(all_barriers)

    # Handle MultiPolygon result (barriers may split arena into corridors)
    # Take the largest polygon if result is MultiPolygon
    if isinstance(navigable_geometry, MultiPolygon):
        # Get the largest polygon by area
        navigable_polygon = max(navigable_geometry.geoms, key=lambda p: p.area)
    elif isinstance(navigable_geometry, Polygon):
        navigable_polygon = navigable_geometry
    else:
        # GeometryCollection or other - extract polygons
        polygons = [g for g in navigable_geometry.geoms if isinstance(g, Polygon)]
        if polygons:
            navigable_polygon = max(polygons, key=lambda p: p.area)
        else:
            navigable_polygon = arena_polygon  # Fallback to arena

    # Convert reward wells to cm
    reward_well_polygons = [
        Polygon([(x * scale, y * scale) for x, y in well]) for well in _REWARD_WELLS_PX
    ]

    # Convert hex centers to cm
    hex_centers = [(x * scale, y * scale) for x, y in _HEX_CENTERS_PX]

    # Center the maze at origin
    centroid = arena_polygon.centroid
    cx, cy = centroid.x, centroid.y

    # Translate all geometries to center at origin
    from shapely.affinity import translate

    navigable_polygon = translate(navigable_polygon, -cx, -cy)
    arena_polygon = translate(arena_polygon, -cx, -cy)
    reward_well_polygons = [translate(p, -cx, -cy) for p in reward_well_polygons]
    barrier_polygons = [translate(p, -cx, -cy) for p in barrier_polygons]
    hex_centers = [(x - cx, y - cy) for x, y in hex_centers]

    # Create 2D environment from navigable polygon
    env_2d = Environment.from_polygon(
        polygon=navigable_polygon,
        bin_size=bin_size,
        name="small_hex_maze",
        connect_diagonal_neighbors=True,
    )
    env_2d.units = "cm"

    # Add reward well regions
    reward_well_names = ["reward_well_0", "reward_well_1", "reward_well_2"]
    for name, poly in zip(reward_well_names, reward_well_polygons, strict=True):
        # Use centroid as point region (polygon may be outside navigable area)
        env_2d.regions.add(name, point=(poly.centroid.x, poly.centroid.y))

    # Create track graph if requested
    env_track = None
    if include_track:
        env_track = _create_hex_track_graph(
            hex_centers, barrier_polygons, bin_size, dims
        )

    return MazeEnvironments(env_2d=env_2d, env_track=env_track)


def _create_hex_track_graph(
    hex_centers: list[tuple[float, float]],
    barrier_polygons: list[Polygon],
    bin_size: float,
    dims: SmallHexDims,
) -> Environment:
    """Create the 1D linearized track graph for Small Hex Maze.

    The track graph uses hex cell centers as nodes. Edges connect centers
    that have line-of-sight (not blocked by barriers).

    Parameters
    ----------
    hex_centers : list[tuple[float, float]]
        List of hex center positions (x, y) in cm.
    barrier_polygons : list[Polygon]
        List of barrier polygons (already centered).
    bin_size : float
        Spatial bin size in cm.
    dims : SmallHexDims
        Maze dimensions.

    Returns
    -------
    Environment
        1D linearized environment representing the hex maze track.
    """
    graph = nx.Graph()

    # Add nodes for each hex center
    for i, pos in enumerate(hex_centers):
        graph.add_node(f"hex_{i}", pos=pos)

    # Compute adjacencies based on distance and barrier intersection
    # Two hex centers are connected if:
    # 1. They are within a threshold distance (nearby in the lattice)
    # 2. The line between them does not intersect any barrier

    # Estimate threshold distance from typical hex spacing
    # Use median nearest-neighbor distance * 1.5
    positions = np.array(hex_centers)
    dist_list = []
    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            if i < j:
                d = np.linalg.norm(pos_j - pos_i)
                dist_list.append(d)

    distances_arr = np.array(dist_list)
    # Threshold: allow connections up to ~1.5x the typical spacing
    threshold = np.percentile(distances_arr, 30) * 1.6

    # Check each potential edge
    adjacencies = []
    for i in range(len(hex_centers)):
        pos_i = Point(hex_centers[i])
        for j in range(i + 1, len(hex_centers)):
            pos_j = Point(hex_centers[j])

            # Check distance threshold
            dist = pos_i.distance(pos_j)
            if dist > threshold:
                continue

            # Check if line-of-sight is blocked by any barrier
            line = LineString([hex_centers[i], hex_centers[j]])
            blocked = False
            for barrier in barrier_polygons:
                if line.intersects(barrier):
                    blocked = True
                    break

            if not blocked:
                adjacencies.append((i, j, dist))
                graph.add_edge(f"hex_{i}", f"hex_{j}", distance=dist)

    # Edge order for linearization
    edge_order = [(f"hex_{i}", f"hex_{j}") for i, j, _ in adjacencies]

    # Create the 1D environment
    env_track = Environment.from_graph(
        graph=graph,
        edge_order=edge_order,
        edge_spacing=0.0,
        bin_size=bin_size,
        name="small_hex_maze_1d",
    )
    env_track.units = "cm"

    return env_track
