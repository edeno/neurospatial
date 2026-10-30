# Simulated mazes

`neurospatial.simulation.mazes` builds the mazes of rodent spatial-navigation
research, so you can test an analysis on a known layout before running it on a
recording. Each factory returns a `MazeEnvironments` with a 2-D environment
(`env_2d`) and, for corridor mazes, a linearized track (`env_track`).

The layouts follow Figure 1 of Wijnen et al. 2024, *Rodent maze studies: from
following simple rules to complex map learning*, Brain Struct Funct 229(4), 823–841
([doi:10.1007/s00429-024-02771-x](https://doi.org/10.1007/s00429-024-02771-x)).
Sizes, reward ports and boxes come from the paper that describes each
apparatus, cited in each factory's docstring. A size no paper gives is an
estimate, marked *estimate* below and in the docstring; for those mazes the
table cites the paper whose protocol the maze follows, not its sizes.

![Every simulated maze at true scale: floor in grey, linearized track in blue,
region outlines in black and landmarks marked by role. Each panel has a 1 m
scale bar.](images/simulated-mazes.png)

The picture is drawn from the package at each docs build. Letters refer to the
panels of Wijnen et al. 2024, Fig. 1.

## Mazes and their sources

| Maze | Source | Default size | Landmarks |
| --- | --- | --- | --- |
| `make_t_maze` | protocol: Deacon & Rawlins 2006 | *estimate* from Wijnen et al. 2024 Fig. 1a: 100 cm stem, 2 x 50 cm arms, 10 cm alleys | `well_left`, `well_right` |
| `make_y_maze` | layout: Wijnen et al. 2024 | *estimate*: three 50 cm arms at 120 degrees, 10 cm alleys | none: spontaneous alternation is unrewarded |
| `make_w_maze` | Karlsson & Frank 2009 | 76 x 76 cm, 7 cm track | `well_1` to `well_3` |
| `make_small_hex_maze` | Krausz et al. 2023 | 49 hexes, 1.30 m sides | `port_A`, `port_B`, `port_C` |
| `make_repeated_y_maze` | Warden 1931; Wijnen et al. 2024 | *estimate*: 4 Y junctions, 10.16 cm alleys | `entrance_box`, `food_box` |
| `make_repeated_t_maze` | Tolman & Nyswander 1927 | six T units, 4 in alleys | `start_box`, `food_box` |
| `make_hampton_court_maze` | Small 1901 | 8 x 6 ft, 4 in galleries | `central_enclosure` |
| `make_radial_arm_maze` | Zoladek & Roberts 1978 | 35.5 cm centre, eight 76 x 9 cm arms | `food_cup_0` to `food_cup_7` |
| `make_barnes_maze` | Barnes 1979 | 122 cm, 18 holes of 9.5 cm | `escape_hole`, `hole_*`, `start_box` |
| `make_cheeseboard_maze` | Mátyás et al. 2025; Roux et al. 2017 | 120 cm, 177 wells 7.5 cm apart | `well_*` |
| `make_watermaze` | Morris 1984; Morris 2008 | 150 cm pool, 15 cm platform | `platform` |
| `make_crossword_maze` | McNamara et al. 2014 | 95 cm square, 8 tracks of 5 cm | `start_box_*`, `cap_*` |
| `make_honeycomb_maze` | Wood et al. 2018 | 37 platforms of 11.5 cm sides, 145 cm across | none: the goal changes each session |
| `make_hamlet_maze` | sizes: Couly 2018; protocol: Crouzier et al. 2018, 2020 | 1.6 m, 8 cm streets, 28 x 16 cm houses; alley layout an *estimate* | `agora` and five houses |
| `make_rat_hexmaze` | rat build: Wijnen et al. 2024; module: Alonso et al. 2021 | *estimate*: four 24-node modules, 48 cm node spacing, about 9 x 5 m | none: the goal changes per animal |
| `make_linear_track` | Kim & Frank 2009 | 150 cm | `well_left`, `well_right` |
| `make_plus_maze` | protocol: Packard & McGaugh 1996 | *estimate*: four 45 cm arms, 10 cm alleys | none: start and goal arms are protocol choices |
| `make_square_maze` | | 30 cm rest box | none |
| `make_sungod_maze` | Gillespie et al. 2021 | about 1 x 1 m | `home_port`, `center_port_*`, `arm_port_1` to `arm_port_8` |

## Walls, drop edges and clear walls

Each maze declares what its edges are, as its source describes the apparatus
(see each factory's docstring). Edges stop the animal in every case, so border
scores, boundary-vector maps, `BorderCellModel` and `BoundaryVectorCellModel`
use them all; a gaze stops only at opaque walls.

- Opaque walls (`arena.outline == "wall"`): the Hampton Court, Hamlet, repeated
  T and Y, T and Y mazes, rat HexMaze and small hex mazes, the watermaze pool
  and the square box.
- Drop edges (`"drop"`), an elevated track or platform the animal sees past:
  the linear track, W and plus mazes, radial arm, crossword, Barnes,
  cheeseboard and honeycomb mazes. A ray-cast gaze leaving one of these views
  nothing in the environment, so its viewed location is NaN.
- The Sungod maze has opaque external walls and transparent walls between its
  arms (Gillespie et al. 2021): `arena.walls` holds the stretches of outline
  between neighbouring arms, each a `"transparent_wall"`.

If your apparatus is built differently, rebuild the floor with its own edges:

<!-- docs-test: run -->
```python
from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.simulation.mazes import make_plus_maze

maze = make_plus_maze()
assert maze.env_2d.arena.outline == "drop"  # an elevated plus maze
walled = (
    Environment.from_polygon(maze.env_2d.arena.polygon, bin_size=2.0, units="cm")
    .with_regions(maze.env_2d.regions)
    .with_landmarks(maze.env_2d.landmarks)
)
assert walled.arena.outline == "wall"
clear = Environment.from_polygon(
    maze.env_2d.arena.polygon,
    bin_size=2.0,
    units="cm",
    arena=OpenArena(maze.env_2d.arena.polygon, outline="transparent_wall"),
)
assert clear.arena.outline == "transparent_wall"
```

## Use the apparatus by name

A reward port, hole, hidden platform or start or goal box is a landmark whose
`metadata["role"]` is one of `MAZE_LANDMARK_ROLES`. Encoding functions that take
`object_positions` accept its name. A region of the same name gives its extent
for visits and occupancy: the real size where the source publishes one,
otherwise a point. A reward port's `metadata["port_type"]` names its hardware
(`"food_well"`, `"food_cup"`, `"plastic_cap"` or `"liquid_port"`). The
linearized track carries the same landmarks.

<!-- docs-test: run -->
```python
from neurospatial.simulation.mazes import make_w_maze

maze = make_w_maze()
env = maze.env_2d

ports = [
    name
    for name, landmark in env.landmarks.items()
    if landmark.metadata["role"] == "reward_port"
]
assert ports == ["well_1", "well_2", "well_3"]

# The same names are regions, for visits and occupancy...
assert len(env.bins_in_region("well_2")) > 0
# ...and landmarks on the linearized track.
assert list(maze.env_track.landmarks) == ports
```

Change a maze through its dimensions class. Sizes are in centimeters, and a
layout that the apparatus cannot have raises an error that says why. A
corridor must be longer than it is wide, for example, so an arm 3 cm long on a
10 cm corridor is refused:

<!-- docs-test: raises ValueError -->
```python
from neurospatial.simulation.mazes import TMazeDims

TMazeDims(arm_length=3.0, corridor_width=10.0)
```

Valid changes build the maze:

<!-- docs-test: run -->
```python
from neurospatial.simulation.mazes import SmallHexDims, make_small_hex_maze

# The empty hex maze, with only its permanent walls.
open_maze = make_small_hex_maze(SmallHexDims(barriers=frozenset()))
assert open_maze.env_2d.n_bins > make_small_hex_maze().env_2d.n_bins
```

## Choose a bin size

Each maze keeps the bins whose centres lie on its floor. Bins wider than about
half the narrowest corridor can leave a stretch of corridor without a bin and
split the maze into pieces, and a region smaller than a bin can hold no bin
centre. Either breaks paths, distances or visits, so the maze warns and names
a bin size that works. The default 2 cm bins are fine enough for every maze.
The rat hexmaze, about 9 × 5 m, still spans a grid of over 100,000 bins at
2 cm, so building it also warns about the grid's memory use (about 120 MB).

<!-- docs-test: run -->
```python
import warnings

from neurospatial.simulation.mazes import make_crossword_maze

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    coarse = make_crossword_maze(bin_size=5.0)  # 5 cm tracks
assert len(coarse.env_2d.components()) == 2
assert "use bin_size at most 2.5 cm" in str(caught[0].message).lower()
assert len(make_crossword_maze().env_2d.components()) == 1
```
