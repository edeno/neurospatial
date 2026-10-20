# Simulated mazes

`neurospatial.simulation.mazes` builds the mazes of rodent spatial-navigation
research, so you can test an analysis on a known layout before running it on a
recording. Each factory returns a `MazeEnvironments` with a 2-D environment
(`env_2d`) and, for corridor mazes, a linearized track (`env_track`).

The layouts follow Figure 1 of Wijnen et al. 2024, *Rodent maze studies: from
following simple rules to complex map learning*, Brain Struct Funct 229:823-841
([doi:10.1007/s00429-024-02771-x](https://doi.org/10.1007/s00429-024-02771-x)).
Sizes, reward ports and boxes come from the paper that describes each
apparatus, cited in each factory's docstring. A size no paper gives is marked
there as an estimate.

![Every simulated maze at true scale: floor in grey, linearized track in blue,
region outlines in black and landmarks marked by role. Each panel has a 1 m
scale bar.](images/simulated-mazes.png)

The picture is drawn from the package at each docs build. Letters refer to the
panels of Wijnen et al. 2024, Fig. 1.

## Mazes and their sources

| Maze | Source | Default size | Landmarks |
| --- | --- | --- | --- |
| `make_t_maze` | Deacon & Rawlins 2006 | 100 cm stem, 2 x 50 cm arms, 10 cm alleys | `well_left`, `well_right` |
| `make_y_maze` | Wijnen et al. 2024 | three 50 cm arms at 120 degrees | none: spontaneous alternation is unrewarded |
| `make_w_maze` | Karlsson & Frank 2009 | 76 x 76 cm, 7 cm track | `well_1` to `well_3` |
| `make_small_hex_maze` | Krausz et al. 2023 | 49 hexes, 1.30 m sides | `port_A`, `port_B`, `port_C` |
| `make_repeated_y_maze` | Warden 1929 | 4 Y junctions, 10.16 cm alleys | `entrance_box`, `food_box` |
| `make_repeated_t_maze` | Tolman & Nyswander 1927 | six T units, 4 in alleys | `start_box`, `food_box` |
| `make_hampton_court_maze` | Small 1901 | 8 x 6 ft, 4 in galleries | `central_enclosure` |
| `make_radial_arm_maze` | Zoladek & Roberts 1978 | 35.5 cm centre, eight 76 x 9 cm arms | `food_cup_0` to `food_cup_7` |
| `make_barnes_maze` | Barnes 1979 | 122 cm, 18 holes of 9.5 cm | `escape_hole`, `hole_*`, `start_box` |
| `make_cheeseboard_maze` | Mátyás et al. 2025; Roux et al. 2017 | 120 cm, 177 wells 7.5 cm apart | `well_*` |
| `make_watermaze` | Morris 1984; Morris 2008 | 150 cm pool, 15 cm platform | `platform` |
| `make_crossword_maze` | McNamara et al. 2014 | 95 cm square, 8 tracks of 5 cm | `start_box_*`, `cap_*` |
| `make_honeycomb_maze` | Wood et al. 2018 | 37 platforms of 11.5 cm sides, 145 cm across | none: the goal changes each session |
| `make_hamlet_maze` | Crouzier et al. 2018 | 1.6 m, 8 cm streets, 28 x 16 cm houses | `agora` and five houses |
| `make_rat_hexmaze` | Alonso et al. 2021 | four 24-node modules, about 9 x 5 m | none: the goal changes per animal |
| `make_linear_track` | Kim & Frank 2009 | 150 cm | `well_left`, `well_right` |
| `make_plus_maze` | Packard & McGaugh 1996 | four 45 cm arms | none: start and goal arms are protocol choices |
| `make_square_maze` | | 30 cm rest box | none |
| `make_sungod_maze` | Gillespie et al. 2021 | about 1 x 1 m | `home_port`, `center_port_*`, `arm_port_1` to `arm_port_8` |

## Use the apparatus by name

A reward port, hole, hidden platform or start or goal box is a landmark whose
`metadata["role"]` is one of `MAZE_LANDMARK_ROLES`. Encoding functions that take
`object_positions` accept its name. A region of the same name gives its extent
for visits and occupancy: the real size where the source publishes one,
otherwise a point. The linearized track carries the same landmarks.

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
layout that the apparatus cannot have raises an error that says why:

<!-- docs-test: run -->
```python
from neurospatial.simulation.mazes import SmallHexDims, make_small_hex_maze

# The empty hex maze, with only its permanent walls.
open_maze = make_small_hex_maze(SmallHexDims(barriers=frozenset()))
assert open_maze.env_2d.n_bins > make_small_hex_maze().env_2d.n_bins
```
