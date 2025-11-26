# Simulation Mazes Implementation Tasks

> **Reference**: See [PLAN_MAZE.md](PLAN_MAZE.md) for detailed context, dimensions, and API specifications.
>
> **Goal**: Implement 15 standardized maze environments for spatial navigation research based on Wijnen et al. 2024.

---

## Milestone 1: Foundation

Create base module structure and shared utilities.

### 1.1 Create Module Structure

- [x] Create `src/neurospatial/simulation/mazes/` directory
- [x] Create `src/neurospatial/simulation/mazes/__init__.py` with placeholder exports
- [x] Verify import works: `from neurospatial.simulation.mazes import MazeEnvironments`

### 1.2 Implement Base Classes

**File**: `src/neurospatial/simulation/mazes/_base.py`

- [x] Create `MazeDims` frozen dataclass (base class for dimension specs)
- [x] Create `MazeEnvironments` dataclass with `env_2d: Environment` and `env_track: Environment | None`
- [x] Add NumPy-style docstrings with examples
- [x] Verify: `uv run pytest --doctest-modules src/neurospatial/simulation/mazes/_base.py`

### 1.3 Implement Geometry Helpers

**File**: `src/neurospatial/simulation/mazes/_geometry.py`

- [x] Implement `make_corridor_polygon(start, end, width)` → Shapely Polygon
- [x] Implement `make_buffered_line(start, end, width)` → Shapely Polygon (buffer around LineString)
- [x] Implement `union_polygons(polygons)` → combined Polygon with cleanup
- [x] Implement `make_circular_arena(center, radius)` → circular Polygon
- [x] Implement `make_star_graph(center, arm_endpoints, spacing)` → nx.Graph for star topology
- [x] Add unit tests: `tests/simulation/mazes/test_geometry.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_geometry.py -v`

---

## Milestone 2: Simple Corridor Mazes (Phase 2 Partial)

Implement the simplest mazes first to establish patterns.

### 2.1 Linear Track

**File**: `src/neurospatial/simulation/mazes/linear_track.py`

- [x] Create `LinearTrackDims(length=150.0, width=10.0)` frozen dataclass
- [x] Implement `make_linear_track(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment using `Environment.from_polygon()` (single rectangle)
- [x] Create track graph with `start` and `end` nodes + intermediate nodes
- [x] Add regions: `reward_left` (0, 0), `reward_right` (length, 0)
- [x] Set `env.units = "cm"`
- [x] Add doctests demonstrating usage
- [x] Add tests: `tests/simulation/mazes/test_linear_track.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_linear_track.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- `make_linear_track()` returns valid MazeEnvironments
- `env_2d.units == "cm"`
- Track graph is connected from start to end
- Regions are queryable: `env_2d.bins_in_region("reward_left")`

### 2.2 T-Maze

**File**: `src/neurospatial/simulation/mazes/t_maze.py`

- [x] Create `TMazeDims(stem_length=100.0, arm_length=50.0, width=10.0)` frozen dataclass
- [x] Implement `make_t_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: union of 3 rectangles (stem + left arm + right arm)
- [x] Create track graph with nodes: `start`, `junction`, `left_end`, `right_end`
- [x] Add edges: start→junction, junction→left_end, junction→right_end
- [x] Add regions: `start`, `junction`, `left_end`, `right_end`
- [x] Set `env.units = "cm"` and center at (0, 0)
- [x] Add doctests
- [x] Add tests: `tests/simulation/mazes/test_t_maze.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_t_maze.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Junction is at T-intersection
- Track graph has 3 edges from junction node
- All regions queryable

### 2.3 Y-Maze

**File**: `src/neurospatial/simulation/mazes/y_maze.py`

- [x] Create `YMazeDims(arm_length=50.0, width=10.0)` frozen dataclass *(Note: arm_angle hardcoded internally)*
- [x] Implement `make_y_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: 3 buffered lines at 120° separation (90°, 210°, 330°)
- [x] Use `shapely.buffer()` on LineString for corridor polygons
- [x] Create track graph: star with 3 arms from center (Y-junction)
- [x] Add regions: `center`, `arm1_end`, `arm2_end`, `arm3_end`
- [x] Set `env.units = "cm"`
- [x] Add doctests
- [x] Add tests: `tests/simulation/mazes/test_y_maze.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_y_maze.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Arms are at correct 120° angles
- Track graph has 3-way connectivity at center
- Arm endpoints at correct positions (trigonometry verified)

---

## Milestone 3: Simple Open-Field Mazes (Phase 4 Partial)

Implement circular arenas without complex topology.

### 3.1 Morris Water Maze

**File**: `src/neurospatial/simulation/mazes/watermaze.py`

- [x] Create `WatermazeDims(pool_diameter=150.0, platform_radius=5.0)` frozen dataclass
- [x] Implement `make_watermaze(dims, platform_position, bin_size)` → MazeEnvironments
- [x] Create 2D environment: `Environment.from_polygon()` with circular arena
- [x] `env_track = None` (open field has no track topology)
- [x] Add regions: `platform` (point or small circle), `NE`, `NW`, `SE`, `SW` quadrants
- [x] Platform defaults to center of one quadrant if not specified
- [x] Set `env.units = "cm"`
- [x] Add doctests
- [x] Add tests: `tests/simulation/mazes/test_watermaze.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_watermaze.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Pool is circular with correct diameter
- Platform region is queryable
- Quadrant regions partition the pool

### 3.2 Barnes Maze

**File**: `src/neurospatial/simulation/mazes/barnes.py`

- [x] Create `BarnesDims(diameter=120.0, n_holes=18, hole_radius=2.5)` frozen dataclass *(Note: holes_on_perimeter hardcoded)*
- [x] Implement `make_barnes_maze(dims, escape_hole_index, bin_size)` → MazeEnvironments
- [x] Create 2D environment: circular arena (holes don't affect navigable space)
- [x] `env_track = None` (open field)
- [x] Add regions: `escape_hole` (goal), `hole_0` through `hole_{n-1}` evenly spaced on perimeter
- [x] Calculate hole positions using angles: `2π * i / n_holes`
- [x] Set `env.units = "cm"`
- [x] Add doctests
- [x] Add tests: `tests/simulation/mazes/test_barnes.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_barnes.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Holes are evenly distributed on perimeter
- Escape hole is one of the holes
- 18 holes by default (original Barnes 1979)

---

## Milestone 4: Remaining Corridor Mazes (Phase 2 Remaining)

### 4.1 W-Maze

**File**: `src/neurospatial/simulation/mazes/w_maze.py`

- [x] Create `WMazeDims(width=120.0, height=80.0, corridor_width=10.0, n_wells=3)` frozen dataclass
- [x] Implement `make_w_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: 3 parallel vertical corridors connected at base (horizontal corridor)
- [x] Create track graph: chain of nodes through W pattern
- [x] Add regions: `start`, `well_1`, `well_2`, `well_3`
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_w_maze.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_w_maze.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- W shape is correct (3 vertical wells + horizontal base)
- Track graph follows corridor path
- All wells are accessible

### 4.2 Small Hex Maze

**File**: `src/neurospatial/simulation/mazes/hex_small.py`

- [x] Create `SmallHexDims(hex_spacing=14.0, corridor_width=10.0)` frozen dataclass
- [x] Implement `make_small_hex_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: `HexagonalLayout` with triangular cluster mask (~7-10 hexes)
- [x] Create track graph: hex grid connectivity (constrained by barriers)
- [x] Add regions for key hex platforms
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_hex_small.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_hex_small.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Hexagonal arrangement is correct
- Triangular cluster shape (4-3-2-1 or similar)
- Connectivity respects barrier constraints

---

## Milestone 5: Repeated Alleyway Mazes (Phase 3)

### 5.1 Repeated Y-Maze

**File**: `src/neurospatial/simulation/mazes/repeated_y.py`

- [x] Create `RepeatedYDims(n_junctions=3, segment_length=50.0, width=10.0)` frozen dataclass
- [x] Implement `make_repeated_y_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment using corridor mask pattern (3 Y-junctions in series)
- [x] Implement Warner-Warden trick: dead ends split into two small corridors
- [x] Create track graph: chain of Y-junction nodes
- [x] Add regions for junctions and endpoints
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_repeated_y.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_repeated_y.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- 3 sequential Y-junctions
- Dead ends have Warner-Warden split (two small corridors)
- Track graph is connected

### 5.2 Repeated T-Maze

**File**: `src/neurospatial/simulation/mazes/repeated_t.py`

- [x] Create `RepeatedTDims(spine_length=150.0, arm_length=40.0, n_junctions=3, width=10.0)` frozen dataclass
- [x] Implement `make_repeated_t_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: horizontal spine + perpendicular T-arms (comb shape)
- [x] Create track graph: linear spine with branch nodes at each T-junction
- [x] Add regions: `start`, `junction_0` through `junction_2`, `arm_0_end` through `arm_2_end`
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_repeated_t.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_repeated_t.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Comb/rake shape with 3 perpendicular arms
- 90° angles at all junctions
- Track graph correctly represents topology

### 5.3 Hampton Court Maze

**File**: `src/neurospatial/simulation/mazes/hampton_court.py`

- [x] Create `HamptonCourtDims(size=300.0, corridor_width=11.0)` frozen dataclass
- [x] Implement `make_hampton_court_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create maze pattern (options: embedded binary image, procedural generation)
- [x] Create 2D environment: `Environment.from_image()` or `Environment.from_mask()`
- [x] Create track graph: skeletonize corridor mask → graph
- [x] Add regions: `start`, `goal` (center of maze)
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_hampton_court.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_hampton_court.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Complex labyrinth structure with dead ends
- Track graph navigable from start to goal
- ~300 × 300 cm size

---

## Milestone 6: Remaining Open-Field Mazes (Phase 4 Remaining)

### 6.1 Radial Arm Maze

**File**: `src/neurospatial/simulation/mazes/radial_arm.py`

- [x] Create `RadialArmDims(center_radius=15.0, arm_length=50.0, arm_width=10.0, n_arms=8)` frozen dataclass
- [x] Implement `make_radial_arm_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: octagonal center + arms at equal angular spacing (45° for 8 arms)
- [x] Create track graph: star graph (center → each arm end)
- [x] Add regions: `center`, `arm_0` through `arm_{n-1}`
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_radial_arm.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_radial_arm.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- 8 arms by default (6 for mice variant)
- Arms at correct angular spacing
- Star graph topology from center

### 6.2 Cheeseboard Maze

**File**: `src/neurospatial/simulation/mazes/cheeseboard.py`

- [x] Create `CheeseboardDims(diameter=110.0, grid_spacing=9.0, well_radius=1.5)` frozen dataclass
- [x] Implement `make_cheeseboard_maze(dims, bin_size)` → MazeEnvironments
- [x] Create 2D environment: circular arena
- [x] `env_track = None` (open field)
- [x] Add regions: grid of `well_i_j` point regions across entire surface
- [x] Calculate well positions on regular grid, filter to those within circular boundary
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_cheeseboard.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_cheeseboard.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Wells distributed across entire surface (not just perimeter)
- Regular grid spacing
- Wells within circular boundary only

---

## Milestone 7: Structured Lattice Mazes (Phase 5)

### 7.1 Crossword Maze

**File**: `src/neurospatial/simulation/mazes/crossword.py`

- [x] Create `CrosswordDims(grid_spacing=30.0, corridor_width=10.0, n_rows=4, n_cols=4)` frozen dataclass
- [x] Implement `make_crossword_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: 4×4 Manhattan-style grid pattern
- [x] Create track graph: Manhattan grid (4-connectivity)
- [x] Add regions: `node_i_j` at each intersection, `box_0` through `box_3` (corner boxes)
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_crossword.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_crossword.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- 4×4 grid structure
- 90° angles throughout
- Four corner boxes identified as start/goal locations

### 7.2 Honeycomb Maze

**File**: `src/neurospatial/simulation/mazes/honeycomb.py`

- [x] Create `HoneycombDims(spacing=25.0, n_rings=3)` frozen dataclass (37 platforms: 1 + 6 + 12 + 18)
- [x] Implement `make_honeycomb_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: `HexagonalLayout` with all 37 platforms active
- [x] Create track graph: hexagonal 6-connectivity
- [x] Add regions: `platform_0` through `platform_36`
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_honeycomb.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_honeycomb.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- 37 hexagonal platforms (1 center + 3 rings)
- 6-connectivity in track graph
- All platforms have region labels

### 7.3 Hamlet Maze

**File**: `src/neurospatial/simulation/mazes/hamlet.py`

- [x] Create `HamletDims(central_radius=30.0, arm_length=40.0, corridor_width=10.0, n_peripheral_arms=5)` frozen dataclass
- [x] Implement `make_hamlet_maze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: pentagon ring + 5 radiating arms, each splitting into 2 terminal boxes
- [x] Create track graph: pentagon ring nodes + arm nodes + 10 terminal goal nodes
- [x] Add regions: `ring_0` through `ring_4`, `goal_0` through `goal_9`
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_hamlet.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_hamlet.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- Pentagonal ring center with 5 arms
- Each arm splits into 2 terminal boxes (10 total goals)
- Track graph correctly represents connectivity

---

## Milestone 8: Rat HexMaze (Phase 6)

**File**: `src/neurospatial/simulation/mazes/rat_hexmaze.py`

### 8.1 Mouse HexMaze (Base Unit)

- [x] Create `MouseHexmazeDims(n_nodes=24, n_corridors=30, corridor_width=11.0)` frozen dataclass
- [x] Implement `make_mouse_hexmaze(dims, bin_size, include_track)` as internal helper
- [x] 24 crossing nodes, 30 corridors, 120° angles
- [x] Implement Warner-Warden dead-end trick (edges split before ending)

### 8.2 Rat HexMaze (4× Mouse)

- [x] Create `RatHexmazeDims(module_width=90.0, corridor_width=11.0, n_modules=3, nodes_per_module=24)` frozen dataclass
- [x] Implement `make_rat_hexmaze(dims, bin_size, include_track)` → MazeEnvironments
- [x] Create 2D environment: 3 hex clusters (A, B, C) with bridging corridors
- [x] Create track graph: 96 junction nodes + corridor edges
- [x] Add regions: `module_A`, `module_B`, `module_C`, `corridor_AB`, `corridor_BC`
- [x] Set `env.units = "cm"`
- [x] Add tests: `tests/simulation/mazes/test_rat_hexmaze.py`
- [x] Verify: `uv run pytest tests/simulation/mazes/test_rat_hexmaze.py -v`
- [x] Add to visualization script and run: `uv run python scripts/visualize_mazes.py`

**Success criteria**:

- 96 total nodes (3 × 32 from figure, or 4 × 24 from text)
- 120° junction angles throughout
- 3 distinct modules connected by bridges
- All nodes look identical (Warner-Warden trick at dead ends)
- Full maze spans ~9 × 5 m (900 × 500 cm)

---

## Milestone 9: Integration & Testing (Phase 7)

### 9.1 Update Module Exports

- [x] Update `src/neurospatial/simulation/mazes/__init__.py` with all factory functions
- [x] Update `src/neurospatial/simulation/__init__.py` to include maze exports
- [x] Verify: `from neurospatial.simulation.mazes import make_t_maze, make_watermaze, ...`

### 9.2 Create Test Infrastructure

- [x] Create `tests/simulation/mazes/conftest.py` with shared fixtures
- [x] Create base test class or helper for common maze assertions

### 9.3 Comprehensive Test Suite

- [x] `tests/simulation/mazes/test_base.py` - MazeEnvironments, MazeDims
- [x] `tests/simulation/mazes/test_geometry.py` - Helper functions (already done in M1.3)
- [x] `tests/simulation/mazes/test_corridor_mazes.py` - Integration tests for T, Y, W, Linear, Small Hex
- [x] `tests/simulation/mazes/test_repeated_mazes.py` - Integration tests for Repeated Y, T, Hampton Court
- [x] `tests/simulation/mazes/test_openfield_mazes.py` - Integration tests for Watermaze, Barnes, Radial, Cheeseboard
- [x] `tests/simulation/mazes/test_lattice_mazes.py` - Integration tests for Crossword, Honeycomb, Hamlet
- [x] `tests/simulation/mazes/test_rat_hexmaze.py` - Rat HexMaze specific tests

### 9.4 Validation Tests

For all mazes, verify:

- [x] Default dimensions create valid environment: `validate_environment(env, strict=True)`
- [x] Custom dimensions work without errors
- [x] `env.units == "cm"`
- [x] All regions exist and are queryable
- [x] Track graph (if present) is connected
- [x] Bin centers within expected spatial extent

### 9.5 Doctest Verification

- [x] All modules have working doctests: `uv run pytest --doctest-modules src/neurospatial/simulation/mazes/`

### 9.6 Integration with Simulation

- [ ] Verify compatibility with `simulate_trajectory_ou()` from existing simulation module
- [ ] Create example notebook or script demonstrating maze usage

---

## Validation Checklist

Run after each milestone:

- [x] Full test suite passes: `uv run pytest tests/simulation/mazes/ -v`
- [x] No regressions: `uv run pytest`
- [x] Linting passes: `uv run ruff check . && uv run ruff format .`
- [x] Type checking passes: `uv run mypy src/neurospatial/simulation/mazes/`
- [x] Doctests pass: `uv run pytest --doctest-modules src/neurospatial/simulation/mazes/`

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Mazes implemented | 15/15 | ✅ Complete |
| Test coverage for factories | 100% | ✅ Complete |
| All environments pass `validate_environment()` | 15/15 | ✅ Complete |
| Doctests pass in all modules | 15/15 | ✅ Complete |
| Compatible with trajectory simulation | Yes | ⬜ Not verified |

---

## File Summary

```
src/neurospatial/simulation/mazes/
├── __init__.py           # Public API exports
├── _base.py              # MazeEnvironments, MazeDims
├── _geometry.py          # Polygon/graph helpers
├── linear_track.py       # Linear Track
├── t_maze.py             # T-Maze
├── y_maze.py             # Y-Maze
├── w_maze.py             # W-Maze
├── hex_small.py          # Small Hex Maze
├── repeated_y.py         # Repeated Y-Maze
├── repeated_t.py         # Repeated T-Maze
├── hampton_court.py      # Hampton Court Maze
├── radial_arm.py         # Radial Arm Maze
├── barnes.py             # Barnes Maze
├── cheeseboard.py        # Cheeseboard Maze
├── watermaze.py          # Morris Water Maze
├── crossword.py          # Crossword Maze
├── honeycomb.py          # Honeycomb Maze
├── hamlet.py             # Hamlet Maze
└── rat_hexmaze.py        # Rat HexMaze

tests/simulation/mazes/
├── conftest.py           # Shared fixtures
├── test_base.py          # Base classes
├── test_geometry.py      # Helpers
├── test_linear_track.py
├── test_t_maze.py
├── test_y_maze.py
├── test_w_maze.py
├── test_hex_small.py
├── test_repeated_y.py
├── test_repeated_t.py
├── test_hampton_court.py
├── test_radial_arm.py
├── test_barnes.py
├── test_cheeseboard.py
├── test_watermaze.py
├── test_crossword.py
├── test_honeycomb.py
├── test_hamlet.py
├── test_rat_hexmaze.py
├── test_corridor_mazes.py    # Integration
├── test_repeated_mazes.py    # Integration
├── test_openfield_mazes.py   # Integration
└── test_lattice_mazes.py     # Integration
```

---

## Dependencies

**Existing (no new dependencies needed)**:

- `shapely` - Polygon operations (buffer, union)
- `networkx` - Track graphs
- `numpy` - Numerical operations

**Existing neurospatial APIs**:

- `Environment.from_polygon()` - Polygon-bounded grids
- `Environment.from_mask()` - Pre-defined N-D boolean mask
- `Environment.from_image()` - Binary image mask
- `Environment.from_graph()` - 1D track-based environments (requires `track-linearization`)
- `HexagonalLayout` - Hexagonal tessellations
- `Regions` - Named regions of interest
