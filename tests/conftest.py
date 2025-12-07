"""Shared test fixtures for neurospatial test suite.

Fixture Naming Convention
=========================

**General-purpose grid fixtures** follow the pattern:
    {size}_{dims}d_{connectivity}_{layout}_env

Where:
    - size: tiny, small, medium, large (relative scale)
    - dims: 1d, 2d, 3d (dimensionality)
    - connectivity: ortho (orthogonal), diag (diagonal) - optional
    - layout: grid, hex, graph - optional if obvious

Examples:
    - small_2d_env: Small 2D grid with orthogonal connectivity
    - medium_2d_env_with_diagonal: Medium 2D grid with diagonal connectivity
    - large_2d_env: Large 2D grid for stress tests

**Domain-specific fixtures** use descriptive names:
    - Maze environments: {maze_type}_env (e.g., tmaze_env, ymaze_env)
    - Purpose fixtures: {purpose}_env_{detail} (e.g., spike_field_env_100)
    - Layout fixtures: {layout}_env (e.g., simple_hex_env, simple_graph_env)

Size Guidelines (approximate bin counts):
    - tiny: <10 bins (validation tests)
    - small: 10-50 bins (quick tests)
    - medium: 50-1000 bins (standard tests)
    - large: 1000+ bins (stress tests, mark as @pytest.mark.slow)
"""

import os

import networkx as nx
import numpy as np
import pytest
from hypothesis import Phase, Verbosity, settings
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.layout.engines.graph import GraphLayout

# =============================================================================
# Hypothesis Configuration for Performance
# =============================================================================
# Register Hypothesis profiles for different testing scenarios:
# - "ci": Fast profile for CI pipelines (fewer examples, no deadline)
# - "dev": Standard development profile (moderate examples)
# - "thorough": Full property testing (many examples, for pre-release)

settings.register_profile(
    "ci",
    max_examples=10,
    deadline=None,  # Disable deadline in CI (variable performance)
    suppress_health_check=[],
    phases=[Phase.explicit, Phase.reuse, Phase.generate],
    verbosity=Verbosity.quiet,
)

settings.register_profile(
    "dev",
    max_examples=25,
    deadline=5000,  # 5 second deadline
    verbosity=Verbosity.normal,
)

settings.register_profile(
    "thorough",
    max_examples=100,
    deadline=None,
    verbosity=Verbosity.verbose,
)

# Load profile based on environment variable (default to "dev")
# Set HYPOTHESIS_PROFILE=ci in CI environments for faster tests
_profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(_profile)

# =============================================================================
# Test Configuration Constants
# =============================================================================
# Use these instead of magic numbers in tests

# Bin sizes (in spatial units - typically cm)
SMALL_BIN_SIZE = 1.0
MEDIUM_BIN_SIZE = 2.0
LARGE_BIN_SIZE = 10.0

# Position sample counts for Environment.from_samples()
# Used to create trajectory data for environment discretization
SMALL_N_POSITIONS = 100  # Quick tests
MEDIUM_N_POSITIONS = 1000  # Standard tests
LARGE_N_POSITIONS = 5000  # Stress tests / slow tests

# Random seeds for reproducibility
DEFAULT_SEED = 42
ALT_SEED_1 = 43
ALT_SEED_2 = 44

# Tolerance levels for assertions
TIGHT_TOLERANCE = 0.01
MEDIUM_TOLERANCE = 0.05
LOOSE_TOLERANCE = 0.1

# Spatial extents (cm)
SMALL_EXTENT = 10.0
MEDIUM_EXTENT = 50.0
LARGE_EXTENT = 100.0


# =============================================================================
# --- Fixtures ---
# =============================================================================
@pytest.fixture(scope="session")
def plus_maze_graph() -> nx.Graph:
    """
    Defines a simple plus-shaped maze graph.
    Center node (0) at (0, 0)
    Arm 1 (North): Node 1 at (0, 2)
    Arm 2 (East): Node 2 at (2, 0)
    Arm 3 (South): Node 3 at (0, -2)
    Arm 4 (West): Node 4 at (-2, 0)
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(0.0, 2.0))  # North
    graph.add_node(2, pos=(2.0, 0.0))  # East
    graph.add_node(3, pos=(0.0, -2.0))  # South
    graph.add_node(4, pos=(-2.0, 0.0))  # West

    # Add edge_id, as expected by track_linearization
    graph.add_edge(0, 1, distance=2.0, edge_id=0)
    graph.add_edge(0, 2, distance=2.0, edge_id=1)
    graph.add_edge(0, 3, distance=2.0, edge_id=2)
    graph.add_edge(0, 4, distance=2.0, edge_id=3)
    return graph


@pytest.fixture(scope="session")
def plus_maze_edge_order() -> list[tuple[int, int]]:
    """Edge order for linearizing the plus maze."""
    # Path: West arm -> Center -> North arm -> Center -> East arm -> Center -> South arm
    return [(4, 0), (0, 1), (0, 2), (0, 3)]


@pytest.fixture(scope="session")
def plus_maze_positions() -> NDArray[np.float64]:
    """Regularly spaced position samples along the plus maze arms."""
    samples = [
        # Center
        [0.0, 0.0],
        # West arm: (-2,0) to (0,0)
        [-2.0, 0.0],
        [-1.5, 0.0],
        [-1.0, 0.0],
        [-0.5, 0.0],
        # North arm: (0,0) to (0,2)
        [0.0, 0.5],
        [0.0, 1.0],
        [0.0, 1.5],
        [0.0, 2.0],
        # East arm: (0,0) to (2,0)
        [0.5, 0.0],
        [1.0, 0.0],
        [1.5, 0.0],
        [2.0, 0.0],
        # South arm: (0,0) to (0,-2)
        [0.0, -0.5],
        [0.0, -1.0],
        [0.0, -1.5],
        [0.0, -2.0],
        # Some off-track points
        [3.0, 3.0],
        [-3.0, -3.0],
    ]
    return np.array(samples, dtype=np.float64)


@pytest.fixture(scope="session")
def graph_env(
    plus_maze_graph: nx.Graph, plus_maze_edge_order: list[tuple[int, int]]
) -> Environment:
    """Environment created from the plus maze graph.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Capture parameters explicitly to pass to Environment constructor
    # for correct serialization testing.
    layout_build_params = {
        "graph_definition": plus_maze_graph,
        "edge_order": plus_maze_edge_order,
        "edge_spacing": 0.0,
        "bin_size": 0.5,
    }
    layout_instance = GraphLayout()
    layout_instance.build(**layout_build_params)

    return Environment(
        name="PlusMazeGraph",
        layout=layout_instance,
        layout_type_used="Graph",
        layout_params_used=layout_build_params,
    )


@pytest.fixture(scope="session")
def grid_env_from_samples(
    plus_maze_positions: NDArray[np.float64],
) -> Environment:
    """Environment created as a RegularGrid from plus maze position samples.

    Session-scoped for performance: Environment.from_samples() is expensive,
    and this fixture is read-only in tests.
    """
    return Environment.from_samples(
        positions=plus_maze_positions,
        bin_size=0.5,
        infer_active_bins=True,
        bin_count_threshold=0,  # A single sample makes a bin active
        dilate=False,  # Keep it simple, no dilation
        fill_holes=False,
        close_gaps=False,
        name="PlusMazeGrid",
        connect_diagonal_neighbors=False,  # Only orthogonal for easier neighbor check
    )


@pytest.fixture(scope="session")
def simple_graph_for_layout() -> nx.Graph:
    """Minimal graph with pos and distance attributes for GraphLayout.

    Session-scoped for performance: Graph is read-only in tests,
    safe to share across all tests.
    """
    G = nx.Graph()
    G.add_node(0, pos=(0.0, 0.0))
    G.add_node(1, pos=(1.0, 0.0))
    G.add_edge(0, 1, distance=1.0, edge_id=0)  # Add edge_id
    return G


@pytest.fixture(scope="session")
def simple_hex_env(plus_maze_positions) -> Environment:
    """Basic hexagonal environment for mask testing.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    return Environment.from_samples(
        positions=plus_maze_positions,  # Use existing samples
        layout="Hexagonal",
        bin_size=2.0,  # Reasonably large hexes
        name="SimpleHexEnvForMask",
        infer_active_bins=True,  # Important for source_flat_to_active_node_id_map
        bin_count_threshold=0,
    )


@pytest.fixture(scope="session")
def simple_graph_env(simple_graph_for_layout) -> Environment:
    """Basic graph environment for mask testing.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    edge_order = [(0, 1)]
    # For serialization to pass correctly, ensure layout_params_used are captured
    layout_build_params = {
        "graph_definition": simple_graph_for_layout,
        "edge_order": edge_order,
        "edge_spacing": 0.0,
        "bin_size": 0.5,
    }
    layout_instance = GraphLayout()
    layout_instance.build(**layout_build_params)
    return Environment(
        name="SimpleGraphEnvForMask",
        layout=layout_instance,
        layout_type_used="Graph",
        layout_params_used=layout_build_params,
    )


@pytest.fixture(scope="session")
def grid_env_for_indexing(plus_maze_positions) -> Environment:
    """A 2D RegularGrid environment suitable for index testing.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    return Environment.from_samples(
        positions=plus_maze_positions,  # Creates a reasonable grid
        bin_size=1.0,
        infer_active_bins=True,
        bin_count_threshold=0,
        name="GridForIndexing",
    )


@pytest.fixture(scope="session")
def env_all_active_2x2() -> Environment:
    """A 2x2 grid where all 4 cells are active.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    active_mask = np.array([[True, True], [True, True]], dtype=bool)
    grid_edges = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    return Environment.from_mask(
        active_mask=active_mask,
        grid_edges=grid_edges,
        name="AllActive2x2",
        connect_diagonal_neighbors=False,  # Orthogonal connections for simpler graph
    )


@pytest.fixture(scope="session")
def ymaze_graph() -> nx.Graph:
    """
    Defines a Y-shaped maze graph for turn sequence testing.

    Structure:
        Node 0 (center/junction): (0, 0)
        Node 1 (straight/North arm): (0, 10)
        Node 2 (left/Northwest arm): (-7, 7)
        Node 3 (right/Northeast arm): (7, 7)

    The animal starts at center (0) and can choose between:
    - Straight ahead (0→1): No turn
    - Left (0→2): Left turn (~45 degrees)
    - Right (0→3): Right turn (~45 degrees)
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))  # Center/junction
    graph.add_node(1, pos=(0.0, 10.0))  # Straight ahead (North)
    graph.add_node(2, pos=(-7.0, 7.0))  # Left arm (Northwest)
    graph.add_node(3, pos=(7.0, 7.0))  # Right arm (Northeast)

    # Add edges with distances and edge_id
    graph.add_edge(0, 1, distance=10.0, edge_id=0)  # Center → Straight
    graph.add_edge(
        0, 2, distance=np.sqrt(7**2 + 7**2), edge_id=1
    )  # Center → Left (~9.9)
    graph.add_edge(
        0, 3, distance=np.sqrt(7**2 + 7**2), edge_id=2
    )  # Center → Right (~9.9)
    return graph


@pytest.fixture(scope="session")
def ymaze_env(ymaze_graph: nx.Graph) -> Environment:
    """Y-maze environment for testing turn sequence detection.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    edge_order = [(0, 1), (0, 2), (0, 3)]
    layout_build_params = {
        "graph_definition": ymaze_graph,
        "edge_order": edge_order,
        "edge_spacing": 0.0,
        "bin_size": 1.0,
    }
    layout_instance = GraphLayout()
    layout_instance.build(**layout_build_params)

    return Environment(
        name="YMaze",
        layout=layout_instance,
        layout_type_used="Graph",
        layout_params_used=layout_build_params,
    )


@pytest.fixture(scope="session")
def tmaze_graph() -> nx.Graph:
    """
    Defines a T-shaped maze graph for turn sequence testing.

    Structure:
        Node 0 (start/bottom): (0, -10)
        Node 1 (center/junction): (0, 0)
        Node 2 (left arm): (-10, 0)
        Node 3 (right arm): (10, 0)

    The animal starts at bottom (0), travels to junction (1),
    then must choose left (1→2) or right (1→3).
    """
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, -10.0))  # Start (bottom)
    graph.add_node(1, pos=(0.0, 0.0))  # Center/junction
    graph.add_node(2, pos=(-10.0, 0.0))  # Left arm
    graph.add_node(3, pos=(10.0, 0.0))  # Right arm

    # Add edges with distances and edge_id
    graph.add_edge(0, 1, distance=10.0, edge_id=0)  # Start → Center
    graph.add_edge(1, 2, distance=10.0, edge_id=1)  # Center → Left
    graph.add_edge(1, 3, distance=10.0, edge_id=2)  # Center → Right
    return graph


@pytest.fixture(scope="session")
def tmaze_env(tmaze_graph: nx.Graph) -> Environment:
    """T-maze environment for testing turn sequence detection.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    edge_order = [(0, 1), (1, 2), (1, 3)]
    layout_build_params = {
        "graph_definition": tmaze_graph,
        "edge_order": edge_order,
        "edge_spacing": 0.0,
        "bin_size": 1.0,
    }
    layout_instance = GraphLayout()
    layout_instance.build(**layout_build_params)

    return Environment(
        name="TMaze",
        layout=layout_instance,
        layout_type_used="Graph",
        layout_params_used=layout_build_params,
    )


# =============================================================================
# Dense Grid Fixtures for Metrics/Boundary Tests
# =============================================================================
# These fixtures replace inefficient nested-loop grid generation patterns.
# Session-scoped for performance since environments are read-only in tests.


@pytest.fixture(scope="session")
def dense_rectangular_grid_env() -> Environment:
    """Dense 50x50 rectangular grid for boundary/place field tests.

    Uses vectorized meshgrid instead of nested loops for 100x faster creation.
    Creates a grid with ~156 active bins at bin_size=4.0.

    Use for: border_score tests, place field tests, region coverage tests.
    """
    x = np.linspace(0, 50, 500)
    y = np.linspace(0, 50, 500)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=4.0)


@pytest.fixture(scope="session")
def dense_40x40_grid_env() -> Environment:
    """Dense 40x40 rectangular grid for corner/edge tests.

    Similar to dense_rectangular_grid_env but smaller extent.
    Creates a grid with ~100 active bins at bin_size=4.0.

    Use for: Corner field tests, edge case validation.
    """
    x = np.linspace(0, 40, 400)
    y = np.linspace(0, 40, 400)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=4.0)


@pytest.fixture(scope="session")
def dense_50x50_bin5_env() -> Environment:
    """Dense 50x50 rectangular grid with bin_size=5.0.

    Same extent as dense_rectangular_grid_env but larger bins.
    Creates a grid with ~100 active bins at bin_size=5.0.

    Use for: Central field tests, euclidean distance tests.
    """
    x = np.linspace(0, 50, 500)
    y = np.linspace(0, 50, 500)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=5.0)


# =============================================================================
# Spike Field Test Fixtures
# =============================================================================
# These fixtures replace the 39 identical Environment.from_samples() calls
# in tests/test_spike_field.py.


@pytest.fixture(scope="session")
def spike_field_env_100() -> Environment:
    """100x100 environment with bin_size=10 for spike field tests.

    Standard diagonal trajectory covering the environment.
    Creates ~100 active bins.

    Use for: spike_to_field tests, compute_place_field tests.
    """
    positions = np.column_stack([np.linspace(0, 100, 1000), np.linspace(0, 100, 1000)])
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture(scope="session")
def spike_field_trajectory() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Standard trajectory for spike field tests (1000 points, 10 seconds).

    Returns
    -------
    tuple[NDArray, NDArray]
        (times, positions) where times is shape (1000,) and positions is (1000, 2).
    """
    positions = np.column_stack([np.linspace(0, 100, 1000), np.linspace(0, 100, 1000)])
    times = np.linspace(0, 10, 1000)
    return times, positions


@pytest.fixture(scope="session")
def spike_field_env_random() -> Environment:
    """Random uniform 60x60 environment with bin_size=5 for compute_place_field tests.

    Uses seeded RNG for reproducibility.
    Positions uniformly distributed in [20, 80] range.

    Use for: compute_place_field tests with random coverage.
    """
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.uniform(20, 80, (500, 2))
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture(scope="session")
def spike_field_random_trajectory() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Random trajectory for compute_place_field tests (500 points, 50 seconds).

    Uses seeded RNG for reproducibility (DEFAULT_SEED=42).
    Positions uniformly distributed in [20, 80] range.

    Returns
    -------
    tuple[NDArray, NDArray]
        (times, positions) where times is shape (500,) and positions is (500, 2).
    """
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.uniform(20, 80, (500, 2))
    times = np.linspace(0, 50, 500)
    return times, positions


# =============================================================================
# Minimal Grid Fixtures for Occupancy/Transitions Tests
# =============================================================================
# These fixtures replace the 63 tiny environment creations
# in test_occupancy.py (23) and test_transitions.py (40).


@pytest.fixture(scope="session")
def minimal_2d_grid_env() -> Environment:
    """Minimal 10x10 grid for quick validation tests.

    Creates a small environment from just 2 corner points.
    Very fast to create, suitable for basic functionality tests.

    Use for: Basic occupancy tests, simple validation.
    """
    data = np.array([[0, 0], [10, 10]], dtype=np.float64)
    return Environment.from_samples(data, bin_size=2.0)


@pytest.fixture(scope="session")
def minimal_20x20_grid_env() -> Environment:
    """Small 20x20 grid for trajectory tests.

    Slightly larger than minimal_2d_grid_env for tests needing
    more spatial resolution.

    Use for: Trajectory occupancy tests, gap handling tests.
    """
    data = np.array([[0, 0], [20, 20]], dtype=np.float64)
    return Environment.from_samples(data, bin_size=5.0)


@pytest.fixture(scope="session")
def linear_track_1d_env() -> Environment:
    """Simple 1D linear track (6 positions, bin_size=2.5).

    Creates a 1D environment suitable for transition testing.
    Has ~4-5 bins along the track.

    Use for: 1D transition tests, sequence validation.
    """
    return Environment.from_samples(
        np.array([[i] for i in range(0, 11, 2)], dtype=np.float64),
        bin_size=2.5,
    )


@pytest.fixture(scope="session")
def minimal_1d_env() -> Environment:
    """Minimal 1D environment from two endpoints (bin_size=5.0).

    Creates a small 1D environment from just 2 corner points [0] and [10].
    Very fast to create, suitable for validation/error tests.

    Use for: Input validation tests, error handling tests.
    """
    return Environment.from_samples(
        np.array([[0.0], [10.0]], dtype=np.float64),
        bin_size=5.0,
    )


@pytest.fixture(scope="session")
def simple_3d_env() -> Environment:
    """A simple 3D environment for comprehensive 3D testing.

    Creates a fully populated 5x5x5 grid (125 bins) with diagonal connectivity.
    Uses bin_size=2.0 covering [0, 10] in each dimension.

    Deterministic: Uses from_mask() with all-True mask for reproducible
    bin structure across all runs.

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Fully populated 5x5x5 grid (125 bins total)
    # Bin edges at [0, 2, 4, 6, 8, 10] → 5 bins per dimension
    mask = np.ones((5, 5, 5), dtype=bool)
    edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    grid_edges = (edges, edges, edges)

    env = Environment.from_mask(
        active_mask=mask,
        grid_edges=grid_edges,
        name="Simple3DEnv",
        connect_diagonal_neighbors=True,  # Enable full 3D connectivity (up to 26 neighbors)
    )
    env.units = "cm"
    env.frame = "session1"
    return env


# =============================================================================
# General-Purpose Reusable Environment Fixtures (Priority 2.1)
# =============================================================================
# These fixtures reduce duplicate Environment.from_samples() calls across tests
# All are session-scoped since environments are read-only in tests


@pytest.fixture(scope="session")
def small_2d_env() -> Environment:
    """Small 2D environment for quick tests.

    - Size: 10x10 cm (5x5 bins at bin_size=2.0 = 25 bins total)
    - Layout: RegularGrid (supports rebin() and other grid operations)
    - Connectivity: Orthogonal only (faster, simpler)

    Deterministic: Uses meshgrid positions for fully reproducible
    bin structure across all runs and parallel workers.

    Use for: Fast unit tests, basic functionality tests

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Deterministic meshgrid covering [0, 10] x [0, 10]
    # Dense enough to ensure all bins are active
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,  # 2.0
        name="Small2DEnv",
        connect_diagonal_neighbors=False,
    )


@pytest.fixture(scope="session")
def medium_2d_env() -> Environment:
    """Medium 2D environment for standard tests.

    - Size: 50x50 cm (25x25 bins at bin_size=2.0 = 625 bins total)
    - Layout: RegularGrid (supports rebin() and other grid operations)
    - Connectivity: Orthogonal only

    Deterministic: Uses meshgrid positions for fully reproducible
    bin structure across all runs and parallel workers.

    Use for: Standard feature tests, integration tests

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Deterministic meshgrid covering [0, 50] x [0, 50]
    # Dense enough to ensure all bins are active
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,  # 2.0
        name="Medium2DEnv",
        connect_diagonal_neighbors=False,
    )


@pytest.fixture(scope="session")
def large_2d_env() -> Environment:
    """Large 2D environment for stress tests.

    - Size: 100x100 cm (50x50 bins at bin_size=2.0 = 2500 bins total)
    - Layout: RegularGrid (supports rebin() and other grid operations)
    - Connectivity: Orthogonal only

    Deterministic: Uses meshgrid positions for fully reproducible
    bin structure across all runs and parallel workers.

    Use for: Performance tests, large-scale validation
    Mark tests using this as @pytest.mark.slow

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Deterministic meshgrid covering [0, 100] x [0, 100]
    # Dense enough to ensure all bins are active
    x = np.linspace(0, 100, 200)
    y = np.linspace(0, 100, 200)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,  # 2.0
        name="Large2DEnv",
        connect_diagonal_neighbors=False,
    )


@pytest.fixture(scope="session")
def small_1d_env() -> Environment:
    """Small 1D environment for quick linear track tests.

    - Length: 10 cm (5 bins at bin_size=2.0)
    - Connectivity: Linear (each bin connects to neighbors)

    Deterministic: Uses linspace positions for fully reproducible
    bin structure across all runs and parallel workers.

    Use for: Fast tests of 1D-specific functionality

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Deterministic linear positions covering [0, 10]
    # Dense enough to ensure all bins are active
    positions = np.linspace(0, 10, 50).reshape(-1, 1)

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,  # 2.0
        name="Small1DEnv",
    )


@pytest.fixture(scope="session")
def medium_2d_env_with_diagonal() -> Environment:
    """Medium 2D environment with diagonal connectivity.

    - Size: 50x50 cm (25x25 bins at bin_size=2.0 = 625 bins total)
    - Layout: RegularGrid (supports rebin() and other grid operations)
    - Connectivity: Diagonal enabled (8-connectivity)

    Deterministic: Uses meshgrid positions for fully reproducible
    bin structure across all runs and parallel workers.

    Use for: Tests requiring diagonal neighbor relationships

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Deterministic meshgrid covering [0, 50] x [0, 50]
    # Dense enough to ensure all bins are active
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,  # 2.0
        name="Medium2DEnvDiagonal",
        connect_diagonal_neighbors=True,
    )
