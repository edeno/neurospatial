from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.layout.engines.graph import GraphLayout

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


@pytest.fixture
def simple_graph_for_layout() -> nx.Graph:
    """Minimal graph with pos and distance attributes for GraphLayout."""
    G = nx.Graph()
    G.add_node(0, pos=(0.0, 0.0))
    G.add_node(1, pos=(1.0, 0.0))
    G.add_edge(0, 1, distance=1.0, edge_id=0)  # Add edge_id
    return G


@pytest.fixture
def simple_hex_env(plus_maze_positions) -> Environment:
    """Basic hexagonal environment for mask testing."""
    return Environment.from_samples(
        positions=plus_maze_positions,  # Use existing samples
        layout="Hexagonal",
        bin_size=2.0,  # Reasonably large hexes
        name="SimpleHexEnvForMask",
        infer_active_bins=True,  # Important for source_flat_to_active_node_id_map
        bin_count_threshold=0,
    )


@pytest.fixture
def simple_graph_env(simple_graph_for_layout) -> Environment:
    """Basic graph environment for mask testing."""
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


@pytest.fixture
def grid_env_for_indexing(plus_maze_positions) -> Environment:
    """A 2D RegularGrid environment suitable for index testing."""
    return Environment.from_samples(
        positions=plus_maze_positions,  # Creates a reasonable grid
        bin_size=1.0,
        infer_active_bins=True,
        bin_count_threshold=0,
        name="GridForIndexing",
    )


@pytest.fixture
def env_all_active_2x2() -> Environment:
    """A 2x2 grid where all 4 cells are active."""
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


@pytest.fixture(scope="session")
def simple_3d_env() -> Environment:
    """A simple 3D environment for comprehensive 3D testing.

    Creates a 3D grid from random samples in a 10x10x10 space.
    Uses bin_size=2.0 and enables diagonal connectivity to test
    full 3D neighbor connectivity (up to 26 neighbors per bin).

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    # Generate random 3D position samples
    rng = np.random.default_rng(42)  # Local RNG for test isolation
    positions = rng.random((200, 3)) * 10.0  # 200 points in [0, 10] cube

    return Environment.from_samples(
        positions=positions,
        bin_size=2.0,
        name="Simple3DEnv",
        connect_diagonal_neighbors=True,  # Enable full 3D connectivity (up to 26 neighbors)
        infer_active_bins=True,
        bin_count_threshold=1,  # At least 1 sample per active bin
    )


# =============================================================================
# General-Purpose Reusable Environment Fixtures (Priority 2.1)
# =============================================================================
# These fixtures reduce duplicate Environment.from_samples() calls across tests
# All are session-scoped since environments are read-only in tests


@pytest.fixture(scope="session")
def small_2d_env() -> Environment:
    """Small 2D environment for quick tests.

    - Size: ~10x10 cm (SMALL_EXTENT)
    - Bins: ~25 bins (bin_size=2.0)
    - Samples: 100 positions (SMALL_N_POSITIONS)
    - Connectivity: Orthogonal only (faster, simpler)

    Use for: Fast unit tests, basic functionality tests

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.standard_normal((SMALL_N_POSITIONS, 2)) * (SMALL_EXTENT / 2)

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,
        name="Small2DEnv",
        connect_diagonal_neighbors=False,
        infer_active_bins=True,
        bin_count_threshold=1,
    )


@pytest.fixture(scope="session")
def medium_2d_env() -> Environment:
    """Medium 2D environment for standard tests.

    - Size: ~50x50 cm (MEDIUM_EXTENT)
    - Bins: ~625 bins (bin_size=2.0)
    - Samples: 1000 positions (MEDIUM_N_POSITIONS)
    - Connectivity: Orthogonal only

    Use for: Standard feature tests, integration tests

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.standard_normal((MEDIUM_N_POSITIONS, 2)) * (MEDIUM_EXTENT / 2)

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,
        name="Medium2DEnv",
        connect_diagonal_neighbors=False,
        infer_active_bins=True,
        bin_count_threshold=1,
    )


@pytest.fixture(scope="session")
def large_2d_env() -> Environment:
    """Large 2D environment for stress tests.

    - Size: ~100x100 cm (LARGE_EXTENT)
    - Bins: ~2500 bins (bin_size=2.0)
    - Samples: 5000 positions (LARGE_N_POSITIONS)
    - Connectivity: Orthogonal only

    Use for: Performance tests, large-scale validation
    Mark tests using this as @pytest.mark.slow

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.standard_normal((LARGE_N_POSITIONS, 2)) * (LARGE_EXTENT / 2)

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,
        name="Large2DEnv",
        connect_diagonal_neighbors=False,
        infer_active_bins=True,
        bin_count_threshold=1,
    )


@pytest.fixture(scope="session")
def small_1d_env() -> Environment:
    """Small 1D environment for quick linear track tests.

    - Length: ~10 cm (SMALL_EXTENT)
    - Bins: ~5 bins (bin_size=2.0)
    - Samples: 100 positions (SMALL_N_POSITIONS)

    Use for: Fast tests of 1D-specific functionality

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.uniform(0, SMALL_EXTENT, size=(SMALL_N_POSITIONS, 1))

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,
        name="Small1DEnv",
        infer_active_bins=True,
        bin_count_threshold=1,
    )


@pytest.fixture(scope="session")
def medium_2d_env_with_diagonal() -> Environment:
    """Medium 2D environment with diagonal connectivity.

    - Size: ~50x50 cm (MEDIUM_EXTENT)
    - Bins: ~625 bins (bin_size=2.0)
    - Samples: 1000 positions (MEDIUM_N_POSITIONS)
    - Connectivity: Diagonal enabled (8-connectivity)

    Use for: Tests requiring diagonal neighbor relationships

    Session-scoped for performance: Environment is read-only in tests,
    safe to share across all tests.
    """
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.standard_normal((MEDIUM_N_POSITIONS, 2)) * (MEDIUM_EXTENT / 2)

    return Environment.from_samples(
        positions=positions,
        bin_size=MEDIUM_BIN_SIZE,
        name="Medium2DEnvDiagonal",
        connect_diagonal_neighbors=True,
        infer_active_bins=True,
        bin_count_threshold=1,
    )


# =============================================================================
# Video Overlay Fixtures (Milestone 6.2)
# =============================================================================


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Create a sample test video (16x16 pixels, 10 frames, 10 fps).

    Each frame has a distinct pattern for verification:
    - Frame i has brightness level i * 25 (0, 25, 50, ..., 225)

    Parameters
    ----------
    tmp_path : Path
        pytest's temporary directory fixture.

    Returns
    -------
    Path
        Path to the created video file.

    Notes
    -----
    Uses OpenCV for video creation. The video is created fresh for each test
    to ensure isolation. Use `sample_video_array` for tests that don't need
    a file on disk.
    """
    import cv2

    video_path = tmp_path / "sample_video.mp4"

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    frame_size = (16, 16)  # width, height
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, frame_size)

    # Write 10 frames with different brightness levels
    for i in range(10):
        # Create frame with brightness = i * 25 (0, 25, 50, ..., 225)
        frame = np.full((16, 16, 3), i * 25, dtype=np.uint8)
        writer.write(frame)  # OpenCV uses BGR

    writer.release()
    return video_path


@pytest.fixture
def sample_video_array() -> np.ndarray:
    """Create a sample video as a numpy array (16x16 pixels, 10 frames).

    Returns
    -------
    np.ndarray
        Video array with shape (10, 16, 16, 3) and dtype uint8.
        Frame i has brightness level i * 25.

    Notes
    -----
    This fixture is faster than `sample_video` since it doesn't create a file.
    Use for tests that accept array input directly.
    """
    frames = []
    for i in range(10):
        frame = np.full((16, 16, 3), i * 25, dtype=np.uint8)
        frames.append(frame)
    return np.array(frames, dtype=np.uint8)


@pytest.fixture
def sample_calibration():
    """Create a sample VideoCalibration for testing.

    Returns a calibration that maps a 16x16 pixel video to a 16x16 cm
    environment (1:1 cm per pixel with Y-flip).

    Returns
    -------
    VideoCalibration
        Calibration object with 1:1 pixel-to-cm mapping.

    Notes
    -----
    The calibration includes Y-axis flip (video origin top-left to
    environment origin bottom-left), which is standard for video overlays.
    """
    from neurospatial.transforms import VideoCalibration, calibrate_from_scale_bar

    # Create 1:1 mapping (1 cm per pixel) with Y-flip
    # 16 pixels = 16 cm
    transform = calibrate_from_scale_bar(
        p1_px=(0.0, 0.0),
        p2_px=(16.0, 0.0),
        known_length_cm=16.0,
        frame_size_px=(16, 16),
    )

    return VideoCalibration(
        transform_px_to_cm=transform,
        frame_size_px=(16, 16),
    )


# =============================================================================
# Video Overlay Environment Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def linearized_env() -> Environment:
    """1D linearized track environment for video overlay rejection tests.

    This environment has n_dims=1 and should be rejected by VideoOverlay
    validation since video frames are 2D images.

    Returns
    -------
    Environment
        1D track environment with 15 bins along a linear path.
    """
    import networkx as nx

    # Create a simple linear 1D track
    graph = nx.Graph()
    # Add nodes with 1D positions (tuples of length 1)
    graph.add_nodes_from(
        [
            (0, {"pos": (0.0,)}),
            (1, {"pos": (10.0,)}),
            (2, {"pos": (20.0,)}),
            (3, {"pos": (30.0,)}),
        ]
    )
    # Add edges with required distance attributes
    graph.add_edge(0, 1, distance=10.0)
    graph.add_edge(1, 2, distance=10.0)
    graph.add_edge(2, 3, distance=10.0)

    edge_order = [(0, 1), (1, 2), (2, 3)]
    return Environment.from_graph(
        graph, edge_order, edge_spacing=0.0, bin_size=2.0, name="LinearizedTrack"
    )


@pytest.fixture(scope="session")
def polygon_env() -> Environment:
    """Non-grid 2D polygon environment for video overlay fallback tests.

    This environment uses from_polygon which creates a MaskedGridLayout.
    It has dimension_ranges but may not have a perfect rectangular grid_shape
    (bins are masked by polygon boundary).

    Returns
    -------
    Environment
        2D polygon environment covering 100x80 cm rectangle.
    """
    from shapely.geometry import box

    polygon = box(0, 0, 100, 80)  # Rectangle 100x80 cm
    return Environment.from_polygon(polygon, bin_size=5.0, name="PolygonEnv")


@pytest.fixture(scope="session")
def masked_env() -> Environment:
    """Grid 2D environment with mask for full video overlay support tests.

    This environment has complete grid metadata (grid_shape, dimension_ranges)
    and represents the standard use case for video overlays.

    Returns
    -------
    Environment
        2D masked grid environment covering 100x80 cm rectangle.
    """
    # Create a boolean mask for a rectangular region
    # mask shape is (dim0_bins, dim1_bins) matching grid_edges order
    # grid_edges = (dim0_edges, dim1_edges)
    # For a 100x80 cm region with bin_size=5:
    # - dim0 (x): 20 bins -> 21 edges
    # - dim1 (y): 16 bins -> 17 edges
    x_edges = np.linspace(0, 100, 21)  # 20 bins
    y_edges = np.linspace(0, 80, 17)  # 16 bins

    # mask shape must match (len(x_edges)-1, len(y_edges)-1) = (20, 16)
    mask = np.ones((20, 16), dtype=bool)

    return Environment.from_mask(
        mask,
        grid_edges=(x_edges, y_edges),
        name="MaskedGridEnv",
    )
