"""Shared fixtures for animation tests.

This module contains fixtures specifically for animation-related testing,
including video overlays, video calibration, and environment fixtures
for video overlay validation.

Fixtures defined here:
- sample_video: Test video file (16x16 pixels, 10 frames)
- sample_video_array: Test video as numpy array
- sample_calibration: VideoCalibration for testing
- linearized_env: 1D environment for video overlay rejection tests
- polygon_env: 2D polygon environment for fallback tests
- masked_env: 2D masked grid for full video overlay support tests
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment

# =============================================================================
# Video Overlay Fixtures
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
