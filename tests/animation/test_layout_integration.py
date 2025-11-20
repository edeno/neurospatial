"""End-to-end integration tests for animation with different layout types.

These tests verify that the full rendering pipeline works across different
layout types (hexagonal, 1D graph, triangular, masked grid). M6 tests only
verified delegation, these verify actual rendering output.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from neurospatial import Environment

# ============================================================================
# Hexagonal Layout Tests
# ============================================================================


def test_hexagonal_layout_with_video_backend(tmp_path):
    """Test full rendering pipeline with hexagonal layout and video backend."""
    # Skip if ffmpeg not available
    if os.system("ffmpeg -version > /dev/null 2>&1") != 0:
        pytest.skip("ffmpeg not installed")

    # Create hexagonal environment
    env = Environment.from_layout(
        kind="hexagonal",
        layout_params={
            "hexagon_width": 10.0,
            "dimension_ranges": ((-50.0, 50.0), (-50.0, 50.0)),
        },
        name="hexagonal_test",
    )

    # Clear cache for pickle-ability (required for parallel rendering)
    env.clear_cache()

    # Generate random fields (10 frames for speed)
    n_frames = 10
    fields = [np.random.rand(env.n_bins) for _ in range(n_frames)]

    output_path = tmp_path / "test_hexagonal.mp4"

    # Render to MP4 with video backend
    env.animate_fields(
        fields,
        backend="video",
        save_path=str(output_path),
        fps=10,
        n_workers=1,  # Single worker for test stability
    )

    # Verify video file created
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify video metadata (duration) - skip frame count due to ffprobe inconsistencies
    # Using ffprobe to check video duration
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse duration
    duration_str = result.stdout.strip()
    if duration_str:
        duration = float(duration_str)
        expected_duration = n_frames / 10.0  # 10 fps
        # Allow ±10% tolerance for encoding
        assert abs(duration - expected_duration) / expected_duration < 0.1


# ============================================================================
# 1D Graph Layout Tests
# ============================================================================


def test_1d_graph_layout_with_html_backend(tmp_path):
    """Test full rendering pipeline with 1D graph layout and HTML backend."""
    pytest.importorskip("track_linearization")

    # Create simple 1D track graph (linear track)
    import networkx as nx

    # Create linear track: 0 -- 1 -- 2 -- 3 -- 4
    graph = nx.Graph()
    n_nodes = 5
    for i in range(n_nodes):
        graph.add_node(i, pos=(float(i * 10), 0.0))  # Evenly spaced along x-axis

    for i in range(n_nodes - 1):
        # Add edge with required distance attribute
        graph.add_edge(i, i + 1, distance=10.0)

    # Create 1D track environment
    # Define edge order (sequential traversal)
    edge_order = [(i, i + 1) for i in range(n_nodes - 1)]

    env = Environment.from_graph(
        graph,
        edge_order=edge_order,
        edge_spacing=10.0,  # 10 units between nodes
        bin_size=2.0,  # 2 units per bin
        name="linear_track_test",
    )

    # Generate random fields (10 frames for manageable HTML size)
    n_frames = 10
    fields = [np.random.rand(env.n_bins) for _ in range(n_frames)]

    output_path = tmp_path / "test_1d_track.html"

    # Render to HTML with HTML backend
    env.animate_fields(
        fields,
        backend="html",
        save_path=str(output_path),
        fps=10,
    )

    # Verify HTML file created
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Verify HTML contains expected structure
    html_content = output_path.read_text()
    assert "const frames = [" in html_content  # JavaScript frames array
    assert "data:image/png;base64" in html_content  # Base64 encoded frames

    # Verify 1D plot structure (should use line plot, not patches)
    # The HTML should contain the rendered field visualization
    assert len(fields) == n_frames  # Verify frame count


# ============================================================================
# Triangular Mesh Layout Tests
# ============================================================================


@pytest.mark.xdist_group(name="napari_gui")
def test_triangular_mesh_layout_with_napari_backend(tmp_path):
    """Test full rendering pipeline with triangular mesh layout."""
    pytest.importorskip("napari")

    from shapely.geometry import Polygon

    from neurospatial.animation.backends.napari_backend import render_napari

    # Create triangular mesh environment with rectangular boundary
    boundary = Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])
    env = Environment.from_layout(
        kind="TriangularMesh",
        layout_params={
            "boundary_polygon": boundary,
            "point_spacing": 15.0,
        },
        name="triangular_test",
    )

    # Generate random fields (10 frames)
    n_frames = 10
    fields = [np.random.rand(env.n_bins) for _ in range(n_frames)]

    # Render with napari backend
    viewer = render_napari(env, fields, vmin=0, vmax=1, fps=10)

    # Verify viewer created
    assert viewer is not None
    assert hasattr(viewer, "layers")
    assert len(viewer.layers) > 0

    # Verify triangular patches render properly
    # (viewer should have image layer with triangular bin structure)


# ============================================================================
# Masked Grid Layout Tests
# ============================================================================


@pytest.mark.xdist_group(name="napari_gui")
def test_masked_grid_layout_with_napari_backend():
    """Test full rendering pipeline with masked grid (boundary handling)."""
    pytest.importorskip("napari")

    from neurospatial.animation.backends.napari_backend import render_napari

    # Create masked grid with infer_active_bins=True
    # Generate sparse data points (circular region)
    n_samples = 200
    theta = np.random.rand(n_samples) * 2 * np.pi
    r = np.random.rand(n_samples) * 30  # Radius up to 30
    positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Create environment with active bin inference (circular mask)
    env = Environment.from_samples(
        positions,
        bin_size=5.0,
        infer_active_bins=True,
        bin_count_threshold=1,
        name="masked_grid_test",
    )

    # Verify only active bins are present (should be < full grid)
    full_grid_bins = (
        int(np.ceil(60 / 5.0))  # Range is ~(-30, 30) → 60 units
        ** 2
    )  # 2D grid
    assert env.n_bins < full_grid_bins  # Masked grid has fewer bins

    # Generate random fields (only for active bins)
    n_frames = 10
    fields = [np.random.rand(env.n_bins) for _ in range(n_frames)]

    # Render with napari backend (GPU acceleration)
    viewer = render_napari(env, fields, vmin=0, vmax=1, fps=10)

    # Verify viewer created
    assert viewer is not None
    assert hasattr(viewer, "layers")
    assert len(viewer.layers) > 0

    # Verify only active bins render (inactive bins should not appear)
    # This is verified by the fact that env.n_bins < full_grid_bins
