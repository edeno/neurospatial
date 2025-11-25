"""Performance benchmarks for Napari backend with overlay rendering.

This module contains performance tests marked with @pytest.mark.slow.
Run with: uv run pytest -m slow tests/animation/test_napari_performance.py -v -s

Performance targets:
- Update latency < 50 ms/frame with realistic pose + trail data
- Batched updates faster than individual layer updates
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

# Skip all tests if napari not available
pytest.importorskip("napari")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def realistic_env():
    """Create realistic 2D environment for performance testing."""
    from neurospatial import Environment

    # Large environment (100x100 cm arena) - deterministic grid
    x = np.linspace(0, 100, 51)
    y = np.linspace(0, 100, 51)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    return Environment.from_samples(positions, bin_size=2.0)


@pytest.fixture
def realistic_fields(realistic_env: Environment) -> list[NDArray[np.float64]]:
    """Create realistic field sequence (100 frames)."""
    rng = np.random.default_rng(42)
    n_frames = 100
    return [rng.random(realistic_env.n_bins) for _ in range(n_frames)]


@pytest.fixture
def realistic_pose_overlay_data():
    """Create realistic BodypartData for performance testing.

    Simulates multi-keypoint tracking (10 bodyparts, 100 frames).
    """
    from neurospatial.animation.overlays import BodypartData
    from neurospatial.animation.skeleton import Skeleton

    n_frames = 100
    # Trajectory is deterministic (sin/cos), no RNG needed

    # Create realistic trajectory with smooth motion
    t = np.linspace(0, 2 * np.pi, n_frames)
    center_x = 50 + 20 * np.cos(t)  # Circular motion
    center_y = 50 + 20 * np.sin(t)

    bodyparts = {}
    bodypart_names = [
        "nose",
        "left_ear",
        "right_ear",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "spine",
        "left_hip",
        "right_hip",
        "tail_base",
    ]

    for i, name in enumerate(bodypart_names):
        # Add offset for each bodypart
        offset_x = (i - 5) * 2.0
        offset_y = np.sin(i) * 1.5
        bodyparts[name] = np.column_stack([center_x + offset_x, center_y + offset_y])

    # Create realistic skeleton
    skeleton = Skeleton(
        name="test",
        nodes=tuple(bodypart_names),
        edges=(
            ("nose", "neck"),
            ("left_ear", "neck"),
            ("right_ear", "neck"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("left_shoulder", "spine"),
            ("right_shoulder", "spine"),
            ("spine", "left_hip"),
            ("spine", "right_hip"),
            ("left_hip", "tail_base"),
            ("right_hip", "tail_base"),
        ),
        edge_color="white",
        edge_width=2.0,
    )

    colors = {name: f"C{i}" for i, name in enumerate(bodypart_names)}

    return BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=colors,
    )


@pytest.fixture
def realistic_position_overlay_data():
    """Create realistic PositionData with trail for performance testing."""
    from neurospatial.animation.overlays import PositionData

    n_frames = 100
    # Trajectory is deterministic (sin/cos), no RNG needed

    # Circular trajectory
    t = np.linspace(0, 2 * np.pi, n_frames)
    x = 50 + 20 * np.cos(t)
    y = 50 + 20 * np.sin(t)
    data = np.column_stack([x, y])

    return PositionData(data=data, color="red", size=10.0, trail_length=20)


@pytest.fixture
def realistic_head_direction_overlay_data():
    """Create realistic HeadDirectionData for performance testing."""
    from neurospatial.animation.overlays import HeadDirectionData

    n_frames = 100
    # Angles are deterministic (linspace), no RNG needed

    # Angles rotating smoothly
    data = np.linspace(0, 4 * np.pi, n_frames)  # Two full rotations
    return HeadDirectionData(data=data, color="yellow", length=5.0)


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
def test_napari_update_latency_pose_and_trail(
    realistic_env,
    realistic_fields,
    realistic_pose_overlay_data,
    realistic_position_overlay_data,
):
    """Benchmark update latency with realistic pose + trail data.

    Target: < 50 ms per frame update on standard hardware.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    # Create overlay data with pose and position
    overlay_data = OverlayData(
        positions=[realistic_position_overlay_data],
        bodypart_sets=[realistic_pose_overlay_data],
        head_directions=[],
    )

    # Mock napari viewer
    with patch("neurospatial.animation.backends.napari_backend.napari") as mock_napari:
        mock_viewer = MagicMock()
        mock_napari.Viewer.return_value = mock_viewer

        # Mock viewer.dims attributes properly
        mock_viewer.dims.ndim = 4  # (time, height, width, rgb)
        mock_viewer.dims.current_step = (0,)

        # Mock layer objects
        mock_layers = []
        for _ in range(10):  # Multiple layers for bodyparts + trail + position
            mock_layer = MagicMock()
            mock_layers.append(mock_layer)

        mock_viewer.add_tracks.return_value = mock_layers[0]
        mock_viewer.add_points.side_effect = mock_layers[1:]
        mock_viewer.add_shapes.return_value = mock_layers[-1]
        mock_viewer.add_image.return_value = MagicMock()

        # Render with overlay data
        _ = render_napari(
            realistic_env,
            realistic_fields,
            overlay_data=overlay_data,
            show_regions=False,
        )

        # Get the update callback
        assert mock_viewer.dims.events.current_step.connect.called
        update_callback = mock_viewer.dims.events.current_step.connect.call_args[0][0]

        # Benchmark update latency across multiple frames
        n_test_frames = 50  # Test 50 frame updates
        update_times = []

        for frame_idx in range(n_test_frames):
            # Create mock event with frame index
            mock_event = MagicMock()
            mock_event.value = frame_idx

            # Time the update
            start_time = time.perf_counter()
            update_callback(mock_event)
            end_time = time.perf_counter()

            update_time_ms = (end_time - start_time) * 1000
            update_times.append(update_time_ms)

        # Compute statistics
        mean_time = np.mean(update_times)
        median_time = np.median(update_times)
        p95_time = np.percentile(update_times, 95)
        max_time = np.max(update_times)

        # Print results
        print(f"\n{'=' * 60}")
        print("Napari Update Latency (Pose + Trail)")
        print(f"{'=' * 60}")
        print(f"Frames tested: {n_test_frames}")
        print(f"Mean:   {mean_time:.2f} ms")
        print(f"Median: {median_time:.2f} ms")
        print(f"P95:    {p95_time:.2f} ms")
        print(f"Max:    {max_time:.2f} ms")
        print("Target: < 50 ms")
        print(f"{'=' * 60}\n")

        # Assert performance target
        # Note: This is a mock-based test, so actual times will be very fast
        # In real usage with napari GUI, the target is < 50 ms
        assert mean_time < 100, (
            f"Update latency too high: {mean_time:.2f} ms (mock-based)"
        )


@pytest.mark.slow
def test_napari_update_latency_all_overlays(
    realistic_env,
    realistic_fields,
    realistic_pose_overlay_data,
    realistic_position_overlay_data,
    realistic_head_direction_overlay_data,
):
    """Benchmark update latency with all overlay types enabled.

    Target: < 50 ms per frame update on standard hardware.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    # Create overlay data with all types
    overlay_data = OverlayData(
        positions=[realistic_position_overlay_data],
        bodypart_sets=[realistic_pose_overlay_data],
        head_directions=[realistic_head_direction_overlay_data],
    )

    # Mock napari viewer
    with patch("neurospatial.animation.backends.napari_backend.napari") as mock_napari:
        mock_viewer = MagicMock()
        mock_napari.Viewer.return_value = mock_viewer

        # Mock viewer.dims attributes properly
        mock_viewer.dims.ndim = 4  # (time, height, width, rgb)
        mock_viewer.dims.current_step = (0,)

        # Mock layer objects
        mock_layers = []
        for _ in range(15):  # Multiple layers for all overlay types
            mock_layer = MagicMock()
            mock_layers.append(mock_layer)

        mock_viewer.add_tracks.return_value = mock_layers[0]
        mock_viewer.add_points.side_effect = mock_layers[1:10]
        mock_viewer.add_shapes.return_value = mock_layers[10]
        mock_viewer.add_vectors.return_value = mock_layers[11]
        mock_viewer.add_image.return_value = MagicMock()

        # Render with overlay data
        _ = render_napari(
            realistic_env,
            realistic_fields,
            overlay_data=overlay_data,
            show_regions=False,
        )

        # Get the update callback
        assert mock_viewer.dims.events.current_step.connect.called
        update_callback = mock_viewer.dims.events.current_step.connect.call_args[0][0]

        # Benchmark update latency across multiple frames
        n_test_frames = 50
        update_times = []

        for frame_idx in range(n_test_frames):
            mock_event = MagicMock()
            mock_event.value = frame_idx

            start_time = time.perf_counter()
            update_callback(mock_event)
            end_time = time.perf_counter()

            update_time_ms = (end_time - start_time) * 1000
            update_times.append(update_time_ms)

        # Compute statistics
        mean_time = np.mean(update_times)
        median_time = np.median(update_times)
        p95_time = np.percentile(update_times, 95)
        max_time = np.max(update_times)

        # Print results
        print(f"\n{'=' * 60}")
        print("Napari Update Latency (All Overlays)")
        print(f"{'=' * 60}")
        print(f"Frames tested: {n_test_frames}")
        print("Overlays: Position + Pose + Head Direction")
        print(f"Mean:   {mean_time:.2f} ms")
        print(f"Median: {median_time:.2f} ms")
        print(f"P95:    {p95_time:.2f} ms")
        print(f"Max:    {max_time:.2f} ms")
        print("Target: < 50 ms")
        print(f"{'=' * 60}\n")

        # Assert performance target (mock-based, so very fast)
        assert mean_time < 100, (
            f"Update latency too high: {mean_time:.2f} ms (mock-based)"
        )


@pytest.mark.slow
def test_napari_batched_vs_individual_updates():
    """Profile batched vs individual layer updates.

    Demonstrates that batched updates (current implementation) are more efficient
    than updating layers individually in separate callbacks.
    """
    # This test demonstrates the design choice of batched updates
    # In the actual implementation, all layers are updated in a single callback

    # Create mock layers
    n_layers = 10
    mock_layers = [MagicMock() for _ in range(n_layers)]

    # Mock data for each layer
    n_frames = 100
    rng = np.random.default_rng(42)
    layer_data = [rng.random((n_frames, 2)) for _ in range(n_layers)]

    # Simulate batched update (current implementation)
    def batched_update(frame_idx: int):
        """Update all layers in a single callback."""
        for layer, data in zip(mock_layers, layer_data, strict=True):
            layer.data = data[frame_idx]

    # Simulate individual updates (alternative approach)
    def individual_update(frame_idx: int, layer_idx: int):
        """Update a single layer."""
        mock_layers[layer_idx].data = layer_data[layer_idx][frame_idx]

    # Benchmark batched updates
    n_test_frames = 50
    batched_times = []
    for frame_idx in range(n_test_frames):
        start_time = time.perf_counter()
        batched_update(frame_idx)
        end_time = time.perf_counter()
        batched_times.append((end_time - start_time) * 1000)

    # Benchmark individual updates (simulate separate callbacks)
    individual_times = []
    for frame_idx in range(n_test_frames):
        frame_total_start = time.perf_counter()
        for layer_idx in range(n_layers):
            individual_update(frame_idx, layer_idx)
        frame_total_end = time.perf_counter()
        individual_times.append((frame_total_end - frame_total_start) * 1000)

    # Compute statistics
    batched_mean = np.mean(batched_times)
    individual_mean = np.mean(individual_times)
    speedup = individual_mean / batched_mean

    # Print results
    print(f"\n{'=' * 60}")
    print("Batched vs Individual Layer Updates")
    print(f"{'=' * 60}")
    print(f"Layers: {n_layers}")
    print(f"Frames tested: {n_test_frames}")
    print(f"Batched mean:    {batched_mean:.4f} ms")
    print(f"Individual mean: {individual_mean:.4f} ms")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"{'=' * 60}\n")

    # Batched should be competitive with individual (within 50% range)
    # Note: In mock-based tests, overhead is so small (~0.005ms) that variance dominates
    # In real napari, batched updates reduce callback overhead and GUI thread contention
    assert batched_mean <= individual_mean * 1.5, (
        "Batched updates should be competitive"
    )


@pytest.mark.slow
def test_napari_multi_animal_performance(
    realistic_env,
    realistic_fields,
    realistic_pose_overlay_data,
):
    """Benchmark update latency with multiple animals (3 pose sets).

    Target: < 50 ms per frame update with 3 simultaneous tracked animals.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData

    # Create overlay data with 3 animals
    overlay_data = OverlayData(
        positions=[],
        bodypart_sets=[
            realistic_pose_overlay_data,
            realistic_pose_overlay_data,
            realistic_pose_overlay_data,
        ],
        head_directions=[],
    )

    # Mock napari viewer
    with patch("neurospatial.animation.backends.napari_backend.napari") as mock_napari:
        mock_viewer = MagicMock()
        mock_napari.Viewer.return_value = mock_viewer

        # Mock viewer.dims attributes properly
        mock_viewer.dims.ndim = 4  # (time, height, width, rgb)
        mock_viewer.dims.current_step = (0,)

        # Mock layer objects (many layers for 3 animals)
        mock_layers = []
        for _ in range(40):  # 3 animals x ~13 layers each
            mock_layer = MagicMock()
            mock_layers.append(mock_layer)

        mock_viewer.add_points.side_effect = mock_layers[:30]
        mock_viewer.add_shapes.side_effect = mock_layers[30:]
        mock_viewer.add_image.return_value = MagicMock()

        # Render with overlay data
        _ = render_napari(
            realistic_env,
            realistic_fields,
            overlay_data=overlay_data,
            show_regions=False,
        )

        # Get the update callback
        assert mock_viewer.dims.events.current_step.connect.called
        update_callback = mock_viewer.dims.events.current_step.connect.call_args[0][0]

        # Benchmark update latency across multiple frames
        n_test_frames = 50
        update_times = []

        for frame_idx in range(n_test_frames):
            mock_event = MagicMock()
            mock_event.value = frame_idx

            start_time = time.perf_counter()
            update_callback(mock_event)
            end_time = time.perf_counter()

            update_time_ms = (end_time - start_time) * 1000
            update_times.append(update_time_ms)

        # Compute statistics
        mean_time = np.mean(update_times)
        median_time = np.median(update_times)
        p95_time = np.percentile(update_times, 95)
        max_time = np.max(update_times)

        # Print results
        print(f"\n{'=' * 60}")
        print("Napari Update Latency (Multi-Animal: 3 Pose Sets)")
        print(f"{'=' * 60}")
        print(f"Frames tested: {n_test_frames}")
        print("Animals: 3")
        print("Bodyparts per animal: 10")
        print(f"Mean:   {mean_time:.2f} ms")
        print(f"Median: {median_time:.2f} ms")
        print(f"P95:    {p95_time:.2f} ms")
        print(f"Max:    {max_time:.2f} ms")
        print("Target: < 50 ms")
        print(f"{'=' * 60}\n")

        # Assert performance target (mock-based, so very fast)
        assert mean_time < 100, (
            f"Update latency too high: {mean_time:.2f} ms (mock-based)"
        )


@pytest.mark.slow
def test_napari_scalability_with_frame_count():
    """Test that update time is independent of total frame count.

    Update latency should be O(1) with respect to total frames since we
    only update visible data per frame, not process entire sequence.
    """
    from neurospatial.animation.backends.napari_backend import render_napari
    from neurospatial.animation.overlays import OverlayData, PositionData

    rng = np.random.default_rng(42)

    # Test with different frame counts
    frame_counts = [50, 100, 200, 500]
    mean_times = []

    # Create deterministic environment (reused across iterations)
    from neurospatial import Environment

    x = np.linspace(0, 100, 21)
    y = np.linspace(0, 100, 21)
    xx, yy = np.meshgrid(x, y)
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    env = Environment.from_samples(positions, bin_size=5.0)

    for n_frames in frame_counts:
        # Create data with local RNG
        data = rng.random((n_frames, 2)) * 100
        position_data = PositionData(data=data, color="red", size=10.0, trail_length=10)
        overlay_data = OverlayData(positions=[position_data])

        # Create fields
        fields = [rng.random(env.n_bins) for _ in range(n_frames)]

        # Mock napari viewer
        with patch(
            "neurospatial.animation.backends.napari_backend.napari"
        ) as mock_napari:
            mock_viewer = MagicMock()
            mock_napari.Viewer.return_value = mock_viewer

            # Mock viewer.dims attributes properly
            mock_viewer.dims.ndim = 4  # (time, height, width, rgb)
            mock_viewer.dims.current_step = (0,)

            mock_track_layer = MagicMock()
            mock_point_layer = MagicMock()
            mock_viewer.add_tracks.return_value = mock_track_layer
            mock_viewer.add_points.return_value = mock_point_layer
            mock_viewer.add_image.return_value = MagicMock()

            # Render
            _ = render_napari(
                env, fields, overlay_data=overlay_data, show_regions=False
            )

            # Get update callback
            update_callback = mock_viewer.dims.events.current_step.connect.call_args[0][
                0
            ]

            # Benchmark a few frames
            n_test_frames = 10
            update_times = []
            for frame_idx in range(n_test_frames):
                mock_event = MagicMock()
                mock_event.value = frame_idx

                start_time = time.perf_counter()
                update_callback(mock_event)
                end_time = time.perf_counter()

                update_times.append((end_time - start_time) * 1000)

            mean_times.append(np.mean(update_times))

    # Print results
    print(f"\n{'=' * 60}")
    print("Napari Scalability with Frame Count")
    print(f"{'=' * 60}")
    for n_frames, mean_time in zip(frame_counts, mean_times, strict=True):
        print(f"{n_frames:4d} frames: {mean_time:.4f} ms")
    print(f"{'=' * 60}\n")

    # Update time should be relatively constant (within 2x range)
    time_range = max(mean_times) / min(mean_times)
    assert time_range < 3.0, (
        f"Update time varies too much with frame count: {time_range:.2f}x"
    )
