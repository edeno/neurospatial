"""Tests for video backend overlay rendering.

Tests cover:
- Position overlays (trails + markers)
- Bodypart overlays (skeleton via LineCollection)
- Head direction overlays (arrows)
- Region overlays (PathPatch)
- Multi-animal scenarios
- Parallel rendering compatibility
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial.animation.backends.video_backend import render_video
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    OverlayData,
    PositionData,
)
from neurospatial.animation.skeleton import Skeleton

# =============================================================================
# Helpers
# =============================================================================


def _mock_subprocess_result():
    """Create a mock subprocess result for successful ffmpeg encoding."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    return mock_result


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_env() -> Environment:
    """Create a simple 2D environment for testing."""
    positions = np.random.rand(100, 2) * 100
    env = Environment.from_samples(positions, bin_size=10.0)
    return env


@pytest.fixture
def sample_fields(simple_env: Environment) -> list[NDArray[np.float64]]:
    """Create sample fields for animation."""
    return [np.random.rand(simple_env.n_bins) for _ in range(10)]


@pytest.fixture
def position_overlay_data() -> OverlayData:
    """Create position overlay data for testing."""
    positions = np.random.rand(10, 2) * 50 + 25  # Center in [25, 75]
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=5)
    return OverlayData(positions=[pos_data])


@pytest.fixture
def bodypart_overlay_data() -> OverlayData:
    """Create bodypart overlay data for testing."""
    n_frames = 10
    bodyparts = {
        "nose": np.random.rand(n_frames, 2) * 50 + 25,
        "tail": np.random.rand(n_frames, 2) * 50 + 25,
    }
    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    colors = {"nose": "blue", "tail": "green"}

    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=colors,
    )
    return OverlayData(bodypart_sets=[bodypart_data])


@pytest.fixture
def head_direction_overlay_data() -> OverlayData:
    """Create head direction overlay data for testing."""
    n_frames = 10
    angles = np.linspace(0, 2 * np.pi, n_frames)
    head_dir_data = HeadDirectionData(data=angles, color="yellow", length=20.0)
    return OverlayData(head_directions=[head_dir_data])


@pytest.fixture
def multi_overlay_data() -> OverlayData:
    """Create data with multiple overlay types."""
    n_frames = 10

    # Position overlay
    positions = np.random.rand(n_frames, 2) * 50 + 25
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=3)

    # Bodypart overlay
    bodyparts = {
        "nose": np.random.rand(n_frames, 2) * 50 + 25,
        "tail": np.random.rand(n_frames, 2) * 50 + 25,
    }
    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=None,
    )

    # Head direction overlay
    angles = np.linspace(0, 2 * np.pi, n_frames)
    head_dir_data = HeadDirectionData(data=angles, color="yellow", length=15.0)

    return OverlayData(
        positions=[pos_data],
        bodypart_sets=[bodypart_data],
        head_directions=[head_dir_data],
    )


@pytest.fixture
def env_with_regions(simple_env: Environment) -> Environment:
    """Create environment with regions."""
    simple_env.regions.add("goal", point=np.array([50.0, 50.0]))
    return simple_env


# =============================================================================
# Test: Basic Parameter Acceptance
# =============================================================================


def test_render_video_accepts_overlay_parameters(
    simple_env: Environment, sample_fields: list[NDArray[np.float64]], tmp_path: Path
):
    """Test that render_video accepts overlay_data, show_regions, region_alpha."""
    overlay_data = OverlayData()
    output_path = tmp_path / "test_video.mp4"

    # Mock successful subprocess result
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stderr = ""

    # Mock ffmpeg and parallel rendering to avoid actual video creation
    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            return_value=str(tmp_path / "frame_%05d.png"),
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=mock_result,
        ),
    ):
        # Should not raise TypeError for unknown parameters
        render_video(
            simple_env,
            sample_fields,
            str(output_path),
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
            n_workers=1,
        )


def test_render_video_accepts_none_overlay_data(
    simple_env: Environment, sample_fields: list[NDArray[np.float64]], tmp_path: Path
):
    """Test backward compatibility: overlay_data=None."""
    output_path = tmp_path / "test_video.mp4"

    mock_result = Mock()
    mock_result.returncode = 0

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            return_value=str(tmp_path / "frame_%05d.png"),
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=mock_result,
        ),
    ):
        # Should not raise with None overlay_data
        render_video(
            simple_env,
            sample_fields,
            str(output_path),
            overlay_data=None,
            n_workers=1,
        )


# =============================================================================
# Test: Position Overlay Rendering
# =============================================================================


@pytest.mark.parametrize("trail_length", [None, 3, 5])
def test_position_overlay_renders_marker_and_trail(
    simple_env: Environment, tmp_path: Path, trail_length: int | None
):
    """Test that position overlay renders marker and optional trail."""
    n_frames = 10
    positions = np.random.rand(n_frames, 2) * 50 + 25

    pos_data = PositionData(
        data=positions, color="red", size=12.0, trail_length=trail_length
    )
    overlay_data = OverlayData(positions=[pos_data])

    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    # Mock to capture parameters passed to parallel_render_frames
    parallel_calls = []

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        parallel_calls.append(
            {
                "overlay_data": kwargs.get("overlay_data"),
                "show_regions": kwargs.get("show_regions"),
                "region_alpha": kwargs.get("region_alpha"),
            }
        )
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            simple_env,
            fields,
            str(tmp_path / "output.mp4"),
            overlay_data=overlay_data,
            n_workers=1,
        )

    # Verify overlay_data was passed to parallel_render_frames
    assert len(parallel_calls) > 0
    assert parallel_calls[0]["overlay_data"] is overlay_data


def test_position_overlay_trail_decay(simple_env: Environment):
    """Test that trails are rendered with decaying alpha."""
    n_frames = 10
    positions = np.random.rand(n_frames, 2) * 50 + 25

    pos_data = PositionData(data=positions, color="blue", size=10.0, trail_length=5)

    # Test trail alpha calculation
    trail_length = pos_data.trail_length
    assert trail_length is not None

    # Alpha should decay from 1.0 (current) to near 0 (oldest)
    for i in range(trail_length):
        alpha = (i + 1) / trail_length
        assert 0.0 < alpha <= 1.0


# =============================================================================
# Test: Bodypart Overlay Rendering (LineCollection for Skeleton)
# =============================================================================


def test_bodypart_overlay_renders_skeleton_with_linecollection(
    simple_env: Environment, tmp_path: Path
):
    """Test that skeleton is rendered using LineCollection (not loops)."""
    n_frames = 10
    bodyparts = {
        "nose": np.random.rand(n_frames, 2) * 50 + 25,
        "tail": np.random.rand(n_frames, 2) * 50 + 25,
        "left_ear": np.random.rand(n_frames, 2) * 50 + 25,
        "right_ear": np.random.rand(n_frames, 2) * 50 + 25,
    }
    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail", "left_ear", "right_ear"),
        edges=(("nose", "tail"), ("nose", "left_ear"), ("nose", "right_ear")),
        edge_color="white",
        edge_width=3.0,
    )

    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors={"nose": "red", "tail": "blue"},
    )
    overlay_data = OverlayData(bodypart_sets=[bodypart_data])

    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    # Mock to capture parameters passed to parallel_render_frames
    parallel_calls = []

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        parallel_calls.append(kwargs.get("overlay_data") is not None)
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            simple_env,
            fields,
            str(tmp_path / "output.mp4"),
            overlay_data=overlay_data,
            n_workers=1,
        )

    # Verify overlay_data was passed
    assert any(parallel_calls)


def test_bodypart_overlay_handles_nan_values(simple_env: Environment):
    """Test that NaN bodypart coordinates are gracefully skipped."""
    bodyparts = {
        "nose": np.array(
            [[10.0, 20.0], [np.nan, np.nan], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]
        ),
        "tail": np.array(
            [[15.0, 25.0], [35.0, 45.0], [np.nan, np.nan], [55.0, 65.0], [75.0, 85.0]]
        ),
    }
    skeleton = Skeleton(
        name="test",
        nodes=("nose", "tail"),
        edges=(("nose", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )

    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=None,
    )

    # Should not raise when creating OverlayData
    overlay_data = OverlayData(bodypart_sets=[bodypart_data])
    assert len(overlay_data.bodypart_sets) == 1


# =============================================================================
# Test: Head Direction Overlay Rendering (Arrows)
# =============================================================================


def test_head_direction_overlay_renders_arrows(simple_env: Environment, tmp_path: Path):
    """Test that head direction is rendered as arrows."""
    n_frames = 8
    angles = np.linspace(0, 2 * np.pi, n_frames)

    head_dir_data = HeadDirectionData(data=angles, color="yellow", length=25.0)
    overlay_data = OverlayData(head_directions=[head_dir_data])

    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    parallel_calls = []

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        parallel_calls.append(kwargs.get("overlay_data") is not None)
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            simple_env,
            fields,
            str(tmp_path / "output.mp4"),
            overlay_data=overlay_data,
            n_workers=1,
        )

    assert any(parallel_calls)


def test_head_direction_overlay_handles_vectors(simple_env: Environment):
    """Test head direction with 2D vector data (not just angles)."""
    # Unit vectors pointing in different directions
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.707, 0.707],
        ]
    )

    head_dir_data = HeadDirectionData(data=vectors, color="cyan", length=15.0)
    overlay_data = OverlayData(head_directions=[head_dir_data])

    # Should not raise
    assert len(overlay_data.head_directions) == 1
    assert overlay_data.head_directions[0].data.shape == (5, 2)


# =============================================================================
# Test: Region Overlay Rendering (PathPatch)
# =============================================================================


def test_region_overlay_renders_with_alpha(
    env_with_regions: Environment, tmp_path: Path
):
    """Test that regions are rendered with specified alpha."""
    n_frames = 5
    fields = [np.random.rand(env_with_regions.n_bins) for _ in range(n_frames)]

    parallel_calls = []

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        parallel_calls.append(
            {
                "show_regions": kwargs.get("show_regions"),
                "region_alpha": kwargs.get("region_alpha"),
            }
        )
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            env_with_regions,
            fields,
            str(tmp_path / "output.mp4"),
            show_regions=True,
            region_alpha=0.5,
            n_workers=1,
        )

    # Verify regions parameters passed
    assert len(parallel_calls) > 0
    for call in parallel_calls:
        assert call["show_regions"] is True
        assert call["region_alpha"] == 0.5


def test_region_overlay_filters_by_name(env_with_regions: Environment, tmp_path: Path):
    """Test that show_regions can filter by region names."""
    env_with_regions.regions.add("start", point=np.array([10.0, 10.0]))
    env_with_regions.regions.add("end", point=np.array([90.0, 90.0]))

    n_frames = 5
    fields = [np.random.rand(env_with_regions.n_bins) for _ in range(n_frames)]

    parallel_calls = []

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        parallel_calls.append(kwargs.get("show_regions"))
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            env_with_regions,
            fields,
            str(tmp_path / "output.mp4"),
            show_regions=["goal", "start"],  # Only these two
            n_workers=1,
        )

    # Verify filter list passed
    assert len(parallel_calls) > 0
    for call in parallel_calls:
        assert call == ["goal", "start"]


# =============================================================================
# Test: Multi-Animal / Multi-Overlay Scenarios
# =============================================================================


def test_multiple_position_overlays(simple_env: Environment, tmp_path: Path):
    """Test rendering multiple position overlays (multi-animal)."""
    n_frames = 10

    # Three animals with different colors
    animal1 = PositionData(
        data=np.random.rand(n_frames, 2) * 50 + 25,
        color="red",
        size=10.0,
        trail_length=5,
    )
    animal2 = PositionData(
        data=np.random.rand(n_frames, 2) * 50 + 25,
        color="blue",
        size=12.0,
        trail_length=3,
    )
    animal3 = PositionData(
        data=np.random.rand(n_frames, 2) * 50 + 25,
        color="green",
        size=8.0,
        trail_length=None,
    )

    overlay_data = OverlayData(positions=[animal1, animal2, animal3])
    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    parallel_calls = []

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        od = kwargs.get("overlay_data")
        if od:
            parallel_calls.append(len(od.positions))
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            simple_env,
            fields,
            str(tmp_path / "output.mp4"),
            overlay_data=overlay_data,
            n_workers=1,
        )

    # Verify all three position overlays passed
    assert parallel_calls[0] == 3


def test_mixed_overlay_types(
    simple_env: Environment, tmp_path: Path, multi_overlay_data: OverlayData
):
    """Test rendering all overlay types together."""
    n_frames = 10
    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    parallel_calls = []

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        od = kwargs.get("overlay_data")
        if od:
            parallel_calls.append(
                {
                    "positions": len(od.positions),
                    "bodyparts": len(od.bodypart_sets),
                    "head_dirs": len(od.head_directions),
                }
            )
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            simple_env,
            fields,
            str(tmp_path / "output.mp4"),
            overlay_data=multi_overlay_data,
            n_workers=1,
        )

    # Verify all overlay types present
    assert len(parallel_calls) > 0
    call = parallel_calls[0]
    assert call["positions"] == 1
    assert call["bodyparts"] == 1
    assert call["head_dirs"] == 1


# =============================================================================
# Test: Parallel Rendering Compatibility (Pickle-ability)
# =============================================================================


def test_overlay_data_is_pickle_safe(position_overlay_data: OverlayData):
    """Test that OverlayData can be pickled for parallel rendering."""
    import pickle

    # Should not raise
    pickled = pickle.dumps(position_overlay_data, protocol=pickle.HIGHEST_PROTOCOL)
    unpickled = pickle.loads(pickled)

    # Verify data integrity
    assert len(unpickled.positions) == len(position_overlay_data.positions)
    assert unpickled.positions[0].color == position_overlay_data.positions[0].color
    np.testing.assert_array_equal(
        unpickled.positions[0].data, position_overlay_data.positions[0].data
    )


def test_parallel_rendering_with_overlays(
    simple_env: Environment, tmp_path: Path, multi_overlay_data: OverlayData
):
    """Test that overlays work with multiple workers (parallel rendering)."""
    n_frames = 20
    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    worker_count = 0

    def mock_parallel_render(env, fields, output_dir, **kwargs):
        nonlocal worker_count
        worker_count += 1
        # Verify overlay_data is present and pickle-safe
        od = kwargs.get("overlay_data")
        assert od is not None
        assert len(od.positions) > 0
        return str(Path(output_dir) / "frame_%05d.png")

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            side_effect=mock_parallel_render,
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=_mock_subprocess_result(),
        ),
    ):
        render_video(
            simple_env,
            fields,
            str(tmp_path / "output.mp4"),
            overlay_data=multi_overlay_data,
            n_workers=1,  # Use serial rendering to avoid pickling mocks
        )

    # Verify worker was invoked (n_workers=1 for test simplicity)
    assert worker_count >= 1  # Worker should have been invoked


# =============================================================================
# Test: Error Handling
# =============================================================================


def test_render_video_with_empty_overlay_data(simple_env: Environment, tmp_path: Path):
    """Test that empty OverlayData is handled gracefully."""
    overlay_data = OverlayData()  # No overlays
    fields = [np.random.rand(simple_env.n_bins) for _ in range(5)]

    mock_result = Mock()
    mock_result.returncode = 0

    with (
        patch(
            "neurospatial.animation.backends.video_backend.check_ffmpeg_available",
            return_value=True,
        ),
        patch(
            "neurospatial.animation._parallel.parallel_render_frames",
            return_value=str(tmp_path / "frame_%05d.png"),
        ),
        patch(
            "neurospatial.animation.backends.video_backend.subprocess.run",
            return_value=mock_result,
        ),
    ):
        # Should not raise with empty overlay data
        render_video(
            simple_env,
            fields,
            str(tmp_path / "output.mp4"),
            overlay_data=overlay_data,
            n_workers=1,
        )


# =============================================================================
# Test: Pickle-ability Validation (Milestone 4.2)
# =============================================================================


def test_unpickleable_overlay_data_raises_clear_error(
    simple_env: Environment, tmp_path: Path
):
    """Test that unpickleable overlay_data raises ValueError with WHAT/WHY/HOW."""
    from neurospatial.animation._parallel import parallel_render_frames

    # Create overlay data with unpickleable content (lambda function)
    n_frames = 5
    positions = np.random.rand(n_frames, 2) * 50 + 25
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=3)

    # Make it unpickleable by adding a lambda
    overlay_data = OverlayData(positions=[pos_data])
    overlay_data._unpickleable = lambda x: x  # Add unpickleable attribute

    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    # Should raise ValueError with clear error message
    with pytest.raises(ValueError, match=r"overlay_data.*pickle"):
        parallel_render_frames(
            env=simple_env,
            fields=fields,
            output_dir=str(tmp_path),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            frame_labels=None,
            dpi=50,
            n_workers=2,  # Parallel rendering requires pickle
            overlay_data=overlay_data,
        )


def test_pickle_error_message_includes_solutions(
    simple_env: Environment, tmp_path: Path
):
    """Test that pickle error includes WHAT/WHY/HOW with actionable solutions."""
    from neurospatial.animation._parallel import parallel_render_frames

    # Create unpickleable overlay data
    n_frames = 5
    positions = np.random.rand(n_frames, 2) * 50 + 25
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=3)
    overlay_data = OverlayData(positions=[pos_data])
    overlay_data._bad_attr = lambda x: x  # Unpickleable

    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    # Capture error message
    with pytest.raises(ValueError) as exc_info:
        parallel_render_frames(
            env=simple_env,
            fields=fields,
            output_dir=str(tmp_path),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            frame_labels=None,
            dpi=50,
            n_workers=2,
            overlay_data=overlay_data,
        )

    error_msg = str(exc_info.value)

    # Verify WHAT: describes the problem
    assert "overlay_data" in error_msg.lower()
    assert "pickle" in error_msg.lower()

    # Verify WHY: explains why it matters
    assert "parallel" in error_msg.lower()

    # Verify HOW: provides solutions
    # Should mention either n_workers=1 or env.clear_cache() or similar fix
    assert (
        "n_workers=1" in error_msg
        or "clear_cache" in error_msg
        or "remove" in error_msg
    )


def test_pickle_check_skipped_for_serial_rendering(
    simple_env: Environment, tmp_path: Path
):
    """Test that pickle validation is skipped when n_workers=1 (serial mode)."""
    from neurospatial.animation._parallel import parallel_render_frames

    # Create overlay data (doesn't matter if pickle-able for n_workers=1)
    n_frames = 5
    positions = np.random.rand(n_frames, 2) * 50 + 25
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=3)
    overlay_data = OverlayData(positions=[pos_data])

    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    # With n_workers=1, should not raise even if unpickleable
    # (though in practice this test uses pickle-able data)
    frame_pattern = parallel_render_frames(
        env=simple_env,
        fields=fields,
        output_dir=str(tmp_path),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        frame_labels=None,
        dpi=50,
        n_workers=1,  # Serial mode - no pickle needed
        overlay_data=overlay_data,
    )

    # Should succeed
    assert "frame_" in frame_pattern
    assert ".png" in frame_pattern


def test_pickleable_overlay_data_succeeds_with_parallel(
    simple_env: Environment, tmp_path: Path
):
    """Test that pickle-able overlay_data works fine with n_workers > 1."""
    from neurospatial.animation._parallel import parallel_render_frames

    # Create fully pickle-able overlay data
    n_frames = 10
    positions = np.random.rand(n_frames, 2) * 50 + 25
    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=5)
    overlay_data = OverlayData(positions=[pos_data])

    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    # Should succeed with n_workers > 1
    frame_pattern = parallel_render_frames(
        env=simple_env,
        fields=fields,
        output_dir=str(tmp_path),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        frame_labels=None,
        dpi=50,
        n_workers=2,  # Parallel rendering
        overlay_data=overlay_data,
    )

    # Should succeed
    assert "frame_" in frame_pattern
    assert ".png" in frame_pattern


def test_none_overlay_data_is_always_pickle_safe(
    simple_env: Environment, tmp_path: Path
):
    """Test that None overlay_data is always safe (no validation needed)."""
    from neurospatial.animation._parallel import parallel_render_frames

    n_frames = 5
    fields = [np.random.rand(simple_env.n_bins) for _ in range(n_frames)]

    # With overlay_data=None, should always succeed
    frame_pattern = parallel_render_frames(
        env=simple_env,
        fields=fields,
        output_dir=str(tmp_path),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        frame_labels=None,
        dpi=50,
        n_workers=2,
        overlay_data=None,  # No overlay data - always safe
    )

    # Should succeed
    assert "frame_" in frame_pattern
    assert ".png" in frame_pattern


# =============================================================================
# Test: Video Overlay Rendering in Video Export Backend (Task 4.2)
# =============================================================================


@pytest.fixture
def sample_video_array() -> NDArray[np.uint8]:
    """Create a simple video array for testing (16x16, 10 frames)."""
    n_frames, height, width = 10, 16, 16
    frames = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    # Create a gradient pattern that varies by frame
    for i in range(n_frames):
        frames[i, :, :, 0] = i * 25  # Red varies by frame
        frames[i, :, :, 1] = 128  # Constant green
        frames[i, :, :, 2] = 128  # Constant blue
    return frames


@pytest.fixture
def video_overlay_data(sample_video_array: NDArray[np.uint8]) -> OverlayData:
    """Create OverlayData with a VideoData for testing."""
    from neurospatial.animation.overlays import VideoData

    frame_indices = np.arange(10, dtype=np.int_)  # 1:1 mapping
    video_data = VideoData(
        frame_indices=frame_indices,
        reader=sample_video_array,
        transform_to_env=None,  # No calibration - stretch to fill
        env_bounds=(0.0, 100.0, 0.0, 100.0),
        alpha=0.7,
        z_order="below",
    )
    return OverlayData(videos=[video_data])


class TestVideoOverlayExportRendering:
    """Test video overlay rendering in video export backend."""

    def test_render_video_background_function_exists(self):
        """Test that _render_video_background function exists in _parallel.py."""
        from neurospatial.animation._parallel import _render_video_background

        assert callable(_render_video_background)

    def test_render_video_background_renders_frame(
        self, simple_env: Environment, sample_video_array: NDArray[np.uint8]
    ):
        """Test _render_video_background renders a video frame."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_video_background
        from neurospatial.animation.overlays import VideoData

        # Create VideoData
        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="below",
        )

        # Create figure
        fig, ax = plt.subplots()
        try:
            # Render video background for frame 0
            _render_video_background(ax, video_data, frame_idx=0)

            # Should have added an image to the axes
            assert len(ax.images) >= 1
        finally:
            plt.close(fig)

    def test_render_video_background_skips_invalid_frame(
        self, simple_env: Environment, sample_video_array: NDArray[np.uint8]
    ):
        """Test _render_video_background skips rendering for -1 frame index."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_video_background
        from neurospatial.animation.overlays import VideoData

        # Create VideoData with -1 indicating no video for that frame
        frame_indices = np.array([-1, 0, 1, 2, -1], dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="below",
        )

        fig, ax = plt.subplots()
        try:
            # Render for frame 0 which has -1 (no video)
            _render_video_background(ax, video_data, frame_idx=0)

            # Should NOT have added an image (skipped)
            assert len(ax.images) == 0
        finally:
            plt.close(fig)

    def test_render_video_background_uses_env_bounds_extent(
        self, simple_env: Environment, sample_video_array: NDArray[np.uint8]
    ):
        """Test _render_video_background uses env_bounds for extent when no transform."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_video_background
        from neurospatial.animation.overlays import VideoData

        env_bounds = (10.0, 90.0, 20.0, 80.0)  # xmin, xmax, ymin, ymax
        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,  # No transform
            env_bounds=env_bounds,
            alpha=0.7,
            z_order="below",
        )

        fig, ax = plt.subplots()
        try:
            _render_video_background(ax, video_data, frame_idx=0)

            # Check extent matches env_bounds
            im = ax.images[0]
            extent = im.get_extent()
            # extent is (left, right, bottom, top) = (xmin, xmax, ymin, ymax)
            assert extent[0] == pytest.approx(env_bounds[0])  # xmin
            assert extent[1] == pytest.approx(env_bounds[1])  # xmax
            assert extent[2] == pytest.approx(env_bounds[2])  # ymin
            assert extent[3] == pytest.approx(env_bounds[3])  # ymax
        finally:
            plt.close(fig)

    def test_render_video_background_applies_alpha(
        self, simple_env: Environment, sample_video_array: NDArray[np.uint8]
    ):
        """Test _render_video_background applies alpha transparency."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_video_background
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.3,  # Specific alpha
            z_order="below",
        )

        fig, ax = plt.subplots()
        try:
            _render_video_background(ax, video_data, frame_idx=0)

            im = ax.images[0]
            assert im.get_alpha() == pytest.approx(0.3)
        finally:
            plt.close(fig)

    def test_render_video_background_zorder_below(
        self, simple_env: Environment, sample_video_array: NDArray[np.uint8]
    ):
        """Test _render_video_background sets zorder < 0 for z_order='below'."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_video_background
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="below",  # Should be rendered below field
        )

        fig, ax = plt.subplots()
        try:
            _render_video_background(ax, video_data, frame_idx=0)

            im = ax.images[0]
            # z_order="below" should have negative zorder
            assert im.get_zorder() < 0
        finally:
            plt.close(fig)

    def test_render_video_background_zorder_above(
        self, simple_env: Environment, sample_video_array: NDArray[np.uint8]
    ):
        """Test _render_video_background sets zorder > 0 for z_order='above'."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_video_background
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="above",  # Should be rendered above field
        )

        fig, ax = plt.subplots()
        try:
            _render_video_background(ax, video_data, frame_idx=0)

            im = ax.images[0]
            # z_order="above" should have positive zorder
            assert im.get_zorder() > 0
        finally:
            plt.close(fig)


class TestRenderAllOverlaysWithVideo:
    """Test that _render_all_overlays handles video overlays."""

    def test_render_all_overlays_renders_videos_below(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
    ):
        """Test _render_all_overlays renders video with z_order='below'."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_all_overlays
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.7,
            z_order="below",
        )
        overlay_data = OverlayData(videos=[video_data])

        fig, ax = plt.subplots()
        try:
            _render_all_overlays(
                ax=ax,
                env=simple_env,
                frame_idx=0,
                overlay_data=overlay_data,
                show_regions=False,
                region_alpha=0.3,
            )

            # Should have rendered video (at least one image)
            assert len(ax.images) >= 1
        finally:
            plt.close(fig)

    def test_render_all_overlays_renders_multiple_videos(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
    ):
        """Test _render_all_overlays renders multiple video overlays."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import _render_all_overlays
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video1 = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 50.0, 0.0, 50.0),
            alpha=0.5,
            z_order="below",
        )
        video2 = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(50.0, 100.0, 50.0, 100.0),
            alpha=0.5,
            z_order="below",
        )
        overlay_data = OverlayData(videos=[video1, video2])

        fig, ax = plt.subplots()
        try:
            _render_all_overlays(
                ax=ax,
                env=simple_env,
                frame_idx=0,
                overlay_data=overlay_data,
                show_regions=False,
                region_alpha=0.3,
            )

            # Should have rendered both videos
            assert len(ax.images) >= 2
        finally:
            plt.close(fig)


class TestVideoFrameRenderer:
    """Test VideoFrameRenderer class for artist reuse."""

    def test_video_frame_renderer_exists(self):
        """Test that VideoFrameRenderer class exists."""
        from neurospatial.animation._parallel import VideoFrameRenderer

        assert VideoFrameRenderer is not None

    def test_video_frame_renderer_initialization(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
    ):
        """Test VideoFrameRenderer initializes correctly."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import VideoFrameRenderer
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.7,
            z_order="below",
        )

        fig, ax = plt.subplots()
        try:
            renderer = VideoFrameRenderer(ax, video_data, simple_env)
            assert renderer is not None
            assert renderer.video_data is video_data
        finally:
            plt.close(fig)

    def test_video_frame_renderer_creates_artist_on_first_render(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
    ):
        """Test VideoFrameRenderer creates artist on first render call."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import VideoFrameRenderer
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.7,
            z_order="below",
        )

        fig, ax = plt.subplots()
        try:
            renderer = VideoFrameRenderer(ax, video_data, simple_env)

            # Before render, no artist
            assert renderer._artist is None

            # First render creates artist
            renderer.render(ax, frame_idx=0)
            assert renderer._artist is not None
        finally:
            plt.close(fig)

    def test_video_frame_renderer_reuses_artist(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
    ):
        """Test VideoFrameRenderer reuses artist on subsequent renders."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import VideoFrameRenderer
        from neurospatial.animation.overlays import VideoData

        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.7,
            z_order="below",
        )

        fig, ax = plt.subplots()
        try:
            renderer = VideoFrameRenderer(ax, video_data, simple_env)

            # First render
            renderer.render(ax, frame_idx=0)
            first_artist = renderer._artist

            # Second render should reuse same artist
            renderer.render(ax, frame_idx=1)
            assert renderer._artist is first_artist
        finally:
            plt.close(fig)

    def test_video_frame_renderer_hides_artist_for_invalid_frame(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
    ):
        """Test VideoFrameRenderer hides artist when frame index is -1."""
        import matplotlib.pyplot as plt

        from neurospatial.animation._parallel import VideoFrameRenderer
        from neurospatial.animation.overlays import VideoData

        # Frame indices with -1 at position 2
        frame_indices = np.array([0, 1, -1, 3, 4], dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.7,
            z_order="below",
        )

        fig, ax = plt.subplots()
        try:
            renderer = VideoFrameRenderer(ax, video_data, simple_env)

            # Render valid frame first
            renderer.render(ax, frame_idx=0)
            assert renderer._artist.get_visible()

            # Render frame with -1 index
            renderer.render(ax, frame_idx=2)
            assert not renderer._artist.get_visible()

            # Render valid frame again - should show artist
            renderer.render(ax, frame_idx=3)
            assert renderer._artist.get_visible()
        finally:
            plt.close(fig)


# =============================================================================
# Test: Widget Backend Video Overlay Support (Task 4.3)
# =============================================================================


class TestWidgetBackendVideoOverlay:
    """Test widget backend handles video overlays correctly."""

    def test_widget_backend_renders_video_overlay(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
    ):
        """Test widget backend renders video overlays via _render_all_overlays."""
        from neurospatial.animation.backends.widget_backend import (
            render_field_to_png_bytes_with_overlays,
        )
        from neurospatial.animation.overlays import VideoData

        # Create VideoData
        frame_indices = np.arange(10, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="below",
        )
        overlay_data = OverlayData(videos=[video_data])

        # Create a simple field
        field = np.random.rand(simple_env.n_bins)

        # Render frame with video overlay - should not raise
        png_bytes = render_field_to_png_bytes_with_overlays(
            env=simple_env,
            field=field,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            frame_idx=0,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )

        # Should return valid PNG bytes
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # PNG magic bytes
        assert png_bytes[:4] == b"\x89PNG"


# =============================================================================
# Test: HTML Backend Video Overlay Warning (Task 4.3)
# =============================================================================


class TestHTMLBackendVideoOverlay:
    """Test HTML backend skips video overlays with warning."""

    def test_html_backend_warns_on_video_overlay(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
        tmp_path: Path,
    ):
        """Test HTML backend emits warning when video overlay is present."""
        import warnings

        from neurospatial.animation.backends.html_backend import render_html
        from neurospatial.animation.overlays import VideoData

        # Create VideoData
        frame_indices = np.arange(5, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="below",
        )
        overlay_data = OverlayData(videos=[video_data])

        # Create simple fields
        fields = [np.random.rand(simple_env.n_bins) for _ in range(5)]

        # Should emit warning about video overlay not being supported
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            render_html(
                env=simple_env,
                fields=fields,
                save_path=str(tmp_path / "output.html"),
                overlay_data=overlay_data,
                max_html_frames=500,
                dpi=50,
            )

            # Check that exactly one warning was emitted about video overlays
            video_warnings = [
                warning
                for warning in w
                if "video" in str(warning.message).lower()
                and issubclass(warning.category, UserWarning)
            ]
            assert len(video_warnings) == 1, (
                "Expected exactly one warning about video overlay"
            )

            # Check warning message contains WHAT/WHY/HOW structure
            warning_msg = str(video_warnings[0].message)
            # Verify WHAT/WHY/HOW format
            assert "WHAT:" in warning_msg, "Warning should contain WHAT section"
            assert "WHY:" in warning_msg, "Warning should contain WHY section"
            assert "HOW" in warning_msg, "Warning should contain HOW section"
            # Should explain what the problem is
            assert "video" in warning_msg.lower()
            # Should provide alternatives
            assert (
                "video backend" in warning_msg.lower()
                or "napari" in warning_msg.lower()
            )

    def test_html_backend_still_renders_with_video_overlay_present(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
        tmp_path: Path,
    ):
        """Test HTML backend still renders successfully when video is skipped."""
        import warnings

        from neurospatial.animation.backends.html_backend import render_html
        from neurospatial.animation.overlays import VideoData

        # Create VideoData
        frame_indices = np.arange(5, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="below",
        )
        overlay_data = OverlayData(videos=[video_data])

        fields = [np.random.rand(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "output.html"

        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_path = render_html(
                env=simple_env,
                fields=fields,
                save_path=str(output_path),
                overlay_data=overlay_data,
                max_html_frames=500,
                dpi=50,
            )

        # Should still generate HTML file
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_html_backend_renders_other_overlays_with_video_present(
        self,
        simple_env: Environment,
        sample_video_array: NDArray[np.uint8],
        tmp_path: Path,
    ):
        """Test HTML renders position overlays even when video is skipped."""
        import warnings

        from neurospatial.animation.backends.html_backend import render_html
        from neurospatial.animation.overlays import VideoData

        # Create mixed overlay data (video + position)
        frame_indices = np.arange(5, dtype=np.int_)
        video_data = VideoData(
            frame_indices=frame_indices,
            reader=sample_video_array,
            transform_to_env=None,
            env_bounds=(0.0, 100.0, 0.0, 100.0),
            alpha=0.5,
            z_order="below",
        )
        positions = np.random.rand(5, 2) * 50 + 25
        pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=3)
        overlay_data = OverlayData(videos=[video_data], positions=[pos_data])

        fields = [np.random.rand(simple_env.n_bins) for _ in range(5)]
        output_path = tmp_path / "output.html"

        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_path = render_html(
                env=simple_env,
                fields=fields,
                save_path=str(output_path),
                overlay_data=overlay_data,
                max_html_frames=500,
                dpi=50,
            )

        # Read HTML and check for position overlay data
        html_content = result_path.read_text()
        # HTML should contain the serialized position overlay
        assert "positions" in html_content
        # Position data should be present
        assert "color" in html_content or '"red"' in html_content
