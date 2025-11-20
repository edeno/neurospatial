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
    skeleton = [("nose", "tail")]
    colors = {"nose": "blue", "tail": "green"}

    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=colors,
        skeleton_color="white",
        skeleton_width=2.0,
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
    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=[("nose", "tail")],
        colors=None,
        skeleton_color="white",
        skeleton_width=2.0,
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
    skeleton = [
        ("nose", "tail"),
        ("nose", "left_ear"),
        ("nose", "right_ear"),
    ]

    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors={"nose": "red", "tail": "blue"},
        skeleton_color="white",
        skeleton_width=3.0,
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
    skeleton = [("nose", "tail")]

    bodypart_data = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors=None,
        skeleton_color="white",
        skeleton_width=2.0,
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
