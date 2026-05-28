"""Visual regression tests for overlay rendering.

These tests drive the production video renderer
(``neurospatial.animation.backends.video_backend.render_video``), read a frame
back with OpenCV, and compare it against a golden baseline via ``pytest-mpl``.
Rendering through the real backend (rather than a parallel matplotlib
reimplementation) means the baselines pin what users actually see.

Run baseline generation:
    uv run pytest tests/animation/test_overlay_visual_regression.py \
        --mpl-generate-path=tests/animation/baseline

Run comparison tests:
    uv run pytest tests/animation/test_overlay_visual_regression.py --mpl

Visual regression tests verify:
- Position overlays with trails render correctly
- Bodypart overlays with skeletons render correctly
- Head direction arrows render correctly
- Regions with alpha transparency render correctly
- Mixed overlay combinations render consistently
"""

from __future__ import annotations

import shutil

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from shapely.geometry import Polygon

from neurospatial import Environment
from neurospatial.animation.backends.video_backend import render_video
from neurospatial.animation.overlays import (
    BodypartData,
    HeadDirectionData,
    OverlayData,
    PositionData,
)
from neurospatial.animation.skeleton import Skeleton
from neurospatial.regions import Regions

# Use non-interactive backend for testing
matplotlib.use("Agg")

cv2 = pytest.importorskip("cv2")

HAS_FFMPEG = shutil.which("ffmpeg") is not None

# render_video raises RuntimeError without ffmpeg; gate the whole module.
pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


# =============================================================================
# Fixtures for Visual Regression Tests
# =============================================================================


@pytest.fixture
def env_2d():
    """Create a simple 2D environment for visual tests."""
    # Create a regular grid environment
    positions = np.array(
        [
            [10, 10],
            [10, 90],
            [90, 10],
            [90, 90],
            [50, 50],
        ]
    )
    env = Environment.from_samples(positions, bin_size=10.0)
    env.units = "cm"
    env.frame = "test"

    # Add regions for testing
    env.regions = Regions()
    env.regions.add("goal", point=np.array([80, 80]))
    env.regions.add(
        "arena",
        polygon=Polygon([[10, 10], [90, 10], [90, 90], [10, 90]]),
    )

    return env


@pytest.fixture
def simple_field_2d(env_2d):
    """Create a simple spatial field for the 2D environment."""
    rng = np.random.default_rng(42)
    # Create a Gaussian-like field centered in the environment
    n_bins = env_2d.n_bins
    field = rng.random(n_bins) * 0.5 + 0.2  # Random baseline

    # Add a "hot spot" near the center
    center_idx = n_bins // 2
    field[center_idx] = 1.0
    if center_idx > 0:
        field[center_idx - 1] = 0.8
    if center_idx < n_bins - 1:
        field[center_idx + 1] = 0.8

    return field


# =============================================================================
# Production-renderer helpers
# =============================================================================


def _read_last_video_frame(path) -> np.ndarray:
    """Read the final frame of a video as a uint8 RGB array via OpenCV.

    Reads sequentially rather than seeking: frame seeking is unreliable across
    codecs, and these test videos are only a handful of frames.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        frame_rgb: np.ndarray | None = None
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    assert frame_rgb is not None, f"no frames decoded from {path}"
    return frame_rgb


def _frame_to_figure(frame_rgb: np.ndarray) -> Figure:
    """Wrap a decoded RGB frame in a borderless figure for pytest-mpl."""
    dpi = 100
    height, width = frame_rgb.shape[:2]
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(frame_rgb)
    ax.axis("off")
    return fig


def render_overlay_frame(
    env: Environment,
    field: np.ndarray,
    save_path,
    *,
    n_frames: int,
    overlay_data: OverlayData | None = None,
    show_regions: bool | list[str] = False,
    region_alpha: float = 0.3,
) -> Figure:
    """Render a static field through the production video backend.

    The same ``field`` is repeated for ``n_frames`` so any animated overlay
    (e.g. a position trail) is fully built up by the final frame, which is the
    frame returned for comparison.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    field : np.ndarray
        Spatial field to render (1D array of values per bin).
    save_path : path-like
        Where to write the temporary ``.mp4``.
    n_frames : int
        Number of frames to render. Overlay arrays must match this length.
    overlay_data : OverlayData or None
        Overlay data passed straight to ``render_video``.
    show_regions : bool or list of str
        Region rendering flag passed to ``render_video``.
    region_alpha : float
        Region transparency passed to ``render_video``.

    Returns
    -------
    Figure
        Borderless figure wrapping the final rendered frame.
    """
    fields = [np.asarray(field, dtype=np.float64) for _ in range(n_frames)]
    render_video(
        env,
        fields,
        str(save_path),
        fps=10,
        vmin=0.0,
        vmax=1.0,
        n_workers=1,
        overlay_data=overlay_data,
        show_regions=show_regions,
        region_alpha=region_alpha,
    )
    return _frame_to_figure(_read_last_video_frame(save_path))


# =============================================================================
# Visual Regression Tests
# =============================================================================


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="position_overlay_with_trail.png",
    tolerance=5,
)
def test_position_overlay_with_trail(env_2d, simple_field_2d, tmp_path):
    """Test visual rendering of position overlay with trail."""
    trajectory = np.array(
        [
            [50, 50],  # Start at center
            [55, 55],
            [60, 58],
            [65, 60],
            [70, 62],
            [75, 65],
            [80, 68],
            [85, 70],
        ],
        dtype=float,
    )

    overlay_data = OverlayData(
        positions=[
            PositionData(
                data=trajectory,
                color="red",
                size=15.0,
                trail_length=5,
            )
        ]
    )

    return render_overlay_frame(
        env_2d,
        simple_field_2d,
        tmp_path / "position_trail.mp4",
        n_frames=len(trajectory),
        overlay_data=overlay_data,
    )


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="bodypart_overlay_with_skeleton.png",
    tolerance=5,
)
def test_bodypart_overlay_with_skeleton(env_2d, simple_field_2d, tmp_path):
    """Test visual rendering of bodypart overlay with skeleton."""
    # A simple 3-bodypart pose (nose, left_ear, right_ear) across 2 frames.
    bodyparts = {
        "nose": np.array([[50, 60], [52, 62]], dtype=float),
        "left_ear": np.array([[45, 65], [47, 67]], dtype=float),
        "right_ear": np.array([[55, 65], [57, 67]], dtype=float),
    }

    skeleton = Skeleton(
        name="test",
        nodes=("nose", "left_ear", "right_ear"),
        edges=(("nose", "left_ear"), ("nose", "right_ear")),
        edge_color="white",
        edge_width=2.0,
    )

    colors = {
        "nose": "red",
        "left_ear": "blue",
        "right_ear": "green",
    }

    overlay_data = OverlayData(
        bodypart_sets=[
            BodypartData(bodyparts=bodyparts, skeleton=skeleton, colors=colors)
        ]
    )

    return render_overlay_frame(
        env_2d,
        simple_field_2d,
        tmp_path / "bodypart_skeleton.mp4",
        n_frames=2,
        overlay_data=overlay_data,
    )


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="head_direction_overlay_angle.png",
    tolerance=5,
)
def test_head_direction_overlay_angle(env_2d, simple_field_2d, tmp_path):
    """Test visual rendering of head direction overlay (angle format)."""
    # Head direction as angles (radians): 45 deg then 90 deg.
    angles = np.array([np.pi / 4, np.pi / 2], dtype=float)

    overlay_data = OverlayData(
        head_directions=[HeadDirectionData(data=angles, color="yellow", length=15.0)]
    )

    return render_overlay_frame(
        env_2d,
        simple_field_2d,
        tmp_path / "head_direction_angle.mp4",
        n_frames=len(angles),
        overlay_data=overlay_data,
    )


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="head_direction_overlay_vector.png",
    tolerance=5,
)
def test_head_direction_overlay_vector(env_2d, simple_field_2d, tmp_path):
    """Test visual rendering of head direction overlay (vector format)."""
    # Head direction as unit vectors: 45 deg then 90 deg.
    vectors = np.array(
        [
            [0.707, 0.707],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    overlay_data = OverlayData(
        head_directions=[HeadDirectionData(data=vectors, color="orange", length=15.0)]
    )

    return render_overlay_frame(
        env_2d,
        simple_field_2d,
        tmp_path / "head_direction_vector.mp4",
        n_frames=len(vectors),
        overlay_data=overlay_data,
    )


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="regions_with_alpha.png",
    tolerance=5,
)
def test_regions_with_alpha(env_2d, simple_field_2d, tmp_path):
    """Test visual rendering of regions with alpha transparency."""
    return render_overlay_frame(
        env_2d,
        simple_field_2d,
        tmp_path / "regions_alpha.mp4",
        n_frames=1,
        show_regions=True,
        region_alpha=0.3,
    )


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="mixed_overlays_all_types.png",
    tolerance=5,
)
def test_mixed_overlays_all_types(env_2d, simple_field_2d, tmp_path):
    """Test visual rendering with all overlay types combined."""
    n_frames = 4

    trajectory = np.array(
        [
            [30, 30],
            [35, 35],
            [40, 38],
            [45, 40],
        ],
        dtype=float,
    )
    position = PositionData(
        data=trajectory,
        color="red",
        size=12.0,
        trail_length=3,
    )

    # Bodyparts held static across the 4 frames.
    bodyparts = {
        "head": np.tile(np.array([60.0, 60.0]), (n_frames, 1)),
        "tail": np.tile(np.array([55.0, 55.0]), (n_frames, 1)),
    }
    skeleton = Skeleton(
        name="test",
        nodes=("head", "tail"),
        edges=(("head", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    bodypart = BodypartData(
        bodyparts=bodyparts,
        skeleton=skeleton,
        colors={"head": "cyan", "tail": "magenta"},
    )

    # Head direction held static (60 deg) across the 4 frames.
    angles = np.full(n_frames, np.pi / 3, dtype=float)
    head_direction = HeadDirectionData(data=angles, color="yellow", length=12.0)

    overlay_data = OverlayData(
        positions=[position],
        bodypart_sets=[bodypart],
        head_directions=[head_direction],
    )

    return render_overlay_frame(
        env_2d,
        simple_field_2d,
        tmp_path / "mixed_overlays.mp4",
        n_frames=n_frames,
        overlay_data=overlay_data,
        show_regions=True,
        region_alpha=0.2,
    )


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="position_overlay_no_trail.png",
    tolerance=5,
)
def test_position_overlay_no_trail(env_2d, simple_field_2d, tmp_path):
    """Test visual rendering of position overlay without trail."""
    position = np.array([[70, 70]], dtype=float)

    overlay_data = OverlayData(
        positions=[
            PositionData(
                data=position,
                color="green",
                size=20.0,
                trail_length=None,  # No trail
            )
        ]
    )

    return render_overlay_frame(
        env_2d,
        simple_field_2d,
        tmp_path / "position_no_trail.mp4",
        n_frames=1,
        overlay_data=overlay_data,
    )
