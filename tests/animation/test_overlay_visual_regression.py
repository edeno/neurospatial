"""Visual regression tests for overlay rendering.

This module uses pytest-mpl to generate and compare golden baseline images
for overlay rendering. Tests ensure visual consistency across code changes.

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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon

from neurospatial import (
    BodypartOverlay,
    Environment,
    HeadDirectionOverlay,
    PositionOverlay,
)
from neurospatial.animation.skeleton import Skeleton
from neurospatial.regions import Regions

# Use non-interactive backend for testing
matplotlib.use("Agg")


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
    # Create a Gaussian-like field centered in the environment
    n_bins = env_2d.n_bins
    field = np.random.rand(n_bins) * 0.5 + 0.2  # Random baseline

    # Add a "hot spot" near the center
    center_idx = n_bins // 2
    field[center_idx] = 1.0
    if center_idx > 0:
        field[center_idx - 1] = 0.8
    if center_idx < n_bins - 1:
        field[center_idx + 1] = 0.8

    return field


# =============================================================================
# Helper Functions for Rendering
# =============================================================================


def render_field_with_overlays(
    env: Environment,
    field: np.ndarray,
    position_overlay: PositionOverlay | None = None,
    bodypart_overlay: BodypartOverlay | None = None,
    head_direction_overlay: HeadDirectionOverlay | None = None,
    show_regions: bool = False,
    region_alpha: float = 0.3,
    figsize: tuple[float, float] = (6, 5),
) -> Figure:
    """Render a spatial field with overlays for visual regression testing.

    This is a simplified rendering function that creates matplotlib figures
    for visual comparison. It mimics the video backend's rendering approach.

    Parameters
    ----------
    env : Environment
        The spatial environment.
    field : np.ndarray
        The spatial field to render (1D array of values per bin).
    position_overlay : PositionOverlay | None
        Optional position overlay to render.
    bodypart_overlay : BodypartOverlay | None
        Optional bodypart overlay to render.
    head_direction_overlay : HeadDirectionOverlay | None
        Optional head direction overlay to render.
    show_regions : bool
        Whether to show environment regions.
    region_alpha : float
        Alpha transparency for regions (0-1).
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    Figure
        The matplotlib figure with rendered overlays.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get dimension ranges for plotting
    assert env.dimension_ranges is not None  # type narrowing
    x_min, x_max = env.dimension_ranges[0]
    y_min, y_max = env.dimension_ranges[1]

    # Plot the spatial field as scatter points
    bin_centers = env.bin_centers
    colors = field
    scatter = ax.scatter(
        bin_centers[:, 0],
        bin_centers[:, 1],
        c=colors,
        cmap="viridis",
        s=100,
        vmin=0,
        vmax=1,
        alpha=0.6,
        zorder=1,
    )

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Field Value")

    # Render regions if requested
    if show_regions and env.regions is not None:
        for region in env.regions.values():
            if region.kind == "point":
                circle = Circle(
                    tuple(region.data),
                    radius=3.0,
                    color="red",
                    alpha=region_alpha,
                    zorder=2,
                )
                ax.add_patch(circle)
            elif region.kind == "polygon":
                coords = np.array(region.data.exterior.coords)  # type: ignore[union-attr]
                path = MplPath(coords)
                patch = PathPatch(
                    path,
                    facecolor="blue",
                    edgecolor="blue",
                    alpha=region_alpha,
                    linewidth=2,
                    zorder=2,
                )
                ax.add_patch(patch)

    # Render position overlay
    if position_overlay is not None:
        # For visual regression, render first point + trail
        pos_data = position_overlay.data
        current_pos = pos_data[0]  # First position

        # Render trail (if enough points)
        trail_length = position_overlay.trail_length or 5
        trail_end = min(trail_length, len(pos_data))
        if trail_end > 1:
            trail = pos_data[:trail_end]
            # Decaying alpha for trail
            for i in range(len(trail) - 1):
                alpha = 0.3 + 0.7 * (i / (len(trail) - 1))
                ax.plot(
                    trail[i : i + 2, 0],
                    trail[i : i + 2, 1],
                    color=position_overlay.color,
                    alpha=alpha,
                    linewidth=2,
                    zorder=3,
                )

        # Render current position marker
        ax.scatter(
            current_pos[0],
            current_pos[1],
            color=position_overlay.color,
            s=position_overlay.size * 10,  # Scale for visibility
            marker="o",
            edgecolors="white",
            linewidths=1,
            zorder=4,
        )

    # Render bodypart overlay
    if bodypart_overlay is not None:
        # Render first frame of bodyparts
        for part_name, part_data in bodypart_overlay.data.items():
            pos = part_data[0]  # First frame

            # Get color for this part
            if bodypart_overlay.colors and part_name in bodypart_overlay.colors:
                color = bodypart_overlay.colors[part_name]
            else:
                color = "cyan"

            # Plot bodypart point
            ax.scatter(
                pos[0],
                pos[1],
                color=color,
                s=50,
                marker="o",
                edgecolors="white",
                linewidths=1,
                zorder=4,
            )

            # Add label
            ax.text(
                pos[0] + 2,
                pos[1] + 2,
                part_name,
                color=color,
                fontsize=8,
                zorder=5,
            )

        # Render skeleton if provided
        if bodypart_overlay.skeleton:
            skeleton = bodypart_overlay.skeleton
            for part_a, part_b in skeleton.edges:
                if part_a in bodypart_overlay.data and part_b in bodypart_overlay.data:
                    pos_a = bodypart_overlay.data[part_a][0]
                    pos_b = bodypart_overlay.data[part_b][0]
                    ax.plot(
                        [pos_a[0], pos_b[0]],
                        [pos_a[1], pos_b[1]],
                        color=skeleton.edge_color,
                        linewidth=skeleton.edge_width,
                        zorder=3,
                    )

    # Render head direction overlay
    if head_direction_overlay is not None:
        # Assume head direction is at center of environment
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Get direction (first frame)
        direction_data = head_direction_overlay.data

        # Check if it's an angle or vector
        if direction_data.ndim == 1:
            # Angle in radians
            angle = direction_data[0]
            dx = head_direction_overlay.length * np.cos(angle)
            dy = head_direction_overlay.length * np.sin(angle)
        else:
            # Unit vector (needs scaling)
            vector = direction_data[0]
            dx = vector[0] * head_direction_overlay.length
            dy = vector[1] * head_direction_overlay.length

        # Render arrow
        ax.arrow(
            center_x,
            center_y,
            dx,
            dy,
            color=head_direction_overlay.color,
            width=2.0,
            head_width=5.0,
            head_length=3.0,
            zorder=4,
        )

    # Set limits and labels
    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Spatial Field with Overlays")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# =============================================================================
# Visual Regression Tests
# =============================================================================


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="position_overlay_with_trail.png",
    tolerance=5,
)
def test_position_overlay_with_trail(env_2d, simple_field_2d):
    """Test visual rendering of position overlay with trail."""
    # Create position overlay with trail
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
        ]
    )

    position_overlay = PositionOverlay(
        data=trajectory,
        color="red",
        size=15.0,
        trail_length=5,
    )

    fig = render_field_with_overlays(
        env_2d,
        simple_field_2d,
        position_overlay=position_overlay,
    )

    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="bodypart_overlay_with_skeleton.png",
    tolerance=5,
)
def test_bodypart_overlay_with_skeleton(env_2d, simple_field_2d):
    """Test visual rendering of bodypart overlay with skeleton."""
    # Create a simple 3-bodypart pose (nose, left_ear, right_ear)
    bodyparts = {
        "nose": np.array([[50, 60], [52, 62]]),  # 2 frames
        "left_ear": np.array([[45, 65], [47, 67]]),
        "right_ear": np.array([[55, 65], [57, 67]]),
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

    bodypart_overlay = BodypartOverlay(
        data=bodyparts,
        skeleton=skeleton,
        colors=colors,
    )

    fig = render_field_with_overlays(
        env_2d,
        simple_field_2d,
        bodypart_overlay=bodypart_overlay,
    )

    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="head_direction_overlay_angle.png",
    tolerance=5,
)
def test_head_direction_overlay_angle(env_2d, simple_field_2d):
    """Test visual rendering of head direction overlay (angle format)."""
    # Create head direction as angles (radians)
    # 45 degrees = π/4 radians
    angles = np.array([np.pi / 4, np.pi / 2])  # 2 frames

    head_direction_overlay = HeadDirectionOverlay(
        data=angles,
        color="yellow",
        length=15.0,
    )

    fig = render_field_with_overlays(
        env_2d,
        simple_field_2d,
        head_direction_overlay=head_direction_overlay,
    )

    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="head_direction_overlay_vector.png",
    tolerance=5,
)
def test_head_direction_overlay_vector(env_2d, simple_field_2d):
    """Test visual rendering of head direction overlay (vector format)."""
    # Create head direction as unit vectors
    # Pointing northeast
    vectors = np.array(
        [
            [0.707, 0.707],  # 45 degrees normalized
            [0.0, 1.0],  # 90 degrees
        ]
    )

    head_direction_overlay = HeadDirectionOverlay(
        data=vectors,
        color="orange",
        length=15.0,
    )

    fig = render_field_with_overlays(
        env_2d,
        simple_field_2d,
        head_direction_overlay=head_direction_overlay,
    )

    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="regions_with_alpha.png",
    tolerance=5,
)
def test_regions_with_alpha(env_2d, simple_field_2d):
    """Test visual rendering of regions with alpha transparency."""
    fig = render_field_with_overlays(
        env_2d,
        simple_field_2d,
        show_regions=True,
        region_alpha=0.3,
    )

    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="mixed_overlays_all_types.png",
    tolerance=5,
)
def test_mixed_overlays_all_types(env_2d, simple_field_2d):
    """Test visual rendering with all overlay types combined."""
    # Position overlay
    trajectory = np.array(
        [
            [30, 30],
            [35, 35],
            [40, 38],
            [45, 40],
        ]
    )
    position_overlay = PositionOverlay(
        data=trajectory,
        color="red",
        size=12.0,
        trail_length=3,
    )

    # Bodypart overlay
    bodyparts = {
        "head": np.array([[60, 60]]),
        "tail": np.array([[55, 55]]),
    }
    skeleton = Skeleton(
        name="test",
        nodes=("head", "tail"),
        edges=(("head", "tail"),),
        edge_color="white",
        edge_width=2.0,
    )
    bodypart_overlay = BodypartOverlay(
        data=bodyparts,
        skeleton=skeleton,
        colors={"head": "cyan", "tail": "magenta"},
    )

    # Head direction overlay
    angles = np.array([np.pi / 3])  # 60 degrees
    head_direction_overlay = HeadDirectionOverlay(
        data=angles,
        color="yellow",
        length=12.0,
    )

    fig = render_field_with_overlays(
        env_2d,
        simple_field_2d,
        position_overlay=position_overlay,
        bodypart_overlay=bodypart_overlay,
        head_direction_overlay=head_direction_overlay,
        show_regions=True,
        region_alpha=0.2,
    )

    return fig


@pytest.mark.mpl_image_compare(
    baseline_dir="baseline",
    filename="position_overlay_no_trail.png",
    tolerance=5,
)
def test_position_overlay_no_trail(env_2d, simple_field_2d):
    """Test visual rendering of position overlay without trail."""
    # Single position, no trail
    position = np.array([[70, 70]])

    position_overlay = PositionOverlay(
        data=position,
        color="green",
        size=20.0,
        trail_length=None,  # No trail
    )

    fig = render_field_with_overlays(
        env_2d,
        simple_field_2d,
        position_overlay=position_overlay,
    )

    return fig
