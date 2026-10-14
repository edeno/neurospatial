"""Heading colormaps and fixed colors work in both Matplotlib rendering paths."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from matplotlib.quiver import Quiver

from neurospatial import Environment
from neurospatial.animation._parallel import (
    OverlayArtistManager,
    _render_head_direction_overlay_matplotlib,
)
from neurospatial.animation.overlays import HeadDirectionOverlay, OverlayData


@pytest.mark.parametrize("renderer", ["redraw", "persistent"])
@pytest.mark.parametrize("data_format", ["angles", "vectors"])
@pytest.mark.parametrize("color", [None, "twilight", "yellow", "#336699"])
def test_heading_colors_follow_direction_across_frame_updates(
    renderer: str, data_format: str, color: str | None
) -> None:
    """The public default renders; changing heading changes hue, not fixed colors."""
    env = Environment.from_grid_mask(
        active_mask=np.ones((2, 2), dtype=bool),
        grid_edges=(np.arange(3, dtype=float), np.arange(3, dtype=float)),
    )
    angles = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2, 3 * np.pi / 2, 2 * np.pi])
    headings = (
        angles
        if data_format == "angles"
        else 3.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    )
    overlay = (
        HeadDirectionOverlay(headings=headings, length=1.0)
        if color is None
        else HeadDirectionOverlay(headings=headings, color=color, length=1.0)
    )
    frame_times = np.arange(len(angles), dtype=float)
    data = overlay.convert_to_data(frame_times, len(frame_times), env)
    fig, ax = plt.subplots()
    manager = OverlayArtistManager(
        ax=ax,
        env=env,
        overlay_data=OverlayData(head_directions=[data]),
        show_regions=False,
        region_alpha=0.3,
    )
    try:
        for frame, angle in enumerate(angles):
            if renderer == "persistent":
                manager.update_frame(frame)
            else:
                ax.clear()
                _render_head_direction_overlay_matplotlib(ax, data, frame, env)
            fig.canvas.draw()
            arrows = [artist for artist in ax.collections if isinstance(artist, Quiver)]
            assert len(arrows) == 1
            arrow = arrows[0]
            expected = (
                colormaps[color or "hsv"]((angle % (2 * np.pi)) / (2 * np.pi))
                if color in {None, "twilight"}
                else to_rgba(color)
            )
            np.testing.assert_allclose(arrow.get_facecolors()[0], expected, atol=1e-12)
            np.testing.assert_allclose(
                [arrow.U[0], arrow.V[0]], [np.cos(angle), np.sin(angle)], atol=1e-12
            )
    finally:
        plt.close(fig)
