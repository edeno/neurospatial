"""Tests for position-trail alpha computation with NaN (occluded) frames.

Regression tests for a bug where NaN-filtering the trail positions before
computing per-segment alpha corrupted the alpha indexing: the alpha denominator
used the post-filter length, so with occluded frames the oldest visible segment
received the wrong alpha and the trail appeared to "jump" in opacity.

The correct behavior: each segment's alpha must reflect its true age (its
position within the *original*, unfiltered trail window), not its compacted
index among the surviving valid positions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from neurospatial.animation._parallel import (
    OverlayArtistManager,
    _render_position_overlay_matplotlib,
)
from neurospatial.animation.overlays import OverlayData, PositionData


def _trail_alphas(ax) -> np.ndarray:
    """Extract per-segment alpha values from the trail LineCollection on ax."""
    line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(line_collections) == 1, "expected exactly one trail LineCollection"
    return line_collections[0].get_colors()[:, 3]


def test_trail_alpha_matches_true_age_with_nan_gap():
    """With a NaN gap, segment alpha must track true age, not compacted index.

    Build an 8-frame trail where one interior frame is occluded (NaN). When the
    NaN frame is dropped, the alpha denominator must still reflect the full trail
    window length so that the oldest visible segment keeps a low alpha and the
    newest keeps a high alpha.
    """
    n = 8
    positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])
    # Occlude an interior frame (index 3).
    positions[3] = [np.nan, np.nan]

    pos_data = PositionData(data=positions, color="red", size=10.0, trail_length=n)

    fig, ax = plt.subplots()
    try:
        # Render at the last frame so the full window is the trail.
        _render_position_overlay_matplotlib(ax, pos_data, frame_idx=n - 1)
        alphas = _trail_alphas(ax)
    finally:
        plt.close(fig)

    # Valid window indices are [0, 1, 2, 4, 5, 6, 7] (index 3 occluded).
    # Segments connect consecutive valid positions:
    #   (0->1), (1->2), (2->4), (4->5), (5->6), (6->7)
    # True-age alpha = (newer_window_index / (L-1)) * 0.7 with L = 8 frames.
    max_alpha = 0.7
    denom = n - 1  # 7
    expected = np.array([1, 2, 4, 5, 6, 7], dtype=float) / denom * max_alpha
    np.testing.assert_allclose(alphas, expected, atol=1e-9)

    # The newest segment reaches the current frame -> full 0.7 alpha.
    assert alphas[-1] == max(alphas)
    np.testing.assert_allclose(alphas[-1], max_alpha, atol=1e-9)

    # The segment bridging the NaN gap must jump (skip the occluded frame's
    # age), proving the alpha indexes the original window, not the compacted
    # valid sequence.
    np.testing.assert_allclose(alphas[2], 4 / denom * max_alpha, atol=1e-9)


def test_trail_alpha_no_gap_matches_window_length():
    """Without gaps, alpha must span the trail evenly up to the 0.7 maximum."""
    n = 6
    positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])
    pos_data = PositionData(data=positions, color="blue", size=10.0, trail_length=n)

    fig, ax = plt.subplots()
    try:
        _render_position_overlay_matplotlib(ax, pos_data, frame_idx=n - 1)
        alphas = _trail_alphas(ax)
    finally:
        plt.close(fig)

    # n positions in the window -> n-1 segments.
    assert len(alphas) == n - 1
    assert np.all(np.diff(alphas) > 0)
    assert alphas[-1] == max(alphas)
    # Newest segment reaches the current frame -> full 0.7 alpha.
    np.testing.assert_allclose(alphas[-1], 0.7, atol=1e-9)


def test_artist_manager_trail_alpha_matches_true_age_with_nan_gap():
    """OverlayArtistManager trail update must use true-age alphas too."""
    n = 8
    positions = np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])
    positions[3] = [np.nan, np.nan]

    overlay_data = OverlayData(
        positions=[PositionData(data=positions, color="red", size=10.0, trail_length=n)]
    )

    fig, ax = plt.subplots()
    try:
        manager = OverlayArtistManager(
            ax=ax,
            env=None,
            overlay_data=overlay_data,
            show_regions=False,
            region_alpha=0.3,
        )
        manager.initialize(frame_idx=0)
        manager.update_frame(n - 1)

        trail_lc = manager._position_trails[0]
        alphas = trail_lc.get_colors()[:, 3]
    finally:
        plt.close(fig)

    denom = n - 1
    max_alpha = 0.7
    expected = np.array([1, 2, 4, 5, 6, 7], dtype=float) / denom * max_alpha
    np.testing.assert_allclose(alphas, expected, atol=1e-9)
    assert alphas[-1] == max(alphas)
    np.testing.assert_allclose(alphas[-1], max_alpha, atol=1e-9)
