"""Backend-independent overlay visibility, trail geometry, and color rules.

Matplotlib color utilities are loaded only when resolving a color; importing
these helpers does not load a rendering backend or its artist machinery.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray

_TRAIL_MAX_ALPHA = 0.7


def _event_visibility(
    frame_indices: NDArray[np.int_], frame_idx: int, decay: int | None
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Which events show on a frame, and their opacity.

    Parameters
    ----------
    frame_indices : ndarray of int, shape (n_events,)
        The frame each event appears on.
    frame_idx : int
        The frame drawn.
    decay : int or None
        None shows every event so far; 0 only the events on this frame; a
        positive count the events of the last ``decay`` frames, fading with
        age (newest 1.0; the + 1 keeps the oldest visible).

    Returns
    -------
    shown : ndarray of bool, shape (n_events,)
    alphas : ndarray of float64, shape (n_events,)
        Opacity of each event; meaningful where ``shown``.
    """
    if decay is None:
        shown = frame_indices <= frame_idx
    elif decay == 0:
        shown = frame_indices == frame_idx
    else:
        shown = (frame_indices >= max(0, frame_idx - decay)) & (
            frame_indices <= frame_idx
        )
    if decay:
        alphas = 1.0 - (frame_idx - frame_indices) / (decay + 1)
    else:
        alphas = np.ones(len(frame_indices))
    return shown, alphas


def _build_trail_segments(
    trail_positions: NDArray[np.float64],
    color: Any,
    *,
    interval_valid: NDArray[np.bool_] | None = None,
) -> tuple[list[NDArray[np.float64]], list[tuple[float, float, float, float]]]:
    """Build decaying-alpha trail line segments from a position window.

    Each segment connects adjacent observed positions within the
    trail window. The per-segment alpha encodes the segment's true age: it is
    derived from the segment's newer endpoint index within the *original*
    window (length ``len(trail_positions)``), NOT from the compacted index
    among surviving positions. This keeps the alpha gradient stable when
    occluded (NaN) frames drop out of the window, so the trail does not "jump"
    in opacity.

    Parameters
    ----------
    trail_positions : NDArray[np.float64], shape (window_len, n_dims)
        Positions in the trail window, ordered oldest-first. May contain NaN
        rows for occluded frames; the newest frame is the last row.
    color : Any
        Matplotlib color spec for the trail (converted to RGBA).

    Returns
    -------
    segments : list[NDArray[np.float64]]
        Line segments, each of shape ``(2, n_dims)``. Empty if fewer than two
        visible, consecutive positions exist.
    colors : list[tuple[float, float, float, float]]
        Per-segment RGBA colors with the age-based alpha applied. Same length
        as ``segments``.
    """
    from matplotlib.colors import to_rgba

    window_len = len(trail_positions)
    if window_len < 2:
        return [], []

    valid_mask = np.isfinite(trail_positions).all(axis=1)
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < 2:
        return [], []

    # Normalize alpha by the full window span so age reflects elapsed frames,
    # not the count of surviving positions. window_len >= 2 here.
    denom = window_len - 1
    base_rgba = to_rgba(color)

    segments: list[NDArray[np.float64]] = []
    colors: list[tuple[float, float, float, float]] = []
    for older, newer in itertools.pairwise(valid_indices):
        if newer != older + 1 or (
            interval_valid is not None and not interval_valid[older]
        ):
            continue
        segments.append(trail_positions[[older, newer]])
        # Newer endpoint's original window index sets the segment's age.
        alpha = float(newer) / denom * _TRAIL_MAX_ALPHA
        colors.append((base_rgba[0], base_rgba[1], base_rgba[2], alpha))

    return segments, colors


def _head_direction_color(
    color: str, direction: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    """Resolve heading hue or a fixed color to RGBA for Matplotlib.

    Colormaps use counterclockwise heading in environment (x, y) coordinates:
    east is 0, north is 0.25, west is 0.5, and south is 0.75. Angles wrap at
    one full turn. Names registered as colormaps take precedence over colors.
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_rgba

    if color in colormaps:
        hue = float((np.arctan2(direction[1], direction[0]) / (2 * np.pi)) % 1.0)
        # Floating-point sin(2*pi) can put due east just below the wrap point.
        if 1.0 - hue <= 4 * np.finfo(float).eps:
            hue = 0.0
        return colormaps[color](hue)
    return to_rgba(color)


def _finite_rate_peak(rates: NDArray[np.float64] | None) -> float:
    """A finite positive peak computed once for a prepared source recording."""
    if rates is None:
        return 1.0
    finite = np.asarray(rates)[np.isfinite(rates)]
    peak = float(np.max(finite, initial=0.0))
    return peak if peak > 0 else 1.0


def _rate_opacity(rate: float, peak: float) -> float:
    """Rate-modulated opacity: zero activity is faint and missing is hidden."""
    if not np.isfinite(rate):
        return 0.0
    # The prepared peak is positive. Clamp callers' changed or negative values
    # so every backend receives a valid alpha with the same interpretation.
    return float(np.clip(0.1 + 0.9 * rate / peak, 0.0, 1.0))
