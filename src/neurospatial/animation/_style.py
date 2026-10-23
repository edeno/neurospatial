"""Backend-independent overlay visibility, trail geometry, and color rules.

Matplotlib color utilities are loaded only when resolving a color; importing
these helpers does not load a rendering backend or its artist machinery.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_TRAIL_MAX_ALPHA = 0.7


def _marker_area(size: float, *, dpi: float) -> float:
    """Convert a screen-pixel marker diameter to Matplotlib points squared.

    Overlay sizes describe the display, independently of coordinate units and
    zoom. Matplotlib's scatter API instead expects an area in typographic
    points, so the canvas DPI belongs in this backend-boundary conversion.
    """
    return (size * 72.0 / dpi) ** 2


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


def _heading_displacement(
    heading: NDArray[np.float64] | float, length: float, *, n_dims: int
) -> NDArray[np.float64] | None:
    """Return a heading's displacement in environment coordinates, or None.

    A finite angle describes a direction in the x-y plane. A finite nonzero
    vector describes a direction in all its coordinates, regardless of its
    magnitude. Zero vectors have no direction (including cancellation during
    interpolation), so renderers must omit their arrows. Scaling before the
    norm preserves even very small or large finite vectors without underflow
    or overflow. Coordinate conversion belongs to each backend.
    """
    value = np.asarray(heading, dtype=np.float64)
    samples = value.reshape(1) if value.ndim == 0 else value[np.newaxis, :]
    direction = _heading_displacements(samples, length, n_dims=n_dims)[0]
    return direction if np.isfinite(direction).all() else None


def _heading_displacements(
    headings: NDArray[np.float64], length: float, *, n_dims: int
) -> NDArray[np.float64]:
    """Batch form of heading geometry; undefined rows are entirely NaN.

    One-dimensional input contains angles; two-dimensional input contains
    vectors. Batch normalization keeps long-recording backend setup vectorized.
    """
    values = np.asarray(headings, dtype=np.float64)
    directions = np.full((len(values), n_dims), np.nan)
    if values.ndim == 1:
        valid = np.isfinite(values)
        directions[valid] = 0
        directions[valid, 0] = np.cos(values[valid])
        directions[valid, 1] = np.sin(values[valid])
    elif values.ndim == 2 and values.shape[1] == n_dims:
        scales = np.max(np.abs(values), axis=1)
        valid = np.isfinite(values).all(axis=1) & (scales > 0)
        scaled = values[valid] / scales[valid, np.newaxis]
        directions[valid] = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    return directions * length


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


def see_through_alpha(
    values: NDArray[np.float64], vmin: float, vmax: float, opaque_from: float
) -> NDArray[np.float64]:
    """Opacity of field values drawn over a video: low values see-through.

    Opacity rises linearly from zero at ``vmin`` to one at ``opaque_from``
    (``vmax`` at the latest), so what is drawn below, such as the camera's
    view of the animal, shows where values are low. A threshold at or
    below ``vmin`` leaves every value opaque.

    Parameters
    ----------
    values : ndarray of float64
        Field values.
    vmin, vmax : float
        The color limits.
    opaque_from : float
        The value from which colors are opaque.

    Returns
    -------
    ndarray of float64, same shape as ``values``
        Opacity in [0, 1], as blended in displayed (sRGB) color.
    """
    opaque = min(opaque_from, vmax)
    if opaque <= vmin:
        return np.ones(np.shape(values))
    return np.clip(
        (np.asarray(values, dtype=np.float64) - vmin) / (opaque - vmin), 0, 1
    )


def field_colors(
    cmap: str,
    vmin: float,
    vmax: float,
    opaque_from: float | None,
    *,
    n_colors: int = 256,
) -> NDArray[np.float64]:
    """A field's colors from ``vmin`` to ``vmax``, shape (n_colors, 4).

    With ``opaque_from``, low values fade (see :func:`see_through_alpha`).
    """
    from matplotlib import colormaps

    colors: NDArray[np.float64] = colormaps[cmap](np.linspace(0.0, 1.0, n_colors))
    if opaque_from is not None:
        colors[:, 3] *= see_through_alpha(
            np.linspace(vmin, vmax, n_colors), vmin, vmax, opaque_from
        )
    return colors


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
