"""Scale bar utilities for spatial visualizations.

This module provides scale bar functionality for matplotlib and napari visualizations,
including automatic "nice" length calculation using the 1-2-5 rule.

Note: This module provides **visualization scale bars** (showing physical scale on plots).
This is distinct from the existing **calibration scale bar** feature
(`calibrate_from_scale_bar()`, `calibrate_video(scale_bar=...)`) which is used for
video-to-environment coordinate transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.font_manager as fm
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

if TYPE_CHECKING:
    import matplotlib.axes
    import napari

__all__ = [
    "ScaleBarConfig",
    "add_scale_bar_to_axes",
    "compute_nice_length",
    "configure_napari_scale_bar",
    "format_scale_label",
]


@dataclass(frozen=True)
class ScaleBarConfig:
    """Configuration for scale bar rendering.

    Attributes
    ----------
    length : float | None
        Scale bar length in environment units. If None, auto-calculated
        using the 1-2-5 rule to produce aesthetically pleasing values.
        **Matplotlib-only**: Napari auto-sizes based on layer scale.
    position : {"lower right", "lower left", "upper right", "upper left"}
        Scale bar position. Works on all backends.
    color : str | None
        Bar and text color. If None, auto-selects for contrast.
        Works on all backends.
    background : str | None
        Background box color. None for transparent. **Matplotlib-only**.
    background_alpha : float
        Background transparency (0-1). **Matplotlib-only**.
    font_size : int
        Text font size in points. Works on all backends.
    pad : float
        Padding from axes edge. AnchoredSizeBar uses this as a multiple
        of font size, not inches. **Matplotlib-only**.
    sep : float
        Separation between bar and label in points. **Matplotlib-only**.
    label_top : bool
        Label above bar (True) or below (False). **Matplotlib-only**.
    box_style : {"round", "square", None}
        Background box style. None for no box. **Matplotlib-only**.
    show_label : bool
        Whether to show "10 cm" text label. Works on all backends.

    Examples
    --------
    >>> # Default configuration
    >>> config = ScaleBarConfig()
    >>> config.position
    'lower right'

    >>> # Custom configuration
    >>> config = ScaleBarConfig(length=20.0, color="white", position="upper left")
    >>> config.length
    20.0
    """

    length: float | None = None
    position: Literal["lower right", "lower left", "upper right", "upper left"] = (
        "lower right"
    )
    color: str | None = None
    background: str | None = "white"
    background_alpha: float = 0.7
    font_size: int = 10
    pad: float = 0.5
    sep: float = 5.0
    label_top: bool = True
    box_style: Literal["round", "square", None] = "round"
    show_label: bool = True


def compute_nice_length(extent: float, target_fraction: float = 0.2) -> float:
    """Compute aesthetically pleasing scale bar length using 1-2-5 rule.

    Calculates a "nice" length that is approximately `target_fraction` of the
    total extent, rounded to the nearest value in the 1-2-5 sequence
    (1, 2, 5, 10, 20, 50, 100, ...).

    Parameters
    ----------
    extent : float
        The total extent of the axis (max - min) in environment units.
        Must be positive.
    target_fraction : float, optional
        Target scale bar length as a fraction of extent (default 0.2 = 20%).

    Returns
    -------
    float
        A "nice" length value following the 1-2-5 rule.

    Raises
    ------
    ValueError
        If extent is zero, negative, or non-finite.

    Examples
    --------
    >>> compute_nice_length(100)
    20

    >>> compute_nice_length(50)
    10

    >>> compute_nice_length(73)
    10

    >>> compute_nice_length(0.5)
    0.1
    """
    # Validate extent
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError(f"extent must be a positive finite number, got {extent}")

    target = extent * target_fraction

    # Find order of magnitude
    magnitude = 10 ** np.floor(np.log10(target))

    # Normalized value (in range [1, 10))
    normalized = target / magnitude

    # Round to nearest 1, 2, or 5
    if normalized < 1.5:
        nice = 1.0
    elif normalized < 3.5:
        nice = 2.0
    elif normalized < 7.5:
        nice = 5.0
    else:
        nice = 10.0

    return float(nice * magnitude)


def format_scale_label(length: float, units: str | None) -> str:
    """Format scale bar label text.

    Formats the length value with optional units, using integer display
    for whole numbers (e.g., "10 cm" not "10.0 cm").

    Parameters
    ----------
    length : float
        Scale bar length in environment units.
    units : str | None
        Unit label (e.g., "cm", "um"). If None, no unit shown.

    Returns
    -------
    str
        Formatted label string.

    Examples
    --------
    >>> format_scale_label(10, "cm")
    '10 cm'

    >>> format_scale_label(10.0, "cm")
    '10 cm'

    >>> format_scale_label(2.5, "cm")
    '2.5 cm'

    >>> format_scale_label(10, None)
    '10'
    """
    # Integer if whole number
    length_str = str(int(length)) if length == int(length) else str(length)

    if units:
        return f"{length_str} {units}"
    return length_str


def _auto_contrast_color(ax: matplotlib.axes.Axes) -> str:
    """Determine scale bar color based on axes background.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to analyze.

    Returns
    -------
    str
        "white" for dark backgrounds, "black" for light backgrounds.
    """
    import matplotlib.colors as mcolors

    bg_color = ax.get_facecolor()
    # Convert to RGBA tuple using matplotlib's color converter
    rgba = mcolors.to_rgba(bg_color)
    # Convert to grayscale luminance
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if luminance < 0.5 else "black"


def add_scale_bar_to_axes(
    ax: matplotlib.axes.Axes,
    extent: float,
    units: str | None = None,
    config: ScaleBarConfig | None = None,
) -> AnchoredSizeBar:
    """Add a scale bar to matplotlib axes.

    Uses `AnchoredSizeBar` for precise positioning that survives
    figure resizing and zoom operations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to.
    extent : float
        The total extent of the axis (max - min) in environment units.
        Used to auto-calculate length if not specified in config.
    units : str | None, optional
        Unit label (e.g., "cm", "um"). If None, no unit shown.
    config : ScaleBarConfig | None, optional
        Configuration object. If None, uses default ScaleBarConfig().

    Returns
    -------
    AnchoredSizeBar
        The scale bar artist added to the axes.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.set_xlim(0, 100)
    >>> ax.set_ylim(0, 100)
    >>> artist = add_scale_bar_to_axes(ax, extent=100, units="cm")
    >>> plt.close(fig)
    """
    if config is None:
        config = ScaleBarConfig()

    # Calculate length if not specified
    length = compute_nice_length(extent) if config.length is None else config.length

    # Determine color
    color = _auto_contrast_color(ax) if config.color is None else config.color

    # Format label
    label = format_scale_label(length, units) if config.show_label else ""

    # Position mapping to matplotlib loc codes
    loc_map = {
        "lower right": "lower right",
        "lower left": "lower left",
        "upper right": "upper right",
        "upper left": "upper left",
    }
    loc = loc_map.get(config.position, "lower right")

    # label_top is just config.label_top (bool)
    label_top = config.label_top

    # Create font properties
    fontprops = fm.FontProperties(size=config.font_size)

    # Create scale bar
    scalebar = AnchoredSizeBar(
        ax.transData,
        size=length,
        label=label,
        loc=loc,
        pad=config.pad,
        color=color,
        frameon=config.box_style is not None,
        sep=config.sep,
        label_top=label_top,
        fontproperties=fontprops,
    )

    # Set background if specified
    if config.box_style is not None and config.background is not None:
        scalebar.patch.set_facecolor(config.background)
        scalebar.patch.set_alpha(config.background_alpha)
        if config.box_style == "round":
            scalebar.patch.set_boxstyle("round,pad=0.3")

    ax.add_artist(scalebar)
    return scalebar


def configure_napari_scale_bar(
    viewer: napari.Viewer,
    units: str | None = None,
    config: ScaleBarConfig | None = None,
) -> None:
    """Configure napari's native scale bar.

    Napari has built-in scale bar support with automatic sizing based on
    the current zoom level and layer scale.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    units : str | None
        Unit label (e.g., "cm", "um"). If None, no unit shown.
    config : ScaleBarConfig | None
        Configuration object. Note: ``config.length`` is ignored because
        napari auto-sizes based on layer scale and zoom level.

    Notes
    -----
    **Napari scale bar auto-sizing**: Unlike matplotlib where we set an
    explicit length, napari calculates the scale bar length dynamically
    based on the current view. This is actually beneficial for interactive
    use but means ``ScaleBarConfig.length`` has no effect in napari.

    **Layer scale**: For napari's scale bar to show correct units, the
    image layer must have the correct ``scale`` attribute set. This is
    typically set via ``EnvScale`` in ``animation/transforms.py``. If scale
    is (1, 1), the scale bar shows pixels regardless of ``unit`` setting.

    Supported config attributes:

    - position: Maps to napari's position names
    - color: Scale bar color
    - font_size: Text size

    Ignored config attributes (matplotlib-only):

    - length, background, background_alpha, pad, sep, label_top, box_style

    Examples
    --------
    >>> import napari
    >>> viewer = napari.Viewer(show=False)  # doctest: +SKIP
    >>> configure_napari_scale_bar(viewer, units="cm")  # doctest: +SKIP
    >>> viewer.close()  # doctest: +SKIP
    """
    if config is None:
        config = ScaleBarConfig()

    # Position mapping: our API -> napari names
    position_map = {
        "lower right": "bottom_right",
        "lower left": "bottom_left",
        "upper right": "top_right",
        "upper left": "top_left",
    }

    viewer.scale_bar.visible = True
    viewer.scale_bar.position = position_map.get(config.position, "bottom_right")
    viewer.scale_bar.font_size = config.font_size

    if units:
        viewer.scale_bar.unit = units

    if config.color:
        viewer.scale_bar.color = config.color
