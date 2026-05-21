"""Shared matplotlib styling for neurospatial example notebooks.

This module centralizes the colour palette, figure size, and font
sizes used across the tutorial notebooks. Each notebook can call
``apply_style()`` once at the top to opt in, instead of repeating
the same rcParams block in every file.

The palette is **Okabe-Ito** (Wong, 2011 - *Points of view: Color
blindness*, Nature Methods 8:441). It is colour-vision-deficiency
safe and prints well in greyscale.

Usage
-----
At the top of an example notebook::

    from _style import apply_style, OKABE_ITO

    apply_style()

To override the figure size for a single notebook::

    apply_style(figsize=(14, 10))

References
----------
Wong, B. (2011). Points of view: Color blindness. Nature Methods,
8(6), 441. https://doi.org/10.1038/nmeth.1618
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from cycler import cycler

OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)


def apply_style(
    *,
    figsize: tuple[float, float] = (10, 8),
    font_size: float = 11,
    palette: Sequence[str] = OKABE_ITO,
) -> None:
    """Apply the shared neurospatial-examples matplotlib style.

    Parameters
    ----------
    figsize : tuple of float, optional
        Default figure size in inches. Default is ``(10, 8)``.
    font_size : float, optional
        Base font size. Title / label / tick sizes scale from this.
        Default is ``11``.
    palette : sequence of str, optional
        Hex colour list used as the property cycle. Default is the
        Okabe-Ito colour-vision-deficiency-safe palette.
    """
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.titlesize"] = font_size + 2
    plt.rcParams["axes.labelsize"] = font_size + 1
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["figure.titlesize"] = font_size + 3
    plt.rcParams["axes.prop_cycle"] = cycler(color=list(palette))
