"""Utility functions for the animation module.

This module provides shared helper functions used across multiple animation
components to ensure consistent behavior and messaging.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure


@contextmanager
def managed_figure(fig: Figure) -> Iterator[Figure]:
    """Yield a matplotlib figure, guaranteeing it is closed on exit.

    Centralizes the ``try/finally: plt.close(fig)`` cleanup pattern used by
    the rendering helpers so that an exception raised while drawing (for
    example, inside ``env.plot_field()`` or an overlay renderer) cannot leak
    the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to manage. It is closed via ``plt.close(fig)`` when the
        ``with`` block exits, whether normally or because of an exception.

    Yields
    ------
    matplotlib.figure.Figure
        The same figure that was passed in, for convenient binding.

    Examples
    --------
    >>> import matplotlib

    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> from neurospatial.animation._utils import managed_figure
    >>> before = len(plt.get_fignums())
    >>> with managed_figure(plt.figure()) as fig:
    ...     _ = fig.add_subplot(1, 1, 1)
    >>> len(plt.get_fignums()) == before
    True
    """
    import matplotlib.pyplot as plt

    try:
        yield fig
    finally:
        plt.close(fig)


def _pickling_guidance(n_workers: int | None = None) -> str:
    """Generate consistent HOW guidance for pickle-related errors.

    This helper centralizes the guidance text for pickle validation errors,
    ensuring both `_validate_pickle_ability()` and `_validate_env_pickleable()`
    provide identical solutions to users.

    Parameters
    ----------
    n_workers : int | None, optional
        Number of parallel workers being used. If provided, included in
        example code snippets. If None, uses placeholder.

    Returns
    -------
    str
        Formatted guidance text with numbered options for resolving
        pickle-ability issues.

    Examples
    --------
    >>> guidance = _pickling_guidance(n_workers=4)
    >>> print(guidance)  # doctest: +ELLIPSIS
    HOW: Fix the issue using one of these approaches:
    ...

    Notes
    -----
    The guidance includes three main options:
    1. Clear caches before rendering (most common fix)
    2. Use serial rendering with n_workers=1 (slower but always works)
    3. Use a backend that doesn't require pickling (html)

    This function is internal and should not be used outside the animation
    module.
    """
    n_workers_str = str(n_workers) if n_workers is not None else "N"

    return (
        "HOW: Fix the issue using one of these approaches:\n"
        "  1. Clear caches before rendering (most common fix):\n"
        "     env.clear_cache()  # Remove cached unpickleable objects\n"
        f"     env.animate_fields(..., n_workers={n_workers_str})\n\n"
        "  2. Use serial/single-threaded rendering (slower but always works):\n"
        "     env.animate_fields(..., n_workers=1)\n\n"
        "  3. Use a backend that doesn't require pickling:\n"
        "     env.animate_fields(..., backend='html')  # HTML doesn't pickle"
    )


__all__ = ["_pickling_guidance", "managed_figure"]
