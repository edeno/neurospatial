"""Utility functions for the animation module.

This module provides shared helper functions used across multiple animation
components to ensure consistent behavior and messaging.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


def _emit_status(message: str) -> None:
    """Emit a user-facing status/progress message.

    Written to ``stderr`` so it is visible by default -- ``logger.info`` output
    is silent unless the application configured logging, which silently hid the
    dry-run estimate and render/encode progress in a plain notebook or script.
    Also logged at INFO for callers that consume the module logger.
    """
    print(message, file=sys.stderr, flush=True)
    logger.info(message)


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


def _pickling_guidance(subject: Literal["environment", "overlays"]) -> str:
    """Return the HOW part of an error raised when parallel rendering cannot pickle.

    Parallel video rendering sends the environment and the overlay data to
    worker processes, which requires pickling them. Only objects the user
    supplied can make that fail: functions defined inline (lambdas, local
    functions), open files or readers, locks and threads. An environment's
    cached analyses are never pickled, so they are never the cause.

    Parameters
    ----------
    subject : {"environment", "overlays"}
        What failed to pickle. It selects where to look for the object.

    Returns
    -------
    str
        Numbered fixes: replace the object with plain data, render serially,
        or use a backend that does not pickle.

    Examples
    --------
    >>> print(_pickling_guidance("overlays"))  # doctest: +ELLIPSIS
    Fix: use one of these approaches:
      1. Keep only plain data in the overlays...
    """
    if subject == "environment":
        replace = (
            "  1. Replace what you attached to the environment that cannot be\n"
            "     pickled (a lambda or local function, an open file, a lock) with\n"
            "     plain data such as numbers, strings and NumPy arrays. Look in\n"
            "     region metadata, graph node and edge attributes, and attributes\n"
            "     you set on the environment. Check with: pickle.dumps(env)\n\n"
        )
    else:
        replace = (
            "  1. Keep only plain data in the overlays: NumPy arrays, numbers and\n"
            "     strings. Define functions at module level instead of lambdas or\n"
            "     local functions, and pass file paths instead of open readers.\n"
            "     Check with: pickle.dumps(overlay)\n\n"
        )
    return (
        "Fix: use one of these approaches:\n"
        f"{replace}"
        "  2. Render serially (slower; nothing is pickled):\n"
        "     env.animate_fields(..., n_workers=1)\n\n"
        "  3. Use a backend that does not pickle:\n"
        "     env.animate_fields(..., backend='html')"
    )


__all__ = ["_pickling_guidance", "managed_figure"]
