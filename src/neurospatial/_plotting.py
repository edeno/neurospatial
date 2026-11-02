"""Helpers shared by the matplotlib plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_polar(
    ax: Axes, caller: str, *, alternative: str = "pass `polar=False`"
) -> None:
    """Refuse an axes that is not polar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes the caller is about to draw angles on.
    caller : str
        Name of the calling function, for the message.
    alternative : str, default "pass `polar=False`"
        What to do instead of making a polar axes, for the message.

    Raises
    ------
    ValueError
        If ``ax`` does not use the polar projection.
    """
    if ax.name != "polar":
        raise ValueError(
            _format_error(
                f"{caller} draws on a polar axes, but ax has the "
                f"'{ax.name}' projection.",
                why="angles are drawn around a circle, which a Cartesian axes "
                "cannot show",
                fix="create the axes with `fig, ax = plt.subplots(subplot_kw="
                f"{{'projection': 'polar'}})`, or {alternative}",
            )
        )
