"""Resolve the ``show_regions`` choice of an animation to region names."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from neurospatial._exceptions import RegionNotFoundError

if TYPE_CHECKING:
    from neurospatial.environment import Environment


def shown_region_names(
    env: Environment, show_regions: bool | Sequence[str]
) -> list[str]:
    """Return the names ``show_regions`` selects, in order.

    ``True`` selects every region and ``False`` none. A string is one name. A
    list must name
    regions of ``env``: an unknown name raises
    :class:`~neurospatial.RegionNotFoundError` with the closest name, instead
    of the region silently missing from the animation.
    """
    if isinstance(show_regions, bool):
        return list(env.regions) if show_regions else []
    names = [show_regions] if isinstance(show_regions, str) else list(show_regions)
    for name in names:
        if name not in env.regions:
            raise RegionNotFoundError(
                name,
                available=list(env.regions),
                argument="show_regions",
                in_list=True,
            )
    return names
