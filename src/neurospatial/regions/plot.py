"""Draw the regions of a :class:`~neurospatial.Regions` collection on Matplotlib axes.

Matplotlib is imported only when :func:`plot_regions` is called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry.polygon import orient

from neurospatial._exceptions import RegionNotFoundError, _format_error

from .core import Region
from .ops import _named_regions, _transform_points, _transform_polygon

if TYPE_CHECKING:
    import matplotlib.axes
    from matplotlib.path import Path as MplPath
    from shapely.geometry import Polygon

    from ..ops.transforms import SpatialTransform


def _polygon_path(polygon: Polygon) -> MplPath:
    """Return a compound path that leaves the polygon's holes unfilled.

    Matplotlib fills by the nonzero winding rule, so a hole drawn in the same
    direction as the exterior is filled. ``orient`` makes the exterior
    counter-clockwise and every hole clockwise first.
    """
    from matplotlib.path import Path as MplPath

    oriented = orient(polygon, sign=1.0)
    paths = []
    for ring in (oriented.exterior, *oriented.interiors):
        vertices = np.asarray(ring.coords)
        codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
        codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
        paths.append(MplPath(vertices, codes))
    return MplPath.make_compound_path(*paths)


def plot_regions(
    regions: Mapping[str, Region],
    *,
    ax: matplotlib.axes.Axes | None = None,
    region_names: Sequence[str] | None = None,
    default_kwargs: Mapping[str, Any] | None = None,
    world_to_pixel: SpatialTransform | None = None,
    add_legend: bool = True,
    **per_region_kwargs: Mapping[str, Any],
) -> matplotlib.axes.Axes:
    """Draw regions on Matplotlib axes: points as markers, polygons as patches.

    Parameters
    ----------
    regions : Mapping[str, Region]
        Collection to draw. Point regions must be 2-D.
    ax : matplotlib.axes.Axes, optional
        Destination Axes. If None, uses the current Axes (``plt.gca()``).
    region_names : str or sequence of str, optional
        Regions to draw. A single string is one name. If None, every region is
        drawn. A name not in ``regions`` raises
        :class:`~neurospatial.RegionNotFoundError` with the closest name.
    default_kwargs : dict, optional
        Matplotlib keyword arguments applied to every region.
    world_to_pixel : SpatialTransform, optional
        Maps ``(n, 2)`` region coordinates to plot coordinates before drawing,
        for example cm to pixels to overlay regions on a video frame.
    add_legend : bool, default True
        Add a legend labeling each drawn region, placed outside the Axes, to
        the right (``loc="center left", bbox_to_anchor=(1, 0.5)``).
    **per_region_kwargs
        Per-region Matplotlib keyword arguments, keyed by region name::

            plot_regions(regions, Stem={"alpha": 0.1}, Reward={"edgecolor": "red"})

        A name not in ``regions`` raises
        :class:`~neurospatial.RegionNotFoundError`.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on.

    Raises
    ------
    RegionNotFoundError
        If a name in ``region_names`` or ``per_region_kwargs`` is not in
        ``regions``.
    ValueError
        If a point region to draw is not 2-D.

    Notes
    -----
    Options are merged per region in increasing precedence:
    ``default_kwargs``, then ``region.metadata["plot_kwargs"]``, then
    ``per_region_kwargs[name]``. Region metadata therefore overrides
    ``default_kwargs``; :func:`~neurospatial.regions.load_cvat_xml`, for
    example, stores each label's color there. Unless overridden, ``alpha`` is
    0.5, the legend label is the region name, and point regions use
    ``marker="x"`` and ``s=100``. ``color`` sets a polygon's face color when
    ``facecolor`` is not given.

    Axis limits that are still autoscaling grow to include the regions; limits
    you set beforehand (for example the frame of a video image) are kept.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from shapely.geometry import box
    >>> from neurospatial import Regions
    >>> from neurospatial.regions import plot_regions
    >>> regions = Regions()
    >>> _ = regions.add("start_box", polygon=box(0.0, 0.0, 20.0, 20.0))
    >>> _ = regions.add("reward_well", point=(80.0, 80.0))
    >>> fig, ax = plt.subplots()
    >>> ax = plot_regions(regions, ax=ax, start_box={"facecolor": "tab:green"})
    >>> [text.get_text() for text in ax.get_legend().get_texts()]
    ['start_box', 'reward_well']
    >>> plt.close(fig)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch

    region_names = _named_regions(regions, region_names)
    for name in per_region_kwargs:
        if name not in regions:
            raise RegionNotFoundError(name, available=list(regions), argument=None)
    for name in region_names:
        region = regions[name]
        if region.n_dims != 2:
            raise ValueError(
                _format_error(
                    f"Region {name!r} is a {region.n_dims}-D point; "
                    "plot_regions draws 2-D regions only.",
                    fix="leave it out with region_names=, or draw it on an "
                    "environment of matching dimension with env.plot()",
                )
            )

    if ax is None:
        ax = plt.gca()

    for name in region_names:
        region = regions[name]
        opts: dict[str, Any] = dict(default_kwargs or {})
        opts.update(region.metadata.get("plot_kwargs", {}))
        opts.update(per_region_kwargs.get(name, {}))

        label = opts.pop("label", name)
        alpha = opts.pop("alpha", 0.5)

        where = f"region {name!r}"
        if region.kind == "point":
            xy = _transform_points(
                region.data[None], world_to_pixel, name="world_to_pixel", where=where
            )
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                marker=opts.pop("marker", "x"),
                s=opts.pop("s", 100),
                alpha=alpha,
                label=label,
                **opts,
            )
        else:
            polygon = _transform_polygon(
                region.data, world_to_pixel, name="world_to_pixel", where=where
            )
            ax.add_patch(
                PathPatch(
                    _polygon_path(polygon),
                    label=label,
                    facecolor=opts.pop("facecolor", opts.pop("color", None)),
                    alpha=alpha,
                    **opts,
                )
            )

    handles, labels = ax.get_legend_handles_labels()
    if handles and add_legend:
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    # add_patch does not rescale; autoscale_view grows only the axes still
    # autoscaling, so limits the caller set (a video frame, env.plot) stay.
    ax.autoscale_view()
    return ax
