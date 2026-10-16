"""Construction adapter turning a built layout into publishable owned data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import networkx as nx
import numpy as np

from neurospatial.geometry._binning import Binning
from neurospatial.geometry._context import GridIndex
from neurospatial.layout.base import LayoutEngine


def built_layout_data(
    layout: LayoutEngine,
) -> tuple[
    Binning,
    nx.Graph,
    Sequence[tuple[float, float]] | None,
    str,
    str,
    dict[str, Any],
    GridIndex | None,
]:
    """Take a built layout's binning, connectivity, ranges and provenance once.

    A layout engine supplies its geometry through ``binning()``. An external
    object without that method needs an installed snapshot adapter that
    converts it to a ``Binning``; neither the object nor its parameters are
    retained.

    Parameters
    ----------
    layout : LayoutEngine
        Built layout engine, or an external provider with an installed
        ``"binning"`` snapshot adapter.

    Returns
    -------
    tuple
        The binning, navigation graph, plotting ranges, layout type tag,
        layout category, build parameters and logical grid index (``None``
        when the index follows from the binning alone).

    Raises
    ------
    LayoutNotBuiltError
        If a layout engine has not been built.
    TypeError
        If the object has no ``binning()`` method and no installed adapter.
    """
    produce = getattr(layout, "binning", None)
    if not callable(produce):
        from neurospatial.extensions import _snapshot_provider
        from neurospatial.geometry._explicit import prepare_bins

        binning, graph, ranges, category = prepare_bins(
            _snapshot_provider(layout, "binning")
        )
        return binning, graph, ranges, binning.kind, category, {}, None
    binning = cast("Binning", produce())
    index = GridIndex(
        layout.grid_edges or (),
        layout.grid_shape or (binning.bins.n_bins,),
        np.ones(binning.bins.n_bins, dtype=bool)
        if layout.active_mask is None
        else layout.active_mask,
    )
    return (
        binning,
        layout.connectivity,
        layout.dimension_ranges,
        binning.kind,
        layout.layout_type,
        layout._build_params_used,
        index,
    )
