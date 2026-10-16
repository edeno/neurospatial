"""Construction adapter turning a built layout into publishable owned data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from neurospatial._exceptions import _format_error
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._explicit import _category, binning_ranges
from neurospatial.geometry._graph import _graph_from_binning
from neurospatial.layout.base import LayoutEngine


def built_layout_data(
    layout: LayoutEngine,
) -> tuple[
    Binning,
    nx.Graph,
    tuple[tuple[float, float], ...] | None,
    str,
    str,
    Mapping[str, Any],
]:
    """Take a built layout's binning once and derive everything else from it.

    A layout engine supplies its geometry through ``binning()`` alone. The
    navigation graph, plotting ranges and grid index all follow from that
    record, so an engine needs only ``build(...)`` and ``binning()``. Two
    descriptive labels are read when an engine has them: its ``layout_type``
    category (otherwise inferred from the binning) and the build arguments it
    records with ``capture_build_params`` (otherwise none). An external
    object without ``binning()`` needs an installed snapshot adapter that
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
        layout category and build parameters.

    Raises
    ------
    LayoutNotBuiltError
        If a layout engine has not been built.
    TypeError
        If the object has no ``binning()`` method and no installed adapter.
    """
    produce = getattr(layout, "binning", None)
    if not callable(produce):
        from neurospatial.geometry._adapters import _snapshot_provider
        from neurospatial.geometry._explicit import prepare_bins

        binning, graph, ranges, category = prepare_bins(
            _snapshot_provider(layout, "binning")
        )
        return binning, graph, ranges, binning.kind, category, {}
    binning = produce()
    if type(binning) is not Binning:
        raise TypeError(
            _format_error(
                f"{type(layout).__name__}.binning() returned "
                f"{type(binning).__name__}, not a Binning.",
                why=(
                    "Why: the environment derives its graph, ranges and grid "
                    "index from the Binning record alone."
                ),
                fix=(
                    "return neurospatial.binning.Binning(kind, bins, locator, "
                    "faces, ...) built from the engine's state"
                ),
            )
        )
    label = getattr(layout, "layout_type", None)
    params = getattr(layout, "_build_params_used", None)
    return (
        binning,
        _graph_from_binning(binning),
        binning_ranges(binning),
        binning.kind,
        label if isinstance(label, str) else _category(binning),
        params if isinstance(params, Mapping) else {},
    )
