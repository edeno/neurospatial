"""Construction adapter turning a built layout into publishable owned data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from neurospatial._exceptions import _format_error
from neurospatial.geometry import Binning
from neurospatial.layout._labels import canonical_label
from neurospatial.layout.base import LayoutEngine


def _built_layout_data(
    layout: LayoutEngine,
) -> tuple[Binning, Mapping[str, Any]]:
    """Take a built layout's binning once, with its build choices.

    A layout engine supplies its geometry through ``binning()`` alone. The
    navigation graph, plotting ranges and grid index all follow from that
    record, so an engine needs only ``build(...)`` and ``binning()``. Two
    descriptive members are read when an engine has them: its
    ``layout_type`` label, which becomes the binning's ``kind`` (otherwise
    the binning's own ``kind`` is the label), and the build arguments it
    records with ``capture_build_params`` (otherwise none). Either label is
    spelled canonically when it names a ``LayoutType``. An external
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
    binning : Binning
        The layout's bins, with the layout label as ``kind``.
    build_params : mapping of str to object
        The recorded build choices.

    Raises
    ------
    LayoutNotBuiltError
        If a layout engine has not been built.
    TypeError
        If the object has no ``binning()`` method and no installed adapter.
    """
    produce = getattr(layout, "binning", None)
    if not callable(produce):
        from neurospatial.geometry import prepare_bins, snapshot_provider

        binning = prepare_bins(snapshot_provider(layout, "binning"))
        return replace(binning, kind=canonical_label(binning.kind)), {}
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
    kind = canonical_label(label if isinstance(label, str) else binning.kind)
    return (
        binning if kind == binning.kind else replace(binning, kind=kind),
        params if isinstance(params, Mapping) else {},
    )
