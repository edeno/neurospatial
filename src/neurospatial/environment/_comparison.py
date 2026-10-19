"""Whether two environments' bins mean the same places, connected the same way.

Analyses that pair arrays over bins from two environments (decoding with rates
estimated elsewhere, comparing rate maps between conditions) first ask whether
the bins mean the same places in the same coordinates, and then whether the
space is connected the same way. A door or barrier experiment keeps the bins
but changes connectivity or walls, so the two answers are reported separately.
Equal bin counts or array shapes never establish either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from shapely.ops import unary_union

from neurospatial.geometry import (
    PolygonCells,
    encode,
    encode_arena,
    materialize_cells,
)

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.geometry import Arena, Binning, SpatialBins


class EnvironmentDifferences(NamedTuple):
    """The first difference in bin meaning and in topology, or None for each.

    Attributes
    ----------
    bins : str or None
        First difference in what the bins mean: ``"coordinate systems"``,
        ``"units or coordinate frames"``, ``"bin coordinates or ordering"``,
        ``"bin measures"``, ``"polygon cell geometry"`` or
        ``"bin-boundary ownership"``.
    topology : str or None
        First difference in how the space is connected:
        ``"weighted connectivity"`` or ``"continuous arena or physical walls"``.
    """

    bins: str | None
    topology: str | None


def _environment_differences(
    first: Environment, second: Environment
) -> EnvironmentDifferences:
    """Compare two environments' bin meaning and topology.

    Parameters
    ----------
    first, second : Environment
        The environments to compare.

    Returns
    -------
    EnvironmentDifferences
        ``(None, None)`` when the bins mean the same places in the same
        coordinates and the space is connected the same way.

    Notes
    -----
    Bin and topology checks run independently, so a topology difference never
    hides a bin difference. Environments that share published geometry, such
    as ``env.with_metadata(...)`` copies, differ at most in their coordinate
    labels.
    """
    if first is second:
        return EnvironmentDifferences(None, None)
    if first.coordinate_system != second.coordinate_system:
        labels = "coordinate systems"
    elif first.units != second.units or first.frame != second.frame:
        labels = "units or coordinate frames"
    else:
        labels = None
    if first._geometry is second._geometry:
        return EnvironmentDifferences(labels, None)
    return EnvironmentDifferences(
        labels or _bin_difference(first, second), _topology_difference(first, second)
    )


def _bin_difference(first: Environment, second: Environment) -> str | None:
    """Return the first difference between published bins with equal labels."""
    if not np.array_equal(first.bin_centers, second.bin_centers):
        return "bin coordinates or ordering"
    first_binning, second_binning = first._binning, second._binning
    if not _same_measures(first_binning.bins, second_binning.bins):
        return "bin measures"
    if isinstance(first_binning.shapes, PolygonCells) or isinstance(
        second_binning.shapes, PolygonCells
    ):
        return _polygon_cell_difference(first_binning, second_binning)
    return None


def _same_measures(first: SpatialBins, second: SpatialBins) -> bool:
    """Whether two sets of bins have equal measures, in equal dimensions."""
    return np.array_equal(first.sizes, second.sizes) and np.array_equal(
        first.measure_dims, second.measure_dims
    )


def _polygon_cell_difference(first: Binning, second: Binning) -> str | None:
    """Compare exact polygon cells, then which bin owns each cell boundary."""
    first_cells, second_cells = materialize_cells(first), materialize_cells(second)
    if len(first_cells) != len(second_cells) or any(
        not left.equals(right)
        for left, right in zip(first_cells, second_cells, strict=True)
    ):
        return "polygon cell geometry"
    try:
        first_meta, first_arrays = encode(first.locator, "locator")
        second_meta, second_arrays = encode(second.locator, "locator")
    except ValueError:
        return "bin-boundary ownership"
    if (
        first_meta != second_meta
        or first_arrays.keys() != second_arrays.keys()
        or any(
            not np.array_equal(value, second_arrays[key])
            for key, value in first_arrays.items()
        )
    ):
        return "bin-boundary ownership"
    return None


def _topology_difference(first: Environment, second: Environment) -> str | None:
    """Return the first difference in connectivity, arena or walls."""
    first_graph, second_graph = first.connectivity, second.connectivity
    if first_graph.number_of_edges() != second_graph.number_of_edges() or any(
        not second_graph.has_edge(u, v)
        or data["distance"] != second_graph.edges[u, v]["distance"]
        for u, v, data in first_graph.edges(data=True)
    ):
        return "weighted connectivity"
    if not _same_arena(first._arena, second._arena):
        return "continuous arena or physical walls"
    return None


def _same_arena(first: Arena, second: Arena) -> bool:
    """Compare continuous meaning independently of polygon ring storage order."""
    if first is second:
        return True
    if any(
        getattr(first, name) != getattr(second, name)
        for name in (
            "n_dims",
            "coordinate_kind",
            "domain_known",
            "physical_footprint_known",
        )
    ):
        return False
    first_domain, second_domain = first.polygonal_domain(), second.polygonal_domain()
    if first_domain is not None or second_domain is not None:
        return (
            first_domain is not None
            and second_domain is not None
            and first_domain.equals(second_domain)
            and unary_union(first.walls).equals(unary_union(second.walls))
        )
    try:
        first_meta, first_arrays = encode_arena(first)
        second_meta, second_arrays = encode_arena(second)
    except ValueError:
        return False
    return (
        first_meta == second_meta
        and first_arrays.keys() == second_arrays.keys()
        and all(
            np.array_equal(value, second_arrays[key])
            for key, value in first_arrays.items()
        )
    )
