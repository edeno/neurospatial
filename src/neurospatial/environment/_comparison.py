"""Whether two environments' bins mean the same places, connected the same way.

Analyses that pair arrays over bins from two environments (decoding with rates
estimated elsewhere, comparing rate maps between conditions) first ask whether
the bins mean the same places in the same coordinates, and then whether the
space is connected the same way. A door or barrier experiment keeps the bins
but changes connectivity or walls, so the two answers are reported separately.
Equal bin counts or array shapes never establish either.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
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


class _EnvironmentDifferences(NamedTuple):
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
) -> _EnvironmentDifferences:
    """Compare two environments' bin meaning and topology.

    Parameters
    ----------
    first, second : Environment
        The environments to compare.

    Returns
    -------
    _EnvironmentDifferences
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
        return _EnvironmentDifferences(None, None)
    if first.coordinate_system != second.coordinate_system:
        labels = "coordinate systems"
    elif first.units != second.units or first.frame != second.frame:
        labels = "units or coordinate frames"
    else:
        labels = None
    if first._geometry is second._geometry:
        return _EnvironmentDifferences(labels, None)
    return _EnvironmentDifferences(
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
    if not _same_weighted_edges(first.connectivity, second.connectivity):
        return "weighted connectivity"
    if not _same_arena(first._arena, second._arena):
        return "continuous arena or physical walls"
    return None


def _mapped_topology_difference(
    first: Environment, second: Environment, bin_pairs: NDArray[np.integer]
) -> str | None:
    """Compare topology through a one-to-one pairing of bins.

    Parameters
    ----------
    first, second : Environment
        The environments to compare.
    bin_pairs : ndarray of int, shape (n_pairs, 2)
        Rows of ``(bin of first, bin of second)`` that are the same place;
        each bin appears at most once in its column.

    Returns
    -------
    str or None
        The first topology difference, or None:

        - ``"weighted connectivity"``: two paired bins are neighbors in one
          environment and not the other, or neighbors at different distances.
        - ``"connections from mapped bins to bins outside the mapping"``: a
          paired bin has a neighbor outside the mapping, in either
          environment, with no counterpart in the other: no unpaired
          neighbor at the same center and the same distance.
        - ``"continuous arena or physical walls"``: the arenas or walls
          differ.

    Notes
    -----
    The topology is the same when every paired bin's full set of neighbors,
    with edge distances, corresponds in the two environments: a paired
    neighbor through ``bin_pairs``, an unpaired neighbor by its center
    (compared exactly, as bin centers are in ``_environment_differences``).
    A smoothing over the graph then used the same neighbors for every paired
    bin. Identical environments under any partial mapping, a renumbered copy,
    and a mapping that covers whole connected components keep the topology;
    a subset compared with its parent does not, because the parent's border
    bins have neighbors the subset lacks. The arena and walls are compared
    whole, as in ``_environment_differences``.
    """
    first_bins, second_bins = bin_pairs[:, 0].tolist(), bin_pairs[:, 1].tolist()
    to_first = dict(zip(second_bins, first_bins, strict=True))
    first_graph = first.connectivity.subgraph(first_bins)
    second_graph = nx.relabel_nodes(second.connectivity.subgraph(second_bins), to_first)
    if not _same_weighted_edges(first_graph, second_graph):
        return "weighted connectivity"
    own_labels = {bin_id: bin_id for bin_id in first_bins}
    if _outside_neighbors(first, own_labels) != _outside_neighbors(second, to_first):
        return "connections from mapped bins to bins outside the mapping"
    if not _same_arena(first._arena, second._arena):
        return "continuous arena or physical walls"
    return None


def _outside_neighbors(
    env: Environment, labels: dict[int, int]
) -> Counter[tuple[int, tuple[float, ...], float]]:
    """Mapped bins' unmapped neighbors as ``(label, center, distance)`` counts.

    ``labels`` maps each mapped bin of ``env`` to its ID in the shared
    numbering, so the same neighbor of the same place counts alike in both
    environments.
    """
    centers = env.bin_centers
    return Counter(
        (labels[inside], tuple(centers[outside].tolist()), distance)
        for inside, outside, distance in nx.edge_boundary(
            env.connectivity, labels, data="distance"
        )
    )


def _same_weighted_edges(first: nx.Graph, second: nx.Graph) -> bool:
    """Whether two graphs have the same edges with the same distances."""
    return first.number_of_edges() == second.number_of_edges() and all(
        second.has_edge(u, v) and data["distance"] == second.edges[u, v]["distance"]
        for u, v, data in first.edges(data=True)
    )


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
