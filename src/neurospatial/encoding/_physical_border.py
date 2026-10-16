"""Border scores from measured wall contact and distance to actual walls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import unary_union

if TYPE_CHECKING:
    from neurospatial.arena import Arena
    from neurospatial.environment._protocols import BorderConsumer


@dataclass(frozen=True)
class _PhysicalBorderContact:
    """Exact cell and wall contact geometry in a shared source frame."""

    cells: tuple[shapely.Geometry, ...]
    walls: shapely.Geometry
    forward: NDArray[np.float64] | None


def _prepare_physical_border(
    env: BorderConsumer,
    firing_rate: NDArray[np.float64],
    walls: shapely.Geometry,
) -> _PhysicalBorderContact:
    """Require represented, observed wall coverage before interpreting activity."""
    from neurospatial.arena import _CombinedArena, _TransformedArena
    from neurospatial.layout._composition import CombinedLocator, composition_view
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator
    from neurospatial.layout._polygon import ClippedLocator, materialize_cells

    def independent_frame(locator: object) -> bool:
        if isinstance(locator, TransformedLocator):
            return True
        if isinstance(locator, (SubsetLocator, ClippedLocator)):
            return independent_frame(locator.base)
        if isinstance(locator, CombinedLocator):
            return any(independent_frame(child.locator) for child in locator.children)
        return False

    def declares_walls(arena: Arena) -> bool:
        if isinstance(arena, _TransformedArena):
            return declares_walls(arena.base)
        if isinstance(arena, _CombinedArena):
            return all(declares_walls(child) for child in arena.children)
        return bool(arena.walls) or arena.physical_footprint_known

    # A field in a component without declared walls would otherwise be
    # scored against the walls of other components.
    composition = composition_view(env._binning.locator)
    arena = env.arena
    while isinstance(arena, _TransformedArena):
        arena = arena.base
    if isinstance(arena, _CombinedArena) and not declares_walls(arena):
        missing = [
            index
            for index, child in enumerate(arena.children)
            if not declares_walls(child)
        ]
        keys = composition[0].keys if composition is not None else ()
        named = (
            ", ".join(repr(keys[index]) for index in missing)
            if len(keys) == len(arena.children)
            else f"{len(missing)} of {len(arena.children)}"
        )
        raise NotImplementedError(
            f"Physical border scores need declared walls for every combined component; component(s) {named} have none, "
            "so fields there would be scored against walls in other components. Build each component with "
            "measured walls (Environment.from_polygon(...), OpenArena or MaskArena(physical_footprint_known=True)) "
            "before combining, or explicitly use boundary_source='analysis'."
        )
    if composition is not None and any(
        independent_frame(child.locator) for child in composition[0].children
    ):
        raise NotImplementedError(
            "Physical border contact needs a shared source frame for combined cells and walls. Combine components before applying a shared transform, or explicitly use boundary_source='analysis'."
        )

    try:
        cells = materialize_cells(env._binning)
    except ValueError as exc:
        raise NotImplementedError(
            "Physical border coverage needs exact planar cells; supply supported grid or polygon cells, or choose boundary_source='analysis'."
        ) from exc
    from neurospatial.environment._arena import _wall_source_frame
    from neurospatial.ops._boundaries import physical_walls

    frame = _wall_source_frame(env._binning.locator, env.arena, env.bin_centers)
    contact_cells, contact_walls = cells, walls
    forward = None
    if frame is not None:
        contact_cells, _internal, _centers, inverse, frames = frame
        contact_walls = physical_walls(frames[-1].base)
        forward = np.asarray(np.linalg.inv(inverse), dtype=np.float64)

    represented = unary_union(contact_cells)
    wall_contact = (
        shapely.length(
            shapely.intersection(np.asarray(contact_cells, dtype=object), contact_walls)
        )
        > 0
    )
    if contact_walls.difference(represented).length > 0 or np.any(
        wall_contact & ~np.isfinite(firing_rate)
    ):
        raise NotImplementedError(
            "Physical-wall coverage is unavailable: some measured wall segments have no retained cells "
            "with finite firing rates. Build bins that reach every measured wall and collect sufficient "
            "observations there. For rotated rectangular arenas, build bins in the arena-aligned frame "
            "and transform the bins, arena and observations together. Adding OpenArena or clipping "
            "existing bins cannot supply missing bins or observations. Alternatively, explicitly use "
            "boundary_source='analysis' to score the retained analysis boundary."
        )
    return _PhysicalBorderContact(tuple(contact_cells), contact_walls, forward)


def _physical_border_score(
    env: BorderConsumer,
    field_bins: NDArray[np.int64],
    walls: shapely.Geometry,
    metric: Literal["geodesic", "euclidean"],
    geometry: _PhysicalBorderContact,
) -> float:
    """Use observed wall-length coverage and mean center-to-wall distance.

    Geodesic distances include a center-to-wall seed length and then paths
    through the retained navigation graph. Unrepresented routes are unknown,
    not zero distance, and any unreachable field bin makes that score NaN.
    """
    contact = geometry.walls.intersection(
        unary_union([geometry.cells[index] for index in field_bins])
    )
    if geometry.forward is not None:
        from neurospatial.arena import _mapped_geometry

        contact = _mapped_geometry(contact, geometry.forward)
    coverage = contact.length / walls.length
    direct = np.asarray(
        shapely.distance(shapely.points(env.bin_centers), walls), dtype=float
    )
    if metric == "euclidean":
        distances = direct[field_bins]
    else:
        # Contact with a wall is a seed, not an invented zero-distance bin.
        seeds = [
            i
            for i, cell in enumerate(geometry.cells)
            if cell.intersection(geometry.walls).length > 0
        ]
        if not seeds:
            return float("nan")
        graph = env.connectivity.copy()
        source = object()
        graph.add_weighted_edges_from(
            [(source, index, direct[index]) for index in seeds], weight="distance"
        )
        try:
            paths = nx.single_source_dijkstra_path_length(
                graph, source, weight="distance"
            )
        except (ValueError, nx.NetworkXException) as exc:
            raise RuntimeError(
                "border_score: physical-wall geodesic computation failed; check navigation distances or use metric='euclidean'."
            ) from exc
        distances = np.array([paths.get(int(index), np.nan) for index in field_bins])
        if not np.all(np.isfinite(distances)):
            return float("nan")
    # The wall diameter is invariant to rigid frame changes. Work one hull
    # vertex at a time instead of allocating a dense all-pairs distance array.
    hull = shapely.get_coordinates(walls.convex_hull)
    extent = max(float(np.hypot.reduce(hull - point, axis=1).max()) for point in hull)
    normalized = float(np.mean(distances)) / extent
    denominator = coverage + normalized
    if denominator == 0:
        return float("nan")
    return float(np.clip((coverage - normalized) / denominator, -1, 1))
