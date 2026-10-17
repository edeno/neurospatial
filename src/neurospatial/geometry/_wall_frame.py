"""Recover shared stored coordinates for physical walls and retained cells."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from shapely.geometry import LineString, MultiPolygon, Polygon

    from neurospatial.geometry._arena import Arena, _TransformedArena
    from neurospatial.geometry._locators import Locator


def _wall_source_frame(
    locator: Locator, arena: Arena, centers: NDArray[np.float64]
) -> (
    tuple[
        tuple[Polygon | MultiPolygon, ...],
        tuple[LineString, ...],
        NDArray[np.float64],
        NDArray[np.float64],
        tuple[_TransformedArena, ...],
    ]
    | None
):
    """Recover a shared stored frame through retained cell-owner mappings."""
    from neurospatial.geometry._arena import _TransformedArena
    from neurospatial.geometry._locators import SubsetLocator, TransformedLocator
    from neurospatial.geometry._polygon import ClippedLocator, cells_from_locator

    if isinstance(locator, TransformedLocator) and isinstance(arena, _TransformedArena):
        if np.array_equal(locator.inverse, arena.inverse):
            original = (centers - arena.forward[:-1, -1]) @ arena.inverse[:-1, :-1].T
            inner = _wall_source_frame(locator.base, arena.base, original)
            if inner is None:
                return (
                    cells_from_locator(locator.base),
                    arena.base.walls,
                    original,
                    locator.inverse,
                    (arena,),
                )
            cells, walls, original, inverse, frames = inner
            return cells, walls, original, inverse @ locator.inverse, (arena, *frames)
    elif isinstance(locator, (SubsetLocator, ClippedLocator)):
        inner = _wall_source_frame(locator.base, arena, centers)
        if inner is not None:
            cells, walls, original, inverse, frames = inner
            if isinstance(locator, SubsetLocator):
                selected = np.flatnonzero(locator.parent_to_child >= 0)
                selected = selected[np.argsort(locator.parent_to_child[selected])]
                return (
                    tuple(cells[index] for index in selected),
                    walls,
                    original,
                    inverse,
                    frames,
                )
            else:
                import shapely
                from shapely.geometry import MultiPolygon, Polygon

                from neurospatial.geometry._validation import (
                    _polygon_coordinate_errors,
                )

                # Retain the *current* clipped support in the exact wall frame.
                # Project shared vertices once through the actual stored frames.
                # Their existing forward-residual bounds include inverse and
                # large-origin arithmetic without expanding physical support.
                polygons = np.asarray(locator.cells.polygons, dtype=object)
                coordinates = shapely.get_coordinates(polygons)
                unique, indices = np.unique(coordinates, axis=0, return_inverse=True)
                mapped = unique
                projection_errors = np.zeros_like(unique)
                for frame in frames:
                    mapped, projection_errors = frame._project(
                        mapped, projection_errors
                    )
                current_parents = cells_from_locator(locator.base)
                current_errors = [
                    _polygon_coordinate_errors(p) for p in current_parents
                ]
                source_errors = [_polygon_coordinate_errors(p) for p in cells]
                projected = shapely.set_coordinates(polygons, mapped[indices])
                clipped: list[Polygon | MultiPolygon] = []
                signed_permutation = all(
                    np.all(np.isin(frame.inverse[:2, :2], (-1.0, 0.0, 1.0)))
                    and np.all(np.count_nonzero(frame.inverse[:2, :2], axis=0) == 1)
                    and np.all(np.count_nonzero(frame.inverse[:2, :2], axis=1) == 1)
                    for frame in frames
                )
                axis_aligned = all(
                    np.all(
                        np.any(
                            np.diff(shapely.get_coordinates(ring), axis=0) == 0, axis=1
                        )
                    )
                    for polygon in (*cells, *current_parents)
                    for ring in shapely.get_rings(shapely.get_parts(polygon))
                )
                # Signed permutations recover the stored axis coordinates
                # exactly on axis-aligned parent geometry. Other inverses or
                # rounded diagonal boundaries may leave a parent-edge gap.
                # Refuse uncertain wall clearance instead of expanding cells.
                exact_axes = signed_permutation and axis_aligned
                projection_error = float(
                    np.max(np.linalg.norm(projection_errors, axis=1))
                )
                projection_error += float(
                    np.max(
                        np.linalg.norm(
                            np.asarray(current_errors) @ np.abs(inverse[:2, :2]).T,
                            axis=1,
                        )
                    )
                )
                for child, (polygon, parent) in enumerate(
                    zip(projected, locator.mapping.parent_bin_indices, strict=True)
                ):
                    retained = polygon.intersection(cells[parent])
                    if (
                        not isinstance(retained, (Polygon, MultiPolygon))
                        or retained.is_empty
                    ):
                        raise ValueError(
                            "Current clipped cells lose their source-frame support."
                        )
                    if not exact_axes:
                        error = projection_error + float(
                            np.linalg.norm(source_errors[parent])
                        )
                        for wall in walls:
                            interior = wall.intersection(cells[parent]).difference(
                                cells[parent].boundary
                            )
                            if (
                                interior.length > 0
                                and interior.intersection(retained)
                                .difference(retained.boundary)
                                .length
                                == 0
                                and interior.distance(retained) <= error
                            ):
                                raise ValueError(
                                    f"Physical wall clearance from clipped bin {child} is unresolved in the source frame. "
                                    "Declare walls before clipping or transforming, or recenter coordinates."
                                )
                    clipped.append(retained)
                return tuple(clipped), walls, original, inverse, frames
    return None
