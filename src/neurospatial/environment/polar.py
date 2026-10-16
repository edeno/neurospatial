"""Construction adapter for egocentric radius/angle coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from neurospatial.geometry._coordinates import (
    _POLAR_COORDINATES,
    EgocentricPolarSystem,
)
from neurospatial.layout.engines._polar import _PolarGridLayout

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.layout.base import LayoutEngine

_POLAR_EXPLANATION = (
    f"{_POLAR_COORDINATES} Use connectivity-graph operations instead (neighbors, "
    "path_between, reachable_from, smooth, or distance_to(metric='geodesic')), "
    "or build an allocentric Cartesian env via Environment.from_samples / "
    "from_polygon / from_grid_mask."
)


def _bin_count(span: float, bin_size: float) -> int:
    """Cover a range without adding a bin for one-ULP division round-off."""
    quotient = np.nextafter(span / bin_size, -np.inf)
    return max(1, int(np.ceil(quotient)))


def _create_polar_environment(
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    distance_bin_size: float,
    angle_bin_size: float,
    *,
    circular_angle: bool | None = None,
    connect_diagonal_neighbors: bool = True,
    name: str = "",
    units: str | None = None,
    frame: str | None = None,
) -> Environment[EgocentricPolarSystem]:
    """Build owned annular sectors with physical polar distances and measures."""
    from neurospatial.environment.core import Environment

    if distance_bin_size <= 0:
        raise ValueError(f"distance_bin_size must be positive, got {distance_bin_size}")
    if angle_bin_size <= 0:
        raise ValueError(f"angle_bin_size must be positive, got {angle_bin_size}")
    if distance_range[0] >= distance_range[1]:
        raise ValueError(f"distance_range must have min < max, got {distance_range}")
    if angle_range[0] >= angle_range[1]:
        raise ValueError(f"angle_range must have min < max, got {angle_range}")

    angle_span = angle_range[1] - angle_range[0]
    full_circle = bool(
        np.isclose(angle_span, 2 * np.pi, rtol=8 * np.finfo(float).eps, atol=0.0)
    )
    if circular_angle is None:
        circular_angle = full_circle
    elif circular_angle and not full_circle:
        raise ValueError(
            "circular_angle=True requires a full circle (2π radians). "
            "Use circular_angle=False or None for a partial angular range."
        )

    n_distance = _bin_count(distance_range[1] - distance_range[0], distance_bin_size)
    n_angle = _bin_count(angle_span, angle_bin_size)

    # Create grid edges and an all-active mask
    distance_edges = np.linspace(distance_range[0], distance_range[1], n_distance + 1)
    angle_edges = np.linspace(angle_range[0], angle_range[1], n_angle + 1)
    grid_edges = (distance_edges, angle_edges)
    active_mask = np.ones((n_distance, n_angle), dtype=bool)

    params: dict[str, Any] = {
        "active_mask": active_mask,
        "grid_edges": grid_edges,
        "connect_diagonal_neighbors": connect_diagonal_neighbors,
        "circular_angle": circular_angle,
    }
    layout = _PolarGridLayout()
    layout.build(**params)
    return Environment._from_built_layout(
        cast("LayoutEngine", layout),
        name=name,
        layout_type_used="MaskedGrid",
        layout_params_used=params,
        units=units,
        frame=frame,
        _coordinates=EgocentricPolarSystem(circular_angle),
    )
