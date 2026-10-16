"""Cartesian-only capability names, guarded by the common Environment class."""

from __future__ import annotations

from neurospatial._exceptions import CartesianOnlyError
from neurospatial.geometry._coordinates import (
    _POLAR_COORDINATES,
    CartesianSystem,
    EgocentricPolarSystem,
)

CARTESIAN_OPERATIONS: frozenset[str] = frozenset(
    [
        "component_keys",
        "bin_sources",
        "get_bridge_attributes",
        "bin_at",
        "contains",
        "distance_between",
        "to_linear",
        "linear_to_nd",
        "occupancy",
        "bin_sequence",
        "bin_sequence_with_runs",
        "interpolate",
        "rebin",
        "subset",
        "clip",
        "apply_transform",
    ]
)
CARTESIAN_GRID_OPERATIONS: frozenset[str] = frozenset(
    {"to_linear", "linear_to_nd", "rebin", "subset"}
)

_POLAR_EXPLANATION = (
    f"{_POLAR_COORDINATES} Use connectivity-graph operations instead (neighbors, "
    "path_between, reachable_from, smooth, or distance_to(metric='geodesic')), "
    "or build an allocentric Cartesian env via Environment.from_samples / "
    "from_polygon / from_grid_mask."
)


def require_cartesian(
    coordinates: CartesianSystem | EgocentricPolarSystem, operation: str
) -> None:
    """Refuse Cartesian coordinate math on polar radius/angle bins.

    Parameters
    ----------
    coordinates : CartesianSystem or EgocentricPolarSystem
        The environment's coordinate descriptor.
    operation : str
        The refused call, e.g. ``"bin_at()"``; the name before ``(`` decides
        whether the refusal names a Cartesian grid or Cartesian coordinates.

    Raises
    ------
    CartesianOnlyError
        If ``coordinates`` is polar.
    """
    if isinstance(coordinates, EgocentricPolarSystem):
        raise CartesianOnlyError(
            "Environment",
            operation,
            reason="Cartesian-grid operation"
            if operation.partition("(")[0] in CARTESIAN_GRID_OPERATIONS
            else "Cartesian-coordinate operation",
            explanation=_POLAR_EXPLANATION,
        )
