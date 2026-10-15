"""Cartesian-only capability names, guarded by the common Environment class."""

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
