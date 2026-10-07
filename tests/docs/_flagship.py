"""Public objects whose documentation is checked and executed."""

import importlib

FLAGSHIP = (
    "neurospatial",
    *(
        f"neurospatial.Environment.{name}"
        for name in (
            "from_samples",
            "open_field",
            "linear_track",
            "maze",
            "from_graph",
            "from_polygon",
            "occupancy",
            "bin_at",
            "neighbors",
            "to_file",
            "from_file",
            "plot_field",
            "animate_fields",
        )
    ),
    *(
        f"neurospatial.encoding.{name}"
        for name in (
            "compute_spatial_rate",
            "compute_spatial_rates",
            "compute_directional_rate",
            "compute_directional_rates",
            "compute_view_rate",
            "compute_view_rates",
            "compute_egocentric_rate",
            "compute_egocentric_rates",
            "is_place_cell",
            "is_head_direction_cell",
            "is_object_vector_cell",
            "is_spatial_view_cell",
            "detect_place_fields",
        )
    ),
    *(
        f"neurospatial.decoding.{name}"
        for name in (
            "decode_position",
            "decode_session",
            "bin_spikes_in_time",
            "decoding_error",
            "DecodingResult",
        )
    ),
    *(
        f"neurospatial.events.{name}"
        for name in (
            "peri_event_histogram",
            "population_peri_event_histogram",
            "PeriEventResult",
            "PopulationPeriEventResult",
        )
    ),
    *(
        f"neurospatial.behavior.{name}"
        for name in (
            "detect_laps",
            "segment_trials",
            "detect_region_crossings",
            "detect_runs_between_regions",
        )
    ),
)
NWB_FLAGSHIP = tuple(
    f"neurospatial.io.nwb.{name}"
    for name in (
        "read_position",
        "read_units",
        "write_environment",
        "read_environment",
    )
)


def resolve(dotted: str) -> object:
    """Resolve the longest importable module followed by its public attributes."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError as exc:
            candidate = ".".join(parts[:i])
            if exc.name is None or not (
                candidate == exc.name or candidate.startswith(exc.name + ".")
            ):
                raise
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)
