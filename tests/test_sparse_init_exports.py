"""Pin the surface of the top-level ``neurospatial`` package.

v0.4 ships a sparse top-level surface — four core classes plus the public
exception hierarchy. Everything else lives in submodules,
on purpose, so that ``from neurospatial import *`` doesn't haul in
hundreds of names.

The earlier version of this file had ~18 tests: per-name "is importable"
tautologies, per-domain "submodule imports still work" passthroughs, and
a docstring-prose audit. Those are subsumed by:

- ``test_all_has_core_classes_and_exceptions`` — pin the exact ``__all__``.
- ``test_old_export_not_in_all`` (parametrized) — pin the removed names
  to "not in __all__"; this is the real regression check that catches
  accidental re-exports.
"""

import pytest


def test_all_has_core_classes_and_exceptions():
    """``neurospatial.__all__`` is exactly the core surface, no more."""
    import neurospatial

    expected = {
        # Core classes
        "Environment",
        "Region",
        "Regions",
        "CompositeEnvironment",
        # Public exception hierarchy
        "EnvironmentNotFittedError",
        "GraphValidationError",
        "RegionNotFoundError",
        "BinIndexOutOfRangeError",
        "IncompatibleEnvironmentError",
        "LayoutNotBuiltError",
        # Eagerly-exported primitives
        "bin_spikes_in_time",
        # Lazily-accessible public container (PEP 562 __getattr__); the one
        # justified new container (ragged spike times don't fit an array).
        "SpikeTrains",
        # Lazily-accessible analysis submodules (PEP 562 __getattr__)
        "encoding",
        "decoding",
        "behavior",
        "events",
        "ops",
        "layout",
        "regions",
        "stats",
        "simulation",
        "annotation",
        "animation",
        "io",
    }
    actual = set(neurospatial.__all__)
    assert actual == expected, (
        f"Top-level surface drift. Expected: {expected}. Got: {actual}."
    )


@pytest.mark.parametrize(
    "old_export",
    [
        # Decoding
        "decode_position",
        "DecodingResult",
        "decoding_error",
        "median_decoding_error",
        # Animation overlays
        "PositionOverlay",
        "EventOverlay",
        "SpikeOverlay",
        "HeadDirectionOverlay",
        "BodypartOverlay",
        "VideoOverlay",
        "ScaleBarConfig",
        # Annotation
        "annotate_video",
        "AnnotationResult",
        # Layout
        "LayoutType",
        "list_available_layouts",
        "get_layout_parameters",
        # I/O
        "to_file",
        "from_file",
        "to_dict",
        "from_dict",
        # Encoding metrics
        "detect_place_fields",
        "spatial_information",
        "sparsity",
        "selectivity",
        "border_score",
        "grid_score",
        "population_vector_correlation",
        # Behavioral
        "detect_laps",
        "segment_trials",
        "detect_region_crossings",
        "path_progress",
        "cost_to_goal",
        # Events
        "peri_event_histogram",
        "align_spikes_to_events",
        "time_to_nearest_event",
        # Ops
        "map_points_to_bins",
        "distance_field",
        "normalize_field",
        "gradient",
        "divergence",
        "heading_from_velocity",
        "compute_viewshed",
        # Simulation
        "SpatialViewCellModel",
    ],
)
def test_old_export_not_in_all(old_export):
    """Names that used to live at the top level must require an explicit submodule import."""
    import neurospatial

    assert old_export not in neurospatial.__all__, (
        f"{old_export!r} should not be in top-level __all__; import from its submodule."
    )
