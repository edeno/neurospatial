"""Test construction helpers using owned geometry publication."""

from neurospatial import Environment


def publish_environment(
    name="",
    layout=None,
    layout_type_used=None,
    layout_params_used=None,
    regions=None,
    **kwargs,
):
    coordinates = kwargs.pop("_coordinates", None)
    if isinstance(layout, Environment):
        return Environment._from_binning(
            layout._binning,
            connectivity=layout.connectivity,
            dimension_ranges=layout.dimension_ranges,
            layout_type_tag=layout_type_used or layout.layout_type,
            layout_type=layout._layout_category,
            build_params=layout.layout_parameters
            if layout_params_used is None
            else layout_params_used,
            is_linearized_track=layout.is_linearized_track,
            coordinate_kind=layout.coordinate_kind,
            name=name,
            regions=regions,
            **kwargs,
        )
    return Environment._from_built_layout(
        layout,
        name=name,
        layout_type_used=layout_type_used,
        layout_params_used=layout_params_used,
        regions=regions,
        _coordinates=coordinates,
        **kwargs,
    )
