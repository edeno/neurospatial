"""Test construction helpers using owned geometry publication."""

import copy
import weakref

from neurospatial import Environment
from neurospatial.environment import _publication
from neurospatial.environment._publication import _Metadata, _Provenance


def publish_binning(
    binning,
    *,
    connectivity,
    dimension_ranges,
    layout_type,
    category,
    build_params=None,
    is_linearized_track=None,
    name="",
    units=None,
    frame=None,
    cls=Environment,
    **kwargs,
):
    """Publish an explicit binning and graph, as factories and readers do."""
    return _publication.publish(
        cls,
        binning,
        connectivity=connectivity,
        dimension_ranges=dimension_ranges,
        provenance=_Provenance(layout_type, category, dict(build_params or {})),
        metadata=_Metadata(name, units, frame),
        is_linearized_track=binning.linearization is not None
        if is_linearized_track is None
        else is_linearized_track,
        **kwargs,
    )


def publish_environment(
    name="",
    layout=None,
    layout_type_used=None,
    layout_params_used=None,
    regions=None,
    *,
    units=None,
    frame=None,
    **kwargs,
):
    """Publish a built layout, or republish another environment's geometry."""
    coordinates = kwargs.pop("_coordinates", None)
    if isinstance(layout, Environment):
        return publish_binning(
            layout._binning,
            connectivity=layout.connectivity,
            dimension_ranges=layout.dimension_ranges,
            layout_type=layout_type_used or layout.layout_type,
            category=layout._provenance.category,
            build_params=layout.layout_parameters
            if layout_params_used is None
            else layout_params_used,
            is_linearized_track=layout.is_linearized_track,
            coordinate_kind=layout.coordinate_kind,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            **kwargs,
        )
    return _publication.publish_layout(
        Environment,
        layout,
        metadata=_Metadata(name, units, frame),
        layout_type=layout_type_used,
        build_params=layout_params_used,
        regions=regions,
        coordinates=coordinates,
        **kwargs,
    )


def record_publications(monkeypatch):
    """Record every environment the publisher returns while patched."""
    published = []
    original = _publication.publish

    def recording(*args, **kwargs):
        env = original(*args, **kwargs)
        published.append(env)
        return env

    monkeypatch.setattr(_publication, "publish", recording)
    return published


_SOURCE_LAYOUTS = weakref.WeakKeyDictionary()


def record_source_layouts():
    """Remember the layout engine each layout-built environment came from.

    Environments keep no layout or input data, so tests that compare a
    published environment against its native layout engine look the engine
    up here instead of rebuilding it from recorded parameters.
    """
    original = _publication.publish_layout
    if getattr(original, "records_source_layouts", False):
        return

    def recording(cls, layout, **kwargs):
        env = original(cls, layout, **kwargs)
        # Only built-in engines: tests check that external providers are freed.
        if type(layout).__module__.startswith("neurospatial."):
            _SOURCE_LAYOUTS[env] = layout
        return env

    recording.records_source_layouts = True
    _publication.publish_layout = recording


def native_layout(env):
    """Return an independent copy of the layout engine ``env`` was built from."""
    return copy.deepcopy(_SOURCE_LAYOUTS[env])
