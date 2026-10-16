"""Test construction helpers using owned geometry publication."""

import copy
import inspect
import weakref
from unittest import mock

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


_BUILD_RECIPES = weakref.WeakKeyDictionary()


def built_from(build):
    """Return ``build()`` and remember how it was built, for ``native_layout``."""
    env = build()
    _BUILD_RECIPES[env] = build
    return env


def layout_of(build):
    """Return the layout engine ``build()`` publishes, copied before publication.

    The publisher is patched only while ``build`` runs. Copying the engine
    before publication keeps the oracle independent of anything publication
    might do to it.
    """
    captured = []
    original = _publication.publish_layout

    def recording(cls, layout, **kwargs):
        captured.append(copy.deepcopy(layout))
        return original(cls, layout, **kwargs)

    with mock.patch.object(_publication, "publish_layout", recording):
        build()
    (layout,) = captured
    return layout


def native_layout(env):
    """Rebuild the layout engine of an environment made with ``built_from``.

    Environments keep no layout or input data, so engine-versus-snapshot
    tests rebuild the engine from the same deterministic recipe.
    """
    return layout_of(_BUILD_RECIPES[env])


def call_unbound(env, name):
    """Call an Environment member through the class, as ``Environment.f(env, ...)``.

    Required arguments are passed as None, and a property's getter is called
    directly, so a refusal must come before any argument is read.
    """
    member = inspect.getattr_static(Environment, name)
    if isinstance(member, property):
        return member.fget(env)
    args, kwargs = [env], {}
    for parameter in list(inspect.signature(member).parameters.values())[1:]:
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = None
        else:
            args.append(None)
    return member(*args, **kwargs)
