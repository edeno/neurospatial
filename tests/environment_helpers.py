"""Test construction helpers using owned geometry publication."""

import copy
import inspect
import weakref
from dataclasses import replace
from unittest import mock

import networkx as nx
import numpy as np
from shapely.geometry import LineString, box

import neurospatial.environment
from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.environment import _publication
from neurospatial.environment._publication import _Metadata, _Provenance
from neurospatial.layout._construction import _built_layout_data


def publish_binning(
    binning,
    *,
    layout_type=None,
    dimension_ranges=None,
    build_params=None,
    derivations=(),
    coordinate_kind="cartesian",
    name="",
    units=None,
    frame=None,
    cls=Environment,
    **kwargs,
):
    """Publish an explicit binning, as factories and readers do.

    ``layout_type`` and ``dimension_ranges``, when given, replace the
    binning's own label and ranges. A polar ``coordinate_kind`` needs an
    ``arena=`` keyword holding the environment's ``EgocentricSurround``.
    """
    if layout_type is not None:
        binning = replace(binning, kind=layout_type)
    if dimension_ranges is not None:
        binning = replace(binning, dimension_ranges=tuple(dimension_ranges))
    return _publication._publish(
        cls,
        binning,
        coordinate_kind=coordinate_kind,
        provenance=_Provenance(dict(build_params or {}), derivations),
        metadata=_Metadata(name, units, frame),
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
    """Publish a built layout, or republish another environment's geometry.

    A built layout records its own build choices; ``layout_params_used``
    replaces the choices only when republishing an environment.
    ``layout_type_used`` replaces the layout label. A republished polar
    environment keeps its arena unless ``arena=`` replaces it.
    """
    coordinate_kind = kwargs.pop("_coordinate_kind", "cartesian")
    if isinstance(layout, Environment):
        if layout.coordinate_kind == "polar":
            kwargs.setdefault("arena", layout.arena)
        return publish_binning(
            layout._binning,
            layout_type=layout_type_used,
            build_params=layout.layout_parameters
            if layout_params_used is None
            else layout_params_used,
            coordinate_kind=layout.coordinate_kind,
            name=name,
            units=units,
            frame=frame,
            regions=regions,
            **kwargs,
        )
    if layout_type_used is not None:
        binning, build_params = _built_layout_data(layout)
        return publish_binning(
            binning,
            layout_type=layout_type_used,
            build_params=build_params,
            coordinate_kind=coordinate_kind,
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
        regions=regions,
        coordinate_kind=coordinate_kind,
        **kwargs,
    )


def record_publications(monkeypatch):
    """Record every environment the publisher returns while patched."""
    published = []
    original = _publication._publish

    def recording(*args, **kwargs):
        env = original(*args, **kwargs)
        published.append(env)
        return env

    monkeypatch.setattr(_publication, "_publish", recording)
    # io imports _publish from neurospatial.environment inside its reader, at
    # call time, so the package's binding must be patched as well.
    monkeypatch.setattr(neurospatial.environment, "_publish", recording)
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
    if len(captured) != 1:
        raise AssertionError(
            f"layout_of() needs build() to publish exactly one layout engine, "
            f"but it published {len(captured)}. Pass a callable that runs one "
            "layout-based factory, such as "
            "lambda: Environment.from_samples(positions, bin_size=2.0); "
            "from_bins, from_file and combine publish no layout engine."
        )
    return captured[0]


def face_graph(binning):
    """Return a binning's finite-volume faces as a graph, and its cell measures.

    Each edge carries the face measure ``A`` and the normal ``distance``, in
    neighbor-pair order; this is the input ``DiffusionGeometry.from_graph``
    and ``compute_diffusion_kernels`` take.
    """
    faces = binning.faces.build()
    graph = nx.Graph()
    graph.add_nodes_from(range(binning.bins.n_bins))
    graph.add_edges_from(
        (int(u), int(v), {"A": float(area), "distance": float(distance)})
        for (u, v), area, distance in zip(
            faces.edges, faces.A, faces.distance, strict=True
        )
    )
    return graph, np.array(faces.volumes)


def native_layout(env):
    """Rebuild the layout engine of an environment made with ``built_from``.

    Environments keep no layout or input data, so engine-versus-snapshot
    tests rebuild the engine from the same deterministic recipe.
    """
    return layout_of(_BUILD_RECIPES[env])


def call_unbound(env, name, *, bound=False):
    """Call an Environment member through the class, as ``Environment.f(env, ...)``.

    Required arguments are passed as None, and a property's getter is called
    directly, so a refusal must come before any argument is read. With
    ``bound=True`` the member is called on the instance, as ``env.f(...)``.
    """
    member = inspect.getattr_static(Environment, name)
    if isinstance(member, property):
        return getattr(env, name) if bound else member.fget(env)
    args, kwargs = [] if bound else [env], {}
    for parameter in list(inspect.signature(member).parameters.values())[1:]:
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = None
        else:
            args.append(None)
    function = getattr(env, name) if bound else member
    return function(*args, **kwargs)


def rotation_matrix(angle, offset=(0, 0)):
    """The 3 x 3 matrix of a rotation by ``angle`` radians, then a shift by ``offset``."""
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), offset[0]],
            [np.sin(angle), np.cos(angle), offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def renumbered(env, order, distances=None, *, max_distance=5.0):
    """Publish ``env``'s bins and open connections in a new bin order.

    New bin ``i`` is ``env``'s bin ``order[i]``; centers, measures, cell
    faces, units, frame and arena are unchanged, so the copy smooths a field
    as ``env`` does, and point lookup is nearest-bin within
    ``max_distance``. ``distances`` replaces the open pairs' navigation
    distances, in ``env.edge_pairs`` order, to make a copy that differs only
    in them.
    """
    from neurospatial.binning import (
        BinConnections,
        Binning,
        FaceGeometry,
        KnownFaces,
        MissingFaces,
        NearestBinLocator,
        SpatialBins,
    )

    binning = env.binning
    centers = binning.bins.centers[order]
    connections = binning.connections
    pairs = np.argsort(order)[connections.pairs]
    if binning.faces.available:
        faces = binning.faces.build()
        new_faces = KnownFaces(
            FaceGeometry(pairs, faces.A, faces.distance, faces.volumes[order])
        )
    else:
        new_faces = MissingFaces("renumbered copy of bins without faces")
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(
                centers, binning.bins.sizes[order], binning.bins.measure_dims[order]
            ),
            BinConnections.from_arrays(
                pairs,
                connections.pair_distances if distances is None else distances,
            ),
            NearestBinLocator(centers, max_distance=max_distance),
            new_faces,
        ),
        units=env.units,
        frame=env.frame,
        arena=env.arena,
    )


def renumbered_cells(env, order):
    """Publish a clipped ``env``'s polygon cells in a new bin order.

    New bin ``i`` is ``env``'s bin ``order[i]``, with the same polygon, parent
    bin, center, measure and faces; lookup is the same clip of the same
    parent bins. The clip gives a point on the line between two pieces of one
    parent bin to the lower-numbered piece, so reordering such pieces moves
    that line from one piece to the other.
    """
    from neurospatial.binning import (
        BinConnections,
        Binning,
        ClipMapping,
        ClippedLocator,
        FaceGeometry,
        KnownFaces,
        PolygonCells,
        SpatialBins,
    )

    binning = env.binning
    locator = binning.locator
    cells = PolygonCells(tuple(binning.shapes.polygons[row] for row in order))
    mapping = ClipMapping(
        locator.mapping.parent_bin_indices[order], locator.mapping.n_parent_bins
    )
    connections = binning.connections
    pairs = np.argsort(order)[connections.pairs]
    if binning.faces.available:
        faces = binning.faces.build()
        new_faces = KnownFaces(
            FaceGeometry(pairs, faces.A, faces.distance, faces.volumes[order])
        )
    else:
        new_faces = binning.faces
    bins = binning.bins
    return Environment.from_bins(
        Binning(
            binning.kind,
            SpatialBins.from_arrays(
                bins.centers[order], bins.sizes[order], bins.measure_dims[order]
            ),
            BinConnections.from_arrays(pairs, connections.pair_distances),
            ClippedLocator(locator.base, cells, mapping),
            new_faces,
            shapes=cells,
        ),
        units=env.units,
        frame=env.frame,
        arena=env.arena,
    )


def wall_split_arena():
    """A box whose wall cuts parent cells 12 to 15 in two (bins 12 to 19)."""
    return Environment.from_samples(
        np.random.default_rng(0).uniform(0, 30, (4000, 2)),
        bin_size=5.0,
        units="cm",
        arena=OpenArena(box(0, 0, 30, 30), walls=[LineString([(12.5, 0), (12.5, 20)])]),
        wall_fit="split",
    )


# Bins of the pieces of each parent cell that the wall cuts in two.
WALL_SPLIT_PIECES = ([12, 13], [14, 15], [16, 17], [18, 19])


def order_preserving_permutation(n_bins):
    """A seeded bin order that keeps each wall-split cell's pieces in order."""
    order = np.random.default_rng(1).permutation(n_bins)
    for pieces in WALL_SPLIT_PIECES:
        positions = np.flatnonzero(np.isin(order, pieces))
        order[positions] = np.sort(order[positions])
    return order
