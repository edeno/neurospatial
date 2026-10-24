"""Ownership boundaries for published binning and continuous domains."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import NoReturn, cast

from neurospatial.geometry._arena import (
    Arena,
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
    _CombinedArena,
    _TransformedArena,
)
from neurospatial.geometry._binning import (
    BinConnections,
    Binning,
    KnownFaces,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
)
from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.geometry._codecs import snapshot_provider
from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.geometry._composition import CombinedLocator
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._grid_frame import FramedGrid
from neurospatial.geometry._locators import (
    HexLocator,
    Locator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.geometry._native_faces import NativeFaces
from neurospatial.geometry._polygon_cells import (
    ClipMapping,
    ClippedLocator,
    PolygonCells,
)
from neurospatial.geometry._rebin_spec import GridBinningSpec


def _migration(provider: object) -> NoReturn:
    raise TypeError(
        f"{type(provider).__name__} cannot be published as an owned immutable "
        "snapshot. Migrate custom providers to supported owned binning/domain "
        "data before constructing an environment; retaining delegated mutable "
        "geometry is no longer supported."
    )


def _check_locator(locator: Locator) -> None:
    if type(locator) is CombinedLocator:
        for child in locator.children:
            _check_binning(child)
        for arena in locator.arenas:
            _snapshot_arena(arena)
        return
    if isinstance(
        locator, (SubsetLocator, TransformedLocator, ClippedLocator)
    ) and type(locator) in (SubsetLocator, TransformedLocator, ClippedLocator):
        if isinstance(locator, ClippedLocator):
            if type(locator.cells) is not PolygonCells:
                _migration(locator.cells)
            if type(locator.mapping) is not ClipMapping:
                _migration(locator.mapping)
        _check_locator(locator.base)
    elif type(locator) not in (
        RectilinearLocator,
        HexLocator,
        MeshLocator,
        NearestBinLocator,
    ):
        _migration(locator)


def _check_binning(binning: Binning) -> None:
    """Refuse opaque callbacks before any provider can be deep-copied."""
    if (
        type(binning) is not Binning
        or type(binning.bins) is not SpatialBins
        or type(binning.connections) is not BinConnections
    ):
        _migration(binning)
    _check_locator(binning.locator)
    faces = binning.faces
    if type(faces) not in (KnownFaces, MissingFaces, NativeFaces):
        _migration(faces)
    for component, allowed in (
        (binning.grid, (RectilinearGrid, FramedGrid)),
        (binning.linearization, (TrackLinearization,)),
        (binning.shapes, (HexagonShapes, TriangleShapes, PolygonCells)),
        (binning.outline, (Outline,)),
        (binning.grid_spec, (GridBinningSpec,)),
        (binning.face_normals, (FaceNormals,)),
    ):
        if component is not None and type(component) not in allowed:
            _migration(component)


def _snapshot_locator(locator: Locator) -> Locator:
    """Adapt external lookup data, including bases of built-in wrappers."""
    if isinstance(
        locator, (SubsetLocator, TransformedLocator, ClippedLocator)
    ) and type(locator) in (SubsetLocator, TransformedLocator, ClippedLocator):
        base = _snapshot_locator(locator.base)
        return locator if base is locator.base else replace(locator, base=base)
    if type(locator) is CombinedLocator:
        children = tuple(_snapshot_binning(child) for child in locator.children)
        arenas = tuple(_snapshot_arena(arena) for arena in locator.arenas)
        if all(
            new is old
            for new, old in zip(
                (*children, *arenas), (*locator.children, *locator.arenas), strict=True
            )
        ):
            return locator
        return replace(locator, children=children, arenas=arenas)
    if type(locator) in (
        RectilinearLocator,
        HexLocator,
        MeshLocator,
        NearestBinLocator,
    ):
        return locator

    return cast("Locator", snapshot_provider(locator, "locator"))


def _snapshot_binning(binning: Binning) -> Binning:
    """Normalize external component data before cross-component validation."""
    locator = _snapshot_locator(binning.locator)
    track = binning.linearization
    if track is not None and type(track) is not TrackLinearization:
        track = cast("TrackLinearization", snapshot_provider(track, "track"))
    if locator is binning.locator and track is binning.linearization:
        return binning
    return replace(binning, locator=locator, linearization=track)


def own_binning(binning: Binning) -> Binning:
    """Own supported component data after normalizing registered adapters."""
    normalized = _snapshot_binning(binning)
    _check_binning(normalized)
    return copy.deepcopy(normalized)


def _snapshot_arena(value: Arena) -> Arena:
    """Adapt a registered arena provider, keeping built-in arenas as they are."""
    if type(value) is _CombinedArena:
        children = tuple(_snapshot_arena(child) for child in value.children)
        if all(new is old for new, old in zip(children, value.children, strict=True)):
            return value
        return _CombinedArena(children)
    if type(value) is _TransformedArena:
        base = _snapshot_arena(value.base)
        return value if base is value.base else replace(value, base=base)
    elif type(value) not in (
        UnknownArena,
        OpenArena,
        MaskArena,
        TrackArena,
        EgocentricSurround,
    ):
        return cast("Arena", snapshot_provider(value, "arena"))
    return value


def own_arena(arena: Arena) -> Arena:
    """Own a native continuous domain, refusing delegated custom behavior.

    The environment retains this private snapshot; public arena access must
    separately detach it so mutable NumPy headers cannot reach its authority.
    """
    return copy.deepcopy(_snapshot_arena(arena))
