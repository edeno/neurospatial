"""Private ownership boundaries for published layouts and continuous domains."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, NoReturn, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.arena import (
    Arena,
    EgocentricSurround,
    MaskArena,
    OpenArena,
    TrackArena,
    UnknownArena,
    _CombinedArena,
    _TransformedArena,
)
from neurospatial.layout import (
    GraphLayout,
    HexagonalLayout,
    ImageMaskLayout,
    LayoutEngine,
    MaskedGridLayout,
    RegularGridLayout,
    ShapelyPolygonLayout,
    TriangularMeshLayout,
)
from neurospatial.layout._binning import (
    Binning,
    KnownFaces,
    LayoutFaces,
    Locator,
    MissingFaces,
    SpatialBins,
    TrackLinearization,
    binning_from_layout,
)
from neurospatial.layout._binning_layout import BinningLayout
from neurospatial.layout._capabilities import RectilinearGrid
from neurospatial.layout._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.layout._composition import CombinedLocator
from neurospatial.layout._face_normals import FaceNormals
from neurospatial.layout._grid_frame import FramedGrid
from neurospatial.layout._locators import (
    HexLocator,
    MeshLocator,
    NearestBinLocator,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
)
from neurospatial.layout._polygon import ClipMapping, ClippedLocator, PolygonCells
from neurospatial.layout._rebin_spec import GridBinningSpec
from neurospatial.layout.engines._polar import _PolarGridLayout

_NATIVE_LAYOUTS = (
    RegularGridLayout,
    MaskedGridLayout,
    ImageMaskLayout,
    ShapelyPolygonLayout,
    HexagonalLayout,
    GraphLayout,
    TriangularMeshLayout,
    _PolarGridLayout,
)


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
            own_arena(arena)
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


def _check_native_layout(layout: LayoutEngine) -> None:
    """Admit native data, not instance-level replacements of geometry methods."""
    if type(layout) not in _NATIVE_LAYOUTS:
        _migration(layout)
    if any(callable(getattr(type(layout), name, None)) for name in vars(layout)):
        _migration(layout)


def _check_binning(binning: Binning) -> None:
    """Refuse opaque callbacks before any provider can be deep-copied."""
    if type(binning) is not Binning or type(binning.bins) is not SpatialBins:
        _migration(binning)
    if (
        binning.clip_mapping is not None
        and type(binning.clip_mapping) is not ClipMapping
    ):
        _migration(binning.clip_mapping)
    _check_locator(binning.locator)
    faces = binning.faces
    if type(faces) is LayoutFaces:
        _check_native_layout(faces.layout)
    elif type(faces) not in (KnownFaces, MissingFaces):
        _migration(faces)
    for component, allowed in (
        (binning.rectilinear, (RectilinearGrid,)),
        (binning.linearization, (TrackLinearization,)),
        (binning.shapes, (HexagonShapes, TriangleShapes, PolygonCells)),
        (binning.outline, (Outline,)),
        (binning.grid_spec, (GridBinningSpec,)),
        (binning.face_normals, (FaceNormals,)),
        (binning.grid_frame, (FramedGrid,)),
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
        return replace(
            locator,
            children=tuple(_snapshot_binning(child) for child in locator.children),
            arenas=tuple(own_arena(arena) for arena in locator.arenas),
        )
    if type(locator) in (
        RectilinearLocator,
        HexLocator,
        MeshLocator,
        NearestBinLocator,
    ):
        return locator
    from neurospatial.extensions import _snapshot_provider

    return cast("Locator", _snapshot_provider(locator, "locator"))


def _snapshot_binning(binning: Binning) -> Binning:
    """Normalize external component data before cross-component validation."""
    locator = _snapshot_locator(binning.locator)
    track = binning.linearization
    if track is not None and type(track) is not TrackLinearization:
        from neurospatial.extensions import _snapshot_provider

        track = cast("TrackLinearization", _snapshot_provider(track, "track"))
    if locator is binning.locator and track is binning.linearization:
        return binning
    return replace(binning, locator=locator, linearization=track)


def _array_export(array: NDArray[Any]) -> NDArray[Any]:
    """Detach both data and mutable ndarray headers from private authority."""
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _graph_export(graph: nx.Graph) -> nx.Graph:
    # Imported at use time because core imports this publication boundary.
    from neurospatial.environment.core import _freeze_graph_attributes

    exported = copy.deepcopy(graph)
    _freeze_graph_attributes(exported)
    return exported


class _OwnedLayout(BinningLayout):
    """Readonly layout interface over an inaccessible native builder snapshot.

    Internal publication reads ``_binning`` and ``_owned_connectivity`` directly.
    Public exports are detached, including the adapter's lazy face builder.
    """

    def __init__(self, layout: LayoutEngine, binning: Binning) -> None:
        # These are trusted data extracted from a private built-in copy. Saved
        # BinningLayout inputs have already run the strong constructor checks.
        object.__setattr__(self, "_binning", binning)
        object.__setattr__(self, "bin_centers", binning.bins.centers)
        object.__setattr__(self, "_owned_connectivity", layout.connectivity)
        object.__setattr__(self, "connectivity", layout.connectivity)
        ranges = layout.dimension_ranges
        object.__setattr__(
            self,
            "dimension_ranges",
            tuple(map(tuple, ranges)) if ranges is not None else None,
        )
        object.__setattr__(self, "_layout_type_tag", layout._layout_type_tag)
        object.__setattr__(self, "_layout_type", layout.layout_type)
        object.__setattr__(self, "_build_params_used", layout._build_params_used)
        object.__setattr__(self, "_is_linearized_track", layout.is_linearized_track)
        object.__setattr__(self, "_native_grid_compatible", layout.is_grid_compatible)
        object.__setattr__(self, "_native_grid_edges", layout.grid_edges or ())
        object.__setattr__(
            self, "_native_grid_shape", layout.grid_shape or (binning.bins.n_bins,)
        )
        mask = layout.active_mask
        object.__setattr__(
            self,
            "_native_active_mask",
            np.ones(binning.bins.n_bins, dtype=bool) if mask is None else mask,
        )

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name == "bin_centers":
            return _array_export(value)
        if name == "connectivity":
            return _graph_export(value)
        if name == "_build_params_used":
            return copy.deepcopy(value)
        return value

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("Published environment layouts are read-only.")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Published environment layouts are read-only.")

    def __getstate__(self) -> dict[str, Any]:
        return vars(self).copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)

    def binning(self) -> Binning:
        """Export independent binning data, including any private face builder."""
        return copy.deepcopy(self._binning)

    def bin_sizes(self) -> NDArray[np.float64]:
        """Export independent cell measures in active-bin order."""
        return _array_export(self._binning.bins.sizes)

    def build(self, **kwargs: Any) -> NoReturn:
        """Refuse rebuilding an already published value."""
        raise TypeError("Published layouts are read-only; build a new environment.")

    @property
    def is_grid_compatible(self) -> bool:
        return bool(self._native_grid_compatible)

    @property
    def grid_edges(self) -> tuple[NDArray[np.float64], ...]:
        return tuple(_array_export(edge) for edge in self._native_grid_edges)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(self._native_grid_shape)

    @property
    def active_mask(self) -> NDArray[np.bool_]:
        return _array_export(self._native_active_mask)


def own_layout(layout: LayoutEngine) -> BinningLayout:
    """Own a built native layout without eagerly constructing its faces.

    Exact built-ins are copied privately. Custom subclasses and delegated
    providers need an owned-data migration instead of generic callback copying.
    An already owned bridge is immutable and can be shared by value operations.
    """
    if type(layout) is _OwnedLayout:
        return layout
    if type(layout) is BinningLayout:
        if any(callable(getattr(type(layout), name, None)) for name in vars(layout)):
            _migration(layout)
        binning = _snapshot_binning(layout._binning)
        _check_binning(binning)
        if binning is not layout._binning:
            layout = BinningLayout(
                binning,
                connectivity=layout.connectivity,
                dimension_ranges=layout.dimension_ranges,
                layout_type_tag=layout._layout_type_tag,
                layout_type=layout.layout_type,
                build_params=layout._build_params_used,
                is_linearized_track=layout.is_linearized_track,
            )
    elif type(layout) not in _NATIVE_LAYOUTS:
        from neurospatial.environment._from_bins import layout_from_bins
        from neurospatial.extensions import _snapshot_provider

        return own_layout(
            layout_from_bins(cast("Binning", _snapshot_provider(layout, "binning")))
        )
    else:
        _check_native_layout(layout)
    private = copy.deepcopy(layout)
    if private.connectivity is None:
        raise ValueError("Build the layout before publishing its owned snapshot.")
    binning = (
        private._binning
        if type(private) is BinningLayout
        else binning_from_layout(private, private.connectivity)
    )
    _check_binning(binning)
    return _OwnedLayout(private, binning)


def own_arena(arena: Arena) -> Arena:
    """Own a native continuous domain, refusing delegated custom behavior.

    The environment retains this private snapshot; public arena access must
    separately detach it so mutable NumPy headers cannot reach its authority.
    """

    def snapshot(value: Arena) -> Arena:
        if type(value) is _CombinedArena:
            return _CombinedArena(tuple(snapshot(child) for child in value.children))
        if type(value) is _TransformedArena:
            base = snapshot(value.base)
            return value if base is value.base else replace(value, base=base)
        elif type(value) not in (
            UnknownArena,
            OpenArena,
            MaskArena,
            TrackArena,
            EgocentricSurround,
        ):
            from neurospatial.extensions import _snapshot_provider

            return cast("Arena", _snapshot_provider(value, "arena"))
        return value

    return copy.deepcopy(snapshot(arena))
