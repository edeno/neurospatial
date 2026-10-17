"""Owned bin geometry, lookup and finite-volume face capabilities."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arrays import ReadOnlyState
from neurospatial.geometry._capabilities import (
    RectilinearGrid,
)
from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.geometry._diffusion import (
    FaceGeometry,
)
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._grid_frame import FramedGrid
from neurospatial.geometry._rebin_spec import GridBinningSpec

if TYPE_CHECKING:
    from neurospatial.geometry._locators import LookupIndexes
    from neurospatial.geometry._polygon import ClipMapping, PolygonCells


@dataclass(frozen=True, eq=False)
class SpatialBins(ReadOnlyState):
    """Discrete bins with owned, read-only geometry arrays.

    Attributes
    ----------
    centers : ndarray of float64, shape (n_bins, n_dims)
        Finite bin centers in the environment's coordinates.
    sizes : ndarray of float64, shape (n_bins,)
        Finite, strictly positive bin measures in ``units ** measure_dim``.
    measure_dim : int
        Dimension of the bin measures: one for track lengths, otherwise the
        coordinate dimension for areas or volumes.
    pairs : ndarray of intp, shape (n_pairs, 2)
        Neighbor pairs in authoritative edge order and orientation, exposed as
        ``env.edge_pairs``. NetworkX edge iteration may use a different order.
    pair_distances : ndarray of float64, shape (n_pairs,)
        Finite, non-negative navigation distances, distinct from the flux
        distances used by finite-volume faces at track junctions.
    """

    centers: NDArray[np.float64]
    sizes: NDArray[np.float64]
    measure_dim: int
    pairs: NDArray[np.intp]
    pair_distances: NDArray[np.float64]

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=np.float64, copy=True)
        sizes = np.array(self.sizes, dtype=np.float64, copy=True)
        raw_pairs = np.asarray(self.pairs)
        distances = np.array(self.pair_distances, dtype=np.float64, copy=True)

        if centers.ndim != 2:
            raise ValueError(f"centers must be 2D, got shape {centers.shape}.")
        index: int | tuple[int, ...]
        invalid = ~np.isfinite(centers)
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"centers{index} must be finite.")
        n_bins, n_dims = centers.shape
        if sizes.shape != (n_bins,):
            raise ValueError(f"sizes must have shape ({n_bins},), got {sizes.shape}.")
        invalid = ~np.isfinite(sizes) | (sizes <= 0.0)
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(f"sizes[{index}] must be finite and strictly positive.")
        if (
            isinstance(self.measure_dim, (bool, np.bool_))
            or not isinstance(self.measure_dim, (int, np.integer))
            or not 1 <= self.measure_dim <= n_dims
        ):
            raise ValueError(
                f"measure_dim must be an integer between 1 and {n_dims}, "
                f"got {self.measure_dim!r}."
            )
        if raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2:
            raise ValueError(
                f"pairs must have shape (n_pairs, 2), got {raw_pairs.shape}."
            )
        if raw_pairs.size and raw_pairs.dtype.kind not in "iu":
            if raw_pairs.dtype.kind == "f":
                invalid = ~np.isfinite(raw_pairs) | (raw_pairs != np.floor(raw_pairs))
                if np.any(invalid):
                    index = tuple(int(i) for i in np.argwhere(invalid)[0])
                    raise ValueError(f"pairs{index} must contain integer bin indices.")
            else:
                raise ValueError("pairs(0, 0) must contain integer bin indices.")
        invalid = (raw_pairs < 0) | (raw_pairs >= n_bins)
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"pairs{index} must be in [0, {n_bins}).")
        pairs = np.array(raw_pairs, dtype=np.intp, copy=True)
        self_pairs = pairs[:, 0] == pairs[:, 1]
        if np.any(self_pairs):
            index = int(np.flatnonzero(self_pairs)[0])
            raise ValueError(f"pairs[{index}] is a self pair.")
        _, first_indices = np.unique(np.sort(pairs, axis=1), axis=0, return_index=True)
        duplicates = np.ones(len(pairs), dtype=bool)
        duplicates[first_indices] = False
        if np.any(duplicates):
            index = int(np.flatnonzero(duplicates)[0])
            raise ValueError(f"pairs[{index}] duplicates an earlier unordered pair.")
        if distances.shape != (len(pairs),):
            raise ValueError(
                f"pair_distances must have shape ({len(pairs)},), "
                f"got {distances.shape}."
            )
        invalid = ~np.isfinite(distances) | (distances < 0.0)
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"pair_distances[{index}] must be finite and non-negative."
            )

        for name, array in (
            ("centers", centers),
            ("sizes", sizes),
            ("pairs", pairs),
            ("pair_distances", distances),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def from_arrays(
        cls,
        centers: ArrayLike,
        sizes: ArrayLike,
        measure_dim: int,
        pairs: ArrayLike,
        pair_distances: ArrayLike,
    ) -> SpatialBins:
        """Validate and own bin centers, measures and ordered neighbor pairs.

        Parameters
        ----------
        centers : array-like, shape (n_bins, n_dims)
            Finite bin coordinates.
        sizes : array-like, shape (n_bins,)
            Finite, strictly positive bin measures.
        measure_dim : int
            Measure dimension in ``[1, n_dims]``.
        pairs : array-like of int, shape (n_pairs, 2)
            Distinct unordered neighbor pairs, kept in their input order.
        pair_distances : array-like, shape (n_pairs,)
            Finite, non-negative navigation distances.

        Returns
        -------
        SpatialBins
            Validated independent snapshot with read-only arrays.

        Raises
        ------
        ValueError
            If shapes, dimensions or values are invalid. Elementwise errors
            identify the offending array index.
        """
        return cls(
            np.asarray(centers),
            np.asarray(sizes),
            measure_dim,
            np.asarray(pairs),
            np.asarray(pair_distances),
        )

    @property
    def n_bins(self) -> int:
        """Number of active bins."""
        return int(self.centers.shape[0])

    @property
    def n_dims(self) -> int:
        """Number of coordinate columns."""
        return int(self.centers.shape[1])


class Locator(Protocol):
    """Find the bin that contains each query point."""

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Return bin indices for ``(n_points, n_dims)`` coordinates, or -1.

        Non-finite rows return -1; a wrong number of columns raises ValueError.
        The input is not modified. ``indexes`` keeps the search structures a
        locator builds (an environment passes its analysis cache); without
        it they are built for this call only.
        """
        ...


class FacesSource(Protocol):
    """Source of an environment's finite-volume cell faces."""

    @property
    def available(self) -> bool:
        """Whether validated finite-volume faces can be built."""
        ...

    def build(self) -> FaceGeometry:
        """Build validated faces in connectivity pair order, without caching."""
        ...

    def missing_error(self, operation: str) -> Exception:
        """Return an actionable error when the named operation needs faces."""
        ...


@dataclass(frozen=True, eq=False)
class KnownFaces(ReadOnlyState):
    """Owned finite-volume faces that no longer need their originating layout."""

    faces: FaceGeometry

    def __post_init__(self) -> None:
        raw = self.faces
        edges = np.asarray(raw.edges)
        areas = np.array(raw.A, dtype=np.float64, copy=True)
        distances = np.array(raw.distance, dtype=np.float64, copy=True)
        volumes = np.array(raw.volumes, dtype=np.float64, copy=True)
        if volumes.ndim != 1 or np.any(~np.isfinite(volumes) | (volumes <= 0)):
            raise ValueError(
                "face volumes must be finite, positive and one-dimensional."
            )
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
            raise ValueError(
                "face edges must have shape (n_faces, 2) and integer indices."
            )
        if np.any((edges < 0) | (edges >= len(volumes))):
            raise ValueError("face edges contain an out-of-range bin index.")
        if areas.shape != (len(edges),) or np.any(~np.isfinite(areas) | (areas < 0)):
            raise ValueError(
                "face areas must have shape (n_faces,) and be finite and non-negative."
            )
        if distances.shape != (len(edges),) or np.any(
            ~np.isfinite(distances) | (distances <= 0)
        ):
            raise ValueError(
                "face distances must have shape (n_faces,) and be finite and positive."
            )
        owned_edges = np.array(edges, dtype=np.intp, copy=True)
        for array in (owned_edges, areas, distances, volumes):
            array.setflags(write=False)
        object.__setattr__(
            self, "faces", FaceGeometry(owned_edges, areas, distances, volumes)
        )

    @property
    def available(self) -> bool:
        """Whether validated face geometry can be obtained."""
        return True

    def build(self) -> FaceGeometry:
        """Return the stored faces in published pair order."""
        return self.faces

    def missing_error(self, operation: str) -> NotImplementedError:
        """Return an error only relevant to unavailable face sources."""
        return NotImplementedError(f"{operation}: stored faces are available.")


@dataclass(frozen=True, eq=False)
class MissingFaces:
    """The reason finite-volume operations cannot use a binning."""

    reason: str

    @property
    def available(self) -> bool:
        """Whether validated face geometry can be obtained."""
        return False

    def build(self) -> FaceGeometry:
        """Refuse construction because valid face geometry is unavailable."""
        raise self.missing_error("This operation")

    def missing_error(self, operation: str) -> NotImplementedError:
        """Name the operation, missing geometry and supported recovery."""
        from neurospatial._exceptions import _format_error

        return NotImplementedError(
            _format_error(
                f"{operation} needs finite-volume cell geometry. {self.reason}",
                why="Why: finite-volume operations require valid cell faces, measures and normal distances.",
                fix="use the original environment or rebuild an environment from the changed geometry",
            )
        )


class _ReadOnlyTrackAttributes(dict[str, Any]):
    """Protect stored track geometry while supporting pickle and editable copies."""

    def _refuse_edit(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("Stored track graph attributes are read-only.")

    __setitem__ = _refuse_edit
    __delitem__ = _refuse_edit
    clear = _refuse_edit
    pop = _refuse_edit
    popitem = _refuse_edit
    setdefault = _refuse_edit
    update = _refuse_edit
    __ior__ = _refuse_edit

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self), (dict(self),)


def graph_attribute_items(graph: nx.Graph) -> Iterator[tuple[str, Any, Any]]:
    """Yield ``(element, key, value)`` for every graph, node and edge attribute.

    ``element`` names where the attribute is, ``"metadata"`` for
    ``graph.graph``, ``"node 3"`` or ``"edge (0, 1)"``, for error messages.
    """
    for key, value in graph.graph.items():
        yield "metadata", key, value
    for node, attributes in graph.nodes(data=True):
        for key, value in attributes.items():
            yield f"node {node!r}", key, value
    for u, v, attributes in graph.edges(data=True):
        for key, value in attributes.items():
            yield f"edge ({u!r}, {v!r})", key, value


def _refuse_unsaveable_attributes(graph: nx.Graph) -> None:
    """Refuse attribute values that no environment format can store.

    Pickle stores the most kinds of values of any format an environment is
    written to, so a value pickle refuses (a lambda, a lock, an open file)
    could never be saved or sent to a worker process.
    """
    for element, key, value in graph_attribute_items(graph):
        try:
            pickle.dumps(value)
        # Any failure to pickle, whatever the value's own __reduce__ raises,
        # means the attribute cannot be saved; name it and chain the cause.
        except Exception as exc:
            raise TypeError(
                _format_error(
                    f"Track graph {element} attribute {key!r} holds a "
                    f"{type(value).__name__} that cannot be saved ({exc}).",
                    why=(
                        "Why: the environment keeps the track graph, and neither "
                        "pickle (used to send it to worker processes) nor "
                        "to_file, to_dict or NWB can store this value."
                    ),
                    fix=(
                        f"remove {key!r} from the graph, or store a number, "
                        "string or list instead, before building the environment"
                    ),
                )
            ) from exc


@dataclass(frozen=True, eq=False)
class TrackLinearization(ReadOnlyState):
    """Owned track geometry and its absolute linear coordinate system.

    Parameters
    ----------
    graph : networkx.Graph
        Original track graph with node positions and enumerated edge IDs.
    edge_order : sequence of pairs
        Track segments in linear coordinate order.
    edge_spacing : float or sequence of float
        Gaps between successive segments.
    grid_edges : tuple of ndarray
        Full linear bin edges, including gaps.
    active_mask : ndarray of bool, shape (n_linear_bins,)
        Active linear cells in their original coordinate order.

    Raises
    ------
    TypeError
        If a graph, node or edge attribute holds a value that pickle cannot
        store, such as a lambda or a lock.
    """

    graph: nx.Graph
    edge_order: Sequence[tuple[Any, Any]]
    edge_spacing: float | Sequence[float]
    grid_edges: tuple[NDArray[np.float64], ...]
    active_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        _refuse_unsaveable_attributes(self.graph)
        graph = copy.deepcopy(self.graph)
        for node, attrs in graph.nodes(data=True):
            graph._node[node] = _ReadOnlyTrackAttributes(
                attrs, pos=tuple(float(value) for value in attrs["pos"])
            )
        for expected_id, (u, v, attrs) in enumerate(graph.edges(data=True)):
            edge_id = attrs.get("edge_id")
            if (
                not isinstance(edge_id, (int, np.integer))
                or isinstance(edge_id, (bool, np.bool_))
                or edge_id != expected_id
            ):
                raise ValueError(
                    f"track edge ({u!r}, {v!r}) edge_id must equal its graph "
                    f"enumeration index {expected_id}, got {edge_id!r}"
                )
            frozen = _ReadOnlyTrackAttributes(attrs)
            graph._adj[u][v] = graph._adj[v][u] = frozen
        graph.graph = _ReadOnlyTrackAttributes(graph.graph)
        object.__setattr__(self, "graph", nx.freeze(graph))
        object.__setattr__(self, "edge_order", tuple(tuple(e) for e in self.edge_order))
        spacing = self.edge_spacing
        object.__setattr__(
            self,
            "edge_spacing",
            float(spacing)
            if isinstance(spacing, (int, float, np.number))
            else tuple(float(x) for x in spacing),
        )
        edges = tuple(np.array(e, dtype=np.float64, copy=True) for e in self.grid_edges)
        mask = np.array(self.active_mask, dtype=bool, copy=True)
        for array in (*edges, mask):
            array.setflags(write=False)
        object.__setattr__(self, "grid_edges", edges)
        object.__setattr__(self, "active_mask", mask)

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(n_points, n_dims)`` coordinates onto the track."""
        from track_linearization import get_linearized_position

        result = get_linearized_position(
            data_points,
            self.graph,
            list(self.edge_order),
            self.edge_spacing
            if isinstance(self.edge_spacing, (int, float))
            else list(self.edge_spacing),
        ).linear_position.to_numpy()
        return np.asarray(result, dtype=np.float64)

    def linear_to_nd(
        self, linear_coordinates: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map ``(n_points,)`` absolute linear positions into track coordinates."""
        from neurospatial.geometry._track import _project_1d_to_2d

        return _project_1d_to_2d(
            linear_coordinates,
            self.graph,
            list(self.edge_order),
            self.edge_spacing,
        )


@dataclass(frozen=True, eq=False)
class Binning:
    """Published bins, lookup, faces and optional structural components.

    ``face_normals`` holds physical directions in ``bins.pairs`` order. It is
    independent of solver availability: only ``faces`` grants that capability.

    ``dimension_ranges`` is the ``(low, high)`` range of each coordinate axis
    that the bins were built over, for example the data range a hexagonal
    tiling covers. When it is None, an environment uses the extent of the
    cells (the grid, outline or cell shapes), or no range for bins without
    cells. An environment's binning carries the environment's
    ``dimension_ranges``.
    """

    kind: str
    bins: SpatialBins
    locator: Locator
    faces: FacesSource
    rectilinear: RectilinearGrid | None = None
    linearization: TrackLinearization | None = None
    shapes: HexagonShapes | TriangleShapes | PolygonCells | None = None
    outline: Outline | None = None
    clip_mapping: ClipMapping | None = None
    grid_spec: GridBinningSpec | None = None
    face_normals: FaceNormals | None = None
    grid_frame: FramedGrid | None = None
    dimension_ranges: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.dimension_ranges is None:
            return
        try:
            ranges = tuple(
                (float(low), float(high)) for low, high in self.dimension_ranges
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Binning.dimension_ranges must be one (low, high) pair of numbers "
                f"per coordinate axis, got {self.dimension_ranges!r}."
            ) from error
        if len(ranges) != self.bins.n_dims:
            raise ValueError(
                f"Binning.dimension_ranges has {len(ranges)} axes but the bins "
                f"have {self.bins.n_dims}; give one (low, high) pair per axis."
            )
        object.__setattr__(self, "dimension_ranges", ranges)
