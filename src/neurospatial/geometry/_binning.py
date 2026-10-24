"""Owned bin geometry, lookup and finite-volume face capabilities."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial.geometry._arrays import ReadOnlyState, owned_grid
from neurospatial.geometry._capabilities import RectilinearGrid
from neurospatial.geometry._components import HexagonShapes, Outline, TriangleShapes
from neurospatial.geometry._diffusion import (
    FaceGeometry,
)
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._grid_frame import FramedGrid, framed_locator
from neurospatial.geometry._locators import Locator
from neurospatial.geometry._rebin_spec import GridBinningSpec
from neurospatial.geometry._track import project_1d_to_2d

if TYPE_CHECKING:
    from neurospatial.geometry._polygon_cells import PolygonCells


@dataclass(frozen=True, eq=False)
class SpatialBins(ReadOnlyState):
    """Ordered bins and their physical measures, without their connections.

    A bin's index is its ID everywhere: fields, regions, faces and
    :class:`BinConnections` all refer to bins by their position here.

    Bins can differ in what they measure: a track joined to an open field
    has bins measured by length next to bins measured by area. Each bin's
    size is in ``units ** measure_dims[i]``.

    Attributes
    ----------
    centers : ndarray of float64, shape (n_bins, n_dims)
        Finite bin centers in the environment's coordinates.
    sizes : ndarray of float64, shape (n_bins,)
        Finite, strictly positive bin measures in ``units ** measure_dims``.
    measure_dims : ndarray of intp, shape (n_bins,)
        Dimension of each bin's measure: one for track lengths, otherwise the
        coordinate dimension for areas or volumes. An integer given at
        construction applies to every bin.
    """

    centers: NDArray[np.float64]
    sizes: NDArray[np.float64]
    measure_dims: NDArray[np.intp]

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=np.float64, copy=True)
        sizes = np.array(self.sizes, dtype=np.float64, copy=True)
        raw_dims = np.asarray(self.measure_dims)

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
        if raw_dims.ndim == 0:
            if raw_dims.dtype.kind not in "iu" or not 1 <= raw_dims <= n_dims:
                raise ValueError(
                    f"measure_dim must be an integer between 1 and {n_dims}, "
                    f"got {self.measure_dims!r}."
                )
            raw_dims = np.full(n_bins, raw_dims)
        if raw_dims.shape != (n_bins,) or (
            raw_dims.size and raw_dims.dtype.kind not in "iu"
        ):
            raise ValueError(
                f"measure_dims must be one integer per bin, shape ({n_bins},), "
                f"got {raw_dims.dtype} with shape {raw_dims.shape}."
            )
        invalid = (raw_dims < 1) | (raw_dims > n_dims)
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"measure_dims[{index}] must be between 1 and {n_dims}, "
                f"got {raw_dims[index]}."
            )
        dims = np.array(raw_dims, dtype=np.intp, copy=True)
        for name, array in (
            ("centers", centers),
            ("sizes", sizes),
            ("measure_dims", dims),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def from_arrays(
        cls,
        centers: ArrayLike,
        sizes: ArrayLike,
        measure_dim: int | ArrayLike,
    ) -> SpatialBins:
        """Validate and own bin centers and measures.

        Parameters
        ----------
        centers : array-like, shape (n_bins, n_dims)
            Finite bin coordinates.
        sizes : array-like, shape (n_bins,)
            Finite, strictly positive bin measures.
        measure_dim : int or array-like of int, shape (n_bins,)
            Measure dimension in ``[1, n_dims]``, for every bin or per bin.

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
        return cls(np.asarray(centers), np.asarray(sizes), np.asarray(measure_dim))

    def has_measure_dim(self, dim: int) -> bool:
        """Whether every bin's measure has dimension ``dim`` (True with no bins)."""
        return bool(np.all(self.measure_dims == dim))

    @property
    def measure_dim(self) -> int | None:
        """The measure dimension every bin shares, or None when bins mix them."""
        dims = self.measure_dims
        if not len(dims) or np.any(dims != dims[0]):
            return None
        return int(dims[0])

    @property
    def n_bins(self) -> int:
        """Number of active bins."""
        return int(self.centers.shape[0])

    @property
    def n_dims(self) -> int:
        """Number of coordinate columns."""
        return int(self.centers.shape[1])


def refuse_mixed_measure_conductance(
    measure_dims: NDArray[np.intp],
    pairs: NDArray[np.intp],
    conducting: NDArray[np.bool_],
    *,
    link: str,
    fix: str,
) -> None:
    """Refuse a conducting link between bins measured in different dimensions.

    Diffusion through it would exchange mass between densities of different
    units, per length and per area, so its result would depend on the length
    unit whatever the conductance.

    Parameters
    ----------
    measure_dims : ndarray of intp, shape (n_bins,)
        Each bin's measure dimension.
    pairs : ndarray of intp, shape (n_links, 2)
        The links' bins.
    conducting : ndarray of bool, shape (n_links,)
        Which links carry diffusion.
    link, fix : str
        What a link is (``"bridge"``, ``"face"``) and how to avoid the error.

    Raises
    ------
    ValueError
        Naming the first such link.
    """
    ends = pairs[conducting]
    mixed = np.flatnonzero(measure_dims[ends[:, 0]] != measure_dims[ends[:, 1]])
    if len(mixed):
        u, v = ends[mixed[0]]
        raise ValueError(
            _format_error(
                f"A conducting {link} joins bin {u}, measured in dimension "
                f"{measure_dims[u]}, to bin {v}, measured in dimension "
                f"{measure_dims[v]}.",
                why="Why: diffusion through it would exchange mass between a "
                "density per length and a density per area, so the result "
                "would change with the units (cm or mm), whatever the "
                "conductance.",
                fix=fix,
            )
        )


@dataclass(frozen=True, eq=False)
class BinConnections(ReadOnlyState):
    """Neighbor pairs between bins, and which of them movement can cross.

    A binning defines its neighbor pairs once. Walls close pairs instead of
    removing them, so two environments over the same bins list their pairs in
    the same order and differ only in ``is_open``. Everything that moves
    between bins (paths, distances, smoothing, gradients) uses the open pairs,
    ``pairs``, which is ``env.edge_pairs``.

    Attributes
    ----------
    all_pairs : ndarray of intp, shape (n_all_pairs, 2)
        Every neighbor pair, open or closed, in authoritative order and
        orientation.
    all_distances : ndarray of float64, shape (n_all_pairs,)
        Finite, non-negative navigation distances, distinct from the flux
        distances used by finite-volume faces at track junctions.
    is_open : ndarray of bool, shape (n_all_pairs,)
        False where a wall closes the edge between the pair's bins.
    closed_by : ndarray of intp, shape (n_all_pairs,)
        Index into the arena's ``walls`` of the wall that closes each pair:
        the one covering most of the edge the pair's cells share, or for
        cells that meet only at a corner, the first wall that corner touches,
        else the first wall the link between their centers crosses. -1 for
        open pairs and for closed pairs no wall of the arena closes.
        Publication records it when it applies walls.
    pairs : ndarray of intp, shape (n_pairs, 2)
        The open pairs, in the order of ``all_pairs``.
    pair_distances : ndarray of float64, shape (n_pairs,)
        Navigation distances of the open pairs.
    """

    all_pairs: NDArray[np.intp]
    all_distances: NDArray[np.float64]
    is_open: NDArray[np.bool_]
    closed_by: NDArray[np.intp]
    pairs: NDArray[np.intp] = field(init=False, repr=False)
    pair_distances: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw_pairs = np.asarray(self.all_pairs)
        distances = np.array(self.all_distances, dtype=np.float64, copy=True)
        if raw_pairs.ndim != 2 or raw_pairs.shape[1] != 2:
            raise ValueError(
                f"pairs must have shape (n_pairs, 2), got {raw_pairs.shape}."
            )
        index: int | tuple[int, ...]
        if raw_pairs.size and raw_pairs.dtype.kind not in "iu":
            if raw_pairs.dtype.kind == "f":
                invalid = ~np.isfinite(raw_pairs) | (raw_pairs != np.floor(raw_pairs))
                if np.any(invalid):
                    index = tuple(int(i) for i in np.argwhere(invalid)[0])
                    raise ValueError(f"pairs{index} must contain integer bin indices.")
            else:
                raise ValueError("pairs(0, 0) must contain integer bin indices.")
        invalid = raw_pairs < 0
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"pairs{index} must be a nonnegative bin index.")
        pairs = np.array(raw_pairs, dtype=np.intp, copy=True)
        self_pairs = pairs[:, 0] == pairs[:, 1]
        if np.any(self_pairs):
            index = int(np.flatnonzero(self_pairs)[0])
            raise ValueError(f"pairs[{index}] is a self pair.")
        # One integer per unordered pair; much faster than unique rows.
        low, high = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(*pairs.T)
        keys = low * (int(high.max(initial=0)) + 1) + high
        _, first_indices = np.unique(keys, return_index=True)
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
        is_open = np.asarray(self.is_open)
        if is_open.dtype.kind != "b" or is_open.shape != (len(pairs),):
            raise ValueError(
                f"is_open must be a boolean array of shape ({len(pairs)},), "
                f"got {is_open.dtype} with shape {is_open.shape}."
            )
        is_open = np.array(is_open, dtype=bool, copy=True)
        closed_by = np.asarray(self.closed_by)
        if (
            closed_by.shape != (len(pairs),)
            or (closed_by.size and closed_by.dtype.kind not in "iu")
            or np.any(closed_by < -1)
            or np.any(closed_by[is_open] != -1)
        ):
            raise ValueError(
                f"closed_by must hold one wall index >= -1 per pair, shape "
                f"({len(pairs)},), and -1 for every open pair."
            )
        closed_by = np.array(closed_by, dtype=np.intp, copy=True)
        values = {
            "all_pairs": pairs,
            "all_distances": distances,
            "is_open": is_open,
            "closed_by": closed_by,
            "pairs": pairs if is_open.all() else pairs[is_open],
            "pair_distances": distances if is_open.all() else distances[is_open],
        }
        for name, array in values.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def from_arrays(
        cls,
        pairs: ArrayLike,
        pair_distances: ArrayLike,
        is_open: ArrayLike | None = None,
        closed_by: ArrayLike | None = None,
    ) -> BinConnections:
        """Validate and own neighbor pairs, their distances and openness.

        Parameters
        ----------
        pairs : array-like of int, shape (n_pairs, 2)
            Distinct unordered neighbor pairs, kept in their input order.
        pair_distances : array-like, shape (n_pairs,)
            Finite, non-negative navigation distances.
        is_open : array-like of bool, shape (n_pairs,), optional
            Which pairs movement can cross. All pairs are open by default.
        closed_by : array-like of int, shape (n_pairs,), optional
            Index of the arena wall closing each pair, -1 for none (the
            default). Publication recomputes it from the arena's walls.

        Returns
        -------
        BinConnections
            Validated independent snapshot with read-only arrays.

        Raises
        ------
        ValueError
            If shapes or values are invalid. Elementwise errors identify the
            offending array index.
        """
        all_pairs = np.asarray(pairs)
        if all_pairs.size == 0:
            all_pairs = all_pairs.reshape(-1, 2)
        return cls(
            all_pairs,
            np.asarray(pair_distances),
            np.ones(len(all_pairs), dtype=bool)
            if is_open is None
            else np.asarray(is_open),
            np.full(len(all_pairs), -1, dtype=np.intp)
            if closed_by is None
            else np.asarray(closed_by),
        )

    def closed(self, keep_open: NDArray[np.bool_]) -> BinConnections:
        """Close the open pairs where ``keep_open`` is False.

        Parameters
        ----------
        keep_open : ndarray of bool, shape (n_pairs,)
            One flag per open pair, in ``pairs`` order.

        Returns
        -------
        BinConnections
            The same pairs, with the given open pairs closed. Closed pairs
            stay closed; newly closed pairs have no closing wall (-1) until
            one is recorded.
        """
        is_open = self.is_open.copy()
        is_open[np.flatnonzero(self.is_open)[~keep_open]] = False
        return BinConnections(
            self.all_pairs, self.all_distances, is_open, self.closed_by
        )

    def select(
        self, child_of: NDArray[np.intp]
    ) -> tuple[BinConnections, NDArray[np.bool_]]:
        """Keep the pairs between kept bins, renumbered as the kept bins are.

        Parameters
        ----------
        child_of : ndarray of intp, shape (n_bins,)
            New index of each bin, or -1 for a dropped bin.

        Returns
        -------
        connections : BinConnections
            Pairs whose bins are both kept, in their original order, with
            their openness.
        open_kept : ndarray of bool, shape (n_pairs,)
            Which open pairs survive, in ``pairs`` order, to select arrays
            aligned with the open pairs such as faces and normals.
        """
        mapped = child_of[self.all_pairs]
        kept = np.all(mapped >= 0, axis=1)
        connections = BinConnections(
            mapped[kept].reshape(-1, 2),
            self.all_distances[kept],
            self.is_open[kept],
            self.closed_by[kept],
        )
        return connections, kept[self.is_open]

    def check_bins(self, n_bins: int) -> None:
        """Raise if a pair names a bin outside ``range(n_bins)``."""
        invalid = self.all_pairs >= n_bins
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"pairs{index} must be in [0, {n_bins}).")


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
        """Return an actionable error when the named operation needs faces.

        A source that is not ``available`` returns a
        ``MissingCellGeometryError``, the error that marks bins which cannot
        smooth.
        """
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


class MissingCellGeometryError(NotImplementedError):
    """An operation needs finite-volume cell faces the bins do not have."""


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

    def missing_error(self, operation: str) -> MissingCellGeometryError:
        """Name the operation, missing geometry and supported recovery."""
        return MissingCellGeometryError(
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


def _node_position(node: Any, attributes: dict[str, Any]) -> tuple[float, ...]:
    """A track node's ``pos`` as finite floats, or an error naming the node."""
    try:
        position = tuple(float(value) for value in attributes["pos"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            _format_error(
                f"Track graph node {node!r} has no numeric 'pos' attribute "
                f"(got {attributes.get('pos')!r}).",
                why="Why: the linear coordinate runs along the segments "
                "between node positions.",
                fix=f"set graph.nodes[{node!r}]['pos'] = (x, y) for every node",
            )
        ) from error
    if not position or not np.all(np.isfinite(position)):
        raise ValueError(
            f"Track graph node {node!r} 'pos' must be finite coordinates, "
            f"got {position!r}."
        )
    return position


def _edge_spacing(spacing: Any, n_segments: int) -> float | tuple[float, ...]:
    """Validate the gaps between ``n_segments`` successive track segments."""
    scalar = np.ndim(spacing) == 0
    try:
        values = np.asarray(spacing, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"TrackLinearization.edge_spacing must be numbers, got {spacing!r}."
        ) from error
    if isinstance(spacing, (str, bytes, bool, np.bool_)) or (
        not scalar and values.shape != (max(n_segments - 1, 0),)
    ):
        raise ValueError(
            "TrackLinearization.edge_spacing must be one number or one gap per "
            f"pair of successive segments ({max(n_segments - 1, 0)}), got "
            f"{spacing!r}."
        )
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError(
            "TrackLinearization.edge_spacing must be finite and non-negative, "
            f"got {spacing!r}."
        )
    return float(values) if scalar else tuple(float(value) for value in values)


@dataclass(frozen=True, eq=False)
class TrackLinearization(ReadOnlyState):
    """Owned track geometry and its absolute linear coordinate system.

    Parameters
    ----------
    graph : networkx.Graph
        Original track graph with node positions and enumerated edge IDs.
    edge_order : sequence of pairs
        Track segments in linear coordinate order; each pair is an edge of
        ``graph``.
    edge_spacing : float or sequence of float
        Finite, non-negative gaps between successive segments: one value for
        every gap, or ``len(edge_order) - 1`` values.
    grid_edges : tuple of ndarray
        Full linear bin edges, including gaps: one finite, strictly
        increasing axis.
    active_mask : ndarray of bool, shape (n_linear_bins,)
        Active linear cells in their original coordinate order.

    Raises
    ------
    TypeError
        If a graph, node or edge attribute holds a value that pickle cannot
        store, such as a lambda or a lock.
    ValueError
        If a node has no finite ``pos``, ``edge_order`` names a pair that is
        not a graph edge, ``edge_spacing`` is negative or has the wrong
        count, or ``grid_edges`` is not one linear axis matching
        ``active_mask``.
    """

    graph: nx.Graph
    edge_order: Sequence[tuple[Any, Any]]
    edge_spacing: float | Sequence[float]
    grid_edges: tuple[NDArray[np.float64], ...]
    active_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, nx.Graph) or (
            self.graph.is_directed() or self.graph.is_multigraph()
        ):
            raise ValueError(
                "TrackLinearization.graph must be an undirected simple "
                f"networkx.Graph, got {type(self.graph).__name__}."
            )
        _refuse_unsaveable_attributes(self.graph)
        graph = copy.deepcopy(self.graph)
        for node, attrs in graph.nodes(data=True):
            graph._node[node] = _ReadOnlyTrackAttributes(
                attrs, pos=_node_position(node, attrs)
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
        edge_order = tuple(tuple(edge) for edge in self.edge_order)
        for index, edge in enumerate(edge_order):
            if len(edge) != 2 or not graph.has_edge(*edge):
                raise ValueError(
                    f"TrackLinearization.edge_order[{index}] = {edge!r} is not "
                    "an edge of the track graph; list the graph's edges in "
                    "linear order."
                )
        object.__setattr__(self, "edge_order", edge_order)
        object.__setattr__(
            self, "edge_spacing", _edge_spacing(self.edge_spacing, len(edge_order))
        )
        if len(self.grid_edges) != 1:
            raise ValueError(
                "TrackLinearization.grid_edges must hold one linear axis, got "
                f"{len(self.grid_edges)}."
            )
        edges, _, mask = owned_grid(
            self.grid_edges, None, self.active_mask, name="TrackLinearization"
        )
        object.__setattr__(self, "grid_edges", edges)
        object.__setattr__(self, "active_mask", mask)

    def to_linear(self, data_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map ``(n_points, n_dims)`` coordinates onto the track."""
        # Imported on use: track_linearization loads pandas and matplotlib (~0.6 s).
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
        """Map ``(n_points,)`` absolute linear positions into track coordinates.

        Returns ``(n_points, n_dims)`` coordinates. A position on no segment
        (below 0, past the end, inside a gap, or not finite) is a row of NaN.
        """

        return project_1d_to_2d(
            linear_coordinates,
            self.graph,
            list(self.edge_order),
            self.edge_spacing,
        )


@dataclass(frozen=True, eq=False)
class Binning:
    """Published bins, their connections, lookup, faces and optional components.

    ``faces`` and ``face_normals`` are aligned with the open pairs,
    ``connections.pairs``. Face normals are independent of solver
    availability: only ``faces`` grants that capability.

    ``dimension_ranges`` is the ``(low, high)`` extent of each coordinate
    axis that the bins' cells cover: a grid's active cells, the outline or
    cell shapes otherwise, and the stretch of centerline a track's bins
    cover. An environment computes that extent from the cells; only bins
    without cells keep a declared value (or None). An environment's
    binning carries the environment's ``dimension_ranges``.

    A ``grid_frame`` is handed to the affine step of ``locator`` (a
    :class:`TransformedLocator`), so ``locator.locate`` assigns points by the
    frame's half-open source cells, as linear time allocation does.
    """

    kind: str
    bins: SpatialBins
    connections: BinConnections
    locator: Locator
    faces: FacesSource
    rectilinear: RectilinearGrid | None = None
    linearization: TrackLinearization | None = None
    shapes: HexagonShapes | TriangleShapes | PolygonCells | None = None
    outline: Outline | None = None
    grid_spec: GridBinningSpec | None = None
    face_normals: FaceNormals | None = None
    grid_frame: FramedGrid | None = None
    dimension_ranges: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.grid_frame is not None:
            # Lookup and linear allocation share the frame's cell rule.
            object.__setattr__(
                self, "locator", framed_locator(self.locator, self.grid_frame)
            )
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


def require_rectilinear(
    binning: Binning, operation: str, *, fix: str
) -> RectilinearGrid:
    """Return the published rectangular cells, or raise naming ``operation``.

    Raises
    ------
    NotImplementedError
        If the binning has no axis-aligned rectangular cells.
    """
    if binning.rectilinear is not None:
        return binning.rectilinear
    raise NotImplementedError(
        _format_error(
            f"{operation} needs a grid of axis-aligned rectangular cells; this "
            f"environment's {binning.kind} layout does not have one.",
            why=(
                "Why: it reads the cell edges along each axis, which grids built "
                "by from_samples, from_grid_mask, from_pixel_mask and "
                "from_polygon provide."
            ),
            fix=fix,
        )
    )


def require_grid_context(
    binning: Binning, operation: str, *, fix: str
) -> tuple[RectilinearGrid, FramedGrid | None]:
    """Read source axes and their optional world frame without copying grids."""
    if binning.grid_frame is not None:
        return binning.grid_frame.source, binning.grid_frame
    return require_rectilinear(binning, operation, fix=fix), None
