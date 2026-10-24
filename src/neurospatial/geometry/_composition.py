"""Owned component snapshots and lookup for combined environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from neurospatial.geometry._affine import linear_scale
from neurospatial.geometry._arena import Arena, contains_with_error
from neurospatial.geometry._arrays import ReadOnlyState, active_order, per_dimension
from neurospatial.geometry._binning import (
    BinConnections,
    Binning,
    KnownFaces,
    MissingCellGeometryError,
    MissingFaces,
    SpatialBins,
    refuse_mixed_measure_conductance,
)
from neurospatial.geometry._components import TriangleShapes
from neurospatial.geometry._diffusion import DiffusionGeometry, FaceGeometry
from neurospatial.geometry._face_builders import triangle_skew_message
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._locators import (
    LookupIndexes,
    SubsetLocator,
    TransformedLocator,
    locate_within,
)
from neurospatial.geometry._polygon_cells import ClippedLocator

if TYPE_CHECKING:
    from neurospatial.geometry._locators import Locator


@dataclass(frozen=True, eq=False)
class CombinedLocator(ReadOnlyState):
    """Keep child geometry and domains without retaining live environments.

    Child order fixes bin offsets and boundary ties. Bridge arrays describe
    navigation and optional geometric conductance; they never change lookup.
    """

    keys: tuple[str, ...]
    children: tuple[Binning, ...]
    arenas: tuple[Arena, ...]
    bridge_pairs: NDArray[np.intp]
    bridge_distances: NDArray[np.float64]
    bridge_conductances: NDArray[np.float64]

    def __post_init__(self) -> None:
        keys, children, arenas = (
            tuple(self.keys),
            tuple(self.children),
            tuple(self.arenas),
        )
        if (
            not keys
            or len(set(keys)) != len(keys)
            or any(not isinstance(key, str) or not key or "/" in key for key in keys)
        ):
            raise ValueError(
                "Component keys must be unique nonempty strings without '/'."
            )
        if len(children) != len(keys) or len(arenas) != len(keys):
            raise ValueError(
                "Component keys, children and arenas must have equal lengths."
            )
        if any(type(child) is not Binning for child in children):
            raise TypeError("Composition children must be built-in Binning snapshots.")
        for child in children:
            child.connections.check_bins(child.bins.n_bins)
        dims = children[0].bins.n_dims
        if any(
            child.bins.n_dims != dims
            or arena.n_dims != dims
            or arena.coordinate_kind != "cartesian"
            for child, arena in zip(children, arenas, strict=True)
        ):
            raise ValueError("Components need matching Cartesian dimensions.")
        centers = np.concatenate([child.bins.centers for child in children])
        bridges = BinConnections.from_arrays(self.bridge_pairs, self.bridge_distances)
        bridges.check_bins(len(centers))
        geometric = np.linalg.norm(
            centers[bridges.pairs[:, 1]] - centers[bridges.pairs[:, 0]], axis=1
        )
        if not np.allclose(
            bridges.pair_distances, geometric, rtol=32 * np.finfo(float).eps, atol=0
        ):
            raise ValueError(
                "Inferred bridge distances must match endpoint separation."
            )
        for child, arena in zip(children, arenas, strict=True):
            if arena.domain_known and not np.all(arena.contains(child.bins.centers)):
                raise ValueError("A component arena excludes its own bin centers.")
        conductances = np.array(self.bridge_conductances, dtype=np.float64, copy=True)
        if conductances.shape != (len(bridges.pairs),) or np.any(
            ~np.isfinite(conductances) | (conductances < 0)
        ):
            raise ValueError(
                "Bridge conductances must be finite nonnegative edge values."
            )
        offsets = np.cumsum([0, *[child.bins.n_bins for child in children]])
        owners = np.repeat(np.arange(len(children)), np.diff(offsets))
        if len(bridges.pairs) and np.any(
            owners[bridges.pairs[:, 0]] == owners[bridges.pairs[:, 1]]
        ):
            raise ValueError("Bridges must join bins in different components.")
        refuse_mixed_measure_conductance(
            np.concatenate([child.bins.measure_dims for child in children]),
            bridges.pairs,
            conductances > 0,
            link="bridge",
            fix="leave that bridge navigation-only (no conductance), or combine "
            "components whose bins are measured the same way",
        )
        conductances.setflags(write=False)
        for name, value in (
            ("keys", keys),
            ("children", children),
            ("arenas", arenas),
            ("bridge_pairs", bridges.pairs),
            ("bridge_distances", bridges.pair_distances),
            ("bridge_conductances", conductances),
        ):
            object.__setattr__(self, name, value)

    @property
    def offsets(self) -> NDArray[np.intp]:
        """Return cumulative child bin offsets, including the final count."""
        return np.cumsum(
            [0, *[child.bins.n_bins for child in self.children]], dtype=np.intp
        )

    def spatial_bins(self) -> SpatialBins:
        """Assemble the authoritative concatenated bins."""
        return SpatialBins.from_arrays(
            np.concatenate([child.bins.centers for child in self.children]),
            np.concatenate([child.bins.sizes for child in self.children]),
            np.concatenate([child.bins.measure_dims for child in self.children]),
        )

    def bin_connections(self) -> BinConnections:
        """Each child's pairs, open and closed, then the open bridges."""
        offsets = self.offsets
        links = [child.connections for child in self.children]
        return BinConnections(
            np.concatenate(
                [
                    *[
                        link.all_pairs + offset
                        for link, offset in zip(links, offsets[:-1], strict=True)
                    ],
                    self.bridge_pairs,
                ]
            ),
            np.concatenate(
                [*[link.all_distances for link in links], self.bridge_distances]
            ),
            np.concatenate(
                [
                    *[link.is_open for link in links],
                    np.ones(len(self.bridge_pairs), dtype=bool),
                ]
            ),
            # Each child's wall indices move to its place in the combined
            # arena's walls, which list every child's walls in order.
            np.concatenate(
                [
                    *[
                        np.where(link.closed_by >= 0, link.closed_by + start, -1)
                        for link, start in zip(links, self.wall_offsets, strict=True)
                    ],
                    np.full(len(self.bridge_pairs), -1, dtype=np.intp),
                ]
            ),
        )

    @property
    def wall_offsets(self) -> NDArray[np.intp]:
        """Where each child's walls start in the combined arena's walls."""
        return np.cumsum(
            [0, *[len(arena.walls) for arena in self.arenas[:-1]]], dtype=np.intp
        )

    def locate(
        self, points: NDArray[np.float64], indexes: LookupIndexes | None = None
    ) -> NDArray[np.intp]:
        """Locate native points with each child's original arena and lookup."""
        query = np.asarray(points, dtype=np.float64)
        return self.locate_within(query, np.zeros(query.shape), indexes)

    def locate_within(
        self,
        points: NDArray[np.float64],
        error: NDArray[np.float64],
        indexes: LookupIndexes | None = None,
        *,
        half_open: bool = False,
    ) -> NDArray[np.intp]:
        """Locate points known to within ``error`` in each child's domain.

        Parameters
        ----------
        points : ndarray of float64, shape (n_points, n_dims)
            Query coordinates.
        error : ndarray of float64, shape (n_points, n_dims)
            Absolute bound on how far each true coordinate may be from the
            given one; zeros for exact coordinates.
        indexes : LookupIndexes, optional
            Keeps the children's search structures.
        half_open : bool, default=False
            Leave points on a grid child's outline unowned, so an enclosing
            combined environment can ask every component for half-open
            owners first.

        Returns
        -------
        ndarray of intp, shape (n_points,)
            Combined bin index of each point, or -1.
        """
        query = np.asarray(points, dtype=np.float64)
        dims = self.children[0].bins.n_dims
        if query.ndim != 2 or query.shape[1] != dims:
            raise ValueError(f"Expected positions of shape (n_points, {dims}).")
        result = np.full(len(query), -1, dtype=np.intp)
        finite = np.all(np.isfinite(query), axis=1)
        # A grid child's outline is closed, but a face it shares with another
        # child keeps its half-open owner. The first pass asks every child
        # with grid outlines left open (other locators answer in full); only
        # the second pass lets a grid child claim a point on its outline.
        # With half_open, the enclosing environment runs that second pass.
        for interior in (True,) if half_open else (True, False):
            for child, arena, offset in zip(
                self.children, self.arenas, self.offsets[:-1], strict=True
            ):
                rows = np.flatnonzero(finite & (result < 0))
                if arena.domain_known:
                    rows = rows[contains_with_error(arena, query[rows], error[rows])]
                local = locate_within(
                    child.locator,
                    query[rows],
                    error[rows],
                    indexes,
                    half_open=interior,
                )
                valid = local >= 0
                result[rows[valid]] = offset + local[valid]
        return result


def combined_faces(
    locator: CombinedLocator, connections: BinConnections, sizes: NDArray[np.float64]
) -> KnownFaces | MissingFaces:
    """Join the components' faces, in the combined open-pair order.

    Parameters
    ----------
    locator : CombinedLocator
        The components and their bridges.
    connections : BinConnections
        ``locator.bin_connections()``.
    sizes : ndarray of float64, shape (n_bins,)
        ``locator.spatial_bins().sizes``.

    Returns
    -------
    KnownFaces or MissingFaces
        Missing when a bridge joins components, since a bridge has no
        physical face, or when a component has no faces.
    """
    if len(locator.bridge_pairs) or not all(
        child.faces.available for child in locator.children
    ):
        return MissingFaces(
            "Combined navigation bridges have no physical face geometry; use "
            "scalar diffusion when child geometry is certified."
        )
    parts = [child.build_faces() for child in locator.children]
    return KnownFaces(
        FaceGeometry(
            connections.pairs,
            np.concatenate([part.A for part in parts]),
            np.concatenate([part.distance for part in parts]),
            sizes,
        )
    )


def combined_normals(locator: CombinedLocator) -> FaceNormals | None:
    """Combine authoritative child normals; bridges have no physical normal."""

    if any(child.face_normals is None for child in locator.children):
        return None
    return FaceNormals(
        np.concatenate(
            [
                *[
                    child.face_normals.values
                    for child in locator.children
                    if child.face_normals is not None
                ],
                np.zeros((len(locator.bridge_pairs), locator.children[0].bins.n_dims)),
            ]
        )
    )


def triangle_skew_messages(binning: Binning) -> tuple[str, ...]:
    """Skew warning texts for the triangle meshes a binning smooths on.

    A triangle-mesh binning gives at most one text. A combined binning is
    walked as :func:`combined_diffusion` walks it, through subsets and
    transforms, and gives one text per skewed mesh component, naming the
    component. A subset counts only the edges between retained triangles.
    Skew is an angle, so the similarity transforms that combined smoothing
    accepts do not change it.

    Parameters
    ----------
    binning : Binning
        The binning a field is smoothed on.

    Returns
    -------
    tuple of str
        One warning text per skewed mesh; empty when none is skewed.
    """
    messages: list[str] = []

    def visit(
        child: Binning, active: NDArray[np.bool_] | None, path: tuple[str, ...]
    ) -> None:
        shapes = child.shapes
        if not isinstance(shapes, TriangleShapes):
            walk(child.locator, active, path)
            return
        pairs = child.connections.pairs
        if active is not None:
            pairs = pairs[active[pairs].all(axis=1)]
        message = triangle_skew_message(
            shapes.points, shapes.triangles, child.bins.centers, pairs
        )
        if message is not None:
            if path:
                message += f" (combined component {'/'.join(path)!r})"
            messages.append(message)

    def walk(
        locator: Locator, active: NDArray[np.bool_] | None, path: tuple[str, ...]
    ) -> None:
        if isinstance(locator, CombinedLocator):
            for key, child, start, end in zip(
                locator.keys,
                locator.children,
                locator.offsets[:-1],
                locator.offsets[1:],
                strict=True,
            ):
                retained = None if active is None else active[start:end]
                if retained is None or np.any(retained):
                    visit(child, retained, (*path, key))
        elif isinstance(locator, SubsetLocator):
            selected = active_order(locator.parent_to_child)
            retained = np.zeros(len(locator.parent_to_child), dtype=np.bool_)
            retained[selected if active is None else selected[active]] = True
            walk(locator.base, retained, path)
        elif isinstance(locator, TransformedLocator):
            walk(locator.base, active, path)

    visit(binning, None, ())
    return tuple(messages)


def combined_diffusion(binning: Binning) -> DiffusionGeometry | None:
    """Build scalar diffusion from child blocks and explicit geometric couplings.

    Return None for ordinary geometry. Missing child faces remain a whole-value
    refusal; navigation bridges never acquire a fabricated physical face.
    Clipped cells must supply their own certified faces. Do not unwrap a
    ClippedLocator here: its parent's weights describe different cell geometry.
    """

    def weights(
        locator: Locator, active: NDArray[np.bool_] | None = None
    ) -> tuple[sparse.csr_matrix, NDArray[np.intp]] | None:
        if isinstance(locator, CombinedLocator):
            blocks = []
            for child, start, end in zip(
                locator.children, locator.offsets[:-1], locator.offsets[1:], strict=True
            ):
                retained = None if active is None else active[start:end]
                if retained is not None and not np.any(retained):
                    blocks.append(
                        sparse.csr_matrix((child.bins.n_bins, child.bins.n_bins))
                    )
                    continue
                if child.faces.available:
                    block = DiffusionGeometry.from_faces(child.build_faces()).W
                else:
                    nested = weights(child.locator, retained)
                    if nested is None:
                        child.build_faces()  # Preserve the actionable missing-child refusal.
                        raise AssertionError(
                            "missing faces unexpectedly returned geometry"
                        )
                    block = nested[0]
                blocks.append(block)
            result = sparse.block_diag(blocks, format="csr")
            if len(locator.bridge_pairs):
                u, v = locator.bridge_pairs.T
                c = locator.bridge_conductances
                result += sparse.csr_matrix(
                    (
                        np.concatenate([c, c]),
                        (np.concatenate([u, v]), np.concatenate([v, u])),
                    ),
                    shape=result.shape,
                )
            return result, np.concatenate(
                [child.bins.measure_dims for child in locator.children]
            )
        if isinstance(locator, SubsetLocator):
            selected = active_order(locator.parent_to_child)
            retained = np.zeros(len(locator.parent_to_child), dtype=np.bool_)
            retained[selected if active is None else selected[active]] = True
            result = weights(locator.base, retained)
            if result is None:
                return None
            matrix, dims = result
            return matrix[selected][:, selected].tocsr(), dims[selected]
        if isinstance(locator, TransformedLocator):
            result = weights(locator.base, active)
            if result is None:
                return None
            matrix, dims = result
            measured = linear_scale(locator.forward[:-1, :-1])
            scale = measured.scale
            if not measured.is_similarity():
                raise MissingCellGeometryError(
                    "Combined scalar diffusion requires a similarity transform; use the original coordinates."
                )
            # Conductance has units length ** (measure_dim - 2) of the bins it
            # joins; positive couplings only join bins of one measure.
            coupling = matrix.tocoo()
            data = coupling.data * per_dimension(
                dims[coupling.row], lambda dim: scale ** (dim - 2)
            )
            matrix = sparse.csr_matrix(
                (data, (coupling.row, coupling.col)), shape=matrix.shape
            )
            if not np.all(np.isfinite(data)) or np.any(
                (coupling.data > 0) & (data == 0)
            ):
                raise ValueError(
                    "Transformed conductances exceed floating-point range; rescale the coordinates."
                )
            return matrix, dims
        return None

    if binning.faces.available:
        return None
    result = weights(binning.locator)
    if result is None:
        return None
    matrix, _dim = result
    matrix.eliminate_zeros()
    n_components, labels = connected_components(matrix, directed=False)
    for array in (matrix.data, matrix.indices, matrix.indptr, labels):
        array.setflags(write=False)
    return DiffusionGeometry(matrix, binning.bins.sizes, int(n_components), labels)


def composition_view(
    locator: object,
) -> tuple[CombinedLocator, NDArray[np.intp], float] | None:
    """Find the combined binning below a locator and its bins' places in it.

    Parameters
    ----------
    locator : object
        A binning's locator.

    Returns
    -------
    tuple or None
        ``(root, indices, scale)``: the innermost :class:`CombinedLocator`
        reached through subsets, affine maps and clips; for each current bin,
        the index of the root bin it comes from, shape ``(n_bins,)``; and the
        geometric-mean linear scale of the affine maps on the way. None when
        no combined binning lies below ``locator``.
    """

    if isinstance(locator, CombinedLocator):
        return locator, np.arange(locator.offsets[-1], dtype=np.intp), 1.0
    if isinstance(locator, (SubsetLocator, TransformedLocator, ClippedLocator)):
        result = composition_view(locator.base)
        if result is None:
            return None
        root, indices, scale = result
        if isinstance(locator, SubsetLocator):
            selected = active_order(locator.parent_to_child)
            indices = indices[selected]
        elif isinstance(locator, ClippedLocator):
            indices = indices[locator.mapping.parent_bin_indices]
        else:
            scale *= linear_scale(locator.forward[:-1, :-1]).scale
        return root, indices, scale
    return None


def bridge_values(binning: Binning) -> dict[frozenset[int], float]:
    """Map retained navigation bridges, including nested children, to conductance."""
    view = composition_view(binning.locator)
    if view is None:
        return {}
    root, indices, scale = view
    original = {
        frozenset(map(int, pair)): float(value)
        for pair, value in zip(root.bridge_pairs, root.bridge_conductances, strict=True)
    }
    for child, offset in zip(root.children, root.offsets[:-1], strict=True):
        original.update(
            {
                frozenset(int(index + offset) for index in pair): value
                for pair, value in bridge_values(child).items()
            }
        )
    result = {}
    for u, v in binning.connections.pairs:
        pair = frozenset((int(indices[u]), int(indices[v])))
        if pair in original:
            # Positive conductances only join bins of one measure.
            dim = int(binning.bins.measure_dims[u])
            result[frozenset((int(u), int(v)))] = original[pair] * scale ** (dim - 2)
    return result


def composition_component_keys(binning: Binning) -> tuple[str, ...]:
    """Return stable component keys from retained composition data."""

    result = composition_view(binning.locator)
    return () if result is None else result[0].keys


def composition_bin_sources(binning: Binning) -> tuple[tuple[str, int], ...] | None:
    """Resolve retained component and local bin IDs in current bin order."""

    result = composition_view(binning.locator)
    if result is None:
        return None
    root, indices, _scale = result
    offsets = root.offsets
    owners = np.searchsorted(offsets[1:], indices, side="right")
    return tuple(
        (root.keys[owner], int(index - offsets[owner]))
        for index, owner in zip(indices, owners, strict=True)
    )
