"""Owned component snapshots and lookup for combined environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from neurospatial.geometry._arena import Arena
from neurospatial.geometry._arrays import ReadOnlyState, per_dimension
from neurospatial.geometry._binning import (
    BinConnections,
    Binning,
    SpatialBins,
    refuse_mixed_measure_conductance,
)
from neurospatial.geometry._diffusion import DiffusionGeometry
from neurospatial.geometry._face_normals import FaceNormals
from neurospatial.geometry._locators import (
    LookupIndexes,
    RectilinearLocator,
    SubsetLocator,
    TransformedLocator,
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
        dims = self.children[0].bins.n_dims
        if query.ndim != 2 or query.shape[1] != dims:
            raise ValueError(f"Expected positions of shape (n_points, {dims}).")
        result = np.full(len(query), -1, dtype=np.intp)
        finite = np.all(np.isfinite(query), axis=1)
        # A grid child's outline is closed, but a face it shares with another
        # child keeps its half-open owner: every child is asked for interior
        # owners before any child claims a point on its outline.
        for interior in (True, False):
            for child, arena, offset in zip(
                self.children, self.arenas, self.offsets[:-1], strict=True
            ):
                rows = np.flatnonzero(finite & (result < 0))
                if arena.domain_known:
                    rows = rows[arena.contains(query[rows])]
                locator = child.locator
                if interior and isinstance(locator, RectilinearLocator):
                    local = locator.locate_half_open(query[rows])
                else:
                    local = locator.locate(query[rows], indexes)
                valid = local >= 0
                result[rows[valid]] = offset + local[valid]
        return result


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
                    block = DiffusionGeometry.from_faces(child.faces.build()).W
                else:
                    nested = weights(child.locator, retained)
                    if nested is None:
                        child.faces.build()  # Preserve the actionable missing-child refusal.
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
            selected = np.flatnonzero(locator.parent_to_child >= 0)
            selected = selected[np.argsort(locator.parent_to_child[selected])]
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
            linear = np.linalg.inv(locator.inverse)[:-1, :-1]
            scale = float(np.exp(np.linalg.slogdet(linear)[1] / len(linear)))
            normalized = linear / scale
            if not np.allclose(
                normalized @ normalized.T, np.eye(len(linear)), rtol=0, atol=1e-12
            ):
                raise NotImplementedError(
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
    """Resolve current bins to root component IDs and a similarity scale."""

    if isinstance(locator, CombinedLocator):
        return locator, np.arange(locator.offsets[-1], dtype=np.intp), 1.0
    if isinstance(locator, (SubsetLocator, TransformedLocator, ClippedLocator)):
        result = composition_view(locator.base)
        if result is None:
            return None
        root, indices, scale = result
        if isinstance(locator, SubsetLocator):
            selected = np.flatnonzero(locator.parent_to_child >= 0)
            selected = selected[np.argsort(locator.parent_to_child[selected])]
            indices = indices[selected]
        elif isinstance(locator, ClippedLocator):
            indices = indices[locator.mapping.parent_bin_indices]
        else:
            linear = np.linalg.inv(locator.inverse)[:-1, :-1]
            scale *= float(np.exp(np.linalg.slogdet(linear)[1] / len(linear)))
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
