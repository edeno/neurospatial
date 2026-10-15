"""Owned component snapshots and lookup for combined environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from neurospatial.arena import Arena
from neurospatial.layout._binning import Binning, SpatialBins
from neurospatial.layout._locators import _LocatorState

if TYPE_CHECKING:
    from neurospatial.layout._binning import Locator
    from neurospatial.layout._diffusion import DiffusionGeometry
    from neurospatial.layout._face_normals import FaceNormals


@dataclass(frozen=True, eq=False)
class CombinedLocator(_LocatorState):
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
        dims, measure = children[0].bins.n_dims, children[0].bins.measure_dim
        if any(
            child.bins.n_dims != dims
            or child.bins.measure_dim != measure
            or arena.n_dims != dims
            or arena.coordinate_kind != "cartesian"
            for child, arena in zip(children, arenas, strict=True)
        ):
            raise ValueError(
                "Components need matching Cartesian dimensions and measures."
            )
        centers = np.concatenate([child.bins.centers for child in children])
        sizes = np.concatenate([child.bins.sizes for child in children])
        bridge_bins = SpatialBins.from_arrays(
            centers, sizes, measure, self.bridge_pairs, self.bridge_distances
        )
        geometric = np.linalg.norm(
            centers[bridge_bins.pairs[:, 1]] - centers[bridge_bins.pairs[:, 0]], axis=1
        )
        if not np.allclose(
            bridge_bins.pair_distances, geometric, rtol=32 * np.finfo(float).eps, atol=0
        ):
            raise ValueError(
                "Inferred bridge distances must match endpoint separation."
            )
        for child, arena in zip(children, arenas, strict=True):
            if arena.domain_known and not np.all(arena.contains(child.bins.centers)):
                raise ValueError("A component arena excludes its own bin centers.")
        conductances = np.array(self.bridge_conductances, dtype=np.float64, copy=True)
        if conductances.shape != (len(bridge_bins.pairs),) or np.any(
            ~np.isfinite(conductances) | (conductances < 0)
        ):
            raise ValueError(
                "Bridge conductances must be finite nonnegative edge values."
            )
        offsets = np.cumsum([0, *[child.bins.n_bins for child in children]])
        owners = np.repeat(np.arange(len(children)), np.diff(offsets))
        if len(bridge_bins.pairs) and np.any(
            owners[bridge_bins.pairs[:, 0]] == owners[bridge_bins.pairs[:, 1]]
        ):
            raise ValueError("Bridges must join bins in different components.")
        conductances.setflags(write=False)
        for name, value in (
            ("keys", keys),
            ("children", children),
            ("arenas", arenas),
            ("bridge_pairs", bridge_bins.pairs),
            ("bridge_distances", bridge_bins.pair_distances),
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
        """Assemble the authoritative concatenated bins and navigation edges."""
        offsets = self.offsets
        return SpatialBins.from_arrays(
            np.concatenate([child.bins.centers for child in self.children]),
            np.concatenate([child.bins.sizes for child in self.children]),
            self.children[0].bins.measure_dim,
            np.concatenate(
                [
                    *[
                        child.bins.pairs + offset
                        for child, offset in zip(
                            self.children, offsets[:-1], strict=True
                        )
                    ],
                    self.bridge_pairs,
                ]
            ),
            np.concatenate(
                [
                    *[child.bins.pair_distances for child in self.children],
                    self.bridge_distances,
                ]
            ),
        )

    def locate(self, points: NDArray[np.float64]) -> NDArray[np.intp]:
        """Locate native points with each child's original arena and lookup."""
        query = np.asarray(points, dtype=np.float64)
        dims = self.children[0].bins.n_dims
        if query.ndim != 2 or query.shape[1] != dims:
            raise ValueError(f"Expected positions of shape (n_points, {dims}).")
        result = np.full(len(query), -1, dtype=np.intp)
        finite = np.all(np.isfinite(query), axis=1)
        for child, arena, offset in zip(
            self.children, self.arenas, self.offsets[:-1], strict=True
        ):
            rows = np.flatnonzero(finite & (result < 0))
            if arena.domain_known:
                rows = rows[arena.contains(query[rows])]
            local = child.locator.locate(query[rows])
            valid = local >= 0
            result[rows[valid]] = offset + local[valid]
        return result


def combined_normals(locator: CombinedLocator) -> FaceNormals | None:
    """Combine authoritative child normals; bridges have no physical normal."""
    from neurospatial.layout._face_normals import FaceNormals

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
    """
    from scipy.sparse.csgraph import connected_components

    from neurospatial.layout._diffusion import DiffusionGeometry
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator

    def weights(
        locator: Locator, active: NDArray[np.bool_] | None = None
    ) -> tuple[sparse.csr_matrix, int] | None:
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
            return result, locator.children[0].bins.measure_dim
        if isinstance(locator, SubsetLocator):
            selected = np.flatnonzero(locator.parent_to_child >= 0)
            selected = selected[np.argsort(locator.parent_to_child[selected])]
            retained = np.zeros(len(locator.parent_to_child), dtype=np.bool_)
            retained[selected if active is None else selected[active]] = True
            result = weights(locator.base, retained)
            if result is None:
                return None
            matrix, dim = result
            return matrix[selected][:, selected].tocsr(), dim
        if isinstance(locator, TransformedLocator):
            result = weights(locator.base, active)
            if result is None:
                return None
            matrix, dim = result
            linear = np.linalg.inv(locator.inverse)[:-1, :-1]
            scale = float(np.exp(np.linalg.slogdet(linear)[1] / len(linear)))
            normalized = linear / scale
            if not np.allclose(
                normalized @ normalized.T, np.eye(len(linear)), rtol=0, atol=1e-12
            ):
                raise NotImplementedError(
                    "Combined scalar diffusion requires a similarity transform; use the original coordinates."
                )
            factor = scale ** (dim - 2)
            old = matrix.data.copy()
            matrix = matrix * factor
            if not np.all(np.isfinite(matrix.data)) or np.any(
                (old > 0) & (matrix.data == 0)
            ):
                raise ValueError(
                    "Transformed conductances exceed floating-point range; rescale the coordinates."
                )
            return matrix, dim
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


def expected_composition_binning(locator: Locator) -> tuple[Binning, bool] | None:
    """Recover physical face authority through subsets and affine wrappers.

    This is validation of stored duplicates, not a layout rebuild or bridge
    inference. The flag distinguishes affine arithmetic from exact selection.
    """
    from types import SimpleNamespace

    from neurospatial.layout._binning import KnownFaces, MissingFaces
    from neurospatial.layout._derive import transform_binning
    from neurospatial.layout._diffusion import FaceGeometry
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator

    if isinstance(locator, CombinedLocator):
        bins = locator.spatial_bins()
        faces: KnownFaces | MissingFaces = MissingFaces(
            "Navigation bridges have no physical faces."
        )
        if not len(locator.bridge_pairs) and all(
            child.faces.available for child in locator.children
        ):
            parts = [child.faces.build() for child in locator.children]
            faces = KnownFaces(
                FaceGeometry(
                    bins.pairs,
                    np.concatenate([part.A for part in parts]),
                    np.concatenate([part.distance for part in parts]),
                    bins.sizes,
                )
            )
        normals = combined_normals(locator)
        return Binning("combined", bins, locator, faces, face_normals=normals), False
    if isinstance(locator, SubsetLocator):
        result = expected_composition_binning(locator.base)
        if result is None:
            return None
        parent, computed = result
        mapping = locator.parent_to_child
        selected = np.flatnonzero(mapping >= 0)
        selected = selected[np.argsort(mapping[selected])]
        mapped = mapping[parent.bins.pairs]
        kept = np.all(mapped >= 0, axis=1)
        bins = SpatialBins.from_arrays(
            parent.bins.centers[selected],
            parent.bins.sizes[selected],
            parent.bins.measure_dim,
            mapped[kept],
            parent.bins.pair_distances[kept],
        )
        subset_faces = parent.faces
        if subset_faces.available:
            old = subset_faces.build()
            subset_faces = KnownFaces(
                FaceGeometry(bins.pairs, old.A[kept], old.distance[kept], bins.sizes)
            )
        from neurospatial.layout._face_normals import FaceNormals

        normals = (
            FaceNormals(parent.face_normals.values[kept])
            if parent.face_normals is not None
            else None
        )
        return Binning(
            "combined_subset", bins, locator, subset_faces, face_normals=normals
        ), computed
    if isinstance(locator, TransformedLocator):
        result = expected_composition_binning(locator.base)
        if result is None:
            return None
        parent, _computed = result
        return transform_binning(
            parent, SimpleNamespace(A=np.linalg.inv(locator.inverse))
        ), True
    return None


def composition_view(
    locator: object,
) -> tuple[CombinedLocator, NDArray[np.intp], float] | None:
    """Resolve current bins to root component IDs and a similarity scale."""
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator
    from neurospatial.layout._polygon import ClippedLocator

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
    for u, v in binning.bins.pairs:
        pair = frozenset((int(indices[u]), int(indices[v])))
        if pair in original:
            result[frozenset((int(u), int(v)))] = original[pair] * scale ** (
                binning.bins.measure_dim - 2
            )
    return result


def _component_keys(binning: Binning) -> tuple[str, ...]:
    """Return stable component keys from retained composition data."""

    result = composition_view(binning.locator)
    return () if result is None else result[0].keys


def _bin_sources(binning: Binning) -> tuple[tuple[str, int], ...] | None:
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
