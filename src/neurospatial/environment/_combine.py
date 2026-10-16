"""Construct ordinary environments from owned component snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from neurospatial._exceptions import IncompatibleEnvironmentError
from neurospatial._logging import log_environment_combine
from neurospatial.environment._publication import (
    _Derivation,
    _Provenance,
)
from neurospatial.geometry._arena import _CombinedArena
from neurospatial.geometry._binning import Binning, KnownFaces, MissingFaces
from neurospatial.geometry._codecs import decode, encode
from neurospatial.geometry._combine_geometry import (
    certify_components,
    segment_enters_component,
)
from neurospatial.geometry._composition import CombinedLocator, combined_normals
from neurospatial.geometry._coordinates import require_cartesian
from neurospatial.geometry._diffusion import FaceGeometry
from neurospatial.geometry._explicit import prepare_bins
from neurospatial.geometry._polygon import (
    PolygonCells,
    cells_from_locator,
)
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

BridgeEndpoint = tuple[str, int]
BridgeKey = tuple[BridgeEndpoint, BridgeEndpoint]


def _nearest_indices(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Query nearest centers, resolving exact ties by the lowest local bin ID."""
    tree = cKDTree(target)
    count = min(2, len(target))
    distances, indices = tree.query(source, k=count)
    if count == 1:
        return np.asarray(indices, dtype=np.intp), np.asarray(
            distances, dtype=np.float64
        )
    nearest = np.asarray(indices[:, 0], dtype=np.intp)
    lengths = np.asarray(distances[:, 0], dtype=np.float64)
    for row in np.flatnonzero(distances[:, 0] == distances[:, 1]):
        candidates = np.asarray(
            tree.query_ball_point(source[row], np.nextafter(lengths[row], np.inf)),
            dtype=np.intp,
        )
        actual = np.linalg.norm(target[candidates] - source[row], axis=1)
        nearest[row] = np.min(candidates[actual == actual.min()])
        lengths[row] = actual.min()
    return nearest, lengths


def _combine(
    components: Mapping[str, Environment],
    *,
    auto_bridge: bool,
    max_mnn_distance: float | None,
    bridge_conductances: Mapping[BridgeKey, float] | None,
) -> _Derivation:
    """Validate components and infer path connections between them.

    The result is published as a derivation of the first component, whose
    units and frame every component shares.
    """
    from neurospatial.environment.core import Environment  # core imports this module

    if not isinstance(components, Mapping) or not components:
        raise ValueError(
            "components must be a nonempty ordered mapping of keys to environments."
        )
    keys, values = tuple(components), tuple(components.values())
    if any(not isinstance(key, str) or not key or "/" in key for key in keys):
        raise ValueError("Component keys must be nonempty strings without '/'.")
    if any(not isinstance(env, Environment) for env in values):
        raise TypeError("Environment.combine() takes Environment values.")
    for key, env in zip(keys, values, strict=True):
        require_cartesian(
            env._coordinate_system,
            f"Environment.combine(components[{key!r}])",
            why="It joins the cells of each component, and the bridges between "
            "them, in one shared Cartesian frame.",
            fix="combine allocentric Cartesian environments, and analyze each "
            "polar environment on its own",
        )
    first = values[0]
    for env in values[1:]:
        if (env.n_dims, env._binning.bins.measure_dim) != (
            first.n_dims,
            first._binning.bins.measure_dim,
        ):
            raise IncompatibleEnvironmentError(
                f"[E1003] Components must share coordinate and measure dimensions: "
                f"first has n_dims={first.n_dims}, measure_dim={first._binning.bins.measure_dim}; "
                f"another has n_dims={env.n_dims}, measure_dim={env._binning.bins.measure_dim}.",
                fix="build components with matching coordinate columns and physical measure dimensions",
            )
        if (env.units, env.frame) != (first.units, first.frame):
            raise ValueError(
                "Component units and frame must match; calibrate and transform coordinates before combining."
            )
    if type(auto_bridge) is not bool:
        raise TypeError("auto_bridge must be boolean.")
    if max_mnn_distance is not None and (
        not isinstance(max_mnn_distance, (int, float, np.integer, np.floating))
        or isinstance(max_mnn_distance, (bool, np.bool_))
        or not np.isfinite(max_mnn_distance)
        or max_mnn_distance < 0
    ):
        raise ValueError("max_mnn_distance must be finite and nonnegative, or None.")
    children = []
    for env in values:
        source = env._binning
        meta, arrays = encode(source, "child")
        child = decode(meta, arrays, "child")
        assert isinstance(child, Binning)
        children.append(child)
    arenas = tuple(env.arena for env in values)
    children_tuple = tuple(children)
    certify_components(children_tuple, arenas, keys)
    offsets = np.cumsum([0, *[child.bins.n_bins for child in children]])
    pairs, distances = [], []
    if auto_bridge:
        for i, child in enumerate(children):
            for j in range(i + 1, len(children)):
                other = children[j]
                if not child.bins.n_bins or not other.bins.n_bins:
                    continue
                right, lengths = _nearest_indices(
                    child.bins.centers, other.bins.centers
                )
                left, _reverse_lengths = _nearest_indices(
                    other.bins.centers, child.bins.centers
                )
                if not np.all(np.isfinite(lengths)):
                    raise ValueError(
                        "Bridge distances exceed coordinate precision; rescale the inputs."
                    )
                for u, v in enumerate(right):
                    if left[v] != u:
                        continue
                    distance = float(lengths[u])
                    if max_mnn_distance is not None and distance > max_mnn_distance:
                        continue
                    if first.n_dims == 2:
                        link = LineString(
                            [child.bins.centers[u], other.bins.centers[v]]
                        )
                        if any(
                            link.intersects(wall)
                            for arena in arenas
                            for wall in arena.walls
                        ):
                            continue
                    elif any(arena.walls for arena in arenas):
                        raise NotImplementedError(
                            "Automatic bridges cannot certify wall crossings outside 2D; disable auto_bridge."
                        )
                    segment = np.array([child.bins.centers[u], other.bins.centers[v]])
                    if any(
                        segment_enters_component(segment, third, arenas[k])
                        for k, third in enumerate(children)
                        if k not in (i, j)
                    ):
                        continue
                    pairs.append((int(offsets[i] + u), int(offsets[j] + v)))
                    distances.append(distance)
    bridge_pairs = np.array(pairs, dtype=np.intp).reshape(-1, 2)
    conductances = np.zeros(len(pairs), dtype=np.float64)
    by_pair = {frozenset(pair): index for index, pair in enumerate(pairs)}
    assigned: set[frozenset[int]] = set()
    for endpoints, value in (bridge_conductances or {}).items():
        try:
            (source_key, source_bin), (target_key, target_bin) = endpoints
            i, j = keys.index(source_key), keys.index(target_key)
        except (ValueError, TypeError) as error:
            raise ValueError(
                "Conductance keys must be pairs of (component_key, local_bin_id) endpoints."
            ) from error
        if any(
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or not 0 <= index < children[owner].bins.n_bins
            for index, owner in ((source_bin, i), (target_bin, j))
        ):
            raise ValueError(
                "Conductance endpoint bin IDs must be valid local integers."
            )
        edge = frozenset((int(offsets[i] + source_bin), int(offsets[j] + target_bin)))
        if edge not in by_pair:
            raise ValueError(
                "Conductance endpoints must identify an inferred bridge; inspect get_bridge_attributes()."
            )
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("Bridge conductance must be finite and nonnegative.")
        if edge in assigned:
            raise ValueError(
                "Supply one conductance per unordered bridge; reversed endpoints duplicate the same connection."
            )
        assigned.add(edge)
        conductances[by_pair[edge]] = value
    locator = CombinedLocator(
        keys, children_tuple, arenas, bridge_pairs, np.array(distances), conductances
    )
    bins = locator.spatial_bins()
    faces: KnownFaces | MissingFaces
    if not len(pairs) and all(child.faces.available for child in children):
        child_faces = [child.faces.build() for child in children]
        faces = KnownFaces(
            FaceGeometry(
                bins.pairs,
                np.concatenate([part.A for part in child_faces]),
                np.concatenate([part.distance for part in child_faces]),
                bins.sizes,
            )
        )
    else:
        faces = MissingFaces(
            "Combined navigation bridges have no physical face geometry; use scalar diffusion when child geometry is certified."
        )
    shapes = None
    if first.n_dims == 2 and bins.measure_dim == 2:
        with suppress(ValueError):
            shapes = PolygonCells(cells_from_locator(locator))
    regions = Regions(
        [
            replace(region, name=f"{key}/{region.name}")
            for key, env in components.items()
            for region in env.regions.values()
        ]
    )

    log_environment_combine(len(children), bins.n_bins, len(pairs))

    normals = combined_normals(locator) if not len(pairs) else None

    snapshot, category = prepare_bins(
        Binning("combined", bins, locator, faces, shapes=shapes, face_normals=normals)
    )
    return _Derivation(
        snapshot,
        _Provenance(snapshot.kind, category, {}),
        regions=regions,
        arena=_CombinedArena(arenas),
    )
