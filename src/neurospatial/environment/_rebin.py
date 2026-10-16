"""Construct new grids using factor or physical-resolution input policies.

Factor mode retains integer truncation, native lazy faces and legacy construction
metadata. Resolution mode rebuilds over the retained arena in its owned affine
frame and materializes faces before transforming them. These construction paths
remain distinct. Both return a derivation, which Environment.rebin publishes
(validating and owning geometry, applying physical walls and creating a fresh
analysis cache), and a :class:`RebinCheck` that ``warn_rebin_losses`` applies to
the published result to report source bins and barriers the new bins lose.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._publication import _Derivation
from neurospatial.geometry._arena import (
    Arena,
)
from neurospatial.geometry._arena_binning import apply_walls
from neurospatial.geometry._binning import Binning, KnownFaces, SpatialBins
from neurospatial.geometry._derive import transform_binning
from neurospatial.geometry._explicit import prepare_bins
from neurospatial.geometry._native_faces import NativeFaces
from neurospatial.geometry._polygon import materialize_cells
from neurospatial.layout._construction import built_layout_data
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.engines.masked_grid import MaskedGridLayout
from neurospatial.layout.engines.regular_grid import regular_grid_from_cells
from neurospatial.layout.helpers.utils import check_grid_size_safety

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.geometry._coordinates import CartesianSystem


@dataclass(frozen=True)
class RebinCheck:
    """How to check rebinned bins against the source bins after publication.

    Parameters
    ----------
    considered : ndarray of bool, shape (n_source_bins,) or None
        Source bins whose loss is reported by this check; ``None`` checks all.
    lost_remedy : str
        Mode-specific advice for covering source bins without a new bin.
    crossing_remedy : str
        Mode-specific advice for connections that cross an arena barrier.
    """

    considered: NDArray[np.bool_] | None
    lost_remedy: str
    crossing_remedy: str


def warn_rebin_losses(
    binning: Binning,
    arena: Arena,
    rebinned: Environment[CartesianSystem],
    check: RebinCheck,
) -> None:
    """Warn about source bins and arena barriers the published new bins lose.

    Parameters
    ----------
    binning : Binning
        The source bins.
    arena : Arena
        The source arena, kept by the new bins.
    rebinned : Environment
        The published rebinned environment.
    check : RebinCheck
        Which source bins to check and the advice to give.
    """
    lost = rebinned.bin_at(binning.bins.centers) < 0
    if check.considered is not None:
        lost = check.considered & lost
    _warn_unbinned(binning, arena, lost, check.lost_remedy)
    _warn_outline_crossings(arena, rebinned, check.crossing_remedy)


def _warn_unbinned(
    binning: Binning, arena: Arena, lost: NDArray[np.bool_], remedy: str
) -> None:
    """Warn that original bins have no new bin, with their count and share.

    Parameters
    ----------
    binning : Binning
        The original bins.
    arena : Arena
        The original arena, kept by the new bins.
    lost : ndarray of bool, shape (n_bins,)
        Original bins whose centers map to no bin after rebinning.
    remedy : str
        Mode-specific advice for covering the lost bins.
    """
    if not np.any(lost):
        return
    sizes = binning.bins.sizes
    n_dims = binning.bins.n_dims
    measure = {1: "length", 2: "area"}.get(n_dims, "volume")
    if arena.polygonal_domain() is not None:
        remedy += (
            " For exact cells along the arena boundary, build a grid that "
            "covers the arena at the new size and clip it: "
            "Environment.from_bins(grid.clip(polygon).binning, arena=env.arena)."
        )
    warnings.warn(
        f"Rebinning leaves {np.count_nonzero(lost)} of {len(lost)} original bins "
        f"({sizes[lost].sum() / sizes.sum():.1%} of their {measure}) without a "
        "new bin, because new bins are kept only where their centers lie in the "
        "arena. Positions in those bins map to -1, so their time and spikes drop "
        f"out of totals on the new bins. {remedy}",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _warn_outline_crossings(
    arena: Arena, rebinned: Environment[CartesianSystem], remedy: str
) -> None:
    """Warn when new connections join cells that touch only outside the arena.

    Grid connections join neighboring cells whose centers lie in the arena,
    without checking where the cells meet. Bins wider than a gap in the
    outline, such as the barrier between two maze arms, are joined across it,
    so path lengths, distances and smoothing pass through the barrier. Cells
    whose shared face (or corner, for diagonal neighbors) lies partly in the
    arena remain connected through it.

    Parameters
    ----------
    arena : Arena
        The original arena, kept by the new bins.
    rebinned : Environment
        The new environment, published with the original arena.
    remedy : str
        Mode-specific advice for resolving the barrier.
    """
    domain = arena.polygonal_domain()
    pairs = rebinned._binning.bins.pairs
    if domain is None or len(pairs) == 0:
        return
    centers = rebinned.bin_centers
    shapely.prepare(domain)
    # A center link inside the arena already meets the shared face there.
    candidates = np.flatnonzero(
        ~shapely.covers(domain, shapely.linestrings(centers[pairs]))
    )
    if len(candidates) == 0:
        return
    cells = np.asarray(materialize_cells(rebinned._binning), dtype=object)
    contacts = shapely.intersection(
        shapely.boundary(cells[pairs[candidates, 0]]),
        shapely.boundary(cells[pairs[candidates, 1]]),
    )
    face = shapely.length(contacts)
    # Tolerances ignore coordinate rounding along the outline.
    scale = float(np.sqrt(np.min(rebinned._binning.bins.sizes)))
    separated = np.where(
        face > 0,
        shapely.length(shapely.intersection(contacts, domain)) <= 1e-9 * face,
        shapely.distance(domain, contacts) > 1e-9 * scale,
    )
    crossing = candidates[separated & ~shapely.is_empty(contacts)]
    if len(crossing) == 0:
        return
    u, v = pairs[crossing[0]]
    count = (
        "1 connection joins"
        if len(crossing) == 1
        else f"{len(crossing)} connections join"
    )
    warnings.warn(
        f"{count} new bins that meet only outside the arena, for example "
        f"bins {u} at {centers[u].tolist()} and {v} at "
        f"{centers[v].tolist()}. The bins are wider than a gap in the arena "
        "outline, such as a barrier between two arms, so path lengths, "
        f"distances and smoothing pass through it. {remedy} Or declare the "
        "barrier's edges as internal walls, OpenArena(polygon, walls=...), so "
        "rebinning removes connections across them or refuses cells they cut.",
        UserWarning,
        stacklevel=_external_stacklevel(),
    )


def _rebin_resolution(
    binning: Binning, arena: Arena, bin_size: float | tuple[float, ...]
) -> tuple[_Derivation, RebinCheck]:
    """Construct new bins over the retained arena, preserving lattice frame."""
    spec = binning.grid_spec
    if spec is None:
        raise NotImplementedError(
            "rebin(bin_size=...) needs a supported stored binning specification. Recreate a rectangular grid with a factory or supply GridBinningSpec in from_bins; a layout name or provenance is insufficient."
        )
    dims = len(spec.origin)
    raw = np.asarray(bin_size)
    if raw.dtype.kind not in "iuf" or raw.ndim > 1:
        raise ValueError(
            "bin_size must be a finite positive scalar or one value per grid axis."
        )
    sizes = np.full(dims, float(raw)) if raw.ndim == 0 else np.asarray(raw, dtype=float)
    if sizes.shape != (dims,) or np.any(~np.isfinite(sizes) | (sizes <= 0)):
        raise ValueError(
            f"bin_size must give {dims} finite positive physical axis lengths."
        )
    lengths = np.hypot.reduce(spec.forward[:-1, :-1], axis=0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        spacing = sizes / lengths
    if np.any(~np.isfinite(spacing) | (spacing <= 0)):
        raise ValueError(
            "bin_size is not representable in the stored grid frame; rescale coordinates."
        )
    # Bound the domain in lattice coordinates. A conservative box is sufficient:
    # every candidate center is subsequently checked against the actual arena.
    local_arena = arena.transformed(
        np.asarray(np.linalg.inv(spec.forward), dtype=np.float64)
    )
    if not local_arena.domain_known:
        raise NotImplementedError(
            "rebin(bin_size=...) requires a known arena domain; supply a mask or explicit arena."
        )
    limits = local_arena.domain_bounds()
    if limits is None:
        raise NotImplementedError(
            "rebin(bin_size=...) cannot bound this arena; supply supported Cartesian domain geometry."
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        low = np.floor((limits[0] - spec.origin) / spacing)
        high = np.ceil((limits[1] - spec.origin) / spacing)
        counts = high - low
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts < 1)
        or np.any(counts >= np.iinfo(np.intp).max)
    ):
        raise ValueError(
            "bin_size produces unrepresentable grid dimensions; use larger bins or recenter coordinates."
        )
    shape = tuple(int(count) for count in counts)
    check_grid_size_safety(shape, dims)
    edges = tuple(
        origin + (start + np.arange(count + 1)) * step
        for origin, start, count, step in zip(
            spec.origin, low, shape, spacing, strict=True
        )
    )
    if any(
        np.any(np.diff(edge) <= 0) or not np.all(np.isfinite(edge)) for edge in edges
    ):
        raise ValueError(
            "bin_size is too small at this coordinate precision; recenter coordinates or use larger bins."
        )
    axes = [(edge[:-1] + edge[1:]) / 2 for edge in edges]
    centers = np.column_stack(
        [axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")]
    )
    mapped = centers @ spec.forward[:-1, :-1].T + spec.forward[:-1, -1]
    domain = arena.polygonal_domain()
    if domain is not None:
        keep = shapely.contains_xy(domain, mapped[:, 0], mapped[:, 1])
    else:
        keep = arena.contains(mapped)
    if not np.any(keep):
        raise ValueError(
            "Rebinning retains no bin center inside the original arena; use a smaller bin_size."
        )
    layout = MaskedGridLayout()
    layout.build(
        active_mask=keep.reshape(shape),
        grid_edges=edges,
        connect_diagonal_neighbors=spec.connect_diagonal,
    )
    grid = layout.binning()
    # The new bins keep the source's layout label.
    grid = replace(
        grid,
        kind=binning.kind,
        faces=KnownFaces(grid.faces.build()),
        grid_spec=None,
    )
    if not np.array_equal(spec.forward, np.eye(dims + 1)):
        grid = transform_binning(grid, SimpleNamespace(A=spec.forward))
    grid = replace(grid, grid_spec=spec)
    # This native construction deliberately lets the arena remove new wall-
    # crossing connections. Public from_bins must preserve supplied topology.
    choices = {
        "bin_size": tuple(float(size) for size in sizes),
        "connect_diagonal_neighbors": spec.connect_diagonal,
    }
    derivation = _Derivation(prepare_bins(grid), "rebin", build_params=choices)
    return derivation, RebinCheck(
        None,
        "Use a smaller bin_size, or one that puts cell edges on the arena "
        "boundary, such as a divisor of a rectangular arena's side lengths.",
        "Use a bin_size smaller than the barrier's width.",
    )


def _require_native_grid_navigation(
    binning: Binning, arena: Arena, *, connect_diagonal: bool
) -> None:
    """Require the native grid paths that factor rebinning knows how to rebuild."""
    bins = binning.bins
    direct_distances = np.hypot.reduce(
        bins.centers[bins.pairs[:, 1]] - bins.centers[bins.pairs[:, 0]], axis=1
    )
    message = (
        "rebin() cannot reconstruct custom navigation paths or connections. "
        "Use subset() to retain them, or explicitly construct the target bins "
        "and their navigation geometry."
    )
    if not np.allclose(
        bins.pair_distances,
        direct_distances,
        rtol=64 * np.finfo(np.float64).eps,
        atol=0.0,
    ):
        raise NotImplementedError(message)

    grid = binning.rectilinear
    assert grid is not None
    # Only adjacency is needed from this grid. Unit cells in index coordinates
    # avoid computing unused physical edge lengths at extreme coordinate scales.
    index_centers = (
        np.column_stack(
            np.unravel_index(np.arange(grid.active_mask.size), grid.shape)
        ).astype(np.float64)
        + 0.5
    )
    graph = regular_grid_from_cells(
        tuple(np.arange(n + 1, dtype=np.float64) for n in grid.shape),
        grid.active_mask,
        index_centers,
        connect_diagonal_neighbors=connect_diagonal,
    ).connectivity
    assert graph is not None
    pairs = np.asarray(list(graph.edges()), dtype=np.intp).reshape(-1, 2)
    candidate_bins = SpatialBins(
        bins.centers,
        bins.sizes,
        bins.measure_dim,
        pairs,
        np.hypot.reduce(bins.centers[pairs[:, 1]] - bins.centers[pairs[:, 0]], axis=1),
    )
    # Apply the same physical wall rule as native construction, with fresh faces
    # aligned to the candidate graph rather than the source's selected pairs.
    candidate = replace(
        binning,
        bins=candidate_bins,
        faces=NativeFaces(
            candidate_bins,
            "grid",
            (np.array([edge[1] - edge[0] for edge in grid.edges]),),
        ),
        face_normals=None,
    )
    candidate = apply_walls(candidate, arena)
    expected_pairs = {frozenset(pair) for pair in candidate.bins.pairs}
    if expected_pairs != {frozenset(pair) for pair in bins.pairs}:
        raise NotImplementedError(message)


def _rebin_factor(
    binning: Binning,
    connectivity: nx.Graph,
    arena: Arena,
    factor: int | tuple[int, ...],
) -> tuple[_Derivation, RebinCheck]:
    """Coarsen a regular grid by an integer factor."""
    # --- Input validation ---

    grid = binning.rectilinear
    if (
        grid is None
        or any(
            np.max(np.abs(np.diff(edge) - np.diff(edge)[0]))
            > 1e-7 * np.diff(edge)[0] + 4 * np.spacing(np.max(np.abs(edge)))
            for edge in grid.edges
        )
        or not np.allclose(
            binning.bins.sizes,
            np.prod([edge[1] - edge[0] for edge in grid.edges]),
            rtol=64 * np.finfo(np.float64).eps,
            atol=0.0,
        )
    ):
        raise NotImplementedError(
            "rebin() requires a uniform rectilinear grid; descriptive layout "
            "names cannot establish cell geometry. Supply uniform grid data "
            "or use subset() for spatial cropping."
        )

    # Parse factor
    grid_shape = grid.shape
    grid_edges = grid.edges
    n_dims = len(grid_shape)

    factor_tuple = (factor,) * n_dims if isinstance(factor, int) else tuple(factor)

    # Validate factor dimensions
    if len(factor_tuple) != n_dims:
        raise ValueError(
            f"factor has {len(factor_tuple)} elements but environment has "
            f"{n_dims} dimensions. factor must be int or tuple matching "
            "environment dimensionality."
        )

    # Validate factor values
    for i, f in enumerate(factor_tuple):
        if not isinstance(f, (int, np.integer)):
            raise ValueError(
                f"factor[{i}] = {f} must be an integer, got {type(f).__name__}"
            )
        if f <= 0:
            raise ValueError(
                f"factor[{i}] = {f} must be positive. "
                "Coarsening factor must be at least 1."
            )
        if f > grid_shape[i]:
            raise ValueError(
                f"factor[{i}] = {f} is too large for grid shape {grid_shape}. "
                f"Dimension {i} has only {grid_shape[i]} bins."
            )

    # The stored choice governs future bins even if a sparse source currently
    # contains no diagonal edges. Legacy grids infer it from their actual edges.
    source_centers = binning.bins.centers
    connect_diagonal = (
        binning.grid_spec.connect_diagonal
        if binning.grid_spec is not None
        else any(
            np.count_nonzero(source_centers[u] != source_centers[v]) > 1
            for u, v in connectivity.edges()
        )
    )
    _require_native_grid_navigation(binning, arena, connect_diagonal=connect_diagonal)

    # Check for non-divisible dimensions
    truncated_shape = tuple(
        s // f * f for s, f in zip(grid_shape, factor_tuple, strict=True)
    )
    if truncated_shape != grid_shape:
        warnings.warn(
            f"Grid shape {grid_shape} is not evenly divisible by factor "
            f"{factor_tuple}. Grid will be truncated to {truncated_shape} "
            f"before coarsening.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    # --- Compute new grid parameters ---

    # Truncate grid edges if needed
    truncated_edges = []
    for edges, trunc_size in zip(grid_edges, truncated_shape, strict=True):
        # Keep edges up to truncated_size + 1 (edges define bins)
        truncated_edges.append(edges[: trunc_size + 1])

    # Compute new coarsened edges
    coarse_edges = tuple(
        edges[::f] for edges, f in zip(truncated_edges, factor_tuple, strict=True)
    )

    # New grid shape
    coarse_shape = tuple(len(edges) - 1 for edges in coarse_edges)

    # --- Compute new bin centers from coarsened edges ---

    # For each coarse bin, compute center from the coarse grid edges
    # This avoids issues with active/inactive bins from the original grid

    # Create bin centers from coarse edges using meshgrid
    coarse_grid_centers = []
    for edges in coarse_edges:
        # Bin centers are midpoints between edges
        centers = (edges[:-1] + edges[1:]) / 2
        coarse_grid_centers.append(centers)

    # Create meshgrid of bin centers
    if n_dims == 1:
        center_coords = [coarse_grid_centers[0]]
    else:
        center_grids = np.meshgrid(*coarse_grid_centers, indexing="ij")
        center_coords = [grid.ravel() for grid in center_grids]

    # Stack into (n_bins, n_dims)
    coarse_bin_centers = np.column_stack(center_coords)

    # Finalize active cells before constructing the graph or publishing an
    # environment. The original arena remains the continuous authority. Its
    # closed containment keeps a coarse cell centered on the domain boundary,
    # such as the corner beside a missing fine cell, in every dimension.
    keep = np.ones(len(coarse_bin_centers), dtype=bool)
    if arena.domain_known:
        keep = arena.contains(coarse_bin_centers)
        if not np.any(keep):
            raise ValueError(
                "Rebinning retains no coarse bin center inside the original arena. "
                "Use a smaller factor or rebuild a grid with centers in the domain."
            )

    # --- Build the coarse grid and publish it ---

    # Connect only eligible coarse cells. The coarse grid keeps the source's
    # binning specification, when it has one, for later re-binning.
    layout = regular_grid_from_cells(
        coarse_edges,
        keep.reshape(coarse_shape),
        coarse_bin_centers,
        connect_diagonal_neighbors=connect_diagonal,
        grid_spec=binning.grid_spec,
    )
    coarse, _ = built_layout_data(cast("LayoutEngine", layout))
    derivation = _Derivation(
        # The coarse bins keep the source's layout label.
        replace(coarse, kind=binning.kind),
        "rebin",
        build_params={
            "bin_size": tuple(
                (edges[-1] - edges[0]) / (len(edges) - 1) for edges in coarse_edges
            ),
            "connect_diagonal_neighbors": connect_diagonal,
        },
    )

    # Bins beyond the truncated extent are already reported above.
    within = np.all(
        source_centers < np.array([edges[-1] for edges in coarse_edges]), axis=1
    )
    return derivation, RebinCheck(
        within,
        "Use a smaller factor.",
        "Use a factor whose bins are narrower than the barrier.",
    )
