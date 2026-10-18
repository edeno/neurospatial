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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np
import shapely
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._publication import _Derivation
from neurospatial.geometry import (
    Arena,
    Binning,
    KnownFaces,
    NativeFaces,
    SpatialBins,
    apply_walls,
    get_centers,
    materialize_cells,
    prepare_bins,
    transform_binning,
)
from neurospatial.layout import _built_layout_data
from neurospatial.layout.base import LayoutEngine
from neurospatial.layout.engines.masked_grid import MaskedGridLayout
from neurospatial.layout.engines.regular_grid import regular_grid_from_cells
from neurospatial.layout.helpers.utils import check_grid_size_safety

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.geometry import CartesianSystem


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


def _grid_centers(edges: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
    """Centers of every cell of a grid, shape ``(n_cells, n_dims)``, C order."""
    axes = np.meshgrid(*(get_centers(edge) for edge in edges), indexing="ij")
    return np.column_stack([axis.ravel() for axis in axes])


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


def _rebin(
    binning: Binning,
    arena: Arena,
    connectivity: nx.Graph,
    layout_parameters: Mapping[str, Any],
    *,
    factor: int | tuple[int, ...] | None,
    bin_size: float | tuple[float, ...] | None,
) -> tuple[_Derivation, RebinCheck]:
    """Build the new bins for ``Environment.rebin`` and the check to run on them.

    Exactly one of ``factor`` (integer coarsening of a rectangular grid) and
    ``bin_size`` (a new physical bin size over the retained arena) is given.
    Unsupported bins are refused before any bins are built.

    Returns
    -------
    derivation : _Derivation
        The new bins, to publish as a derivation of the source environment.
    check : RebinCheck
        Which source bins to check for loss once the new bins are published.
    """
    if (factor is None) == (bin_size is None):
        raise ValueError("Pass exactly one of factor or bin_size to rebin().")
    refuse_unsupported_rebin(
        binning, arena, layout_parameters, factor=factor, bin_size=bin_size
    )
    if bin_size is not None:
        return _rebin_resolution(binning, arena, bin_size)
    assert factor is not None
    return _rebin_factor(binning, connectivity, arena, factor)


def refuse_unsupported_rebin(
    binning: Binning,
    arena: Arena,
    layout_parameters: Mapping[str, Any],
    *,
    factor: object,
    bin_size: object,
) -> None:
    """Refuse a rebin these bins cannot take, with a call that works on them.

    Factor rebinning merges whole cells of a rectangular grid; size rebinning
    redraws a grid from its recorded lattice. Bins without either get advice
    for their own kind: the factory that rebuilds them at the new size, or
    ``env.smooth`` for a coarser scale.
    """
    call = "rebin(bin_size=...)" if bin_size is not None else "rebin(factor)"
    needs_lattice = bin_size is not None
    if (binning.grid_spec if needs_lattice else binning.rectilinear) is not None:
        return
    size = _requested_size(layout_parameters, binning, factor, bin_size)
    if not needs_lattice and binning.grid_spec is not None:
        # A clipped grid keeps its lattice but not whole rectangular cells.
        target = f"env.rebin(bin_size={size:g})" if size else "env.rebin(bin_size=...)"
        raise NotImplementedError(
            _format_error(
                "rebin(factor) merges blocks of whole grid cells, and clipping "
                "cut these cells to the arena outline.",
                why="Why: a cut cell cannot be merged with its neighbors into "
                "a larger rectangle.",
                fix=f"give the new cell size instead, which redraws the grid "
                f"and cuts it to the same outline: {target}",
            )
        )
    if needs_lattice and binning.rectilinear is not None:
        raise NotImplementedError(
            _format_error(
                "rebin(bin_size=...) redraws a grid from its origin and cell "
                "size, which these rectangular bins do not record.",
                why="Why: bins built with from_bins without a grid "
                "specification carry their cells but not the lattice they "
                "came from, so a new lattice cannot be aligned to it.",
                fix="merge whole cells with a factor instead: env.rebin(2)",
            )
        )
    rebuild = _rebuild_call(binning, arena, size)
    smooth = (
        f"; or smooth fields at that scale instead: env.smooth(field, "
        f"bandwidth={size:g})"
        if size
        else ""
    )
    raise NotImplementedError(
        _format_error(
            f"{call} works on rectangular grids, and these {binning.kind!r} bins "
            "have no grid edges.",
            why="Why: rebinning merges or redraws the cells of a rectangular "
            "grid. Hexagons, track segments and mesh triangles do not tile "
            "into larger cells of the same shape.",
            fix=f"{rebuild}{smooth}",
        )
    )


# The recorded build option that sets the cell size of each native kind.
_SIZE_OPTION = {
    "Hexagonal": "hexagon_width",
    "Graph": "bin_size",
    "TriangularMesh": "point_spacing",
}


def _requested_size(
    layout_parameters: Mapping[str, Any],
    binning: Binning,
    factor: object,
    bin_size: object,
) -> float | None:
    """The new cell size the call asks for, when it is one known number."""
    if bin_size is not None:
        sizes = np.ravel(np.asarray(bin_size, dtype=float))
        return float(sizes[0]) if sizes.size and np.all(sizes == sizes[0]) else None
    factors = np.ravel(np.asarray(factor, dtype=float))
    current = layout_parameters.get(_SIZE_OPTION.get(str(binning.kind), "bin_size"))
    if (
        isinstance(current, (int, float))
        and factors.size
        and np.all(factors == factors[0])
    ):
        return float(current) * float(factors[0])
    return None


def _rebuild_call(binning: Binning, arena: Arena, size: float | None) -> str:
    """Advice that rebuilds bins of this kind at ``size``, as code when known."""
    new = f"{size:g}" if size else "..."
    linearization = binning.linearization
    if linearization is not None:
        track = arena.as_track()
        width = None if track is None else track.width
        track_width = "" if width is None else f", track_width={width:g}"
        spacing = linearization.edge_spacing
        return (
            "build the track again at the new bin size: "
            "Environment.from_graph(env.binning.linearization.graph, "
            "list(env.binning.linearization.edge_order), "
            f"bin_size={new}, edge_spacing={spacing!r}{track_width})"
        )
    if binning.kind == "Hexagonal":
        return (
            "build the hexagons again from the positions you built them from: "
            f"Environment.from_samples(positions, bin_size={new}, "
            "layout='Hexagonal')"
        )
    if binning.kind == "TriangularMesh" and arena.polygonal_domain() is not None:
        return (
            "build the mesh again at the new spacing: from neurospatial.layout "
            "import create_layout; Environment.from_layout(create_layout("
            "'TriangularMesh', boundary_polygon=env.arena.polygonal_domain(), "
            f"point_spacing={new}))"
        )
    return "build the bins again at the new size with the factory or layout engine that made them"


def _rebin_resolution(
    binning: Binning, arena: Arena, bin_size: float | tuple[float, ...]
) -> tuple[_Derivation, RebinCheck]:
    """Construct new bins over the retained arena, preserving lattice frame."""
    spec = binning.grid_spec
    assert spec is not None  # refuse_unsupported_rebin refuses bins without one
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
    centers = _grid_centers(edges)
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
        assert grid is not None  # refuse_unsupported_rebin refuses bins without one
        raise NotImplementedError(
            _format_error(
                "rebin(factor) merges blocks of equal cells, but this grid's "
                "cells are not all the same size.",
                why="Why: merging unequal cells would give coarse bins of "
                "different sizes, so occupancy and rates per bin would not be "
                "comparable.",
                fix="build the grid again with even cells at the coarser size, "
                "for example with Environment.from_grid_mask and its "
                "grid_edges",
            )
        )

    # Parse factor
    grid_shape = grid.shape
    grid_edges = grid.edges
    n_dims = len(grid_shape)

    requested: tuple[object, ...] = (
        (factor,) * n_dims if np.ndim(factor) == 0 else tuple(cast("Any", factor))
    )

    # Validate factor dimensions
    if len(requested) != n_dims:
        raise ValueError(
            _format_error(
                f"rebin(factor={factor!r}) gives {len(requested)} factors for a "
                f"{n_dims}-D grid.",
                why="Why: a tuple gives one merge factor per grid axis.",
                fix=f"pass one integer for every axis, such as env.rebin(2), or "
                f"{n_dims} integers, such as env.rebin({(2,) * n_dims})",
            )
        )

    # A factor merges blocks of whole cells, so it must be an integer.
    if any(
        isinstance(f, (bool, np.bool_)) or not isinstance(f, (int, np.integer))
        for f in requested
    ):
        try:
            values = [float(cast("Any", f)) for f in requested]
        except (TypeError, ValueError):
            values = []
        widths = [float(edges[1] - edges[0]) for edges in grid_edges]
        sizes = [f"{w * v:.3g}" for w, v in zip(widths, values, strict=False)]
        target = (
            "..."
            if len(sizes) != n_dims
            else sizes[0]
            if len(set(sizes)) == 1
            else f"({', '.join(sizes)})"
        )
        merged = [max(1, round(v)) for v in values] or [2]
        whole = merged[0] if len(set(merged)) == 1 else tuple(merged)
        integral = bool(values) and all(v == round(v) for v in values)
        raise ValueError(
            _format_error(
                f"rebin(factor={factor!r}) takes integer factors, the number of "
                "cells to merge along each axis"
                + (", got a float." if integral else "."),
                why="Why: factor coarsens the grid by merging blocks of whole "
                "cells, so a fractional factor would split cells.",
                fix=f"pass an integer, such as env.rebin({whole}), or give the "
                f"new cell size in physical units, such as "
                f"env.rebin(bin_size={target})",
            )
        )
    factor_tuple = tuple(int(cast("int", f)) for f in requested)
    for i, f in enumerate(factor_tuple):
        if f <= 0 or f > grid_shape[i]:
            raise ValueError(
                _format_error(
                    f"rebin(factor={factor!r}) asks to merge {f} cells along "
                    f"axis {i}, which has {grid_shape[i]}.",
                    why="Why: a factor is the number of cells merged into one, "
                    "so it is at least 1 and at most the cells along that axis.",
                    fix=f"pass a factor from 1 to {grid_shape[i]}, such as "
                    f"env.rebin({min(2, min(grid_shape))})",
                )
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

    # Keep the edges of the truncated grid, then every factor-th of them.
    coarse_edges = tuple(
        edges[: size + 1 : f]
        for edges, size, f in zip(
            grid_edges, truncated_shape, factor_tuple, strict=True
        )
    )
    coarse_shape = tuple(len(edges) - 1 for edges in coarse_edges)

    # Centers come from the coarse edges, not from the original grid's
    # active bins.
    coarse_bin_centers = _grid_centers(coarse_edges)

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
    coarse, _ = _built_layout_data(cast("LayoutEngine", layout))
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
