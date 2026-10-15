"""Re-bin known arenas using owned rectangular-lattice construction data."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial.arena import (
    Arena,
    MaskArena,
    TrackArena,
    _CombinedArena,
    _TransformedArena,
)

if TYPE_CHECKING:
    from neurospatial.environment._protocols import RebinConsumer
    from neurospatial.environment.core import Environment


def _arena_bounds(arena: Arena) -> NDArray[np.float64]:
    """Enclose a known domain in its coordinate frame, without inventing walls."""
    if not arena.domain_known:
        raise NotImplementedError(
            "rebin(bin_size=...) requires a known arena domain; supply a mask or explicit arena."
        )
    if isinstance(arena, _TransformedArena):
        limits = _arena_bounds(arena.base)
        corners = np.array(list(product(*limits.T)))
        mapped = corners @ arena.forward[:-1, :-1].T + arena.forward[:-1, -1]
        return np.array([mapped.min(axis=0), mapped.max(axis=0)])
    polygon = arena.polygonal_domain()
    if polygon is not None:
        return np.array([polygon.bounds[:2], polygon.bounds[2:]])
    if isinstance(arena, MaskArena):
        cells = np.argwhere(arena.active_mask)
        return np.array(
            [
                [
                    edge[int(low)]
                    for edge, low in zip(arena.edges, cells.min(axis=0), strict=True)
                ],
                [
                    edge[int(high) + 1]
                    for edge, high in zip(arena.edges, cells.max(axis=0), strict=True)
                ],
            ]
        )
    if isinstance(arena, TrackArena) and arena.width is not None:
        return np.array(
            [
                arena.segments.min(axis=(0, 1)) - arena.width / 2,
                arena.segments.max(axis=(0, 1)) + arena.width / 2,
            ]
        )
    if isinstance(arena, _CombinedArena):
        children = np.array([_arena_bounds(child) for child in arena.children])
        return np.array([children[:, 0].min(axis=0), children[:, 1].max(axis=0)])
    raise NotImplementedError(
        "rebin(bin_size=...) cannot bound this arena; supply supported Cartesian domain geometry."
    )


def _rebin_resolution(
    env: RebinConsumer, bin_size: float | tuple[float, ...]
) -> Environment:
    """Construct new bins over the retained arena, preserving lattice frame."""
    from neurospatial.environment.core import Environment
    from neurospatial.layout._binning import KnownFaces, binning_from_layout
    from neurospatial.layout._derive import transform_binning
    from neurospatial.layout.base import LayoutEngine
    from neurospatial.layout.engines.masked_grid import MaskedGridLayout
    from neurospatial.layout.helpers.utils import check_grid_size_safety

    spec = env._binning.grid_spec
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
    local_arena = env.arena.transformed(
        np.asarray(np.linalg.inv(spec.forward), dtype=np.float64)
    )
    limits = _arena_bounds(local_arena)
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
    domain = env.arena.polygonal_domain()
    if domain is not None:
        from shapely import contains_xy

        keep = contains_xy(domain, mapped[:, 0], mapped[:, 1])
    else:
        keep = env.arena.contains(mapped)
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
    assert layout.connectivity is not None
    binning = binning_from_layout(cast("LayoutEngine", layout), layout.connectivity)
    binning = replace(
        binning,
        kind="RegularGrid",
        faces=KnownFaces(binning.faces.build()),
        grid_spec=None,
    )
    if not np.array_equal(spec.forward, np.eye(dims + 1)):
        binning = transform_binning(binning, SimpleNamespace(A=spec.forward))
    binning = replace(binning, grid_spec=spec)
    return Environment.from_bins(
        binning,
        name=f"{env.name}_rebinned" if env.name else "",
        units=env.units,
        frame=env.frame,
        regions=env.regions,
        arena=env.arena,
    )
