"""Binning layer for view encoding (spatial view cells).

This module converts spike trains and gaze data into discrete spike counts
and view occupancy arrays that can be processed by the smoothing layer.

The key difference from spatial binning (``_spatial_counts.py``) is that:
- Position binning: bins spikes by *where the animal was*
- View binning: bins spikes by *where the animal was looking*

``prepare_view_binning`` handles:
1. Computation of viewed locations from positions, headings, and gaze model
2. The viewed bin of each sample and the interval support shared by view
   occupancy (time spent *viewing* each bin) and spike counts

Its ``FrameBinning.count`` then counts each neuron's spikes by the location
viewed at the most recent frame and accumulates view occupancy from the same
intervals.

Output shapes:
- Spike counts: (n_neurons, n_bins)
- View occupancy: (n_bins,) - always shared across neurons

Gaze models supported:
- "ray_cast": the first wall along the gaze (default)
- "fixed_distance": the point a fixed distance along the gaze, or the first
  wall when one is nearer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import (
    DEFAULT_MAX_GAP,
    hold_last_valid,
    interval_valid_mask,
)
from neurospatial.encoding._spatial_counts import FrameBinning, prepare_frame_binning
from neurospatial.encoding._validation import require_planar, validate_trajectory
from neurospatial.geometry import wall_hit_bins
from neurospatial.ops.visibility import gaze_points

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource

__all__ = [
    "prepare_view_binning",
    "require_allocentric_view",
]


def _view_observation_mask(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    gaze_offsets: NDArray[np.float64] | None,
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
    break_before: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Resolve selected exposure before requiring bins at physical wall hits.

    A sample with a non-finite position, heading or gaze offset is missing
    and is bridged like a deleted row. Returns the interval mask, shape
    ``(n_samples - 1,)``, and the observed samples, shape ``(n_samples,)``.
    """
    observed = np.isfinite(positions).all(axis=1) & np.isfinite(headings)
    if gaze_offsets is not None:
        observed &= np.isfinite(gaze_offsets)
    mask = interval_valid_mask(
        times,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
        sample_valid=observed,
    )
    mask &= np.diff(times) > 0
    return mask, observed


def require_allocentric_view(env: Environment, operation: str) -> None:
    """Refuse a polar environment for rates indexed by the viewed location.

    Parameters
    ----------
    env : Environment
        The environment the viewed locations are binned in.
    operation : str
        The refused call, e.g. ``"compute_view_rate()"``.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar.
    ValueError
        If ``env`` is not two-dimensional.
    """
    require_planar(
        env,
        operation,
        why="It casts gaze from the animal's allocentric positions and headings "
        "and bins the viewed Cartesian locations.",
        fix="compute view rates on the allocentric environment the positions "
        "were recorded in",
        plane_why="gaze is cast in the plane from (x, y) positions and headings",
    )


def _precompute_view_bins(
    env: Environment,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast"] = "ray_cast",
    view_distance: float | None = None,
    gaze_offsets: NDArray[np.float64] | None = None,
    boundary_source: BoundarySource = "physical",
    interval_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.intp]:
    """Precompute view bin indices for all trajectory samples.

    This internal helper computes viewed locations and maps them to bins
    once, so they can be reused for multiple neurons in batch processing.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast"}, default="ray_cast"
        Method for computing viewed location.
    view_distance : float or None, default=None
        Distance for the fixed_distance gaze model (environment units);
        required by it and unused by ray_cast.
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. For ray-based rate maps, a finite
        physical hit without a retained bin raises NotImplementedError; provide
        bins at the wall or explicitly select the analysis approximation.
        Fixed-distance gaze stops at the same walls.

    Returns
    -------
    view_bins : ndarray, shape (n_samples,), dtype=intp
        Bin index for each sample. -1 indicates invalid view (outside env).

    Notes
    -----
    Without ``interval_mask``, this is a geometry-only lookup and unowned hits stay
    -1. Allocation callers provide canonical observation support; only finite
    physical hits at starts of selected positive-duration intervals require
    retained bins. Final samples and excluded intervals supply no exposure.
    """
    n_samples = len(positions)

    # Viewed locations for all timepoints; a gaze never passes through a wall.
    viewed_locations, at_wall = gaze_points(
        positions,
        headings,
        method=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        env=env,
        max_distance=None,
        boundary_source=boundary_source,
    )

    view_bins = np.full(n_samples, -1, dtype=np.intp)
    finite = np.all(np.isfinite(viewed_locations), axis=1)
    # A point in open space lies in a bin or outside the environment (-1).
    open_space = finite & ~at_wall
    if np.any(open_space):
        view_bins[open_space] = env.bin_at(viewed_locations[open_space])
    # A point on a wall belongs to the bin on the observer's side.
    wall = finite & at_wall
    if np.any(wall):
        hits = viewed_locations[wall]
        origins = positions[wall]
        wall_bins = env.bin_at(np.nextafter(hits, origins))
        wall_bins = wall_hit_bins(
            env.binning,
            env.arena,
            hits,
            origins,
            wall_bins,
            boundary_source,
        )
        required = np.zeros(n_samples, dtype=bool)
        if interval_mask is not None:
            required[:-1] = interval_mask
        if boundary_source == "physical" and np.any((wall_bins < 0) & required[wall]):
            raise NotImplementedError(
                "Physical wall hit has no retained bin on the observer's side. "
                "Build bins that reach the measured wall before computing view rates; "
                "adding OpenArena or clipping existing cells cannot add missing bins. "
                "For curved or slanted walls, clip a grid that covers the arena to the "
                "measured polygon and attach its walls: Environment.from_bins("
                "grid.clip(polygon).binning, arena=OpenArena(polygon)). "
                "For rotated rectangular arenas, construct bins in the arena-aligned frame "
                "and transform bins, arena and observations together. Alternatively, explicitly "
                "choose boundary_source='analysis' to encode the retained analysis boundary. "
                "compute_viewed_location(...) remains available for geometry-only wall contacts."
            )
        view_bins[wall] = wall_bins

    return view_bins


def prepare_view_binning(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast"] = "ray_cast",
    view_distance: float | None = None,
    gaze_offsets: NDArray[np.float64] | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    view_bins: NDArray[np.intp] | None = None,
    boundary_source: BoundarySource = "physical",
    break_before: NDArray[np.bool_] | None = None,
) -> FrameBinning:
    """Prepare geometry and exact temporal support before counting spikes.

    ``break_before`` is already validated; the interval ending at each
    declared recording break is dropped from counts and occupancy.
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)
    if gaze_offsets is not None:
        gaze_offsets = np.asarray(gaze_offsets, dtype=np.float64)

    validate_trajectory(
        times, positions, context="prepare_view_binning", n_dims=env.n_dims
    )
    # Headings are checked here, not by validate_trajectory, so a degree-like
    # heading warns once, at the public entry point.
    for name, values in (("headings", headings), ("gaze_offsets", gaze_offsets)):
        if values is not None and len(values) != len(times):
            raise ValueError(
                _format_error(
                    f"prepare_view_binning: {name} has {len(values)} samples but "
                    f"times has {len(times)}.",
                    why=f"row k of {name} describes the sample at times[k]",
                    fix=f"pass one {name} row per timestamp (NaN marks a "
                    "missing sample)",
                )
            )

    # Precompute view bins ONCE (shared across all neurons) unless the caller
    # already holds them for these exact samples and gaze settings.
    # This is the expensive computation - computed once instead of per-neuron
    mask, observed = _view_observation_mask(
        times,
        positions,
        headings,
        gaze_offsets,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        break_before=break_before,
    )
    if view_bins is None:
        view_bins = _precompute_view_bins(
            env,
            positions,
            headings,
            gaze_model=gaze_model,
            view_distance=view_distance,
            gaze_offsets=gaze_offsets,
            boundary_source=boundary_source,
            interval_mask=mask,
        )

    mask &= hold_last_valid(view_bins, observed, fill=-1)[:-1] >= 0
    n_bins = env.n_bins
    prepared = prepare_frame_binning(
        times,
        view_bins,
        n_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        interval_mask=mask,
        break_before=break_before,
        sample_valid=observed,
    )
    return prepared
