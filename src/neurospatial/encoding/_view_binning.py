"""Binning layer for view encoding (spatial view cells).

This module converts spike trains and gaze data into discrete spike counts
and view occupancy arrays that can be processed by the smoothing layer.

The key difference from spatial binning (_binning.py) is that:
- Position binning: bins spikes by *where the animal was*
- View binning: bins spikes by *where the animal was looking*

The functions in this module handle:
1. Computation of viewed locations from positions, headings, and gaze model
2. View occupancy computation (time spent *viewing* each bin)
3. Spike binning based on viewed location at spike time
4. Batch processing of multiple neurons with joblib parallelization

Output shapes:
- Spike counts (single neuron): (n_bins,)
- Spike counts (batch): (n_neurons, n_bins)
- View occupancy: (n_bins,) - always shared across neurons

Gaze models supported:
- "fixed_distance": Point at fixed distance in gaze direction (default)
- "ray_cast": Intersection with environment boundary
- "boundary": Nearest boundary point in gaze direction
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import TimeSupport
from neurospatial.encoding._binning import (
    FrameBinning,
    count_spikes_by_frame,
    prepare_frame_binning,
)
from neurospatial.encoding._validation import validate_times as _validate_times
from neurospatial.environment.trajectory import (
    interval_valid_mask,
    start_allocated_occupancy,
)
from neurospatial.ops.visibility import compute_viewed_location

if TYPE_CHECKING:
    import shapely

    from neurospatial.arena import Arena
    from neurospatial.environment import Environment
    from neurospatial.layout._binning import Locator

__all__ = [
    "bin_view_spike_train",
    "bin_view_spike_trains",
    "compute_occupancy",
]


def _resolve_wall_contacts(
    locator: Locator,
    arena: Arena,
    cells: Sequence[shapely.Geometry],
    walls: shapely.Geometry,
    hits: NDArray[np.float64],
    origins: NDArray[np.float64],
    bins: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Resolve rounded boundary hits using cells that actually touch the wall.

    A single representable step can remain outside a slanted polygon. For a
    failed hit, intersect its approach segment with a retained wall-touching
    cell, and query an interior point of the terminal segment. Contact must be
    within the cell's bounded coordinate uncertainty; a missing strip between
    a cell and the wall is never bridged by an inward search.
    """
    import shapely

    from neurospatial.arena import _feature_cap
    from neurospatial.layout._locators import RectilinearLocator, SubsetLocator
    from neurospatial.layout._polygon import ClippedLocator
    from neurospatial.layout._validation import _polygon_coordinate_errors

    def same_parent(target: NDArray[np.float64], query: NDArray[np.float64]) -> bool:
        parent = locator
        while True:
            if isinstance(parent, RectilinearLocator):
                # Optimized grid subsets retain inactive cells in their grid
                # mask, rather than a SubsetLocator wrapper. Preserve that
                # full-grid owner even when its active-bin index is -1.
                return all(
                    np.searchsorted(edge, target[0, axis], side="right")
                    == np.searchsorted(edge, query[0, axis], side="right")
                    for axis, edge in enumerate(parent.edges)
                )
            original = parent.locate(target)[0]
            if original >= 0 and parent.locate(query)[0] != original:
                return False
            if not isinstance(parent, (SubsetLocator, ClippedLocator)):
                return True
            parent = parent.base

    missing = np.flatnonzero((bins < 0) & np.all(np.isfinite(hits), axis=1))
    if not len(missing):
        return bins
    polygons = np.asarray(cells, dtype=object)
    # Union first: adjacent clipped cells can carry independently rounded
    # fragments of the same wall. Their shared edges cancel in the union.
    contact = shapely.union_all(polygons).intersection(walls)
    if contact.is_empty:
        return bins
    budgets = np.array(
        [
            np.hypot.reduce(
                np.minimum(_polygon_coordinate_errors(cell), _feature_cap(arena))
            )
            for cell in cells
        ]
    )
    for row in missing:
        point = shapely.Point(hits[row])
        distances = shapely.distance(polygons, point)
        candidates = np.flatnonzero(
            (distances <= budgets) & (contact.distance(point) <= budgets)
        )
        end = shapely.get_point(shapely.shortest_line(point, contact), -1)
        segment = shapely.LineString([origins[row], [end.x, end.y]])
        for index in candidates:
            pieces = shapely.get_parts(segment.intersection(cells[index]))
            for piece in pieces:
                if piece.geom_type != "LineString" or piece.length == 0:
                    continue
                # Only the terminal approach interval establishes ownership.
                if piece.distance(end) > budgets[index]:
                    continue
                middle = piece.interpolate(0.5, normalized=True)
                query = np.array([[middle.x, middle.y]])
                if arena.domain_known and not arena.contains(query)[0]:
                    continue
                original = np.nextafter(hits[row : row + 1], origins[row : row + 1])
                if locator.locate(query)[0] == index and same_parent(original, query):
                    bins[row] = index
                    break
            if bins[row] >= 0:
                break

    return bins


def _wall_view_bins(
    env: Environment,
    hits: NDArray[np.float64],
    origins: NDArray[np.float64],
    source: Literal["physical", "analysis"],
) -> NDArray[np.intp]:
    """Assign the observer-side limit in the retained cell-owner frame."""
    import shapely
    from shapely.ops import unary_union

    from neurospatial.arena import _feature_cap, _TransformedArena
    from neurospatial.environment._arena import _wall_source_frame
    from neurospatial.layout._composition import CombinedLocator
    from neurospatial.layout._locators import SubsetLocator, TransformedLocator
    from neurospatial.layout._polygon import (
        ClippedLocator,
        PolygonCells,
        materialize_cells,
    )
    from neurospatial.ops._boundaries import physical_walls

    def source_locator(
        locator: Locator, arena: Arena, centers: NDArray[np.float64]
    ) -> Locator:
        if isinstance(locator, TransformedLocator):
            if isinstance(arena, _TransformedArena) and np.array_equal(
                locator.inverse, arena.inverse
            ):
                original, _error = arena._project(centers)
                return source_locator(locator.base, arena.base, original)
            if source == "physical":
                raise NotImplementedError(
                    "Physical wall-bin assignment needs a shared source frame; retain the arena's transform with the bins, or explicitly choose boundary_source='analysis'."
                )
        if isinstance(locator, SubsetLocator):
            return SubsetLocator(
                source_locator(locator.base, arena, centers), locator.parent_to_child
            )
        if isinstance(locator, ClippedLocator):
            local = _wall_source_frame(locator, arena, centers)
            if local is not None:
                return ClippedLocator(
                    source_locator(locator.base, arena, centers),
                    PolygonCells(local[0]),
                    locator.mapping,
                )
        if isinstance(locator, CombinedLocator) and source == "physical":
            # A child with its own affine frame has no one shared owner frame.
            pending = [child.locator for child in locator.children]
            while pending:
                child = pending.pop()
                if isinstance(child, TransformedLocator):
                    raise NotImplementedError(
                        "Physical wall-bin assignment needs a shared source frame for combined components. Combine before a shared transform, or explicitly choose boundary_source='analysis'."
                    )
                if isinstance(child, (SubsetLocator, ClippedLocator)):
                    pending.append(child.base)
                elif isinstance(child, CombinedLocator):
                    pending.extend(part.locator for part in child.children)
        return locator

    locator = source_locator(env._binning.locator, env.arena, env.bin_centers)
    frame = _wall_source_frame(env._binning.locator, env.arena, env.bin_centers)
    if frame is None:
        bins = env.bin_at(np.nextafter(hits, origins))
        if source == "physical":
            walls = physical_walls(env.arena)
        else:
            from neurospatial.ops._boundaries import boundary_geometry

            walls = boundary_geometry(env, source)
        return _resolve_wall_contacts(
            locator,
            env.arena,
            materialize_cells(env._binning),
            walls,
            hits,
            origins,
            bins,
        )
    cells, _walls, _centers, _inverse, frames = frame
    query = np.vstack((hits, origins))
    error = None
    for arena in frames:
        query, error = arena._project(query, error)
    base = frames[-1].base
    if source == "physical":
        walls = physical_walls(base)
    else:
        domain = unary_union(cells)
        restriction = base.polygonal_domain() if base.domain_known else None
        if restriction is not None:
            domain = domain.intersection(restriction)
        walls = unary_union([domain.boundary, *base.walls])
    assert error is not None
    points = shapely.points(query[: len(hits)])
    allowance = np.hypot.reduce(
        np.minimum(error[: len(hits)], _feature_cap(base)), axis=1
    )
    close = shapely.distance(points, walls) <= allowance
    nearest = shapely.get_point(shapely.shortest_line(points[close], walls), -1)
    query[np.flatnonzero(close)] = np.column_stack(
        (shapely.get_x(nearest), shapely.get_y(nearest))
    )
    inside = np.nextafter(query[: len(hits)], query[len(hits) :])
    bins = locator.locate(inside)
    if base.domain_known:
        bins[~base.contains(inside)] = -1
    return _resolve_wall_contacts(
        locator, base, cells, walls, query[: len(hits)], query[len(hits) :], bins
    )


def _view_observation_support(
    times: NDArray[np.float64],
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
) -> TimeSupport:
    """Resolve selected exposure before requiring bins at physical wall hits."""
    mask = interval_valid_mask(
        times, max_gap=max_gap, epochs=epochs, spike_window=spike_window
    )
    mask &= np.diff(times) > 0
    return TimeSupport.from_mask(
        times, mask, epochs=epochs, spike_window=spike_window, max_gap=max_gap
    )


def _precompute_view_bins(
    env: Environment,
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
    support: TimeSupport | None = None,
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
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. For ray-based rate maps, a finite
        physical hit without a retained bin raises NotImplementedError; provide
        bins at the wall or explicitly select the analysis approximation.
        Fixed-distance gaze ignores this.

    Returns
    -------
    view_bins : ndarray, shape (n_samples,), dtype=intp
        Bin index for each sample. -1 indicates invalid view (outside env).

    Notes
    -----
    Without ``support``, this is a geometry-only lookup and unowned hits stay
    -1. Allocation callers provide canonical observation support; only finite
    physical hits at starts of selected positive-duration intervals require
    retained bins. Final samples and excluded intervals supply no exposure.
    """
    n_samples = len(positions)

    max_distance = 100.0
    if gaze_model != "fixed_distance":
        from neurospatial.ops._boundaries import boundary_geometry, search_distance

        max_distance = search_distance(
            boundary_geometry(env, boundary_source), positions
        )

    # Compute viewed locations for all timepoints
    viewed_locations = compute_viewed_location(
        positions,
        headings,
        method=gaze_model,
        boundary_source=boundary_source,
        max_distance=max_distance,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        env=env if gaze_model in ("ray_cast", "boundary") else None,
    )

    # Identify valid viewed locations (finite and inside environment)
    valid_view_mask = np.all(np.isfinite(viewed_locations), axis=1)

    # Map viewed locations to bins
    view_bins = np.full(n_samples, -1, dtype=np.intp)
    if np.any(valid_view_mask):
        valid_viewed = viewed_locations[valid_view_mask]
        if gaze_model != "fixed_distance":
            valid_bins = _wall_view_bins(
                env, valid_viewed, positions[valid_view_mask], boundary_source
            )
            required = np.zeros(n_samples, dtype=bool)
            if support is not None:
                required[:-1] = support.mask
            if boundary_source == "physical" and np.any(
                (valid_bins < 0) & required[valid_view_mask]
            ):
                raise NotImplementedError(
                    "Physical wall hit has no retained bin on the observer's side. "
                    "Build bins that reach the measured wall before computing view rates; "
                    "adding OpenArena or clipping existing cells cannot add missing bins. "
                    "For rotated rectangular arenas, construct bins in the arena-aligned frame "
                    "and transform bins, arena and observations together. Alternatively, explicitly "
                    "choose boundary_source='analysis' to encode the retained analysis boundary. "
                    "compute_viewed_location(...) remains available for geometry-only wall contacts."
                )
        else:
            valid_bins = env.bin_at(valid_viewed)
        view_bins[valid_view_mask] = valid_bins
        # Bins outside environment return -1 from bin_at, which is what we want

    return view_bins


def compute_occupancy(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> NDArray[np.float64]:
    """Compute view occupancy (time spent viewing each bin).

    Computes the total time spent *viewing* each spatial bin by computing
    viewed locations from positions and headings, then accumulating time
    intervals per viewed bin.

    This differs from position occupancy which measures time spent *at* each bin.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location:
        - "fixed_distance": Point at fixed distance in gaze direction
        - "ray_cast": Intersection with environment boundary
        - "boundary": Nearest boundary point in gaze direction
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction (e.g., from eye tracking).
        If None, gaze is aligned with heading.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : ndarray of shape (n, 2), or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    spike_window : ndarray of shape (n, 2), or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
        The calling public encoder records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
        Windows must already be normalized by ``resolve_time_windows``;
        public encoders accept and normalize the other supported input forms.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. For ray-based rate maps, a finite
        physical hit without a retained bin raises NotImplementedError; provide
        bins at the wall or explicitly select the analysis approximation.
        Fixed-distance gaze ignores this.

    Returns
    -------
    ndarray, shape (n_bins,)
        Time in seconds spent viewing each spatial bin.

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths.
        If gaze_model is invalid.
        If fewer than 2 samples provided.
        If times are not monotonically non-decreasing.

    Notes
    -----
    View occupancy is computed by:
    1. Computing viewed locations for each time sample using the gaze model
    2. Mapping viewed locations to spatial bins
    3. Accumulating time intervals (dt) per bin using np.add.at

    Viewed locations outside the environment (invalid bins) do not contribute
    to occupancy. This means total view occupancy may be less than recording
    duration if many views fall outside the environment.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._view_binning import compute_occupancy

    >>> # Create environment
    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory with positions and headings
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.column_stack(
    ...     [
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),  # x
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),  # y
    ...     ]
    ... )
    >>> trajectory = np.clip(trajectory, 20, 80)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)

    >>> # Compute view occupancy
    >>> view_occ = compute_occupancy(env, times, trajectory, headings)
    >>> view_occ.shape[0] == env.n_bins
    True
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Validate input shapes
    n_samples = len(times)
    if len(positions) != n_samples:
        raise ValueError(
            f"times length ({n_samples}) must match positions length ({len(positions)})"
        )
    if len(headings) != n_samples:
        raise ValueError(
            f"times length ({n_samples}) must match headings length ({len(headings)})"
        )

    # Validate times (minimum samples and monotonicity)
    _validate_times(times, context="compute_occupancy")

    # Validate gaze_model
    valid_gaze_models = {"fixed_distance", "ray_cast", "boundary"}
    if gaze_model not in valid_gaze_models:
        raise ValueError(
            f"Invalid gaze_model: '{gaze_model}'. "
            f"Must be one of {sorted(valid_gaze_models)}"
        )

    # Validate gaze_offsets if provided
    if gaze_offsets is not None:
        gaze_offsets = np.asarray(gaze_offsets, dtype=np.float64)
        if len(gaze_offsets) != n_samples:
            raise ValueError(
                f"gaze_offsets length ({len(gaze_offsets)}) must match "
                f"times length ({n_samples})"
            )

    view_bins = _precompute_view_bins(
        env,
        positions,
        headings,
        gaze_model=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        boundary_source=boundary_source,
        support=_view_observation_support(
            times, max_gap=max_gap, epochs=epochs, spike_window=spike_window
        ),
    )
    mask = interval_valid_mask(
        times,
        start_bin=view_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    return start_allocated_occupancy(view_bins, np.diff(times), mask, env.n_bins)


def bin_view_spike_train(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> NDArray[np.float64]:
    """Bin spike train by viewed location.

    Converts continuous spike times to spike counts per spatial bin based on
    where the animal was *looking* at each spike time, not where the animal
    was located.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : ndarray of shape (n, 2), or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    spike_window : ndarray of shape (n, 2), or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
        The calling public encoder records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
        Windows must already be normalized by ``resolve_time_windows``;
        public encoders accept and normalize the other supported input forms.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. For ray-based rate maps, a finite
        physical hit without a retained bin raises NotImplementedError; provide
        bins at the wall or explicitly select the analysis approximation.
        Fixed-distance gaze ignores this.

    Returns
    -------
    ndarray, shape (n_bins,)
        Number of spikes in each spatial bin based on viewed location
        (float64 for compatibility with smoothing operations).

    Raises
    ------
    ValueError
        If fewer than 2 trajectory samples provided.
        If times are not monotonically non-decreasing.

    Notes
    -----
    Spikes are assigned to bins based on the viewed location at the nearest
    behavioral frame to each spike time. This uses nearest-neighbor lookup
    (via np.searchsorted) rather than interpolation because gaze direction
    cannot be meaningfully interpolated.

    Spikes where the viewed location is outside the environment or invalid
    are excluded from counts.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._view_binning import bin_view_spike_train

    >>> # Create environment
    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.column_stack(
    ...     [
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...     ]
    ... )
    >>> trajectory = np.clip(trajectory, 20, 80)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5])

    >>> # Bin spikes by viewed location
    >>> spike_counts = bin_view_spike_train(
    ...     env, spike_times, times, trajectory, headings
    ... )
    >>> spike_counts.shape[0] == env.n_bins
    True
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Validate times (minimum samples and monotonicity)
    _validate_times(times, context="bin_view_spike_train")

    view_bins = _precompute_view_bins(
        env,
        positions,
        headings,
        gaze_model=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        boundary_source=boundary_source,
        support=_view_observation_support(
            times, max_gap=max_gap, epochs=epochs, spike_window=spike_window
        ),
    )
    mask = interval_valid_mask(
        times,
        start_bin=view_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    return count_spikes_by_frame(spike_times, times, view_bins, mask, env.n_bins)


def prepare_view_binning(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    view_bins: NDArray[np.intp] | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> FrameBinning:
    """Prepare geometry and exact temporal support before counting spikes."""
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64)

    # Validate times (minimum samples and monotonicity)
    _validate_times(times, context="bin_view_spike_trains")

    # Precompute view bins ONCE (shared across all neurons) unless the caller
    # already holds them for these exact samples and gaze settings.
    # This is the expensive computation - computed once instead of per-neuron
    if view_bins is None:
        view_bins = _precompute_view_bins(
            env,
            positions,
            headings,
            gaze_model=gaze_model,
            view_distance=view_distance,
            gaze_offsets=gaze_offsets,
            boundary_source=boundary_source,
            support=_view_observation_support(
                times, max_gap=max_gap, epochs=epochs, spike_window=spike_window
            ),
        )

    n_bins = env.n_bins
    prepared = prepare_frame_binning(
        times,
        view_bins,
        n_bins,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
    )
    return prepared


def bin_view_spike_trains(
    env: Environment,
    spike_times: Sequence[NDArray[np.float64]] | NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    headings: NDArray[np.float64],
    *,
    gaze_model: Literal["fixed_distance", "ray_cast", "boundary"] = "fixed_distance",
    view_distance: float = 10.0,
    gaze_offsets: NDArray[np.float64] | None = None,
    max_gap: float | None = 0.5,
    epochs: NDArray[np.float64] | None = None,
    spike_window: NDArray[np.float64] | None = None,
    n_jobs: int = 1,
    view_bins: NDArray[np.intp] | None = None,
    boundary_source: Literal["physical", "analysis"] = "physical",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin multiple spike trains by viewed location.

    Batch version of bin_view_spike_train that efficiently processes multiple
    neurons. Precomputes shared quantities (viewed locations, view bins,
    view occupancy) and optionally parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment
        The spatial environment defining bin structure.
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Can be:
        - List/tuple of 1D arrays (one per neuron)
        - 2D array shape (n_neurons, max_spikes) with NaN padding
        Input is coerced to per-neuron spike trains via as_spike_trains().
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds.
    positions : ndarray, shape (n_samples, 2)
        Position coordinates at each time sample.
    headings : ndarray, shape (n_samples,)
        Head direction at each time sample (radians, 0=East).
    gaze_model : {"fixed_distance", "ray_cast", "boundary"}, default="fixed_distance"
        Method for computing viewed location.
    view_distance : float, default=10.0
        Distance for fixed_distance gaze model (environment units).
    gaze_offsets : ndarray, shape (n_samples,), optional
        Offset from heading to gaze direction.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : ndarray of shape (n, 2), or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
    spike_window : ndarray of shape (n, 2), or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
        The calling public encoder records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
        Windows must already be normalized by ``resolve_time_windows``;
        public encoders accept and normalize the other supported input forms.
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    view_bins : ndarray of int, shape (n_samples,), optional
        Viewed bin per sample from ``_precompute_view_bins`` for exactly these
        samples and gaze settings. Passing it skips the gaze computation, so a
        caller that bins many spike sets on one trajectory computes it once.

    boundary_source : {'physical', 'analysis'}, default='physical'
        Declared physical walls, or explicitly requested analysis-cell boundaries.
        Physical mode requires exact supported wall geometry; missing analysis
        bins do not occlude a physical ray. For ray-based rate maps, a finite
        physical hit without a retained bin raises NotImplementedError; provide
        bins at the wall or explicitly select the analysis approximation.
        Fixed-distance gaze ignores this.

    Returns
    -------
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron,
        based on viewed location at spike time.
    occupancy : ndarray, shape (n_bins,)
        Time in seconds spent viewing each spatial bin (shared across neurons).

    Raises
    ------
    ValueError
        If fewer than 2 trajectory samples provided.
        If times are not monotonically non-decreasing.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._view_binning import bin_view_spike_trains

    >>> # Create environment
    >>> positions = np.random.rand(100, 2) * 100
    >>> env = Environment.from_samples(positions, bin_size=10.0)

    >>> # Create trajectory
    >>> times = np.linspace(0, 10, 1000)
    >>> trajectory = np.column_stack(
    ...     [
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...         50 + np.cumsum(np.random.randn(1000) * 0.5),
    ...     ]
    ... )
    >>> trajectory = np.clip(trajectory, 20, 80)
    >>> headings = np.random.uniform(0, 2 * np.pi, 1000)
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0]),  # Neuron 2
    ... ]

    >>> # Bin spikes by viewed location
    >>> spike_counts, view_occ = bin_view_spike_trains(
    ...     env, spike_times, times, trajectory, headings, n_jobs=2
    ... )
    >>> spike_counts.shape == (3, env.n_bins)
    True
    >>> view_occ.shape == (env.n_bins,)
    True

    See Also
    --------
    bin_view_spike_train : Single-neuron version
    compute_occupancy : Compute view occupancy only
    """
    from neurospatial.encoding._spikes import as_spike_trains

    # Normalize spike times to canonical list-of-arrays format
    spike_times_list = as_spike_trains(spike_times)

    prepared = prepare_view_binning(
        env,
        times,
        positions,
        headings,
        gaze_model=gaze_model,
        view_distance=view_distance,
        gaze_offsets=gaze_offsets,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        view_bins=view_bins,
        boundary_source=boundary_source,
    )
    spike_counts, occupancy = prepared.count(spike_times_list, n_jobs=n_jobs)
    return spike_counts, occupancy
