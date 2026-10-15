"""Spatial field operations for Environment class.

This module provides methods for processing spatial fields, including
kernel computation, smoothing, and interpolation.

Key Features
------------
- Diffusion kernel computation with caching
- Graph-based field smoothing
- Field interpolation at arbitrary points (nearest/linear modes)

Notes
-----
This is a mixin class designed to be used with Environment. It should NOT
be decorated with @dataclass. Only the main Environment class in core.py
should be a dataclass.

TYPE_CHECKING Pattern
---------------------
To avoid circular imports, we import Environment only for type checking.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial import _constants
from neurospatial._lru import lru_get, lru_put
from neurospatial.environment._protocols import (
    FieldConsumer,
)
from neurospatial.layout._diffusion import (
    DiffusionGeometry,
    FaceGeometry,
)
from neurospatial.layout._grid_frame import require_grid_context

if TYPE_CHECKING:
    from neurospatial import Environment
    from neurospatial.environment._protocols import InterpolationConsumer
    from neurospatial.layout._binning import Binning
    from neurospatial.ops._types import (
        DiffusionEigenbasisCache,
        KernelMode,
        MRFEigenbasisCache,
    )
    from neurospatial.ops.diffusion import MRFBasis


def _interpolate(
    env: InterpolationConsumer,
    field: NDArray[np.float64],
    points: NDArray[np.float64],
    *,
    mode: Literal["nearest", "linear"] = "nearest",
) -> NDArray[np.float64]:
    """Interpolate a bin-valued field at continuous query points."""
    # Input validation - field
    field = np.asarray(field, dtype=np.float64)

    # Check field dimensionality
    if field.ndim != 1:
        raise ValueError(
            f"Field must be 1-D array (got {field.ndim}-D array). "
            f"Expected shape (n_bins,) = ({env.n_bins},), got shape {field.shape}."
        )

    # Check field shape matches n_bins
    if field.shape[0] != env.n_bins:
        raise ValueError(
            f"Field shape {field.shape} must match n_bins={env.n_bins}. "
            f"Expected shape (n_bins,) = ({env.n_bins},), got ({field.shape[0]},)."
        )

    # Check for NaN/Inf values in field
    if np.any(np.isnan(field)):
        raise ValueError(
            "Field contains NaN values. "
            f"Found {np.sum(np.isnan(field))} NaN values out of {len(field)} bins. "
            "NaN values are not supported in interpolation operations."
        )

    if np.any(np.isinf(field)):
        raise ValueError(
            "Field contains infinite values. "
            f"Found {np.sum(np.isinf(field))} infinite values out of {len(field)} bins. "
            "Infinite values are not supported in interpolation operations."
        )

    # Input validation - points
    points = np.asarray(points, dtype=np.float64)

    # Check points dimensionality
    if points.ndim != 2:
        raise ValueError(
            f"Points must be 2-D array (got {points.ndim}-D array). "
            f"Expected shape (n_points, n_dims), got shape {points.shape}."
        )

    # Check points dimension matches environment
    n_dims = env.bin_centers.shape[1]
    if points.shape[1] != n_dims:
        raise ValueError(
            f"Points dimension {points.shape[1]} must match environment "
            f"dimension {n_dims}. Expected shape (n_points, {n_dims}), "
            f"got shape {points.shape}."
        )

    # Check for NaN/Inf values in points
    if np.any(~np.isfinite(points)):
        n_invalid = np.sum(~np.isfinite(points))
        raise ValueError(
            f"Points array contains {n_invalid} non-finite value(s) (NaN or Inf). "
            f"All point coordinates must be finite. Check your input data for "
            f"missing values or infinities."
        )

    # Validate mode
    valid_modes = {"nearest", "linear"}
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes} (got '{mode}'). "
            "Use 'nearest' for nearest-neighbor interpolation or 'linear' "
            "for bilinear/trilinear interpolation (grids only)."
        )

    # Handle empty points array
    if points.shape[0] == 0:
        return np.array([], dtype=np.float64)

    # Dispatch based on mode
    if mode == "nearest":
        values = _interpolate_nearest(env, field, points)
    else:  # mode == "linear"
        values = _interpolate_linear(env._binning, field, points)
    owners = env._locate(points)
    values[owners < 0] = np.nan
    return values


def _interpolate_nearest(
    env: InterpolationConsumer,
    field: NDArray[np.float64],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Nearest-neighbor interpolation using KDTree.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values.
    points : NDArray[np.float64], shape (n_points, n_dims)
        Query points.

    Returns
    -------
    values : NDArray[np.float64], shape (n_points,)
        Interpolated values (NaN for points outside).

    """
    from typing import cast

    from neurospatial.ops.binning import map_points_to_bins

    # Map points to bins (-1 for outside points)
    # With return_dist=False, we get just the indices (not a tuple)
    bin_indices = cast(
        "NDArray[np.int64]",
        map_points_to_bins(
            cast("Environment", env),
            points,
            tie_break="lowest_index",
            return_dist=False,
        ),
    )

    # Initialize result with NaN
    result = np.full(points.shape[0], np.nan, dtype=np.float64)

    # Fill in values for points inside environment
    inside_mask = bin_indices >= 0
    result[inside_mask] = field[bin_indices[inside_mask]]

    return result


def _interpolate_linear(
    binning: Binning,
    field: NDArray[np.float64],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linear interpolation using scipy RegularGridInterpolator.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Field values.
    points : NDArray[np.float64], shape (n_points, n_dims)
        Query points.

    Returns
    -------
    values : NDArray[np.float64], shape (n_points,)
        Interpolated values (NaN for points outside).

    Raises
    ------
    NotImplementedError
        If layout has no rectangular grid cells.

    Notes
    -----
    On a holed/masked grid (where ``n_bins < prod(grid_shape)``), a missing
    corner contributes NaN only when its interpolation weight is positive.
    Active-bin centers therefore retain their known values, while points
    requiring an inactive cell's value and out-of-bounds points return NaN.

    """
    grid, frame = require_grid_context(
        binning,
        "interpolate(mode='linear')",
        fix="use mode='nearest', which works on every layout",
    )

    # Import scipy
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as e:
        raise ImportError(
            "Linear interpolation requires scipy. Install with: pip install scipy"
        ) from e

    grid_shape = grid.shape
    grid_edges = grid.edges
    n_dims = len(grid_shape)

    # Scatter active-bin values into a full-grid array before reshaping.
    # `field` is indexed by active-bin id (length n_bins); `grid_shape`
    # is the FULL grid. On a holed/masked grid n_bins < prod(grid_shape),
    # so a direct reshape would raise. Use finite placeholders for inactive
    # cells and mask their positive-weight contributions separately below:
    # 0 * NaN would otherwise erase even exact active-bin center values.
    n_full = int(np.prod(grid_shape))
    active_mask = grid.active_mask
    if field.shape[0] == n_full:
        # Fully-active grid: field already covers every cell.
        field_grid = field.reshape(grid_shape)
    else:
        full_field = np.zeros(n_full, dtype=np.float64)
        full_field[np.flatnonzero(active_mask.ravel())] = field
        field_grid = full_field.reshape(grid_shape)

    # Create grid points for each dimension (bin centers)
    grid_points: list[NDArray[np.float64]] = []
    for dim in range(n_dims):
        edges = grid_edges[dim]
        # Bin centers are midpoints between edges
        centers = (edges[:-1] + edges[1:]) / 2
        grid_points.append(centers)

    if frame is not None:
        points = frame._project_to_planes(
            points, grid_points, operation="Linear interpolation"
        )

    # Create interpolator
    # bounds_error=False, fill_value=np.nan → outside points return NaN
    interpolator = RegularGridInterpolator(
        grid_points,
        field_grid,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Evaluate at query points
    result: NDArray[np.float64] = interpolator(points)
    if field.shape[0] != n_full:
        missing_weight = RegularGridInterpolator(
            grid_points,
            (~active_mask).astype(np.float64),
            method="linear",
            bounds_error=False,
            fill_value=1.0,
        )(points)
        result[missing_weight > 0.0] = np.nan

    return result


def _compute_kernel(
    env: FieldConsumer,
    bandwidth: float,
    *,
    mode: KernelMode = "density",
    cache: bool = True,
) -> NDArray[np.float64]:
    """Implement Environment.compute_kernel using explicit fields inputs."""
    from neurospatial.ops.diffusion import diffusion_kernel

    # Validate mode.
    valid_modes = {"transition", "density", "average"}
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes} (got '{mode}'). "
            "Use 'transition' for mass-conserving smoothing of extensive "
            "quantities, 'density' for count-to-density (KDE), or 'average' "
            "for volume-unbiased averaging of an intensive field (rate map)."
        )

    # Initialize cache if it doesn't exist
    # (for backward compatibility with environments deserialized from older versions)
    if not hasattr(env, "_kernel_cache"):
        empty: dict[tuple[float, KernelMode], NDArray[np.float64]] = {}
        env._kernel_cache = empty

    # Check cache first if enabled
    cache_key = (bandwidth, mode)
    if cache:
        cached = lru_get(env._kernel_cache, cache_key)
        if cached is not None:
            return cast("NDArray[np.float64]", cached.copy())

    # Compute kernel from the cached diffusion geometry, whose cell measures
    # (the layout's bin sizes) are the mass matrix M used by every mode.
    kernel = diffusion_kernel(env, bandwidth, mode=mode)

    # Store in cache if enabled
    if cache:
        lru_put(
            env._kernel_cache,
            cache_key,
            kernel,
            limit=_constants.MAX_CACHED_KERNELS,
        )

    return kernel.copy() if cache else kernel


def _smooth(
    env: FieldConsumer,
    field: NDArray[np.float64],
    bandwidth: float,
    *,
    mode: KernelMode = "average",
) -> NDArray[np.float64]:
    """Implement Environment.smooth using explicit fields inputs."""
    field = np.asarray(field, dtype=np.float64)

    # Check field dimensionality
    if field.ndim != 1:
        raise ValueError(
            f"Field must be 1-D array (got {field.ndim}-D array). "
            f"Expected shape (n_bins,) = ({env.n_bins},), got shape {field.shape}."
        )

    # Check field shape matches n_bins
    if field.shape[0] != env.n_bins:
        raise ValueError(
            f"Field shape {field.shape} must match n_bins={env.n_bins}. "
            f"Expected shape (n_bins,) = ({env.n_bins},), got ({field.shape[0]},)."
        )

    # Check for NaN/Inf values
    if np.any(np.isnan(field)):
        raise ValueError(
            "Field contains NaN values. "
            f"Found {np.sum(np.isnan(field))} NaN values out of {len(field)} bins. "
            "NaN values are not supported in smoothing operations."
        )

    if np.any(np.isinf(field)):
        raise ValueError(
            "Field contains infinite values. "
            f"Found {np.sum(np.isinf(field))} infinite values out of {len(field)} bins. "
            "Infinite values are not supported in smoothing operations."
        )

    # Validate bandwidth
    if bandwidth <= 0:
        raise ValueError(
            f"bandwidth must be positive (got {bandwidth}). "
            "Bandwidth controls the spatial scale of smoothing."
        )

    # Validate mode
    valid_modes = {"transition", "density", "average"}
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes} (got '{mode}'). "
            "Use 'transition' for mass-conserving smoothing of extensive "
            "data, 'density' for count-to-density, or 'average' for "
            "volume-unbiased averaging of an intensive field."
        )

    # Apply the diffusion smoothing via the matrix-free apply-path
    # (env.diffuse). This is a PURE LINEAR operator -- no clip / renorm --
    # so smoothing a SIGNED field is preserved. It reproduces the dense
    # kernel to within the truncation tolerance (see the "Approximation"
    # note below); positivity, where needed, is the caller's job.
    return env.diffuse(field, bandwidth, mode=mode)


def _diffuse(
    env: FieldConsumer,
    fields: Any,
    bandwidth: float,
    *,
    mode: KernelMode = "density",
    backend: Literal["numpy", "jax"] = "numpy",
) -> NDArray[np.float64]:
    """Implement Environment.diffuse using explicit fields inputs."""
    from neurospatial.ops.diffusion import diffusion_apply

    if mode not in ("transition", "density", "average"):
        raise ValueError(
            f"mode must be one of {{'transition', 'density', 'average'}} "
            f"(got '{mode}')."
        )
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'numpy' or 'jax' (got '{backend}').")
    if not bandwidth > 0:
        raise ValueError(
            f"bandwidth must be positive (got {bandwidth}). "
            "Bandwidth controls the spatial scale of smoothing."
        )

    # Coerce to 2-D (n_bins, n_fields); a 1-D field is squeezed back on
    # return. np.ndim / np.shape avoid forcing a JAX array to NumPy.
    if backend == "numpy":
        fields = np.asarray(fields, dtype=np.float64)
    ndim = int(np.ndim(fields))
    if ndim not in (1, 2):
        raise ValueError(
            f"fields must be 1-D (n_bins,) or 2-D (n_bins, n_fields), got "
            f"{ndim}-D array."
        )
    shape = np.shape(fields)
    if shape[0] != env.n_bins:
        raise ValueError(
            f"fields leading dimension {shape[0]} must match n_bins={env.n_bins}."
        )
    was_1d = ndim == 1
    fields_2d = fields.reshape(-1, 1) if was_1d else fields

    W, volumes, n_components, _labels = env._diffusion_geometry
    holder = env._diffusion_eigenbasis
    result = diffusion_apply(
        holder,
        W,
        volumes,
        n_components,
        fields_2d,
        float(bandwidth),
        mode,
        backend=backend,
    )
    # numpy backend -> ndarray; jax backend -> jax.Array (documented above).
    # The NDArray annotation is the numpy-path type; cast covers both.
    return cast("NDArray[np.float64]", result[:, 0] if was_1d else result)


def _face_geometry(env: FieldConsumer) -> FaceGeometry:
    """Implement Environment._face_geometry using explicit fields inputs."""
    return env._binning.faces.build()


def _require_face_geometry(env: FieldConsumer, operation: str) -> FaceGeometry:
    """Implement Environment._require_face_geometry using explicit fields inputs."""
    if not env._binning.faces.available:
        raise env._binning.faces.missing_error(operation)
    return env._face_geometry


def _diffusion_geometry(env: FieldConsumer) -> DiffusionGeometry:
    """Implement Environment._diffusion_geometry using explicit fields inputs."""
    from neurospatial.layout._composition import combined_diffusion

    combined = combined_diffusion(env._binning)
    if combined is not None:
        return combined
    return DiffusionGeometry.from_faces(env._require_face_geometry("smoothing"))


def _diffusion_eigenbasis(env: FieldConsumer) -> DiffusionEigenbasisCache:
    """Implement Environment._diffusion_eigenbasis using explicit fields inputs."""
    return {}


def _gaussian_kernels(env: FieldConsumer) -> dict[float, NDArray[np.float64]]:
    """Implement Environment._gaussian_kernels using explicit fields inputs."""
    return {}


def _mrf_eigenbasis(env: FieldConsumer) -> MRFEigenbasisCache:
    """Implement Environment._mrf_eigenbasis using explicit fields inputs."""
    n = env.n_bins
    return {
        "Q": np.zeros((n, 0)),
        "Lam": np.zeros(0),
        "mode_comp": np.zeros(0, dtype=np.intp),
        "G": 0,
    }


def _mrf_basis(
    env: FieldConsumer, occupancy: NDArray[np.float64], *, rank: int | None
) -> MRFBasis:
    """Implement Environment._mrf_basis using explicit fields inputs."""
    from neurospatial.ops.diffusion import (
        _ensure_global_modes,
        _live_components,
        _null_mode_indices,
        _symmetric_conjugate,
        _target_live_rank,
        _validate_occupancy,
        select_live_basis,
    )

    _validate_occupancy(occupancy)  # fail fast, before the eigensolve
    _W, volumes, n_components, labels = env._diffusion_geometry
    S = _symmetric_conjugate(_W, volumes)  # cheap; the eigensolve is the cost
    holder = env._mrf_eigenbasis
    live = _live_components(labels, occupancy)
    n_live_components = int(live.size)
    r_eff = _target_live_rank(labels, occupancy, rank)
    n_fill = r_eff - n_live_components
    # ``_symmetric_eigenbasis`` raises on rank < n_components (total geometry
    # components), but ``r_eff`` counts only LIVE components -- so 1 live +
    # several dead would start ``G`` too small and raise on the first solve.
    # Floor ``G`` at ``n_components`` to keep the first solve legal.
    G = max(holder["G"], r_eff, n_components)
    while True:  # grow until every live null is present AND enough fills
        Q, Lam, mode_comp = _ensure_global_modes(holder, S, labels, G)
        # Verify the structural null the selector requires is actually present
        # for every live component (its numerical null can sort AFTER one of
        # its positive modes when both are near machine-epsilon), and count
        # the non-null fill-eligible live modes -- using the same overlap
        # criterion select_live_basis uses, so the two never disagree.
        is_null, missing = _null_mode_indices(Q, mode_comp, volumes, labels, live)
        n_fill_available = int((np.isin(mode_comp, live) & ~is_null).sum())
        covered = missing.size == 0
        if (covered and n_fill_available >= n_fill) or Q.shape[1] >= S.shape[0]:
            break
        G = min(2 * Q.shape[1], S.shape[0])  # double, capped at full rank
    return select_live_basis(Q, Lam, mode_comp, volumes, labels, occupancy, rank=rank)
