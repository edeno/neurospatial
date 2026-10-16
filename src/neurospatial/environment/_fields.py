"""Private spatial field operations on owned geometry and its analysis cache.

Cached analyses (cell faces, diffusion geometry, kernels and eigenbases) are
read and stored through the environment's :class:`AnalysisCache`, so every
function here takes the cache and the published :class:`Binning` explicitly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neurospatial import _constants
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.environment._cache import (
    DIFFUSION_EIGENBASIS,
    DIFFUSION_GEOMETRY,
    DIFFUSION_KERNELS,
    FACE_GEOMETRY,
    GAUSSIAN_KERNELS,
    MRF_EIGENBASIS,
    AnalysisCache,
)
from neurospatial.environment._lookup import nearest_bin
from neurospatial.environment._publication import (
    _array_header,
    _freeze_diffusion_geometry,
    _immutable_array,
)
from neurospatial.geometry._binning import Binning
from neurospatial.geometry._composition import combined_diffusion
from neurospatial.geometry._coordinates import CartesianSystem, EgocentricPolarSystem
from neurospatial.geometry._diffusion import (
    DiffusionGeometry,
    FaceGeometry,
)
from neurospatial.geometry._grid_frame import require_grid_context
from neurospatial.geometry._lookup import locate
from neurospatial.ops.diffusion import (
    MRFBasis,
    _ensure_global_modes,
    _live_components,
    _null_mode_indices,
    _symmetric_conjugate,
    _target_live_rank,
    _validate_occupancy,
    diffusion_apply,
    heat_kernel_from_W,
    select_live_basis,
)
from neurospatial.ops.smoothing import _gaussian_kernel

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.ops._types import (
        DiffusionEigenbasisCache,
        KernelMode,
        MRFEigenbasisCache,
    )


def _interpolate(
    field: NDArray[np.float64],
    points: NDArray[np.float64],
    *,
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    mode: Literal["nearest", "linear"] = "nearest",
) -> NDArray[np.float64]:
    """Interpolate a bin-valued field at continuous query points."""
    n_bins = binning.bins.n_bins
    # Input validation - field
    field = np.asarray(field, dtype=np.float64)

    # Check field dimensionality
    if field.ndim != 1:
        raise ValueError(
            f"Field must be 1-D array (got {field.ndim}-D array). "
            f"Expected shape (n_bins,) = ({n_bins},), got shape {field.shape}."
        )

    # Check field shape matches n_bins
    if field.shape[0] != n_bins:
        raise ValueError(
            f"Field shape {field.shape} must match n_bins={n_bins}. "
            f"Expected shape (n_bins,) = ({n_bins},), got ({field.shape[0]},)."
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
    n_dims = binning.bins.n_dims
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
        values = _interpolate_nearest(cache, binning, field, points)
    else:  # mode == "linear"
        values = _interpolate_linear(binning, field, points)
    owners = locate(binning, arena, points)
    values[owners < 0] = np.nan
    return values


def _interpolate_nearest(
    cache: AnalysisCache,
    binning: Binning,
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
    # Map points to bins (-1 for outside points)
    # With return_dist=False, we get just the indices (not a tuple)
    bin_indices = cast(
        "NDArray[np.int64]",
        nearest_bin(
            cache,
            binning,
            points,
            tie_break="lowest_index",
            return_dist=False,
            max_distance=None,
            max_distance_factor=None,
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
    # Deferred: scipy.interpolate adds about 40 ms to importing neurospatial.
    from scipy.interpolate import RegularGridInterpolator

    grid, frame = require_grid_context(
        binning,
        "interpolate(mode='linear')",
        fix="use mode='nearest', which works on every layout",
    )

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


def face_geometry(cache: AnalysisCache, binning: Binning) -> FaceGeometry:
    """Validated cell faces, built once per published geometry."""
    return cache.get(FACE_GEOMETRY, binning.faces.build)


def require_face_geometry(
    cache: AnalysisCache, binning: Binning, operation: str
) -> FaceGeometry:
    """Cached cell faces, or an actionable error naming ``operation``."""
    if not binning.faces.available:
        raise binning.faces.missing_error(operation)
    return face_geometry(cache, binning)


def _build_diffusion_geometry(
    cache: AnalysisCache, binning: Binning
) -> DiffusionGeometry:
    """Conductances and cell measures from combined components or cell faces."""
    combined = combined_diffusion(binning)
    if combined is not None:
        return _freeze_diffusion_geometry(combined)
    return _freeze_diffusion_geometry(
        DiffusionGeometry.from_faces(require_face_geometry(cache, binning, "smoothing"))
    )


def diffusion_geometry(cache: AnalysisCache, binning: Binning) -> DiffusionGeometry:
    """Cached diffusion geometry ``(W, volumes, n_components, labels)``."""
    return cache.get(
        DIFFUSION_GEOMETRY, lambda: _build_diffusion_geometry(cache, binning)
    )


def diffusion_eigenbasis(cache: AnalysisCache) -> DiffusionEigenbasisCache:
    """Growable truncated-eigenbasis holder for :func:`diffuse`, empty at first."""
    return cache.get(DIFFUSION_EIGENBASIS, lambda: {})


def mrf_eigenbasis(cache: AnalysisCache, n_bins: int) -> MRFEigenbasisCache:
    """Growable global eigenbasis holder for the MRF penalty basis."""

    def build() -> MRFEigenbasisCache:
        return {
            "Q": np.zeros((n_bins, 0)),
            "Lam": np.zeros(0),
            "mode_comp": np.zeros(0, dtype=np.intp),
            "G": 0,
        }

    return cache.get(MRF_EIGENBASIS, build)


def _compute_kernel(
    cache: AnalysisCache,
    binning: Binning,
    bandwidth: float,
    *,
    mode: KernelMode = "density",
    use_cache: bool = True,
) -> NDArray[np.float64]:
    """Dense finite-volume diffusion kernel, optionally cached by bandwidth and mode."""
    # Validate mode.
    valid_modes = {"transition", "density", "average"}
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes} (got '{mode}'). "
            "Use 'transition' for mass-conserving smoothing of extensive "
            "quantities, 'density' for count-to-density (KDE), or 'average' "
            "for volume-unbiased averaging of an intensive field (rate map)."
        )

    # The kernel uses the cached diffusion geometry, whose cell measures
    # (the layout's bin sizes) are the mass matrix M used by every mode.
    def build() -> NDArray[np.float64]:
        geometry = diffusion_geometry(cache, binning)
        return heat_kernel_from_W(geometry.W, geometry.volumes, bandwidth, mode=mode)

    if not use_cache:
        return build()
    kernel = cache.get_recent(DIFFUSION_KERNELS, (bandwidth, mode), build)
    return cast("NDArray[np.float64]", kernel.copy())


def _smooth(
    cache: AnalysisCache,
    binning: Binning,
    field: NDArray[np.float64],
    bandwidth: float,
    *,
    mode: KernelMode = "average",
) -> NDArray[np.float64]:
    """Validate one bin-valued field and smooth it matrix-free."""
    n_bins = binning.bins.n_bins
    field = np.asarray(field, dtype=np.float64)

    # Check field dimensionality
    if field.ndim != 1:
        raise ValueError(
            f"Field must be 1-D array (got {field.ndim}-D array). "
            f"Expected shape (n_bins,) = ({n_bins},), got shape {field.shape}."
        )

    # Check field shape matches n_bins
    if field.shape[0] != n_bins:
        raise ValueError(
            f"Field shape {field.shape} must match n_bins={n_bins}. "
            f"Expected shape (n_bins,) = ({n_bins},), got ({field.shape[0]},)."
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

    # Apply the diffusion smoothing via the matrix-free apply-path. This is a
    # PURE LINEAR operator -- no clip / renorm -- so smoothing a SIGNED field
    # is preserved. It reproduces the dense kernel to within the truncation
    # tolerance; positivity, where needed, is the caller's job.
    return _diffuse(cache, binning, field, bandwidth, mode=mode)


def _diffuse(
    cache: AnalysisCache,
    binning: Binning,
    fields: Any,
    bandwidth: float,
    *,
    mode: KernelMode = "density",
    backend: Literal["numpy", "jax"] = "numpy",
) -> NDArray[np.float64]:
    """Validate a field batch and apply the cached truncated heat operator."""
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
    n_bins = binning.bins.n_bins
    if shape[0] != n_bins:
        raise ValueError(
            f"fields leading dimension {shape[0]} must match n_bins={n_bins}."
        )
    was_1d = ndim == 1
    fields_2d = fields.reshape(-1, 1) if was_1d else fields

    W, volumes, n_components, _labels = diffusion_geometry(cache, binning)
    # The eigenbasis grows in place while it is applied.
    with cache.exclusive(DIFFUSION_EIGENBASIS):
        result = diffusion_apply(
            diffusion_eigenbasis(cache),
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


def _mrf_basis(
    cache: AnalysisCache,
    binning: Binning,
    occupancy: NDArray[np.float64],
    *,
    rank: int | None,
) -> MRFBasis:
    """Reduced-rank MRF smoothness basis covering the occupied support."""
    _validate_occupancy(occupancy)  # fail fast, before the eigensolve
    _W, volumes, n_components, labels = diffusion_geometry(cache, binning)
    S = _symmetric_conjugate(_W, volumes)  # cheap; the eigensolve is the cost
    # The eigenbasis grows in place until it covers the occupied support.
    with cache.exclusive(MRF_EIGENBASIS):
        holder = mrf_eigenbasis(cache, binning.bins.n_bins)
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


def _compute_gaussian_kernel(
    cache: AnalysisCache,
    bin_centers: NDArray[np.float64],
    coordinates: CartesianSystem | EgocentricPolarSystem,
    bandwidth: float,
) -> NDArray[np.float64]:
    """Build/cache Gaussian weights privately and export an immutable header."""
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f"bandwidth must be finite and positive (got {bandwidth}).")

    def build() -> NDArray[np.float64]:
        n_bins = bin_centers.shape[0]
        # Warn (never raise) for large environments. The dense weight matrix
        # exp(-d^2/2sigma^2) costs n_bins**2 * 8 bytes of float64 memory; we
        # estimate that and proceed. Mirrors the diffusion-kernel warning and
        # shares its threshold. It fires once per build, not per call.
        if n_bins > _constants.LARGE_KERNEL_THRESHOLD:
            estimated_gb = n_bins * n_bins * 8 / 1e9
            warnings.warn(
                f"Computing a dense Gaussian-KDE kernel for {n_bins} bins. The "
                f"weight matrix exp(-d^2/2sigma^2) is dense by construction (every "
                f"entry > 0), so it requires an {n_bins} x {n_bins} float64 matrix "
                f"(~{estimated_gb:.1f} GB) -- O(n^2) memory. Proceeding anyway; this "
                f"may be slow and memory-intensive. To avoid a dense kernel, use "
                f"method='diffusion_kde' (matrix-free, boundary-aware, "
                f"O(n * rank)) or increase bin_size (fewer bins).",
                UserWarning,
                stacklevel=_external_stacklevel(),
            )
        return _immutable_array(_gaussian_kernel(bin_centers, coordinates, bandwidth))

    return _array_header(cache.get_recent(GAUSSIAN_KERNELS, float(bandwidth), build))
