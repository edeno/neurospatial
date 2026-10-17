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
from neurospatial._exceptions import _format_error
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
from neurospatial.environment._publication import _freeze_diffusion_geometry
from neurospatial.geometry import (
    Binning,
    CartesianSystem,
    DiffusionGeometry,
    EgocentricPolarSystem,
    FaceGeometry,
    combined_diffusion,
    locate,
    require_grid_context,
)
from neurospatial.ops import (
    HEAT_KERNEL_RANK_TOL,
    RESOLUTION_MARGIN,
    DiffusionEigenbasis,
    MRFEigenbasis,
    diffusion_apply,
    ensure_global_modes,
    gaussian_kernel,
    heat_kernel_from_W,
    live_components,
    masked_diffusion_average,
    null_mode_indices,
    resolve_basis,
    select_live_basis,
    symmetric_conjugate,
    target_live_rank,
    validate_occupancy,
)
from neurospatial.ops.diffusion import MRFBasis

if TYPE_CHECKING:
    from neurospatial.geometry import Arena
    from neurospatial.ops import KernelMode

# Smoothing a field with NaN (missing) bins leaves a bin NaN when its smoothed
# support -- the share of its smoothing weight on observed bins -- is at most
# this fraction of the largest support in its connected part. The truncated
# operator reproduces the heat kernel to HEAT_KERNEL_RANK_TOL (1e-6) of its
# largest output, so above a support 1e3 times that the averaged value is
# accurate to about 0.1% of the field's largest magnitude in the component
# (not of each value: a 1000:1 field can be off by ~1% at its low end); below
# it the ratio is truncation noise.
MISSING_SUPPORT_FLOOR = RESOLUTION_MARGIN * HEAT_KERNEL_RANK_TOL


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

    # NaN marks a missing bin value; infinity has no interpolation meaning.
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

    # A NaN coordinate is a missing sample and an infinite one lies outside:
    # both read NaN, and only finite points are interpolated.
    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        values = np.full(points.shape[0], np.nan)
        if finite.any():
            values[finite] = _interpolate_points(
                field, points[finite], binning, arena, cache, mode
            )
        return values
    return _interpolate_points(field, points, binning, arena, cache, mode)


def _interpolate_points(
    field: NDArray[np.float64],
    points: NDArray[np.float64],
    binning: Binning,
    arena: Arena,
    cache: AnalysisCache,
    mode: Literal["nearest", "linear"],
) -> NDArray[np.float64]:
    """Interpolate at finite points; points outside the environment read NaN."""
    if mode == "nearest":
        values = _interpolate_nearest(cache, binning, field, points)
    else:  # mode == "linear"
        values = _interpolate_linear(binning, field, points)
    owners = locate(binning, arena, points, cache)
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
            # interpolate is Cartesian-only, so the centers are positions.
            coordinates=CartesianSystem(),
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
    A missing corner (an inactive cell of a holed/masked grid, or a bin whose
    field value is NaN) contributes NaN only when its interpolation weight is
    positive. Other bin centers therefore retain their known values, while
    points requiring a missing corner's value and out-of-bounds points return
    NaN.

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
    # NaN (missing) field values are handled the same way as inactive cells.
    n_full = int(np.prod(grid_shape))
    active_mask = grid.active_mask
    missing_values = np.isnan(field)
    has_missing = field.shape[0] != n_full or bool(missing_values.any())
    if not has_missing:
        # Fully-active grid: field already covers every cell.
        field_grid = field.reshape(grid_shape)
    else:
        active_cells = np.flatnonzero(active_mask.ravel())
        full_field = np.zeros(n_full, dtype=np.float64)
        full_field[active_cells] = np.where(missing_values, 0.0, field)
        field_grid = full_field.reshape(grid_shape)
        missing_cells = ~active_mask.ravel()
        missing_cells[active_cells[missing_values]] = True

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
    if has_missing:
        missing_weight = RegularGridInterpolator(
            grid_points,
            missing_cells.reshape(grid_shape).astype(np.float64),
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


def diffusion_eigenbasis(cache: AnalysisCache) -> DiffusionEigenbasis:
    """The current truncated-eigenbasis holder for :func:`_diffuse`, empty at first."""
    return cache.get(DIFFUSION_EIGENBASIS, DiffusionEigenbasis)


def mrf_eigenbasis(cache: AnalysisCache, n_bins: int) -> MRFEigenbasis:
    """The current global eigenbasis holder for the MRF penalty basis."""
    return cache.get(MRF_EIGENBASIS, lambda: MRFEigenbasis.empty(n_bins))


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


def _field_batch(
    fields: Any, n_bins: int, operation: str, argument: str, bandwidth: float
) -> Any:
    """Check a field or batch of fields indexed by bin first.

    Returns the input as an array of shape ``(n_bins,)`` or
    ``(n_bins, n_fields)``; a rate result or a ``(n_fields, n_bins)`` batch is
    refused with the corrected call.
    """
    if hasattr(fields, "firing_rate") or hasattr(fields, "firing_rates"):
        call = (
            f"env.smooth(result.firing_rates.T, bandwidth={bandwidth!r}).T, "
            "which keeps each unit's rates in a column"
            if hasattr(fields, "firing_rates")
            else f"env.smooth(result.firing_rate, bandwidth={bandwidth!r})"
        )
        raise TypeError(
            _format_error(
                f"{operation} takes an array of values per bin, got a "
                f"{type(fields).__name__}.",
                why="Why: it returns an array, so it smooths the rates "
                "themselves; a rate result also holds occupancy and settings "
                "that a smoothed copy would no longer match.",
                fix=f"pass the rates: {call}; smooth's default mode='average' "
                "is the one for rates. To smooth while estimating the rates, "
                "pass bandwidth= to compute_spatial_rate instead",
            )
        )
    ndim = int(np.ndim(fields))
    shape = np.shape(fields)
    if ndim not in (1, 2):
        raise ValueError(
            _format_error(
                f"{operation} takes fields of shape (n_bins,) or "
                f"(n_bins, n_fields), got a {ndim}-D array of shape {shape}.",
                why="Why: each row holds one bin's value in every field.",
                fix="reshape the values to (n_bins, n_fields), one column per field",
            )
        )
    if shape[0] != n_bins:
        transposed = ndim == 2 and shape[1] == n_bins
        raise ValueError(
            _format_error(
                f"{operation} got fields of shape {shape}, but this environment "
                f"has {n_bins} bins, which must be the first axis.",
                why="Why: fields are indexed by bin first, one column per field; "
                + (
                    "an array of shape (n_fields, n_bins), as population rates "
                    "are stored, has its bins on the second axis."
                    if transposed
                    else "the values must come from an environment with the same bins."
                ),
                fix=(
                    f"pass {argument}.T with the same options and transpose the "
                    f"result back, such as env.{operation[:-2]}({argument}.T, "
                    f"bandwidth={bandwidth!r}).T"
                    if transposed
                    else "compute the field on this environment, or map it here "
                    "with neurospatial.ops.resample_field"
                ),
            )
        )
    return fields


def _smooth(
    cache: AnalysisCache,
    binning: Binning,
    field: NDArray[np.float64],
    bandwidth: float,
    *,
    mode: KernelMode = "average",
) -> NDArray[np.float64]:
    """Validate bin-valued fields and smooth them matrix-free."""
    n_bins = binning.bins.n_bins
    field = _field_batch(field, n_bins, "smooth()", "field", bandwidth)
    field = np.asarray(field, dtype=np.float64)

    # NaN marks a missing bin and is handled by _diffuse; infinity is invalid.
    if np.any(np.isinf(field)):
        raise ValueError(
            "Field contains infinite values. "
            f"Found {np.sum(np.isinf(field))} infinite values out of {field.size} "
            "field values. Infinite values are not supported in smoothing "
            "operations."
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

    # Apply the diffusion smoothing via the matrix-free apply-path. On a
    # NaN-free field this is a pure linear operator -- no clip / renorm -- so
    # smoothing a SIGNED field is preserved; a field with NaN is averaged over
    # its observed bins (mode="average" only). It reproduces the dense kernel to
    # within the truncation tolerance; positivity, where needed, is the
    # caller's job.
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
    n_bins = binning.bins.n_bins
    fields = _field_batch(fields, n_bins, "diffuse()", "fields", bandwidth)
    if backend == "numpy":
        fields = np.asarray(fields, dtype=np.float64)
    was_1d = np.ndim(fields) == 1
    fields_2d = fields.reshape(-1, 1) if was_1d else fields

    W, volumes, n_components, labels = diffusion_geometry(cache, binning)
    # Resolving a new bandwidth replaces the cached holder under its lock;
    # the apply itself runs on the resolved, read-only basis without it.
    basis = cache.update(
        DIFFUSION_EIGENBASIS,
        DiffusionEigenbasis,
        lambda holder: resolve_basis(
            holder, W, volumes, n_components, float(bandwidth)
        ),
    )

    def apply(columns: Any) -> Any:
        return diffusion_apply(
            basis, volumes, columns, float(bandwidth), mode, backend=backend
        )

    # NaN marks a missing bin. A NaN-free batch takes the plain linear apply
    # unchanged; a traced JAX input cannot be inspected, so it is not.
    missing = _missing_bins(fields_2d, backend)
    if missing is None or not missing.any():
        result = apply(fields_2d)
    elif mode != "average":
        raise ValueError(
            _format_error(
                f"Cannot smooth a field with {int(missing.sum())} NaN (missing) "
                f"bins in mode={mode!r}.",
                why=(
                    "Why: missing bins have no defensible mass under a conserving "
                    "operator, so their counts or density cannot be smoothed "
                    "around them."
                ),
                fix=(
                    "fill the missing bins explicitly (for example "
                    "np.nan_to_num(field) if they are truly zero), or use "
                    "mode='average' if the field is intensive, such as a rate map"
                ),
            )
        )
    elif backend == "jax":
        raise ValueError(
            _format_error(
                f"diffuse(backend='jax') got {int(missing.sum())} NaN (missing) bins.",
                why=(
                    "Why: the JAX apply is a plain linear operator, so NaN would "
                    "spread through its whole connected part."
                ),
                fix=(
                    "use backend='numpy', which averages over the observed bins, "
                    "or fill the missing bins first"
                ),
            )
        )
    else:
        # Holed fields are averaged over their observed bins. NaN-free fields
        # in the same batch go through one plain apply of the original shape,
        # so they stay identical; that apply is skipped when every field has
        # holes, since its holed columns would be overwritten.
        holed = np.flatnonzero(missing.any(axis=0))
        if holed.size == fields_2d.shape[1]:
            result = np.empty_like(fields_2d)
        else:
            result = apply(np.where(missing, 0.0, fields_2d))
        result[:, holed] = masked_diffusion_average(
            apply,
            fields_2d[:, holed],
            ~missing[:, holed],
            labels,
            n_components,
            floor=MISSING_SUPPORT_FLOOR,
        )
    # numpy backend -> ndarray; jax backend -> jax.Array (documented above).
    # The NDArray annotation is the numpy-path type; cast covers both.
    return cast("NDArray[np.float64]", result[:, 0] if was_1d else result)


def _missing_bins(fields: Any, backend: str) -> NDArray[np.bool_] | None:
    """NaN bins of a concrete field batch; None for a NaN-free or traced JAX one."""
    if backend == "numpy":
        return np.asarray(np.isnan(fields), dtype=np.bool_)
    import jax
    import jax.numpy as jnp

    # Check on the device first, so a NaN-free JAX batch is not copied to host.
    if isinstance(fields, jax.core.Tracer) or not bool(jnp.isnan(fields).any()):
        return None
    return np.asarray(np.isnan(np.asarray(fields)), dtype=np.bool_)


def _mrf_basis(
    cache: AnalysisCache,
    binning: Binning,
    occupancy: NDArray[np.float64],
    *,
    rank: int | None,
) -> MRFBasis:
    """Reduced-rank MRF smoothness basis covering the occupied support."""
    validate_occupancy(occupancy)  # fail fast, before the eigensolve
    _W, volumes, n_components, labels = diffusion_geometry(cache, binning)
    S = symmetric_conjugate(_W, volumes)  # cheap; the eigensolve is the cost
    live = live_components(labels, occupancy)
    n_live_components = int(live.size)
    r_eff = target_live_rank(labels, occupancy, rank)
    n_fill = r_eff - n_live_components

    def grow(
        holder: MRFEigenbasis,
    ) -> tuple[
        MRFEigenbasis,
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]],
    ]:
        """Grow the cached modes until they cover the occupied support."""
        # ``_symmetric_eigenbasis`` raises on rank < n_components (total geometry
        # components), but ``r_eff`` counts only LIVE components -- so 1 live +
        # several dead would start ``G`` too small and raise on the first solve.
        # Floor ``G`` at ``n_components`` to keep the first solve legal.
        G = max(holder.G, r_eff, n_components)
        while True:  # grow until every live null is present AND enough fills
            holder, modes = ensure_global_modes(holder, S, labels, G)
            Q, _, mode_comp = modes
            # Verify the structural null the selector requires is actually present
            # for every live component (its numerical null can sort AFTER one of
            # its positive modes when both are near machine-epsilon), and count
            # the non-null fill-eligible live modes -- using the same overlap
            # criterion select_live_basis uses, so the two never disagree.
            is_null, missing = null_mode_indices(Q, mode_comp, volumes, labels, live)
            n_fill_available = int((np.isin(mode_comp, live) & ~is_null).sum())
            covered = missing.size == 0
            if (covered and n_fill_available >= n_fill) or Q.shape[1] >= S.shape[0]:
                return holder, modes
            G = min(2 * Q.shape[1], S.shape[0])  # double, capped at full rank

    # Growth replaces the cached modes under their lock, so concurrent
    # requests share one eigensolve.
    Q, Lam, mode_comp = cache.update(
        MRF_EIGENBASIS, lambda: MRFEigenbasis.empty(binning.bins.n_bins), grow
    )
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
        kernel = gaussian_kernel(bin_centers, coordinates, bandwidth)
        kernel.setflags(write=False)
        return kernel

    kernel: NDArray[np.float64] = cache.get_recent(
        GAUSSIAN_KERNELS, float(bandwidth), build
    ).view()
    return kernel
