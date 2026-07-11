"""Shared smoothing implementations for encoding rate map computation.

This module provides smoothing functions that are used by the encoding
compute functions (compute_spatial_rate, compute_directional_rate, etc.).

The functions in this module operate on dense arrays:
- Single neuron: spike_counts (n_bins,), occupancy (n_bins,)
- Batch: spike_counts (n_neurons, n_bins), occupancy (n_bins,)

Three smoothing methods are supported:
- **diffusion_kde**: Graph-based boundary-aware KDE (recommended)
- **gaussian_kde**: Standard Euclidean KDE (ignores boundaries)
- **binned**: Bin-then-smooth order

The key difference between methods is the order of operations:
- diffusion_kde/gaussian_kde: Smooth counts → Smooth occupancy → Normalize
- binned: Normalize → Smooth result

**Backend Support**:

The smoothing functions support both NumPy and JAX backends via the ``backend``
parameter. When ``backend="jax"``, the ``diffusion_kde`` smoothing runs entirely
in JAX via ``env.diffuse(backend="jax")`` (the cached NumPy eigenbasis is cast to
``jnp``, so the eigenbasis *build* stays on CPU but the *apply* runs on device),
and the rate computation uses JAX array operations. This keeps ``jit`` / ``grad``
/ GPU working through the smoothing. (``binned`` keeps a NumPy round-trip for its
masked average.)

**Performance**:

``diffusion_kde`` and ``binned`` smooth **matrix-free** via ``env.diffuse``: a
cached, bandwidth-aware truncated eigenbasis of the finite-volume operator is
applied without ever materializing the dense ``(n_bins, n_bins)`` heat kernel.
Time and memory scale with ``n_bins × rank`` (``rank ~ measure(domain)/σ^d``),
not ``n_bins²``, so these methods scale to large/fine grids. The eigenbasis is
geometry-only, so it is built once per environment and reused across every
bandwidth, mode, and neuron. (A near-full-rank request — a light bandwidth on a
large grid — genuinely needs most modes, so ``env.diffuse`` falls back to a
transient dense basis there, with the same large-matrix ``UserWarning`` as
``compute_kernel``.)

``gaussian_kde`` still builds a dense ``(n_bins, n_bins)`` Gaussian weight matrix
(Euclidean, boundary-ignoring), costing O(n_bins²) memory and time; it is cached
per ``(environment, bandwidth)`` (see ``_get_gaussian_kernel``) and reused across
neurons, and warns above ``_LARGE_KERNEL_THRESHOLD`` bins. For very large
environments prefer ``diffusion_kde`` (matrix-free and boundary-aware).

References
----------
.. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
       An information-theoretic approach to deciphering the hippocampal code.
.. [2] Barry, C., et al. (2006). The boundary vector cell model of
       place cell firing and spatial memory. Reviews in the Neurosciences.
"""

from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Single shared high-bin warn threshold. Reuse the diffusion-kernel warn
# threshold so both dense O(n_bins^2) smoothing paths (diffusion + gaussian)
# warn at the same bin count -- one threshold, not a divergent copy. Neither
# path imposes a hard limit; above the threshold each warns (with a GB
# estimate) and proceeds.
from neurospatial.ops.diffusion import (
    _DIFFUSE_DENOM_EPS,
    component_support_mask,
    diffusion_component_labels,
)
from neurospatial.ops.smoothing import _LARGE_KERNEL_THRESHOLD

if TYPE_CHECKING:
    from neurospatial.environment._protocols import EnvironmentProtocol
    from neurospatial.environment.core import _BaseEnvironment

__all__ = [
    "smooth_rate_map",
    "smooth_rate_maps_batch",
]


# Cache for the dense Gaussian-KDE kernel. Keyed by (id(env), bandwidth);
# value is the (n_bins, n_bins) weight matrix, its bin-count, and a weakref
# to the owning environment. The weakref is the real id-reuse guard: ``id()``
# is only unique among *live* objects, so once an Environment is GC'd a freshly
# built one can reuse its address. A different env reusing the id (or the same
# id after GC) would otherwise return a stale kernel that does not match the new
# env's geometry (e.g. a circular vs. open angular axis with identical n_bins).
# Validating ``ref() is env`` on lookup turns any such reuse into a cache miss.
_GAUSSIAN_KERNEL_CACHE: dict[
    tuple[int, float], tuple[NDArray[np.float64], int, weakref.ref[Any]]
] = {}
_GAUSSIAN_KERNEL_CACHE_MAX = 32


def _get_gaussian_kernel(
    env: _BaseEnvironment, bandwidth: float
) -> NDArray[np.float64]:
    """Return the dense Gaussian-KDE weight matrix for ``env`` at ``bandwidth``.

    The matrix is ``(n_bins, n_bins)`` and was previously rebuilt at every
    call site via ``np.exp(-pairwise_dist_sq / (2*sigma^2))``. For
    ``n_bins`` of a few thousand that materialization plus exp is
    measurable; cache the result keyed on ``(id(env), bandwidth)`` and
    verify ``n_bins`` to defend against id reuse after GC.

    The weight matrix is dense ``(n_bins, n_bins)`` -- O(n_bins**2) memory --
    so when ``n_bins`` exceeds ``_LARGE_KERNEL_THRESHOLD`` a loud memory
    ``UserWarning`` (with a GB estimate) is emitted and the matrix is built
    anyway. There is no hard limit. The warning fires once per kernel *build*
    (a cache hit returns before warning), mirroring the diffusion-kernel warning
    in ``ops/smoothing.py``.

    Parameters
    ----------
    env : Environment
        The spatial environment whose bin centers define the kernel geometry.
    bandwidth : float
        Gaussian ``sigma`` in environment units.
    """
    key = (id(env), float(bandwidth))
    cached = _GAUSSIAN_KERNEL_CACHE.get(key)
    bin_centers = env.bin_centers
    n_bins = bin_centers.shape[0]

    # Require the cached weakref to still resolve to *this* exact env. A dead
    # weakref (env GC'd) or one resolving to a different object (id reused by a
    # new env) means the entry belongs to a now-gone environment -- treat as a
    # miss and recompute rather than returning a geometrically wrong kernel.
    if cached is not None and cached[1] == n_bins and cached[2]() is env:
        return cached[0]

    # Warn (never raise) for large environments. The dense weight matrix
    # exp(-d^2/2sigma^2) costs n_bins**2 * 8 bytes of float64 memory; we
    # estimate that and proceed. Mirrors the diffusion-kernel warning and shares
    # its threshold. Placed after the cache-hit check so it fires once per
    # build, not per call.
    if n_bins > _LARGE_KERNEL_THRESHOLD:
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
            stacklevel=2,
        )

    two_sigma_sq = 2.0 * bandwidth**2

    if getattr(env, "_POLAR", False):
        # Egocentric polar env: bin_centers[:, 0] is distance (length units),
        # bin_centers[:, 1] is angle (radians). A naive Euclidean norm on
        # these columns collapses cm and radians into one scalar. Instead use
        # the physical polar distance between bin centers:
        #     d² = Δr² + (r̄ · Δθ)²
        # with r̄ the mean radius of the two bins. ``bandwidth`` is then a
        # single physical length (e.g. cm), consistent with the corrected
        # connectivity edge geometry.
        r = bin_centers[:, 0]
        theta = bin_centers[:, 1]
        d_r = r[:, None] - r[None, :]
        r_mean = 0.5 * (r[:, None] + r[None, :])
        d_theta_raw = theta[:, None] - theta[None, :]
        # Wrap the angular difference into [-pi, pi] ONLY when the angular axis
        # is circular, so bins straddling the -pi/+pi seam are treated as
        # adjacent (Delta theta ~ 0) rather than ~2*pi apart. Without this wrap
        # the seam gets a vanishing Gaussian weight, a hard artifact for a
        # full-circle egocentric-polar gaussian_kde. When the angular axis is
        # OPEN (circular_angle=False -- no seam edges in the graph), bins at -pi
        # and +pi are genuinely far apart, so wrapping would leak smoothing
        # across a boundary the caller deliberately left open; use the raw
        # angular difference instead. Circularity is derived from the
        # connectivity graph (presence of seam edges), so it stays consistent
        # with the graph and survives serialization. The diffusion_kde path is
        # unaffected: it smooths over the environment graph directly.
        if getattr(env, "_angular_is_circular", False):
            d_theta = (d_theta_raw + np.pi) % (2.0 * np.pi) - np.pi
        else:
            d_theta = d_theta_raw
        arc = r_mean * d_theta
        dist_sq = d_r**2 + arc**2
    else:
        bin_sq_norm = np.sum(bin_centers**2, axis=1, keepdims=True)
        dist_sq = bin_sq_norm + bin_sq_norm.T - 2 * (bin_centers @ bin_centers.T)
    dist_sq = np.maximum(dist_sq, 0)
    kernel: NDArray[np.float64] = np.exp(-dist_sq / two_sigma_sq).astype(
        np.float64, copy=False
    )

    try:
        env_ref: weakref.ref[Any] = weakref.ref(env)
    except TypeError:
        # Not weakref-able (e.g. __slots__ without __weakref__): skip caching
        # rather than risk an id-reuse collision we cannot detect. Correctness
        # over the (optional) speed-up.
        return kernel

    if len(_GAUSSIAN_KERNEL_CACHE) >= _GAUSSIAN_KERNEL_CACHE_MAX:
        # Evict an arbitrary oldest-ish entry. dict insertion order makes
        # iter(...) return the oldest key first.
        _GAUSSIAN_KERNEL_CACHE.pop(next(iter(_GAUSSIAN_KERNEL_CACHE)))
    _GAUSSIAN_KERNEL_CACHE[key] = (kernel, n_bins, env_ref)
    return kernel


def smooth_rate_map(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax"] = "numpy",
) -> ArrayLike:
    """Compute smoothed firing rate map from spike counts and occupancy.

    This function applies smoothing to spike counts and occupancy to compute
    a firing rate map. The smoothing method determines the order of operations
    and the type of smoothing applied.

    Parameters
    ----------
    env : Environment
        The spatial environment. Used for graph structure and kernel computation.
    spike_counts : ndarray, shape (n_bins,)
        Number of spikes in each spatial bin.
    occupancy : ndarray, shape (n_bins,)
        Time spent in each spatial bin (seconds).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph. Order: smooth → normalize.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers. Ignores boundaries (mass
          can "bleed through" walls). Order: smooth → normalize.
        - **binned**: Bin-then-smooth method. Computes raw rate first, then smooths.
          Order: normalize → smooth. Can introduce discretization artifacts.

    bandwidth : float, default=5.0
        Smoothing bandwidth in the same units as bin_size. Larger values
        produce more smoothing. For diffusion_kde, this is the kernel
        bandwidth σ. For gaussian_kde, this is the Gaussian σ. For binned,
        this is passed to env.smooth().
    min_occupancy : float, default=0.0
        Minimum occupancy in **seconds** for a bin to be kept; bins whose *raw*
        per-bin occupancy is below the threshold are set to NaN. Applied
        identically for every method (``diffusion_kde``, ``gaussian_kde``,
        ``binned``), so ``min_occupancy`` means the same thing across methods and
        the masked bins are exactly ``occupancy < min_occupancy``. The default
        ``0.0`` applies no occupancy masking. If the threshold would mask every
        occupied bin -- usually a units/scale mistake, since it is seconds -- a
        ``UserWarning`` is emitted and the rate map is empty.
    backend : {"numpy", "jax"}, default="numpy"
        Computation backend. When "jax", the ``diffusion_kde`` smoothing runs
        in JAX via ``env.diffuse(backend="jax")`` (the cached eigenbasis is cast
        to ``jnp``), so ``jit`` / ``grad`` / GPU work through it.

    Returns
    -------
    ArrayLike, shape (n_bins,)
        Smoothed firing rate in Hz (spikes/second). Bins with zero or
        low occupancy are NaN. Returns ndarray for numpy backend, jax.Array
        for jax backend.

    Raises
    ------
    ValueError
        If method is not one of the valid options.
        If bandwidth is negative.
        If spike_counts and occupancy have different shapes.
        If spike_counts shape doesn't match env.n_bins.

    Notes
    -----
    **Method Comparison**:

    +--------------+----------------+------------------------+--------------+
    | Method       | Boundaries     | Complexity             | Artifacts    |
    +==============+================+========================+==============+
    | diffusion_kde| Respects       | O(n_bins·rank) / neuron| None         |
    +--------------+----------------+------------------------+--------------+
    | gaussian_kde | Ignores        | O(n_bins²) per neuron  | Wall bleed   |
    +--------------+----------------+------------------------+--------------+
    | binned       | Respects*      | O(n_bins·rank) / neuron| Discretization|
    +--------------+----------------+------------------------+--------------+

    *binned computes the rate first, then smooths it (bin-then-smooth); the
    other methods smooth the counts, then normalize (smooth-then-normalize).

    ``diffusion_kde`` and ``binned`` smooth **matrix-free** via ``env.diffuse``
    (a cached bandwidth-aware truncated eigenbasis), so they never build the
    dense ``(n_bins, n_bins)`` heat kernel: time and memory scale with
    ``n_bins × rank`` (``rank ~ measure(domain)/σ^d``), not ``n_bins²``. The
    eigenbasis is geometry-only, so it is built once per environment and reused
    across bandwidths, modes, and neurons. ``gaussian_kde`` still builds a dense
    ``(n_bins, n_bins)`` Gaussian matrix (O(n_bins²), warns above
    ``_LARGE_KERNEL_THRESHOLD`` bins).

    **Performance recommendation**: For most analyses use ``diffusion_kde``
    (default) -- it is boundary-aware and matrix-free, so it scales to large/fine
    grids where the dense ``gaussian_kde`` kernel would not fit in RAM.

    **Backend behavior**: When ``backend="jax"``, ``diffusion_kde`` runs the
    smoothing **in JAX** via ``env.diffuse(backend="jax")`` (the cached eigenbasis
    is cast to ``jnp``), enabling GPU acceleration and ``jit`` / ``grad`` through
    the smoothing. ``binned`` keeps a NumPy round-trip for its masked average.

    **JAX backend limitation with binned method**: When using ``backend="jax"``
    with ``method="binned"``, the masked-average smoothing step
    (``_binned_gate``) runs on NumPy, so it requires a round-trip out of JAX.
    This may be slower than pure NumPy for this method. For optimal JAX
    performance, use ``diffusion_kde`` (which runs the smoothing in JAX via
    ``env.diffuse(backend="jax")``) or ``gaussian_kde``.

    **Algorithm Details**:

    For diffusion_kde and gaussian_kde (correct KDE order):
    1. Smooth spike counts using kernel
    2. Smooth occupancy using kernel
    3. Compute rate: smoothed_spikes / smoothed_occupancy

    For binned (bin-then-smooth order):
    1. Compute raw rate: spike_counts / occupancy
    2. Smooth the rate map

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding._smoothing import smooth_rate_map

    >>> # Create environment
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.random((1000, 2)) * 100
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Simulate spike counts and occupancy
    >>> spike_counts = rng.poisson(5, env.n_bins).astype(float)
    >>> occupancy = np.ones(env.n_bins) * 1.0  # 1 second per bin

    >>> # Compute smoothed rate map
    >>> rate_map = smooth_rate_map(
    ...     env, spike_counts, occupancy, method="diffusion_kde", bandwidth=10.0
    ... )
    >>> rate_map.shape == (env.n_bins,)
    True

    See Also
    --------
    smooth_rate_maps_batch : Batch version for multiple neurons
    Environment.compute_kernel : Compute diffusion kernel
    Environment.smooth : Apply smoothing to a field
    """
    # Input validation
    _validate_smoothing_inputs(env, spike_counts, occupancy, method, bandwidth)

    # Dispatch to JAX or NumPy implementation.
    if backend == "jax":
        return _smooth_rate_map_jax(  # type: ignore[no-any-return]
            env, spike_counts, occupancy, method, bandwidth, min_occupancy
        )

    # NumPy path only: warn if the seconds threshold masks every occupied bin.
    # (Skipped on the jax path, where occupancy may be a traced array under
    # jit/grad and np.asarray would raise TracerArrayConversionError.)
    _warn_if_fully_masked(occupancy, min_occupancy)

    # Dispatch to appropriate NumPy method
    match method:
        case "diffusion_kde":
            return _diffusion_kde(
                env, spike_counts, occupancy, bandwidth, min_occupancy
            )
        case "gaussian_kde":
            return _gaussian_kde(env, spike_counts, occupancy, bandwidth, min_occupancy)
        case "binned":
            return _binned(env, spike_counts, occupancy, bandwidth, min_occupancy)
        case _:
            # This should never be reached due to validation
            raise ValueError(f"Unknown method: {method}")


def smooth_rate_maps_batch(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: Literal["numpy", "jax"] = "numpy",
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> ArrayLike:
    """Compute smoothed firing rate maps for multiple neurons.

    Batch version of smooth_rate_map that efficiently processes multiple
    neurons using vectorized matrix operations (BLAS Level 3).

    Parameters
    ----------
    env : Environment
        The spatial environment.
    spike_counts : ndarray, shape (n_neurons, n_bins)
        Number of spikes in each spatial bin for each neuron.
    occupancy : ndarray, shape (n_bins,)
        Shared time spent in each spatial bin (seconds).
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Smoothing method. See smooth_rate_map for details.
    bandwidth : float, default=5.0
        Smoothing bandwidth.
    min_occupancy : float, default=0.0
        Minimum occupancy in **seconds** for a bin to be kept; bins whose *raw*
        per-bin occupancy is below the threshold are set to NaN, identically for
        every method (masked bins are exactly ``occupancy < min_occupancy``).
        The default ``0.0`` applies no occupancy masking. See ``smooth_rate_map``.
    backend : {"numpy", "jax"}, default="numpy"
        Computation backend. When "jax", uses JAX array operations for the
        core rate computation (smoothing/division). See smooth_rate_map for
        details on backend behavior.
    dtype : {np.float32, np.float64}, default=np.float64
        Storage dtype of the returned rate-map array. The matmul/division is
        always done in float64 for accuracy; only the final returned array is
        cast to ``dtype``. ``np.float32`` halves the ``(n_neurons, n_bins)``
        storage. Default ``np.float64`` leaves every existing caller
        byte-for-byte unchanged. Honored on the NumPy path; on the JAX path the
        cast is applied to the returned array as well.

    Returns
    -------
    ArrayLike, shape (n_neurons, n_bins)
        Smoothed firing rates in Hz for each neuron. Returns ndarray for
        numpy backend, jax.Array for jax backend.

    Raises
    ------
    ValueError
        If spike_counts is not 2D.
        If spike_counts.shape[1] != occupancy.shape[0].
        If spike_counts.shape[1] != env.n_bins.
    """
    # Validate batch-specific requirements. Use np.shape/np.ndim (which read
    # .shape/.ndim) rather than np.asarray, so a traced JAX array passes through
    # untouched: coercing it here would raise TracerArrayConversionError and
    # break jit/grad on the backend="jax" path. The NumPy impls re-coerce.
    sc_shape = np.shape(spike_counts)
    occ_shape = np.shape(occupancy)

    if len(sc_shape) != 2:
        raise ValueError(
            f"spike_counts must be 2D (n_neurons, n_bins), got shape {sc_shape}"
        )

    if sc_shape[1] != occ_shape[0]:
        raise ValueError(
            f"spike_counts has {sc_shape[1]} bins but occupancy has {occ_shape[0]} bins"
        )

    if sc_shape[1] != env.n_bins:
        raise ValueError(
            f"spike_counts has {sc_shape[1]} bins but "
            f"env has {env.n_bins} bins (n_bins mismatch)"
        )

    # Dispatch to JAX or NumPy implementation.
    if backend == "jax":
        return _smooth_rate_maps_batch_jax(  # type: ignore[no-any-return]
            env,
            spike_counts,
            occupancy,
            method,
            bandwidth,
            min_occupancy,
            dtype=dtype,
        )

    # NumPy path only: warn if the seconds threshold masks every occupied bin.
    # (Skipped on the jax path above, where occupancy may be a traced array.)
    _warn_if_fully_masked(occupancy, min_occupancy)

    # Dispatch to NumPy vectorized implementations
    if method == "diffusion_kde":
        return _diffusion_kde_batch(
            env, spike_counts, occupancy, bandwidth, min_occupancy, dtype=dtype
        )
    elif method == "gaussian_kde":
        return _gaussian_kde_batch(
            env, spike_counts, occupancy, bandwidth, min_occupancy, dtype=dtype
        )
    elif method == "binned":
        return _binned_batch(
            env, spike_counts, occupancy, bandwidth, min_occupancy, dtype=dtype
        )
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Private Implementation Functions
# =============================================================================


def _validate_smoothing_inputs(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    method: str,
    bandwidth: float,
) -> None:
    """Validate inputs for smoothing functions."""
    _validate_smoothing_parameters(method, bandwidth)

    # Read shapes via np.shape (not np.asarray), so a traced JAX array passes
    # through untouched -- coercing it here would raise TracerArrayConversionError
    # and break jit/grad on the backend="jax" path.
    sc_shape = np.shape(spike_counts)
    occ_shape = np.shape(occupancy)

    # Check shapes match
    if sc_shape != occ_shape:
        raise ValueError(
            f"spike_counts shape {sc_shape} does not match occupancy shape {occ_shape}"
        )

    # Check matches environment
    if sc_shape[0] != env.n_bins:
        raise ValueError(
            f"spike_counts has {sc_shape[0]} elements but "
            f"env has {env.n_bins} bins (n_bins mismatch)"
        )


def _validate_smoothing_parameters(method: str, bandwidth: float) -> None:
    """Validate smoothing method and bandwidth without requiring count arrays."""
    valid_methods = {"diffusion_kde", "gaussian_kde", "binned"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    # Validate bandwidth (binned allows 0). Bandwidth is in environment units
    # (e.g. cm); the tuning curves and rate maps inherit those units.
    if method == "binned":
        if bandwidth < 0:
            raise ValueError(
                "bandwidth must be non-negative (in environment units, e.g., cm), "
                f"got {bandwidth}"
            )
    else:
        if bandwidth <= 0:
            raise ValueError(
                "bandwidth must be positive (in environment units, e.g., cm), "
                f"got {bandwidth}"
            )


def _apply_min_occupancy_mask(
    firing_rate: NDArray[np.floating[Any]],
    occupancy: NDArray[np.float64],
    min_occupancy: float,
) -> NDArray[np.floating[Any]]:
    """Mask bins whose RAW occupancy (seconds) is below ``min_occupancy``.

    ``min_occupancy`` is a threshold in seconds, applied identically to every
    smoothing method against the *raw* per-bin occupancy (not any smoothed
    density). This makes ``min_occupancy`` mean the same thing everywhere and
    keeps the public contract that masked bins are exactly recoverable via
    ``result.occupancy < min_occupancy``. ``occupancy`` (shape ``(n_bins,)``)
    broadcasts over a leading neuron axis for batch rate maps.

    A non-positive ``min_occupancy`` is a no-op, so the default (0.0) leaves the
    smoothed rate untouched.
    """
    if min_occupancy <= 0.0:
        return firing_rate
    return np.where(occupancy >= min_occupancy, firing_rate, np.nan)


def _warn_if_fully_masked(occupancy: NDArray[np.float64], min_occupancy: float) -> None:
    """Warn when ``min_occupancy`` would mask every occupied bin.

    Masking all bins the animal actually visited yields an empty rate map
    (all-NaN, or all-``fill_value``) with no other signal -- almost always a
    sign that ``min_occupancy`` is too large or in the wrong units (it is
    seconds). This defense-in-depth guard turns that silent empty result into a
    loud, actionable warning.
    """
    if min_occupancy <= 0.0:
        return
    occupancy = np.asarray(occupancy, dtype=np.float64)
    occupied = occupancy > 0.0
    n_occupied = int(np.count_nonzero(occupied))
    if n_occupied == 0:
        return
    n_masked = int(np.count_nonzero(occupied & (occupancy < min_occupancy)))
    if n_masked == n_occupied:
        warnings.warn(
            f"min_occupancy={min_occupancy} s masks ALL {n_occupied} occupied "
            f"bins (the highest occupancy is {float(occupancy.max()):.4g} s), so "
            "the rate map will be empty (all-NaN, or all-fill_value). "
            "min_occupancy is a threshold in SECONDS -- this usually means it is "
            "too large for this session. Lower it (or use the default 0.0 for no "
            "occupancy masking).",
            UserWarning,
            stacklevel=3,
        )


def _diffusion_kde(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
) -> NDArray[np.float64]:
    """Apply diffusion KDE smoothing via the matrix-free apply-path.

    Algorithm (correct KDE order):
    1. Spread spike counts and occupancy with ``env.diffuse(mode="density")``
       (one matrix-free pass, no ``(n_bins, n_bins)`` kernel).
    2. Normalize: spike_density / occupancy_density.

    Notes
    -----
    The firing-rate denominator is the *smoothed* occupancy density, so a bin
    with zero raw occupancy but a well-defined denominator from neighboring
    occupancy still gets a finite rate here. The ``min_occupancy`` cut is then
    applied separately, on the *raw* occupancy in seconds (see
    :func:`_apply_min_occupancy_mask`), so it means the same thing across every
    smoothing method and matches the public
    ``result.occupancy < min_occupancy`` contract.

    ``env.diffuse`` is a pure linear operator (unlike the shipped clipped dense
    kernel), so under truncation the smoothed occupancy density can carry a
    tolerance-level negative lobe; the **magnitude gate** floors it
    (``max(occupancy_density, 0)``) before the ``> 0`` divide guard, and the
    output rate is clipped ``>= 0`` (decode nonnegativity, as the shipped
    clipped-kernel path guaranteed).
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Smooth counts and occupancy in one matrix-free density-mode pass.
    stacked = np.column_stack([spike_counts, occupancy])  # (n_bins, 2)
    smoothed = np.asarray(
        cast("EnvironmentProtocol", env).diffuse(stacked, bandwidth, mode="density")
    )
    spike_density = smoothed[:, 0]
    occupancy_density = smoothed[:, 1]

    # Magnitude gate: floor the (possibly tolerance-negative) denominator, then
    # divide only where it is strictly positive (avoids 0/0 and negative-lobe
    # blowups). This gate is purely numerical -- it is NOT the min_occupancy cut.
    occ_floor = np.maximum(occupancy_density, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            occ_floor > 0.0,
            spike_density / occupancy_density,
            np.nan,
        )

    # Clip the output rate >= 0 (np.clip preserves NaN), then apply the
    # raw-occupancy (seconds) min_occupancy mask, consistent across methods.
    firing_rate = np.clip(firing_rate, 0.0, None)
    firing_rate = _apply_min_occupancy_mask(firing_rate, occupancy, min_occupancy)
    return firing_rate.astype(np.float64)


def _gaussian_kde(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
) -> NDArray[np.float64]:
    """Apply Gaussian KDE smoothing (Euclidean distance).

    Algorithm:
    1. For each bin, compute Gaussian-weighted spike density from all bins
    2. For each bin, compute Gaussian-weighted occupancy density
    3. Normalize: spike_density / occupancy_density

    Note: This ignores graph connectivity and uses Euclidean distance.
    Mass can "bleed through" walls.

    Notes
    -----
    As in :func:`_diffusion_kde`, the firing-rate denominator is the *smoothed*
    occupancy density and the ``min_occupancy`` cut is applied separately, on
    the *raw* occupancy in seconds (see :func:`_apply_min_occupancy_mask`), so
    it is consistent across smoothing methods.
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    weights = _get_gaussian_kernel(env, bandwidth)
    spike_density = weights @ spike_counts
    occupancy_density = weights @ occupancy

    # Divide only where the smoothed denominator is strictly positive (numerical
    # guard, NOT the min_occupancy cut -- that is applied on raw occupancy below).
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            occupancy_density > 0.0,
            spike_density / occupancy_density,
            np.nan,
        )

    firing_rate = _apply_min_occupancy_mask(firing_rate, occupancy, min_occupancy)
    return firing_rate.astype(np.float64)


def _binned(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
) -> NDArray[np.float64]:
    """Apply binned smoothing.

    Algorithm (bin-then-smooth order):
    1. Compute raw rate: spike_counts / occupancy
    2. Apply the masked diffusion average to the rate (see :func:`_binned_gate`).

    This order can introduce discretization artifacts.
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Step 1: Compute raw firing rate
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_rate = np.where(occupancy > 0, spike_counts / occupancy, np.nan)

    # Exclude sub-threshold bins from the smoothing average so their noisy
    # ratios do not bleed into neighbors. The output-side mask below then keeps
    # them NaN even if the masked average would re-fill them from neighbors.
    if min_occupancy > 0:
        raw_rate = np.where(occupancy >= min_occupancy, raw_rate, np.nan)

    # Step 2: Smooth the rate map (if bandwidth > 0)
    if bandwidth <= 0 or np.all(np.isnan(raw_rate)):
        return raw_rate.astype(np.float64)

    # A firing rate is non-negative. The shipped dense-kernel binned path was
    # structurally >= 0 (clipped kernel times non-negative raw rate); the linear
    # apply-path can leave tolerance-level negatives at far/unvisited bins, so
    # clip here (as _diffusion_kde does). _binned_gate itself stays sign-
    # preserving -- resample uses the same masked-average pattern on signed
    # fields -- so the clip lives in this rate wrapper, not the helper.
    smoothed = np.clip(_binned_gate(env, raw_rate, bandwidth), 0.0, None)
    # Re-apply the raw-occupancy mask: the masked average can re-fill a masked
    # bin from its neighbors, but result.occupancy < min_occupancy must stay NaN.
    masked = _apply_min_occupancy_mask(smoothed, occupancy, min_occupancy)
    return masked.astype(np.float64)


def _binned_gate(
    env: _BaseEnvironment,
    raw_rate: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Masked diffusion average of a (possibly NaN) intensive rate.

    Nadaraya-Watson with the row-stochastic average operator on the input's
    valid (finite) bins: ``diffuse(rate * valid) / diffuse(valid)``, so uncovered
    / NaN bins contribute no weight (they do not pull covered neighbours toward
    zero) and an interior NaN is interpolated rather than propagating. Both the
    filled rate and the validity mask are smoothed in one matrix-free
    ``env.diffuse(mode="average")`` pass.

    The final ``> 0`` support is derived from the **W-component structure**, NOT
    the smoothed weight's sign: within a connected component the average operator
    is entrywise positive, so a bin is supported iff its component holds a valid
    input bin. This is exact and truncation-proof (a ``max(den, 0)`` floor would
    still fail ``> 0`` where truncation flipped a tiny-positive dense denominator,
    spuriously emitting NaN). Where support holds we divide by ``max(den, eps)``.

    This helper is **sign-preserving** (the numerator is not floored), so it is
    also the pattern ``resample_field`` uses for signed intensive fields; callers
    that need a non-negative result (the ``binned`` firing-rate wrappers) clip
    the output themselves.

    **Far-field caveat.** Value-equality with the dense masked average holds only
    where the dense denominator is comfortably above the truncation floor (well
    within a bandwidth of valid input). For a bin **deep inside a supported
    component but many bandwidths from any valid bin** (sparse coverage), both
    ``num`` and ``den`` are truncation-noise-dominated, so the ratio can deviate
    from the dense value by well more than ``tol`` (the dense operator extends the
    matched-ratio constant across the whole component; the truncated one decays
    toward zero). Such bins are finite and supported but should be treated as
    extrapolation, not interpolation. Realistic dense coverage with scattered gaps
    is unaffected (matches the dense average to ~1e-13).

    Handles a 1-D ``(n_bins,)`` rate and a 2-D ``(n_neurons, n_bins)`` batch.
    """
    nan_mask = np.isnan(raw_rate)
    valid = ~nan_mask
    rate_filled = np.where(nan_mask, 0.0, raw_rate)
    weights = valid.astype(np.float64)

    # Smooth [rate_filled | weights] together (columns of an (n_bins, 2*k) batch).
    if raw_rate.ndim == 1:
        stacked = np.column_stack([rate_filled, weights])  # (n_bins, 2)
    else:
        stacked = np.concatenate([rate_filled.T, weights.T], axis=1)
    smoothed = np.asarray(
        cast("EnvironmentProtocol", env).diffuse(stacked, bandwidth, mode="average")
    )
    if raw_rate.ndim == 1:
        rate_smoothed = smoothed[:, 0]
        weights_smoothed = smoothed[:, 1]
    else:
        n_neurons = raw_rate.shape[0]
        rate_smoothed = smoothed[:, :n_neurons].T
        weights_smoothed = smoothed[:, n_neurons:].T

    n_components, labels = diffusion_component_labels(cast("EnvironmentProtocol", env))
    support = component_support_mask(labels, n_components, valid)
    den = np.maximum(weights_smoothed, _DIFFUSE_DENOM_EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(support, rate_smoothed / den, np.nan)


def _diffusion_kde_batch(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> NDArray[np.floating[Any]]:
    """Apply diffusion KDE smoothing for multiple neurons (matrix-free)."""
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)
    n_neurons = spike_counts.shape[0]

    # env.diffuse takes (n_bins, n_fields): stack every neuron's counts (as
    # columns, spike_counts.T) plus the shared occupancy as one extra column, so
    # all neurons + occupancy diffuse in a single matrix-free density-mode pass.
    cols = np.column_stack([spike_counts.T, occupancy])  # (n_bins, n_neurons + 1)
    smoothed = np.asarray(
        cast("EnvironmentProtocol", env).diffuse(cols, bandwidth, mode="density")
    )
    spike_density = smoothed[:, :n_neurons].T  # (n_neurons, n_bins)
    occupancy_density = smoothed[:, n_neurons]  # (n_bins,)

    # Magnitude gate (numerical, not the min_occupancy cut) + nonneg clip, shared
    # across neurons (see _diffusion_kde). occupancy broadcasts over the neuron
    # axis in the raw-occupancy mask below.
    occ_floor = np.maximum(occupancy_density, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(
            occ_floor > 0.0,
            spike_density / occupancy_density,
            np.nan,
        )

    firing_rates = np.clip(firing_rates, 0.0, None)
    firing_rates = _apply_min_occupancy_mask(firing_rates, occupancy, min_occupancy)
    return firing_rates.astype(dtype)


def _gaussian_kde_batch(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> NDArray[np.floating[Any]]:
    """Apply Gaussian KDE smoothing for multiple neurons."""
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    weights = _get_gaussian_kernel(env, bandwidth)
    spike_density = spike_counts @ weights.T
    occupancy_density = weights @ occupancy

    # Divide where the denominator is strictly positive (numerical guard, shared
    # across neurons); the min_occupancy cut is applied on raw occupancy below.
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(
            occupancy_density > 0.0,
            spike_density / occupancy_density,
            np.nan,
        )

    firing_rates = _apply_min_occupancy_mask(firing_rates, occupancy, min_occupancy)
    return firing_rates.astype(dtype)


def _binned_batch(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> NDArray[np.floating[Any]]:
    """Apply binned smoothing for multiple neurons (single batched pass)."""
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_rates = np.where(occupancy > 0, spike_counts / occupancy, np.nan)

    # Exclude sub-threshold bins from the average (see _binned); the output-side
    # mask below keeps them NaN even if the masked average re-fills them.
    if min_occupancy > 0:
        raw_rates = np.where(occupancy >= min_occupancy, raw_rates, np.nan)

    if bandwidth <= 0:
        return raw_rates.astype(dtype)

    # One matrix-free pass over all neurons (a fully-NaN neuron has no valid bins,
    # so the W-component support gate yields an all-NaN row, matching the raw
    # input). Clip >= 0 (firing rate; see _binned).
    smoothed = np.clip(_binned_gate(env, raw_rates, bandwidth), 0.0, None)
    masked = _apply_min_occupancy_mask(smoothed, occupancy, min_occupancy)
    return masked.astype(dtype)


# =============================================================================
# JAX Implementation Functions
# =============================================================================


def _smooth_rate_map_jax(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float,
    min_occupancy: float,
) -> Any:
    """JAX implementation of smooth_rate_map.

    ``diffusion_kde`` smoothing runs IN JAX via ``env.diffuse(backend="jax")``
    (the cached eigenbasis is cast to ``jnp``), so ``jit`` / ``grad`` / GPU work
    through it. ``binned`` keeps the documented NumPy round-trip (its masked
    average is not a matrix multiply).
    """
    import jax.numpy as jnp

    from neurospatial.encoding._core_jax import compute_firing_rate_single

    # Convert inputs to JAX with explicit float64
    spike_counts_j = jnp.asarray(spike_counts, dtype=jnp.float64)
    occupancy_j = jnp.asarray(occupancy, dtype=jnp.float64)

    if method == "binned":
        # Binned: compute rate first, then smooth (NumPy round-trip; the masked
        # average + W-component support gate live in _binned_gate).
        firing_rate = compute_firing_rate_single(
            spike_counts_j, occupancy_j, min_occupancy=min_occupancy
        )
        if bandwidth <= 0:
            return firing_rate
        firing_rate_np = np.asarray(firing_rate)
        if np.all(np.isnan(firing_rate_np)):
            return firing_rate  # All NaN, return as-is
        gated = np.clip(_binned_gate(env, firing_rate_np, bandwidth), 0.0, None)
        # Re-apply the raw-occupancy mask (the masked average can re-fill masked
        # bins); consistent with the NumPy binned path.
        masked = _apply_min_occupancy_mask(gated, np.asarray(occupancy), min_occupancy)
        return jnp.asarray(masked, dtype=jnp.float64)

    # For diffusion_kde and gaussian_kde: smooth then normalize.
    if method == "diffusion_kde":
        # Run the smoothing IN JAX via env.diffuse(backend="jax"): counts and
        # occupancy diffuse together in one density-mode pass, no (n, n) kernel.
        stacked = jnp.stack([spike_counts_j, occupancy_j], axis=1)  # (n_bins, 2)
        smoothed = jnp.asarray(
            cast("EnvironmentProtocol", env).diffuse(
                stacked, bandwidth, mode="density", backend="jax"
            )
        )
        spike_density = smoothed[:, 0]
        occupancy_density = smoothed[:, 1]
        # Magnitude gate (numerical divide guard, not the min_occupancy cut)
        # + nonneg clip (see _diffusion_kde).
        occ_floor = jnp.maximum(occupancy_density, 0.0)
        firing_rate = jnp.where(
            occ_floor > 0.0,
            spike_density / occupancy_density,
            jnp.nan,
        )
        firing_rate = jnp.clip(firing_rate, 0.0, None)
        # Raw-occupancy (seconds) mask, consistent with the NumPy path.
        if min_occupancy > 0:
            firing_rate = jnp.where(occupancy_j >= min_occupancy, firing_rate, jnp.nan)
        return firing_rate

    # gaussian_kde: dense Gaussian weight matrix (nonneg), unchanged.
    kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
    spike_density = kernel_j @ spike_counts_j
    occupancy_density = kernel_j @ occupancy_j
    firing_rate = jnp.where(
        occupancy_density > 0.0,
        spike_density / occupancy_density,
        jnp.nan,
    )
    if min_occupancy > 0:
        firing_rate = jnp.where(occupancy_j >= min_occupancy, firing_rate, jnp.nan)
    return firing_rate


def _smooth_rate_maps_batch_jax(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float,
    min_occupancy: float,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> Any:
    """JAX implementation of smooth_rate_maps_batch.

    ``diffusion_kde`` smoothing runs IN JAX via ``env.diffuse(backend="jax")``.
    The core matmul/division runs in float64; only the returned array is cast to
    ``dtype``. When JAX x64 is disabled the float64 request is naturally narrowed
    by JAX; the final defensive cast in ``compute_spatial_rates`` guarantees the
    requested dtype regardless. ``binned`` keeps the documented NumPy round-trip.
    """
    import jax.numpy as jnp

    from neurospatial.encoding._core_jax import compute_firing_rates_batch

    # Resolve the matching jnp dtype for the final cast.
    jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64

    # Convert inputs to JAX with explicit float64
    spike_counts_j = jnp.asarray(spike_counts, dtype=jnp.float64)
    occupancy_j = jnp.asarray(occupancy, dtype=jnp.float64)

    if method == "binned":
        # Binned: compute rate first, then masked-average (NumPy round-trip; the
        # W-component support gate is in _binned_gate, handled in one batched pass).
        firing_rates = compute_firing_rates_batch(
            spike_counts_j, occupancy_j, min_occupancy=min_occupancy
        )
        if bandwidth <= 0:
            return jnp.asarray(firing_rates, dtype=jnp_dtype)
        gated = np.clip(
            _binned_gate(env, np.asarray(firing_rates), bandwidth), 0.0, None
        )
        # Re-apply the raw-occupancy mask (consistent with the NumPy binned path).
        masked = _apply_min_occupancy_mask(gated, np.asarray(occupancy), min_occupancy)
        return jnp.asarray(masked, dtype=jnp_dtype)

    # For diffusion_kde and gaussian_kde: smooth then normalize.
    if method == "diffusion_kde":
        # Run the batch smoothing IN JAX via env.diffuse(backend="jax"): every
        # neuron's counts (columns of spike_counts.T) plus the shared occupancy
        # diffuse together in one density-mode pass, no (n, n) kernel.
        n_neurons = spike_counts_j.shape[0]
        cols = jnp.concatenate(
            [spike_counts_j.T, occupancy_j[:, None]], axis=1
        )  # (n_bins, n_neurons + 1)
        smoothed = jnp.asarray(
            cast("EnvironmentProtocol", env).diffuse(
                cols, bandwidth, mode="density", backend="jax"
            )
        )
        spike_density = smoothed[:, :n_neurons].T
        occupancy_density = smoothed[:, n_neurons]
        # Magnitude gate (numerical divide guard, not the min_occupancy cut)
        # + nonneg clip (see _diffusion_kde).
        occ_floor = jnp.maximum(occupancy_density, 0.0)
        firing_rates = jnp.where(
            occ_floor > 0.0,
            spike_density / occupancy_density,
            jnp.nan,
        )
        firing_rates = jnp.clip(firing_rates, 0.0, None)
        # Raw-occupancy (seconds) mask (occupancy broadcasts over neurons).
        if min_occupancy > 0:
            firing_rates = jnp.where(
                occupancy_j >= min_occupancy, firing_rates, jnp.nan
            )
        return jnp.asarray(firing_rates, dtype=jnp_dtype)

    # gaussian_kde: dense Gaussian weight matrix (nonneg), unchanged.
    kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
    spike_density = spike_counts_j @ kernel_j.T
    occupancy_density = kernel_j @ occupancy_j
    firing_rates = jnp.where(
        occupancy_density > 0.0,
        spike_density / occupancy_density,
        jnp.nan,
    )
    if min_occupancy > 0:
        firing_rates = jnp.where(occupancy_j >= min_occupancy, firing_rates, jnp.nan)
    return jnp.asarray(firing_rates, dtype=jnp_dtype)
