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
parameter. When ``backend="jax"``, the core rate computation (spike_counts /
occupancy) is performed using JAX array operations from ``_core_jax.py``.

Note that the diffusion kernel computation uses Environment methods which are
NumPy-based, so the kernel is computed on CPU and then transferred to JAX.
The rate computation itself uses JAX operations.

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
            f"may be slow and memory-intensive. To reduce the cost, increase "
            f"bin_size (fewer bins). (smoothing_method='binned' also builds a "
            f"dense kernel, so it is not a memory mitigation.)",
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
        Minimum occupancy (seconds) for a bin to be included; bins below the
        threshold are set to NaN. The threshold is applied to the *same
        occupancy quantity used as the firing-rate denominator*. For the KDE
        methods (``diffusion_kde``, ``gaussian_kde``) the denominator is the
        smoothed occupancy density, so a bin with zero raw occupancy but a
        smoothed denominator above the threshold reports a finite rate. For
        ``binned`` the denominator is the raw per-bin occupancy, so the raw
        occupancy is thresholded.
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
    with ``smoothing_method="binned"``, the smoothing step requires a round-trip
    to NumPy (Environment.smooth uses NumPy). This may be slower than pure NumPy
    for this method. For optimal JAX performance, use ``diffusion_kde`` or
    ``gaussian_kde`` which keep the rate computation entirely in JAX.

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
        Minimum occupancy threshold.
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
    # Validate batch-specific requirements
    spike_counts = np.asarray(spike_counts)
    occupancy = np.asarray(occupancy)

    if spike_counts.ndim != 2:
        raise ValueError(
            f"spike_counts must be 2D (n_neurons, n_bins), got shape {spike_counts.shape}"
        )

    if spike_counts.shape[1] != occupancy.shape[0]:
        raise ValueError(
            f"spike_counts has {spike_counts.shape[1]} bins but "
            f"occupancy has {occupancy.shape[0]} bins"
        )

    if spike_counts.shape[1] != env.n_bins:
        raise ValueError(
            f"spike_counts has {spike_counts.shape[1]} bins but "
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

    # Convert to arrays
    spike_counts = np.asarray(spike_counts)
    occupancy = np.asarray(occupancy)

    # Check shapes match
    if spike_counts.shape != occupancy.shape:
        raise ValueError(
            f"spike_counts shape {spike_counts.shape} does not match "
            f"occupancy shape {occupancy.shape}"
        )

    # Check matches environment
    if spike_counts.shape[0] != env.n_bins:
        raise ValueError(
            f"spike_counts has {spike_counts.shape[0]} elements but "
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
    The firing-rate denominator is the *smoothed* occupancy density, not the raw
    occupancy, so the ``min_occupancy`` threshold is applied to it: a bin is NaN
    when its smoothed denominator is below ``min_occupancy``. Thresholding the raw
    occupancy would spuriously NaN-out bins never directly traversed yet with a
    well-defined denominator from neighboring occupancy.

    ``env.diffuse`` is a pure linear operator (unlike the shipped clipped dense
    kernel), so under truncation the smoothed occupancy density can carry a
    tolerance-level negative lobe; the **magnitude gate** floors it
    (``max(occupancy_density, 0)``) before the ``> threshold`` comparison, and the
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

    # Magnitude gate: floor the (possibly tolerance-negative) denominator before
    # comparing to the threshold; divide by the unfloored density where it holds
    # (there the floor is a no-op, so it equals the dense denominator).
    occupancy_threshold = max(min_occupancy, 0.0)
    occ_floor = np.maximum(occupancy_density, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            occ_floor > occupancy_threshold,
            spike_density / occupancy_density,
            np.nan,
        )

    # Clip the output rate >= 0 (np.clip preserves NaN).
    return np.clip(firing_rate, 0.0, None).astype(np.float64)


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
    As in :func:`_diffusion_kde`, the ``min_occupancy`` threshold is applied
    to the *smoothed* occupancy density (the firing-rate denominator), not
    the raw occupancy. See that function's notes for the rationale.
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    weights = _get_gaussian_kernel(env, bandwidth)
    spike_density = weights @ spike_counts
    occupancy_density = weights @ occupancy

    # Normalize, thresholding the smoothed occupancy density at min_occupancy.
    occupancy_threshold = max(min_occupancy, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            occupancy_density > occupancy_threshold,
            spike_density / occupancy_density,
            np.nan,
        )

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

    # Apply min_occupancy threshold before smoothing
    if min_occupancy > 0:
        raw_rate = np.where(occupancy >= min_occupancy, raw_rate, np.nan)

    # Step 2: Smooth the rate map (if bandwidth > 0)
    if bandwidth <= 0 or np.all(np.isnan(raw_rate)):
        return raw_rate.astype(np.float64)

    return _binned_gate(env, raw_rate, bandwidth).astype(np.float64)


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
    spuriously emitting NaN). Where support holds we divide by
    ``max(den, eps)``; a bin whose dense denominator is itself truncation-tiny is
    kept finite but not value-equal to the dense path (part of the approximation
    contract).

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

    # Magnitude gate + nonneg clip, shared across neurons (see _diffusion_kde).
    occupancy_threshold = max(min_occupancy, 0.0)
    occ_floor = np.maximum(occupancy_density, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(
            occ_floor > occupancy_threshold,
            spike_density / occupancy_density,
            np.nan,
        )

    return np.clip(firing_rates, 0.0, None).astype(dtype)


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

    # Threshold the smoothed occupancy density (the denominator), shared
    # across neurons. See _diffusion_kde notes for the rationale.
    occupancy_threshold = max(min_occupancy, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(
            occupancy_density > occupancy_threshold,
            spike_density / occupancy_density,
            np.nan,
        )

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

    if min_occupancy > 0:
        raw_rates = np.where(occupancy >= min_occupancy, raw_rates, np.nan)

    if bandwidth <= 0:
        return raw_rates.astype(dtype)

    # One matrix-free pass over all neurons (a fully-NaN neuron has no valid bins,
    # so the W-component support gate yields an all-NaN row, matching the raw input).
    return _binned_gate(env, raw_rates, bandwidth).astype(dtype)


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
        gated = _binned_gate(env, firing_rate_np, bandwidth)
        return jnp.asarray(gated, dtype=jnp.float64)

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
        # Magnitude gate + nonneg clip (see _diffusion_kde).
        occupancy_threshold = max(min_occupancy, 0.0)
        occ_floor = jnp.maximum(occupancy_density, 0.0)
        firing_rate = jnp.where(
            occ_floor > occupancy_threshold,
            spike_density / occupancy_density,
            jnp.nan,
        )
        return jnp.clip(firing_rate, 0.0, None)

    # gaussian_kde: dense Gaussian weight matrix (nonneg), unchanged.
    kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
    spike_density = kernel_j @ spike_counts_j
    occupancy_density = kernel_j @ occupancy_j
    occupancy_threshold = max(min_occupancy, 0.0)
    return jnp.where(
        occupancy_density > occupancy_threshold,
        spike_density / occupancy_density,
        jnp.nan,
    )


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
        gated = _binned_gate(env, np.asarray(firing_rates), bandwidth)
        return jnp.asarray(gated, dtype=jnp_dtype)

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
        # Magnitude gate + nonneg clip (see _diffusion_kde).
        occupancy_threshold = max(min_occupancy, 0.0)
        occ_floor = jnp.maximum(occupancy_density, 0.0)
        firing_rates = jnp.where(
            occ_floor > occupancy_threshold,
            spike_density / occupancy_density,
            jnp.nan,
        )
        firing_rates = jnp.clip(firing_rates, 0.0, None)
        return jnp.asarray(firing_rates, dtype=jnp_dtype)

    # gaussian_kde: dense Gaussian weight matrix (nonneg), unchanged.
    kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
    spike_density = spike_counts_j @ kernel_j.T
    occupancy_density = kernel_j @ occupancy_j
    occupancy_threshold = max(min_occupancy, 0.0)
    firing_rates = jnp.where(
        occupancy_density > occupancy_threshold,
        spike_density / occupancy_density,
        jnp.nan,
    )
    return jnp.asarray(firing_rates, dtype=jnp_dtype)
