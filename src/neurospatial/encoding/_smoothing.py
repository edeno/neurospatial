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

**Performance Warning**:

``gaussian_kde`` builds a dense ``(n_bins, n_bins)`` Gaussian weight matrix,
costing O(n_bins²) memory and O(n_bins²) time to materialize. This matrix is
cached per ``(environment, bandwidth)`` (see ``_get_gaussian_kernel``), so a
batch call computes it once and reuses it across all neurons; the per-neuron
cost is then a single O(n_bins²) matmul, i.e. O(n_neurons × n_bins²) for the
matmuls but only one weight-matrix build.

``diffusion_kde`` is **not** sparse: the diffusion heat kernel ``exp(-tL)`` is
dense by construction (every entry is positive), so it too is a dense
``(n_bins, n_bins)`` matrix costing O(n_bins²) memory. It is built once via a
matrix exponential (a one-time O(n_bins³) cost) and cached per
``(environment, bandwidth)``; the per-neuron smoothing is then a dense
O(n_bins²) matmul. Its advantage over ``gaussian_kde`` is boundary-awareness
(it diffuses over the environment graph and so respects walls), not lower
asymptotic cost.

**All three methods materialize a dense ``(n_bins, n_bins)`` kernel** —
``binned`` smooths its normalized rate map through the same diffusion kernel
(via ``env.smooth``), so it has no memory advantage — and each emits a loud
memory ``UserWarning`` (with a GB estimate) for very large environments, above
``_LARGE_KERNEL_THRESHOLD`` bins, then proceeds. There is **no** hard limit.
For environments that would not fit in RAM, reduce the bin count (increase
``bin_size``).

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
    kernel: NDArray[np.float64] | None = None,
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
    kernel : ndarray, shape (n_bins, n_bins), optional
        Precomputed diffusion kernel for efficiency when processing multiple
        neurons. Only used with method="diffusion_kde". If None, the kernel
        is computed from the environment.
    backend : {"numpy", "jax"}, default="numpy"
        Computation backend. When "jax", uses JAX array operations for the
        core rate computation (smoothing/division). The kernel computation
        from the Environment is always NumPy-based.

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

    +--------------+----------------+-----------------------+--------------+
    | Method       | Boundaries     | Complexity            | Artifacts    |
    +==============+================+=======================+==============+
    | diffusion_kde| Respects       | O(n_bins²) per neuron | None         |
    +--------------+----------------+-----------------------+--------------+
    | gaussian_kde | Ignores        | O(n_bins²) per neuron | Wall bleed   |
    +--------------+----------------+-----------------------+--------------+
    | binned       | Respects*      | O(n_bins²) per neuron | Discretization|
    +--------------+----------------+-----------------------+--------------+

    *binned computes the rate first, then smooths it (bin-then-smooth); the
    other methods smooth the counts, then normalize (smooth-then-normalize).

    The ``diffusion_kde`` kernel is dense ``(n_bins, n_bins)`` and built once
    per ``(env, bandwidth)`` via a matrix exponential (a one-time O(n_bins³)
    build); the per-neuron smoothing is then the dense O(n_bins²) matmul
    ``kernel @ counts``. **All three methods build a dense
    ``(n_bins, n_bins)`` kernel** — ``binned`` smooths its rate map through the
    same diffusion kernel (via ``env.smooth``), so it has no memory advantage —
    and each emits a loud memory ``UserWarning`` (with a GB estimate) above
    ``_LARGE_KERNEL_THRESHOLD`` bins, then proceeds (no hard limit).

    **Performance recommendation**: For most analyses use ``diffusion_kde``
    (default) -- it is boundary-aware. All three methods build a dense
    ``(n_bins, n_bins)`` kernel (cached per ``(env, bandwidth)`` and reused
    across neurons), so all cost O(n_bins²) memory. For environments too large
    for a dense kernel, increase ``bin_size`` to reduce the bin count.

    **Backend behavior**: When ``backend="jax"``, the kernel smoothing and
    rate computation use JAX operations. The kernel itself is computed from
    the Environment using NumPy and converted to JAX. This enables GPU
    acceleration for the matrix operations and compatibility with JAX
    transformations like ``jit`` and ``grad``.

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
            env, spike_counts, occupancy, method, bandwidth, min_occupancy, kernel
        )

    # Dispatch to appropriate NumPy method
    match method:
        case "diffusion_kde":
            return _diffusion_kde(
                env, spike_counts, occupancy, bandwidth, min_occupancy, kernel
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
    kernel: NDArray[np.float64] | None = None,
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
    kernel : ndarray, shape (n_bins, n_bins), optional
        Precomputed diffusion kernel.
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
            kernel,
            dtype=dtype,
        )

    # Dispatch to NumPy vectorized implementations
    if method == "diffusion_kde":
        if kernel is None:
            kernel = cast("EnvironmentProtocol", env).compute_kernel(
                bandwidth, mode="density", cache=True
            )
        return _diffusion_kde_batch(
            spike_counts, occupancy, bandwidth, min_occupancy, kernel, dtype=dtype
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
    kernel: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Apply diffusion KDE smoothing.

    Algorithm (correct KDE order):
    1. Spread spike counts using diffusion kernel
    2. Spread occupancy using diffusion kernel
    3. Normalize: spike_density / occupancy_density

    Notes
    -----
    The firing-rate denominator is the *smoothed* occupancy density
    (``kernel @ occupancy``), not the raw occupancy. The ``min_occupancy``
    threshold is therefore applied to that same smoothed occupancy density:
    a bin is NaN when its smoothed denominator is below ``min_occupancy``.
    Thresholding the raw occupancy instead would spuriously NaN-out bins
    that were never directly traversed yet have a well-defined denominator
    from neighboring occupancy (a discontinuity between the masking quantity
    and the division quantity).
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Get or compute kernel
    if kernel is None:
        kernel = cast("EnvironmentProtocol", env).compute_kernel(
            bandwidth, mode="density", cache=True
        )

    # Spread spike counts using kernel (kernel @ counts)
    spike_density = kernel @ spike_counts

    # Spread occupancy using kernel
    occupancy_density = kernel @ occupancy

    # Compute firing rate with safe division, thresholding the smoothed
    # occupancy density (the denominator) at min_occupancy.
    occupancy_threshold = max(min_occupancy, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            occupancy_density > occupancy_threshold,
            spike_density / occupancy_density,
            np.nan,
        )

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
    2. Apply diffusion smoothing to the rate

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
    if bandwidth <= 0:
        return raw_rate.astype(np.float64)

    # Handle NaN values by smoothing with weight normalization
    nan_mask = np.isnan(raw_rate)

    if np.all(nan_mask):
        # All NaN, nothing to smooth
        return raw_rate.astype(np.float64)

    # Fill NaN with 0 for smoothing
    rate_filled = raw_rate.copy()
    rate_filled[nan_mask] = 0.0

    # Create weights (1 where valid, 0 where NaN)
    weights = np.ones_like(raw_rate)
    weights[nan_mask] = 0.0

    # Smooth both rate and weights with the row-stochastic average kernel: this
    # is a masked (valid-bin-normalized) average of an intensive rate, so it is
    # volume-unbiased on non-uniform M. (On uniform M the per-cell volume factor
    # cancels in the rate/weights ratio, so the result is unchanged there.)
    rate_smoothed = cast("EnvironmentProtocol", env).smooth(
        rate_filled, bandwidth=bandwidth, mode="average"
    )
    weights_smoothed = cast("EnvironmentProtocol", env).smooth(
        weights, bandwidth=bandwidth, mode="average"
    )

    # Normalize by smoothed weights
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            weights_smoothed > 0,
            rate_smoothed / weights_smoothed,
            np.nan,
        )

    return firing_rate.astype(np.float64)


def _diffusion_kde_batch(
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bandwidth: float,
    min_occupancy: float,
    kernel: NDArray[np.float64],
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> NDArray[np.floating[Any]]:
    """Apply diffusion KDE smoothing for multiple neurons."""
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    spike_density = (kernel @ spike_counts.T).T
    occupancy_density = kernel @ occupancy

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
    """Apply binned smoothing for multiple neurons."""
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_rates = np.where(occupancy > 0, spike_counts / occupancy, np.nan)

    if min_occupancy > 0:
        raw_rates = np.where(occupancy >= min_occupancy, raw_rates, np.nan)

    if bandwidth <= 0:
        return raw_rates.astype(dtype)

    n_neurons = raw_rates.shape[0]
    result = np.empty_like(raw_rates, dtype=np.float64)
    env_protocol = cast("EnvironmentProtocol", env)

    for i in range(n_neurons):
        raw_rate = raw_rates[i]
        nan_mask = np.isnan(raw_rate)

        if np.all(nan_mask):
            result[i] = raw_rate
            continue

        rate_filled = raw_rate.copy()
        rate_filled[nan_mask] = 0.0

        weights = np.ones_like(raw_rate)
        weights[nan_mask] = 0.0

        rate_smoothed = env_protocol.smooth(
            rate_filled, bandwidth=bandwidth, mode="average"
        )
        weights_smoothed = env_protocol.smooth(
            weights, bandwidth=bandwidth, mode="average"
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            result[i] = np.where(
                weights_smoothed > 0,
                rate_smoothed / weights_smoothed,
                np.nan,
            )

    return result.astype(dtype)


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
    kernel: NDArray[np.float64] | None,
) -> Any:
    """JAX implementation of smooth_rate_map.

    Uses JAX array operations for the core computation while keeping
    kernel computation on NumPy (from Environment).
    """
    import jax.numpy as jnp

    from neurospatial.encoding._core_jax import compute_firing_rate_single

    # Convert inputs to JAX with explicit float64
    spike_counts_j = jnp.asarray(spike_counts, dtype=jnp.float64)
    occupancy_j = jnp.asarray(occupancy, dtype=jnp.float64)

    if method == "binned":
        # Binned: compute rate first, then smooth
        # Rate computation uses JAX
        firing_rate = compute_firing_rate_single(
            spike_counts_j, occupancy_j, min_occupancy=min_occupancy
        )

        if bandwidth <= 0:
            return firing_rate

        # Smoothing still uses NumPy (Environment.smooth)
        # Convert back to NumPy for smoothing, then back to JAX
        firing_rate_np = np.asarray(firing_rate)
        nan_mask = np.isnan(firing_rate_np)

        if np.all(nan_mask):
            return firing_rate  # All NaN, return as-is

        rate_filled = firing_rate_np.copy()
        rate_filled[nan_mask] = 0.0
        weights = np.ones_like(firing_rate_np)
        weights[nan_mask] = 0.0

        env_protocol = cast("EnvironmentProtocol", env)
        rate_smoothed = env_protocol.smooth(
            rate_filled, bandwidth=bandwidth, mode="average"
        )
        weights_smoothed = env_protocol.smooth(
            weights, bandwidth=bandwidth, mode="average"
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            result_np = np.where(
                weights_smoothed > 0,
                rate_smoothed / weights_smoothed,
                np.nan,
            )
        return jnp.asarray(result_np, dtype=jnp.float64)

    # For diffusion_kde and gaussian_kde: smooth then normalize
    # Get kernel (computed by Environment, so NumPy)
    if method == "diffusion_kde":
        if kernel is None:
            kernel = cast("EnvironmentProtocol", env).compute_kernel(
                bandwidth, mode="density", cache=True
            )
        kernel_j = jnp.asarray(kernel, dtype=jnp.float64)

        # JAX matrix operations for smoothing
        spike_density = kernel_j @ spike_counts_j
        occupancy_density = kernel_j @ occupancy_j

    else:  # gaussian_kde
        kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
        spike_density = kernel_j @ spike_counts_j
        occupancy_density = kernel_j @ occupancy_j

    # Compute firing rate using JAX, thresholding the smoothed occupancy
    # density (the denominator) at min_occupancy. See _diffusion_kde notes.
    occupancy_threshold = max(min_occupancy, 0.0)
    firing_rate = jnp.where(
        occupancy_density > occupancy_threshold,
        spike_density / occupancy_density,
        jnp.nan,
    )

    return firing_rate


def _smooth_rate_maps_batch_jax(
    env: _BaseEnvironment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    method: Literal["diffusion_kde", "gaussian_kde", "binned"],
    bandwidth: float,
    min_occupancy: float,
    kernel: NDArray[np.float64] | None,
    *,
    dtype: type[np.float32] | type[np.float64] = np.float64,
) -> Any:
    """JAX implementation of smooth_rate_maps_batch.

    Uses JAX array operations for the core computation while keeping
    kernel computation on NumPy (from Environment). The core matmul/division
    runs in float64; only the returned array is cast to ``dtype``. When JAX
    x64 is disabled, the float64 request is naturally narrowed by JAX; the
    final defensive cast in ``compute_spatial_rates`` guarantees the requested
    dtype regardless.
    """
    import jax.numpy as jnp

    from neurospatial.encoding._core_jax import compute_firing_rates_batch

    # Resolve the matching jnp dtype for the final cast.
    jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64

    # Convert inputs to JAX with explicit float64
    spike_counts_j = jnp.asarray(spike_counts, dtype=jnp.float64)
    occupancy_j = jnp.asarray(occupancy, dtype=jnp.float64)

    if method == "binned":
        # Binned: compute rate first, then smooth
        # Rate computation uses JAX
        firing_rates = compute_firing_rates_batch(
            spike_counts_j, occupancy_j, min_occupancy=min_occupancy
        )

        if bandwidth <= 0:
            return jnp.asarray(firing_rates, dtype=jnp_dtype)

        # Smoothing uses NumPy (Environment.smooth) - per-neuron loop
        # This method is not optimized for JAX
        firing_rates_np = np.asarray(firing_rates)
        n_neurons = firing_rates_np.shape[0]
        result = np.empty_like(firing_rates_np, dtype=np.float64)
        env_protocol = cast("EnvironmentProtocol", env)

        for i in range(n_neurons):
            raw_rate = firing_rates_np[i]
            nan_mask = np.isnan(raw_rate)

            if np.all(nan_mask):
                result[i] = raw_rate
                continue

            rate_filled = raw_rate.copy()
            rate_filled[nan_mask] = 0.0
            weights = np.ones_like(raw_rate)
            weights[nan_mask] = 0.0

            rate_smoothed = env_protocol.smooth(
                rate_filled, bandwidth=bandwidth, mode="average"
            )
            weights_smoothed = env_protocol.smooth(
                weights, bandwidth=bandwidth, mode="average"
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                result[i] = np.where(
                    weights_smoothed > 0,
                    rate_smoothed / weights_smoothed,
                    np.nan,
                )

        return jnp.asarray(result, dtype=jnp_dtype)

    # For diffusion_kde and gaussian_kde: smooth then normalize
    # Get kernel (computed by Environment, so NumPy)
    if method == "diffusion_kde":
        if kernel is None:
            kernel = cast("EnvironmentProtocol", env).compute_kernel(
                bandwidth, mode="density", cache=True
            )
        kernel_j = jnp.asarray(kernel, dtype=jnp.float64)

        # JAX matrix operations for batch smoothing
        # (kernel @ spike_counts.T).T = spike_counts @ kernel.T
        spike_density = (kernel_j @ spike_counts_j.T).T
        occupancy_density = kernel_j @ occupancy_j

    else:  # gaussian_kde
        kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
        spike_density = spike_counts_j @ kernel_j.T
        occupancy_density = kernel_j @ occupancy_j

    # Compute firing rates using JAX (broadcasting over neurons), thresholding
    # the smoothed occupancy density (the denominator) at min_occupancy.
    # See _diffusion_kde notes.
    occupancy_threshold = max(min_occupancy, 0.0)
    firing_rates = jnp.where(
        occupancy_density > occupancy_threshold,
        spike_density / occupancy_density,
        jnp.nan,
    )

    return jnp.asarray(firing_rates, dtype=jnp_dtype)
