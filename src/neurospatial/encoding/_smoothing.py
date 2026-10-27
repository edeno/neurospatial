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
large grid — genuinely needs most modes, so ``env.diffuse`` solves the full
dense basis once there, with the same large-matrix ``UserWarning`` as
``compute_kernel``, and reuses it for later calls.)

``gaussian_kde`` still builds a dense ``(n_bins, n_bins)`` Gaussian weight matrix
(Euclidean, boundary-ignoring), costing O(n_bins²) memory and time; it is cached
per ``(environment, bandwidth)`` (see ``_get_gaussian_kernel``) and reused across
neurons, and warns above ``LARGE_KERNEL_THRESHOLD`` bins. For very large
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
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._stacklevel import _external_stacklevel
from neurospatial.ops import (
    _RESOLUTION_MARGIN,
    _above_component_floor,
    _masked_diffusion_average,
)

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.ops._types import ResolvedBackend

__all__ = [
    "smooth_rate_map",
    "smooth_rate_maps_batch",
]


def resolve_spatial_bandwidth(env: Any, bandwidth: float | None) -> float:
    """Resolve a length-scale default and diagnose smoothing across the arena.

    The default is 5 cm in declared metric units. For unspecified, pixel or
    custom units, it is 2.5 median graph-edge lengths (or bin widths when no
    edges exist). Polar environments use physical graph lengths and radial
    extent; their angular coordinate never enters a length comparison.
    An explicit value is always in environment coordinates.
    """
    if bandwidth is None:
        from neurospatial._units import units_per_centimeter

        per_cm = units_per_centimeter(env.units)
        if per_cm is not None:
            bandwidth = 5.0 * per_cm
        else:
            lengths = np.asarray(
                [
                    edge["distance"]
                    for _, _, edge in env.connectivity.edges(data=True)
                    if edge.get("distance", 0) > 0
                ]
            )
            if lengths.size == 0:
                lengths = env.coordinate_system.fallback_lengths(env)
            bandwidth = 2.5 * float(np.median(lengths))
    # Point bins built from_bins without declared ranges have no known
    # extent; the warning below needs one.
    ranges = (
        ()
        if env.dimension_ranges is None
        else env.coordinate_system.length_ranges(env.dimension_ranges)
    )
    extent = max((stop - start for start, stop in ranges), default=0.0)
    if extent > 0 and bandwidth >= extent:
        warnings.warn(
            f"bandwidth={bandwidth:g} spans the environment (extent {extent:g} "
            f"{env.units or 'coordinate units'}), which can erase spatial tuning. "
            "Choose an explicit bandwidth in the same units as positions; "
            "check env.units and the scale of your recording.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    return float(bandwidth)


def _get_gaussian_kernel(env: Environment, bandwidth: float) -> NDArray[np.float64]:
    """Compatibility wrapper for environment-owned Gaussian weights."""
    return env.compute_gaussian_kernel(bandwidth=bandwidth)


def smooth_rate_map(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: ResolvedBackend = "numpy",
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
          on Euclidean distance between bin centers, so it smooths through
          walls (21-60% of a field's peak 10 cm away across a thin wall at 5-10 cm
          bandwidths in a simulated maze); prefer ``diffusion_kde`` or
          ``binned`` in walled mazes. Order: smooth → normalize.
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
        ``binned``), so ``min_occupancy`` means the same thing across methods.
        The default ``0.0`` keeps every visited bin; unvisited bins are NaN
        whatever the threshold. If the threshold would mask every occupied
        bin -- usually a units/scale mistake, since it is seconds -- a
        ``UserWarning`` is emitted and the rate map is empty.
    backend : {"numpy", "jax"}, default="numpy"
        Computation backend. When "jax", the ``diffusion_kde`` smoothing runs
        in JAX via ``env.diffuse(backend="jax")`` (the cached eigenbasis is cast
        to ``jnp``), so ``jit`` / ``grad`` / GPU work through it.

    Returns
    -------
    ArrayLike, shape (n_bins,)
        Smoothed firing rate in Hz (spikes/second). NaN where the rate is
        missing: every bin the animal never visited and every bin whose raw
        occupancy is below ``min_occupancy``, for every method. The kernel
        reaches unvisited bins, but a rate there would be extrapolated from
        visits a bandwidth or more away. ``diffusion_kde`` is also NaN in a
        visited bin whose smoothed occupancy is below the accuracy of a
        truncated ``env.diffuse`` (1e-3 of the largest in its component).
        Returns ndarray for numpy backend, jax.Array for jax backend.

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
    ``LARGE_KERNEL_THRESHOLD`` bins).

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
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float = 5.0,
    min_occupancy: float = 0.0,
    backend: ResolvedBackend = "numpy",
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
        every method. Unvisited bins are always NaN. See ``smooth_rate_map``.
    backend : {"numpy", "jax"}, default="numpy"
        Computation backend. When "jax", uses JAX array operations for the
        core rate computation (smoothing/division). See smooth_rate_map for
        details on backend behavior.
    dtype : {np.float32, np.float64}, default=np.float64
        Storage dtype of the returned rate-map array. The matmul/division is
        always done in float64 for accuracy; only the final returned array is
        cast to ``dtype``. ``np.float32`` halves the ``(n_neurons, n_bins)``
        storage. Honored on the NumPy path; on the JAX path the cast is
        applied to the returned array as well.

    Returns
    -------
    ArrayLike, shape (n_neurons, n_bins)
        Smoothed firing rates in Hz for each neuron, NaN in the bins described
        in ``smooth_rate_map``. Returns ndarray for numpy backend, jax.Array
        for jax backend.

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
    env: Environment,
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


_SPATIAL_METHODS: tuple[str, ...] = (
    "diffusion_kde",
    "gaussian_kde",
    "binned",
    "glm",
)
_RATIO_METHOD_PARAMS: tuple[str, ...] = ("bandwidth", "min_occupancy", "fill_value")


def _validate_glm_penalty(penalty: Any) -> float:
    """Validate a glm ``penalty`` value: reject ``bool`` first, then finite & >= 0.

    ``bool`` is checked BEFORE any numeric coercion because ``True`` / ``False``
    are ``int`` in Python (``float(True) == 1.0``), so a bare numeric check would
    silently accept them as ``lambda`` values.
    """
    if isinstance(penalty, bool):
        raise ValueError(
            f"penalty must be a real number (a smoothing weight lambda), not bool; "
            f"got {penalty!r}."
        )
    try:
        value = float(penalty)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"penalty must be a real number (a smoothing weight lambda); got "
            f"{penalty!r}."
        ) from exc
    except OverflowError as exc:
        # A Python int too large to represent as a finite float (e.g. 10**1000):
        # penalty is NOT clamped, so an unrepresentable value is rejected cleanly
        # rather than leaking a raw OverflowError.
        raise ValueError(
            f"penalty must be finite and >= 0 (a smoothing weight lambda); got "
            f"{penalty!r}, which is too large to represent as a finite float."
        ) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"penalty must be finite and >= 0 (a smoothing weight lambda; 0 = no "
            f"penalty, None = choose by REML); got {value}."
        )
    return value


def _validate_glm_rank(rank: Any) -> int:
    """Validate a glm ``rank`` value: reject ``bool`` first, then integral & >= 1.

    Magnitude is NOT rejected here -- an out-of-range ``rank`` is clamped to the
    effective rank ``r_eff = max(n_live_components, min(n_live_bins, rank))`` when
    the basis is built, and the clamp is reported via ``result.rank``.
    """
    if isinstance(rank, bool):
        raise ValueError(
            f"rank must be an integer (the requested basis rank), not bool; got "
            f"{rank!r}."
        )
    # Any integral value is accepted WITHOUT a float round-trip: magnitude is
    # clamped (never rejected) to the effective rank when the basis is built, and
    # a huge Python int (e.g. 10**1000) overflows float64 -- coercing it would
    # leak an OverflowError and wrongly reject a value the clamp handles.
    if isinstance(rank, (int, np.integer)):
        value = int(rank)
    else:
        try:
            as_float = float(rank)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"rank must be an integer (the requested basis rank); got {rank!r}."
            ) from exc
        if not np.isfinite(as_float) or as_float != int(as_float):
            raise ValueError(
                f"rank must be an integer (the requested basis rank); got {rank!r}."
            )
        value = int(as_float)
    if value < 1:
        raise ValueError(f"rank must be >= 1 (the requested basis rank); got {value}.")
    return value


def validate_pooled(pooled: Any, method: str) -> None:
    """Validate the ``pooled`` flag of ``compute_spatial_rate(s)``.

    ``pooled`` is a **strict** ``bool`` and **glm-only**:

    - reject any non-``bool`` (``isinstance(pooled, bool)`` -- rejects strings,
      ``int`` including ``0`` / ``1``, arrays, and ``None``), because ``True`` /
      ``False`` are the only meaningful values and ``1`` / ``0`` would silently
      pass a bare truthiness check;
    - reject ``pooled=False`` with a **ratio method** (``pooled`` selects between
      shared / per-unit ``lambda``, which only the glm estimator has).
      ``pooled=True`` with a ratio method is the harmless default and does not
      raise.

    Parameters
    ----------
    pooled : Any
        The value passed by the caller.
    method : str
        The requested estimator (already validated as a known method).

    Raises
    ------
    ValueError
        If ``pooled`` is not a real ``bool``, or ``pooled=False`` is combined
        with a ratio method.
    """
    if not isinstance(pooled, bool):
        raise ValueError(
            "pooled must be a bool (True = one shared smoothing penalty lambda, "
            "False = an independent per-unit lambda_k); got "
            f"{pooled!r} of type {type(pooled).__name__}."
        )
    if method != "glm" and pooled is False:
        raise ValueError(
            "pooled is a method='glm' parameter (it selects shared vs per-unit "
            f"lambda) and pooled=False cannot be combined with method={method!r}. "
            "Ratio methods have no smoothing penalty; use method='glm' for "
            "per-unit smoothing."
        )


def _validate_spatial_method_params(
    method: str,
    *,
    bandwidth: float | None,
    min_occupancy: float | None,
    fill_value: float | None,
    penalty: float | None,
    rank: int | None,
) -> tuple[float | None, int | None]:
    """Validate the method-specific params of ``compute_spatial_rate(s)``.

    Applies, in order:

    1. **Method validity** -- ``method`` must be one of the four spatial methods
       (``diffusion_kde`` / ``gaussian_kde`` / ``binned`` / ``glm``).
    2. **Mutual exclusivity** -- an *explicitly set* (``not None``) ratio param
       (``bandwidth`` / ``min_occupancy`` / ``fill_value``) with ``method="glm"``,
       or ``penalty`` / ``rank`` with a ratio method, raises ``ValueError`` naming
       which param belongs to which method. Keyed on ``is None`` (the sentinel
       default), so bare ``method="glm"`` does not false-trip on a resolved
       ``bandwidth``.
    3. **Value domains** (only when set) -- ``penalty`` and ``rank`` are validated
       and returned (``bool`` rejected before numeric coercion). ``rank`` magnitude
       is clamped, not rejected, when the basis is built.

    Ratio-default resolution (``bandwidth -> 5.0``, ``min_occupancy -> 0.0``) is
    the caller's job, done *after* this returns.

    Parameters
    ----------
    method : str
        The requested estimator.
    bandwidth, min_occupancy, fill_value : float or None
        Ratio-method params (``None`` = "not set").
    penalty : float or None
        glm penalty (``None`` = choose by REML).
    rank : int or None
        glm requested rank (``None`` = default cap).

    Returns
    -------
    penalty : float or None
        The validated penalty (``None`` if not set).
    rank : int or None
        The validated rank (``None`` if not set); magnitude clamping is deferred.

    Raises
    ------
    ValueError
        On an unknown method, a mutual-exclusivity violation, or an out-of-domain
        ``penalty`` / ``rank``.
    """
    if method not in _SPATIAL_METHODS:
        raise ValueError(
            f"method must be one of {list(_SPATIAL_METHODS)}, got {method!r}"
        )

    if method == "glm":
        ratio_values = {
            "bandwidth": bandwidth,
            "min_occupancy": min_occupancy,
            "fill_value": fill_value,
        }
        for name, value in ratio_values.items():
            if value is not None:
                raise ValueError(
                    f"{name} is a ratio-method (diffusion_kde/gaussian_kde/binned) "
                    f"parameter and cannot be combined with method='glm'. The glm "
                    f"estimator is configured with penalty= and rank=."
                )
    else:
        for name, value in {"penalty": penalty, "rank": rank}.items():
            if value is not None:
                raise ValueError(
                    f"{name} is a method='glm' parameter and cannot be combined with "
                    f"method={method!r}. Ratio methods use bandwidth=, "
                    f"min_occupancy=, and fill_value=."
                )

    validated_penalty = None if penalty is None else _validate_glm_penalty(penalty)
    validated_rank = None if rank is None else _validate_glm_rank(rank)
    return validated_penalty, validated_rank


def _validate_smoothing_parameters(method: str, bandwidth: float) -> None:
    """Validate smoothing method and bandwidth without requiring count arrays."""
    valid_methods = ["diffusion_kde", "gaussian_kde", "binned"]
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


def validate_min_occupancy(min_occupancy: float) -> float:
    """Return ``min_occupancy`` as a float after checking it is finite and >= 0.

    Raises
    ------
    ValueError
        If ``min_occupancy`` is NaN, infinite or negative. A NaN threshold
        compares False with every occupancy, so it would mask every bin
        without the fully-masked warning.
    """
    value = float(min_occupancy)
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(
            _format_error(
                f"min_occupancy must be a finite number of seconds >= 0, got {min_occupancy}.",
                why="Why: it is the minimum time the animal must spend in a bin for its rate to count",
                fix="pass min_occupancy=0.0 for no masking, or a dwell time in seconds such as 0.5",
            )
        )
    return value


def _mask_outside_support(
    firing_rate: NDArray[np.floating[Any]],
    occupancy: NDArray[np.float64],
    min_occupancy: float,
) -> NDArray[np.floating[Any]]:
    """Set the rate to NaN outside the occupancy support.

    A bin is supported when its *raw* occupancy is positive and at least
    ``min_occupancy`` seconds, the ``spatial_support`` of a rate result. A
    smoothed rate in a never-visited bin is an extrapolation from visits a
    bandwidth or more away, not an estimate, so every method leaves it NaN.
    The cut is on raw seconds for every method, never on a smoothed density,
    so the masked bins are exactly ``~result.spatial_support``. ``occupancy``
    (shape ``(n_bins,)``) broadcasts over a leading neuron axis for batch rate
    maps.
    """
    min_occupancy = validate_min_occupancy(min_occupancy)
    supported = (occupancy > 0) & (occupancy >= min_occupancy)
    return np.where(supported, firing_rate, np.nan)


def _mask_outside_support_jax(
    firing_rate: Any, occupancy: Any, min_occupancy: float
) -> Any:
    """JAX version of :func:`_mask_outside_support` (traceable under ``jit``)."""
    import jax.numpy as jnp

    supported = (occupancy > 0) & (occupancy >= min_occupancy)
    return jnp.where(supported, firing_rate, jnp.nan)


def _warn_if_fully_masked(occupancy: NDArray[np.float64], min_occupancy: float) -> None:
    """Warn when ``min_occupancy`` would mask every occupied bin.

    Masking all bins the animal actually visited yields an empty rate map
    (all-NaN, or all-``fill_value``) with no other signal -- almost always a
    sign that ``min_occupancy`` is too large or in the wrong units (it is
    seconds). This defense-in-depth guard turns that silent empty result into a
    loud, actionable warning.
    """
    if validate_min_occupancy(min_occupancy) == 0.0:
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
            stacklevel=_external_stacklevel(),
        )


def _resolution_floor(env: Environment, bandwidth: float) -> float:
    """Fraction of its component maximum below which a smoothed value is noise.

    ``_RESOLUTION_MARGIN`` times the relative accuracy of ``env.diffuse`` at
    ``bandwidth`` (1e-3 for a truncated eigenbasis, 1e-10 for the full basis).
    """
    return _RESOLUTION_MARGIN * env.diffusion_accuracy(bandwidth=bandwidth)


def _resolved_density_mask(
    env: Environment,
    occupancy_density: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.bool_]:
    """Bins whose smoothed occupancy density is resolved above the apply's noise.

    ``env.diffuse`` reproduces the dense heat operator only to a relative
    accuracy (``env.diffusion_accuracy``: the truncation tolerance, or float64
    roundoff for the full basis). Where the smoothed occupancy density is near
    that noise, the smoothed spike density is too, and their ratio is noise (it
    reached 20,000 Hz on a 50 Hz field). A bin is resolved when its smoothed
    occupancy density exceeds :func:`_resolution_floor` times the largest
    density in its ``W``-component, which keeps the ratio's error near 0.1% of
    the bin's rate or less. A visited bin fails this only with a truncated
    eigenbasis, when its dwell is a tiny fraction of the busiest region's.

    Parameters
    ----------
    env : Environment
        The environment whose ``diffuse`` produced ``occupancy_density``.
    occupancy_density : NDArray[np.float64], shape (n_bins,)
        Smoothed occupancy density.
    bandwidth : float
        Smoothing bandwidth of the apply.

    Returns
    -------
    NDArray[np.bool_], shape (n_bins,)
        True where the KDE ratio is numerically meaningful.
    """
    geometry = env.get_diffusion_geometry()
    return _above_component_floor(
        geometry.labels,
        geometry.n_components,
        occupancy_density,
        _resolution_floor(env, bandwidth),
    )


def _resolved_density_mask_jax(
    env: Environment, occupancy_density: Any, bandwidth: float
) -> Any:
    """JAX version of :func:`_resolved_density_mask` (traceable under ``jit``)."""
    import jax
    import jax.numpy as jnp

    geometry = env.get_diffusion_geometry()
    labels_j = jnp.asarray(geometry.labels)
    component_max = jax.ops.segment_max(
        occupancy_density, labels_j, num_segments=geometry.n_components
    )
    threshold = _resolution_floor(env, bandwidth) * component_max[labels_j]
    return occupancy_density > threshold


def _diffusion_kde(
    env: Environment,
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
    The firing-rate denominator is the *smoothed* occupancy density. The rate
    is reported only in supported bins, those with raw occupancy above zero
    and at least ``min_occupancy`` seconds (see :func:`_mask_outside_support`):
    a never-visited bin next to visited ones has a finite smoothed ratio, but
    it is an extrapolation, not an estimate. A supported bin whose smoothed
    occupancy density is below the diffusion apply's resolution (see
    :func:`_resolved_density_mask`) is NaN too, since there both densities are
    truncation noise.

    ``env.diffuse`` is a linear operator, so under truncation the smoothed
    densities carry tolerance-level noise, including negative lobes; the
    **resolution gate** divides only where the occupancy density is resolved
    above that noise, and the output rate is clipped ``>= 0`` (a firing rate
    is non-negative, which decoding relies on).
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    # Smooth counts and occupancy in one matrix-free density-mode pass.
    stacked = np.column_stack([spike_counts, occupancy])  # (n_bins, 2)
    smoothed = np.asarray(env.diffuse(stacked, bandwidth=bandwidth, mode="density"))
    spike_density = smoothed[:, 0]
    occupancy_density = smoothed[:, 1]

    # Resolution gate: divide only where the denominator is resolved above the
    # apply's truncation noise (avoids 0/0, negative-lobe and noise/noise
    # blowups). This gate is purely numerical -- it is NOT the support mask.
    resolved = _resolved_density_mask(env, occupancy_density, bandwidth)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rate = np.where(
            resolved,
            spike_density / occupancy_density,
            np.nan,
        )

    # Clip the output rate >= 0 (np.clip preserves NaN), then keep only the
    # occupancy support (raw seconds), consistent across methods.
    firing_rate = np.clip(firing_rate, 0.0, None)
    firing_rate = _mask_outside_support(firing_rate, occupancy, min_occupancy)
    return firing_rate.astype(np.float64)


def _gaussian_kde(
    env: Environment,
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
    occupancy density and the rate is reported only in supported bins (see
    :func:`_mask_outside_support`). The dense kernel is exact, so a supported
    bin, whose own occupancy enters its denominator, needs no resolution gate.
    """
    spike_counts = np.asarray(spike_counts, dtype=np.float64)
    occupancy = np.asarray(occupancy, dtype=np.float64)

    weights = _get_gaussian_kernel(env, bandwidth)
    spike_density = weights @ spike_counts
    occupancy_density = weights @ occupancy

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = spike_density / occupancy_density
    return _mask_outside_support(ratio, occupancy, min_occupancy).astype(np.float64)


def _binned(
    env: Environment,
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

    # A firing rate is non-negative. The linear apply-path can leave
    # tolerance-level negatives at far/unvisited bins, so clip here (as
    # _diffusion_kde does). _binned_gate itself stays sign-
    # preserving -- resample uses the same masked-average pattern on signed
    # fields -- so the clip lives in this rate wrapper, not the helper.
    smoothed = np.clip(_binned_gate(env, raw_rate, bandwidth), 0.0, None)
    # Re-apply the raw-occupancy mask: the masked average can re-fill a masked
    # bin from its neighbors, but result.occupancy < min_occupancy must stay NaN.
    masked = _mask_outside_support(smoothed, occupancy, min_occupancy)
    return masked.astype(np.float64)


def _binned_gate(
    env: Environment,
    raw_rate: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Masked diffusion average of a (possibly NaN) intensive rate.

    The shared :func:`~neurospatial.ops._masked_diffusion_average`
    over the rate's valid (finite) bins, so NaN bins contribute no weight and an
    interior NaN is interpolated rather than propagating. It is
    **sign-preserving**; the ``binned`` firing-rate wrappers clip the output.

    **Far field.** Many bandwidths from any valid bin, both ``num`` and ``den``
    approach the truncated apply's noise, and their ratio is noise (it reached
    295,658 Hz from raw rates of at most 50 Hz). Bins whose smoothed weight is
    below :func:`_resolution_floor` of the largest in their component are
    therefore NaN (see :func:`~neurospatial.ops._above_component_floor`);
    elsewhere the result matches the dense masked average.

    Handles a 1-D ``(n_bins,)`` rate and a 2-D ``(n_neurons, n_bins)`` batch.
    """
    geometry = env.get_diffusion_geometry()
    n_components, labels = geometry.n_components, geometry.labels
    rates = np.atleast_2d(raw_rate)  # (n_neurons, n_bins)
    averaged = _masked_diffusion_average(
        lambda columns: np.asarray(
            env.diffuse(columns, bandwidth=bandwidth, mode="average")
        ),
        rates.T,
        ~np.isnan(rates.T),
        labels,
        n_components,
        # Far from every valid bin both smoothed sums approach the apply's
        # truncation noise; leave those bins NaN rather than report noise/noise.
        floor=lambda: _resolution_floor(env, bandwidth),
    ).T
    result: NDArray[np.float64] = averaged[0] if raw_rate.ndim == 1 else averaged
    return result


def _diffusion_kde_batch(
    env: Environment,
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
    smoothed = np.asarray(env.diffuse(cols, bandwidth=bandwidth, mode="density"))
    spike_density = smoothed[:, :n_neurons].T  # (n_neurons, n_bins)
    occupancy_density = smoothed[:, n_neurons]  # (n_bins,)

    # Resolution gate (numerical, not the support mask) + nonneg clip, shared
    # across neurons (see _diffusion_kde). occupancy broadcasts over the
    # neuron axis in the support mask below.
    resolved = _resolved_density_mask(env, occupancy_density, bandwidth)
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = np.where(
            resolved,
            spike_density / occupancy_density,
            np.nan,
        )

    firing_rates = np.clip(firing_rates, 0.0, None)
    firing_rates = _mask_outside_support(firing_rates, occupancy, min_occupancy)
    return firing_rates.astype(dtype)


def _gaussian_kde_batch(
    env: Environment,
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

    # Occupancy support shared across neurons (see _gaussian_kde).
    with np.errstate(divide="ignore", invalid="ignore"):
        firing_rates = spike_density / occupancy_density
    firing_rates = _mask_outside_support(firing_rates, occupancy, min_occupancy)
    return firing_rates.astype(dtype)


def _binned_batch(
    env: Environment,
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
    masked = _mask_outside_support(smoothed, occupancy, min_occupancy)
    return masked.astype(dtype)


# =============================================================================
# JAX Implementation Functions
# =============================================================================


def _smooth_rate_map_jax(
    env: Environment,
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
        masked = _mask_outside_support(gated, np.asarray(occupancy), min_occupancy)
        return jnp.asarray(masked, dtype=jnp.float64)

    # For diffusion_kde and gaussian_kde: smooth then normalize.
    if method == "diffusion_kde":
        # Run the smoothing IN JAX via env.diffuse(backend="jax"): counts and
        # occupancy diffuse together in one density-mode pass, no (n, n) kernel.
        stacked = jnp.stack([spike_counts_j, occupancy_j], axis=1)  # (n_bins, 2)
        smoothed = jnp.asarray(
            env.diffuse(stacked, bandwidth=bandwidth, mode="density", backend="jax")
        )
        spike_density = smoothed[:, 0]
        occupancy_density = smoothed[:, 1]
        # Resolution gate (numerical divide guard, not the support mask)
        # + nonneg clip (see _diffusion_kde).
        resolved = _resolved_density_mask_jax(env, occupancy_density, bandwidth)
        firing_rate = jnp.where(
            resolved,
            spike_density / occupancy_density,
            jnp.nan,
        )
        firing_rate = jnp.clip(firing_rate, 0.0, None)
        # Occupancy support (raw seconds), consistent with the NumPy path.
        return _mask_outside_support_jax(firing_rate, occupancy_j, min_occupancy)

    # gaussian_kde: dense Gaussian weight matrix with the same support mask.
    kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
    spike_density = kernel_j @ spike_counts_j
    occupancy_density = kernel_j @ occupancy_j
    return _mask_outside_support_jax(
        spike_density / occupancy_density, occupancy_j, min_occupancy
    )


def _smooth_rate_maps_batch_jax(
    env: Environment,
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
        masked = _mask_outside_support(gated, np.asarray(occupancy), min_occupancy)
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
            env.diffuse(cols, bandwidth=bandwidth, mode="density", backend="jax")
        )
        spike_density = smoothed[:, :n_neurons].T
        occupancy_density = smoothed[:, n_neurons]
        # Resolution gate (numerical divide guard, not the support mask)
        # + nonneg clip (see _diffusion_kde).
        resolved = _resolved_density_mask_jax(env, occupancy_density, bandwidth)
        firing_rates = jnp.where(
            resolved,
            spike_density / occupancy_density,
            jnp.nan,
        )
        firing_rates = jnp.clip(firing_rates, 0.0, None)
        # Occupancy support (raw seconds; occupancy broadcasts over neurons).
        firing_rates = _mask_outside_support_jax(
            firing_rates, occupancy_j, min_occupancy
        )
        return jnp.asarray(firing_rates, dtype=jnp_dtype)

    # gaussian_kde: dense Gaussian weight matrix with the same support mask.
    kernel_j = jnp.asarray(_get_gaussian_kernel(env, bandwidth), dtype=jnp.float64)
    spike_density = spike_counts_j @ kernel_j.T
    occupancy_density = kernel_j @ occupancy_j
    firing_rates = _mask_outside_support_jax(
        spike_density / occupancy_density, occupancy_j, min_occupancy
    )
    return jnp.asarray(firing_rates, dtype=jnp_dtype)
