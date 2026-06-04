"""Smoothing operations for spatial fields.

This module provides diffusion-based kernel computation and application
for smoothing spatial fields on graphs. The main functions are:

- ``compute_diffusion_kernels``: Compute diffusion kernel from graph structure
- ``apply_kernel``: Apply kernel in forward or adjoint mode

Examples
--------
>>> import networkx as nx
>>> import numpy as np
>>> from neurospatial.ops.smoothing import compute_diffusion_kernels, apply_kernel

Create a simple graph and compute kernel:

>>> graph = nx.path_graph(5)
>>> for u, v in graph.edges():
...     graph.edges[u, v]["distance"] = 1.0
>>> kernel = compute_diffusion_kernels(graph, bandwidth_sigma=1.0, mode="transition")
>>> kernel.shape
(5, 5)

Apply kernel to a field:

>>> field = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
>>> smoothed = apply_kernel(field, kernel, mode="forward")
>>> bool(smoothed[2] < 1.0)  # Original spike reduced
True
"""

from __future__ import annotations

import warnings
from typing import Literal

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

__all__ = [
    "apply_kernel",
    "compute_diffusion_kernels",
]

# Threshold for warning about large kernel computation
# Matrix exponential is O(n³) and dense, so warn for large environments
_LARGE_KERNEL_THRESHOLD = 3000

# Hard ceiling above which kernel computation refuses to run by default.
# The diffusion heat kernel exp(-tL) of a connected graph is mathematically
# dense (every entry > 0) and is built dense, so it costs n_bins**2 * 8 bytes
# of float64 memory. At 20,000 bins that is 20_000**2 * 8 / 1e9 ≈ 3.2 GB for a
# single kernel. Above this ceiling we raise (turning a silent OOM into an
# actionable error) unless the caller passes ``allow_large=True``.
_KERNEL_HARD_LIMIT_BINS = 20_000


def _assign_gaussian_weights_from_distance(
    graph: nx.Graph,
    bandwidth_sigma: float,
) -> None:
    """
    Overwrites each edge's "weight" attribute with
        w_uv = exp( - (distance_uv)^2 / (2 * sigma^2) ).
    Assumes each edge already has "distance" = Euclidean length.
    """
    two_sigma2 = 2.0 * (bandwidth_sigma**2)
    for u, v, data in graph.edges(data=True):
        d = data.get("distance", None)
        if d is None:
            raise KeyError(f"Edge ({u},{v}) has no 'distance' attribute.")
        data["weight"] = float(np.exp(-(d * d) / two_sigma2))


def compute_diffusion_kernels(
    graph: nx.Graph,
    bandwidth_sigma: float,
    *,
    bin_sizes: NDArray | None = None,
    mode: Literal["transition", "density"] = "transition",
    allow_large: bool = False,
) -> NDArray[np.float64]:
    """
    Computes a diffusion-based kernel for all bins (nodes) of `graph` via
    matrix-exponential of a (possibly volume-corrected) graph-Laplacian.

    Parameters
    ----------
    graph : nx.Graph
        Nodes = bins.  Each edge must have a "distance" attribute (Euclidean length).
    bandwidth_sigma : float
        The Gaussian-bandwidth (σ), must be > 0.  We exponentiate with t = σ^2 / 2.
    bin_sizes : ndarray of shape (n_bins,), dtype float64, optional
        If provided, bin_sizes[i] is the physical "area/volume" of node i.
        If not provided, we treat all bins as unit-mass.
    mode : {"transition", "density"}, default="transition"
        - "transition":  Return a purely discrete transition-matrix P so that ∑_i P[i,j] = 1.
                         (You do *not* need `bin_sizes` in this mode; if you pass it,
                         it will only be used in the exponent step to form L_vol = M^{-1} L,
                         but the final column-normalization is "sum→1".)
        - "density":     Return a continuous-KDE kernel so that ∑_i [K[i,j] * bin_sizes[i]] = 1.
                         Requires `bin_sizes` ≢ None.  (You exponentiate M^{-1} L, then rescale
                         each column so that its weighted-sum by bin_areas is 1.)
    allow_large : bool, default=False
        Escape hatch for the high-bin memory gate. By default this function
        **raises** ``MemoryError`` when ``n_bins`` exceeds
        ``_KERNEL_HARD_LIMIT_BINS`` (20,000), because the dense kernel would
        require an ``n_bins**2 * 8`` byte allocation (≈ 3.2 GB at 20,000 bins).
        Pass ``allow_large=True`` to override the gate and attempt the
        allocation anyway (only if you have the RAM). Setting this flag does
        **not** suppress the ``UserWarning`` issued above
        ``_LARGE_KERNEL_THRESHOLD`` (3,000) bins.

    Returns
    -------
    kernel : ndarray of shape (n_bins, n_bins), dtype float64
        Diffusion kernel matrix. Column normalization depends on mode:
        - mode="transition": each column j sums to 1 (∑_i K[i,j] = 1)
        - mode="density": each column j integrates to 1 over area
                         (∑_i K[i,j] * bin_sizes[i] = 1)

    Raises
    ------
    MemoryError
        If ``n_bins > _KERNEL_HARD_LIMIT_BINS`` (20,000) and ``allow_large`` is
        ``False``. The kernel would not fit comfortably in memory; the message
        names ``n_bins``, the estimated dense size in GB, and the escape
        hatches. This check runs *before* the expensive matrix exponential, so
        an over-large request fails fast.

    Notes
    -----
    **Memory cost is O(n²) and unavoidable.** The diffusion heat kernel
    ``exp(-t L)`` of a connected graph is *mathematically dense* — every entry
    is strictly positive, decaying only with graph distance — and SciPy builds
    it dense. The returned matrix therefore occupies

    .. math::
        \\text{bytes} = n\\_bins^2 \\times 8

    of float64 memory. For example a 20,000-bin environment needs
    ``20000**2 * 8 / 1e9 ≈ 3.2 GB`` for a single kernel. There is no sparse or
    truncated form that preserves the numerical result, so this peak cannot be
    avoided while using the dense diffusion kernel.

    Two guard rails bound this cost:

    - **Warn threshold (3,000 bins):** a ``UserWarning`` estimating the GB cost
      is issued when ``n_bins > _LARGE_KERNEL_THRESHOLD``.
    - **Hard gate (20,000 bins):** the call **raises** ``MemoryError`` when
      ``n_bins > _KERNEL_HARD_LIMIT_BINS`` unless ``allow_large=True`` is
      passed. This turns a silent out-of-memory crash into an actionable error.

    Performance: the matrix exponential is also O(n³) in time, so large
    environments are slow as well as memory-hungry.

    **Mitigations for large environments:**

    - Use ``smoothing_method="binned"`` in higher-level encoding functions
      (e.g. :func:`~neurospatial.encoding.spatial.compute_spatial_rate`) to
      avoid building the dense kernel entirely.
    - Reduce the number of bins by increasing ``bin_size`` when constructing
      the environment.
    - For population decoding at scale, the memory-safe paths this release are
      float32 rate maps and the summary decode
      (:func:`~neurospatial.decoding.posterior.decode_position_summary`), which
      avoid materializing a full dense posterior.

    A faster, lower-peak ``expm_multiply`` / Chebyshev rewrite of this kernel is
    a deferred stretch goal and is **not** implemented in this release.
    """
    # 1) Validate bandwidth is positive
    if bandwidth_sigma <= 0:
        raise ValueError(f"bandwidth_sigma must be positive (got {bandwidth_sigma}).")

    n_bins = graph.number_of_nodes()

    # 2) Hard memory gate -- runs BEFORE the expensive expm so an over-large
    #    request fails fast instead of attempting (and crashing on) the dense
    #    n_bins x n_bins float64 allocation.
    if n_bins > _KERNEL_HARD_LIMIT_BINS and not allow_large:
        estimated_gb = n_bins * n_bins * 8 / 1e9
        raise MemoryError(
            f"Refusing to build a diffusion kernel for {n_bins} bins: the heat "
            f"kernel exp(-tL) is dense by construction (every entry > 0), so it "
            f"requires an {n_bins} x {n_bins} float64 matrix "
            f"(~{estimated_gb:.1f} GB) -- O(n^2) memory. This exceeds the "
            f"{_KERNEL_HARD_LIMIT_BINS}-bin safety ceiling. To proceed, either "
            f"reduce the number of bins (increase bin_size), use "
            f"smoothing_method='binned' in the higher-level encoding function, "
            f"or pass allow_large=True to override this gate if you have enough "
            f"RAM."
        )

    # 3) Issue warning for large environments
    if n_bins > _LARGE_KERNEL_THRESHOLD:
        warnings.warn(
            f"Computing diffusion kernel for {n_bins} bins. "
            f"Matrix exponential has O(n³) complexity and O(n²) memory. "
            f"This may be slow and memory-intensive (estimated "
            f"{n_bins * n_bins * 8 / 1e9:.1f} GB for kernel matrix). "
            f"Consider using method='binned' or reducing bin_size to decrease "
            f"the number of bins.",
            UserWarning,
            stacklevel=2,
        )

    # 4) Re-compute edge "weight" = exp( - dist^2/(2σ^2) )
    #    Operate on a copy so the caller's graph (and any "weight" attributes it
    #    relies on) is never mutated as a side effect.
    working_graph = graph.copy()
    _assign_gaussian_weights_from_distance(working_graph, bandwidth_sigma)

    # 5) Build unnormalized Laplacian L = D - W
    laplacian = nx.laplacian_matrix(
        working_graph, nodelist=range(n_bins), weight="weight"
    )

    # 6) If bin_sizes is given, form M⁻¹ = diag(1/bin_sizes),
    #    then replace L ← M⁻¹ @ L (so we solve du/dt = - M⁻¹ L u).
    #    IMPORTANT: Use sparse diagonal matrix to avoid O(n²) dense matrix creation
    if bin_sizes is not None:
        if bin_sizes.shape != (n_bins,):
            raise ValueError(
                f"bin_sizes must have shape ({n_bins},), but got {bin_sizes.shape}."
            )
        # Use scipy.sparse.diags for O(n) memory instead of np.diag's O(n²)
        mass_inv = scipy.sparse.diags(1.0 / bin_sizes, format="csr")
        laplacian = mass_inv @ laplacian  # Sparse @ Sparse = Sparse

    # 7) Exponentiate: kernel = exp( - (σ^2 / 2) * L )
    t = bandwidth_sigma**2 / 2.0
    # expm returns a dense numpy array
    kernel = scipy.sparse.linalg.expm(-t * laplacian)

    # Convert to dense array if it's somehow still sparse
    if hasattr(kernel, "toarray"):
        kernel = kernel.toarray()

    # 8) Clip tiny negative noise to zero
    kernel = np.clip(kernel, a_min=0.0, a_max=None)

    # 9) Final normalization:
    #   - If mode="transition":  ∑_i K[i,j] = 1  (pure discrete)
    #   - If mode="density":     ∑_i [K[i,j] * areas[i]] = 1  (continuous KDE)
    match mode:
        case "transition":
            # Just normalize each column so it sums to 1
            mass_out = kernel.sum(axis=0)  # shape = (n_bins,)
            # scale = 1 / mass_out[j]  (so that ∑_i K[i,j] = 1)
        case "density":
            if bin_sizes is None:
                raise ValueError("bin_sizes is required when mode='density'.")
            # Compute mass_out[j] = ∑_i [kernel[i,j] * areas[i]]
            # shape = (n_bins,)
            mass_out = (kernel * bin_sizes[:, None]).sum(axis=0)
            # scale[j] = 1 / mass_out[j]  (so that ∑_i [K[i,j]*areas[i]] = 1)
        case _:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose 'transition' or 'density'."
            )

    # Avoid division by zero
    scale = np.where(mass_out == 0.0, 0.0, 1.0 / mass_out)
    kernel_normalized: NDArray[np.float64] = (
        kernel * scale[None, :]
    )  # Broadcast scale across rows

    return kernel_normalized


def apply_kernel(
    field: NDArray[np.float64],
    kernel: NDArray[np.float64],
    *,
    mode: Literal["forward", "adjoint"] = "forward",
    bin_sizes: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply diffusion kernel in forward or adjoint mode.

    This function provides a unified interface for applying diffusion kernels
    to fields, with explicit support for adjoint operations. The adjoint is
    essential for likelihood computations, Bayesian inference, and gradient-based
    analyses on spatial fields.

    Parameters
    ----------
    field : NDArray[np.float64], shape (n_bins,)
        Input field to transform.
    kernel : NDArray[np.float64], shape (n_bins, n_bins)
        Diffusion kernel matrix (from compute_kernel or compute_diffusion_kernels).
    mode : "forward" | "adjoint", default="forward"
        Direction of operation:
        - "forward": Standard kernel application (K @ field)
        - "adjoint": Transpose operation with optional mass weighting
    bin_sizes : NDArray[np.float64], shape (n_bins,), optional
        Bin sizes (areas/volumes) for mass-weighted adjoint in density mode.
        Typically obtained from ``env.bin_sizes``.
        - If None: adjoint is simple transpose (K.T @ field)
        - If provided: adjoint is mass-weighted (M^{-1} K.T M @ field)
          where M = diag(bin_sizes)

    Returns
    -------
    result : NDArray[np.float64], shape (n_bins,)
        Transformed field.
        - Forward: result = K @ field
        - Adjoint (no bin_sizes): result = K.T @ field
        - Adjoint (with bin_sizes): result = M^{-1} K.T M @ field

    Raises
    ------
    ValueError
        If mode is not 'forward' or 'adjoint'.
        If field size doesn't match kernel dimensions.
        If bin_sizes size doesn't match field size.
        If bin_sizes has non-positive values (adjoint mode only).
        If kernel is not square.

    Notes
    -----
    **Quick Guide**:

    - **Forward mode**: Standard smoothing/diffusion (result = K @ field)
    - **Adjoint mode**: Used for likelihood calculations in Bayesian inference

      - Without bin_sizes: Use for transition kernels (K.T @ field)
      - With bin_sizes: Use for density kernels (pass ``env.bin_sizes``)

    **When to use adjoint mode**:

    - Computing likelihood fields for spatial decoding
    - Gradient-based optimization on spatial fields
    - Bayesian inference with density representations
    - Backpropagation-style analyses on place fields

    **Forward Mode**:

    Applies the kernel as a linear operator: result = K @ field
    Use this for smoothing, diffusion, or prediction on spatial fields.

    **Adjoint Mode**:

    The adjoint depends on whether bin_sizes is provided:

    - **Without bin_sizes** (transition mode):
      Adjoint is the matrix transpose: result = K.T @ field
      This is the standard adjoint for the Euclidean inner product.
      Use with transition kernels (from ``mode="transition"``).

    - **With bin_sizes** (density mode):
      Adjoint is mass-weighted: result = M^{-1} K.T M @ field
      where M = diag(bin_sizes) is the mass matrix.
      This is the adjoint with respect to the weighted inner product:
      <u, v>_M = sum(u * M * v)
      Use with density kernels (from ``mode="density"``).

    **Mathematical Properties** (optional reading):

    - Inner product preservation (transition):
      <K x, y> = <x, K^T y>

    - Weighted inner product preservation (density):
      <K x, y>_M = <x, K^* y>_M
      where K^* = M^{-1} K^T M is the mass-weighted adjoint

    - Mass conservation (density forward):
      sum((K @ field) * bin_sizes) = sum(field * bin_sizes)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops.smoothing import apply_kernel

    Create environment and kernel:

    >>> data = np.array([[i, j] for i in range(11) for j in range(11)])
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> K = env.compute_kernel(bandwidth=1.0, mode="transition")

    Forward application (smoothing):

    >>> field = np.zeros(env.n_bins)
    >>> field[env.n_bins // 2] = 1.0  # Spike at center
    >>> smoothed = apply_kernel(field, K, mode="forward")
    >>> smoothed.shape
    (36,)

    Adjoint application (for likelihoods):

    >>> observation = np.random.rand(env.n_bins)
    >>> adjoint_result = apply_kernel(observation, K, mode="adjoint")
    >>> adjoint_result.shape
    (36,)

    Density mode with mass weighting:

    >>> bin_sizes = env.bin_sizes
    >>> K_density = env.compute_kernel(bandwidth=1.0, mode="density")
    >>> field_density = np.random.rand(env.n_bins)
    >>> result = apply_kernel(
    ...     field_density, K_density, mode="adjoint", bin_sizes=bin_sizes
    ... )
    >>> result.shape
    (36,)

    See Also
    --------
    compute_diffusion_kernels : Compute diffusion kernel from graph
    Environment.compute_kernel : Compute kernel for environment
    Environment.smooth : Smooth field using kernel (forward mode)
    resample_field : Resample fields across environments (spatial resampling)
    regions_to_mask : Convert regions to bin masks (spatial discretization)

    References
    ----------
    .. [1] Coifman, R. R., & Lafon, S. (2006). Diffusion maps.
           Applied and Computational Harmonic Analysis, 21(1), 5-30.
    .. [2] Grigor'yan, A. (2009). Heat Kernel and Analysis on Manifolds. AMS.

    """
    # Input validation
    if mode not in ("forward", "adjoint"):
        raise ValueError(f"mode must be 'forward' or 'adjoint', got '{mode}'")

    # Check kernel is square
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Kernel must be square, got shape {kernel.shape}")

    n_bins = kernel.shape[0]

    # Check field size
    if field.shape != (n_bins,):
        raise ValueError(
            f"Field size {field.shape[0]} does not match kernel size {n_bins}"
        )

    # Check bin_sizes size and validate positive values if provided
    if bin_sizes is not None:
        if bin_sizes.shape != (n_bins,):
            raise ValueError(
                f"bin_sizes size {bin_sizes.shape[0]} does not match field size {n_bins}. "
                f"bin_sizes should come from env.bin_sizes for the environment."
            )
        # Validate bin_sizes for adjoint mode (division required)
        if mode == "adjoint" and np.any(bin_sizes <= 0):
            raise ValueError(
                f"bin_sizes must have strictly positive values for adjoint mode. "
                f"Found {np.sum(bin_sizes <= 0)} non-positive values. "
                f"Check that env.bin_sizes contains valid bin areas/volumes."
            )

    # Apply kernel based on mode
    if mode == "forward":
        # Standard matrix-vector product
        result = kernel @ field

    else:  # mode == "adjoint"
        if bin_sizes is None:
            # Simple transpose
            result = kernel.T @ field
        else:
            # Mass-weighted adjoint: M^{-1} K^T M @ field
            # First apply mass matrix: M @ field
            m_field = bin_sizes * field

            # Then apply transpose: K^T @ (M @ field)
            kt_m_field = kernel.T @ m_field

            # Finally apply inverse mass matrix: M^{-1} @ (K^T @ M @ field)
            result = kt_m_field / bin_sizes

    return result
