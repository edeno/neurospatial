"""Smoothing operations for spatial fields.

This module provides the low-level diffusion-kernel primitive and kernel
application for smoothing spatial fields on graphs. The main functions are:

- ``compute_diffusion_kernels``: assemble a finite-volume diffusion kernel from
  a working graph carrying ``"A"`` (shared-face measure) and ``"distance"``
  (center-to-center) on each edge, plus a node-ordered ``volumes`` array.
- ``apply_kernel``: apply a kernel in forward or adjoint mode.

The operator itself (``H = exp(-t L)``, ``L = M^-1 (D - W)``,
``W[i, j] = A[i, j] / d[i, j]``) lives in
:mod:`neurospatial.ops.diffusion`; the per-geometry face measures and the
``Environment``-level dispatch live there too. This module is the graph-level
seam: given ``"A"`` already on the edges, it builds ``W`` and normalizes the
kernel to the requested ``mode``.

Examples
--------
>>> import networkx as nx
>>> import numpy as np
>>> from neurospatial.ops.smoothing import compute_diffusion_kernels, apply_kernel

Create a simple graph (unit face measure, unit spacing) and compute a kernel:

>>> graph = nx.path_graph(5)
>>> for u, v in graph.edges():
...     graph.edges[u, v]["distance"] = 1.0
...     graph.edges[u, v]["A"] = 1.0
>>> volumes = np.ones(5)
>>> kernel = compute_diffusion_kernels(
...     graph, volumes=volumes, sigma=1.0, mode="transition"
... )
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

from neurospatial.ops.diffusion import heat_kernel_from_W

__all__ = [
    "apply_kernel",
    "compute_diffusion_kernels",
]

# Threshold for warning about large kernel computation
# Matrix exponential is O(n³) and dense, so warn for large environments. This
# is the SOLE high-bin guard: there is no hard limit -- a large request always
# warns (with a GB estimate) and proceeds.
_LARGE_KERNEL_THRESHOLD = 3000


def compute_diffusion_kernels(
    graph: nx.Graph,
    *,
    volumes: NDArray[np.float64],
    sigma: float,
    mode: Literal["transition", "density", "average"],
) -> NDArray[np.float64]:
    """Assemble a finite-volume diffusion kernel from a face-measure graph.

    Builds the finite-volume weight matrix ``W[i, j] = A[i, j] / d[i, j]`` from
    the graph's edge attributes and returns the requested normalized view of the
    heat operator ``H = exp(-t L)``, ``L = M^-1 (D - W)``, ``t = sigma^2 / 2``.
    ``bandwidth`` is thereby the true physical standard deviation (sigma) of the
    smoothing on any K-orthogonal layout, independent of bin size.

    This is the low-level primitive. Most callers should use
    :meth:`Environment.compute_kernel` / :meth:`Environment.smooth`, which
    resolve the per-geometry face measures via
    :func:`neurospatial.ops.diffusion.diffusion_kernel`.

    Parameters
    ----------
    graph : nx.Graph
        Nodes are bins and MUST be contiguous integer labels ``0..n-1`` (they
        are used directly as sparse-matrix indices). Each edge MUST carry:

        - ``"distance"``: center-to-center distance ``d`` (finite, > 0).
        - ``"A"``: measure of the face shared by the two bins (finite, >= 0).
          An explicit ``A == 0`` means no diffusion across that edge (e.g.
          corner-touching Cartesian bins); a **missing** ``"A"`` raises.
    volumes : NDArray[np.float64], shape (n_bins,)
        Per-bin cell volumes ``M``, node-ordered ``0..n-1``. Every entry must be
        finite and strictly positive (the operator divides by ``volumes``).
    sigma : float
        Physical smoothing standard deviation. Must be finite and > 0 (it feeds
        ``sigma**2`` into the matrix exponential).
    mode : {"transition", "density", "average"}
        Normalized view of the operator:

        - ``"transition"``: ``Hᵀ`` column-stochastic (``sum_i K[i, j] = 1``);
          mass-conserving smoothing of an extensive field (``K @ counts``).
        - ``"density"``: ``H·M⁻¹`` (``sum_i M_i K[i, j] = 1``); count -> density.
        - ``"average"``: ``H`` row-stochastic (``sum_j K[i, j] = 1``); averages
          an intensive field (``K @ rate``).

    Returns
    -------
    kernel : NDArray[np.float64], shape (n_bins, n_bins)
        Dense diffusion kernel, normalized per ``mode``.

    Raises
    ------
    ValueError
        If ``sigma`` is non-finite or ``<= 0``; if ``volumes`` has the wrong
        shape or any non-finite / non-positive entry; if node labels are not
        contiguous integers ``0..n-1`` (float or bool labels are rejected); if
        any edge is missing ``"A"`` or ``"distance"``, or has a non-finite /
        negative ``"A"`` or a non-finite / non-positive ``"distance"``; or if
        ``mode`` is invalid.

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

    **There is no hard limit — high-bin kernels warn and proceed.** A
    ``UserWarning`` estimating the GB cost is issued when
    ``n_bins > _LARGE_KERNEL_THRESHOLD`` (3,000 bins); the kernel is then built
    regardless of size. The warning names ``n_bins``, the GB estimate, the
    dense O(n²) reason, and the mitigation (reduce bins by increasing
    ``bin_size``). It is the sole high-bin guard.

    Performance: the matrix exponential is also O(n³) in time, so large
    environments are slow as well as memory-hungry.

    **Mitigations for large environments:**

    - Reduce the number of bins by increasing ``bin_size`` when constructing
      the environment. (Every ``smoothing_method`` -- ``diffusion_kde``,
      ``gaussian_kde``, and ``binned`` -- builds a dense kernel, so switching
      method is not a memory mitigation.)
    - For population decoding at scale, the memory-safe paths this release are
      float32 rate maps and the summary decode
      (:func:`~neurospatial.decoding.posterior.decode_position_summary`), which
      avoid materializing a full dense posterior.

    A faster, lower-peak ``expm_multiply`` / Chebyshev rewrite of this kernel is
    a deferred stretch goal and is **not** implemented in this release.
    """
    # 1) Validate mode and sigma up front, before any O(n^2)/O(n^3) work (the
    #    high-bin warning, W assembly, or the dense matrix exponential): a typo
    #    on a large graph must fail fast, not after building the kernel.
    if mode not in ("transition", "density", "average"):
        raise ValueError(
            f"Invalid mode {mode!r}. Choose 'transition', 'density', or 'average'."
        )
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be finite and > 0, got {sigma}.")

    n_bins = graph.number_of_nodes()

    # 2) Validate node labels: contiguous integers 0..n-1 (used directly as
    #    sparse-matrix indices). bool subclasses int and 0.0 == 0, so require
    #    true integer labels AND the full contiguous 0..n-1 set.
    nodes = list(graph.nodes)
    if not all(
        isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in nodes
    ) or {int(x) for x in nodes} != set(range(n_bins)):
        raise ValueError(
            "graph nodes must be contiguous integer labels 0..n-1 (no float/bool)."
        )

    # 3) Validate volumes: correct shape, finite and strictly positive (the
    #    operator divides by volumes, so a zero/negative/NaN must fail loudly).
    volumes = np.asarray(volumes, dtype=np.float64)
    if volumes.shape != (n_bins,):
        raise ValueError(
            f"volumes must have shape ({n_bins},), but got {volumes.shape}."
        )
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError(
            "volumes must be finite and strictly positive (per-bin cell volume)."
        )

    # 4) Warn (never raise) for large environments. The dense heat kernel
    #    exp(-tL) costs n_bins**2 * 8 bytes of float64 memory; we estimate that
    #    and proceed. There is no hard limit -- the call always returns a kernel.
    if n_bins > _LARGE_KERNEL_THRESHOLD:
        estimated_gb = n_bins * n_bins * 8 / 1e9
        warnings.warn(
            f"Computing a dense diffusion kernel for {n_bins} bins. The heat "
            f"kernel exp(-tL) is dense by construction (every entry > 0), so it "
            f"requires an {n_bins} x {n_bins} float64 matrix "
            f"(~{estimated_gb:.1f} GB) -- O(n^2) memory (and O(n^3) time for the "
            f"matrix exponential). Proceeding anyway; this may be slow and "
            f"memory-intensive. To reduce the cost, increase bin_size (fewer "
            f"bins). Every smoothing_method (diffusion_kde, gaussian_kde, "
            f"binned) builds a dense kernel, so switching method does not avoid "
            f"this cost.",
            UserWarning,
            stacklevel=2,
        )

    # 5) Build the finite-volume weight matrix W[i, j] = A[i, j] / d[i, j].
    #    "A" and "distance" are both edge-contract fields; a missing or invalid
    #    value raises rather than silently degrading the kernel.
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, data in graph.edges(data=True):
        if "A" not in data or "distance" not in data:
            raise ValueError(
                f"edge ({u},{v}) is missing 'A' and/or 'distance' attribute."
            )
        A = float(data["A"])
        d = float(data["distance"])
        if not (np.isfinite(A) and A >= 0.0):
            raise ValueError(f"edge ({u},{v}) has invalid face measure A={A}.")
        if not (np.isfinite(d) and d > 0.0):
            raise ValueError(f"edge ({u},{v}) has invalid distance d={d}.")
        if A == 0.0:
            continue  # explicit A == 0 => no diffusion across this edge
        w = A / d
        rows += [int(u), int(v)]
        cols += [int(v), int(u)]
        vals += [w, w]
    W = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n_bins, n_bins), dtype=np.float64
    )

    # 6) Assemble and normalize the heat operator for the requested mode.
    return heat_kernel_from_W(W, volumes, sigma, mode=mode)


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

    - Count-to-density conservation (density forward): a density kernel maps an
      extensive input (counts) to a density whose integral under bin volumes
      equals the input total,
      sum((K @ field) * bin_sizes) = sum(field)
      (each column integrates to 1 under bin volumes: sum_i K[i, j] * M_i = 1).

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
