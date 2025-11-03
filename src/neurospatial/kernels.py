from typing import Literal

import networkx as nx
import numpy as np
import scipy.sparse.linalg
from numpy.typing import NDArray


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
    bin_sizes: NDArray | None = None,
    mode: Literal["transition", "density"] = "transition",
) -> NDArray[np.float64]:
    """
    Computes a diffusion-based kernel for all bins (nodes) of `graph` via
    matrix-exponential of a (possibly volume-corrected) graph-Laplacian.

    Parameters
    ----------
    graph : nx.Graph
        Nodes = bins.  Each edge must have a "distance" attribute (Euclidean length).
    bandwidth_sigma : float
        The Gaussian-bandwidth (σ).  We exponentiate with t = σ^2 / 2.
    bin_sizes : Optional[NDArray], shape (n_bins,)
        If provided, bin_sizes[i] is the physical "area/volume" of node i.
        If not provided, we treat all bins as unit-mass.
    mode : "transition" or "density"
        - "transition":  Return a purely discrete transition-matrix P so that ∑_i P[i,j] = 1.
                         (You do *not* need `bin_sizes` in this mode; if you pass it,
                         it will only be used in the exponent step to form L_vol = M^{-1} L,
                         but the final column-normalization is "sum→1".)
        - "density":     Return a continuous-KDE kernel so that ∑_i [K[i,j] * bin_sizes[i]] = 1.
                         Requires `bin_sizes` ≢ None.  (You exponentiate M^{-1} L, then rescale
                         each column so that its weighted-sum by bin_areas is 1.)

    Returns
    -------
    kernel : np.ndarray, shape (n_bins, n_bins)
        If mode="transition":   each column j sums to 1 (∑_i K[i,j] = 1).
        If mode="density":      each column j integrates to 1 over area
                                 (∑_i K[i,j] * bin_sizes[i] = 1).

    Notes
    -----
    Performance warning: Matrix exponential has O(n³) complexity where n is the
    number of bins. For large environments (>1000 bins), computation may be slow.
    """
    # 1) Validate bandwidth is positive
    if bandwidth_sigma <= 0:
        raise ValueError(f"bandwidth_sigma must be positive (got {bandwidth_sigma}).")

    n_bins = graph.number_of_nodes()

    # 2) Re-compute edge "weight" = exp( - dist^2/(2σ^2) )
    _assign_gaussian_weights_from_distance(graph, bandwidth_sigma)

    # 3) Build unnormalized Laplacian L = D - W
    laplacian = nx.laplacian_matrix(graph, nodelist=range(n_bins), weight="weight")

    # 4) If bin_sizes is given, form M⁻¹ = diag(1/bin_sizes),
    #    then replace L ← M⁻¹ @ L (so we solve du/dt = - M⁻¹ L u).
    if bin_sizes is not None:
        if bin_sizes.shape != (n_bins,):
            raise ValueError(
                f"bin_sizes must have shape ({n_bins},), but got {bin_sizes.shape}."
            )
        mass_inv = np.diag(1.0 / bin_sizes)  # shape = (n_bins, n_bins)
        laplacian = mass_inv @ laplacian  # now L = M⁻¹ (D - W)

    # 5) Exponentiate: kernel = exp( - (σ^2 / 2) * L )
    t = bandwidth_sigma**2 / 2.0
    # expm returns a dense numpy array
    kernel = scipy.sparse.linalg.expm(-t * laplacian)

    # Convert to dense array if it's somehow still sparse
    if hasattr(kernel, "toarray"):
        kernel = kernel.toarray()

    # 6) Clip tiny negative noise to zero
    kernel = np.clip(kernel, a_min=0.0, a_max=None)

    # 7) Final normalization:
    #   - If mode="transition":  ∑_i K[i,j] = 1  (pure discrete)
    #   - If mode="density":     ∑_i [K[i,j] * areas[i]] = 1  (continuous KDE)
    if mode == "transition":
        # Just normalize each column so it sums to 1
        mass_out = kernel.sum(axis=0)  # shape = (n_bins,)
        # scale = 1 / mass_out[j]  (so that ∑_i K[i,j] = 1)
    elif mode == "density":
        if bin_sizes is None:
            raise ValueError("bin_sizes is required when mode='density'.")
        # Compute mass_out[j] = ∑_i [kernel[i,j] * areas[i]]
        # shape = (n_bins,)
        mass_out = (kernel * bin_sizes[:, None]).sum(axis=0)
        # scale[j] = 1 / mass_out[j]  (so that ∑_i [K[i,j]*areas[i]] = 1)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'transition' or 'density'.")

    # Avoid division by zero
    scale = np.where(mass_out == 0.0, 0.0, 1.0 / mass_out)
    kernel_normalized: NDArray[np.float64] = (
        kernel * scale[None, :]
    )  # Broadcast scale across rows

    return kernel_normalized
