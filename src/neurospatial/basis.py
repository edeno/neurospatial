"""
Maze-aware spatial basis functions for regression.

This module provides localized basis functions that respect environment
topology (walls, corridors) for use in GLMs and other spatial models.

Which Basis Should I Use?
-------------------------
**First time / Simple place cell models** → `geodesic_rbf_basis`
    Start here. Works like familiar Gaussian kernels but respects walls.
    Fast computation. Easy to visualize and interpret.
    Good for: GLMs, simple decoding models, place field decomposition.

**Multi-scale spatial structure** → `heat_kernel_wavelet_basis`
    Captures both local bumps and compartment-level structure simultaneously.
    Smoother than RBF. Based on diffusion physics.
    Good for: Environments with rooms/corridors, hierarchical place cells.
    Cost: 2-3× slower than RBF (still fast for typical environments).

**Very large environments (>10,000 bins)** → `chebyshev_filter_basis`
    Fastest computation (no distance matrix). Strict k-hop locality.
    Good for: Large arenas, online/real-time applications.
    Tradeoff: Less intuitive parameterization (degree vs spatial extent).

**Not sure?** Start with `geodesic_rbf_basis(env, n_centers=50, sigma=10.0)`.
You can always switch later—all three work the same way in GLMs.

**Want automatic parameter selection?** Use `spatial_basis()`:
    >>> basis = spatial_basis(env)  # Automatic, sensible defaults  # doctest: +SKIP

Scale Parameters Across Bases
------------------------------
The three basis types use different scale parameters:

- geodesic_rbf_basis(sigma=...): RBF bandwidth in environment units (cm)
- heat_kernel_wavelet_basis(scales=...): Diffusion times (proportional to sigma²)
- chebyshev_filter_basis(max_degree=...): How many bins away from center the
  function extends (degree 5 = nonzero within 5 bins of center)

Rough equivalence for spatial extent:
    sigma ≈ sqrt(2*scale) * bin_size ≈ max_degree * bin_size

For example, with 2cm bins and 10cm desired spatial extent:
    sigma=10.0  or  scales=[12.5]  or  max_degree=5

Quick Start
-----------
>>> from neurospatial import Environment  # doctest: +SKIP
>>> from neurospatial.basis import geodesic_rbf_basis  # doctest: +SKIP
>>>
>>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
>>> basis = geodesic_rbf_basis(env, n_centers=50, sigma=5.0)  # doctest: +SKIP
>>>
>>> # Create GLM design matrix
>>> bin_indices = env.bin_sequence(trajectory, times)  # doctest: +SKIP
>>> X = basis[:, bin_indices].T  # Shape: (n_times, n_basis)  # doctest: +SKIP

Full GLM Workflow
-----------------
>>> from neurospatial import Environment  # doctest: +SKIP
>>> from neurospatial.basis import geodesic_rbf_basis  # doctest: +SKIP
>>> import numpy as np  # doctest: +SKIP
>>>
>>> # 1. Create environment and basis
>>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
>>> basis = geodesic_rbf_basis(env, n_centers=50, sigma=[5.0, 10.0])  # doctest: +SKIP
>>>
>>> # 2. Create design matrix from trajectory
>>> bin_indices = env.bin_sequence(trajectory, times)  # doctest: +SKIP
>>> X_spatial = basis[:, bin_indices].T  # (n_times, n_basis)  # doctest: +SKIP
>>>
>>> # 3. Combine with other features (optional)
>>> speed = compute_speed(positions, times)  # doctest: +SKIP
>>> X = np.column_stack([X_spatial, speed[:, None]])  # doctest: +SKIP
>>>
>>> # 4. Fit GLM (example with statsmodels)
>>> import statsmodels.api as sm  # doctest: +SKIP
>>> X = sm.add_constant(X)  # doctest: +SKIP
>>> model = sm.GLM(spike_counts, X, family=sm.families.Poisson())  # doctest: +SKIP
>>> result = model.fit()  # doctest: +SKIP
>>>
>>> # 5. Visualize fitted place field
>>> beta_spatial = result.params[
...     1 : len(basis) + 1
... ]  # Spatial coefficients  # doctest: +SKIP
>>> place_field = beta_spatial @ basis  # Project back to space  # doctest: +SKIP
>>> env.plot(place_field, title="Fitted Place Field")  # doctest: +SKIP

References
----------
.. [1] Coifman, R.R. & Maggioni, M. (2006). Diffusion wavelets.
.. [2] Hammond, D.K. et al. (2011). Wavelets on graphs via spectral graph theory.
.. [3] Defferrard, M. et al. (2016). Convolutional neural networks on graphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial import Environment

__all__ = [
    "chebyshev_filter_basis",
    "geodesic_rbf_basis",
    "heat_kernel_wavelet_basis",
    "plot_basis_functions",
    "select_basis_centers",
    "spatial_basis",
]


def select_basis_centers(
    env: Environment,
    n_centers: int,
    method: Literal["kmeans", "farthest_point", "random", "grid"] = "kmeans",
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """
    Select well-distributed center nodes for basis functions.

    Parameters
    ----------
    env : Environment
        Fitted environment with bin_centers.
    n_centers : int
        Number of centers to select.

        **Rule of thumb**: n_centers ≈ (environment area) / (sigma²)

        - For 100cm × 100cm maze with sigma=10cm: n_centers ≈ 100
        - For 50cm linear track with sigma=5cm: n_centers ≈ 10

        Too few centers: May miss local spatial structure
        Too many centers: Redundant coverage, overfitting risk

        Start conservative (30-50) and increase if place field
        reconstructions are too smooth.
    method : {"kmeans", "farthest_point", "random", "grid"}
        Selection strategy:

        - "kmeans": K-means clustering on bin_centers, return nearest bins
          to cluster centroids. Good general-purpose choice.
        - "farthest_point": Greedy farthest-point sampling using geodesic
          distance. Guarantees maximum spread but O(n_centers * n_bins).
        - "random": Uniform random selection. Fast but may cluster.
        - "grid": Regular grid subsampling (only for grid-based environments).
    random_state : int, Generator, or None
        Random state for reproducibility (kmeans, random methods).

    Returns
    -------
    centers : NDArray[np.int_], shape (n_centers,)
        Node indices of selected centers.

    Raises
    ------
    ValueError
        If n_centers > n_bins.
        If n_centers < 1.
        If method="grid" but environment is not grid-based.

    Examples
    --------
    >>> centers = select_basis_centers(
    ...     env, n_centers=50, method="kmeans"
    ... )  # doctest: +SKIP
    >>> basis = geodesic_rbf_basis(env, centers=centers, sigma=5.0)  # doctest: +SKIP
    """
    # Input validation
    if n_centers < 1:
        raise ValueError(f"n_centers must be at least 1, got {n_centers}.")

    if n_centers > env.n_bins:
        raise ValueError(
            f"Cannot select {n_centers} centers from environment "
            f"with only {env.n_bins} bins.\n"
            f"\n"
            f"Options:\n"
            f"  1. Reduce n_centers to {env.n_bins} or fewer\n"
            f"  2. Decrease bin_size when creating environment (more bins)\n"
            f"\n"
            f"Your environment has {env.n_bins} bins covering "
            f"{env.n_dims}D space."
        )

    # Handle random state
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if method == "kmeans":
        return _select_centers_kmeans(env, n_centers, rng)
    elif method == "farthest_point":
        return _select_centers_farthest_point(env, n_centers, rng)
    elif method == "random":
        return _select_centers_random(env, n_centers, rng)
    elif method == "grid":
        return _select_centers_grid(env, n_centers)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Expected one of: 'kmeans', 'farthest_point', 'random', 'grid'."
        )


def _select_centers_kmeans(
    env: Environment,
    n_centers: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Select centers using k-means clustering on bin_centers."""
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans

    # Run k-means on bin centers
    kmeans = KMeans(
        n_clusters=n_centers,
        random_state=int(rng.integers(0, 2**31)),
        n_init=10,
    )
    kmeans.fit(env.bin_centers)

    # Find nearest bin to each cluster centroid
    tree = cKDTree(env.bin_centers)
    _, centers = tree.query(kmeans.cluster_centers_)

    return np.asarray(centers, dtype=np.int_)


def _select_centers_random(
    env: Environment,
    n_centers: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Select centers uniformly at random."""
    centers = rng.choice(env.n_bins, size=n_centers, replace=False)
    return np.asarray(centers, dtype=np.int_)


def _select_centers_farthest_point(
    env: Environment,
    n_centers: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Select centers using greedy farthest-point sampling with geodesic distance."""
    import networkx as nx
    from scipy.sparse.csgraph import shortest_path

    # Get adjacency matrix for shortest path computation
    adjacency = nx.to_scipy_sparse_array(
        env.connectivity, weight="distance", format="csr"
    )

    # Start with a random point
    centers = [int(rng.integers(0, env.n_bins))]

    # Compute distances from first center
    distances_from_centers = shortest_path(
        csgraph=adjacency,
        method="auto",
        directed=False,
        indices=centers[0],
    )

    # Greedy selection: iteratively add the farthest point
    for _ in range(n_centers - 1):
        # Find bin with maximum minimum distance to current centers
        farthest_idx = np.argmax(distances_from_centers)
        centers.append(int(farthest_idx))

        # Update distances: compute distance from new center
        new_distances = shortest_path(
            csgraph=adjacency,
            method="auto",
            directed=False,
            indices=farthest_idx,
        )

        # Keep minimum distance to any center
        distances_from_centers = np.minimum(distances_from_centers, new_distances)

    return np.asarray(centers, dtype=np.int_)


def _select_centers_grid(
    env: Environment,
    n_centers: int,
) -> NDArray[np.int_]:
    """Select centers using regular grid subsampling."""
    # For now, raise NotImplementedError - grid method requires
    # knowing the grid structure which not all environments have
    raise NotImplementedError(
        "Grid method for center selection is not yet implemented.\n"
        "\n"
        "Use one of the available methods:\n"
        "  - 'kmeans': Good general-purpose choice\n"
        "  - 'farthest_point': Maximizes spatial spread\n"
        "  - 'random': Fast but may cluster"
    )


def _normalize_basis(
    basis: NDArray[np.float64],
    mode: Literal["unit", "max", "none"],
) -> NDArray[np.float64]:
    """
    Normalize basis functions.

    Parameters
    ----------
    basis : NDArray, shape (n_basis, n_bins)
        Unnormalized basis.
    mode : {"unit", "max", "none"}
        - "unit": L2 norm = 1 for each basis function
        - "max": Max value = 1 for each basis function
        - "none": No normalization

    Returns
    -------
    NDArray, shape (n_basis, n_bins)
        Normalized basis.
    """
    if mode == "none":
        return basis

    if mode == "unit":
        norms = np.linalg.norm(basis, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        result: NDArray[np.float64] = basis / norms
        return result

    if mode == "max":
        maxes = np.max(np.abs(basis), axis=1, keepdims=True)
        maxes = np.where(maxes == 0, 1.0, maxes)
        result_max: NDArray[np.float64] = basis / maxes
        return result_max

    raise ValueError(f"Unknown normalization mode: {mode!r}")


def geodesic_rbf_basis(
    env: Environment,
    centers: NDArray[np.int_] | None = None,
    sigma: float | Sequence[float] = 5.0,
    n_centers: int | None = None,
    center_method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
    normalize: Literal["unit", "max", "none"] = "unit",
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Compute maze-aware RBF basis using geodesic distances.

    For each center c and bandwidth sigma, computes:
        B(j) = exp(-d_geo(j, c)^2 / (2 * sigma^2))

    where d_geo is the shortest-path distance respecting walls.

    Parameters
    ----------
    env : Environment
        Fitted environment.
    centers : NDArray[np.int_] or None
        Node indices for basis centers. If None, auto-select using
        n_centers and center_method.
    sigma : float or sequence of float
        RBF bandwidth in environment units (typically cm).
        Controls spatial spread of each basis function.

        **Rule of thumb**: sigma ≈ (desired coverage diameter) / 2

        - sigma=5.0cm: Localized to ~10cm diameter (covers 2*sigma)
        - sigma=20.0cm: Broad coverage, ~40cm diameter

        Multiple values create multi-scale basis (recommended):
        Example: [5.0, 10.0, 20.0] for 10cm, 20cm, 40cm coverage.
    n_centers : int or None
        Number of centers for auto-selection. Required if centers is None.
        See `select_basis_centers` for guidance on choosing n_centers.
    center_method : {"kmeans", "farthest_point", "random"}
        Method for auto center selection.
    normalize : {"unit", "max", "none"}, default="unit"
        Normalization mode:

        - "unit": L2 norm = 1 (RECOMMENDED for GLMs)
          Ensures all basis functions contribute equally regardless
          of spatial extent. Prevents bias toward larger-scale features.
        - "max": Max value = 1
          Useful for visualization (all functions on same color scale).
          May bias GLM toward small-scale features (they're "peakier").
        - "none": Raw RBF values
          Only use if you're normalizing differently downstream.
    random_state : int, Generator, or None
        Random state for center selection.

    Returns
    -------
    basis : NDArray[np.float64], shape (n_centers * n_sigmas, n_bins)
        Basis matrix with rows ordered by (center, sigma):

        - basis[0:n_sigmas] = all scales for center 0
        - basis[n_sigmas:2*n_sigmas] = all scales for center 1
        - ...

        Example with n_centers=3, sigma=[5, 10]:
            Row 0: center 0, sigma=5
            Row 1: center 0, sigma=10
            Row 2: center 1, sigma=5
            Row 3: center 1, sigma=10
            Row 4: center 2, sigma=5
            Row 5: center 2, sigma=10

        To create GLM design matrix from trajectory:
        >>> bin_indices = env.bin_sequence(trajectory, times)  # doctest: +SKIP
        >>> X = basis[:, bin_indices].T  # (n_timepoints, n_basis)  # doctest: +SKIP

    Raises
    ------
    ValueError
        If sigma contains non-positive values.
        If centers is None and n_centers is None.
        If environment graph is disconnected and some bins are unreachable.

    Notes
    -----
    **Why geodesic instead of Euclidean?**

    In a maze, two points may be spatially close but far apart in terms
    of navigation (separated by a wall). Geodesic distance captures the
    actual path length, so basis functions don't "leak" through walls.

    **Computational complexity**:

    - Time: O(n_centers * n_bins * log(n_bins)) using batch shortest paths
    - Space: O(n_centers * n_sigmas * n_bins) for output basis

    Typical runtimes (2023 MacBook Pro):

    - 50cm × 50cm maze, 2cm bins, 50 centers: ~0.1s
    - 100cm × 100cm maze, 2cm bins, 100 centers: ~1.0s
    - 200cm × 200cm maze, 1cm bins, 200 centers: ~30s

    For very large environments (>10,000 bins), consider:

    - Using fewer centers (n_centers=30 instead of 100)
    - Larger bin_size to reduce n_bins
    - Switching to chebyshev_filter_basis (no distance matrix)

    Examples
    --------
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.basis import geodesic_rbf_basis  # doctest: +SKIP
    >>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
    >>>
    >>> # Single scale
    >>> basis = geodesic_rbf_basis(env, n_centers=50, sigma=5.0)  # doctest: +SKIP
    >>> basis.shape  # doctest: +SKIP
    (50, n_bins)
    >>>
    >>> # Multi-scale (recommended)
    >>> basis = geodesic_rbf_basis(
    ...     env, n_centers=30, sigma=[5.0, 10.0, 20.0]
    ... )  # doctest: +SKIP
    >>> basis.shape  # doctest: +SKIP
    (90, n_bins)  # 30 centers * 3 scales

    See Also
    --------
    heat_kernel_wavelet_basis : Diffusion-based alternative
    chebyshev_filter_basis : Strictly hop-local alternative
    select_basis_centers : Manual center selection
    """
    import networkx as nx
    from scipy.sparse.csgraph import shortest_path

    # Handle centers
    if centers is None:
        if n_centers is None:
            raise ValueError(
                "Must specify basis function locations.\n"
                "\n"
                "Two options:\n"
                "  1. Provide centers: specific bin indices for basis centers\n"
                "     Example: centers=np.array([5, 10, 15, 20])\n"
                "\n"
                "  2. Provide n_centers: auto-select centers using k-means\n"
                "     Example: n_centers=50\n"
                "     (Recommended for most use cases)\n"
                "\n"
                "See select_basis_centers() for manual center selection."
            )
        centers = select_basis_centers(
            env, n_centers, method=center_method, random_state=random_state
        )
    centers = np.asarray(centers, dtype=np.int_)

    # Handle sigma and validate
    sigmas = np.atleast_1d(sigma).astype(np.float64)
    if np.any(sigmas <= 0):
        invalid = sigmas[sigmas <= 0]
        raise ValueError(
            f"sigma values must be positive, got: {invalid}.\n"
            "\n"
            "sigma controls spatial spread of basis functions:\n"
            "  - sigma=5.0: ~10cm diameter coverage\n"
            "  - sigma=10.0: ~20cm diameter coverage\n"
            "\n"
            "Rule of thumb: sigma ≈ (desired coverage diameter) / 2"
        )

    n_centers_actual = len(centers)
    n_sigmas = len(sigmas)
    n_bins = env.n_bins

    # Compute distances from all centers at once using scipy's optimized implementation
    # This is ~5-10x faster than per-center Dijkstra
    adjacency = nx.to_scipy_sparse_array(
        env.connectivity, weight="distance", format="csr"
    )
    distances = shortest_path(
        csgraph=adjacency,
        method="auto",
        directed=False,
        indices=centers,  # Only compute from these sources
    )
    # Shape: (n_centers, n_bins)

    # Check for disconnected graph (unreachable bins)
    if np.any(np.isinf(distances)):
        n_inf = np.sum(np.isinf(distances))
        n_affected_centers = np.sum(np.any(np.isinf(distances), axis=1))
        raise ValueError(
            f"Found {n_inf} unreachable (center, bin) pairs.\n"
            f"{n_affected_centers} of {n_centers_actual} centers have unreachable bins.\n"
            "\n"
            "This usually means the environment graph is disconnected.\n"
            "\n"
            "To diagnose:\n"
            "  >>> import networkx as nx\n"
            "  >>> nx.is_connected(env.connectivity)\n"
            "  >>> list(nx.connected_components(env.connectivity))\n"
            "\n"
            "To fix, ensure all bins are reachable when creating the environment."
        )

    # Compute RBF for each (center, sigma) combination
    # Rows ordered by (center, sigma): all sigmas for center 0, then center 1, etc.
    # Shape: (n_centers * n_sigmas, n_bins)
    basis = np.zeros((n_centers_actual * n_sigmas, n_bins), dtype=np.float64)

    for c_idx in range(n_centers_actual):
        for s_idx, s in enumerate(sigmas):
            row_idx = c_idx * n_sigmas + s_idx
            two_sigma_sq = 2.0 * s * s
            basis[row_idx] = np.exp(-(distances[c_idx] ** 2) / two_sigma_sq)

    # Normalize
    basis = _normalize_basis(basis, normalize)

    return basis


def heat_kernel_wavelet_basis(
    env: Environment,
    centers: NDArray[np.int_] | None = None,
    scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    n_centers: int | None = None,
    center_method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
    normalize: Literal["unit", "max", "none"] = "unit",
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Compute heat kernel wavelet basis via graph diffusion.

    For each center c and diffusion time s, computes:
        B(j) = [exp(-s * L)]_{j,c}

    where L is the graph Laplacian. This represents heat diffusion from
    center c for time s, respecting the graph topology.

    Parameters
    ----------
    env : Environment
        Fitted environment with connectivity graph.
    centers : NDArray[np.int_] or None
        Node indices for wavelet centers. If None, auto-select.
    scales : sequence of float, default=(0.5, 1.0, 2.0, 4.0)
        Diffusion time scales in graph-intrinsic units.
        Larger values = wider spatial spread.

        Default values (0.5, 1, 2, 4) provide octave-spaced multi-scale
        coverage similar to wavelets. For most environments:

        - scale=0.5: ~1-2 bins radius (very localized)
        - scale=1.0: ~2-3 bins radius
        - scale=2.0: ~3-5 bins radius
        - scale=4.0: ~5-8 bins radius (compartment-level)

        Adjust if your bin_size is unusual:

        - Finer bins (bin_size < 1cm): increase scales 2-4×
        - Coarser bins (bin_size > 5cm): decrease scales 2-4×

        **Relationship to RBF sigma**: Roughly scale ≈ sigma² / (2 * bin_size²)
    n_centers : int or None
        Number of centers for auto-selection.
    center_method : {"kmeans", "farthest_point", "random"}
        Method for auto center selection.
    normalize : {"unit", "max", "none"}, default="unit"
        Normalization mode (see geodesic_rbf_basis for details).
    random_state : int, Generator, or None
        Random state for center selection.

    Returns
    -------
    basis : NDArray[np.float64], shape (n_centers * n_scales, n_bins)
        Wavelet basis matrix. Rows ordered by (center, scale):
        all scales for center 0, then all scales for center 1, etc.

    Raises
    ------
    ValueError
        If scales contains non-positive values.
        If centers is None and n_centers is None.

    Notes
    -----
    **Physical interpretation**: Each basis function represents how heat
    would spread from a point source at the center after diffusion time s.
    Walls block heat flow; narrow corridors slow it down.

    **Multi-scale behavior**:

    - Small s (e.g., 0.5): Highly localized bumps, captures local structure
    - Large s (e.g., 4.0): Broad bumps, captures compartment-level structure

    **Computational complexity**: O(n_scales * n_centers * n_bins) using
    scipy's expm_multiply, which avoids forming the full matrix exponential.
    About 2-3× slower than geodesic RBF for typical environments.

    **Laplacian weighting**: Uses edge 'distance' weights in the Laplacian.
    For physical diffusion interpretation, these represent edge conductance
    (inverse resistance). If uniform diffusion is preferred, the Laplacian
    can be computed without weights.

    **Comparison with geodesic RBF**:

    - Heat kernel: Smooth transition, infinite but decaying support
    - Geodesic RBF: Gaussian profile along shortest paths

    Both respect maze topology; heat kernel is more "diffuse".

    Examples
    --------
    >>> basis = heat_kernel_wavelet_basis(
    ...     env, n_centers=30, scales=[0.5, 1, 2, 4]
    ... )  # doctest: +SKIP
    >>> basis.shape  # doctest: +SKIP
    (120, n_bins)  # 30 centers * 4 scales

    References
    ----------
    .. [1] Coifman, R.R. & Maggioni, M. (2006). Diffusion wavelets.
           Applied and Computational Harmonic Analysis, 21(1), 53-94.
    .. [2] Hammond, D.K. et al. (2011). Wavelets on graphs via spectral
           graph theory. Applied and Computational Harmonic Analysis, 30(2), 129-150.
    """
    import networkx as nx
    from scipy.sparse.linalg import expm_multiply

    # Handle centers
    if centers is None:
        if n_centers is None:
            raise ValueError(
                "Must specify basis function locations.\n"
                "\n"
                "Two options:\n"
                "  1. Provide centers: specific bin indices for wavelet centers\n"
                "     Example: centers=np.array([5, 10, 15, 20])\n"
                "\n"
                "  2. Provide n_centers: auto-select centers using k-means\n"
                "     Example: n_centers=50\n"
                "\n"
                "See select_basis_centers() for manual center selection."
            )
        centers = select_basis_centers(
            env, n_centers, method=center_method, random_state=random_state
        )
    centers = np.asarray(centers, dtype=np.int_)

    # Validate scales
    scales_arr = np.asarray(scales, dtype=np.float64)
    if np.any(scales_arr <= 0):
        invalid = scales_arr[scales_arr <= 0]
        raise ValueError(
            f"scales values must be positive, got: {invalid}.\n"
            "\n"
            "scales controls diffusion time (spatial spread):\n"
            "  - scale=0.5: ~1-2 bins radius\n"
            "  - scale=2.0: ~3-5 bins radius\n"
            "  - scale=4.0: ~5-8 bins radius\n"
            "\n"
            "Default (0.5, 1.0, 2.0, 4.0) works well for most environments."
        )

    n_centers_actual = len(centers)
    n_scales = len(scales_arr)
    n_bins = env.n_bins

    # Get graph Laplacian (sparse)
    laplacian = nx.laplacian_matrix(
        env.connectivity, nodelist=range(n_bins), weight="distance"
    )

    # Create delta vectors at centers
    # Shape: (n_bins, n_centers)
    delta_mat = np.zeros((n_bins, n_centers_actual), dtype=np.float64)
    delta_mat[centers, np.arange(n_centers_actual)] = 1.0

    # Compute heat kernel wavelets using expm_multiply
    # Rows ordered by (center, scale): all scales for center 0, then center 1, etc.
    # Shape: (n_centers * n_scales, n_bins)
    basis = np.zeros((n_centers_actual * n_scales, n_bins), dtype=np.float64)

    # Compute exp(-s * L) @ delta for each scale
    # expm_multiply computes exp(A) @ B efficiently without forming exp(A)
    for s_idx, s in enumerate(scales_arr):
        wavelets = expm_multiply(
            -s * laplacian, delta_mat
        )  # Shape: (n_bins, n_centers)
        # Place in output array with (center, scale) ordering
        for c_idx in range(n_centers_actual):
            row_idx = c_idx * n_scales + s_idx
            basis[row_idx] = wavelets[:, c_idx]

    # Normalize
    basis = _normalize_basis(basis, normalize)

    return basis


def chebyshev_filter_basis(
    env: Environment,
    centers: NDArray[np.int_] | None = None,
    max_degree: int = 5,
    n_centers: int | None = None,
    center_method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
    normalize: Literal["unit", "max", "none"] = "unit",
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Compute Chebyshev polynomial filter basis on graph Laplacian.

    For each center c and polynomial degree k, computes:
        B(j) = [T_k(L_scaled)]_{j,c}

    where T_k is the Chebyshev polynomial of degree k and L_scaled is the
    Laplacian rescaled to have eigenvalues in [-1, 1].

    Parameters
    ----------
    env : Environment
        Fitted environment with connectivity graph.
    centers : NDArray[np.int_] or None
        Node indices for filter centers. If None, auto-select.
    max_degree : int, default=5
        Maximum polynomial degree, which controls spatial extent.
        A degree-k basis function is nonzero only for bins within k
        steps of its center (where each "step" is a bin-to-bin connection).

        **Choosing max_degree**:

        - Degree 1-2: Very local, only immediate neighbors
        - Degree 3-5: Typical choice, captures local place field structure
        - Degree 10+: Wide spatial extent, approaches global

        **Rule of thumb**: max_degree ≈ desired_radius / bin_size

        For 2cm bins and 10cm desired radius: max_degree ≈ 5
    n_centers : int or None
        Number of centers for auto-selection.
    center_method : {"kmeans", "farthest_point", "random"}
        Method for auto center selection.
    normalize : {"unit", "max", "none"}, default="unit"
        Normalization mode (see geodesic_rbf_basis for details).
    random_state : int, Generator, or None
        Random state for center selection.

    Returns
    -------
    basis : NDArray[np.float64], shape (n_centers * (max_degree + 1), n_bins)
        Filter basis matrix. Rows ordered by (center, degree):
        all degrees for center 0, then all degrees for center 1, etc.

        Includes degree 0 (delta function) through degree max_degree.

    Raises
    ------
    ValueError
        If max_degree < 0.
        If centers is None and n_centers is None.

    Notes
    -----
    **Strict locality guarantee**: A degree-k polynomial of L depends only
    on nodes within k hops. This is stronger than heat kernel or RBF bases,
    which have infinite (though decaying) support.

    **No eigendecomposition needed**: Computed via Chebyshev recurrence:

        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2x * T_k(x) - T_{k-1}(x)

    Each step is a sparse matrix-vector product: O(n_edges).

    **Computational complexity**: O(max_degree * n_edges * n_centers).
    Fastest of the three basis types for large environments.

    **Comparison with other bases**:

    - Chebyshev: Strictest locality (exactly k hops), fastest to compute
    - Heat kernel: Smooth decay, infinite support
    - Geodesic RBF: Gaussian along paths, infinite support

    Examples
    --------
    >>> basis = chebyshev_filter_basis(
    ...     env, n_centers=40, max_degree=5
    ... )  # doctest: +SKIP
    >>> basis.shape  # doctest: +SKIP
    (240, n_bins)  # 40 centers * 6 degrees (0 through 5)

    References
    ----------
    .. [1] Defferrard, M. et al. (2016). Convolutional neural networks on
           graphs with fast localized spectral filtering. NeurIPS.
    .. [2] Hammond, D.K. et al. (2011). Wavelets on graphs via spectral
           graph theory. Applied and Computational Harmonic Analysis.

    See Also
    --------
    PyGSP : Graph Signal Processing library with similar functionality
        https://pygsp.readthedocs.io/
    """

    import networkx as nx
    from scipy import sparse

    # Handle centers
    if centers is None:
        if n_centers is None:
            raise ValueError(
                "Must specify basis function locations.\n"
                "\n"
                "Two options:\n"
                "  1. Provide centers: specific bin indices for filter centers\n"
                "     Example: centers=np.array([5, 10, 15, 20])\n"
                "\n"
                "  2. Provide n_centers: auto-select centers using k-means\n"
                "     Example: n_centers=50\n"
                "\n"
                "See select_basis_centers() for manual center selection."
            )
        centers = select_basis_centers(
            env, n_centers, method=center_method, random_state=random_state
        )
    centers = np.asarray(centers, dtype=np.int_)

    # Validate max_degree
    if max_degree < 0:
        raise ValueError(
            f"max_degree must be non-negative, got {max_degree}.\n"
            "\n"
            "max_degree controls spatial extent (in hops):\n"
            "  - max_degree=0: Delta function (no spread)\n"
            "  - max_degree=3: ~3 bins radius\n"
            "  - max_degree=5: ~5 bins radius (typical choice)\n"
        )

    n_centers_actual = len(centers)
    n_bins = env.n_bins
    n_degrees = max_degree + 1

    # Get graph Laplacian (sparse)
    laplacian = nx.laplacian_matrix(
        env.connectivity, nodelist=range(n_bins), weight="distance"
    )
    laplacian = laplacian.tocsr()  # Ensure CSR for efficient matrix-vector products

    # Estimate largest eigenvalue for rescaling
    lambda_max = _estimate_spectral_radius(laplacian)

    # Rescale Laplacian to [-1, 1]: L_scaled = 2*L/lambda_max - I
    laplacian_scaled = (2.0 / lambda_max) * laplacian - sparse.identity(
        n_bins, format="csr"
    )

    # Create delta vectors at centers
    # Shape: (n_bins, n_centers)
    delta = np.zeros((n_bins, n_centers_actual), dtype=np.float64)
    delta[centers, np.arange(n_centers_actual)] = 1.0

    # Chebyshev recurrence: compute T_k(L_scaled) @ delta for k = 0..max_degree
    # Standard recurrence:
    #   T_0(x) = 1
    #   T_1(x) = x
    #   T_{k+1}(x) = 2x * T_k(x) - T_{k-1}(x)
    #
    # Rows ordered by (center, degree): all degrees for center 0, then center 1

    basis = np.zeros((n_centers_actual * n_degrees, n_bins), dtype=np.float64)

    cheby_prev = delta.copy()  # T_0 @ delta = delta (identity)
    # Store degree 0 for each center
    for c_idx in range(n_centers_actual):
        basis[c_idx * n_degrees + 0] = cheby_prev[:, c_idx]

    if max_degree >= 1:
        cheby_curr = laplacian_scaled @ delta  # T_1 @ delta = L_scaled @ delta
        # Store degree 1 for each center
        for c_idx in range(n_centers_actual):
            basis[c_idx * n_degrees + 1] = cheby_curr[:, c_idx]

        for k in range(2, n_degrees):
            cheby_next = 2.0 * (laplacian_scaled @ cheby_curr) - cheby_prev
            # Store degree k for each center
            for c_idx in range(n_centers_actual):
                basis[c_idx * n_degrees + k] = cheby_next[:, c_idx]
            cheby_prev, cheby_curr = cheby_curr, cheby_next

    # Normalize
    basis = _normalize_basis(basis, normalize)

    return basis


def _estimate_spectral_radius(laplacian: Any) -> float:
    """
    Estimate largest eigenvalue of graph Laplacian.

    Uses scipy.sparse.linalg.eigsh for efficiency. Falls back to
    max-degree bound if eigsh fails.

    Parameters
    ----------
    laplacian : sparse matrix
        Graph Laplacian matrix.

    Returns
    -------
    float
        Estimated largest eigenvalue.
    """
    import warnings

    from scipy.sparse.linalg import eigsh

    n = laplacian.shape[0]
    if n <= 2:
        # Small matrix: compute directly
        return float(np.max(np.linalg.eigvalsh(laplacian.toarray())))

    # Use eigsh with which='LM' (largest magnitude)
    try:
        eigenvalues = eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)
        return float(eigenvalues[0])
    except Exception as e:
        # Fallback: use max-degree bound (2 * max node degree)
        # For graph Laplacian, lambda_max <= 2 * max_degree
        # Diagonal of L contains node degrees
        max_degree_bound = 2.0 * float(np.max(laplacian.diagonal()))
        warnings.warn(
            f"eigsh failed ({e}), using max-degree bound {max_degree_bound:.2f}. "
            f"This may be slightly inaccurate for highly irregular graphs.",
            stacklevel=3,
        )
        return max_degree_bound


def spatial_basis(
    env: Environment,
    coverage: Literal["local", "medium", "global"] = "medium",
    n_features: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Create maze-aware spatial basis with automatic parameter selection.

    This is a convenience wrapper around geodesic_rbf_basis() that
    automatically chooses n_centers and sigma based on your environment
    size and desired coverage scale.

    Use this when you want sensible defaults without tuning parameters.

    Parameters
    ----------
    env : Environment
        Fitted environment.
    coverage : {"local", "medium", "global"}, default="medium"
        Spatial scale of basis functions:

        - "local": Fine-grained (sigma ≈ 5% of environment size)
          Good for: Detailed place fields, small receptive fields
        - "medium": Moderate (sigma ≈ 10% of environment size)
          Good for: General-purpose spatial modeling (recommended)
        - "global": Broad (sigma ≈ 20% of environment size)
          Good for: Compartment-level structure, coarse encoding
    n_features : int, default=100
        Approximate number of basis functions.
        Actual count may vary slightly due to multi-scale construction.
    random_state : int, Generator, or None
        Random state for reproducibility.

    Returns
    -------
    basis : NDArray[np.float64], shape (n_basis, n_bins)
        Ready-to-use basis matrix.

    Examples
    --------
    >>> from neurospatial import Environment  # doctest: +SKIP
    >>> from neurospatial.basis import spatial_basis  # doctest: +SKIP
    >>>
    >>> env = Environment.from_samples(positions, bin_size=2.0)  # doctest: +SKIP
    >>> basis = spatial_basis(env)  # Automatic defaults  # doctest: +SKIP
    >>>
    >>> # Use in GLM immediately
    >>> bin_indices = env.bin_sequence(trajectory, times)  # doctest: +SKIP
    >>> X = basis[:, bin_indices].T  # doctest: +SKIP

    Notes
    -----
    For more control over parameters, use geodesic_rbf_basis() directly.

    The function creates a multi-scale basis with 3 octave-spaced sigma
    values centered around the coverage-determined scale.
    """
    # Compute environment extent
    # dimension_ranges is a sequence of tuples: [(min1, max1), (min2, max2), ...]
    dim_ranges = env.dimension_ranges
    if dim_ranges is None:
        raise ValueError(
            "Environment has no dimension_ranges. "
            "Ensure the environment was created with valid data."
        )

    if env.n_dims == 1:
        min_val, max_val = dim_ranges[0]
        extent = max_val - min_val
    else:
        # Use geometric mean of extents for multi-dimensional environments
        extents = [max_val - min_val for min_val, max_val in dim_ranges]
        extent = float(np.sqrt(np.prod(extents)))

    # Determine base sigma from coverage setting
    sigma_fraction = {"local": 0.05, "medium": 0.10, "global": 0.20}
    sigma_base = extent * sigma_fraction[coverage]

    # Create multi-scale basis with 3 octave-spaced sigmas
    sigmas = [sigma_base / 2, sigma_base, sigma_base * 2]

    # Compute n_centers to achieve approximately n_features total
    n_centers = max(1, int(np.ceil(n_features / len(sigmas))))

    return geodesic_rbf_basis(
        env,
        n_centers=n_centers,
        sigma=sigmas,
        normalize="unit",
        random_state=random_state,
    )


def plot_basis_functions(
    env: Environment,
    basis: NDArray[np.float64],
    indices: list[int] | None = None,
    n_examples: int = 6,
    figsize: tuple[float, float] = (12, 8),
    cmap: str = "viridis",
) -> Any:
    """
    Visualize selected basis functions for quality checking.

    Displays a grid of basis functions overlaid on the environment.
    Helps verify:

    - Spatial coverage is appropriate
    - Basis functions respect walls
    - Multi-scale structure is present (if using multiple sigmas/scales)

    Parameters
    ----------
    env : Environment
        Environment used to create basis.
    basis : NDArray[np.float64], shape (n_basis, n_bins)
        Basis matrix from geodesic_rbf_basis() etc.
    indices : list[int] or None
        Specific basis function indices to plot.
        If None, randomly selects n_examples.
    n_examples : int, default=6
        Number of example basis functions to show (if indices is None).
    figsize : tuple, default=(12, 8)
        Figure size (width, height) in inches.
    cmap : str, default="viridis"
        Colormap for basis function values.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with subplots.

    Examples
    --------
    >>> basis = geodesic_rbf_basis(
    ...     env, n_centers=50, sigma=[5, 10, 20]
    ... )  # doctest: +SKIP
    >>> fig = plot_basis_functions(env, basis, n_examples=9)  # doctest: +SKIP
    >>> fig.savefig("basis_functions.png")  # doctest: +SKIP

    Notes
    -----
    This function is useful for:

    - Debugging: Check that basis functions have expected spatial extent
    - Parameter tuning: Visualize effect of different sigma/scale values
    - Presentation: Create figures showing basis function coverage
    """
    import matplotlib.pyplot as plt

    n_basis = basis.shape[0]

    # Select indices to plot
    if indices is None:
        if n_examples >= n_basis:
            indices = list(range(n_basis))
        else:
            rng = np.random.default_rng()
            indices = sorted(rng.choice(n_basis, size=n_examples, replace=False))

    n_plot = len(indices)
    n_cols = min(3, n_plot)
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plot == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(indices):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        env.plot_field(basis[idx], ax=ax, cmap=cmap, colorbar=False)
        ax.set_title(f"Basis {idx}", fontsize=10)

    # Hide empty subplots
    for i in range(n_plot, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Basis Functions ({n_basis} total)", fontsize=12)
    fig.tight_layout()

    return fig
