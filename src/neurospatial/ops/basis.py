"""
Maze-aware spatial basis functions for regression.

This module provides localized basis functions that respect environment
topology (walls, corridors) for use in GLMs and other spatial models.

Which Basis Should I Use?
-------------------------
**First time / Simple place cell models** → `geodesic_rbf_basis`
    Start here. Works like familiar Gaussian kernels but respects walls.
    Easy to visualize and interpret.
    Good for: GLMs, simple decoding models, place field decomposition.

**Multi-scale spatial structure** → `heat_kernel_wavelet_basis`
    Captures both local bumps and compartment-level structure simultaneously.
    Smoother than RBF. Based on diffusion physics.
    Good for: Environments with rooms/corridors, hierarchical place cells.
    Cost: one sparse matrix exponential per scale.

**Very large environments (>10,000 bins)** → `chebyshev_filter_basis`
    Fastest computation (no distance matrix). Strict k-hop locality.
    Good for: Large arenas, online/real-time applications.
    Tradeoff: Less intuitive parameterization (degree vs spatial extent).

**Not sure?** Start with `geodesic_rbf_basis(env, n_centers=50, bandwidth=10.0)`.
You can always switch later—all three work the same way in GLMs.

**Want automatic parameter selection?** Use ``spatial_basis(env)``, which
picks centers and bandwidths from the environment's size.

Scale Parameters Across Bases
------------------------------
The three basis types use different scale parameters:

- geodesic_rbf_basis(bandwidth=...): RBF bandwidth in environment units (cm)
- heat_kernel_wavelet_basis(scales=...): Diffusion time in bin-spacing² units;
  standard deviation sqrt(2*scale) bins
- chebyshev_filter_basis(max_degree=...): How many bins away from center the
  function extends (degree 5 = nonzero within 5 bins of center)

Rough equivalence for spatial extent:
    bandwidth ≈ sqrt(2*scale) * bin_size ≈ max_degree * bin_size

For example, with 2cm bins and 10cm desired spatial extent:
    bandwidth=10.0  or  scales=[12.5]  or  max_degree=5

Quick Start
-----------
A GLM needs one design-matrix row per time bin, aligned with the spike counts
of that time bin. Look up the bin of every position sample with
``env.bin_at`` (not ``env.bin_sequence``, which drops repeated bins) and keep
the samples inside the environment: ``bin_at`` marks the others with -1,
which as an index would select the last bin.

>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.ops import geodesic_rbf_basis
>>> rng = np.random.default_rng(0)
>>> positions = rng.uniform(0, 100, size=(2000, 2))  # (n_time, 2), cm
>>> env = Environment.from_samples(positions, bin_size=5.0)
>>> basis = geodesic_rbf_basis(env, n_centers=50, bandwidth=10.0, rng=0)
>>> bins = env.bin_at(positions)  # (n_time,), -1 outside the environment
>>> inside = bins >= 0
>>> X = basis[:, bins[inside]].T  # (n_time_inside, n_basis)
>>> X.shape == (int(inside.sum()), 50)
True

Full GLM Workflow
-----------------
Fit a simulated place cell (field at (30, 60) cm) with a Poisson GLM and
project the coefficients back to a rate map.

>>> import numpy as np
>>> from sklearn.linear_model import PoissonRegressor
>>> from neurospatial import Environment
>>> from neurospatial.ops import geodesic_rbf_basis
>>> rng = np.random.default_rng(0)
>>> dt = 0.02  # s per time bin
>>> positions = rng.uniform(0, 100, size=(5000, 2))  # one row per time bin, cm
>>> distance_sq = np.sum((positions - [30.0, 60.0]) ** 2, axis=1)
>>> rate = 1.0 + 20.0 * np.exp(-distance_sq / (2 * 10.0**2))  # Hz
>>> spike_counts = rng.poisson(rate * dt)  # (n_time,)
>>>
>>> # 1. Environment and a two-scale basis
>>> env = Environment.from_samples(positions, bin_size=5.0)
>>> basis = geodesic_rbf_basis(env, n_centers=30, bandwidth=[10.0, 20.0], rng=0)
>>>
>>> # 2. Design matrix, one row per time bin inside the environment
>>> bins = env.bin_at(positions)
>>> inside = bins >= 0
>>> X = basis[:, bins[inside]].T  # (n_time_inside, n_basis)
>>> y = spike_counts[inside]
>>>
>>> # 3. Fit the GLM
>>> model = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X, y)
>>>
>>> # 4. Fitted rate in every bin (Hz); plot it with env.plot_field
>>> fitted_rate = np.exp(model.intercept_ + basis.T @ model.coef_) / dt
>>> peak = env.bin_centers[np.argmax(fitted_rate)]
>>> bool(np.linalg.norm(peak - [30.0, 60.0]) < 5.0)
True

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

from neurospatial._exceptions import _format_error
from neurospatial.geometry import MissingFaces, require_cartesian

if TYPE_CHECKING:
    import networkx as nx
    from scipy import sparse
    from scipy.spatial import cKDTree

    from neurospatial.geometry import DiffusionGeometry
    from neurospatial.ops._protocols import (
        BasisCentersConsumer,
        FieldPlotConsumer,
        SpatialBasisConsumer,
        WaveletBasisConsumer,
    )

__all__ = [
    "chebyshev_filter_basis",
    "geodesic_rbf_basis",
    "heat_kernel_wavelet_basis",
    "plot_basis_functions",
    "select_basis_centers",
    "spatial_basis",
]


def select_basis_centers(
    env: BasisCentersConsumer,
    *,
    n_centers: int,
    method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """
    Select well-distributed center nodes for basis functions.

    Parameters
    ----------
    env : Environment
        Environment with bin_centers.
    n_centers : int
        Number of centers to select.

        **Rule of thumb**: n_centers ≈ (environment area) / (bandwidth²)

        - For 100cm × 100cm maze with bandwidth=10cm: n_centers ≈ 100
        - For 50cm linear track with bandwidth=5cm: n_centers ≈ 10

        Too few centers: May miss local spatial structure
        Too many centers: Redundant coverage, overfitting risk

        Start conservative (30-50) and increase if place field
        reconstructions are too smooth.
    method : {"kmeans", "farthest_point", "random"}
        Selection strategy:

        - "kmeans": K-means clustering on bin_centers, return nearest bins
          to cluster centroids. Good general-purpose choice.
        - "farthest_point": Greedy farthest-point sampling using geodesic
          distance, starting from a random bin. Each new center is the bin
          farthest from those already chosen: a greedy heuristic that
          spreads centers well, not a guaranteed maximum spread. Runs one
          shortest-path search per center.
        - "random": Uniform random selection. Fast but may cluster.
    rng : int, Generator, or None
        Random seed or generator for reproducibility. Seeds k-means, the
        first farthest-point center, and random selection.

    Returns
    -------
    centers : NDArray[np.int_], shape (n_centers,)
        Node indices of selected centers.

    Raises
    ------
    ValueError
        If n_centers > n_bins.
        If n_centers < 1.
        If method is not one of the methods above.
    CartesianOnlyError
        If method="kmeans" and ``env`` is polar: k-means would cluster
        ``(distance, angle)`` bin centers as ``(x, y)``. Use
        ``method="farthest_point"`` or ``method="random"``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import geodesic_rbf_basis, select_basis_centers
    >>> positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> centers = select_basis_centers(env, n_centers=50, method="kmeans", rng=0)
    >>> centers.shape
    (50,)
    >>> basis = geodesic_rbf_basis(env, centers=centers, bandwidth=5.0)
    >>> basis.shape == (50, env.n_bins)
    True
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
    generator = np.random.default_rng(rng)  # a Generator is returned as is

    if method == "kmeans":
        require_cartesian(
            env.coordinate_system,
            "select_basis_centers(method='kmeans')",
            why="k-means clusters bin centers as Cartesian positions.",
            fix=(
                "use method='farthest_point', which spreads centers by graph "
                "distance, or method='random' (center_method= in the basis functions)"
            ),
        )
        return _select_centers_kmeans(
            env.bin_centers, env.bin_center_tree(), n_centers, generator
        )
    elif method == "farthest_point":
        return _select_centers_farthest_point(
            env.connectivity, env.n_bins, n_centers, generator
        )
    elif method == "random":
        return _select_centers_random(env.n_bins, n_centers, generator)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Expected one of: 'kmeans', 'farthest_point', 'random'."
        )


def _select_centers_kmeans(
    bin_centers: NDArray[np.float64],
    tree: cKDTree,
    n_centers: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Select centers by k-means on ``bin_centers``, shape (n_bins, n_dims).

    ``tree`` is the environment's KD-tree over ``bin_centers``.
    """
    from sklearn.cluster import KMeans

    # Run k-means on bin centers
    kmeans = KMeans(
        n_clusters=n_centers,
        random_state=int(rng.integers(0, 2**31)),
        n_init=10,
    )
    kmeans.fit(bin_centers)

    # Find nearest bin to each cluster centroid
    _, centers = tree.query(kmeans.cluster_centers_)

    return np.asarray(centers, dtype=np.int_)


def _select_centers_random(
    n_bins: int,
    n_centers: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Select centers uniformly at random."""
    centers = rng.choice(n_bins, size=n_centers, replace=False)
    return np.asarray(centers, dtype=np.int_)


def _select_centers_farthest_point(
    connectivity: nx.Graph,
    n_bins: int,
    n_centers: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Select centers using greedy farthest-point sampling with geodesic distance."""
    from scipy.sparse.csgraph import shortest_path

    from neurospatial.ops.distance import _bin_adjacency

    adjacency = _bin_adjacency(
        connectivity, weight="distance", operation="select_basis_centers()"
    )

    # Start with a random point
    centers = [int(rng.integers(0, n_bins))]

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


def _validate_centers(centers: Any, n_bins: int) -> NDArray[np.int_]:
    """Return ``centers`` as bin indices, refusing values that are not bins.

    Parameters
    ----------
    centers : array-like of int, shape (n_centers,)
        Bin indices of the basis centers.
    n_bins : int
        Number of bins in the environment.

    Returns
    -------
    NDArray[np.int_], shape (n_centers,)
        The same indices.

    Raises
    ------
    TypeError
        If ``centers`` is not an integer array: a fractional value would be
        truncated to a different bin.
    ValueError
        If ``centers`` is not 1-D or holds an index outside ``[0, n_bins)``,
        such as the outside marker -1, which would wrap to the last bin.
    """
    array = np.asarray(centers)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            _format_error(
                f"centers must be integer bin indices, got dtype {array.dtype}.",
                why="A fractional center would be truncated to a different bin.",
                fix="pass bin indices, e.g. env.bin_at(points) or "
                "select_basis_centers(env, n_centers=...)",
            )
        )
    if array.ndim != 1:
        raise ValueError(
            _format_error(
                f"centers must be a 1-D array of bin indices, got shape {array.shape}.",
                fix="pass one bin index per basis center",
            )
        )
    invalid = array[(array < 0) | (array >= n_bins)]
    if invalid.size:
        raise ValueError(
            _format_error(
                f"centers holds {invalid.size} value(s) that are not bins of this "
                f"environment (valid: 0 to {n_bins - 1}), e.g. {int(invalid[0])}.",
                why="-1 is env.bin_at's marker for a point outside the "
                "environment, and a negative index would wrap to the last bin.",
                fix="drop centers whose point is outside the environment, or "
                "choose centers with select_basis_centers(env, n_centers=...)",
            )
        )
    return array.astype(np.int_, copy=False)


def _resolve_centers(
    env: BasisCentersConsumer,
    centers: NDArray[np.int_] | None,
    n_centers: int | None,
    center_method: Literal["kmeans", "farthest_point", "random"],
    rng: int | np.random.Generator | None,
    noun: str,
) -> NDArray[np.int_]:
    """Validated basis centers: ``centers`` as given, or ``n_centers`` selected.

    ``noun`` names the basis functions in the error raised when neither is given.
    """
    if centers is None:
        if n_centers is None:
            raise ValueError(
                "Must specify basis function locations.\n"
                "\n"
                "Two options:\n"
                f"  1. Provide centers: specific bin indices for {noun} centers\n"
                "     Example: centers=np.array([5, 10, 15, 20])\n"
                "\n"
                "  2. Provide n_centers: auto-select centers using k-means\n"
                "     Example: n_centers=50\n"
                "\n"
                "See select_basis_centers() for manual center selection."
            )
        centers = select_basis_centers(
            env, n_centers=n_centers, method=center_method, rng=rng
        )
    return _validate_centers(centers, env.n_bins)


def _require_faces(env: WaveletBasisConsumer, analysis: str) -> None:
    """Refuse a layout without cell faces, naming ``analysis`` (not smoothing)."""
    faces = env.binning.faces
    if isinstance(faces, MissingFaces):
        raise faces.missing_error(analysis)


def _center_deltas(n_bins: int, centers: NDArray[np.int_]) -> NDArray[np.float64]:
    """Unit impulses at ``centers``, one per column: shape ``(n_bins, n_centers)``."""
    delta = np.zeros((n_bins, len(centers)), dtype=np.float64)
    delta[centers, np.arange(len(centers))] = 1.0
    return delta


def _center_major(
    responses: NDArray[np.float64], normalize: Literal["unit", "max", "none"]
) -> NDArray[np.float64]:
    """Basis rows ordered by center, then by parameter, normalized.

    Parameters
    ----------
    responses : NDArray[np.float64], shape (n_params, n_bins, n_centers)
        One response per parameter value (scale or degree) and center.
    normalize : {"unit", "max", "none"}
        Row normalization (see :func:`_normalize_basis`).

    Returns
    -------
    NDArray[np.float64], shape (n_centers * n_params, n_bins)
        Every parameter for center 0, then center 1, and so on.
    """
    n_bins = responses.shape[1]
    basis = responses.transpose(2, 0, 1).reshape(-1, n_bins)
    return _normalize_basis(basis, normalize)


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
    env: BasisCentersConsumer,
    centers: NDArray[np.int_] | None = None,
    *,
    bandwidth: float | Sequence[float] = 5.0,
    n_centers: int | None = None,
    center_method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
    normalize: Literal["unit", "max", "none"] = "unit",
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Compute maze-aware RBF basis using geodesic distances.

    For each center c and bandwidth (physical standard deviation), computes:
        B(j) = exp(-d_geo(j, c)^2 / (2 * bandwidth^2))

    where d_geo is the shortest-path distance respecting walls.

    Parameters
    ----------
    env : Environment
        Environment.
    centers : NDArray[np.int_] or None
        Bin indices (0 to ``n_bins - 1``) for basis centers. If None, auto-select using
        n_centers and center_method.
    bandwidth : float or sequence of float
        RBF bandwidth in environment units (typically cm).
        Controls spatial spread of each basis function.

        **Rule of thumb**: bandwidth ≈ (desired coverage diameter) / 2

        - bandwidth=5.0cm: Localized to ~10cm diameter (covers 2*bandwidth)
        - bandwidth=20.0cm: Broad coverage, ~40cm diameter

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
    rng : int, Generator, or None
        Random seed or generator for center selection.

    Returns
    -------
    basis : NDArray[np.float64], shape (n_centers * n_bandwidths, n_bins)
        Basis matrix with rows ordered by (center, bandwidth):

        - basis[0:n_bandwidths] = all scales for center 0
        - basis[n_bandwidths:2*n_bandwidths] = all scales for center 1
        - ...

        Example with n_centers=3, bandwidth=[5, 10]:
            Row 0: center 0, bandwidth=5
            Row 1: center 0, bandwidth=10
            Row 2: center 1, bandwidth=5
            Row 3: center 1, bandwidth=10
            Row 4: center 2, bandwidth=5
            Row 5: center 2, bandwidth=10

        ``basis[:, bins].T`` with ``bins = env.bin_at(positions)`` restricted
        to ``bins >= 0`` is a GLM design matrix with one row per time bin
        (see Examples).

    Raises
    ------
    ValueError
        If bandwidth contains non-positive values.
        If centers is None and n_centers is None.
        If centers holds an index that is not a bin, such as the outside
        marker -1.
        If environment graph is disconnected and some bins are unreachable.
    TypeError
        If centers is not an integer array.

    Notes
    -----
    **Why geodesic instead of Euclidean?**

    In a maze, two points may be spatially close but far apart in terms
    of navigation (separated by a wall). Geodesic distance captures the
    actual path length, so basis functions don't "leak" through walls.

    **Computational complexity**:

    - Time: O(n_centers * n_bins * log(n_bins)) using batch shortest paths
    - Space: O(n_centers * n_bandwidths * n_bins) for output basis

    Runtime grows with ``n_centers * n_bins``. For very large environments
    (>10,000 bins), consider:

    - Using fewer centers (n_centers=30 instead of 100)
    - Larger bin_size to reduce n_bins
    - Switching to chebyshev_filter_basis (no distance matrix)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import geodesic_rbf_basis
    >>> positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    Single scale:

    >>> basis = geodesic_rbf_basis(env, n_centers=50, bandwidth=10.0, rng=0)
    >>> basis.shape == (50, env.n_bins)
    True

    Multi-scale (recommended), 30 centers x 3 bandwidths:

    >>> basis = geodesic_rbf_basis(
    ...     env, n_centers=30, bandwidth=[5.0, 10.0, 20.0], rng=0
    ... )
    >>> basis.shape == (90, env.n_bins)
    True

    GLM design matrix, one row per position sample (time bin). ``bin_at``
    marks samples outside the environment with -1; drop them, and the same
    time bins from the spike counts:

    >>> bins = env.bin_at(positions)
    >>> inside = bins >= 0
    >>> X = basis[:, bins[inside]].T  # (n_time_inside, n_basis)
    >>> X.shape == (int(inside.sum()), 90)
    True

    See Also
    --------
    heat_kernel_wavelet_basis : Diffusion-based alternative
    chebyshev_filter_basis : Strictly hop-local alternative
    select_basis_centers : Manual center selection
    """
    from scipy.sparse.csgraph import shortest_path

    from neurospatial.ops.distance import _bin_adjacency

    centers = _resolve_centers(env, centers, n_centers, center_method, rng, "basis")

    # Handle bandwidth and validate
    bandwidths = np.atleast_1d(bandwidth).astype(np.float64)
    invalid_mask = ~(np.isfinite(bandwidths) & (bandwidths > 0))
    if np.any(invalid_mask):
        invalid = bandwidths[invalid_mask]
        raise ValueError(
            f"bandwidth values must be positive and finite, got: {invalid}.\n"
            "\n"
            "bandwidth controls spatial spread of basis functions:\n"
            "  - bandwidth=5.0: ~10cm diameter coverage\n"
            "  - bandwidth=10.0: ~20cm diameter coverage\n"
            "\n"
            "Rule of thumb: bandwidth ≈ (desired coverage diameter) / 2"
        )

    n_centers_actual = len(centers)
    n_bandwidths = len(bandwidths)
    n_bins = env.n_bins

    # Compute distances from all centers at once using scipy's optimized implementation
    # This is ~5-10x faster than per-center Dijkstra
    adjacency = _bin_adjacency(
        env.connectivity, weight="distance", operation="geodesic_rbf_basis()"
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

    # Compute RBF for each (center, bandwidth) combination
    # Rows ordered by (center, bandwidth): all bandwidths for center 0, then center 1, etc.
    # Shape: (n_centers * n_bandwidths, n_bins)
    #
    # Vectorized computation using broadcasting:
    # distances: (n_centers, n_bins)
    # bandwidths: (n_bandwidths,)
    # Result: (n_centers, n_bandwidths, n_bins) -> reshape to (n_centers * n_bandwidths, n_bins)
    two_bandwidth_sq = 2.0 * bandwidths**2  # (n_bandwidths,)

    # Broadcast: (n_centers, 1, n_bins) / (1, n_bandwidths, 1) -> (n_centers, n_bandwidths, n_bins)
    basis_3d = np.exp(
        -(distances[:, np.newaxis, :] ** 2)
        / two_bandwidth_sq[np.newaxis, :, np.newaxis]
    )
    # Reshape to (n_centers * n_bandwidths, n_bins) with (center, bandwidth) ordering
    basis = basis_3d.reshape(n_centers_actual * n_bandwidths, n_bins)

    # Normalize
    basis = _normalize_basis(basis, normalize)

    return basis


def _diffusion_generator(
    geometry: DiffusionGeometry, spacing: float
) -> sparse.csr_matrix:
    """Finite-volume generator in bin-spacing² units, as used by env.smooth.

    Parameters
    ----------
    geometry : DiffusionGeometry
        Conductances ``W`` and cell volumes of the bins.
    spacing : float
        The bins' typical spacing, ``env.bin_spacing``.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_bins, n_bins)
        ``spacing**2 * M**-1 (Deg - W)``.
    """
    from neurospatial.ops.diffusion import _generator

    return spacing**2 * _generator(geometry.W, geometry.volumes)


def heat_kernel_wavelet_basis(
    env: WaveletBasisConsumer,
    centers: NDArray[np.int_] | None = None,
    *,
    scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    n_centers: int | None = None,
    center_method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
    normalize: Literal["unit", "max", "none"] = "unit",
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Compute heat kernel wavelet basis via graph diffusion.

    For each center c and diffusion time s, computes:
        B(j) = [exp(-s * L)]_{j,c}

    where L is the finite-volume diffusion generator in bin-spacing² units.
    This represents heat diffusion from
    center c for time s, respecting the graph topology.

    Parameters
    ----------
    env : Environment
        Environment with connectivity graph.
    centers : NDArray[np.int_] or None
        Bin indices (0 to ``n_bins - 1``) for wavelet centers. If None, auto-select.
    scales : sequence of float, default=(0.5, 1.0, 2.0, 4.0)
        Diffusion time in units of bin spacing². The kernel's standard
        deviation on a uniform Cartesian grid is ``sqrt(2 * scale)`` bins,
        independent of ``bin_size``: 0.5 → 1, 1 → 1.41, 2 → 2, 4 → 2.83 bins.
        Larger values give wider spread; walls and boundaries constrain it.
        For a physical bandwidth, scale = bandwidth² / (2 * spacing²).
    n_centers : int or None
        Number of centers for auto-selection.
    center_method : {"kmeans", "farthest_point", "random"}
        Method for auto center selection.
    normalize : {"unit", "max", "none"}, default="unit"
        Normalization mode (see geodesic_rbf_basis for details).
    rng : int, Generator, or None
        Random seed or generator for center selection.

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
        If centers holds an index that is not a bin, such as the outside
        marker -1.
    TypeError
        If centers is not an integer array.
    NotImplementedError
        If the layout has no finite-volume geometry builder.

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

    **Laplacian weighting**: Uses the finite-volume generator of
    ``env.smooth`` (shared-face measure over center distance, divided by cell
    volume), scaled by the squared bin spacing, ``env.bin_spacing``.
    Spread is isotropic on a uniform Cartesian grid and expressed in bin units.

    **Comparison with geodesic RBF**:

    - Heat kernel: Smooth transition, infinite but decaying support
    - Geodesic RBF: Gaussian profile along shortest paths

    Both respect maze topology; heat kernel is more "diffuse".

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import heat_kernel_wavelet_basis
    >>> positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> basis = heat_kernel_wavelet_basis(
    ...     env, n_centers=30, scales=[0.5, 1, 2, 4], rng=0
    ... )
    >>> basis.shape == (120, env.n_bins)  # 30 centers x 4 scales
    True

    References
    ----------
    .. [1] Coifman, R.R. & Maggioni, M. (2006). Diffusion wavelets.
           Applied and Computational Harmonic Analysis, 21(1), 53-94.
    .. [2] Hammond, D.K. et al. (2011). Wavelets on graphs via spectral
           graph theory. Applied and Computational Harmonic Analysis, 30(2), 129-150.
    """
    from scipy.sparse.linalg import expm_multiply

    centers = _resolve_centers(env, centers, n_centers, center_method, rng, "wavelet")

    # Validate scales
    scales_arr = np.asarray(scales, dtype=np.float64)
    invalid_mask = ~(np.isfinite(scales_arr) & (scales_arr > 0))
    if np.any(invalid_mask):
        invalid = scales_arr[invalid_mask]
        raise ValueError(
            f"scales values must be positive and finite, got: {invalid}.\n"
            "\n"
            "scales controls diffusion time (spatial spread):\n"
            "  - scale=0.5: standard deviation 1 bin\n"
            "  - scale=2.0: standard deviation 2 bins\n"
            "  - scale=4.0: standard deviation 2.83 bins\n"
            "\n"
            "Fix: pass positive scales, e.g. scales=[0.5, 1.0, 2.0, 4.0]."
        )

    _require_faces(env, "heat_kernel_wavelet_basis")
    laplacian = _diffusion_generator(env.get_diffusion_geometry(), env.bin_spacing)

    # exp(-s L) @ delta for each scale, without forming exp(-s L).
    # Shape: (n_scales, n_bins, n_centers)
    delta = _center_deltas(env.n_bins, centers)
    wavelets = np.stack(
        [expm_multiply(-s * laplacian, delta) for s in scales_arr], axis=0
    )
    return _center_major(wavelets, normalize)


def chebyshev_filter_basis(
    env: WaveletBasisConsumer,
    centers: NDArray[np.int_] | None = None,
    *,
    max_degree: int = 5,
    n_centers: int | None = None,
    center_method: Literal["kmeans", "farthest_point", "random"] = "kmeans",
    normalize: Literal["unit", "max", "none"] = "unit",
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Compute Chebyshev polynomial filter basis on the diffusion generator.

    For each center c and polynomial degree k, computes:
        B(j) = [T_k(L_scaled)]_{j,c}

    where T_k is the Chebyshev polynomial of degree k and L_scaled is the
    finite-volume diffusion generator of ``env.smooth`` rescaled to have
    eigenvalues in [-1, 1].

    Parameters
    ----------
    env : Environment
        Environment with finite-volume cell geometry.
    centers : NDArray[np.int_] or None
        Bin indices (0 to ``n_bins - 1``) for filter centers. If None, auto-select.
    max_degree : int, default=5
        Maximum polynomial degree, which controls spatial extent.
        A degree-k basis function is nonzero only for bins within k
        steps of its center, where each step crosses a face shared by two
        bins (bins that touch only at a corner are not one step apart).

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
    rng : int, Generator, or None
        Random seed or generator for center selection.

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
        If centers holds an index that is not a bin, such as the outside
        marker -1.
    TypeError
        If centers is not an integer array.
    NotImplementedError
        If the layout has no finite-volume geometry builder.

    Notes
    -----
    The operator is ``L = M**-1 (Deg - W)``, the generator ``env.smooth`` and
    ``heat_kernel_wavelet_basis`` diffuse with: ``W = A / d`` is the shared-face
    measure over the center distance and ``M`` the cell volume. It is rescaled
    by its largest eigenvalue, so the filters do not depend on ``bin_size``.
    Neighbors across a face of the same size are coupled equally, and bins
    that only touch at a corner, or sit on opposite sides of a wall, are not
    coupled.

    **Strict locality guarantee**: A degree-k polynomial of L depends only
    on bins within k shared-face steps. This is stronger than heat kernel or
    RBF bases, which have infinite (though decaying) support.

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
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import chebyshev_filter_basis
    >>> positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> basis = chebyshev_filter_basis(env, n_centers=40, max_degree=5, rng=0)
    >>> basis.shape == (240, env.n_bins)  # 40 centers x 6 degrees (0 to 5)
    True

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

    from scipy import sparse

    from neurospatial.ops.diffusion import _generator, _symmetric_conjugate

    centers = _resolve_centers(env, centers, n_centers, center_method, rng, "filter")

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

    _require_faces(env, "chebyshev_filter_basis")
    geometry = env.get_diffusion_geometry()
    # L = M^-1 (Deg - W) is not symmetric on unequal volumes; its symmetric
    # conjugate S = M^-1/2 (Deg - W) M^-1/2 has the same eigenvalues.
    symmetric = _symmetric_conjugate(geometry.W, geometry.volumes)
    laplacian = _generator(geometry.W, geometry.volumes)

    # Estimate largest eigenvalue for rescaling
    lambda_max = _estimate_spectral_radius(symmetric)

    # Rescale Laplacian to [-1, 1]: L_scaled = 2*L/lambda_max - I. When no two
    # bins share a face, L = 0 and every eigenvalue is 0, which any positive
    # scale maps to -1, so L_scaled = -I.
    identity = sparse.identity(env.n_bins, format="csr")
    laplacian_scaled = (
        (2.0 / lambda_max) * laplacian - identity if lambda_max > 0 else -identity
    )

    # Chebyshev recurrence T_k(L_scaled) @ delta for k = 0..max_degree:
    #   T_0(x) = 1,  T_1(x) = x,  T_{k+1}(x) = 2x * T_k(x) - T_{k-1}(x)
    # Each term has shape (n_bins, n_centers).
    delta = _center_deltas(env.n_bins, centers)
    cheby_results = [delta]
    if max_degree >= 1:
        cheby_prev, cheby_curr = delta, laplacian_scaled @ delta
        cheby_results.append(cheby_curr)
        for _k in range(2, max_degree + 1):
            cheby_next = 2.0 * (laplacian_scaled @ cheby_curr) - cheby_prev
            cheby_results.append(cheby_next)
            cheby_prev, cheby_curr = cheby_curr, cheby_next

    return _center_major(np.stack(cheby_results, axis=0), normalize)


def _estimate_spectral_radius(laplacian: Any) -> float:
    """
    Estimate largest eigenvalue of a symmetric positive semi-definite Laplacian.

    Uses scipy.sparse.linalg.eigsh for efficiency. Falls back to
    max-degree bound if eigsh fails.

    Parameters
    ----------
    laplacian : sparse matrix, shape (n_bins, n_bins)
        Symmetric Laplacian, e.g. ``M**-1/2 (Deg - W) M**-1/2``.

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

    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence

    # Use eigsh with which='LM' (largest magnitude)
    try:
        eigenvalues = eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)
        return float(eigenvalues[0])
    except (ArpackNoConvergence, ArpackError) as e:
        # ARPACK failed to converge on the largest eigenvalue. Fall back to the
        # Gershgorin upper bound lambda_max <= 2 * max(diagonal): the diagonal
        # Deg_i / M_i equals the off-diagonal row sum of M^-1 (Deg - W), which
        # has the same eigenvalues. Any *other* exception (malformed matrix,
        # wrong dtype) is a real bug and propagates.
        max_degree_bound = 2.0 * float(np.max(laplacian.diagonal()))
        warnings.warn(
            f"eigsh did not converge ({e}); using max-degree bound "
            f"{max_degree_bound:.2f}. This is an exact upper bound but may be "
            f"loose for highly irregular graphs.",
            stacklevel=3,
            category=UserWarning,
        )
        return max_degree_bound


def spatial_basis(
    env: SpatialBasisConsumer,
    *,
    coverage: Literal["local", "medium", "global"] = "medium",
    n_features: int = 100,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Create maze-aware spatial basis with automatic parameter selection.

    This is a convenience wrapper around geodesic_rbf_basis() that
    automatically chooses n_centers and bandwidth based on your environment
    size and desired coverage scale.

    Use this when you want sensible defaults without tuning parameters.

    Parameters
    ----------
    env : Environment
        Environment.
    coverage : {"local", "medium", "global"}, default="medium"
        Spatial scale of basis functions:

        Environment size is the track length on a linearized track and the
        geometric mean of the axis extents (``dimension_ranges``) otherwise.

        - "local": Fine-grained (bandwidth ≈ 5% of environment size)
          Good for: Detailed place fields, small receptive fields
        - "medium": Moderate (bandwidth ≈ 10% of environment size)
          Good for: General-purpose spatial modeling (recommended)
        - "global": Broad (bandwidth ≈ 20% of environment size)
          Good for: Compartment-level structure, coarse encoding
    n_features : int, default=100
        Approximate number of basis functions.
        Actual count may vary slightly due to multi-scale construction.
    rng : int, Generator, or None
        Random seed or generator for reproducibility.

    Returns
    -------
    basis : NDArray[np.float64], shape (n_basis, n_bins)
        Ready-to-use basis matrix.

    Raises
    ------
    CartesianOnlyError
        If ``env`` is polar: the automatic scale multiplies distance and angle
        extents, and centers come from k-means on Cartesian bin centers.
    ValueError
        If ``env.dimension_ranges`` is None, so there is no extent to size the
        basis from.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import spatial_basis
    >>> positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> basis = spatial_basis(env, rng=0)  # automatic defaults
    >>> basis.shape == (102, env.n_bins)  # 34 centers x 3 bandwidths
    True

    GLM design matrix, one row per time bin inside the environment:

    >>> bins = env.bin_at(positions)
    >>> X = basis[:, bins[bins >= 0]].T  # (n_time_inside, n_basis)

    Notes
    -----
    For more control over parameters, use geodesic_rbf_basis() directly.

    The function creates a multi-scale basis with 3 octave-spaced bandwidth
    values centered around the coverage-determined scale.
    """
    require_cartesian(
        env.coordinate_system,
        "spatial_basis()",
        why="Its automatic scale comes from the Cartesian extent of the bins.",
        fix=(
            "use geodesic_rbf_basis(env, bandwidth=..., center_method='farthest_point') "
            "with bandwidth in distance units"
        ),
    )
    # Compute environment extent
    # dimension_ranges is a sequence of tuples: [(min1, max1), (min2, max2), ...]
    dim_ranges = env.dimension_ranges
    if dim_ranges is None:
        raise ValueError(
            "Environment has no dimension_ranges. "
            "Ensure the environment was created with valid data."
        )

    if env.is_linearized_track:
        # A track's size is the length of centerline its bins cover; the
        # bounding box of its centerline is flat for a straight track and
        # unrelated to the length for a bent one.
        extent = float(np.sum(env.bin_sizes))
    else:
        # Geometric mean of the axis extents: a length in every dimension
        # (the extent itself in 1-D).
        extents = [max_val - min_val for min_val, max_val in dim_ranges]
        extent = float(np.prod(extents) ** (1.0 / len(extents)))

    # Determine base bandwidth from coverage setting
    bandwidth_fraction = {"local": 0.05, "medium": 0.10, "global": 0.20}
    bandwidth_base = extent * bandwidth_fraction[coverage]

    # Create multi-scale basis with 3 octave-spaced bandwidths
    bandwidths = [bandwidth_base / 2, bandwidth_base, bandwidth_base * 2]

    # Compute n_centers to achieve approximately n_features total
    n_centers = max(1, int(np.ceil(n_features / len(bandwidths))))

    return geodesic_rbf_basis(
        env,
        n_centers=n_centers,
        bandwidth=bandwidths,
        normalize="unit",
        rng=rng,
    )


def plot_basis_functions(
    env: FieldPlotConsumer,
    basis: NDArray[np.float64],
    *,
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
    - Multi-scale structure is present (if using multiple bandwidths/scales)

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
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.ops import geodesic_rbf_basis, plot_basis_functions
    >>> positions = np.random.default_rng(0).uniform(0, 100, size=(2000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> basis = geodesic_rbf_basis(
    ...     env, n_centers=20, bandwidth=[5.0, 10.0, 20.0], rng=0
    ... )
    >>> fig = plot_basis_functions(env, basis, indices=[0, 1, 2])
    >>> len(fig.axes) >= 3
    True
    >>> plt.close(fig)

    Notes
    -----
    This function is useful for:

    - Debugging: Check that basis functions have expected spatial extent
    - Parameter tuning: Visualize effect of different bandwidth/scale values
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

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
