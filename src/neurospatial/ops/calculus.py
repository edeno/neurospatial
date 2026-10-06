"""Finite-volume differential operators on spatial environments.

For an oriented edge i -> j of length d, the gradient is (f[j] - f[i]) / d.
Divergence weights edge flux by the shared-face measure A and divides by the
cell volume M. Their composition is the continuum-sign Laplacian:
``div(grad(f)) = -M**-1 (Deg - W) f``, with ``W = A / d``, the generator
used by ``env.smooth``. The pair satisfies discrete Gauss-Green with edge
inner-product weights A*d and node inner-product weights M.

Import from ``neurospatial.ops`` or ``neurospatial.ops.calculus``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    import networkx as nx

    from neurospatial import Environment
    from neurospatial.environment._protocols import EnvironmentProtocol

__all__ = ["compute_differential_operator", "divergence", "gradient"]


def _fv_edges(env: Environment) -> tuple[nx.Graph, NDArray[np.float64]]:
    """Resolve face measures, along-edge distances and cell volumes."""
    from neurospatial.ops.diffusion import _finite_volume_geometry

    try:
        return _finite_volume_geometry(cast("EnvironmentProtocol", env))
    except NotImplementedError as err:
        raise NotImplementedError(
            f"gradient/divergence need finite-volume cell geometry, which layout "
            f"{type(env.layout).__name__!r} does not provide ({err}).\n"
            "Fix: build the environment with a factory method, e.g. "
            "Environment.from_samples(positions, bin_size=...)."
        ) from err


def compute_differential_operator(env: Environment) -> sparse.csc_matrix:
    """Build the inverse-distance oriented edge operator.

    For edge e = (i -> j), ``D[i, e] = -1/d_e`` and ``D[j, e] = 1/d_e``.
    ``D.T @ f`` is the gradient in field units per environment length unit.

    Parameters
    ----------
    env : Environment
        Fitted environment with finite-volume cell geometry. Edge distances
        use the same geometry as ``env.smooth``, including track junctions.

    Returns
    -------
    D : scipy.sparse.csc_matrix, shape (n_bins, n_edges)
        Sparse operator, with columns in ``env.connectivity.edges()`` order.

    Raises
    ------
    NotImplementedError
        If the layout has no finite-volume geometry builder.

    Notes
    -----
    Divergence is a separate operator: ``-M**-1 D diag(A*d)``. Thus
    ``div(grad(f)) = -M**-1 (Deg - W) f``, with ``W = A/d``. It is
    negative semidefinite in the volume-weighted node inner product.
    ``D @ D.T`` alone is not the physical Laplacian.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(np.arange(4.0)[:, None], bin_size=1.0)
    >>> D = compute_differential_operator(env)
    >>> D.shape
    (4, 3)
    >>> np.allclose(D.T @ env.bin_centers[:, 0], 1.0)
    True

    See Also
    --------
    gradient : Compute the directional derivative along each edge.
    divergence : Compute the net outward flux per cell volume.
    Environment.get_differential_operator : Cached access to this matrix.
    """
    graph, _ = _fv_edges(env)
    edges = list(graph.edges(data="distance"))
    n_bins, n_edges = env.n_bins, len(edges)
    if n_edges == 0:
        return sparse.csc_matrix((n_bins, 0), dtype=np.float64)
    src = np.fromiter((u for u, _, _ in edges), dtype=np.int64, count=n_edges)
    dst = np.fromiter((v for _, v, _ in edges), dtype=np.int64, count=n_edges)
    inv_d = 1.0 / np.fromiter((d for *_, d in edges), dtype=np.float64, count=n_edges)
    cols = np.arange(n_edges)
    return sparse.csc_matrix(
        (
            np.concatenate([-inv_d, inv_d]),
            (np.concatenate([src, dst]), np.concatenate([cols, cols])),
        ),
        shape=(n_bins, n_edges),
    )


def _compute_divergence_operator(env: Environment) -> sparse.csc_matrix:
    """Build the negative volume-weighted adjoint of the gradient."""
    graph, volumes = _fv_edges(env)
    grad_t = env.get_differential_operator()
    area = np.fromiter(
        (a for *_, a in graph.edges(data="A")), dtype=np.float64, count=grad_t.shape[1]
    )
    length = np.fromiter(
        (d for *_, d in graph.edges(data="distance")),
        dtype=np.float64,
        count=grad_t.shape[1],
    )
    return (
        -sparse.diags(1.0 / np.asarray(volumes, dtype=np.float64))
        @ grad_t
        @ sparse.diags(area * length)
    ).tocsc()


def gradient(env: Environment, field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the directional derivative of a scalar field along every edge.

    Parameters
    ----------
    env : Environment
        Fitted environment with finite-volume cell geometry.
    field : NDArray[np.float64], shape (n_bins,)
        Scalar value at each bin center.

    Returns
    -------
    gradient_field : NDArray[np.float64], shape (n_edges,)
        ``(field[j] - field[i]) / d_e`` on each oriented edge i -> j, in
        field units per length unit. Edge order is ``env.connectivity.edges()``.

    Raises
    ------
    ValueError
        If field shape does not match the environment's bins.
    NotImplementedError
        If the layout has no finite-volume geometry builder.

    Notes
    -----
    The gradient and divergence satisfy discrete Gauss-Green:
    ``sum(A*d*grad(f)*q) = -sum(M*f*div(q))``. Their composition is the
    continuum-sign Laplacian, the negative of the diffusion generator.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(np.arange(5.0)[:, None], bin_size=1.0)
    >>> np.allclose(gradient(env, env.bin_centers[:, 0]), 1.0)
    True
    >>> np.allclose(gradient(env, np.ones(env.n_bins)), 0.0)
    True

    See Also
    --------
    divergence : Compute outward flux per cell volume.
    compute_differential_operator : Build the inverse-distance edge operator.
    """
    if field.shape != (env.n_bins,):
        raise ValueError(
            f"field must have shape ({env.n_bins},) to match environment bins, "
            f"but got shape {field.shape}; a scalar value is needed at each bin.\n"
            "Fix: pass a 1-D field with one value per environment bin."
        )
    return np.asarray(
        env.get_differential_operator().T @ field, dtype=np.float64
    ).ravel()


def divergence(
    env: Environment, edge_field: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute net outward flux per cell volume.

    Parameters
    ----------
    env : Environment
        Fitted environment with finite-volume cell geometry.
    edge_field : NDArray[np.float64], shape (n_edges,)
        Flux density along each oriented edge in ``env.connectivity.edges()``
        order. Positive values flow from the first endpoint to the second.

    Returns
    -------
    divergence_field : NDArray[np.float64], shape (n_bins,)
        Net outward face flux divided by cell volume, in edge-field units per
        length unit. Positive at sources and negative at sinks.

    Raises
    ------
    ValueError
        If edge_field shape does not match the connectivity graph's edges.
    NotImplementedError
        If the layout has no finite-volume geometry builder.

    Notes
    -----
    With oriented incidence matrix B (negative at source, positive at
    destination), ``div(q) = -M**-1 B diag(A) q``. Consequently
    ``div(grad(f)) = -M**-1 (Deg - W) f``, ``W = A/d``. For a field in Hz
    and an environment in cm, this Laplacian has units Hz/cm².

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> env = Environment.from_samples(np.arange(5.0)[:, None], bin_size=1.0)
    >>> field = env.bin_centers[:, 0] ** 2
    >>> np.allclose(divergence(env, gradient(env, field))[1:-1], 2.0)
    True
    >>> np.allclose(divergence(env, np.zeros(env.connectivity.number_of_edges())), 0.0)
    True

    See Also
    --------
    gradient : Compute directional derivatives on edges.
    Environment.smooth : Smooth a field using the same finite-volume geometry.
    """
    n_edges = env.connectivity.number_of_edges()
    if edge_field.shape != (n_edges,):
        raise ValueError(
            f"edge_field must have shape ({n_edges},) to match connectivity graph edges, "
            f"but got shape {edge_field.shape}; a flux is needed for each edge.\n"
            "Fix: pass a 1-D edge_field in env.connectivity.edges() order."
        )
    return np.asarray(
        env._divergence_operator_cached @ edge_field, dtype=np.float64
    ).ravel()
