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

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    from neurospatial.geometry._diffusion import FaceGeometry
    from neurospatial.ops._protocols import (
        DivergenceConsumer,
        GradientConsumer,
    )

__all__ = ["divergence", "gradient"]


def _compute_differential_operator(
    faces: FaceGeometry, n_bins: int
) -> sparse.csc_matrix:
    """Build an oriented inverse-distance operator from explicit face data."""
    n_edges = faces.distance.size
    if n_edges == 0:
        return sparse.csc_matrix((n_bins, 0), dtype=np.float64)
    src, dst = faces.edges[:, 0], faces.edges[:, 1]
    inv_d = 1.0 / faces.distance
    cols = np.arange(n_edges)
    return sparse.csc_matrix(
        (
            np.concatenate([-inv_d, inv_d]),
            (np.concatenate([src, dst]), np.concatenate([cols, cols])),
        ),
        shape=(n_bins, n_edges),
    )


def _compute_divergence_operator(
    faces: FaceGeometry, differential_operator: sparse.csc_matrix
) -> sparse.csc_matrix:
    """Build the negative volume-weighted adjoint of the gradient."""
    return (
        -sparse.diags(1.0 / faces.volumes)
        @ differential_operator
        @ sparse.diags(faces.A * faces.distance)
    ).tocsc()


def gradient(env: GradientConsumer, field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the directional derivative of a scalar field along every edge.

    Parameters
    ----------
    env : Environment
        Environment with finite-volume cell geometry.
    field : NDArray[np.float64], shape (n_bins,)
        Scalar value at each bin center.

    Returns
    -------
    gradient_field : NDArray[np.float64], shape (n_edges,)
        ``(field[j] - field[i]) / d_e`` on each oriented edge i -> j, in
        field units per length unit. Edge order is ``env.edge_pairs``.

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
    Environment.get_differential_operator : The inverse-distance edge operator.
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
    env: DivergenceConsumer, edge_field: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute net outward flux per cell volume.

    Parameters
    ----------
    env : Environment
        Environment with finite-volume cell geometry.
    edge_field : NDArray[np.float64], shape (n_edges,)
        Flux density along each oriented edge in ``env.edge_pairs``
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
            "Fix: pass a 1-D edge_field in env.edge_pairs order."
        )
    return np.asarray(
        env.get_divergence_operator() @ edge_field, dtype=np.float64
    ).ravel()
