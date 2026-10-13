"""Shared low-level types for kernel and spectral operations.

Kept in a dependency-free leaf module (imports only ``typing``) so both the
``ops`` implementations and the ``environment`` methods/Protocol can name the
same alias without triggering import cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Kernel normalization mode, shared by the kernel cache, ``compute_kernel``,
# ``smooth``, and ``diffuse`` across the ops and environment layers. One alias so
# the accepted set cannot drift between the type annotations. (Runtime validation
# lives next to each consumer as an explicit membership check.)
KernelMode = Literal["transition", "density", "average"]


class DiffusionEigenbasisCache(TypedDict, total=False):
    """Optional spectral basis and bandwidth resolutions for one geometry."""

    basis: tuple[NDArray[np.float64], NDArray[np.float64]]
    resolved: dict[tuple[float, float], int | Literal["dense"]]


class MRFEigenbasisCache(TypedDict):
    """Global eigenvectors, eigenvalues and component IDs for the MRF basis."""

    Q: NDArray[np.float64]
    Lam: NDArray[np.float64]
    mode_comp: NDArray[np.intp]
    G: int
