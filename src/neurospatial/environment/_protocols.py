"""Narrow structural contracts for domain functions consuming environment data.

Index-dtype convention: point/region *index* arrays (``bin_at``,
``bins_in_region``, ``boundary_bins``, ``point_to_bin_index``,
``trajectory._allocate_time_linear``) are ``np.intp`` -- pointer-sized (64-bit on every
platform, including Windows where ``np.int_`` is 32-bit) and the dtype NumPy
fancy-indexing returns. Trajectory *sequence* arrays (``bin_sequence``,
``BinSequenceWithRuns.bins``) stay compact ``np.int32`` on purpose, and the
``run_starts`` / ``run_lengths`` offsets are ``np.int64``. The split is
intentional -- portability for indices, memory for long sequences -- so do not
collapse them to one dtype.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from neurospatial.geometry._coordinates import CellLengthConsumer as CellLengthConsumer

if TYPE_CHECKING:
    from neurospatial.geometry._arena import Arena
    from neurospatial.geometry._binning import Binning
    from neurospatial.geometry._coordinates import CoordinateSystem
    from neurospatial.geometry._diffusion import DiffusionGeometry
    from neurospatial.ops._types import (
        KernelMode,
    )
    from neurospatial.regions import Region, Regions


class FieldMetricsConsumer(Protocol):
    """Environment members read by encoding/_field_metrics.py."""

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def bin_sizes(self) -> NDArray[np.float64]:
        """Per-bin cell volume; the canonical mass ``M``, shape ``(n_bins,)``."""
        ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of spatial dimensions (1, 2, or 3)."""
        ...


class BorderConsumer(FieldMetricsConsumer, Protocol):
    """Environment members read by encoding/border.py, encoding/_physical_border.py."""

    @property
    def binning(self) -> Binning: ...

    @property
    def arena(self) -> Arena: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def boundary_bins(self) -> NDArray[np.intp]:
        """Integer indices of bins at the environment boundary."""
        ...

    @property
    def connectivity(self) -> nx.Graph: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    def region_mask(
        self,
        regions: str | list[str] | Region | Regions,
        *,
        include_boundary: bool = True,
    ) -> NDArray[np.bool_]:
        """Boolean mask (shape ``(n_bins,)``) for one or more regions."""
        ...

    @property
    def regions(self) -> Regions: ...


class SmoothingConsumer(Protocol):
    """Environment members read by encoding/_smoothing.py."""

    def get_diffusion_geometry(self) -> DiffusionGeometry: ...

    def diffusion_accuracy(self, bandwidth: float) -> float: ...

    def compute_gaussian_kernel(self, bandwidth: float) -> NDArray[np.float64]: ...

    @property
    def coordinate_system(self) -> CoordinateSystem: ...

    @property
    def bin_centers(self) -> NDArray[np.float64]: ...

    @property
    def connectivity(self) -> nx.Graph: ...

    def diffuse(
        self,
        # ``Any`` is load-bearing: on ``backend="jax"`` diffuse accepts (and
        # returns) a JAX ``Array``, and jax is an optional dependency whose array
        # type is not importable here -- so ``NDArray | jax.Array`` cannot be
        # spelled. ``Any`` keeps the backend-polymorphic surface without breaking
        # jax-absent type checks.
        fields: Any,
        bandwidth: float,
        *,
        mode: KernelMode = "density",
        backend: Literal["numpy", "jax"] = "numpy",
    ) -> NDArray[np.float64]:
        """Matrix-free diffusion smoothing (no dense ``(n_bins, n_bins)`` kernel)."""
        ...

    @property
    def dimension_ranges(self) -> Sequence[tuple[float, float]] | None: ...

    @property
    def n_bins(self) -> int:
        """Number of spatial bins in the environment."""
        ...

    @property
    def units(self) -> str | None: ...
