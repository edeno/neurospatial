"""Shared protocols, mixins, and array helpers for encoding result classes.

This module provides shared infrastructure for encoding result classes:

- `_to_numpy`: Convert arrays (NumPy or JAX) to NumPy for host-only operations
- `_get_array_module`: Detect array backend (numpy or jax.numpy) for backend-aware ops
- `HasOccupancy`: Protocol for result classes with occupancy data
- `HasEnvironment`: Protocol for result classes with spatial environment
- `SpatialResultMixin`: Shared methods for spatial result classes

These utilities enable result classes to work transparently with both NumPy
and JAX arrays, routing to the appropriate implementation based on input type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from neurospatial import Environment


def _to_numpy(arr: ArrayLike) -> NDArray:
    """Convert array to NumPy for host-only operations.

    Use this for plotting, DataFrame export, and other convenience methods
    that are not part of the JAX-traced compute graph.

    Parameters
    ----------
    arr : ArrayLike
        Input array. Can be NumPy array, JAX array, list, tuple, or any
        array-like object that np.asarray can convert.

    Returns
    -------
    NDArray
        NumPy array. If input was already a NumPy array, it may be returned
        as-is (no copy). If input was a JAX array, it is transferred to host
        memory.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._base import _to_numpy
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> result = _to_numpy(arr)
    >>> isinstance(result, np.ndarray)
    True
    """
    return np.asarray(arr)


def _get_array_module(arr: ArrayLike) -> Any:
    """Get the array module (numpy or jax.numpy) for backend-aware operations.

    Use this for backend-aware methods that should preserve array type:
    NumPy in → NumPy out, JAX in → JAX out. This enables JAX-traced
    compute graphs to remain on-device.

    Parameters
    ----------
    arr : ArrayLike
        Input array. JAX arrays are detected via isinstance check with
        jax.Array.

    Returns
    -------
    ModuleType
        Either `numpy` or `jax.numpy` module, depending on the input type.
        For non-array types (lists, tuples), returns numpy.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._base import _get_array_module
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> xp = _get_array_module(arr)
    >>> xp is np
    True
    """
    # Check if JAX is available and if arr is a JAX array
    try:
        import jax
        import jax.numpy as jnp

        if isinstance(arr, jax.Array):
            return jnp
    except ImportError:
        pass
    return np


@runtime_checkable
class HasOccupancy(Protocol):
    """Protocol for result classes with occupancy data.

    This protocol is runtime-checkable, enabling isinstance() checks
    for duck-typing validation.

    Attributes
    ----------
    occupancy : ArrayLike
        Time spent in each spatial bin (seconds). Shape is typically (n_bins,).
    """

    occupancy: ArrayLike


@runtime_checkable
class HasEnvironment(Protocol):
    """Protocol for result classes with spatial environment.

    This protocol is runtime-checkable, enabling isinstance() checks
    for duck-typing validation.

    Attributes
    ----------
    env : Environment
        The spatial environment used for the computation.
    """

    env: Environment


class SpatialResultMixin:
    """Mixin providing common spatial result methods.

    Requires subclass to have:
    - `self.firing_rate` (1D array) OR `self.firing_rates` (2D array)
    - `self.occupancy` (1D array)
    - `self.env` (Environment instance)

    Result classes should inherit this mixin to get consistent implementations
    of `peak_locations()` and `peak_firing_rates()`. Do NOT reimplement these
    methods in subclasses.

    Notes
    -----
    All mixin methods are host-only: they use `_to_numpy()` internally and
    always return NumPy arrays. This is intentional as these methods are
    for convenience/visualization, not part of JAX-traced compute graphs.
    """

    def _get_rates(self) -> Any:
        """Get firing rate(s), handling both single and batch results.

        Returns
        -------
        ArrayLike
            1D array for single neuron, 2D array (n_neurons, n_bins) for batch.
        """
        if hasattr(self, "firing_rates"):
            return self.firing_rates
        return self.firing_rate  # type: ignore[attr-defined]

    def peak_locations(self) -> NDArray[np.float64]:
        """Peak firing locations in physical coordinates.

        Returns the environment coordinates of the bin with maximum firing rate.

        For single-neuron results, returns shape (n_dims,).
        For batch results, returns shape (n_neurons, n_dims).

        This is a host-only method: always returns NumPy arrays.
        Uses `_to_numpy()` internally for JAX compatibility.

        Returns
        -------
        NDArray[np.float64]
            Coordinates of peak firing location(s).
            Shape is (n_dims,) for single neuron, (n_neurons, n_dims) for batch.
        """
        rates = _to_numpy(self._get_rates())
        env: Any = self.env  # type: ignore[attr-defined]
        bin_centers: NDArray[np.float64] = env.bin_centers

        if rates.ndim == 1:
            # Single neuron: return (n_dims,)
            peak_idx = int(np.nanargmax(rates))
            result: NDArray[np.float64] = bin_centers[peak_idx]
            return result
        else:
            # Batch: return (n_neurons, n_dims)
            peak_indices = np.nanargmax(rates, axis=1)
            result = bin_centers[peak_indices]
            return result

    def peak_firing_rates(self) -> NDArray[np.float64] | float:
        """Peak firing rate values.

        Returns the maximum firing rate for each neuron.

        For single-neuron results, returns a scalar (float).
        For batch results, returns shape (n_neurons,).

        This is a host-only method: always returns NumPy arrays.
        Uses `_to_numpy()` internally for JAX compatibility.

        Returns
        -------
        NDArray[np.float64] | float
            Peak firing rate(s).
            Scalar for single neuron, (n_neurons,) array for batch.
        """
        rates = _to_numpy(self._get_rates())

        if rates.ndim == 1:
            # Single neuron: return scalar
            return float(np.nanmax(rates))
        else:
            # Batch: return (n_neurons,)
            result: NDArray[np.float64] = np.nanmax(rates, axis=1)
            return result
