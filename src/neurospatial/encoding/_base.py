"""Shared protocols, mixins, and array helpers for encoding result classes.

This module provides shared infrastructure for encoding result classes:

- `_is_jax_array`: Check if an array is a JAX array
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

from neurospatial._results import ResultMixin

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial import Environment


def _is_jax_array(arr: ArrayLike) -> bool:
    """Check if an array is a JAX array.

    Use this to detect JAX arrays for backend-aware dispatch. Returns False
    if JAX is not installed.

    Parameters
    ----------
    arr : ArrayLike
        Input array to check.

    Returns
    -------
    bool
        True if arr is a JAX array, False otherwise (including if JAX is
        not installed).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding._base import _is_jax_array
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> _is_jax_array(arr)
    False
    """
    try:
        import jax

        return isinstance(arr, jax.Array)
    except ImportError:
        return False


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
    if _is_jax_array(arr):
        import jax.numpy as jnp

        return jnp
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


class SpatialResultMixin(ResultMixin):
    """Mixin providing common spatial result methods.

    Extends :class:`neurospatial._results.ResultMixin`, so every spatial result
    class also carries the uniform ``to_dataframe()`` / ``summary()`` / ``plot()``
    surface. This mixin supplies sensible spatial defaults for ``summary()`` and
    a tidy-form ``to_dataframe()`` on top of the peak helpers; ``plot()`` is left
    to each concrete result (and is optional).

    Requires subclass to have:
    - `self.firing_rate` (1D array) OR `self.firing_rates` (2D array)
    - `self.occupancy` (1D array)
    - `self._bin_centers` returning the bin-centers array (default reads
      ``self.env.bin_centers``; subclasses may override).

    Result classes should inherit this mixin to get consistent implementations
    of `peak_location()` and `peak_firing_rate()`. Do NOT reimplement these
    methods in subclasses.

    Notes
    -----
    All mixin methods are host-only: they use `_to_numpy()` internally and
    always return NumPy arrays. This is intentional as these methods are
    for convenience/visualization, not part of JAX-traced compute graphs.

    The mixin works for any result class whose bin centers are available
    via a single attribute access. The default reads ``self.env.bin_centers``,
    suitable for spatial / view results. Result classes that index a different
    coordinate space (egocentric polar, directional angular) should override
    ``_bin_centers`` to return the appropriate centers array.
    """

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        """Bin centers in the relevant coordinate space.

        Default implementation reads ``self.env.bin_centers``. Subclasses
        may override to read from a directly stored ``bin_centers``
        (e.g. for directional results that have no Environment).
        """
        env: Any = self.env  # type: ignore[attr-defined]
        bin_centers: NDArray[np.float64] = env.bin_centers
        return bin_centers

    def _bin_center_columns(self) -> dict[str, NDArray[np.float64]]:
        """Per-bin center coordinates as consistently-named DataFrame columns.

        Maps each bin's center to one or more named columns following the
        shared vocabulary used across all result classes:

        - Cartesian environments: ``bin_center_x`` / ``bin_center_y`` /
          ``bin_center_z`` (and ``bin_center_dim_3`` ... for >3-D).
        - Polar egocentric environments: ``bin_center_distance`` and
          ``bin_center_angle``.
        - Directional results (no environment, angular centers): subclasses
          override to emit ``bin_center_angle`` directly.

        The default reads ``self.env`` and dispatches Cartesian vs. polar via
        :func:`neurospatial._results._bin_center_coords`, which raises if the
        environment's ``bin_centers`` disagree with the result's bin count.

        Returns
        -------
        dict
            Mapping ``column_name -> values`` (each a 1-D array of length
            ``n_bins``). Empty when no environment is available.
        """
        from neurospatial._results import _bin_center_coords

        env: Any = getattr(self, "env", None)
        if env is None:
            return {}
        rates = _to_numpy(self._get_rates())
        n_bins = int(rates.shape[-1])
        coords = _bin_center_coords(env, n_bins)
        return {name: np.asarray(values) for name, (_dim, values) in coords.items()}

    def _row_unit_ids(self) -> NDArray[Any]:
        """Per-unit identity labels, one per unit (row).

        Batch results expose ``unit_ids`` (length ``n_units``); single-unit
        results expose a scalar ``unit_id`` (possibly ``None``). This returns a
        1-D array of length ``n_units`` suitable for building the dense
        ``unit_id`` column or indexing the per-unit ``summary_table``.
        """
        import pandas as pd

        unit_ids = getattr(self, "unit_ids", None)
        if unit_ids is not None:
            return np.asarray(unit_ids)
        unit_id = getattr(self, "unit_id", None)
        # When a standalone single-unit result has no identity, represent the
        # absence of identity as absence (pd.NA), not a fabricated label like 0
        # (which would be a real, selectable index value the unit never had).
        return np.asarray([pd.NA if unit_id is None else unit_id], dtype=object)

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

    def peak_location(self) -> NDArray[np.float64]:
        """Peak firing locations in the result's coordinate space.

        Returns the bin-center coordinates of the maximum firing rate.

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
        bin_centers = self._bin_centers
        n_dims = int(bin_centers.shape[1]) if bin_centers.ndim > 1 else 1

        if rates.ndim == 1:
            # A fully-NaN (dead) unit has no defined peak; np.nanargmax would
            # raise "All-NaN slice encountered". Return NaN coordinates instead.
            if not np.any(np.isfinite(rates)):
                return np.full(n_dims, np.nan, dtype=np.float64)
            peak_idx = int(np.nanargmax(rates))
            result: NDArray[np.float64] = bin_centers[peak_idx]
            return result

        # Batch: guard each row so an all-NaN unit yields NaN coordinates
        # rather than crashing the whole np.nanargmax call.
        finite_mask = np.any(np.isfinite(rates), axis=1)
        safe_rates = np.where(finite_mask[:, None], rates, 0.0)
        peak_indices = np.nanargmax(safe_rates, axis=1)
        result = bin_centers[peak_indices].astype(np.float64, copy=True)
        result[~finite_mask] = np.nan
        return result

    def peak_firing_rate(self) -> NDArray[np.float64] | float:
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
            return float(np.nanmax(rates))
        result: NDArray[np.float64] = np.nanmax(rates, axis=1)
        return result

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics for this spatial result.

        Provides a uniform, specialization of
        :meth:`neurospatial._results.ResultMixin.summary` for spatial results.
        Reports the number of bins, peak firing rate, and total occupancy.
        For batch results (with ``firing_rates``), the peak is the maximum
        across all neurons and ``n_neurons`` is included.

        Returns
        -------
        dict
            Mapping with keys ``n_bins`` (int), ``peak_firing_rate`` (float,
            Hz), and ``total_occupancy`` (float, seconds). Batch results also
            include ``n_neurons`` (int).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> s = result.summary()
        >>> sorted(s)
        ['method', 'n_bins', 'peak_firing_rate', 'total_occupancy']
        """
        rates = _to_numpy(self._get_rates())
        occupancy = _to_numpy(self.occupancy)  # type: ignore[attr-defined]

        # An empty result (0 neurons in a batch, or 0 bins) has no firing-rate
        # values to reduce over. np.nanmax over a zero-size axis raises
        # "zero-size array to reduction operation fmax which has no identity",
        # so report a NaN peak instead of crashing.
        if rates.size == 0:
            peak_value = float("nan")
        else:
            peak_value = float(np.nanmax(np.asarray(self.peak_firing_rate())))

        out: dict[str, Any] = {
            "n_bins": int(rates.shape[-1]),
            "peak_firing_rate": peak_value,
            "total_occupancy": float(np.nansum(occupancy)),
        }
        if rates.ndim > 1:
            out["n_neurons"] = int(rates.shape[0])
        # Carry the estimator on result classes that record it (the spatial/view
        # rate results); egocentric/directional/place-field results have no
        # `method` field and are left unchanged.
        if hasattr(self, "method"):
            out["method"] = self.method
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Dense tidy table of per-bin firing rate and occupancy.

        Specializes :meth:`neurospatial._results.ResultMixin.to_dataframe` as
        the **dense** terminal verb: one row per ``(unit, bin)`` (single-unit
        results: one row per ``bin``), **always** carrying a ``unit_id``
        column. This is the frame for plotting / detailed per-bin inspection.
        For the per-unit scalar summary (peak location, spatial info, cell
        type, ...) use :meth:`summary_table` instead.

        The ``unit_id`` column carries the real per-unit identity labels
        (:attr:`unit_ids` for batch results, :attr:`unit_id` for single-unit
        results). Bin-center coordinates are emitted with the shared
        vocabulary: ``bin_center_x`` / ``bin_center_y`` / ``bin_center_z`` for
        Cartesian environments, ``bin_center_distance`` /
        ``bin_center_angle`` for polar egocentric environments, and
        ``bin_center_angle`` for directional results.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``unit_id``, ``bin`` (int),
            the ``bin_center_*`` coordinate columns (float), ``firing_rate``
            (float, Hz), and ``occupancy`` (float, seconds).

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> df = result.to_dataframe()
        >>> {"unit_id", "bin", "firing_rate"} <= set(df.columns)
        True
        >>> len(df) == env.n_bins
        True
        """
        import pandas as pd

        rates = _to_numpy(self._get_rates())
        occupancy = _to_numpy(self.occupancy)  # type: ignore[attr-defined]

        rates_2d = rates[None, :] if rates.ndim == 1 else rates
        n_neurons, n_bins = rates_2d.shape

        row_unit_ids = self._row_unit_ids()
        unit_col = np.repeat(np.asarray(row_unit_ids), n_bins)
        bin_col = np.tile(np.arange(n_bins), n_neurons)

        data: dict[str, Any] = {
            "unit_id": unit_col,
            "bin": bin_col,
        }
        for name, values in self._bin_center_columns().items():
            data[name] = np.tile(np.asarray(values), n_neurons)
        data["firing_rate"] = rates_2d.reshape(-1)
        # Occupancy is shared across neurons for batch spatial results.
        occ = np.asarray(occupancy).reshape(-1)
        data["occupancy"] = np.tile(occ, n_neurons)
        # Carry the estimator (spatial/view rate results only); broadcasts to
        # every row. Absent on results with no `method` field.
        if hasattr(self, "method"):
            data["method"] = self.method

        return pd.DataFrame(data)

    def summary_table(self) -> pd.DataFrame:
        """Per-unit scalar summary, one row per unit, ``unit_id``-indexed.

        Complements :meth:`to_dataframe` (dense, one row per ``(unit, bin)``)
        as the **per-unit summary** terminal verb: one row per unit with scalar
        metric columns (peak location, peak rate, and the spatial metrics each
        result class can compute). This is the table a many-neuron user wants
        for filtering, sorting, and population summaries.

        The base implementation provides the shared columns
        (``peak_*`` coordinates and ``peak_rate``); concrete batch result
        classes extend it with their domain metrics (spatial information,
        grid/border scores, cell type, preferred direction, etc.).

        Returns
        -------
        pandas.DataFrame
            One row per unit, indexed by ``unit_id``, with at least
            ``peak_<coord>`` columns and ``peak_rate`` (float, Hz).
        """
        import pandas as pd

        row_unit_ids = self._row_unit_ids()
        peaks = np.atleast_2d(self.peak_location())
        peak_rates = np.atleast_1d(self.peak_firing_rate())

        data: dict[str, Any] = {}
        coord_names = list(self._bin_center_columns().keys())
        n_dims = peaks.shape[1]
        for d in range(n_dims):
            # Reuse the bin-center vocabulary for peak columns:
            # bin_center_x -> peak_x, bin_center_distance -> peak_distance.
            if d < len(coord_names):
                col = "peak_" + coord_names[d].removeprefix("bin_center_")
            else:
                col = f"peak_coord_{d}"
            data[col] = peaks[:, d]
        data["peak_rate"] = peak_rates
        # Carry the estimator (spatial/view rate results only), one value per
        # unit. Absent on results with no `method` field.
        if hasattr(self, "method"):
            data["method"] = self.method

        df = pd.DataFrame(data, index=pd.Index(row_unit_ids, name="unit_id"))
        return df
