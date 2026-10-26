"""Shared protocols, mixins, and array helpers for encoding result classes.

This module provides shared infrastructure for encoding result classes:

- `_is_jax_array`: Check if an array is a JAX array
- `_to_numpy`: Convert arrays (NumPy or JAX) to NumPy for host-only operations
- `_get_array_module`: Detect array backend (numpy or jax.numpy) for backend-aware ops
- `SpatialResultMixin`: Shared methods of every rate result
- `SingleUnitRateMixin` / `PopulationRateMixin`: Construction checks and unit
  identity of single-unit and population rate results

These utilities enable result classes to work transparently with both NumPy
and JAX arrays, routing to the appropriate implementation based on input type.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._results import ReadOnlyArraysMixin, ResultMixin, UnitSelectionMixin

if TYPE_CHECKING:
    import pandas as pd


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


class SpatialResultMixin(ResultMixin, ReadOnlyArraysMixin):
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

    spike_window: NDArray[np.float64] | None = None
    analysis_settings: dict[str, Any] | None = None

    if TYPE_CHECKING:
        # Members every rate result provides, as fields or properties.
        @property
        def occupancy(self) -> Any: ...

        @property
        def method(self) -> str: ...

        @property
        def bandwidth(self) -> float | None: ...

    @property
    def spatial_support(self) -> NDArray[np.bool_]:
        """Observed, sufficiently occupied bins with finite rates.

        The mask follows the result's coordinates: spatial locations, viewed
        locations, polar object vectors or angular bins. Zero-filling a rate
        never supplies missing coverage.
        """
        from neurospatial.encoding._provenance import spatial_support

        return spatial_support(self)

    def _analysis_attrs(self) -> dict[str, Any]:
        """Independent estimator metadata for table exports and indexed results."""
        from copy import deepcopy

        return {
            "method": self.method,
            "bandwidth": self.bandwidth,
            "analysis_settings": deepcopy(self.analysis_settings),
        }

    @property
    def spike_window_assumed(self) -> bool:
        """True when spikes were assumed recorded wherever position was.

        No ``spike_window`` was passed. The population-silence warning catches
        one common violation of this assumption; it cannot establish recording
        coverage.
        """
        return self.spike_window is None

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
        name = "firing_rates" if hasattr(self, "firing_rates") else "firing_rate"
        return getattr(self, name)

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
        across all units as ``max_peak_firing_rate`` and ``n_units`` is included.

        Returns
        -------
        dict
            Mapping with keys ``n_bins`` (int), ``peak_firing_rate`` (float,
            Hz), and ``total_occupancy`` (float, seconds). Batch results also
            include ``n_units`` (int) and use ``max_peak_firing_rate`` for
            the maximum per-unit peak. Total occupancy is seconds in the
            shared map, never summed over units. Singular results add cheap
            family metrics; grid/border scores are excluded from the repr. ``spike_window`` is the normalized
            acquisition-window list or None, and ``spike_window_assumed``
            records whether coverage was assumed.

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
        ['method', 'n_bins', 'peak_firing_rate', 'sparsity', 'spatial_info', 'spike_window', 'spike_window_assumed', 'total_occupancy']
        """
        rates = _to_numpy(self._get_rates())
        occupancy = _to_numpy(self.occupancy)

        # An empty result (0 neurons in a batch, or 0 bins) has no firing-rate
        # values to reduce over. np.nanmax over a zero-size axis raises
        # "zero-size array to reduction operation fmax which has no identity",
        # so report a NaN peak instead of crashing.
        if rates.size == 0:
            peak_value = float("nan")
        else:
            peak_value = float(np.nanmax(np.asarray(self.peak_firing_rate())))

        out: dict[str, Any] = {}
        if rates.ndim > 1:
            out["n_units"] = int(rates.shape[0])
        out["n_bins"] = int(rates.shape[-1])
        out["max_peak_firing_rate" if rates.ndim > 1 else "peak_firing_rate"] = (
            peak_value
        )
        if rates.ndim == 1:
            out.update(self._headline_metrics())
        out["total_occupancy"] = float(np.nansum(occupancy))
        out["spike_window_assumed"] = self.spike_window_assumed
        out["spike_window"] = (
            None if self.spike_window is None else self.spike_window.tolist()
        )
        out["method"] = self.method
        if hasattr(self, "direction_frame"):
            out["direction_frame"] = self.direction_frame
        return out

    def _headline_metrics(self) -> dict[str, float]:
        """O(n_bins) singular metrics displayed by summary and repr."""
        return {}

    def to_dataframe(self) -> pd.DataFrame:
        """Dense tidy table of per-bin firing rate and occupancy.

        The **dense** terminal verb: one row per ``(unit, bin)`` (single-unit
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
            (float, Hz), and ``occupancy`` (float, seconds). Spatial results
            also include ``spatial_support``. Units and available analysis
            settings are stored in ``attrs``; CSV does not preserve these.

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
        occupancy = _to_numpy(self.occupancy)

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
        # The estimator broadcasts to every row.
        data["method"] = self.method

        frame = pd.DataFrame(data)
        coordinate_units = getattr(getattr(self, "env", None), "units", None)
        frame.attrs["units"] = {
            "firing_rate": "Hz",
            "occupancy": "s",
            **{
                name: "rad" if name.endswith("angle") else coordinate_units or ""
                for name in self._bin_center_columns()
            },
        }
        frame.attrs.update(self._analysis_attrs())
        frame["spatial_support"] = np.tile(self.spatial_support, n_neurons)
        return frame

    def to_xarray(self) -> Any:
        """Export a labeled Dataset with dimensions (unit_id, bin).

        A singular result has one unit. A standalone result with no unit_id
        uses <NA>; pass a unit_id or index a population result for a NetCDF-safe
        label. Occupancy is shared across units. Coordinates, method metadata,
        physical units, direction frame and acquisition windows are retained.

        Returns
        -------
        xarray.Dataset
            Firing rates, shared occupancy, unit labels and bin coordinates.

        Raises
        ------
        ImportError
            If the optional xarray dependency is unavailable.
        ValueError
            If labels are duplicated or coordinates/occupancy do not match rates.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0)
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> single = result[0]
        >>> dataset = single.to_xarray()  # doctest: +SKIP
        >>> dataset.sizes["unit_id"]  # doctest: +SKIP
        1
        """
        from neurospatial._results import build_population_dataset
        from neurospatial.encoding._provenance import rate_xarray_attrs

        rates = np.asarray(_to_numpy(self._get_rates()), dtype=np.float64)
        env = getattr(self, "env", None)
        dataset = build_population_dataset(
            np.atleast_2d(rates),
            self._row_unit_ids(),
            env=env,
            bin_centers=None if env is not None else self._bin_centers,
            occupancy=np.asarray(_to_numpy(self.occupancy), dtype=np.float64),
            attrs={**rate_xarray_attrs(self), **self._xarray_attrs()},
        )
        dataset["spatial_support"] = ("bin", self.spatial_support)
        return dataset

    def _xarray_attrs(self) -> dict[str, Any]:
        """Family metadata; concrete rate results override this hook."""
        return {}


_UnitT = TypeVar("_UnitT", bound="SingleUnitRateMixin")
_UnitResultT = TypeVar("_UnitResultT", bound="SingleUnitRateMixin")


BandwidthRule = Literal["none", "nonnegative", "positive"]
"""What an estimator requires of ``bandwidth``: ``None``, ``>= 0`` or ``> 0``."""

KERNEL_ESTIMATORS: Mapping[str, BandwidthRule] = MappingProxyType(
    {"diffusion_kde": "positive", "gaussian_kde": "positive"}
)

_BANDWIDTH_REQUIREMENT: Mapping[BandwidthRule, str] = MappingProxyType(
    {
        "none": "None",
        "nonnegative": "a finite number >= 0 in the coordinates' units",
        "positive": "a finite number > 0 in the coordinates' units",
    }
)


def _as_result_array(value: Any) -> Any:
    """``value`` as a NumPy array; a JAX array is kept as given."""
    return value if _is_jax_array(value) else np.asarray(value)


def _real_scalar(value: Any) -> float | None:
    """``value`` as a float if it is one real number (not a bool), else None."""
    if isinstance(value, bool | np.bool_) or np.ndim(value) != 0:
        return None
    try:
        dtype = np.asarray(value).dtype
    except (TypeError, ValueError):
        return None
    return float(value) if dtype.kind in "fiu" else None


def _bandwidth_meets(bandwidth: float | None, rule: BandwidthRule) -> bool:
    """Whether a (normalized) ``bandwidth`` satisfies an estimator's rule."""
    if rule == "none" or bandwidth is None:
        return rule == "none" and bandwidth is None
    return bool(
        np.isfinite(bandwidth)
        and (bandwidth > 0 or (rule == "nonnegative" and bandwidth == 0))
    )


class _RateResultMixin(SpatialResultMixin):
    """Construction shared by every rate result dataclass.

    ``__post_init__`` checks the rates, occupancy, estimator and recording
    windows (:meth:`_check_rate_fields`), runs the family's own checks
    (:meth:`_check_family`), resolves unit identity (:meth:`_resolve_units`)
    and finally takes read-only ownership of the arrays.

    Subclasses set ``_RATES_FIELD`` / ``_RATES_NDIM`` (single-unit or
    population) and the family sets ``_ESTIMATORS``, mapping each estimator
    to what it requires of ``bandwidth``.
    """

    _RATES_FIELD: ClassVar[str]
    _RATES_NDIM: ClassVar[int]
    _ESTIMATORS: ClassVar[Mapping[str, BandwidthRule]]

    def __post_init__(self) -> None:
        self._check_rate_fields()
        self._check_family()
        self._resolve_units()
        self._own_arrays()

    def _n_bins(self) -> int:
        """Number of bins the rates must cover (the environment's by default)."""
        return int(np.shape(self._bin_centers)[0])

    def _check_rate_fields(self) -> None:
        """Refuse rates, occupancy, estimator or windows that do not agree.

        Arrays are stored as NumPy arrays (JAX arrays are kept as given) so a
        list cannot bypass read-only ownership.
        """
        from neurospatial._exceptions import _format_error
        from neurospatial._intervals import as_intervals

        kind = type(self).__name__
        rates = _as_result_array(getattr(self, self._RATES_FIELD))
        occupancy = _as_result_array(self.occupancy)
        object.__setattr__(self, self._RATES_FIELD, rates)
        object.__setattr__(self, "occupancy", occupancy)
        n_bins = self._n_bins()
        expected = "(n_bins,)" if self._RATES_NDIM == 1 else "(n_units, n_bins)"
        why = (
            "Why: each rate and occupancy value belongs to one bin, so both "
            "must cover the result's bins in the same order."
        )
        fix = (
            "build the result from the rates, occupancy and bins of one "
            "computation, or use the family's compute_* function."
        )
        for name, array in ((self._RATES_FIELD, rates), ("occupancy", occupancy)):
            if np.dtype(array.dtype).kind not in "fiu":
                raise ValueError(
                    _format_error(
                        f"{kind}: {name} must hold numbers, got dtype {array.dtype}.",
                        why=why,
                        fix=fix,
                    )
                )
        if np.ndim(rates) != self._RATES_NDIM or np.shape(rates)[-1] != n_bins:
            raise ValueError(
                _format_error(
                    f"{kind}: {self._RATES_FIELD} has shape {np.shape(rates)}, "
                    f"not {expected} over the result's {n_bins} bins.",
                    why=why,
                    fix=fix,
                )
            )
        if np.shape(occupancy) != (n_bins,):
            raise ValueError(
                _format_error(
                    f"{kind}: occupancy has shape {np.shape(occupancy)}, not "
                    f"({n_bins},) to match the {n_bins} bins of the rates.",
                    why=why,
                    fix=fix,
                )
            )
        method = self.method
        if method not in self._ESTIMATORS:
            raise ValueError(
                _format_error(
                    f"{kind}: method={method!r} is not an estimator of this "
                    "result family.",
                    why="Why: the estimator decides how the rates, occupancy "
                    "and bandwidth are read.",
                    fix=f"pass one of {sorted(self._ESTIMATORS)}.",
                )
            )
        rule = self._ESTIMATORS[method]
        bandwidth = None if self.bandwidth is None else _real_scalar(self.bandwidth)
        if not _bandwidth_meets(bandwidth, rule) or (
            bandwidth is None and self.bandwidth is not None
        ):
            raise ValueError(
                _format_error(
                    f"{kind}: bandwidth={self.bandwidth!r} is not valid for "
                    f"method={method!r}.",
                    why=f"Why: method={method!r} needs a bandwidth of "
                    f"{_BANDWIDTH_REQUIREMENT[rule]}.",
                    fix="pass the bandwidth the rates were estimated with.",
                )
            )
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(
            self, "spike_window", as_intervals(self.spike_window, name="spike_window")
        )

    def _check_family(self) -> None:
        """Family-specific invariants; results override this hook."""

    def _resolve_units(self) -> None:
        """Record unit identity; single-unit and population results differ."""


class SingleUnitRateMixin(_RateResultMixin):
    """A rate result for one unit, labelled by ``unit_id``."""

    _RATES_FIELD: ClassVar[str] = "firing_rate"
    _RATES_NDIM: ClassVar[int] = 1

    unit_id: int | str | None
    _unit_ids_generated: bool

    if TYPE_CHECKING:
        # Each family's dataclass defines the constructor.
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def _resolve_units(self) -> None:
        # No label is a generated identity; a label from a population with
        # generated labels arrives already marked.
        if self.unit_id is None:
            object.__setattr__(self, "_unit_ids_generated", True)


class PopulationRateMixin(_RateResultMixin, UnitSelectionMixin, Generic[_UnitResultT]):
    """A rate result for several units sharing occupancy and bins.

    Resolves ``unit_ids`` (default ``np.arange(n_units)``) and aligns
    ``unit_table`` to them. ``len()``, ``[i]`` (by position), ``sel(unit_id=)``
    and iteration give single-unit results that carry each unit's label and
    the population's shared settings.
    """

    _RATES_FIELD: ClassVar[str] = "firing_rates"
    _RATES_NDIM: ClassVar[int] = 2

    unit_ids: Any
    unit_table: pd.DataFrame | None
    _unit_ids_generated: bool

    def _resolve_units(self) -> None:
        from neurospatial._results import align_unit_table, resolve_unit_ids

        kind = type(self).__name__
        # Generated labels keep positional pairing (e.g. with a decoder's
        # spike input); dataclasses.replace carries the mark forward.
        if self.unit_ids is None:
            object.__setattr__(self, "_unit_ids_generated", True)
        object.__setattr__(
            self, "unit_ids", resolve_unit_ids(self.unit_ids, len(self), context=kind)
        )
        object.__setattr__(
            self,
            "unit_table",
            align_unit_table(self.unit_table, self.unit_ids, context=kind),
        )

    def __len__(self) -> int:
        """Number of units (rows of ``firing_rates``)."""
        return int(np.shape(self._get_rates())[0])

    def __iter__(self) -> Iterator[_UnitResultT]:
        """Each unit's single-unit result, in stored order."""
        for i in range(len(self)):
            yield self[i]

    if TYPE_CHECKING:

        def __getitem__(self, idx: int) -> _UnitResultT: ...

    def _unit_result(self, unit_class: type[_UnitT], idx: int, **fields: Any) -> _UnitT:
        """Build the single-unit result at position ``idx``.

        Parameters
        ----------
        unit_class : type
            The family's single-unit result class.
        idx : int
            Unit position; label lookups go through ``sel(unit_id=...)``.
        **fields
            Family fields of the single-unit result (environment, estimator,
            per-unit diagnostics).

        Returns
        -------
        object
            The unit's result with its label, the shared occupancy, recording
            windows and settings, and the population's unit-label provenance.
        """
        from neurospatial._results import label_at

        position = self._unit_position(idx)
        return unit_class(
            firing_rate=self._get_rates()[position],
            occupancy=self.occupancy,
            unit_id=label_at(self.unit_ids, position),
            spike_window=self.spike_window,
            analysis_settings=self._analysis_attrs()["analysis_settings"],
            _unit_ids_generated=self._unit_ids_generated,
            **fields,
        )
