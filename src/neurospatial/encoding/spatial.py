"""Spatial rate computation for place, grid, and border cells.

This module provides result classes and compute functions for spatial firing
rate analysis. The result classes wrap firing rate maps with their metadata
and provide convenient methods for analysis and visualization.

Result Classes
--------------
SpatialRateResult
    Single-neuron spatial rate map with convenience methods
SpatialRatesResult
    Multi-neuron spatial rate maps with batch methods and iteration

Compute Functions
-----------------
compute_spatial_rate
    Compute spatial firing rate for one neuron
compute_spatial_rates
    Compute spatial firing rates for multiple neurons

Examples
--------
>>> import numpy as np
>>> from neurospatial import Environment
>>> from neurospatial.encoding.spatial import compute_spatial_rate

>>> # Create environment from a seeded trajectory
>>> rng = np.random.default_rng(0)
>>> positions = rng.uniform(0, 50, (500, 2))
>>> env = Environment.from_samples(positions, bin_size=5.0)

>>> # Compute a single-neuron spatial rate map (returns SpatialRateResult)
>>> times = np.linspace(0, 50, 500)
>>> spike_times = np.sort(rng.uniform(0, 50, 30))
>>> result = compute_spatial_rate(env, spike_times, times, positions, bandwidth=10.0)

>>> # Use inherited mixin methods
>>> peak = result.peak_location()  # (n_dims,) coordinates of peak
>>> peak_rate = result.peak_firing_rate()  # scalar max firing rate
>>> peak.shape
(2,)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._intervals import DEFAULT_MAX_GAP, TimeWindows, resolve_time_windows
from neurospatial._keyword_hint import keyword_hint
from neurospatial._observation import validate_break_before
from neurospatial._pynapple import reads_pynapple_samples
from neurospatial._results import (
    ReadOnlyArraysMixin,
    ResultMixin,
    UnitSelectionMixin,
    _copy_analysis_settings,
    _copy_unit_table,
    read_only_copy,
    unit_index_required,
)
from neurospatial._spikes import spike_unit_table
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._validation import refuse_none_thresholds, validate_choice
from neurospatial.encoding._base import (
    KERNEL_ESTIMATORS,
    BandwidthRule,
    PopulationRateMixin,
    SingleUnitRateMixin,
    _to_numpy,
)
from neurospatial.encoding._field_metrics import _seed_component, field_stability
from neurospatial.encoding._metrics import BatchScoresResult
from neurospatial.encoding._provenance import spikes_outside_intervals
from neurospatial.encoding._significance import (
    _SHUFFLE_DEFAULTS,
    check_criterion,
    check_mode_keywords,
    shuffle_information_verdict,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial._intervals import TimeSupport
    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.encoding._spatial_counts import (
        LinearSpatialIntervals,
        _SpikeTrack,
    )
    from neurospatial.encoding.grid import GridProperties
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource
    from neurospatial.ops._types import BackendType, ResolvedBackend, StraightOrGeodesic
    from neurospatial.stats.shuffle import ShuffleTestResult

# ruff: noqa: RUF022 - intentionally grouped by category
__all__ = [
    "place_cell_significance",
    "PlaceCellSignificanceResult",
    "spatial_cell_type_significance",
    "CellTypeSignificanceResult",
    # Result classes
    "SpatialRateResult",
    "SpatialRatesResult",
    "PlaceFieldsResult",
    # Compute functions
    "compute_spatial_rate",
    "compute_spatial_rates",
    # Directional place fields
    "DirectionalPlaceFields",
    "compute_directional_place_fields",
    # Field detection
    "detect_place_fields",
    # Classification predicates
    "is_place_cell",
    "has_place_field",
]


PLACE_FIELD_DETECTION_DEFAULTS = MappingProxyType(
    {
        "threshold": 0.2,
        "min_peak_rate": 1.0,
        "min_secondary_peak_fraction": 0.0,
        "max_mean_rate": np.inf,
        "subfields": "core",
    }
)
# The ways detect_place_fields separates fields that merge at the threshold.
_SUBFIELD_RULES = ("core", "watershed", None)
# Statistics recomputed from every shifted spike train, in result column order.
_CELL_TYPE_STATISTICS = ("spatial_information", "grid_score", "border_score")
_CELL_TYPE_STATISTIC_UNITS = MappingProxyType(
    {"spatial_information": "bits/spike", "grid_score": "", "border_score": ""}
)
PLACE_SPATIAL_INFO_THRESHOLDS = MappingProxyType({"min_info": 0.5})
# Every place verdict and label applies the same information cutoff.
PLACE_GRID_BORDER_THRESHOLDS = MappingProxyType(
    {
        "min_spatial_info": PLACE_SPATIAL_INFO_THRESHOLDS["min_info"],
        "min_grid_score": 0.4,
        "min_border_score": 0.5,
    }
)


def _owned_field_bins(
    fields: Sequence[ArrayLike], *, owner: str
) -> tuple[NDArray[np.int64], ...]:
    """Validate place-field bin arrays and return read-only int64 copies.

    Parameters
    ----------
    fields : sequence of array-like
        One 1-D array of bin indices per field.
    owner : str
        The class or function the fields are passed to, for error messages.

    Returns
    -------
    tuple of ndarray
        One read-only ``int64`` copy per field.

    Raises
    ------
    ValueError
        If a field is not a 1-D array of non-negative integer bin indices.
    """
    owned = []
    for i, bins in enumerate(fields):
        array = np.asarray(bins)
        numeric = array.size == 0 or (
            np.issubdtype(array.dtype, np.number)
            and not np.issubdtype(array.dtype, np.complexfloating)
        )
        if (
            array.ndim != 1
            or not numeric
            or not np.all(np.isfinite(array))
            or not np.array_equal(array, np.round(array))
        ):
            raise ValueError(
                _format_error(
                    f"{owner}: fields[{i}] must be a 1-D array of integer bin "
                    f"indices, got shape {array.shape} and dtype {array.dtype}.",
                    why="Why: each field is the set of bins it covers",
                    fix="pass integer bin indices, e.g. np.flatnonzero(mask)",
                )
            )
        if array.size and array.min() < 0:
            raise ValueError(
                _format_error(
                    f"{owner}: fields[{i}] contains negative bin indices.",
                    why="Why: bin indices count from 0",
                    fix="pass non-negative bin indices",
                )
            )
        owned.append(read_only_copy(array.astype(np.int64)))
    return tuple(owned)


def _measure_units(env: Environment) -> str:
    """Unit of a bin measure: ``cm`` on a track, ``cm^2`` in an arena, ...

    ``""`` when the environment has no units or its bins mix measure
    dimensions.
    """
    dims = np.unique(np.asarray(env.binning.bins.measure_dims))
    if not env.units or dims.size != 1:
        return ""
    return env.units if int(dims[0]) == 1 else f"{env.units}^{int(dims[0])}"


@dataclass(frozen=True, repr=False, eq=False)
class PlaceFieldsResult(ReadOnlyArraysMixin, ResultMixin):
    """Result of ``detect_place_fields()``: one unit's fields and their rate map.

    Returned by :func:`detect_place_fields`. Distinguishes "this map
    has no detectable fields" (``fields=()``, ``excluded_reason=None``)
    from "this map was excluded by an explicitly requested spatial-map-mean filter"
    (``fields=()``, ``excluded_reason="mean_rate_above_threshold"``) so a
    population pipeline can branch without listening for warnings.

    Parameters
    ----------
    fields : sequence of NDArray[np.int64], length n_fields
        One 1-D array of bin indices per detected field, in order of
        decreasing peak rate. Empty if no field was detected or the map was
        excluded by the ``max_mean_rate`` filter. Stored as a tuple of
        read-only arrays.
    env : Environment
        The environment the map was binned in.
    firing_rate : NDArray[np.float64], shape (n_bins,)
        The rate map the fields were cut from (Hz); NaN where unsupported.
    bandwidth : float
        Smoothing bandwidth of the map (environment units), which set the
        smallest field; 0 for an unsmoothed map.
    excluded_reason : {"mean_rate_above_threshold"} or None, default=None
        ``None`` when ``fields`` is the detection result.
        ``"mean_rate_above_threshold"`` when the explicit ``max_mean_rate``
        filter excluded the map before detection.
    unit_id : int or str or None, default=None
        The unit the map belongs to: a rate result's ``unit_id``. None for a
        map without one, shown as ``<NA>`` in tables.

    Attributes
    ----------
    n_fields : int
        Number of detected fields.
    total_bins : int
        Number of bins across all fields.
    n_excluded : int
        ``1`` if ``excluded_reason`` is set, else ``0``, so population
        aggregation can sum exclusions without parsing the reason.

    Notes
    -----
    The result is iterable, indexable and sized like its ``fields``, and is
    truthy when at least one field was detected.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import detect_place_fields
    >>> env = Environment.from_samples(
    ...     np.arange(0.0, 101.0, 2.0)[:, None], bin_size=2.0, units="cm"
    ... )
    >>> rate = 8.0 * np.exp(-((env.bin_centers[:, 0] - 40.0) ** 2) / (2 * 6.0**2))
    >>> fields = detect_place_fields(env, rate, bandwidth=0.0)
    >>> fields.n_fields
    1
    >>> list(fields.fields_table().columns[:5])
    ['unit_id', 'field', 'n_bins', 'area', 'peak_rate']
    >>> table = fields.to_dataframe()  # one row per bin, labeled by field
    >>> list(table.columns)
    ['unit_id', 'bin', 'bin_center_x', 'field', 'firing_rate']
    >>> len(table) == env.n_bins
    True
    >>> int(table["field"].notna().sum()) == fields.total_bins
    True
    """

    fields: Sequence[NDArray[np.int64]]
    env: Environment
    firing_rate: NDArray[np.float64]
    bandwidth: float
    excluded_reason: Literal["mean_rate_above_threshold"] | None = None
    unit_id: int | str | None = None

    def __post_init__(self) -> None:
        """Validate, own each field's bins and the map, prevent replacing fields."""
        if self.excluded_reason not in (None, "mean_rate_above_threshold"):
            raise ValueError(
                _format_error(
                    f"excluded_reason must be None or 'mean_rate_above_threshold', "
                    f"got {self.excluded_reason!r}.",
                    why="Why: it names the filter that excluded the map",
                    fix="pass excluded_reason=None for a detection result",
                )
            )
        if self.excluded_reason is not None and len(self.fields) > 0:
            raise ValueError(
                _format_error(
                    "An excluded map cannot have fields.",
                    why="Why: the map-mean filter excludes the map before detection",
                    fix="pass fields=[] with excluded_reason, or excluded_reason=None",
                )
            )
        fields = _owned_field_bins(self.fields, owner="PlaceFieldsResult")
        every = np.concatenate(fields) if fields else np.empty(0, dtype=np.int64)
        shared, counts = np.unique(every, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(
                _format_error(
                    f"PlaceFieldsResult: fields share bin "
                    f"{int(shared[np.argmax(counts > 1)])}.",
                    why="Why: detection assigns each bin to at most one field, "
                    "and to_dataframe() labels each bin with its one field",
                    fix="pass disjoint fields, each bin listed once, as "
                    "detect_place_fields returns them",
                )
            )
        rate = np.asarray(self.firing_rate, dtype=np.float64)
        n_bins = self.env.n_bins
        if rate.shape != (n_bins,) or any(
            bins.size and bins.max() >= n_bins for bins in fields
        ):
            raise ValueError(
                _format_error(
                    f"PlaceFieldsResult: firing_rate has shape {rate.shape} and "
                    f"the fields reach bin {max((int(b.max()) for b in fields if b.size), default=-1)}, "
                    f"but env has {n_bins} bins.",
                    why="Why: the fields are bins of the rate map, which has one "
                    "rate per bin of env",
                    fix="build the result with detect_place_fields(env, rates)",
                )
            )
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "firing_rate", rate)
        object.__setattr__(self, "bandwidth", float(self.bandwidth))
        self._own_arrays()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore owned field arrays after copying."""
        super().__setstate__(state)
        self.__post_init__()

    @property
    def n_excluded(self) -> int:
        """``1`` if a filter excluded the map, else ``0``."""
        return int(self.excluded_reason is not None)

    @property
    def n_fields(self) -> int:
        """Number of detected fields."""
        return len(self.fields)

    @property
    def total_bins(self) -> int:
        """Number of bins across all fields."""
        return int(sum(len(bins) for bins in self.fields))

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, idx: int) -> NDArray[np.int64]:
        return self.fields[idx]

    def __iter__(self) -> Iterator[NDArray[np.int64]]:
        return iter(self.fields)

    def __bool__(self) -> bool:
        return len(self.fields) > 0

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics for the detected place fields.

        Returns
        -------
        dict
            Mapping with keys ``n_fields`` (int, number of detected fields),
            ``total_bins`` (int, bins across all fields), ``n_excluded``
            (int), and ``excluded_reason`` (str or ``None``).
        """
        return {
            "n_fields": self.n_fields,
            "total_bins": self.total_bins,
            "n_excluded": self.n_excluded,
            "excluded_reason": self.excluded_reason,
        }

    def _table_unit_id(self) -> Any:
        import pandas as pd

        return pd.NA if self.unit_id is None else self.unit_id

    def _columns_units(self, columns: Sequence[str]) -> dict[str, str]:
        """Unit of each table column: Hz, env units, measure units or none."""
        length = self.env.units or ""
        units = {
            "field": "",
            "bin": "",
            "n_bins": "",
            "area": _measure_units(self.env),
            "firing_rate": "Hz",
            "peak_rate": "Hz",
            "mean_rate": "Hz",
            "n_fields": "",
            "largest_field_area": _measure_units(self.env),
        }
        return {
            name: units.get(
                name,
                "rad"
                if name.endswith("angle")
                else length
                if name.startswith(("centroid_", "bin_center_"))
                else "",
            )
            for name in columns
        }

    def to_dataframe(self) -> pd.DataFrame:
        """One row per bin: its rate and the field it belongs to, if any.

        Returns
        -------
        pandas.DataFrame
            One row per bin of the environment, in bin order, with columns
            ``unit_id`` (``<NA>`` without one), ``bin`` (the bin index),
            ``bin_center_x``, ``bin_center_y``, ... (environment units;
            ``bin_center_distance`` and ``bin_center_angle`` (rad) on a polar
            environment), ``field`` (nullable integer: 0 for the field with
            the highest peak, ``<NA>`` for a bin outside every field) and
            ``firing_rate`` (Hz; NaN where the map is unsupported). One
            ``field`` column suffices because fields are disjoint: detection
            assigns each bin to at most one field, and a coalescent field
            (``subfields``) replaces the region it was found in rather than
            nesting inside it. Select ``df[df["field"] == k]`` for the
            bins of field ``k``. ``df.attrs`` holds ``units`` (per column),
            ``bandwidth`` and ``excluded_reason``. :meth:`fields_table` has
            one row per field.
        """
        import pandas as pd

        from neurospatial._results import _coord_dim_names

        names = [f"bin_center_{name}" for name in _coord_dim_names(self.env)]
        n_bins = self.env.n_bins
        field = pd.array([pd.NA] * n_bins, dtype="Int64")
        for k, bins in enumerate(self.fields):
            field[bins] = k
        centers = np.asarray(self.env.bin_centers, dtype=np.float64)
        columns = ["unit_id", "bin", *names, "field", "firing_rate"]
        df = pd.DataFrame(
            {
                "unit_id": pd.Series([self._table_unit_id()] * n_bins, dtype=object),
                "bin": np.arange(n_bins, dtype=np.int64),
                **{name: centers[:, i] for i, name in enumerate(names)},
                "field": field,
                "firing_rate": self.firing_rate.copy(),
            },
            columns=columns,
        )
        df.attrs.update(
            units=self._columns_units(columns[1:]),
            bandwidth=self.bandwidth,
            excluded_reason=self.excluded_reason,
        )
        return df

    def fields_table(self) -> pd.DataFrame:
        """One row per detected field, with its size, rates and centroid.

        Returns
        -------
        pandas.DataFrame
            Columns ``unit_id`` (``<NA>`` without one), ``field`` (0 for the
            field with the highest peak), ``n_bins``, ``area`` (summed bin
            measures, :func:`~neurospatial.encoding.field_size`: an area in a
            2-D arena, a length on a track), ``peak_rate`` (Hz), ``mean_rate``
            (Hz, mean over the field's bins weighted by their measure) and
            ``centroid_x``, ``centroid_y``, ... (the rate-weighted centroid,
            :func:`~neurospatial.encoding.rate_map_centroid`, in environment
            units). On a track, where the mean of a field around a corner
            lies off the track, the centroid is the rate-weighted mean
            distance along the track (``env.to_linear``), mapped back onto it
            (``env.linear_to_nd``); a field over a branch point or an
            ``edge_spacing`` gap has no such mean, and its centroid is the
            geodesic one, a field bin. On a polar environment the columns are
            ``centroid_distance`` and ``centroid_angle`` (rad) of the field
            bin nearest the rest along the bin graph
            (``rate_map_centroid(method="bin_graph")``), because averaging
            angles and distances is not a position. No
            rows when no field was detected. ``df.attrs`` holds
            ``units`` (per column), ``bandwidth`` and ``excluded_reason``.
            :meth:`to_dataframe` has one row per bin, labeled by field.
        """
        import pandas as pd

        from neurospatial._results import _coord_dim_names
        from neurospatial.encoding._field_metrics import (
            _track_centroid,
            field_size,
            rate_map_centroid,
        )

        measures = np.asarray(self.env.bin_sizes, dtype=np.float64)
        names = [f"centroid_{name}" for name in _coord_dim_names(self.env)]
        # Averaging (distance, angle) bin centers is not a position: a polar
        # centroid is the field bin nearest the rest along the bin graph, as
        # the any-angle path is not defined there. The mean of a field around
        # a track's corner is off the track, so a track's centroid is the mean
        # distance along it, mapped back onto it.
        method: Literal["euclidean", "geodesic", "bin_graph"] = (
            "bin_graph"
            if self.env.coordinate_kind == "polar"
            else "geodesic"
            if self.env.is_linearized_track
            else "euclidean"
        )
        rows = []
        for k, bins in enumerate(self.fields):
            rates = self.firing_rate[bins]
            centroid = (
                _track_centroid(self.env, self.firing_rate, bins)
                if self.env.is_linearized_track
                else None
            )
            if centroid is None:
                centroid = rate_map_centroid(
                    self.env, self.firing_rate, bins, method=method
                )
            rows.append(
                {
                    "unit_id": self._table_unit_id(),
                    "field": k,
                    "n_bins": len(bins),
                    "area": field_size(self.env, bins),
                    "peak_rate": float(np.max(rates)),
                    "mean_rate": float(np.average(rates, weights=measures[bins])),
                    **dict(zip(names, np.asarray(centroid, dtype=float), strict=True)),
                }
            )
        columns = [
            "unit_id",
            "field",
            "n_bins",
            "area",
            "peak_rate",
            "mean_rate",
            *names,
        ]
        df = pd.DataFrame(rows, columns=columns)
        df.attrs.update(
            units=self._columns_units(columns[2:]),
            bandwidth=self.bandwidth,
            excluded_reason=self.excluded_reason,
        )
        return df

    def summary_table(self) -> pd.DataFrame:
        """One row for the unit: its field count, largest field and peak.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``unit_id`` (``<NA>`` without one), with ``n_fields``,
            ``largest_field_area`` (the largest :func:`field_size` among the
            fields; NaN without a field) and ``peak_rate`` (Hz, the highest
            field peak; NaN without a field). ``df.attrs`` holds ``units``,
            ``bandwidth`` and ``excluded_reason``.
        """
        import pandas as pd

        from neurospatial.encoding._field_metrics import field_size

        areas = [field_size(self.env, bins) for bins in self.fields]
        peaks = [float(np.max(self.firing_rate[bins])) for bins in self.fields]
        columns = ["n_fields", "largest_field_area", "peak_rate"]
        df = pd.DataFrame(
            {
                "n_fields": [self.n_fields],
                "largest_field_area": [max(areas, default=np.nan)],
                "peak_rate": [max(peaks, default=np.nan)],
            },
            index=pd.Index([self._table_unit_id()], name="unit_id"),
        )
        df.attrs.update(
            units=self._columns_units(columns),
            bandwidth=self.bandwidth,
            excluded_reason=self.excluded_reason,
        )
        return df

    def peak_location(self) -> NDArray[np.float64]:
        """Coordinates of the highest field peak, in environment units.

        Returns
        -------
        ndarray, shape (n_dims,)
            Center of the bin with the highest rate among the field bins;
            NaN without a field.
        """
        if not self.fields:
            return np.full(self.env.n_dims, np.nan)
        bins = self.fields[0]
        peak = int(bins[np.argmax(self.firing_rate[bins])])
        return np.array(self.env.bin_centers[peak], dtype=np.float64)

    def plot(self, *, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Draw the rate map with each field's bins outlined.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when None.
        **kwargs
            Passed to ``env.plot_field`` for the rate map.

        Returns
        -------
        matplotlib.axes.Axes
            The rate map (colorbar "Firing rate (Hz)") with the field bins'
            centers marked, one color per field; on a track or 1-D
            environment, the rate curve with each field's bins marked at their
            linear position and rate.
        """
        kwargs.setdefault("colorbar_label", "Firing rate (Hz)")
        ax = self.env.plot_field(self.firing_rate, ax=ax, **kwargs)
        centers = np.asarray(self.env.bin_centers, dtype=np.float64)
        # A track's map is drawn as a curve along its linear coordinate.
        curve = self.env.is_linearized_track or self.env.n_dims == 1
        for k, bins in enumerate(self.fields):
            if curve:
                x = (
                    self.env.to_linear(centers[bins])
                    if self.env.is_linearized_track
                    else centers[bins, 0]
                )
                y = self.firing_rate[bins]
            else:
                x, y = centers[bins, 0], centers[bins, 1]
            ax.scatter(
                x,
                y,
                s=12,
                facecolors="none",
                edgecolors=f"C{k % 10}",
                label=f"field {k}",
            )
        if self.fields:
            ax.legend(loc="best", fontsize="small")
        return ax


#: Required GAM diagnostics -- non-``None`` for ``method="glm"``, ``None`` for
#: the ratio methods. ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` /
#: ``penalty_selected_by_reml`` are excluded because they are legitimately
#: ``None`` even for glm (fixed penalty, REML-skip, no data, or ``pooled=True``).
_GAM_REQUIRED_FIELDS = (
    "coefficients",
    "penalty_weights",
    "rank",
    "deviance",
    "converged",
    "n_iter",
)


def _check_gam_result_invariant(
    kind: str, fields: dict[str, Any], *, n_units: int | None
) -> None:
    """Enforce the ``None``-iff-glm invariant on a spatial-rate result.

    The GAM diagnostics travel together: they are all populated for
    ``method="glm"`` and all ``None`` for the ratio methods, and ``bandwidth`` is
    ``None`` exactly for glm. This makes the illegal states the type cannot
    express -- a ratio result carrying a stray GAM array, or a glm result missing
    or mis-shaping a diagnostic -- unrepresentable at construction (the classes
    are frozen and public, so this closes the gap the type alone leaves open).

    Parameters
    ----------
    kind : str
        Class name, for error messages.
    fields : dict
        The result's ``method``, ``bandwidth``, and the eleven GAM fields.
    n_units : int or None, keyword-only
        Number of units (``firing_rates.shape[0]``) for the plural class, or
        ``None`` for the singular class (whose GAM fields are a per-unit slice).
        Selects the expected glm array shapes.

    Raises
    ------
    ValueError
        If ``method`` is unknown, or the GAM fields / ``bandwidth`` are
        inconsistent with ``method`` (presence, ``bandwidth``, or shape).
    """
    from neurospatial.encoding._smoothing import _SPATIAL_METHODS

    method = fields["method"]
    if method not in _SPATIAL_METHODS:
        msg = (
            f"{kind} has method={method!r}, which is not a known estimator; "
            f"expected one of {list(_SPATIAL_METHODS)}."
        )
        raise ValueError(msg)

    is_glm = method == "glm"
    required = {name: fields[name] for name in _GAM_REQUIRED_FIELDS}
    # ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` /
    # ``penalty_selected_by_reml`` are legitimately ``None`` even for glm (fixed
    # penalty, REML-skip, no data, or pooled=True), so they are "optional".
    optional = {
        "penalty": fields["penalty"],
        "reml_objective": fields["reml_objective"],
        "reml_at_boundary": fields["reml_at_boundary"],
        "penalty_selected_by_reml": fields["penalty_selected_by_reml"],
    }
    if is_glm:
        missing = [name for name, value in required.items() if value is None]
        if missing:
            msg = (
                f"{kind} with method='glm' is missing required GAM field(s): "
                f"{missing}. glm results carry all of {list(_GAM_REQUIRED_FIELDS)}."
            )
            raise ValueError(msg)
        if fields["bandwidth"] is not None:
            msg = (
                f"{kind} with method='glm' must have bandwidth=None (glm has no "
                f"bandwidth); got {fields['bandwidth']!r}."
            )
            raise ValueError(msg)
        # Full shape coupling: a mis-shaped diagnostic (e.g. coefficients
        # (rank, 1) on a 2-unit result) would construct and only IndexError when a
        # unit is later indexed. Pin every GAM array shape here. The singular
        # class holds a per-unit slice; the plural holds the population arrays.
        rank = int(fields["rank"])
        if n_units is None:  # singular: per-unit slice
            expected = {
                "coefficients": (rank,),
                "penalty_weights": (rank,),
                "deviance": (),  # scalar
            }
        else:  # plural: population arrays
            expected = {
                "coefficients": (rank, n_units),
                "penalty_weights": (rank,),
                "deviance": (n_units,),
            }
        for name, want in expected.items():
            got = tuple(np.shape(fields[name]))
            if got != want:
                raise ValueError(
                    f"{kind} glm field {name!r} has shape {got} but expected "
                    f"{want} (rank={rank}"
                    + (f", n_units={n_units}" if n_units is not None else "")
                    + ")."
                )
        # Per-unit fields (pooled=False) are scalar on the singular slice and
        # either scalar or a ``(n_units,)`` vector on the plural class; a stray
        # extra axis is a construction bug (indexing / summary_table rely on it).
        for name in (
            "penalty",
            "reml_objective",
            "reml_at_boundary",
            "penalty_selected_by_reml",
        ):
            value = fields[name]
            if value is None:
                continue
            ndim = np.ndim(value)
            if n_units is None:  # singular: must be a scalar
                if ndim != 0:
                    msg = (
                        f"{kind} (single-unit) glm field {name!r} must be a scalar; "
                        f"got shape {tuple(np.shape(value))}."
                    )
                    raise ValueError(msg)
            elif ndim not in (0, 1) or (
                ndim == 1 and tuple(np.shape(value)) != (n_units,)
            ):  # plural: scalar or (n_units,)
                msg = (
                    f"{kind} glm field {name!r} must be a scalar or a ({n_units},) "
                    f"vector; got shape {tuple(np.shape(value))}."
                )
                raise ValueError(msg)
        # State-machine coupling: the per-unit-lambda fields travel together, and
        # NWB persistence keys "is this per-unit?" off the provenance mask alone.
        # So the mask's presence must agree with the vector-ness of the other
        # three (and with pooled=False) -- otherwise an inconsistent result (e.g.
        # pooled=True with a vector penalty and no mask) constructs cleanly and
        # only fails deep in the NWB writer. per-unit here = a per-unit-shaped
        # slot: a ``(n_units,)`` vector on the plural class, a scalar on the
        # singular slice.
        mask = fields["penalty_selected_by_reml"]
        per_unit_slots = ("penalty", "reml_objective", "reml_at_boundary")

        def _is_per_unit_shaped(value: Any) -> bool:
            if value is None:
                return False
            return np.ndim(value) == (0 if n_units is None else 1)

        if mask is not None:
            # Per-unit result: only under pooled=False, with the mask AND all
            # three slots populated and per-unit-shaped (a scalar mask alongside
            # vector slots is exactly what crashes the NWB writer at len()).
            if fields["pooled"] is not False:
                msg = (
                    f"{kind} carries penalty_selected_by_reml (a per-unit result) "
                    f"but pooled={fields['pooled']!r}; per-unit lambda is only "
                    "produced under pooled=False."
                )
                raise ValueError(msg)
            missing_vec = [
                n
                for n in (*per_unit_slots, "penalty_selected_by_reml")
                if not _is_per_unit_shaped(fields[n])
            ]
            if missing_vec:
                msg = (
                    f"{kind} carries penalty_selected_by_reml (a per-unit result) "
                    f"but {missing_vec} are not per-unit values; penalty, "
                    "reml_objective, reml_at_boundary, and penalty_selected_by_reml "
                    "must all be per-unit when the provenance mask is present."
                )
                raise ValueError(msg)
            # Semantic coupling of the per-unit provenance (values, not just
            # shapes), so a corrupt record cannot round-trip through NWB:
            selected = np.asarray(mask, dtype=bool)
            penalty_vals = np.asarray(fields["penalty"], dtype=np.float64)
            reml_vals = np.asarray(fields["reml_objective"], dtype=np.float64)
            # Every applied lambda is a finite, positive smoothing penalty.
            if not np.all(np.isfinite(penalty_vals) & (penalty_vals > 0.0)):
                msg = (
                    f"{kind} per-unit penalty must be finite and > 0 (an applied "
                    f"smoothing lambda); got {penalty_vals}."
                )
                raise ValueError(msg)
            # A REML-selected unit carries a FINITE objective; a pooled-fallback
            # unit carries the ``nan`` sentinel specifically (not ``inf``). Check
            # the two partitions explicitly -- ``isfinite`` alone would let a
            # non-finite ``inf`` masquerade as the fallback sentinel. ``atleast_1d``
            # keeps the boolean-partition indexing valid on the singular scalar
            # slice.
            sel_1d = np.atleast_1d(selected)
            reml_1d = np.atleast_1d(reml_vals)
            if not (
                np.all(np.isfinite(reml_1d[sel_1d]))
                and np.all(np.isnan(reml_1d[~sel_1d]))
            ):
                msg = (
                    f"{kind} reml_objective must be finite for REML-selected units "
                    "and the nan sentinel for pooled-fallback units; got "
                    f"selected={selected}, reml_objective={reml_vals}."
                )
                raise ValueError(msg)
            # The per-unit vector path only exists with >= 1 informative unit
            # (population level). A singular slice may legitimately be a lone
            # fallback unit, so only pin this on the plural class.
            if n_units is not None and not selected.any():
                msg = (
                    f"{kind} is a per-unit result but penalty_selected_by_reml is "
                    "all False; the per-unit lambda path requires at least one "
                    "informative (REML-selected) unit."
                )
                raise ValueError(msg)
        else:
            # Non-per-unit result: none of the three may be a per-unit vector
            # (this is exactly the state that crashes NWB writing).
            stray_vec = [
                n
                for n in per_unit_slots
                if fields[n] is not None and np.ndim(fields[n]) != 0
            ]
            if stray_vec:
                msg = (
                    f"{kind} has vector {stray_vec} but no penalty_selected_by_reml "
                    "mask; per-unit penalty / reml_objective / reml_at_boundary "
                    "require the provenance mask (a per-unit pooled=False result). "
                    "For a shared/fixed lambda these must be scalar or None."
                )
                raise ValueError(msg)
        # A boundary flag only exists when REML produced an objective, so
        # ``reml_at_boundary`` being set requires ``reml_objective`` set too. The
        # reverse is NOT required: a schema-2.1 file records ``reml_objective`` but
        # no ``reml_at_boundary`` (it predates the flag), and reads back with the
        # objective set and the boundary flag ``None`` -- a legitimate state.
        if fields["reml_at_boundary"] is not None and fields["reml_objective"] is None:
            msg = (
                f"{kind} has reml_at_boundary={fields['reml_at_boundary']!r} but "
                "reml_objective=None; a boundary flag requires REML to have run "
                "(so reml_objective must be populated)."
            )
            raise ValueError(msg)
        # Symmetrically, a REML objective only exists when a lambda was applied, so
        # a set ``reml_objective`` requires a set ``penalty`` (both come from the
        # same selection; None together on REML-skip / no-data).
        if fields["reml_objective"] is not None and fields["penalty"] is None:
            msg = (
                f"{kind} has reml_objective={fields['reml_objective']!r} but "
                "penalty=None; a REML objective requires an applied penalty."
            )
            raise ValueError(msg)
        # The boolean diagnostics must be boolean (a float boundary flag or an
        # integer provenance mask would round-trip wrong through NWB).
        for name in ("reml_at_boundary", "penalty_selected_by_reml"):
            value = fields[name]
            if value is not None and np.asarray(value).dtype != np.bool_:
                msg = (
                    f"{kind} glm field {name!r} must be boolean; got dtype "
                    f"{np.asarray(value).dtype}."
                )
                raise ValueError(msg)
        # ``pooled`` is a concrete bool for every glm result (the only reliable
        # NWB source, since scalar cases are value-identical under both settings).
        # Checked last so a more fundamental shape/required error surfaces first.
        if not isinstance(fields["pooled"], bool):
            msg = (
                f"{kind} with method='glm' must carry pooled=True/False; got "
                f"{fields['pooled']!r}."
            )
            raise ValueError(msg)
    else:
        stray = [
            name
            for name, value in {**required, **optional}.items()
            if value is not None
        ]
        if fields["pooled"] is not None:
            stray.append("pooled")
        if stray:
            msg = (
                f"{kind} with ratio method={fields['method']!r} must not carry GAM "
                f"field(s): {stray}. GAM diagnostics belong only to method='glm'."
            )
            raise ValueError(msg)
        if fields["bandwidth"] is None:
            msg = (
                f"{kind} with ratio method={fields['method']!r} must have a float "
                "bandwidth; got None (bandwidth=None is reserved for method='glm')."
            )
            raise ValueError(msg)


SPATIAL_ESTIMATORS: Mapping[str, BandwidthRule] = MappingProxyType(
    {**KERNEL_ESTIMATORS, "binned": "nonnegative", "glm": "none"}
)


def _index_per_unit(value: Any, idx: int) -> Any:
    """Slice a possibly-per-unit GAM field to a single unit (for indexing).

    ``None`` and shared **scalars** carry through unchanged (a shared ``lambda``
    or ``None`` is the same for every unit); a per-unit ``(n_units,)`` vector
    (``pooled=False``) is sliced to its ``idx`` element as a Python scalar. Keeps
    ``rates[i]`` field-for-field equal to the singular ``compute_spatial_rate``.
    """
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:  # shared scalar -> preserve the original Python value/type
        return value
    return arr[idx].item()


def _test_outcome(passed: NDArray[np.bool_], *, unknown: bool) -> NDArray[np.float64]:
    """One test's outcome per unit: 1.0 passes, 0.0 fails, NaN unknown."""
    if unknown:
        return np.full(len(passed), np.nan)
    return np.asarray(passed, dtype=np.float64)


def _cell_type_labels(
    outcomes: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.str_]:
    """Label each unit from its spatial-information, grid and border tests.

    Parameters
    ----------
    outcomes : mapping of str to ndarray, shape (n_units,)
        For ``"spatial_information"``, ``"grid_score"`` and
        ``"border_score"``: 1.0 where the unit passes that test, 0.0 where it
        fails, NaN where the test applies to the environment but could not be
        scored there.

    Returns
    -------
    ndarray of str, shape (n_units,)
        ``"grid"``, ``"border"`` or ``"grid+border"`` from the grid and border
        tests alone, each judged by its own statistic; ``"place"`` for a unit
        that passes the information test and neither of the others;
        ``"unclassified"`` for one that passes none; ``"undetermined"`` when a
        test that applies could not be scored, so the label is unknown.
    """
    info = outcomes["spatial_information"]
    grid = outcomes["grid_score"]
    border = outcomes["border_score"]
    labels = np.full(len(info), "unclassified", dtype="<U12")
    labels[info == 1] = "place"
    labels[grid == 1] = "grid"
    labels[border == 1] = "border"
    labels[(grid == 1) & (border == 1)] = "grid+border"
    labels[np.isnan(info) | np.isnan(grid) | np.isnan(border)] = "undetermined"
    return labels


def _cell_type_columns(
    outcomes: Mapping[str, NDArray[np.float64]],
) -> dict[str, Any]:
    """``passes_<statistic>``, ``is_place`` and ``cell_type`` columns.

    ``passes_<statistic>`` is ``<NA>`` where the outcome is unknown.
    ``is_place`` is ``passes_spatial_information`` and not a known grid or
    border cell: a grid or border test that could not be scored does not
    block it, unlike ``cell_type``, which is ``"undetermined"`` then.
    ``<NA>`` information gives ``<NA>`` unless the unit is a known grid or
    border cell (False).
    """
    import pandas as pd

    columns: dict[str, Any] = {
        f"passes_{name}": pd.array(
            [None if np.isnan(value) else bool(value) for value in values],
            dtype="boolean",
        )
        for name, values in outcomes.items()
    }
    grid_or_border = (outcomes["grid_score"] == 1) | (outcomes["border_score"] == 1)
    # Kleene logic: <NA> information stays <NA> unless the unit is grid or border.
    columns["is_place"] = columns["passes_spatial_information"] & pd.array(
        ~grid_or_border, dtype="boolean"
    )
    columns["cell_type"] = _cell_type_labels(outcomes)
    return columns


def _screen_scores(
    result: SpatialRateResult | SpatialRatesResult,
    *,
    boundary_source: BoundarySource,
    grid_scores: NDArray[np.float64] | None = None,
    border_scores: NDArray[np.float64] | None = None,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, str], tuple[str, ...]]:
    """Spatial information, grid and border scores of every unit's map.

    Returns
    -------
    scores : dict of str to ndarray, shape (n_units,)
        Scores keyed by statistic; NaN where a statistic cannot be scored.
    unavailable : dict of str to str
        Why each unscored statistic has no score.
    inapplicable : tuple of str
        Unscored statistics that do not apply to the environment.
    """
    from neurospatial.encoding._metrics import batch_spatial_information

    rates = np.atleast_2d(_to_numpy(result._supported_rates()))
    scores = {
        "spatial_information": np.asarray(
            batch_spatial_information(rates, _to_numpy(result.occupancy))
        )
    }
    unavailable: dict[str, str] = {}
    inapplicable: list[str] = []
    for name, given in (("grid_score", grid_scores), ("border_score", border_scores)):
        if given is not None:
            scores[name] = given
            continue
        scores[name], reason, applies = _cell_type_score(
            result.env, rates, name, boundary_source
        )
        if reason is not None:
            unavailable[name] = reason
        if not applies:
            inapplicable.append(name)
    return scores, unavailable, tuple(inapplicable)


def _grid_spacings(env: Environment, rates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Spacing of each map's hexagonal first ring of autocorrelogram peaks.

    Parameters
    ----------
    env : Environment
        Environment with a regular 2-D grid of rectangular cells.
    rates : ndarray, shape (n_units, n_bins)
        Rates in Hz, NaN outside the occupancy support.

    Returns
    -------
    ndarray, shape (n_units,)
        :func:`~neurospatial.encoding.grid.grid_scale` of each map, in
        environment units. NaN when the peaks nearest zero lag are not the
        first ring of one hexagon, and for an all-NaN or constant map.
    """
    from neurospatial.encoding.grid import (
        autocorrelation_pixel_size,
        grid_scale,
        spatial_autocorrelation,
    )

    spacing = np.full(len(rates), np.nan)
    if len(rates) == 0:
        return spacing
    pixel_size = autocorrelation_pixel_size(env)
    for i, rate in enumerate(rates):
        finite = rate[np.isfinite(rate)]
        if finite.size and np.ptp(finite) > 0:
            spacing[i] = grid_scale(
                spatial_autocorrelation(env, rate), pixel_size=pixel_size
            )
    return spacing


def _screen_outcomes(
    result: SpatialRateResult | SpatialRatesResult,
    scores: Mapping[str, NDArray[np.float64]],
    unavailable: Mapping[str, str],
    inapplicable: tuple[str, ...],
    *,
    information: NDArray[np.float64],
    min_spatial_info: float,
    min_grid_score: float,
    min_border_score: float,
) -> dict[str, NDArray[np.float64]]:
    """Fixed-threshold outcome of each test; a NaN score fails its test.

    ``information`` is the bias-corrected spatial information the place
    screen compares with its cutoff, shape (n_units,). A unit with fewer
    spikes than its map has independent bins has no defined bias, so its
    information test is unknown (NaN outcome, label ``"undetermined"``), not
    failed. A grid score at its
    cutoff passes only when the unit's map has a hexagonal first ring of
    autocorrelogram peaks (see :func:`_grid_spacings`); the ring is read
    from ``result``'s maps for those units alone.
    """
    from neurospatial.encoding._metrics import _information_bias_terms

    too_few = _information_bias_terms(result).too_few
    with np.errstate(invalid="ignore"):
        grid = scores["grid_score"] >= min_grid_score
    if grid.any():
        rates = np.atleast_2d(_to_numpy(result._supported_rates()))
        grid[grid] = np.isfinite(_grid_spacings(result.env, rates[grid]))
    passed = {
        "spatial_information": np.asarray(information) >= min_spatial_info,
        "grid_score": grid,
        "border_score": scores["border_score"] >= min_border_score,
    }
    outcomes = {
        name: _test_outcome(
            values, unknown=name in unavailable and name not in inapplicable
        )
        for name, values in passed.items()
    }
    outcomes["spatial_information"][too_few] = np.nan
    return outcomes


def _glm_summary_columns(
    result: SpatialRateResult | SpatialRatesResult,
) -> dict[str, Any]:
    """Existing GAM diagnostics, broadcasting scalar slices to a table row."""
    data: dict[str, Any] = {}
    if result.method == "glm":
        data["penalty"] = result.penalty
        data["rank"] = result.rank
        data["deviance"] = np.asarray(result.deviance)
        data["converged"] = result.converged
        data["n_iter"] = result.n_iter
        data["reml_objective"] = result.reml_objective
        data["reml_at_boundary"] = result.reml_at_boundary
        data["pooled"] = result.pooled
        # Per-unit provenance mask only exists for the per-unit (pooled=False)
        # path; skip it (rather than write an all-None column) otherwise.
        if result.penalty_selected_by_reml is not None:
            data["penalty_selected_by_reml"] = np.asarray(
                result.penalty_selected_by_reml
            )

    return data


def _border_baseline(
    result: SpatialRateResult | SpatialRatesResult, boundary_source: BoundarySource
) -> dict[str, float]:
    """How much wall the occupancy observes, and the untuned border score.

    Both depend on the environment and the occupancy support only, so they
    are the same for every unit of a summary table.
    """
    from neurospatial.encoding._provenance import occupancy_support
    from neurospatial.encoding.border import (
        observed_wall_fraction,
        untuned_border_score,
    )

    support = np.asarray(occupancy_support(result), dtype=bool)
    observed = np.where(support, 1.0, np.nan)
    return {
        "border_observed_wall_fraction": observed_wall_fraction(
            result.env, observed, boundary_source=boundary_source
        ),
        "border_score_untuned": untuned_border_score(
            result.env, support=support, boundary_source=boundary_source
        ),
    }


def _spatial_summary_frame(
    result: SpatialRateResult | SpatialRatesResult,
    *,
    index: Sequence[Hashable],
    include_classification: bool,
    boundary_source: BoundarySource,
) -> pd.DataFrame:
    """Build spatial metric columns for single and population results."""
    import pandas as pd

    from neurospatial.encoding._metrics import (
        _debiased_information,
        _information_bias,
        batch_sparsity,
    )
    from neurospatial.encoding.border import _border_field_minimum

    rates = np.atleast_2d(_to_numpy(result._supported_rates()))
    occupancy = _to_numpy(result.occupancy)
    peaks = np.atleast_2d(result.peak_location())
    scores, unavailable, inapplicable = _screen_scores(
        result, boundary_source=boundary_source
    )
    border_attrs = (
        {}
        if "border_score" in unavailable
        else _border_baseline(result, boundary_source)
    )
    bias = _information_bias(result)
    columns: dict[str, Any] = {
        "peak_firing_rate": np.atleast_1d(result.peak_firing_rate()),
        "spatial_information": scores["spatial_information"],
        "spatial_information_bias": bias,
        "sparsity": np.asarray(batch_sparsity(rates, occupancy)),
        "grid_score": scores["grid_score"],
        "border_score": scores["border_score"],
        "peak_x": peaks[:, 0],
        "peak_y": peaks[:, 1] if peaks.shape[1] > 1 else np.full(len(rates), np.nan),
        **_glm_summary_columns(result),
        **result._accounting_columns(),
    }
    if include_classification:
        columns.update(
            _cell_type_columns(
                _screen_outcomes(
                    result,
                    scores,
                    unavailable,
                    inapplicable,
                    information=_debiased_information(result, bias),
                    **PLACE_GRID_BORDER_THRESHOLDS,
                )
            )
        )
    df = pd.DataFrame(columns, index=pd.Index(list(index), name="unit_id"))
    df.attrs["border_boundary_source"] = boundary_source
    df.attrs["border_field_minimum"] = _border_field_minimum(result.env)
    df.attrs.update(border_attrs)
    if unavailable:
        df.attrs["unavailable_metrics"] = unavailable
    df.attrs["units"] = {
        "peak_firing_rate": "Hz",
        "spatial_information": "bits/spike",
        "spatial_information_bias": "bits/spike",
        "peak_x": result.env.units or "",
        "peak_y": result.env.units or "",
    }
    df.attrs["classification_thresholds"] = dict(PLACE_GRID_BORDER_THRESHOLDS)
    df.attrs.update(result._analysis_attrs())
    return df


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class SpatialRateResult(SingleUnitRateMixin):
    """Result of spatial rate computation for a single neuron.

    This class wraps a spatial firing rate map with its associated metadata
    (occupancy, environment, smoothing parameters). It inherits from
    `SpatialResultMixin` for common methods like `peak_location()` and
    `peak_firing_rate()`.

    Parameters
    ----------
    firing_rate : ArrayLike
        Firing rate map in Hz. Shape is (n_bins,) where n_bins is the
        number of active bins in the environment. Can contain NaN for
        bins with insufficient occupancy.
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The immutable environment these rates were estimated on. It is their
        spatial context: bin coordinates and order, bin measures, units, frame
        and connectivity. Environments never change; ``with_metadata`` returns
        a new one, so relabelling later cannot change this result.
    method : str
        Estimator used: "diffusion_kde", "gaussian_kde", "binned", or "glm".
    bandwidth : float or None
        Smoothing bandwidth in the same units as the environment's bin_size, or
        ``None`` for ``method="glm"`` (which has no bandwidth).

    Attributes
    ----------
    firing_rate : ArrayLike
        Firing rate map in Hz. Shape is (n_bins,).
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment.
    method : str
        Estimator used.
    bandwidth : float or None
        Smoothing bandwidth (``None`` for ``method="glm"``).
    unit_id : int or str or None
        Identifier for this unit. Set automatically when indexing/iterating a
        population result (``rates[i].unit_id == rates.unit_ids[i]``); ``None``
        for a standalone single-unit computation.
    coefficients, penalty, penalty_weights, rank, deviance, converged, n_iter, \
reml_objective, reml_at_boundary, penalty_selected_by_reml, pooled
        GAM diagnostics for ``method="glm"`` (all ``None`` for the ratio
        methods): the per-unit slice of a population fit -- ``coefficients``
        ``(rank,)``, scalar ``deviance``, and the shared/batch-scalar
        ``penalty`` / ``penalty_weights`` ``(rank,)`` / ``rank`` / ``converged``
        / ``n_iter`` / ``reml_objective``. Under a per-unit fit (``pooled=False``)
        ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` are this unit's
        own scalar values and ``penalty_selected_by_reml`` its provenance
        (``False`` = pooled-``λ`` fallback zero-spike unit); ``pooled`` records
        the flag (``None`` for ratio results). See :class:`SpatialRatesResult`.
    _unit_ids_generated : bool, default=False
        True when the unit labels were generated (no ``unit_ids`` or
        ``unit_id`` supplied), so pairing with spike inputs stays positional.
        Set by the compute functions and by indexing; not an analysis choice.
    n_spikes_outside_intervals : int or ndarray of int64, shape (n_units,), or None
        Each unit's spikes outside the analyzed time intervals
        (``analysis_settings["observed_intervals"]``), which are in no rate:
        an int for one unit, one count per unit for a population. ``None``
        when unknown, as for a result built by hand. Also a
        ``summary_table()`` column.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to be in ``spatial_support``: the
        threshold the rates were estimated with. ``analysis_settings`` only
        records it.
    analysis_settings : dict or None
        Resolved epochs, observed intervals, gap/speed/occupancy gates and
        estimator settings. None for manually constructed results.
    spatial_support : ndarray of bool, shape (n_bins,)
        Observed bins meeting the occupancy threshold with finite rates.
    spike_window : NDArray[np.float64] or None
        Normalized spike-recording windows in seconds, copied read-only.
        ``None`` records the assumption that spikes were observed whenever
        position was recorded.
    spike_window_assumed : bool
        Whether spike-recording coverage was assumed rather than supplied.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    Inherits from `SpatialResultMixin`, which provides:

    - `peak_location()`: Returns (n_dims,) coordinates of peak firing
    - `peak_firing_rate()`: Returns scalar max firing rate

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rate

    >>> # Create a simple environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 50, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Compute result (returns SpatialRateResult)
    >>> times = np.linspace(0, 50, 500)
    >>> spike_times = np.sort(rng.uniform(0, 50, 30))
    >>> result = compute_spatial_rate(
    ...     env, spike_times, times, positions, bandwidth=10.0
    ... )

    >>> # Access fields
    >>> result.firing_rate.shape == (env.n_bins,)
    True
    >>> result.method
    'diffusion_kde'

    >>> # Use mixin methods
    >>> peak_coords = result.peak_location()  # (n_dims,)
    >>> max_rate = result.peak_firing_rate()  # float

    See Also
    --------
    SpatialRatesResult : Batch version for multiple neurons
    compute_spatial_rate : Function to compute this result
    SpatialResultMixin : Provides peak_location() and peak_firing_rate()
    """

    firing_rate: ArrayLike
    occupancy: ArrayLike
    env: Environment
    method: str
    bandwidth: float | None
    unit_id: int | str | None = None
    # GAM (``method="glm"``) fields -- all ``None`` for the ratio methods. See
    # ``SpatialRatesResult`` for the population-level shapes; the singular fields
    # are the per-unit slices stamped when indexing a population result.
    coefficients: NDArray[np.float64] | None = None
    penalty: float | NDArray[np.float64] | None = None
    penalty_weights: NDArray[np.float64] | None = None
    rank: int | None = None
    deviance: float | NDArray[np.float64] | None = None
    converged: bool | None = None
    n_iter: int | None = None
    reml_objective: float | None = None
    # Per-unit-lambda (``pooled=False``) diagnostics: the per-unit slice of the
    # population fit (scalars here). ``None`` for ratio methods. For glm:
    # ``reml_at_boundary`` is a scalar bool when REML ran (both pooled settings)
    # and ``None`` when it did not (fixed penalty / r==0 / no data);
    # ``penalty_selected_by_reml`` is ``None`` except for a per-unit
    # (``pooled=False``) slice; ``pooled`` is ``True`` / ``False`` for glm.
    reml_at_boundary: bool | None = None
    penalty_selected_by_reml: bool | None = None
    pooled: bool | None = None

    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: int | None = field(default=None, kw_only=True)
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )

    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, compare=False, repr=False
    )

    _ESTIMATORS: ClassVar[Mapping[str, BandwidthRule]] = SPATIAL_ESTIMATORS

    def summary_table(
        self,
        include_classification: bool = True,
        *,
        boundary_source: BoundarySource = "physical",
    ) -> pd.DataFrame:
        """Per-unit metrics with the same columns as the population table.

        Parameters
        ----------
        include_classification : bool, default=True
            Include the fixed-threshold screen columns ``passes_*``,
            ``is_place`` and ``cell_type``.

        boundary_source : {'physical', 'analysis'}, default='physical'
            Border geometry source. Unsupported physical metrics are NaN and
            explained in ``df.attrs['unavailable_metrics']``; other metrics
            remain available. The source is recorded in table attributes.

        Returns
        -------
        pandas.DataFrame
            One row indexed by this unit label, or <NA> when no label was supplied.
            The columns match :meth:`SpatialRatesResult.summary_table`.

        Notes
        -----
        The ``passes_*``, ``is_place`` and ``cell_type`` columns are a
        fixed-threshold screen with known false positives; see :meth:`SpatialRatesResult.label_cell_types` for the cutoffs, their
        provenance and measured false-positive rates. The cutoffs are in
        ``df.attrs["classification_thresholds"]``. For a verdict, call
        :func:`spatial_cell_type_significance` with the raw arrays.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.
        The border score's baseline is ``df.attrs["border_score_untuned"]``,
        the score of a map that fires everywhere the animal was (0.5 in a
        convex arena with every wall visited, less where parts of the walls
        were not), and ``df.attrs["border_observed_wall_fraction"]`` is the
        fraction of the walls' length that was observed. Unobserved wall
        counts as not covered, so when that fraction is below 1 a border score
        can understate a field's coverage but never overstate it.

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
        >>> table = result[0].summary_table()
        >>> len(table)
        1
        """
        return _spatial_summary_frame(
            self,
            index=self._row_unit_ids().tolist(),
            include_classification=include_classification,
            boundary_source=boundary_source,
        )

    def _headline_metrics(self) -> dict[str, float]:
        """Cheap metrics for the singular summary (NaN when undefined)."""
        return {
            "spatial_information": float(self.spatial_information()),
            "sparsity": float(self.sparsity()),
        }

    def _check_family(self) -> None:
        # Enforce the None-iff-glm invariant: the GAM diagnostics are all present
        # (and correctly per-unit-shaped) for method="glm" with bandwidth=None, or
        # all absent for a ratio method. n_units=None -> the singular per-unit slice.
        _check_gam_result_invariant(
            "SpatialRateResult",
            {
                "method": self.method,
                "bandwidth": self.bandwidth,
                "coefficients": self.coefficients,
                "penalty": self.penalty,
                "penalty_weights": self.penalty_weights,
                "rank": self.rank,
                "deviance": self.deviance,
                "converged": self.converged,
                "n_iter": self.n_iter,
                "reml_objective": self.reml_objective,
                "reml_at_boundary": self.reml_at_boundary,
                "penalty_selected_by_reml": self.penalty_selected_by_reml,
                "pooled": self.pooled,
            },
            n_units=None,
        )

    def plot(self, *, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Plot the spatial rate map.

        Delegates to the environment's plot_field method for consistent
        visualization across the codebase.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - colorbar : bool, default=True
            - colorbar_label : str, default="Firing rate (Hz)"

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
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
        >>> ax = result.plot()
        >>> type(ax).__name__
        'Axes'
        """
        kwargs.setdefault("colorbar_label", "Firing rate (Hz)")
        return self.env.plot_field(_to_numpy(self.firing_rate), ax=ax, **kwargs)

    def spatial_information(self) -> float | Any:
        """Skaggs spatial information (bits per spike).

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        float | jax.Array
            Spatial information in bits/spike. Always non-negative, and 0.0
            for uniform firing. NaN for a silent cell (no spikes) or a map with
            no supported bins: bits per spike are undefined without spikes.
            Bins outside ``spatial_support`` are excluded even when filled.

            **Backend-aware**: Returns float for NumPy input,
            JAX scalar for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        **Interpretation**:

        - Place cells typically have 1-3 bits/spike
        - Higher values indicate more spatially selective firing
        - Zero means uniform firing (no spatial selectivity)

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
               An information-theoretic approach to deciphering the hippocampal code.

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
        >>> info = result.spatial_information()
        >>> bool(info >= 0.0)
        True

        See Also
        --------
        neurospatial.encoding._metrics.spatial_information : Underlying computation
        """
        from neurospatial.encoding._metrics import spatial_information

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return spatial_information(self._supported_rates(), occupancy=self.occupancy)

    def spatial_information_bias(self) -> float:
        """Expected spatial information of an untuned cell (bits/spike).

        Plug-in Skaggs information is positive even for a cell that fires at
        the same rate everywhere, because Poisson noise makes its rate map
        uneven. This is the information such an untuned cell, firing at this
        unit's mean rate, is expected to show on this occupancy with this
        estimator and bandwidth. ``spatial_information() -
        spatial_information_bias()`` estimates the information beyond chance,
        the value :meth:`is_place_cell` compares with its cutoff.

        Returns
        -------
        float
            Expected untuned information in bits/spike. NaN for a silent unit
            and, without a warning, when the unit has fewer spikes than the
            map has effectively independent bins: the bias is then
            undetermined, not zero (see Notes).

        Notes
        -----
        **Derivation.** Every ratio estimator here is linear in spike counts at
        fixed occupancy: ``r = A n``, where column ``j`` of ``A`` is the map one
        spike in bin ``j`` produces, and a constant rate is reproduced
        (``sum_j A_ij t_j = 1``). For an untuned Poisson cell,
        ``n_j ~ Poisson(lam t_j)``. With ``x_i = r_i / rbar`` and
        ``sum_i p_i x_i = 1``, a second-order expansion of ``x log2 x`` about 1
        gives ``I ~= sum_i p_i (r_i - rbar)^2 / (2 ln2 rbar^2)``, and

        ``E[sum_i p_i (r_i - rbar)^2] = lam V``,
        ``V = sum_j t_j (sum_i p_i A_ij^2 - (sum_i p_i A_ij)^2)``,

        so the expected information is ``V / (2 ln2 lam)``, evaluated at the
        unit's mean rate. Writing ``K = V T`` (``T`` the total occupancy) gives
        ``K / (2 ln2 N)`` for ``N`` spikes: ``K`` is the number of effectively
        independent bins. Without smoothing (``A = diag(1/t)``), ``K`` is the
        number of supported bins minus one, the form of the Miller (1955) and
        Treves & Panzeri (1995) correction; occupancy weights cancel. Smoothing
        correlates neighboring bins, so ``K`` is the occupancy-weighted
        variance of each one-spike footprint, summed: about 46 for a 1 m box
        with 5 cm bins and a 5 cm kernel (441 bins), 11 with a 10 cm kernel.
        ``method="glm"`` uses the same expression with the GAM's response to
        one spike at a constant-rate fit, ``A = lam B H^-1 B^T``,
        ``H = lam B^T T B + penalty * diag(d)``; unpenalized, ``K`` is the basis
        rank minus one.

        The extension beyond the unsmoothed map, to kernel smoothing and the
        GAM, is neurospatial's own derivation. On an unsmoothed map it gives
        the ``(K - 1) / (2 ln 2 N)`` form, which a test checks; the smoothed
        and GAM cases rest on simulation.

        **Range.** The expansion is a series in the relative variance of a
        bin's rate, ``K / N`` on average, and fails once it reaches 1. With
        fewer spikes than ``K`` the bias is NaN, so screens refuse the unit.

        **Validation.** Untuned Poisson units (40 per spike count) on an
        Ornstein-Uhlenbeck trajectory: predicted and simulated mean
        information agreed within 10% for ``N >= K`` with ``diffusion_kde``
        and ``gaussian_kde`` on 1 m and 2 m boxes (2.5 and 5 cm bins, 5 and
        10 cm kernels) and a 200 cm track. A repeat for ``binned`` on the
        same arenas, with N = K, 2K and 4K spikes on 20-minute trajectories,
        agreed within 14% for unsmoothed maps (the default; boxes within 10%)
        and within 33% when smoothed with a 5 or 10 cm bandwidth; that repeat
        is harsher than the first check (``diffusion_kde`` came within 16%
        in it). On two 10-minute Ornstein-Uhlenbeck trajectories in a 1 m box
        (5 cm bins), 400 untuned cells per setting: with ``diffusion_kde``
        and ``gaussian_kde`` at 2.5, 5 and 10 cm and 1000 spikes, predicted
        and simulated means agreed within 1 standard error of the simulated
        mean (0.3-1.6%); at 100 spikes within 4% (2.5 cm leaves more
        independent bins than spikes, so the bias is NaN). A GAM at a fixed
        penalty of 1 agreed within 0.1% (60 cells, 1000 spikes). With REML
        the penalty chosen for untuned cells is so large that both the bias
        and the information are near 1e-4 bits/spike.

        ``method="binned"`` is less accurate when bins are visited unevenly.
        On four pairs of those 10-minute Ornstein-Uhlenbeck trajectories
        (1 m box, 5 cm bins, 379-394 of the 400 bins visited, for 0.03 to
        6 s each; 400 untuned cells per setting), maps smoothed with a 5 or
        10 cm bandwidth predicted 6-20% more than the simulated mean at 100
        spikes and 0-6% more at 1000 spikes. The rate is formed in each bin
        before smoothing, so one spike in a briefly visited bin gives a
        large rate there; the correction counts its square, but information
        grows more slowly. The screens then subtract too much and are
        conservative. With every bin visited about equally (positions drawn
        uniformly) the prediction was within about 2%. Unsmoothed maps at
        1000 spikes came out 7-8% below the simulated mean (11% below with
        evenly visited bins): the Miller-Madow form is a first correction
        and underestimates once ``K / N`` is not small, so the screens
        subtract 0.02-0.04 bits/spike too little. With 100 spikes these
        maps have more independent bins than spikes and the bias is NaN.

        For a tuned cell the plug-in bias is smaller (its spikes come from
        fewer bins), so the correction is conservative.

        References
        ----------
        .. [1] Miller, G. A. (1955). Note on the bias of information estimates.
               In Information Theory in Psychology, 95-100.
        .. [2] Treves, A., & Panzeri, S. (1995). The upward bias in measures of
               information derived from limited data samples. Neural
               Computation, 7(2), 399-407.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 100, (36000, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
        >>> times = np.arange(36000) / 30.0
        >>> spike_times = np.sort(rng.uniform(0, 1200, 600))  # untuned, 0.5 Hz
        >>> result = compute_spatial_rate(env, spike_times, times, positions)
        >>> info = result.spatial_information()
        >>> bias = result.spatial_information_bias()
        >>> bool(abs(info - bias) < 0.02)
        True
        """
        from neurospatial.encoding._metrics import _information_bias

        return float(_information_bias(self)[0])

    def sparsity(self) -> float | Any:
        """Sparsity of spatial firing.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        float | jax.Array
            Sparsity value in range [0, 1]. NaN for a silent cell or a map
            with no supported bins. Bins outside ``spatial_support`` are
            excluded even when filled.

            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

            **Backend-aware**: Returns float for NumPy input,
            JAX scalar for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1996) formula:

        .. math::

            S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
               precession in hippocampal neuronal populations.

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
        >>> spars = result.sparsity()
        >>> bool(0.0 <= spars <= 1.0)
        True

        See Also
        --------
        neurospatial.encoding._metrics.sparsity : Underlying computation
        """
        from neurospatial.encoding._metrics import sparsity

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return sparsity(self._supported_rates(), occupancy=self.occupancy)

    def grid_score(self) -> float:
        """Grid score (hexagonal periodicity).

        Quantifies the hexagonal periodicity of the firing rate map, which
        is characteristic of grid cells. Higher values indicate stronger
        hexagonal grid patterns.

        Returns
        -------
        float
            Grid score in range [-2, 2]; higher means stronger sixfold
            rotational symmetry, about 0 none, below 0 another symmetry.
            0.4 is a common screening value, not a verdict: on circularly
            time-shifted spike trains of simulated cells 1-8% of shuffled
            maps reach it. Calling a grid cell needs a shuffle null from
            your own recording (circular time shifts of the spike train).
            NaN when the map is all NaN or constant, or its autocorrelogram
            has no central peak bounded inside the map.

        Raises
        ------
        NotImplementedError
            If the environment has no regular 2D grid of rectangular cells
            (hexagonal, mesh, track or 3D layouts); use
            ``spatial_autocorrelation_radial`` there.

        Notes
        -----
        Computes the spatial autocorrelation of the firing rate map and
        extracts the grid score based on rotational symmetry:

        1. Compute the 2D spatial autocorrelogram
        2. Rotate it by 30°, 60°, 90°, 120°, 150° and correlate it with
           each rotation in annuli growing outward from the central peak
        3. Grid score = min(r60, r120) - max(r30, r90, r150), the best mean
           over three consecutive annulus radii

        This method delegates to ``neurospatial.encoding.grid.grid_score()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.

        See Also
        --------
        grid_properties : Full grid cell metrics (score, scale, orientation)

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> from neurospatial.simulation import GridCellModel, generate_poisson_spikes
        >>> times = np.arange(0.0, 900.0, 1 / 30)
        >>> # A Lissajous path covers a 100 cm box smoothly.
        >>> positions = 50 + 49 * np.column_stack(
        ...     [np.sin(0.31 * times), np.sin(0.47 * times + 0.4)]
        ... )
        >>> env = Environment.from_samples(positions, bin_size=4.0)
        >>> cell = GridCellModel(env, grid_spacing=40.0, max_rate=20.0)
        >>> rate = cell.firing_rate(times, positions)
        >>> spike_times = generate_poisson_spikes(rate, times, rng=1)
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=5.0
        ... )
        >>> bool(result.grid_score() > 0.4)
        True
        """
        from neurospatial.encoding.grid import grid_score as gs_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(self._supported_rates())
        finite = firing_rate[np.isfinite(firing_rate)]
        if finite.size == 0 or np.all(finite == finite[0]):
            # No spatial structure to autocorrelate: the score is undefined.
            return np.nan
        return gs_func(spatial_autocorrelation(self.env, firing_rate))

    def grid_properties(self) -> GridProperties:
        """Full grid cell metrics (score, scale, orientation).

        Returns a comprehensive set of grid cell metrics computed from the
        spatial autocorrelation of the firing rate map.

        Returns
        -------
        GridProperties
            Dataclass containing:

            - score : float
                Grid score in range [-2, 2]
            - scale : float
                Grid spacing in the environment's units
            - orientation : float
                Lattice angle in radians [0, pi/3), from +x (East)
                counterclockwise toward +y (North), modulo pi/3
            - orientation_std : float
                Standard deviation of orientation estimates, in radians
            - peak_coords : NDArray, shape (n_peaks, 2)
                Autocorrelogram peaks as ``(x, y)`` offsets from zero lag,
                in the environment's units
            - n_peaks : int
                Number of peaks found

        Raises
        ------
        NotImplementedError
            If the environment has no regular 2D grid of rectangular cells.

        Notes
        -----
        Delegates to ``neurospatial.encoding.grid.grid_properties()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.
        .. [2] Hafting, T., Fyhn, M., et al. (2005). Microstructure of a
               spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.

        See Also
        --------
        grid_score : Just the grid score
        neurospatial.encoding.grid.GridProperties : Return type details

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> from neurospatial.simulation import GridCellModel, generate_poisson_spikes
        >>> times = np.arange(0.0, 900.0, 1 / 30)
        >>> # A Lissajous path covers a 100 cm box smoothly.
        >>> positions = 50 + 49 * np.column_stack(
        ...     [np.sin(0.31 * times), np.sin(0.47 * times + 0.4)]
        ... )
        >>> env = Environment.from_samples(positions, bin_size=4.0)
        >>> cell = GridCellModel(env, grid_spacing=40.0, max_rate=20.0)
        >>> rate = cell.firing_rate(times, positions)
        >>> spike_times = generate_poisson_spikes(rate, times, rng=1)
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=5.0
        ... )
        >>> props = result.grid_properties()
        >>> round(float(props.scale) / 4.0)  # spacing in 4 cm pixels, 40 cm
        10
        """
        from neurospatial.encoding.grid import (
            autocorrelation_pixel_size,
            spatial_autocorrelation,
        )
        from neurospatial.encoding.grid import grid_properties as gp_func

        firing_rate = _to_numpy(self._supported_rates())
        autocorr = spatial_autocorrelation(self.env, firing_rate)
        # The autocorrelogram's square pixel side, a length; ``env.bin_sizes``
        # holds cell areas off a track.
        return gp_func(autocorr, pixel_size=autocorrelation_pixel_size(self.env))

    def border_score(
        self,
        threshold: float = 0.3,
        min_area: float | None = None,
        metric: StraightOrGeodesic = "geodesic",
        *,
        boundary_source: BoundarySource = "physical",
    ) -> float:
        """Border score of Solstad et al. (2008).

        Compares the largest fraction of one wall covered by one firing field
        with the firing-rate-weighted mean distance of the fields from the
        walls. Higher values indicate firing along a wall.

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to segment fields.
            Bins with firing rate >= threshold * peak are field bins.
        min_area : float, optional
            Smallest area of one field, in squared position units (for
            example cm²). Smaller fields are ignored. The default, None, is
            the 200 cm² of Solstad et al. (2008) in the environment's units,
            or at least 9 bins (the BNT/opexebo default) when the units are
            not a metric length.
        metric : {'geodesic', 'euclidean'}, default 'geodesic'
            Distance from field bins to the nearest wall.
            - 'geodesic': path through the environment (respects obstacles)
            - 'euclidean': straight-line distance

        boundary_source : {'physical', 'analysis'}, default='physical'
            The measured arena walls, or explicitly the outline of the
            retained bins. Physical mode requires measured wall geometry.

        Returns
        -------
        float
            Border score in range [-1, 1]: near +1 for a thin field along a
            whole wall, -1 for firing away from every wall. A uniform map
            scores about 0.5 on a square. NaN if no field reaches the
            minimum size.

        Notes
        -----
        Delegates to ``neurospatial.encoding.border.border_score()``, which
        describes the walls, fields and normalization.

        References
        ----------
        .. [1] Solstad, T., Boccara, C. N., et al. (2008). Representation of
               geometric borders in the entorhinal cortex. Science, 322(5909),
               1865-1868.

        See Also
        --------
        region_coverage : Coverage of specific spatial regions

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rate
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = np.sort(rng.uniform(0, 50, 30))
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> score = result.border_score(boundary_source="analysis")
        >>> bool(-1.0 <= score <= 1.0)
        True
        """
        from neurospatial.encoding.border import border_score as bs_func

        firing_rate = _to_numpy(self._supported_rates())
        env = self.env
        return bs_func(
            env,
            firing_rate,
            threshold=threshold,
            min_area=min_area,
            metric=metric,
            boundary_source=boundary_source,
        )

    def region_coverage(
        self,
        threshold: float = 0.3,
        regions: list[str] | None = None,
    ) -> dict[str, float]:
        """Coverage of each spatial region by the firing field.

        Computes what fraction of each region's area is covered by the
        firing field (bins where firing_rate >= threshold * peak), weighting
        each bin by its measure (``env.bin_sizes``).

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to define the field.
            Bins with firing rate >= threshold * peak are included.
        regions : list of str, optional
            Region names to analyze. If None, analyzes all regions
            defined in env.regions.

        Returns
        -------
        dict[str, float]
            Mapping from region name to coverage fraction [0, 1], NaN for
            every region when the map has no positive rate.
            Coverage = area(region in field) / area(region); on a regular
            grid, the fraction of the region's bins.

        Notes
        -----
        This method is useful for:

        - **Border cell analysis**: Determine which wall a border cell prefers
        - **Task-relevant regions**: Check if fields overlap with reward zones
        - **Multi-zone analysis**: Quantify field distribution across zones

        Delegates to ``neurospatial.encoding.border.region_coverage()``.

        See Also
        --------
        border_score : Overall border preference score
        neurospatial.encoding.border.region_coverage : Direct call

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
        >>> coverage = result.region_coverage()  # no regions defined -> {}
        >>> isinstance(coverage, dict)
        True
        """
        from neurospatial.encoding.border import region_coverage

        firing_rate = _to_numpy(self._supported_rates())

        # Threshold field at fraction of peak
        peak_rate = np.nanmax(firing_rate)
        if not peak_rate > 0:
            # Without firing there is no field, so its coverage is undefined.
            if regions is None:
                regions = list(self.env.regions.keys())
            return dict.fromkeys(regions, np.nan)

        field_mask = firing_rate >= threshold * peak_rate
        field_bins = np.where(field_mask)[0]

        env = self.env
        return region_coverage(env, field_bins, regions=regions)

    def has_place_field(
        self,
        *,
        threshold: float = 0.2,
        min_peak_rate: float = 1.0,
        min_secondary_peak_fraction: float = 0.0,
        max_mean_rate: float = np.inf,
        subfields: Literal["core", "watershed"] | None = "core",
    ) -> bool:
        """Detect whether the rate map has a place field; this is not a cell-type verdict.

        Runs :func:`detect_place_fields` on this map. Untuned Poisson units
        firing at 1 Hz had a detected field in 20 of 20 units at 1, 2, 5, 10
        and 20 minutes, and at 0.5 Hz in 20, 20, 20, 19 and 11 of 20 (30 Hz
        Ornstein-Uhlenbeck trajectory at 15 cm/s in a 100 x 100 cm arena,
        5 cm bins, diffusion KDE with a 5 cm bandwidth). Use
        ``is_place_cell(..., criterion="shuffle")``, or the labels of
        :func:`spatial_cell_type_significance` for a population, for a verdict
        that controls false positives under the circular-shift null.

        Parameters
        ----------
        threshold : float, default=0.2
            Fraction of each field's own peak rate that defines its boundary
            (0-1). A field must be larger than one noisy bin smoothed with
            this map's ``bandwidth`` (see :func:`detect_place_fields`); for
            ``method="glm"``, which has no kernel bandwidth, larger than one
            bin.
        min_peak_rate : float, default=1.0
            Smallest peak rate (Hz) that can start a field.
        min_secondary_peak_fraction : float, default=0.0
            A field after the first must peak at least this fraction of the
            map's maximum rate. The default keeps every peak above
            ``min_peak_rate``; see :func:`detect_place_fields`.
        max_mean_rate : float, default=np.inf
            Optional upper bound on the unweighted mean of non-NaN spatial-bin
            rates (Hz). The default disables exclusion. This is not the
            recording-time mean firing rate and does not identify cell type.
        subfields : {"core", "watershed"} or None, default="core"
            How to separate two fields that merge at ``threshold``:
            ``"core"`` is the FMAToolbox ``MapStats`` rule, ``"watershed"``
            divides them at the valley between them, ``None`` does not
            separate them (see :func:`detect_place_fields`).

        Returns
        -------
        bool
            True when the detector finds at least one field.

        Notes
        -----
        A detected field says the map has a region of elevated rate, not that
        the rate differs from chance. Short recordings give spiky maps, and
        smoothing turns a few spikes into a field-sized bump. For publication,
        report a circular-shift test and its assumptions instead.

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
        >>> isinstance(result.has_place_field(), bool)
        True
        """
        fields = detect_place_fields(
            self.env,
            self,
            threshold=threshold,
            min_peak_rate=min_peak_rate,
            min_secondary_peak_fraction=min_secondary_peak_fraction,
            max_mean_rate=max_mean_rate,
            subfields=subfields,
        )
        return len(fields) > 0

    def is_place_cell(self, *, min_info: float = 0.5) -> bool:
        """Screen one neuron for place tuning by bias-corrected spatial information.

        This is a fast threshold screen, not a verdict: it thresholds spatial
        information minus the information an untuned cell shows by chance
        (``spatial_information_bias``) and runs no null test. For a verdict
        that controls false positives against circularly shifted spike
        trains, call the free function :func:`is_place_cell` with the raw
        arrays (it runs the shuffle test by default); a result does not keep
        the spike times it was computed from. To detect a field, use
        ``has_place_field()``; field detection also flags most noise units.
        It tests information alone, so it can disagree with the ``is_place``
        column of :meth:`summary_table`, which also excludes known grid and
        border cells: a grid cell that passes information is True here and
        False there, and a unit with too few spikes for an information bias
        is False here and ``<NA>`` there.

        Parameters
        ----------
        min_info : float, default=0.5
            Inclusive cutoff on bias-corrected spatial information
            (bits/spike).

        Returns
        -------
        bool
            Whether bias-corrected spatial information meets the cutoff.

        Notes
        -----
        The screen compares bias-corrected information,
        ``spatial_information() - spatial_information_bias()``, with the
        cutoff: the information an untuned cell shows from Poisson noise alone
        is subtracted first (derivation in
        :meth:`SpatialRateResult.spatial_information_bias`). Units with fewer
        spikes than the map has effectively independent bins have no estimable
        information bias, so the boolean is False for them, without a warning:
        False includes "too few spikes to tell". ``label_cell_types()`` and
        ``summary_table()`` mark them "undetermined". For 40 untuned Poisson units
        per spike count in a 100 x 100 cm
        arena (5 cm bins, diffusion KDE bandwidth 5 cm), uncorrected
        information passed 0.5 bits/spike in 100%, 25-55%, 0% and 0% of units
        at about 20, 60, 120 and 600 spikes; corrected, none passed. Place
        cells with a 2 Hz peak passed at 10 minutes (about 110 spikes) and
        were refused at 2 minutes (about 18 spikes). The cutoff itself is a
        common screening convention, not derived or calibrated. For
        publication, report a circular-shift test and its assumptions
        instead of a cutoff verdict.
        """
        from neurospatial.encoding._metrics import _information_screen

        return bool(_information_screen(self, min_info)[0])


@_copy_unit_table
@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class SpatialRatesResult(PopulationRateMixin[SpatialRateResult]):
    """Result of spatial rate computation for multiple neurons.

    This class wraps spatial firing rate maps for a population of neurons
    with shared occupancy and environment. It inherits from `SpatialResultMixin`
    for common methods and provides iteration over individual neuron results.

    Parameters
    ----------
    firing_rates : ArrayLike
        Firing rate maps in Hz. Shape is (n_neurons, n_bins) where n_bins
        is the number of active bins in the environment. Each row is a
        single neuron's rate map.
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,). Shared
        across all neurons.
    env : Environment
        The immutable environment these rates were estimated on. It is their
        spatial context: bin coordinates and order, bin measures, units, frame
        and connectivity. Environments never change; ``with_metadata`` returns
        a new one, so relabelling later cannot change this result.
    method : str
        Estimator used: "diffusion_kde", "gaussian_kde", "binned", or "glm".
    bandwidth : float or None
        Smoothing bandwidth in the same units as the environment's bin_size, or
        ``None`` for ``method="glm"`` (which has no bandwidth).

    Attributes
    ----------
    firing_rates : ArrayLike
        Firing rate maps in Hz. Shape is (n_neurons, n_bins).
    occupancy : ArrayLike
        Time spent in each bin in seconds. Shape is (n_bins,).
    env : Environment
        The spatial environment.
    method : str
        Estimator used.
    bandwidth : float or None
        Smoothing bandwidth (``None`` for ``method="glm"``).
    coefficients : NDArray or None
        (``method="glm"`` only; ``None`` for ratio methods.) Fitted GAM
        coefficients ``gamma`` on the live basis, shape ``(rank, n_units)``. Like
        every GLM diagnostic here, this is the ``float64`` fit result -- the
        ``dtype`` argument governs only the ``firing_rates`` storage, not the
        diagnostics.
    penalty : float or None
        Smoothness penalty ``lambda`` actually applied (scalar for the shared-λ
        fit; ``None`` for the REML-skip and no-data cases, or for ratio methods).
    penalty_weights : NDArray or None
        Basis penalty weights ``d``, shape ``(rank,)`` (``None`` for ratio).
    rank : int or None
        Effective basis rank ``r_eff`` (``None`` for ratio methods).
    deviance : NDArray or None
        Per-unit unpenalized Poisson deviance, shape ``(n_units,)`` (``None`` for
        ratio methods).
    converged : bool or None
        Batch-level convergence flag (``None`` for ratio methods). ``False`` on
        nonconvergence -- a line-search failure, the Newton iteration cap, or
        out-of-domain data (empirical rate above ``exp(30)``).
    n_iter : int or None
        Batch-level Newton iteration count (``None`` for ratio methods).
    reml_objective : float or NDArray or None
        Minimized REML objective, or ``None`` when REML did not run (a fixed
        ``penalty`` was supplied, the basis has no penalized modes, or the data
        was all-zero) -- and ``None`` for ratio methods. Under a per-unit fit
        (``pooled=False``) it is a ``(n_units,)`` vector, ``nan`` for a zero-spike
        fallback unit.
    reml_at_boundary : bool or NDArray or None
        Whether the selected ``log(λ)`` is **near** a (log-λ) REML search bound
        (within ``5·xatol`` of it, in log-λ space): ``λ`` is weakly identified
        there (its optimum may lie at or beyond the bound). The fitted field
        remains finite, but its sensitivity to ``λ`` is not measured -- treat this
        as a flag to check that sensitivity, not a stability guarantee. Scalar for
        the shared
        (``pooled=True``) fit, a ``(n_units,)`` bool vector for the per-unit
        (``pooled=False``) fit, and ``None`` when REML did not run or for ratio
        methods.
    penalty_selected_by_reml : NDArray or None
        Per-unit provenance mask for ``pooled=False`` only: ``True`` where
        ``λ_k`` is the unit's own REML minimum, ``False`` for a zero-spike unit
        carrying the pooled-``λ`` fallback. ``None`` under ``pooled=True`` and for
        ratio methods.
    pooled : bool or None
        The smoothing-pool flag actually applied: ``True`` (shared ``λ``) /
        ``False`` (per-unit ``λ``) for glm, ``None`` for ratio methods. Persisted
        because scalar-output cases are value-identical under both settings, so it
        cannot be reconstructed from the values (the only reliable NWB source).
    unit_ids : NDArray, shape (n_units,)
        Identifier for each unit (row), e.g. from ``read_units`` or passed via
        ``unit_ids=``. Defaults to ``np.arange(n_units)``. Carried into
        indexed/iterated single-unit results and into xarray exports.
    unit_table : pandas.DataFrame or None
        Optional per-unit metadata aligned to ``unit_ids`` (e.g. region,
        quality, depth, inclusion flags), one row per unit; ``None`` when not
        provided. A unit_id column or named/non-default index aligns rows
        by identity; an unnamed default RangeIndex uses unit order.
        Mismatched or duplicate labels raise ValueError.
    _unit_ids_generated : bool, default=False
        True when the unit labels were generated (no ``unit_ids`` or
        ``unit_id`` supplied), so pairing with spike inputs stays positional.
        Set by the compute functions and by indexing; not an analysis choice.
    n_spikes_outside_intervals : int or ndarray of int64, shape (n_units,), or None
        Each unit's spikes outside the analyzed time intervals
        (``analysis_settings["observed_intervals"]``), which are in no rate:
        an int for one unit, one count per unit for a population. ``None``
        when unknown, as for a result built by hand. Also a
        ``summary_table()`` column.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to be in ``spatial_support``: the
        threshold the rates were estimated with. ``analysis_settings`` only
        records it.
    analysis_settings : dict or None
        Resolved epochs, observed intervals, gap/speed/occupancy gates and
        estimator settings. None for manually constructed results.
    spatial_support : ndarray of bool, shape (n_bins,)
        Observed bins meeting the occupancy threshold with finite rates.
    spike_window : NDArray[np.float64] or None
        Normalized spike-recording windows in seconds, copied read-only.
        Shared by the population and preserved on indexed single-unit results.
        ``None`` records assumed recording coverage.
    spike_window_assumed : bool
        Whether spike-recording coverage was assumed rather than supplied.

    Notes
    -----
    This is a frozen dataclass (immutable). All fields are set at construction
    and cannot be modified afterward.

    **Iteration Support**:

    This class supports len(), indexing, and iteration:

    - `len(result)`: Number of neurons
    - `result[i]`: Returns `SpatialRateResult` for neuron i
    - `for r in result`: Iterates over single-neuron results

    **Inherited Methods from SpatialResultMixin**:

    - `peak_location()`: Returns (n_neurons, n_dims) coordinates of peaks
    - `peak_firing_rate()`: Returns (n_neurons,) max firing rates

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rates

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 50, (500, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Compute batch result for 3 neurons (returns SpatialRatesResult)
    >>> times = np.linspace(0, 50, 500)
    >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
    >>> result = compute_spatial_rates(
    ...     env, spike_times, times, positions, bandwidth=10.0
    ... )

    >>> # Access fields
    >>> result.firing_rates.shape == (3, env.n_bins)
    True
    >>> len(result)
    3

    >>> # Index to get single-neuron result
    >>> single = result[0]
    >>> type(single).__name__
    'SpatialRateResult'

    >>> # Iterate over neurons
    >>> peaks = [round(float(r.peak_firing_rate()), 2) for r in result]
    >>> len(peaks)
    3

    >>> # Use mixin methods (batch)
    >>> peak_coords = result.peak_location()  # (n_neurons, n_dims)
    >>> max_rates = result.peak_firing_rate()  # (n_neurons,)
    >>> peak_coords.shape
    (3, 2)

    See Also
    --------
    SpatialRateResult : Single-neuron version
    compute_spatial_rates : Function to compute this result
    SpatialResultMixin : Provides peak_location() and peak_firing_rate()
    """

    firing_rates: ArrayLike
    occupancy: ArrayLike
    env: Environment
    method: str
    bandwidth: float | None
    unit_ids: NDArray[Any] | Sequence[Any] | None = field(default=None, compare=False)
    unit_table: pd.DataFrame | None = field(default=None, compare=False)
    # GAM (``method="glm"``) fields -- all ``None`` for the ratio methods,
    # populated for ``method="glm"``. Shapes (``rank == r_eff`` effective basis
    # rank, ``n_units`` neurons):
    #   coefficients      (rank, n_units)   gamma on the live basis
    #   penalty           scalar or None    lambda actually applied (None = REML skip / no data)
    #   penalty_weights   (rank,)           basis penalty weights d
    #   rank              int               effective rank r_eff
    #   deviance          (n_units,)        per-unit Poisson deviance
    #   converged         bool              batch-level convergence flag
    #   n_iter            int               batch-level Newton iterations
    #   reml_objective    scalar, (n_units,), or None   minimized REML objective
    #   reml_at_boundary  scalar bool, (n_units,) bool, or None  weak-lambda flag
    #   penalty_selected_by_reml  (n_units,) bool or None  pooled=False provenance
    #   pooled            bool or None      shared vs per-unit lambda (None ratio)
    coefficients: NDArray[np.float64] | None = None
    penalty: float | NDArray[np.float64] | None = None
    penalty_weights: NDArray[np.float64] | None = None
    rank: int | None = None
    deviance: NDArray[np.float64] | None = None
    converged: bool | None = None
    n_iter: int | None = None
    reml_objective: float | NDArray[np.float64] | None = None
    # Per-unit-lambda (``pooled=False``) diagnostics. ``penalty`` /
    # ``reml_objective`` / ``reml_at_boundary`` widen to ``(n_units,)`` vectors on
    # the per-unit automatic-REML path (informative units their ``lambda_k``,
    # zero-spike units the pooled fallback with ``reml_objective=nan``);
    # ``penalty_selected_by_reml`` is the ``(n_units,)`` bool provenance mask
    # (``True`` informative, ``False`` fallback), ``None`` otherwise.
    reml_at_boundary: bool | NDArray[np.bool_] | None = None
    penalty_selected_by_reml: NDArray[np.bool_] | None = None
    pooled: bool | None = None

    min_occupancy: float = field(default=0.0, kw_only=True)
    n_spikes_outside_intervals: NDArray[np.int64] | None = field(
        default=None, kw_only=True
    )
    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )

    _unit_ids_generated: bool = field(
        default=False, kw_only=True, repr=False, compare=False
    )

    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, compare=False, repr=False
    )

    _ESTIMATORS: ClassVar[Mapping[str, BandwidthRule]] = SPATIAL_ESTIMATORS

    def _check_family(self) -> None:
        n_units = int(np.asarray(self.firing_rates).shape[0])
        # Enforce the None-iff-glm invariant (GAM diagnostics travel together):
        # populated and correctly (rank, n_units)/(rank,)/(n_units,)-shaped for
        # method="glm" (bandwidth=None), all absent for a ratio method.
        _check_gam_result_invariant(
            "SpatialRatesResult",
            {
                "method": self.method,
                "bandwidth": self.bandwidth,
                "coefficients": self.coefficients,
                "penalty": self.penalty,
                "penalty_weights": self.penalty_weights,
                "rank": self.rank,
                "deviance": self.deviance,
                "converged": self.converged,
                "n_iter": self.n_iter,
                "reml_objective": self.reml_objective,
                "reml_at_boundary": self.reml_at_boundary,
                "penalty_selected_by_reml": self.penalty_selected_by_reml,
                "pooled": self.pooled,
            },
            n_units=n_units,
        )

    def __getitem__(self, idx: int) -> SpatialRateResult:
        """Get single-neuron result by index.

        Parameters
        ----------
        idx : int
            Unit position (0-based), the row of :attr:`firing_rates`. Select
            by label with ``sel(unit_id=...)``.

        Returns
        -------
        SpatialRateResult
            Result for the specified neuron with shared occupancy,
            environment, and smoothing parameters.

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
        >>> single.firing_rate.shape == (env.n_bins,)
        True
        """
        idx = self._unit_position(idx)
        # Slice the per-unit GAM fields; shared / batch-scalar fields carry
        # through unchanged. All ``None`` for ratio results (the ``is not None``
        # guards keep them ``None``).
        coefficients = (
            None if self.coefficients is None else np.asarray(self.coefficients)[:, idx]
        )
        deviance = None if self.deviance is None else np.asarray(self.deviance)[idx]
        # ``penalty`` / ``reml_objective`` / ``reml_at_boundary`` /
        # ``penalty_selected_by_reml`` slice per-unit when they are per-unit
        # vectors (pooled=False); a shared scalar or ``None`` carries through.
        return self._unit_result(
            SpatialRateResult,
            idx,
            env=self.env,
            method=self.method,
            bandwidth=self.bandwidth,
            coefficients=coefficients,
            penalty=_index_per_unit(self.penalty, idx),
            penalty_weights=self.penalty_weights,
            rank=self.rank,
            deviance=deviance,
            converged=self.converged,
            n_iter=self.n_iter,
            reml_objective=_index_per_unit(self.reml_objective, idx),
            reml_at_boundary=_index_per_unit(self.reml_at_boundary, idx),
            penalty_selected_by_reml=_index_per_unit(
                self.penalty_selected_by_reml, idx
            ),
            pooled=self.pooled,
        )

    def peak_locations(self) -> NDArray[np.float64]:
        """Locations of peak firing for all neurons (batch accessor).

        Plural batch counterpart to :meth:`peak_location`. Returns the
        bin-center coordinates of the maximum firing rate for each neuron in
        the population. This is the canonical plural accessor mandated by the
        v0.6 result-class contract; it returns the same array as the batch
        form of :meth:`peak_location`.

        Returns
        -------
        ndarray, shape (n_neurons, n_dims)
            Spatial coordinates of the bins with maximum firing rate for
            each neuron. Uses ``nanargmax`` to handle NaN values.

        See Also
        --------
        SpatialRateResult.peak_location : Single-neuron version.
        peak_firing_rate : Peak firing rate values.

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
        >>> result.peak_locations().shape
        (3, 2)
        """
        return self.peak_location()

    def plot(
        self, idx: int | None = None, *, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Plot the spatial rate map for a specific neuron.

        Delegates to the environment's plot_field method for consistent
        visualization across the codebase.

        Parameters
        ----------
        idx : int
            Index of the neuron to plot (0-indexed). Required: this batch result
            holds N units, so there is no single map to plot without a choice of
            unit. Passing ``None`` raises a ``ValueError`` explaining how to pick
            one.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure and axes.
        **kwargs
            Additional keyword arguments passed to env.plot_field().
            Common options include:
            - cmap : str or Colormap, default="viridis"
            - vmin, vmax : float, colorbar limits
            - colorbar : bool, default=True
            - colorbar_label : str, default="Firing rate (Hz)"

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Raises
        ------
        ValueError
            If `idx` is None. A batch result holds N units, so a unit index is
            required (e.g. ``result.plot(0)`` or iterate ``result[i].plot()``).

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
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
        >>> ax = result.plot(idx=0)
        >>> type(ax).__name__
        'Axes'
        """
        if idx is None:
            raise unit_index_required(len(self), what="rate map")
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        kwargs.setdefault("colorbar_label", "Firing rate (Hz)")
        return self.env.plot_field(_to_numpy(rates[idx]), ax=ax, **kwargs)

    def spatial_information(self) -> NDArray[np.float64] | Any:
        """Skaggs spatial information (bits per spike) for all neurons.

        Quantifies how much information each spike conveys about the
        animal's spatial location. Higher values indicate more spatially
        selective firing.

        Returns
        -------
        ndarray | jax.Array, shape (n_neurons,)
            Spatial information in bits/spike for each neuron. Always
            non-negative, and 0.0 for uniform firing. NaN for a silent neuron
            or a map with no supported bins. Bins outside the occupancy
            support are excluded even when filled.

            **Backend-aware**: Returns NumPy array for NumPy input,
            JAX array for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1993) formula:

        .. math::

            I = \\sum_i p_i \\frac{r_i}{\\bar{r}} \\log_2 \\left( \\frac{r_i}{\\bar{r}} \\right)

        **Interpretation**:

        - Place cells typically have 1-3 bits/spike
        - Higher values indicate more spatially selective firing
        - Zero means uniform firing (no spatial selectivity)

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
               An information-theoretic approach to deciphering the hippocampal code.

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
        >>> info = result.spatial_information()
        >>> info.shape
        (3,)

        See Also
        --------
        neurospatial.encoding._metrics.batch_spatial_information : Underlying computation
        """
        from neurospatial.encoding._metrics import batch_spatial_information

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return batch_spatial_information(self._supported_rates(), self.occupancy)

    def spatial_information_bias(self) -> NDArray[np.float64]:
        """Expected spatial information of an untuned cell, per unit (bits/spike).

        The information a cell firing at each unit's mean rate everywhere is
        expected to show on this occupancy with this estimator, from Poisson
        noise alone. ``spatial_information() - spatial_information_bias()``
        is what :meth:`classify` and :meth:`label_cell_types` compare with
        their cutoff. The derivation and its validation are in
        :meth:`SpatialRateResult.spatial_information_bias`.

        Returns
        -------
        ndarray, shape (n_neurons,)
            Expected untuned information in bits/spike. NaN for silent units
            and, without a warning, for units with fewer spikes than
            the maps have effectively independent bins: the bias is then
            undetermined, not zero.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 100, (36000, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
        >>> times = np.arange(36000) / 30.0
        >>> spike_times = [np.sort(rng.uniform(0, 1200, n)) for n in (600, 1200)]
        >>> result = compute_spatial_rates(env, spike_times, times, positions)
        >>> bias = result.spatial_information_bias()
        >>> bool(bias[0] > bias[1] > 0)  # twice the spikes, half the bias
        True
        """
        from neurospatial.encoding._metrics import _information_bias

        return _information_bias(self)

    def sparsity(self) -> NDArray[np.float64] | Any:
        """Sparsity of spatial firing for all neurons.

        Measures what fraction of the environment elicits significant
        firing. Lower values indicate sparser, more selective place fields.

        Returns
        -------
        ndarray | jax.Array, shape (n_neurons,)
            Sparsity values in range [0, 1] for each neuron. NaN for a silent
            neuron or a map with no supported bins. Bins outside the
            occupancy support are excluded even when filled.

            - Low (0.1-0.3): Sparse, selective place field
            - High (~1.0): Uniform firing throughout environment

            **Backend-aware**: Returns NumPy array for NumPy input,
            JAX array for JAX input.

        Notes
        -----
        Uses the Skaggs et al. (1996) formula:

        .. math::

            S = \\frac{\\left( \\sum_i p_i r_i \\right)^2}{\\sum_i p_i r_i^2}

        References
        ----------
        .. [1] Skaggs, W. E., McNaughton, B. L., et al. (1996). Theta phase
               precession in hippocampal neuronal populations.

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
        >>> spars = result.sparsity()
        >>> spars.shape
        (3,)

        See Also
        --------
        neurospatial.encoding._metrics.batch_sparsity : Underlying computation
        """
        from neurospatial.encoding._metrics import batch_sparsity

        # Pass arrays directly - _metrics.py handles JAX dispatch
        return batch_sparsity(self._supported_rates(), self.occupancy)

    def grid_scores(self) -> BatchScoresResult:
        """Grid scores (hexagonal periodicity) for all neurons.

        Quantifies the hexagonal periodicity of each neuron's firing rate map,
        which is characteristic of grid cells. Higher values indicate stronger
        hexagonal grid patterns.

        Returns
        -------
        BatchScoresResult
            Container with ``scores`` (shape ``(n_neurons,)``, range [-2, 2])
            and ``failures`` (boolean mask, ``True`` for neurons whose grid
            score computation raised an exception that was caught and
            converted to NaN). Use ``result.scores`` for the raw array if
            your downstream code expects a plain ndarray.

        Raises
        ------
        NotImplementedError
            If the environment has no regular 2D grid of rectangular cells
            (hexagonal, mesh, track or 3D layouts). :meth:`summary_table`
            records the score as unavailable instead.

        Notes
        -----
        For each neuron, computes the spatial autocorrelation and extracts
        the grid score based on rotational symmetry, as
        :meth:`SpatialRateResult.grid_score` does.

        Delegates to ``neurospatial.encoding._metrics.batch_grid_scores()``.

        References
        ----------
        .. [1] Sargolini, F., Fyhn, M., et al. (2006). Conjunctive
               representation of position, direction, and velocity in
               entorhinal cortex. Science, 312(5774), 758-762.

        See Also
        --------
        SpatialRateResult.grid_score : Single-neuron grid score
        SpatialRateResult.grid_properties : Full grid cell metrics

        Examples
        --------
        >>> result = SpatialRatesResult(...)  # doctest: +SKIP
        >>> scores = result.grid_scores()  # doctest: +SKIP
        >>> # The result wraps an ndarray; reach in via .scores for math
        >>> # operations that need a real array.
        >>> print(f"Mean grid score: {np.nanmean(scores.scores):.3f}")  # doctest: +SKIP
        >>> n_grid_cells = int(np.sum(scores.scores > 0.4))  # doctest: +SKIP
        """
        from neurospatial.encoding._metrics import batch_grid_scores

        return batch_grid_scores(self.env, _to_numpy(self._supported_rates()))

    def border_scores(
        self,
        threshold: float = 0.3,
        min_area: float | None = None,
        metric: StraightOrGeodesic = "geodesic",
        *,
        boundary_source: BoundarySource = "physical",
    ) -> BatchScoresResult:
        """Border scores of Solstad et al. (2008) for all neurons.

        Compares the largest fraction of one wall covered by one firing field
        with the firing-rate-weighted mean distance of the fields from the
        walls. Higher values indicate firing along a wall.

        Parameters
        ----------
        threshold : float, default 0.3
            Fraction of peak firing rate used to segment fields.
            Bins with firing rate >= threshold * peak are field bins.
        min_area : float, optional
            Smallest area of one field, in squared position units (for
            example cm²). Smaller fields are ignored. The default, None, is
            the 200 cm² of Solstad et al. (2008) in the environment's units,
            or at least 9 bins (the BNT/opexebo default) when the units are
            not a metric length.
        metric : {'geodesic', 'euclidean'}, default 'geodesic'
            Distance from field bins to the nearest wall.
            - 'geodesic': path through the environment (respects obstacles)
            - 'euclidean': straight-line distance

        boundary_source : {'physical', 'analysis'}, default='physical'
            The measured arena walls, or explicitly the outline of the
            retained bins. Physical mode requires measured wall geometry.

        Returns
        -------
        BatchScoresResult
            Border scores in range [-1, 1] for each neuron (``.scores``):
            near +1 for a thin field along a whole wall, -1 for firing away
            from every wall. A uniform map scores about 0.5 on a square. NaN
            where no field reaches the minimum size.

        Notes
        -----
        Uses :func:`neurospatial.encoding.border.border_score` for each
        neuron.

        Delegates to ``neurospatial.encoding._metrics.batch_border_scores()``.

        References
        ----------
        .. [1] Solstad, T., Boccara, C. N., et al. (2008). Representation of
               geometric borders in the entorhinal cortex. Science, 322(5909),
               1865-1868.

        See Also
        --------
        SpatialRateResult.border_score : Single-neuron border score

        Examples
        --------
        >>> result = SpatialRatesResult(...)  # doctest: +SKIP
        >>> scores = result.border_scores()  # doctest: +SKIP
        >>> # The result wraps an ndarray; reach in via .scores for math
        >>> # operations that need a real array.
        >>> print(
        ...     f"Mean border score: {np.nanmean(scores.scores):.3f}"
        ... )  # doctest: +SKIP
        >>> n_border_cells = int(np.sum(scores.scores > 0.5))  # doctest: +SKIP
        """
        from neurospatial.encoding._metrics import batch_border_scores

        return batch_border_scores(
            self.env,
            _to_numpy(self._supported_rates()),
            threshold=threshold,
            min_area=min_area,
            metric=metric,
            boundary_source=boundary_source,
        )

    def label_cell_types(
        self,
        *,
        min_spatial_info: float = 0.5,
        min_grid_score: float = 0.4,
        min_border_score: float = 0.5,
        grid_scores: NDArray[np.float64] | None = None,
        border_scores: NDArray[np.float64] | None = None,
        boundary_source: BoundarySource = "physical",
    ) -> NDArray[np.str_]:
        """Screen neurons for place, grid and border tuning with fixed cutoffs.

        Each label is judged by its own statistic: a grid label needs a grid
        score at the cutoff and a hexagonal first ring of autocorrelogram
        peaks (a finite :meth:`SpatialRateResult.grid_properties` ``scale``),
        and a border label only a border score at the cutoff. ``"place"`` is a cell whose bias-corrected spatial information
        (``spatial_information() - spatial_information_bias()``) reaches the
        cutoff and that is neither a grid nor a border cell. This is a fast
        screen with known false positives (see Notes); for a verdict, test
        each statistic against circularly shifted spike trains with
        :func:`spatial_cell_type_significance`.

        This is the **multi-class labeler** (returns string labels). It is
        distinct from the single-type :meth:`classify` boolean predicate.

        Parameters
        ----------
        min_spatial_info : float, default=0.5
            Inclusive cutoff on bias-corrected spatial information
            (bits/spike).
        min_grid_score : float, default=0.4
            Minimum grid score to be classified as a grid cell, a common
            screening convention. On circularly time-shifted
            spike trains of simulated cells 1-8% of shuffled maps reach 0.4,
            so it is a screen, not a verdict; use
            :func:`spatial_cell_type_significance` for a verdict.
        min_border_score : float, default=0.5
            Inclusive border-score cutoff. A uniform map scores
            about 0.5 in a convex arena but differently on other shapes (see
            ``untuned_border_score``).
        grid_scores : ndarray of shape (n_neurons,), optional
            Precomputed grid scores, one per neuron, used instead of
            recomputing :meth:`grid_scores`.
        border_scores : ndarray of shape (n_neurons,), optional
            Precomputed border scores, one per neuron, used instead of
            recomputing :meth:`border_scores`.
        boundary_source : {'physical', 'analysis'}, default='physical'
            Source used when recomputing border scores. Supply measured walls
            or explicitly request the analysis-boundary approximation.

        Returns
        -------
        ndarray, shape (n_neurons,)
            String labels for each neuron. One of:

            - ``"grid"``, ``"border"``: the grid or border score reaches its
              cutoff; a grid map also has a hexagonal first ring.
            - ``"grid+border"``: both do; the cell is reported as both.
            - ``"place"``: spatial information reaches its cutoff and neither
              the grid nor the border score does.
            - ``"unclassified"``: no statistic reaches its cutoff.
            - ``"undetermined"``: the grid or border score applies to this
              environment but cannot be computed (no regular grid of
              rectangular cells for the grid score; no measured walls, or
              walls without observed bins, for the physical border score), so
              any label is possible. ``summary_table()`` records the reason in
              ``attrs['unavailable_metrics']``; no warning is raised. Its
              ``is_place`` column is True when spatial information reaches
              its cutoff and the unit was not shown to be a grid or border
              cell. Without the border score, border cells are not
              excluded, so a True is a place-or-border candidate.

            Grid and border cells are defined on 2-D maps, so on a 1-D track
            neither applies and a tuned cell is ``"place"``. A neuron whose
            own score is NaN (a silent unit, or no field of the minimum size
            for the border score) does not pass that cutoff.

        Notes
        -----
        The cutoffs are conventions, not calibrated values. The grid cutoff
        0.4 and the border cutoff 0.5 are carried over from rectangular-box
        recordings; 0.5 bits/spike is a common information cutoff. In
        simulated sessions:

        - Spatial information: the screen subtracts the information an untuned
          cell shows from Poisson noise alone (derivation in
          :meth:`SpatialRateResult.spatial_information_bias`); units with fewer
          spikes than the map has effectively independent bins fail, with a
          warning. For 40 untuned Poisson units per spike count in a
          100 x 100 cm arena (5 cm bins, diffusion KDE bandwidth 5 cm),
          uncorrected information passed 0.5 bits/spike in 100%, 25-55%, 0%
          and 0% of units at about 20, 60, 120 and 600 spikes; corrected, none
          passed.
        - Grid score: about 10% of rate maps from circularly shifted spike
          trains reach 0.4; the 95th percentile of that null was 0.46-0.57
          across square, rectangular and circular arenas, and 11% of place
          cells reached 0.4. The score measures sixfold symmetry of the
          autocorrelogram, which a map without a lattice can have, so a grid
          label also needs the peaks nearest zero lag to be one hexagon's
          first ring (:func:`~neurospatial.encoding.grid.grid_scale`); a grid
          whose spacing exceeds about 0.6 of a square arena's side has no
          ring inside the autocorrelogram and is not labeled grid.
        - Border score: a uniform (untuned) map scores 0.50 in convex arenas
          and 0.58-0.79 on corridor mazes, so untuned cells pass the 0.5
          cutoff often; on a radial-arm maze, 6 of 12 arm place cells did.
        - Untuned cells: 40,000 Poisson cells firing 0.5-3 Hz in a 1 m
          square box and as many in a 1 m circle (400 ten-minute
          trajectories of 100 cells in each arena, default 5 cm diffusion
          kernel). None was labeled place, and 1.8% were labeled grid in
          either arena. 14.8% were labeled border in the square and 11.5%
          in the circle, but the border rate is not a per-cell chance.
          Between sessions it ranged from 0% to 63% in the square
          (standard deviation 14 percentage points; no border label in
          16% of sessions) and from 0% to 46% in the circle (8 points;
          none in 7%). Over 20 sessions it ranged from 9% to 23% in the
          square and from 6% to 16% in the circle, so 20 sessions do not
          tell the two arenas apart. An untuned cell seldom scores above
          the session's
          ``untuned_border_score(env, support=result.spatial_support)``:
          95% of cells scored below it plus 0.012, and the median cell of
          a session was within 0.015 of it in 98% of sessions. That
          baseline was 0.39-0.51 in the square and 0.45-0.50 in the
          circle, depending on which bins along the walls the animal
          visited, and it sets the session's rate: 1 of 4,500 cells was
          labeled border in the sessions where it was below 0.47, and 21%
          (square) and 14% (circle) where it was 0.49 or more. Compare
          border scores with that baseline.

        Pass precomputed scores via ``grid_scores=``/``border_scores=`` to avoid
        recomputation when you've already computed them.

        See Also
        --------
        spatial_cell_type_significance : Shuffle test of all three statistics
        spatial_information : Compute spatial information
        grid_scores : Compute grid scores
        border_scores : Compute border scores
        classify : Single-type place-cell boolean predicate

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import compute_spatial_rates
        >>> rng = np.random.default_rng(0)
        >>> positions = rng.uniform(0, 50, (500, 2))
        >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
        >>> times = np.linspace(0, 50, 500)
        >>> spike_times = [np.sort(rng.uniform(0, 50, n)) for n in (30, 40, 20)]
        >>> result = compute_spatial_rates(
        ...     env, spike_times, times, positions, bandwidth=10.0
        ... )
        >>> labels = result.label_cell_types(boundary_source="analysis")
        >>> labels.shape
        (3,)
        >>> valid = {"place", "grid", "border", "grid+border", "unclassified"}
        >>> set(labels.tolist()).issubset(valid)
        True
        """
        thresholds = {
            "min_spatial_info": min_spatial_info,
            "min_grid_score": min_grid_score,
            "min_border_score": min_border_score,
        }
        refuse_none_thresholds(
            min_spatial_info=min_spatial_info,
            min_grid_score=min_grid_score,
            min_border_score=min_border_score,
        )
        given: dict[str, NDArray[np.float64] | None] = {}
        for name, value in (
            ("grid_scores", grid_scores),
            ("border_scores", border_scores),
        ):
            given[name] = None
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.ndim != 1 or array.shape[0] != len(self):
                    msg = (
                        f"{name} must be a 1-D array of length {len(self)} (one "
                        f"per neuron), got shape {array.shape}"
                    )
                    raise ValueError(msg)
                given[name] = array
        scores, unavailable, inapplicable = _screen_scores(
            self, boundary_source=boundary_source, **given
        )
        from neurospatial.encoding._metrics import _debiased_information

        return _cell_type_labels(
            _screen_outcomes(
                self,
                scores,
                unavailable,
                inapplicable,
                information=_debiased_information(self),
                **thresholds,
            )
        )

    def classify(self, *, min_info: float = 0.5) -> NDArray[np.bool_]:
        """Screen neurons for place tuning by bias-corrected spatial information.

        A neuron passes the place-cell screen if its bias-corrected spatial
        information, ``spatial_information() - spatial_information_bias()``,
        meets the minimum threshold. This is the single-type boolean
        predicate ("is this a place cell") sibling of
        :meth:`ObjectVectorRatesResult.classify` and
        :meth:`ViewRatesResult.classify`.

        For place-field detection use has_place_field().

        For multi-class labels (``"place"``, ``"grid"``, ``"border"``,
        ``"grid+border"``, ...) use :meth:`label_cell_types` instead.

        It tests information alone, so it can disagree with the ``is_place``
        column of :meth:`summary_table`, which also excludes known grid and
        border cells: a grid cell that passes information is True here and
        False there, and a unit with too few spikes for an information bias
        is False here and ``<NA>`` there.


        Parameters
        ----------
        min_info : float, default=0.5
            Inclusive cutoff on bias-corrected Skaggs spatial information
            (bits/spike).

        Returns
        -------
        ndarray, shape (n_neurons,)
            Boolean array where True indicates the neuron is classified as a
            place cell.

        See Also
        --------
        label_cell_types : Multi-class string labeler
        is_place_cell : Free-function single-neuron place predicate

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
        >>> is_place = result.classify()
        >>> is_place.shape
        (3,)
        >>> is_place.dtype == bool
        True

        Notes
        -----
        The screen compares bias-corrected information,
        ``spatial_information() - spatial_information_bias()``, with the
        cutoff: the information an untuned cell shows from Poisson noise alone
        is subtracted first (derivation in
        :meth:`SpatialRateResult.spatial_information_bias`). Units with fewer
        spikes than the map has effectively independent bins have no estimable
        information bias, so the boolean is False for them, without a warning:
        False includes "too few spikes to tell". ``label_cell_types()`` and
        ``summary_table()`` mark them "undetermined". For 40 untuned Poisson units
        per spike count in a 100 x 100 cm
        arena (5 cm bins, diffusion KDE bandwidth 5 cm), uncorrected
        information passed 0.5 bits/spike in 100%, 25-55%, 0% and 0% of units
        at about 20, 60, 120 and 600 spikes; corrected, none passed. Place
        cells with a 2 Hz peak passed at 10 minutes (about 110 spikes) and
        were refused at 2 minutes (about 18 spikes). The cutoff itself is a
        common screening convention, not derived or calibrated. For
        publication, report a circular-shift test and its assumptions
        instead of a cutoff verdict.
        For a shuffle test, call place_cell_significance(...)
        with the raw arrays; a result does not keep the arrays it was computed from.
        """
        from neurospatial.encoding._metrics import _information_screen

        return _information_screen(self, min_info)

    def summary_table(
        self,
        unit_ids: Sequence[str | int] | None = None,
        include_classification: bool = True,
        *,
        boundary_source: BoundarySource = "physical",
    ) -> pd.DataFrame:
        """Per-unit scalar summary: one row per unit, ``unit_id``-indexed.

        Computes all spatial metrics and returns one row per unit, indexed by
        ``unit_id``, with scalar metric columns. This is the per-unit summary a
        many-neuron user wants for filtering, sorting, and population tables.
        For the dense per-bin frame (one row per ``(unit, bin)``) use
        :meth:`to_dataframe` instead.

        This is a host-only method; all metrics are computed as NumPy arrays
        (not JAX).

        Parameters
        ----------
        unit_ids : sequence of str or int, optional
            Identity labels for the index, one per unit. If ``None``, the
            result's own :attr:`unit_ids` are used. A wrong length or a
            repeated label raises ``ValueError``.
        include_classification : bool, default True
            Whether to include the fixed-threshold screen columns
            (``passes_*``, ``is_place`` and ``cell_type``, as from
            ``label_cell_types()``).

        boundary_source : {'physical', 'analysis'}, default='physical'
            Border geometry source. Unsupported physical metrics are NaN and
            explained in ``df.attrs['unavailable_metrics']``; other metrics
            remain available. The source is recorded in table attributes.

        Returns
        -------
        pd.DataFrame
            One row per unit, indexed by ``unit_id``, with columns:

            - peak_x: x-coordinate of peak firing location
            - peak_y: y-coordinate of peak firing location (NaN for 1D)
            - peak_firing_rate: maximum firing rate (Hz)
            - spatial_information: spatial information (bits/spike)
            - spatial_information_bias: the information an untuned cell
              would show (bits/spike); ``passes_spatial_information``
              compares ``spatial_information - spatial_information_bias``
              with its cutoff, and a NaN bias (too few spikes) fails it
            - sparsity: sparsity measure (0-1)
            - grid_score: grid score (hexagonal periodicity)
            - border_score: border score (boundary proximity tuning)
            - n_spikes_outside_intervals: spikes outside the analyzed time
            - passes_spatial_information, passes_grid_score,
              passes_border_score (if include_classification=True): whether
              each score reaches its screen cutoff (a grid score only with a
              hexagonal first ring); ``<NA>`` when the score applies to this
              environment but cannot be computed
            - is_place (if include_classification=True): True when the unit
              passes the spatial-information screen and was not shown to be
              a grid or border cell. A grid or border score that cannot be
              computed does not block it, so it stays defined where
              ``cell_type`` is ``"undetermined"``; but without measured
              walls border cells are not excluded, and a True is a
              place-or-border candidate. ``<NA>`` when the information test
              is unknown, except False for a known grid or border cell. It
              can disagree with :meth:`classify`, which tests information
              alone: a grid cell that passes information is True there and
              False here, and a unit with too few spikes for an information
              bias is False there and ``<NA>`` here. The shuffle verdicts
              :func:`is_place_cell` and the ``is_place_cell`` column of
              :class:`PlaceCellSignificanceResult` also test information
              alone
            - cell_type (if include_classification=True): the
              ``label_cell_types()`` label: ``"grid"``, ``"border"``,
              ``"grid+border"``, ``"place"``, ``"unclassified"`` or
              ``"undetermined"``

        Notes
        -----
        The ``passes_*``, ``is_place`` and ``cell_type`` columns are a
        fixed-threshold screen with known false positives; see :meth:`label_cell_types` for the cutoffs, their
        provenance and measured false-positive rates. The cutoffs are in
        ``df.attrs["classification_thresholds"]``. For a verdict, call
        :func:`spatial_cell_type_significance` with the raw arrays.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.
        The border score's baseline is ``df.attrs["border_score_untuned"]``,
        the score of a map that fires everywhere the animal was (0.5 in a
        convex arena with every wall visited, less where parts of the walls
        were not), and ``df.attrs["border_observed_wall_fraction"]`` is the
        fraction of the walls' length that was observed. Unobserved wall
        counts as not covered, so when that fraction is below 1 a border score
        can understate a field's coverage but never overstate it.

        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``spatial_information()``, ``grid_scores()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["cell_type"] == "place"]``, or ``df[df["is_place"]]``
          for place-or-border candidates when the border test is unavailable
          (``<NA>`` rows are dropped)
        - Sort: ``df.sort_values("spatial_information", ascending=False)``
        - Top-N: ``df.nlargest(10, "peak_firing_rate")``

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
        >>> df = result.summary_table()
        >>> "cell_type" in df.columns
        True
        >>> len(df)
        3
        >>> df.index.name
        'unit_id'

        >>> # Filter for place cells
        >>> place_cells = df[df["cell_type"] == "place"]

        >>> # This environment has no measured walls, so there is no border
        >>> # score and every label is "undetermined". is_place still screens
        >>> # information; these untuned units fail it.
        >>> df["cell_type"].unique().tolist()
        ['undetermined']
        >>> df["is_place"].tolist()
        [False, False, False]

        >>> # Sort by spatial information
        >>> top_cells = df.sort_values("spatial_information", ascending=False)

        >>> # Custom unit identifiers
        >>> df = result.summary_table(unit_ids=["unit_0", "unit_1", "unit_2"])
        >>> df.index.tolist()
        ['unit_0', 'unit_1', 'unit_2']

        See Also
        --------
        to_dataframe : Dense per-bin frame (one row per (unit, bin)).
        label_cell_types : Cell type classification
        spatial_information : Batch spatial information computation
        grid_scores : Batch grid score computation
        border_scores : Batch border score computation
        """
        n_neurons = len(self)

        if unit_ids is None:
            index_ids: list[str | int] = list(np.asarray(self.unit_ids))
        else:
            from neurospatial._results import resolve_unit_ids

            # Validate as an object array so mixed int/str labels are
            # neither coerced to strings nor merged; keep them as given.
            index_ids = list(unit_ids)
            resolve_unit_ids(
                np.asarray(index_ids, dtype=object),
                n_neurons,
                context="SpatialRatesResult.summary_table",
            )

        return _spatial_summary_frame(
            self,
            index=index_ids,
            include_classification=include_classification,
            boundary_source=boundary_source,
        )


# ==============================================================================
# Compute Functions
# ==============================================================================


def _compute_glm_spatial_rates(
    env: Environment,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    *,
    penalty: float | None,
    rank: int | None,
    pooled: bool = True,
    unit_ids: Any = None,
    resolved_backend: ResolvedBackend,
    dtype: type[np.float32] | type[np.float64],
) -> tuple[ArrayLike, Any]:
    """Fit ``method="glm"`` and assemble the full active-bin firing-rate array.

    Orchestrates the ``method="glm"`` estimator: it owns the **unit-major <->
    bin-major** boundary, the float64 core / dtype-at-the-boundary split, and the
    two-concern backend handling. It composes the reduced-rank penalty basis
    (:meth:`Environment.get_mrf_basis`) and the penalized-Poisson fit
    (:func:`neurospatial.encoding._glm.fit_mrf_gam`) without modifying either.

    The encoding side is **unit-major** ``(n_units, n_bins)``; the fit is
    **bin-major** ``(n_live_bins, n_units)``. The transpose + the restriction to
    ``basis.live_bins`` happen here (the only live-bin restriction; the fit never
    re-slices), and the bin-major ``log_rate`` is scattered back into a
    ``_RATE_FLOOR``-filled ``(n_units, n_bins)`` array whose rows are units.

    Parameters
    ----------
    env : Environment
        Environment that supplies the diffusion geometry and the reduced-rank
        MRF penalty basis.
    spike_counts : NDArray[np.float64], shape (n_units, n_bins)
        Per-unit binned spike counts in active-bin order (unit-major).
    occupancy : NDArray[np.float64], shape (n_bins,)
        Dwell time per active bin (seconds); the Poisson log-offset.
    penalty : float or None, keyword-only
        Fixed ``lambda`` (echoed on the fit) or ``None`` to select by REML.
    rank : int or None, keyword-only
        Requested basis rank cap; ``None`` resolves to the module default. The
        effective rank is reported via ``MRFFit.rank`` after clamping.
    pooled : bool, keyword-only, default True
        Forwarded to ``fit_mrf_gam``: ``True`` shares one REML ``lambda``,
        ``False`` selects an independent per-unit ``lambda`` (validated glm-only
        upstream).
    unit_ids : sequence or None, keyword-only, default None
        Per-unit identity labels aligned to the rows of ``spike_counts``, passed
        to ``fit_mrf_gam`` **only** to name the affected units in per-unit
        (``pooled=False``) convergence / boundary warnings. ``None`` names them by
        unit-axis index.
    resolved_backend : {"numpy", "jax"}, keyword-only
        The already-resolved backend (via ``get_backend_name``). Forwarded to
        ``fit_mrf_gam``, which routes the fit compute to the float32 JAX mirror
        when this is ``"jax"`` and JAX is installed (else the float64 core); either
        way ``MRFFit`` arrays come back NumPy float64. The returned rate array is
        converted to a JAX array when this is ``"jax"``, matching the ratio path's
        return contract.
    dtype : {np.float32, np.float64}, keyword-only
        Storage dtype of the returned rate array (applied at the boundary; the
        glm core stays float64).

    Returns
    -------
    firing_rates : ArrayLike, shape (n_units, n_bins)
        Assembled firing-rate map(s) (dtype-cast, JAX-converted when needed).
        ``max(exp(eta), _RATE_FLOOR)`` on live bins, ``_RATE_FLOOR`` elsewhere.
    fit : MRFFit
        The raw (always-NumPy, float64) fit result; the caller reads the GAM
        result fields off it.
    """
    from neurospatial.encoding._backend import _to_backend
    from neurospatial.encoding._glm import _RATE_FLOOR, fit_mrf_gam

    spike_counts = np.asarray(spike_counts, dtype=np.float64)  # (n_units, n_bins)
    occupancy = np.asarray(occupancy, dtype=np.float64)  # (n_bins,)
    n_units, n_bins = spike_counts.shape

    # Reduced-rank penalty basis from occupancy (live-bin order).
    basis = env.get_mrf_basis(occupancy, rank=rank)
    live_bins = np.asarray(basis.live_bins, dtype=np.intp)

    # Bins of components the animal never occupied are not in the basis; they
    # keep ``_RATE_FLOOR`` and are outside ``spatial_support`` (zero occupancy),
    # so metrics and peaks read them as unsupported, not as zero firing.
    # Boundary in: unit-major (n_units, n_bins) -> bin-major (n_live_bins,
    # n_units), restricted to live bins (the ONLY live-bin restriction).
    counts_fit = spike_counts.T[live_bins, :]  # (n_live_bins, n_units)
    occ_fit = occupancy[live_bins]  # (n_live_bins,)

    # Fit at the chosen penalty. ``pooled`` selects shared (``True``) vs per-unit
    # (``False``) lambda; it is forwarded from the public estimator, validated
    # there. ``backend`` dispatches the fit compute in ``fit_mrf_gam`` (the float32
    # JAX mirror when resolved to ``"jax"`` and available, else the float64 core);
    # MRFFit arrays come back NumPy float64 either way.
    fit = fit_mrf_gam(
        basis,
        counts_fit,
        occ_fit,
        penalty=penalty,
        pooled=pooled,
        backend=resolved_backend,
        unit_ids=unit_ids,
    )

    # Boundary out: floor-fill, then scatter max(exp(eta), _RATE_FLOOR).T into the
    # live-bin columns so rows are units.
    firing_rates = np.full((n_units, n_bins), _RATE_FLOOR, dtype=np.float64)
    if live_bins.size and n_units:
        rate_live = np.maximum(np.exp(np.asarray(fit.log_rate)), _RATE_FLOOR)
        firing_rates[:, live_bins] = rate_live.T  # (n_units, n_live_bins)

    # dtype at the result boundary (the core stayed float64), like the ratio path.
    # Return-array-type contract: a JAX array iff the resolved backend is "jax",
    # matching what the ratio path returns for backend="jax".
    if resolved_backend == "jax":
        return _to_backend(firing_rates, resolved_backend, dtype=dtype), fit
    return firing_rates.astype(dtype, copy=False), fit


def _check_spatial_env(
    env: Environment,
    time_allocation: str,
    *,
    context: str,
) -> None:
    """Check the environment and the time allocation it must support.

    Parameters
    ----------
    env : Environment
        Environment passed to the spatial entry point.
    time_allocation : str
        Requested time allocation, ``"start"`` or ``"linear"``.
    context : str
        Name of the calling entry point, used in error messages.

    Raises
    ------
    TypeError
        If ``env`` is not an environment.
    ValueError
        If ``time_allocation`` is not ``"start"`` or ``"linear"``.
    CartesianOnlyError
        If the environment is polar (``time_allocation="start"`` needs an
        allocentric environment, ``"linear"`` a Cartesian one).
    """
    from neurospatial.encoding._spatial_counts import (
        require_allocentric,
        require_linear_allocation,
    )
    from neurospatial.encoding._validation import _validate_env_argument

    _validate_env_argument(
        env, context=context, arguments="spike_times, times, positions"
    )
    if time_allocation not in ("start", "linear"):
        msg = "time_allocation must be 'start' or 'linear'."
        raise ValueError(msg)
    # Refuse polar before any default (such as the bandwidth) is resolved.
    if time_allocation == "start":
        require_allocentric(env, f"{context}()")
    else:
        require_linear_allocation(env)


def _resolve_spatial_method(
    env: Environment,
    method: str,
    *,
    context: str,
    bandwidth: float | None,
    min_occupancy: float,
    penalty: float | None,
    rank: int | None,
    pooled: bool,
) -> tuple[float, float, float | None, int | None]:
    """Validate the method parameters and resolve the ratio defaults.

    Parameters
    ----------
    env : Environment
        Environment that sets the default bandwidth.
    method : str
        Spatial rate estimator.
    context : str
        The public function, named in errors.
    bandwidth : float or None
        Ratio-method smoothing as passed by the caller.
    min_occupancy : float
        Seconds a bin must be occupied; ``None`` is refused.
    penalty : float or None
        GLM smoothing penalty as passed by the caller.
    rank : int or None
        GLM basis rank as passed by the caller.
    pooled : bool
        Whether the GLM shares one penalty across units.

    Returns
    -------
    bandwidth : float
        Resolved bandwidth in environment units (``0.0`` for ``"glm"``, and
        for ``"binned"`` without a bandwidth).
    min_occupancy : float
        Resolved minimum occupancy in seconds.
    penalty : float or None
        Validated GLM penalty.
    rank : int or None
        Validated GLM basis rank.
    """
    from neurospatial.encoding._smoothing import (
        _validate_smoothing_parameters,
        _validate_spatial_method_params,
        resolve_spatial_bandwidth,
        validate_min_occupancy,
        validate_pooled,
    )

    # Method-specific validation (mutual exclusivity + value domains), in the
    # contract order, then resolve the ratio defaults.
    penalty, rank = _validate_spatial_method_params(
        method,
        owner=context,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
    )
    # ``pooled`` (glm-only, strict bool): shared vs per-unit lambda. pooled=True
    # with a ratio method is the harmless default; pooled=False with one raises.
    validate_pooled(pooled, method)
    # Resolve the ratio defaults. glm ignores bandwidth/min_occupancy (its branch
    # returns before any smoothing and stamps bandwidth=None on the result).
    # "binned" is the plain occupancy-normalized histogram unless a bandwidth
    # is passed; only the kernel methods default to a smoothing length.
    if method == "glm" or (method == "binned" and bandwidth is None):
        resolved_bandwidth = 0.0
    else:
        resolved_bandwidth = resolve_spatial_bandwidth(env, bandwidth)
    resolved_min_occupancy = validate_min_occupancy(min_occupancy)
    if method != "glm":
        _validate_smoothing_parameters(method, resolved_bandwidth, owner=f"{context}()")
    return resolved_bandwidth, resolved_min_occupancy, penalty, rank


class _SpatialSupport(NamedTuple):
    """Speed gate, training support and provenance shared by spatial rates."""

    speed: NDArray[np.float64] | None  # (n_time,) resolved gate, or None
    linear_intervals: LinearSpatialIntervals | None  # time_allocation="linear"
    time_support: TimeSupport
    analysis_settings: dict[str, Any]
    drop_message: str | None  # excluded-interval message, warned with spike drops


def _prepare_spatial_support(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    windows: TimeWindows,
    break_before: NDArray[np.bool_] | None,
    warn_on_drop: bool,
    time_allocation: Literal["start", "linear"],
    method: str,
    bandwidth: float,
    min_occupancy: float,
    penalty: float | None,
    rank: int | None,
    pooled: bool,
    backend: str,
    raise_if_empty: bool = True,
) -> _SpatialSupport:
    """Resolve the speed gate, training support and provenance once.

    Parameters
    ----------
    env : Environment
        Environment whose active bins define spatial support.
    times : ndarray of shape (n_time,)
        Validated trajectory timestamps in seconds.
    positions : ndarray of shape (n_time, n_dims)
        Validated positions in environment units.
    speed : ndarray of shape (n_time,) or None
        Speed supplied by the caller, recorded as given in the provenance.
    min_speed, max_gap : float or None
        Speed gate and maximum timestamp gap in seconds.
    windows : TimeWindows
        The resolved ``epochs`` and ``spike_window``.
    break_before : ndarray of bool, shape (n_time,), or None
        Validated declared recording breaks, applied to the speed smoothing
        and the interval gate and recorded in the provenance.
    warn_on_drop : bool
        Whether to warn when support is dropped.
    time_allocation : {"start", "linear"}
        How each observation interval is assigned to bins.
    method : str
        Spatial rate estimator.
    bandwidth, min_occupancy : float
        Resolved ratio parameters.
    penalty : float or None
        Validated GLM penalty.
    rank : int or None
        Validated GLM basis rank.
    pooled : bool
        Whether the GLM shares one penalty across units.
    backend : str
        Resolved backend recorded in the provenance.
    raise_if_empty : bool, default=True
        Raise when no interval is usable; False warns, for a caller whose
        empty map is a valid answer.

    Returns
    -------
    _SpatialSupport
        The resolved speed gate, linear intervals (``None`` for
        ``time_allocation="start"``), time support and analysis settings.
    """
    from neurospatial._velocity import resolve_speed
    from neurospatial.encoding._provenance import rate_analysis_settings
    from neurospatial.encoding._spatial_counts import (
        _resolve_spatial_time_support,
        prepare_linear_spatial_intervals,
    )

    # Resolve the speed gate ONCE so the SAME concrete array feeds both the
    # spike numerator and the occupancy denominator (numerator/denominator
    # alignment by construction). When min_speed is None this returns None and
    # no speed gate is applied.
    resolved_speed = resolve_speed(
        times, positions, speed, min_speed, max_gap=max_gap, break_before=break_before
    )

    linear_intervals = (
        prepare_linear_spatial_intervals(
            env,
            times,
            positions,
            speed=resolved_speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=windows.epochs,
            spike_window=windows.spike_window,
            break_before=break_before,
        )
        if time_allocation == "linear"
        else None
    )

    time_support, drop_message = _resolve_spatial_time_support(
        env,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=windows.epochs,
        spike_window=windows.spike_window,
        warn_on_drop=warn_on_drop,
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
        break_before=break_before,
        raise_if_empty=raise_if_empty,
    )

    settings = rate_analysis_settings(
        time_support,
        epoch_interval_ids=windows.epoch_interval_ids,
        min_speed=min_speed,
        speed=speed,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=backend,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        time_allocation=time_allocation,
    )
    return _SpatialSupport(
        speed=resolved_speed,
        linear_intervals=linear_intervals,
        time_support=time_support,
        analysis_settings=settings,
        drop_message=drop_message,
    )


class _SpatialRateMaps(NamedTuple):
    """Population rate maps and their provenance, before wrapping as a result."""

    firing_rates: ArrayLike  # (n_units, n_bins) Hz, in the backend's array type
    occupancy: ArrayLike  # (n_bins,) seconds, float64, in the backend's array type
    spike_times: list[NDArray[np.float64]]  # (n_units,) validated trains
    unit_ids: NDArray[Any] | None  # caller-supplied labels, else None
    support: _SpatialSupport
    bandwidth: float  # resolved ratio bandwidth (0.0 for method="glm")
    min_occupancy: float  # resolved seconds threshold
    fit: Any  # MRFFit for method="glm", else None


def _spatial_rate_maps(
    env: Environment,
    spike_times: Any,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"],
    bandwidth: float | None,
    min_occupancy: float,
    penalty: float | None,
    rank: int | None,
    pooled: bool,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    time_allocation: Literal["start", "linear"],
    windows: TimeWindows,
    break_before: ArrayLike | None,
    n_jobs: int,
    backend: BackendType,
    warn_on_drop: bool,
    dtype: type[np.float32] | type[np.float64],
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    context: str,
    single_unit: bool,
    window_messages: bool = True,
) -> _SpatialRateMaps:
    """Validate the inputs and estimate every unit's spatial rate map.

    The one implementation behind :func:`compute_spatial_rate` and
    :func:`compute_spatial_rates`; the parameters are theirs, except:

    Parameters
    ----------
    windows : TimeWindows
        The resolved ``epochs`` and ``spike_window``.
    context : str
        Name of the public entry point, used in error messages.
    single_unit : bool
        True for :func:`compute_spatial_rate`: ``spike_times`` is one train,
        validated without a neuron index, drop warnings do not say "across
        all neurons", and GLM warnings name the unit by position because a
        standalone unit has no label.
    window_messages : bool, default=True
        False when ``windows`` are a piece of the recording the caller chose,
        such as one direction label's runs: a piece with no usable interval
        is an empty map, not a refusal, and the excluded-interval,
        spike-time-window and bandwidth messages are left to the caller, which
        checks the whole recording once.

    Returns
    -------
    _SpatialRateMaps
        Rates of shape ``(n_units, n_bins)``, with ``n_units == 1`` when
        ``single_unit``.
    """
    from neurospatial._spikes import as_labelled_spike_trains, as_spike_train
    from neurospatial._validation import validate_float_dtype
    from neurospatial.encoding._backend import (
        _to_backend,
        validate_and_resolve_backend,
    )
    from neurospatial.encoding._smoothing import (
        smooth_rate_maps_batch,
        warn_about_bandwidth,
    )
    from neurospatial.encoding._spatial_counts import (
        _warn_if_population_silent,
        bin_spike_trains,
        tracked_samples,
    )
    from neurospatial.encoding._validation import validate_trajectory

    _check_spatial_env(env, time_allocation, context=context)

    # Raises ValueError for unknown names and ImportError if jax is unavailable;
    # "auto" resolves to "numpy" or "jax".
    resolved_backend = validate_and_resolve_backend(backend)

    dtype = validate_float_dtype(dtype, stored="rate maps")

    resolved_bandwidth, resolved_min_occupancy, penalty, rank = _resolve_spatial_method(
        env,
        method,
        context=context,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
    )

    # Normalize spike times to canonical list-of-arrays format and resolve the
    # per-unit identity labels (defaults to arange). A labelled input keeps its
    # own labels; a differing unit_ids= raises. A standalone unit has no label.
    if single_unit:
        spike_times_list = [
            as_spike_train(spike_times, call=context, population=f"{context}s")
        ]
        result_unit_ids = None
    else:
        spike_times_list, result_unit_ids = as_labelled_spike_trains(
            spike_times, unit_ids, context=context
        )
    n_neurons = len(spike_times_list)

    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    validate_trajectory(times, positions=positions, context=context, n_dims=env.n_dims)
    breaks = validate_break_before(break_before, len(times), call=context)

    # The speed gate is resolved once for the population, so a gate that
    # excludes every interval raises once rather than per neuron.
    support = _prepare_spatial_support(
        env,
        times,
        positions,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        windows=windows,
        break_before=breaks,
        warn_on_drop=warn_on_drop and window_messages,
        time_allocation=time_allocation,
        method=method,
        bandwidth=resolved_bandwidth,
        min_occupancy=resolved_min_occupancy,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        backend=resolved_backend,
        raise_if_empty=window_messages,
    )
    # Recording coverage uses tracked runs, independently of speed or frame
    # bins. Fewer than five units never warn, so a single unit never does.
    _warn_if_population_silent(
        spike_times_list,
        times,
        max_gap=max_gap,
        epochs=support.time_support.epochs,
        spike_window=support.time_support.spike_window,
        tracked=tracked_samples(len(times), positions),
        break_before=breaks,
    )

    def maps(firing_rates: ArrayLike, occupancy: Any, fit: Any) -> _SpatialRateMaps:
        return _SpatialRateMaps(
            firing_rates=firing_rates,
            occupancy=_to_backend(occupancy, resolved_backend, dtype=np.float64),
            spike_times=spike_times_list,
            unit_ids=result_unit_ids,
            support=support,
            bandwidth=resolved_bandwidth,
            min_occupancy=resolved_min_occupancy,
            fit=fit,
        )

    # Bin spike trains and compute the shared occupancy (always NumPy -
    # CPU/joblib); an empty population still has occupancy.
    spike_counts, occupancy = bin_spike_trains(
        env,
        spike_times_list,
        times,
        positions,
        speed=support.speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=support.time_support.epochs,
        spike_window=support.time_support.spike_window,
        interval_mask=support.time_support.mask,
        n_jobs=n_jobs,
        warn_on_drop=warn_on_drop,
        time_allocation=time_allocation,
        linear_intervals=support.linear_intervals,
        drop_scope="" if single_unit else "across all neurons ",
        drop_message=support.drop_message,
        check_time_window=window_messages,
    )

    # method="glm": fit the penalized-Poisson GAM (occupancy as a log-offset).
    # It handles the no-neurons case too (fit_mrf_gam returns an (r_eff, 0) fit).
    if method == "glm":
        glm_firing_rates, fit = _compute_glm_spatial_rates(
            env,
            spike_counts,
            occupancy,
            penalty=penalty,
            rank=rank,
            pooled=pooled,
            # Population labels flow in so a per-unit convergence / boundary
            # warning names unit_ids, not bare unit-axis indices.
            unit_ids=None
            if single_unit
            else np.arange(n_neurons)
            if result_unit_ids is None
            else result_unit_ids,
            resolved_backend=resolved_backend,
            dtype=dtype,
        )
        return maps(glm_firing_rates, occupancy, fit)

    # The inputs are valid, so a bandwidth below a bin or a units-free
    # default is announced here rather than ahead of an input error, and
    # also for a population with no units: the bandwidth is wrong either way.
    if window_messages:
        warn_about_bandwidth(env, bandwidth, resolved_bandwidth, method)

    if n_neurons == 0:
        empty = _to_backend(
            np.empty((0, env.n_bins), dtype=dtype), resolved_backend, dtype=dtype
        )
        return maps(empty, occupancy, None)

    # Apply batch smoothing to compute firing rates
    # When backend="jax", uses JAX for the core rate computation
    firing_rates = smooth_rate_maps_batch(
        env,
        spike_counts,
        occupancy,
        method=method,
        bandwidth=resolved_bandwidth,
        min_occupancy=resolved_min_occupancy,
        backend=resolved_backend,
        dtype=dtype,
    )

    # Store the requested dtype on either backend.
    if isinstance(firing_rates, np.ndarray):
        firing_rates = firing_rates.astype(dtype, copy=False)
    else:
        firing_rates = _to_backend(firing_rates, resolved_backend, dtype=dtype)
    return maps(firing_rates, occupancy, None)


def _spatial_rates_result(
    env: Environment, maps: _SpatialRateMaps, *, method: str, unit_table: Any
) -> SpatialRatesResult:
    """Wrap population rate maps from :func:`_spatial_rate_maps` as a result.

    Parameters
    ----------
    env : Environment
        Environment the maps were estimated on.
    maps : _SpatialRateMaps
        Rates and provenance of every unit.
    method : str
        Estimator that produced the maps.
    unit_table : object or None
        Unit metadata carried by a labelled spike group, else None.
    """
    common: dict[str, Any] = {
        "n_spikes_outside_intervals": spikes_outside_intervals(
            maps.spike_times, maps.support.time_support
        ),
        "analysis_settings": maps.support.analysis_settings,
        "unit_table": unit_table,
        "spike_window": maps.support.time_support.spike_window,
        "firing_rates": maps.firing_rates,
        "occupancy": maps.occupancy,
        "env": env,
        "method": method,
        "unit_ids": maps.unit_ids,
    }
    fit = maps.fit
    if fit is None:
        return SpatialRatesResult(
            **common, min_occupancy=maps.min_occupancy, bandwidth=maps.bandwidth
        )
    # The per-unit lambda fields (``penalty`` / ``reml_objective`` /
    # ``reml_at_boundary`` vectors, ``penalty_selected_by_reml`` mask) carry
    # straight through from the fit; they are scalar/None under pooled=True.
    # dtype governs the (n_units, n_bins) rate-map storage only. The GLM
    # diagnostics are the float64 fit result and are kept float64 -- so
    # they do not lose precision (deviance/coefficients) and rates[i]
    # matches compute_spatial_rate, whose diagnostics are always float64.
    return SpatialRatesResult(
        **common,
        bandwidth=None,
        coefficients=fit.coefficients,
        penalty=fit.penalty,
        penalty_weights=fit.penalty_weights,
        rank=fit.rank,
        deviance=fit.deviance,
        converged=fit.converged,
        n_iter=fit.n_iter,
        reml_objective=fit.reml_objective,
        reml_at_boundary=fit.reml_at_boundary,
        penalty_selected_by_reml=fit.penalty_selected_by_reml,
        pooled=fit.pooled,
    )


@keyword_hint()
@reads_pynapple_samples
def compute_spatial_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    time_allocation: Literal["start", "linear"] = "start",
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    backend: BackendType = "numpy",
    warn_on_drop: bool = True,
) -> SpatialRateResult:
    """Compute spatial firing rate map for one neuron.

    This function computes a smoothed firing rate map from spike times
    and trajectory data. The result is a SpatialRateResult object containing
    the firing rate map, occupancy, and metadata.

    Computing rate maps for many neurons? Use
    :func:`compute_spatial_rates` (plural) instead — it shares the occupancy
    and smoothing-kernel computation across the whole population in one call
    rather than recomputing it per neuron.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure, for example
        one built with ``Environment.from_samples()``.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds. For pynapple tracking,
        pass ``tsd.t`` and ``tsd.values`` as separate arrays.
    positions : ndarray, shape (n_samples, n_dims)
        Required position coordinates at each time sample. NaN values are
        missing observations excluded from occupancy and rate computation.
    method : {"diffusion_kde", "gaussian_kde", "binned", "glm"}, \
default="diffusion_kde"
        Estimator to use:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph. Like ``gaussian_kde`` and
          ``binned``, it leaves every bin the animal never visited NaN: the
          kernel reaches them, but a rate there would be extrapolated.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers, so by construction it
          smooths through walls. In a simulated maze, 10 cm from a field
          center across a thin wall, the rate reached 21% of the field's peak
          at a 5 cm bandwidth and 60% at 10 cm, and the leak was detected as
          a second field (``diffusion_kde`` and ``binned``: 0-0.3%, no
          leaked field). Use ``diffusion_kde`` or ``binned`` in walled mazes.
        - **binned**: Occupancy-normalized histogram, spike count divided by
          occupancy in each bin, with no smoothing by default. Pass
          ``bandwidth`` to smooth that raw rate afterwards; smoothing after
          the division can introduce discretization artifacts. An unsmoothed
          map has as many independent bins as visited bins, so on a short
          recording a unit can have fewer spikes than that: its information
          bias is then unknown, and the place screen and verdict are False
          ("undetermined" in labels). Pass ``bandwidth`` or record longer.
        - **glm**: Penalized-Poisson GAM. Occupancy enters as a **log-offset**
          (never a denominator) and the smoothness penalty λ is chosen by REML,
          so the fit returns **finite rates everywhere** -- including
          low-occupancy and unvisited bins where the ratio estimators NaN. Tuned
          with ``penalty`` and ``rank`` (not ``bandwidth`` / ``min_occupancy``, which are mutually exclusive
          with ``method="glm"``).

        Note: ``diffusion_kde``, and ``binned`` with a ``bandwidth``, smooth
        matrix-free via the cached finite-volume eigenbasis (O(n_bins·rank) per neuron) — they never build a
        dense kernel and scale to large/fine grids. ``glm`` also avoids a dense
        O(n_bins²) kernel (it fits on the rank-``r`` basis), but its
        penalized-Poisson fit is heavier and **not** linear in ``rank``: each
        Newton step builds a per-unit (r, r) Hessian ``Bᵀ diag(μ) B`` (≈
        O(n_units·n_bins·rank²)) and solves it (≈ O(n_units·rank³)), and this is
        repeated across Newton iterations and REML λ candidates. Keep ``rank``
        modest for large populations. Only ``gaussian_kde`` builds a dense
        O(n_bins²) matrix; for very large grids prefer ``diffusion_kde``, or
        increase ``bin_size``.

    bandwidth : float | None, default=None
        Smoothing bandwidth in environment units; ``None`` is 0 (no
        smoothing) for ``method="binned"``, and 5 cm in declared metric units
        (0.05 m / 50 mm) or 2.5 median graph-edge lengths for unspecified or
        custom units for ``"diffusion_kde"`` and ``"gaussian_kde"``; the
        units-free default warns with its resolved value. Larger values smooth
        more; an explicit value smooths ``"binned"`` maps too.
        Mutually exclusive with ``method="glm"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included; bins below the
        threshold are set to NaN. ``0.0`` masks nothing (unvisited bins are
        NaN regardless). ``method="glm"`` fits every bin and accepts only
        ``0.0``.
    penalty : float | None, default=None
        (``method="glm"`` only.) Fixed smoothness penalty ``λ`` (≥ 0; ``0`` = no
        penalty). ``None`` (the default) selects ``λ`` by REML; on pathologically
        under-sampled data where no ``λ`` yields a converged, positive-definite
        fit, REML raises ``ValueError`` (supply a fixed ``penalty``, coarsen the
        grid, or reduce ``rank``). Mutually exclusive with the ratio methods.
    rank : int | None, default=None
        (``method="glm"`` only.) Requested rank of the reduced-rank penalty basis
        (≥ 1). ``None`` uses the module default cap. An out-of-range value is
        **clamped** (never rejected) to the effective rank
        ``max(n_live_components, min(n_live_bins, rank))``, reported via
        ``result.rank``. Mutually exclusive with the ratio methods.
    pooled : bool, default=True
        (``method="glm"`` only; strict ``bool``.) Whether REML selects **one
        shared** smoothing penalty ``λ`` for the population (``True``, the
        default) or an **independent per-unit** ``λ`` (``False``). ``pooled=False``
        runs the REML search once per unit, so the fit costs roughly one REML per
        neuron; ``result.penalty`` / ``reml_objective`` / ``reml_at_boundary``
        then hold that unit's own value (a single-unit call unwraps them to
        scalars). Zero-spike units (whose ``λ`` is unidentified) fall back to the
        pooled ``λ`` over the informative units, flagged
        ``penalty_selected_by_reml=False`` with ``reml_objective=nan``. A supplied
        fixed ``penalty`` beats ``pooled`` (REML is skipped and one scalar ``λ``
        is recorded); ``pooled`` is likewise a no-op at ``penalty_rank == 0`` (a
        shared-basis property) or when no unit spikes. A boundary warning means
        the selected ``λ`` is near a search bound -- ``λ`` itself is weakly
        identified; the fitted field is finite but its sensitivity to ``λ`` should
        be checked. Under ``pooled=False`` that warning names the affected
        ``unit_ids``; the shared ``pooled=True`` warning refers to "the pooled
        fit". Passing ``pooled=False`` with a ratio method raises ``ValueError``.
    speed : ndarray, shape (n_samples,), optional
        Precomputed instantaneous speed at each trajectory sample (physical
        units / second). Only used when ``min_speed`` is set. When
        ``min_speed`` is set and ``speed`` is ``None``, speed is auto-derived
        from the trajectory (see ``min_speed``). Pass your own ``speed`` for
        geodesic / linearized-track environments where the Euclidean
        auto-default is not appropriate. A wrong-length array raises
        ``ValueError``.
    min_speed : float, optional
        Minimum speed threshold (physical units / second). When set, low-speed
        periods are excluded from BOTH the spike numerator AND the occupancy
        denominator using ONE shared per-interval speed gate, so the firing
        rate stays correct (gating only one side would bias the rate). When
        ``None`` (the default) NO speed filtering is applied.

        **Auto-speed convention.** When ``min_speed`` is set and ``speed`` is
        ``None``, speed is :func:`neurospatial.behavior.compute_speed`: the
        magnitude of the 0.1 s Gaussian-smoothed velocity, so tracking jitter
        at rest is not counted as running. Pass ``speed`` for another
        definition, such as speed along a linearized track.

        **Choosing it.** Without a gate, time and spikes at rest (reward
        sites, track ends) enter the map. On a simulated track where the
        animal rested 27% of the time and fired 3 Hz at rest, spatial
        information was 0.85-0.94 bits/spike without a gate and 1.55-1.84
        with ``min_speed=4`` cm/s, and the rest firing made extra fields at
        the track ends. No speed suits every recording: place maps are
        usually built from running, so pick the threshold from your animal's
        speed distribution, between the resting and running modes.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    time_allocation : {"start", "linear"}, default="start"
        How trajectory intervals contribute dwell time. ``"start"`` assigns
        each eligible interval, and the spikes in it, to its starting bin and
        requires both endpoints inside the environment. ``"linear"`` follows the straight segment
        between finite endpoints and retains its exact portions inside the
        effective planar cells and arena, including entry and re-entry. Spike
        counts and occupancy use the same retained half-open time spans;
        span entries are included, exits and the final trajectory timestamp
        are excluded. Spatial boundaries retain ``env.bin_at`` ownership. Linear
        allocation supports Cartesian 2-D area geometry; other dimensions
        raise ``NotImplementedError``. Gap, speed and recording-window gates
        still apply to each complete observation interval. Linear allocation
        raises ``ValueError`` if a boundary-owned spike has zero dwell or if
        absolute timestamps cannot represent a retained span. Adjust spatial
        resolution or subtract a common time origin from all time arrays.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        An :class:`~neurospatial.Intervals`, such as the trials of one
        condition, is read through its ``windows``, and its ``interval_ids``
        are recorded in ``analysis_settings["epoch_interval_ids"]``.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``, and the speed derived for ``min_speed`` is not
        smoothed across it. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend. For
        ``method="glm"``, a resolved ``jax`` backend runs the penalized-Poisson
        fit + REML through an optional **float32** JAX mirror of the NumPy/SciPy
        core (``backend="jax"`` requires the ``jax`` extra, like the ratio
        methods; ``"auto"`` uses it when available and otherwise the NumPy core).
        The float32 mirror matches the float64 core to ~1e-6 at a fixed penalty (a
        touch looser under automatic REML, which picks a slightly different
        ``lambda``) and is markedly faster on populations. The returned
        diagnostics stay float64 either way.
    warn_on_drop : bool, default=True
        Warn when more than 5% of the spikes inside ``epochs`` and
        ``spike_window`` fall outside the tracked time (often spike times on
        another clock or in milliseconds), when more than 5% map outside the
        environment's bins, or when the interval gates discard much of the
        recording. Spikes excluded by ``min_speed`` are intentional and do
        not count. ``False`` silences these warnings; a selection that leaves
        no usable interval still raises. Spikes outside ``epochs`` and
        ``spike_window`` are not checked, so a clock offset that leaves some
        spikes inside them goes unreported: compare ``spike_times.min()`` and
        ``.max()`` with ``times`` yourself.

    Returns
    -------
    SpatialRateResult
        Result object containing:

        - ``firing_rate``: Firing rate map in Hz, shape (n_bins,); NaN where
          the rate is missing (see ``spatial_support``)
        - ``occupancy``: Time in each bin in seconds, shape (n_bins,)
        - ``env``: The environment used
        - ``method``: Estimator used
        - ``bandwidth``: Smoothing bandwidth (``None`` for ``method="glm"``)
        - ``spatial_support``: Bins with occupancy at least
          ``min_occupancy`` and a finite rate, shape (n_bins,)
        - ``analysis_settings``: Resolved epochs, observed intervals,
          gap/speed/occupancy gates and estimator settings
        - ``spike_window`` and ``spike_window_assumed``: The spike-recording
          windows applied, and whether they were assumed
        - GAM diagnostics for ``method="glm"`` (``coefficients``,
          ``penalty``, ``penalty_weights``, ``rank``, ``deviance``,
          ``converged``, ``n_iter``, ``reml_objective``,
          ``reml_at_boundary``, ``penalty_selected_by_reml``, ``pooled``);
          ``None`` for the ratio methods

    See Also
    --------
    compute_spatial_rates : Batch version for multiple neurons
    SpatialRateResult : Result class with convenience methods

    Notes
    -----
    An interval is analyzed only if it passes the gap and speed checks and
    lies inside ``epochs ∩ spike_window``. Both endpoints must be tracked.
    With ``time_allocation="start"``, both must also be inside the environment.
    Linear allocation instead retains the portions crossing effective cells.
    Counts and occupancy always share the same retained temporal support.

    **Algorithm**:

    1. Map trajectory positions to spatial bins
    2. Compute occupancy (time spent in each bin) from the retained intervals
    3. Count each spike in the bin that receives its interval's time (the
       start sample's bin under ``"start"`` allocation, the interpolated
       position's bin under ``"linear"``)
    4. Estimate the rate with ``method``: smooth counts and occupancy, then
       divide (``diffusion_kde``, ``gaussian_kde``), or divide, then smooth
       only when a ``bandwidth`` is passed (``binned``)

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rate

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory timestamps and (sorted) spike times
    >>> times = np.linspace(0, 10, 1000)
    >>> spike_times = np.array([1.0, 2.5, 4.0, 7.5, 8.2])

    >>> # Compute spatial rate
    >>> result = compute_spatial_rate(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     method="diffusion_kde",
    ...     bandwidth=10.0,
    ... )

    >>> # Access results
    >>> result.firing_rate.shape == (env.n_bins,)
    True
    >>> bool(result.peak_firing_rate() >= 0.0)
    True
    >>> result.peak_location().shape
    (2,)

    >>> # Penalized-Poisson GAM: occupancy as a log-offset, lambda by REML.
    >>> # Returns finite rates everywhere (no NaN in low-occupancy bins).
    >>> glm = compute_spatial_rate(env, spike_times, times, positions, method="glm")
    >>> bool(np.all(np.isfinite(glm.firing_rate)))
    True
    >>> glm.bandwidth is None  # ratio-only param; glm uses penalty/rank instead
    True

    >>> # Follow straight segments through cells for both spikes and dwell.
    >>> linear = compute_spatial_rate(
    ...     env, spike_times, times, positions, method="binned",
    ...     time_allocation="linear",
    ... )
    >>> linear.analysis_settings["time_allocation"]
    'linear'
    """
    maps = _spatial_rate_maps(
        env,
        spike_times,
        times,
        positions,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        time_allocation=time_allocation,
        windows=resolve_time_windows(epochs, spike_window),
        break_before=break_before,
        n_jobs=1,
        backend=backend,
        warn_on_drop=warn_on_drop,
        dtype=np.float64,
        unit_ids=None,
        context="compute_spatial_rate",
        single_unit=True,
    )
    common: dict[str, Any] = {
        "n_spikes_outside_intervals": int(
            spikes_outside_intervals(maps.spike_times, maps.support.time_support)[0]
        ),
        "analysis_settings": maps.support.analysis_settings,
        "spike_window": maps.support.time_support.spike_window,
        "firing_rate": maps.firing_rates[0],  # type: ignore[index]
        "occupancy": maps.occupancy,
        "env": env,
        "method": method,
    }
    fit = maps.fit
    if fit is None:
        return SpatialRateResult(
            **common, min_occupancy=maps.min_occupancy, bandwidth=maps.bandwidth
        )
    # Singular result: UNWRAP the one-element per-unit vectors (pooled=False)
    # to scalars so ``compute_spatial_rate(pooled=False)`` equals
    # ``compute_spatial_rates([spikes], pooled=False)[0]`` field-for-field.
    # ``pooled=True`` fields are already scalars and pass through untouched.
    return SpatialRateResult(
        **common,
        bandwidth=None,
        coefficients=np.asarray(fit.coefficients)[:, 0],
        penalty=_index_per_unit(fit.penalty, 0),
        penalty_weights=fit.penalty_weights,
        rank=fit.rank,
        deviance=fit.deviance[0],
        converged=fit.converged,
        n_iter=fit.n_iter,
        reml_objective=_index_per_unit(fit.reml_objective, 0),
        reml_at_boundary=_index_per_unit(fit.reml_at_boundary, 0),
        penalty_selected_by_reml=_index_per_unit(fit.penalty_selected_by_reml, 0),
        pooled=fit.pooled,
    )


@keyword_hint()
@reads_pynapple_samples
def compute_spatial_rates(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    time_allocation: Literal["start", "linear"] = "start",
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    n_jobs: int = 1,
    backend: BackendType = "numpy",
    warn_on_drop: bool = True,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
) -> SpatialRatesResult:
    """Compute spatial firing rate maps for multiple neurons.

    This is the batch version of ``compute_spatial_rate()`` that efficiently
    processes multiple neurons with shared trajectory data. It precomputes
    shared quantities (occupancy, position bins, diffusion kernel) once and
    optionally parallelizes spike counting with joblib.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure, for example
        one built with ``Environment.from_samples()``.
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        A ``unit_ids`` passed with a labelled group must equal the group's
        index.
    times : ndarray, shape (n_samples,)
        Timestamps of trajectory samples in seconds. For pynapple tracking,
        pass ``tsd.t`` and ``tsd.values`` as separate arrays.
    positions : ndarray, shape (n_samples, n_dims)
        Required position coordinates at each time sample. NaN values are
        missing observations excluded from occupancy and rate computation.
    method : {"diffusion_kde", "gaussian_kde", "binned", "glm"}, \
default="diffusion_kde"
        Estimator to use. See ``compute_spatial_rate()`` for details, including
        why unsmoothed ``"binned"`` maps of short recordings can leave a unit's
        place verdict undetermined. In addition
        to the three ratio methods, ``method="glm"`` fits a penalized-Poisson GAM
        (occupancy as a log-offset, ``λ`` by REML) and returns finite rates
        everywhere; it is tuned with ``penalty`` / ``rank`` (mutually exclusive
        with ``bandwidth`` / ``min_occupancy``). This function is
        the batched entry point the decoder consumes; ``method="glm"`` flows
        through both the decoder (``decode_session`` / ``BayesianDecoder``) and
        NWB persistence (``write_spatial_rates`` round-trips the GAM diagnostics).
        ``diffusion_kde`` (and ``binned``, unsmoothed unless given a
        ``bandwidth``) are matrix-free (O(n_bins·rank) per neuron); ``glm`` also avoids a dense
        O(n_bins²) kernel but its penalized-Poisson fit is **not** linear in
        ``rank`` — each Newton step builds and solves a per-unit (r, r) Hessian
        (≈ O(n_units·n_bins·rank²) + O(n_units·rank³)), repeated across Newton
        iterations and REML λ candidates, so keep ``rank`` modest for large
        populations (see ``compute_spatial_rate`` for details). Only
        ``gaussian_kde`` builds a dense O(n_bins²) kernel.
    bandwidth : float | None, default=None
        Smoothing bandwidth in environment units; ``None`` is 0 (no
        smoothing) for ``method="binned"``, and 5 cm in declared metric units
        (0.05 m / 50 mm) or 2.5 median graph-edge lengths for unspecified or
        custom units for ``"diffusion_kde"`` and ``"gaussian_kde"``; the
        units-free default warns with its resolved value. Larger values smooth
        more; an explicit value smooths ``"binned"`` maps too.
        Mutually exclusive with ``method="glm"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included; ``0.0`` masks
        nothing (unvisited bins are NaN regardless). ``method="glm"`` fits
        every bin and accepts only ``0.0``.
    penalty : float | None, default=None
        (``method="glm"`` only.) Fixed smoothness penalty ``λ`` (≥ 0). ``None``
        selects ``λ`` by REML, which raises ``ValueError`` on pathologically
        under-sampled data where no ``λ`` yields a converged fit (see
        ``compute_spatial_rate``). Mutually exclusive with the ratio methods.
    rank : int | None, default=None
        (``method="glm"`` only.) Requested rank of the reduced-rank penalty basis
        (≥ 1). ``None`` uses the module default cap; an out-of-range value is
        clamped (never rejected) to the effective rank reported via
        ``result.rank``. Mutually exclusive with the ratio methods.
    pooled : bool, default=True
        (``method="glm"`` only; strict ``bool``.) One **shared** smoothing penalty
        ``λ`` for the whole population (``True``, the default) or an **independent
        per-unit** ``λ`` (``False``). Under ``pooled=False`` the REML search runs
        once per informative unit (cost ~ one REML per neuron), so
        ``result.penalty`` / ``reml_objective`` / ``reml_at_boundary`` become
        ``(n_units,)`` vectors and ``penalty_selected_by_reml`` a per-unit mask;
        zero-spike units fall back to the pooled ``λ`` over the informative units
        (``penalty_selected_by_reml=False``, ``reml_objective=nan``). A fixed
        ``penalty`` beats ``pooled`` (scalar ``λ``); ``pooled`` is a no-op at
        ``penalty_rank == 0`` or when no unit spikes. ``pooled=False`` with a ratio
        method raises ``ValueError``.
    speed : ndarray, shape (n_samples,), optional
        Precomputed instantaneous speed at each trajectory sample (physical
        units / second). Only used when ``min_speed`` is set; auto-derived from
        the trajectory when ``None``. See ``min_speed``. A wrong-length array
        raises ``ValueError``.
    min_speed : float, optional
        Minimum speed threshold (physical units / second). When set, low-speed
        periods are excluded from BOTH the (shared) occupancy denominator AND
        every per-neuron spike numerator using ONE shared per-interval speed
        gate, so firing rates stay correct. When ``None`` (default) NO speed
        filtering is applied. Auto-speed is :func:`neurospatial.behavior.compute_speed`, the
        magnitude of the 0.1 s Gaussian-smoothed velocity; pass an explicit
        ``speed`` for geodesic / linearized-track environments.

        **Choosing it.** Without a gate, time and spikes at rest (reward
        sites, track ends) enter the map. On a simulated track where the
        animal rested 27% of the time and fired 3 Hz at rest, spatial
        information was 0.85-0.94 bits/spike without a gate and 1.55-1.84
        with ``min_speed=4`` cm/s, and the rest firing made extra fields at
        the track ends. No speed suits every recording: place maps are
        usually built from running, so pick the threshold from your animal's
        speed distribution, between the resting and running modes.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    time_allocation : {"start", "linear"}, default="start"
        How trajectory intervals contribute dwell time. ``"start"`` assigns
        each eligible interval, and the spikes in it, to its starting bin and
        requires both endpoints inside the environment. ``"linear"`` follows the straight segment
        between finite endpoints and retains its exact portions inside the
        effective planar cells and arena, including entry and re-entry. All
        units and the shared occupancy use the same retained half-open spans;
        span entries are included, exits and the final trajectory timestamp
        are excluded. Spatial boundaries retain ``env.bin_at`` ownership.
        Linear allocation supports Cartesian 2-D area geometry; other dimensions raise
        ``NotImplementedError``. Gap, speed and recording-window gates still
        apply to each complete observation interval. Linear allocation raises
        ``ValueError`` if a boundary-owned spike has zero dwell or if absolute
        timestamps cannot represent a retained span. Adjust spatial resolution
        or subtract a common time origin from all time arrays.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        An :class:`~neurospatial.Intervals`, such as the trials of one
        condition, is read through its ``windows``, and its ``interval_ids``
        are recorded in ``analysis_settings["epoch_interval_ids"]``.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``, and the speed derived for ``min_speed`` is not
        smoothed across it. ``analysis_settings["break_times"]`` records
        ``times[k]`` of each break (``None`` when not given).
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    n_jobs : int, default=1
        Number of parallel jobs for spike counting. Use -1 for all CPUs.
        1 means sequential processing (no parallelization overhead).
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend. For
        ``method="glm"``, a resolved ``jax`` backend runs the penalized-Poisson
        fit + REML through an optional **float32** JAX mirror of the NumPy/SciPy
        core (``backend="jax"`` requires the ``jax`` extra, like the ratio
        methods; ``"auto"`` uses it when available and otherwise the NumPy core).
        The float32 mirror matches the float64 core to ~1e-6 at a fixed penalty (a
        touch looser under automatic REML, which picks a slightly different
        ``lambda``) and is markedly faster on populations. The returned
        diagnostics stay float64 either way.
    warn_on_drop : bool, default=True
        If ``True`` (the default), emit a single ``UserWarning`` (per drop
        cause) when a large fraction of spikes are silently dropped across
        all neurons.  The warning is computed in the main process from
        aggregate statistics, so it fires exactly once even when
        ``n_jobs != 1`` (joblib worker warnings are commonly swallowed).
        Set to ``False`` to suppress all drop-related warnings. Spikes
        outside ``epochs`` and ``spike_window`` are not checked, so a clock
        offset that leaves some spikes inside them goes unreported: compare
        ``spike_times.min()`` and ``.max()`` with ``times`` yourself.
    dtype : {np.float32, np.float64}, default=np.float64
        Storage dtype of the returned ``(n_units, n_bins)`` rate-map array.
        ``np.float32`` halves the stored rate-map array. The rate computation
        is still performed in float64 and only the final result is cast, so
        float32 values match float64 within float32 tolerance. To halve the
        decoding working set as well, pass ``dtype=np.float32`` to
        ``decode_session`` / ``decode_session_summary``. Any other dtype
        raises ``ValueError``.
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.

    Returns
    -------
    SpatialRatesResult
        Result object containing:

        - ``firing_rates``: Firing rate maps, shape ``(n_neurons, n_bins)``
        - ``occupancy``: Time in each bin in seconds, shape ``(n_bins,)``
        - ``env``: The environment used
        - ``method``: Method used for smoothing
        - ``bandwidth``: Bandwidth used for smoothing

        The result supports iteration: ``for single in result: ...``
        and indexing: ``single = result[0]``.

    Warns
    -----
    UserWarning
        When at least five units are all silent for at least 60 seconds of
        continuously tracked time and ``spike_window`` was not supplied.
        This is a heuristic for possible recording outages: it cannot detect
        an outage for a single unit or one shorter than 60 seconds, and its
        absence is not proof that recording coverage is correct.

    See Also
    --------
    compute_spatial_rate : Single-neuron version
    SpatialRatesResult : Result class with batch methods

    Notes
    -----
    An interval is analyzed only if it passes the gap and speed checks and
    lies inside ``epochs ∩ spike_window``. Both endpoints must be tracked.
    With ``time_allocation="start"``, both must also be inside the environment.
    Linear allocation instead retains the portions crossing effective cells.
    Counts and occupancy always share the same retained temporal support.

    **Efficiency advantages over calling ``compute_spatial_rate()`` in a loop**:

    1. Occupancy is computed once and shared across all neurons
    2. Diffusion kernel (for ``diffusion_kde`` method) is computed once
    3. Position-to-bin mapping is done once
    4. Spike binning can be parallelized with joblib

    **When to use batch vs single**:

    - **Batch** (this function): Processing 3+ neurons, or any case where
      efficiency matters. The overhead of precomputing shared quantities
      is amortized over multiple neurons.
    - **Single** (``compute_spatial_rate``): Processing 1-2 neurons, or when
      you need fine-grained control over individual neurons.

    **Memory (``dtype``).** Passing ``dtype=np.float32`` halves the stored
    ``(n_units, n_bins)`` rate-map array. The rate computation (GEMM /
    division) is still done in float64 and only the final result is cast to
    ``dtype``, so values match the float64 default within float32 tolerance.
    ``decode_session`` / ``decode_session_summary`` now accept their own
    ``dtype`` parameter (default float64) that honors float32 end-to-end --
    the encoding-model working set AND the posterior -- so
    ``decode_session(dtype=np.float32)`` halves the decode working set on the
    golden path.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_spatial_rates

    >>> # Create environment from a seeded trajectory
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")

    >>> # Create trajectory timestamps
    >>> times = np.linspace(0, 10, 1000)

    >>> # Spike times for 3 neurons (each sorted ascending); enough spikes
    >>> # per neuron for the summary table's information screen
    >>> spike_times = [np.sort(rng.uniform(0, 10, n)) for n in (400, 500, 600)]

    >>> # Compute spatial rates for all neurons
    >>> result = compute_spatial_rates(
    ...     env,
    ...     spike_times,
    ...     times,
    ...     positions,
    ...     method="diffusion_kde",
    ...     bandwidth=10.0,
    ... )

    >>> # Access results
    >>> len(result)
    3
    >>> result.firing_rates.shape == (3, env.n_bins)
    True

    >>> # Iterate over neurons
    >>> peaks = [round(float(single.peak_firing_rate()), 2) for single in result]
    >>> len(peaks)
    3

    >>> # Per-unit scalar summary (one row per unit)
    >>> summary = result.summary_table()
    >>> len(summary)
    3
    >>> # Dense per-bin frame (one row per (unit, bin))
    >>> df = result.to_dataframe()
    >>> len(df) == 3 * env.n_bins
    True

    >>> # Use 2D array with NaN padding
    >>> spike_times_2d = np.array(
    ...     [
    ...         [0.1, 0.5, 1.0, np.nan],
    ...         [0.2, 0.3, 0.8, 1.2],
    ...     ]
    ... )
    >>> result2 = compute_spatial_rates(env, spike_times_2d, times, positions)
    >>> len(result2)
    2

    >>> # A decoder can consume rates fitted on retained linear support.
    >>> linear = compute_spatial_rates(
    ...     env, spike_times, times, positions, method="binned",
    ...     time_allocation="linear",
    ... )
    >>> linear.analysis_settings["time_allocation"]
    'linear'
    """
    maps = _spatial_rate_maps(
        env,
        spike_times,
        times,
        positions,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        time_allocation=time_allocation,
        windows=resolve_time_windows(epochs, spike_window),
        break_before=break_before,
        n_jobs=n_jobs,
        backend=backend,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
        unit_ids=unit_ids,
        context="compute_spatial_rates",
        single_unit=False,
    )
    return _spatial_rates_result(
        env, maps, method=method, unit_table=spike_unit_table(spike_times)
    )


# ==============================================================================
# Directional Place Fields
# ==============================================================================


@dataclass(frozen=True, repr=False, eq=False)
class DirectionalPlaceFields(ReadOnlyArraysMixin, ResultMixin):
    """Container for direction-conditioned place field results.

    Stores firing rate maps computed separately for different movement
    directions or trial types. This enables analysis of directional
    tuning in place cells.

    Attributes
    ----------
    firing_rates : Mapping[Hashable, NDArray[np.float64]]
        Mapping from direction label (e.g. ``"A→B"``, ``"forward"``, ``1``)
        to that direction's firing rate map, shape (n_bins,), in Hz. Keys are
        the label values themselves.
    occupancy : Mapping[Hashable, NDArray[np.float64]]
        Per-direction occupancy (time spent in each bin) in seconds, shape
        (n_bins,). Same keys as ``firing_rates``.
    env : Environment
        Spatial environment used to compute the per-direction fields.
        Shared across all labels (the per-direction split is over time,
        not over space).
    labels : tuple of hashable
        The keys of ``firing_rates`` in iteration order.
    spike_window : NDArray[np.float64] or None, shape (n_windows, 2)
        The spike-recording windows applied; None when spikes were assumed
        recorded wherever position was.
    missing_label : hashable or None
        The label value treated as missing, besides ``None`` and ``NaN``.
    unit_id : int or str or None, default=None
        The unit the maps belong to, written to the ``unit_id`` column of
        :meth:`to_dataframe` (``<NA>`` when None).

    Raises
    ------
    ValueError
        If the two mappings have different labels, or an array is not
        shaped (env.n_bins,).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
    >>> env = Environment.from_samples(np.linspace(0, 10, 11)[:, None], bin_size=1.0)
    >>> outbound = np.linspace(1.0, 5.0, env.n_bins)
    >>> result = DirectionalPlaceFields(
    ...     firing_rates={"home→goal": outbound, "goal→home": outbound[::-1]},
    ...     occupancy={
    ...         "home→goal": np.ones(env.n_bins),
    ...         "goal→home": np.ones(env.n_bins),
    ...     },
    ...     env=env,
    ... )
    >>> result.labels
    ('home→goal', 'goal→home')

    See Also
    --------
    compute_directional_place_fields : Compute directional place fields from spike data.
    """

    firing_rates: Mapping[Hashable, NDArray[np.float64]]
    occupancy: Mapping[Hashable, NDArray[np.float64]]
    env: Environment

    spike_window: NDArray[np.float64] | None = field(
        default=None, kw_only=True, compare=False
    )
    missing_label: Hashable | None = field(default=None, kw_only=True, compare=False)
    unit_id: int | str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate, own each direction's arrays and prevent replacing entries."""
        rate_labels = list(self.firing_rates)
        if set(rate_labels) != set(self.occupancy) or len(rate_labels) != len(
            self.occupancy
        ):
            raise ValueError(
                _format_error(
                    f"firing_rates has labels {tuple(rate_labels)} but occupancy has "
                    f"{tuple(self.occupancy)}.",
                    why="Why: each direction's rate map needs its own occupancy",
                    fix="pass both mappings with the same direction labels",
                )
            )
        n_bins = self.env.n_bins
        for name in ("firing_rates", "occupancy"):
            arrays = getattr(self, name)
            owned = {}
            for label in rate_labels:
                array = read_only_copy(arrays[label])
                if array.shape != (n_bins,):
                    raise ValueError(
                        _format_error(
                            f"{name}[{label!r}] has shape {array.shape}, not "
                            f"({n_bins},) for the env's {n_bins} bins.",
                            why="Why: each direction's map is one value per bin of env",
                            fix="compute the maps with compute_directional_place_fields on this env",
                        )
                    )
                owned[label] = array
            object.__setattr__(self, name, MappingProxyType(owned))
        self._own_arrays()

    @property
    def labels(self) -> tuple[Hashable, ...]:
        """Direction labels in iteration order."""
        return tuple(self.firing_rates)

    @property
    def n_directions(self) -> int:
        """Number of direction labels."""
        return len(self.firing_rates)

    @property
    def n_bins(self) -> int:
        """Number of spatial bins of each direction's map (``env.n_bins``)."""
        return int(self.env.n_bins)

    @property
    def peak_firing_rates(self) -> Mapping[Hashable, float]:
        """Each direction's peak firing rate in Hz, keyed by label.

        The largest rate over the bins with a rate (not NaN); NaN for a
        direction with none.
        """
        peaks: dict[Hashable, float] = {}
        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label])
            supported = rate[~np.isnan(rate)]
            peaks[label] = float(supported.max()) if supported.size else np.nan
        return MappingProxyType(peaks)

    def __getstate__(self) -> dict[str, Any]:
        """Serialize read-only mappings as ordinary dictionaries."""
        state = vars(self).copy()
        for name in ("firing_rates", "occupancy"):
            state[name] = dict(state[name])
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore owned arrays and read-only mappings after copying."""
        super().__setstate__(state)
        self.__post_init__()

    @property
    def spike_window_assumed(self) -> bool:
        """True when spikes were assumed recorded wherever position was.

        No ``spike_window`` was passed. The population-silence warning catches
        one common violation of this assumption; it cannot establish recording
        coverage.
        """
        return self.spike_window is None

    def correlation(self, label_a: Hashable, label_b: Hashable) -> float:
        """Pearson correlation between two directions' rate maps.

        Quantifies how similar the place-field map is between two movement
        directions (or trial types). A correlation near ``1.0`` means the cell
        fires in the same locations regardless of direction; values near
        ``0`` (or negative) indicate direction-specific tuning.

        Bins where either map is NaN are excluded pairwise before the
        correlation is computed. This is ``field_stability`` with
        ``method="pearson"``, which shares its rule for an undefined
        correlation with ``compare_spatial_rates``.

        Parameters
        ----------
        label_a, label_b : hashable
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Pearson correlation coefficient in ``[-1, 1]``. Returns ``nan``
            if fewer than three bins are finite in both maps (two points
            always give exactly +-1) or if either map is flat over those
            bins: its highest and lowest rates differ by at most ``1e-12``
            times its largest absolute rate, the rounding a smoothed flat map
            carries.

        Raises
        ------
        KeyError
            If ``label_a`` or ``label_b`` is not a known direction label.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> rate = np.linspace(1.0, 5.0, n)
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={"fwd": rate, "rev": rate.copy()},
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ... )
        >>> bool(np.isclose(result.correlation("fwd", "rev"), 1.0))
        True
        """
        for label in (label_a, label_b):
            if label not in self.firing_rates:
                msg = (
                    f"Unknown direction label {label!r}. Known labels: {tuple(self.firing_rates)}."
                    f" Fix: pass one of {tuple(self.firing_rates)}."
                )
                raise KeyError(msg)
        return field_stability(
            _to_numpy(self.firing_rates[label_a]).ravel(),
            _to_numpy(self.firing_rates[label_b]).ravel(),
            method="pearson",
        )

    def directionality_index(self, label_a: Hashable, label_b: Hashable) -> float:
        """Rate-weighted directionality index between two directions.

        The difference in total rate between the two directions' maps over
        their total rate, summed over the bins observed in both:

        .. math::

            \\mathrm{DI} = \\frac{\\sum_i (r^a_i - r^b_i)}{\\sum_i (r^a_i + r^b_i)}

        This is the index of Navratilova et al. [1]_, the difference in the
        spikes fired in the two running directions over the spikes fired in
        both, with each direction's rate map in place of its spike count, so
        a direction traversed more often does not count more. ``0`` means the
        cell fires as much in both directions, ``+1`` that it fires only in
        ``label_a`` and ``-1`` only in ``label_b``. Bins with low rate in both
        directions add little to either sum, so out-of-field noise does not
        dominate. The index compares how much a cell fires, not where: a
        field that moves between directions at the same rate has DI near 0
        and is detected by :meth:`correlation` instead.

        Parameters
        ----------
        label_a, label_b : hashable
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Directionality index in ``[-1, 1]``, positive when the cell fires
            more in ``label_a``. ``nan`` if no bin is finite in both maps or
            the cell is silent in both there.

        Raises
        ------
        KeyError
            If ``label_a`` or ``label_b`` is not a known direction label.

        References
        ----------
        .. [1] Navratilova, Z., Hoang, L. T., Schwindel, C. D., Tatsuno, M., &
               McNaughton, B. L. (2012). Experience-dependent firing rate
               remapping generates directional selectivity in hippocampal place
               cells. Frontiers in Neural Circuits, 6, 6.
               https://doi.org/10.3389/fncir.2012.00006

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> rate = np.linspace(1.0, 5.0, n)
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={"fwd": rate, "rev": rate.copy()},
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ... )
        >>> bool(np.isclose(result.directionality_index("fwd", "rev"), 0.0))
        True
        """
        for label in (label_a, label_b):
            if label not in self.firing_rates:
                msg = (
                    f"Unknown direction label {label!r}. Known labels: {tuple(self.firing_rates)}."
                    f" Fix: pass one of {tuple(self.firing_rates)}."
                )
                raise KeyError(msg)
        a = _to_numpy(self.firing_rates[label_a]).ravel()
        b = _to_numpy(self.firing_rates[label_b]).ravel()
        observed = np.isfinite(a) & np.isfinite(b)
        rate_a, rate_b = float(a[observed].sum()), float(b[observed].sum())
        if rate_a + rate_b <= 0.0:
            return float("nan")
        return (rate_a - rate_b) / (rate_a + rate_b)

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics across directions.

        Returns
        -------
        dict
            Mapping with ``n_directions`` (int), ``n_bins`` (int), the
            analysis settings ``spike_window``, ``spike_window_assumed`` and
            ``missing_label``, and one ``peak_firing_rates[<label>]`` entry
            per direction giving that direction's peak firing rate (Hz), as
            read from :attr:`peak_firing_rates`. Compare two directions'
            maps with :meth:`correlation` and :meth:`directionality_index`.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ... )
        >>> result.summary()["n_directions"]
        2
        >>> result.summary()["missing_label"] is None
        True
        """
        out: dict[str, Any] = {
            "n_directions": self.n_directions,
            "spike_window_assumed": self.spike_window_assumed,
            "spike_window": None
            if self.spike_window is None
            else self.spike_window.tolist(),
            "missing_label": self.missing_label,
            "n_bins": self.n_bins,
        }
        for label, peak in self.peak_firing_rates.items():
            out[f"peak_firing_rates[{label}]"] = peak
        return out

    def _repr_unit(self, key: str) -> str:
        """``"Hz"`` for each direction's peak firing rate."""
        if key.startswith("peak_firing_rates["):
            return "Hz"
        return super()._repr_unit(key)

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy/long-form table: one row per (unit, direction, bin).

        Stacks the per-direction rate maps into long form with a ``direction``
        identifier column, so directional results ``pandas.concat`` cleanly
        with other tidy result tables.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``unit_id`` (``<NA>`` without one),
            ``direction`` (the label), ``bin`` (int), the bin-center columns
            (``bin_center_x``, ``bin_center_y``, ...), ``firing_rate``
            (float, Hz) and ``occupancy`` (float, seconds).
            ``df.attrs["units"]`` gives each column's unit: the environment's
            ``units`` for the bin centers (``""`` when unset), ``"Hz"`` and
            ``"s"``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ...     unit_id=7,
        ... )
        >>> df = result.to_dataframe()
        >>> list(df.columns)
        ['unit_id', 'direction', 'bin', 'bin_center_x', 'firing_rate', 'occupancy']
        >>> sorted(df["direction"].unique())
        ['fwd', 'rev']
        >>> len(df) == 2 * n
        True
        """
        import pandas as pd

        from neurospatial._results import _bin_center_coords

        n_bins = self.env.n_bins
        centers = {
            name: np.asarray(values)
            for name, (_dim, values) in _bin_center_coords(self.env, n_bins).items()
        }
        unit_id = pd.NA if self.unit_id is None else self.unit_id
        n_rows = n_bins * len(self.labels)
        direction = np.empty(n_rows, dtype=object)
        direction[:] = [label for label in self.labels for _ in range(n_bins)]

        def stacked(maps: Mapping[Hashable, NDArray[np.float64]]) -> NDArray[Any]:
            if not self.labels:
                return np.array([], dtype=np.float64)
            return np.concatenate(
                [_to_numpy(maps[label]).ravel() for label in self.labels]
            )

        df = pd.DataFrame(
            {
                "unit_id": pd.Series([unit_id] * n_rows, dtype=object),
                "direction": direction,
                "bin": np.tile(np.arange(n_bins, dtype=np.int64), len(self.labels)),
                **{
                    name: np.tile(values.astype(np.float64), len(self.labels))
                    for name, values in centers.items()
                },
                "firing_rate": stacked(self.firing_rates),
                "occupancy": stacked(self.occupancy),
            }
        )
        df.attrs["units"] = {
            "unit_id": "",
            "direction": "",
            "bin": "",
            **{
                name: "rad" if name == "bin_center_angle" else self.env.units or ""
                for name in centers
            },
            "firing_rate": "Hz",
            "occupancy": "s",
        }
        return df

    def plot(
        self, *, ax: Axes | Sequence[Axes] | None = None, **kwargs: Any
    ) -> Axes | tuple[Axes, ...]:
        """Plot each direction's firing rate map.

        On a track or 1-D environment, one line per direction is drawn
        against position in the environment's units, so the directions
        overlay on one axes. In a 2-D environment each direction gets its own
        panel, drawn with :meth:`Environment.plot_field` on shared color
        limits, so the maps can be compared.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or sequence of Axes, optional
            Axes to draw on. A 2-D environment needs one axes per direction.
            If None, a new figure is created.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot`` for each
            direction's line in 1-D, or to ``env.plot_field`` for each
            panel in 2-D.

        Returns
        -------
        matplotlib.axes.Axes or tuple of Axes
            The axes with one line per direction on a track; one axes per
            direction in 2-D.

        Raises
        ------
        ValueError
            If the environment has three or more dimensions, or ``ax`` does
            not hold one axes per direction in 2-D.
        TypeError
            If ``ax`` is not a single Axes on a track.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding.spatial import DirectionalPlaceFields
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 9, 100)[:, None], bin_size=1.0, units="cm"
        ... )
        >>> n = env.n_bins
        >>> result = DirectionalPlaceFields(
        ...     firing_rates={
        ...         "fwd": np.linspace(1.0, 5.0, n),
        ...         "rev": np.linspace(5.0, 1.0, n),
        ...     },
        ...     occupancy={"fwd": np.ones(n), "rev": np.ones(n)},
        ...     env=env,
        ... )
        >>> ax = result.plot()
        >>> len(ax.get_lines())
        2
        >>> ax.get_xlabel()
        'Position (cm)'
        """
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes

        from neurospatial.environment._queries import _track_extents

        track = _track_extents(self.env)
        if track is None and self.env.n_dims != 2:
            msg = _format_error(
                f"DirectionalPlaceFields.plot() cannot draw {self.env.n_dims}-D maps.",
                why="it draws lines along a track or one map per direction in 2-D",
                fix="use result.to_dataframe() to inspect the maps",
            )
            raise ValueError(msg)
        created_axes = ax is None
        units = f" ({self.env.units})" if self.env.units else ""

        if track is not None:
            if ax is None:
                _, ax = plt.subplots()
            if not isinstance(ax, Axes):
                msg = _format_error(
                    "DirectionalPlaceFields.plot() draws a 1-D environment on one "
                    "Axes, but ax is not an Axes.",
                    fix="pass a single Axes, for example `fig, ax = plt.subplots()`",
                )
                raise TypeError(msg)
            starts, ends = track
            position = (starts + ends) / 2
            order = np.argsort(position)
            # Break the lines across gaps between track segments.
            gaps = np.flatnonzero(starts[order][1:] > ends[order][:-1] + 1e-9) + 1
            for label in self.labels:
                rate = _to_numpy(self.firing_rates[label]).ravel()
                ax.plot(
                    np.insert(position[order], gaps, np.nan),
                    np.insert(rate[order], gaps, np.nan),
                    label=str(label),
                    **kwargs,
                )
            ax.set_xlabel(f"Position{units}")
            ax.set_ylabel("Firing rate (Hz)")
            if created_axes:
                ax.set_title("Directional place fields")
            ax.legend()
            return ax

        n_panels = len(self.labels)
        if ax is None:
            _, made = plt.subplots(
                1, n_panels, figsize=(4.5 * n_panels, 4.5), squeeze=False
            )
            panels = list(made.ravel())
        else:
            panels = list(np.atleast_1d(np.asarray(ax, dtype=object)).ravel())
        if len(panels) != n_panels:
            msg = _format_error(
                f"DirectionalPlaceFields.plot() got {len(panels)} axes for "
                f"{n_panels} directions.",
                fix="pass one axes per direction, for example "
                "`fig, ax = plt.subplots(1, n_directions)`",
            )
            raise ValueError(msg)
        rates = [_to_numpy(self.firing_rates[label]).ravel() for label in self.labels]
        finite = np.concatenate([rate[np.isfinite(rate)] for rate in rates] or [[]])
        if finite.size:
            kwargs.setdefault("vmin", 0.0)
            kwargs.setdefault("vmax", float(finite.max()))
        kwargs.setdefault("colorbar_label", "Firing rate (Hz)")
        for panel, label, rate in zip(panels, self.labels, rates, strict=True):
            self.env.plot_field(rate, ax=panel, **kwargs)
            if created_axes:
                panel.set_title(str(label))
        return tuple(panels)


def _check_directional_recording(
    env: Environment,
    spike_times: Any,
    times: Any,
    positions: Any,
    *,
    max_gap: float | None,
    epochs: NDArray[np.float64] | None,
    spike_window: NDArray[np.float64] | None,
    break_before: NDArray[np.bool_] | None,
    warn_on_drop: bool,
) -> None:
    """Check the whole recording once before the per-label maps.

    Raises :class:`~neurospatial.InputError` when no interval of the
    recording is usable (positions in another frame, every gap longer than
    ``max_gap``, ``spike_window`` on another clock), as
    :func:`compute_spatial_rate` does, and warns once about the recording's
    excluded intervals and spikes outside the tracked time. The per-label
    maps leave these to this check, so a label without analyzable time is an
    empty map and the warning does not depend on which label a unit fires in.
    """
    from neurospatial._spikes import as_spike_train
    from neurospatial.encoding._spatial_counts import (
        _WARN_ON_DROP_HINT,
        _resolve_spatial_time_support,
        _time_window_message,
        time_window_drops,
        warn_discarded,
    )
    from neurospatial.encoding._validation import validate_trajectory

    call = "compute_directional_place_fields"
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    validate_trajectory(times, positions=positions, context=call, n_dims=env.n_dims)
    _, drop_message = _resolve_spatial_time_support(
        env,
        times,
        positions,
        speed=None,
        min_speed=None,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        warn_on_drop=warn_on_drop,
        break_before=break_before,
    )
    if not warn_on_drop:
        return
    spikes = as_spike_train(spike_times, call=call, population=None)
    t_min, t_max = float(times[0]), float(times[-1])
    n_dropped, n_considered, within = time_window_drops(
        spikes,
        (spikes < t_min) | (spikes >= t_max),
        t_min,
        t_max,
        epochs=epochs,
        spike_window=spike_window,
    )
    window_message = _time_window_message(
        n_dropped, n_considered, t_min, t_max, spikes, scope=within
    )
    warn_discarded([drop_message, window_message], suppress_hint=_WARN_ON_DROP_HINT)


@keyword_hint()
@reads_pynapple_samples
def compute_directional_place_fields(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    direction_labels: NDArray[np.object_],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    missing_label: Hashable | None = None,
    warn_on_drop: bool = True,
) -> DirectionalPlaceFields:
    """Compute place fields conditioned on movement direction or trial type.

    Separates trajectory data by direction labels and computes independent
    place fields for each direction. This enables analysis of directional
    tuning in place cells, where firing rates differ based on which way
    the animal is moving through a location.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Timestamps of spike occurrences (seconds).
    times : NDArray[np.float64], shape (n_timepoints,)
        Timestamps of trajectory samples (seconds). Must be sorted.
    positions : NDArray[np.float64], shape (n_timepoints, n_dims) or (n_timepoints,)
        Position trajectory. For 1D, can be shape (n_timepoints,) or (n_timepoints, 1).
    direction_labels : NDArray[object], shape (n_timepoints,)
        Direction label for each timepoint. Each label is hashable (e.g.,
        "A→B", "forward", "CW", or a lap number). Labels are kept as given:
        ``1`` and ``"1"`` are different directions. ``None``, ``NaN`` and
        ``missing_label`` mark samples without a direction and are excluded. The direction-label
        functions give ``None`` to samples outside every lap, run or trial.
        Labels from ``heading_direction_labels`` and
        ``headings_to_direction_labels`` include ``"stationary"``, an
        ordinary label here that gets its own map; pass
        ``missing_label="stationary"`` to drop it.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Estimation method passed through to the place-field helper.
    bandwidth : float or None, default=None
        Smoothing bandwidth in environment units (e.g., cm); ``None`` is 0 (no
        smoothing) for ``method="binned"``, and 5 cm in declared metric units
        or 2.5 median graph-edge lengths for unspecified or custom units for
        the kernel methods; the units-free default warns with its resolved
        value.
    min_occupancy : float, default=0.0
        Minimum occupancy threshold in seconds. Bins below this threshold are
        set to NaN.

    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_timepoints,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    missing_label : hashable, optional
        A label value that means "no label" (for example ``"none"`` or
        ``-1`` from a labeling function), excluded like ``None``.
    warn_on_drop : bool, default=True
        Warn when more than 5% of the spikes inside ``epochs`` and
        ``spike_window`` fall outside the tracked time (often spike times on
        another clock or in milliseconds), or when the interval gates
        discard much of the recording. ``False`` silences these warnings; a
        selection that leaves no usable interval still raises.
        Spikes outside ``epochs`` and ``spike_window`` are not checked, so a
        clock offset that leaves some spikes inside them goes unreported:
        compare ``spike_times.min()`` and ``.max()`` with ``times`` yourself.

    Returns
    -------
    DirectionalPlaceFields
        Container with:

        - ``firing_rates``: Mapping from each direction label to its firing
          rate map (n_bins,)
        - ``occupancy``: Mapping from each direction label to its per-bin
          occupancy in seconds (n_bins,)
        - ``env``: Spatial environment shared across labels
        - ``labels``: Every label that occurs in ``direction_labels``, sorted
          (numbers numerically, then strings alphabetically; labels that
          cannot be compared keep first-seen order). A label with no
          analyzable time (it occurs only at the last sample, outside
          ``epochs``, or only where positions are missing, outside the
          environment or across gaps longer than ``max_gap``) has zero
          occupancy and an all-NaN rate map.

    Raises
    ------
    ValueError
        If ``direction_labels`` length doesn't match ``times`` length.
    ValueError
        If ``bandwidth`` is not positive.

    See Also
    --------
    compute_spatial_rate : Compute single (non-directional) spatial rate.

    Notes
    -----
    An interval is analyzed only if it passes the gap and bounds checks
    and lies inside ``epochs ∩ spike_window``. The bounds check
    requires both of the interval's samples to be tracked inside the
    environment, so an interval ending in a tracking dropout is left out.
    The same intervals are removed from the spike counts and the occupancy.

    Each interval carries the direction label of its start sample. Separate
    runs of the same label become analysis epochs on the original trajectory,
    so filtering never joins nonadjacent samples into artificial intervals.
    The caller's epochs intersect the label windows; ``spike_window`` remains
    the caller's acquisition coverage.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import compute_directional_place_fields
    >>>
    >>> # Create environment and trajectory (seeded for reproducibility)
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> times = np.linspace(0, 100, 1000)
    >>> env = Environment.from_samples(positions, bin_size=10.0)
    >>>
    >>> # Forward, then a pause without a direction (None), then backward
    >>> labels = np.array(
    ...     ["forward"] * 400 + [None] * 200 + ["backward"] * 400, dtype=object
    ... )
    >>> spike_times = np.sort(rng.uniform(0, 100, 50))
    >>>
    >>> # Compute directional place fields
    >>> result = compute_directional_place_fields(
    ...     env, spike_times, times, positions, labels, bandwidth=10.0
    ... )
    >>> "forward" in result.firing_rates
    True
    >>> result.labels
    ('backward', 'forward')
    """
    # Validate direction_labels length matches times
    from neurospatial._intervals import intersect_intervals
    from neurospatial.encoding._smoothing import (
        resolve_spatial_bandwidth,
        warn_about_bandwidth,
    )
    from neurospatial.encoding._spatial_counts import require_allocentric

    require_allocentric(env, "compute_directional_place_fields()")
    from neurospatial._validation import _refuse_wrong_kind

    _refuse_wrong_kind(
        spike_times,
        name="spike_times",
        caller="compute_directional_place_fields",
        expected="spike_train",
        population=None,
    )

    if len(direction_labels) != len(times):
        msg = (
            f"direction_labels must have same length as times, "
            f"got {len(direction_labels)} and {len(times)}"
        )
        raise ValueError(msg)
    breaks = validate_break_before(
        break_before, len(times), call="compute_directional_place_fields"
    )

    # Label code of each interval (the label at its start sample). None, NaN
    # and missing_label are missing (-1); consecutive intervals with one code
    # form one window, so separate runs never join.
    from neurospatial._observation import label_codes, label_runs

    labels_arr = np.empty(len(direction_labels), dtype=object)
    for i, value in enumerate(direction_labels):
        labels_arr[i] = value
    sample_codes, values = label_codes(labels_arr, missing_label)
    codes = sample_codes[:-1]
    first, last, run_codes = label_runs(codes, np.ones(codes.size, dtype=bool))
    sample_times = np.asarray(times, dtype=np.float64)
    run_windows = np.column_stack([sample_times[first], sample_times[last + 1]])

    # Natural order of the labels themselves: numbers numerically, then strings
    # alphabetically. Labels that cannot be compared keep first-seen order.
    sort_keys: list[Any] = [(isinstance(value, str), value) for value in values]
    try:
        order = sorted(range(len(sort_keys)), key=sort_keys.__getitem__)
    except TypeError:
        order = list(range(len(values)))

    firing_rates_dict: dict[Hashable, NDArray[np.float64]] = {}
    occupancy_dict: dict[Hashable, NDArray[np.float64]] = {}

    resolved_epochs, resolved_spike_window, _ = resolve_time_windows(
        epochs, spike_window
    )
    _check_directional_recording(
        env,
        spike_times,
        times,
        positions,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=breaks,
        warn_on_drop=warn_on_drop,
    )
    # The bandwidth is announced here, once per call: the per-label maps run
    # with window_messages=False.
    if bandwidth is not None or method in ("diffusion_kde", "gaussian_kde"):
        warn_about_bandwidth(
            env, bandwidth, resolve_spatial_bandwidth(env, bandwidth), method
        )
    for code in order:
        label = values[code]
        label_windows = run_windows[run_codes == code]
        windows = (
            label_windows
            if resolved_epochs is None
            else intersect_intervals(label_windows, resolved_epochs)
        )
        if windows.shape[0] == 0:
            # The label occurs but no interval of it can be analyzed: report
            # zero dwell time and a missing (not zero) rate.
            firing_rates_dict[label] = np.full(env.n_bins, np.nan)
            occupancy_dict[label] = np.zeros(env.n_bins)
            continue

        # compute_spatial_rate on this label's windows, already resolved. A
        # label whose samples are all missing, outside the environment or
        # across long gaps has no analyzable time either: a NaN map with
        # zero occupancy. The recording as a whole was checked above.
        single = _spatial_rate_maps(
            env,
            spike_times,
            times,
            positions,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            penalty=None,
            rank=None,
            pooled=True,
            speed=None,
            min_speed=None,
            max_gap=max_gap,
            time_allocation="start",
            windows=TimeWindows(windows, resolved_spike_window, None),
            break_before=breaks,
            n_jobs=1,
            backend="numpy",
            warn_on_drop=warn_on_drop,
            dtype=np.float64,
            unit_ids=None,
            context="compute_spatial_rate",
            single_unit=True,
            window_messages=False,
        )
        firing_rates_dict[label] = np.asarray(single.firing_rates[0], dtype=np.float64)  # type: ignore[index]
        occupancy_dict[label] = np.asarray(single.occupancy, dtype=np.float64)

    return DirectionalPlaceFields(
        spike_window=resolved_spike_window,
        missing_label=missing_label,
        firing_rates=firing_rates_dict,
        occupancy=occupancy_dict,
        env=env,
    )


# ==============================================================================
# Place Field Detection
# ==============================================================================


def detect_place_fields(
    env: Environment,
    firing_rate: SpatialRateResult | ArrayLike,
    *,
    bandwidth: float | None = None,
    threshold: float = 0.2,
    min_peak_rate: float = 1.0,
    min_secondary_peak_fraction: float = 0.0,
    max_mean_rate: float = np.inf,
    subfields: Literal["core", "watershed"] | None = "core",
) -> PlaceFieldsResult:
    """Segment a rate map into place fields around successive peaks.

    Each field is the connected region around a peak where the rate is at
    least ``threshold`` times that field's own peak. Peaks are taken from the
    highest down; the search stops at the first peak below
    ``max(min_peak_rate, min_secondary_peak_fraction * map maximum)``. This
    follows the field search of the AyA Lab neurocode toolbox [1]_,
    generalized from its 1-D track maps to any environment by following
    ``env.connectivity``. Segmentation does not classify cells.

    Parameters
    ----------
    env : Environment
        Spatial environment for binning.
    firing_rate : SpatialRateResult or array, shape (n_bins,)
        One unit's rate map (Hz): a result of :func:`compute_spatial_rate`,
        one unit of :func:`compute_spatial_rates` (``rates[i]``, or iterate:
        ``[detect_place_fields(env, unit) for unit in rates]``, each with its
        ``unit_id``), or one neuron's array. NaN bins (unvisited, below the
        occupancy threshold, or outside a result's occupancy support) never
        belong to a field and do not connect fields.
    bandwidth : float or None, default=None
        Smoothing bandwidth (standard deviation, environment units) the map
        was estimated with. It sets the smallest field, the region one noisy
        bin fills after smoothing (see Notes). Must be finite and >= 0.
        ``None`` with a rate result uses the bandwidth it records (0 for an
        unsmoothed ``"binned"`` map and for ``"glm"``, which records none); a
        different value raises. With an array it is required: pass
        ``result.bandwidth`` of the rate result the map came from, or 0 for
        an unsmoothed map.
    threshold : float, default=0.2
        Fraction of each field's own peak rate that defines its boundary
        (0-1). 0.2 is the neurocode ``MapStats`` default [1]_.
    min_peak_rate : float, default=1.0
        Smallest peak rate (Hz) that can start a field. 1 Hz is ``minPeak``
        of ``MapStats`` in FMAToolbox [2]_ and neurocode [1]_ and opexebo's
        ``firing_field_min_peak`` [3]_; neurocode's 1-D
        ``findPlaceFieldsAvg1D`` uses 2 Hz. Must be finite and >= 0.
    min_secondary_peak_fraction : float, default=0.0
        A field after the first must peak at least this fraction of the
        map's maximum rate. Must be in [0, 1]. The default keeps every peak
        above ``min_peak_rate``. neurocode's 1-D ``findPlaceFieldsAvg1D``
        uses 0.6 (``minPeak2nd``) [1]_, together with a maximum field size
        and a dip below threshold between fields. On simulated tracks and
        mazes 0.6 missed every secondary field at 20-50% of the main peak
        (a 40% arm field on a W-maze, a 50% third arm on a radial maze) and
        gave untuned maps as many fields as 0.0 did, so it is no protection
        against noise.
    max_mean_rate : float, default=np.inf
        Optional upper bound on the unweighted mean of non-NaN spatial-bin
        rates (Hz). The default disables exclusion. A map above the bound
        returns no fields and no warning; ``excluded_reason`` is
        ``"mean_rate_above_threshold"``. This is not the recording-time mean
        firing rate and does not identify cell type.
    subfields : {"core", "watershed"} or None, default="core"
        How to separate two fields that merge at the ``threshold`` level
        (see Notes, step 4).

        - ``"core"``: the rule of FMAToolbox ``MapStats`` [2]_. When the
          field's core is less than half the field, the core is the field
          and the rest is searched again from its own peak. The weaker
          field can then grow through the stronger one's leftover edge and
          around it, which pulls its centroid toward the stronger field.
        - ``"watershed"``: the same fields as ``"core"``, but each bin goes
          to the field whose peak it is reached from by flooding down the
          map, and each field ends at ``threshold`` times its own peak. Two
          touching fields are divided at the valley between them. It never
          reports more fields than ``"core"``, so two fields that ``"core"``
          reports as one stay one field.
        - ``None``: no separation. A field is the whole connected region at
          ``threshold`` times its peak.

    Returns
    -------
    PlaceFieldsResult
        Detected fields, one bin-index array each, in order of decreasing
        peak rate, with the map, its bandwidth and unit, and
        ``excluded_reason``, which says whether the ``max_mean_rate`` filter
        excluded the map. The result is iterable, indexable and sized like a
        list of fields; ``fields_table()`` tabulates each field's size, rates
        and centroid, and ``to_dataframe()`` lists each field's bins.

    Raises
    ------
    TypeError
        If ``firing_rate`` is a population result (:class:`SpatialRatesResult`):
        segment one unit at a time.
    ValueError
        If ``firing_rate`` does not have one value per bin, ``bandwidth`` is
        missing for an array or disagrees with a result's, or a parameter is
        outside its stated range.

    Notes
    -----
    **Algorithm**:

    1. **Optional map-mean filter**: if an explicit finite ``max_mean_rate``
       is exceeded, return no fields.
    2. **Peak**: take the highest bin not yet assigned. Stop if its rate is
       below ``max(min_peak_rate, min_secondary_peak_fraction * map_max)``.
       Because peaks are taken in decreasing order, every later peak would
       also fail.
    3. **Field**: the bins connected to the peak through ``env.connectivity``
       whose rate is at least ``threshold`` times this peak.
    4. **Coalescent fields**: with ``subfields="core"``, repeat step 3 on the
       map minus the field's floor ``m = threshold * peak``, at ``threshold``
       of the remaining height: the core is the connected region above
       ``m + threshold * (peak - m)`` (0.36 of the peak at the default 0.2).
       If the core is less than half the field, a second field shares the
       region, and the core replaces the field; the rest of the region stays
       unassigned, so the neighboring field is found from its own peak with
       the same rules. This is the subfield rule of FMAToolbox ``MapStats``
       [2]_ (``Analyses/MapStats.m``, lines 194-203).

       With ``subfields="watershed"``, the search runs as with ``"core"``
       and the fields it keeps (step 5) give the peaks. Every bin then goes
       to the peak it is reached from by flooding down the map: from all
       peaks at once, the highest bin flooded so far (at equal rates, the
       lowest bin index) passes its peak on to its unflooded neighbors
       through ``env.connectivity``. NaN bins are not flooded, so they do
       not connect basins. Each field is the part of its peak's basin
       connected to the peak, with a rate from ``threshold`` times that peak
       up to the peak. A higher bin in a basin is a bump the search did not
       keep, and it belongs to no field. If a field is then no larger than
       the smallest field (step 5), the lowest such peak is dropped and the
       map is flooded again, so its bins go to its neighbors. A field whose
       region holds no other kept peak is its whole region, as with
       ``None``. This is neurospatial's own construction; the closest
       published method is the FMAToolbox ``MapStats`` subfield rule [2]_.
       Hockeimer et al. [4]_ also split fields with a watershed, but they
       keep a split only when the highest rate on the line between two
       subfields is at most 0.75 times the subfield's own peak (in their
       code; the paper's text says the larger peak), and they do not cut a
       subfield at its own threshold.
    5. **Keep or discard**: mark the field's bins as assigned. Keep it only
       if it is larger than the smallest field (below) and its peak bin is a
       local maximum of the full map (no neighbor has a higher rate). A candidate
       that rises toward a neighbor is the flank of a field already assigned,
       left behind when that field was cut at its own threshold. This is the
       n-D form of neurocode's 1-D check that the rate dips below threshold
       between two fields, and of the ``localmax`` option of ``MapStats``.
    6. Repeat from step 2 until a peak fails the floor or every finite bin
       is assigned.

    The floor in step 2 stops the search on noise. Without it, each
    successive threshold is a fraction of a smaller peak, so small bumps far
    below the main field would be reported as fields.

    **Smallest field.** A field must be larger than what one noisy bin
    becomes after smoothing; anything smaller cannot be told apart from a
    single bin's excess rate. One bin's excess smoothed by a Gaussian kernel
    of standard deviation ``σ = bandwidth`` stays above ``threshold`` times
    its own peak within ``r = σ·sqrt(2·ln(1/threshold))`` of the bin (1.79σ
    at the default 0.2): a segment of length ``2r`` on a track (18 cm at
    σ = 5 cm) or a disk of area ``πr²`` in an open field (253 cm² at
    σ = 5 cm). A real field of width ``s`` smoothed the same way is
    ``sqrt(s² + σ²)/σ`` times wider, so it is always larger. The detector
    measures this region on the environment: it diffuses one unit at the
    candidate's peak bin (``env.diffuse``), keeps the connected bins with a
    finite rate above ``threshold`` times the result's maximum, and sums their
    measures (``env.binning.bins.sizes``). Walls, track ends and arena edges
    shrink it as they shrink a field there. For an unsmoothed map
    (``bandwidth=0``) it is the peak bin. Field and region are compared by
    measure, so the rule does not depend on bin size. This rule is
    neurospatial's choice; ``MapStats`` in FMAToolbox [2]_ and neurocode [1]_
    instead drops fields smaller than a fixed number of bins (100 in 2-D,
    10 in 1-D).

    **Segmentation is not a test.** The detector finds regions of elevated
    rate; it does not ask whether they differ from chance. Every untuned
    Poisson unit firing at 1 Hz or more had a field at every recording length
    tried (1-20 min in a 1 m open field, 2-20 min on a 2 m track), and so did
    every interneuron-like unit with a high, broadly modulated rate (15 Hz
    plus a 5 Hz bump 40 cm wide on the track), whose field is the whole
    track. Untuned 0.5 Hz units lose their field as the map converges below
    ``min_peak_rate`` (7, 2, 0 and 0 of 20 on the track at 2, 5, 10 and
    20 min). Use ``is_place_cell(..., criterion="shuffle")`` for a verdict.
    neurocode's 1-D ``findPlaceFieldsAvg1D`` also drops fields longer than
    60% of the track (``maxSize``) [1]_; a fraction of a track has no
    counterpart in an open field, so no maximum is applied here. Filter the
    fields on :func:`~neurospatial.encoding.field_size` if your protocol
    needs one. An explicit finite ``max_mean_rate`` filters the unweighted
    mean of spatial-bin rates; unequal occupancy makes that different from
    total spikes divided by recording duration. Neither the filter nor the
    resulting field count establishes cell identity.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import detect_place_fields
    >>> xx, yy = np.meshgrid(np.arange(-20.0, 21.0, 2.0), np.arange(-20.0, 21.0, 2.0))
    >>> env = Environment.from_samples(
    ...     np.c_[xx.ravel(), yy.ravel()], bin_size=2.0, units="cm"
    ... )
    >>> # An 8 Hz Gaussian place field (3 cm width) at the center
    >>> distance = np.linalg.norm(env.bin_centers, axis=1)
    >>> firing_rate = 8.0 * np.exp(-(distance**2) / (2 * 3.0**2))
    >>> fields = detect_place_fields(env, firing_rate, bandwidth=0.0)
    >>> len(fields)
    1

    A rate result carries its bandwidth and unit:

    >>> from neurospatial.encoding import compute_spatial_rates
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(-20, 20, (3000, 2))
    >>> times = np.arange(3000) * 0.02
    >>> spikes = [times[np.linalg.norm(positions, axis=1) < 8][::3]]
    >>> rates = compute_spatial_rates(env, spikes, times, positions)
    >>> fields = [detect_place_fields(env, unit) for unit in rates]
    >>> fields[0].bandwidth == rates.bandwidth, fields[0].unit_id
    (True, 0)

    See Also
    --------
    SpatialRateResult.has_place_field : Whether a result's map has a field
    neurospatial.encoding.population.population_coverage : Fields across a population

    References
    ----------
    .. [1] AyA Lab neurocode toolbox, ``PlaceCells/MapStats.m`` and
           ``PlaceCells/findPlaceFieldsAvg1D.m``,
           https://github.com/ayalab1/neurocode
    .. [2] Zugaro, M. FMAToolbox, ``Analyses/MapStats.m``,
           https://github.com/michael-zugaro/FMAToolbox
    .. [3] Kavli Institute (NTNU) opexebo, ``opexebo/defaults.py``,
           https://github.com/kavli-ntnu/opexebo
    .. [4] Hockeimer, W., Lai, R.-Y., Natrajan, M., Snider, W., & Knierim,
           J. J. (2025). Leveraging place field repetition to understand
           positional versus nonpositional inputs to hippocampal field CA1.
           eLife, 14, e85599. https://doi.org/10.7554/eLife.85599; code:
           ``RatterdamOpen_Project/RateMapClass_William_20190308.py``
           (``findPlaceFields``), https://github.com/whock3/ratterdam
    """
    if isinstance(firing_rate, SpatialRatesResult):
        raise TypeError(
            _format_error(
                f"detect_place_fields segments one rate map, but got a population "
                f"of {len(firing_rate)} units (SpatialRatesResult).",
                why="Why: each unit's map has its own fields, so the result is "
                "one PlaceFieldsResult per call",
                fix="call it once per unit, e.g. "
                "[detect_place_fields(env, unit) for unit in rates]; iterating "
                "gives each unit its unit_id",
            )
        )
    unit_ids: list[Any] | None
    if isinstance(firing_rate, SpatialRateResult):
        # A glm map has no kernel and records None; it is unsmoothed (0).
        recorded = (
            0.0 if firing_rate.bandwidth is None else float(firing_rate.bandwidth)
        )
        if bandwidth is None:
            bandwidth = recorded
        elif bandwidth != recorded:
            # glm takes no bandwidth, so its fix names a kernel method.
            glm = firing_rate.bandwidth is None
            raise ValueError(
                _format_error(
                    f"detect_place_fields got bandwidth={bandwidth!r}, but the "
                    "rate result was estimated with "
                    f"bandwidth={firing_rate.bandwidth!r} "
                    f"(method={firing_rate.method!r}).",
                    why="Why: the bandwidth sets the smallest field one noisy "
                    "bin makes after smoothing, so it must be the map's own"
                    + (
                        "; a glm map has no kernel, so its fields are detected "
                        "at bandwidth 0"
                        if glm
                        else ""
                    ),
                    fix="drop bandwidth= to use the result's"
                    + (
                        "; to choose a smoothing scale, compute the rates with a "
                        "kernel method, compute_spatial_rate(..., "
                        f"method='diffusion_kde', bandwidth={bandwidth!r})"
                        if glm
                        else ", or recompute the rates with the bandwidth you want"
                    ),
                )
            )
        rates = np.atleast_2d(
            np.asarray(_to_numpy(firing_rate._supported_rates()), dtype=np.float64)
        )
        unit_ids = [firing_rate.unit_id]
    else:
        if bandwidth is None:
            raise ValueError(
                _format_error(
                    "detect_place_fields needs bandwidth= with a rate-map array.",
                    why="Why: the bandwidth the map was smoothed with sets the "
                    "smallest field one noisy bin makes",
                    fix="pass the rate result itself, detect_place_fields(env, "
                    "rates[i]), which records it; or bandwidth=rates.bandwidth "
                    "of the result the array came from, or bandwidth=0.0 for "
                    "an unsmoothed map",
                )
            )
        rates = np.asarray(firing_rate, dtype=np.float64)[np.newaxis]
        unit_ids = None
    if rates.shape[1:] != (env.n_bins,):
        msg = (
            f"firing_rate shape {rates.shape[1:]} does not match "
            f"env.n_bins ({env.n_bins})"
        )
        raise ValueError(msg)
    _check_place_field_arguments(
        threshold=threshold,
        min_peak_rate=min_peak_rate,
        min_secondary_peak_fraction=min_secondary_peak_fraction,
        bandwidth=bandwidth,
        subfields=subfields,
    )
    searches = [
        _search_place_fields(
            env,
            rate,
            threshold=threshold,
            min_peak_rate=min_peak_rate,
            min_secondary_peak_fraction=min_secondary_peak_fraction,
            max_mean_rate=max_mean_rate,
            subfields=subfields,
        )
        for rate in rates
    ]
    return _keep_place_fields(env, searches, bandwidth, threshold, unit_ids)[0]


def _check_place_field_arguments(
    *,
    threshold: float,
    min_peak_rate: float,
    min_secondary_peak_fraction: float,
    bandwidth: float,
    subfields: object,
) -> None:
    """Refuse invalid :func:`detect_place_fields` arguments."""
    validate_choice(
        subfields, _SUBFIELD_RULES, name="subfields", owner="detect_place_fields()"
    )
    if not 0 < threshold < 1:
        msg = f"threshold must be in (0, 1), got {threshold}"
        raise ValueError(msg)
    if not (np.isfinite(min_peak_rate) and min_peak_rate >= 0):
        raise ValueError(
            _format_error(
                f"min_peak_rate must be a finite rate >= 0 Hz, got {min_peak_rate}.",
                why="Why: it is the smallest peak rate that can start a field",
                fix="pass min_peak_rate=1.0 (the MapStats and opexebo default) or another rate in Hz",
            )
        )
    if not 0 <= min_secondary_peak_fraction <= 1:
        raise ValueError(
            _format_error(
                "min_secondary_peak_fraction must be in [0, 1], got "
                f"{min_secondary_peak_fraction}.",
                why="Why: it is a fraction of the map's maximum rate",
                fix="pass 0.0 to keep every peak above min_peak_rate, or a fraction such as 0.6",
            )
        )

    if not (np.isfinite(bandwidth) and bandwidth >= 0):
        raise ValueError(
            _format_error(
                f"bandwidth must be a finite length >= 0, got {bandwidth}.",
                why="Why: it is the smoothing the map was estimated with, which "
                "sets the smallest field",
                fix="pass result.bandwidth of the rate map, or 0.0 for an "
                "unsmoothed map",
            )
        )


class _FieldSearch(NamedTuple):
    """Candidate fields of one rate map, before the smallest-field test.

    ``peaks[i]`` is the peak bin of ``candidates[i]``; ``observed`` marks the
    map's finite bins. ``excluded_reason`` is set when the map-mean filter
    excluded the whole map.
    """

    peaks: list[int]
    candidates: list[NDArray[np.int64]]
    observed: NDArray[np.bool_]
    firing_rate: NDArray[np.float64]
    excluded_reason: Literal["mean_rate_above_threshold"] | None = None
    watershed: bool = False


def _search_place_fields(
    env: Environment,
    firing_rate: NDArray[np.float64],
    *,
    threshold: float,
    min_peak_rate: float,
    min_secondary_peak_fraction: float,
    max_mean_rate: float,
    subfields: Literal["core", "watershed"] | None,
) -> _FieldSearch:
    """Segment one validated rate map into candidate fields around its peaks."""

    finite = np.isfinite(firing_rate)
    if not np.any(finite):
        return _FieldSearch([], [], finite, firing_rate)

    # A finite cutoff is an explicit map-level filter, independent of cell type.
    if np.isfinite(max_mean_rate):
        mean_rate = np.nanmean(firing_rate)
        if mean_rate > max_mean_rate:
            return _FieldSearch(
                [], [], finite, firing_rate, "mean_rate_above_threshold"
            )

    min_peak = max(
        min_peak_rate, min_secondary_peak_fraction * float(np.max(firing_rate[finite]))
    )
    measures = np.asarray(env.binning.bins.sizes, dtype=np.float64)
    # Unassigned bins keep their rate; assigned and missing bins are NaN.
    remaining = np.where(finite, firing_rate, np.nan)
    peaks: list[int] = []
    candidates: list[NDArray[np.int64]] = []
    while np.any(np.isfinite(remaining)):
        peak_idx = int(np.nanargmax(remaining))
        peak_rate = float(remaining[peak_idx])
        if peak_rate <= 0 or peak_rate < min_peak:
            break
        if _rises_toward_a_neighbor(peak_idx, peak_rate, firing_rate, env):
            # A flank of a field already cut, not a peak. Only this bin is
            # set aside: the region above its own floor can run into a
            # neighbouring field, which must still be found from its own peak.
            remaining[peak_idx] = np.nan
            continue

        # NaN compares False, so assigned and missing bins are excluded.
        floor = peak_rate * threshold
        field_bins = _seed_component(peak_idx, remaining >= floor, env)
        # The watershed takes its peaks from the "core" search.
        if subfields is not None:
            # FMAToolbox MapStats: the same search on the map minus the floor.
            core_level = floor + (peak_rate - floor) * threshold
            core = _seed_component(peak_idx, remaining >= core_level, env)
            if measures[core].sum() < measures[field_bins].sum() / 2:
                field_bins = core
        remaining[field_bins] = np.nan
        peaks.append(peak_idx)
        candidates.append(field_bins)
    return _FieldSearch(
        peaks, candidates, finite, firing_rate, watershed=subfields == "watershed"
    )


def _keep_place_fields(
    env: Environment,
    searches: Sequence[_FieldSearch],
    bandwidth: float,
    threshold: float,
    unit_ids: Sequence[Any] | None = None,
) -> list[PlaceFieldsResult]:
    """Keep each map's candidates larger than the field one noisy bin makes.

    The peaks of every map are tested together, so a population shares one
    batch of impulse diffusions (see :func:`_smoothed_bin_measures`).
    ``unit_ids`` labels the maps (None: no labels).
    """
    measures = np.asarray(env.binning.bins.sizes, dtype=np.float64)
    peaks = np.array(
        [peak for search in searches for peak in search.peaks], dtype=np.int64
    )
    owners = np.repeat(np.arange(len(searches)), [len(s.peaks) for s in searches])
    observed = np.zeros((env.n_bins, len(searches)), dtype=bool)
    for column, search in enumerate(searches):
        observed[:, column] = search.observed
    smallest = _smoothed_bin_measures(
        env, peaks, owners, observed, bandwidth, threshold, measures
    )
    labels = [None] * len(searches) if unit_ids is None else list(unit_ids)
    results = []
    start = 0
    for search, label in zip(searches, labels, strict=True):
        minima = smallest[start : start + len(search.candidates)]
        start += len(search.candidates)
        kept = [
            (peak, field_bins, minimum)
            for peak, field_bins, minimum in zip(
                search.peaks, search.candidates, minima, strict=True
            )
            if measures[field_bins].sum() > minimum
        ]
        fields = (
            _watershed_fields(env, search.firing_rate, kept, threshold, measures)
            if search.watershed
            else [field_bins for _, field_bins, _ in kept]
        )
        results.append(
            PlaceFieldsResult(
                fields=fields,
                env=env,
                firing_rate=search.firing_rate,
                bandwidth=bandwidth,
                excluded_reason=search.excluded_reason,
                unit_id=label,
            )
        )
    return results


@keyword_hint()
@reads_pynapple_samples
def has_place_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    threshold: float = 0.2,
    min_peak_rate: float = 1.0,
    min_secondary_peak_fraction: float = 0.0,
    max_mean_rate: float = np.inf,
    subfields: Literal["core", "watershed"] | None = "core",
    min_occupancy: float = 0.0,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    backend: BackendType = "numpy",
    warn_on_drop: bool = True,
) -> bool:
    """Detect whether the rate map has a place field; this is not a cell-type verdict.

    Computes the rate map with :func:`compute_spatial_rate` and runs
    :func:`detect_place_fields` on it. Untuned Poisson units firing at 1 Hz
    had a detected field in 20 of 20 units at 1, 2, 5, 10 and 20 minutes,
    and at 0.5 Hz in 20, 20, 20, 19 and 11 of 20 (30 Hz Ornstein-Uhlenbeck
    trajectory at 15 cm/s in a 100 x 100 cm arena, 5 cm bins, diffusion KDE
    with a 5 cm bandwidth). Use ``is_place_cell(..., criterion="shuffle")``,
    or the labels of :func:`spatial_cell_type_significance` for a population,
    for a verdict that controls false positives under the circular-shift null.

    Parameters
    ----------
    env : Environment
        Spatial environment defining the discretization.
    spike_times : NDArray[np.float64], shape (n_spikes,)
        Times of spikes.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    positions : ndarray, shape (n_samples, n_dims)
        Raw animal coordinates aligned with times, in environment units.

    method : {"diffusion_kde", "gaussian_kde", "binned", "glm"}, default="diffusion_kde"
        Rate-map estimator; see :func:`compute_spatial_rate`.
    bandwidth : float or None, default=None
        Smoothing bandwidth in environment units; None is 0 (no smoothing)
        for ``method="binned"``, and 5 cm in declared metric units or 2.5
        median graph-edge lengths for unspecified or custom units for the
        kernel methods; the units-free default warns with its resolved value.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``, and the speed derived for ``min_speed`` is not
        smoothed across it.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording.
    threshold : float, default=0.2
        Fraction of each field's own peak rate that defines its boundary (0-1).
        A field must be larger than one noisy bin smoothed with ``bandwidth``
        (see :func:`detect_place_fields`); for ``method="glm"``, larger than
        one bin.
    min_peak_rate : float, default=1.0
        Smallest peak rate (Hz) that can start a field.
    min_secondary_peak_fraction : float, default=0.0
        A field after the first must peak at least this fraction of the map's
        maximum rate. The default keeps every peak above ``min_peak_rate``;
        see :func:`detect_place_fields`.
    max_mean_rate : float, default=np.inf
        Optional upper bound on the unweighted mean of non-NaN spatial-bin
        rates (Hz). The default disables exclusion. A map above the bound
        returns no fields and no warning; ``excluded_reason`` is
        ``"mean_rate_above_threshold"``. This is not the recording-time mean
        firing rate and does not identify cell type.
    subfields : {"core", "watershed"} or None, default="core"
        How to separate two fields that merge at ``threshold``: ``"core"``
        is the FMAToolbox ``MapStats`` rule, ``"watershed"`` divides them at
        the valley between them, ``None`` does not separate them (see
        :func:`detect_place_fields`).

    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included; bins below the
        threshold are set to NaN. ``0.0`` masks nothing (unvisited bins are
        NaN regardless). ``method="glm"`` fits every bin and accepts only
        ``0.0``.
    penalty : float | None, default=None
        (``method="glm"`` only.) Fixed smoothness penalty ``λ`` (≥ 0; ``0`` = no
        penalty). ``None`` (the default) selects ``λ`` by REML; on pathologically
        under-sampled data where no ``λ`` yields a converged, positive-definite
        fit, REML raises ``ValueError`` (supply a fixed ``penalty``, coarsen the
        grid, or reduce ``rank``). Mutually exclusive with the ratio methods.
    rank : int | None, default=None
        (``method="glm"`` only.) Requested rank of the reduced-rank penalty basis
        (≥ 1). ``None`` uses the module default cap. An out-of-range value is
        **clamped** (never rejected) to the effective rank
        ``max(n_live_components, min(n_live_bins, rank))``, reported via
        ``result.rank``. Mutually exclusive with the ratio methods.
    pooled : bool, default=True
        (``method="glm"`` only; strict ``bool``.) Whether REML selects **one
        shared** smoothing penalty ``λ`` for the population (``True``, the
        default) or an **independent per-unit** ``λ`` (``False``). ``pooled=False``
        runs the REML search once per unit, so the fit costs roughly one REML per
        neuron; ``result.penalty`` / ``reml_objective`` / ``reml_at_boundary``
        then hold that unit's own value (a single-unit call unwraps them to
        scalars). Zero-spike units (whose ``λ`` is unidentified) fall back to the
        pooled ``λ`` over the informative units, flagged
        ``penalty_selected_by_reml=False`` with ``reml_objective=nan``. A supplied
        fixed ``penalty`` beats ``pooled`` (REML is skipped and one scalar ``λ``
        is recorded); ``pooled`` is likewise a no-op at ``penalty_rank == 0`` (a
        shared-basis property) or when no unit spikes. A boundary warning means
        the selected ``λ`` is near a search bound -- ``λ`` itself is weakly
        identified; the fitted field is finite but its sensitivity to ``λ`` should
        be checked. Under ``pooled=False`` that warning names the affected
        ``unit_ids``; the shared ``pooled=True`` warning refers to "the pooled
        fit". Passing ``pooled=False`` with a ratio method raises ``ValueError``.
    speed : ndarray, shape (n_samples,), optional
        Precomputed instantaneous speed at each trajectory sample (physical
        units / second). Only used when ``min_speed`` is set. When
        ``min_speed`` is set and ``speed`` is ``None``, speed is auto-derived
        from the trajectory (see ``min_speed``). Pass your own ``speed`` for
        geodesic / linearized-track environments where the Euclidean
        auto-default is not appropriate. A wrong-length array raises
        ``ValueError``.
    min_speed : float, optional
        Minimum speed threshold (physical units / second). When set, low-speed
        periods are excluded from BOTH the spike numerator AND the occupancy
        denominator using ONE shared per-interval speed gate, so the firing
        rate stays correct (gating only one side would bias the rate). When
        ``None`` (the default) NO speed filtering is applied.

        **Auto-speed convention.** When ``min_speed`` is set and ``speed`` is
        ``None``, speed is :func:`neurospatial.behavior.compute_speed`: the
        magnitude of the 0.1 s Gaussian-smoothed velocity, so tracking jitter
        at rest is not counted as running. Pass ``speed`` for another
        definition, such as speed along a linearized track.

        **Choosing it.** Without a gate, time and spikes at rest (reward
        sites, track ends) enter the map. On a simulated track where the
        animal rested 27% of the time and fired 3 Hz at rest, spatial
        information was 0.85-0.94 bits/spike without a gate and 1.55-1.84
        with ``min_speed=4`` cm/s, and the rest firing made extra fields at
        the track ends. No speed suits every recording: place maps are
        usually built from running, so pick the threshold from your animal's
        speed distribution, between the resting and running modes.
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend. For
        ``method="glm"``, a resolved ``jax`` backend runs the penalized-Poisson
        fit + REML through an optional **float32** JAX mirror of the NumPy/SciPy
        core (``backend="jax"`` requires the ``jax`` extra, like the ratio
        methods; ``"auto"`` uses it when available and otherwise the NumPy core).
        The float32 mirror matches the float64 core to ~1e-6 at a fixed penalty (a
        touch looser under automatic REML, which picks a slightly different
        ``lambda``) and is markedly faster on populations. The returned
        diagnostics stay float64 either way.
    warn_on_drop : bool, default=True
        Warn when more than 5% of the spikes inside ``epochs`` and
        ``spike_window`` fall outside the tracked time (often spike times on
        another clock or in milliseconds), when more than 5% map outside the
        environment's bins, or when the interval gates discard much of the
        recording. Spikes excluded by ``min_speed`` are intentional and do
        not count. ``False`` silences these warnings; a selection that leaves
        no usable interval still raises. Spikes outside ``epochs`` and
        ``spike_window`` are not checked, so a clock offset that leaves some
        spikes inside them goes unreported: compare ``spike_times.min()`` and
        ``.max()`` with ``times`` yourself.


    Returns
    -------
    bool
        True when a place field is detected.

    Notes
    -----
    A detected field says the map has a region of elevated rate, not that the
    rate differs from chance. Short recordings give spiky maps, and smoothing
    turns a few spikes into a field-sized bump. For publication, report a
    circular-shift test and its assumptions instead.

    The "Which place-cell function?" table in the user guide compares this
    function with its neighbors:
    <https://edeno.github.io/neurospatial/user-guide/neuroscience-metrics/#which-place-cell-function>

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import has_place_field
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    >>> times = np.linspace(0, 40, 1000)
    >>> spike_times = np.sort(rng.uniform(0, 40, 50))
    >>> result = has_place_field(env, spike_times, times, positions)
    >>> type(result)
    <class 'bool'>

    See Also
    --------
    compute_spatial_rate : Full spatial rate computation
    detect_place_fields : The field detector, returning the fields themselves
    SpatialRateResult.has_place_field : The same check on a computed rate map
    is_place_cell : Place-cell verdict by information screen or shuffle test
    """
    result = compute_spatial_rate(
        env,
        spike_times,
        times,
        positions,
        method=method,
        bandwidth=bandwidth,
        max_gap=max_gap,
        epochs=epochs,
        break_before=validate_break_before(
            break_before, len(times), call="has_place_field"
        ),
        spike_window=spike_window,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        speed=speed,
        min_speed=min_speed,
        backend=backend,
        warn_on_drop=warn_on_drop,
    )
    return result.has_place_field(
        threshold=threshold,
        min_peak_rate=min_peak_rate,
        min_secondary_peak_fraction=min_secondary_peak_fraction,
        max_mean_rate=max_mean_rate,
        subfields=subfields,
    )


@keyword_hint()
@reads_pynapple_samples
def is_place_cell(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    criterion: Literal["threshold", "shuffle"] = "shuffle",
    min_info: float = 0.5,
    alpha: float = 0.05,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
    unit_id: Hashable | None = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    backend: BackendType = "numpy",
    warn_on_drop: bool = True,
) -> bool:
    """Test whether one neuron is a place cell with a circular-shift shuffle (default).

    By default (``criterion="shuffle"``) the neuron is a place cell when its
    spatial information is significant (``p_value < alpha``) against
    ``n_shuffles=1000`` circularly shifted spike trains and its
    bias-corrected spatial information (spatial information minus the
    information an untuned cell shows by chance,
    ``spatial_information_bias``) reaches ``min_info``: the rule of
    ``passes_spatial_information`` in
    :meth:`CellTypeSignificanceResult.summary_table`. The shuffle costs about
    ``n_shuffles`` rate-map recomputes; pass ``rng`` (an integer seed or a
    Generator) to make the verdict reproducible. ``criterion="threshold"`` is
    the fast screen: it applies only the ``min_info`` cutoff, like
    :meth:`SpatialRateResult.is_place_cell`. To detect a field, use
    ``has_place_field()``; field detection also flags most noise units.
    Both criteria test information alone; the ``is_place`` column of
    :meth:`SpatialRatesResult.summary_table` and of
    :meth:`CellTypeSignificanceResult.summary_table` also excludes known grid
    and border cells, so a grid cell that passes information is True here and
    False there.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure, for example
        one built with ``Environment.from_samples()``.
    spike_times : ndarray, shape (n_spikes,)
        Times of spike events in seconds. Can be empty.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    positions : ndarray, shape (n_samples, n_dims)
        Raw animal coordinates aligned with times, in environment units.

    criterion : {"shuffle", "threshold"}, default="shuffle"
        Test circular-shift significance (a verdict), or screen the observed
        statistic only (fast).
    min_info : float, default=0.5
        Inclusive cutoff on bias-corrected spatial information (bits/spike),
        for both criteria; a common screening convention, not derived or
        calibrated.
    alpha : float, default=0.05
        P-value level for ``criterion="shuffle"``.
    n_shuffles : int, default=1000
        Number of circular shifts in shuffle mode.
    min_shift : float, default=20.0
        Minimum shift in analyzed seconds, in shuffle mode.
    rng : numpy.random.Generator, int or None, default=None
        Shuffle random source; an integer seed with the same unit label is stable
        across single and population calls.
    unit_id : hashable or None, default=None
        Shuffle stream label; None uses label 0. Match the population's label.
    method : {"diffusion_kde", "gaussian_kde", "binned", "glm"}, default="diffusion_kde"
        Estimator to use:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph. Like ``gaussian_kde`` and
          ``binned``, it leaves every bin the animal never visited NaN: the
          kernel reaches them, but a rate there would be extrapolated.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers, so by construction it
          smooths through walls. In a simulated maze, 10 cm from a field
          center across a thin wall, the rate reached 21% of the field's peak
          at a 5 cm bandwidth and 60% at 10 cm, and the leak was detected as
          a second field (``diffusion_kde`` and ``binned``: 0-0.3%, no
          leaked field). Use ``diffusion_kde`` or ``binned`` in walled mazes.
        - **binned**: Occupancy-normalized histogram, spike count divided by
          occupancy in each bin, with no smoothing by default. Pass
          ``bandwidth`` to smooth that raw rate afterwards; smoothing after
          the division can introduce discretization artifacts. An unsmoothed
          map has as many independent bins as visited bins, so on a short
          recording a unit can have fewer spikes than that: its information
          bias is then unknown, and the place screen and verdict are False
          ("undetermined" in labels). Pass ``bandwidth`` or record longer.
        - **glm**: Penalized-Poisson GAM. Occupancy enters as a **log-offset**
          (never a denominator) and the smoothness penalty λ is chosen by REML,
          so the fit returns **finite rates everywhere** -- including
          low-occupancy and unvisited bins where the ratio estimators NaN. Tuned
          with ``penalty`` and ``rank`` (not ``bandwidth`` / ``min_occupancy``, which are mutually exclusive
          with ``method="glm"``).

        Note: ``diffusion_kde``, and ``binned`` with a ``bandwidth``, smooth
        matrix-free via the cached finite-volume eigenbasis (O(n_bins·rank) per neuron) — they never build a
        dense kernel and scale to large/fine grids. ``glm`` also avoids a dense
        O(n_bins²) kernel (it fits on the rank-``r`` basis), but its
        penalized-Poisson fit is heavier and **not** linear in ``rank``: each
        Newton step builds a per-unit (r, r) Hessian ``Bᵀ diag(μ) B`` (≈
        O(n_units·n_bins·rank²)) and solves it (≈ O(n_units·rank³)), and this is
        repeated across Newton iterations and REML λ candidates. Keep ``rank``
        modest for large populations. Only ``gaussian_kde`` builds a dense
        O(n_bins²) matrix; for very large grids prefer ``diffusion_kde``, or
        increase ``bin_size``.

    bandwidth : float | None, default=None
        Smoothing bandwidth in environment units; ``None`` is 0 (no
        smoothing) for ``method="binned"``, and 5 cm in declared metric units
        (0.05 m / 50 mm) or 2.5 median graph-edge lengths for unspecified or
        custom units for ``"diffusion_kde"`` and ``"gaussian_kde"``; the
        units-free default warns with its resolved value. Larger values smooth
        more; an explicit value smooths ``"binned"`` maps too.
        Mutually exclusive with ``method="glm"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included; bins below the
        threshold are set to NaN. ``0.0`` masks nothing (unvisited bins are
        NaN regardless). ``method="glm"`` fits every bin and accepts only
        ``0.0``.
    penalty : float | None, default=None
        (``method="glm"`` only.) Fixed smoothness penalty ``λ`` (≥ 0; ``0`` = no
        penalty). ``None`` (the default) selects ``λ`` by REML; on pathologically
        under-sampled data where no ``λ`` yields a converged, positive-definite
        fit, REML raises ``ValueError`` (supply a fixed ``penalty``, coarsen the
        grid, or reduce ``rank``). Mutually exclusive with the ratio methods.
    rank : int | None, default=None
        (``method="glm"`` only.) Requested rank of the reduced-rank penalty basis
        (≥ 1). ``None`` uses the module default cap. An out-of-range value is
        **clamped** (never rejected) to the effective rank
        ``max(n_live_components, min(n_live_bins, rank))``, reported via
        ``result.rank``. Mutually exclusive with the ratio methods.
    pooled : bool, default=True
        (``method="glm"`` only; strict ``bool``.) Whether REML selects **one
        shared** smoothing penalty ``λ`` for the population (``True``, the
        default) or an **independent per-unit** ``λ`` (``False``). ``pooled=False``
        runs the REML search once per unit, so the fit costs roughly one REML per
        neuron; ``result.penalty`` / ``reml_objective`` / ``reml_at_boundary``
        then hold that unit's own value (a single-unit call unwraps them to
        scalars). Zero-spike units (whose ``λ`` is unidentified) fall back to the
        pooled ``λ`` over the informative units, flagged
        ``penalty_selected_by_reml=False`` with ``reml_objective=nan``. A supplied
        fixed ``penalty`` beats ``pooled`` (REML is skipped and one scalar ``λ``
        is recorded); ``pooled`` is likewise a no-op at ``penalty_rank == 0`` (a
        shared-basis property) or when no unit spikes. A boundary warning means
        the selected ``λ`` is near a search bound -- ``λ`` itself is weakly
        identified; the fitted field is finite but its sensitivity to ``λ`` should
        be checked. Under ``pooled=False`` that warning names the affected
        ``unit_ids``; the shared ``pooled=True`` warning refers to "the pooled
        fit". Passing ``pooled=False`` with a ratio method raises ``ValueError``.
    speed : ndarray, shape (n_samples,), optional
        Precomputed instantaneous speed at each trajectory sample (physical
        units / second). Only used when ``min_speed`` is set. When
        ``min_speed`` is set and ``speed`` is ``None``, speed is auto-derived
        from the trajectory (see ``min_speed``). Pass your own ``speed`` for
        geodesic / linearized-track environments where the Euclidean
        auto-default is not appropriate. A wrong-length array raises
        ``ValueError``.
    min_speed : float, optional
        Minimum speed threshold (physical units / second). When set, low-speed
        periods are excluded from BOTH the spike numerator AND the occupancy
        denominator using ONE shared per-interval speed gate, so the firing
        rate stays correct (gating only one side would bias the rate). When
        ``None`` (the default) no speed filtering is applied.

        **Auto-speed convention.** When ``min_speed`` is set and ``speed`` is
        ``None``, speed is :func:`neurospatial.behavior.compute_speed`: the
        magnitude of the 0.1 s Gaussian-smoothed velocity, so tracking jitter
        at rest is not counted as running. Pass ``speed`` for another
        definition, such as speed along a linearized track.

        **Choosing it.** Without a gate, time and spikes at rest (reward
        sites, track ends) enter the map. On a simulated track where the
        animal rested 27% of the time and fired 3 Hz at rest, spatial
        information was 0.85-0.94 bits/spike without a gate and 1.55-1.84
        with ``min_speed=4`` cm/s, and the rest firing made extra fields at
        the track ends. No speed suits every recording: place maps are
        usually built from running, so pick the threshold from your animal's
        speed distribution, between the resting and running modes.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``, and the speed derived for ``min_speed`` is not
        smoothed across it.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Computation backend for rate map smoothing:

        - ``"numpy"``: Use NumPy for all computations. Works everywhere.
        - ``"jax"``: Use JAX for rate computation. Requires JAX installation.
          Enables GPU acceleration and JAX transformations (jit, grad).
        - ``"auto"``: Use JAX if available, otherwise NumPy.

        Note: Binning operations (spike counting, occupancy) always use NumPy.
        Only the smoothing/rate computation uses the selected backend. For
        ``method="glm"``, a resolved ``jax`` backend runs the penalized-Poisson
        fit + REML through an optional **float32** JAX mirror of the NumPy/SciPy
        core (``backend="jax"`` requires the ``jax`` extra, like the ratio
        methods; ``"auto"`` uses it when available and otherwise the NumPy core).
        The float32 mirror matches the float64 core to ~1e-6 at a fixed penalty (a
        touch looser under automatic REML, which picks a slightly different
        ``lambda``) and is markedly faster on populations. The returned
        diagnostics stay float64 either way.
    warn_on_drop : bool, default=True
        Warn when more than 5% of the spikes inside ``epochs`` and
        ``spike_window`` fall outside the tracked time (often spike times on
        another clock or in milliseconds), when more than 5% map outside the
        environment's bins, or when the interval gates discard much of the
        recording. Spikes excluded by ``min_speed`` are intentional and do
        not count. ``False`` silences these warnings; a selection that leaves
        no usable interval still raises. Spikes outside ``epochs`` and
        ``spike_window`` are not checked, so a clock offset that leaves some
        spikes inside them goes unreported: compare ``spike_times.min()`` and
        ``.max()`` with ``times`` yourself.


    Returns
    -------
    bool
        Whether the chosen criterion is met.

    Raises
    ------
    ValueError
        If the criterion, inputs, or mode-specific keywords are invalid.

    Notes
    -----
    ``criterion="threshold"`` compares
    ``spatial_information() - spatial_information_bias()`` of the rate map
    with ``min_info``; see :meth:`SpatialRateResult.is_place_cell` for its
    false- and true-positive rates. Units with fewer spikes than the map has
    effectively independent bins fail, with a warning. For publication,
    report a circular-shift test and its assumptions instead of a cutoff verdict.

    ``criterion="shuffle"`` is True when the information test of
    :func:`place_cell_significance` has ``p_value < alpha`` and the
    bias-corrected information reaches ``min_info``. In 2-minute recordings,
    untuned 0.5 Hz units passed the information test by chance (p = 0.02 and
    0.04 against 99 shifts) with only 0.04 bits/spike beyond their bias; the
    cutoff refuses them. A significant unit with fewer spikes than the map
    has independent bins has an unknown bias and is not a place cell, with a
    warning.

    **Calibration (simulation).** Untuned Poisson cells firing 0.5-3 Hz in
    10-minute explorations of a 1 m box (5 cm bins, the default 5 cm
    diffusion kernel): the p-values of :func:`place_cell_significance` were
    uniform, and 61 of 1400 (4.4%) fell below 0.05 with 199 shifts, where
    an exact test rejects 4.5%. No untuned cell passed the verdict (0 of
    200), because none reached ``min_info`` beyond its bias: the verdict's
    false-positive rate is well below ``alpha``. The cutoff costs power. In
    the same sessions, of 20 place cells with a Gaussian field (standard deviation 10 cm,
    0.1 Hz baseline), the verdict found 2 at a 1 Hz peak (about 100
    spikes), 15 at 2 Hz (about 140) and 20 at 5 Hz (199 shifts throughout).

    Circular shifting costs about n_shuffles recomputes of the
    plural map. Its null assumes stable firing statistics on the joined
    analyzed clock; recording gaps and excluded epochs are never shift
    destinations. Results keep no raw arrays or recompute closures.
    Shuffle keywords belong only to the shuffle; passing one with a value
    other than its default with ``criterion="threshold"`` raises.

    The "Which place-cell function?" table in the user guide compares this
    function with its neighbors:
    <https://edeno.github.io/neurospatial/user-guide/neuroscience-metrics/#which-place-cell-function>

    See Also
    --------
    place_cell_significance : Population significance on raw arrays.
    spatial_cell_type_significance : Place, grid and border tests together.
    compute_spatial_rate : Compute the map without classification.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import is_place_cell
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (3000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0, units="cm")
    >>> times = np.linspace(0, 120, 3000)  # 2 minutes, in seconds
    >>> spike_times = np.sort(rng.uniform(0, 120, 150))  # untuned unit
    >>> is_place_cell(env, spike_times, times, positions, rng=0)
    False

    The threshold screen is fast and runs no null test:

    >>> is_place_cell(env, spike_times, times, positions, criterion="threshold")
    False
    """
    check_criterion(criterion, ("threshold", "shuffle"), call="is_place_cell")
    from neurospatial._validation import _refuse_wrong_kind

    _refuse_wrong_kind(
        spike_times,
        name="spike_times",
        caller="is_place_cell",
        expected="spike_train",
        population="compute_spatial_rates",
    )
    check_mode_keywords(
        criterion,
        threshold={},
        shuffle={
            "n_shuffles": n_shuffles,
            "min_shift": min_shift,
            "rng": rng,
            "unit_id": unit_id,
            "alpha": alpha,
        },
        defaults=_SHUFFLE_DEFAULTS,
        call="is_place_cell",
    )
    breaks = validate_break_before(break_before, len(times), call="is_place_cell")

    def rate_map() -> SpatialRateResult:
        return compute_spatial_rate(
            env,
            spike_times,
            times,
            positions,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            penalty=penalty,
            rank=rank,
            pooled=pooled,
            speed=speed,
            min_speed=min_speed,
            max_gap=max_gap,
            epochs=epochs,
            break_before=breaks,
            spike_window=spike_window,
            backend=backend,
            warn_on_drop=warn_on_drop,
        )

    if criterion == "shuffle":
        label = 0 if unit_id is None else unit_id
        return shuffle_information_verdict(
            lambda: place_cell_significance(
                env,
                [spike_times],
                times,
                positions,
                unit_ids=[label],
                n_shuffles=n_shuffles,
                min_shift=min_shift,
                rng=rng,
                method=method,
                bandwidth=bandwidth,
                min_occupancy=min_occupancy,
                penalty=penalty,
                rank=rank,
                pooled=pooled,
                speed=speed,
                min_speed=min_speed,
                max_gap=max_gap,
                epochs=epochs,
                break_before=breaks,
                spike_window=spike_window,
                backend=backend,
                warn_on_drop=warn_on_drop,
            ).sel(unit_id=label),
            rate_map,
            alpha=alpha,
            min_info=min_info,
            kind="place",
        )
    return rate_map().is_place_cell(min_info=min_info)


def _rises_toward_a_neighbor(
    peak_idx: int,
    peak_rate: float,
    firing_rate: NDArray[np.float64],
    env: Environment,
) -> bool:
    """Whether a neighbor of the candidate peak has a higher rate in the full map.

    Such a candidate is the flank of an already assigned field, not a peak.
    """
    neighbors = list(env.connectivity.neighbors(peak_idx))
    return bool(np.any(firing_rate[neighbors] > peak_rate))


def _watershed_basins(
    env: Environment, firing_rate: NDArray[np.float64], peaks: Sequence[int]
) -> NDArray[np.intp]:
    """Basin of every bin: the peak it is reached from by flooding down the map.

    From all peaks at once, the flood takes the highest flooded bin next (at
    equal rates the lowest bin index) and gives each of its unflooded finite
    neighbors that bin's basin. NaN bins are never flooded, so they do not
    connect basins.

    Parameters
    ----------
    env : Environment
        Environment whose neighbor pairs connect the bins.
    firing_rate : ndarray of float64, shape (n_bins,)
        Rate map (Hz).
    peaks : sequence of int, length (n_peaks,)
        Bin of each basin's peak.

    Returns
    -------
    ndarray of intp, shape (n_bins,)
        Index into ``peaks`` of each bin's basin; -1 for a bin no peak
        reaches through finite bins.
    """
    import heapq

    n_bins = env.n_bins
    finite = np.isfinite(firing_rate)
    pairs = env.edge_pairs
    pairs = pairs[finite[pairs[:, 0]] & finite[pairs[:, 1]]]
    both = np.concatenate([pairs, pairs[:, ::-1]])
    both = both[np.lexsort((both[:, 1], both[:, 0]))]
    starts = np.searchsorted(both[:, 0], np.arange(n_bins + 1)).tolist()
    neighbors = both[:, 1].tolist()
    rates = firing_rate.tolist()
    labels = [-1] * n_bins
    for k, peak in enumerate(peaks):
        labels[peak] = k
    heap = [(-rates[peak], peak) for peak in peaks]
    heapq.heapify(heap)
    while heap:
        _, b = heapq.heappop(heap)
        for neighbor in neighbors[starts[b] : starts[b + 1]]:
            if labels[neighbor] < 0:
                labels[neighbor] = labels[b]
                heapq.heappush(heap, (-rates[neighbor], neighbor))
    return np.asarray(labels, dtype=np.intp)


def _watershed_fields(
    env: Environment,
    firing_rate: NDArray[np.float64],
    kept: Sequence[tuple[int, NDArray[np.int64], float]],
    threshold: float,
    measures: NDArray[np.float64],
) -> list[NDArray[np.int64]]:
    """Fields of the kept ``"core"`` peaks, each inside its own watershed basin.

    Each field is the part of its peak's basin (:func:`_watershed_basins`)
    connected to the peak with a rate between ``threshold`` times that peak
    and the peak; a higher bin in the basin is a bump the search did not
    keep, and it stays outside every field. If a field is then no larger than the smallest field of its peak,
    the lowest such peak is dropped and the basins are flooded again, so its
    bins go to the neighboring fields. See :func:`detect_place_fields`, Notes.

    Parameters
    ----------
    env : Environment
        Environment of the map.
    firing_rate : ndarray of float64, shape (n_bins,)
        Rate map (Hz).
    kept : sequence of (int, ndarray of int64, float)
        Peak bin, ``"core"`` field and smallest field measure of each kept
        field, in order of decreasing peak.
    threshold : float
        Fraction of each field's own peak that bounds it.
    measures : ndarray of float64, shape (n_bins,)
        Measure of each bin.

    Returns
    -------
    list of ndarray of int64
        Bins of each field, in order of decreasing peak.
    """
    peaks = [peak for peak, _, _ in kept]
    minima = [minimum for _, _, minimum in kept]
    while True:
        basins = _watershed_basins(env, firing_rate, peaks)
        fields = [
            _seed_component(
                peak,
                (basins == k)
                & (firing_rate >= threshold * firing_rate[peak])
                & (firing_rate <= firing_rate[peak]),
                env,
            )
            for k, peak in enumerate(peaks)
        ]
        small = [
            k
            for k, (field_bins, minimum) in enumerate(zip(fields, minima, strict=True))
            if measures[field_bins].sum() <= minimum
        ]
        if not small:
            return fields
        del peaks[small[-1]], minima[small[-1]]


# Impulses are diffused this many map entries at a time. It bounds memory
# (32 MiB of float64 per array), not the computed value.
_IMPULSE_BLOCK_ELEMENTS = 1 << 22


def _smoothed_bin_measures(
    env: Environment,
    peaks: NDArray[np.int64],
    owners: NDArray[np.intp],
    observed: NDArray[np.bool_],
    bandwidth: float,
    threshold: float,
    measures: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Measure of the field one noisy bin at each peak makes after smoothing.

    For each peak, the connected observed bins of its map where one unit
    diffused from the peak stays at or above ``threshold`` times its maximum;
    the peak bin alone for an unsmoothed map. The impulses diffuse together,
    as the columns of a few batches. See :func:`detect_place_fields`, Notes.

    Parameters
    ----------
    env : Environment
        Environment of the maps.
    peaks : ndarray of int64, shape (n_peaks,)
        Peak bins.
    owners : ndarray of intp, shape (n_peaks,)
        Column of ``observed`` that holds each peak's map.
    observed : ndarray of bool, shape (n_bins, n_maps)
        Finite bins of each map.
    bandwidth : float
        Smoothing bandwidth of the maps; 0 for unsmoothed maps.
    threshold : float
        Fraction of the peak that bounds a field.
    measures : ndarray of float64, shape (n_bins,)
        Measure of each bin.

    Returns
    -------
    ndarray of float64, shape (n_peaks,)
        The smallest field measure for each peak.
    """
    if bandwidth == 0 or len(peaks) == 0:
        return measures[peaks]
    smallest = np.empty(len(peaks))
    block = max(1, _IMPULSE_BLOCK_ELEMENTS // env.n_bins)
    for start in range(0, len(peaks), block):
        stop = min(start + block, len(peaks))
        impulses = np.zeros((env.n_bins, stop - start))
        impulses[peaks[start:stop], np.arange(stop - start)] = 1.0
        spreads = np.asarray(env.diffuse(impulses, bandwidth=bandwidth, mode="density"))
        above = observed[:, owners[start:stop]] & (
            spreads >= threshold * spreads.max(axis=0)
        )
        for column, peak in enumerate(peaks[start:stop]):
            region = _seed_component(int(peak), above[:, column], env)
            smallest[start + column] = measures[region].sum()
    return smallest


@keyword_hint()
@reads_pynapple_samples
def place_cell_significance(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    backend: BackendType = "numpy",
    warn_on_drop: bool = False,
    dtype: type[np.float32] | type[np.float64] = np.float64,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> PlaceCellSignificanceResult:
    """Test place tuning against circular spike-time shifts.

    The test statistic is Skaggs spatial information (bits/spike) of each
    unit's rate map. The trajectory is binned once; each shifted train is
    recounted and smoothed with exactly the observed map's analyzed intervals,
    occupancy and bandwidth. Shuffles preserve counts on the compressed
    analyzed clock and never enter recording gaps.

    A significant test is not a place verdict on its own: in a short
    recording about alpha of untuned units pass it by chance, with almost no
    information beyond what an untuned cell shows on the same map. A place
    cell also needs bias-corrected information,
    ``spatial_information() - spatial_information_bias()`` of
    :func:`compute_spatial_rates` with the same settings, of at least 0.5
    bits/spike. ``result.classify()``, ``is_place_cell(...,
    criterion="shuffle")`` and the labels of
    :func:`spatial_cell_type_significance` apply both.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure, for example
        one built with ``Environment.from_samples()``.
    spike_times : sequence of arrays or 2D array
        Spike times for each neuron. Accepted formats:

        - List/tuple of 1D arrays: ``[spikes_0, spikes_1, ...]`` (canonical)
        - 2D array with NaN padding: shape ``(n_neurons, max_spikes)``
        - 1D array (single neuron): wrapped in list automatically
        - A pynapple ``TsGroup`` (or a group exposing an ``.index`` of unit
          labels): its index becomes the result's ``unit_ids``

        A ``unit_ids`` passed with a labelled group must equal the group's
        index.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with the raw coordinates.

    positions : ndarray, shape (n_samples, n_dims)
        Raw animal coordinates aligned with times, in environment units.

    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Rate-map estimator; see ``compute_spatial_rate()``. ``"glm"`` raises:
        its pooled REML penalty couples units, so a shifted single-unit fit
        would not be the observed model.
    bandwidth : float | None, default=None
        Smoothing bandwidth in environment units; ``None`` is 0 (no
        smoothing) for ``method="binned"``, and 5 cm in declared metric units
        (0.05 m / 50 mm) or 2.5 median graph-edge lengths for unspecified or
        custom units for ``"diffusion_kde"`` and ``"gaussian_kde"``; the
        units-free default warns with its resolved value. Larger values smooth
        more; an explicit value smooths ``"binned"`` maps too.
        Mutually exclusive with ``method="glm"``.
    min_occupancy : float, default=0.0
        Minimum occupancy (seconds) for a bin to be included; ``0.0`` masks
        nothing (unvisited bins are NaN regardless). ``method="glm"`` fits
        every bin and accepts only ``0.0``.
    penalty, rank : None, default=None
        GLM-only settings; ``method="glm"`` is not supported here, so any
        value raises ``ValueError``.
    pooled : bool, default=True
        GLM-only setting; ``pooled=False`` raises ``ValueError``.
    speed : ndarray, shape (n_samples,), optional
        Precomputed instantaneous speed at each trajectory sample (physical
        units / second). Only used when ``min_speed`` is set; auto-derived from
        the trajectory when ``None``. See ``min_speed``. A wrong-length array
        raises ``ValueError``.
    min_speed : float, optional
        Minimum speed threshold (physical units / second). When set, low-speed
        periods are excluded from BOTH the (shared) occupancy denominator AND
        every per-neuron spike numerator using ONE shared per-interval speed
        gate, so firing rates stay correct. When ``None`` (default) NO speed
        filtering is applied. Auto-speed is :func:`neurospatial.behavior.compute_speed`, the
        magnitude of the 0.1 s Gaussian-smoothed velocity; pass an explicit
        ``speed`` for geodesic / linearized-track environments.

        **Choosing it.** Without a gate, time and spikes at rest (reward
        sites, track ends) enter the map. On a simulated track where the
        animal rested 27% of the time and fired 3 Hz at rest, spatial
        information was 0.85-0.94 bits/spike without a gate and 1.55-1.84
        with ``min_speed=4`` cm/s, and the rest firing made extra fields at
        the track ends. No speed suits every recording: place maps are
        usually built from running, so pick the threshold from your animal's
        speed distribution, between the resting and running modes.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The interval from sample ``k - 1`` to ``k``
        is excluded from occupancy and its spikes are not counted, as for a gap
        longer than ``max_gap``, and the speed derived for ``min_speed`` is not
        smoothed across it.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording. Intervals outside it are
        excluded from occupancy (and their spikes are not counted). ``None``
        (default) assumes spikes were recorded whenever position was; this is an
        assumption, not something the function checks. Pass it when tracking
        started before, or continued after, the spike recording. The result
        records the window applied (``result.spike_window``) and whether it was
        assumed (``result.spike_window_assumed``).
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Backend for rate-map smoothing (``"auto"`` uses JAX when installed);
        binning always uses NumPy.
    warn_on_drop : bool, default=False
        Report dropped spikes in the observed fit when True. Null-fit warnings are suppressed.
    dtype : {np.float32, np.float64}, default=np.float64
        Storage dtype of the returned ``(n_units, n_bins)`` rate-map array.
        ``np.float32`` halves the stored rate-map array. The rate computation
        is still performed in float64 and only the final result is cast, so
        float32 values match float64 within float32 tolerance. To halve the
        decoding working set as well, pass ``dtype=np.float32`` to
        ``decode_session`` / ``decode_session_summary``. Any other dtype
        raises ``ValueError``.
    unit_ids : ndarray or sequence, optional
        Per-unit identity labels (integers or strings), one per neuron in
        the same order as ``spike_times``. Stored on the result's
        ``unit_ids`` field and stamped onto each child's ``unit_id`` when
        indexing/iterating. Defaults to the labels of a labelled
        ``spike_times`` group, else ``np.arange(n_neurons)``. With a labelled
        group, a ``unit_ids`` that differs from the group's index raises
        ``ValueError`` (relabel the group itself instead). A wrong-length
        value or a repeated label raises ``ValueError``.
    n_shuffles : int, default=1000
        Positive number of null map recomputes.
    min_shift : float, default=20.0
        Minimum shift in either direction, seconds of analyzed time.
    rng : numpy.random.Generator, int or None, default=None
        Integer seeds give stable per-unit shifts regardless of population order.

    Returns
    -------
    PlaceCellSignificanceResult
        Per-unit observed spatial information (bits/spike), null scores,
        p-value, null mean and standard deviation, and z-score, in input
        order with the units' ``unit_ids``. ``summary_table()`` gives one row
        per unit, ``classify()`` the place screen, ``result[i]`` or
        ``result.sel(unit_id=label)`` one unit's
        :class:`~neurospatial.stats.ShuffleTestResult` and ``plot(i)`` its
        null distribution. An undefined observed score (a silent unit) or a
        null with no finite score gives a NaN p-value.

    Raises
    ------
    ValueError
        For invalid inputs/settings, duplicate/conflicting unit labels,
        insufficient analyzed time, or unsupported method="glm".

    See Also
    --------
    spatial_cell_type_significance : This test with grid and border scores
        recomputed from the same shifted trains, and cell-type labels.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import place_cell_significance
    >>> rng = np.random.default_rng(7)
    >>> times = np.arange(0.0, 60.0, 0.1)
    >>> xx, yy = np.meshgrid(np.arange(0, 101, 5), np.arange(0, 101, 5))
    >>> env = Environment.from_samples(
    ...     np.c_[xx.ravel(), yy.ravel()], bin_size=5.0, units="cm"
    ... )
    >>> positions = np.c_[50 + 20 * np.sin(times / 3), 50 + 20 * np.cos(times / 4)]
    >>> headings = np.sin(times / 5)
    >>> objects = np.array([[50.0, 50.0]])
    >>> trains = [np.sort(rng.uniform(0, 59.9, 40))]
    >>> result = place_cell_significance(
    ...     env, trains, times, positions, n_shuffles=20, rng=0
    ... )
    >>> result.unit_ids.tolist()
    [0]
    >>> result.n_shuffles, result.statistic_units
    (20, 'bits/spike')
    >>> list(result.summary_table().columns[:3])
    ['spatial_information', 'spatial_information_bias', 'p_value']
    """
    shuffled = _spatial_shuffle(
        env,
        spike_times,
        times,
        positions,
        statistics=("spatial_information",),
        call="place_cell_significance",
        boundary_source="physical",
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        backend=backend,
        warn_on_drop=warn_on_drop,
        dtype=dtype,
        unit_ids=unit_ids,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        rng=rng,
    )
    from neurospatial.encoding._metrics import _information_bias_terms

    return PlaceCellSignificanceResult(
        observed=shuffled.observed[:, 0],
        null_scores=shuffled.null[:, :, 0].T,
        unit_ids=shuffled.unit_ids,
        spatial_information_bias=_information_bias_terms(shuffled.reference).bias,
        analysis_settings={
            **(shuffled.reference.analysis_settings or {}),
            "n_shuffles": n_shuffles,
            "min_shift": float(min_shift),
            "shuffle_type": "circular_time_shift",
        },
    )


def _holm_adjusted(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values within each row.

    Parameters
    ----------
    p_values : ndarray, shape (n_units, n_tests)
        NaN marks a test outside the unit's family.

    Returns
    -------
    ndarray, shape (n_units, n_tests)
        With the m finite p-values of a row sorted, p_(1) <= ... <= p_(m),
        the adjusted value of p_(j) is ``max_{i <= j} min(1, (m - i + 1)
        p_(i))``; a test passes Holm's procedure at level alpha exactly when
        its adjusted value is below alpha. NaN where the input is NaN.
    """
    order = np.argsort(p_values, axis=1)  # NaN sorts last
    ranked = np.take_along_axis(p_values, order, axis=1)
    m = np.isfinite(p_values).sum(axis=1, keepdims=True)
    scaled = ranked * (m - np.arange(p_values.shape[1]))
    adjusted = np.minimum(np.fmax.accumulate(scaled, axis=1), 1.0)
    adjusted[np.isnan(ranked)] = np.nan
    result: NDArray[np.float64] = np.empty_like(adjusted)
    np.put_along_axis(result, order, adjusted, axis=1)
    return result


def _benjamini_hochberg_adjusted(
    p_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values over every finite entry.

    Parameters
    ----------
    p_values : ndarray, any shape
        NaN marks a test outside the family.

    Returns
    -------
    ndarray, same shape
        With the M finite p-values sorted, p_(1) <= ... <= p_(M), the
        adjusted value of p_(j) is ``min_{i >= j} min(1, M p_(i) / i)``, as
        :func:`scipy.stats.false_discovery_control` computes it. NaN where
        the input is NaN.
    """
    result = np.full(p_values.shape, np.nan)
    finite = np.isfinite(p_values)
    values = p_values[finite]
    order = np.argsort(values)
    ranked = values[order] * values.size / np.arange(1, values.size + 1)
    adjusted = np.empty(values.size)
    adjusted[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    result[finite] = adjusted
    return result


def _smallest_null_count(passes: Callable[[float], bool], estimate: float) -> int:
    """Smallest n whose floor p-value ``1 / (1 + n)`` passes a cutoff.

    Parameters
    ----------
    passes : callable
        Whether a p-value passes, evaluated with the same floating-point
        expression as the decision it predicts (for example
        ``lambda p: p * m < alpha`` for Holm's first step), so rounding at the
        boundary cannot name a count that still fails.
    estimate : float
        Approximate answer, such as ``m / alpha``; the search starts here.

    Returns
    -------
    int
        The number of finite null scores needed.
    """
    n = max(int(np.ceil(estimate)) - 1, 0)
    while not passes(1.0 / (1.0 + n)):
        n += 1
    while n > 0 and passes(1.0 / n):
        n -= 1
    return n


def _adjusted_p_values(
    p_values: NDArray[np.float64], correction: str | None
) -> NDArray[np.float64]:
    """P-values adjusted for the tests of one result; NaN outside the family."""
    if correction is None:
        return np.array(p_values, dtype=np.float64, copy=True)
    if correction == "holm":
        return _holm_adjusted(p_values)
    if correction == "fdr_bh":
        return _benjamini_hochberg_adjusted(p_values)
    raise ValueError(
        _format_error(
            f"correction={correction!r} is not a multiple-comparison correction.",
            why="Why: the correction decides which error rate the labels control",
            fix=(
                "pass correction='holm' (each unit's chance of any false label "
                "at most alpha), correction='fdr_bh' (expected fraction of "
                "false labels across the units at most alpha) or None (each "
                "test at alpha)"
            ),
        )
    )


@dataclass(frozen=True, repr=False, eq=False)
class CellTypeSignificanceResult(UnitSelectionMixin, ResultMixin):
    """Circular-shift tests of place, grid and border tuning for each unit.

    Returned by :func:`spatial_cell_type_significance`. Every statistic was
    recomputed from the same shifted spike trains, so the three tests of one
    unit share one null.

    Parameters
    ----------
    observed_scores : ndarray, shape (n_units, 3)
        Observed spatial information (bits/spike), grid score and border
        score, in the order of :attr:`statistics`. A column is NaN when its
        statistic cannot be scored on the environment (see
        ``unavailable_metrics``).
    null_scores : ndarray, shape (n_units, n_shuffles, 3)
        Scores of the circularly shifted spike trains.
    unit_ids : ndarray, shape (n_units,)
        Unit labels in input order.
    min_shift : float
        Smallest shift, in seconds of analyzed time.
    method : str
        Rate-map estimator.
    bandwidth : float
        Smoothing bandwidth in environment units.
    boundary_source : {'physical', 'analysis'}
        Wall geometry of the border score.
    grid_spacing : ndarray, shape (n_units,)
        Spacing of the hexagonal first ring of each observed map's
        autocorrelogram, in environment units, as
        :meth:`SpatialRateResult.grid_properties` reads it. NaN when the map
        has no such ring or the grid score is unscored; such a unit is never
        labeled grid.
    spatial_information_bias : ndarray, shape (n_units,)
        Information an untuned cell would show on each observed map
        (bits/spike), as :meth:`SpatialRatesResult.spatial_information_bias`
        computes it. NaN for a silent unit or one with fewer spikes than the
        map has independent bins; such a unit is never labeled place.
    unavailable_metrics : dict of str to str
        Why each unscored statistic has no score.
    inapplicable_metrics : tuple of str
        Unscored statistics that do not apply to the environment: grid and
        border cells are defined on 2-D maps.

    Attributes
    ----------
    p_values : ndarray, shape (n_units, 3)
        Upper-tail Monte Carlo p-values (k + 1) / (n + 1), where k counts
        finite null scores at least the observed score and n is the number of
        finite null scores; NaN null scores (a shifted map with no defined
        score) are left out. NaN when the observed score is undefined (a
        silent unit, or a border map with no field of the minimum size), the
        statistic is unscored, or no null score is finite. With n finite null
        scores the smallest p-value is 1 / (n + 1), so a test with
        1 / (n + 1) >= alpha cannot reject and is undetermined; under a
        correction, so is a test that fails the corrected cutoff even with
        every test at its smallest p-value.
    n_shuffles : int
        Number of shifted trains per unit.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding.spatial import CellTypeSignificanceResult
    >>> result = CellTypeSignificanceResult(
    ...     observed_scores=np.array([[1.2, 0.9, 0.1]]),
    ...     null_scores=np.zeros((1, 19, 3)),
    ...     unit_ids=np.array([7]),
    ...     min_shift=20.0,
    ...     method="diffusion_kde",
    ...     bandwidth=5.0,
    ...     boundary_source="physical",
    ...     grid_spacing=np.array([40.0]),
    ...     spatial_information_bias=np.array([0.05]),
    ... )
    >>> result.p_values.tolist()
    [[0.05, 0.05, 0.05]]
    >>> result.sel(unit_id=7)["grid_score"].p_value
    0.05
    """

    observed_scores: NDArray[np.float64]
    null_scores: NDArray[np.float64]
    unit_ids: NDArray[Any]
    min_shift: float
    method: str
    bandwidth: float
    boundary_source: str
    grid_spacing: NDArray[np.float64]
    spatial_information_bias: NDArray[np.float64]
    unavailable_metrics: Mapping[str, str] = field(default_factory=dict)
    inapplicable_metrics: tuple[str, ...] = ()
    p_values: NDArray[np.float64] = field(init=False)

    statistics: ClassVar[tuple[str, ...]] = _CELL_TYPE_STATISTICS

    def __post_init__(self) -> None:
        from neurospatial.encoding._significance import shuffle_pvalues

        observed = np.asarray(self.observed_scores, dtype=np.float64)
        null = np.asarray(self.null_scores, dtype=np.float64)
        spacing = np.asarray(self.grid_spacing, dtype=np.float64)
        bias = np.asarray(self.spatial_information_bias, dtype=np.float64)
        n_stats = len(self.statistics)
        if (
            observed.ndim != 2
            or observed.shape[1] != n_stats
            or null.ndim != 3
            or null.shape[0] != observed.shape[0]
            or null.shape[2] != n_stats
            or len(self.unit_ids) != observed.shape[0]
            or spacing.shape != (observed.shape[0],)
            or bias.shape != (observed.shape[0],)
        ):
            msg = (
                "observed_scores must have shape (n_units, 3), null_scores "
                "(n_units, n_shuffles, 3), and unit_ids, grid_spacing and "
                "spatial_information_bias (n_units,); got "
                f"{observed.shape}, {null.shape}, {np.shape(self.unit_ids)}, "
                f"{spacing.shape} and {bias.shape}."
            )
            raise ValueError(msg)
        object.__setattr__(self, "grid_spacing", spacing)
        object.__setattr__(self, "spatial_information_bias", bias)
        object.__setattr__(self, "observed_scores", observed)
        object.__setattr__(self, "null_scores", null)
        object.__setattr__(self, "unit_ids", np.asarray(self.unit_ids))
        object.__setattr__(
            self,
            "unavailable_metrics",
            MappingProxyType(dict(self.unavailable_metrics)),
        )
        object.__setattr__(
            self, "p_values", shuffle_pvalues(observed, null.transpose(1, 0, 2))
        )
        self._own_arrays()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["unavailable_metrics"] = dict(self.unavailable_metrics)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        state["unavailable_metrics"] = MappingProxyType(state["unavailable_metrics"])
        super().__setstate__(state)

    @property
    def n_shuffles(self) -> int:
        """Number of shifted trains per unit."""
        return int(self.null_scores.shape[1])

    @property
    def n_units(self) -> int:
        """Number of units."""
        return len(self.unit_ids)

    def __len__(self) -> int:
        return len(self.unit_ids)

    def _summary_for_repr(self) -> dict[str, Any]:
        """Unit and shuffle counts; the label counts depend on ``alpha``.

        ``summary()`` counts each label at the default ``alpha``; the repr
        shows only what is an attribute of the result.
        """
        return {"n_units": self.n_units, "n_shuffles": self.n_shuffles}

    def __getitem__(self, idx: int) -> dict[str, ShuffleTestResult]:
        """One unit's test of each statistic, keyed by statistic name."""
        from neurospatial.encoding._significance import shuffle_test_result

        i = self._unit_position(idx)
        return {
            name: shuffle_test_result(
                float(self.observed_scores[i, k]),
                self.null_scores[i, :, k],
                float(self.p_values[i, k]),
                statistic=name,
                statistic_units=_CELL_TYPE_STATISTIC_UNITS[name],
            )
            for k, name in enumerate(self.statistics)
        }

    def _adjusted(
        self, level: float, correction: str | None
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Adjusted p-values and the tests that cannot reject at ``level``.

        The smallest p-value of n finite null scores is its floor
        1 / (n + 1); when the floor is not below ``level`` the test cannot
        reject, whatever the data, and is left out of the correction's family
        along with unscored tests.

        A test in the family can still be unable to pass the corrected
        cutoff. Holm and Benjamini-Hochberg are monotone: lowering any
        p-value never removes a rejection, because both act on the sorted
        p-values, which can only fall, through a running maximum (Holm) or a
        step-up search (BH). Every p-value is at least its floor, so the most
        favorable data put every test of the family at its floor; a test the
        correction rejects for no data set fails there, and is also a test
        that cannot reject. For Holm, a unit whose m tests share one floor
        needs ``m / (n + 1) < level`` at the first step. For BH, the largest
        of M p-values passes when ``M p / M < level``, so a floor below
        ``level`` can always pass and BH adds no such test.
        """
        floor = 1.0 / (1.0 + self._n_finite_null())
        cannot_reject = np.isfinite(self.observed_scores) & (floor >= level)
        family = np.where(cannot_reject, np.nan, self.p_values)
        in_family = np.isfinite(family)
        best_case = _adjusted_p_values(np.where(in_family, floor, np.nan), correction)
        cannot_reject |= in_family & ~(best_case < level)
        return _adjusted_p_values(family, correction), cannot_reject

    def _outcomes(
        self,
        level: float,
        min_info: float,
        correction: Literal["holm", "fdr_bh"] | None,
    ) -> dict[str, NDArray[np.float64]]:
        refuse_none_thresholds(alpha=level, min_spatial_info=min_info)
        adjusted, cannot_reject = self._adjusted(level, correction)
        with np.errstate(invalid="ignore"):
            passed = adjusted < level
        if correction is not None:
            self._warn_resolution_limit(
                adjusted, passed, cannot_reject, level, correction
            )
        from neurospatial.encoding._metrics import _warn_unknown_bias

        debiased = self.observed_scores[:, 0] - self.spatial_information_bias
        _warn_unknown_bias(passed[:, 0], debiased, kind="place")
        # Significant information with an unknown bias is undetermined below.
        unknown_bias = passed[:, 0] & np.isnan(debiased)
        met = np.ones_like(passed)
        # Significant information is a place cell only beyond the untuned bias.
        met[:, 0] = debiased >= min_info
        # A significant grid score is a grid cell only with a lattice to show.
        met[:, 1] = np.isfinite(self.grid_spacing)
        undetermined = cannot_reject & met
        undetermined_bias = np.zeros_like(undetermined)
        undetermined_bias[:, 0] = unknown_bias
        outcomes = np.where(
            undetermined | undetermined_bias, np.nan, (passed & met).astype(np.float64)
        )
        for k, name in enumerate(self.statistics):
            if (
                name in self.unavailable_metrics
                and name not in self.inapplicable_metrics
            ):
                outcomes[:, k] = np.nan
        return {name: outcomes[:, k] for k, name in enumerate(self.statistics)}

    def _warn_resolution_limit(
        self,
        adjusted: NDArray[np.float64],
        passed: NDArray[np.bool_],
        cannot_reject: NDArray[np.bool_],
        level: float,
        correction: str,
    ) -> None:
        """Warn when tests at the smallest attainable p-value fail a correction.

        Such a test is as extreme as its shuffles can show, so the number of
        shuffles, not the data, decided it. Tests that cannot reject for any
        data are undetermined and warned about separately. Holm passes the smallest of a
        unit's m p-values when ``m p < alpha``, and Benjamini-Hochberg passes
        k equal p-values of M tests when ``M p / k < alpha``; the message
        names the smallest n whose floor p-value ``1 / (n + 1)`` passes,
        evaluated as the adjusted p-values are.
        """

        n_finite = self._n_finite_null()
        at_floor = np.isfinite(adjusted) & (self.p_values <= 1.0 / (1.0 + n_finite))
        limited = at_floor & ~passed & ~cannot_reject
        if not limited.any():
            return
        if correction == "holm":
            m = int(np.isfinite(adjusted).sum(axis=1)[limited.any(axis=1)].max())
            needed = _smallest_null_count(lambda p: p * m < level, m / level)
        else:
            total, tied = int(np.isfinite(adjusted).sum()), int(at_floor.sum())
            needed = _smallest_null_count(
                lambda p: p * total / tied < level, total / (tied * level)
            )
        warnings.warn(
            f"{int(limited.sum())} tests reached the smallest p-value their "
            f"shuffles allow, 1 / (n + 1), but fail correction={correction!r} "
            f"at alpha={level}, so the number of shuffles, not the data, "
            "decided them.\n"
            "Why: a correction compares p-values with a fraction of alpha, and "
            "with n finite null scores a Monte Carlo p-value cannot go below "
            "1 / (n + 1).\n"
            "Fix: rerun with more shuffles, so that each test has at least "
            f"{needed} finite null scores (n_shuffles >= {needed} when every "
            "shifted map has a score).",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )

    def _n_finite_null(self) -> NDArray[np.int64]:
        """Null scores behind each p-value, shape (n_units, 3).

        The p-values count the defined (non-NaN) null scores; these
        statistics are bounded, so a defined score is finite.
        """
        counts: NDArray[np.int64] = (~np.isnan(self.null_scores)).sum(axis=1)
        return counts

    def label_cell_types(
        self,
        *,
        alpha: float = 0.05,
        min_spatial_info: float = 0.5,
        correction: Literal["holm", "fdr_bh"] | None = None,
    ) -> NDArray[np.str_]:
        """Label each unit from its three shuffle tests.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level; a test passes when ``p_value < alpha``.
        min_spatial_info : float, default=0.5
            Inclusive cutoff on bias-corrected spatial information
            (``spatial_information - spatial_information_bias``, bits/spike)
            that a significant unit must also reach to be ``"place"``: the
            cutoff of the fixed-threshold screen
            (:meth:`SpatialRatesResult.label_cell_types`), a common screening
            convention, not derived or calibrated.
        correction : {"holm", "fdr_bh"} or None, default=None
            Multiple-comparison correction across the shuffle tests; a test
            passes when its adjusted p-value is below ``alpha``. Only tests
            that can reject are in the family: unscored tests and tests with
            too few finite null scores are left out.

            - None: each test at ``alpha``, as most studies report. Each
              unit is tested three times, so an untuned unit receives some
              label more often than ``alpha`` (measured rates in the Notes of
              :func:`spatial_cell_type_significance`).
            - ``"holm"``: Holm-Bonferroni step-down across each unit's own
              tests. Each unit's chance of any false label is at most
              ``alpha``, whatever the dependence between its tests. Use it
              when each unit's label must stand on its own.
            - ``"fdr_bh"``: Benjamini-Hochberg across every test of every
              unit in this result. The expected fraction of false passes
              among all passing tests is at most ``alpha`` when the tests are
              independent or positively dependent. Use it to count cell types
              in a population. A unit's verdict depends on the other units
              in the result, so a unit tested alone and in a population can
              be labeled differently.

        Returns
        -------
        ndarray of str, shape (n_units,)
            ``"grid"``, ``"border"`` or ``"grid+border"`` from the grid and
            border tests, each judged by its own p-value, and a grid label
            also needs a hexagonal first ring (finite :attr:`grid_spacing`);
            ``"place"`` for a
            unit whose spatial information is significant, whose
            bias-corrected information reaches ``min_spatial_info``, and that
            is neither a grid nor a border cell; ``"unclassified"`` for none;
            ``"undetermined"`` when a grid or border test applies to the
            environment but its score could not be computed, or when a test
            has too few finite null scores to pass and its other requirement
            holds; ``summary_table()`` gives the reason (``attrs["unavailable_metrics"]``
            and the ``<statistic>_n_finite_null`` columns). No warning is
            raised. Too few means
            ``1 / (n_finite + 1) >= alpha``, or, with a ``correction``, that
            the test fails the correction even with every test at its
            smallest p-value ``1 / (n_finite + 1)``: under Holm, a unit's
            ``m`` tests with equal ``n_finite`` need
            ``m / (n_finite + 1) < alpha``. On a 1-D environment
            grid and border do not apply, so a significant cell is
            ``"place"``. ``summary_table()["is_place"]`` is True for a unit
            that passes the place test and was not shown to be a grid or
            border cell, even when a grid or border test is undetermined;
            without a border test border cells are not excluded, so a True
            is then a place-or-border candidate.
        """
        return _cell_type_labels(self._outcomes(alpha, min_spatial_info, correction))

    def summary_table(
        self,
        *,
        alpha: float = 0.05,
        min_spatial_info: float = 0.5,
        correction: Literal["holm", "fdr_bh"] | None = None,
    ) -> pd.DataFrame:
        """Per-unit observed scores, null percentiles, p-values and labels.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for ``passes_*``, ``is_place`` and
            ``cell_type``, and the null percentile reported, ``100 * (1 - alpha)``.
        min_spatial_info : float, default=0.5
            Bias-corrected information cutoff of the place label
            (bits/spike); see :meth:`label_cell_types`.
        correction : {"holm", "fdr_bh"} or None, default=None
            Multiple-comparison correction of ``passes_*``, ``is_place`` and
            ``cell_type``; see :meth:`label_cell_types`.

        Returns
        -------
        pandas.DataFrame
            One row per unit, indexed by ``unit_id``: for each statistic in
            :attr:`statistics` the observed score, then
            ``spatial_information_bias``, ``<statistic>_p_value``, with a
            ``correction`` its ``<statistic>_p_adjusted`` (NaN for a test
            outside the family), ``<statistic>_null_percentile`` (the
            ``100 * (1 - alpha)``
            percentile of the unit's finite null scores) and
            ``<statistic>_n_finite_null`` (how many null scores were finite and
            entered the p-value), then ``passes_<statistic>``
            (adjusted ``p_value < alpha``, with the place and grid
            requirements of
            :meth:`label_cell_types`; ``<NA>`` when the statistic applies but
            could not be computed or the test cannot reach ``alpha``),
            ``is_place`` (``passes_spatial_information`` and not shown to be
            a grid or border cell; an undetermined grid or border test does
            not block it, so without a border test a True is a
            place-or-border candidate; ``<NA>`` when the place test is
            undetermined, except False for a known grid or border cell; it
            can disagree with :meth:`PlaceCellSignificanceResult.classify`,
            which tests information alone) and ``cell_type`` from
            :meth:`label_cell_types`. ``df.attrs`` records ``alpha``,
            ``correction``, ``classification_thresholds``, ``null_percentile``,
            ``n_shuffles``, ``min_shift``,
            ``shuffle_type``, ``method``, ``bandwidth``, ``boundary_source``,
            ``units`` and any ``unavailable_metrics``.
        """
        import pandas as pd

        # Refuse None before the percentile and the correction use alpha.
        refuse_none_thresholds(alpha=alpha, min_spatial_info=min_spatial_info)
        level = alpha
        percentile = 100.0 * (1.0 - level)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN nulls
            thresholds = np.nanpercentile(self.null_scores, percentile, axis=1)
        columns: dict[str, Any] = {}
        for k, name in enumerate(self.statistics):
            columns[name] = self.observed_scores[:, k].copy()
        columns["spatial_information_bias"] = self.spatial_information_bias.copy()
        for k, name in enumerate(self.statistics):
            columns[f"{name}_p_value"] = self.p_values[:, k].copy()
        if correction is not None:
            adjusted, _ = self._adjusted(level, correction)
            for k, name in enumerate(self.statistics):
                columns[f"{name}_p_adjusted"] = adjusted[:, k]
        for k, name in enumerate(self.statistics):
            columns[f"{name}_null_percentile"] = thresholds[:, k]
        n_finite = self._n_finite_null()
        for k, name in enumerate(self.statistics):
            columns[f"{name}_n_finite_null"] = n_finite[:, k]
        min_info = min_spatial_info
        columns.update(_cell_type_columns(self._outcomes(level, min_info, correction)))
        df = pd.DataFrame(
            columns, index=pd.Index(self.unit_ids.tolist(), name="unit_id")
        )
        df.attrs.update(
            alpha=level,
            correction=correction,
            classification_thresholds={"min_spatial_info": min_info},
            null_percentile=percentile,
            n_shuffles=self.n_shuffles,
            min_shift=self.min_shift,
            shuffle_type="circular_time_shift",
            method=self.method,
            bandwidth=self.bandwidth,
            boundary_source=self.boundary_source,
            units={
                "spatial_information": "bits/spike",
                "spatial_information_bias": "bits/spike",
                "min_shift": "s",
            },
        )
        if self.unavailable_metrics:
            df.attrs["unavailable_metrics"] = dict(self.unavailable_metrics)
        return df

    def summary(self) -> dict[str, Any]:
        """Unit and shuffle counts and the number of units with each label.

        The labels are :meth:`label_cell_types` at its defaults; call it with
        your ``alpha`` and ``correction`` for others.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            labels = self.label_cell_types()
        counts = {
            f"n_{label.replace('+', '_and_')}": int(np.sum(labels == label))
            for label in (
                "place",
                "grid",
                "border",
                "grid+border",
                "unclassified",
                "undetermined",
            )
        }
        return {"n_units": len(self), "n_shuffles": self.n_shuffles, **counts}


@_copy_analysis_settings
@dataclass(frozen=True, repr=False, eq=False)
class PlaceCellSignificanceResult(UnitSelectionMixin, ResultMixin):
    """Circular-shift tests of each unit's spatial information.

    Returned by :func:`place_cell_significance`. Each unit's spike train was
    shifted circularly against the behavior ``n_shuffles`` times, and the
    Skaggs spatial information (bits/spike) of each shifted train's rate map
    is that unit's null distribution.

    Parameters
    ----------
    observed : ndarray, shape (n_units,)
        Spatial information of each unit's rate map, bits/spike; NaN for a
        silent unit.
    null_scores : ndarray, shape (n_units, n_shuffles)
        Spatial information of the shifted trains' maps, bits/spike; NaN for a
        shifted map without a defined score.
    unit_ids : ndarray, shape (n_units,)
        Unit labels in input order.
    spatial_information_bias : ndarray, shape (n_units,)
        Information an untuned cell would show on each observed map
        (bits/spike), as :meth:`SpatialRatesResult.spatial_information_bias`
        computes it; NaN when unknown (fewer spikes than independent bins).
    analysis_settings : dict, optional
        The rate-map settings of the observed maps, with ``n_shuffles``,
        ``min_shift`` and ``shuffle_type``.

    Attributes
    ----------
    statistic : str
        ``"spatial_information"``.
    statistic_units : str
        ``"bits/spike"``.
    p_value : ndarray, shape (n_units,)
        Upper-tail Monte Carlo p-value (k + 1) / (n + 1), where k counts the
        defined null scores at least the observed score and n is the number
        of defined null scores. NaN when the observed score is undefined or
        no null score is defined.
    null_mean, null_std : ndarray, shape (n_units,)
        Mean and (population) standard deviation of each unit's defined null
        scores, bits/spike.
    z_score : ndarray, shape (n_units,)
        ``(observed - null_mean) / null_std``, as
        :attr:`~neurospatial.stats.ShuffleTestResult.z_score`; NaN when the
        null has no spread.
    n_units, n_shuffles : int
        Number of units, and of shifted trains per unit.

    Notes
    -----
    ``result[i]`` is unit ``i``'s :class:`~neurospatial.stats.ShuffleTestResult`
    and ``result.sel(unit_id=label)`` selects it by label.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.encoding import PlaceCellSignificanceResult
    >>> result = PlaceCellSignificanceResult(
    ...     observed=np.array([1.2, 0.1]),
    ...     null_scores=np.array([[0.1, 0.2, 0.3, 0.2], [0.1, 0.2, 0.3, 0.2]]),
    ...     unit_ids=np.array([7, 9]),
    ...     spatial_information_bias=np.array([0.05, 0.05]),
    ... )
    >>> result.p_value.tolist()
    [0.2, 1.0]
    >>> result.sel(unit_id=7).statistic_units
    'bits/spike'
    """

    observed: NDArray[np.float64]
    null_scores: NDArray[np.float64]
    unit_ids: NDArray[Any]
    spatial_information_bias: NDArray[np.float64]
    analysis_settings: dict[str, Any] | None = None
    p_value: NDArray[np.float64] = field(init=False)

    statistic: ClassVar[str] = "spatial_information"
    statistic_units: ClassVar[str] = "bits/spike"

    def __post_init__(self) -> None:
        from neurospatial.encoding._significance import shuffle_pvalues

        observed = np.asarray(self.observed, dtype=np.float64)
        null = np.asarray(self.null_scores, dtype=np.float64)
        bias = np.asarray(self.spatial_information_bias, dtype=np.float64)
        unit_ids = np.asarray(self.unit_ids)
        n_units = observed.shape[0] if observed.ndim == 1 else -1
        if (
            observed.ndim != 1
            or null.ndim != 2
            or null.shape[0] != n_units
            or unit_ids.shape != (n_units,)
            or bias.shape != (n_units,)
        ):
            raise ValueError(
                _format_error(
                    "PlaceCellSignificanceResult needs observed, unit_ids and "
                    "spatial_information_bias of shape (n_units,) and "
                    "null_scores of shape (n_units, n_shuffles); got "
                    f"{observed.shape}, {unit_ids.shape}, {bias.shape} and "
                    f"{null.shape}.",
                    why="each unit's observed score is tested against its own "
                    "row of null scores",
                    fix="build the result with place_cell_significance(...)",
                )
            )
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "null_scores", null)
        object.__setattr__(self, "spatial_information_bias", bias)
        object.__setattr__(self, "unit_ids", unit_ids)
        p_value = shuffle_pvalues(observed[:, None], null.T[:, :, None])[:, 0]
        object.__setattr__(self, "p_value", p_value)
        self._own_arrays()

    @property
    def n_units(self) -> int:
        """Number of units."""
        return len(self.unit_ids)

    @property
    def n_shuffles(self) -> int:
        """Number of shifted trains per unit."""
        return int(self.null_scores.shape[1])

    @property
    def null_mean(self) -> NDArray[np.float64]:
        """Mean of each unit's defined null scores (bits/spike); NaN without any."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN nulls
            mean: NDArray[np.float64] = np.nanmean(self.null_scores, axis=1)
        return mean

    @property
    def null_std(self) -> NDArray[np.float64]:
        """Standard deviation of each unit's defined null scores (bits/spike)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN nulls
            std: NDArray[np.float64] = np.nanstd(self.null_scores, axis=1)
        return std

    @property
    def z_score(self) -> NDArray[np.float64]:
        """Each unit's observed score in standard deviations of its null.

        As :attr:`~neurospatial.stats.ShuffleTestResult.z_score`: NaN with
        fewer than two defined null scores or no spread.
        """
        from neurospatial.stats.shuffle import _shuffle_zscore

        return np.array(
            [
                _shuffle_zscore(float(score), null)
                for score, null in zip(self.observed, self.null_scores, strict=True)
            ]
        )

    def __len__(self) -> int:
        return len(self.unit_ids)

    def __iter__(self) -> Iterator[ShuffleTestResult]:
        """Each unit's test, in ``unit_ids`` order."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> ShuffleTestResult:
        """Unit ``idx``'s test, as a :class:`~neurospatial.stats.ShuffleTestResult`."""
        from neurospatial.encoding._significance import shuffle_test_result

        i = self._unit_position(idx)
        return shuffle_test_result(
            float(self.observed[i]),
            self.null_scores[i],
            float(self.p_value[i]),
            statistic=self.statistic,
            statistic_units=self.statistic_units,
        )

    def classify(
        self, *, alpha: float = 0.05, min_info: float = 0.5
    ) -> NDArray[np.bool_]:
        """Screen each unit for place tuning from its shuffle test.

        A unit passes when its information test is significant
        (``p_value < alpha``) and its bias-corrected information
        (``observed - spatial_information_bias``) is at least ``min_info``,
        the rule of ``is_place_cell(..., criterion="shuffle")``, which gives
        the same answer for the same seeded unit. It tests information alone;
        the ``is_place`` column of
        :meth:`CellTypeSignificanceResult.summary_table` also excludes units
        with a significant grid or border score.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level of each unit's test; no correction across
            units.
        min_info : float, default=0.5
            Inclusive cutoff on bias-corrected information (bits/spike), a
            common screening convention, not derived or calibrated.

        Returns
        -------
        ndarray of bool, shape (n_units,)
            False for a unit with an undefined p-value or an unknown bias.

        Warns
        -----
        UserWarning
            When a significant unit's bias is unknown, so it fails.
        """
        from neurospatial.encoding._metrics import _warn_unknown_bias

        refuse_none_thresholds(alpha=alpha, min_info=min_info)
        with np.errstate(invalid="ignore"):
            significant = self.p_value < alpha
        debiased = self.observed - self.spatial_information_bias
        _warn_unknown_bias(significant, debiased, kind="place")
        with np.errstate(invalid="ignore"):
            passed: NDArray[np.bool_] = significant & (debiased >= min_info)
        return passed

    def summary_table(
        self, *, alpha: float = 0.05, min_info: float = 0.5
    ) -> pd.DataFrame:
        """One row per unit: the observed score, its null and the screen.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level of ``is_place_cell``; see :meth:`classify`.
        min_info : float, default=0.5
            Bias-corrected information cutoff of ``is_place_cell``
            (bits/spike); see :meth:`classify`.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``unit_id``, with ``spatial_information`` (the
            observed score), ``spatial_information_bias``, ``p_value``,
            ``null_mean``, ``null_std``, ``z_score``, ``n_finite_null`` (the
            null scores the p-value counts) and ``is_place_cell`` from
            :meth:`classify`. ``df.attrs`` records ``units``, ``statistic``,
            ``alpha``, ``classification_thresholds``, ``n_shuffles``,
            ``shuffle_type`` and ``analysis_settings``.
        """
        import pandas as pd

        df = pd.DataFrame(
            {
                "spatial_information": self.observed.copy(),
                "spatial_information_bias": self.spatial_information_bias.copy(),
                "p_value": self.p_value.copy(),
                "null_mean": self.null_mean,
                "null_std": self.null_std,
                "z_score": self.z_score,
                "n_finite_null": (~np.isnan(self.null_scores)).sum(axis=1),
                "is_place_cell": self.classify(alpha=alpha, min_info=min_info),
            },
            index=pd.Index(self.unit_ids.tolist(), name="unit_id"),
        )
        df.attrs.update(
            statistic=self.statistic,
            alpha=alpha,
            classification_thresholds={"min_info": min_info},
            n_shuffles=self.n_shuffles,
            shuffle_type="circular_time_shift",
            analysis_settings=self.analysis_settings,
            units={
                "spatial_information": "bits/spike",
                "spatial_information_bias": "bits/spike",
                "p_value": "",
                "null_mean": "bits/spike",
                "null_std": "bits/spike",
                "z_score": "",
                "n_finite_null": "",
                "is_place_cell": "",
            },
        )
        return df

    def summary(self) -> dict[str, Any]:
        """The statistic, its unit, and the unit and shuffle counts."""
        return {
            "statistic": self.statistic,
            "statistic_units": self.statistic_units,
            "n_units": self.n_units,
            "n_shuffles": self.n_shuffles,
        }

    def plot(
        self, idx: int | None = None, *, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Plot one unit's null distribution with its observed score.

        Parameters
        ----------
        idx : int
            Unit position. Required: each unit has its own null
            distribution, so ``None`` raises a ``ValueError`` that says how
            to pick a unit.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure when None.
        **kwargs
            Passed to :meth:`~neurospatial.stats.ShuffleTestResult.plot`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from neurospatial._results import unit_index_required

        if idx is None:
            raise unit_index_required(
                len(self),
                what="null distribution",
                why="each unit is tested against its own null distribution.",
            )
        return self[idx].plot(ax=ax, **kwargs)


@keyword_hint()
@reads_pynapple_samples
def spatial_cell_type_significance(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    boundary_source: BoundarySource = "physical",
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float = 0.0,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    backend: BackendType = "numpy",
    warn_on_drop: bool = False,
    unit_ids: NDArray[Any] | Sequence[Any] | None = None,
    n_shuffles: int = 1000,
    min_shift: float = 20.0,
    rng: np.random.Generator | int | None = None,
) -> CellTypeSignificanceResult:
    """Test place, grid and border tuning against circular spike-time shifts.

    Each unit's spike train is circularly shifted against the behavior, and
    spatial information, grid score and border score are recomputed from the
    same shifted trains. Each statistic gets its own p-value, and each label
    is judged by its own test: a grid cell needs a significant grid score and
    a hexagonal first ring of autocorrelogram peaks, a border cell a
    significant border score, and a cell that passes both is
    reported as ``"grid+border"``. A place cell has significant spatial
    information, bias-corrected information
    (``spatial_information - spatial_information_bias``) of at least
    ``min_spatial_info`` (0.5 bits/spike by default, as on the screen), and
    is neither. Use this for a verdict; the fixed cutoffs of
    :meth:`SpatialRatesResult.label_cell_types` are a screen.

    Parameters
    ----------
    env : Environment
        The spatial environment defining the bin structure.
    spike_times : sequence of arrays or 2D array
        Spike times in seconds for each unit, in any form
        :func:`compute_spatial_rates` accepts. A labelled group (such as a
        :class:`~neurospatial.encoding.SpikeTrains` or a pynapple ``TsGroup``)
        supplies the unit labels.
    times : ndarray, shape (n_samples,)
        Sample timestamps in seconds, sorted and aligned with ``positions``.
    positions : ndarray, shape (n_samples, n_dims)
        Animal coordinates aligned with ``times``, in environment units.
    boundary_source : {'physical', 'analysis'}, default='physical'
        Wall geometry of the border score: the measured arena walls, or
        explicitly the outline of the retained bins.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Rate-map estimator; see :func:`compute_spatial_rates`.
    bandwidth : float or None, default=None
        Smoothing bandwidth in environment units; None is 0 (no smoothing)
        for ``method="binned"``, and 5 cm in metric units or 2.5 median
        graph-edge lengths otherwise for the kernel methods; the units-free
        default warns with its resolved value.
    min_occupancy : float, default=0.0
        Seconds a bin must be occupied to count; ``0.0`` masks nothing.
    speed : ndarray, shape (n_samples,), optional
        Speed for the ``min_speed`` gate; derived from ``positions`` when None.
    min_speed : float or None, default=None
        Exclude intervals slower than this (environment units per second)
        from spikes and occupancy alike. None applies no speed gate.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Analyze only these half-open windows, in seconds.
        When ``positions`` is a pynapple ``TsdFrame`` or ``Tsd``, ``None`` uses its
        ``time_support``.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a recording break before it.
    spike_window : same forms as ``epochs``, or None
        When the electrophysiology was recording; None assumes it recorded
        whenever position did.
    backend : {"numpy", "jax", "auto"}, default="numpy"
        Backend for rate-map smoothing.
    warn_on_drop : bool, default=False
        Report spikes dropped from the observed maps.
    unit_ids : ndarray or sequence, optional
        Unit labels, one per spike train; defaults to the labels of a
        labelled group, else ``0..n_units - 1``. They also key each unit's
        random stream, so a seeded unit gets the same shifts alone or in a
        population.
    n_shuffles : int, default=1000
        Number of shifted trains per unit.
    min_shift : float, default=20.0
        Smallest shift in either direction, in seconds of analyzed time, as
        for the other ``*_significance`` functions.
    rng : numpy.random.Generator, int or None, default=None
        Random source. With an integer seed, a unit gets the same shifts, and
        so the same p-values, alone, in any population order and in
        :func:`place_cell_significance`.

    Returns
    -------
    CellTypeSignificanceResult
        Per-unit observed scores, null scores and p-values for spatial
        information, grid score and border score. Its ``summary_table()``
        adds null percentiles and labels, and ``label_cell_types()`` returns
        the labels.

    Raises
    ------
    ValueError
        For invalid inputs or settings, duplicate or conflicting unit labels,
        or too little analyzed time for ``min_shift``.

    Notes
    -----
    The null assumes firing is stationary on the joined analyzed clock;
    shifts never enter recording gaps or excluded epochs. Occupancy, analyzed
    intervals and bandwidth come from the observed maps, so the trajectory is
    binned once. A score that cannot be computed is recorded, not raised:
    grid and border cells are defined on 2-D maps and do not apply to a 1-D
    track, the grid score needs a regular grid of rectangular cells, and the
    physical border score needs measured walls with observed bins along
    them. Those statistics have NaN p-values and their reason in
    ``unavailable_metrics``. Where such a statistic applies but cannot be
    computed the label is ``"undetermined"``; ``summary_table()["is_place"]``
    is then True for a unit whose information test passes and that was not
    shown to be a grid or border cell, a place-or-border candidate when the
    border test is missing. A map with no field of the minimum size has no
    border score, so its p-value is NaN and it is not a border cell. A null
    map without a defined score is left out of that unit's p-value, which is
    (1 + k) / (1 + n) with n the finite null scores and k those at least the
    observed score. Its smallest value is 1 / (1 + n), so when
    1 / (1 + n) >= alpha the test cannot reject and the unit's label is
    ``"undetermined"`` rather than ``"unclassified"``; ``summary_table()``
    reports n as ``<statistic>_n_finite_null``. With a correction the same
    holds for a test that fails it even with every test at its smallest
    p-value (see :meth:`CellTypeSignificanceResult.label_cell_types`).

    Each test controls its own false-positive rate at ``alpha``, and each
    unit is tested three times. Calibration: three simulated 20-minute
    sessions in a 100 cm box (2.5 cm bins, 1000 shuffles, analysis
    boundary), each with 100 untuned Poisson units (0.5-3 Hz), 10 place cells
    and 5 grid cells. Of the 300 untuned units 5.3%, 3.7% and 5.3% had
    p < 0.05 for spatial information, grid score and border score; a unit's
    three p-values were nearly independent (rank correlations -0.11 to
    0.03), and 12.3% had at least one p < 0.05. The place label also needs
    bias-corrected information and the grid label a hexagonal ring, so
    6.7% (standard error 1.4%) of the untuned units received a label without
    a correction, 2.3% (0.9%) with ``correction="holm"`` and 1.0% (0.6%)
    with ``correction="fdr_bh"`` (3 false labels among 48); all 30 place
    cells and 15 grid cells kept their labels under every option. With 200
    shuffles ``"fdr_bh"`` labeled no unit at all, tuned cells included: their
    p-values sat at the floor 1 / 201, which the correction cannot use, and
    a warning names the number of shuffles needed. The fixed-cutoff screen
    labeled 8-53% of 100 untuned units border cells in each of four 20- to
    40-minute sessions, fewer where more of the walls went unvisited. Of 12
    simulated place cells on a radial-arm maze none was labeled a border cell (the screen labeled 6). A significant grid score says the
    map's autocorrelogram is more sixfold-symmetric than those of its own
    shifted spike trains, not that it has a lattice of fields: 2 of 30
    simulated single-field place cells had significant grid scores (0.92 and
    1.07). A grid label therefore also needs the observed map's
    autocorrelogram peaks nearest zero lag to be one hexagon's first ring
    (``grid_spacing`` finite, as in :meth:`SpatialRateResult.grid_properties`).
    A grid whose spacing exceeds about 0.6 of a square arena's side has no
    ring inside the autocorrelogram and is not labeled grid.
    Scoring all three statistics for 50 units with 200 shuffles in that box
    took about 16 s on one core, against 3 s for spatial information alone.

    See Also
    --------
    place_cell_significance : The spatial-information test alone.
    SpatialRatesResult.label_cell_types : Fixed-cutoff screen.

    Examples
    --------
    >>> import numpy as np
    >>> import shapely
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import spatial_cell_type_significance
    >>> env = Environment.from_polygon(
    ...     shapely.box(0, 0, 100, 100), bin_size=5.0, units="cm"
    ... )
    >>> times = np.arange(0.0, 120.0, 0.1)
    >>> positions = np.c_[50 + 40 * np.sin(times / 3), 50 + 40 * np.cos(times / 4)]
    >>> rng = np.random.default_rng(0)
    >>> trains = [np.sort(rng.uniform(0, 119.9, 120)) for _ in range(2)]
    >>> result = spatial_cell_type_significance(
    ...     env, trains, times, positions, n_shuffles=39, rng=0
    ... )
    >>> result.p_values.shape
    (2, 3)
    >>> table = result.summary_table()
    >>> list(table.columns[:3])
    ['spatial_information', 'grid_score', 'border_score']
    """
    shuffled = _spatial_shuffle(
        env,
        spike_times,
        times,
        positions,
        statistics=_CELL_TYPE_STATISTICS,
        call="spatial_cell_type_significance",
        boundary_source=boundary_source,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        penalty=None,
        rank=None,
        pooled=True,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        spike_window=spike_window,
        backend=backend,
        warn_on_drop=warn_on_drop,
        dtype=np.float64,
        unit_ids=unit_ids,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        rng=rng,
    )
    from neurospatial.encoding._metrics import _information_bias_terms

    # Labeling reports the units whose bias is unknown, in terms of labels.
    bias = _information_bias_terms(shuffled.reference).bias
    grid_spacing = np.full(len(shuffled.unit_ids), np.nan)
    if "grid_score" not in shuffled.unavailable:
        grid_spacing = _grid_spacings(
            env, np.atleast_2d(shuffled.reference._supported_rates())
        )
    return CellTypeSignificanceResult(
        observed_scores=shuffled.observed,
        null_scores=shuffled.null.transpose(1, 0, 2),
        unit_ids=shuffled.unit_ids,
        min_shift=float(min_shift),
        method=method,
        bandwidth=shuffled.bandwidth,
        boundary_source=boundary_source,
        grid_spacing=grid_spacing,
        spatial_information_bias=bias,
        unavailable_metrics=shuffled.unavailable,
        inapplicable_metrics=shuffled.inapplicable,
    )


class _ShuffleScores(NamedTuple):
    """Observed and null scores of a circular-shift test of spatial maps."""

    observed: NDArray[np.float64]  # (n_units, n_statistics); NaN when unscored
    null: NDArray[np.float64]  # (n_shuffles, n_units, n_statistics)
    unit_ids: NDArray[Any]  # (n_units,) resolved labels in input order
    unavailable: dict[str, str]  # why each unscored statistic has no score
    inapplicable: tuple[str, ...]  # unscored statistics that do not apply
    bandwidth: float  # resolved smoothing bandwidth
    reference: SpatialRatesResult  # the observed rate maps


class _ShuffleMaps(NamedTuple):
    """Trajectory binning and estimator settings shared by every shifted map."""

    track: _SpikeTrack  # frame bins on the analyzed intervals, built once
    occupancy: NDArray[np.float64]  # (n_bins,) seconds
    occupied: NDArray[np.bool_]  # (n_bins,) occupancy support of the metrics
    method: str
    bandwidth: float
    min_occupancy: float
    backend: ResolvedBackend
    dtype: type[np.float32] | type[np.float64]


def _shuffle_rate_maps(
    env: Environment, spike_times: list[NDArray[np.float64]], maps: _ShuffleMaps
) -> NDArray[np.float64]:
    """Supported rate maps of spike trains on the shared trajectory binning.

    Matches ``compute_spatial_rates(...)._supported_rates()`` for the same
    trains and settings, without re-binning the trajectory.

    Returns
    -------
    ndarray, shape (n_units, n_bins)
        Rates in Hz, NaN outside the occupancy support.
    """
    from neurospatial.encoding._smoothing import smooth_rate_maps_batch
    from neurospatial.encoding._spatial_counts import _bin_spike_train_with_stats

    counts = np.zeros((len(spike_times), env.n_bins), dtype=np.float64)
    for i, train in enumerate(spike_times):
        counts[i] = _bin_spike_train_with_stats(env, train, maps.track)[0]
    rates = smooth_rate_maps_batch(
        env,
        counts,
        maps.occupancy,
        method=cast("Any", maps.method),
        bandwidth=maps.bandwidth,
        min_occupancy=maps.min_occupancy,
        backend=maps.backend,
        dtype=maps.dtype,
    )
    stored = np.asarray(rates).astype(maps.dtype, copy=False)
    return np.where(maps.occupied, stored.astype(np.float64), np.nan)


def _spatial_scores(
    env: Environment,
    spike_times: list[NDArray[np.float64]],
    maps: _ShuffleMaps,
    statistics: tuple[str, ...],
    boundary_source: BoundarySource,
) -> NDArray[np.float64]:
    """Score each unit's rate map with each named statistic.

    Returns
    -------
    ndarray, shape (n_units, len(statistics))
        Spatial information (bits/spike), grid score and border score, in the
        order of ``statistics``.
    """
    from neurospatial.encoding._metrics import (
        batch_border_scores,
        batch_grid_scores,
        batch_spatial_information,
    )

    rates = _shuffle_rate_maps(env, spike_times, maps)
    columns = []
    for name in statistics:
        if name == "spatial_information":
            columns.append(np.asarray(batch_spatial_information(rates, maps.occupancy)))
        elif name == "grid_score":
            columns.append(batch_grid_scores(env, rates).scores)
        else:
            columns.append(
                batch_border_scores(env, rates, boundary_source=boundary_source).scores
            )
    return np.column_stack(columns).reshape(len(spike_times), len(statistics))


def _cell_type_score(
    env: Environment,
    rates: NDArray[np.float64],
    name: str,
    boundary_source: BoundarySource,
) -> tuple[NDArray[np.float64], str | None, bool]:
    """Grid or border scores of supported rate maps, or why there are none.

    Parameters
    ----------
    env : Environment
        Environment of the rate maps.
    rates : ndarray, shape (n_units, n_bins)
        Rates in Hz, NaN outside the occupancy support.
    name : {"grid_score", "border_score"}
        Statistic to score.
    boundary_source : {"physical", "analysis"}
        Wall geometry of the border score.

    Returns
    -------
    scores : ndarray, shape (n_units,)
        The scores; all NaN when the statistic cannot be scored.
    reason : str or None
        Why the statistic cannot be scored, or None when it was scored.
    applies : bool
        False when grid or border cells do not exist in this environment:
        both are defined on 2-D maps, so a 1-D track or a 3-D volume has none.
        True when the statistic applies, even if it could not be scored.
    """
    from neurospatial.encoding._metrics import batch_border_scores, batch_grid_scores

    unscored = np.full(len(rates), np.nan)
    if env.n_dims != 2:
        kind = "grid" if name == "grid_score" else "border"
        return (
            unscored,
            f"{kind} cells are defined on 2-D maps; this environment is "
            f"{env.n_dims}-D, so {name} does not apply.",
            False,
        )
    try:
        if name == "grid_score":
            return batch_grid_scores(env, rates).scores, None, True
        return (
            batch_border_scores(env, rates, boundary_source=boundary_source).scores,
            None,
            True,
        )
    except NotImplementedError as exc:
        return unscored, str(exc), True


def _spatial_shuffle(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    statistics: tuple[str, ...],
    call: str,
    boundary_source: BoundarySource,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"],
    bandwidth: float | None,
    min_occupancy: float,
    penalty: float | None,
    rank: int | None,
    pooled: bool,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    epochs: Any,
    break_before: ArrayLike | None,
    spike_window: Any,
    backend: BackendType,
    warn_on_drop: bool,
    dtype: type[np.float32] | type[np.float64],
    unit_ids: NDArray[Any] | Sequence[Any] | None,
    n_shuffles: int,
    min_shift: float,
    rng: np.random.Generator | int | None,
) -> _ShuffleScores:
    """Circularly shift each unit's spikes and rescore its rate map.

    The trajectory is binned once: the analyzed intervals, frame bins,
    occupancy and smoothing settings of the observed map are reused for every
    shifted train, and each shift keeps the unit's spike count on the
    analyzed clock without entering recording gaps. Every requested statistic
    is computed from the same shifted trains.

    Returns
    -------
    _ShuffleScores
        Observed scores ``(n_units, len(statistics))`` and null scores
        ``(n_shuffles, n_units, len(statistics))``, NaN for statistics that
        cannot be scored, with the reasons (see ``_cell_type_score``).
    """
    from neurospatial.encoding._backend import validate_and_resolve_backend
    from neurospatial.encoding._provenance import occupancy_support
    from neurospatial.encoding._significance import (
        run_shuffle_test,
        shuffle_inputs,
        snapshot_options,
    )
    from neurospatial.encoding._spatial_counts import (
        _spike_track,
        require_allocentric,
    )
    from neurospatial.encoding._validation import (
        _validate_env_argument,
        validate_trajectory,
    )

    if method == "glm":
        raise ValueError(
            _format_error(
                f"{call} does not support method='glm'.",
                why="Why: pooled REML couples units, so shifted single-unit fits would not be the observed model",
                fix="pass method='diffusion_kde', 'gaussian_kde' or 'binned'",
            )
        )
    inputs = shuffle_inputs(
        spike_times,
        times,
        (positions,),
        unit_ids=unit_ids,
        epochs=epochs,
        spike_window=spike_window,
        call=call,
    )
    trains, times, ids = inputs.trains, inputs.times, inputs.unit_ids
    (positions,) = inputs.copies
    options: dict[str, Any] = {
        "method": method,
        "bandwidth": bandwidth,
        "min_occupancy": min_occupancy,
        "penalty": penalty,
        "rank": rank,
        "pooled": pooled,
        "speed": speed,
        "min_speed": min_speed,
        "max_gap": max_gap,
        "backend": backend,
        "warn_on_drop": warn_on_drop,
        "dtype": dtype,
    }
    options = snapshot_options(options, unit_ids=ids)
    _validate_env_argument(env, context=call, arguments="spike_times, times, positions")
    require_allocentric(env, f"{call}()")
    validate_trajectory(times, positions=positions, context=call, n_dims=env.n_dims)
    options["break_before"] = validate_break_before(break_before, len(times), call=call)
    positions_2d = positions.reshape(-1, 1) if positions.ndim == 1 else positions
    # The observed map resolves the estimator defaults, the analyzed
    # intervals, its occupancy and the caller's data-quality warnings once;
    # every shifted map reuses them.
    rate_maps = _spatial_rate_maps(
        env,
        trains,
        times,
        positions,
        **options,
        windows=inputs.windows,
        time_allocation="start",
        n_jobs=1,
        context="compute_spatial_rates",
        single_unit=False,
    )
    reference = _spatial_rates_result(env, rate_maps, method=method, unit_table=None)
    time_support = rate_maps.support.time_support
    occupancy = np.asarray(reference.occupancy, dtype=np.float64)
    maps = _ShuffleMaps(
        track=_spike_track(
            env,
            times,
            positions_2d,
            interval_mask=time_support.mask,
            linear_intervals=None,
        ),
        occupancy=occupancy,
        occupied=occupancy_support(reference),
        method=method,
        bandwidth=cast("float", reference.bandwidth),
        min_occupancy=reference.min_occupancy,
        backend=validate_and_resolve_backend(backend),
        dtype=options["dtype"],
    )
    # The observed maps decide which statistics can be scored; the observed
    # scores are recomputed, with their warnings, by the shuffle test below.
    observed_rates = np.asarray(reference._supported_rates(), dtype=np.float64)
    unavailable: dict[str, str] = {}
    inapplicable: list[str] = []
    for name in statistics:
        if name == "spatial_information":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, reason, applies = _cell_type_score(
                env, observed_rates, name, boundary_source
            )
        if reason is not None:
            unavailable[name] = reason
        if not applies:
            inapplicable.append(name)
    scored = tuple(name for name in statistics if name not in unavailable)

    def statistic(shifted: list[NDArray[np.float64]]) -> ArrayLike:
        return _spatial_scores(env, shifted, maps, scored, boundary_source)

    observed_scored, null_scored = run_shuffle_test(
        statistic,
        trains,
        time_support.runs,
        ids,
        n_shuffles=n_shuffles,
        min_shift=min_shift,
        rng=rng,
    )
    columns = [statistics.index(name) for name in scored]
    observed = np.full((len(trains), len(statistics)), np.nan)
    null = np.full((n_shuffles, len(trains), len(statistics)), np.nan)
    observed[:, columns] = observed_scored.reshape(len(trains), len(scored))
    null[:, :, columns] = null_scored.reshape(n_shuffles, len(trains), len(scored))
    return _ShuffleScores(
        observed=observed,
        null=null,
        unit_ids=ids,
        unavailable=unavailable,
        inapplicable=tuple(inapplicable),
        bandwidth=maps.bandwidth,
        reference=reference,
    )
