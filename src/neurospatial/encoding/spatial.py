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
from collections import deque
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurospatial._exceptions import _format_error
from neurospatial._observation import validate_break_before
from neurospatial._results import (
    ReadOnlyArraysMixin,
    ResultMixin,
    UnitSelectionMixin,
    _copy_analysis_settings,
    _copy_unit_table,
    read_only_copy,
)
from neurospatial.encoding._base import (
    KERNEL_ESTIMATORS,
    BandwidthRule,
    PopulationRateMixin,
    SingleUnitRateMixin,
    _is_jax_array,
    _to_numpy,
)
from neurospatial.encoding._metrics import BatchScoresResult
from neurospatial.encoding._provenance import spikes_outside_intervals
from neurospatial.encoding._significance import (
    _SHUFFLE_DEFAULTS,
    check_criterion,
    check_mode_keywords,
    resolve_shuffle_settings,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial._intervals import TimeSupport
    from neurospatial._typing import SpikeTrainsLike
    from neurospatial.encoding._spatial_counts import LinearSpatialIntervals
    from neurospatial.encoding.grid import GridProperties
    from neurospatial.environment import Environment
    from neurospatial.geometry._boundaries import BoundarySource
    from neurospatial.ops._types import BackendType, DistanceMetric, ResolvedBackend
    from neurospatial.stats.shuffle import ShuffleTestResult

# ruff: noqa: RUF022 - intentionally grouped by category
__all__ = [
    "place_cell_significance",
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
        "min_size": None,
        "min_peak_rate": 1.0,
        "min_secondary_peak_fraction": 0.6,
        "max_mean_rate": np.inf,
        "detect_subfields": True,
    }
)
# Statistics recomputed from every shifted spike train, in result column order.
_CELL_TYPE_STATISTICS = ("spatial_information", "grid_score", "border_score")
PLACE_SPATIAL_INFO_THRESHOLDS = MappingProxyType({"min_info": 0.5})
PLACE_GRID_BORDER_THRESHOLDS = MappingProxyType(
    {"min_spatial_info": 0.5, "min_grid_score": 0.4, "min_border_score": 0.5}
)


@dataclass(frozen=True, repr=False, eq=False)
class PlaceFieldsResult(ReadOnlyArraysMixin, ResultMixin):
    """Result of ``detect_place_fields()``: detected fields plus exclusion metadata.

    Returned by :func:`detect_place_fields`. Distinguishes "this map
    has no detectable fields" (``fields=[]``, ``excluded_reason=None``)
    from "this map was excluded by an explicitly requested spatial-map-mean filter"
    (``fields=[]``, ``excluded_reason="mean_rate_above_threshold"``) so a
    population pipeline can branch without listening for warnings.

    Attributes
    ----------
    fields : tuple of NDArray[np.int64], length n_fields
        One read-only 1-D array of bin indices per detected field, in order
        of decreasing peak rate. Empty if no field was detected or the map
        was excluded by the ``max_mean_rate`` filter.
    excluded_reason : {"mean_rate_above_threshold"} or None
        ``None`` when ``fields`` is the detection result.
        ``"mean_rate_above_threshold"`` when the explicit ``max_mean_rate``
        filter excluded the map before detection.
    n_excluded : int
        ``1`` if ``excluded_reason`` is set, else ``0``, so population
        aggregation can sum exclusions without parsing the reason.

    Notes
    -----
    The result is iterable, indexable and sized like its ``fields``, and is
    truthy when at least one field was detected.
    """

    fields: Sequence[NDArray[np.int64]]
    excluded_reason: Literal["mean_rate_above_threshold"] | None = None

    def __post_init__(self) -> None:
        """Validate and own each field's bins and prevent replacing fields."""
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
        owned = []
        for i, bins in enumerate(self.fields):
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
                        f"fields[{i}] must be a 1-D array of integer bin indices, "
                        f"got shape {array.shape} and dtype {array.dtype}.",
                        why="Why: each field is the set of bins it covers",
                        fix="pass integer bin indices, e.g. np.flatnonzero(mask)",
                    )
                )
            if array.size and array.min() < 0:
                raise ValueError(
                    _format_error(
                        f"fields[{i}] contains negative bin indices.",
                        why="Why: bin indices count from 0",
                        fix="pass non-negative bin indices",
                    )
                )
            owned.append(read_only_copy(array.astype(np.int64)))
        object.__setattr__(self, "fields", tuple(owned))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore owned field arrays after copying."""
        super().__setstate__(state)
        self.__post_init__()

    @property
    def n_excluded(self) -> int:
        """``1`` if a filter excluded the map, else ``0``."""
        return int(self.excluded_reason is not None)

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

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.spatial import PlaceFieldsResult
        >>> result = PlaceFieldsResult(fields=[np.array([0, 1, 2])])
        >>> result.summary()["n_fields"]
        1
        >>> result.summary()["total_bins"]
        3
        """
        total_bins = int(sum(len(f) for f in self.fields))
        return {
            "n_fields": len(self.fields),
            "total_bins": total_bins,
            "n_excluded": self.n_excluded,
            "excluded_reason": self.excluded_reason,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy/long-form table of field membership: one row per (field, bin).

        Each detected place field contributes one row per member bin, with a
        ``field`` index column. Neurons excluded by a filter (``fields=[]``)
        yield an empty table; ``excluded_reason`` is exposed via
        :meth:`summary`.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``field`` (int, field index) and
            ``bin`` (int, member bin index). Empty (with those columns) when
            no fields were detected.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.encoding.spatial import PlaceFieldsResult
        >>> result = PlaceFieldsResult(fields=[np.array([0, 1]), np.array([5])])
        >>> df = result.to_dataframe()
        >>> df["field"].tolist()
        [0, 0, 1]
        >>> df["bin"].tolist()
        [0, 1, 5]
        """
        import pandas as pd

        field_col: list[int] = []
        bin_col: list[int] = []
        for field_idx, field_bins in enumerate(self.fields):
            for b in np.asarray(field_bins).ravel():
                field_col.append(field_idx)
                bin_col.append(int(b))
        return pd.DataFrame(
            {
                "field": np.asarray(field_col, dtype=np.int64),
                "bin": np.asarray(bin_col, dtype=np.int64),
            }
        )


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
        raise ValueError(
            f"{kind} has method={method!r}, which is not a known estimator; "
            f"expected one of {list(_SPATIAL_METHODS)}."
        )

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
            raise ValueError(
                f"{kind} with method='glm' is missing required GAM field(s): "
                f"{missing}. glm results carry all of {list(_GAM_REQUIRED_FIELDS)}."
            )
        if fields["bandwidth"] is not None:
            raise ValueError(
                f"{kind} with method='glm' must have bandwidth=None (glm has no "
                f"bandwidth); got {fields['bandwidth']!r}."
            )
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
                    raise ValueError(
                        f"{kind} (single-unit) glm field {name!r} must be a scalar; "
                        f"got shape {tuple(np.shape(value))}."
                    )
            elif ndim not in (0, 1) or (
                ndim == 1 and tuple(np.shape(value)) != (n_units,)
            ):  # plural: scalar or (n_units,)
                raise ValueError(
                    f"{kind} glm field {name!r} must be a scalar or a ({n_units},) "
                    f"vector; got shape {tuple(np.shape(value))}."
                )
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
                raise ValueError(
                    f"{kind} carries penalty_selected_by_reml (a per-unit result) "
                    f"but pooled={fields['pooled']!r}; per-unit lambda is only "
                    "produced under pooled=False."
                )
            missing_vec = [
                n
                for n in (*per_unit_slots, "penalty_selected_by_reml")
                if not _is_per_unit_shaped(fields[n])
            ]
            if missing_vec:
                raise ValueError(
                    f"{kind} carries penalty_selected_by_reml (a per-unit result) "
                    f"but {missing_vec} are not per-unit values; penalty, "
                    "reml_objective, reml_at_boundary, and penalty_selected_by_reml "
                    "must all be per-unit when the provenance mask is present."
                )
            # Semantic coupling of the per-unit provenance (values, not just
            # shapes), so a corrupt record cannot round-trip through NWB:
            selected = np.asarray(mask, dtype=bool)
            penalty_vals = np.asarray(fields["penalty"], dtype=np.float64)
            reml_vals = np.asarray(fields["reml_objective"], dtype=np.float64)
            # Every applied lambda is a finite, positive smoothing penalty.
            if not np.all(np.isfinite(penalty_vals) & (penalty_vals > 0.0)):
                raise ValueError(
                    f"{kind} per-unit penalty must be finite and > 0 (an applied "
                    f"smoothing lambda); got {penalty_vals}."
                )
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
                raise ValueError(
                    f"{kind} reml_objective must be finite for REML-selected units "
                    "and the nan sentinel for pooled-fallback units; got "
                    f"selected={selected}, reml_objective={reml_vals}."
                )
            # The per-unit vector path only exists with >= 1 informative unit
            # (population level). A singular slice may legitimately be a lone
            # fallback unit, so only pin this on the plural class.
            if n_units is not None and not selected.any():
                raise ValueError(
                    f"{kind} is a per-unit result but penalty_selected_by_reml is "
                    "all False; the per-unit lambda path requires at least one "
                    "informative (REML-selected) unit."
                )
        else:
            # Non-per-unit result: none of the three may be a per-unit vector
            # (this is exactly the state that crashes NWB writing).
            stray_vec = [
                n
                for n in per_unit_slots
                if fields[n] is not None and np.ndim(fields[n]) != 0
            ]
            if stray_vec:
                raise ValueError(
                    f"{kind} has vector {stray_vec} but no penalty_selected_by_reml "
                    "mask; per-unit penalty / reml_objective / reml_at_boundary "
                    "require the provenance mask (a per-unit pooled=False result). "
                    "For a shared/fixed lambda these must be scalar or None."
                )
        # A boundary flag only exists when REML produced an objective, so
        # ``reml_at_boundary`` being set requires ``reml_objective`` set too. The
        # reverse is NOT required: a schema-2.1 file records ``reml_objective`` but
        # no ``reml_at_boundary`` (it predates the flag), and reads back with the
        # objective set and the boundary flag ``None`` -- a legitimate state.
        if fields["reml_at_boundary"] is not None and fields["reml_objective"] is None:
            raise ValueError(
                f"{kind} has reml_at_boundary={fields['reml_at_boundary']!r} but "
                "reml_objective=None; a boundary flag requires REML to have run "
                "(so reml_objective must be populated)."
            )
        # Symmetrically, a REML objective only exists when a lambda was applied, so
        # a set ``reml_objective`` requires a set ``penalty`` (both come from the
        # same selection; None together on REML-skip / no-data).
        if fields["reml_objective"] is not None and fields["penalty"] is None:
            raise ValueError(
                f"{kind} has reml_objective={fields['reml_objective']!r} but "
                "penalty=None; a REML objective requires an applied penalty."
            )
        # The boolean diagnostics must be boolean (a float boundary flag or an
        # integer provenance mask would round-trip wrong through NWB).
        for name in ("reml_at_boundary", "penalty_selected_by_reml"):
            value = fields[name]
            if value is not None and np.asarray(value).dtype != np.bool_:
                raise ValueError(
                    f"{kind} glm field {name!r} must be boolean; got dtype "
                    f"{np.asarray(value).dtype}."
                )
        # ``pooled`` is a concrete bool for every glm result (the only reliable
        # NWB source, since scalar cases are value-identical under both settings).
        # Checked last so a more fundamental shape/required error surfaces first.
        if not isinstance(fields["pooled"], bool):
            raise ValueError(
                f"{kind} with method='glm' must carry pooled=True/False; got "
                f"{fields['pooled']!r}."
            )
    else:
        stray = [
            name
            for name, value in {**required, **optional}.items()
            if value is not None
        ]
        if fields["pooled"] is not None:
            stray.append("pooled")
        if stray:
            raise ValueError(
                f"{kind} with ratio method={fields['method']!r} must not carry GAM "
                f"field(s): {stray}. GAM diagnostics belong only to method='glm'."
            )
        if fields["bandwidth"] is None:
            raise ValueError(
                f"{kind} with ratio method={fields['method']!r} must have a float "
                "bandwidth; got None (bandwidth=None is reserved for method='glm')."
            )


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


def _passes_information_screen(info: ArrayLike, min_info: float) -> Any:
    """Inclusive spatial-information screen shared by every place-cell screen."""
    return np.asarray(info) >= min_info


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
    """``passes_<statistic>`` columns (``<NA>`` when unknown) and ``cell_type``."""
    import pandas as pd

    columns: dict[str, Any] = {
        f"passes_{name}": pd.array(
            [None if np.isnan(value) else bool(value) for value in values],
            dtype="boolean",
        )
        for name, values in outcomes.items()
    }
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

    rates = np.atleast_2d(_to_numpy(_supported_rates(result)))
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


def _screen_outcomes(
    scores: Mapping[str, NDArray[np.float64]],
    unavailable: Mapping[str, str],
    inapplicable: tuple[str, ...],
    *,
    min_spatial_info: float,
    min_grid_score: float,
    min_border_score: float,
) -> dict[str, NDArray[np.float64]]:
    """Fixed-threshold outcome of each test; a NaN score fails its test."""
    passed = {
        "spatial_information": _passes_information_screen(
            scores["spatial_information"], min_spatial_info
        ),
        "grid_score": scores["grid_score"] >= min_grid_score,
        "border_score": scores["border_score"] >= min_border_score,
    }
    return {
        name: _test_outcome(
            values, unknown=name in unavailable and name not in inapplicable
        )
        for name, values in passed.items()
    }


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


def _spatial_summary_frame(
    result: SpatialRateResult | SpatialRatesResult,
    *,
    index: Sequence[Hashable],
    include_classification: bool,
    boundary_source: BoundarySource,
) -> pd.DataFrame:
    """Build spatial metric columns for single and population results."""
    import pandas as pd

    from neurospatial.encoding._metrics import batch_sparsity
    from neurospatial.encoding.border import _border_field_minimum

    rates = np.atleast_2d(_to_numpy(_supported_rates(result)))
    occupancy = _to_numpy(result.occupancy)
    peaks = np.atleast_2d(result.peak_location())
    scores, unavailable, inapplicable = _screen_scores(
        result, boundary_source=boundary_source
    )
    columns: dict[str, Any] = {
        "peak_firing_rate": np.atleast_1d(result.peak_firing_rate()),
        "spatial_information": scores["spatial_information"],
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
                    scores, unavailable, inapplicable, **PLACE_GRID_BORDER_THRESHOLDS
                )
            )
        )
    df = pd.DataFrame(columns, index=pd.Index(list(index), name="unit_id"))
    df.attrs["border_boundary_source"] = boundary_source
    df.attrs["border_field_minimum"] = _border_field_minimum(result.env)
    if unavailable:
        df.attrs["unavailable_metrics"] = unavailable
    df.attrs["units"] = {
        "peak_firing_rate": "Hz",
        "spatial_information": "bits/spike",
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
            Include the fixed-threshold screen columns ``passes_*`` and
            ``cell_type``.

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
        The ``passes_*`` columns and ``cell_type`` are a fixed-threshold
        screen with known false positives; see
        :meth:`SpatialRatesResult.label_cell_types`. For a verdict, call
        :func:`spatial_cell_type_significance` with the raw arrays.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.

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

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
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
            - colorbar_label : str, default="Firing Rate (Hz)"

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
        kwargs.setdefault("colorbar_label", "Firing Rate (Hz)")
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
        return spatial_information(_supported_rates(self), self.occupancy)

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
        return sparsity(_supported_rates(self), self.occupancy)

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
        >>> rate = cell.firing_rate(positions)
        >>> spike_times = generate_poisson_spikes(rate, times, seed=1)
        >>> result = compute_spatial_rate(
        ...     env, spike_times, times, positions, bandwidth=5.0
        ... )
        >>> bool(result.grid_score() > 0.4)
        True
        """
        from neurospatial.encoding.grid import grid_score as gs_func
        from neurospatial.encoding.grid import spatial_autocorrelation

        firing_rate = _to_numpy(_supported_rates(self))
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
        >>> rate = cell.firing_rate(positions)
        >>> spike_times = generate_poisson_spikes(rate, times, seed=1)
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

        firing_rate = _to_numpy(_supported_rates(self))
        autocorr = spatial_autocorrelation(self.env, firing_rate)
        # The autocorrelogram's square pixel side, a length; ``env.bin_sizes``
        # holds cell areas off a track.
        return gp_func(autocorr, pixel_size=autocorrelation_pixel_size(self.env))

    def border_score(
        self,
        threshold: float = 0.3,
        min_area: float | None = None,
        metric: DistanceMetric = "geodesic",
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

        firing_rate = _to_numpy(_supported_rates(self))
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

        Computes what fraction of each region's bins are covered by the
        firing field (bins where firing_rate >= threshold * peak).

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
            Coverage = (region bins in field) / (total region bins).

        Notes
        -----
        This method is useful for:

        - **Border cell analysis**: Determine which wall a border cell prefers
        - **Task-relevant regions**: Check if fields overlap with reward zones
        - **Multi-zone analysis**: Quantify field distribution across zones

        Delegates to ``neurospatial.encoding.border.compute_region_coverage()``.

        See Also
        --------
        border_score : Overall border preference score
        neurospatial.encoding.border.compute_region_coverage : Direct call

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
        from neurospatial.encoding.border import compute_region_coverage

        firing_rate = _to_numpy(_supported_rates(self))

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
        return compute_region_coverage(env, field_bins, regions=regions)

    def has_place_field(
        self,
        *,
        threshold: float = 0.2,
        min_size: int | None = None,
        min_peak_rate: float = 1.0,
        min_secondary_peak_fraction: float = 0.6,
        max_mean_rate: float = np.inf,
        detect_subfields: bool = True,
    ) -> bool:
        """Detect whether the rate map has a place field; this is not a cell-type verdict.

        Runs :func:`detect_place_fields` on this map. Independent 0.5 Hz
        Poisson units with no spatial tuning had a detected
        field in 11, 11, 20, 20 and 16 of 20 units at 1, 2, 5, 10 and 20 minutes
        (30 Hz Ornstein-Uhlenbeck trajectory in a 100 x 100 cm arena, 5 cm bins,
        diffusion KDE with a 5 cm bandwidth). Use
        ``is_place_cell(..., criterion="shuffle")`` for a verdict that controls false
        positives under the circular-shift null.

        Parameters
        ----------
        threshold : float, default=0.2
            Fraction of each field's own peak rate that defines its boundary
            (0-1).
        min_size : int, optional
            Minimum number of bins for a valid field. If None, defaults to 9.
        min_peak_rate : float, default=1.0
            Smallest peak rate (Hz) that can start a field.
        min_secondary_peak_fraction : float, default=0.6
            A field after the first must peak at least this fraction of the
            map's maximum rate.
        max_mean_rate : float, default=np.inf
            Optional upper bound on the unweighted mean of non-NaN spatial-bin
            rates (Hz). The default disables exclusion. This is not the
            recording-time mean firing rate and does not identify cell type.
        detect_subfields : bool, default=True
            If True, split a large field at 0.5 and then 0.7 of its peak when
            that separates two neighboring fields.

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
        firing_rate = _to_numpy(_supported_rates(self))
        fields = detect_place_fields(
            self.env,
            firing_rate,
            threshold=threshold,
            min_size=min_size,
            min_peak_rate=min_peak_rate,
            min_secondary_peak_fraction=min_secondary_peak_fraction,
            max_mean_rate=max_mean_rate,
            detect_subfields=detect_subfields,
        )
        return len(fields) > 0

    def is_place_cell(
        self, *, criterion: Literal["threshold"], min_info: float | None = None
    ) -> bool:
        """Classify one neuron using an explicitly chosen criterion.

        There is no default criterion. ``criterion="threshold"`` thresholds
        plug-in spatial information, which flagged 19/20 and 14/20 untuned
        0.5 Hz Poisson units at 1 and 2 minutes. ``criterion="shuffle"`` tests
        it against circularly shifted spike trains. Use the shuffle when the
        verdict must control false positives under that null (slow), or spatial
        information for a fast screen biased upward at low counts. To detect a
        field, use ``has_place_field()``; field detection also flags most noise units.

        Parameters
        ----------
        criterion : {"threshold"}
            Required screen. Shuffle verdicts require the raw-array free function.
        min_info : float or None, default=None
            Inclusive information cutoff; None resolves to 0.5 bits/spike.

        Returns
        -------
        bool
            Whether spatial information meets the cutoff.

        Notes
        -----
        Plug-in information is biased upward by approximately
        (n_bins - 1) / (2 ln(2) N_spikes). The threshold is a screening heuristic.
        For 20 untuned 0.5 Hz Poisson units in a 100 x 100 cm arena with 5 cm
        bins and diffusion KDE (bandwidth 5), median information at 1, 2, 5,
        10 and 20 minutes was 0.86, 0.54, 0.22, 0.11 and 0.05 bits/spike.
        The 0.5 screen flagged 19/20, 14/20, 0/20, 0/20 and 0/20, respectively.
        Field detection flagged 11, 11, 20, 20 and 16 of 20. For publication,
        report a circular-shift test and its assumptions instead of a cutoff verdict.
        For a shuffle test, call place_cell_significance(...) or
        is_place_cell(..., criterion='shuffle') with the raw arrays; a result
        does not keep the arrays it was computed from.
        """
        if str(criterion) == "shuffle":
            raise ValueError(
                _format_error(
                    "SpatialRateResult.is_place_cell cannot run criterion='shuffle'.",
                    why="Why: a result does not keep the spike times it was computed from",
                    fix="is_place_cell(env, spike_times, times, positions, criterion='shuffle') or place_cell_significance(...)",
                )
            )
        check_criterion(
            criterion, ("threshold",), call="SpatialRateResult.is_place_cell"
        )
        min_info = (
            PLACE_SPATIAL_INFO_THRESHOLDS["min_info"] if min_info is None else min_info
        )
        return bool(_passes_information_screen(self.spatial_information(), min_info))


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
        self, idx: int | None = None, ax: Axes | None = None, **kwargs: Any
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
            - colorbar_label : str, default="Firing Rate (Hz)"

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
            n_units = np.asarray(self.firing_rates).shape[0]
            raise ValueError(
                f"plot() requires a unit index: this batch result holds "
                f"{n_units} units, so there is no single rate map to plot. "
                f"Pass a unit index, e.g. result.plot(0), or iterate the units "
                f"with `for r in result: r.plot()` / `result[i].plot()`."
            )
        rates: NDArray[np.float64] = np.asarray(self.firing_rates)
        kwargs.setdefault("colorbar_label", "Firing Rate (Hz)")
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
        return batch_spatial_information(_supported_rates(self), self.occupancy)

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
        return batch_sparsity(_supported_rates(self), self.occupancy)

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

        return batch_grid_scores(self.env, _to_numpy(_supported_rates(self)))

    def border_scores(
        self,
        threshold: float = 0.3,
        min_area: float | None = None,
        metric: DistanceMetric = "geodesic",
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
            _to_numpy(_supported_rates(self)),
            threshold=threshold,
            min_area=min_area,
            metric=metric,
            boundary_source=boundary_source,
        )

    def label_cell_types(
        self,
        *,
        min_spatial_info: float | None = None,
        min_grid_score: float | None = None,
        min_border_score: float | None = None,
        grid_scores: NDArray[np.float64] | None = None,
        border_scores: NDArray[np.float64] | None = None,
        boundary_source: BoundarySource = "physical",
    ) -> NDArray[np.str_]:
        """Screen neurons for place, grid and border tuning with fixed cutoffs.

        Each label is judged by its own statistic: a grid label needs only a
        grid score at the cutoff and a border label only a border score at the
        cutoff. ``"place"`` is a cell whose spatial information reaches the
        cutoff and that is neither a grid nor a border cell. This is a fast
        screen with known false positives (see Notes); for a verdict, test
        each statistic against circularly shifted spike trains with
        :func:`spatial_cell_type_significance`.

        This is the **multi-class labeler** (returns string labels). It is
        distinct from the single-type :meth:`classify` boolean predicate.

        Parameters
        ----------
        min_spatial_info : float or None, default=None
            Inclusive spatial-information cutoff (bits/spike); None is 0.5.
        min_grid_score : float or None, default=None
            Minimum grid score to be classified as a grid cell; ``None`` uses
            0.4, a common screening convention. On circularly time-shifted
            spike trains of simulated cells 1-8% of shuffled maps reach 0.4,
            so it is a screen, not a verdict; use
            :func:`spatial_cell_type_significance` for a verdict.
        min_border_score : float or None, default=None
            Inclusive border-score cutoff; None is 0.5.
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

            - ``"grid"``, ``"border"``: the grid or border score reaches its cutoff.
            - ``"grid+border"``: both do; the cell is reported as both.
            - ``"place"``: spatial information reaches its cutoff and neither
              the grid nor the border score does.
            - ``"unclassified"``: no statistic reaches its cutoff.
            - ``"undetermined"``: the grid or border score applies to this
              environment but cannot be computed (no regular grid of
              rectangular cells for the grid score; no measured walls, or
              walls without observed bins, for the physical border score), so
              any label is possible. A ``UserWarning`` gives the reason, as
              ``summary_table()`` records it in ``attrs['unavailable_metrics']``.

            Grid and border cells are defined on 2-D maps, so on a 1-D track
            neither applies and a tuned cell is ``"place"``. A neuron whose
            own score is NaN (a silent unit, or no field of the minimum size
            for the border score) does not pass that cutoff.

        Notes
        -----
        The cutoffs are conventions, not calibrated values. The grid cutoff
        0.4 and the border cutoff 0.5 are carried over from rectangular-box
        recordings; 0.5 bits/spike is a common information cutoff. In
        simulated open-field sessions they produce these false positives:

        - Spatial information: plug-in information is biased upward by about
          (n_bins - 1) / (2 ln(2) N_spikes). For 20 untuned 0.5 Hz Poisson
          units in a 100 x 100 cm arena with 5 cm bins and diffusion KDE
          (bandwidth 5), the 0.5 cutoff flagged 19/20, 14/20 and 0/20 at 1, 2
          and 5 minutes.
        - Grid score: about 10% of rate maps from circularly shifted spike
          trains reach 0.4; the 95th percentile of that null was 0.46-0.57
          across square, rectangular and circular arenas, and 11% of place
          cells reached 0.4.
        - Border score: a uniform (untuned) map scores 0.50 in convex arenas
          and 0.58-0.79 on corridor mazes, so untuned cells pass the 0.5
          cutoff often; on a radial-arm maze, 6 of 12 arm place cells did.

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
            name: PLACE_GRID_BORDER_THRESHOLDS[name] if value is None else value
            for name, value in (
                ("min_spatial_info", min_spatial_info),
                ("min_grid_score", min_grid_score),
                ("min_border_score", min_border_score),
            )
        }
        given: dict[str, NDArray[np.float64] | None] = {}
        for name, value in (
            ("grid_scores", grid_scores),
            ("border_scores", border_scores),
        ):
            given[name] = None
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.ndim != 1 or array.shape[0] != len(self):
                    raise ValueError(
                        f"{name} must be a 1-D array of length {len(self)} (one "
                        f"per neuron), got shape {array.shape}"
                    )
                given[name] = array
        scores, unavailable, inapplicable = _screen_scores(
            self, boundary_source=boundary_source, **given
        )
        unknown = {
            name: reason
            for name, reason in unavailable.items()
            if name not in inapplicable
        }
        if unknown:
            reasons = " ".join(f"{name}: {reason}" for name, reason in unknown.items())
            warnings.warn(
                "label_cell_types: "
                + " and ".join(unknown)
                + " cannot be computed on this environment, so cells are "
                f"labeled 'undetermined'. Reason: {reasons}",
                UserWarning,
                stacklevel=2,
            )
        return _cell_type_labels(
            _screen_outcomes(scores, unavailable, inapplicable, **thresholds)
        )

    def classify(self, *, min_info: float | None = None) -> NDArray[np.bool_]:
        """Classify neurons as place cells (single-type boolean predicate).

        A neuron is classified as a place cell if its spatial information
        meets the minimum threshold. This is the single-type boolean
        predicate ("is this a place cell") sibling of
        :meth:`ObjectVectorRatesResult.classify` and
        :meth:`ViewRatesResult.classify`.

        For place-field detection use has_place_field().

        For multi-class labels (``"place"``, ``"grid"``, ``"border"``,
        ``"grid+border"``, ...) use :meth:`label_cell_types` instead.


        Parameters
        ----------
        min_info : float or None, default=None
            Inclusive minimum Skaggs spatial information (bits/spike) to be
            classified as a place cell; None is 0.5 bits/spike.

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
        Plug-in information is biased upward by approximately
        (n_bins - 1) / (2 ln(2) N_spikes). The threshold is a screening heuristic.
        For 20 untuned 0.5 Hz Poisson units in a 100 x 100 cm arena with 5 cm
        bins and diffusion KDE (bandwidth 5), median information at 1, 2, 5,
        10 and 20 minutes was 0.86, 0.54, 0.22, 0.11 and 0.05 bits/spike.
        The 0.5 screen flagged 19/20, 14/20, 0/20, 0/20 and 0/20, respectively.
        Field detection flagged 11, 11, 20, 20 and 16 of 20. For publication,
        report a circular-shift test and its assumptions instead of a cutoff verdict.
        ``min_info=None`` is 0.5 bits/spike.
        For a shuffle test, call place_cell_significance(...)
        with the raw arrays; a result does not keep the arrays it was computed from.
        """
        min_info = (
            PLACE_SPATIAL_INFO_THRESHOLDS["min_info"] if min_info is None else min_info
        )
        return cast(
            "NDArray[np.bool_]",
            _passes_information_screen(self.spatial_information(), min_info),
        )

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
            (``passes_*`` and ``cell_type``, as from ``label_cell_types()``).

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
            - sparsity: sparsity measure (0-1)
            - grid_score: grid score (hexagonal periodicity)
            - border_score: border score (boundary proximity tuning)
            - n_spikes_outside_intervals: spikes outside the analyzed time
            - passes_spatial_information, passes_grid_score,
              passes_border_score (if include_classification=True): whether
              each score reaches its screen cutoff; ``<NA>`` when the score
              applies to this environment but cannot be computed
            - cell_type (if include_classification=True): the
              ``label_cell_types()`` label: ``"grid"``, ``"border"``,
              ``"grid+border"``, ``"place"``, ``"unclassified"`` or
              ``"undetermined"``

        Notes
        -----
        The ``passes_*`` columns and ``cell_type`` are a fixed-threshold
        screen with known false positives; see :meth:`label_cell_types` for
        the cutoffs, their provenance and measured false-positive rates. The
        cutoffs are in ``df.attrs["classification_thresholds"]``. For a
        verdict, call :func:`spatial_cell_type_significance` with the raw
        arrays.
        Physical units are in ``df.attrs["units"]``; the estimator is in
        ``df.attrs["method"]`` and its bandwidth in ``df.attrs["bandwidth"]``.

        This method computes all metrics at once, which may be slow for
        large populations. For selective metric computation, use the
        individual methods (``spatial_information()``, ``grid_scores()``, etc.).

        **Common pandas workflows**:

        - Filter: ``df[df["cell_type"] == "place"]``
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


def _supported_rates(result: SpatialRateResult | SpatialRatesResult) -> Any:
    """Rates with every bin outside the occupancy support set to NaN.

    A filled rate (``fill_value``) is a display or decoding convenience; the
    bin was still unobserved, so metrics must not count it as observed
    firing. JAX rates stay JAX arrays.

    Returns
    -------
    array, shape (n_bins,) or (n_neurons, n_bins)
        ``result``'s rates, NaN where the occupancy support is False.
    """
    from neurospatial.encoding._provenance import occupancy_support

    rates = result._get_rates()
    support = occupancy_support(result)
    if _is_jax_array(rates):
        import jax.numpy as jnp

        return jnp.where(jnp.asarray(support), rates, jnp.nan)
    return np.where(support, np.asarray(rates, dtype=np.float64), np.nan)


def _fill_nan(rates: ArrayLike, fill_value: float) -> ArrayLike:
    """Replace NaN entries of a rate map with ``fill_value``.

    Works for both NumPy and JAX arrays. The masked/low-occupancy bins set
    to NaN by ``min_occupancy`` are the only NaN entries in a rate map, so
    this targets exactly those bins.

    Parameters
    ----------
    rates : ArrayLike
        Firing rate map (NumPy or JAX array), any shape.
    fill_value : float
        Value substituted wherever ``rates`` is NaN.

    Returns
    -------
    ArrayLike
        A new array of the same type/shape with NaN replaced by ``fill_value``.
    """
    # JAX arrays expose .at/.dtype but are not numpy ndarrays; dispatch on
    # whether the object is a numpy array. Both backends support np.isnan via
    # the array's own namespace, but jnp.where keeps the result on-device.
    if isinstance(rates, np.ndarray):
        return np.where(np.isnan(rates), fill_value, rates)
    import jax.numpy as jnp

    rates_jax = cast("Any", rates)
    return cast("ArrayLike", jnp.where(jnp.isnan(rates_jax), fill_value, rates_jax))


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
    from neurospatial.encoding._backend import is_jax_available
    from neurospatial.encoding._glm import _RATE_FLOOR, fit_mrf_gam

    spike_counts = np.asarray(spike_counts, dtype=np.float64)  # (n_units, n_bins)
    occupancy = np.asarray(occupancy, dtype=np.float64)  # (n_bins,)
    n_units, n_bins = spike_counts.shape

    # Reduced-rank penalty basis from occupancy (live-bin order).
    basis = env.get_mrf_basis(occupancy, rank=rank)
    live_bins = np.asarray(basis.live_bins, dtype=np.intp)

    # Dead-component warning is owned here (not by ``fit_mrf_gam``): it needs env's
    # TOTAL component count, which the basis (live-only) does not carry. Warn only
    # when there ARE units and there is at least one live component but fewer than
    # the total: the fully-dead (zero total occupancy) case is already covered by
    # ``fit_mrf_gam``'s own warning, and a no-neuron call has no rates to floor.
    n_components = int(env.get_diffusion_geometry().n_components)
    if n_units and 0 < basis.n_live_components < n_components:
        n_dead = n_components - basis.n_live_components
        warnings.warn(
            f"MRF-GAM fit: {n_dead} of {n_components} environment components were "
            f"never occupied (dead); their bins are set to _RATE_FLOOR "
            f"({_RATE_FLOOR:.0e} Hz). Fit runs on the live bins only.",
            UserWarning,
            # 3 frames out: warn -> _compute_glm_spatial_rates -> the public
            # compute_spatial_rate(s), so it points at the caller's line.
            stacklevel=3,
        )

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
    firing_rates_out: ArrayLike = firing_rates.astype(dtype, copy=False)

    # Return-array-type contract: convert to a JAX array iff the resolved backend
    # is "jax" (resolved via get_backend_name upstream -- NOT a raw backend check),
    # matching what the ratio path returns for backend="jax".
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64
        firing_rates_out = jnp.asarray(firing_rates, dtype=jnp_dtype)

    return firing_rates_out, fit


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
        raise ValueError("time_allocation must be 'start' or 'linear'.")
    # Refuse polar before any default (such as the bandwidth) is resolved.
    if time_allocation == "start":
        require_allocentric(env, f"{context}()")
    else:
        require_linear_allocation(env)


def _resolve_spatial_method(
    env: Environment,
    method: str,
    *,
    bandwidth: float | None,
    min_occupancy: float | None,
    fill_value: float | None,
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
    bandwidth, min_occupancy, fill_value : float or None
        Ratio-method parameters as passed by the caller.
    penalty : float or None
        GLM smoothing penalty as passed by the caller.
    rank : int or None
        GLM basis rank as passed by the caller.
    pooled : bool
        Whether the GLM shares one penalty across units.

    Returns
    -------
    bandwidth : float
        Resolved bandwidth in environment units (``0.0`` for ``"glm"``).
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
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
    )
    # ``pooled`` (glm-only, strict bool): shared vs per-unit lambda. pooled=True
    # with a ratio method is the harmless default; pooled=False with one raises.
    validate_pooled(pooled, method)
    # Resolve the ratio defaults. glm ignores bandwidth/min_occupancy (its branch
    # returns before any smoothing and stamps bandwidth=None on the result).
    resolved_bandwidth = (
        resolve_spatial_bandwidth(env, bandwidth) if method != "glm" else 0.0
    )
    resolved_min_occupancy = (
        0.0 if min_occupancy is None else validate_min_occupancy(min_occupancy)
    )
    if method != "glm":
        _validate_smoothing_parameters(method, resolved_bandwidth)
    return resolved_bandwidth, resolved_min_occupancy, penalty, rank


class _SpatialSupport(NamedTuple):
    """Speed gate, training support and provenance shared by spatial rates."""

    speed: NDArray[np.float64] | None  # (n_time,) resolved gate, or None
    linear_intervals: LinearSpatialIntervals | None  # time_allocation="linear"
    time_support: TimeSupport
    analysis_settings: dict[str, Any]


def _prepare_spatial_support(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    speed: NDArray[np.float64] | None,
    min_speed: float | None,
    max_gap: float | None,
    epochs: Any,
    break_before: NDArray[np.bool_] | None,
    spike_window: Any,
    warn_on_drop: bool,
    time_allocation: Literal["start", "linear"],
    method: str,
    bandwidth: float,
    min_occupancy: float,
    fill_value: float | None,
    penalty: float | None,
    rank: int | None,
    pooled: bool,
    backend: str,
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
    epochs, spike_window
        Recording windows as passed by the caller.
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
    fill_value, penalty : float or None
        Validated fill value and GLM penalty.
    rank : int or None
        Validated GLM basis rank.
    pooled : bool
        Whether the GLM shares one penalty across units.
    backend : str
        Resolved backend recorded in the provenance.

    Returns
    -------
    _SpatialSupport
        The resolved speed gate, linear intervals (``None`` for
        ``time_allocation="start"``), time support and analysis settings.
    """
    from neurospatial.encoding._provenance import rate_analysis_settings
    from neurospatial.encoding._spatial_counts import (
        _resolve_spatial_time_support,
        prepare_linear_spatial_intervals,
        resolve_speed,
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
            epochs=epochs,
            spike_window=spike_window,
            break_before=break_before,
        )
        if time_allocation == "linear"
        else None
    )

    time_support = _resolve_spatial_time_support(
        env,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        spike_window=spike_window,
        warn_on_drop=warn_on_drop,
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
        break_before=break_before,
    )

    settings = rate_analysis_settings(
        time_support,
        min_speed=min_speed,
        speed=speed,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
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
    )


def compute_spatial_rate(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float | None = None,
    fill_value: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
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
          kernel computed from environment graph. Bins too far from all
          occupancy for the smoothing to resolve (typically 4-7 bandwidths
          away) are NaN; so are such bins for ``binned``.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers. Ignores boundaries (mass
          can "bleed through" walls).
        - **binned**: Bin-then-smooth method. Computes raw rate first, then
          smooths. Can introduce discretization artifacts.
        - **glm**: Penalized-Poisson GAM. Occupancy enters as a **log-offset**
          (never a denominator) and the smoothness penalty λ is chosen by REML,
          so the fit returns **finite rates everywhere** -- including
          low-occupancy and unvisited bins where the ratio estimators NaN. Tuned
          with ``penalty`` and ``rank`` (not ``bandwidth`` / ``min_occupancy`` /
          ``fill_value``, which are mutually exclusive with ``method="glm"``).

        Note: ``diffusion_kde`` and ``binned`` smooth matrix-free via the cached
        finite-volume eigenbasis (O(n_bins·rank) per neuron) — they never build a
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
        (Ratio methods only.) Smoothing bandwidth in the same units as bin_size;
        larger values produce more smoothing. ``None`` uses 5 cm in declared metric units (0.05 m / 50 mm),
        or 2.5 median graph-edge lengths for unspecified/custom units.
        Mutually exclusive with ``method="glm"``.
    min_occupancy : float | None, default=None
        (Ratio methods only.) Minimum occupancy (seconds) for a bin to be
        included; bins below the threshold are set to NaN. ``None`` resolves to
        ``0.0`` (no masking). Mutually exclusive with ``method="glm"``.
    fill_value : float | None, default=None
        Ratio methods only: replace missing or masked stored rates with this
        value. None preserves NaN. Filling never changes ``spatial_support``,
        and result metrics still treat filled bins as missing.
        Prefer ``BayesianDecoder.from_rates(rates)`` for population decoding;
        explicit arrays require ``prior=rates.spatial_support.astype(float)``.
        Zero filling without that prior treats unobserved space as silence.
        Mutually exclusive with ``method="glm"``.
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
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
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
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
        If ``True`` (the default), emit a ``UserWarning`` when a large
        fraction of spikes are silently dropped — either because they
        fall outside the position time window or because they map to
        inactive/out-of-environment bins.  A warning is always emitted
        when **all** spikes are dropped (regardless of threshold).  This
        guards against common unit mismatches (e.g. spike_times in
        milliseconds while times is in seconds).  Set to ``False`` to
        suppress all drop-related warnings. Speed-excluded spikes (via
        ``min_speed``) are intentional exclusions and do NOT trigger this
        warning.

    Returns
    -------
    SpatialRateResult
        Result object containing:

        - ``firing_rate``: Firing rate map in Hz, shape (n_bins,); NaN where
          the rate is missing (see ``spatial_support``) unless ``fill_value``
          is given
        - ``occupancy``: Time in each bin in seconds, shape (n_bins,)
        - ``env``: The environment used
        - ``method``: Estimator used
        - ``bandwidth``: Smoothing bandwidth (``None`` for ``method="glm"``)
        - ``spatial_support``: Bins with occupancy at least
          ``min_occupancy`` and a finite rate, shape (n_bins,); unchanged by
          ``fill_value``
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
       (``binned``)

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
    from neurospatial.encoding._backend import (
        is_jax_available,
        validate_and_resolve_backend,
    )
    from neurospatial.encoding._smoothing import smooth_rate_map
    from neurospatial.encoding._spatial_counts import (
        bin_spike_train,
        compute_occupancy,
    )
    from neurospatial.encoding._validation import (
        validate_spike_times,
        validate_trajectory,
    )

    _check_spatial_env(env, time_allocation, context="compute_spatial_rate")

    # Raises ValueError for unknown names and ImportError if jax is unavailable;
    # "auto" resolves to "numpy" or "jax".
    resolved_backend = validate_and_resolve_backend(backend)

    bandwidth, min_occupancy, penalty, rank = _resolve_spatial_method(
        env,
        method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
    )

    # Convert inputs to arrays
    spike_times = np.asarray(spike_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    validate_trajectory(
        times, positions=positions, context="compute_spatial_rate", n_dims=env.n_dims
    )
    validate_spike_times(spike_times, context="compute_spatial_rate")
    breaks = validate_break_before(
        break_before, len(times), call="compute_spatial_rate"
    )

    support = _prepare_spatial_support(
        env,
        times,
        positions,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        spike_window=spike_window,
        warn_on_drop=warn_on_drop,
        time_allocation=time_allocation,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        backend=resolved_backend,
    )
    resolved_speed = support.speed
    linear_intervals = support.linear_intervals
    interval_mask = support.time_support.mask
    resolved_spike_window = support.time_support.spike_window
    settings = support.analysis_settings

    # Bin spike train into spatial bins (always NumPy - CPU/joblib)
    spike_counts = bin_spike_train(
        env,
        spike_times,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        interval_mask=interval_mask,
        context="compute_spatial_rate",
        warn_on_drop=warn_on_drop,
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
    )

    # Compute occupancy (always NumPy) using the SAME resolved speed gate and
    # the SAME max_gap, so the numerator and denominator drop identical
    # intervals.
    occupancy = compute_occupancy(
        env,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        interval_mask=interval_mask,
        context="compute_spatial_rate",
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
    )

    # method="glm": fit the penalized-Poisson GAM (occupancy as a log-offset) and
    # return finite rates everywhere -- no ratio smoothing / min_occupancy / NaN.
    # The single-neuron counts (n_bins,) become a 1-unit unit-major batch.
    if method == "glm":
        glm_firing_rates, fit = _compute_glm_spatial_rates(
            env,
            spike_counts[np.newaxis, :],
            occupancy,
            penalty=penalty,
            rank=rank,
            pooled=pooled,
            resolved_backend=resolved_backend,
            dtype=np.float64,
        )
        # ``glm_firing_rates`` already has the right array type from the helper
        # (JAX for backend="jax", NumPy otherwise); slice row 0 once, preserving
        # that type -- no redundant round-trip. Use dedicated ``ArrayLike`` output
        # variables so the JAX branch does not reassign the NumPy-typed
        # ``occupancy`` (which would not type-check under a JAX-present run).
        single_firing_rate: ArrayLike
        single_occupancy: ArrayLike = occupancy
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            single_firing_rate = jnp.asarray(glm_firing_rates)[0]
            single_occupancy = jnp.asarray(occupancy, dtype=jnp.float64)
        else:
            single_firing_rate = np.asarray(glm_firing_rates)[0]
        # Singular result: UNWRAP the one-element per-unit vectors (pooled=False)
        # to scalars so ``compute_spatial_rate(pooled=False)`` equals
        # ``compute_spatial_rates([spikes], pooled=False)[0]`` field-for-field.
        # ``pooled=True`` fields are already scalars and pass through untouched.
        return SpatialRateResult(
            n_spikes_outside_intervals=int(
                spikes_outside_intervals([spike_times], support.time_support)[0]
            ),
            analysis_settings=settings,
            spike_window=resolved_spike_window,
            firing_rate=single_firing_rate,
            occupancy=single_occupancy,
            env=env,
            method=method,
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

    # Apply smoothing to compute firing rate
    # When backend="jax", uses JAX for the core rate computation
    firing_rate = smooth_rate_map(
        env,
        spike_counts,
        occupancy,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
    )

    # Replace masked/low-occupancy NaN bins with fill_value when requested;
    # None keeps them NaN (missing).
    if fill_value is not None:
        firing_rate = _fill_nan(firing_rate, fill_value)

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rate is already JAX from smooth_rate_map)
    occupancy_out: ArrayLike = occupancy
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy_out = jnp.asarray(occupancy, dtype=jnp.float64)

    # Return result
    return SpatialRateResult(
        n_spikes_outside_intervals=int(
            spikes_outside_intervals([spike_times], support.time_support)[0]
        ),
        min_occupancy=min_occupancy,
        analysis_settings=settings,
        spike_window=resolved_spike_window,
        firing_rate=firing_rate,
        occupancy=occupancy_out,
        env=env,
        method=method,
        bandwidth=bandwidth,
    )


def compute_spatial_rates(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float | None = None,
    fill_value: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
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
        Estimator to use. See ``compute_spatial_rate()`` for details. In addition
        to the three ratio methods, ``method="glm"`` fits a penalized-Poisson GAM
        (occupancy as a log-offset, ``λ`` by REML) and returns finite rates
        everywhere; it is tuned with ``penalty`` / ``rank`` (mutually exclusive
        with ``bandwidth`` / ``min_occupancy`` / ``fill_value``). This function is
        the batched entry point the decoder consumes; ``method="glm"`` flows
        through both the decoder (``decode_session`` / ``BayesianDecoder``) and
        NWB persistence (``write_spatial_rates`` round-trips the GAM diagnostics).
        ``diffusion_kde`` and ``binned`` are
        matrix-free (O(n_bins·rank) per neuron); ``glm`` also avoids a dense
        O(n_bins²) kernel but its penalized-Poisson fit is **not** linear in
        ``rank`` — each Newton step builds and solves a per-unit (r, r) Hessian
        (≈ O(n_units·n_bins·rank²) + O(n_units·rank³)), repeated across Newton
        iterations and REML λ candidates, so keep ``rank`` modest for large
        populations (see ``compute_spatial_rate`` for details). Only
        ``gaussian_kde`` builds a dense O(n_bins²) kernel.
    bandwidth : float | None, default=None
        (Ratio methods only.) Smoothing bandwidth in the same units as bin_size;
        ``None`` uses 5 cm in declared metric units (0.05 m / 50 mm),
        or 2.5 median graph-edge lengths for unspecified/custom units. Mutually exclusive with ``method="glm"``.
    min_occupancy : float | None, default=None
        (Ratio methods only.) Minimum occupancy (seconds) for a bin to be
        included; ``None`` resolves to ``0.0`` (no masking). Mutually exclusive
        with ``method="glm"``.
    fill_value : float | None, default=None
        Ratio methods only: replace missing or masked stored rates with this
        value. None preserves NaN. Filling never changes ``spatial_support``,
        and result metrics still treat filled bins as missing.
        Prefer ``BayesianDecoder.from_rates(rates)`` for population decoding;
        explicit arrays require ``prior=rates.spatial_support.astype(float)``.
        Zero filling without that prior treats unobserved space as silence.
        Mutually exclusive with ``method="glm"``.
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
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
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
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
        Set to ``False`` to suppress all drop-related warnings.
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
    >>> env = Environment.from_samples(positions, bin_size=5.0)

    >>> # Create trajectory timestamps
    >>> times = np.linspace(0, 10, 1000)

    >>> # Spike times for 3 neurons (each sorted ascending)
    >>> spike_times = [
    ...     np.array([1.0, 2.5, 4.0]),  # Neuron 0
    ...     np.array([0.5, 1.5, 2.5, 3.5]),  # Neuron 1
    ...     np.array([5.0, 8.0]),  # Neuron 2
    ... ]

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
    from neurospatial._spikes import as_labelled_spike_trains
    from neurospatial.encoding._backend import (
        is_jax_available,
        validate_and_resolve_backend,
    )
    from neurospatial.encoding._smoothing import smooth_rate_maps_batch
    from neurospatial.encoding._spatial_counts import (
        _warn_if_population_silent,
        bin_spike_trains,
    )
    from neurospatial.encoding._validation import (
        validate_spike_trains,
        validate_trajectory,
    )

    _check_spatial_env(env, time_allocation, context="compute_spatial_rates")

    # Raises ValueError for unknown names and ImportError if jax is unavailable;
    # "auto" resolves to "numpy" or "jax".
    resolved_backend = validate_and_resolve_backend(backend)

    # Validate dtype: only single/double precision rate maps are supported.
    # Wrap the parse so an unparseable dtype string (e.g. "bogus") raises this
    # clean ValueError naming `dtype`, not a raw NumPy
    # ``TypeError: data type 'bogus' not understood``.
    _dtype_msg = (
        f"dtype must be np.float32 or np.float64, got {dtype!r}. "
        "Only single- and double-precision rate maps are supported "
        "(float32 halves the (n_units, n_bins) storage and the downstream "
        "decode working set)."
    )
    try:
        _resolved_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(_dtype_msg) from exc
    if _resolved_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(_dtype_msg)
    # Normalize to the canonical numpy scalar type for downstream casts.
    dtype = np.float32 if _resolved_dtype == np.dtype(np.float32) else np.float64

    bandwidth, min_occupancy, penalty, rank = _resolve_spatial_method(
        env,
        method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
    )

    # Normalize spike times to canonical list-of-arrays format and resolve the
    # per-unit identity labels (defaults to arange). A labelled input keeps its
    # own labels; a differing unit_ids= raises.
    spike_times_list, resolved_unit_ids, labels_supplied = as_labelled_spike_trains(
        spike_times, unit_ids, context="compute_spatial_rates"
    )
    n_neurons = len(spike_times_list)
    result_unit_ids = resolved_unit_ids if labels_supplied else None

    # Convert inputs to arrays
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    validate_trajectory(
        times, positions=positions, context="compute_spatial_rates", n_dims=env.n_dims
    )
    validate_spike_trains(spike_times_list, context="compute_spatial_rates")
    breaks = validate_break_before(
        break_before, len(times), call="compute_spatial_rates"
    )

    # The speed gate is resolved once for the population, so a gate that
    # excludes every interval warns once rather than per neuron.
    support = _prepare_spatial_support(
        env,
        times,
        positions,
        speed=speed,
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=epochs,
        break_before=breaks,
        spike_window=spike_window,
        warn_on_drop=warn_on_drop,
        time_allocation=time_allocation,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
        penalty=penalty,
        rank=rank,
        pooled=pooled,
        backend=resolved_backend,
    )
    resolved_speed = support.speed
    linear_intervals = support.linear_intervals
    interval_mask = support.time_support.mask
    resolved_epochs = support.time_support.epochs
    resolved_spike_window = support.time_support.spike_window
    settings = support.analysis_settings

    # Recording coverage uses tracked runs, independently of speed or frame bins.
    _warn_if_population_silent(
        spike_times_list,
        times,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=breaks,
    )

    # method="glm": fit the penalized-Poisson GAM (occupancy as a log-offset).
    # Handles the no-neurons case too (fit_mrf_gam returns an (r_eff, 0) fit), so
    # it sits before the ratio no-neurons short-circuit and returns early, leaving
    # the ratio path below untouched.
    if method == "glm":
        from neurospatial.encoding._spatial_counts import (
            bin_spike_trains as _bin_spike_trains,
        )
        from neurospatial.encoding._spatial_counts import (
            compute_occupancy as _compute_occupancy,
        )

        if n_neurons == 0:
            spike_counts = np.zeros((0, env.n_bins), dtype=np.float64)
            occupancy = _compute_occupancy(
                env,
                times,
                positions,
                speed=resolved_speed,
                min_speed=min_speed,
                max_gap=max_gap,
                interval_mask=interval_mask,
                context="compute_spatial_rates",
                time_allocation=time_allocation,
                linear_intervals=linear_intervals,
            )
        else:
            spike_counts, occupancy = _bin_spike_trains(
                env,
                spike_times_list,
                times,
                positions,
                speed=resolved_speed,
                min_speed=min_speed,
                max_gap=max_gap,
                interval_mask=interval_mask,
                n_jobs=n_jobs,
                warn_on_drop=warn_on_drop,
                time_allocation=time_allocation,
                linear_intervals=linear_intervals,
            )

        glm_firing_rates, fit = _compute_glm_spatial_rates(
            env,
            spike_counts,
            occupancy,
            penalty=penalty,
            rank=rank,
            pooled=pooled,
            # Resolved labels flow in so a per-unit convergence / boundary warning
            # names the caller's unit_ids, not bare unit-axis indices.
            unit_ids=resolved_unit_ids,
            resolved_backend=resolved_backend,
            dtype=dtype,
        )
        # Dedicated ArrayLike output var so the JAX branch does not reassign the
        # NumPy-typed ``occupancy`` (which would not type-check JAX-present).
        batch_occupancy: ArrayLike = occupancy
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            batch_occupancy = jnp.asarray(occupancy, dtype=jnp.float64)
        # The per-unit lambda fields (``penalty`` / ``reml_objective`` /
        # ``reml_at_boundary`` vectors, ``penalty_selected_by_reml`` mask) carry
        # straight through from the fit; they are scalar/None under pooled=True.
        return SpatialRatesResult(
            n_spikes_outside_intervals=spikes_outside_intervals(
                spike_times_list, support.time_support
            ),
            analysis_settings=settings,
            unit_table=getattr(spike_times, "unit_table", None),
            spike_window=resolved_spike_window,
            firing_rates=glm_firing_rates,
            occupancy=batch_occupancy,
            env=env,
            method=method,
            bandwidth=None,
            unit_ids=result_unit_ids,
            # dtype governs the (n_units, n_bins) rate-map storage only. The GLM
            # diagnostics are the float64 fit result and are kept float64 -- so
            # they do not lose precision (deviance/coefficients) and rates[i]
            # matches compute_spatial_rate, whose diagnostics are always float64.
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

    # Handle edge case: no neurons
    # Still compute occupancy from trajectory (occupancy is independent of neural data)
    if n_neurons == 0:
        from neurospatial.encoding._spatial_counts import compute_occupancy

        # Use compute_occupancy which handles 1D position reshaping
        occupancy = compute_occupancy(
            env,
            times,
            positions,
            speed=resolved_speed,
            min_speed=min_speed,
            max_gap=max_gap,
            interval_mask=interval_mask,
            context="compute_spatial_rates",
            time_allocation=time_allocation,
            linear_intervals=linear_intervals,
        )

        # Convert to JAX if needed
        firing_rates_result: ArrayLike = np.empty((0, env.n_bins), dtype=dtype)
        occupancy_result: ArrayLike = occupancy
        if resolved_backend == "jax" and is_jax_available():
            import jax.numpy as jnp

            jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64
            firing_rates_result = jnp.asarray(firing_rates_result, dtype=jnp_dtype)
            occupancy_result = jnp.asarray(occupancy, dtype=jnp.float64)

        return SpatialRatesResult(
            n_spikes_outside_intervals=spikes_outside_intervals(
                spike_times_list, support.time_support
            ),
            min_occupancy=min_occupancy,
            analysis_settings=settings,
            unit_table=getattr(spike_times, "unit_table", None),
            spike_window=resolved_spike_window,
            firing_rates=firing_rates_result,
            occupancy=occupancy_result,
            env=env,
            method=method,
            bandwidth=bandwidth,
            unit_ids=result_unit_ids,
        )

    # Bin spike trains and compute occupancy (always NumPy - CPU/joblib)
    # bin_spike_trains returns (spike_counts, occupancy)
    spike_counts, occupancy = bin_spike_trains(
        env,
        spike_times_list,
        times,
        positions,
        speed=resolved_speed,
        min_speed=min_speed,
        max_gap=max_gap,
        interval_mask=interval_mask,
        n_jobs=n_jobs,
        warn_on_drop=warn_on_drop,
        time_allocation=time_allocation,
        linear_intervals=linear_intervals,
    )

    # Apply batch smoothing to compute firing rates
    # When backend="jax", uses JAX for the core rate computation
    firing_rates = smooth_rate_maps_batch(
        env,
        spike_counts,
        occupancy,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        backend=resolved_backend,
        dtype=dtype,
    )

    # Replace masked/low-occupancy NaN bins with fill_value when requested;
    # None keeps them NaN (missing).
    if fill_value is not None:
        firing_rates = _fill_nan(firing_rates, fill_value)

    # Defensive final cast to the requested dtype. Under NEP 50,
    # np.where(mask, 0.0, f32_array) keeps float32, but this guarantees the
    # stored dtype regardless of the fill_value / backend path.
    if isinstance(firing_rates, np.ndarray):
        firing_rates = firing_rates.astype(dtype, copy=False)
    elif is_jax_available():
        import jax.numpy as jnp

        jnp_dtype = jnp.float32 if dtype is np.float32 else jnp.float64
        firing_rates = jnp.asarray(firing_rates, dtype=jnp_dtype)

    # Convert occupancy to JAX if JAX backend is selected
    # (firing_rates is already JAX from smooth_rate_maps_batch)
    occupancy_out: ArrayLike = occupancy
    if resolved_backend == "jax" and is_jax_available():
        import jax.numpy as jnp

        occupancy_out = jnp.asarray(occupancy, dtype=jnp.float64)

    # Return result
    return SpatialRatesResult(
        n_spikes_outside_intervals=spikes_outside_intervals(
            spike_times_list, support.time_support
        ),
        min_occupancy=min_occupancy,
        analysis_settings=settings,
        unit_table=getattr(spike_times, "unit_table", None),
        spike_window=resolved_spike_window,
        firing_rates=firing_rates,
        occupancy=occupancy_out,
        env=env,
        method=method,
        bandwidth=bandwidth,
        unit_ids=result_unit_ids,
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
        correlation is computed.

        Parameters
        ----------
        label_a, label_b : hashable
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Pearson correlation coefficient in ``[-1, 1]``. Returns ``nan``
            if fewer than two finite overlapping bins exist or if either map
            has zero variance over the overlap.

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
                raise KeyError(
                    f"Unknown direction label {label!r}. Known labels: {tuple(self.firing_rates)}."
                    + f" Fix: pass one of {tuple(self.firing_rates)}."
                )
        a = _to_numpy(self.firing_rates[label_a]).ravel()
        b = _to_numpy(self.firing_rates[label_b]).ravel()
        finite = np.isfinite(a) & np.isfinite(b)
        if int(finite.sum()) < 2:
            return float("nan")
        a, b = a[finite], b[finite]
        if a.std() == 0.0 or b.std() == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def directionality_index(self, label_a: Hashable, label_b: Hashable) -> float:
        """Per-bin directionality index between two directions.

        Returns the mean absolute normalized rate difference between the two
        directions across bins:

        .. math::

            \\mathrm{DI} = \\mathrm{mean}_i
                \\frac{|r^a_i - r^b_i|}{r^a_i + r^b_i}

        Values near ``0`` indicate direction-independent firing; values near
        ``1`` indicate strongly direction-selective firing. Bins where either
        map is NaN, or where both rates are zero, are excluded.

        Parameters
        ----------
        label_a, label_b : hashable
            Direction labels to compare. Must be present in ``labels``.

        Returns
        -------
        float
            Mean directionality index in ``[0, 1]``. Returns ``nan`` if no
            bin has a positive summed rate.

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
        >>> bool(np.isclose(result.directionality_index("fwd", "rev"), 0.0))
        True
        """
        for label in (label_a, label_b):
            if label not in self.firing_rates:
                raise KeyError(
                    f"Unknown direction label {label!r}. Known labels: {tuple(self.firing_rates)}."
                    + f" Fix: pass one of {tuple(self.firing_rates)}."
                )
        a = _to_numpy(self.firing_rates[label_a]).ravel()
        b = _to_numpy(self.firing_rates[label_b]).ravel()
        total = a + b
        valid = np.isfinite(a) & np.isfinite(b) & (total > 0.0)
        if not valid.any():
            return float("nan")
        di = np.abs(a[valid] - b[valid]) / total[valid]
        return float(np.mean(di))

    def summary(self) -> dict[str, Any]:
        """Scalar headline metrics across directions.

        Returns
        -------
        dict
            Mapping with ``n_directions`` (int), ``n_bins`` (int), the
            analysis settings ``spike_window``, ``spike_window_assumed`` and
            ``missing_label``, and one ``peak_<label>`` entry per direction
            giving that direction's peak firing rate (Hz). When exactly two
            directions are present, a ``correlation`` entry (Pearson r between
            the two maps) is also included.

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
            "n_directions": len(self.labels),
            "spike_window_assumed": self.spike_window_assumed,
            "spike_window": None
            if self.spike_window is None
            else self.spike_window.tolist(),
            "missing_label": self.missing_label,
            "n_bins": int(self.env.n_bins),
        }
        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label])
            out[f"peak_{label}"] = float(np.nanmax(rate))
        if len(self.labels) == 2:
            out["correlation"] = self.correlation(self.labels[0], self.labels[1])
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy/long-form table: one row per (direction, bin).

        Stacks the per-direction rate maps into long form with a ``direction``
        identifier column, so directional results ``pandas.concat`` cleanly
        with other tidy result tables. Bin-center coordinates are emitted as
        ``coord_0``, ``coord_1``, ... columns.

        Returns
        -------
        pandas.DataFrame
            Long-form table with columns ``direction`` (the label), ``bin`` (int),
            ``coord_0`` ... (float), ``firing_rate`` (float, Hz), and
            ``occupancy`` (float, seconds).

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
        >>> df = result.to_dataframe()
        >>> sorted(df["direction"].unique())
        ['fwd', 'rev']
        >>> len(df) == 2 * n
        True
        """
        import pandas as pd

        bin_centers = np.asarray(self.env.bin_centers)
        if bin_centers.ndim == 1:
            bin_centers = bin_centers[:, None]
        n_bins, n_dims = bin_centers.shape

        frames: list[pd.DataFrame] = []
        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label]).ravel()
            occ = _to_numpy(self.occupancy[label]).ravel()
            direction = np.empty(n_bins, dtype=object)
            direction[:] = [label] * n_bins
            data: dict[str, Any] = {
                "direction": direction,
                "bin": np.arange(n_bins, dtype=np.int64),
            }
            for d in range(n_dims):
                data[f"coord_{d}"] = bin_centers[:, d]
            data["firing_rate"] = rate
            data["occupancy"] = occ
            frames.append(pd.DataFrame(data))
        if not frames:
            # No labelled directions (e.g. every label passed to
            # compute_directional_place_fields was missing, leaving
            # labels == ()). Return an empty frame with the documented column
            # schema instead of letting pd.concat([]) raise "No objects to
            # concatenate".
            empty: dict[str, NDArray[Any]] = {
                "direction": np.array([], dtype=object),
                "bin": np.array([], dtype=np.int64),
            }
            for d in range(n_dims):
                empty[f"coord_{d}"] = np.array([], dtype=np.float64)
            empty["firing_rate"] = np.array([], dtype=np.float64)
            empty["occupancy"] = np.array([], dtype=np.float64)
            return pd.DataFrame(empty)
        return pd.concat(frames, ignore_index=True)

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> Axes:
        """Overlay the per-direction firing rate maps on a single axis.

        Plots each direction's firing rate against bin index as one line,
        producing one artist per direction. This is most informative for
        1-D / linearized environments where bin index maps to track position.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, a new figure and axes are created.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot`` for each
            direction's line.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the overlay (one line per direction).

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")  # non-interactive backend for doctest
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
        >>> ax = result.plot()
        >>> len(ax.get_lines())
        2
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        for label in self.labels:
            rate = _to_numpy(self.firing_rates[label]).ravel()
            ax.plot(np.arange(rate.shape[0]), rate, label=str(label), **kwargs)

        ax.set_xlabel("Bin index")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title("Directional place fields")
        ax.legend()
        return ax


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
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    missing_label: Hashable | None = None,
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
        Labels from ``heading_direction_labels`` include ``"stationary"``, an
        ordinary label here that gets its own map; pass
        ``missing_label="stationary"`` to drop it.
    method : {"diffusion_kde", "gaussian_kde", "binned"}, default="diffusion_kde"
        Estimation method passed through to the place-field helper.
    bandwidth : float or None, default=None
        Smoothing bandwidth in environment units (e.g., cm).
        None uses 5 cm in declared metric units, or 2.5 median graph-edge
        lengths for unspecified/custom units.
    min_occupancy : float, default=0.0
        Minimum occupancy threshold in seconds. Bins below this threshold are
        set to NaN.

    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
          analyzable time (it occurs only at the last sample, or outside
          ``epochs``) has zero occupancy and an all-NaN rate map.

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
    >>> spike_times = np.sort(rng.uniform(0, 100, 50))  # spikes must be sorted
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
    from neurospatial._intervals import (
        intersect_intervals,
        resolve_time_windows,
    )
    from neurospatial.encoding._spatial_counts import require_allocentric

    require_allocentric(env, "compute_directional_place_fields()")

    if len(direction_labels) != len(times):
        raise ValueError(
            f"direction_labels must have same length as times, "
            f"got {len(direction_labels)} and {len(times)}"
        )
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

    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
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

        single = compute_spatial_rate(
            env,
            spike_times,
            times,
            positions,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            max_gap=max_gap,
            epochs=windows,
            break_before=breaks,
            spike_window=resolved_spike_window,
        )
        firing_rates_dict[label] = np.asarray(single.firing_rate, dtype=np.float64)
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
    firing_rate: NDArray[np.float64],
    *,
    threshold: float = 0.2,
    min_size: int | None = None,
    min_peak_rate: float = 1.0,
    min_secondary_peak_fraction: float = 0.6,
    max_mean_rate: float = np.inf,
    detect_subfields: bool = True,
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
    firing_rate : array, shape (n_bins,)
        Firing rate map (Hz) from neuron. NaN bins (unvisited or below the
        occupancy threshold) never belong to a field and do not connect
        fields.
    threshold : float, default=0.2
        Fraction of each field's own peak rate that defines its boundary
        (0-1). 0.2 is the neurocode ``MapStats`` default [1]_.
    min_size : int, optional
        Minimum number of bins for a valid field. If None, defaults to 9 bins
        (3 x 3 bins in 2-D).
    min_peak_rate : float, default=1.0
        Smallest peak rate (Hz) that can start a field. 1 Hz is the minimum
        peak in neurocode's ``MapStats`` and its place-cell pipeline [1]_
        (also the FMAToolbox and opexebo default). Must be finite and >= 0.
    min_secondary_peak_fraction : float, default=0.6
        A field after the first must peak at least this fraction of the
        map's maximum rate. 0.6 is neurocode's ``minPeak2nd`` [1]_. Pass
        0.0 to keep every peak above ``min_peak_rate``. Must be in [0, 1].
    max_mean_rate : float, default=np.inf
        Optional upper bound on the unweighted mean of non-NaN spatial-bin
        rates (Hz). The default disables exclusion. This is not the
        recording-time mean firing rate and does not identify cell type.
    detect_subfields : bool, default=True
        If True, try to split each field larger than ``2 * min_size`` bins
        into separate fields at 0.5 and then 0.7 of its peak. This separates
        two neighboring fields that merge at the ``threshold`` level.

    Returns
    -------
    PlaceFieldsResult
        Detected fields, one bin-index array each, in order of decreasing
        peak rate, plus ``excluded_reason``, which says whether the
        ``max_mean_rate`` filter excluded the map. The result is iterable,
        indexable and sized like a list of fields.

    Raises
    ------
    ValueError
        If ``firing_rate`` does not have one value per bin, or a parameter is
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
    4. **Keep or discard**: mark those bins as assigned. Keep the field only
       if it has at least ``min_size`` bins and its peak bin is a local
       maximum of the full map (no neighbor has a higher rate). A candidate
       that rises toward a neighbor is the flank of a field already assigned,
       left behind when that field was cut at its own threshold. This is the
       n-D form of neurocode's 1-D check that the rate dips below threshold
       between two fields, and of the ``localmax`` option of ``MapStats``.
    5. **Subfields**: if ``detect_subfields`` is True and the field has more
       than ``2 * min_size`` bins, threshold it at 0.5 and then 0.7 of its
       peak. The first level that leaves two or more connected parts of at
       least ``min_size`` bins replaces the field by those parts. This is one
       pass over two levels, not a recursion; bins of the field outside
       every part are assigned to no field.
    6. Repeat from step 2 until a peak fails the floor or every finite bin
       is assigned.

    The floor in step 2 stops the search on noise. Without it, each
    successive threshold is a fraction of a smaller peak, so small bumps far
    below the main field would be reported as fields.

    **Segmentation is not cell classification.** An unmodulated or high-rate
    map can contain detected regions. An explicit finite ``max_mean_rate``
    filters the unweighted mean of spatial-bin rates; unequal occupancy makes
    that different from total spikes divided by recording duration. Neither
    the filter nor the resulting field count establishes cell identity.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import detect_place_fields
    >>> xx, yy = np.meshgrid(np.arange(-20.0, 21.0, 2.0), np.arange(-20.0, 21.0, 2.0))
    >>> env = Environment.from_samples(np.c_[xx.ravel(), yy.ravel()], bin_size=2.0)
    >>> # An 8 Hz Gaussian place field (3 cm width) at the center
    >>> distance = np.linalg.norm(env.bin_centers, axis=1)
    >>> firing_rate = 8.0 * np.exp(-(distance**2) / (2 * 3.0**2))
    >>> fields = detect_place_fields(env, firing_rate)
    >>> len(fields)
    1

    See Also
    --------
    SpatialRateResult.has_place_field : Whether a result's map has a field
    neurospatial.encoding.population.population_coverage : Fields across a population

    References
    ----------
    .. [1] AyA Lab neurocode toolbox, ``PlaceCells/MapStats.m`` and
           ``PlaceCells/findPlaceFieldsAvg1D.m``,
           https://github.com/ayalab1/neurocode
    """
    firing_rate = np.asarray(firing_rate, dtype=np.float64)
    if firing_rate.shape != (env.n_bins,):
        raise ValueError(
            f"firing_rate shape {firing_rate.shape} does not match "
            f"env.n_bins ({env.n_bins})"
        )

    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if not (np.isfinite(min_peak_rate) and min_peak_rate >= 0):
        raise ValueError(
            _format_error(
                f"min_peak_rate must be a finite rate >= 0 Hz, got {min_peak_rate}.",
                why="Why: it is the smallest peak rate that can start a field",
                fix="pass min_peak_rate=1.0 (the neurocode default) or another rate in Hz",
            )
        )
    if not 0 <= min_secondary_peak_fraction <= 1:
        raise ValueError(
            _format_error(
                "min_secondary_peak_fraction must be in [0, 1], got "
                f"{min_secondary_peak_fraction}.",
                why="Why: it is a fraction of the map's maximum rate",
                fix="pass 0.6 (the neurocode default) or 0.0 to keep every peak above min_peak_rate",
            )
        )

    if min_size is None:
        min_size = 9  # 3 x 3 bins in 2-D

    finite = np.isfinite(firing_rate)
    if not np.any(finite):
        return PlaceFieldsResult(fields=[])

    # A finite cutoff is an explicit map-level filter, independent of cell type.
    if np.isfinite(max_mean_rate):
        mean_rate = np.nanmean(firing_rate)
        if mean_rate > max_mean_rate:
            warnings.warn(
                f"detect_place_fields: map excluded by the spatial-map-mean filter "
                f"(unweighted mean {float(mean_rate):.2f} Hz > max_mean_rate "
                f"{max_mean_rate} Hz). Returning empty field list. This is not "
                "a recording-time mean or a cell-type verdict. Pass "
                "max_mean_rate=np.inf to disable the filter.",
                UserWarning,
                stacklevel=2,
            )
            return PlaceFieldsResult(
                fields=[], excluded_reason="mean_rate_above_threshold"
            )

    min_peak = max(
        min_peak_rate, min_secondary_peak_fraction * float(np.max(firing_rate[finite]))
    )
    # Unassigned bins keep their rate; assigned and missing bins are NaN.
    remaining = np.where(finite, firing_rate, np.nan)
    fields: list[NDArray[np.int64]] = []
    while np.any(np.isfinite(remaining)):
        peak_idx = int(np.nanargmax(remaining))
        peak_rate = float(remaining[peak_idx])
        if peak_rate <= 0 or peak_rate < min_peak:
            break

        # NaN compares False, so assigned and missing bins are excluded.
        above_threshold = remaining >= peak_rate * threshold
        field_bins = _extract_connected_component(peak_idx, above_threshold, env)
        remaining[field_bins] = np.nan
        if len(field_bins) < min_size or _rises_toward_a_neighbor(
            peak_idx, peak_rate, firing_rate, env
        ):
            continue

        if detect_subfields and len(field_bins) > min_size * 2:
            fields.extend(
                _detect_subfields(
                    firing_rate[field_bins], field_bins, peak_rate, env, min_size
                )
            )
        else:
            fields.append(field_bins)

    return PlaceFieldsResult(fields=fields)


def has_place_field(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    threshold: float = 0.2,
    min_size: int | None = None,
    min_peak_rate: float = 1.0,
    min_secondary_peak_fraction: float = 0.6,
    max_mean_rate: float = np.inf,
    detect_subfields: bool = True,
    min_occupancy: float | None = None,
    fill_value: float | None = None,
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
    :func:`detect_place_fields` on it. Independent 0.5 Hz Poisson units with
    no spatial tuning had a detected
    field in 11, 11, 20, 20 and 16 of 20 units at 1, 2, 5, 10 and 20 minutes
    (30 Hz Ornstein-Uhlenbeck trajectory in a 100 x 100 cm arena, 5 cm bins,
    diffusion KDE with a 5 cm bandwidth). Use
    ``is_place_cell(..., criterion="shuffle")`` for a verdict that controls false
    positives under the circular-shift null.

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
        Smoothing bandwidth in environment units.
        None uses 5 cm in declared metric units, or 2.5 median graph-edge
        lengths for unspecified/custom units.
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
    min_size : int, optional
        Minimum number of bins for a valid field. If None, defaults to 9.
    min_peak_rate : float, default=1.0
        Smallest peak rate (Hz) that can start a field.
    min_secondary_peak_fraction : float, default=0.6
        A field after the first must peak at least this fraction of the map's
        maximum rate.
    max_mean_rate : float, default=np.inf
        Optional upper bound on the unweighted mean of non-NaN spatial-bin
        rates (Hz). The default disables exclusion. This is not the
        recording-time mean firing rate and does not identify cell type.
    detect_subfields : bool, default=True
        If True, split a large field at 0.5 and then 0.7 of its peak when that
        separates two neighboring fields.

    min_occupancy : float | None, default=None
        (Ratio methods only.) Minimum occupancy (seconds) for a bin to be
        included; bins below the threshold are set to NaN. ``None`` resolves to
        ``0.0`` (no masking). Mutually exclusive with ``method="glm"``.
    fill_value : float | None, default=None
        Ratio methods only: replace missing or masked stored rates with this
        value. None preserves NaN. Filling never changes ``spatial_support``,
        and result metrics still treat filled bins as missing.
        Prefer ``BayesianDecoder.from_rates(rates)`` for population decoding;
        explicit arrays require ``prior=rates.spatial_support.astype(float)``.
        Zero filling without that prior treats unobserved space as silence.
        Mutually exclusive with ``method="glm"``.
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
        If ``True`` (the default), emit a ``UserWarning`` when a large
        fraction of spikes are silently dropped — either because they
        fall outside the position time window or because they map to
        inactive/out-of-environment bins.  A warning is always emitted
        when **all** spikes are dropped (regardless of threshold).  This
        guards against common unit mismatches (e.g. spike_times in
        milliseconds while times is in seconds).  Set to ``False`` to
        suppress all drop-related warnings. Speed-excluded spikes (via
        ``min_speed``) are intentional exclusions and do NOT trigger this
        warning.


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

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding.spatial import has_place_field
    >>> rng = np.random.default_rng(0)
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
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
        fill_value=fill_value,
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
        min_size=min_size,
        min_peak_rate=min_peak_rate,
        min_secondary_peak_fraction=min_secondary_peak_fraction,
        max_mean_rate=max_mean_rate,
        detect_subfields=detect_subfields,
    )


def is_place_cell(
    env: Environment,
    spike_times: NDArray[np.float64],
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    criterion: Literal["threshold", "shuffle"],
    min_info: float | None = None,
    alpha: float | None = None,
    n_shuffles: int | None = None,
    min_shift: float | None = None,
    rng: np.random.Generator | int | None = None,
    unit_id: Hashable | None = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float | None = None,
    fill_value: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
    spike_window: Any = None,
    backend: BackendType = "numpy",
    warn_on_drop: bool = True,
) -> bool:
    """Classify one neuron using an explicitly chosen criterion.

    There is no default criterion. ``criterion="threshold"`` thresholds
    plug-in spatial information, which flagged 19/20 and 14/20 untuned
    0.5 Hz Poisson units at 1 and 2 minutes. ``criterion="shuffle"`` tests
    it against circularly shifted spike trains. Use the shuffle when the
    verdict must control false positives under that null (slow), or spatial
    information for a fast screen biased upward at low counts. To detect a
    field, use ``has_place_field()``; field detection also flags most noise units.

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

    criterion : {"threshold", "shuffle"}
        Screen the observed statistic or test circular-shift significance.
    min_info : float or None, default=None
        Inclusive spatial-information cutoff for ``criterion="threshold"``
        (bits/spike); None is 0.5 bits/spike.
    alpha : float or None, default=None
        P-value level for ``criterion="shuffle"``; None resolves to 0.05.
    n_shuffles : int or None, default=None
        Number of circular shifts in shuffle mode; None resolves to 1000.
    min_shift : float or None, default=None
        Minimum shift in analyzed seconds; None resolves to 20.0.
    rng : numpy.random.Generator, int or None, default=None
        Shuffle random source; an integer seed with the same unit label is stable
        across single and population calls.
    unit_id : hashable or None, default=None
        Shuffle stream label; None uses label 0. Match the population's label.
    method : {"diffusion_kde", "gaussian_kde", "binned", "glm"}, default="diffusion_kde"
        Estimator to use:

        - **diffusion_kde** (recommended): Graph-based boundary-aware KDE.
          Respects environment boundaries (walls, obstacles). Uses diffusion
          kernel computed from environment graph. Bins too far from all
          occupancy for the smoothing to resolve (typically 4-7 bandwidths
          away) are NaN; so are such bins for ``binned``.
        - **gaussian_kde**: Standard Euclidean KDE. Uses Gaussian kernel based
          on Euclidean distance between bin centers. Ignores boundaries (mass
          can "bleed through" walls).
        - **binned**: Bin-then-smooth method. Computes raw rate first, then
          smooths. Can introduce discretization artifacts.
        - **glm**: Penalized-Poisson GAM. Occupancy enters as a **log-offset**
          (never a denominator) and the smoothness penalty λ is chosen by REML,
          so the fit returns **finite rates everywhere** -- including
          low-occupancy and unvisited bins where the ratio estimators NaN. Tuned
          with ``penalty`` and ``rank`` (not ``bandwidth`` / ``min_occupancy`` /
          ``fill_value``, which are mutually exclusive with ``method="glm"``).

        Note: ``diffusion_kde`` and ``binned`` smooth matrix-free via the cached
        finite-volume eigenbasis (O(n_bins·rank) per neuron) — they never build a
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
        (Ratio methods only.) Smoothing bandwidth in the same units as bin_size;
        larger values produce more smoothing. ``None`` uses 5 cm in declared metric units (0.05 m / 50 mm),
        or 2.5 median graph-edge lengths for unspecified/custom units.
        Mutually exclusive with ``method="glm"``.
    min_occupancy : float | None, default=None
        (Ratio methods only.) Minimum occupancy (seconds) for a bin to be
        included; bins below the threshold are set to NaN. ``None`` resolves to
        ``0.0`` (no masking). Mutually exclusive with ``method="glm"``.
    fill_value : float | None, default=None
        Ratio methods only: replace missing or masked stored rates with this
        value. None preserves NaN. Filling never changes ``spatial_support``,
        and result metrics still treat filled bins as missing.
        Prefer ``BayesianDecoder.from_rates(rates)`` for population decoding;
        explicit arrays require ``prior=rates.spatial_support.astype(float)``.
        Zero filling without that prior treats unobserved space as silence.
        Mutually exclusive with ``method="glm"``.
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
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
        If ``True`` (the default), emit a ``UserWarning`` when a large
        fraction of spikes are silently dropped — either because they
        fall outside the position time window or because they map to
        inactive/out-of-environment bins.  A warning is always emitted
        when **all** spikes are dropped (regardless of threshold).  This
        guards against common unit mismatches (e.g. spike_times in
        milliseconds while times is in seconds).  Set to ``False`` to
        suppress all drop-related warnings. Speed-excluded spikes (via
        ``min_speed``) are intentional exclusions and do NOT trigger this
        warning.


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
    Plug-in information is biased upward by approximately
    (n_bins - 1) / (2 ln(2) N_spikes). The threshold is a screening heuristic.
    For 20 untuned 0.5 Hz Poisson units in a 100 x 100 cm arena with 5 cm
    bins and diffusion KDE (bandwidth 5), median information at 1, 2, 5,
    10 and 20 minutes was 0.86, 0.54, 0.22, 0.11 and 0.05 bits/spike.
    The 0.5 screen flagged 19/20, 14/20, 0/20, 0/20 and 0/20, respectively.
    Field detection flagged 11, 11, 20, 20 and 16 of 20. For publication,
    report a circular-shift test and its assumptions instead of a cutoff verdict.

    Circular shifting costs about n_shuffles recomputes of the plural map.
    Its null assumes stable firing statistics on the joined analyzed clock;
    recording gaps and excluded epochs are never shift destinations.
    Compare p_value < alpha; a significant association alone does not establish
    cell identity. Results keep no raw arrays or recompute closures.
    Threshold keywords belong only to the screen; shuffle keywords belong
    only to the shuffle. Passing a keyword for the other mode raises.

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
    >>> positions = rng.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(positions, bin_size=5.0)
    >>> times = np.linspace(0, 40, 1000)
    >>> spike_times = np.sort(rng.uniform(0, 40, 50))
    >>> result = is_place_cell(
    ...     env, spike_times, times, positions, criterion="threshold"
    ... )
    >>> type(result)
    <class 'bool'>
    """
    check_criterion(criterion, ("threshold", "shuffle"), call="is_place_cell")
    check_mode_keywords(
        criterion,
        threshold={"min_info": min_info},
        shuffle={
            "n_shuffles": n_shuffles,
            "min_shift": min_shift,
            "rng": rng,
            "unit_id": unit_id,
            "alpha": alpha,
        },
        call="is_place_cell",
    )
    breaks = validate_break_before(break_before, len(times), call="is_place_cell")
    if criterion == "shuffle":
        label = 0 if unit_id is None else unit_id
        level = _SHUFFLE_DEFAULTS["alpha"] if alpha is None else alpha
        shuffle_result = place_cell_significance(
            env,
            [spike_times],
            times,
            positions,
            unit_ids=[label],
            **resolve_shuffle_settings(n_shuffles, min_shift),
            rng=rng,
            method=method,
            bandwidth=bandwidth,
            min_occupancy=min_occupancy,
            fill_value=fill_value,
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
        )[label]
        return shuffle_result.p_value < level
    min_info = (
        PLACE_SPATIAL_INFO_THRESHOLDS["min_info"] if min_info is None else min_info
    )
    result = compute_spatial_rate(
        env,
        spike_times,
        times,
        positions,
        method=method,
        bandwidth=bandwidth,
        min_occupancy=min_occupancy,
        fill_value=fill_value,
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
    return result.is_place_cell(criterion="threshold", min_info=min_info)


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


def _extract_connected_component_graph(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Extract the seed component using the environment's connectivity graph.

    Breadth-first search follows only published edges between candidate bins,
    preserving diagonal connections and respecting physical walls.

    Parameters
    ----------
    seed_idx : int
        Starting bin index.
    mask : array, shape (n_bins,)
        Boolean mask of candidate bins.
    env : Environment
        Spatial environment for connectivity.

    Returns
    -------
    component : array
        Bin indices in connected component (sorted).
    """
    # Flood fill using graph connectivity (BFS)
    component_set = {seed_idx}
    frontier = deque([seed_idx])

    while frontier:
        current = frontier.popleft()
        # Get neighbors from graph
        neighbors = list(env.connectivity.neighbors(current))
        for neighbor in neighbors:
            if mask[neighbor] and neighbor not in component_set:
                component_set.add(neighbor)
                frontier.append(neighbor)

    return np.array(sorted(component_set), dtype=np.int64)


def _extract_connected_component(
    seed_idx: int,
    mask: NDArray[np.bool_],
    env: Environment,
) -> NDArray[np.int64]:
    """Return sorted candidate bins connected to the seed by published edges.

    Grid geometry does not establish connectivity: physical walls can remove
    lattice edges, while subsets and restored grids can retain diagonal edges.
    The environment graph is authoritative for every binning family.
    """
    return _extract_connected_component_graph(seed_idx, mask, env)


def _detect_subfields(
    field_rates: NDArray[np.float64],
    field_bins: NDArray[np.int64],
    peak_rate: float,
    env: Environment,
    min_size: int,
) -> list[NDArray[np.int64]]:
    """Recursively detect subfields using higher thresholds.

    Parameters
    ----------
    field_rates : array
        Firing rates within field bins.
    field_bins : array
        Bin indices of field.
    peak_rate : float
        Peak firing rate in field.
    env : Environment
        Spatial environment.
    min_size : int
        Minimum field size.

    Returns
    -------
    subfields : list of arrays
        List of subfield bin indices. If only one subfield found,
        returns list with original field.
    """
    # Try thresholds: 0.5 and 0.7 of peak
    subfield_thresholds = [0.5, 0.7]

    for thresh in subfield_thresholds:
        threshold_rate = peak_rate * thresh
        above_threshold = field_rates >= threshold_rate

        # Find connected components
        subfields = []
        remaining_mask = above_threshold.copy()

        while remaining_mask.any():
            # Find a seed
            seed_local_idx = np.where(remaining_mask)[0][0]
            seed_global_idx = field_bins[seed_local_idx]

            # Build mask in global coordinates
            global_mask = np.zeros(env.n_bins, dtype=bool)
            global_mask[field_bins[above_threshold]] = True

            # Extract component
            component_global = _extract_connected_component(
                seed_global_idx, global_mask, env
            )

            if len(component_global) >= min_size:
                subfields.append(component_global)

            # Remove from remaining mask
            for bin_idx in component_global:
                # Find local index
                local_indices = np.where(field_bins == bin_idx)[0]
                if len(local_indices) > 0:
                    remaining_mask[local_indices[0]] = False

        # If found multiple subfields, return them
        if len(subfields) > 1:
            return subfields

    # No subfields found
    return [field_bins]


def place_cell_significance(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    method: Literal["diffusion_kde", "gaussian_kde", "binned", "glm"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float | None = None,
    fill_value: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
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
) -> dict[Hashable, ShuffleTestResult]:
    """Test place tuning against circular spike-time shifts.

    The test statistic is Skaggs spatial information (bits/spike) of each
    unit's rate map. The trajectory is binned once; each shifted train is
    recounted and smoothed with exactly the observed map's analyzed intervals,
    occupancy and bandwidth. Shuffles preserve counts on the compressed
    analyzed clock and never enter recording gaps.
    The result's is_significant property uses 0.05; compare p_value < alpha
    for another level.

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
        (Ratio methods only.) Smoothing bandwidth in the same units as bin_size;
        ``None`` uses 5 cm in declared metric units (0.05 m / 50 mm),
        or 2.5 median graph-edge lengths for unspecified/custom units. Mutually exclusive with ``method="glm"``.
    min_occupancy : float | None, default=None
        (Ratio methods only.) Minimum occupancy (seconds) for a bin to be
        included; ``None`` resolves to ``0.0`` (no masking). Mutually exclusive
        with ``method="glm"``.
    fill_value : float | None, default=None
        Ratio methods only: replace missing or masked stored rates with this
        value. None preserves NaN. Filling never changes ``spatial_support``,
        and result metrics still treat filled bins as missing.
        Prefer ``BayesianDecoder.from_rates(rates)`` for population decoding;
        explicit arrays require ``prior=rates.spatial_support.astype(float)``.
        Zero filling without that prior treats unobserved space as silence.
        Mutually exclusive with ``method="glm"``.
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
    max_gap : float or None, default=0.5
        Longest sampling interval (seconds) treated as continuous recording.
        Longer intervals (dropped frames, pauses between sessions) are excluded
        from occupancy and their spikes are not counted. ``None`` disables the
        gap check.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). An interval counts only if it lies entirely
        inside one window. ``None`` (default) means unrestricted.
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
    dict[Hashable, ShuffleTestResult]
        Per-unit observed spatial information (bits/spike), null scores,
        corrected p-value and z-score, keyed by unit label in input order. An
        undefined observed score (a silent unit) or a null with no finite
        score gives a NaN p-value.

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
    >>> env = Environment.from_samples(np.c_[xx.ravel(), yy.ravel()], bin_size=5.0)
    >>> positions = np.c_[50 + 20 * np.sin(times / 3), 50 + 20 * np.cos(times / 4)]
    >>> headings = np.sin(times / 5)
    >>> objects = np.array([[50.0, 50.0]])
    >>> trains = [np.sort(rng.uniform(0, 59.9, 40))]
    >>> results = place_cell_significance(
    ...     env, trains, times, positions, n_shuffles=20, rng=0
    ... )
    >>> list(results)
    [0]
    >>> results[0].n_shuffles
    20
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
        fill_value=fill_value,
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
    from neurospatial.encoding._significance import to_shuffle_results

    return to_shuffle_results(shuffled.observed, shuffled.null, shuffled.unit_ids)


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
        finite null scores. NaN when the observed score is undefined (a silent
        unit, or a border map with no field of the minimum size) or the
        statistic is unscored.
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
    unavailable_metrics: Mapping[str, str] = field(default_factory=dict)
    inapplicable_metrics: tuple[str, ...] = ()
    p_values: NDArray[np.float64] = field(init=False)

    statistics: ClassVar[tuple[str, ...]] = _CELL_TYPE_STATISTICS

    def __post_init__(self) -> None:
        from neurospatial.encoding._significance import shuffle_pvalues

        observed = np.asarray(self.observed_scores, dtype=np.float64)
        null = np.asarray(self.null_scores, dtype=np.float64)
        n_stats = len(self.statistics)
        if (
            observed.ndim != 2
            or observed.shape[1] != n_stats
            or null.ndim != 3
            or null.shape[0] != observed.shape[0]
            or null.shape[2] != n_stats
            or len(self.unit_ids) != observed.shape[0]
        ):
            raise ValueError(
                "observed_scores must have shape (n_units, 3), null_scores "
                "(n_units, n_shuffles, 3) and unit_ids (n_units,); got "
                f"{observed.shape}, {null.shape} and {np.shape(self.unit_ids)}."
            )
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

    def __len__(self) -> int:
        return len(self.unit_ids)

    def __getitem__(self, idx: int) -> dict[str, ShuffleTestResult]:
        """One unit's test of each statistic, keyed by statistic name."""
        from neurospatial.encoding._significance import shuffle_test_result

        i = self._unit_position(idx)
        return {
            name: shuffle_test_result(
                float(self.observed_scores[i, k]),
                self.null_scores[i, :, k],
                float(self.p_values[i, k]),
            )
            for k, name in enumerate(self.statistics)
        }

    def _outcomes(self, alpha: float | None) -> dict[str, NDArray[np.float64]]:
        level = _SHUFFLE_DEFAULTS["alpha"] if alpha is None else alpha
        with np.errstate(invalid="ignore"):
            passed = self.p_values < level
        return {
            name: _test_outcome(
                passed[:, k],
                unknown=name in self.unavailable_metrics
                and name not in self.inapplicable_metrics,
            )
            for k, name in enumerate(self.statistics)
        }

    def label_cell_types(self, *, alpha: float | None = None) -> NDArray[np.str_]:
        """Label each unit from its three shuffle tests.

        Parameters
        ----------
        alpha : float or None, default=None
            Significance level; a test passes when ``p_value < alpha``. None
            is 0.05.

        Returns
        -------
        ndarray of str, shape (n_units,)
            ``"grid"``, ``"border"`` or ``"grid+border"`` from the grid and
            border tests, each judged by its own p-value; ``"place"`` for a
            unit whose spatial information is significant and that is neither
            a grid nor a border cell; ``"unclassified"`` for none;
            ``"undetermined"`` when a grid or border test applies to the
            environment but its score could not be computed (a
            ``UserWarning`` gives the reason). On a 1-D environment grid and
            border do not apply, so a significant cell is ``"place"``.
        """
        unknown = {
            name: reason
            for name, reason in self.unavailable_metrics.items()
            if name not in self.inapplicable_metrics
        }
        if unknown:
            reasons = " ".join(f"{name}: {reason}" for name, reason in unknown.items())
            warnings.warn(
                "label_cell_types: "
                + " and ".join(unknown)
                + " could not be computed on this environment, so cells are "
                f"labeled 'undetermined'. Reason: {reasons}",
                UserWarning,
                stacklevel=2,
            )
        return _cell_type_labels(self._outcomes(alpha))

    def summary_table(self, *, alpha: float | None = None) -> pd.DataFrame:
        """Per-unit observed scores, null percentiles, p-values and labels.

        Parameters
        ----------
        alpha : float or None, default=None
            Significance level for ``passes_*`` and ``cell_type``, and the
            null percentile reported, ``100 * (1 - alpha)``. None is 0.05.

        Returns
        -------
        pandas.DataFrame
            One row per unit, indexed by ``unit_id``: for each statistic in
            :attr:`statistics` the observed score, ``<statistic>_p_value``
            and ``<statistic>_null_percentile`` (the ``100 * (1 - alpha)``
            percentile of the unit's finite null scores), then
            ``passes_<statistic>`` (``p_value < alpha``; ``<NA>`` when the
            statistic applies but could not be computed) and ``cell_type``
            from :meth:`label_cell_types`. ``df.attrs`` records ``alpha``,
            ``null_percentile``, ``n_shuffles``, ``min_shift``,
            ``shuffle_type``, ``method``, ``bandwidth``, ``boundary_source``,
            ``units`` and any ``unavailable_metrics``.
        """
        import pandas as pd

        level = _SHUFFLE_DEFAULTS["alpha"] if alpha is None else alpha
        percentile = 100.0 * (1.0 - level)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN nulls
            thresholds = np.nanpercentile(self.null_scores, percentile, axis=1)
        columns: dict[str, Any] = {}
        for k, name in enumerate(self.statistics):
            columns[name] = self.observed_scores[:, k].copy()
        for k, name in enumerate(self.statistics):
            columns[f"{name}_p_value"] = self.p_values[:, k].copy()
        for k, name in enumerate(self.statistics):
            columns[f"{name}_null_percentile"] = thresholds[:, k]
        columns.update(_cell_type_columns(self._outcomes(level)))
        df = pd.DataFrame(
            columns, index=pd.Index(self.unit_ids.tolist(), name="unit_id")
        )
        df.attrs.update(
            alpha=level,
            null_percentile=percentile,
            n_shuffles=self.n_shuffles,
            min_shift=self.min_shift,
            shuffle_type="circular_time_shift",
            method=self.method,
            bandwidth=self.bandwidth,
            boundary_source=self.boundary_source,
            units={"spatial_information": "bits/spike", "min_shift": "s"},
        )
        if self.unavailable_metrics:
            df.attrs["unavailable_metrics"] = dict(self.unavailable_metrics)
        return df

    def summary(self) -> dict[str, Any]:
        """Unit and shuffle counts and the number of units with each label."""
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


def spatial_cell_type_significance(
    env: Environment,
    spike_times: SpikeTrainsLike,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    boundary_source: BoundarySource = "physical",
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    bandwidth: float | None = None,
    min_occupancy: float | None = None,
    speed: NDArray[np.float64] | None = None,
    min_speed: float | None = None,
    max_gap: float | None = 0.5,
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
    is judged by its own test: a grid cell needs a significant grid score, a
    border cell a significant border score, and a cell that passes both is
    reported as ``"grid+border"``. A place cell has significant spatial
    information and is neither. Use this for a verdict; the fixed cutoffs of
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
        Smoothing bandwidth in environment units; None uses 5 cm in metric
        units, or 2.5 median graph-edge lengths otherwise.
    min_occupancy : float or None, default=None
        Seconds a bin must be occupied to count; None is 0.0.
    speed : ndarray, shape (n_samples,), optional
        Speed for the ``min_speed`` gate; derived from ``positions`` when None.
    min_speed : float or None, default=None
        Exclude intervals slower than this (environment units per second)
        from spikes and occupancy alike. None applies no speed gate.
    max_gap : float or None, default=0.5
        Longest sampling interval, in seconds, treated as continuous.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Analyze only these half-open windows, in seconds.
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
    ``unavailable_metrics``. A map with no field of the minimum size has no
    border score, so its p-value is NaN and it is not a border cell; null
    maps without a border score are left out of that unit's p-value.

    Each test controls its own false-positive rate at ``alpha``; a unit is
    tested three times, so an untuned unit receives some label more often
    than ``alpha``. For 100 untuned Poisson units (0.5-3 Hz) in a 20-minute
    simulated session in a 100 cm box (2.5 cm bins, 200 shuffles, analysis
    boundary), 6%, 1% and 3% had p < 0.05 for spatial information, grid
    score and border score, and 10% received a label other than
    ``"unclassified"``; the fixed-cutoff screen labeled 46% of them border
    cells. Of 12 simulated place cells on a radial-arm maze none was labeled
    a border cell (the screen labeled 6). A significant grid score says the
    map is more hexagonal than its own shifted spike trains, not that it has
    several fields: 2 of 30 simulated single-field place cells had
    significant grid scores (0.92 and 1.07), so check the autocorrelogram
    (:meth:`SpatialRateResult.grid_properties`) before reporting a grid cell.
    Scoring all three statistics for 50 units with 200 shuffles in that box
    took about 50 s on one core, against 9 s for spatial information alone.

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
    ...     env, trains, times, positions, n_shuffles=19, rng=0
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
        fill_value=None,
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
    return CellTypeSignificanceResult(
        observed_scores=shuffled.observed,
        null_scores=shuffled.null.transpose(1, 0, 2),
        unit_ids=shuffled.unit_ids,
        min_shift=float(min_shift),
        method=method,
        bandwidth=shuffled.bandwidth,
        boundary_source=boundary_source,
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


class _ShuffleMaps(NamedTuple):
    """Trajectory binning and estimator settings shared by every shifted map."""

    times: NDArray[np.float64]  # (n_time,)
    positions: NDArray[np.float64]  # (n_time, n_dims)
    interval_mask: NDArray[np.bool_]  # (n_time - 1,) analyzed intervals
    occupancy: NDArray[np.float64]  # (n_bins,) seconds
    occupied: NDArray[np.bool_]  # (n_bins,) occupancy support of the metrics
    method: str
    bandwidth: float
    min_occupancy: float
    fill_value: float | None
    backend: ResolvedBackend
    dtype: type[np.float32] | type[np.float64]


def _shuffle_rate_maps(
    env: Environment, spike_times: list[NDArray[np.float64]], maps: _ShuffleMaps
) -> NDArray[np.float64]:
    """Supported rate maps of spike trains on the shared trajectory binning.

    Matches ``_supported_rates(compute_spatial_rates(...))`` for the same
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
        counts[i] = _bin_spike_train_with_stats(
            env,
            train,
            maps.times,
            maps.positions,
            interval_mask=maps.interval_mask,
        )[0]
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
    if maps.fill_value is not None:
        rates = _fill_nan(rates, maps.fill_value)
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
    min_occupancy: float | None,
    fill_value: float | None,
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

    The trajectory is binned once: the analyzed intervals, occupancy and
    smoothing settings of the observed map are reused for every shifted
    train, and each shift keeps the unit's spike count on the analyzed clock
    without entering recording gaps. Every requested statistic is computed
    from the same shifted trains.

    Returns
    -------
    _ShuffleScores
        Observed scores ``(n_units, len(statistics))`` and null scores
        ``(n_shuffles, n_units, len(statistics))``, NaN for statistics that
        cannot be scored, with the reasons (see ``_cell_type_score``).
    """
    from neurospatial._intervals import resolve_time_windows, run_time_bounds
    from neurospatial._results import resolve_unit_ids
    from neurospatial._spikes import _as_spike_trains_with_ids
    from neurospatial.encoding._backend import validate_and_resolve_backend
    from neurospatial.encoding._provenance import occupancy_support
    from neurospatial.encoding._significance import run_shuffle_test, snapshot_options
    from neurospatial.encoding._spatial_counts import (
        _spatial_interval_mask,
        require_allocentric,
        resolve_speed,
    )
    from neurospatial.encoding._validation import (
        _validate_env_argument,
        validate_spike_times,
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
    trains, input_ids = _as_spike_trains_with_ids(spike_times)
    trains = [np.array(train, dtype=np.float64, copy=True) for train in trains]
    times = np.array(times, dtype=np.float64, copy=True)
    positions = np.array(positions, dtype=np.float64, copy=True)
    ids = np.array(
        resolve_unit_ids(unit_ids, len(trains), input_ids=input_ids, context=call),
        copy=True,
    )
    resolved_epochs, resolved_spike_window = resolve_time_windows(epochs, spike_window)
    options: dict[str, Any] = {
        "method": method,
        "bandwidth": bandwidth,
        "min_occupancy": min_occupancy,
        "fill_value": fill_value,
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
    options = snapshot_options(
        options,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        unit_ids=ids,
    )
    _validate_env_argument(env, context=call, arguments="spike_times, times, positions")
    require_allocentric(env, f"{call}()")
    validate_trajectory(times, positions=positions, context=call, n_dims=env.n_dims)
    for train in trains:
        validate_spike_times(train, context=call)
    options["break_before"] = validate_break_before(break_before, len(times), call=call)
    positions_2d = positions.reshape(-1, 1) if positions.ndim == 1 else positions
    mask = _spatial_interval_mask(
        env,
        times,
        positions_2d,
        speed=resolve_speed(
            times,
            positions,
            options["speed"],
            min_speed,
            max_gap=max_gap,
            break_before=options["break_before"],
        ),
        min_speed=min_speed,
        max_gap=max_gap,
        epochs=resolved_epochs,
        spike_window=resolved_spike_window,
        break_before=options["break_before"],
    )
    # The observed map resolves the estimator defaults, its occupancy and the
    # caller's data-quality warnings once; every shifted map reuses them.
    reference = compute_spatial_rates(env, trains, times, positions, **options)
    occupancy = np.asarray(reference.occupancy, dtype=np.float64)
    maps = _ShuffleMaps(
        times=times,
        positions=positions_2d,
        interval_mask=mask,
        occupancy=occupancy,
        occupied=occupancy_support(reference),
        method=method,
        bandwidth=cast("float", reference.bandwidth),
        min_occupancy=reference.min_occupancy,
        fill_value=fill_value,
        backend=validate_and_resolve_backend(backend),
        dtype=options["dtype"],
    )
    observed_rates = _shuffle_rate_maps(env, trains, maps)
    unavailable: dict[str, str] = {}
    inapplicable: list[str] = []
    for name in statistics:
        if name == "spatial_information":
            continue
        # The observed scores are recomputed, with their warnings, below.
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
        run_time_bounds(times, mask),
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
    )
