"""Compare spatial rate maps of the same units between two conditions.

A comparison pairs units by their ``unit_ids`` and bins by the places they
cover, then scores each unit over the bins both conditions observed. It
describes how similar the maps are; it does not decide whether a cell
remapped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from neurospatial._exceptions import IncompatibleEnvironmentError, _format_error
from neurospatial._results import (
    ResultMixin,
    UnitSelectionMixin,
    _copy_analysis_settings,
)
from neurospatial.encoding._base import _to_numpy
from neurospatial.encoding._provenance import occupancy_support
from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult
from neurospatial.environment import _environment_differences
from neurospatial.environment._targets import _validate_target_bins

if TYPE_CHECKING:
    from neurospatial.environment import Environment

__all__ = ["RateComparisonResult", "compare_spatial_rates"]

_OPERATION = "compare_spatial_rates()"
_METHODS = ("pearson", "spearman")
_UNIT_SELECTIONS = ("intersection", "exact")
_TOPOLOGIES = ("same", "may_differ")
_MERGED_RATE_EXAMPLE = (
    "Merging bins by averaging rates is wrong: rates [0, 100] Hz with occupancy "
    "[100, 1] s merge to 0.99 Hz from counts and occupancy, not 50 Hz."
)


@_copy_analysis_settings
@dataclass(frozen=True, eq=False, repr=False)
class RateComparisonResult(ResultMixin, UnitSelectionMixin):
    """Per-unit similarity of two conditions' rate maps over shared support.

    Each unit is scored by correlating its rate in condition A with its rate
    in condition B across the bins where both conditions have support: the
    animal occupied the bin for at least each estimate's ``min_occupancy``
    and both rates are finite. A high correlation says the two maps have a
    similar shape over the places both conditions sampled. It is descriptive:
    it is not a remapping verdict, has no p-value, and says nothing about
    places only one condition visited. Units are paired by ``unit_ids``, so
    the labels must already identify the same sorted cells in both
    conditions; the comparison does not register cells.

    Build it with :func:`compare_spatial_rates`.

    Attributes
    ----------
    unit_ids : ndarray, shape (n_units,)
        Labels of the compared units, in ``rates_a`` order.
    correlation : ndarray of float64, shape (n_units,)
        Pearson or Spearman correlation over each unit's common bins. NaN when
        the unit was not scored; ``status`` says why.
    status : ndarray of str, shape (n_units,)
        ``"scored"``, ``"insufficient_common_bins"`` (fewer common bins than
        ``min_common_bins``), ``"constant_rates_a"``, ``"constant_rates_b"``
        or ``"constant_rates_both"`` (a map is flat over the common bins, so
        the correlation is undefined).
    n_common_bins : ndarray of intp, shape (n_units,)
        Number of bins each unit was scored over.
    bins_a, bins_b : ndarray of intp, shape (n_bins,)
        The compared bins, as bin IDs in ``rates_a.env`` and ``rates_b.env``.
        Column ``j`` of every per-bin array is the place ``bins_a[j]``.
    firing_rates_a_on_bins, firing_rates_b_on_bins : ndarray of float64, \
shape (n_units, n_bins)
        Each condition's rates in Hz on the compared bins, unsupported bins
        included as estimated.
    occupancy_a, occupancy_b : ndarray of float64, shape (n_bins,)
        Each condition's occupancy in seconds on the compared bins.
    occupied_a, occupied_b : ndarray of bool, shape (n_bins,)
        Bins occupied for at least the condition's ``min_occupancy``.
    common_support : ndarray of bool, shape (n_units, n_bins)
        Bins entering each unit's score: occupied in both conditions with
        finite rates in both.
    env : Environment
        ``rates_a.env``: coordinates and units of ``bins_a``.
    method : str
        ``"pearson"`` or ``"spearman"``.
    unit_ids_only_a, unit_ids_only_b : ndarray
        Units recorded in only one condition, which were not compared.
    analysis_settings : dict or None
        The comparison's choices: ``method``, ``unit_selection``,
        ``min_common_bins``, ``topology``, ``topology_differs``,
        ``bin_mapping`` (``None`` or ``"explicit"``), ``n_compared_bins``,
        ``bins_selected``, each condition's estimator settings
        (``estimator_a``, ``estimator_b``) and ``units``.

    Examples
    --------
    Split one session into halves and ask whether each cell's map is stable:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import compare_spatial_rates, compute_spatial_rates
    >>> times = np.arange(3000) / 30.0
    >>> positions = 50 + 40 * np.column_stack(
    ...     [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    ... )
    >>> env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    >>> rng = np.random.default_rng(0)
    >>> spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (300, 200, 400)]
    >>> halves = [
    ...     compute_spatial_rates(
    ...         env, spike_times, times, positions,
    ...         epochs=[window], unit_ids=[10, 11, 12],
    ...     )
    ...     for window in ([0.0, 50.0], [50.0, 100.0])
    ... ]
    >>> result = compare_spatial_rates(*halves)
    >>> result.unit_ids.tolist()
    [10, 11, 12]
    >>> result.status.tolist()
    ['scored', 'scored', 'scored']
    """

    unit_ids: NDArray[Any]
    correlation: NDArray[np.float64]
    status: NDArray[np.str_]
    n_common_bins: NDArray[np.intp]
    bins_a: NDArray[np.intp]
    bins_b: NDArray[np.intp]
    firing_rates_a_on_bins: NDArray[np.float64]
    firing_rates_b_on_bins: NDArray[np.float64]
    occupancy_a: NDArray[np.float64]
    occupancy_b: NDArray[np.float64]
    occupied_a: NDArray[np.bool_]
    occupied_b: NDArray[np.bool_]
    common_support: NDArray[np.bool_]
    env: Environment
    method: str
    unit_ids_only_a: NDArray[Any]
    unit_ids_only_b: NDArray[Any]
    analysis_settings: dict[str, Any] | None = field(
        default=None, kw_only=True, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        n_units, n_bins = len(self.unit_ids), len(self.bins_a)
        expected = {
            "correlation": (n_units,),
            "status": (n_units,),
            "n_common_bins": (n_units,),
            "bins_b": (n_bins,),
            "firing_rates_a_on_bins": (n_units, n_bins),
            "firing_rates_b_on_bins": (n_units, n_bins),
            "occupancy_a": (n_bins,),
            "occupancy_b": (n_bins,),
            "occupied_a": (n_bins,),
            "occupied_b": (n_bins,),
            "common_support": (n_units, n_bins),
        }
        for name, shape in expected.items():
            actual = np.shape(getattr(self, name))
            if actual != shape:
                raise ValueError(
                    _format_error(
                        f"RateComparisonResult: {name} has shape {actual}, not "
                        f"{shape} for {n_units} unit(s) and {n_bins} compared "
                        "bin(s).",
                        why="Why: every per-unit array has one entry per unit and "
                        "every per-bin array one entry per compared bin.",
                        fix="build comparisons with compare_spatial_rates()",
                    )
                )
        self._own_arrays()

    def __len__(self) -> int:
        """Number of compared units.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0
        ... )
        >>> def rates(maps):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.ones(5),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> len(
        ...     compare_spatial_rates(
        ...         rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...         rates([[1, 2, 3, 5, 4], [4, 1, 5, 2, 3]]),
        ...     )
        ... )
        2
        """
        return len(self.unit_ids)

    def __getitem__(self, idx: int) -> RateComparisonResult:
        """The ``idx``-th compared unit, as a one-unit comparison.

        Parameters
        ----------
        idx : int
            Position in ``unit_ids``. Select by label with ``sel(unit_id=)``.

        Returns
        -------
        RateComparisonResult
            One unit; the compared bins and settings are unchanged.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0
        ... )
        >>> def rates(maps):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.ones(5),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...     rates([[1, 2, 3, 5, 4], [5, 1, 4, 2, 3]]),
        ... )
        >>> result[1].unit_ids.tolist(), result[1].correlation.tolist()
        ([9], [1.0])
        >>> result.sel(unit_id=9).correlation.tolist()
        [1.0]
        """
        rows = [self._unit_position(idx)]
        return replace(
            self,
            unit_ids=self.unit_ids[rows],
            correlation=self.correlation[rows],
            status=self.status[rows],
            n_common_bins=self.n_common_bins[rows],
            firing_rates_a_on_bins=self.firing_rates_a_on_bins[rows],
            firing_rates_b_on_bins=self.firing_rates_b_on_bins[rows],
            common_support=self.common_support[rows],
        )

    @property
    def rate_difference(self) -> NDArray[np.float64]:
        """Rate in condition B minus rate in condition A, in Hz.

        NaN outside each unit's common support, so a bin only one condition
        observed never reads as a change.

        Returns
        -------
        ndarray of float64, shape (n_units, n_bins)
            Read-only; copy with ``np.array(...)`` to modify.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0
        ... )
        >>> def rates(maps, occupancy):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.array(occupancy, dtype=float),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5]], [1, 1, 1, 1, 1]),
        ...     rates([[2, 2, 5, 4, 9]], [1, 1, 1, 1, 0]),
        ... )
        >>> result.rate_difference.tolist()
        [[1.0, 0.0, 2.0, 0.0, nan]]
        """
        with np.errstate(invalid="ignore"):
            change = self.firing_rates_b_on_bins - self.firing_rates_a_on_bins
        difference = np.where(self.common_support, change, np.nan)
        difference.flags.writeable = False
        return difference


def compare_spatial_rates(
    rates_a: SpatialRatesResult | SpatialRateResult,
    rates_b: SpatialRatesResult | SpatialRateResult,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
    unit_selection: Literal["intersection", "exact"] = "intersection",
    bin_mapping: ArrayLike | None = None,
    bins: ArrayLike | None = None,
    topology: Literal["same", "may_differ"] = "same",
    min_common_bins: int = 3,
) -> RateComparisonResult:
    """Correlate each unit's rate map between two conditions.

    Typical questions: is a cell's map the same on outbound and inbound runs,
    in the first and second half of a session, or before and after a cue
    rotation? Estimate each condition with ``compute_spatial_rates(...,
    epochs=..., unit_ids=...)`` on the same environment, then compare.

    Units are paired by ``unit_ids``, never by row position. Each unit is
    scored only over its common support: bins the animal occupied for at
    least ``min_occupancy`` in both conditions, where both rates are finite.
    A smoothed rate in a bin the animal never visited is an extrapolation,
    not an observation, so it never enters a score. One unit's missing bins
    do not remove bins from another unit.

    Parameters
    ----------
    rates_a, rates_b : SpatialRatesResult or SpatialRateResult
        Rate maps of the two conditions: two population results or two
        single-unit results.
    method : {"pearson", "spearman"}, default="pearson"
        Correlation of rates (``"pearson"``) or of their ranks within the
        common bins (``"spearman"``, insensitive to monotonic rate changes).
    unit_selection : {"intersection", "exact"}, default="intersection"
        ``"intersection"`` compares the units present in both conditions and
        reports the others in ``unit_ids_only_a`` / ``unit_ids_only_b``.
        ``"exact"`` raises unless both conditions hold the same units.
    bin_mapping : array-like of int, shape (n_pairs, 2), optional
        Rows of ``(bin in rates_a.env, bin in rates_b.env)`` pairing bins that
        are the same place, for maps estimated on different environments such
        as a subset of the arena. Each bin appears at most once on each side;
        paired bins must have identical centers, sizes and dimensions. By
        default the environments must have the same bins.
    bins : array-like of int, optional
        Bin IDs of ``rates_a.env`` to restrict the comparison to, for example
        ``env.bins_in_region("center")``.
    topology : {"same", "may_differ"}, default="same"
        ``"may_differ"`` accepts environments whose bins are the same places
        but whose connectivity or walls differ, as when a door opens between
        conditions. Smoothing then used different neighbors in each
        condition; the choice is recorded in ``analysis_settings``.
    min_common_bins : int, default=3
        Fewest common bins a unit needs to be scored. At least 3: a
        correlation over 2 bins is always +1 or -1.

    Returns
    -------
    RateComparisonResult
        Per-unit ``correlation``, ``status``, ``n_common_bins``,
        ``common_support`` and ``rate_difference`` (B - A, Hz).

    Raises
    ------
    TypeError
        If the inputs are not two ``SpatialRatesResult`` or two
        ``SpatialRateResult``.
    ValueError
        If a population result has generated ``unit_ids``, the units cannot
        be matched, ``bin_mapping`` is not one-to-one, or an option is
        invalid.
    IncompatibleEnvironmentError
        If the environments' bins are not the same places in the same
        coordinates, or their topology differs and ``topology="same"``.

    Warns
    -----
    UserWarning
        If the conditions were estimated with different ``method``,
        ``bandwidth`` or ``min_occupancy``: the maps then differ partly
        because of the estimation.

    See Also
    --------
    compute_spatial_rates : Estimate the rate maps of each condition.
    field_stability : Correlation of two bare rate-map arrays.

    Notes
    -----
    An undefined score is NaN with a reason in ``status``, never 0: too few
    common bins (``"insufficient_common_bins"``) or a map that is flat over
    the common bins (``"constant_rates_a"``, ``"constant_rates_b"``,
    ``"constant_rates_both"``).

    Maps are never coarsened or merged here. Rates of several bins cannot be
    averaged into one bin: 0 Hz over 100 s and 100 Hz over 1 s hold 100
    spikes in 101 s, a rate of 0.99 Hz, not the 50 Hz mean of the rates. To
    compare on coarser bins, re-estimate both conditions on that environment.

    Examples
    --------
    Is each cell's map stable between the two halves of a session?

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.encoding import compare_spatial_rates, compute_spatial_rates
    >>> times = np.arange(3000) / 30.0
    >>> positions = 50 + 40 * np.column_stack(
    ...     [np.sin(2 * np.pi * times / 20), np.cos(2 * np.pi * times / 13)]
    ... )
    >>> env = Environment.from_samples(positions, bin_size=10.0, units="cm")
    >>> rng = np.random.default_rng(0)
    >>> spike_times = [np.sort(rng.uniform(0, 100, n)) for n in (300, 200, 400)]
    >>> first_half, second_half = (
    ...     compute_spatial_rates(
    ...         env,
    ...         spike_times,
    ...         times,
    ...         positions,
    ...         epochs=[window],
    ...         unit_ids=[10, 11, 12],
    ...     )
    ...     for window in ([0.0, 50.0], [50.0, 100.0])
    ... )
    >>> result = compare_spatial_rates(first_half, second_half)
    >>> result.unit_ids.tolist()
    [10, 11, 12]
    >>> result.status.tolist()
    ['scored', 'scored', 'scored']
    >>> bool(np.all(np.abs(result.correlation) <= 1))
    True
    """
    population = _check_pair(rates_a, rates_b)
    _check_choice("method", method, _METHODS)
    _check_choice("unit_selection", unit_selection, _UNIT_SELECTIONS)
    _check_choice("topology", topology, _TOPOLOGIES)
    _check_min_common_bins(min_common_bins)

    if population:
        unit_ids, rows_a, rows_b, only_a, only_b = _match_units(
            cast("SpatialRatesResult", rates_a),
            cast("SpatialRatesResult", rates_b),
            unit_selection,
        )
    else:
        label = _single_unit_label(
            cast("SpatialRateResult", rates_a), cast("SpatialRateResult", rates_b)
        )
        unit_ids = np.array([label])
        rows_a = rows_b = np.zeros(1, dtype=np.intp)
        only_a = only_b = unit_ids[:0]

    env_a, env_b = rates_a.env, rates_b.env
    pairs, topology_differs = _bin_pairs(env_a, env_b, bin_mapping, topology)
    selected = None
    if bins is not None:
        selected = _validate_target_bins(
            bins,
            n_bins=env_a.n_bins,
            coordinates=env_a.coordinate_system,
            operation=_OPERATION,
            argument="bins",
        )
        pairs = pairs[np.isin(pairs[:, 0], selected)]
    bins_a = np.ascontiguousarray(pairs[:, 0], dtype=np.intp)
    bins_b = np.ascontiguousarray(pairs[:, 1], dtype=np.intp)

    rate_a = _rates_2d(rates_a)[np.ix_(rows_a, bins_a)]
    rate_b = _rates_2d(rates_b)[np.ix_(rows_b, bins_b)]
    occupied_a = occupancy_support(rates_a)[bins_a]
    occupied_b = occupancy_support(rates_b)[bins_b]
    common = occupied_a & occupied_b & np.isfinite(rate_a) & np.isfinite(rate_b)
    correlation, status, n_common = _correlate(
        rate_a, rate_b, common, method=method, min_common_bins=min_common_bins
    )

    estimator_a, estimator_b = _estimator(rates_a), _estimator(rates_b)
    _warn_on_estimator_differences(estimator_a, estimator_b)
    settings = {
        "method": method,
        "unit_selection": unit_selection,
        "min_common_bins": int(min_common_bins),
        "topology": topology,
        "topology_differs": topology_differs,
        "bin_mapping": None if bin_mapping is None else "explicit",
        "n_compared_bins": len(bins_a),
        "bins_selected": None if selected is None else int(np.unique(selected).size),
        "estimator_a": estimator_a,
        "estimator_b": estimator_b,
        "units": env_a.units,
    }
    return RateComparisonResult(
        unit_ids=unit_ids,
        correlation=correlation,
        status=status,
        n_common_bins=n_common,
        bins_a=bins_a,
        bins_b=bins_b,
        firing_rates_a_on_bins=rate_a,
        firing_rates_b_on_bins=rate_b,
        occupancy_a=_occupancy(rates_a)[bins_a],
        occupancy_b=_occupancy(rates_b)[bins_b],
        occupied_a=occupied_a,
        occupied_b=occupied_b,
        common_support=common,
        env=env_a,
        method=method,
        unit_ids_only_a=only_a,
        unit_ids_only_b=only_b,
        analysis_settings=settings,
    )


def _check_pair(rates_a: object, rates_b: object) -> bool:
    """Refuse unsupported inputs; return whether the pair is a population."""
    supported = (
        "two SpatialRatesResult (populations) or two SpatialRateResult (single units)"
    )
    for name, rates in (("rates_a", rates_a), ("rates_b", rates_b)):
        if not isinstance(rates, SpatialRatesResult | SpatialRateResult):
            raise TypeError(
                _format_error(
                    f"{_OPERATION} got {name} of type {type(rates).__name__}; it "
                    f"compares {supported}.",
                    why="Why: directional, view, object-vector and boundary-vector "
                    "maps are not maps of the animal's position, and comparing "
                    "them needs their own coordinates.",
                    fix="pass the results of compute_spatial_rates() (or "
                    "compute_spatial_rate()) for each condition",
                )
            )
    population = isinstance(rates_a, SpatialRatesResult)
    if population != isinstance(rates_b, SpatialRatesResult):
        raise TypeError(
            _format_error(
                f"{_OPERATION} got a population SpatialRatesResult and a "
                "single-unit SpatialRateResult.",
                why="Why: the comparison pairs like with like.",
                fix="select the matching unit of the population with "
                "rates.sel(unit_id=...), or pass both populations",
            )
        )
    return population


def _check_choice(argument: str, value: object, allowed: tuple[str, ...]) -> None:
    """Refuse an option outside its allowed values."""
    if value not in allowed:
        options = ", ".join(repr(option) for option in allowed)
        raise ValueError(
            _format_error(
                f"{_OPERATION} got {argument}={value!r}.",
                fix=f"pass {argument}= one of {options}",
            )
        )


def _check_min_common_bins(min_common_bins: object) -> None:
    """Require an integer of at least 3."""
    if isinstance(min_common_bins, bool) or not isinstance(
        min_common_bins, int | np.integer
    ):
        raise TypeError(
            _format_error(
                f"{_OPERATION} got min_common_bins={min_common_bins!r}, not an "
                "integer.",
                fix="pass a whole number of bins, e.g. min_common_bins=3",
            )
        )
    if min_common_bins < 3:
        raise ValueError(
            _format_error(
                f"{_OPERATION} got min_common_bins={min_common_bins}.",
                why="Why: a correlation over two bins is always +1 or -1 whatever "
                "the maps, so it measures nothing.",
                fix="use min_common_bins=3 or more",
            )
        )


def _match_units(
    rates_a: SpatialRatesResult,
    rates_b: SpatialRatesResult,
    unit_selection: str,
) -> tuple[
    NDArray[Any], NDArray[np.intp], NDArray[np.intp], NDArray[Any], NDArray[Any]
]:
    """Pair population rows by unit label, keeping ``rates_a`` order."""
    for name, rates in (("rates_a", rates_a), ("rates_b", rates_b)):
        if rates._unit_ids_generated:
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: {name} has no unit_ids; its labels are row "
                    "positions generated by the estimator.",
                    why="Why: units are matched between conditions by label, and "
                    "row positions do not identify the same cell in two results.",
                    fix="pass the same sorted-unit labels as unit_ids= to "
                    "compute_spatial_rates() for both conditions",
                )
            )
    ids_a, ids_b = np.asarray(rates_a.unit_ids), np.asarray(rates_b.unit_ids)
    labels_a, labels_b = ids_a.tolist(), ids_b.tolist()
    row_of_b = {label: row for row, label in enumerate(labels_b)}
    set_a = set(labels_a)
    in_b = np.array([label in row_of_b for label in labels_a], dtype=bool)
    in_a = np.array([label in set_a for label in labels_b], dtype=bool)
    only_a, only_b = ids_a[~in_b], ids_b[~in_a]
    if unit_selection == "exact" and (only_a.size or only_b.size):
        raise ValueError(
            _format_error(
                f"{_OPERATION}: unit_selection='exact' needs the same unit_ids in "
                f"both conditions; only in rates_a: {only_a.tolist()}, only in "
                f"rates_b: {only_b.tolist()}.",
                fix="pass unit_selection='intersection' to compare the shared "
                "units and report the others",
            )
        )
    if not in_b.any():
        raise ValueError(
            _format_error(
                f"{_OPERATION}: rates_a and rates_b have no unit_ids in common "
                f"(rates_a: {labels_a[:10]}, rates_b: {labels_b[:10]}).",
                why="Why: units are matched by label, so the same cell must carry "
                "the same label in both conditions.",
                fix="pass the same sorted-unit labels as unit_ids= to both "
                "compute_spatial_rates() calls",
            )
        )
    rows_a = np.flatnonzero(in_b).astype(np.intp)
    rows_b = np.array([row_of_b[labels_a[row]] for row in rows_a], dtype=np.intp)
    return ids_a[rows_a], rows_a, rows_b, only_a, only_b


def _single_unit_label(rates_a: SpatialRateResult, rates_b: SpatialRateResult) -> Any:
    """The pair's label: the one set, the shared one, or None."""
    labels = [
        rates.unit_id for rates in (rates_a, rates_b) if not rates._unit_ids_generated
    ]
    if len(labels) == 2 and labels[0] != labels[1]:
        raise ValueError(
            _format_error(
                f"{_OPERATION}: rates_a has unit_id {labels[0]!r} and rates_b has "
                f"unit_id {labels[1]!r}.",
                why="Why: a comparison scores one cell across two conditions.",
                fix="pass the same unit from each condition, e.g. "
                "rates.sel(unit_id=...) on both populations",
            )
        )
    return labels[0] if labels else None


def _bin_pairs(
    env_a: Environment,
    env_b: Environment,
    bin_mapping: ArrayLike | None,
    topology: str,
) -> tuple[NDArray[np.intp], bool]:
    """Pair bins that are the same place; return pairs and topology_differs."""
    differences = _environment_differences(env_a, env_b)
    if bin_mapping is None:
        if differences.bins is not None:
            raise IncompatibleEnvironmentError(
                f"{_OPERATION}: rates_a and rates_b were estimated on "
                f"environments that differ in {differences.bins}, so bin j of "
                "one map is not the same place as bin j of the other. Equal "
                "numbers of bins do not make them the same places.",
                fix="estimate both conditions on the same environment, or pass a "
                "one-to-one bin_mapping between them",
                first=env_a,
                second=env_b,
            )
        identity = np.arange(env_a.n_bins, dtype=np.intp)
        pairs = np.column_stack([identity, identity])
    else:
        pairs = _validate_bin_mapping(bin_mapping, env_a, env_b)
    if differences.topology is not None and topology == "same":
        raise IncompatibleEnvironmentError(
            f"{_OPERATION}: rates_a and rates_b were estimated on environments "
            f"that differ in {differences.topology}, so their maps were smoothed "
            "over different neighbors.",
            fix='pass topology="may_differ" if the bins are the same places and '
            "only connectivity or walls changed, e.g. a door opened; smoothing "
            "then used different neighbors",
            first=env_a,
            second=env_b,
        )
    return pairs, differences.topology is not None


def _validate_bin_mapping(
    bin_mapping: ArrayLike, env_a: Environment, env_b: Environment
) -> NDArray[np.intp]:
    """Validate a one-to-one ``(n_pairs, 2)`` mapping between identical bins."""
    mapping = np.asarray(bin_mapping)
    if mapping.ndim != 2 or mapping.shape[1] != 2 or mapping.shape[0] == 0:
        raise ValueError(
            _format_error(
                f"{_OPERATION} got bin_mapping with shape {mapping.shape}, not "
                "(n_pairs, 2).",
                why="Why: each row pairs a bin of rates_a.env with the bin of "
                "rates_b.env at the same place.",
                fix="pass rows of (bin in rates_a, bin in rates_b), e.g. "
                "np.column_stack([bins_a, bins_b])",
            )
        )
    if mapping.dtype.kind not in "iu":
        raise ValueError(
            _format_error(
                f"{_OPERATION} got bin_mapping of dtype {mapping.dtype}; it holds "
                "integer bin IDs.",
                fix="pass integer bin IDs, e.g. np.asarray(bin_mapping, dtype=int) "
                "after checking the values are whole numbers",
            )
        )
    for column, name, env in ((0, "rates_a", env_a), (1, "rates_b", env_b)):
        ids = mapping[:, column]
        bad = (ids < 0) | (ids >= env.n_bins)
        if bad.any():
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: bin_mapping column {column} holds bin "
                    f"{int(ids[bad][0])}, out of range for {name}.env with "
                    f"{env.n_bins} bins.",
                    fix=f"use bin IDs from 0 to {env.n_bins - 1} in column {column}",
                )
            )
    _refuse_repeats(mapping)
    reason: str | None
    if env_a.coordinate_system != env_b.coordinate_system:
        reason = "coordinate systems"
    elif env_a.units != env_b.units or env_a.frame != env_b.frame:
        reason = (
            f"units or coordinate frames ({env_a.units!r}, {env_a.frame!r} vs "
            f"{env_b.units!r}, {env_b.frame!r})"
        )
    else:
        reason = _mapped_bin_difference(mapping, env_a, env_b)
    if reason is not None:
        raise IncompatibleEnvironmentError(
            f"{_OPERATION}: bin_mapping pairs bins that differ in {reason}, so "
            "paired bins are not the same place.",
            fix="pair only bins with identical centers and sizes in the same "
            "units and frame, or re-estimate both conditions on one common "
            "environment",
            first=env_a,
            second=env_b,
        )
    return mapping.astype(np.intp)


def _refuse_repeats(mapping: NDArray[np.integer]) -> None:
    """Refuse many-to-one merges, then one-to-many splits."""
    for column, kind in ((1, "merge"), (0, "split")):
        values, counts = np.unique(mapping[:, column], return_counts=True)
        if np.all(counts == 1):
            continue
        repeated = int(values[np.argmax(counts > 1)])
        others = mapping[mapping[:, column] == repeated, 1 - column].tolist()
        if kind == "merge":
            raise ValueError(
                _format_error(
                    f"{_OPERATION}: bin_mapping is many-to-one: it sends bins "
                    f"{others} of rates_a to bin {repeated} of rates_b.",
                    why=f"Why: {_MERGED_RATE_EXAMPLE}",
                    fix="re-estimate both conditions on one common environment",
                )
            )
        raise ValueError(
            _format_error(
                f"{_OPERATION}: bin_mapping is not one-to-one: it sends bin "
                f"{repeated} of rates_a to bins {others} of rates_b.",
                why="Why: splitting one bin's rate copies a single measurement "
                "into several places.",
                fix="pair each bin at most once, or re-estimate both conditions "
                "on one common environment",
            )
        )


def _mapped_bin_difference(
    mapping: NDArray[np.integer], env_a: Environment, env_b: Environment
) -> str | None:
    """First difference between paired bins' centers, then measures."""
    rows_a, rows_b = mapping[:, 0], mapping[:, 1]
    if not np.array_equal(env_a.bin_centers[rows_a], env_b.bin_centers[rows_b]):
        return "bin coordinates"
    bins_a, bins_b = env_a.binning.bins, env_b.binning.bins
    if not (
        np.array_equal(bins_a.sizes[rows_a], bins_b.sizes[rows_b])
        and np.array_equal(bins_a.measure_dims[rows_a], bins_b.measure_dims[rows_b])
    ):
        return "bin measures"
    return None


def _rates_2d(rates: SpatialRatesResult | SpatialRateResult) -> NDArray[np.float64]:
    """Firing rates as float64 ``(n_units, n_bins)``."""
    values = np.asarray(_to_numpy(rates._get_rates()), dtype=np.float64)
    return values.reshape(-1, rates.env.n_bins)


def _occupancy(rates: SpatialRatesResult | SpatialRateResult) -> NDArray[np.float64]:
    """Occupancy in seconds as float64 ``(n_bins,)``."""
    return np.asarray(_to_numpy(rates.occupancy), dtype=np.float64)


def _correlate(
    rate_a: NDArray[np.float64],
    rate_b: NDArray[np.float64],
    common: NDArray[np.bool_],
    *,
    method: str,
    min_common_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.str_], NDArray[np.intp]]:
    """Per-unit correlation over each row's ``common`` bins, with status.

    Parameters
    ----------
    rate_a, rate_b : ndarray, shape (n_units, n_bins)
        Rates of the two conditions.
    common : ndarray of bool, shape (n_units, n_bins)
        Bins entering each unit's score.
    method : {"pearson", "spearman"}
    min_common_bins : int

    Returns
    -------
    correlation : ndarray, shape (n_units,)
        NaN unless the status is ``"scored"``.
    status : ndarray of str, shape (n_units,)
    n_common : ndarray of intp, shape (n_units,)
    """
    n_common = common.sum(axis=1).astype(np.intp)
    constant_a = _is_constant(rate_a, common)
    constant_b = _is_constant(rate_b, common)
    status = np.select(
        [
            n_common < min_common_bins,
            constant_a & constant_b,
            constant_a,
            constant_b,
        ],
        [
            "insufficient_common_bins",
            "constant_rates_both",
            "constant_rates_a",
            "constant_rates_b",
        ],
        default="scored",
    ).astype(np.str_)
    values_a, values_b = rate_a, rate_b
    if method == "spearman":
        values_a = stats.rankdata(
            np.where(common, rate_a, np.nan), axis=1, nan_policy="omit"
        )
        values_b = stats.rankdata(
            np.where(common, rate_b, np.nan), axis=1, nan_policy="omit"
        )
    centered_a = _centered(values_a, common, n_common)
    centered_b = _centered(values_b, common, n_common)
    covariance = (centered_a * centered_b).sum(axis=1)
    scale = np.sqrt((centered_a**2).sum(axis=1) * (centered_b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / scale
    correlation = np.where(status == "scored", np.clip(correlation, -1.0, 1.0), np.nan)
    return correlation, status, n_common


def _is_constant(
    values: NDArray[np.float64], common: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Whether each row takes a single value over its ``common`` bins."""
    highest = np.where(common, values, -np.inf).max(axis=1, initial=-np.inf)
    lowest = np.where(common, values, np.inf).min(axis=1, initial=np.inf)
    constant: NDArray[np.bool_] = highest == lowest
    return constant


def _centered(
    values: NDArray[np.float64],
    common: NDArray[np.bool_],
    n_common: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Values minus their row mean over ``common`` bins; 0 elsewhere."""
    kept = np.where(common, values, 0.0)
    mean = kept.sum(axis=1) / np.maximum(n_common, 1)
    centered: NDArray[np.float64] = np.where(common, values - mean[:, None], 0.0)
    return centered


def _estimator(rates: SpatialRatesResult | SpatialRateResult) -> dict[str, Any]:
    """The estimator settings that shape a rate map, JSON-safe."""
    settings = rates.analysis_settings or {}
    min_occupancy = settings.get("min_occupancy")
    return {
        "method": rates.method,
        "bandwidth": None if rates.bandwidth is None else float(rates.bandwidth),
        "min_occupancy": None if min_occupancy is None else float(min_occupancy),
    }


def _warn_on_estimator_differences(
    estimator_a: dict[str, Any], estimator_b: dict[str, Any]
) -> None:
    """Warn once, naming every estimator setting that differs."""
    differing = [key for key in estimator_a if estimator_a[key] != estimator_b[key]]
    if not differing:
        return
    details = "; ".join(
        f"{key}: {estimator_a[key]!r} vs {estimator_b[key]!r}" for key in differing
    )
    warnings.warn(
        f"{_OPERATION}: rates_a and rates_b were estimated with different "
        f"{', '.join(differing)} ({details}). Smoothing and occupancy thresholds "
        "change rate maps by themselves, so part of any difference comes from the "
        "estimation, not the cells. Estimate both conditions with the same "
        "settings to compare the cells alone.",
        UserWarning,
        stacklevel=3,
    )
