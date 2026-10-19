"""Compare spatial rate maps of the same units between two conditions.

A comparison pairs units by their ``unit_ids`` and bins by the places they
cover, then scores each unit over the bins both conditions observed. It
describes how similar the maps are; it does not decide whether a cell
remapped.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from neurospatial._exceptions import _format_error
from neurospatial._results import (
    ResultMixin,
    UnitSelectionMixin,
    _copy_analysis_settings,
)
from neurospatial.encoding._base import _to_numpy
from neurospatial.encoding._comparison_inputs import (
    _OPERATION,
    _bin_pairs,
    _check_choice,
    _check_min_common_bins,
    _check_pair,
    _match_units,
    _single_unit_label,
)
from neurospatial.encoding._provenance import occupancy_support
from neurospatial.encoding.spatial import SpatialRateResult, SpatialRatesResult
from neurospatial.environment import _validate_target_bins

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd
    from matplotlib.axes import Axes

    from neurospatial.environment import Environment

__all__ = ["RateComparisonResult", "compare_spatial_rates"]

_METHODS = ("pearson", "spearman")
_UNIT_SELECTIONS = ("intersection", "exact")
_TOPOLOGIES = ("same", "may_differ")
# A map whose spread over the common bins is at most this fraction of its
# largest rate is flat. Dividing smoothed counts by smoothed occupancy leaves
# rounding of order 1e-14 relative (a flat map from the package's estimators
# spreads tens to hundreds of ulps), and correlating that noise reports a
# spurious score.
# A real field is many orders of magnitude deeper than 1e-12 of its peak.
_CONSTANT_RELATIVE_SPREAD = 1e-12


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
        or ``"constant_rates_both"`` (a map is flat over the common bins, up
        to floating-point rounding, so the correlation is undefined).
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
        (``estimator_a``, ``estimator_b``: ``method``, ``bandwidth`` and
        ``min_occupancy`` in seconds, ``0.0`` when no threshold was set) and
        ``units``.

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

    def __iter__(self) -> Iterator[RateComparisonResult]:
        """Iterate over the compared units as one-unit comparisons.

        Yields
        ------
        RateComparisonResult
            ``self[i]`` for each unit, in ``unit_ids`` order.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
        ... )
        >>> def rates(maps, occupancy=(1, 1, 1, 1, 1)):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.array(occupancy, dtype=float),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...     rates([[1, 2, 3, 5, 4], [4, 1, 5, 2, 3]], (1, 1, 1, 1, 0)),
        ... )
        >>> [unit.unit_ids[0].item() for unit in result]
        [7, 9]
        """
        for position in range(len(self)):
            yield self[position]

    def _row_unit_labels(self) -> NDArray[Any]:
        """Unit labels for table indexes; a missing label is ``pd.NA``."""
        labels = np.asarray(self.unit_ids)
        if labels.dtype != object:
            return labels
        import pandas as pd

        return np.array(
            [pd.NA if label is None else label for label in labels], dtype=object
        )

    def _coordinate_columns(self) -> dict[str, NDArray[np.float64]]:
        """Center of each compared bin, one array per ``bin_center_*`` name."""
        from neurospatial._results import _bin_center_coords

        coords = _bin_center_coords(self.env, self.env.n_bins)
        return {
            name: np.asarray(values)[self.bins_a]
            for name, (_, values) in coords.items()
        }

    def _common_means(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean of ``values`` over each unit's common bins; NaN without any."""
        counts = self.n_common_bins
        total = np.where(self.common_support, values, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)

    def summary_table(self) -> pd.DataFrame:
        """One row per unit: the score and what it was computed from.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``unit_id`` (a missing label shows as ``<NA>``) with
            columns ``correlation``, ``n_common_bins``, ``mean_rate_a``,
            ``mean_rate_b``, ``mean_rate_difference`` (B - A),
            ``common_occupancy_a``, ``common_occupancy_b`` and ``status``.
            Means are over each unit's common bins, in Hz, and NaN when the
            unit has none. Occupancies are summed over the common bins, in
            seconds. ``attrs`` holds ``method``, ``units`` and a copy of
            ``analysis_settings``; CSV does not preserve them.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
        ... )
        >>> def rates(maps, occupancy=(1, 1, 1, 1, 1)):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.array(occupancy, dtype=float),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...     rates([[1, 2, 3, 5, 4], [4, 1, 5, 2, 3]], (1, 1, 1, 1, 0)),
        ... )
        >>> table = result.summary_table()
        >>> table.index.tolist()
        [7, 9]
        >>> table["n_common_bins"].tolist()
        [4, 4]
        >>> table["common_occupancy_a"].tolist()
        [4.0, 4.0]
        >>> table.attrs["method"]
        'pearson'
        """
        import pandas as pd

        support = self.common_support
        table = pd.DataFrame(
            {
                "correlation": self.correlation,
                "n_common_bins": self.n_common_bins,
                "mean_rate_a": self._common_means(self.firing_rates_a_on_bins),
                "mean_rate_b": self._common_means(self.firing_rates_b_on_bins),
                "mean_rate_difference": self._common_means(
                    np.where(support, self.rate_difference, 0.0)
                ),
                "common_occupancy_a": support @ self.occupancy_a,
                "common_occupancy_b": support @ self.occupancy_b,
                "status": self.status,
            },
            index=pd.Index(self._row_unit_labels(), name="unit_id"),
        )
        table.attrs["method"] = self.method
        table.attrs["units"] = {
            "mean_rate_a": "Hz",
            "mean_rate_b": "Hz",
            "mean_rate_difference": "Hz",
            "common_occupancy_a": "s",
            "common_occupancy_b": "s",
            "coordinates": self.env.units,
        }
        table.attrs["analysis_settings"] = self.analysis_settings
        return table

    def summary(self) -> dict[str, Any]:
        """Headline numbers of the comparison as flat scalars.

        Returns
        -------
        dict
            ``n_units`` compared, ``n_units_scored``, ``median_correlation``
            over the scored units (NaN when none was scored),
            ``n_units_only_a`` and ``n_units_only_b`` (recorded in one
            condition only), ``n_compared_bins``, ``method`` and
            ``topology_differs``.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
        ... )
        >>> def rates(maps, occupancy=(1, 1, 1, 1, 1)):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.array(occupancy, dtype=float),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...     rates([[1, 2, 3, 5, 4], [4, 1, 5, 2, 3]], (1, 1, 1, 1, 0)),
        ... )
        >>> summary = result.summary()
        >>> summary["n_units"], summary["n_units_scored"], summary["n_compared_bins"]
        (2, 2, 5)
        >>> summary["method"], summary["topology_differs"]
        ('pearson', False)
        """
        scored = self.status == "scored"
        settings = self.analysis_settings or {}
        return {
            "n_units": len(self),
            "n_units_scored": int(scored.sum()),
            "median_correlation": float(np.median(self.correlation[scored]))
            if scored.any()
            else float("nan"),
            "n_units_only_a": len(self.unit_ids_only_a),
            "n_units_only_b": len(self.unit_ids_only_b),
            "n_compared_bins": len(self.bins_a),
            "method": self.method,
            "topology_differs": bool(settings.get("topology_differs", False)),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Dense table with one row per ``(unit, bin)`` of the compared bins.

        Returns
        -------
        pandas.DataFrame
            Columns ``unit_id``, ``bin`` (bin ID in ``rates_a.env``),
            ``bin_b`` (the same place in ``rates_b.env``), ``bin_center_*``
            (coordinates of the bin, in the units of ``env``),
            ``firing_rate_a``, ``firing_rate_b``, ``rate_difference`` (B - A,
            NaN outside the unit's common support), ``occupancy_a``,
            ``occupancy_b`` (seconds), ``occupied_a``, ``occupied_b`` and
            ``common_support``. It is a writable copy; ``attrs`` holds
            ``units`` and a copy of ``analysis_settings``, which CSV does not
            preserve.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
        ... )
        >>> def rates(maps, occupancy=(1, 1, 1, 1, 1)):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.array(occupancy, dtype=float),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...     rates([[1, 2, 3, 5, 4], [4, 1, 5, 2, 3]], (1, 1, 1, 1, 0)),
        ... )
        >>> frame = result.to_dataframe()
        >>> len(frame)
        10
        >>> frame.loc[frame["unit_id"] == 9, "common_support"].tolist()
        [True, True, True, True, False]
        >>> int(frame["rate_difference"].isna().sum())
        2
        """
        import pandas as pd

        n_units, n_bins = self.common_support.shape
        centers = self._coordinate_columns()
        data: dict[str, Any] = {
            "unit_id": np.repeat(self._row_unit_labels(), n_bins),
            "bin": np.tile(self.bins_a, n_units),
            "bin_b": np.tile(self.bins_b, n_units),
        }
        for name, values in centers.items():
            data[name] = np.tile(values, n_units)
        data["firing_rate_a"] = self.firing_rates_a_on_bins.ravel()
        data["firing_rate_b"] = self.firing_rates_b_on_bins.ravel()
        data["rate_difference"] = self.rate_difference.ravel()
        data["occupancy_a"] = np.tile(self.occupancy_a, n_units)
        data["occupancy_b"] = np.tile(self.occupancy_b, n_units)
        data["occupied_a"] = np.tile(self.occupied_a, n_units)
        data["occupied_b"] = np.tile(self.occupied_b, n_units)
        data["common_support"] = self.common_support.ravel()
        frame = pd.DataFrame(data)
        frame.attrs["units"] = {
            "firing_rate_a": "Hz",
            "firing_rate_b": "Hz",
            "rate_difference": "Hz",
            "occupancy_a": "s",
            "occupancy_b": "s",
            **dict.fromkeys(centers, self.env.units or ""),
        }
        frame.attrs["analysis_settings"] = self.analysis_settings
        return frame

    def to_xarray(self) -> Any:
        """Export a labeled Dataset with dimensions ``("unit_id", "bin")``.

        Returns
        -------
        xarray.Dataset
            The ``bin`` coordinate holds the compared bin IDs of
            ``rates_a.env``; ``bin_b`` and ``bin_center_*`` are coordinates on
            it. ``firing_rate_a``, ``firing_rate_b``, ``rate_difference`` and
            ``common_support`` are ``(unit_id, bin)``; ``occupancy_a/b`` and
            ``occupied_a/b`` are ``(bin,)``; ``correlation``,
            ``n_common_bins`` and ``status`` are ``(unit_id,)``. ``attrs``
            hold ``method``, ``units`` (coordinate units, when set),
            ``analysis_settings`` (JSON text) and ``software_version``. The
            arrays are copies.

        Raises
        ------
        ImportError
            If the optional xarray dependency is unavailable.

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
        ... )
        >>> def rates(maps, occupancy=(1, 1, 1, 1, 1)):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.array(occupancy, dtype=float),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...     rates([[1, 2, 3, 5, 4], [4, 1, 5, 2, 3]], (1, 1, 1, 1, 0)),
        ... )
        >>> dataset = result.to_xarray()  # doctest: +SKIP
        >>> dataset["correlation"].sel(unit_id=9).item()  # doctest: +SKIP
        0.9
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "to_xarray() requires the optional 'xarray' dependency, which "
                "is not installed. Install it with "
                "'pip install neurospatial[xarray]' or 'pip install xarray'."
            ) from exc
        from neurospatial._results import software_version, units_attr

        unit_dim, bin_dim = "unit_id", "bin"
        per_bin = {
            name: (bin_dim, values)
            for name, values in self._coordinate_columns().items()
        }
        dataset = xr.Dataset(
            data_vars={
                "firing_rate_a": ((unit_dim, bin_dim), self.firing_rates_a_on_bins),
                "firing_rate_b": ((unit_dim, bin_dim), self.firing_rates_b_on_bins),
                "rate_difference": ((unit_dim, bin_dim), self.rate_difference),
                "common_support": ((unit_dim, bin_dim), self.common_support),
                "occupancy_a": ((bin_dim,), self.occupancy_a),
                "occupancy_b": ((bin_dim,), self.occupancy_b),
                "occupied_a": ((bin_dim,), self.occupied_a),
                "occupied_b": ((bin_dim,), self.occupied_b),
                "correlation": ((unit_dim,), self.correlation),
                "n_common_bins": ((unit_dim,), self.n_common_bins),
                "status": ((unit_dim,), self.status),
            },
            coords={
                unit_dim: self._row_unit_labels(),
                bin_dim: self.bins_a,
                "bin_b": (bin_dim, self.bins_b),
                **per_bin,
            },
            attrs={
                "method": self.method,
                **units_attr(self.env),
                "software_version": software_version(),
                **(
                    {}
                    if self.analysis_settings is None
                    else {
                        "analysis_settings": json.dumps(
                            self.analysis_settings, sort_keys=True
                        )
                    }
                ),
            },
        )
        # Datasets are the caller's to edit; results are read-only.
        return dataset.copy(deep=True)

    def plot(
        self, ax: Axes | None = None, *, unit_id: Any = None, **kwargs: Any
    ) -> Axes:
        """Plot how well the units' maps agree, or where one unit's map changed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is made when omitted.
        unit_id : optional
            Label of one unit. Without it, each scored unit is a point of
            correlation against the number of common bins it was scored over;
            units without a score (see ``status``) are not drawn, never drawn
            as 0. With it, the unit's rate difference (B - A, Hz) is mapped
            over the environment on a color scale symmetric about 0, with
            bins outside the unit's common support left blank.
        **kwargs
            Passed to ``Axes.scatter`` (all units) or to
            ``Environment.plot_field`` (one unit; overrides ``cmap``, ``vmin``,
            ``vmax`` and ``colorbar_label``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.

        Raises
        ------
        KeyError
            If ``unit_id`` is not one of ``unit_ids``.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import numpy as np
        >>> from neurospatial import Environment
        >>> from neurospatial.encoding import SpatialRatesResult, compare_spatial_rates
        >>> env = Environment.from_samples(
        ...     np.linspace(0, 40, 41)[:, None], bin_size=10.0, units="cm"
        ... )
        >>> def rates(maps, occupancy=(1, 1, 1, 1, 1)):
        ...     return SpatialRatesResult(
        ...         firing_rates=np.array(maps, dtype=float),
        ...         occupancy=np.array(occupancy, dtype=float),
        ...         env=env,
        ...         method="binned",
        ...         bandwidth=5.0,
        ...         unit_ids=[7, 9],
        ...     )
        >>> result = compare_spatial_rates(
        ...     rates([[1, 2, 3, 4, 5], [5, 1, 4, 2, 3]]),
        ...     rates([[1, 2, 3, 5, 4], [4, 1, 5, 2, 3]], (1, 1, 1, 1, 0)),
        ... )
        >>> ax = result.plot()
        >>> ax.get_title()
        '2 of 2 units scored'
        >>> type(result.plot(unit_id=9)).__name__
        'Axes'
        """
        if unit_id is not None:
            unit = self.sel(unit_id=unit_id)
            difference = unit.rate_difference[0]
            values = np.full(self.env.n_bins, np.nan)
            values[self.bins_a] = difference
            finite = np.abs(difference[np.isfinite(difference)])
            bound = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
            options = {
                "cmap": "RdBu_r",
                "vmin": -bound,
                "vmax": bound,
                "colorbar_label": "Rate difference, B \u2212 A (Hz)",
                **kwargs,
            }
            return self.env.plot_field(values, ax=ax, **options)

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        scored = self.status == "scored"
        ax.scatter(self.n_common_bins[scored], self.correlation[scored], **kwargs)
        ax.set_xlabel("Common bins")
        ax.set_ylabel(f"Correlation ({self.method})")
        ax.set_title(f"{int(scored.sum())} of {len(self)} units scored")
        return ax


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
        default the environments must have the same bins. With a mapping,
        topology is compared through it: every mapped bin must have the same
        neighbors, at the same distances, in both environments, and the
        arenas and walls must match. A mapped neighbor must be the mapped
        partner of a neighbor in the other environment; an unmapped neighbor
        must have an unmapped counterpart at the same center. So identical
        environments under a partial mapping (equivalent to ``bins=``), a
        renumbered copy, or a mapping that covers whole connected components
        have the same topology. A subset compared with its full arena does
        not: the full arena's border bins have neighbors the subset lacks, so
        smoothing over the graph used fewer neighbors there. A subset, or a
        barrier between mapped bins, needs ``topology="may_differ"``.
    bins : array-like of int, optional
        Bin IDs of ``rates_a.env`` to restrict the comparison to, for example
        ``env.bins_in_region("center")``. With ``bin_mapping``, selected bins
        that have no mapped partner are dropped; ``analysis_settings`` records
        both counts, ``bins_selected`` and ``n_compared_bins``.
    topology : {"same", "may_differ"}, default="same"
        ``"may_differ"`` accepts environments whose bins are the same places
        but whose connectivity or walls differ, as when a door opens between
        conditions. Smoothing then used different neighbors in each
        condition; the choice is recorded in ``analysis_settings``. Without
        ``bin_mapping``, connectivity is compared over all bins by bin ID,
        with the neighbor distances; with it, each mapped bin's neighbors are
        compared through the mapping (see ``bin_mapping``). Arenas and walls
        are compared whole in both cases. Maps estimated with
        ``method="binned", bandwidth=0.0`` are counts over occupancy with no
        smoothing, so connectivity did not shape them and
        ``"may_differ"`` can be passed knowingly.
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
    ``"constant_rates_both"``). A map counts as flat when its highest and
    lowest common-bin rates differ by at most ``1e-12`` times the larger
    magnitude: a uniform map from the smoothing estimators differs by about
    ``1e-14`` relative through rounding alone, and correlating that rounding
    would report a score for a flat map.

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

    Maps estimated on part of the arena pair with the full arena's bins
    through ``bin_mapping``. The part's border bins lose their neighbors
    outside it, so the topology differs and the default ``topology="same"``
    refuses. Maps counted without smoothing (``method="binned",
    bandwidth=0.0``) did not use the neighbors, so the difference can be
    accepted knowingly:

    >>> keep = np.zeros(env.n_bins, dtype=bool)
    >>> keep[: env.n_bins // 2] = True
    >>> part = env.subset(mask=keep)
    >>> full_raw, part_raw = (
    ...     compute_spatial_rates(
    ...         where,
    ...         spike_times,
    ...         times,
    ...         positions,
    ...         method="binned",
    ...         bandwidth=0.0,
    ...         unit_ids=[10, 11, 12],
    ...     )
    ...     for where in (env, part)
    ... )
    >>> mapping = np.column_stack([np.flatnonzero(keep), np.arange(part.n_bins)])
    >>> on_part = compare_spatial_rates(
    ...     full_raw, part_raw, bin_mapping=mapping, topology="may_differ"
    ... )
    >>> on_part.analysis_settings["topology_differs"]
    True
    >>> on_part.status.tolist()
    ['scored', 'scored', 'scored']
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
    """Whether each row is flat, up to rounding, over its ``common`` bins.

    A row is flat when ``max - min <= _CONSTANT_RELATIVE_SPREAD *
    max(|values|)`` over its common bins: the spread is rounding from
    computing the rates, not a field. Rows without common bins count as flat;
    the caller reports them as having too few bins first.
    """
    highest = np.where(common, values, -np.inf).max(axis=1, initial=-np.inf)
    lowest = np.where(common, values, np.inf).min(axis=1, initial=np.inf)
    magnitude = np.maximum(np.abs(highest), np.abs(lowest))
    tolerance = _CONSTANT_RELATIVE_SPREAD * magnitude
    constant: NDArray[np.bool_] = highest - lowest <= tolerance
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
    """The estimator settings that shape a rate map, JSON-safe.

    ``min_occupancy`` is recorded as the threshold the comparison applied:
    a missing or ``None`` value means no threshold, as in
    ``occupancy_support``, and is recorded as ``0.0``.
    """
    settings = rates.analysis_settings or {}
    return {
        "method": rates.method,
        "bandwidth": None if rates.bandwidth is None else float(rates.bandwidth),
        "min_occupancy": float(settings.get("min_occupancy") or 0.0),
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
