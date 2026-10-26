"""Machinery shared by rate maps over a (distance, direction) polar grid.

Object-vector and boundary-vector results index the same egocentric polar
environment: distance in the first coordinate, direction in radians in the
second, in the result's ``direction_frame``. This module holds what they
compute alike; each family keeps its own public methods and documentation.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from neurospatial.encoding._base import KERNEL_ESTIMATORS, BandwidthRule, _to_numpy

if TYPE_CHECKING:
    import pandas as pd


POLAR_ESTIMATORS: Mapping[str, BandwidthRule] = MappingProxyType(
    {**KERNEL_ESTIMATORS, "binned": "none"}
)
"""Polar estimators: ``"binned"`` is the raw bin rate and does not smooth."""


class _PolarRateMixin:
    """Bin centers, export metadata and peak coordinates of a polar rate map.

    Hosts declare ``env`` (the polar environment) and ``direction_frame``,
    and inherit :class:`~neurospatial.encoding._base.SpatialResultMixin`
    after this class. The grid's extent and shape are read from ``env``, the
    one record of them, so they cannot disagree with the bins.
    """

    env: Any
    direction_frame: str
    _ESTIMATORS: ClassVar[Mapping[str, BandwidthRule]] = POLAR_ESTIMATORS

    @property
    def distance_range(self) -> tuple[float, float]:
        """(min, max) distance covered by the bins, in the environment's units."""
        edges = self.env.grid_edges[0]
        return (float(edges[0]), float(edges[-1]))

    @property
    def n_distance_bins(self) -> int:
        """Number of distance bins (rings) of the polar grid."""
        return int(self.env.grid_shape[0])

    @property
    def n_direction_bins(self) -> int:
        """Number of direction bins of the polar grid."""
        return int(self.env.grid_shape[1])

    @property
    def _bin_centers(self) -> NDArray[np.float64]:
        # Polar results index the (distance, direction) bins of their own
        # environment, not world positions.
        bin_centers: NDArray[np.float64] = self.env.bin_centers
        return bin_centers

    def _family_attrs(self) -> dict[str, Any]:
        """Metadata a family adds to ``to_xarray`` attributes."""
        return {}

    def _xarray_attrs(self) -> dict[str, Any]:
        return {
            "direction_frame": self.direction_frame,
            "distance_range": list(self.distance_range),
            "n_distance_bins": self.n_distance_bins,
            "n_direction_bins": self.n_direction_bins,
            **self._family_attrs(),
        }

    def _peak_coordinates(self, column: int) -> NDArray[np.float64]:
        """Coordinate ``column`` of each unit's peak bin, NaN for an empty map.

        Returns
        -------
        NDArray[np.float64], shape (n_units,)
            Distance (``column=0``) or direction (``column=1``) of the bin
            with the highest finite rate.
        """
        rates = np.atleast_2d(_to_numpy(self._get_rates()))  # type: ignore[attr-defined]
        observed = np.isfinite(rates).any(axis=1)
        peaks = np.full(len(rates), np.nan)
        if observed.any():
            peak_bins = np.nanargmax(rates[observed], axis=1)
            peaks[observed] = self._bin_centers[peak_bins, column]
        return peaks


def _unknown_units(units: str | None) -> str:
    """Why a length default cannot be converted, for error messages."""
    if units is None:
        return "the length units are unknown (no env, or env.units is unset)"
    return f"env.units={units!r} is not a known length unit"


def resolve_distance_range(
    units: str | None,
    distance_range: tuple[float, float] | None,
    *,
    context: str,
) -> tuple[float, float]:
    """The polar grid's distance range, defaulting to 0-50 cm.

    Parameters
    ----------
    units : str or None
        Length units of the positions (``env.units``), or None when unknown.
    distance_range : tuple of float or None
        The caller's range, in those units; None for the default.
    context : str
        Public function name for error messages.

    Returns
    -------
    tuple of float
        ``(min, max)`` in the positions' units.

    Raises
    ------
    ValueError
        If ``distance_range`` is None and ``units`` is not a known length
        unit: 50 has no meaning without a scale.
    """
    from neurospatial._exceptions import _format_error
    from neurospatial._units import units_per_centimeter

    if distance_range is not None:
        low, high = distance_range
        return (float(low), float(high))
    per_cm = units_per_centimeter(units)
    if per_cm is None:
        raise ValueError(
            _format_error(
                f"{context}: distance_range is required because "
                f"{_unknown_units(units)}.",
                why="Why: the default 0-50 cm range cannot be converted to "
                "coordinates of unknown scale",
                fix="pass distance_range=(0.0, max_distance) in the units of "
                "positions, or an env with units='cm' (or 'mm', 'm')",
            )
        )
    return (0.0, 50.0 * per_cm)


def require_known_bandwidth(
    units: str | None,
    bandwidth: float | None,
    method: str,
    *,
    context: str,
) -> None:
    """Refuse a default kernel bandwidth that has no physical length.

    The default is 5 cm converted to ``units``. A polar grid's graph edges
    mix radial steps with arcs that grow with distance, so no length can be
    inferred from them; without known units the caller must choose.

    Raises
    ------
    ValueError
        If a kernel method has no ``bandwidth`` and ``units`` is not a known
        length unit.
    """
    from neurospatial._exceptions import _format_error
    from neurospatial._units import units_per_centimeter

    if method == "binned" or bandwidth is not None:
        return
    if units_per_centimeter(units) is None:
        raise ValueError(
            _format_error(
                f"{context}: method={method!r} needs a bandwidth because "
                f"{_unknown_units(units)}.",
                why="Why: the default 5 cm kernel cannot be converted to "
                "coordinates of unknown scale",
                fix="pass bandwidth in the units of positions, an env with "
                "units='cm' (or 'mm', 'm'), or method='binned'",
            )
        )


def polar_summary_frame(
    result: Any,
    *,
    index: Sequence[Hashable],
    extra: Mapping[str, NDArray[Any]],
    extra_units: Mapping[str, str],
    thresholds: Mapping[str, float] | None,
) -> pd.DataFrame:
    """One row per unit with the peak and its distance and direction.

    The columns are ``peak_firing_rate``, ``preferred_distance``,
    ``preferred_direction_deg`` and ``preferred_direction``, then ``extra``
    (a family's metrics, with any classification last). Singular and
    population results get identical columns.
    """
    import pandas as pd

    peaks = np.atleast_2d(result.peak_location())
    columns = {
        "peak_firing_rate": np.atleast_1d(result.peak_firing_rate()),
        "preferred_distance": peaks[:, 0],
        "preferred_direction_deg": np.degrees(peaks[:, 1]),
        "preferred_direction": peaks[:, 1],
        **extra,
    }
    df = pd.DataFrame(columns, index=pd.Index(list(index), name="unit_id"))
    df.attrs["direction_frame"] = result.direction_frame
    df.attrs["units"] = {
        "peak_firing_rate": "Hz",
        "preferred_distance": result.env.units or "",
        "preferred_direction": "rad",
        "preferred_direction_deg": "deg",
        **extra_units,
    }
    if thresholds is not None:
        df.attrs["classification_thresholds"] = dict(thresholds)
    df.attrs.update(result._analysis_attrs())
    return df
