"""Machinery shared by rate maps over a (distance, direction) polar grid.

Object-vector and boundary-vector results index the same egocentric polar
environment: distance in the first coordinate, direction in radians in the
second, in the result's ``direction_frame``. This module holds what they
compute alike; each family keeps its own public methods and documentation.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from neurospatial.encoding._base import _to_numpy

if TYPE_CHECKING:
    import pandas as pd


class _PolarRateMixin:
    """Bin centers, export metadata and peak coordinates of a polar rate map.

    Hosts declare ``env`` (the polar environment), ``direction_frame``,
    ``distance_range``, ``n_distance_bins`` and ``n_direction_bins``, and
    inherit :class:`~neurospatial.encoding._base.SpatialResultMixin` after
    this class.
    """

    env: Any
    direction_frame: str
    distance_range: tuple[float, float]
    n_distance_bins: int
    n_direction_bins: int

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


def polar_summary_frame(
    result: Any,
    *,
    index: Sequence[Hashable],
    extra: Mapping[str, NDArray[Any]],
    extra_units: Mapping[str, str],
    thresholds: Mapping[str, float] | None,
) -> pd.DataFrame:
    """One row per unit with the peak and its distance and direction.

    The columns are ``peak_rate``, ``preferred_distance``,
    ``preferred_direction_deg`` and ``preferred_direction``, then ``extra``
    (a family's metrics, with any classification last). Singular and
    population results get identical columns.
    """
    import pandas as pd

    peaks = np.atleast_2d(result.peak_location())
    columns = {
        "peak_rate": np.atleast_1d(result.peak_firing_rate()),
        "preferred_distance": peaks[:, 0],
        "preferred_direction_deg": np.degrees(peaks[:, 1]),
        "preferred_direction": peaks[:, 1],
        **extra,
    }
    df = pd.DataFrame(columns, index=pd.Index(list(index), name="unit_id"))
    df.attrs["direction_frame"] = result.direction_frame
    df.attrs["units"] = {
        "peak_rate": "Hz",
        "preferred_distance": result.env.units or "",
        "preferred_direction": "rad",
        "preferred_direction_deg": "deg",
        **extra_units,
    }
    if thresholds is not None:
        df.attrs["classification_thresholds"] = dict(thresholds)
    df.attrs.update(result._analysis_attrs())
    return df
