"""Serializable analysis settings and spatial coverage for rate results."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._intervals import TimeSupport


def rate_analysis_settings(
    support: TimeSupport,
    *,
    method: str,
    bandwidth: float | None,
    backend: str,
    min_occupancy: float | None = None,
    min_speed: float | None = None,
    speed: Any = None,
    fill_value: float | None = None,
    penalty: float | None = None,
    rank: int | None = None,
    pooled: bool = True,
    **family_settings: Any,
) -> dict[str, Any]:
    """Record resolved choices without retaining the input recording arrays."""
    return {
        **support.analysis_settings(),
        "min_speed": None if min_speed is None else float(min_speed),
        "speed_source": None
        if min_speed is None
        else ("provided" if speed is not None else "derived_from_positions"),
        "method": method,
        "bandwidth": None if bandwidth is None or method == "glm" else float(bandwidth),
        "min_occupancy": None
        if min_occupancy is None or method == "glm"
        else float(min_occupancy),
        "fill_value": None if fill_value is None else float(fill_value),
        "backend": backend,
        "penalty": None if penalty is None else float(penalty),
        "rank": None if rank is None else int(rank),
        "pooled": bool(pooled) if method == "glm" else None,
        **deepcopy(family_settings),
    }


def rate_xarray_attrs(result: Any) -> dict[str, Any]:
    """Build common, NetCDF-safe rate metadata in one place."""
    from neurospatial._results import env_fingerprint, software_version, units_attr

    attrs: dict[str, Any] = {
        "software_version": software_version(),
        "spike_window_assumed": int(result.spike_window_assumed),
    }
    env = getattr(result, "env", None)
    if env is not None:
        attrs.update(units_attr(env))
        attrs["env"] = env_fingerprint(env)
    for key in ("method", "bandwidth"):
        value = getattr(result, key, None)
        if value is not None:
            attrs[key] = value
    if result.spike_window is not None:
        attrs["spike_window"] = np.asarray(result.spike_window).ravel().copy()
    if result.analysis_settings is not None:
        attrs["analysis_settings"] = json.dumps(
            result.analysis_settings, sort_keys=True
        )
    return attrs


def occupancy_support(result: Any) -> NDArray[np.bool_]:
    """Bins the animal occupied for at least the result's ``min_occupancy``.

    Parameters
    ----------
    result : object
        A rate result with ``occupancy`` ``(n_bins,)`` and ``min_occupancy``,
        both in seconds.

    Returns
    -------
    ndarray of bool, shape (n_bins,)
        True where occupancy is positive and at least ``min_occupancy``.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> import numpy as np
    >>> result = SimpleNamespace(
    ...     occupancy=np.array([0.0, 0.2, 1.0]),
    ...     min_occupancy=0.5,
    ... )
    >>> occupancy_support(result).tolist()
    [False, False, True]
    """
    occupancy = np.asarray(result.occupancy)
    support: NDArray[np.bool_] = np.asarray(
        (occupancy > 0) & (occupancy >= result.min_occupancy), dtype=bool
    )
    return support


def spatial_support(result: Any) -> NDArray[np.bool_]:
    """Bins with observed occupancy, sufficient dwell and finite model rates."""
    occupied = occupancy_support(result)
    rates = np.asarray(result._get_rates())
    finite = np.isfinite(rates)
    if rates.ndim == 2:
        finite = (
            finite.all(axis=0) if len(rates) else np.zeros_like(occupied, dtype=bool)
        )
    support: NDArray[np.bool_] = np.asarray(occupied & finite, dtype=bool)
    return support
