"""Threshold comparisons that treat round-off as equality.

Replay criteria compare measured values, such as segment durations built
from ``t0 + 0.005 * k``, with thresholds a paper states exactly. A value
within ``ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * |threshold|`` of the
threshold counts as equal to it, so rounding does not flip a strict test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._constants import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE
from neurospatial._validation import validate_choice

# The comparisons a criterion can state, elementwise.
_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def _tolerance(threshold: Any) -> Any:
    """Distance from ``threshold`` within which a value counts as equal."""
    return ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(threshold)


def _compare(values: Any, comparison: str, thresholds: Any) -> NDArray[np.bool_]:
    """Compare ``values`` with ``thresholds``, equal within round-off.

    Parameters
    ----------
    values, thresholds : float or array_like
        Broadcast against each other.
    comparison : {"<", "<=", ">", ">="}
        The comparison, as the paper states it.

    Returns
    -------
    NDArray[np.bool_]
        The broadcast comparison. A value within :func:`_tolerance` of its
        threshold is compared as the threshold itself; a NaN or infinite
        value or threshold fails.
    """
    values = np.asarray(values, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        close = np.abs(values - thresholds) <= _tolerance(thresholds)
        result = _COMPARISONS[comparison](
            np.where(close, thresholds, values), thresholds
        )
    passed: NDArray[np.bool_] = result & np.isfinite(values) & np.isfinite(thresholds)
    return passed


def _check_criterion_settings(
    prefix: str, comparison: Any, relative_to: Any, absolute: Any
) -> None:
    """Refuse an unknown comparison or scale, or a non-bool ``absolute``."""
    validate_choice(comparison, tuple(_COMPARISONS), name="comparison", owner=prefix)
    validate_choice(
        relative_to, (None, "track_length"), name="relative_to", owner=prefix
    )
    if not isinstance(absolute, (bool, np.bool_)):
        raise ValueError(f"{prefix}: absolute must be True or False, got {absolute!r}.")


def _criterion_label(
    quantity: str,
    comparison: str,
    *,
    absolute: bool,
    relative_to: str | None,
    threshold: float | None = None,
) -> str:
    """A readable criterion, such as ``"|score| > 0.6"`` or ``"|score| >"``."""
    name = f"|{quantity}|" if absolute else quantity
    bound = "" if threshold is None else f" {threshold:g}"
    scale = " x track_length" if relative_to else ""
    return f"{name} {comparison}{bound}{scale}"
