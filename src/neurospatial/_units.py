"""Length units the package can convert defaults into.

Defaults stated in centimeters (a 5 cm smoothing bandwidth, a 5 cm/s speed
threshold) are expressed in an environment's own units through
:func:`units_per_centimeter`. Pixels and custom units have no conversion.
"""

from __future__ import annotations

# Centimeters in one unit, for every accepted spelling of a metric length unit.
_CENTIMETERS_PER_UNIT = {
    **dict.fromkeys(["m", "meter", "meters", "metre", "metres"], 100.0),
    **dict.fromkeys(
        ["cm", "centimeter", "centimeters", "centimetre", "centimetres"], 1.0
    ),
    **dict.fromkeys(
        ["mm", "millimeter", "millimeters", "millimetre", "millimetres"], 0.1
    ),
    **dict.fromkeys(["um", "µm", "micrometer", "micrometers", "micrometre"], 1e-4),
}


def units_per_centimeter(units: str | None) -> float | None:
    """How many ``units`` make one centimeter, or None for a non-metric unit.

    Parameters
    ----------
    units : str or None
        An environment's ``units``, any case and surrounding whitespace.

    Returns
    -------
    float or None
        For example 100.0 for ``"mm"`` and 0.01 for ``"m"``; None for pixels,
        custom units and unset units.
    """
    centimeters = _CENTIMETERS_PER_UNIT.get(str(units or "").lower().strip())
    return None if centimeters is None else 1.0 / centimeters
