"""Length units the package can convert defaults into.

Defaults stated in centimeters (a 5 cm smoothing bandwidth, a 5 cm/s speed
threshold) are expressed in an environment's own units through
:func:`units_per_centimeter`. Pixels and custom units have no conversion.
"""

from __future__ import annotations

# Every accepted spelling of a length unit, mapped to its canonical name.
_UNIT_SPELLINGS = {
    **dict.fromkeys(["m", "meter", "meters", "metre", "metres"], "m"),
    **dict.fromkeys(
        ["cm", "centimeter", "centimeters", "centimetre", "centimetres"], "cm"
    ),
    **dict.fromkeys(
        ["mm", "millimeter", "millimeters", "millimetre", "millimetres"], "mm"
    ),
    **dict.fromkeys(
        ["um", "µm", "micrometer", "micrometers", "micrometre", "micrometres"], "um"
    ),
    **dict.fromkeys(["px", "pixel", "pixels"], "px"),
}

# Centimeters in one canonical metric unit. Pixels have no conversion.
_CENTIMETERS_PER_UNIT = {"m": 100.0, "cm": 1.0, "mm": 0.1, "um": 1e-4}


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
    centimeters = _CENTIMETERS_PER_UNIT.get(normalize_position_units(str(units or "")))
    return None if centimeters is None else 1.0 / centimeters


def normalize_position_units(units: str) -> str:
    """Canonical spelling for matching spatial metadata, without conversion.

    Unknown units are returned lowercased and stripped.
    """
    value = units.strip().lower()
    return _UNIT_SPELLINGS.get(value, value)
