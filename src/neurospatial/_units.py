"""Length units the package can convert defaults into.

Defaults stated in centimeters (a 5 cm smoothing bandwidth, a 5 cm/s speed
threshold) are expressed in an environment's own units through
:func:`units_per_centimeter`. Pixels and custom units have no conversion.
"""

from __future__ import annotations

from neurospatial._exceptions import _format_error

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


def same_coordinate_declaration(
    name: str,
    first: str | None,
    second: str | None,
    *,
    undeclared_matches: bool,
) -> bool:
    """Whether two ``units`` or ``frame`` declarations describe the same coordinates.

    Units are compared by canonical spelling (``"metres"`` matches ``"m"``);
    frames verbatim. Nothing is converted.

    Parameters
    ----------
    name : {"units", "frame"}
        Which declaration is compared.
    first, second : str or None
        The two declarations; None is undeclared.
    undeclared_matches : bool
        What an undeclared side means. True: it takes the other side's
        coordinates, as an overlay without units is in its environment's.
        False: one-sided metadata cannot establish a correspondence, so only
        two undeclared sides match.

    Returns
    -------
    bool
    """
    if first is None or second is None:
        return undeclared_matches or first is second
    if name == "units":
        return normalize_position_units(first) == normalize_position_units(second)
    return first == second


def default_speed(
    units: str | None,
    cm_per_s: float,
    *,
    name: str,
    call: str,
    problem: str,
    placeholder: str,
) -> float:
    """A default speed of ``cm_per_s`` expressed in ``units`` per second.

    Parameters
    ----------
    units : str or None
        The environment's ``units``.
    cm_per_s : float
        The default speed in centimeters per second.
    name : str
        The speed argument left as None, such as ``"min_speed"``.
    call : str
        The public call the scientist made, named in the error.
    problem : str
        Why the default cannot be converted, completing "the environment's
        units (...) are ...", such as ``"not a known length, so the 5 cm/s
        default"``.
    placeholder : str
        What the scientist passes instead, such as ``"threshold"``.

    Returns
    -------
    float
        The default speed in position units per second.

    Raises
    ------
    ValueError
        If ``units`` is not a metric length unit.
    """
    per_cm = units_per_centimeter(units)
    if per_cm is None:
        raise ValueError(
            _format_error(
                f"{call} needs {name}: the environment's units ({units!r}) are "
                f"{problem} cannot be expressed in them.",
                why=f"Why: {name} is in position units per second.",
                fix=f"pass {name}=<{placeholder} in position units per second>, "
                "or declare the units with env.with_metadata(units='cm')",
            )
        )
    return cm_per_s * per_cm
