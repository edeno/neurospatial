"""Length units the package can convert defaults into.

Defaults stated in centimeters (a 5 cm smoothing bandwidth, a 5 cm/s speed
threshold) are expressed in an environment's own units through
:func:`units_per_centimeter`. Pixels and custom units have no conversion.
"""

from __future__ import annotations

from neurospatial._exceptions import _format_error

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
