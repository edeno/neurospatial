"""Import optional dependencies on first use."""

from __future__ import annotations

import sys
from types import ModuleType

from neurospatial._exceptions import _format_error


def require_optional(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import and return an optional dependency, naming the extra if it is absent.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"pynwb"`` or ``"ndx_pose"``.
    extra : str
        The neurospatial extra that installs it, e.g. ``"nwb"``.
    purpose : str
        What needs it, completing "``<module> is required for ...``".

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If ``module`` itself is not installed. Any other import failure, such
        as a missing or broken dependency of ``module``, propagates unchanged
        so it is not mistaken for a missing install.
    """
    try:
        __import__(module)
    except ModuleNotFoundError as exc:
        if exc.name != module:
            raise
        raise ImportError(
            _format_error(
                f"{module} is required for {purpose} but is not installed.",
                fix=f"pip install 'neurospatial[{extra}]'",
            )
        ) from exc
    return sys.modules[module]
