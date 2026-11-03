"""
Input/Output utilities.

This module provides tools for saving and loading environments,
regions, and neural data in various formats.

Submodules
----------
files : File I/O (to_file, from_file, to_dict, from_dict)
nwb : NeurodataWithoutBorders integration
pynapple : pynapple ingress/egress (from_pynapple / to_pynapple; optional extra)

Notes
-----
``from_pynapple`` / ``to_pynapple`` are safe to import even when pynapple is
absent: the ``import pynapple`` is lazy (inside the functions), so importing
this package never requires the optional ``pynapple`` extra. Calling them
without pynapple installed raises a clear :class:`ImportError`.
"""

from neurospatial.io import nwb
from neurospatial.io.files import from_dict, from_file, to_dict, to_file
from neurospatial.io.pynapple import from_pynapple, to_pynapple

__all__ = [
    "from_dict",
    "from_file",
    "from_pynapple",
    "nwb",
    "to_dict",
    "to_file",
    "to_pynapple",
]


def __dir__() -> list[str]:
    """List the public names, so ``dir()`` shows what the documentation teaches."""
    return sorted(__all__)
