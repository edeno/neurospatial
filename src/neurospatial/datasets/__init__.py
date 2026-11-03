"""Public spatial neuroscience datasets, downloaded once and checked.

A loaded session is a :class:`Recording`: the arrays neurospatial's
functions take (``times``, ``positions``, ``spike_times``, ...), the
conversions the loader applied (:class:`Conversion`), where the file came
from (:class:`Provenance`), and every known pitfall of the session. Printing
it shows the full summary.

Importing this module needs no optional dependency; the loaders ask for the
``datasets`` extra (``pip install 'neurospatial[datasets]'``) when they need it.
"""

from __future__ import annotations

from neurospatial.datasets._recording import Conversion, Provenance, Recording

__all__ = ["Conversion", "Provenance", "Recording"]


def __dir__() -> list[str]:
    """List the public names."""
    return sorted(__all__)
