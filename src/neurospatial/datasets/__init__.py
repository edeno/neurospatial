"""Public spatial neuroscience datasets, downloaded once and checked.

Importing this module needs no optional dependency; the loaders ask for the
``datasets`` extra (``pip install 'neurospatial[datasets]'``) when they need it.
"""

from __future__ import annotations

__all__: list[str] = []


def __dir__() -> list[str]:
    """List the public names."""
    return sorted(__all__)
