"""Custom exception classes for the neurospatial public API.

This module collects the project's custom exceptions so users have a
single canonical import path for ``except`` blocks. Most exceptions
inherit from a stdlib base (``KeyError``, ``ValueError``,
``RuntimeError``) so that callers who catch the broader stdlib type
keep working.

Two exceptions live elsewhere for historical / dependency reasons but
are re-exported here so that ``from neurospatial import <Error>`` is the
documented import path:

* :class:`EnvironmentNotFittedError` — defined in
  :mod:`neurospatial.environment.decorators` because it ships with the
  ``check_fitted`` decorator that raises it.
* :class:`GraphValidationError` — defined in
  :mod:`neurospatial.layout.validation` next to the validator that
  emits it.
"""

from __future__ import annotations

# Re-export so users have one canonical import path.
from neurospatial.environment.decorators import EnvironmentNotFittedError
from neurospatial.layout.validation import GraphValidationError

__all__ = [
    "BinIndexOutOfRangeError",
    "EnvironmentNotFittedError",
    "GraphValidationError",
    "IncompatibleEnvironmentError",
    "LayoutNotBuiltError",
    "RegionNotFoundError",
]


class RegionNotFoundError(KeyError):
    """Raised when a region name is requested but not in the Regions container.

    Inherits from :class:`KeyError` so ``except KeyError`` blocks (e.g.
    around ``regions[name]`` lookups) keep working.

    Examples
    --------
    >>> from neurospatial._exceptions import RegionNotFoundError
    >>> try:
    ...     raise RegionNotFoundError("goal")
    ... except KeyError as exc:
    ...     print(repr(exc.args[0]))
    "Region 'goal' not found."
    """

    def __init__(self, name: str, *, available: list[str] | None = None) -> None:
        if available:
            msg = (
                f"Region '{name}' not found. Available regions: {sorted(available)!r}."
            )
        else:
            msg = f"Region '{name}' not found."
        super().__init__(msg)
        self.region_name = name
        self.available = available


class BinIndexOutOfRangeError(ValueError):
    """Raised when a bin index falls outside ``[0, n_bins)``.

    Inherits from :class:`ValueError` so existing ``except ValueError``
    blocks around bin lookups keep working.

    Examples
    --------
    >>> from neurospatial._exceptions import BinIndexOutOfRangeError
    >>> raise BinIndexOutOfRangeError(99, n_bins=42)
    Traceback (most recent call last):
        ...
    neurospatial._exceptions.BinIndexOutOfRangeError: Bin index 99 ...
    """

    def __init__(self, index: int, *, n_bins: int) -> None:
        msg = (
            f"Bin index {index} is out of range for an environment with "
            f"{n_bins} bin(s); valid indices are [0, {n_bins})."
        )
        super().__init__(msg)
        self.index = index
        self.n_bins = n_bins


class IncompatibleEnvironmentError(ValueError):
    """Raised when two environments are required to share a property but do not.

    Typical examples: composing a 2D environment with a 3D one,
    requiring matching ``bin_size`` between source and target, requiring
    the same environment type (Cartesian ``Environment`` vs egocentric
    ``EgocentricPolarEnvironment``), etc.

    Inherits from :class:`ValueError` so existing ``except ValueError``
    blocks keep working.
    """

    def __init__(
        self,
        message: str,
        *,
        first: object | None = None,
        second: object | None = None,
    ) -> None:
        super().__init__(message)
        self.first = first
        self.second = second


class LayoutNotBuiltError(RuntimeError):
    """Raised when a :class:`LayoutEngine` is accessed before ``build()``.

    Distinct from :class:`EnvironmentNotFittedError`: this signals that
    the underlying layout engine itself has not been built (e.g. its
    ``connectivity`` is ``None``). It is mostly raised inside layout
    engines and helpers; user code typically sees it surfacing through
    a layout property accessed at the wrong time.

    Inherits from :class:`RuntimeError` to match the broader
    "object-not-ready" exception family.
    """

    def __init__(self, layout_name: str, attribute: str) -> None:
        msg = (
            f"{layout_name}.{attribute} is unavailable: the layout has not been "
            "built yet. Call layout.build() (or use a factory like "
            "Environment.from_samples) before accessing this attribute."
        )
        super().__init__(msg)
        self.layout_name = layout_name
        self.attribute = attribute
