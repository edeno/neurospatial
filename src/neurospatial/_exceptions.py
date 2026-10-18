"""Public exception classes and error-message formatting.

Library errors inherit a standard Python exception first and the common
NeurospatialError base second. This module has no internal dependencies.
"""

from __future__ import annotations

import functools
from difflib import get_close_matches
from typing import Any, Literal

__all__ = [
    "BinIndexOutOfRangeError",
    "CartesianOnlyError",
    "IncompatibleEnvironmentError",
    "LayoutNotBuiltError",
    "NeurospatialError",
    "RegionNotFoundError",
]


class NeurospatialError(Exception):
    """Base class for every exception neurospatial defines.

    Each concrete error also inherits a built-in type, listed first, so
    ``except ValueError`` keeps working. ``except NeurospatialError`` catches
    only problems that neurospatial itself detected.

    Subclasses build their message from structured constructor arguments, so
    the default pickling (which re-calls the class with the formatted message)
    would fail or double-format. Each subclass ``__init__`` therefore records
    its arguments, and ``__reduce__`` rebuilds the error from them, which lets
    errors raised in worker processes reach the caller intact.
    """

    _init_arguments: tuple[tuple[Any, ...], dict[str, Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None:
            return

        @functools.wraps(init)
        def recording_init(self: NeurospatialError, *args: Any, **kw: Any) -> None:
            init(self, *args, **kw)
            self._init_arguments = (args, kw)

        setattr(cls, "__init__", recording_init)  # noqa: B010

    def __reduce__(self) -> str | tuple[Any, ...]:
        arguments = self.__dict__.get("_init_arguments")
        if arguments is None:
            return super().__reduce__()
        args, kwargs = arguments
        return (_rebuild_error, (type(self), args, kwargs), self.__dict__)


def _rebuild_error(
    cls: type[NeurospatialError], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> NeurospatialError:
    """Unpickling hook: call the error class with its original arguments."""
    return cls(*args, **kwargs)


def _format_error(what: str, *, fix: str, why: str | None = None) -> str:
    """Return ``what``, an optional ``why``, and a final ``Fix:`` line."""
    lines = [what.strip()] + ([why.strip()] if why else []) + [f"Fix: {fix.strip()}"]
    return "\n".join(lines)


class RegionNotFoundError(KeyError, ValueError, NeurospatialError):
    """Raised when a region name is requested but not in the Regions container.

    Inherits from :class:`KeyError` and :class:`ValueError` so existing
    catch blocks for region lookups and segmentation keep working.

    Parameters
    ----------
    name : str or None
        Requested region name; ``None`` reports a missing name with guidance.
    available : list of str, optional
        Available region names, used to suggest a close match.
    argument : str or None, default "region_name"
        Calling function's argument name, shown in the corrected call. None
        for a lookup by key, such as ``env.regions[name]``, which suggests
        the corrected name alone.
    in_list : bool, default False
        Whether ``argument`` takes a list of names, so the corrected call
        shows a list.

    Examples
    --------
    >>> from neurospatial._exceptions import RegionNotFoundError
    >>> try:
    ...     raise RegionNotFoundError("goal")
    ... except KeyError as exc:
    ...     print(str(exc))
    Region 'goal' not found.
    Fix: create regions = Regions(env.regions.values()); regions.add('goal', point=(x, y)) (or polygon=...); env = env.with_regions(regions), then pass region_name='goal'.
    """

    def __init__(
        self,
        name: str | None,
        *,
        available: list[str] | None = None,
        argument: str | None = "region_name",
        in_list: bool = False,
    ) -> None:
        if available:
            what = (
                f"Region '{name}' not found. Available regions: {sorted(available)!r}."
            )
        elif available is not None:
            what = f"Region '{name}' not found. This environment has no regions."
        else:
            what = f"Region '{name}' not found."
        matches = (
            get_close_matches(name, available or [], n=1)
            if isinstance(name, str)
            else []
        )

        def use(region: str) -> str:
            if argument is None:
                return f"use {region!r}"
            value = f"[{region!r}]" if in_list else repr(region)
            return f"pass {argument}={value}"

        if not isinstance(name, str):
            what = (
                f"Region name {name!r} was not found; region names must be strings."
                if argument is None
                else f"{argument}={name!r} was not found; region names must be strings."
            )
            suggested = available[0] if available else "home"
            fix = (
                f"{use(suggested)}; if a new region is needed, create "
                "regions = Regions(env.regions.values()); "
                f"regions.add({suggested!r}, point=(x, y)); "
                "env = env.with_regions(regions)"
            )
        elif matches:
            fix = use(matches[0])
        else:
            then = "" if argument is None else f", then {use(name)}"
            fix = (
                "create regions = Regions(env.regions.values()); "
                f"regions.add({name!r}, point=(x, y)) (or polygon=...); "
                f"env = env.with_regions(regions){then}."
            )
        super().__init__(_format_error(what, fix=fix))
        self.region_name = name
        self.available = available

    def __str__(self) -> str:
        return str(self.args[0])


class BinIndexOutOfRangeError(ValueError, NeurospatialError):
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

    Parameters
    ----------
    index : int
        The refused index.
    n_bins : int
        Number of active bins.
    coordinate_kind : {"cartesian", "polar"}, default="cartesian"
        The environment's coordinates, so the fix shows the rows ``bin_at``
        takes there: a point for Cartesian bins, ``(distance, angle)`` for
        egocentric polar bins.
    """

    def __init__(
        self,
        index: int,
        *,
        n_bins: int,
        coordinate_kind: Literal["cartesian", "polar"] = "cartesian",
    ) -> None:
        msg = (
            f"Bin index {index} is out of range for an environment with "
            f"{n_bins} bin(s); valid indices are [0, {n_bins})."
        )
        lookup = (
            "int(env.bin_at([[distance, angle]])[0])"
            if coordinate_kind == "polar"
            else "int(env.bin_at([point])[0])"
        )
        why = (
            "Why: -1 is the value bin_at gives a position outside the "
            "environment; it marks no bin."
            if index == -1
            else None
        )
        super().__init__(
            _format_error(
                msg,
                why=why,
                fix=f"pass a bin index in [0, {n_bins}); obtain one with {lookup}.",
            )
        )
        self.index = index
        self.n_bins = n_bins


class IncompatibleEnvironmentError(ValueError, NeurospatialError):
    """Raised when two environments are required to share a property but do not.

    Typical examples: composing a 2D environment with a 3D one,
    requiring matching ``bin_size`` between source and target, requiring
    compatible coordinate meanings (Cartesian positions or egocentric
    radius/angle), etc.

    Inherits from :class:`ValueError` so existing ``except ValueError``
    blocks keep working.

    Parameters
    ----------
    message : str
        What differs between the environments.
    fix : str, optional
        The ``Fix:`` line for this mismatch. The default names no specific
        remedy; raise sites pass the fix for their own check.
    first, second : object, optional
        The two mismatched objects, kept as attributes for inspection.
    """

    def __init__(
        self,
        message: str,
        *,
        fix: str = "use environments that share the property named above.",
        first: object | None = None,
        second: object | None = None,
    ) -> None:
        super().__init__(_format_error(message, fix=fix))
        self.first = first
        self.second = second


class CartesianOnlyError(TypeError, NeurospatialError):
    """Raised when an operation needs Cartesian positions and the environment is polar.

    A polar ``Environment`` stores ``(distance, angle)`` bins around the
    animal, so an operation that reads positions or bin centers as Cartesian
    ``(x, y[, z])`` coordinates would return plausible but wrong numbers. The
    operation does not apply to this kind of environment, so the error is a
    :class:`TypeError`; static type checkers flag the same calls. Every
    Cartesian-only method, Cartesian-only mode and Cartesian-only analysis
    function raises it with the same message layout: what was refused, why it
    needs Cartesian coordinates, and what works on a polar environment.

    Parameters
    ----------
    operation : str
        The refused call, e.g. ``"Environment.bin_at()"`` or
        ``"Environment.distance_to(metric='euclidean')"``.
    why : str
        Why the operation needs Cartesian coordinates.
    fix : str
        What to use instead on a polar environment.
    example : str or None, optional
        Code for the alternative, shown after the fix: statements separated by
        ``"; "`` whose last part is an expression giving the result. It uses
        ``np`` and names such as ``env``, ``points`` or ``field``. None when
        the alternative needs other data than the refused call had.

    Attributes
    ----------
    operation : str
        The refused call.
    example : str or None
        The code shown after the fix.
    """

    def __init__(
        self, operation: str, *, why: str, fix: str, example: str | None = None
    ) -> None:
        shown = fix if example is None else f"{fix}: {example}"
        super().__init__(
            _format_error(
                f"{operation} does not work on a polar environment.",
                why=(
                    f"Why: {why} This environment's bins are (distance, angle) "
                    "pairs around the animal, not Cartesian (x, y) positions."
                ),
                fix=shown,
            )
        )
        self.operation = operation
        self.example = example


class LayoutNotBuiltError(RuntimeError, NeurospatialError):
    """Raised when a :class:`LayoutEngine` is accessed before ``build()``.

    This signals that the layout engine has not been built (e.g. its
    ``binning()`` is requested before ``build()``). It is mostly raised inside layout
    engines and helpers; user code typically sees it surfacing through
    a layout property accessed at the wrong time.

    Inherits from :class:`RuntimeError` to match the broader
    "object-not-ready" exception family.
    """

    def __init__(self, layout_name: str, attribute: str) -> None:
        msg = f"{layout_name}.{attribute} is unavailable: the layout is not built yet."
        super().__init__(
            _format_error(
                msg,
                fix="call build() on the layout first, or create the environment with a factory such as Environment.from_samples(positions, bin_size=2.0).",
            )
        )
        self.layout_name = layout_name
        self.attribute = attribute
