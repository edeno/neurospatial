"""Public exception classes and error-message formatting.

Library errors inherit a standard Python exception first and the common
NeurospatialError base second. This module has no internal dependencies.
"""

from __future__ import annotations

import functools
from difflib import get_close_matches
from typing import Any

__all__ = [
    "BinIndexOutOfRangeError",
    "EnvironmentNotFittedError",
    "GraphValidationError",
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
    argument : str, default "region_name"
        Calling function's argument name, shown in the corrected call.

    Examples
    --------
    >>> from neurospatial._exceptions import RegionNotFoundError
    >>> try:
    ...     raise RegionNotFoundError("goal")
    ... except KeyError as exc:
    ...     print(str(exc))
    Region 'goal' not found.
    Fix: add it first: env.regions.add('goal', point=(x, y)) (or polygon=...), then pass region_name='goal'.
    """

    def __init__(
        self,
        name: str | None,
        *,
        available: list[str] | None = None,
        argument: str = "region_name",
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
        if not isinstance(name, str):
            what = f"{argument}={name!r} was not found; region names must be strings."
            suggested = available[0] if available else "home"
            value = f"[{suggested!r}]" if argument == "end_regions" else repr(suggested)
            fix = f"pass {argument}={value}; add it first with env.regions.add({suggested!r}, point=(x, y)) if needed"
        elif matches:
            value = (
                f"[{matches[0]!r}]" if argument == "end_regions" else repr(matches[0])
            )
            fix = f"pass {argument}={value}"
        else:
            value = f"[{name!r}]" if argument == "end_regions" else repr(name)
            fix = (
                f"add it first: env.regions.add({name!r}, point=(x, y)) "
                f"(or polygon=...), then pass {argument}={value}."
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
    """

    def __init__(self, index: int, *, n_bins: int) -> None:
        msg = (
            f"Bin index {index} is out of range for an environment with "
            f"{n_bins} bin(s); valid indices are [0, {n_bins})."
        )
        super().__init__(
            _format_error(
                msg,
                fix=f"pass a bin index in [0, {n_bins}); obtain one with int(env.bin_at([point])[0]).",
            )
        )
        self.index = index
        self.n_bins = n_bins


class IncompatibleEnvironmentError(ValueError, NeurospatialError):
    """Raised when two environments are required to share a property but do not.

    Typical examples: composing a 2D environment with a 3D one,
    requiring matching ``bin_size`` between source and target, requiring
    the same environment type (Cartesian ``Environment`` vs egocentric
    ``EgocentricPolarEnvironment``), etc.

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


class LayoutNotBuiltError(RuntimeError, NeurospatialError):
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
        msg = f"{layout_name}.{attribute} is unavailable: the layout is not built yet."
        super().__init__(
            _format_error(
                msg,
                fix="call build() on the layout first, or create the environment with a factory such as Environment.from_samples(positions, bin_size=2.0).",
            )
        )
        self.layout_name = layout_name
        self.attribute = attribute


class EnvironmentNotFittedError(RuntimeError, NeurospatialError):
    """Exception raised when an unfitted Environment is consumed.

    This exception is raised both by the :func:`check_fitted` decorator on
    bound methods and by free functions that receive an :class:`Environment`
    argument. It supports two construction shapes:

    1. Bound-method form: ``EnvironmentNotFittedError(class_name, method_name)``
       — formats the message as ``Environment.method()`` with factory-method
       guidance.
    2. Free-function form:
       ``EnvironmentNotFittedError(function_name, *, is_function=True)`` —
       formats the message as ``function()`` (no class qualifier) and the
       same guidance about factory methods.

    Parameters
    ----------
    class_or_function_name : str
        For the bound-method form, the Environment class name (e.g.
        "Environment"). For the free-function form, the qualified function
        name (e.g. "path_progress" or "neurospatial.behavior.navigation.path_progress").
    method_name : str, optional
        Name of the method requiring initialization. Required for the
        bound-method form; ignored when ``is_function=True``.
    error_code : str, optional
        Error code for documentation reference. Default is "E1004".
    is_function : bool, optional
        If True, format the message as a free function (omit class
        qualifier). Default is False.

    Examples
    --------
    >>> from neurospatial.environment.decorators import EnvironmentNotFittedError
    >>> raise EnvironmentNotFittedError("Environment", "bin_at")
    Traceback (most recent call last):
        ...
    neurospatial._exceptions.EnvironmentNotFittedError: [E1004] Environment.bin_at() requires...

    >>> raise EnvironmentNotFittedError("path_progress", is_function=True)
    Traceback (most recent call last):
        ...
    neurospatial._exceptions.EnvironmentNotFittedError: [E1004] path_progress() requires...

    See Also
    --------
    check_fitted : Decorator that raises this exception for bound methods.

    Notes
    -----
    This exception inherits from ``RuntimeError`` to maintain backward
    compatibility with existing code that catches ``RuntimeError``. Users
    can catch either ``EnvironmentNotFittedError`` for specific handling or
    ``RuntimeError`` for general error handling.

    """

    def __init__(
        self,
        class_or_function_name: str,
        method_name: str | None = None,
        *,
        is_function: bool = False,
        error_code: str = "E1004",
    ) -> None:
        if is_function:
            qualified = f"{class_or_function_name}()"
            class_name: str | None = None
            method_name_resolved = class_or_function_name
        else:
            if method_name is None:
                raise TypeError(
                    "EnvironmentNotFittedError requires `method_name` "
                    "when `is_function=False` (the default bound-method form)."
                )
            qualified = f"{class_or_function_name}.{method_name}()"
            class_name = class_or_function_name
            method_name_resolved = method_name

        message = (
            f"[{error_code}] {qualified} "
            "requires the environment to be fully initialized. "
            "Ensure it was created with a factory method.\n\n"
            "Example (correct usage):\n"
            "    env = Environment.from_samples(data, bin_size=2.0)\n"
            "    result = env.bin_at(points)\n\n"
            "Avoid:\n"
            "    env = Environment()  # This will not work!\n\n"
            "For more information, see: "
            f"https://edeno.github.io/neurospatial/errors/#{error_code.lower()}"
        )
        super().__init__(
            _format_error(
                message, fix="env = Environment.from_samples(positions, bin_size=2.0)"
            )
        )
        self.class_name = class_name
        self.method_name = method_name_resolved
        self.error_code = error_code
        self.is_function = is_function


class GraphValidationError(ValueError, NeurospatialError):
    """Raised when connectivity graph has invalid structure or metadata.

    This error indicates a bug in the layout engine that produced the graph,
    not a user error. All layout engines must produce graphs that pass
    validation.

    See Also
    --------
    validate_connectivity_graph : Main validation function
    """

    pass
