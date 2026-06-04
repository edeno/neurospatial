"""Decorators for Environment methods.

This module provides utility decorators used throughout the Environment class
and its mixins to enforce constraints and add functionality.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment

T = TypeVar("T")


class EnvironmentNotFittedError(RuntimeError):
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
    neurospatial.environment.decorators.EnvironmentNotFittedError: [E1004] Environment.bin_at() requires...

    >>> raise EnvironmentNotFittedError("path_progress", is_function=True)
    Traceback (most recent call last):
        ...
    neurospatial.environment.decorators.EnvironmentNotFittedError: [E1004] path_progress() requires...

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
            f"https://neurospatial.readthedocs.io/errors/#{error_code.lower()}"
        )
        super().__init__(message)
        self.class_name = class_name
        self.method_name = method_name_resolved
        self.error_code = error_code
        self.is_function = is_function


def check_fitted(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that an Environment method is called only after fitting.

    Parameters
    ----------
    method : callable
        Method to decorate.

    Returns
    -------
    callable
        Wrapped method that checks fitted status before execution.

    Raises
    ------
    EnvironmentNotFittedError
        If the method is called on an Environment instance that has not been
        fully initialized (i.e., `_is_fitted` is False).

    Examples
    --------
    >>> from neurospatial.environment.decorators import (
    ...     check_fitted,
    ...     EnvironmentNotFittedError,
    ... )
    >>> class MyEnvironment:
    ...     def __init__(self):
    ...         self._is_fitted = False
    ...
    ...     @check_fitted
    ...     def query_method(self):
    ...         return "success"
    >>> env = MyEnvironment()
    >>> env.query_method()  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    neurospatial.environment.decorators.EnvironmentNotFittedError: [E1004] MyEnvironment.query_method() requires...

    Notes
    -----
    This decorator is typically used on methods that require the Environment
    to be fully initialized via a factory method (e.g., `Environment.from_samples()`).
    It prevents runtime errors by catching attempts to use uninitialized instances.

    The decorator checks for the `_is_fitted` attribute, which is set to `True`
    after the Environment has been properly initialized through `_setup_from_layout()`.

    """

    @wraps(method)
    def _inner(self: Environment, *args, **kwargs) -> T:
        if not getattr(self, "_is_fitted", False):
            raise EnvironmentNotFittedError(self.__class__.__name__, method.__name__)
        return method(self, *args, **kwargs)

    return _inner


# ----------------------------------------------------------------------
# Versioned cached property
# ----------------------------------------------------------------------


class versioned_cached_property(Generic[T]):  # noqa: N801 — mimics functools.cached_property
    """Cached property that auto-invalidates on Environment ``_state_version`` bump.

    Behaves like :func:`functools.cached_property`, but the cached
    value is keyed by the host instance's ``_state_version`` integer.
    When a property is accessed, the wrapper compares the version
    captured at cache time against the instance's current
    ``_state_version``; on mismatch the underlying function is re-run
    and the new value is cached.

    This decorator is the mechanism for invalidating derived quantities
    (bin attributes, distance fields, …) when the Environment is mutated
    through the documented paths (``_setup_from_layout``, ``subset``,
    ``apply_transform``, ``rebin``). The host class must expose an
    integer attribute named ``_state_version`` and must allow
    per-instance attribute storage (no ``__slots__`` without
    ``_state_version`` plus the versioned cache keys).

    Limitations
    -----------
    Invalidation is **only** triggered by ``_state_version`` bumps. The
    descriptor does **not** fingerprint the underlying state, so direct
    in-place mutation of e.g. ``env.bin_centers`` or ``env.connectivity``
    (writing into the array, calling ``add_node`` on the graph) is not
    detected — versioned-cached values will keep returning the
    pre-mutation result. Such mutation is unsupported in v0.4; if a
    caller absolutely must reach in, they must call
    ``env.clear_cache()`` to drop versioned caches before re-reading
    derived attributes. A first-access fingerprint check was considered
    and dropped: it would catch the first stale read but still miss
    later mutations, so it would not actually deliver the invariant the
    user wants.

    Cache layout
    ------------
    The cached value lives at
    ``instance.__dict__[f"_versioned_cache__{attr_name}"]`` and the
    captured version at
    ``instance.__dict__[f"_versioned_cache__{attr_name}__version"]``.
    These keys are intentionally not the bare attribute name so the
    descriptor is invoked on every access (Python only short-circuits
    a non-data descriptor when an instance attribute of the same name
    exists).

    Examples
    --------
    >>> class Env:
    ...     _state_version = 0
    ...
    ...     @versioned_cached_property
    ...     def expensive(self) -> int:
    ...         print("computing")
    ...         return 42
    >>> e = Env()
    >>> e.expensive
    computing
    42
    >>> e.expensive  # cached
    42
    >>> e._state_version += 1
    >>> e.expensive  # recomputes after version bump
    computing
    42
    """

    def __init__(self, func: Callable[[Any], T]) -> None:
        self.func = func
        self.attrname: str | None = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner: type, name: str) -> None:
        if self.attrname is None:
            self.attrname = name
        elif self.attrname != name:
            raise TypeError(
                "Cannot assign the same versioned_cached_property to two "
                f"different names ({self.attrname!r} and {name!r})."
            )

    @overload
    def __get__(
        self, instance: None, owner: type | None = None
    ) -> versioned_cached_property[T]: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> T: ...

    def __get__(
        self, instance: object | None, owner: type | None = None
    ) -> T | versioned_cached_property[T]:
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use versioned_cached_property without calling __set_name__ on it."
            )
        cache = instance.__dict__
        # Use distinct keys so the descriptor is always invoked.
        # Setting `cache[self.attrname]` directly would shadow the
        # descriptor on subsequent attribute access (because non-data
        # descriptors lose to instance __dict__ in MRO lookup).
        value_key = f"_versioned_cache__{self.attrname}"
        version_key = f"{value_key}__version"
        current_version = getattr(instance, "_state_version", 0)
        if value_key in cache and cache.get(version_key) == current_version:
            return cache[value_key]  # type: ignore[no-any-return]
        value = self.func(instance)
        cache[value_key] = value
        cache[version_key] = current_version
        return value
