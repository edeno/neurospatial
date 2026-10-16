"""Decorators for Environment methods.

This module provides utility decorators used throughout the Environment class
to enforce initialization and cache contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar, overload

from neurospatial._exceptions import (
    EnvironmentNotFittedError as EnvironmentNotFittedError,
)

T = TypeVar("T")
P = ParamSpec("P")


def check_fitted(method: Callable[P, T]) -> Callable[P, T]:
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
    after the Environment has been properly initialized through owned geometry publication.

    """

    # ParamSpec keeps the method's signature for type checkers, so a call with
    # a wrong argument or keyword is still reported through the decorator.
    @wraps(method)
    def _inner(*args: P.args, **kwargs: P.kwargs) -> T:
        self = args[0]
        if not getattr(self, "_is_fitted", False):
            raise EnvironmentNotFittedError(self.__class__.__name__, method.__name__)
        return method(*args, **kwargs)

    return _inner


# ----------------------------------------------------------------------
# Versioned cached property
# ----------------------------------------------------------------------


class versioned_cached_property(Generic[T]):  # noqa: N801 — mimics functools.cached_property
    """Cached property that auto-invalidates on a host ``_state_version`` bump.

    Behaves like :func:`functools.cached_property`, but the cached
    value is keyed by the host instance's ``_state_version`` integer.
    When a property is accessed, the wrapper compares the version
    captured at cache time against the instance's current
    ``_state_version``; on mismatch the underlying function is re-run
    and the new value is cached.

    Environment uses ordinary private cached properties over immutable geometry.
    This descriptor remains available for hosts that advance their version
    explicitly. The host class must expose an
    integer attribute named ``_state_version`` and must allow
    per-instance attribute storage (no ``__slots__`` without
    ``_state_version`` plus the versioned cache keys).

    Limitations
    -----------
    Invalidation is **only** triggered by ``_state_version`` bumps. The
    descriptor does **not** fingerprint the underlying state. The host must own
    and protect its state so edits cannot bypass invalidation. Assigning or
    deleting a cached property raises rather than shadowing its descriptor and
    bypassing future version checks.

    Cache layout
    ------------
    The cached value lives at
    ``instance.__dict__[f"_versioned_cache__{attr_name}"]`` and the
    captured version at
    ``instance.__dict__[f"_versioned_cache__{attr_name}__version"]``.
    These private keys keep computed values separate from the read-only
    descriptor's attribute name. The descriptor checks the version on every
    access.

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
        # Keep cache storage separate from the read-only descriptor's name.
        value_key = f"_versioned_cache__{self.attrname}"
        version_key = f"{value_key}__version"
        current_version = getattr(instance, "_state_version", 0)
        if value_key in cache and cache.get(version_key) == current_version:
            return cache[value_key]  # type: ignore[no-any-return]
        value = self.func(instance)
        cache[value_key] = value
        cache[version_key] = current_version
        return value

    def __set__(self, instance: object, value: T) -> None:
        raise AttributeError(f"{type(instance).__name__}.{self.attrname} is read-only.")

    def __delete__(self, instance: object) -> None:
        raise AttributeError(f"{type(instance).__name__}.{self.attrname} is read-only.")
