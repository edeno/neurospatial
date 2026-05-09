"""Decorators for Environment methods.

This module provides utility decorators used throughout the Environment class
and its mixins to enforce constraints and add functionality.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

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
