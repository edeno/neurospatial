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
    """Exception raised when an Environment method is called before initialization.

    This exception is raised by the `@check_fitted` decorator when a user tries
    to call methods on an Environment instance that was not properly initialized
    via a factory method.

    The error message includes:
    - The specific method that was called
    - Example of correct usage with factory methods
    - Guidance on what to avoid

    Parameters
    ----------
    class_name : str
        Name of the Environment class (e.g., "Environment").
    method_name : str
        Name of the method that requires initialization (e.g., "bin_at").
    error_code : str, optional
        Error code for documentation reference. Default is "E1004".

    Examples
    --------
    >>> from neurospatial.environment.decorators import EnvironmentNotFittedError
    >>> raise EnvironmentNotFittedError("Environment", "bin_at")
    Traceback (most recent call last):
        ...
    neurospatial.environment.decorators.EnvironmentNotFittedError: [E1004] Environment.bin_at() requires...

    See Also
    --------
    check_fitted : Decorator that raises this exception

    Notes
    -----
    This exception inherits from `RuntimeError` to maintain backward compatibility
    with existing code that catches `RuntimeError`. Users can catch either:

    - `EnvironmentNotFittedError` for specific handling
    - `RuntimeError` for general error handling

    """

    def __init__(
        self, class_name: str, method_name: str, error_code: str = "E1004"
    ) -> None:
        """Initialize the EnvironmentNotFittedError.

        Parameters
        ----------
        class_name : str
            Name of the Environment class.
        method_name : str
            Name of the method requiring initialization.
        error_code : str, optional
            Error code for documentation. Default is "E1004".

        """
        message = (
            f"[{error_code}] {class_name}.{method_name}() "
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
        self.method_name = method_name
        self.error_code = error_code


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
