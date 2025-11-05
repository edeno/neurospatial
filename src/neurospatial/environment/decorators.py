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
    RuntimeError
        If the method is called on an Environment instance that has not been
        fully initialized (i.e., `_is_fitted` is False).

    Examples
    --------
    >>> from neurospatial.environment.decorators import check_fitted
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
    RuntimeError: MyEnvironment.query_method() requires the environment to be fully initialized...

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
            raise RuntimeError(
                f"{self.__class__.__name__}.{method.__name__}() "
                "requires the environment to be fully initialized. "
                "Ensure it was created with a factory method.\n\n"
                "Example (correct usage):\n"
                "    env = Environment.from_samples(data, bin_size=2.0)\n"
                "    result = env.bin_at(points)\n\n"
                "Avoid:\n"
                "    env = Environment()  # This will not work!",
            )
        return method(self, *args, **kwargs)

    return _inner
