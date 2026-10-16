"""Decorators for Environment methods.

This module provides utility decorators used throughout the Environment class
to enforce initialization contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

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
