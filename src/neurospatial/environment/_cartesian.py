"""Cartesian-only capability: the decorator that marks and guards each member."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

from neurospatial.geometry._coordinates import require_cartesian

_F = TypeVar("_F", bound=Callable[..., Any])

# Every guarded function the decorator returned.
_GUARDED: set[Callable[..., Any]] = set()


def cartesian_only(
    *, why: str, fix: str, example: str | None = None
) -> Callable[[_F], _F]:
    """Mark an Environment member as Cartesian-only and refuse polar calls.

    The guard runs when the member is called, before it reads any
    coordinates, for bound calls and for unbound calls such as
    ``Environment.bin_at(polar_env, points)`` alike. The member stays visible
    on polar environments: ``hasattr`` is True and ``dir`` lists it. The
    restricted ``self: Environment[CartesianSystem]`` annotation lets type
    checkers flag the call before it runs. For a property, decorate the getter
    below ``@property``.

    Parameters
    ----------
    why : str
        One sentence on why this member needs Cartesian coordinates.
    fix : str
        What works on a polar environment instead.
    example : str or None, optional
        Code for that alternative, run by the tests (see
        :class:`~neurospatial.CartesianOnlyError`). None when a polar
        environment has no equivalent.

    Returns
    -------
    callable
        A decorator returning the guarded member, with the original name,
        docstring and signature.
    """

    def decorate(function: _F) -> _F:
        name = function.__name__

        @functools.wraps(function)
        def guarded(self: Any, /, *args: Any, **kwargs: Any) -> Any:
            require_cartesian(
                self._geometry.coordinates,
                f"{type(self).__name__}.{name}()",
                why=why,
                fix=fix,
                example=example,
            )
            return function(self, *args, **kwargs)

        _GUARDED.add(guarded)
        return cast("_F", guarded)

    return decorate


def cartesian_only_members(cls: type) -> frozenset[str]:
    """Name each ``@cartesian_only`` member that ``cls`` or its bases define.

    Parameters
    ----------
    cls : type
        The class whose members, including inherited ones, are read.

    Returns
    -------
    frozenset of str
        Names of the guarded methods and properties.
    """
    members: set[str] = set()
    for klass in cls.__mro__:
        for name, member in vars(klass).items():
            function = member.fget if isinstance(member, property) else member
            if callable(function) and function in _GUARDED:
                members.add(name)
    return frozenset(members)
