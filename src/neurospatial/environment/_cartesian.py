"""Cartesian-only capability: the decorator that marks and guards each member."""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, TypeVar, cast, overload

from neurospatial._exceptions import CartesianOnlyError
from neurospatial.geometry._coordinates import (
    _POLAR_COORDINATES,
    CartesianSystem,
    EgocentricPolarSystem,
)

COORDINATE_REASON = "Cartesian-coordinate operation"
GRID_REASON = "Cartesian-grid operation"

_POLAR_EXPLANATION = (
    f"{_POLAR_COORDINATES} Use connectivity-graph operations instead (neighbors, "
    "path_between, reachable_from, smooth, or distance_to(metric='geodesic')), "
    "or build an allocentric Cartesian env via Environment.from_samples / "
    "from_polygon / from_grid_mask."
)

_F = TypeVar("_F", bound=Callable[..., Any])

# Each guarded function the decorator returned, mapped to its refusal reason.
_REASONS: dict[Callable[..., Any], str] = {}


def cartesian_only_error(
    environment_type: str, operation: str, reason: str
) -> CartesianOnlyError:
    """Build the refusal for a Cartesian operation on polar radius/angle bins.

    Parameters
    ----------
    environment_type : str
        Runtime class name of the refused environment.
    operation : str
        The refused call, e.g. ``"bin_at()"``.
    reason : str
        ``COORDINATE_REASON`` or ``GRID_REASON``.

    Returns
    -------
    CartesianOnlyError
        The error to raise.
    """
    return CartesianOnlyError(
        environment_type, operation, reason=reason, explanation=_POLAR_EXPLANATION
    )


@overload
def cartesian_only(function: _F, /) -> _F: ...


@overload
def cartesian_only(*, grid: bool) -> Callable[[_F], _F]: ...


def cartesian_only(
    function: _F | None = None, /, *, grid: bool = False
) -> _F | Callable[[_F], _F]:
    """Mark an Environment member as Cartesian-only and guard its calls.

    The guard refuses polar environments before the member reads any
    coordinates, including unbound calls such as
    ``Environment.bin_at(polar_env, points)``. Instance attribute access is
    refused separately by ``Environment.__getattribute__``, which reads the
    members collected by :func:`cartesian_only_members`. For a property,
    decorate the getter below ``@property``.

    Parameters
    ----------
    function : callable, optional
        The member to guard. Omit it to pass ``grid``.
    grid : bool, default=False
        Whether the member needs a Cartesian grid rather than only Cartesian
        coordinates; this selects the reason the refusal states.

    Returns
    -------
    callable
        The guarded member, with the original name, docstring and signature.
    """
    reason = GRID_REASON if grid else COORDINATE_REASON

    def decorate(function: _F) -> _F:
        operation = f"{function.__name__}()"

        @functools.wraps(function)
        def guarded(self: Any, /, *args: Any, **kwargs: Any) -> Any:
            if isinstance(self._geometry.coordinates, EgocentricPolarSystem):
                raise cartesian_only_error(type(self).__name__, operation, reason)
            return function(self, *args, **kwargs)

        _REASONS[guarded] = reason
        return cast("_F", guarded)

    return decorate if function is None else decorate(function)


def cartesian_only_members(cls: type) -> Mapping[str, str]:
    """Map each ``@cartesian_only`` member defined on ``cls`` to its refusal reason.

    Parameters
    ----------
    cls : type
        The class whose own members are read; inherited members are not.

    Returns
    -------
    Mapping[str, str]
        Read-only mapping from member name to ``COORDINATE_REASON`` or
        ``GRID_REASON``.
    """
    members: dict[str, str] = {}
    for name, member in vars(cls).items():
        function = member.fget if isinstance(member, property) else member
        if callable(function) and function in _REASONS:
            members[name] = _REASONS[function]
    return MappingProxyType(members)


def require_cartesian(
    coordinates: CartesianSystem | EgocentricPolarSystem, operation: str
) -> None:
    """Refuse one Cartesian mode of an operation that also works on polar bins.

    Members that are Cartesian-only as a whole use :func:`cartesian_only`.

    Parameters
    ----------
    coordinates : CartesianSystem or EgocentricPolarSystem
        The environment's coordinate descriptor.
    operation : str
        The refused call and mode, e.g. ``"distance_to(metric='euclidean')"``.

    Raises
    ------
    CartesianOnlyError
        If ``coordinates`` is polar.
    """
    if isinstance(coordinates, EgocentricPolarSystem):
        raise cartesian_only_error("Environment", operation, COORDINATE_REASON)
