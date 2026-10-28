"""One reading of ``metric="euclidean" | "geodesic"`` for behavior functions.

``"euclidean"`` is the straight line between two positions; ``"geodesic"``
is the shortest path that stays inside the environment, measured by
:mod:`neurospatial.behavior._environment_paths`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.behavior._environment_paths import EnvironmentPaths
    from neurospatial.environment import Environment


def require_metric(metric: object, *, where: str) -> None:
    """Raise unless ``metric`` is ``"euclidean"`` or ``"geodesic"``.

    Parameters
    ----------
    metric : object
        The value passed as ``metric``.
    where : str
        The call or result class named in the message, such as
        ``"traveled_path_length()"`` or ``"PathEfficiencyResult"``.

    Raises
    ------
    ValueError
        If ``metric`` is any other value, including another spelling.
    """
    if metric not in ("euclidean", "geodesic"):
        raise ValueError(
            _format_error(
                f"{where} got metric={metric!r}; it must be 'euclidean' or 'geodesic'.",
                why=(
                    "distances are measured either along straight lines or along "
                    "the shortest path inside the environment."
                ),
                fix=(
                    "use metric='euclidean' for straight-line distances, or "
                    "metric='geodesic' with env= to respect barriers."
                ),
            )
        )


def metric_paths(
    metric: object, env: Environment | None, *, call: str
) -> EnvironmentPaths | None:
    """The shortest-path engine ``metric`` measures with, or None for straight lines.

    Parameters
    ----------
    metric : object
        The value passed as ``metric``.
    env : Environment or None
        Environment the positions were recorded in; required for
        ``metric="geodesic"``.
    call : str
        Public function name, used in messages.

    Returns
    -------
    EnvironmentPaths or None
        ``env``'s shared engine for ``"geodesic"``; None for ``"euclidean"``.

    Raises
    ------
    ValueError
        If ``metric`` is not ``"euclidean"`` or ``"geodesic"``, or it is
        ``"geodesic"`` and ``env`` is None.
    CartesianOnlyError
        For ``"geodesic"`` on an egocentric polar environment.
    """
    require_metric(metric, where=f"{call}()")
    if metric == "euclidean":
        return None
    if env is None:
        raise ValueError(
            _format_error(
                f"{call}(metric='geodesic') needs env=.",
                why="the geodesic distance is a path inside the environment.",
                fix=(
                    "pass the Environment the positions were recorded in, or use "
                    "metric='euclidean' for straight-line distances."
                ),
            )
        )
    from neurospatial.behavior._environment_paths import environment_paths

    return environment_paths(env, call=call)
