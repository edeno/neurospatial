"""One reading of ``metric="euclidean" | "geodesic" | "bin_graph"`` for behavior.

``"euclidean"`` is the straight line between two positions; ``"geodesic"``
is the shortest path that stays inside the environment, at any angle,
measured by :mod:`neurospatial.ops._paths`; ``"bin_graph"`` walks the
environment's connections between the centers of the positions' bins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neurospatial._exceptions import _format_error

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.ops import _EnvironmentPaths

METRICS = ("euclidean", "geodesic", "bin_graph")


def require_metric(metric: object, *, where: str) -> None:
    """Raise unless ``metric`` is ``"euclidean"``, ``"geodesic"`` or ``"bin_graph"``.

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
    if metric not in METRICS:
        raise ValueError(
            _format_error(
                f"{where} got metric={metric!r}; it must be 'euclidean', "
                "'geodesic' or 'bin_graph'.",
                why=(
                    "distances are measured along straight lines, along the "
                    "shortest path inside the environment, or along the "
                    "connections between bin centers."
                ),
                fix=(
                    "use metric='euclidean' for straight-line distances, "
                    "metric='geodesic' with env= to respect barriers, or "
                    "metric='bin_graph' with env= for the faster walk between "
                    "bin centers."
                ),
            )
        )


def metric_paths(
    metric: object, env: Environment | None, *, call: str
) -> _EnvironmentPaths | None:
    """The path engine ``metric`` measures with, or None for straight lines.

    Parameters
    ----------
    metric : object
        The value passed as ``metric``.
    env : Environment or None
        Environment the positions were recorded in; required for
        ``metric="geodesic"`` and ``metric="bin_graph"``.
    call : str
        Public function name, used in messages.

    Returns
    -------
    _EnvironmentPaths or None
        ``env``'s shared any-angle engine for ``"geodesic"``, its bin-graph
        engine for ``"bin_graph"``, None for ``"euclidean"``.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the three, or it needs ``env`` and
        ``env`` is None.
    CartesianOnlyError
        For ``"geodesic"`` on an egocentric polar environment.
    """
    require_metric(metric, where=f"{call}()")
    if metric == "euclidean":
        return None
    if env is None:
        raise ValueError(
            _format_error(
                f"{call}(metric={metric!r}) needs env=.",
                why="the distance is a path inside the environment.",
                fix=(
                    "pass the Environment the positions were recorded in, or use "
                    "metric='euclidean' for straight-line distances."
                ),
            )
        )
    from neurospatial.ops import _bin_graph_paths, _environment_paths

    if metric == "bin_graph":
        return _bin_graph_paths(env)
    return _environment_paths(env, call=call)
