"""Construct ordinary environments from owned component snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from neurospatial._exceptions import IncompatibleEnvironmentError
from neurospatial._logging import log_environment_combine
from neurospatial.environment._publication import _Derivation, publish_derived
from neurospatial.geometry import (
    BridgeKey,
    _CombinedArena,
    combine_binnings,
    require_cartesian,
)
from neurospatial.layout import LayoutType
from neurospatial.regions import Regions

if TYPE_CHECKING:
    from neurospatial.environment.core import Environment
    from neurospatial.geometry import CartesianSystem


def _combine(
    environment_type: type[Environment[Any]],
    components: Mapping[str, Environment],
    *,
    auto_bridge: bool,
    max_mnn_distance: float | None,
    bridge_conductances: Mapping[BridgeKey, float] | None,
    name: str,
) -> Environment[CartesianSystem]:
    """Validate components and publish their joined bins as one environment.

    The result is published as a derivation of the first component, whose
    units and frame every component shares; ``name`` replaces its name.
    ``environment_type`` is the ``Environment`` class, passed in because
    ``core`` imports this module; every component must be an instance of it.
    """
    if not isinstance(components, Mapping) or not components:
        raise ValueError(
            "components must be a nonempty ordered mapping of keys to environments."
        )
    keys, values = tuple(components), tuple(components.values())
    if any(not isinstance(key, str) or not key or "/" in key for key in keys):
        raise ValueError("Component keys must be nonempty strings without '/'.")
    if any(not isinstance(env, environment_type) for env in values):
        raise TypeError("Environment.combine() takes Environment values.")
    for key, env in zip(keys, values, strict=True):
        require_cartesian(
            env._coordinate_system,
            f"Environment.combine(components[{key!r}])",
            why="It joins the cells of each component, and the bridges between "
            "them, in one shared Cartesian frame.",
            fix="combine allocentric Cartesian environments, and analyze each "
            "polar environment on its own",
        )
    first = values[0]
    for env in values[1:]:
        if (env.n_dims, env._binning.bins.measure_dim) != (
            first.n_dims,
            first._binning.bins.measure_dim,
        ):
            raise IncompatibleEnvironmentError(
                f"[E1003] Components must share coordinate and measure dimensions: "
                f"first has n_dims={first.n_dims}, measure_dim={first._binning.bins.measure_dim}; "
                f"another has n_dims={env.n_dims}, measure_dim={env._binning.bins.measure_dim}.",
                fix="build components with matching coordinate columns and physical measure dimensions",
            )
        if (env.units, env.frame) != (first.units, first.frame):
            raise ValueError(
                "Component units and frame must match; calibrate and transform coordinates before combining."
            )
    arenas = tuple(env.arena for env in values)
    binning, n_bridges = combine_binnings(
        keys,
        tuple(env._binning for env in values),
        arenas,
        auto_bridge=auto_bridge,
        max_mnn_distance=max_mnn_distance,
        bridge_conductances=bridge_conductances,
        kind=LayoutType.COMBINED.value,
    )
    regions = Regions(
        [
            replace(region, name=f"{key}/{region.name}")
            for key, env in components.items()
            for region in env.regions.values()
        ]
    )

    log_environment_combine(len(values), binning.bins.n_bins, n_bridges)

    # The components have no single parent: the history starts here and no
    # one layout's build choices describe the joined bins.
    derivation = _Derivation(
        binning,
        "combine",
        build_params={},
        continues_history=False,
        regions=regions,
        arena=_CombinedArena(arenas),
    )
    # Units and frame come from the first component; all components share them.
    return publish_derived(first, derivation, name=name)
