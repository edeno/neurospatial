"""Static contracts: every environment type provides the interface its mixins use.

Run with ``uv run mypy tests/typing/environment_protocol.py``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.environment._protocols import (
        EnvironmentProtocol,
        RebinConsumer,
        SubsetConsumer,
        TrajectoryBinningConsumer,
        TransitionsConsumer,
    )
    from neurospatial.environment.core import _BaseEnvironment
    from neurospatial.environment.polar import EgocentricPolarEnvironment

    def _environment(env: EnvironmentProtocol) -> None: ...

    def _check(
        cartesian: Environment,
        polar: EgocentricPolarEnvironment,
        base: _BaseEnvironment,
        field: NDArray[np.float64],
    ) -> None:
        _environment(cartesian)
        _environment(polar)
        _environment(base)
        assert_type(cartesian.smooth(field, 1.0), NDArray[np.float64])
        assert_type(polar.smooth(field, 1.0), NDArray[np.float64])
        assert_type(cartesian.diffuse(field, 1.0), NDArray[np.float64])
        assert_type(polar.to_dict(), dict[str, Any])
        assert_type(cartesian.neighbors(0), list[int])

    def _check_trajectory(
        cartesian: Environment, polar: EgocentricPolarEnvironment
    ) -> None:
        _binning: TrajectoryBinningConsumer = cartesian
        _cartesian_transitions: TransitionsConsumer = cartesian
        _polar_transitions: TransitionsConsumer = polar

    def _check_transforms(cartesian: Environment) -> None:
        _rebin: RebinConsumer = cartesian
        _subset: SubsetConsumer = cartesian
