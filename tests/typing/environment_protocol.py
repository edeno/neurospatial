"""Static contracts: every environment type provides the interface its mixins use.

Run with ``uv run mypy tests/typing/environment_protocol.py``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.environment._coordinates import (
        CartesianSystem,
        CoordinateSystem,
        EgocentricPolarSystem,
    )
    from neurospatial.environment._protocols import (
        CartesianEnvironmentProtocol,
        CellLengthConsumer,
        EnvironmentProtocol,
        InterpolationConsumer,
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
        assert_type(base.diffuse(field, 1.0), NDArray[np.float64])
        _cartesian: CartesianEnvironmentProtocol = cartesian
        _ = polar.bin_at  # type: ignore[attr-defined]
        assert_type(cartesian.smooth(field, 1.0), NDArray[np.float64])
        assert_type(polar.smooth(field, 1.0), NDArray[np.float64])
        assert_type(cartesian.diffuse(field, 1.0), NDArray[np.float64])
        assert_type(polar.to_dict(), dict[str, Any])
        assert_type(cartesian.neighbors(0), list[int])

    def _check_signatures(
        cartesian: Environment,
        protocol: CartesianEnvironmentProtocol,
        points: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> None:
        # Fitted-state checks must not hide method signatures: each bad call
        # below must stay an error, or its now-unused ignore fails this check.
        cartesian.bin_at("not an array")  # type: ignore[arg-type]
        cartesian.bin_at(points, nonexistent_keyword=True)  # type: ignore[call-arg]
        cartesian.neighbors(0, 1, 2)  # type: ignore[call-arg]
        cartesian.occupancy(times)  # type: ignore[call-arg]
        assert_type(protocol.occupancy(times, points, epochs=None), NDArray[np.float64])

    def _check_trajectory(
        cartesian: Environment, polar: EgocentricPolarEnvironment
    ) -> None:
        _binning: TrajectoryBinningConsumer = cartesian
        _cartesian_transitions: TransitionsConsumer = cartesian
        _polar_transitions: TransitionsConsumer = polar

    def _check_transforms(cartesian: Environment) -> None:
        _rebin: RebinConsumer = cartesian
        _subset: SubsetConsumer = cartesian

    def _check_interpolation(cartesian: Environment) -> None:
        _interpolation: InterpolationConsumer = cartesian

    def _check_coordinates(
        cartesian: Environment, polar: EgocentricPolarEnvironment
    ) -> None:
        _cartesian_lengths: CellLengthConsumer = cartesian
        _polar_lengths: CellLengthConsumer = polar
        _cartesian_system: CoordinateSystem = CartesianSystem()
        _polar_system: CoordinateSystem = EgocentricPolarSystem(circular=True)
        assert_type(cartesian._coordinate_system, CartesianSystem)
        assert_type(polar._coordinate_system, EgocentricPolarSystem)
        assert_type(cartesian._gaussian_kernels, dict[float, NDArray[np.float64]])
        assert_type(polar._gaussian_kernels, dict[float, NDArray[np.float64]])
