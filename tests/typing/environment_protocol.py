"""Static contracts: Environment types, Cartesian restrictions and coordinates.

Run with ``uv run mypy tests/typing/environment_protocol.py``.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import fastplotlib
    import matplotlib.axes
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.environment import CartesianSystem, EgocentricPolarSystem
    from neurospatial.environment._cache import AnalysisCache
    from neurospatial.geometry._coordinates import (
        CellLengthConsumer,
        CoordinateSystem,
    )

    def _check(
        cartesian: Environment[CartesianSystem],
        polar: Environment[EgocentricPolarSystem],
        field: NDArray[np.float64],
    ) -> None:
        _ = polar.interpolate  # type: ignore[misc]
        assert_type(polar.bin_at(field[:, None]), NDArray[np.intp])
        assert_type(polar.contains(field[:, None]), NDArray[np.bool_])
        assert_type(
            cartesian.smooth(field, bandwidth=1.0, mode="average"), NDArray[np.float64]
        )
        assert_type(
            polar.smooth(field, bandwidth=1.0, mode="average"), NDArray[np.float64]
        )
        assert_type(
            cartesian.diffuse(field, bandwidth=1.0, mode="density"), NDArray[np.float64]
        )
        assert_type(polar.to_dict(), dict[str, Any])
        assert_type(cartesian.neighbors(bins=0), list[int])

    def _check_signatures(
        cartesian: Environment[CartesianSystem],
        points: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> None:
        # Each bad call below must stay an error, or its now-unused ignore
        # fails this check.
        cartesian.bin_at("not an array")  # type: ignore[arg-type]
        cartesian.bin_at(points, nonexistent_keyword=True)  # type: ignore[call-arg]
        cartesian.neighbors(0)  # type: ignore[misc]
        cartesian.occupancy(times)  # type: ignore[call-arg]
        assert_type(
            cartesian.occupancy(times, points, epochs=None), NDArray[np.float64]
        )

    def _check_coordinates(
        cartesian: Environment[CartesianSystem],
        polar: Environment[EgocentricPolarSystem],
    ) -> None:
        _cartesian_lengths: CellLengthConsumer = cartesian
        _polar_lengths: CellLengthConsumer = polar
        _cartesian_system: CoordinateSystem = CartesianSystem()
        _polar_system: CoordinateSystem = EgocentricPolarSystem(circular=True)
        assert_type(cartesian.coordinate_system, CartesianSystem)
        assert_type(polar.coordinate_system, EgocentricPolarSystem)
        assert_type(cartesian._cache, AnalysisCache)
        assert_type(polar._cache, AnalysisCache)


if TYPE_CHECKING:

    def _factory_coordinate_inference(points: NDArray[np.float64]) -> None:
        cartesian = Environment.from_samples(points, bin_size=1.0)
        polar = Environment.from_polar_egocentric(
            distance_range=(0.0, 3.0),
            angle_range=(-3.0, 3.0),
            distance_bin_size=1.0,
            angle_bin_size=1.0,
        )
        assert_type(cartesian, Environment[CartesianSystem])
        assert_type(polar, Environment[EgocentricPolarSystem])
        cartesian.bin_at(points)
        polar.bin_at(points)
        Environment.bin_at(polar, points)
        polar.occupancy(points[:, 0], points)
        polar.bin_sequence(points[:, 0], points)
        field = points[:, 0]
        polar.interpolate(field, points)  # type: ignore[misc]
        Environment.interpolate(polar, field, points)  # type: ignore[arg-type]
        _ = polar.apply_transform  # type: ignore[misc]
        assert_type(
            polar.with_metadata(name="copy"), Environment[EgocentricPolarSystem]
        )

    def _check_backends(
        env: Environment[CartesianSystem], field: NDArray[np.float64]
    ) -> None:
        # Matplotlib stays the default and keeps its Axes return.
        assert_type(env.plot(), matplotlib.axes.Axes)
        assert_type(env.plot_field(field), matplotlib.axes.Axes)
        assert_type(env.plot(backend="fastplotlib"), fastplotlib.Figure)
        assert_type(env.plot_field(field, backend="fastplotlib"), fastplotlib.Figure)
        env.plot(backend="fastplotlib", scale_bar=True)  # type: ignore[call-overload]
