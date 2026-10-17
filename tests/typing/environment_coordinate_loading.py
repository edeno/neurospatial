"""Shared annotations and storage do not claim Cartesian coordinates."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from shapely.geometry import Polygon
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.geometry import Binning, CartesianSystem
    from neurospatial.io import from_dict, from_file
    from neurospatial.io.nwb import read_environment

    def _shared(env: Environment, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return env.smooth(values, bandwidth=1.0)

    def _check_loading(data: dict[str, Any], path: str) -> None:
        assert_type(Environment.from_dict(data), Environment[Any])
        assert_type(Environment.from_file(path), Environment[Any])
        assert_type(from_dict(data), Environment[Any])
        assert_type(from_file(path), Environment[Any])

    def _check_nwb_loading(nwbfile: Any) -> None:
        # The stored record decides the coordinate system.
        assert_type(Environment.from_nwb(nwbfile), Environment[Any])
        assert_type(read_environment(nwbfile), Environment[Any])

    def _check_explicit_factories(binning: Binning, outline: Polygon) -> None:
        # from_bins and from_polygon build Cartesian environments.
        assert_type(Environment.from_bins(binning), Environment[CartesianSystem])
        assert_type(
            Environment.from_polygon(outline, bin_size=2.0),
            Environment[CartesianSystem],
        )

    def _check_shared(values: NDArray[np.float64]) -> None:
        polar = Environment.from_polar_egocentric(
            distance_range=(0.0, 3.0),
            angle_range=(-np.pi, np.pi),
            distance_bin_size=1.0,
            angle_bin_size=np.pi / 2,
        )
        _shared(polar, values)
