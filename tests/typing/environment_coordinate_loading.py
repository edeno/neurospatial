"""Shared annotations and storage do not claim Cartesian coordinates."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from typing_extensions import assert_type

    from neurospatial import Environment
    from neurospatial.io import from_dict, from_file

    def _shared(env: Environment, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return env.smooth(values, bandwidth=1.0)

    def _check_loading(data: dict[str, Any], path: str) -> None:
        assert_type(Environment.from_dict(data), Environment[Any])
        assert_type(Environment.from_file(path), Environment[Any])
        assert_type(from_dict(data), Environment[Any])
        assert_type(from_file(path), Environment[Any])

    def _check_shared(values: NDArray[np.float64]) -> None:
        polar = Environment.from_polar_egocentric(
            (0.0, 3.0), (-np.pi, np.pi), 1.0, np.pi / 2
        )
        _shared(polar, values)
