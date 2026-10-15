"""Only a full angular circle can join the first and last polar sectors."""

import numpy as np
import pytest

from neurospatial import Environment


def _polar(angle_range, **kwargs):
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 10.0),
        angle_range=angle_range,
        distance_bin_size=10.0,
        angle_bin_size=(angle_range[1] - angle_range[0]) / 4,
        connect_diagonal_neighbors=False,
        **kwargs,
    )


@pytest.mark.parametrize("option", ["omitted", None])
@pytest.mark.parametrize(
    "angle_range,circular",
    [
        ((-np.pi, np.pi), True),
        ((0.0, 2 * np.pi), True),
        ((1.0, 1.0 + 2 * np.pi), True),
        ((-np.pi / 2, np.pi / 2), False),
        ((0.0, np.pi / 2), False),
    ],
)
def test_default_periodicity_follows_the_angular_extent(option, angle_range, circular):
    kwargs = {} if option == "omitted" else {"circular_angle": option}
    env = _polar(angle_range, **kwargs)
    assert env._angular_is_circular is circular
    assert env.connectivity.has_edge(0, 3) is circular


@pytest.mark.parametrize("option", [False, True])
def test_full_circle_allows_explicit_open_or_circular_topology(option):
    env = _polar((-np.pi, np.pi), circular_angle=option)
    assert env._angular_is_circular is option
    assert env.connectivity.has_edge(0, 3) is option


@pytest.mark.parametrize(
    "angle_range", [(-np.pi / 2, np.pi / 2), (0.0, np.pi), (0.0, 2 * np.pi + 1e-6)]
)
def test_explicit_circular_topology_requires_a_full_circle(angle_range):
    with pytest.raises(ValueError, match=r"circular_angle=True.*full circle"):
        _polar(angle_range, circular_angle=True)


def test_partial_range_has_no_shortcut_across_its_unobserved_angles():
    env = _polar((-np.pi / 2, np.pi / 2))
    np.testing.assert_allclose(
        env.distance_to([0], metric="geodesic")[3], 15 * np.pi / 4
    )


@pytest.mark.parametrize("full_circle", [False, True])
def test_inferred_periodicity_survives_copy_and_storage(tmp_path, full_circle):
    span = 2 * np.pi if full_circle else np.pi
    env = _polar((0.0, span))
    env.to_file(tmp_path / "polar")
    for restored in (
        env.copy(),
        Environment.from_dict(env.to_dict()),
        Environment.from_file(tmp_path / "polar"),
    ):
        assert restored._angular_is_circular is full_circle
        assert restored.connectivity.has_edge(0, 3) is full_circle
