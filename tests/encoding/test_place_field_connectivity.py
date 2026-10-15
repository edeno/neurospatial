"""Place fields follow published graph edges, including walls and derivations."""

from datetime import datetime, timezone

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding.spatial import detect_place_fields


def _grid(*, diagonal: bool, shape: tuple[int, ...] = (2, 2), arena=None):
    return Environment.from_grid_mask(
        np.ones(shape, dtype=bool),
        grid_edges=tuple(np.arange(side + 1, dtype=float) for side in shape),
        connect_diagonal_neighbors=diagonal,
        arena=arena,
    )


def _assert_fields(env: Environment, rates: np.ndarray, expected: list[list[int]]):
    result = detect_place_fields(env, rates, min_size=1, detect_subfields=False)
    assert len(result) == len(expected)
    for field, bins in zip(result, expected, strict=True):
        assert_array_equal(field, bins)


@pytest.mark.parametrize("diagonal", [False, True])
def test_three_dimensional_field_uses_declared_diagonal_edges(diagonal):
    env = _grid(diagonal=diagonal, shape=(2, 2, 2))
    rates = np.zeros(env.n_bins)
    rates[0], rates[-1] = 10.0, 9.0
    expected = [[0, 7]] if diagonal else [[0], [7]]
    _assert_fields(env, rates, expected)


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("derivation", ["native", "subset", "rebin", "arena"])
def test_diagonal_field_connections_survive_grid_derivation(diagonal, derivation):
    """A diagonal is usable exactly when an edge survives publication."""
    if derivation == "rebin":
        env = Environment.from_layout(
            "RegularGrid",
            {
                "bin_size": 1.0,
                "dimension_ranges": [(0, 4), (0, 4)],
                "connect_diagonal_neighbors": diagonal,
            },
        ).rebin(2)
    elif derivation == "arena":
        env = _grid(
            diagonal=diagonal,
            shape=(3, 3),
            arena=OpenArena(box(0, 0, 2, 2)),
        )
    else:
        env = _grid(diagonal=diagonal)
        if derivation == "subset":
            env = env.subset(bins=np.array([True, False, False, True]))

    rates = np.zeros(env.n_bins)
    rates[0], rates[-1] = 10.0, 9.0
    assert env.connectivity.has_edge(0, env.n_bins - 1) is diagonal
    expected = [[0, env.n_bins - 1]] if diagonal else [[0], [env.n_bins - 1]]
    _assert_fields(env, rates, expected)


@pytest.mark.parametrize("storage", ["dict", "file", "nwb"])
@pytest.mark.parametrize("boundary", ["diagonal", "wall"])
def test_serialized_field_detection_uses_restored_edges(tmp_path, storage, boundary):
    """Loading a graph preserves both real diagonals and removed wall edges."""
    if boundary == "wall":
        env = _grid(
            diagonal=False,
            shape=(2, 1),
            arena=OpenArena(box(0, 0, 2, 1), walls=(LineString([(1, 0), (1, 1)]),)),
        )
        rates, expected = np.array([10.0, 9.0]), [[0], [1]]
    else:
        env = _grid(diagonal=True).subset(bins=np.array([True, False, False, True]))
        rates, expected = np.array([10.0, 9.0]), [[0, 1]]

    if storage == "dict":
        restored = Environment.from_dict(env.to_dict())
    elif storage == "file":
        env.to_file(tmp_path / "environment")
        restored = Environment.from_file(tmp_path / "environment")
    else:
        pynwb = pytest.importorskip("pynwb")
        from neurospatial.io.nwb import read_environment, write_environment

        nwbfile = pynwb.NWBFile(
            "field graph", "field-graph", datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        write_environment(nwbfile, env)
        with pynwb.NWBHDF5IO(str(tmp_path / "environment.nwb"), "w") as io:
            io.write(nwbfile)
        with pynwb.NWBHDF5IO(str(tmp_path / "environment.nwb"), "r") as io:
            restored = read_environment(io.read())

    assert set(restored.connectivity.edges) == set(env.connectivity.edges)
    _assert_fields(restored, rates, expected)


def test_declared_wall_splits_a_field_before_and_after_subsetting():
    """Adjacent high-rate cells on opposite sides of a wall are separate fields."""
    env = _grid(
        diagonal=False,
        shape=(2, 1),
        arena=OpenArena(box(0, 0, 2, 1), walls=(LineString([(1, 0), (1, 1)]),)),
    )
    assert env.connectivity.number_of_edges() == 0
    for candidate in (env, env.subset(bins=np.ones(2, dtype=bool))):
        _assert_fields(candidate, np.array([10.0, 9.0]), [[0], [1]])


def test_subfield_discrimination_uses_graph_connections():
    """Diagonal peaks stay connected when applying higher thresholds."""
    env = _grid(diagonal=True)
    rates = np.array([10.0, 3.0, 3.0, 9.0])
    result = detect_place_fields(env, rates, min_size=1, detect_subfields=True)
    assert len(result) == 1
    assert_array_equal(result[0], [0, 1, 2, 3])
