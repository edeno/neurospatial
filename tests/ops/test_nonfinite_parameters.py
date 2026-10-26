"""A NaN length or scale names no choice, so ops refuse it instead of computing.

``value <= 0`` is False for NaN, so a sign check alone let NaN through: a NaN
bandwidth made every basis function NaN and a NaN ``max_distance`` marked no
point outside. Infinite values are refused where they are not a meaningful
choice; ``inf`` stays allowed as "no limit" for cutoffs and radii.
"""

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.ops import (
    calibrate_from_scale_bar,
    distance_field,
    geodesic_rbf_basis,
    heat_kernel_wavelet_basis,
    neighbors_within,
    resample_field,
)


@pytest.fixture(scope="module")
def env():
    edges = np.arange(9, dtype=float) * 2.0
    return Environment.from_grid_mask(np.ones((8, 8), bool), (edges, edges))


def _path_graph():
    graph = nx.path_graph(4)
    nx.set_edge_attributes(graph, 1.0, "distance")
    return graph


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda env, value: geodesic_rbf_basis(
                env, bandwidth=[2.0, value], n_centers=4
            ),
            "bandwidth values must be positive and finite",
        ),
        (
            lambda env, value: heat_kernel_wavelet_basis(
                env, scales=[1.0, value], n_centers=4
            ),
            "scales values must be positive and finite",
        ),
        (
            lambda env, value: resample_field(
                np.ones(env.n_bins), env, env, method="diffuse", bandwidth=value
            ),
            "bandwidth must be positive and finite",
        ),
        (
            lambda env, value: env.nearest_bin(
                env.bin_centers, max_distance_factor=value
            ),
            "max_distance_factor must be positive and finite",
        ),
        (
            lambda env, value: calibrate_from_scale_bar(
                (0.0, 0.0), (100.0, 0.0), value, (640, 480)
            ),
            "known_length_cm must be positive and finite",
        ),
    ],
    ids=[
        "rbf-bandwidth",
        "wavelet-scales",
        "resample-bandwidth",
        "factor",
        "scale-bar",
    ],
)
def test_nonfinite_scale_is_refused(env, call, message, value):
    with pytest.raises(ValueError, match=message):
        call(env, value)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda env: env.nearest_bin(env.bin_centers, max_distance=np.nan),
            "max_distance must be non-negative",
        ),
        (
            lambda env: distance_field(_path_graph(), [0], cutoff=np.nan),
            "cutoff must be non-negative",
        ),
        (
            lambda env: neighbors_within(_path_graph(), [0], radius=np.nan),
            "radius must be non-negative",
        ),
    ],
    ids=["max-distance", "cutoff", "radius"],
)
def test_nan_limit_is_refused(env, call, message):
    with pytest.raises(ValueError, match=message):
        call(env)


def test_infinite_limit_means_no_limit(env):
    graph = _path_graph()
    np.testing.assert_array_equal(
        distance_field(graph, [0], cutoff=np.inf), [0.0, 1.0, 2.0, 3.0]
    )
    assert neighbors_within(graph, [0], radius=np.inf)[0].tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(
        env.nearest_bin(env.bin_centers + 50.0, max_distance=np.inf),
        env.nearest_bin(env.bin_centers + 50.0),
    )
