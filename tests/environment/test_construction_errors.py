"""Common data-shape mistakes at construction say what failed, why and how to fix it."""

from __future__ import annotations

import re
import warnings

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment
from neurospatial.layout import create_layout

_RNG = np.random.default_rng(0)
_POSITIONS = _RNG.uniform(0.0, 100.0, size=(2000, 2))
_ENV = Environment.from_samples(_POSITIONS, bin_size=4.0)
_HEX = Environment.from_samples(_POSITIONS, bin_size=4.0, layout="Hexagonal")
_TRACK = Environment.linear_track(endpoints=[(0.0, 0.0), (100.0, 0.0)], bin_size=2.0)
_MESH = Environment.from_layout(
    create_layout(
        "TriangularMesh", boundary_polygon=box(0, 0, 30, 30), point_spacing=2.0
    )
)


def _graph_without_positions() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("home", "goal", distance=50.0)
    return graph


# (call, exception type, text the fix must contain)
_MISTAKES = {
    "transposed positions": (
        lambda: Environment.from_samples(_POSITIONS.T, bin_size=4.0),
        ValueError,
        "positions.T",
    ),
    "non-numeric positions": (
        lambda: Environment.from_samples(
            np.array([["a", "b"], ["c", "d"]]), bin_size=4.0
        ),
        TypeError,
        "to_numpy(dtype=float)",
    ),
    "polygon as arena": (
        lambda: Environment.from_samples(
            _POSITIONS, bin_size=4.0, arena=box(0, 0, 100, 100)
        ),
        TypeError,
        "arena=OpenArena(polygon)",
    ),
    "from_bins(env.binning)": (
        lambda: Environment.from_bins(_ENV.binning),
        TypeError,
        "faces=KnownFaces(binning.faces.build())",
    ),
    "rebin(1.5)": (lambda: _ENV.rebin(1.5), ValueError, "rebin(2)"),
    "linear_track(endpoints=[0, 200])": (
        lambda: Environment.linear_track(endpoints=[0, 200], bin_size=2.0),
        ValueError,
        "endpoints=[(0, 0.0), (200, 0.0)]",
    ),
    "graph without pos": (
        lambda: Environment.from_graph(
            _graph_without_positions(), [("home", "goal")], bin_size=2.0
        ),
        ValueError,
        "pos=(x, y)",
    ),
    "maze graph without pos": (
        lambda: Environment.maze(
            kind="t", track_graph=_graph_without_positions(), bin_size=2.0
        ),
        ValueError,
        "pos=(x, y)",
    ),
    "bin_size of the wrong length": (
        lambda: Environment.from_samples(_POSITIONS, bin_size=[4.0, 4.0, 4.0]),
        ValueError,
        "2 numbers, one per column",
    ),
    "hexagon bin_size per axis": (
        lambda: Environment.from_samples(
            _POSITIONS, bin_size=[4.0, 4.0], layout="Hexagonal"
        ),
        ValueError,
        "bin_size=4",
    ),
    "flat dimension_ranges": (
        lambda: Environment.from_samples(
            _POSITIONS, bin_size=4.0, dimension_ranges=(0, 100)
        ),
        ValueError,
        "dimension_ranges=[(",
    ),
    "reversed dimension_ranges": (
        lambda: Environment.from_samples(
            _POSITIONS, bin_size=4.0, dimension_ranges=[(100, 0), (0, 100)]
        ),
        ValueError,
        "dimension_ranges=[(",
    ),
    "hexagonal rebin(2)": (
        lambda: _HEX.rebin(2),
        NotImplementedError,
        "layout='Hexagonal'",
    ),
    "hexagonal rebin(bin_size=8)": (
        lambda: _HEX.rebin(bin_size=8.0),
        NotImplementedError,
        "layout='Hexagonal'",
    ),
    "track rebin(2)": (
        lambda: _TRACK.rebin(2),
        NotImplementedError,
        "Environment.from_graph(env.binning.linearization.graph",
    ),
    "mesh rebin(2)": (
        lambda: _MESH.rebin(2),
        NotImplementedError,
        "create_layout('TriangularMesh'",
    ),
    "apply_transform(np.eye(3))": (
        lambda: _ENV.apply_transform(np.eye(3)),
        TypeError,
        "AffineND(matrix)",
    ),
}

# Implementation details a scientist cannot act on.
_INTERNALS = re.compile(
    r"NativeFaces|snapshot adapter|histogramdd|meshgrid|set_int_max_str_digits"
    r"|is not iterable|has no attribute|\b_[a-z]\w*"
)


@pytest.mark.parametrize("mistake", sorted(_MISTAKES))
def test_mistake_says_what_why_and_fix(mistake):
    call, error, fix_text = _MISTAKES[mistake]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the transposed-positions warning
        with pytest.raises(error) as caught:
            call()
    lines = str(caught.value).splitlines()
    assert any(line.startswith("Why: ") for line in lines), lines
    assert lines[-1].startswith("Fix: "), lines
    assert fix_text in lines[-1]
    assert not _INTERNALS.search(str(caught.value)), str(caught.value)


@pytest.mark.parametrize(
    "build",
    [Environment.from_samples, Environment.open_field],
    ids=["from_samples", "open_field"],
)
@pytest.mark.parametrize("layout", ["RegularGrid", "Hexagonal"])
def test_infinite_positions_are_missing_samples_like_nan(build, layout):
    """An infinite coordinate is not a place in the arena: it is left out."""
    with_inf = _POSITIONS.copy()
    with_inf[5, 0], with_inf[6, 1] = np.inf, -np.inf
    with_nan = _POSITIONS.copy()
    with_nan[5, 0] = with_nan[6, 1] = np.nan
    from_inf = build(with_inf, bin_size=4.0, layout=layout)
    from_nan = build(with_nan, bin_size=4.0, layout=layout)
    np.testing.assert_array_equal(from_inf.bin_centers, from_nan.bin_centers)
    np.testing.assert_array_equal(from_inf.edge_pairs, from_nan.edge_pairs)
    assert np.isinf(with_inf[5, 0])  # the caller's array is not changed
