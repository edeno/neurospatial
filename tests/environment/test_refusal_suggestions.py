"""The code a refusal suggests runs on the input that was refused.

Each case triggers a refusal, checks that its ``Fix:`` line shows the code,
then runs that code with the names the caller had (``env``, ``positions``,
the refused argument, ...). A message whose code no longer runs fails here,
whatever its wording.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from neurospatial import Environment, Regions
from neurospatial.binning import KnownFaces
from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import translate_3d

_RNG = np.random.default_rng(0)
_POSITIONS = _RNG.uniform(0.0, 100.0, size=(3000, 2))
_TIMES = np.arange(len(_POSITIONS)) / 30.0


def _run(code: str, names: dict[str, Any]) -> Any:
    """Run ``"stmt; stmt; expr"`` and return the value of the last expression."""
    *statements, last = code.split("; ")
    namespace = {"np": np, "Environment": Environment, **names}
    exec("\n".join(statements), namespace)
    return eval(last, namespace)


def _keyword(code: str, names: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a suggested ``name=value`` argument."""
    name, value = code.split("=", 1)
    return {name: eval(value, {"np": np, **names})}


@dataclass(frozen=True)
class Case:
    """A refused call, the code its fix shows, and how to run that code.

    ``kind`` says what ``code`` is:

    - ``"call"``: statements ending in an expression, run as they are;
    - ``"argument"``: one ``name=value`` argument, passed to ``retry``;
    - ``"value"``: an expression or one assignment, whose value goes to
      ``retry`` in place of the refused input.

    ``prelude`` is an import the fix names in words, run before ``code``.
    """

    names: Callable[[], dict[str, Any]]
    refuse: Callable[[dict[str, Any]], Any]
    code: str
    kind: str = "call"
    retry: Callable[[dict[str, Any], Any], Any] | None = None
    prelude: str = ""


def _follow(case: Case, names: dict[str, Any]) -> Any:
    """Run the suggested code the way its kind says."""
    namespace = {"np": np, "Environment": Environment, **names}
    if case.prelude:
        exec(case.prelude, namespace)
    if case.kind == "call":
        return _run(case.code, namespace)
    assert case.retry is not None
    if case.kind == "argument":
        return case.retry(names, _keyword(case.code, namespace))
    if " = " in case.code:
        target = case.code.split(" = ", 1)[0]
        exec(case.code, namespace)
        return case.retry(names, namespace[target])
    return case.retry(names, eval(case.code, namespace))


def _grid() -> Environment:
    """A 40 x 40 cm grid of 4 cm cells, every cell active."""
    return Environment.from_grid_mask(
        np.ones((10, 10), dtype=bool), grid_edges=(np.arange(0.0, 41.0, 4.0),) * 2
    )


def _sampled() -> dict[str, Any]:
    return {
        "env": Environment.from_samples(_POSITIONS, bin_size=5.0),
        "positions": _POSITIONS,
    }


def _hexagons() -> dict[str, Any]:
    env = Environment.from_samples(_POSITIONS, bin_size=4.0, layout="Hexagonal")
    return {"env": env, "positions": _POSITIONS, "field": np.ones(env.n_bins)}


def _track() -> dict[str, Any]:
    env = Environment.linear_track(
        node_positions=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)],
        bin_size=2.0,
        edge_spacing=5.0,
        track_width=10.0,
    )
    return {"env": env, "field": np.ones(env.n_bins)}


def _widthless_track() -> dict[str, Any]:
    env = Environment.linear_track(
        node_positions=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)],
        bin_size=2.0,
        edge_spacing=5.0,
        units="cm",
    )
    positions = np.array([[10.0, 1.0], [100.0, 30.0], [50.0, 400.0]])
    return {"env": env, "positions": positions, "width": 10.0}


_TRACK_WIDTH_CODE = (
    "Environment.from_graph(env.binning.linearization.graph, "
    "list(env.binning.linearization.edge_order), "
    "bin_size=env.layout_parameters['bin_size'], "
    "edge_spacing=env.binning.linearization.edge_spacing, track_width=width, "
    "name=env.name, units=env.units, frame=env.frame, regions=env.regions)"
)


def _mesh() -> dict[str, Any]:
    env = Environment.from_layout(
        create_layout(
            "TriangularMesh", boundary_polygon=box(0, 0, 30, 30), point_spacing=2.0
        )
    )
    return {"env": env, "field": np.ones(env.n_bins)}


def _clipped() -> dict[str, Any]:
    env = Environment.from_samples(_POSITIONS, bin_size=5.0)
    return {"env": env.clip(Polygon([(0, 0), (100, 0), (0, 100)]))}


def _unrecorded_lattice() -> dict[str, Any]:
    grid = _grid()
    bins = replace(grid.binning, grid_spec=None)
    bins = replace(bins, faces=KnownFaces(grid.binning.faces.build()))
    return {"env": Environment.from_bins(bins, arena=grid.arena)}


def _with_regions() -> dict[str, Any]:
    regions = Regions()
    regions.add("reward", point=(50.0, 50.0))
    return {"env": _grid().with_regions(regions)}


def _rates(population: bool) -> dict[str, Any]:
    env = Environment.from_samples(_POSITIONS, bin_size=10.0)
    spikes = np.sort(_RNG.uniform(0, _TIMES[-1], 200))
    result = (
        compute_spatial_rates(env, [spikes, spikes], _TIMES, _POSITIONS)
        if population
        else compute_spatial_rate(env, spikes, _TIMES, _POSITIONS)
    )
    return {"env": env, "result": result}


def _three_edge_graph() -> dict[str, Any]:
    graph = nx.Graph()
    for node, pos in enumerate([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]):
        graph.add_node(node, pos=pos)
    edges = [(0, 1), (1, 2), (2, 3)]
    for u, v in edges:
        graph.add_edge(u, v, distance=10.0)
    return {"graph": graph, "edge_order": edges}


CASES: dict[str, Case] = {
    # rebin on bins that are not an even rectangular grid
    "hexagonal rebin(2)": Case(
        _hexagons,
        lambda n: n["env"].rebin(2),
        "Environment.from_samples(positions, bin_size=8, layout='Hexagonal')",
    ),
    "hexagonal rebin(2), smoothing instead": Case(
        _hexagons, lambda n: n["env"].rebin(2), "env.smooth(field, bandwidth=8)"
    ),
    "hexagonal rebin(bin_size=10)": Case(
        _hexagons,
        lambda n: n["env"].rebin(bin_size=10.0),
        "Environment.from_samples(positions, bin_size=10, layout='Hexagonal')",
    ),
    "track rebin(2)": Case(
        _track,
        lambda n: n["env"].rebin(2),
        "Environment.from_graph(env.binning.linearization.graph, "
        "list(env.binning.linearization.edge_order), bin_size=4, "
        "edge_spacing=5.0, track_width=10)",
    ),
    "track rebin(2), smoothing instead": Case(
        _track, lambda n: n["env"].rebin(2), "env.smooth(field, bandwidth=4)"
    ),
    "mesh rebin(2)": Case(
        _mesh,
        lambda n: n["env"].rebin(2),
        "from neurospatial.layout import create_layout; Environment.from_layout("
        "create_layout('TriangularMesh', boundary_polygon=env.arena.polygonal_domain(), "
        "point_spacing=4))",
    ),
    "clipped grid rebin(2)": Case(
        _clipped, lambda n: n["env"].rebin(2), "env.rebin(bin_size=10)"
    ),
    "rebin(bin_size=) without a recorded lattice": Case(
        _unrecorded_lattice, lambda n: n["env"].rebin(bin_size=8.0), "env.rebin(2)"
    ),
    # rebin factors
    "rebin(1.5)": Case(
        lambda: {"env": _grid()}, lambda n: n["env"].rebin(1.5), "env.rebin(2)"
    ),
    "rebin(1.5), a cell size instead": Case(
        lambda: {"env": _grid()}, lambda n: n["env"].rebin(1.5), "env.rebin(bin_size=6)"
    ),
    "rebin(0)": Case(
        lambda: {"env": _grid()}, lambda n: n["env"].rebin(0), "env.rebin(2)"
    ),
    "rebin((2, 2, 2))": Case(
        lambda: {"env": _grid()},
        lambda n: n["env"].rebin((2, 2, 2)),
        "env.rebin((2, 2))",
    ),
    # transforms and explicit bins
    "apply_transform(np.eye(3))": Case(
        lambda: {"env": _grid(), "matrix": np.eye(3)},
        lambda n: n["env"].apply_transform(n["matrix"]),
        "from neurospatial.ops import AffineND; env.apply_transform(AffineND(matrix))",
    ),
    "apply_transform with a 3-D transform": Case(
        lambda: {"env": _grid()},
        lambda n: n["env"].apply_transform(translate_3d(1, 2, 3)),
        "from neurospatial.ops import AffineND; env.apply_transform(AffineND(np.eye(3)))",
    ),
    "from_bins(env.binning)": Case(
        lambda: {"env": _grid(), "binning": _grid().binning},
        lambda n: Environment.from_bins(n["binning"]),
        "from dataclasses import replace; from neurospatial.binning import KnownFaces; "
        "binning = replace(binning, faces=KnownFaces(binning.faces.build())); "
        "Environment.from_bins(binning, arena=env.arena)",
    ),
    "from_polygon(points)": Case(
        lambda: {"polygon": [(0, 0), (10, 0), (10, 10)]},
        lambda n: Environment.from_polygon(n["polygon"], bin_size=1.0),
        "from shapely.geometry import Polygon; "
        "Environment.from_polygon(Polygon(polygon), bin_size=1.0)",
    ),
    # sample-based factories
    "transposed positions": Case(
        lambda: {"positions": _POSITIONS.T},
        lambda n: Environment.from_samples(n["positions"], bin_size=4.0),
        "positions.T",
        kind="value",
        retry=lambda n, value: Environment.from_samples(value, bin_size=4.0),
    ),
    "positions with a label column": Case(
        lambda: {
            "df": pd.DataFrame(
                {"x": [0.0, 10.0, 20.0], "y": [0.0, 5.0, 9.0], "label": "run"}
            )
        },
        lambda n: Environment.from_samples(n["df"], bin_size=2.0),
        "positions = df[['x', 'y']].to_numpy(dtype=float)",
        kind="value",
        retry=lambda n, positions: Environment.from_samples(positions, bin_size=2.0),
    ),
    "polygon as arena": Case(
        lambda: {"polygon": box(0, 0, 100, 100)},
        lambda n: Environment.from_samples(
            _POSITIONS, bin_size=4.0, arena=n["polygon"]
        ),
        "arena=OpenArena(polygon)",
        kind="argument",
        prelude="from neurospatial.arena import OpenArena",
        retry=lambda n, kw: Environment.from_samples(_POSITIONS, bin_size=4.0, **kw),
    ),
    "bin_size of the wrong length": Case(
        dict,
        lambda n: Environment.from_samples(_POSITIONS, bin_size=[4.0, 4.0, 4.0]),
        "bin_size=4",
        kind="argument",
        retry=lambda n, kw: Environment.from_samples(_POSITIONS, **kw),
    ),
    "hexagon bin_size per axis": Case(
        dict,
        lambda n: Environment.from_samples(
            _POSITIONS, bin_size=[4.0, 4.0], layout="Hexagonal"
        ),
        "bin_size=4",
        kind="argument",
        retry=lambda n, kw: Environment.from_samples(
            _POSITIONS, layout="Hexagonal", **kw
        ),
    ),
    "bin_size as text": Case(
        dict,
        lambda n: Environment.from_samples(_POSITIONS, bin_size="5"),
        "bin_size=5",
        kind="argument",
        retry=lambda n, kw: Environment.from_samples(_POSITIONS, **kw),
    ),
    "reversed dimension_ranges": Case(
        lambda: {"positions": np.array([[0.0, 0.0], [10.0, 20.0]])},
        lambda n: Environment.from_samples(
            n["positions"], bin_size=2.0, dimension_ranges=[(10, 0), (0, 20)]
        ),
        "dimension_ranges=[(0, 10), (0, 20)]",
        kind="argument",
        retry=lambda n, kw: Environment.from_samples(
            n["positions"], bin_size=2.0, **kw
        ),
    ),
    # track factories
    "linear_track(endpoints=[0, 200])": Case(
        dict,
        lambda n: Environment.linear_track(endpoints=[0, 200], bin_size=2.0),
        "endpoints=[(0, 0.0), (200, 0.0)]",
        kind="argument",
        retry=lambda n, kw: Environment.linear_track(bin_size=2.0, **kw),
    ),
    "linear_track with three endpoints": Case(
        dict,
        lambda n: Environment.linear_track(
            endpoints=[(0, 0), (10, 0), (10, 10)], bin_size=1.0
        ),
        "node_positions=[(0, 0), (10, 0), (10, 10)]",
        kind="argument",
        retry=lambda n, kw: Environment.linear_track(bin_size=1.0, **kw),
    ),
    "maze(node_positions={...: number})": Case(
        dict,
        lambda n: Environment.maze(
            kind="t", node_positions={0: 0, 1: 10, 2: 20, 3: 30}, bin_size=1.0
        ),
        "node_positions={0: (0, 0.0), 1: (10, 0.0), 2: (20, 0.0), 3: (30, 0.0)}",
        kind="argument",
        retry=lambda n, kw: Environment.maze(kind="t", bin_size=1.0, **kw),
    ),
    "edge_spacing with the wrong number of gaps": Case(
        _three_edge_graph,
        lambda n: Environment.from_graph(
            n["graph"], n["edge_order"], bin_size=1.0, edge_spacing=[5.0]
        ),
        "edge_spacing=[5.0, 5.0]",
        kind="argument",
        retry=lambda n, kw: Environment.from_graph(
            n["graph"], n["edge_order"], bin_size=1.0, **kw
        ),
    ),
    "from_grid_mask with edges for other cells": Case(
        lambda: {
            "active_mask": np.ones((3, 3), dtype=bool),
            "grid_edges": (np.arange(3.0), np.arange(4.0)),
        },
        lambda n: Environment.from_grid_mask(n["active_mask"], n["grid_edges"]),
        "grid_edges=tuple(np.linspace(edges[0], edges[-1], n + 1) for edges, n in "
        "zip(grid_edges, active_mask.shape))",
        kind="argument",
        retry=lambda n, kw: Environment.from_grid_mask(n["active_mask"], **kw),
    ),
    # fields
    "smooth with bins last": Case(
        lambda: {"env": _grid(), "field": np.ones((3, 100))},
        lambda n: n["env"].smooth(n["field"], bandwidth=2.0),
        "env.smooth(field.T, bandwidth=2.0).T",
    ),
    "diffuse with bins last": Case(
        lambda: {"env": _grid(), "fields": np.ones((3, 100))},
        lambda n: n["env"].diffuse(n["fields"], bandwidth=2.0),
        "env.diffuse(fields.T, bandwidth=2.0).T",
    ),
    "smooth(rate result)": Case(
        lambda: _rates(population=False),
        lambda n: n["env"].smooth(n["result"], bandwidth=10.0),
        "env.smooth(result.firing_rate, bandwidth=10.0)",
    ),
    "smooth(population rate result)": Case(
        lambda: _rates(population=True),
        lambda n: n["env"].smooth(n["result"], bandwidth=10.0),
        "env.smooth(result.firing_rates.T, bandwidth=10.0).T",
    ),
    # regions
    "region_membership(list of names)": Case(
        _with_regions,
        lambda n: n["env"].region_membership(["reward"]),
        "from neurospatial import Regions; env.region_membership(Regions("
        "env.regions[name] for name in ['reward']))",
    ),
    "bin_at on a track without track_width": Case(
        _widthless_track,
        lambda n: n["env"].bin_at(n["positions"]),
        _TRACK_WIDTH_CODE,
    ),
    "occupancy on a track without track_width": Case(
        _widthless_track,
        lambda n: n["env"].occupancy(np.arange(3) * 0.1, n["positions"]),
        _TRACK_WIDTH_CODE,
    ),
    "points= on a track without track_width": Case(
        _widthless_track,
        lambda n: n["env"].distance_to(points=n["positions"][:1]),
        _TRACK_WIDTH_CODE,
    ),
    "region_membership(list of names), one mask instead": Case(
        _with_regions,
        lambda n: n["env"].region_membership(["reward"]),
        "env.region_mask(['reward'])",
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_suggested_code_runs_on_the_refused_input(name):
    case = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # transposition and rebin-loss warnings
        names = case.names()
        with pytest.raises(Exception) as caught:
            case.refuse(names)
        fix = str(caught.value).splitlines()[-1]
        assert fix.startswith("Fix: ")
        for shown in (case.prelude, case.code):
            assert shown in fix, fix
        result = _follow(case, names)
    assert result is not None


def test_suggested_coarser_bins_are_coarser():
    """The rebuild a non-grid rebin suggests has larger cells than before."""
    for build in (_hexagons, _track, _mesh):
        names = build()
        with pytest.raises(NotImplementedError) as caught:
            names["env"].rebin(2)
        fix = str(caught.value).splitlines()[-1]
        code = next(
            CASES[name].code
            for name in CASES
            if CASES[name].names is build and CASES[name].code in fix
        )
        rebuilt = _run(code, names)
        assert rebuilt.n_bins < names["env"].n_bins
        assert rebuilt.layout_type == names["env"].layout_type


def test_suggested_track_rebuild_keeps_the_bins_and_drops_off_track_positions():
    names = _widthless_track()
    env = names["env"]
    rebuilt = _run(_TRACK_WIDTH_CODE, names)
    np.testing.assert_array_equal(rebuilt.bin_centers, env.bin_centers)
    assert (rebuilt.name, rebuilt.units, rebuilt.frame) == (
        env.name,
        env.units,
        env.frame,
    )
    bins = rebuilt.bin_at(names["positions"])
    # Within 5 cm of the centerline: a bin; 350 cm off the track: none.
    assert (bins[:2] >= 0).all() and bins[2] == -1
