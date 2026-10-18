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
from neurospatial.io import from_dict, from_file, to_dict, to_file
from neurospatial.layout import create_layout
from neurospatial.ops.transforms import AffineND, scale_2d, translate_3d

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


_TRACK_WIDTH_CODE = "env.with_track_width(width)"


def _widthless_track_subset() -> dict[str, Any]:
    names = _widthless_track()
    env = names["env"]
    names["env"] = env.subset(mask=env.bin_centers[:, 0] < 30.0)
    return names


def _widthless_track_transformed() -> dict[str, Any]:
    from neurospatial.ops.transforms import AffineND

    names = _widthless_track()
    rotation = np.array([[0.0, -2.0, 5.0], [2.0, 0.0, -3.0], [0.0, 0.0, 1.0]])
    names["env"] = names["env"].apply_transform(AffineND(rotation))
    names["positions"] = names["positions"] @ rotation[:2, :2].T + rotation[:2, 2]
    names["width"] = 20.0  # the track doubled in size
    return names


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


def _clipped_pixels() -> dict[str, Any]:
    env = Environment.from_pixel_mask(np.ones((20, 20), dtype=bool), pixel_size=5.0)
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
    # track operations on an environment that is not a track
    "to_linear on a grid, a track instead": Case(
        _sampled,
        lambda n: n["env"].to_linear(n["positions"][:3]),
        "Environment.linear_track(endpoints=[(0.0, 0.0), (100.0, 0.0)], "
        "bin_size=2.0, track_width=8.0)",
    ),
    "to_linear on a grid, bins instead": Case(
        _sampled,
        lambda n: n["env"].to_linear(n["positions"][:3]),
        "env.bin_at(positions)",
    ),
    "linear_to_nd on a grid, bin centers instead": Case(
        lambda: {**_sampled(), "bins": [0, 1]},
        lambda n: n["env"].linear_to_nd(np.array([1.0])),
        "env.bin_center_of(bins)",
    ),
    # rebin on bins that are not an even rectangular grid
    "hexagonal rebin(2)": Case(
        _hexagons,
        lambda n: n["env"].rebin(2),
        "Environment.from_samples(positions, bin_size=8, layout='Hexagonal')",
    ),
    "hexagonal rebin(2), smoothing instead": Case(
        _hexagons,
        lambda n: n["env"].rebin(2),
        "env.smooth(field, bandwidth=8, mode='average')",
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
        _track,
        lambda n: n["env"].rebin(2),
        "env.smooth(field, bandwidth=4, mode='average')",
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
    "clipped pixel grid rebin(2)": Case(
        _clipped_pixels, lambda n: n["env"].rebin(2), "env.rebin(bin_size=10)"
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
    "from_polar_egocentric with more than one turn": Case(
        dict,
        lambda n: Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            angle_range=(-np.pi, 2 * np.pi),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
        ),
        f"angle_range=({-np.pi!r}, {-np.pi!r} + 2 * np.pi)",
        kind="argument",
        retry=lambda n, kw: Environment.from_polar_egocentric(
            distance_range=(0.0, 30.0),
            distance_bin_size=10.0,
            angle_bin_size=np.pi / 4,
            **kw,
        ),
    ),
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
        "node_positions={0: (0, 0), 1: (0, 50), 2: (-30, 50), 3: (30, 50)}",
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
        lambda n: n["env"].smooth(n["field"], bandwidth=2.0, mode="average"),
        "env.smooth(field.T, bandwidth=2.0, mode='average').T",
    ),
    "diffuse with bins last": Case(
        lambda: {"env": _grid(), "fields": np.ones((3, 100))},
        lambda n: n["env"].diffuse(n["fields"], bandwidth=2.0, mode="density"),
        "env.diffuse(fields.T, bandwidth=2.0, mode='density').T",
    ),
    "smooth(rate result)": Case(
        lambda: _rates(population=False),
        lambda n: n["env"].smooth(n["result"], bandwidth=10.0, mode="average"),
        "env.smooth(result.firing_rate, bandwidth=10.0, mode='average')",
    ),
    "smooth(population rate result)": Case(
        lambda: _rates(population=True),
        lambda n: n["env"].smooth(n["result"], bandwidth=10.0, mode="average"),
        "env.smooth(result.firing_rates.T, bandwidth=10.0, mode='average').T",
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
    "bin_at on a subset of a track without track_width": Case(
        _widthless_track_subset,
        lambda n: n["env"].bin_at(n["positions"]),
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


# A quarter turn that doubles lengths, then a shift.
_SIMILARITY = np.array([[0.0, -2.0, 5.0], [2.0, 0.0, -3.0], [0.0, 0.0, 1.0]])


def _doubled(build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """The names of ``build`` after ``_SIMILARITY``, positions included."""
    names = build()
    names["env"] = names["env"].apply_transform(AffineND(_SIMILARITY))
    if "positions" in names:
        names["positions"] = (
            names["positions"] @ _SIMILARITY[:2, :2].T + _SIMILARITY[:2, 2]
        )
    return names


def _rebin_fix(env: Environment) -> str:
    with pytest.raises(NotImplementedError) as caught:
        env.rebin(2)
    return str(caught.value).splitlines()[-1]


@pytest.mark.parametrize(
    ("build", "size"),
    [
        (_hexagons, "bin_size=16"),
        (_track, "bin_size=8"),
        (_mesh, "point_spacing=8"),
        (_clipped, "bin_size=20"),
    ],
    ids=["hexagons", "track", "mesh", "clipped grid"],
)
def test_rebin_suggests_sizes_in_transformed_coordinates(build, size):
    """After apply_transform doubles the cells, rebin(2) suggests twice their new size.

    The recorded build size (4, 2, 2 and 5) is in the coordinates the bins
    were built in; the suggestion is in the coordinates they are in now.
    """
    names = _doubled(build)
    env = names["env"]
    fix = _rebin_fix(env)
    assert size in fix, fix
    code = fix.split(": ", 2)[2].split("; or smooth")[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # rebin-loss warnings
        rebuilt = _run(code, names)
    # Merging two bins along each axis: cells 2**measure_dim times larger.
    np.testing.assert_allclose(
        np.median(rebuilt.bin_sizes),
        2**env.binning.bins.measure_dim * np.median(env.bin_sizes),
        rtol=0.25,
    )


def test_rebin_suggests_no_size_after_a_stretch():
    """A stretch leaves no single cell size, so the suggestion gives none."""
    env = _hexagons()["env"].apply_transform(scale_2d(2.0, 3.0))
    fix = _rebin_fix(env)
    assert "bin_size=...," in fix, fix
    assert "bandwidth" not in fix


def test_rebin_suggestion_survives_saving(tmp_path):
    """A saved transformed environment suggests the same size when read back."""
    env = _doubled(_hexagons)["env"]
    to_file(env, tmp_path / "env")
    for restored in (from_dict(to_dict(env)), from_file(tmp_path / "env")):
        assert _rebin_fix(restored) == _rebin_fix(env)


def test_suggested_track_rebuild_keeps_a_transformed_track_width():
    """A track moved and scaled by apply_transform is rebuilt at its new width."""
    from neurospatial.ops.transforms import AffineND

    names = _track()
    rotation = np.array([[0.0, -2.0, 5.0], [2.0, 0.0, -3.0], [0.0, 0.0, 1.0]])
    names["env"] = names["env"].apply_transform(AffineND(rotation))
    with pytest.raises(NotImplementedError) as caught:
        names["env"].rebin(2)
    fix = str(caught.value).splitlines()[-1]
    # The 10 cm wide track doubled in size.
    assert "track_width=20)" in fix, fix
    code = fix[fix.index("Environment.from_graph(") : fix.index("track_width=20)") + 15]
    rebuilt = _run(code, names)
    assert rebuilt.arena.width == 20.0
    # The first segment now runs along +y; a point 9 units to its side is
    # on the 20-wide track (it would be off a 10-wide one).
    nodes = names["env"].binning.linearization.graph.nodes
    middle = (np.asarray(nodes[0]["pos"]) + np.asarray(nodes[1]["pos"])) / 2
    assert rebuilt.bin_at(middle[None] + [9.0, 0.0])[0] >= 0


@pytest.mark.parametrize(
    "build",
    [_widthless_track, _widthless_track_subset, _widthless_track_transformed],
    ids=["native", "subset", "transformed"],
)
def test_suggested_track_rebuild_keeps_the_bins_and_drops_off_track_positions(
    build,
):
    names = build()
    env = names["env"]
    rebuilt = _run(_TRACK_WIDTH_CODE, names)
    # The same bins as the refused environment, subset or transform included.
    np.testing.assert_array_equal(rebuilt.bin_centers, env.bin_centers)
    np.testing.assert_array_equal(rebuilt.bin_sizes, env.bin_sizes)
    assert rebuilt.is_linearized_track
    assert (rebuilt.name, rebuilt.units, rebuilt.frame) == (
        env.name,
        env.units,
        env.frame,
    )
    np.testing.assert_array_equal(
        rebuilt.bin_at(rebuilt.bin_centers), np.arange(rebuilt.n_bins)
    )
    bins = rebuilt.bin_at(names["positions"])
    # The last position is 350 cm off the track: no bin.
    assert bins[-1] == -1
    if build is not _widthless_track_subset:
        # Within half the width of the centerline: a bin.
        assert (bins[:2] >= 0).all()
