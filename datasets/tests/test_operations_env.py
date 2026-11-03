from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from _geometry import (
    declared_circle,
    declared_rectangle,
    distance_to_segments,
    fit_circle,
    fit_rectangle,
    graph_edge_order,
    graph_environment,
    lattice_graph,
)
from operations import _env, env_declared, env_kind

from neurospatial import Environment


@pytest.fixture
def square_loop():
    graph = nx.Graph()
    corners = {0: (0.0, 0.0), 1: (100.0, 0.0), 2: (100.0, 100.0), 3: (0.0, 100.0)}
    for node, pos in corners.items():
        graph.add_node(node, pos=pos)
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    for a, b in graph.edges:
        graph.edges[a, b]["distance"] = 100.0
    return graph


def test_graph_edge_order_lists_every_edge(square_loop):
    order = graph_edge_order(square_loop, root=0)
    assert {frozenset(e) for e in order} == {frozenset(e) for e in square_loop.edges}


def test_graph_env_keeps_every_edge(square_loop):
    env = graph_environment(
        square_loop, bin_size=5.0, track_width=10.0, units="cm", name="loop", root=0
    )
    assert env.n_bins == 80
    # Points on the edge a depth-first order would drop (3 -> 0, along x = 0).
    points = np.array([[0.0, 10.0], [0.0, 50.0]])
    centres = np.asarray(env.bin_centers)[np.asarray(env.bin_at(points))]
    assert np.all(np.linalg.norm(centres - points, axis=1) <= 2.5 + 1e-9)


def test_declared_environment_is_used(simulated_recording):
    declared = Environment.from_polygon(
        __import__("shapely.geometry", fromlist=["box"]).box(0, 0, 100, 100),
        bin_size=2.5,
        units="cm",
    )
    rec = replace(simulated_recording, geometry={"environment": lambda: declared})
    cache: dict = {}
    assert env_kind(rec, cache) == "declared"
    assert _env(rec, cache) is declared
    outcome = env_declared(rec, {})
    assert outcome.status == "ok"
    assert outcome.summary["frac_samples_outside"] < 0.01
    forced = {"inferred_env": True}
    assert env_kind(rec, forced) == "inferred"
    assert _env(rec, forced) is not declared


def test_no_declared_environment_is_na(simulated_recording):
    assert env_declared(simulated_recording, {}).status == "n/a"


def test_fitted_square_recovers_pose():
    rng = np.random.default_rng(0)
    local = rng.uniform(-50, 50, size=(20000, 2))
    angle = np.radians(17.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = local @ rot.T + np.array([30.0, -12.0])
    polygon, fit = fit_rectangle(points, 100.0, 100.0)
    assert fit["centre_x"] == pytest.approx(30.0, abs=1.0)
    assert fit["centre_y"] == pytest.approx(-12.0, abs=1.0)
    # A square's rotation is defined modulo 90 degrees.
    assert np.degrees(fit["rotation_rad"]) % 90 == pytest.approx(17.0, abs=1.0)
    assert polygon.area == pytest.approx(100.0 * 100.0)
    assert (
        np.mean(
            [
                polygon.buffer(1.0).contains(__import__("shapely").Point(p))
                for p in points[:2000]
            ]
        )
        == 1.0
    )


def test_fitted_circle_recovers_centre():
    rng = np.random.default_rng(1)
    r = 48.0 * np.sqrt(rng.uniform(0, 1, 20000))
    theta = rng.uniform(0, 2 * np.pi, 20000)
    points = np.column_stack([r * np.cos(theta) + 55.0, r * np.sin(theta) + 60.0])
    polygon, fit = fit_circle(points, 96.0)
    assert fit["centre_x"] == pytest.approx(55.0, abs=1.0)
    assert fit["centre_y"] == pytest.approx(60.0, abs=1.0)
    assert fit["samples_diameter"] == pytest.approx(96.0, abs=2.0)
    assert polygon.area == pytest.approx(np.pi * 48.0**2, rel=0.01)


@pytest.fixture(scope="module")
def box_samples():
    """Samples filling a 96 x 96 cm square centred at (10, 20): a 100 cm box's floor."""
    rng = np.random.default_rng(2)
    return rng.uniform(-48, 48, size=(20000, 2)) + np.array([10.0, 20.0])


def test_declared_rectangle_records_residuals(box_samples):
    factory, line, fit = declared_rectangle(
        box_samples, 100.0, 100.0, bin_size=2.5, units="cm", name="box", source="test"
    )
    env = factory()
    # 100 cm / 2.5 cm bins: 40 x 40 cells, one more or less where the fitted
    # rotation moves a cell centre across the outline.
    assert env.n_bins == pytest.approx(1600, abs=2)
    assert np.all(np.asarray(env.bin_at(box_samples)) >= 0)
    # The samples' own sides are 96 cm: 4 % short of the stated 100 cm.
    residuals = [fit["size_residual_a"], fit["size_residual_b"]]
    assert residuals == pytest.approx([-0.04, -0.04], abs=0.005)
    assert f"size_residual_a {fit['size_residual_a']:.3g}" in line
    assert line.startswith("environment: declared square (100 x 100, test)")


def test_declared_rectangle_refuses_a_wrong_size(box_samples):
    with pytest.raises(ValueError, match=r"size_residual_a -36\.\d%"):
        declared_rectangle(
            box_samples, 150.0, 150.0, bin_size=2.5, units="cm", name="box", source="t"
        )


def test_declared_circle_at_a_stated_centre():
    rng = np.random.default_rng(3)
    r = 45.0 * np.sqrt(rng.uniform(0, 1, 20000))
    theta = rng.uniform(0, 2 * np.pi, 20000)
    points = np.column_stack([r * np.cos(theta) + 1.0, r * np.sin(theta)])
    factory, line, fit = declared_circle(
        points,
        100.0,
        bin_size=2.0,
        units="cm",
        name="pool",
        source="t",
        centre=(0.0, 0.0),
    )
    env = factory()
    centres = np.asarray(env.bin_centers)
    assert np.hypot(*centres.T).max() <= 50.0
    assert np.hypot(*centres.mean(axis=0)) < 1.0  # centred on the stated centre
    assert fit["centre_offset"] == pytest.approx(1.0, abs=0.3)
    assert fit["size_residual"] == pytest.approx(-0.1, abs=0.01)
    assert f"centre_offset {fit['centre_offset']:.3g}" in line


def test_lattice_graph_cuts_bars_at_crossings():
    # A 2 x 2 x 2 cube lattice of side 20: 3 x 3 bars along each axis.
    ticks = np.array([0.0, 10.0, 20.0])
    bars = []
    for axis in range(3):
        for a in ticks:
            for b in ticks:
                start = np.insert(np.array([a, b]), axis, 0.0)
                bars.append([start, np.insert(np.array([a, b]), axis, 20.0)])
    graph = lattice_graph(np.asarray(bars), n_cubes=2)
    assert graph.number_of_nodes() == 27
    assert graph.number_of_edges() == 3 * 9 * 2
    assert all(d == pytest.approx(10.0) for *_, d in graph.edges(data="distance"))


def test_distance_to_segments_matches_brute_force():
    rng = np.random.default_rng(5)
    segments = rng.uniform(0, 10, size=(7, 2, 3))
    points = rng.uniform(-2, 12, size=(50, 3))
    t = np.linspace(0, 1, 20001)
    dense = segments[:, None, 0] + t[None, :, None] * (
        segments[:, None, 1] - segments[:, None, 0]
    )
    brute = np.linalg.norm(points[:, None, :] - dense.reshape(-1, 3)[None], axis=2).min(
        axis=1
    )
    np.testing.assert_allclose(
        distance_to_segments(points, segments, chunk=8), brute, atol=1e-3
    )


@pytest.mark.parametrize(
    ("side", "accepted"),
    [
        (98.0, True),  # samples 2 % past the wall: within one 2.5 cm bin
        (90.0, False),  # samples 6.7 % past it: more than one bin
        (110.0, True),  # samples 13 % short of the walls: allowed
        (115.0, False),  # samples 17 % short: refused
    ],
)
def test_declared_rectangle_limits_are_one_bin_long_and_15_percent_short(
    box_samples, side, accepted
):
    # box_samples fill a 96 cm square.
    def declare():
        return declared_rectangle(
            box_samples, side, side, bin_size=2.5, units="cm", name="b", source="t"
        )

    if accepted:
        declare()
    else:
        with pytest.raises(ValueError, match="limits -15% short, one bin long"):
            declare()


def test_declared_circle_refuses_a_far_stated_centre():
    rng = np.random.default_rng(6)
    r = 45.0 * np.sqrt(rng.uniform(0, 1, 5000))
    theta = rng.uniform(0, 2 * np.pi, 5000)
    points = np.column_stack([r * np.cos(theta) + 10.0, r * np.sin(theta)])
    # 10 cm off a 50 cm radius is 20 %, past the 15 % limit.
    with pytest.raises(ValueError, match=r"another frame or unit"):
        declared_circle(
            points, 100.0, bin_size=2.0, units="cm", name="p", source="t", centre=(0, 0)
        )
