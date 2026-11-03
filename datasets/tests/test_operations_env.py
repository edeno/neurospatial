from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from _geometry import fit_circle, fit_rectangle, graph_edge_order, graph_environment
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
