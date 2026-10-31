"""Geodesic earth mover's distance as a flow along the environment's graph."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import linprog
from shapely.geometry import LineString, box

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding import compute_field_emd


def _dense_geodesic_emd(env: Environment, p: np.ndarray, q: np.ndarray) -> float:
    """Reference: the full transport plan over shortest-path distances."""
    bins = np.flatnonzero((p > 0) | (q > 0))
    lengths = dict(
        nx.all_pairs_dijkstra_path_length(env.connectivity, weight="distance")
    )
    cost = np.array([[lengths[i][j] for j in bins] for i in bins])
    n = bins.size
    ones = np.ones((1, n))
    constraints = sparse.vstack(
        [sparse.kron(sparse.eye(n), ones), sparse.kron(ones, sparse.eye(n))]
    )
    solved = linprog(
        cost.ravel(),
        A_eq=constraints,
        b_eq=np.r_[p[bins], q[bins]],
        bounds=(0, None),
        method="highs",
    )
    return float(solved.fun)


@pytest.fixture(scope="module")
def walled_room() -> Environment:
    """A 30 x 30 cm room with a wall from the left side to 20 cm."""
    outline = box(0, 0, 30, 30)
    return Environment.from_polygon(
        outline,
        bin_size=3.0,
        arena=OpenArena(outline, walls={"wall": LineString([(0, 15), (20, 15)])}),
        units="cm",
    )


def test_flow_along_the_graph_matches_the_full_transport_plan(walled_room):
    rng = np.random.default_rng(3)
    centers = walled_room.bin_centers
    field_1 = np.exp(-np.sum((centers - [5, 5]) ** 2, axis=1) / 30)
    field_2 = np.exp(-np.sum((centers - [5, 25]) ** 2, axis=1) / 30)
    # Unsampled bins in each map: mass still travels through them.
    field_1[rng.choice(walled_room.n_bins, 6, replace=False)] = np.nan
    field_2[rng.choice(walled_room.n_bins, 6, replace=False)] = np.nan
    common = np.isfinite(field_1) & np.isfinite(field_2)
    measures = walled_room.bin_sizes / walled_room.bin_sizes.max()
    p = np.where(common, np.nan_to_num(field_1) * measures, 0.0)
    q = np.where(common, np.nan_to_num(field_2) * measures, 0.0)
    expected = _dense_geodesic_emd(walled_room, p / p.sum(), q / q.sum())
    assert compute_field_emd(
        walled_room, field_1, field_2, metric="geodesic"
    ) == pytest.approx(expected, rel=1e-6)


def test_a_bump_moved_along_a_corridor_moves_that_distance():
    track = Environment.from_samples(
        np.arange(0.5, 100.0)[:, None],
        bin_size=1.0,
        infer_active_bins=False,
        units="cm",
    )
    x = track.bin_centers[:, 0]
    field_1 = np.exp(-0.5 * ((x - 30) / 4) ** 2)
    field_2 = np.exp(-0.5 * ((x - 55) / 4) ** 2)
    assert compute_field_emd(
        track, field_1, field_2, metric="geodesic"
    ) == pytest.approx(25.0, rel=1e-6)


@pytest.mark.wallclock
def test_geodesic_emd_scales_to_a_finely_binned_arena():
    """3,600 bins (1.5 m at 2.5 cm): the dense plan had 13 million variables."""
    env = Environment.from_polygon(box(0, 0, 150, 150), bin_size=2.5, units="cm")
    centers = env.bin_centers
    field_1 = np.exp(-np.sum((centers - [40, 40]) ** 2, axis=1) / 200)
    field_2 = np.exp(-np.sum((centers - [110, 90]) ** 2, axis=1) / 200)
    start = time.perf_counter()
    emd = compute_field_emd(env, field_1, field_2, metric="geodesic")
    assert time.perf_counter() - start < 10.0
    assert 80 < emd < 95  # 86 cm between the centers, plus the grid's excess


def test_euclidean_emd_solves_a_balanced_problem_on_a_full_box():
    """The pairwise plan fixes every row and column sum, so one constraint is
    redundant; with the two totals equal only to roundoff, the solver called
    this valid 625-bin problem infeasible and the EMD came back NaN."""
    env = Environment.from_polygon(box(0, 0, 150, 150), bin_size=6.0, units="cm")
    centers = env.bin_centers
    field_1 = np.exp(-np.sum((centers - [42, 42]) ** 2, axis=1) / 200)
    field_2 = np.exp(-np.sum((centers - [108, 90]) ** 2, axis=1) / 200)
    # The same field shifted by whole bins: it moves by the shift.
    assert compute_field_emd(
        env, field_1, field_2, metric="euclidean"
    ) == pytest.approx(np.hypot(66, 48), rel=1e-4)
