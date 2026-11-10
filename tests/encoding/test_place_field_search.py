"""The place-field search follows its documented steps, at a cost linear in bins.

``detect_place_fields`` Notes, steps 2 to 6: take the highest bin left; a bin
with a higher neighbor in the full map is a flank and is set aside alone;
otherwise it is a peak and takes its connected region, or its core.
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pytest
import shapely

from neurospatial import Environment
from neurospatial.arena import OpenArena
from neurospatial.encoding import detect_place_fields
from neurospatial.layout import create_layout


def _region(env, seed, mask):
    """Bins of ``mask`` joined to ``seed`` through ``env.connectivity``."""
    return np.array(
        sorted(nx.node_connected_component(env.connectivity.subgraph(mask), seed))
    )


def _fields_one_bin_at_a_time(env, rate, *, threshold, min_peak_rate, subfields):
    """Notes steps 2 to 6, one bin per step, for an unsmoothed map.

    Each step takes the highest bin not assigned or set aside. A field must
    be larger than its peak bin (the smallest field at bandwidth 0).
    """
    sizes = env.bin_sizes
    left = {int(b) for b in np.flatnonzero(np.isfinite(rate))}
    fields = []
    while left:
        peak = min(left, key=lambda b: (-rate[b], b))
        if rate[peak] <= 0 or rate[peak] < min_peak_rate:
            break
        if any(rate[n] > rate[peak] for n in env.connectivity.neighbors(peak)):
            left.discard(peak)  # a flank: this bin only
            continue
        floor = threshold * rate[peak]
        field = _region(env, peak, [b for b in left if rate[b] >= floor])
        if subfields is not None:
            level = floor + threshold * (rate[peak] - floor)
            core = _region(env, peak, [b for b in left if rate[b] >= level])
            if sizes[core].sum() < sizes[field].sum() / 2:
                field = core
        left -= set(field.tolist())
        if sizes[field].sum() > sizes[peak]:
            fields.append(field)
    return fields


def _box(**options):
    edges = np.arange(0.0, 40.1, 2.5)
    return Environment.from_grid_mask(
        np.ones((16, 16), dtype=bool),
        grid_edges=(edges, edges),
        units="cm",
        **options,
    )


def _walled_box():
    outline = shapely.box(0, 0, 40, 40)
    wall = shapely.LineString([(20, 0), (20, 30)])
    return Environment.from_polygon(
        outline, bin_size=2.5, units="cm", arena=OpenArena(outline, walls={"w": wall})
    )


def _hexagons():
    layout = create_layout(
        "Hexagonal", hexagon_width=3.0, dimension_ranges=[(0, 40), (0, 40)]
    )
    return Environment.from_layout(layout, units="cm")


def _track():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(200.0, 0.0))
    graph.add_edge(0, 1)
    return Environment.from_graph(graph, [(0, 1)], bin_size=2.0, units="cm")


LAYOUTS = {
    "box": _box,
    "box without diagonals": lambda: _box(connect_diagonal_neighbors=False),
    "walled box": _walled_box,
    "hexagons": _hexagons,
    "track": _track,
}


def _bump(centers, rng, peak, width):
    center = centers[rng.integers(len(centers))]
    return peak * np.exp(-np.sum((centers - center) ** 2, axis=1) / (2 * width**2))


def _maps(env, rng):
    """Smooth, touching, noisy and tied maps, some with missing bins."""
    centers, n_bins = env.bin_centers, env.n_bins
    smooth = _bump(centers, rng, 10.0, 6.0) + _bump(centers, rng, 6.0, 4.0)
    # A strong and a weak field that merge at the threshold.
    seed = centers[rng.integers(n_bins)]
    offset = np.zeros(centers.shape[1])
    offset[0] = 16.0
    touching = 10.0 * np.exp(-np.sum((centers - seed) ** 2, axis=1) / 50.0)
    touching += 3.0 * np.exp(-np.sum((centers - seed - offset) ** 2, axis=1) / 50.0)
    noisy = rng.gamma(2.0, 1.0, n_bins)
    bumps_in_noise = smooth + rng.random(n_bins)
    # Whole-number rates: plateaus and equal peaks.
    tied = np.round(smooth + rng.random(n_bins))
    counts = rng.poisson(1.5, n_bins).astype(float)
    maps = [smooth, touching, noisy, bumps_in_noise, tied, counts]
    for rate in (noisy.copy(), tied.copy(), touching.copy()):
        rate[rng.choice(n_bins, size=n_bins // 10, replace=False)] = np.nan
        maps.append(rate)
    return maps


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("subfields", ["core", None])
def test_search_gives_the_fields_of_the_documented_steps(layout, subfields):
    """Every map gives the fields of the search taken one bin at a time."""
    env = LAYOUTS[layout]()
    rng = np.random.default_rng(11)
    n_fields = 0
    for rate in _maps(env, rng) + _maps(env, rng):
        for threshold, min_peak_rate in ((0.2, 1.0), (0.5, 0.0)):
            expected = _fields_one_bin_at_a_time(
                env,
                rate,
                threshold=threshold,
                min_peak_rate=min_peak_rate,
                subfields=subfields,
            )
            result = detect_place_fields(
                env,
                rate,
                bandwidth=0.0,
                threshold=threshold,
                min_peak_rate=min_peak_rate,
                subfields=subfields,
            )
            assert len(result.fields) == len(expected)
            for found, wanted in zip(result.fields, expected, strict=True):
                np.testing.assert_array_equal(found, wanted)
            n_fields += len(expected)
    assert n_fields > 20  # the maps do have fields to compare


def _two_smooth_fields(n_side):
    """Two Gaussian fields on an ``n_side`` x ``n_side`` box of 1 cm bins."""
    edges = np.arange(0.0, n_side + 0.5)
    env = Environment.from_grid_mask(
        np.ones((n_side, n_side), dtype=bool), grid_edges=(edges, edges), units="cm"
    )
    centers = env.bin_centers
    rate = np.zeros(env.n_bins)
    for x, peak in ((0.3, 10.0), (0.7, 6.0)):
        squared = np.sum((centers - [x * n_side, 0.5 * n_side]) ** 2, axis=1)
        rate += peak * np.exp(-squared / (2 * (0.08 * n_side) ** 2))
    return env, rate


def _seconds(env, rate):
    """Fastest of three unsmoothed detections."""
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        fields = detect_place_fields(env, rate, bandwidth=0.0)
        best = min(best, time.perf_counter() - start)
    assert len(fields) == 2
    return best


@pytest.mark.wallclock
def test_smooth_map_cost_grows_with_the_number_of_bins():
    """Four times the bins of a smooth map costs about four times the time.

    A smooth field has a flank bin above ``min_peak_rate`` for most of its
    bins. Scanning the whole map once per flank bin made the search
    quadratic: 13 times the time for these 4 times the bins, where one scan
    of the map per peak takes 3.4 to 3.8 times.
    """
    coarse = _seconds(*_two_smooth_fields(150))
    fine = _seconds(*_two_smooth_fields(300))
    assert fine < 8 * coarse
