"""Simulation-based calibration of the Bayesian position decoder.

When positions are drawn from the decoder's prior and spikes from its own
Poisson model, the posterior is the exact conditional distribution of the
true bin. Two consequences hold for any environment:

- a credible region's posterior mass equals, on average, the probability
  that it contains the true bin;
- among (time bin, spatial bin) pairs with posterior probability near p, a
  fraction p hold the true bin.

The default prior is uniform over space, so true bins are drawn with
probability proportional to ``env.bin_sizes``. Bins of unequal size (a track
with short and long bins, polar sectors whose area grows with distance)
check that the decoder compares probability per unit length or area.

Coverage is compared with the mean region mass rather than with the nominal
level: a region of whole bins overshoots ``level``.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)
from neurospatial.decoding import credible_region, decode_position

N_TRIALS = 20_000
N_NEURONS = 10
DT = 0.1
LEVELS = (0.5, 0.8, 0.95)
UNEQUAL_EDGES = [0, 30, 35, 40, 45, 50, 70, 75, 80, 100]


def _unequal_track() -> Environment:
    edges = np.asarray(UNEQUAL_EDGES, dtype=float)
    centers = ((edges[:-1] + edges[1:]) / 2)[:, None]
    n = len(centers)
    pairs = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(centers, np.diff(edges), 1),
            BinConnections.from_arrays(pairs, np.diff(centers[:, 0])),
            NearestBinLocator(centers, max_distance=math.inf),
            MissingFaces("calibration track"),
        ),
        units="cm",
    )


def _polar() -> Environment:
    return Environment.from_polar_egocentric(
        distance_range=(0.0, 30.0),
        angle_range=(-np.pi, np.pi),
        distance_bin_size=5.0,
        angle_bin_size=np.pi / 8,
    )


def _maze() -> Environment:
    """A T-maze graph: a 60 cm stem and two 40 cm arms."""
    graph = nx.Graph()
    for node, pos in {0: (0, 0), 1: (0, 60), 2: (-40, 60), 3: (40, 60)}.items():
        graph.add_node(node, pos=pos)
    edges = [(0, 1), (1, 2), (1, 3)]
    for u, v in edges:
        graph.add_edge(u, v)
    return Environment.from_graph(
        graph, edges, bin_size=6.0, track_width=5.0, units="cm"
    )


def _grid_with_holes() -> Environment:
    mask = np.ones((8, 8), dtype=bool)
    mask[2:4, 2:5] = False
    mask[5:7, 5] = False
    edges = np.arange(9) * 5.0
    return Environment.from_grid_mask(mask, grid_edges=(edges, edges), units="cm")


ENVIRONMENTS = {
    "unequal_track": _unequal_track,
    "polar": _polar,
    "maze": _maze,
    "grid_with_holes": _grid_with_holes,
}


def _planar(env: Environment) -> np.ndarray:
    """Bin centres in Cartesian coordinates, shape (n_bins, n_dims)."""
    centers = np.asarray(env.bin_centers, dtype=float)
    if env.coordinate_kind == "polar":
        distance, angle = centers[:, 0], centers[:, 1]
        return np.column_stack([distance * np.cos(angle), distance * np.sin(angle)])
    return centers


def _simulate(env: Environment, *, prior: np.ndarray | None = None):
    """Draw true bins from the uniform-over-space prior and decode them."""
    xy = _planar(env)
    stimulus = np.random.default_rng(0)
    low, high = xy.min(axis=0), xy.max(axis=0)
    field_centers = stimulus.uniform(low, high, (N_NEURONS, xy.shape[1]))
    width = 0.2 * float(np.max(high - low))
    distance_sq = ((xy[None] - field_centers[:, None]) ** 2).sum(axis=-1)
    rates = 2.0 + 20.0 * np.exp(-distance_sq / (2 * width**2))

    weights = env.bin_sizes / env.bin_sizes.sum()
    true_bin = stimulus.choice(env.n_bins, size=N_TRIALS, p=weights)
    spikes = np.random.default_rng(1)
    counts = spikes.poisson(rates[:, true_bin].T * DT)
    posterior = decode_position(env, counts, rates, dt=DT, prior=prior).posterior
    return np.asarray(posterior), true_bin


def _coverage_and_mass(env, posterior, true_bin, level):
    regions = credible_region(env, posterior, level=level)
    covered = np.array(
        [t in region for t, region in zip(true_bin, regions, strict=True)]
    )
    mass = np.array([posterior[i, region].sum() for i, region in enumerate(regions)])
    return covered.mean(), mass.mean()


@pytest.fixture(scope="module", params=list(ENVIRONMENTS))
def calibration(request):
    env = ENVIRONMENTS[request.param]()
    posterior, true_bin = _simulate(env)
    return env, posterior, true_bin


def test_posterior_is_informative(calibration):
    """Calibration of the prior alone would also pass; the spikes must matter."""
    env, posterior, true_bin = calibration
    prior_mass = env.bin_sizes / env.bin_sizes.sum()
    trials = np.arange(len(true_bin))
    assert posterior[trials, true_bin].mean() > 2 * prior_mass[true_bin].mean()


@pytest.mark.parametrize("level", LEVELS)
def test_credible_region_coverage_matches_its_mass(calibration, level):
    env, posterior, true_bin = calibration
    coverage, mass = _coverage_and_mass(env, posterior, true_bin, level)
    assert mass >= level
    assert abs(coverage - mass) < 0.015, (coverage, mass)


def test_posterior_probabilities_are_reliable(calibration):
    """Bins assigned probability p hold the true position a fraction p of the time."""
    _, posterior, true_bin = calibration
    predicted = posterior.ravel()
    hit = np.zeros_like(posterior, dtype=bool)
    hit[np.arange(len(true_bin)), true_bin] = True
    hit = hit.ravel()
    decile = np.minimum((predicted * 10).astype(int), 9)
    checked = 0
    for d in range(10):
        in_decile = decile == d
        if in_decile.sum() < 2000:
            continue
        checked += 1
        gap = abs(predicted[in_decile].mean() - hit[in_decile].mean())
        assert gap < 0.03, (d, gap)
    assert checked >= 3


def test_flat_per_bin_prior_is_miscalibrated_on_unequal_bins():
    """A prior uniform per bin, not over space, breaks calibration.

    Negative control: true positions are uniform over space, so a prior that
    gives every bin equal mass overweights the short bins.
    """
    env = _unequal_track()
    posterior, true_bin = _simulate(env, prior=np.ones(env.n_bins))
    gaps = [
        abs(np.subtract(*_coverage_and_mass(env, posterior, true_bin, level)))
        for level in LEVELS
    ]
    assert max(gaps) > 0.015, gaps
