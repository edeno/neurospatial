"""Decoding on bins of unequal size compares probability over space.

The posterior holds a probability mass per bin. A prior that knows nothing
about position is uniform over space, so its mass is proportional to bin
length; the MAP bin and the credible region rank bins by density, mass per
unit length. Splitting a bin, or renumbering bins, must not change what is
decoded.
"""

import math

import numpy as np
import pytest
from environment_helpers import renumbered

from neurospatial import Environment
from neurospatial.binning import (
    BinConnections,
    Binning,
    MissingFaces,
    NearestBinLocator,
    SpatialBins,
)
from neurospatial.decoding import (
    BayesianDecoder,
    confusion_matrix,
    credible_region,
    decode_position,
    decode_position_summary,
    log_poisson_likelihood,
    map_position,
    normalize_to_posterior,
)

DT = 0.25


def track_from_edges(edges):
    """1-D track whose bins are the intervals between ``edges`` (cm)."""
    edges = np.asarray(edges, dtype=float)
    centers = ((edges[:-1] + edges[1:]) / 2)[:, None]
    n = len(centers)
    pairs = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    return Environment.from_bins(
        Binning(
            "explicit",
            SpatialBins.from_arrays(centers, np.diff(edges), 1),
            BinConnections.from_arrays(pairs, np.diff(centers[:, 0])),
            NearestBinLocator(centers, max_distance=math.inf),
            MissingFaces("test track"),
        ),
        units="cm",
    )


# First bin 0-50 cm, then five 10 cm bins; the split copy cuts the first bin
# into two 25 cm halves.
COARSE = track_from_edges([0, 50, 60, 70, 80, 90, 100])
SPLIT = track_from_edges([0, 25, 50, 60, 70, 80, 90, 100])


def field(env, center, width=15.0, peak=10.0, base=0.1):
    """One Gaussian place field evaluated at the bin centers, shape (1, n_bins)."""
    x = env.bin_centers[:, 0]
    return (peak * np.exp(-((x - center) ** 2) / (2 * width**2)) + base)[None, :]


def split_rates(rates):
    """Give both halves of the split first bin the coarse first bin's rate."""
    return np.concatenate([rates[:, :1], rates], axis=1)


def test_untuned_cells_decode_to_the_center_of_the_track():
    """A flat likelihood leaves the prior: uniform over 0-100 cm, mean 50 cm."""
    rates = np.full((3, COARSE.n_bins), 2.0)
    result = decode_position(COARSE, np.array([[0, 1, 0], [1, 0, 0]]), rates, dt=DT)

    np.testing.assert_allclose(result.posterior, [[0.5] + [0.1] * 5] * 2)
    np.testing.assert_allclose(result.mean_position[:, 0], 50.0)


def test_rate_result_and_session_decoders_use_a_uniform_over_space_prior():
    """Decoders that intersect the prior with training support agree."""
    rates = np.full((2, COARSE.n_bins), 2.0)
    support = np.ones(COARSE.n_bins, dtype=bool)
    decoder = BayesianDecoder.from_models(COARSE, rates, spatial_support=support, dt=DT)
    spikes = [np.array([0.1]), np.array([0.6])]
    result = decoder.predict(spikes, np.linspace(0.0, 1.0, 11))
    summary = decoder.predict_summary(spikes, np.linspace(0.0, 1.0, 11))

    assert result.n_time_bins > 0
    np.testing.assert_allclose(
        result.posterior, [[0.5] + [0.1] * 5] * result.n_time_bins
    )
    np.testing.assert_allclose(summary.mean_position[:, 0], 50.0)


def test_splitting_a_bin_does_not_change_the_decoded_density():
    """Halving a bin into two equal-rate halves leaves decoding unchanged."""
    counts = np.array([[1], [2], [3]])
    coarse_rates = field(COARSE, center=20.0)
    coarse = decode_position(COARSE, counts, coarse_rates, dt=DT)
    split = decode_position(SPLIT, counts, split_rates(coarse_rates), dt=DT)

    coarse_density = coarse.posterior / COARSE.bin_sizes
    split_density = split.posterior / SPLIT.bin_sizes
    np.testing.assert_allclose(split_density[:, 1:], coarse_density)
    np.testing.assert_allclose(split_density[:, 0], coarse_density[:, 0])
    np.testing.assert_allclose(split.mean_position, coarse.mean_position)

    # The MAP bin lies in the same stretch of track and has the same density.
    np.testing.assert_array_equal(coarse.map_bin, 0)
    assert set(split.map_bin.tolist()) <= {0, 1}
    rows = np.arange(len(counts))
    np.testing.assert_allclose(
        split_density[rows, split.map_bin],
        coarse_density[rows, coarse.map_bin],
    )

    # The credible region covers the same length of track.
    for level in (0.3, 0.5, 0.8, 0.95):
        coarse_regions = credible_region(COARSE, coarse.posterior, level=level)
        split_regions = credible_region(SPLIT, split.posterior, level=level)
        for c, s in zip(coarse_regions, split_regions, strict=True):
            assert SPLIT.bin_sizes[s].sum() == pytest.approx(COARSE.bin_sizes[c].sum())


def test_map_ranks_bins_by_density_not_mass():
    """A large bin holding the most mass is not the mode when it is sparse."""
    # Bin 0 holds 0.45 over 50 cm; bin 2 holds 0.2 over 10 cm.
    posterior = np.array([[0.45, 0.1, 0.2, 0.1, 0.1, 0.05]])

    assert map_position(COARSE, posterior)[0, 0] == pytest.approx(65.0)
    np.testing.assert_array_equal(
        confusion_matrix(COARSE, posterior, np.array([2]), method="map")[2],
        np.eye(COARSE.n_bins)[2],
    )


def test_credible_region_is_the_smallest_region_by_length():
    """The region ranks by density, so it is shorter than ranking by mass."""
    posterior = np.array([[0.45, 0.1, 0.2, 0.1, 0.1, 0.05]])
    region = credible_region(COARSE, posterior, level=0.5)[0]

    # Densities (per 10 cm): 0.09, 0.1, 0.2, 0.1, 0.1, 0.05. Bin 2, then the
    # three tied 0.1 bins, reach 0.5 in 40 cm; ranking by mass would take
    # bin 0 and bin 2 and cover 60 cm.
    assert region.tolist() == [2, 1, 3, 4]
    assert COARSE.bin_sizes[region].sum() == pytest.approx(40.0)


def test_credible_region_does_not_depend_on_bin_numbering():
    """Bins tied in density at the boundary all enter the region."""
    env = track_from_edges(np.arange(0.0, 61.0, 10.0))
    posterior = np.array([[0.4, 0.15, 0.15, 0.15, 0.15, 0.0]])
    order = np.array([5, 3, 0, 4, 1, 2])
    renamed = renumbered(env, order)

    region = credible_region(env, posterior, level=0.5)[0]
    renamed_region = credible_region(renamed, posterior[:, order], level=0.5)[0]

    assert sorted(region.tolist()) == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(
        np.sort(env.bin_centers[region, 0]),
        np.sort(renamed.bin_centers[renamed_region, 0]),
    )


def test_streamed_summary_reports_the_density_mode():
    """``decode_position_summary`` agrees with the full posterior's MAP."""
    counts = np.array([[1], [0], [2], [5]])
    rates = field(COARSE, center=35.0)
    full = decode_position(COARSE, counts, rates, dt=DT)
    summary = decode_position_summary(COARSE, counts, rates, dt=DT, time_chunk=2)

    np.testing.assert_array_equal(summary.map_bin, full.map_bin)
    np.testing.assert_allclose(summary.map_position, full.map_position)


def test_mixed_measure_environment_needs_one_measure():
    """Lengths and areas have no common density, so the default prior refuses."""
    corridor = Environment.linear_track(
        endpoints=[[-3.0, 5.0], [-23.0, 5.0]], bin_size=2.0, track_width=2.0, units="cm"
    )
    edges = np.arange(0.0, 11.0, 2.0)
    arena = Environment.from_grid_mask(
        np.ones((5, 5), dtype=bool), grid_edges=(edges, edges), units="cm"
    )
    env = Environment.combine({"corridor": corridor, "arena": arena})
    rates = np.ones((1, env.n_bins))

    with pytest.raises(ValueError, match=r"(?s)mixes bins.*env\.subset"):
        decode_position(env, np.array([[1]]), rates, dt=DT)

    # An explicit prior decodes; density-free summaries still work.
    result = decode_position(
        env, np.array([[1]]), rates, dt=DT, prior=np.ones(env.n_bins)
    )
    summary = result.summary()
    assert summary["n_undecodable"] == 0
    assert summary["mean_entropy"] == pytest.approx(np.log2(env.n_bins))
    with pytest.raises(ValueError, match="mixes bins"):
        _ = result.map_bin


def test_impossible_spikes_fall_back_to_the_prior_over_space():
    """A time bin no position can explain keeps the uniform-over-space prior."""
    silent = np.zeros((1, COARSE.n_bins))
    result = decode_position(COARSE, np.array([[1]]), silent, dt=DT)

    np.testing.assert_allclose(result.posterior, [[0.5] + [0.1] * 5])


def test_equal_bins_decode_exactly_as_a_flat_prior():
    """Equal bins keep the flat prior and the plain argmax, bit for bit."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=3.0, track_width=2.0, units="cm"
    )
    rng = np.random.default_rng(1)
    rates = rng.uniform(0.5, 20.0, (8, env.n_bins))
    counts = np.random.default_rng(2).poisson(1.0, (50, 8))
    result = decode_position(env, counts, rates, dt=0.05)

    expected = normalize_to_posterior(log_poisson_likelihood(counts, rates, dt=0.05))
    np.testing.assert_array_equal(result.posterior, expected)
    np.testing.assert_array_equal(result.map_bin, expected.argmax(axis=1))
