"""Simulation-based calibration of ``decode_jointly``'s template probability.

Spikes are drawn from the decoder's own model: a template with its prior
probability, a bin of that template with probability proportional to its
length, then Poisson counts from that template's rates. Among time bins
given template probability near p, a fraction p must then come from that
template.

Two tracks of different length check that the default prior is proportional
to track length. One template is trained on part of its track only, which
checks the documented rule that its excluded bins pass their prior weight
to its other bins, so the template keeps its full prior probability.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import decode_jointly

N_TRIALS = 40_000
N_CELLS = 8
DT = 0.05


@pytest.fixture(scope="module")
def joint_decode():
    envs = {
        "long": Environment.linear_track(
            endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
        ),
        "short": Environment.linear_track(
            endpoints=[[0.0, 0.0], [60.0, 0.0]], bin_size=4.0, units="cm"
        ),
    }
    stimulus = np.random.default_rng(0)
    rates = {}
    for name, env in envs.items():
        x = env.bin_centers[:, 0]
        centers = stimulus.uniform(x.min(), x.max(), N_CELLS)
        rates[name] = 1.0 + 15.0 * np.exp(
            -0.5 * ((x[None] - centers[:, None]) / 10.0) ** 2
        )
    rates["long"][:, 14:] = np.nan  # trained on its first 70 cm only

    # Prior probability of each template: its whole track length.
    names = tuple(envs)
    lengths = np.array([envs[name].bin_sizes.sum() for name in names])
    draw = np.random.default_rng(1)
    template = draw.choice(len(names), N_TRIALS, p=lengths / lengths.sum())
    counts = np.empty((N_TRIALS, N_CELLS), dtype=np.int64)
    for k, name in enumerate(names):
        trials = np.flatnonzero(template == k)
        supported = np.all(np.isfinite(rates[name]), axis=0)
        weight = np.where(supported, envs[name].bin_sizes, 0.0)
        true_bin = draw.choice(envs[name].n_bins, len(trials), p=weight / weight.sum())
        counts[trials] = draw.poisson(rates[name][:, true_bin].T * DT)
    with pytest.warns(UserWarning, match="NaN or Inf rate"):
        result = decode_jointly(envs, counts, rates, dt=DT)
    return result.template_probability[:, 0], template == 0, lengths


def test_mean_template_probability_is_its_prior(joint_decode):
    probability, from_long, lengths = joint_decode
    prior = lengths[0] / lengths.sum()  # 100 / 160, not 70 / 130
    se = np.sqrt(prior * (1 - prior) / N_TRIALS)
    assert abs(probability.mean() - prior) < 4 * se
    assert abs(from_long.mean() - prior) < 4 * se


def test_template_probability_is_reliable(joint_decode):
    """Per decile of p, the share of bins from the template is p within 4 SE."""
    probability, from_long, _ = joint_decode
    decile = np.minimum((probability * 10).astype(int), 9)
    checked = 0
    for d in range(10):
        in_decile = decile == d
        if in_decile.sum() < 500:
            continue
        checked += 1
        predicted = probability[in_decile].mean()
        se = np.sqrt(predicted * (1 - predicted) / in_decile.sum())
        gap = from_long[in_decile].mean() - predicted
        assert abs(gap) < 4 * se, (d, predicted, gap, se)
    assert checked >= 8
