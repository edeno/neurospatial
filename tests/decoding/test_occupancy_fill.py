"""Reconcile encoder ``fill_value`` masking with decoder NaN handling.

Covers the contract that low-occupancy bins masked by ``min_occupancy``:

- stay NaN by default (``fill_value=None``, no behavior change),
- become finite zero rate when ``fill_value=0.0`` is passed,
- let the documented encode -> decode golden path compose with no manual
  ``np.nan_to_num`` scrubbing,
- and that a residual-NaN encoding model still decodes (NaN bins excluded)
  while warning exactly once.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding.posterior import decode_position
from neurospatial.encoding.spatial import compute_spatial_rate, compute_spatial_rates


@pytest.fixture
def low_occupancy_session() -> tuple[
    Environment, list[np.ndarray], np.ndarray, np.ndarray
]:
    """A 1D track session with deliberately low occupancy in a subset of bins.

    The animal dwells over most of the track but only briefly touches its far
    end, so the far-end bins fall below ``min_occupancy`` and get masked.

    Returns
    -------
    env : Environment
        1D linear-track environment.
    spike_times_list : list of ndarray
        Spike times for 3 neurons.
    times : ndarray
        Trajectory timestamps.
    positions : ndarray, shape (n_samples, 1)
        Trajectory positions.
    """
    rng = np.random.default_rng(0)
    # Dense sampling over [0, 40], very sparse over (40, 50].
    dense = np.linspace(0.0, 40.0, 4000)
    sparse = np.linspace(40.0, 50.0, 20)
    positions_1d = np.concatenate([dense, sparse])
    positions = positions_1d.reshape(-1, 1)
    times = np.arange(len(positions_1d)) * 0.02  # 50 Hz
    env = Environment.from_samples(positions, bin_size=2.0)

    # Three place cells with fields in the well-sampled region.
    spike_times_list = [
        np.sort(rng.uniform(times[0], times[-1], 200)),
        np.sort(rng.uniform(times[0], times[-1], 150)),
        np.sort(rng.uniform(times[0], times[-1], 250)),
    ]
    return env, spike_times_list, times, positions


def test_spatial_rate_fill_value_zero(
    low_occupancy_session: tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """fill_value=0.0 yields finite (zero) rates in low-occupancy bins."""
    env, spike_times_list, times, positions = low_occupancy_session

    result = compute_spatial_rate(
        env,
        spike_times_list[0],
        times,
        positions,
        smoothing_method="binned",
        bandwidth=0.0,
        min_occupancy=0.5,
        fill_value=0.0,
    )

    firing_rate = np.asarray(result.firing_rate)
    occupancy = np.asarray(result.occupancy)

    # The masking actually fired (there are low-occupancy bins to fill).
    low_occ = occupancy < 0.5
    assert low_occ.any(), "fixture should produce low-occupancy bins"

    # No NaN anywhere, and the previously masked bins are exactly zero.
    assert np.isfinite(firing_rate).all()
    assert np.allclose(firing_rate[low_occ], 0.0)


def test_spatial_rate_fill_value_none_preserves_nan(
    low_occupancy_session: tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Default fill_value=None keeps NaN in masked bins (no behavior change)."""
    env, spike_times_list, times, positions = low_occupancy_session

    # Explicit None and the default must agree, and both must preserve NaN.
    result_default = compute_spatial_rate(
        env,
        spike_times_list[0],
        times,
        positions,
        smoothing_method="binned",
        bandwidth=0.0,
        min_occupancy=0.5,
    )
    result_none = compute_spatial_rate(
        env,
        spike_times_list[0],
        times,
        positions,
        smoothing_method="binned",
        bandwidth=0.0,
        min_occupancy=0.5,
        fill_value=None,
    )

    fr_default = np.asarray(result_default.firing_rate)
    fr_none = np.asarray(result_none.firing_rate)
    occupancy = np.asarray(result_default.occupancy)

    low_occ = occupancy < 0.5
    assert low_occ.any(), "fixture should produce low-occupancy bins"
    assert np.isnan(fr_default[low_occ]).all()
    # Default == explicit None bit-for-bit (treating NaN as equal).
    np.testing.assert_array_equal(
        np.nan_to_num(fr_default, nan=-1.0), np.nan_to_num(fr_none, nan=-1.0)
    )


def test_encode_decode_golden_path(
    low_occupancy_session: tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Documented min_occupancy=0.5, fill_value=0.0 -> decode runs end to end.

    No manual NaN scrubbing; ``decode_position(validate=True)`` does not raise.
    """
    env, spike_times_list, times, positions = low_occupancy_session

    rates = compute_spatial_rates(
        env,
        spike_times_list,
        times,
        positions,
        smoothing_method="binned",
        bandwidth=0.0,
        min_occupancy=0.5,
        fill_value=0.0,
    )
    encoding_models = np.asarray(rates.firing_rates)
    assert np.isfinite(encoding_models).all()

    # Build a few time bins of spike counts.
    dt = 0.25
    n_time_bins = 20
    rng = np.random.default_rng(1)
    spike_counts = rng.poisson(1.0, (n_time_bins, len(spike_times_list))).astype(
        np.int64
    )

    # validate=True is the default; must not raise on this golden-path model.
    result = decode_position(env, spike_counts, encoding_models, dt=dt)
    assert result.posterior.shape == (n_time_bins, env.n_bins)
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-6)


def test_decode_warns_on_nan_model(
    low_occupancy_session: tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """A model with NaN bins decodes (NaN bins excluded) and warns once."""
    env, spike_times_list, times, positions = low_occupancy_session

    # fill_value=None -> NaN bins survive into the encoding model.
    rates = compute_spatial_rates(
        env,
        spike_times_list,
        times,
        positions,
        smoothing_method="binned",
        bandwidth=0.0,
        min_occupancy=0.5,
    )
    encoding_models = np.asarray(rates.firing_rates)
    assert np.isnan(encoding_models).any(), "fixture should leave NaN bins"

    dt = 0.25
    n_time_bins = 20
    rng = np.random.default_rng(2)
    spike_counts = rng.poisson(1.0, (n_time_bins, len(spike_times_list))).astype(
        np.int64
    )

    with pytest.warns(UserWarning) as record:
        result = decode_position(env, spike_counts, encoding_models, dt=dt)

    # Exactly one warning per call.
    nan_warnings = [w for w in record if "NaN" in str(w.message)]
    assert len(nan_warnings) == 1

    # Decoded posterior is well-formed despite the NaN encoding bins.
    assert result.posterior.shape == (n_time_bins, env.n_bins)
    assert np.isfinite(result.posterior).all()
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-6)


def test_decode_partial_nan_bin_excluded_not_penalized() -> None:
    """A partial-NaN bin decodes from its observing neuron(s), not penalized.

    Refined contract (partial- vs all-NaN):

    - **Partial-NaN bin** (NaN for *some* neurons but observed by >=1 other):
      the NaN neurons' ``(neuron, bin)`` terms are *excluded* from the Poisson
      sum at that bin, and the bin still decodes from the observing neurons.
      "Excluded, not penalized" means: relative to a literal zero-fill, a spike
      from the masked neuron must not crush the bin's posterior the way a true
      zero rate clipped to the likelihood floor would.
    - **All-NaN bin** (NaN for *every* neuron) -> zero posterior; tested
      separately in :func:`test_decode_all_nan_bin_gets_zero_posterior`.

    This test exercises the partial-NaN case: two neurons, with the target bin
    NaN for one neuron but observed by the other.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins

    # Two neurons, flat 5 Hz everywhere. Neuron 0 is masked (NaN) at bin 0,
    # but neuron 1 still observes bin 0 -> bin 0 is a *partial*-NaN bin.
    rates_nan = np.full((2, n_bins), 5.0)
    rates_nan[0, 0] = np.nan
    # Both neurons fire in this time bin.
    spike_counts = np.array([[3, 3]], dtype=np.int64)
    dt = 0.1

    with pytest.warns(UserWarning):
        post_nan = decode_position(env, spike_counts, rates_nan, dt=dt).posterior

    # Well-formed posterior; bin 0 still decodes from neuron 1, so it is not
    # spuriously killed: it retains real, non-negligible mass.
    assert np.isfinite(post_nan).all()
    np.testing.assert_allclose(post_nan.sum(axis=1), 1.0, atol=1e-6)
    assert post_nan[0, 0] > 0.0

    # Contrast: a literal zero rate at bin 0 for neuron 0 (clipped to the
    # likelihood floor) makes neuron 0's 3 spikes there astronomically
    # unlikely, crushing bin 0's posterior. Excluding the NaN term must leave
    # bin 0 with vastly more mass than the zero-fill penalty would.
    rates_zero = rates_nan.copy()
    rates_zero[0, 0] = 0.0
    post_zero = decode_position(env, spike_counts, rates_zero, dt=dt).posterior
    assert post_nan[0, 0] > post_zero[0, 0] * 1e3


def test_decode_all_nan_bin_gets_zero_posterior() -> None:
    """An all-NaN bin carries no information and gets zero posterior mass.

    Regression for the all-NaN-bin contract: a bin that is NaN for *every*
    neuron in ``encoding_models`` has no observing neuron, so its excluded-term
    log-likelihood would collapse to a neutral 0 and let an uninformative bin
    win the MAP. Such bins must instead receive ~0 posterior and never be the
    argmax.

    Repro (pre-fix posterior was ``[0.9867, 0.0066, 0.0066]`` with bin 0 the
    spurious MAP): ``encoding_models=[[nan, 5, 5]]``, ``spike_counts=[[0]]``,
    ``dt=1.0``.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins

    # Single neuron -> bin 0 (NaN for that neuron) is an all-NaN bin.
    rates = np.full((1, n_bins), 5.0)
    rates[0, 0] = np.nan
    spike_counts = np.array([[0]], dtype=np.int64)
    dt = 1.0

    with pytest.warns(UserWarning):
        post = decode_position(env, spike_counts, rates, dt=dt).posterior

    assert np.isfinite(post).all()
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    # All-NaN bin gets ~0 mass and is NOT the argmax.
    assert post[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert int(post[0].argmax()) != 0
