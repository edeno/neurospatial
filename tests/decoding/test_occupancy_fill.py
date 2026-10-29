"""Encoder masking of low-occupancy bins and decoder NaN handling.

Covers the contract that low-occupancy bins masked by ``min_occupancy``:

- are stored as NaN (missing) and lie outside ``spatial_support``,
- let the documented encode -> decode golden path (passing the rate result)
  compose with no manual NaN scrubbing,
- and that a raw residual-NaN encoding model still decodes (NaN bins
  excluded) while warning exactly once.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment, encoding
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


def test_low_occupancy_bins_are_stored_as_missing(
    low_occupancy_session: tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """Bins below min_occupancy are NaN in the stored rates, never zero."""
    env, spike_times_list, times, positions = low_occupancy_session

    result = compute_spatial_rate(
        env,
        spike_times_list[0],
        times,
        positions,
        method="binned",
        bandwidth=0.0,
        min_occupancy=0.5,
    )
    firing_rate = np.asarray(result.firing_rate)
    low_occ = np.asarray(result.occupancy) < 0.5
    assert low_occ.any(), "fixture should produce low-occupancy bins"
    assert np.isnan(firing_rate[low_occ]).all()
    np.testing.assert_array_equal(result.spatial_support, np.isfinite(firing_rate))


@pytest.mark.parametrize(
    "function",
    [
        encoding.compute_spatial_rate,
        encoding.compute_spatial_rates,
        encoding.has_place_field,
        encoding.is_place_cell,
        encoding.place_cell_significance,
    ],
)
def test_fill_value_is_not_a_parameter(function, low_occupancy_session) -> None:
    """Rates keep missing bins NaN; there is no option to store a fill."""
    env, spike_times_list, times, positions = low_occupancy_session
    with pytest.raises(TypeError, match="fill_value"):
        function(env, spike_times_list, times, positions, fill_value=0.0)


def test_encode_decode_golden_path(
    low_occupancy_session: tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """min_occupancy=0.5 rates decode end to end when passed as the result.

    No manual NaN scrubbing or fill; ``decode_position(validate=True)`` does
    not raise, and unsupported bins get no posterior mass.
    """
    env, spike_times_list, times, positions = low_occupancy_session

    rates = compute_spatial_rates(
        env,
        spike_times_list,
        times,
        positions,
        method="binned",
        bandwidth=0.0,
        min_occupancy=0.5,
    )
    assert not rates.spatial_support.all(), "fixture should leave unsupported bins"

    # Build a few time bins of spike counts.
    dt = 0.25
    n_time_bins = 20
    rng = np.random.default_rng(1)
    spike_counts = rng.poisson(1.0, (n_time_bins, len(spike_times_list))).astype(
        np.int64
    )

    # validate=True is the default; must not raise on this golden-path model.
    result = decode_position(env, spike_counts, rates, dt=dt)
    assert result.posterior.shape == (n_time_bins, env.n_bins)
    np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(result.posterior[:, ~rates.spatial_support] == 0.0)


def test_decode_warns_on_nan_model(
    low_occupancy_session: tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray],
) -> None:
    """A model with NaN bins decodes (NaN bins excluded) and warns once."""
    env, spike_times_list, times, positions = low_occupancy_session

    # The raw array keeps its NaN bins; passed bare, they reach the decoder.
    rates = compute_spatial_rates(
        env,
        spike_times_list,
        times,
        positions,
        method="binned",
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


def test_decode_partial_nan_bin_gets_zero_posterior() -> None:
    """A bin missing a rate for any neuron is excluded, not decoded from the rest.

    Leaving out only the masked neuron would favour the bin (each observed
    Poisson term is at most 1); a zero fill would treat a missing estimate
    as silence. The bin gets zero mass and the remaining bins keep the
    posterior of the complete model restricted to them.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins

    rng = np.random.default_rng(0)
    rates = rng.uniform(1.0, 10.0, (2, n_bins))
    rates_nan = rates.copy()
    rates_nan[0, 0] = np.nan
    spike_counts = np.array([[3, 3], [0, 0]], dtype=np.int64)
    dt = 0.1

    with pytest.warns(UserWarning, match="1 of"):
        post_nan = decode_position(env, spike_counts, rates_nan, dt=dt).posterior

    np.testing.assert_array_equal(post_nan[:, 0], 0.0)
    prior = np.ones(n_bins)
    prior[0] = 0.0
    expected = decode_position(env, spike_counts, rates, dt=dt, prior=prior).posterior
    np.testing.assert_allclose(post_nan, expected, rtol=1e-12)


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


def test_decode_model_with_no_finite_bins_raises() -> None:
    """A model where EVERY spatial bin is all-NaN is unusable and raises.

    Regression for the "fully-NaN model decodes to a confident uniform over
    invalid bins" finding. When every spatial bin is NaN across all neurons,
    the entire likelihood is -inf and the "uniform" degeneracy handler would
    otherwise hand back a confident-looking uniform posterior over positions
    that carry zero information. ``decode_position`` must instead refuse loudly
    with a ``ValueError`` before producing a meaningless posterior.

    Repro: ``encoding_models = np.full((1, n_bins), nan)``, ``spike_counts =
    [[0]]`` -- previously returned a uniform posterior.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins

    encoding_models = np.full((1, n_bins), np.nan)
    spike_counts = np.array([[0]], dtype=np.int64)

    with pytest.raises(ValueError, match="no bin where every neuron"):
        decode_position(env, spike_counts, encoding_models, dt=1.0)


def test_decode_no_finite_bins_raises_multineuron_all_columns_nan() -> None:
    """No finite bins with several neurons: every column all-NaN still raises.

    The "no finite bins" condition is column-wise: a spatial bin is killed only
    if NaN for ALL neurons. With multiple neurons, if every bin (column) is
    all-NaN, the model still carries zero information and must raise.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins

    encoding_models = np.full((3, n_bins), np.nan)
    spike_counts = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="no bin where every neuron"):
        decode_position(env, spike_counts, encoding_models, dt=1.0)


def test_decode_partial_model_one_finite_bin_still_decodes() -> None:
    """A model with >=1 finite bin must decode normally (no raise).

    Guards the contract boundary: the "no finite bins" raise must NOT fire on a
    model that still has at least one observable bin, even if most bins are
    all-NaN. The model decodes from the finite bin(s); all-NaN bins get ~0 mass.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins
    assert n_bins >= 2

    # All bins NaN for the single neuron EXCEPT the last one (finite, 5 Hz).
    rates = np.full((1, n_bins), np.nan)
    rates[0, -1] = 5.0
    spike_counts = np.array([[2]], dtype=np.int64)

    with pytest.warns(UserWarning):
        post = decode_position(env, spike_counts, rates, dt=0.1).posterior

    assert np.isfinite(post).all()
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    # The only finite (observed) bin carries all the mass and is the MAP.
    assert int(post[0].argmax()) == n_bins - 1


def test_decode_all_inf_model_validate_false_raises() -> None:
    """An all-Inf model carries no finite bins and must raise, even Inf-only.

    Regression for the "no finite bins guard was NaN-only" finding: an all-Inf
    model is invalid input (not a low-occupancy mask), but with validate=False
    the validate=True Inf rejection is bypassed. The unconditional no-finite-
    bins guard must still recognize Inf -- ``np.isnan`` is False for Inf -- and
    refuse rather than decode to a meaningless uniform posterior.

    Repro: ``encoding_models = np.full((1, n_bins), inf)``, ``spike_counts =
    [[0]]``, ``validate=False`` -- previously returned a uniform posterior.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins

    encoding_models = np.full((1, n_bins), np.inf)
    spike_counts = np.array([[0]], dtype=np.int64)

    with pytest.raises(ValueError, match="no bin where every neuron"):
        decode_position(env, spike_counts, encoding_models, dt=1.0, validate=False)


def test_decode_no_finite_bins_mixed_nan_inf_raises() -> None:
    """A mixed model where every bin is non-finite (NaN or Inf) must raise.

    No single bin is finite for any neuron: some columns are all-NaN, the rest
    all-Inf. The model carries zero information and must raise regardless of
    whether the non-finiteness is NaN or Inf, and even with validate=False.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins
    assert n_bins >= 2

    encoding_models = np.full((1, n_bins), np.nan)
    encoding_models[0, n_bins // 2 :] = np.inf  # other half all-Inf
    spike_counts = np.array([[0]], dtype=np.int64)

    with pytest.raises(ValueError, match="no bin where every neuron"):
        decode_position(env, spike_counts, encoding_models, dt=1.0, validate=False)


def test_decode_one_finite_bin_with_inf_elsewhere_validate_false_decodes() -> None:
    """A model with >=1 finite bin must not trip the no-finite-bins guard.

    Even with Inf in other bins and validate=False (so the Inf rejection is
    bypassed), a single finite bin means the model is usable: the guard must
    not fire (no false positive). Partial-Inf handling beyond this is out of
    scope -- this only asserts the guard does not over-reject.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins
    assert n_bins >= 2

    rates = np.full((1, n_bins), np.inf)
    rates[0, -1] = 5.0  # one finite bin
    spike_counts = np.array([[2]], dtype=np.int64)

    # Must not raise the no-finite-bins ValueError (guard sees a finite bin).
    post = decode_position(env, spike_counts, rates, dt=0.1, validate=False).posterior
    assert post.shape == (1, n_bins)


def test_decode_partial_inf_model_concentrates_on_finite_bin() -> None:
    """A partial-Inf model puts mass on the finite bin, not a uniform posterior.

    Regression: with ``validate=False`` a partial-Inf encoding model such as
    ``rates=[inf, inf, 5]`` (>=1 finite bin) previously warned then returned a
    uniform posterior ``[1/3, 1/3, 1/3]``. The per-bin "unobserved" exclusion
    must treat Inf bins like NaN bins (masked/excluded, -inf log-likelihood ->
    zero posterior), so all mass concentrates on the single finite bin.
    """
    positions = np.linspace(0.0, 4.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins
    assert n_bins == 3

    rates = np.array([[np.inf, np.inf, 5.0]])
    spike_counts = np.array([[3]], dtype=np.int64)

    with pytest.warns(UserWarning):
        post = decode_position(
            env, spike_counts, rates, dt=0.1, validate=False
        ).posterior

    assert np.isfinite(post).all()
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    # Mass concentrates on bin 2 (the only finite bin), NOT a uniform 1/3 each.
    assert int(post[0].argmax()) == 2
    np.testing.assert_allclose(post[0], [0.0, 0.0, 1.0], atol=1e-9)


def test_decode_no_spike_valid_model_yields_uniform_posterior() -> None:
    """A no-spike time bin against a VALID (finite) model decodes to ~uniform.

    This is the case that MUST keep working: a legitimate no-spike time bin
    against a fully finite encoding model produces a flat finite likelihood
    that correctly normalizes to uniform. It must NOT be confused with the
    unusable all-NaN model (which raises) nor flipped to NaN.
    """
    positions = np.linspace(0.0, 10.0, 200).reshape(-1, 1)
    env = Environment.from_samples(positions, bin_size=2.0)
    n_bins = env.n_bins

    # Fully finite, flat firing rate; no spikes this time bin.
    rates = np.full((2, n_bins), 5.0)
    spike_counts = np.array([[0, 0]], dtype=np.int64)

    post = decode_position(env, spike_counts, rates, dt=0.1).posterior

    assert np.isfinite(post).all()
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    # Flat likelihood -> uniform posterior.
    np.testing.assert_allclose(post[0], 1.0 / n_bins, atol=1e-9)
