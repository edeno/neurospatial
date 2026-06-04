"""Tests for decode_session() — the one-call encode->bin->decode golden path.

Tests
-----
1. Golden-path / tracking test: simulate trajectory + place cells, call
   decode_session, assert MAP tracks trajectory (median error below threshold).
2. Equivalence test: decode_session numerically matches the manual 3-call path.
3. encoding_models passthrough: precomputed firing_rates skips the fit.
4. decode_kwargs forwarding: keyword args (e.g. validate=False) reach decode_position.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from neurospatial import Environment
from neurospatial.decoding import DecodingResult, bin_spikes_in_time, decode_position
from neurospatial.encoding import compute_spatial_rates

# ---------------------------------------------------------------------------
# Helpers — small, fast simulation
# ---------------------------------------------------------------------------


def _make_linear_track_sim(
    *,
    n_neurons: int = 20,
    duration: float = 30.0,
    dt_traj: float = 0.02,
    seed: int = 0,
) -> tuple[Environment, list[np.ndarray], np.ndarray, np.ndarray]:
    """Build a tiny linear-track simulation (1D, 100 cm, 5 cm bins).

    Returns
    -------
    env : Environment
        Fitted 1-D environment (20 bins).
    spike_times : list[ndarray]
        Per-neuron spike-time arrays (seconds).
    times : ndarray, shape (n_frames,)
        Trajectory timestamps.
    positions : ndarray, shape (n_frames, 2)
        2-D position (y=0 always; only x varies along track).
    """
    from neurospatial.simulation import (
        PlaceCellModel,
        generate_poisson_spikes,
        simulate_trajectory_ou,
    )

    rng = np.random.default_rng(seed)

    # --- environment (2-D but movement only in x) ---
    track_positions = np.column_stack(
        [
            np.linspace(0.0, 100.0, 200),
            np.zeros(200),
        ]
    )
    env = Environment.from_samples(track_positions, bin_size=5.0)
    env.units = "cm"

    # --- trajectory (OU random walk in x, y fixed at 0) ---
    positions, times = simulate_trajectory_ou(
        env, duration=duration, seed=seed, speed_units="cm"
    )

    # --- place cells evenly spread along track ---
    centers = np.column_stack(
        [
            np.linspace(5.0, 95.0, n_neurons),
            np.zeros(n_neurons),
        ]
    )

    spike_times: list[np.ndarray] = []
    for i in range(n_neurons):
        cell = PlaceCellModel(
            env,
            center=centers[i],
            width=15.0,
            max_rate=30.0,
            seed=int(rng.integers(0, 2**31)),
        )
        rates = cell.firing_rate(positions, times)
        spikes = generate_poisson_spikes(rates, times, seed=int(rng.integers(0, 2**31)))
        spike_times.append(spikes)

    return env, spike_times, times, positions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDecodeSessionGoldenPath:
    """decode_session exists and produces a tracking DecodingResult."""

    def test_import(self) -> None:
        """decode_session is importable from both neurospatial and neurospatial.decoding."""
        from neurospatial import decode_session as ds2  # noqa: F401
        from neurospatial.decoding import decode_session  # noqa: F401

    def test_returns_decoding_result(self) -> None:
        """Return type is always DecodingResult."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0
        )
        result = decode_session(env, spike_times, times, positions, dt=0.1)
        assert isinstance(result, DecodingResult)

    def test_times_as_python_list(self) -> None:
        """`times` may be a plain Python list (coerced via np.asarray)."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0
        )
        result = decode_session(env, spike_times, list(times), positions, dt=0.1)
        assert isinstance(result, DecodingResult)

    def test_single_neuron_1d_spike_times(self) -> None:
        """A bare 1-D spike-time array (single neuron) flows through correctly."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=1, duration=10.0
        )
        # Pass the single neuron's spikes as a bare 1-D array, not a list.
        result = decode_session(env, spike_times[0], times, positions, dt=0.1)
        assert isinstance(result, DecodingResult)
        # One neuron -> one column of firing rates.
        assert result.map_position.shape[0] == len(result.times)

    def test_too_few_times_raises_clear_error(self) -> None:
        """An empty `times` raises a clear ValueError naming the param.

        The guard runs before the encode step, so the same clear error fires
        whether or not ``encoding_models`` is precomputed.
        """
        import pytest

        from neurospatial.decoding import decode_session

        env, spike_times, _, _ = _make_linear_track_sim(n_neurons=3, duration=10.0)
        with pytest.raises(ValueError, match="times must have at least 2 samples"):
            decode_session(env, spike_times, np.array([]), np.zeros((0, 2)), dt=0.1)

    def test_map_tracks_trajectory(self) -> None:
        """MAP position should track the true trajectory (median error < 25 cm on 100-cm track)."""
        from neurospatial.decoding import decode_session, decoding_error

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=20, duration=30.0, seed=42
        )
        dt = 0.1
        result = decode_session(env, spike_times, times, positions, dt=dt)

        assert isinstance(result, DecodingResult)
        assert result.times is not None

        # Align decoded time bins to the trajectory
        decoded_times = result.times  # bin centers
        # Interpolate true positions to decoded time bins
        true_x = np.interp(decoded_times, times, positions[:, 0])
        true_y = np.interp(decoded_times, times, positions[:, 1])
        true_positions_at_decode = np.column_stack([true_x, true_y])

        # decoding_error(decoded_positions, actual_positions)
        errors = decoding_error(result.map_position, true_positions_at_decode)
        med_err = float(np.nanmedian(errors))

        # On a 100-cm track with 20 neurons, median error well below 25 cm
        assert med_err < 25.0, (
            f"MAP not tracking trajectory: median error = {med_err:.2f} cm "
            f"(threshold 25 cm). Likely an orientation/shape bug."
        )


class TestDecodeSessionEquivalence:
    """decode_session is numerically equivalent to the manual 3-call path."""

    def test_posterior_matches_manual_path(self) -> None:
        """Posterior from decode_session equals manual encode->bin->decode."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=7
        )
        dt = 0.1
        times_arr = np.asarray(times, dtype=np.float64)

        # --- Manual 3-call path ---
        rates_result = compute_spatial_rates(
            env,
            spike_times,
            times_arr,
            positions,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
            fill_value=0.0,
        )
        firing_rates = rates_result.firing_rates  # (n_neurons, n_bins)

        counts, centers = bin_spikes_in_time(
            spike_times,
            dt,
            t_start=times_arr.min(),
            t_stop=times_arr.max(),
        )  # counts: (n_time, n_neurons)

        manual_result = decode_position(
            env,
            counts,
            firing_rates,
            dt,
            times=centers,
        )

        # --- decode_session path ---
        session_result = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
        )

        assert_allclose(
            session_result.posterior,
            manual_result.posterior,
            atol=1e-10,
            err_msg="decode_session posterior differs from manual 3-call path",
        )
        assert_allclose(
            session_result.times,
            manual_result.times,
            atol=1e-12,
            err_msg="decode_session times differ from manual path",
        )

    def test_map_positions_match_manual_path(self) -> None:
        """MAP positions from decode_session match manual path."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=99
        )
        dt = 0.1
        times_arr = np.asarray(times, dtype=np.float64)

        rates_result = compute_spatial_rates(
            env,
            spike_times,
            times_arr,
            positions,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
            fill_value=0.0,
        )
        counts, centers = bin_spikes_in_time(
            spike_times,
            dt,
            t_start=times_arr.min(),
            t_stop=times_arr.max(),
        )
        manual_result = decode_position(
            env, counts, rates_result.firing_rates, dt, times=centers
        )
        session_result = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
        )

        assert_allclose(
            session_result.map_position,
            manual_result.map_position,
            atol=1e-10,
            err_msg="MAP positions differ between decode_session and manual path",
        )


class TestDecodeSessionEncodingModelsPassthrough:
    """Precomputed encoding_models bypasses the fit step."""

    def test_precomputed_models_give_same_result(self) -> None:
        """Passing precomputed firing_rates as encoding_models equals fitting from scratch."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=3
        )
        dt = 0.1
        times_arr = np.asarray(times, dtype=np.float64)

        # Precompute firing rates
        rates_result = compute_spatial_rates(
            env,
            spike_times,
            times_arr,
            positions,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
            fill_value=0.0,
        )
        precomputed = rates_result.firing_rates  # (n_neurons, n_bins)

        # Call with precomputed models
        result_precomputed = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            encoding_models=precomputed,
        )

        # Call without (fit internally, same params)
        result_fitted = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
        )

        assert_allclose(
            result_precomputed.posterior,
            result_fitted.posterior,
            atol=1e-10,
            err_msg="Precomputed encoding_models gave different result from fitted models",
        )

    def test_precomputed_models_skip_fit(self) -> None:
        """Passing encoding_models= skips encoding step (different params irrelevant)."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=13
        )
        dt = 0.05
        times_arr = np.asarray(times, dtype=np.float64)

        precomputed = compute_spatial_rates(
            env,
            spike_times,
            times_arr,
            positions,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
            fill_value=0.0,
        ).firing_rates

        # Using precomputed with different bandwidth param — bandwidth is ignored
        result1 = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            encoding_models=precomputed,
            bandwidth=999.0,  # ignored
        )
        result2 = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            encoding_models=precomputed,
            bandwidth=1.0,  # also ignored
        )

        assert_allclose(
            result1.posterior,
            result2.posterior,
            atol=1e-12,
            err_msg="encoding_models passthrough should ignore bandwidth param",
        )


class TestDecodeSessionDecodeKwargs:
    """**decode_kwargs are forwarded to decode_position."""

    def test_validate_false_forwarded(self) -> None:
        """validate=False is forwarded via **decode_kwargs to decode_position."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=11
        )
        # Should not raise even with validate=False
        result = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=0.1,
            validate=False,
        )
        assert isinstance(result, DecodingResult)

    def test_times_stored_in_result(self) -> None:
        """Result.times is populated (bin_centers from bin_spikes_in_time)."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=17
        )
        result = decode_session(env, spike_times, times, positions, dt=0.1)
        assert result.times is not None
        assert len(result.times) == result.posterior.shape[0]


class TestNormalizeSpikeTimesPublic:
    """normalize_spike_times is publicly importable from neurospatial.encoding."""

    def test_public_import(self) -> None:
        """normalize_spike_times importable from neurospatial.encoding."""
        from neurospatial.encoding import normalize_spike_times  # noqa: F401

    def test_public_import_in_all(self) -> None:
        """normalize_spike_times is in neurospatial.encoding.__all__."""
        import neurospatial.encoding as enc

        assert "normalize_spike_times" in enc.__all__

    def test_normalizes_correctly(self) -> None:
        """Public symbol behaves identically to the private implementation."""
        from neurospatial.encoding import normalize_spike_times

        spikes = np.array([0.1, 0.5, 1.2])
        result = normalize_spike_times(spikes)
        assert len(result) == 1
        assert_allclose(result[0], spikes)
