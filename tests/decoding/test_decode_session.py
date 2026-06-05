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
import pytest
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
        """decode_session is importable from neurospatial.decoding (canonical path)."""
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
        with pytest.raises(ValueError, match="At least 2 samples required"):
            decode_session(env, spike_times, np.array([]), np.zeros((0, 2)), dt=0.1)

    def test_nonfinite_times_raise_clear_error_in_passthrough(self) -> None:
        """NaN/inf `times` raise a beginner-grade error (not a raw int-conversion).

        The encoding_models passthrough branch skips the encoder's own
        validation, so without an up-front check a NaN/inf in `times` leaked a
        raw 'cannot convert float NaN to integer' from bin_spikes_in_time.
        """
        import pytest

        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=10.0
        )
        models = compute_spatial_rates(
            env, spike_times, times, positions, fill_value=0.0
        ).firing_rates

        for bad_value in (np.nan, np.inf):
            bad_times = times.copy()
            bad_times[3] = bad_value
            with pytest.raises(ValueError, match="finite"):
                decode_session(
                    env,
                    spike_times,
                    bad_times,
                    positions,
                    dt=0.1,
                    encoding_models=models,
                )

    def test_non_1d_times_raise_clear_error(self) -> None:
        """A 2-D `times` array raises a clear shape error, not a cryptic one."""
        import pytest

        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=10.0
        )
        models = compute_spatial_rates(
            env, spike_times, times, positions, fill_value=0.0
        ).firing_rates

        with pytest.raises(ValueError, match="1-D"):
            decode_session(
                env,
                spike_times,
                times.reshape(-1, 1),
                positions,
                dt=0.1,
                encoding_models=models,
            )

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


class TestDecodeSessionOutOfWindowWarning:
    """decode_session warns loudly when spikes fall outside the decode window.

    Guards the units footgun (spike_times in ms while times is in seconds):
    spikes ~1000x the trajectory window all fall outside [t_start, t_stop],
    histogram silently drops them, and the posterior is plausible-but-wrong.
    decode_session must surface this with exactly one UserWarning that covers
    BOTH the encoding_models-provided branch and the None branch.
    """

    # Phrase shared across both encoding paths' warning text.
    _MATCH = r"fell outside the decode time window"

    def test_warns_in_encoding_models_branch(self) -> None:
        """encoding_models= branch (skips compute_spatial_rates) still warns."""
        import pytest

        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=5
        )
        times_arr = np.asarray(times, dtype=np.float64)

        # Build models on the in-window (seconds) data first.
        models = compute_spatial_rates(
            env,
            spike_times,
            times_arr,
            positions,
            bandwidth=5.0,
            smoothing_method="diffusion_kde",
            min_occupancy=0.0,
            fill_value=0.0,
        ).firing_rates

        # Units footgun: spike times in ms (~1000x), so (nearly) all fall
        # outside the seconds-scale decode window.
        ms_spikes = [s * 1000.0 for s in spike_times]

        with pytest.warns(UserWarning, match=self._MATCH):
            decode_session(
                env,
                ms_spikes,
                times,
                positions,
                dt=0.1,
                encoding_models=models,
            )

    def test_single_warning_in_non_passthrough_branch(self) -> None:
        """encoding_models=None branch emits exactly ONE out-of-window warning.

        In this branch the encoder (compute_spatial_rates) owns the
        time-window drop warning; decode_session must NOT add a second one, so
        exactly one out-of-window warning is emitted for the whole golden path.
        """
        import warnings

        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=8
        )
        ms_spikes = [s * 1000.0 for s in spike_times]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decode_session(env, ms_spikes, times, positions, dt=0.1)

        # Either phrasing ("position time window" from the encoder, or "decode
        # time window" from decode_session) — there must be exactly one.
        matching = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and "fell outside the" in str(w.message)
            and "time window" in str(w.message)
        ]
        assert len(matching) == 1, (
            f"Expected exactly one out-of-window warning, got {len(matching)}: "
            f"{[str(w.message) for w in matching]}"
        )

    def test_in_window_is_silent(self) -> None:
        """A normal in-window call raises NO drop warning."""
        import warnings

        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=21
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decode_session(env, spike_times, times, positions, dt=0.1)

        matching = [w for w in caught if "fell outside the" in str(w.message)]
        assert not matching, (
            f"In-window decode should not warn, got: "
            f"{[str(w.message) for w in matching]}"
        )

    def test_inactive_bin_warning_survives_golden_path(self) -> None:
        """The encoder's inactive-bin warning reaches the user via decode_session.

        Spikes whose interpolated positions fall outside the environment (e.g.
        positions in the wrong coordinate frame) are an independent footgun from
        the time-window units mismatch. The encoder warns about it; decode_session
        must not swallow that warning in the encoding_models=None branch.
        """
        import warnings

        from neurospatial.decoding import decode_session

        env, _spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=55
        )
        # In-window spikes (no time-window drop). The interval STARTS stay
        # in-bounds (so the interval-valid mask keeps them), but every other
        # sample jumps far outside the [0, 100] environment. We place every
        # spike at the MIDPOINT of a valid (even-indexed) interval, so each
        # spike interpolates toward the far excursion and maps to an inactive
        # bin. This exercises the inactive-bin drop path (distinct from the
        # interval mask, which gates by the start sample).
        positions_wrong_frame = positions.copy()
        positions_wrong_frame[1::2] = 1000.0
        n_frames = len(times)
        # Midpoints of intervals 0, 2, 4, ... (those starting at in-bounds
        # samples); each interpolates to ~500 → out of the environment.
        even_starts = np.arange(0, n_frames - 1, 2)
        midpoints = 0.5 * (times[even_starts] + times[even_starts + 1])
        oob_spikes = [midpoints.copy() for _ in range(10)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decode_session(env, oob_spikes, times, positions_wrong_frame, dt=0.1)

        inactive = [
            w for w in caught if "interpolated to positions outside" in str(w.message)
        ]
        assert inactive, (
            "decode_session should surface the encoder's inactive-bin warning, "
            f"got: {[str(w.message) for w in caught]}"
        )

    def test_warn_on_drop_false_silences(self) -> None:
        """warn_on_drop=False silences BOTH decode_session and the encoder."""
        import warnings

        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=34
        )
        ms_spikes = [s * 1000.0 for s in spike_times]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decode_session(env, ms_spikes, times, positions, dt=0.1, warn_on_drop=False)

        matching = [
            w
            for w in caught
            if "fell outside" in str(w.message)
            or "fell outside the decode time window" in str(w.message)
        ]
        assert not matching, (
            f"warn_on_drop=False should silence all drop warnings, got: "
            f"{[str(w.message) for w in matching]}"
        )


class TestAsSpikeTrainsPublic:
    """as_spike_trains is publicly importable from neurospatial.encoding."""

    def test_public_import(self) -> None:
        """as_spike_trains importable from neurospatial.encoding."""
        from neurospatial.encoding import as_spike_trains  # noqa: F401

    def test_public_import_in_all(self) -> None:
        """as_spike_trains is in neurospatial.encoding.__all__."""
        import neurospatial.encoding as enc

        assert "as_spike_trains" in enc.__all__

    def test_normalizes_correctly(self) -> None:
        """Public symbol behaves identically to the private implementation."""
        from neurospatial.encoding import as_spike_trains

        spikes = np.array([0.1, 0.5, 1.2])
        result = as_spike_trains(spikes)
        assert len(result) == 1
        assert_allclose(result[0], spikes)


class TestDecodeSessionDtype:
    """The `dtype` knob honors float32 end-to-end (R8).

    "Decode in this dtype": a single ``dtype=np.float32`` controls BOTH the
    encoding-model working set that reaches ``decode_position`` AND the
    posterior dtype, on both the computed and the precomputed-``encoding_models``
    branches. Default ``np.float64`` is byte-for-byte unchanged.
    """

    def _spy_decode_position(self, monkeypatch):
        """Wrap decode_position to capture the encoding-model dtype it receives.

        Returns the ``seen`` dict; ``seen["models_dtype"]`` is the dtype of the
        firing-rate model array passed into the real ``decode_position`` (i.e.
        the working set), and the real function still runs so the decode
        completes.
        """
        import neurospatial.decoding.session as session_mod
        from neurospatial.decoding import posterior as posterior_mod

        seen: dict = {}
        real = posterior_mod.decode_position

        def _spy(env, counts, firing_rates, dt, **kw):
            seen["models_dtype"] = np.asarray(firing_rates).dtype
            seen["posterior_dtype_kw"] = kw.get("dtype", "MISSING")
            return real(env, counts, firing_rates, dt, **kw)

        # decode_session imports decode_position locally from
        # neurospatial.decoding.posterior, so patch it there.
        monkeypatch.setattr(posterior_mod, "decode_position", _spy)
        # Guard against decode_session resolving the name another way.
        if hasattr(session_mod, "decode_position"):
            monkeypatch.setattr(session_mod, "decode_position", _spy, raising=False)
        return seen

    def test_computed_branch_float32_reaches_working_set(self, monkeypatch) -> None:
        """dtype=np.float32 (computed branch) → decode_position gets float32 models."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=8.0, seed=3
        )
        seen = self._spy_decode_position(monkeypatch)

        decode_session(env, spike_times, times, positions, dt=0.1, dtype=np.float32)

        assert seen["models_dtype"] == np.float32
        assert seen["posterior_dtype_kw"] is np.float32

    def test_computed_branch_default_is_float64(self, monkeypatch) -> None:
        """Default dtype (computed branch) → decode_position gets float64 models."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=8.0, seed=3
        )
        seen = self._spy_decode_position(monkeypatch)

        decode_session(env, spike_times, times, positions, dt=0.1)

        assert seen["models_dtype"] == np.float64
        assert seen["posterior_dtype_kw"] is np.float64

    def test_passthrough_branch_float32_reaches_working_set(self, monkeypatch) -> None:
        """float64 encoding_models + dtype=float32 → decode_position gets float32."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=8.0, seed=5
        )
        times_arr = np.asarray(times, dtype=np.float64)
        models64 = compute_spatial_rates(
            env,
            spike_times,
            times_arr,
            positions,
            bandwidth=5.0,
            fill_value=0.0,
            dtype=np.float64,
        ).firing_rates
        assert models64.dtype == np.float64

        seen = self._spy_decode_position(monkeypatch)
        decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=0.1,
            encoding_models=models64,
            dtype=np.float32,
        )

        # dtype is authoritative end-to-end: float64-in is cast down to float32.
        assert seen["models_dtype"] == np.float32
        assert seen["posterior_dtype_kw"] is np.float32

    def test_posterior_dtype_follows_knob(self) -> None:
        """decode_session(dtype=...) sets the posterior dtype; default float64."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=8.0, seed=9
        )

        res32 = decode_session(
            env, spike_times, times, positions, dt=0.1, dtype=np.float32
        )
        res64 = decode_session(env, spike_times, times, positions, dt=0.1)

        assert res32.posterior.dtype == np.float32
        assert res64.posterior.dtype == np.float64

    def test_parity_within_tol(self) -> None:
        """float32 decode map_position matches float64 within tolerance."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=7
        )
        res32 = decode_session(
            env, spike_times, times, positions, dt=0.1, dtype=np.float32
        )
        res64 = decode_session(env, spike_times, times, positions, dt=0.1)

        assert_allclose(
            res32.map_position,
            res64.map_position,
            rtol=1e-4,
            atol=1e-4,
            err_msg="float32 decode_session decoded too far from float64",
        )

    def test_default_byte_for_byte_unchanged(self) -> None:
        """No dtype == explicit float64 == byte-for-byte identical posterior."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=7
        )
        res_default = decode_session(env, spike_times, times, positions, dt=0.1)
        res_f64 = decode_session(
            env, spike_times, times, positions, dt=0.1, dtype=np.float64
        )
        # NaN-aware exact equality.
        np.testing.assert_array_equal(res_default.posterior, res_f64.posterior)

    def test_invalid_dtype_raises(self) -> None:
        """A non-float32/64 dtype raises ValueError naming `dtype`."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=1
        )
        with pytest.raises(
            ValueError, match="dtype must be np\\.float32 or np\\.float64"
        ):
            decode_session(env, spike_times, times, positions, dt=0.1, dtype=np.int32)

    def test_unparseable_dtype_raises_clean_valueerror(self) -> None:
        """An unparseable dtype string raises ValueError naming `dtype`.

        Without the wrapped ``np.dtype(dtype)`` parse, ``dtype="bogus"`` leaked a
        raw ``TypeError: data type 'bogus' not understood`` from NumPy.
        """
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=1
        )
        with pytest.raises(ValueError, match="dtype"):
            decode_session(env, spike_times, times, positions, dt=0.1, dtype="bogus")  # type: ignore[arg-type]

    def test_precomputed_models_decode_within_tol(self) -> None:
        """Precomputed float32 vs float64 encoding_models decode within tol."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=7
        )
        dt = 0.1
        times_arr = np.asarray(times, dtype=np.float64)

        kwargs = {
            "bandwidth": 5.0,
            "smoothing_method": "diffusion_kde",
            "min_occupancy": 0.0,
            "fill_value": 0.0,
        }
        models32 = compute_spatial_rates(
            env, spike_times, times_arr, positions, dtype=np.float32, **kwargs
        ).firing_rates
        models64 = compute_spatial_rates(
            env, spike_times, times_arr, positions, dtype=np.float64, **kwargs
        ).firing_rates

        assert models32.dtype == np.float32
        assert models64.dtype == np.float64

        result32 = decode_session(
            env, spike_times, times, positions, dt=dt, encoding_models=models32
        )
        result64 = decode_session(
            env, spike_times, times, positions, dt=dt, encoding_models=models64
        )

        assert_allclose(
            result32.map_position,
            result64.map_position,
            rtol=1e-4,
            atol=1e-4,
            err_msg="float32 encoding models decoded too far from float64",
        )


# ---------------------------------------------------------------------------
# max_gap passthrough (R1 follow-up)
# ---------------------------------------------------------------------------


class TestDecodeSessionMaxGap:
    """decode_session / decode_session_summary forward max_gap to the encoder."""

    def _gap_session(self):
        """1D track with one >0.5 s tracking gap and a spike inside that gap.

        Returns (env, spike_times, times, positions, t_gap). The lone gap
        interval is dropped by default (max_gap=0.5) but kept under
        max_gap=None, so the encoding firing-rate maps differ between the two.
        """
        env = Environment.from_samples(
            np.linspace(0, 100, 101).reshape(-1, 1), bin_size=10.0
        )
        dt = 0.1
        n_a, n_b = 30, 30
        xa = 10.0 + np.arange(n_a) * 0.1
        xb = 13.0 + np.arange(n_b) * 0.1
        positions = np.concatenate([xa, xb]).reshape(-1, 1)
        ta = np.arange(n_a) * dt
        tb = ta[-1] + 1.0 + np.arange(n_b) * dt  # 1.0 s gap
        times = np.concatenate([ta, tb])
        gap_interval = n_a - 1
        t_gap = 0.5 * (times[gap_interval] + times[gap_interval + 1])
        # Two neurons; neuron 0 fires inside the gap, plus some valid spikes.
        spike_times = [
            np.array([0.5, 1.2, t_gap, 4.5]),
            np.array([1.0, 2.0, 5.0]),
        ]
        return env, spike_times, times, positions, t_gap

    def test_max_gap_forwarded_to_encoder(self, monkeypatch) -> None:
        """The max_gap kwarg reaches compute_spatial_rates verbatim."""
        import neurospatial.encoding.spatial as spatial_mod
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions, _ = self._gap_session()

        seen = {}
        real = spatial_mod.compute_spatial_rates

        def _spy(*args, **kwargs):
            seen["max_gap"] = kwargs.get("max_gap", "MISSING")
            return real(*args, **kwargs)

        monkeypatch.setattr(spatial_mod, "compute_spatial_rates", _spy)

        decode_session(env, spike_times, times, positions, dt=0.1, max_gap=None)
        assert seen["max_gap"] is None

        decode_session(env, spike_times, times, positions, dt=0.1)
        assert seen["max_gap"] == 0.5

    def test_max_gap_none_changes_encoding(self) -> None:
        """max_gap=None keeps the gap spike, changing the posterior vs default."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions, _ = self._gap_session()

        res_default = decode_session(
            env, spike_times, times, positions, dt=0.1, warn_on_drop=False
        )
        res_no_gap = decode_session(
            env,
            spike_times,
            times,
            positions,
            dt=0.1,
            max_gap=None,
            warn_on_drop=False,
        )
        # Dropping vs keeping the in-gap spike changes the encoding model and
        # therefore the posterior.
        assert not np.allclose(res_default.posterior, res_no_gap.posterior)

    def test_summary_accepts_max_gap(self) -> None:
        """decode_session_summary also accepts and forwards max_gap."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions, _ = self._gap_session()
        summary = decode_session_summary(
            env,
            spike_times,
            times,
            positions,
            dt=0.1,
            max_gap=None,
            warn_on_drop=False,
        )
        assert summary.map_position.shape[1] == env.n_dims


# ---------------------------------------------------------------------------
# R5: decode_session_summary streams the time-binning (no full count matrix)
# ---------------------------------------------------------------------------


def _summary_reference(env, spike_times, times, positions, *, dt, time_chunk, **kw):
    """PRE-R5 reference: materialize the FULL count matrix, then stream-decode.

    This is exactly what decode_session_summary did before R5 (encode -> bin
    the WHOLE session into a dense (n_time, n_neurons) count matrix ->
    decode_position_summary). R5 streams the binning, so the new
    decode_session_summary must match this byte-for-byte.
    """
    from neurospatial.decoding import (
        bin_spikes_in_time,
        decode_position_summary,
    )
    from neurospatial.encoding import compute_spatial_rates

    times_arr = np.asarray(times, dtype=np.float64)
    firing_rates = compute_spatial_rates(
        env,
        spike_times,
        times_arr,
        positions,
        bandwidth=5.0,
        smoothing_method="diffusion_kde",
        min_occupancy=0.0,
        fill_value=0.0,
    ).firing_rates
    counts, centers = bin_spikes_in_time(
        spike_times, dt, t_start=times_arr.min(), t_stop=times_arr.max()
    )
    return decode_position_summary(
        env,
        counts,
        firing_rates,
        dt,
        times=centers,
        time_chunk=time_chunk,
        **kw,
    )


class TestDecodeSessionSummaryStreaming:
    """decode_session_summary streams binning; result == materialize-then-stream."""

    def _assert_summary_equal(self, a, b, *, exact=True) -> None:
        """Assert two DecodingSummary objects agree (byte-for-byte or tight)."""
        np.testing.assert_array_equal(a.map_bin, b.map_bin)
        np.testing.assert_array_equal(a.map_position, b.map_position)
        np.testing.assert_array_equal(a.times, b.times)
        if exact:
            np.testing.assert_array_equal(a.mean_position, b.mean_position)
            np.testing.assert_array_equal(a.posterior_entropy, b.posterior_entropy)
            np.testing.assert_array_equal(a.peak_prob, b.peak_prob)
        else:
            assert_allclose(a.mean_position, b.mean_position, rtol=1e-12, atol=1e-12)
            assert_allclose(
                a.posterior_entropy, b.posterior_entropy, rtol=1e-12, atol=1e-12
            )
            assert_allclose(a.peak_prob, b.peak_prob, rtol=1e-12, atol=1e-12)

    def test_parity_dt_divides_evenly(self) -> None:
        """dt evenly divides the session: streamed == materialize-then-stream."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=12, duration=20.0, seed=3
        )
        # times spans [0, ~20]; dt=0.1 -> grid is a clean multiple.
        dt = 0.1
        ref = _summary_reference(
            env, spike_times, times, positions, dt=dt, time_chunk=64
        )
        got = decode_session_summary(
            env, spike_times, times, positions, dt=dt, time_chunk=64
        )
        self._assert_summary_equal(got, ref)

    def test_parity_dt_not_dividing_evenly(self) -> None:
        """dt that does NOT evenly divide the span: trailing partial bin dropped."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=12, duration=20.0, seed=9
        )
        dt = 0.07  # 20 / 0.07 is not integral
        ref = _summary_reference(
            env, spike_times, times, positions, dt=dt, time_chunk=64
        )
        got = decode_session_summary(
            env, spike_times, times, positions, dt=dt, time_chunk=64
        )
        self._assert_summary_equal(got, ref)

    def test_parity_n_time_not_multiple_of_chunk(self) -> None:
        """n_time not a multiple of time_chunk: final short block handled."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=15.0, seed=15
        )
        dt = 0.1  # ~150 bins; chunk=37 leaves a short final block
        ref = _summary_reference(
            env, spike_times, times, positions, dt=dt, time_chunk=37
        )
        got = decode_session_summary(
            env, spike_times, times, positions, dt=dt, time_chunk=37
        )
        self._assert_summary_equal(got, ref)
        # Sanity: the chunking really did straddle a non-multiple boundary.
        assert got.map_bin.shape[0] % 37 != 0

    def test_time_chunk_none_rejected(self) -> None:
        """R12: time_chunk=None is rejected (it would materialize the posterior)."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=21
        )
        dt = 0.1
        with pytest.raises(ValueError, match=r"full.*posterior|decode_session"):
            decode_session_summary(
                env, spike_times, times, positions, dt=dt, time_chunk=None
            )

    @pytest.mark.parametrize("bad", [0, -1])
    def test_time_chunk_non_positive_rejected(self, bad) -> None:
        """R12: time_chunk < 1 is rejected up front with a clear ValueError."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=21
        )
        dt = 0.1
        with pytest.raises(ValueError, match="time_chunk must be a positive integer"):
            decode_session_summary(
                env, spike_times, times, positions, dt=dt, time_chunk=bad
            )

    def test_default_and_explicit_chunk_agree(self) -> None:
        """R12 regression: default time_chunk and an explicit positive chunk agree."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=21
        )
        dt = 0.1
        default = decode_session_summary(env, spike_times, times, positions, dt=dt)
        chunked = decode_session_summary(
            env, spike_times, times, positions, dt=dt, time_chunk=37
        )
        self._assert_summary_equal(chunked, default)

    def test_parity_with_precomputed_encoding_models(self) -> None:
        """encoding_models passthrough also streams correctly."""
        from neurospatial.decoding import (
            bin_spikes_in_time,
            decode_position_summary,
            decode_session_summary,
        )

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=12.0, seed=27
        )
        dt = 0.1
        times_arr = np.asarray(times, dtype=np.float64)
        models = compute_spatial_rates(
            env, spike_times, times_arr, positions, fill_value=0.0
        ).firing_rates

        counts, centers = bin_spikes_in_time(
            spike_times, dt, t_start=times_arr.min(), t_stop=times_arr.max()
        )
        ref = decode_position_summary(
            env, counts, models, dt, times=centers, time_chunk=50
        )
        got = decode_session_summary(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            encoding_models=models,
            time_chunk=50,
        )
        self._assert_summary_equal(got, ref)

    def test_block_centers_match_global_grid(self) -> None:
        """Each streamed block's centers equal the global bin_centers slice.

        The streaming loop asserts block alignment in-line (a streamed block's
        centers must equal the corresponding slice of the global grid). Running
        decode_session_summary across several non-trivial chunk sizes therefore
        exercises that assertion for every block boundary; the returned
        ``.times`` must equal the global grid exactly. Also independently
        reconstruct the per-block global-edge slices and assert they tile the
        grid with no gap / overlap / off-by-one.
        """
        from neurospatial.decoding import bin_spikes_in_time, decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=12.0, seed=31
        )
        dt = 0.1
        times_arr = np.asarray(times, dtype=np.float64)

        # Global grid (reference) via the public binning primitive.
        _, global_centers = bin_spikes_in_time(
            spike_times, dt, t_start=times_arr.min(), t_stop=times_arr.max()
        )
        n_time = len(global_centers)

        # The in-loop alignment assertions fire for every block boundary across
        # these chunk sizes; if any block drifted, the call would raise.
        for chunk in (1, 7, 40, n_time, n_time + 5):
            summ = decode_session_summary(
                env, spike_times, times, positions, dt=dt, time_chunk=chunk
            )
            # The summary's time grid must equal the global grid exactly.
            assert_allclose(summ.times, global_centers, atol=1e-12)

        # Independent reconstruction: per-block global-edge slices must tile the
        # global grid contiguously with no gap/overlap. Reconstruct the global
        # edges the same way the implementation does.
        t_start = float(times_arr.min())
        edges = t_start + dt * np.arange(n_time + 1, dtype=np.float64)
        chunk = 40
        assembled: list[np.ndarray] = []
        for start in range(0, n_time, chunk):
            stop = min(start + chunk, n_time)
            block_edges = edges[start : stop + 1]
            assembled.append(block_edges[:-1] + dt / 2.0)
        assert_allclose(np.concatenate(assembled), global_centers, atol=1e-12)

    def test_boundary_spike_counted_once(self) -> None:
        """A spike exactly on a block boundary is counted exactly once.

        Place a single spike exactly on an interior global edge that also
        happens to be a block boundary. The streamed result must match the
        full-materialization reference (where one histogram call assigns that
        spike to exactly one bin). bin_spikes_in_time right-closes its last bin,
        so without per-block edge-exclusion the spike would be double-counted.
        """
        from neurospatial.decoding import decode_session_summary

        env = Environment.from_samples(
            np.linspace(0.0, 100.0, 201).reshape(-1, 1), bin_size=5.0
        )
        env.units = "cm"
        dt = 0.1
        t0, t1 = 0.0, 10.0
        n_frames = 200
        times = np.linspace(t0, t1, n_frames)
        positions = np.linspace(0.0, 100.0, n_frames).reshape(-1, 1)

        time_chunk = 40
        # Block boundary 0 ends at global edge index 40 -> time = 40*dt = 4.0,
        # an interior edge AND a block boundary.
        boundary_time = time_chunk * dt  # 4.0
        # One neuron with a spike exactly on that boundary (plus a couple of
        # ordinary in-window spikes so the encoding model is non-degenerate).
        spike_times = [np.array([1.05, boundary_time, 7.25])]

        ref = _summary_reference(
            env, spike_times, times, positions, dt=dt, time_chunk=time_chunk
        )
        got = decode_session_summary(
            env,
            spike_times,
            times,
            positions,
            dt=dt,
            time_chunk=time_chunk,
        )
        # If the boundary spike were double-counted, the count matrix (hence the
        # posterior, MAP, entropy) at the boundary bins would diverge.
        self._assert_summary_equal(got, ref)

    def test_streaming_2d_prior_parity(self) -> None:
        """A 2-D time-varying prior streamed == full-materialization reference."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=12.0, seed=44
        )
        dt = 0.1
        # Build the reference first to learn n_time, then build a matching prior.
        ref_no_prior = _summary_reference(
            env, spike_times, times, positions, dt=dt, time_chunk=64
        )
        n_time = ref_no_prior.map_bin.shape[0]
        rng = np.random.default_rng(2)
        prior = rng.uniform(0.5, 2.0, (n_time, env.n_bins))

        ref = _summary_reference(
            env, spike_times, times, positions, dt=dt, time_chunk=64, prior=prior
        )
        got = decode_session_summary(
            env, spike_times, times, positions, dt=dt, time_chunk=64, prior=prior
        )
        self._assert_summary_equal(got, ref)

    def test_streaming_overlong_2d_prior_raises(self) -> None:
        """An over-long 2-D prior raises up front (inherits R2), not truncates."""
        import pytest

        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=10.0, seed=51
        )
        dt = 0.1
        bad_prior = np.ones((100000, env.n_bins))  # clearly over-long
        with pytest.raises(ValueError, match="prior"):
            decode_session_summary(
                env, spike_times, times, positions, dt=dt, prior=bad_prior
            )

    def test_unknown_kwarg_raises(self) -> None:
        """An unknown decode kwarg is not silently dropped."""
        import pytest

        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=6.0, seed=61
        )
        with pytest.raises(TypeError, match="unexpected keyword"):
            decode_session_summary(
                env, spike_times, times, positions, dt=0.1, not_a_real_kwarg=1
            )


class TestDecodeSessionSummaryDtype:
    """decode_session_summary honors the dtype knob (working set + parity, R8)."""

    def _spy_reduce_block(self, monkeypatch):
        """Wrap _decode_and_reduce_block to capture firing-rates + dtype it gets.

        Records the dtype of the per-block working set: ``seen["models_dtype"]``
        is the firing-rate model dtype reaching the block decode, and
        ``seen["dtype"]`` is the working dtype forwarded to it.
        """
        from neurospatial.decoding import posterior as posterior_mod

        seen: dict = {}
        real = posterior_mod._decode_and_reduce_block

        def _spy(counts_block, firing_rates, dt, bin_centers, **kw):
            seen.setdefault("models_dtype", np.asarray(firing_rates).dtype)
            seen.setdefault("dtype", kw.get("dtype", "MISSING"))
            return real(counts_block, firing_rates, dt, bin_centers, **kw)

        monkeypatch.setattr(posterior_mod, "_decode_and_reduce_block", _spy)
        return seen

    def test_float32_reaches_block_working_set(self, monkeypatch) -> None:
        """dtype=np.float32 → the streamed block decode gets float32 models + dtype."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=8.0, seed=4
        )
        seen = self._spy_reduce_block(monkeypatch)

        decode_session_summary(
            env, spike_times, times, positions, dt=0.1, dtype=np.float32
        )

        assert seen["models_dtype"] == np.float32
        assert seen["dtype"] is np.float32

    def test_default_block_working_set_is_float64(self, monkeypatch) -> None:
        """Default dtype → the streamed block decode gets float64 models + dtype."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=8, duration=8.0, seed=4
        )
        seen = self._spy_reduce_block(monkeypatch)

        decode_session_summary(env, spike_times, times, positions, dt=0.1)

        assert seen["models_dtype"] == np.float64
        assert seen["dtype"] is np.float64

    def test_invalid_dtype_raises(self) -> None:
        """A non-float32/64 dtype raises ValueError naming `dtype`."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=2
        )
        with pytest.raises(
            ValueError, match="dtype must be np\\.float32 or np\\.float64"
        ):
            decode_session_summary(
                env, spike_times, times, positions, dt=0.1, dtype=np.int32
            )

    def test_unparseable_dtype_raises_clean_valueerror(self) -> None:
        """An unparseable dtype string raises ValueError naming `dtype`."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=2
        )
        with pytest.raises(ValueError, match="dtype"):
            decode_session_summary(
                env, spike_times, times, positions, dt=0.1, dtype="bogus"
            )  # type: ignore[arg-type]

    def test_dtype_via_decode_kwargs_now_collides(self) -> None:
        """dtype is an explicit param, no longer a decode_kwargs entry.

        Passing dtype both ways is a duplicate-keyword TypeError at call time;
        passing it once via the explicit param is the supported route.
        """
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=5, duration=5.0, seed=2
        )
        # Single explicit dtype works (no longer routed through **decode_kwargs).
        summary = decode_session_summary(
            env, spike_times, times, positions, dt=0.1, dtype=np.float32
        )
        assert summary.map_position.shape[1] == env.n_dims

    def test_parity_within_tol(self) -> None:
        """float32 summary MAP matches float64 summary within tolerance."""
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim(
            n_neurons=10, duration=10.0, seed=7
        )
        summ32 = decode_session_summary(
            env, spike_times, times, positions, dt=0.1, dtype=np.float32
        )
        summ64 = decode_session_summary(
            env, spike_times, times, positions, dt=0.1, dtype=np.float64
        )
        assert_allclose(
            summ32.map_position,
            summ64.map_position,
            rtol=1e-4,
            atol=1e-4,
            err_msg="float32 decode_session_summary decoded too far from float64",
        )


class TestDecodeSessionSummaryTimeChunkValidation:
    """decode_session_summary validates time_chunk as a positive int (Fix B)."""

    @pytest.mark.parametrize("bad", [1.5, "2", True, False, 0, -1])
    def test_rejects_bad_time_chunk(self, bad: object) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match="time_chunk"):
            decode_session_summary(
                env,
                spike_times,
                times,
                positions,
                dt=0.025,
                time_chunk=bad,
            )

    def test_none_time_chunk_specific_message(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match="decode_session"):
            decode_session_summary(
                env,
                spike_times,
                times,
                positions,
                dt=0.025,
                time_chunk=None,
            )

    def test_accepts_valid_int_time_chunk(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        summ = decode_session_summary(
            env,
            spike_times,
            times,
            positions,
            dt=0.025,
            time_chunk=64,
        )
        assert summ.map_position.shape[0] > 0


class TestDecodeSessionDtValidation:
    """decode_session / decode_session_summary validate dt up front.

    Both route through ``_build_encoding_model``, which computes the decode
    time grid directly (bypassing ``bin_spikes_in_time``'s dt guard). Without an
    explicit up-front check, invalid ``dt`` leaks cryptic errors:
    ``dt=0`` → ``ZeroDivisionError``; ``dt=nan`` → "cannot convert float NaN to
    integer"; ``dt<0`` → a MISLEADING "span smaller than one bin" message;
    ``dt=inf`` → a similar cryptic failure. The fix mirrors
    ``bin_spikes_in_time``'s wording: "dt must be finite and > 0, got ...".
    """

    def test_decode_session_dt_zero_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session(env, spike_times, times, positions, dt=0.0)

    def test_decode_session_dt_nan_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session(env, spike_times, times, positions, dt=np.nan)

    def test_decode_session_dt_negative_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session(env, spike_times, times, positions, dt=-0.1)

    def test_decode_session_dt_inf_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session(env, spike_times, times, positions, dt=np.inf)

    def test_decode_session_valid_dt_still_works(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        result = decode_session(env, spike_times, times, positions, dt=0.1)
        assert isinstance(result, DecodingResult)

    def test_decode_session_summary_dt_zero_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session_summary(env, spike_times, times, positions, dt=0.0)

    def test_decode_session_summary_dt_nan_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session_summary(env, spike_times, times, positions, dt=np.nan)

    def test_decode_session_summary_dt_negative_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session_summary(env, spike_times, times, positions, dt=-0.1)

    def test_decode_session_summary_dt_inf_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must be finite and > 0"):
            decode_session_summary(env, spike_times, times, positions, dt=np.inf)

    def test_decode_session_summary_valid_dt_still_works(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        summ = decode_session_summary(env, spike_times, times, positions, dt=0.1)
        assert summ.map_position.shape[0] > 0

    # Non-numeric / bool dt: an isinstance guard runs BEFORE float() coercion so
    # a numeric STRING (e.g. dt="0.1", which float() would silently accept while
    # the caller's dt stays a str and leaks a downstream TypeError) and a bool
    # (dt=True would coerce to 1.0 silently) both raise the clean message early.
    def test_decode_session_dt_numeric_string_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must"):
            decode_session(env, spike_times, times, positions, dt="0.1")  # type: ignore[arg-type]

    def test_decode_session_dt_non_numeric_string_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must"):
            decode_session(env, spike_times, times, positions, dt="abc")  # type: ignore[arg-type]

    def test_decode_session_dt_bool_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must"):
            decode_session(env, spike_times, times, positions, dt=True)  # type: ignore[arg-type]

    def test_decode_session_summary_dt_numeric_string_raises_valueerror(
        self,
    ) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must"):
            decode_session_summary(env, spike_times, times, positions, dt="0.1")  # type: ignore[arg-type]

    def test_decode_session_summary_dt_non_numeric_string_raises_valueerror(
        self,
    ) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must"):
            decode_session_summary(env, spike_times, times, positions, dt="abc")  # type: ignore[arg-type]

    def test_decode_session_summary_dt_bool_raises_valueerror(self) -> None:
        from neurospatial.decoding import decode_session_summary

        env, spike_times, times, positions = _make_linear_track_sim()
        with pytest.raises(ValueError, match=r"dt must"):
            decode_session_summary(env, spike_times, times, positions, dt=True)  # type: ignore[arg-type]
