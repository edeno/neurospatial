"""``method="glm"`` flows through the decoder (both entry points).

Covers the functional path (:func:`decode_session` / :func:`decode_session_summary`)
and the :class:`BayesianDecoder` class: each accepts and forwards the glm params
(``penalty`` / ``rank``), makes ``bandwidth`` / ``min_occupancy`` nullable, and
validates with the SAME validator the encoder uses (so the error messages mirror
``compute_spatial_rate``).
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment


def _grid_open_field() -> Environment:
    """8x8 open-field grid (single component, uniform bins)."""
    edges = np.linspace(0.0, 16.0, 9)
    return Environment.from_grid_mask(
        active_mask=np.ones((8, 8), dtype=bool),
        grid_edges=(edges, edges),
        connect_diagonal_neighbors=True,
    )


def _tiled_session(env: Environment, *, n_units: int = 4, seed: int = 0):
    """A session tiling active-bin centers so occupancy is positive everywhere."""
    rng = np.random.default_rng(seed)
    bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
    positions = np.tile(bin_centers, (10, 1))
    positions = positions[rng.permutation(positions.shape[0])]
    times = np.arange(positions.shape[0], dtype=np.float64) * 0.05

    centers = bin_centers[rng.choice(env.n_bins, size=n_units, replace=False)]
    spike_times: list[np.ndarray] = []
    for center in centers:
        dist2 = np.sum((positions - center) ** 2, axis=1)
        rate = 25.0 * np.exp(-dist2 / (2.0 * 4.0**2))
        counts = rng.poisson(rate * 0.05)
        per_unit = [
            times[i] + rng.uniform(0.0, 0.05, size=int(c))
            for i, c in enumerate(counts)
            if c
        ]
        spike_times.append(
            np.sort(np.concatenate(per_unit)) if per_unit else np.array([], float)
        )
    return env, spike_times, times, positions


class TestDecodeSessionGlm:
    def test_decode_session_summary_glm_matches_full(self) -> None:
        """decode_session_summary(method="glm") shares decode_session's encoding.

        The streamed summary and the full path build the same glm encoding model
        (both route through _fit_encoding_model), so their MAP estimates agree.
        """
        from neurospatial.decoding import decode_session, decode_session_summary

        env, spike_times, times, positions = _tiled_session(_grid_open_field())

        full = decode_session(
            env, spike_times, times, positions, dt=0.1, method="glm", penalty=None
        )
        summary = decode_session_summary(
            env, spike_times, times, positions, dt=0.1, method="glm", penalty=None
        )
        # Same encoding model => identical MAP positions.
        np.testing.assert_array_equal(summary.map_position, full.map_position)

    def test_decode_session_glm_rank_changes_the_maps(self) -> None:
        """rank reaches the encoder: a 2-function basis gives other posteriors."""
        from neurospatial.decoding import decode_session

        env, spike_times, times, positions = _tiled_session(_grid_open_field())
        low, high = (
            decode_session(
                env, spike_times, times, positions, dt=0.1, method="glm", rank=r
            )
            for r in (2, 8)
        )
        assert not np.allclose(low.posterior, high.posterior, atol=1e-3)


class TestBayesianDecoderGlm:
    @pytest.fixture(scope="class")
    def running_session(self):
        """200 s of running in a 40 cm box with 16 place cells."""
        from neurospatial.simulation import (
            PlaceCellModel,
            generate_poisson_spikes,
            simulate_trajectory_ou,
        )

        edges = np.linspace(0.0, 40.0, 11)
        env = Environment.from_grid_mask(
            np.ones((10, 10), dtype=bool), grid_edges=(edges, edges), units="cm"
        )
        positions, times = simulate_trajectory_ou(
            env, duration=200.0, seed=0, speed_units="cm"
        )
        centers = np.random.default_rng(5).uniform(4.0, 36.0, (16, 2))
        spikes = [
            generate_poisson_spikes(
                PlaceCellModel(
                    env, center=c, width=8.0, max_rate=25.0, seed=i
                ).firing_rate(times, positions),
                times,
                seed=50 + i,
            )
            for i, c in enumerate(centers)
        ]
        half = times[len(times) // 2]
        return env, spikes, times, positions, (times[0], half), (half, times[-1] + 1)

    @pytest.mark.parametrize("rank", [2, 8])
    def test_rank_is_recorded(self, running_session, rank) -> None:
        from neurospatial.decoding import BayesianDecoder

        env, spikes, times, positions, train, _ = running_session
        fitted = BayesianDecoder(env, dt=0.25, method="glm", rank=rank).fit(
            spikes, times, positions, epochs=train
        )
        assert fitted.analysis_settings["rank"] == rank

    def test_rank_changes_the_posterior(self, running_session) -> None:
        from neurospatial.decoding import BayesianDecoder

        env, spikes, times, positions, train, test = running_session
        low, high = (
            BayesianDecoder(env, dt=0.25, method="glm", rank=r)
            .fit(spikes, times, positions, epochs=train)
            .predict(spikes, times, epochs=test)
            for r in (2, 8)
        )
        assert np.abs(low.posterior - high.posterior).max() > 0.1

    def test_held_out_error_is_under_two_bins(self, running_session) -> None:
        from neurospatial.decoding import BayesianDecoder

        env, spikes, times, positions, train, test = running_session
        fitted = BayesianDecoder(env, dt=0.25, method="glm", rank=8).fit(
            spikes, times, positions, epochs=train
        )
        assert fitted.bandwidth is None
        assert fitted.min_occupancy is None
        error = fitted.score(
            spikes, times, positions, epochs=test, distance="euclidean"
        )
        assert error < 2 * 4.0, error

    def test_bayesian_decoder_glm_default_rank(self) -> None:
        """method="glm" with default (None) rank/penalty is constructible + fits."""
        from neurospatial.decoding import BayesianDecoder

        env, spike_times, times, positions = _tiled_session(_grid_open_field())
        decoder = BayesianDecoder(env, dt=0.1, method="glm")
        assert decoder.penalty is None
        assert decoder.rank is None
        fitted = decoder.fit(spike_times, times, positions)
        assert fitted.is_fitted

    def test_bayesian_decoder_glm_rejects_bandwidth(self) -> None:
        """bandwidth=5.0 + method="glm" is rejected at construction."""
        from neurospatial.decoding import BayesianDecoder

        env = _grid_open_field()
        with pytest.raises(ValueError, match="bandwidth"):
            BayesianDecoder(env, method="glm", bandwidth=5.0)

    def test_bayesian_decoder_ratio_penalty_rejected(self) -> None:
        """penalty + a ratio method is rejected at construction (mirror)."""
        from neurospatial.decoding import BayesianDecoder

        env = _grid_open_field()
        with pytest.raises(ValueError, match="penalty"):
            BayesianDecoder(env, method="diffusion_kde", penalty=0.5)


class TestDecoderValidationMirrorsEncoder:
    """The decoder's param errors match ``compute_spatial_rate``'s exactly."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"method": "glm", "bandwidth": 5.0},
            {"method": "glm", "min_occupancy": 0.1},
            {"method": "diffusion_kde", "penalty": 0.5},
            {"method": "diffusion_kde", "rank": 10},
            {"method": "glm", "penalty": -1.0},
            {"method": "glm", "penalty": True},
            {"method": "glm", "rank": 0},
        ],
    )
    def test_message_matches(self, kwargs) -> None:
        from neurospatial.decoding import BayesianDecoder
        from neurospatial.encoding.spatial import compute_spatial_rate

        env, spike_times, times, positions = _tiled_session(_grid_open_field())

        with pytest.raises(ValueError) as enc_exc:
            compute_spatial_rate(env, spike_times[0], times, positions, **kwargs)
        with pytest.raises(ValueError) as dec_exc:
            BayesianDecoder(env, **kwargs)

        assert str(dec_exc.value) == str(enc_exc.value)
