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
    def test_decode_session_glm(self) -> None:
        """decode_session(method="glm", penalty=None) -> valid DecodingResult."""
        from neurospatial.decoding import DecodingResult, decode_session

        env, spike_times, times, positions = _tiled_session(_grid_open_field())
        result = decode_session(
            env, spike_times, times, positions, dt=0.1, method="glm", penalty=None
        )
        assert isinstance(result, DecodingResult)
        assert result.posterior.shape[1] == env.n_bins
        assert np.all(np.isfinite(result.posterior))
        # Each posterior row is a normalized distribution over bins.
        np.testing.assert_allclose(result.posterior.sum(axis=1), 1.0, atol=1e-6)

    def test_decode_session_summary_glm_matches_full(self) -> None:
        """decode_session_summary(method="glm") shares decode_session's encoding.

        The streamed summary and the full path build the same glm encoding model
        (both route through _build_encoding_model), so their MAP estimates agree.
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

    def test_decode_session_glm_rank_forwarded(self) -> None:
        """An explicit rank is accepted and reaches the encoder (no error)."""
        from neurospatial.decoding import DecodingResult, decode_session

        env, spike_times, times, positions = _tiled_session(_grid_open_field())
        result = decode_session(
            env, spike_times, times, positions, dt=0.1, method="glm", rank=8
        )
        assert isinstance(result, DecodingResult)


class TestBayesianDecoderGlm:
    def test_bayesian_decoder_glm_fit_predict(self) -> None:
        """BayesianDecoder(method="glm", rank=...).fit(...).predict(...) runs."""
        from neurospatial.decoding import BayesianDecoder, DecodingResult

        env, spike_times, times, positions = _tiled_session(_grid_open_field())
        decoder = BayesianDecoder(env, dt=0.1, method="glm", rank=8)
        # Nullable bandwidth / min_occupancy: default None, not 5.0 / 0.0.
        assert decoder.bandwidth is None
        assert decoder.min_occupancy is None

        fitted = decoder.fit(spike_times, times, positions)
        assert fitted.is_fitted
        result = fitted.predict(spike_times, times)
        assert isinstance(result, DecodingResult)
        assert result.posterior.shape[1] == env.n_bins

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
