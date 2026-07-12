"""NWB round-trip for ``method="glm"`` spatial-rate results.

``write_spatial_rates`` / ``read_place_field`` must persist the GAM diagnostics
that ``method="glm"`` carries -- ``coefficients``, ``penalty``,
``penalty_weights``, ``rank``, ``deviance``, ``converged``, ``n_iter``,
``reml_objective`` -- and read them back field-for-field with shapes intact,
``bandwidth=None``. Ratio results still round-trip with the GAM fields absent
(``None``).
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip every test here when pynwb is absent (HAS_PYNWB gate).
pytest.importorskip("pynwb")


def _make_glm_rates_result(
    env,
    *,
    n_units: int = 3,
    rank: int = 5,
    penalty: float | None = 0.75,
    reml_objective: float | None = -123.45,
    seed: int = 0,
):
    """Synthetic glm ``SpatialRatesResult`` satisfying the None-iff-glm invariant.

    All GAM fields are populated with the exact shapes the invariant pins
    (``coefficients (rank, n_units)``, ``penalty_weights (rank,)``,
    ``deviance (n_units,)``, ``bandwidth=None``), so the round-trip is a pure
    persistence test independent of the estimator's numerics.
    """
    from neurospatial.encoding.spatial import SpatialRatesResult

    rng = np.random.default_rng(seed)
    return SpatialRatesResult(
        firing_rates=rng.uniform(0.01, 10.0, (n_units, env.n_bins)),
        occupancy=rng.uniform(0.0, 5.0, env.n_bins),
        env=env,
        method="glm",
        bandwidth=None,
        coefficients=rng.standard_normal((rank, n_units)),
        penalty=penalty,
        penalty_weights=rng.uniform(0.0, 2.0, rank),
        rank=rank,
        deviance=rng.uniform(0.0, 100.0, n_units),
        converged=True,
        n_iter=7,
        reml_objective=reml_objective,
    )


def _make_ratio_rates_result(env, *, n_units: int = 3, seed: int = 0):
    from neurospatial.encoding.spatial import SpatialRatesResult

    rng = np.random.default_rng(seed)
    return SpatialRatesResult(
        firing_rates=rng.uniform(0.0, 10.0, (n_units, env.n_bins)),
        occupancy=rng.uniform(0.0, 5.0, env.n_bins),
        env=env,
        method="diffusion_kde",
        bandwidth=5.0,
    )


_GAM_FIELDS = (
    "coefficients",
    "penalty",
    "penalty_weights",
    "rank",
    "deviance",
    "converged",
    "n_iter",
    "reml_objective",
)


class TestGlmRoundTrip:
    def test_nwb_glm_roundtrip(self, empty_nwb, sample_environment):
        """A glm result writes and reads back equal -- field-by-field, shapes kept."""
        from neurospatial.encoding.spatial import SpatialRatesResult
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_glm_rates_result(env)

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb, env=env)

        assert isinstance(back, SpatialRatesResult)
        assert back.method == "glm"
        assert back.bandwidth is None

        # Scalars.
        assert back.rank == result.rank
        assert bool(back.converged) == bool(result.converged)
        assert int(back.n_iter) == int(result.n_iter)
        assert float(back.penalty) == pytest.approx(float(result.penalty))
        assert float(back.reml_objective) == pytest.approx(float(result.reml_objective))

        # Vectors / matrices -- shapes preserved.
        assert np.asarray(back.coefficients).shape == (result.rank, 3)
        np.testing.assert_allclose(
            np.asarray(back.coefficients), np.asarray(result.coefficients)
        )
        assert np.asarray(back.penalty_weights).shape == (result.rank,)
        np.testing.assert_allclose(
            np.asarray(back.penalty_weights), np.asarray(result.penalty_weights)
        )
        assert np.asarray(back.deviance).shape == (3,)
        np.testing.assert_allclose(
            np.asarray(back.deviance), np.asarray(result.deviance)
        )

        # firing_rates / occupancy still exact.
        np.testing.assert_array_equal(
            np.asarray(back.firing_rates), np.asarray(result.firing_rates)
        )
        np.testing.assert_array_equal(
            np.asarray(back.occupancy), np.asarray(result.occupancy)
        )

    def test_nwb_glm_none_penalty_roundtrip(self, empty_nwb, sample_environment):
        """penalty=None / reml_objective=None (REML-skip) round-trip as None."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_glm_rates_result(env, penalty=None, reml_objective=None)

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb, env=env)

        assert back.penalty is None
        assert back.reml_objective is None
        # The other GAM fields still populated (invariant: they are required).
        assert back.coefficients is not None
        assert back.rank == result.rank

    def test_nwb_glm_real_fit_roundtrip(self, empty_nwb, sample_environment):
        """A REAL computed glm result (estimator output) round-trips equal."""
        from neurospatial.encoding.spatial import compute_spatial_rates
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        # Tile active-bin centers so occupancy is uniform + positive everywhere.
        bin_centers = np.asarray(env.bin_centers, dtype=np.float64)
        positions = np.tile(bin_centers, (8, 1))
        rng = np.random.default_rng(3)
        positions = positions[rng.permutation(positions.shape[0])]
        times = np.arange(positions.shape[0], dtype=np.float64) * 0.05
        spike_times = [np.sort(rng.uniform(times[0], times[-1], 40)) for _ in range(3)]

        result = compute_spatial_rates(env, spike_times, times, positions, method="glm")
        assert result.method == "glm"

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb, env=env)

        assert back.method == "glm"
        assert back.bandwidth is None
        assert back.rank == result.rank
        np.testing.assert_allclose(
            np.asarray(back.coefficients), np.asarray(result.coefficients)
        )
        np.testing.assert_allclose(
            np.asarray(back.deviance), np.asarray(result.deviance)
        )
        np.testing.assert_allclose(
            np.asarray(back.penalty_weights), np.asarray(result.penalty_weights)
        )

    def test_nwb_glm_disk_roundtrip(self, sample_environment, tmp_path):
        """A glm result survives a real NWBHDF5IO write -> reopen -> read cycle.

        Exercises the on-disk path (the 2-D ``coefficients`` column materialized
        through HDF5), not just the in-memory container.
        """
        from pynwb import NWBHDF5IO

        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        from ._nwb_helpers import create_roundtrip_nwb

        env = sample_environment
        result = _make_glm_rates_result(env)

        nwbfile = create_roundtrip_nwb()
        write_spatial_rates(nwbfile, result, name="glm_rates")

        nwb_path = tmp_path / "glm_rates.nwb"
        with NWBHDF5IO(str(nwb_path), "w") as io:
            io.write(nwbfile)
        with NWBHDF5IO(str(nwb_path), "r") as io:
            reopened = io.read()
            back = read_place_field(reopened, name="glm_rates", env=env)

            assert back.method == "glm"
            assert back.bandwidth is None
            assert back.rank == result.rank
            assert np.asarray(back.coefficients).shape == (result.rank, 3)
            np.testing.assert_allclose(
                np.asarray(back.coefficients), np.asarray(result.coefficients)
            )
            np.testing.assert_allclose(
                np.asarray(back.penalty_weights), np.asarray(result.penalty_weights)
            )
            np.testing.assert_allclose(
                np.asarray(back.deviance), np.asarray(result.deviance)
            )
            assert float(back.penalty) == pytest.approx(float(result.penalty))

    def test_nwb_ratio_roundtrip_unchanged(self, empty_nwb, sample_environment):
        """A diffusion_kde result round-trips with GAM fields absent/None."""
        from neurospatial.io.nwb import read_place_field, write_spatial_rates

        env = sample_environment
        result = _make_ratio_rates_result(env)

        write_spatial_rates(empty_nwb, result)
        back = read_place_field(empty_nwb, env=env)

        assert back.method == "diffusion_kde"
        assert back.bandwidth == 5.0
        for name in _GAM_FIELDS:
            assert getattr(back, name) is None, f"ratio result carried GAM {name}"
        np.testing.assert_array_equal(
            np.asarray(back.firing_rates), np.asarray(result.firing_rates)
        )
