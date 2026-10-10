"""Tests for DecodingResult container.

These tests verify that DecodingResult correctly stores decoding results
and computes derived properties (MAP, mean, entropy) via cached properties.
"""

import numpy as np
import pytest


class TestDecodingResultBasic:
    """Test basic DecodingResult instantiation and properties."""

    def test_n_time_bins_property(self, small_2d_env):
        """n_time_bins property should return correct count."""
        from neurospatial.decoding import DecodingResult

        for n_time_bins in [1, 5, 10, 100]:
            posterior = (
                np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
            )
            result = DecodingResult(posterior=posterior, env=small_2d_env)
            assert result.n_time_bins == n_time_bins


class TestMapEstimate:
    """Test MAP (maximum a posteriori) estimate computation."""

    def test_map_estimate_shape(self, small_2d_env):
        """map_estimate should return (n_time_bins,) array of bin indices."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.map_estimate.shape == (n_time_bins,)
        assert result.map_estimate.dtype in (np.int64, np.intp)

    def test_map_estimate_delta_posterior(self, small_2d_env):
        """map_estimate should return correct bin for delta (one-hot) posterior."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors: all mass on a single bin per time step
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 3, n_bins - 1, 5, 2]
        for t, bin_idx in enumerate(expected_bins):
            posterior[t, bin_idx] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_equal(result.map_estimate, expected_bins)

    def test_map_estimate_identity_posterior(self, small_2d_env):
        """map_estimate on identity posterior should return diagonal indices."""
        from neurospatial.decoding import DecodingResult

        # Use identity matrix (10x10) if env has enough bins
        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        expected = np.arange(n_time_bins)
        np.testing.assert_array_equal(result.map_estimate, expected)


class TestMapPosition:
    """Test MAP position in environment coordinates."""

    def test_map_position_shape(self, small_2d_env):
        """map_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.map_position.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.map_position.dtype == np.float64

    def test_map_position_matches_bin_centers(self, small_2d_env):
        """map_position should equal bin_centers[map_estimate]."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors at known bins
        posterior = np.zeros((n_time_bins, n_bins))
        known_bins = [0, 1, 2, 3, 4]
        for t, bin_idx in enumerate(known_bins):
            posterior[t, bin_idx] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        expected_positions = small_2d_env.bin_centers[known_bins]
        np.testing.assert_array_almost_equal(result.map_position, expected_positions)


class TestMeanPosition:
    """Test posterior mean position computation."""

    def test_mean_position_shape(self, small_2d_env):
        """mean_position should return (n_time_bins, n_dims) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.mean_position.shape == (n_time_bins, small_2d_env.n_dims)
        assert result.mean_position.dtype == np.float64

    def test_mean_position_delta_equals_map(self, small_2d_env):
        """For delta posteriors, mean_position should equal map_position."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_almost_equal(result.mean_position, result.map_position)

    def test_mean_position_uniform_is_centroid(self, small_2d_env):
        """For uniform posterior, mean_position should be centroid of all bins."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Expected: centroid of all bin centers
        expected_centroid = small_2d_env.bin_centers.mean(axis=0)
        for t in range(n_time_bins):
            np.testing.assert_array_almost_equal(
                result.mean_position[t], expected_centroid
            )

    def test_mean_position_two_bin_weighted(self, small_2d_env):
        """Mean position with two bins should be weighted average."""
        from neurospatial.decoding import DecodingResult

        n_bins = small_2d_env.n_bins
        if n_bins < 2:
            pytest.skip("Need at least 2 bins for this test")

        # Put 75% mass on bin 0, 25% on bin 1
        posterior = np.zeros((1, n_bins))
        posterior[0, 0] = 0.75
        posterior[0, 1] = 0.25

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Expected: 0.75 * bin_centers[0] + 0.25 * bin_centers[1]
        expected = (
            0.75 * small_2d_env.bin_centers[0] + 0.25 * small_2d_env.bin_centers[1]
        )
        np.testing.assert_array_almost_equal(result.mean_position[0], expected)


class TestUncertainty:
    """Test posterior entropy (posterior_entropy) computation."""

    def test_uncertainty_shape(self, small_2d_env):
        """posterior_entropy should return (n_time_bins,) array."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.posterior_entropy.shape == (n_time_bins,)
        assert result.posterior_entropy.dtype == np.float64

    def test_uncertainty_delta_is_zero(self, small_2d_env):
        """Delta (one-hot) posterior should have zero entropy."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_almost_equal(
            result.posterior_entropy, np.zeros(n_time_bins)
        )

    def test_uncertainty_uniform_is_maximum(self, small_2d_env):
        """Uniform posterior should have maximum entropy = log2(n_bins)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        posterior = np.ones((n_time_bins, n_bins)) / n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        max_entropy = np.log2(n_bins)
        np.testing.assert_array_almost_equal(
            result.posterior_entropy, np.full(n_time_bins, max_entropy)
        )

    def test_uncertainty_bounds(self, small_2d_env):
        """Uncertainty should be bounded: 0 <= entropy <= log2(n_bins)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        n_bins = small_2d_env.n_bins

        # Random posterior (normalized)
        rng = np.random.default_rng(42)
        posterior = rng.random((n_time_bins, n_bins))
        posterior = posterior / posterior.sum(axis=1, keepdims=True)

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert np.all(result.posterior_entropy >= 0)
        assert np.all(result.posterior_entropy <= np.log2(n_bins) + 1e-10)

    def test_uncertainty_handles_exact_zeros(self, small_2d_env):
        """Uncertainty should handle exact zeros without NaN (mask-based)."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 3
        n_bins = small_2d_env.n_bins
        # Posterior with many exact zeros
        posterior = np.zeros((n_time_bins, n_bins))
        posterior[:, 0] = 0.5
        posterior[:, 1] = 0.5

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        # Should not be NaN
        assert np.all(np.isfinite(result.posterior_entropy))
        # Entropy of 50-50 distribution = 1 bit
        np.testing.assert_array_almost_equal(
            result.posterior_entropy, np.ones(n_time_bins)
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_time_bin(self, small_2d_env):
        """DecodingResult should work with a single time bin."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((1, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.n_time_bins == 1
        assert result.map_estimate.shape == (1,)
        assert result.map_position.shape == (1, small_2d_env.n_dims)
        assert result.mean_position.shape == (1, small_2d_env.n_dims)
        assert result.posterior_entropy.shape == (1,)

    def test_single_bin_environment(self):
        """DecodingResult should work with single-bin environment."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        # Create a minimal single-bin environment
        env = Environment.from_samples(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            bin_size=10.0,  # Large bin size to get single bin
        )

        if env.n_bins == 1:
            n_time_bins = 5
            posterior = np.ones((n_time_bins, 1))  # Only one bin, always prob=1

            result = DecodingResult(posterior=posterior, env=env)

            assert result.n_time_bins == n_time_bins
            np.testing.assert_array_equal(result.map_estimate, np.zeros(n_time_bins))
            # Entropy of single bin is 0 (log2(1) = 0)
            np.testing.assert_array_almost_equal(
                result.posterior_entropy, np.zeros(n_time_bins)
            )
        else:
            pytest.skip("Could not create single-bin environment")


class TestPlotMethod:
    """Test DecodingResult.plot() method.

    Only one test kept: ``plot()`` returns the caller-supplied ``ax``
    object. The previous file had six tests that each asserted
    ``assert ax is not None`` after passing a different kwarg
    (cmap, colorbar, show_map, times, no-ax) — none of them verified
    that the kwarg actually changed anything about the plot.
    """

    def test_plot_uses_provided_axes(self, small_2d_env):
        """plot() should use the provided axes when given."""
        import matplotlib.pyplot as plt

        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        _, provided_ax = plt.subplots()
        returned_ax = result.plot(ax=provided_ax)

        assert returned_ax is provided_ax
        plt.close("all")


class TestToDataFrameMethod:
    """Test DecodingResult.to_dataframe() method."""

    def test_to_dataframe_has_time_column_when_times_provided(self, small_2d_env):
        """to_dataframe() should include time column when times are provided."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins
        times = np.arange(n_time_bins) * 0.025
        result = DecodingResult(posterior=posterior, env=small_2d_env, times=times)

        df = result.to_dataframe()

        assert "time" in df.columns
        np.testing.assert_array_almost_equal(df["time"].values, times)

    def test_to_dataframe_no_time_column_when_times_none(self, small_2d_env):
        """to_dataframe() should not have time column when times is None."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "time" not in df.columns

    def test_to_dataframe_has_map_bin_column(self, small_2d_env):
        """to_dataframe() should include map_bin column."""
        from neurospatial.decoding import DecodingResult

        # Create delta posteriors with known MAP bins
        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 1, 2, 3, 4]
        for t, b in enumerate(expected_bins):
            posterior[t, b] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)
        df = result.to_dataframe()

        assert "map_bin" in df.columns
        np.testing.assert_array_equal(df["map_bin"].values, expected_bins)

    def test_to_dataframe_has_map_position_columns_2d(self, small_2d_env):
        """to_dataframe() should include map_x and map_y for 2D environments."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "map_x" in df.columns
        assert "map_y" in df.columns

    def test_to_dataframe_has_mean_position_columns_2d(self, small_2d_env):
        """to_dataframe() should include mean_x and mean_y for 2D environments."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "mean_x" in df.columns
        assert "mean_y" in df.columns

    def test_to_dataframe_has_uncertainty_column(self, small_2d_env):
        """to_dataframe() should include posterior_entropy column."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)

        df = result.to_dataframe()

        assert "posterior_entropy" in df.columns
        # Uniform posterior has max entropy
        expected_entropy = np.log2(small_2d_env.n_bins)
        np.testing.assert_array_almost_equal(
            df["posterior_entropy"].values, np.full(5, expected_entropy)
        )

    def test_to_dataframe_map_positions_match_property(self, small_2d_env):
        """to_dataframe() map positions should match map_position property."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        posterior = np.zeros((n_time_bins, n_bins))
        for t in range(n_time_bins):
            posterior[t, t % n_bins] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)
        df = result.to_dataframe()

        np.testing.assert_array_almost_equal(
            df["map_x"].values, result.map_position[:, 0]
        )
        np.testing.assert_array_almost_equal(
            df["map_y"].values, result.map_position[:, 1]
        )

    def test_to_dataframe_mean_positions_match_property(self, small_2d_env):
        """to_dataframe() mean positions should match mean_position property."""
        from neurospatial.decoding import DecodingResult

        posterior = np.ones((5, small_2d_env.n_bins)) / small_2d_env.n_bins
        result = DecodingResult(posterior=posterior, env=small_2d_env)
        df = result.to_dataframe()

        np.testing.assert_array_almost_equal(
            df["mean_x"].values, result.mean_position[:, 0]
        )
        np.testing.assert_array_almost_equal(
            df["mean_y"].values, result.mean_position[:, 1]
        )

    def test_to_dataframe_1d_environment(self):
        """to_dataframe() should use 'x' for 1D environments."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        # Create 1D environment
        positions = np.linspace(0, 100, 100).reshape(-1, 1)
        env = Environment.from_samples(positions, bin_size=5.0)

        posterior = np.ones((5, env.n_bins)) / env.n_bins
        result = DecodingResult(posterior=posterior, env=env)
        df = result.to_dataframe()

        assert "map_x" in df.columns
        assert "mean_x" in df.columns
        assert "map_y" not in df.columns
        assert "mean_y" not in df.columns

    def test_to_dataframe_high_dimensional_environment(self):
        """to_dataframe() should use dim_0, dim_1, etc. for >3D environments."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        # Create 4D environment
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 10, (100, 4))
        env = Environment.from_samples(positions, bin_size=5.0)

        posterior = np.ones((5, env.n_bins)) / env.n_bins
        result = DecodingResult(posterior=posterior, env=env)
        df = result.to_dataframe()

        for i in range(4):
            assert f"map_dim_{i}" in df.columns
            assert f"mean_dim_{i}" in df.columns
        assert "map_x" not in df.columns


# ---------------------------------------------------------------------------
# DecodingResult owns its arrays
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env_4bin():
    """1-D environment with four unit-width bins."""
    from neurospatial import Environment

    return Environment.from_grid_mask(
        np.ones(4, dtype=bool), grid_edges=(np.arange(5.0),)
    )


class TestResultOwnsData:
    def test_result_isolated_from_caller_array(self, env_4bin):
        from neurospatial.decoding import DecodingResult

        src = np.array([[0.9, 0.1, 0.0, 0.0]])
        times = np.array([0.5])
        r = DecodingResult(posterior=src, env=env_4bin, times=times)
        assert r.map_estimate.tolist() == [0]

        src[0] = [0.1, 0.9, 0.0, 0.0]
        times[0] = 99.0

        np.testing.assert_array_equal(r.posterior[0], [0.9, 0.1, 0.0, 0.0])
        assert r.map_estimate.tolist() == r.posterior.argmax(axis=1).tolist() == [0]
        np.testing.assert_array_equal(r.times, [0.5])

        # A read-only view of a writeable array is copied, not shared.
        base = np.array([[0.9, 0.1, 0.0, 0.0]])
        view = base.view()
        view.flags.writeable = False
        r_view = DecodingResult(posterior=view, env=env_4bin)
        assert not np.shares_memory(r_view.posterior, base)

    def test_result_fields_cannot_change(self, env_4bin):
        import dataclasses

        from neurospatial.decoding import DecodingResult

        src = np.array([[0.9, 0.1, 0.0, 0.0]])
        r = DecodingResult(posterior=src, env=env_4bin)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.posterior = np.zeros((1, 4))
        with pytest.raises(ValueError, match="read-only"):
            r.posterior[0, 0] = 0.5
        assert src.flags.writeable

    def test_result_copies_owning_array_with_prior_view(self, env_4bin):
        from neurospatial.decoding import DecodingResult

        a = np.array([[0.9, 0.1, 0.0, 0.0]])
        v = a[:]
        a.flags.writeable = False
        r = DecodingResult(posterior=a, env=env_4bin)
        assert r.map_estimate.tolist() == [0]

        v[0] = [0.1, 0.9, 0.0, 0.0]  # the earlier view still writes into `a`

        np.testing.assert_array_equal(r.posterior[0], [0.9, 0.1, 0.0, 0.0])
        assert r.map_estimate.tolist() == [0]

    def test_replace_rebuilds_cache(self, env_4bin):
        import dataclasses

        from neurospatial.decoding import DecodingResult

        r = DecodingResult(posterior=np.array([[0.9, 0.1, 0.0, 0.0]]), env=env_4bin)
        assert r.map_estimate.tolist() == [0]
        new = np.array([[0.0, 0.0, 1.0, 0.0]])
        r2 = dataclasses.replace(r, posterior=new)
        assert r2.map_estimate.tolist() == [2]
        assert r.map_estimate.tolist() == [0]
        assert not np.shares_memory(r2.posterior, new)


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_copied_result_stays_read_only(env_4bin, how):
    import copy
    import pickle

    from neurospatial.decoding import DecodingResult

    r = DecodingResult(
        posterior=np.array([[0.9, 0.1, 0.0, 0.0]]), env=env_4bin, times=np.array([0.5])
    )
    _ = r.map_estimate  # cached values travel with the copy
    r2 = pickle.loads(pickle.dumps(r)) if how == "pickle" else copy.deepcopy(r)
    assert r2.posterior.flags.writeable is False
    assert r2.times.flags.writeable is False
    np.testing.assert_array_equal(r2.posterior, r.posterior)


@pytest.fixture
def record_owned_posterior(monkeypatch):
    """Record the arguments decode_position passes to the trusted constructor."""
    from neurospatial.decoding import DecodingResult

    calls: list[tuple[np.ndarray, dict]] = []
    original = DecodingResult._from_owned_posterior.__func__

    def recording(cls, posterior, **fields):
        calls.append((posterior, fields))
        return original(cls, posterior, **fields)

    monkeypatch.setattr(DecodingResult, "_from_owned_posterior", classmethod(recording))
    return calls


def _decode_small(env, **kwargs):
    from neurospatial.decoding import decode_position

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(20, 3)).astype(np.int64)
    models = rng.uniform(1.0, 10.0, size=(3, env.n_bins))
    return decode_position(
        env, counts, models, 0.1, times=np.arange(20) * 0.1, **kwargs
    )


class TestDecodePositionOwnership:
    @pytest.mark.parametrize(
        "kwargs", [{}, {"time_chunk": 7}, {"dtype": np.float32}], ids=str
    )
    def test_decode_position_posterior_not_copied(
        self, env_4bin, record_owned_posterior, kwargs
    ):
        result = _decode_small(env_4bin, **kwargs)
        ((recorded, _),) = record_owned_posterior
        assert result.posterior is recorded
        assert recorded.flags.writeable is False

    def test_from_owned_posterior_covers_all_fields(
        self, env_4bin, record_owned_posterior
    ):
        import dataclasses

        from neurospatial.decoding import DecodingResult

        _decode_small(env_4bin)
        ((_, fields),) = record_owned_posterior
        expected = {f.name for f in dataclasses.fields(DecodingResult)} - {"posterior"}
        assert set(fields) == expected


@pytest.mark.parametrize(
    ("posterior_shape", "n_times", "detail"),
    [
        pytest.param((10, 3), 10, "n_bins", id="columns-not-env-bins"),
        pytest.param((10,), 10, "2-D", id="one-dimensional"),
        pytest.param((10, None), 4, "times", id="times-length"),
    ],
)
def test_constructor_rejects_misaligned_shapes(posterior_shape, n_times, detail):
    """A misaligned posterior would map columns to the wrong bin centers."""
    from neurospatial import Environment
    from neurospatial.decoding import DecodingResult

    env = Environment.from_samples(
        np.random.default_rng(0).uniform(0, 50, (500, 2)), bin_size=5.0
    )
    shape = tuple(env.n_bins if s is None else s for s in posterior_shape)
    with pytest.raises(ValueError, match=detail) as exc:
        DecodingResult(
            posterior=np.ones(shape), env=env, times=np.arange(float(n_times))
        )
    assert "Fix:" in str(exc.value)
