"""Tests for DecodingResult container.

These tests verify that DecodingResult correctly stores decoding results
and computes derived properties (MAP, mean, entropy) via cached properties.
"""

import numpy as np
import pandas as pd
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

    def test_map_bin_shape(self, small_2d_env):
        """map_bin should return (n_time_bins,) array of bin indices."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 10
        posterior = np.ones((n_time_bins, small_2d_env.n_bins)) / small_2d_env.n_bins

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        assert result.map_bin.shape == (n_time_bins,)
        assert result.map_bin.dtype in (np.int64, np.intp)

    def test_map_bin_delta_posterior(self, small_2d_env):
        """map_bin should return correct bin for delta (one-hot) posterior."""
        from neurospatial.decoding import DecodingResult

        n_time_bins = 5
        n_bins = small_2d_env.n_bins
        # Create delta posteriors: all mass on a single bin per time step
        posterior = np.zeros((n_time_bins, n_bins))
        expected_bins = [0, 3, n_bins - 1, 5, 2]
        for t, bin_idx in enumerate(expected_bins):
            posterior[t, bin_idx] = 1.0

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        np.testing.assert_array_equal(result.map_bin, expected_bins)

    def test_map_bin_identity_posterior(self, small_2d_env):
        """map_bin on identity posterior should return diagonal indices."""
        from neurospatial.decoding import DecodingResult

        # Use identity matrix (10x10) if env has enough bins
        n_time_bins = min(10, small_2d_env.n_bins)
        posterior = np.eye(n_time_bins, small_2d_env.n_bins)

        result = DecodingResult(posterior=posterior, env=small_2d_env)

        expected = np.arange(n_time_bins)
        np.testing.assert_array_equal(result.map_bin, expected)


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
        """map_position should equal bin_centers[map_bin]."""
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
        assert result.map_bin.shape == (1,)
        assert result.map_position.shape == (1, small_2d_env.n_dims)
        assert result.mean_position.shape == (1, small_2d_env.n_dims)
        assert result.posterior_entropy.shape == (1,)

    def test_single_bin_environment(self):
        """DecodingResult should work with single-bin environment."""
        from neurospatial import Environment
        from neurospatial.decoding import DecodingResult

        env = Environment.from_grid_mask(
            np.ones(1, dtype=bool), grid_edges=(np.array([0.0, 10.0]),)
        )
        assert env.n_bins == 1
        n_time_bins = 5
        posterior = np.ones((n_time_bins, 1))  # Only one bin, always prob=1

        result = DecodingResult(posterior=posterior, env=env)

        assert result.n_time_bins == n_time_bins
        np.testing.assert_array_equal(result.map_bin, np.zeros(n_time_bins))
        np.testing.assert_array_equal(result.map_position, 5.0)
        # Entropy of single bin is 0 (log2(1) = 0)
        np.testing.assert_array_equal(result.posterior_entropy, 0.0)
        assert result.summary()["max_entropy"] == 0.0


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
        assert r.map_bin.tolist() == [0]

        src[0] = [0.1, 0.9, 0.0, 0.0]
        times[0] = 99.0

        np.testing.assert_array_equal(r.posterior[0], [0.9, 0.1, 0.0, 0.0])
        assert r.map_bin.tolist() == r.posterior.argmax(axis=1).tolist() == [0]
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
        assert r.map_bin.tolist() == [0]

        v[0] = [0.1, 0.9, 0.0, 0.0]  # the earlier view still writes into `a`

        np.testing.assert_array_equal(r.posterior[0], [0.9, 0.1, 0.0, 0.0])
        assert r.map_bin.tolist() == [0]

    def test_replace_rebuilds_cache(self, env_4bin):
        import dataclasses

        from neurospatial.decoding import DecodingResult

        r = DecodingResult(posterior=np.array([[0.9, 0.1, 0.0, 0.0]]), env=env_4bin)
        assert r.map_bin.tolist() == [0]
        new = np.array([[0.0, 0.0, 1.0, 0.0]])
        r2 = dataclasses.replace(r, posterior=new)
        assert r2.map_bin.tolist() == [2]
        assert r.map_bin.tolist() == [0]
        assert not np.shares_memory(r2.posterior, new)


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_copied_result_stays_read_only(env_4bin, how):
    import copy
    import pickle

    from neurospatial.decoding import DecodingResult

    r = DecodingResult(
        posterior=np.array([[0.9, 0.1, 0.0, 0.0]]), env=env_4bin, times=np.array([0.5])
    )
    _ = r.map_bin  # cached values travel with the copy
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
        env, counts, models, dt=0.1, times=np.arange(20) * 0.1, **kwargs
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


def test_result_equality_is_identity(env_4bin):
    """Comparing or hashing results never asks an array for a truth value."""
    from neurospatial.decoding import DecodingResult

    posterior = np.array([[0.9, 0.1, 0.0, 0.0]])
    r = DecodingResult(posterior=posterior, env=env_4bin)
    same_values = DecodingResult(posterior=posterior, env=env_4bin)
    assert r == r
    assert r != same_values
    assert hash(r) != hash(same_values)
    assert r in [same_values, r]


_ARRAY_ATTRIBUTES = (
    "posterior",
    "times",
    "spike_window",
    "map_bin",
    "map_position",
    "mean_position",
    "posterior_entropy",
)


@pytest.fixture
def windowed_result(env_4bin):
    from neurospatial.decoding import DecodingResult

    return DecodingResult(
        posterior=np.array([[0.9, 0.1, 0.0, 0.0], [0.0, 0.2, 0.8, 0.0]]),
        env=env_4bin,
        times=np.array([0.5, 1.5]),
        spike_window=np.array([[0.0, 2.0]]),
    )


@pytest.mark.parametrize("name", _ARRAY_ATTRIBUTES)
def test_every_result_array_is_read_only(windowed_result, name):
    """Writing into a derived array would corrupt later exports."""
    before = windowed_result.to_dataframe()
    array = getattr(windowed_result, name)
    assert array.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        array[...] = 3
    pd.testing.assert_frame_equal(windowed_result.to_dataframe(), before)


@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_copied_result_keeps_every_array_read_only(windowed_result, how):
    """Cached arrays and spike windows stay read-only across copies."""
    import copy
    import pickle

    for name in _ARRAY_ATTRIBUTES:
        getattr(windowed_result, name)  # populate the cached arrays
    copied = (
        pickle.loads(pickle.dumps(windowed_result))
        if how == "pickle"
        else copy.deepcopy(windowed_result)
    )
    for name in _ARRAY_ATTRIBUTES:
        array = getattr(copied, name)
        assert array.flags.writeable is False, name
        np.testing.assert_array_equal(array, getattr(windowed_result, name))


@pytest.mark.parametrize(
    ("row", "detail"),
    [
        pytest.param([-7.0, 0.0, 0.0, 0.0], "negative", id="sums-to-minus-seven"),
        pytest.param([0.5, 0.5, 0.5, 0.5], "sum", id="sums-to-two"),
        pytest.param([1.2, -0.2, 0.0, 0.0], "negative", id="negative-entry"),
        pytest.param([np.inf, 0.0, 0.0, 0.0], "Inf", id="infinite-entry"),
        pytest.param([np.nan, 1.0, 0.0, 0.0], "NaN", id="partly-nan"),
    ],
)
def test_constructor_rejects_invalid_posterior_rows(env_4bin, row, detail):
    """Every row is a probability mass over bins, or entirely NaN."""
    from neurospatial.decoding import DecodingResult

    posterior = np.array([[0.25, 0.25, 0.25, 0.25], row])
    with pytest.raises(ValueError, match=detail) as exc:
        DecodingResult(posterior=posterior, env=env_4bin)
    assert "row 1" in str(exc.value)
    assert "Fix:" in str(exc.value)


def test_constructor_accepts_undecodable_rows_and_float32_rounding():
    """Entirely-NaN rows mark undecodable time bins; float32 sums are inexact."""
    from neurospatial import Environment
    from neurospatial.decoding import DecodingResult

    n_bins = 20_000
    env = Environment.from_grid_mask(
        np.ones(n_bins, dtype=bool), grid_edges=(np.arange(n_bins + 1.0),)
    )
    posterior = np.random.default_rng(0).random((4, n_bins), dtype=np.float32)
    posterior /= posterior.sum(axis=1, keepdims=True)
    posterior[2] = np.nan
    result = DecodingResult(posterior=posterior, env=env)
    assert result.posterior.dtype == np.float32


@pytest.mark.parametrize(
    ("times", "detail"),
    [
        pytest.param([3.0, 2.0, 1.0, 0.0], "increasing", id="decreasing"),
        pytest.param([0.0, 1.0, 1.0, 2.0], "increasing", id="duplicate"),
        pytest.param([0.0, np.nan, 2.0, 3.0], "finite", id="nan"),
    ],
)
def test_constructor_rejects_unordered_times(env_4bin, times, detail):
    from neurospatial.decoding import DecodingResult

    posterior = np.full((4, 4), 0.25)
    with pytest.raises(ValueError, match=detail) as exc:
        DecodingResult(posterior=posterior, env=env_4bin, times=np.array(times))
    assert "Fix:" in str(exc.value)


@pytest.mark.parametrize(
    "spike_window",
    [
        pytest.param(np.array([0.0, 1.0, 2.0]), id="shape-3"),
        pytest.param(np.array([[2.0, 1.0]]), id="stop-before-start"),
        pytest.param(np.array([[0.0, np.inf]]), id="infinite"),
    ],
)
def test_constructor_rejects_invalid_spike_window(env_4bin, spike_window):
    from neurospatial.decoding import DecodingResult

    with pytest.raises(ValueError, match="spike_window"):
        DecodingResult(
            posterior=np.full((1, 4), 0.25), env=env_4bin, spike_window=spike_window
        )


def test_every_construction_path_enforces_invariants(env_4bin):
    """The trusted constructors validate too, so no path skips the checks."""
    from neurospatial.decoding import DecodingResult

    with pytest.raises(ValueError, match="sum"):
        DecodingResult._from_owned_posterior(
            np.full((2, 4), 0.5), env=env_4bin, times=None, spike_window=None
        )
    result = DecodingResult(posterior=np.full((2, 4), 0.25), env=env_4bin)
    with pytest.raises(ValueError, match="increasing"):
        result._evolve(times=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="spike_window"):
        result._evolve(spike_window=np.array([0.0, 1.0, 2.0]))
    windowed = result._evolve(spike_window=(0.0, 2.0))
    np.testing.assert_array_equal(windowed.spike_window, [[0.0, 2.0]])
    assert windowed.spike_window.flags.writeable is False


def _summary_fields(env, map_bin=(0, 3), **overrides):
    """Consistent DecodingSummary arguments for ``map_bin`` on ``env``."""
    map_bin = np.asarray(map_bin)
    n_time = map_bin.size
    fields = {
        "times": np.arange(n_time, dtype=np.float64),
        "mean_position": np.zeros((n_time, env.n_dims)),
        "posterior_entropy": np.zeros(n_time),
        "peak_probability": np.ones(n_time),
        "env": env,
        "map_bin": map_bin,
    }
    fields.update(overrides)
    return fields


class TestDecodingSummaryInvariants:
    def test_map_position_is_the_map_bin_center(self, env_4bin):
        from neurospatial.decoding import DecodingSummary

        summary = DecodingSummary(**_summary_fields(env_4bin))
        np.testing.assert_array_equal(
            summary.map_position, env_4bin.bin_centers[[0, 3]]
        )
        assert summary.map_position.flags.writeable is False
        with pytest.raises(TypeError, match="map_position"):
            DecodingSummary(**_summary_fields(env_4bin), map_position=np.zeros((2, 1)))

    @pytest.mark.parametrize(
        ("map_bin", "detail"),
        [
            pytest.param([999, -4], "map_bin", id="out-of-range"),
            pytest.param([0.0, 1.0], "integer", id="float"),
        ],
    )
    def test_rejects_invalid_map_bin(self, env_4bin, map_bin, detail):
        from neurospatial.decoding import DecodingSummary

        with pytest.raises(ValueError, match=detail) as exc:
            DecodingSummary(**_summary_fields(env_4bin, map_bin=map_bin))
        assert "Fix:" in str(exc.value)

    def test_rejects_probability_outside_unit_interval(self, env_4bin):
        from neurospatial.decoding import DecodingSummary

        with pytest.raises(ValueError, match="peak_probability") as exc:
            DecodingSummary(
                **_summary_fields(env_4bin, peak_probability=np.array([5.0, -1.0]))
            )
        assert "Fix:" in str(exc.value)

    def test_rejects_unordered_times(self, env_4bin):
        from neurospatial.decoding import DecodingSummary

        with pytest.raises(ValueError, match="increasing"):
            DecodingSummary(**_summary_fields(env_4bin, times=np.array([1.0, 0.0])))

    def test_accepts_lists(self, env_4bin):
        from neurospatial.decoding import DecodingSummary

        summary = DecodingSummary(
            times=[0.0, 1.0],
            mean_position=[[0.5], [3.5]],
            posterior_entropy=[0.0, 0.0],
            peak_probability=[1.0, 1.0],
            env=env_4bin,
            map_bin=[0, 3],
            spike_window=(0.0, 2.0),
        )
        assert summary.map_bin.dtype == np.int64
        assert summary.posterior_entropy.dtype == np.float64
        np.testing.assert_array_equal(summary.spike_window, [[0.0, 2.0]])
        assert summary.summary()["mean_peak_probability"] == 1.0
        assert "peak_probability" in summary.to_dataframe().columns

    def test_owned_construction_validates_without_copying(self, env_4bin):
        from neurospatial.decoding import DecodingSummary

        fields = _summary_fields(env_4bin, spike_window=None)
        summary = DecodingSummary._from_owned(**fields)
        assert summary.mean_position is fields["mean_position"]
        assert summary.mean_position.flags.writeable is False
        assert not np.shares_memory(summary.times, fields["times"])
        with pytest.raises(ValueError, match="map_bin"):
            DecodingSummary._from_owned(
                **_summary_fields(env_4bin, map_bin=[0, 9], spike_window=None)
            )


class TestUndecodableRows:
    """An all-NaN posterior row has no MAP bin and no position."""

    @pytest.fixture
    def result(self, env_4bin):
        from neurospatial.decoding import DecodingResult

        posterior = np.array([[np.nan] * 4, [0.0, 0.0, 1.0, 0.0]])
        return DecodingResult(
            posterior=posterior, env=env_4bin, times=np.array([0, 1.0])
        )

    def test_map_bin_uses_the_outside_sentinel(self, result, env_4bin):
        outside = env_4bin.bin_at(np.array([[100.0]]))[0]
        assert outside == -1
        np.testing.assert_array_equal(result.map_bin, [outside, 2])

    def test_positions_and_entropy_are_nan(self, result, env_4bin):
        np.testing.assert_array_equal(result.map_position[0], [np.nan])
        np.testing.assert_array_equal(result.map_position[1], env_4bin.bin_centers[2])
        np.testing.assert_array_equal(result.mean_position[0], [np.nan])
        assert np.isnan(result.posterior_entropy[0])
        assert result.posterior_entropy[1] == 0.0
        df = result.to_dataframe()
        assert df["map_bin"].tolist() == [-1, 2]
        assert np.isnan(df.loc[0, "map_x"])
        assert np.isnan(df.loc[0, "mean_x"])

    def test_error_against_is_nan(self, result, env_4bin):
        truth = env_4bin.bin_centers[[1, 2]]
        errors = result.error_against(np.array([0.0, 1.0]), truth)
        assert np.isnan(errors[0])
        assert errors[1] == 0.0

    def test_summary_metrics_average_the_decoded_rows(self, result, env_4bin):
        from neurospatial.decoding import DecodingSummary

        s = result.summary()
        assert s["n_undecodable"] == 1
        assert s["mean_entropy"] == 0.0
        streamed = DecodingSummary(
            **_summary_fields(
                env_4bin,
                map_bin=[-1, 2],
                peak_probability=np.array([np.nan, 1.0]),
                posterior_entropy=np.array([np.nan, 0.0]),
            )
        ).summary()
        assert streamed["n_undecodable"] == 1
        assert streamed["mean_entropy"] == 0.0
        assert streamed["mean_peak_probability"] == 1.0

    def test_summary_without_decoded_rows_is_nan(self, env_4bin):
        from neurospatial.decoding import DecodingResult

        s = DecodingResult(posterior=np.full((1, 4), np.nan), env=env_4bin).summary()
        assert s["n_undecodable"] == 1
        assert np.isnan(s["mean_entropy"])

    def test_summary_accepts_sentinel_with_nan_peak(self, env_4bin):
        from neurospatial.decoding import DecodingSummary

        summary = DecodingSummary(
            **_summary_fields(
                env_4bin,
                map_bin=[-1, 2],
                peak_probability=np.array([np.nan, 1.0]),
                posterior_entropy=np.array([np.nan, 0.0]),
            )
        )
        np.testing.assert_array_equal(summary.map_position[0], [np.nan])
        np.testing.assert_array_equal(summary.map_position[1], env_4bin.bin_centers[2])

    @pytest.mark.parametrize(
        ("map_bin", "peak"),
        [
            pytest.param([-1, 2], [1.0, 1.0], id="sentinel-with-peak"),
            pytest.param([0, 2], [np.nan, 1.0], id="bin-without-peak"),
        ],
    )
    def test_summary_rejects_inconsistent_sentinel(self, env_4bin, map_bin, peak):
        from neurospatial.decoding import DecodingSummary

        with pytest.raises(ValueError, match="peak_probability value") as exc:
            DecodingSummary(
                **_summary_fields(
                    env_4bin, map_bin=map_bin, peak_probability=np.array(peak)
                )
            )
        assert "map_bin value" not in str(exc.value)


def test_posterior_entropy_is_float64_like_the_summary(env_4bin):
    """A float32 decode reports float64 entropy on both decode paths."""
    from neurospatial.decoding import decode_position, decode_position_summary

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(20, 3)).astype(np.int64)
    models = rng.uniform(1.0, 10.0, size=(3, env_4bin.n_bins))
    full = decode_position(env_4bin, counts, models, dt=0.1, dtype=np.float32)
    summary = decode_position_summary(
        env_4bin, counts, models, dt=0.1, dtype=np.float32
    )
    assert full.posterior.dtype == np.float32
    assert full.posterior_entropy.dtype == np.float64
    np.testing.assert_allclose(
        full.posterior_entropy, summary.posterior_entropy, rtol=0, atol=1e-12
    )


def test_delta_posterior_entropy_is_positive_zero(env_4bin):
    from neurospatial.decoding import DecodingResult

    result = DecodingResult(posterior=np.eye(4), env=env_4bin)
    assert not np.signbit(result.posterior_entropy).any()
    np.testing.assert_array_equal(result.posterior_entropy, 0.0)
