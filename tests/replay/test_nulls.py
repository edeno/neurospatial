"""Tests for the replay shuffles in neurospatial.replay.nulls.

The seed and degenerate-input contract shared by every shuffle generator is in
``tests/stats/test_shuffle.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from environment_helpers import track_for
from numpy.testing import assert_array_equal

from neurospatial.replay.nulls import (
    shuffle_cell_identity,
    shuffle_place_fields_circular,
    shuffle_place_fields_circular_2d,
    shuffle_posterior_circular,
    shuffle_posterior_weighted_circular,
)


@pytest.fixture
def spike_counts() -> np.ndarray:
    """Sample spike counts: (5 time bins, 3 neurons)."""
    return np.array(
        [
            [0, 1, 2],  # t=0
            [1, 0, 1],  # t=1
            [2, 2, 0],  # t=2
            [0, 0, 3],  # t=3
            [1, 1, 1],  # t=4
        ],
        dtype=np.int64,
    )


# =============================================================================
# Tests for shuffle_cell_identity
# =============================================================================


class TestShuffleCellIdentity:
    """Tests for shuffle_cell_identity (neuron-place field mapping shuffle)."""

    @pytest.fixture
    def encoding_models(self) -> np.ndarray:
        """Sample encoding models: (3 neurons, 4 bins)."""
        return np.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # Neuron 0 place field
                [4.0, 3.0, 2.0, 1.0],  # Neuron 1 place field
                [2.0, 4.0, 1.0, 3.0],  # Neuron 2 place field
            ],
            dtype=np.float64,
        )

    def test_yields_correct_number_of_shuffles(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Should yield exactly n_shuffles tuples."""

        n_shuffles = 10
        shuffles = list(
            shuffle_cell_identity(
                spike_counts, encoding_models, n_shuffles=n_shuffles, rng=42
            )
        )
        assert len(shuffles) == n_shuffles

    def test_yields_tuples(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Should yield (shuffled_counts, encoding_models) tuples."""

        for shuffled_counts, models in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert isinstance(shuffled_counts, np.ndarray)
            assert isinstance(models, np.ndarray)

    def test_encoding_models_unchanged(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Encoding models should be returned unchanged (same object)."""

        for _, models in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert models is encoding_models

    def test_shuffled_counts_shape(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Shuffled counts should have same shape as input."""

        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled_counts.shape == spike_counts.shape

    def test_shuffled_counts_dtype(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Shuffled counts should preserve dtype."""

        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=5, rng=42
        ):
            assert shuffled_counts.dtype == spike_counts.dtype

    def test_preserves_total_spikes(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Total spike count should be preserved."""

        total_original = spike_counts.sum()
        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=10, rng=42
        ):
            assert shuffled_counts.sum() == total_original

    def test_preserves_spikes_per_time_bin(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Total spikes per time bin should be preserved (columns permuted)."""

        spikes_per_time_bin = spike_counts.sum(axis=1)
        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=10, rng=42
        ):
            assert_array_equal(shuffled_counts.sum(axis=1), spikes_per_time_bin)

    def test_columns_are_permuted(
        self, spike_counts: np.ndarray, encoding_models: np.ndarray
    ) -> None:
        """Columns (neurons) should be permutations of original columns."""

        original_cols = {
            tuple(spike_counts[:, i]) for i in range(spike_counts.shape[1])
        }
        for shuffled_counts, _ in shuffle_cell_identity(
            spike_counts, encoding_models, n_shuffles=10, rng=42
        ):
            shuffled_cols = {
                tuple(shuffled_counts[:, i]) for i in range(shuffled_counts.shape[1])
            }
            assert shuffled_cols == original_cols


# =============================================================================
# Tests for shuffle_place_fields_circular
# =============================================================================


class TestShufflePlaceFieldsCircular:
    """Tests for shuffle_place_fields_circular (1D circular shift)."""

    @pytest.fixture
    def encoding_models(self) -> np.ndarray:
        """Sample encoding models: (3 neurons, 8 bins)."""
        return np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # Neuron 0
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],  # Neuron 1
                [1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 5.0, 5.0],  # Neuron 2
            ],
            dtype=np.float64,
        )

    def test_yields_correct_number_of_shuffles(
        self, encoding_models: np.ndarray
    ) -> None:
        """Should yield exactly n_shuffles arrays."""

        n_shuffles = 10
        shuffles = list(
            shuffle_place_fields_circular(
                track_for(encoding_models),
                encoding_models,
                n_shuffles=n_shuffles,
                rng=42,
            )
        )
        assert len(shuffles) == n_shuffles

    def test_preserves_row_values(self, encoding_models: np.ndarray) -> None:
        """Each row should contain same values (multiset) as original.

        Circular shift preserves the shape of each place field.
        """

        for shuffled in shuffle_place_fields_circular(
            track_for(encoding_models), encoding_models, n_shuffles=10, rng=42
        ):
            for i in range(encoding_models.shape[0]):
                original_sorted = sorted(encoding_models[i, :].tolist())
                shuffled_sorted = sorted(shuffled[i, :].tolist())
                assert shuffled_sorted == original_sorted

    def test_rows_are_circular_shifts(self, encoding_models: np.ndarray) -> None:
        """Each row should be a circular shift of the original.

        A circular shift means the row values appear in the same cyclic order.
        """

        for shuffled in shuffle_place_fields_circular(
            track_for(encoding_models), encoding_models, n_shuffles=10, rng=42
        ):
            for i in range(encoding_models.shape[0]):
                # Check if shuffled row is a circular shift of original
                orig = encoding_models[i, :]
                shuf = shuffled[i, :]
                n_bins = len(orig)
                is_shift = any(
                    np.allclose(shuf, np.roll(orig, shift)) for shift in range(n_bins)
                )
                assert is_shift, f"Row {i} is not a circular shift"

    def test_each_neuron_shifted_independently(
        self, encoding_models: np.ndarray
    ) -> None:
        """Different neurons should get different shift amounts."""

        # With enough shuffles, different neurons should have different shifts
        # at least sometimes
        found_different = False
        for shuffled in shuffle_place_fields_circular(
            track_for(encoding_models), encoding_models, n_shuffles=50, rng=42
        ):
            # Check shift amounts for each neuron
            shifts = []
            for i in range(encoding_models.shape[0]):
                orig = encoding_models[i, :]
                shuf = shuffled[i, :]
                # Find the shift amount
                for shift in range(len(orig)):
                    if np.allclose(shuf, np.roll(orig, shift)):
                        shifts.append(shift)
                        break
            # Check if different neurons got different shifts
            if len(set(shifts)) > 1:
                found_different = True
                break
        assert found_different, "Each neuron should be shifted independently"


# =============================================================================
# Tests for shuffle_place_fields_circular_2d
# =============================================================================


class TestShufflePlaceFieldsCircular2D:
    """Tests for shuffle_place_fields_circular_2d (2D circular shift)."""

    @pytest.fixture
    def env_2d(self):
        """Create a 2D environment with full grid (no inactive bins)."""
        from neurospatial import Environment

        # Create a full grid by using infer_active_bins=False
        positions = np.random.default_rng(42).uniform(0, 10, (1000, 2))
        env = Environment.from_samples(positions, bin_size=2.0, infer_active_bins=False)
        return env

    @pytest.fixture
    def encoding_models_2d(self, env_2d) -> np.ndarray:
        """Sample encoding models for 2D environment: (3 neurons, n_bins)."""
        # For 2D shuffle, encoding models must match full grid size
        grid_shape = env_2d.grid_shape
        n_bins = int(np.prod(grid_shape))
        rng = np.random.default_rng(42)
        # Create simple place fields with some structure
        models = rng.random((3, n_bins))
        return models

    def test_yields_correct_number_of_shuffles(
        self, encoding_models_2d: np.ndarray, env_2d
    ) -> None:
        """Should yield exactly n_shuffles arrays."""

        n_shuffles = 10
        shuffles = list(
            shuffle_place_fields_circular_2d(
                env_2d, encoding_models_2d, n_shuffles=n_shuffles, rng=42
            )
        )
        assert len(shuffles) == n_shuffles

    def test_preserves_row_values(self, encoding_models_2d: np.ndarray, env_2d) -> None:
        """Each row should contain same values (multiset) as original.

        2D circular shift preserves the shape of each place field.
        """

        for shuffled in shuffle_place_fields_circular_2d(
            env_2d, encoding_models_2d, n_shuffles=10, rng=42
        ):
            for i in range(encoding_models_2d.shape[0]):
                original_sorted = sorted(encoding_models_2d[i, :].tolist())
                shuffled_sorted = sorted(shuffled[i, :].tolist())
                assert shuffled_sorted == original_sorted

    def test_requires_2d_environment(self) -> None:
        """Should raise ValueError for non-2D environments."""
        from neurospatial import Environment

        # Create a 1D environment
        positions = np.linspace(0, 10, 50).reshape(-1, 1)
        env_1d = Environment.from_samples(positions, bin_size=1.0)
        encoding_models = np.random.default_rng(42).random((3, env_1d.n_bins))

        with pytest.raises(ValueError, match="2D"):
            list(
                shuffle_place_fields_circular_2d(
                    env_1d, encoding_models, n_shuffles=5, rng=42
                )
            )

    def test_rejects_masked_grid(self) -> None:
        """Should raise ValueError if environment has inactive bins (masked grid)."""
        from neurospatial import Environment

        # Create 2D environment with inactive bins (default behavior)
        positions = np.random.default_rng(42).uniform(0, 10, (100, 2))
        env = Environment.from_samples(positions, bin_size=2.0)
        # Create encoding models matching n_bins (not grid_shape product)
        encoding_models = np.random.default_rng(42).random((3, env.n_bins))

        # Should fail because n_bins != prod(grid_shape)
        with pytest.raises(ValueError, match="cells outside the arena") as error:
            list(
                shuffle_place_fields_circular_2d(
                    env, encoding_models, n_shuffles=5, rng=42
                )
            )
        # The 1-D shuffle would break the field on a masked grid; the message
        # points to the spike-time shuffle instead.
        assert "shuffle_spike_times_circular" in str(error.value)
        assert "shuffle_place_fields_circular " not in str(error.value)
        assert "shuffle_place_fields_circular for" not in str(error.value)


# =============================================================================
# Tests for shuffle_posterior_circular
# =============================================================================


class TestShufflePosteriorCircular:
    """Tests for shuffle_posterior_circular (circular shift of posterior rows)."""

    @pytest.fixture
    def posterior(self) -> np.ndarray:
        """Sample normalized posterior: (5 time bins, 8 spatial bins)."""
        rng = np.random.default_rng(42)
        raw = rng.random((5, 8))
        # Normalize each row to sum to 1
        return raw / raw.sum(axis=1, keepdims=True)

    def test_yields_correct_number_of_shuffles(self, posterior: np.ndarray) -> None:
        """Should yield exactly n_shuffles arrays."""

        n_shuffles = 10
        shuffles = list(
            shuffle_posterior_circular(
                track_for(posterior), posterior, n_shuffles=n_shuffles, rng=42
            )
        )
        assert len(shuffles) == n_shuffles

    def test_preserves_normalization(self, posterior: np.ndarray) -> None:
        """Each row should still sum to 1.0 after shuffle."""

        for shuffled in shuffle_posterior_circular(
            track_for(posterior), posterior, n_shuffles=10, rng=42
        ):
            row_sums = shuffled.sum(axis=1)
            assert np.allclose(row_sums, 1.0), f"Row sums should be 1.0, got {row_sums}"

    def test_preserves_row_values(self, posterior: np.ndarray) -> None:
        """Each row should contain same values (multiset) as original.

        Circular shift preserves the shape of each posterior.
        """

        for shuffled in shuffle_posterior_circular(
            track_for(posterior), posterior, n_shuffles=10, rng=42
        ):
            for i in range(posterior.shape[0]):
                original_sorted = sorted(posterior[i, :].tolist())
                shuffled_sorted = sorted(shuffled[i, :].tolist())
                assert np.allclose(shuffled_sorted, original_sorted), (
                    f"Row {i} values changed"
                )

    def test_rows_are_circular_shifts(self, posterior: np.ndarray) -> None:
        """Each row should be a circular shift of the original."""

        for shuffled in shuffle_posterior_circular(
            track_for(posterior), posterior, n_shuffles=10, rng=42
        ):
            for i in range(posterior.shape[0]):
                orig = posterior[i, :]
                shuf = shuffled[i, :]
                n_bins = len(orig)
                is_shift = any(
                    np.allclose(shuf, np.roll(orig, shift)) for shift in range(n_bins)
                )
                assert is_shift, f"Row {i} is not a circular shift"

    def test_each_time_bin_shifted_independently(self, posterior: np.ndarray) -> None:
        """Different time bins should get different shift amounts."""

        # With enough shuffles, different time bins should have different shifts
        # at least sometimes
        found_different = False
        for shuffled in shuffle_posterior_circular(
            track_for(posterior), posterior, n_shuffles=50, rng=42
        ):
            # Check shift amounts for each time bin
            shifts = []
            for i in range(posterior.shape[0]):
                orig = posterior[i, :]
                shuf = shuffled[i, :]
                # Find the shift amount
                for shift in range(len(orig)):
                    if np.allclose(shuf, np.roll(orig, shift)):
                        shifts.append(shift)
                        break
            # Check if different time bins got different shifts
            if len(set(shifts)) > 1:
                found_different = True
                break
        assert found_different, "Each time bin should be shifted independently"


# =============================================================================
# Tests for shuffle_posterior_weighted_circular
# =============================================================================


class TestShufflePosteriorWeightedCircular:
    """Tests for shuffle_posterior_weighted_circular (edge-aware circular shift)."""

    @pytest.fixture
    def posterior(self) -> np.ndarray:
        """Sample normalized posterior: (10 time bins, 20 spatial bins).

        Create a posterior with clear edge structure - higher probability
        mass near the center to test edge effect mitigation.
        """
        rng = np.random.default_rng(42)
        raw = rng.random((10, 20))
        # Add some center-biased structure
        center = 10
        for i in range(raw.shape[0]):
            raw[i, :] += np.exp(-((np.arange(20) - center) ** 2) / 20)
        # Normalize each row to sum to 1
        return raw / raw.sum(axis=1, keepdims=True)

    def test_yields_correct_number_of_shuffles(self, posterior: np.ndarray) -> None:
        """Should yield exactly n_shuffles arrays."""

        n_shuffles = 10
        shuffles = list(
            shuffle_posterior_weighted_circular(
                track_for(posterior), posterior, n_shuffles=n_shuffles, rng=42
            )
        )
        assert len(shuffles) == n_shuffles

    def test_preserves_normalization(self, posterior: np.ndarray) -> None:
        """Each row should still sum to 1.0 after shuffle."""

        for shuffled in shuffle_posterior_weighted_circular(
            track_for(posterior), posterior, n_shuffles=10, rng=42
        ):
            row_sums = shuffled.sum(axis=1)
            assert np.allclose(row_sums, 1.0), f"Row sums should be 1.0, got {row_sums}"

    def test_preserves_row_values(self, posterior: np.ndarray) -> None:
        """Each row should contain same values (multiset) as original."""

        for shuffled in shuffle_posterior_weighted_circular(
            track_for(posterior), posterior, n_shuffles=10, rng=42
        ):
            for i in range(posterior.shape[0]):
                original_sorted = sorted(posterior[i, :].tolist())
                shuffled_sorted = sorted(shuffled[i, :].tolist())
                assert np.allclose(shuffled_sorted, original_sorted), (
                    f"Row {i} values changed"
                )

    def test_rows_are_circular_shifts(self, posterior: np.ndarray) -> None:
        """Each row should be a circular shift of the original."""

        for shuffled in shuffle_posterior_weighted_circular(
            track_for(posterior), posterior, n_shuffles=10, rng=42
        ):
            for i in range(posterior.shape[0]):
                orig = posterior[i, :]
                shuf = shuffled[i, :]
                n_bins = len(orig)
                is_shift = any(
                    np.allclose(shuf, np.roll(orig, shift)) for shift in range(n_bins)
                )
                assert is_shift, f"Row {i} is not a circular shift"

    def test_edge_buffer_restricts_shifts(self, posterior: np.ndarray) -> None:
        """With edge_buffer, shifts near boundaries should be restricted.

        When the MAP position is near an edge, the shift should be limited
        to avoid wrapping probability mass to the other end.
        """

        # Create a posterior with clear edge positions
        # First row: peak at bin 2 (near left edge)
        # Last row: peak at bin 17 (near right edge with 20 bins)
        edge_posterior = np.zeros((3, 20), dtype=np.float64)
        edge_posterior[0, 2] = 0.8  # Near left edge (within edge_buffer=5)
        edge_posterior[1, 10] = 0.8  # Center (not restricted)
        edge_posterior[2, 17] = 0.8  # Near right edge (within edge_buffer=5)
        # Add small uniform noise and normalize
        edge_posterior += 0.01
        edge_posterior /= edge_posterior.sum(axis=1, keepdims=True)

        # With edge_buffer=5, time bins with MAP near edges should have
        # restricted shifts to avoid wrapping to the opposite side
        for shuffled in shuffle_posterior_weighted_circular(
            track_for(edge_posterior),
            edge_posterior,
            edge_buffer=5,
            n_shuffles=100,
            rng=42,
        ):
            assert shuffled.shape == edge_posterior.shape
            assert np.allclose(shuffled.sum(axis=1), 1.0)

            # Verify edge time bins don't wrap to opposite edge
            # Time bin 0: MAP=2 (near left), shouldn't wrap to far right
            map_0 = np.argmax(shuffled[0])
            assert map_0 < 15, (
                f"Left edge MAP shouldn't wrap to far right, got bin {map_0}"
            )

            # Time bin 2: MAP=17 (near right), shouldn't wrap to far left
            map_2 = np.argmax(shuffled[2])
            assert map_2 > 5, (
                f"Right edge MAP shouldn't wrap to far left, got bin {map_2}"
            )

    def test_edge_buffer_zero_allows_full_shifts(self, posterior: np.ndarray) -> None:
        """With edge_buffer=0, should behave like standard circular shuffle."""

        # With edge_buffer=0, any shift should be possible
        for shuffled in shuffle_posterior_weighted_circular(
            track_for(posterior), posterior, edge_buffer=0, n_shuffles=10, rng=42
        ):
            assert shuffled.shape == posterior.shape
            assert np.allclose(shuffled.sum(axis=1), 1.0)

    def test_edge_buffer_larger_than_half_bins(self) -> None:
        """Should handle edge_buffer larger than n_bins/2.

        When edge_buffer is very large relative to n_bins, the allowed
        shift range becomes very restricted.
        """

        # Small posterior (5 bins), large edge_buffer (4)
        small = np.array(
            [[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.2, 0.4, 0.2, 0.1]], dtype=np.float64
        )
        for shuffled in shuffle_posterior_weighted_circular(
            track_for(small), small, edge_buffer=4, n_shuffles=5, rng=42
        ):
            assert shuffled.shape == small.shape
            assert np.allclose(shuffled.sum(axis=1), 1.0)


# =============================================================================
# Circular shifts run along the track
# =============================================================================


class TestCircularShiftsRunAlongTrack:
    """Shifts follow bin positions along the track, not bin numbers."""

    @pytest.fixture
    def renumbered_track(self):
        from environment_helpers import renumbered, strips_1d

        env = strips_1d(np.arange(0.0, 101.0, 5.0))
        order = np.random.default_rng(3).permutation(env.n_bins)
        return env, renumbered(env, order), order

    @pytest.mark.parametrize(
        "name",
        [
            "shuffle_place_fields_circular",
            "shuffle_posterior_circular",
            "shuffle_posterior_weighted_circular",
        ],
    )
    def test_renumbering_bins_gives_the_same_shuffles(self, renumbered_track, name):
        import neurospatial.replay.nulls as shuffle

        env, shuffled_env, order = renumbered_track
        # A field over bins 3-7 (and a small background), normalized per row.
        values = np.full((4, env.n_bins), 0.01)
        values[:, 3:8] = [1.0, 3.0, 5.0, 3.0, 1.0]
        values /= values.sum(axis=1, keepdims=True)

        original = list(getattr(shuffle, name)(env, values, n_shuffles=5, rng=7))
        renumbered = list(
            getattr(shuffle, name)(shuffled_env, values[:, order], n_shuffles=5, rng=7)
        )

        for first, second in zip(original, renumbered, strict=True):
            np.testing.assert_array_equal(second, first[:, order])

    def test_field_stays_one_piece_on_a_renumbered_track(self, renumbered_track):
        _, shuffled_env, order = renumbered_track
        field = np.zeros((1, 20))
        field[0, 3:8] = [1.0, 3.0, 5.0, 3.0, 1.0]
        new_index = np.argsort(order)

        for shuffled in shuffle_place_fields_circular(
            shuffled_env, field[:, order], n_shuffles=20, rng=1
        ):
            along_track = shuffled[0, new_index]
            # Rolling along the track keeps the field's profile, cyclically.
            shift = int(np.argmax(along_track)) - 5
            np.testing.assert_array_equal(along_track, np.roll(field[0], shift))

    @pytest.mark.parametrize(
        "name",
        [
            "shuffle_place_fields_circular",
            "shuffle_posterior_circular",
            "shuffle_posterior_weighted_circular",
        ],
    )
    def test_open_arena_is_refused(self, name):
        import neurospatial.replay.nulls as shuffle
        from neurospatial import Environment

        theta = np.random.default_rng(0).uniform(0, 2 * np.pi, 4000)
        radius = 40 * np.sqrt(np.random.default_rng(1).uniform(0, 1, 4000))
        disc = Environment.from_samples(
            np.c_[radius * np.cos(theta), radius * np.sin(theta)], bin_size=4.0
        )
        values = np.full((2, disc.n_bins), 1.0 / disc.n_bins)

        with pytest.raises(ValueError, match="needs a position along a track"):
            getattr(shuffle, name)(disc, values, n_shuffles=1, rng=0)


class TestShuffleSpikeCountsCircular:
    """Each cell's counts rotate in time, independently of the other cells."""

    def test_each_cell_is_a_rotation_of_itself(self):
        from neurospatial.replay import shuffle_spike_counts_circular

        counts = np.random.default_rng(0).poisson(1.0, (12, 6))
        for shuffled in shuffle_spike_counts_circular(counts, n_shuffles=20, rng=1):
            assert shuffled.dtype == counts.dtype
            for cell in range(counts.shape[1]):
                rotations = [np.roll(counts[:, cell], k) for k in range(12)]
                assert any(np.array_equal(shuffled[:, cell], r) for r in rotations)

    def test_cells_shift_independently_and_uniformly(self):
        """Shifts of 0..n-1 are equally likely, and differ between cells."""
        from neurospatial.replay import shuffle_spike_counts_circular

        n_time = 5
        counts = np.zeros((n_time, 2), dtype=np.int64)
        counts[0] = 1  # a single spike marks each cell's shift
        shifts = np.array(
            [
                np.argmax(shuffled, axis=0)
                for shuffled in shuffle_spike_counts_circular(
                    counts, n_shuffles=5000, rng=2
                )
            ]
        )
        frequencies = np.bincount(shifts[:, 0], minlength=n_time) / len(shifts)
        assert np.allclose(frequencies, 1 / n_time, atol=0.03)
        assert np.mean(shifts[:, 0] == shifts[:, 1]) < 0.3  # ~1/5 if independent

    def test_refuses_one_dimensional_counts(self):
        from neurospatial.replay import shuffle_spike_counts_circular

        with pytest.raises(ValueError, match="2-D"):
            next(shuffle_spike_counts_circular(np.arange(5), n_shuffles=1))


def test_place_field_shift_skips_bins_without_a_rate():
    """NaN bins stay in place and fields move over the other bins only."""
    from neurospatial.replay import shuffle_place_fields_circular

    env = track_for(np.zeros((1, 10)))
    field = np.arange(1.0, 11.0)[None].repeat(2, axis=0)
    field[:, [0, 9]] = np.nan
    for shuffled in shuffle_place_fields_circular(env, field, n_shuffles=20, rng=1):
        assert np.all(np.isnan(shuffled[:, [0, 9]]))
        for row in shuffled:
            rotations = [np.roll(np.arange(2.0, 10.0), k) for k in range(8)]
            assert any(np.array_equal(row[1:9], r) for r in rotations)
