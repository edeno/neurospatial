"""Tests for the replay shuffles in neurospatial.replay.nulls.

The seed and degenerate-input contract shared by every shuffle generator is in
``tests/stats/test_shuffle.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from environment_helpers import track_for
from numpy.testing import assert_allclose, assert_array_equal

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
    """Farooq et al. (2019): each row's peak lands at a weighted position."""

    @pytest.fixture
    def peaked(self) -> np.ndarray:
        """Rows with one clear peak each, at different positions, on 20 bins."""
        rows = np.full((5, 20), 0.01)
        rows[np.arange(5), [0, 4, 9, 15, 19]] = 0.5
        return rows / rows.sum(axis=1, keepdims=True)

    def test_rows_are_circular_shifts_along_the_track(self, peaked):
        shuffles = list(
            shuffle_posterior_weighted_circular(
                track_for(peaked), peaked, np.ones(20), n_shuffles=10, rng=42
            )
        )
        assert len(shuffles) == 10
        for shuffled in shuffles:
            for row, original in zip(shuffled, peaked, strict=True):
                assert any(
                    np.array_equal(row, np.roll(original, shift)) for shift in range(20)
                )

    def test_peaks_are_drawn_from_the_weights(self, peaked):
        """A skewed weight vector: peak frequencies match its normalized mass."""
        weights = np.linspace(0.0, 1.0, 20) ** 2
        n_shuffles = 2000
        peaks = np.concatenate(
            [
                np.argmax(shuffled, axis=1)
                for shuffled in shuffle_posterior_weighted_circular(
                    track_for(peaked), peaked, weights, n_shuffles=n_shuffles, rng=1
                )
            ]
        )
        frequencies = np.bincount(peaks, minlength=20) / peaks.size
        expected = weights / weights.sum()
        # 10,000 draws: the standard error of each frequency is below 0.005.
        np.testing.assert_allclose(frequencies, expected, atol=0.015)
        assert frequencies[0] == 0.0  # a bin without weight is never a peak

    def test_peaks_never_land_within_the_edge_exclusion(self, peaked):
        """1 cm bins: an exclusion of 3 cm leaves out three bins at each end."""
        peaks = set()
        for shuffled in shuffle_posterior_weighted_circular(
            track_for(peaked),
            peaked,
            np.ones(20),
            edge_exclusion=3.0,
            n_shuffles=200,
            rng=2,
        ):
            peaks |= set(np.argmax(shuffled, axis=1).tolist())
        assert peaks == set(range(3, 17))

    def test_edge_exclusion_is_a_distance_along_the_track(self, peaked):
        """The same exclusion leaves out fewer bins of a longer bin size."""
        from neurospatial import Environment

        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [40.0, 0.0]], bin_size=2.0, units="cm"
        )
        peaks = set()
        for shuffled in shuffle_posterior_weighted_circular(
            env, peaked, np.ones(20), edge_exclusion=3.0, n_shuffles=200, rng=2
        ):
            peaks |= set(np.argmax(shuffled, axis=1).tolist())
        # Centers at 1, 3, 5, ... cm: only the first and last are within 3 cm.
        assert peaks == set(range(1, 19))

    def test_a_row_without_mass_stays_zero(self, peaked):
        posterior = peaked.copy()
        posterior[2] = 0.0
        for shuffled in shuffle_posterior_weighted_circular(
            track_for(posterior), posterior, np.ones(20), n_shuffles=5, rng=0
        ):
            np.testing.assert_array_equal(shuffled[2], 0.0)

    @pytest.mark.parametrize(
        ("weights", "edge_exclusion", "match"),
        [
            (np.ones(19), 0.0, "one finite, non-negative weight per bin"),
            (np.r_[-1.0, np.ones(19)], 0.0, "one finite, non-negative"),
            (np.r_[np.nan, np.ones(19)], 0.0, "one finite, non-negative"),
            (np.ones(20), -1.0, "edge_exclusion must be a finite distance"),
            (np.r_[1.0, np.zeros(18), 1.0], 2.0, "has weight"),
            (np.ones(20), 10.0, "has weight"),
        ],
    )
    def test_refuses_weights_without_a_landing_bin(
        self, peaked, weights, edge_exclusion, match
    ):
        with pytest.raises(ValueError, match=match):
            shuffle_posterior_weighted_circular(
                track_for(peaked),
                peaked,
                weights,
                edge_exclusion=edge_exclusion,
                n_shuffles=1,
            )


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

        # The weighted shift also takes peak weights and an edge exclusion,
        # both of which must follow the bins along the track.
        weights = np.linspace(1.0, 2.0, env.n_bins)
        weighted = name == "shuffle_posterior_weighted_circular"
        options = {"edge_exclusion": 12.0} if weighted else {}

        def draw(on_env, rows, peak_weights):
            extra = (peak_weights,) if weighted else ()
            return list(
                getattr(shuffle, name)(
                    on_env, rows, *extra, n_shuffles=5, rng=7, **options
                )
            )

        original = draw(env, values, weights)
        renumbered = draw(shuffled_env, values[:, order], weights[order])

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

        extra = (
            (np.ones(disc.n_bins),)
            if name == "shuffle_posterior_weighted_circular"
            else ()
        )
        with pytest.raises(ValueError, match="needs a position along a track"):
            getattr(shuffle, name)(disc, values, *extra, n_shuffles=1, rng=0)


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


class TestShuffleSpikeCountsUniform:
    """Each cell's spikes spread uniformly over the event (Stella et al. 2019)."""

    def test_each_cell_keeps_its_count_in_the_event(self):
        from neurospatial.replay import shuffle_spike_counts_uniform

        counts = np.random.default_rng(0).poisson(1.0, (12, 6))
        for shuffled in shuffle_spike_counts_uniform(counts, n_shuffles=20, rng=1):
            assert shuffled.dtype == counts.dtype
            np.testing.assert_array_equal(shuffled.sum(axis=0), counts.sum(axis=0))

    def test_counts_are_multinomial_over_the_time_bins(self):
        """Seven spikes in ten bins: mean 0.7, variance 0.63, covariance -0.07.

        All seven spikes start in one bin, so a shuffle that kept the bins
        (or only rotated them) would have a variance of 4.41 and no spikes
        in nine bins out of ten.
        """
        from neurospatial.replay import shuffle_spike_counts_uniform

        counts = np.zeros((10, 1), dtype=np.int64)
        counts[3] = 7
        draws = np.array(
            [
                shuffled[:, 0]
                for shuffled in shuffle_spike_counts_uniform(
                    counts, n_shuffles=20000, rng=2
                )
            ]
        )
        np.testing.assert_allclose(draws.mean(axis=0), 0.7, atol=0.03)
        np.testing.assert_allclose(draws.var(axis=0), 0.63, atol=0.04)
        covariance = np.cov(draws[:, 0], draws[:, 1])[0, 1]
        assert covariance == pytest.approx(-0.07, abs=0.02)

    def test_cells_are_spread_independently(self):
        """Two cells with all spikes in one bin end up in different bins."""
        from neurospatial.replay import shuffle_spike_counts_uniform

        counts = np.zeros((8, 2), dtype=np.int64)
        counts[0] = 1
        same = [
            np.argmax(shuffled[:, 0]) == np.argmax(shuffled[:, 1])
            for shuffled in shuffle_spike_counts_uniform(counts, n_shuffles=4000, rng=3)
        ]
        # Independent cells share a bin with probability 1/8.
        assert np.mean(same) == pytest.approx(1 / 8, abs=0.02)

    def test_refuses_counts_that_are_not_a_count_matrix(self):
        from neurospatial.replay import shuffle_spike_counts_uniform

        with pytest.raises(ValueError, match="must be 2-D"):
            next(shuffle_spike_counts_uniform(np.ones(5, dtype=int), n_shuffles=1))
        with pytest.raises(ValueError, match="non-negative"):
            next(shuffle_spike_counts_uniform(-np.ones((3, 2), dtype=int)))


class TestCellIdentityActiveOnly:
    """Bush et al. (2022) permuted only the cells active in the event."""

    def test_silent_cells_keep_their_columns(self):
        rng = np.random.default_rng(0)
        counts = rng.poisson(1.0, size=(12, 8))
        counts[:, [1, 4, 6]] = 0  # silent in this event
        rates = np.ones((8, 5))
        active = np.array([0, 2, 3, 5, 7])
        moved = set()
        for shuffled, _ in shuffle_cell_identity(
            counts, rates, active_only=True, n_shuffles=200, rng=1
        ):
            assert (shuffled[:, [1, 4, 6]] == 0).all()
            sources = [
                int(np.flatnonzero((counts == shuffled[:, [j]]).all(axis=0))[0])
                for j in active
            ]
            assert sorted(sources) == list(active)
            moved.add(tuple(sources))
        assert len(moved) > 50  # 5! = 120 distinct permutations

    def test_every_cell_moves_by_default(self):
        counts = np.zeros((4, 6), dtype=np.int64)
        counts[:, 0] = 3
        rates = np.ones((6, 5))
        landed = {
            int(np.flatnonzero(shuffled.sum(axis=0))[0])
            for shuffled, _ in shuffle_cell_identity(
                counts, rates, n_shuffles=100, rng=0
            )
        }
        assert len(landed) > 3
        landed_active = {
            int(np.flatnonzero(shuffled.sum(axis=0))[0])
            for shuffled, _ in shuffle_cell_identity(
                counts, rates, active_only=True, n_shuffles=100, rng=0
            )
        }
        assert landed_active == {0}


def test_unit_identity_null_options_are_checked():
    from neurospatial.replay import ReplayMethod, weighted_correlation

    method = ReplayMethod(
        name="active_shuffle",
        reference="test",
        score=weighted_correlation,
        nulls=("unit_identity",),
        null_options={"unit_identity": {"active_only": True}},
    )
    assert method.null_options["unit_identity"]["active_only"] is True
    with pytest.raises(ValueError, match="active_only of unit_identity"):
        ReplayMethod(
            name="bad",
            reference="test",
            score=weighted_correlation,
            nulls=("unit_identity",),
            null_options={"unit_identity": {"active_only": "yes"}},
        )


# ---------------------------------------------------------------------------
# Moved from the stats tests: replay shuffles.
# ---------------------------------------------------------------------------


def test_shuffle_cell_identity_preserves_counts():
    """Test shuffle_cell_identity preserves spike counts."""
    spike_counts = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    encoding_models = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    for shuffled, models in shuffle_cell_identity(
        spike_counts, encoding_models, n_shuffles=5, rng=42
    ):
        assert shuffled.sum() == spike_counts.sum()
        assert models is encoding_models  # Same object


def test_shuffle_place_fields_circular_preserves_shape():
    """Test shuffle_place_fields_circular preserves field shapes."""
    encoding_models = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])

    for shuffled in shuffle_place_fields_circular(
        track_for(encoding_models), encoding_models, n_shuffles=5, rng=42
    ):
        assert shuffled.shape == encoding_models.shape
        # Each row should have same values, just shifted
        for i in range(encoding_models.shape[0]):
            assert set(shuffled[i]) == set(encoding_models[i])


def test_shuffle_posterior_circular_preserves_normalization():
    """Test shuffle_posterior_circular preserves row normalization."""
    raw = np.array(
        [
            [0.1, 0.2, 0.4, 0.2, 0.1],
            [0.5, 0.3, 0.1, 0.05, 0.05],
            [0.05, 0.1, 0.2, 0.4, 0.25],
        ]
    )

    for shuffled in shuffle_posterior_circular(
        track_for(raw), raw, n_shuffles=5, rng=42
    ):
        assert_allclose(shuffled.sum(axis=1), 1.0)


def test_shuffle_posterior_weighted_circular_preserves_normalization():
    """Test shuffle_posterior_weighted_circular preserves row normalization."""
    raw = np.random.default_rng(42).random((3, 10))
    posterior = raw / raw.sum(axis=1, keepdims=True)

    for shuffled in shuffle_posterior_weighted_circular(
        track_for(posterior),
        posterior,
        posterior.mean(axis=0),
        edge_exclusion=2.0,
        n_shuffles=5,
        rng=42,
    ):
        assert_allclose(shuffled.sum(axis=1), 1.0)


def test_cell_identity_preserves_marginal_and_destroys_pairing() -> None:
    from neurospatial.replay.nulls import shuffle_cell_identity

    rng = np.random.default_rng(0)
    n_neurons = 20
    spike_counts = rng.poisson(3.0, (100, n_neurons)).astype(np.int64)
    encoding_models = rng.uniform(0.0, 10.0, (n_neurons, 30))
    original_column_sums = np.sort(spike_counts.sum(axis=0))

    moved = 0
    n_shuffles = 200
    for shuffled, _ in shuffle_cell_identity(
        spike_counts, encoding_models, n_shuffles=n_shuffles, rng=42
    ):
        # Marginal preserved: the multiset of per-neuron column totals is
        # unchanged (columns are permuted, not altered).
        assert np.array_equal(np.sort(shuffled.sum(axis=0)), original_column_sums)
        # Pairing destroyed: the spike-train column now paired with place
        # field 0 differs from the original neuron-0 column.
        if not np.array_equal(shuffled[:, 0], spike_counts[:, 0]):
            moved += 1

    # For a random permutation of N columns, P(column 0 fixed) = 1/N, so the
    # fraction moved should be ~1 - 1/20 = 0.95; require >= 0.9.
    assert moved / n_shuffles >= 0.9


def test_shuffle_cell_identity_neuron_count_mismatch_raises():
    """Mismatched neuron counts raise before yielding any shuffle."""
    spike_counts = np.zeros((5, 3), dtype=np.int64)
    encoding_models = np.zeros((4, 7), dtype=np.float64)
    with pytest.raises(ValueError, match=r"3 neuron.*4"):
        next(shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=2))


def test_shuffle_cell_identity_matched_ok():
    """Matched shapes still yield the right number/shape of arrays."""
    spike_counts = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    encoding_models = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    results = list(
        shuffle_cell_identity(spike_counts, encoding_models, n_shuffles=3, rng=42)
    )
    assert len(results) == 3
    for shuffled, models in results:
        assert shuffled.shape == spike_counts.shape
        assert models is encoding_models


@pytest.mark.parametrize("n_shuffles", [0, -3, 2.0, True])
def test_spike_count_nulls_refuse_a_draw_count_that_yields_nothing(n_shuffles):
    """A non-positive or non-integer count is refused, not an empty null."""
    from neurospatial.replay import (
        shuffle_spike_counts_circular,
        shuffle_spike_counts_uniform,
    )

    counts = np.ones((3, 2), dtype=np.int64)
    for shuffle in (shuffle_spike_counts_circular, shuffle_spike_counts_uniform):
        with pytest.raises(ValueError, match="n_shuffles must be a positive integer"):
            list(shuffle(counts, n_shuffles=n_shuffles, rng=0))
