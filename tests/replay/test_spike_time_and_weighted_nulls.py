"""The spike-time-uniform and weighted column-cycle nulls of replay_significance."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import decode_position
from neurospatial.replay import (
    ReplayMethod,
    ReplaySegmentation,
    detect_replay,
    replay_significance,
    weighted_correlation,
)

DT = 0.02


@pytest.fixture(scope="module")
def place_code():
    """12 place cells on a 100 cm track in 10 cm bins."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=10.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    rates = 0.5 + 30.0 * np.exp(
        -0.5 * ((x - np.linspace(0.0, 100.0, 12)[:, None]) / 10.0) ** 2
    )
    return env, rates


@pytest.fixture(scope="module")
def sweep(place_code):
    """A forward sweep's counts, with two time bins without spikes."""
    _, rates = place_code
    rng = np.random.default_rng(5)
    counts = np.zeros((10, len(rates)), dtype=np.int64)
    for row in range(10):
        if row in (2, 6):
            continue
        counts[row, round(row * (len(rates) - 1) / 9)] = 1 + rng.poisson(1.0)
    return counts, np.arange(10) * DT


class _PeakRecorder:
    """A score that keeps each row's peak; the score itself is irrelevant."""

    def __init__(self):
        self.peaks = []

    def __call__(self, env, posterior, times):
        self.peaks.append(np.argmax(posterior, axis=1))
        return SimpleNamespace(score=float(np.sum(posterior[:, 0])))


@pytest.fixture
def decoded_counts(monkeypatch):
    """Every count matrix the nulls decode, in order."""
    from neurospatial.replay.significance import _EventDecoder

    seen = []
    decode, decode_each = _EventDecoder.decode, _EventDecoder.decode_each

    def spy(self, counts, models=None):
        seen.append(np.array(counts))
        return decode(self, counts, models)

    def spy_each(self, shuffled):
        for posterior, counts in decode_each(self, shuffled):
            seen.append(np.array(counts))
            yield posterior, counts

    monkeypatch.setattr(_EventDecoder, "decode", spy)
    monkeypatch.setattr(_EventDecoder, "decode_each", spy_each)
    return seen


class TestSpikeTimeUniform:
    """Stella et al. (2019): each cell's spikes redrawn uniformly in the event."""

    def test_each_shuffle_keeps_every_cells_count_and_moves_its_spikes(
        self, place_code, sweep, decoded_counts
    ):
        env, rates = place_code
        counts, times = sweep
        posterior = decode_position(env, counts, rates, dt=DT).posterior
        decoded_counts.clear()
        replay_significance(
            env,
            posterior,
            times,
            score=weighted_correlation,
            nulls="spike_time_uniform",
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            n_shuffles=50,
            rng=0,
        )
        # The first decode checks the inputs reproduce the posterior.
        reproduced, *shuffled = decoded_counts
        np.testing.assert_array_equal(reproduced, counts)
        assert len(shuffled) == 50
        for spread in shuffled:
            np.testing.assert_array_equal(spread.sum(axis=0), counts.sum(axis=0))
        # Spikes land in other time bins, also in bins that had none.
        assert sum(not np.array_equal(spread, counts) for spread in shuffled) == 50
        assert any(spread[[2, 6]].sum() > 0 for spread in shuffled)

    def test_p_values_are_calibrated_on_events_without_a_sequence(self, place_code):
        """Each cell fires at a constant rate through each event.

        Its spikes are then uniform over the event given its count, which is
        exactly this null, so p-values are uniform: about a tenth fall at or
        below 0.1 and half at or below 0.5.
        """
        env, rates = place_code
        rng = np.random.default_rng(11)
        p_values = []
        for _ in range(80):
            event_rates = rng.gamma(2.0, 4.0, size=len(rates))  # Hz
            counts = rng.poisson(event_rates * DT, size=(10, len(rates)))
            posterior = decode_position(env, counts, rates, dt=DT).posterior
            result = replay_significance(
                env,
                posterior,
                np.arange(10) * DT,
                score=weighted_correlation,
                nulls="spike_time_uniform",
                spike_counts=counts,
                encoding_models=rates,
                dt=DT,
                n_shuffles=39,
                rng=rng,
            )
            p_values.append(result.p_value)
        p_values = np.asarray(p_values)
        # Loose bounds: 80 events give a binomial standard error of 0.03-0.06.
        assert 0.0 < np.mean(p_values <= 0.1) <= 0.2
        assert 0.35 <= np.mean(p_values <= 0.5) <= 0.65

    def test_a_sequence_is_rejected(self, place_code, sweep):
        env, rates = place_code
        counts, times = sweep
        posterior = decode_position(env, counts, rates, dt=DT).posterior
        result = replay_significance(
            env,
            posterior,
            times,
            score=weighted_correlation,
            nulls="spike_time_uniform",
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            n_shuffles=99,
            rng=0,
        )
        assert result.p_value <= 0.02

    def test_each_shuffle_zeroes_its_own_empty_time_bins(
        self, place_code, sweep, decoded_counts
    ):
        env, rates = place_code
        counts, times = sweep
        posterior = decode_position(env, counts, rates, dt=DT).posterior
        empty_rows = []

        def record(env, posterior, times):
            empty_rows.append(posterior.sum(axis=1) == 0)
            return weighted_correlation(env, posterior, times, row_normalization="none")

        decoded_counts.clear()
        replay_significance(
            env,
            posterior,
            times,
            score=record,
            nulls="spike_time_uniform",
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            empty_time_bins="zero",
            use_absolute=True,
            n_shuffles=30,
            rng=0,
        )
        _, *shuffled_counts = decoded_counts
        observed, *shuffled = empty_rows
        np.testing.assert_array_equal(observed, counts.sum(axis=1) == 0)
        for zeroed, spread in zip(shuffled, shuffled_counts, strict=True):
            np.testing.assert_array_equal(zeroed, spread.sum(axis=1) == 0)
        assert len({tuple(zeroed) for zeroed in shuffled}) > 1

    def test_refuses_overlapping_time_bins(self, place_code, sweep):
        """20 ms windows every 10 ms share spikes, so counts cannot be redrawn."""
        env, rates = place_code
        counts, _ = sweep
        posterior = decode_position(env, counts, rates, dt=DT).posterior
        with pytest.raises(ValueError, match="time bins that do not overlap"):
            replay_significance(
                env,
                posterior,
                np.arange(10) * DT / 2,
                score=weighted_correlation,
                nulls="spike_time_uniform",
                spike_counts=counts,
                encoding_models=rates,
                dt=DT,
                n_shuffles=5,
            )

    @pytest.mark.parametrize("null", ["spike_time_uniform", "spike_train_circular"])
    def test_refuses_gaps_between_time_bins(self, place_code, sweep, null):
        """Spikes cannot be moved into the part of the event left out."""
        env, rates = place_code
        counts, times = sweep
        kept = np.flatnonzero(counts.sum(axis=1) > 0)  # empty time bins dropped
        posterior = decode_position(env, counts[kept], rates, dt=DT).posterior
        with pytest.raises(ValueError, match="cover the event without gaps"):
            replay_significance(
                env,
                posterior,
                times[kept],
                score=weighted_correlation,
                nulls=null,
                spike_counts=counts[kept],
                encoding_models=rates,
                dt=DT,
                n_shuffles=5,
            )

    @pytest.mark.parametrize("null", ["spike_time_uniform", "spike_train_circular"])
    def test_accepts_time_bins_with_round_off(self, place_code, sweep, null):
        """Times built as start + k * dt far from zero still tile the event."""
        env, rates = place_code
        counts, _ = sweep
        times = 5000.0 + np.arange(10) * DT
        assert np.ptp(np.diff(times)) > 0, "the spacing carries round-off"
        posterior = decode_position(env, counts, rates, dt=DT).posterior
        replay_significance(
            env,
            posterior,
            times,
            score=weighted_correlation,
            nulls=null,
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            n_shuffles=5,
            rng=0,
        )

    def test_refuses_spike_counts_in_score_options(self, place_code, sweep):
        from neurospatial.replay import fit_linear_trajectory

        env, rates = place_code
        counts, times = sweep
        posterior = decode_position(env, counts, rates, dt=DT).posterior
        with pytest.raises(ValueError, match="score_options holds spike_counts"):
            replay_significance(
                env,
                posterior,
                times,
                score=fit_linear_trajectory,
                score_options={"method": "map", "spike_counts": counts},
                nulls="spike_time_uniform",
                spike_counts=counts,
                encoding_models=rates,
                dt=DT,
                n_shuffles=5,
            )

    @pytest.mark.parametrize("null", ["spike_time_uniform", "spike_train_circular"])
    def test_a_method_refuses_dropping_empty_time_bins(self, null):
        """Spikes would be spread over the kept time bins, not the event."""
        with pytest.raises(ValueError, match="empty_time_bins='drop' cannot"):
            ReplayMethod(
                name="x",
                reference="y",
                score=weighted_correlation,
                nulls=(null,),
                empty_time_bins="drop",
            )

    @pytest.mark.parametrize("null", ["spike_time_uniform", "spike_train_circular"])
    @pytest.mark.parametrize("rows", ["segment", "trajectory"])
    def test_a_scored_method_refuses_a_segmentation(self, null, rows):
        """The score is tested on the segment, so spikes would move within it."""
        with pytest.raises(ValueError, match="segmentation cannot be used"):
            ReplayMethod(
                name="x",
                reference="y",
                score=weighted_correlation,
                nulls=("column_cycle", null),
                segmentation=ReplaySegmentation({"max_jump": 20.0}, rows=rows),
            )

    def test_a_segment_detection_test_may_use_them(self):
        """Without a score the shuffles are drawn over the whole event."""
        method = ReplayMethod(
            name="x",
            reference="y",
            score=None,
            nulls=("spike_time_uniform",),
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
        )
        assert method.nulls == ("spike_time_uniform",)


class TestSegmentDetectionWithCountChangingNulls:
    """Invalid time bins from spikes cannot stay put while the spikes move.

    A segment-detection test passes the event's valid time bins unchanged to
    every shuffle. Time bins invalid for having no spikes (or too few active
    cells) would then stay where the event had them, while the shuffle puts
    spikes there; such methods are refused.
    """

    @pytest.mark.parametrize(
        ("empty_time_bins", "cells"), [("zero", None), ("drop", None), ("keep", 2)]
    )
    def test_spike_derived_valid_time_bins_are_refused(self, empty_time_bins, cells):
        with pytest.raises(ValueError, match="move or change the spikes"):
            ReplayMethod(
                name="x",
                reference="y",
                score=None,
                nulls=("spike_time_uniform",),
                empty_time_bins=empty_time_bins,
                segmentation=ReplaySegmentation(
                    {"max_jump": 20.0}, min_active_cells_per_time_bin=cells
                ),
            )

    def test_given_valid_time_bins_are_refused(self, place_code, sweep):
        env, rates = place_code
        counts, times = sweep
        posterior = decode_position(env, counts, rates, dt=DT).posterior
        method = ReplayMethod(
            name="x",
            reference="y",
            score=None,
            nulls=("spike_time_uniform",),
            segmentation=ReplaySegmentation({"max_jump": 20.0}),
        )
        with pytest.raises(ValueError, match="valid_time_bins cannot be used"):
            detect_replay(
                env,
                posterior,
                times,
                replay_method=method,
                spike_counts=counts,
                encoding_models=rates,
                dt=DT,
                valid_time_bins=np.ones(10, dtype=bool),
            )


class TestColumnCycleWeighted:
    """Farooq et al. (2019): peaks drawn from the session's mean posterior."""

    @pytest.fixture
    def peaked(self, place_code):
        env, _ = place_code
        posterior = np.full((6, env.n_bins), 0.02)
        posterior[np.arange(6), [0, 2, 4, 5, 7, 9]] = 1.0
        return posterior / posterior.sum(axis=1, keepdims=True)

    def _peaks(self, env, posterior, pool, n_shuffles, **kwargs):
        recorder = _PeakRecorder()
        replay_significance(
            env,
            posterior,
            np.arange(len(posterior)) * DT,
            score=recorder,
            nulls="column_cycle_weighted",
            pseudo_event_pool=pool,
            n_shuffles=n_shuffles,
            rng=0,
            **kwargs,
        )
        _, *shuffled = recorder.peaks
        return np.concatenate(shuffled)

    def test_peaks_follow_the_pools_mean_posterior(self, place_code, peaked):
        """A pool whose mass is skewed toward the far end of the track."""
        env, _ = place_code
        rng = np.random.default_rng(4)
        skew = np.linspace(0.0, 1.0, env.n_bins) ** 2
        pool = rng.dirichlet(50.0 * skew + 0.01, size=200)
        peaks = self._peaks(env, peaked, pool, n_shuffles=1000)
        frequencies = np.bincount(peaks, minlength=env.n_bins) / peaks.size
        # 6,000 draws: each frequency's standard error is below 0.007.
        np.testing.assert_allclose(frequencies, pool.mean(axis=0), atol=0.025)

    def test_peaks_never_land_within_the_edge_exclusion(self, place_code, peaked):
        """Bin centers at 5, 15, ..., 95 cm: 25 cm leaves out two at each end."""
        env, _ = place_code
        pool = np.full((3, env.n_bins), 1.0 / env.n_bins)
        peaks = self._peaks(
            env,
            peaked,
            pool,
            n_shuffles=300,
            null_options={"column_cycle_weighted": {"edge_exclusion": 25.0}},
        )
        assert set(peaks.tolist()) == set(range(2, 8))

    def test_needs_a_pool(self, place_code, peaked):
        env, _ = place_code
        with pytest.raises(ValueError, match="column_cycle_weighted needs"):
            replay_significance(
                env,
                peaked,
                np.arange(6) * DT,
                nulls="column_cycle_weighted",
                n_shuffles=5,
            )

    def test_a_method_forwards_its_null_options(self, place_code, peaked):
        """An exclusion that leaves no weighted bin is refused through detect_replay."""
        env, _ = place_code
        pool = np.zeros((2, env.n_bins))
        pool[:, 0] = 1.0  # all of the session's mass at the first bin
        method = ReplayMethod(
            name="farooq_like",
            reference="test",
            score=weighted_correlation,
            nulls=("column_cycle_weighted",),
            null_options={"column_cycle_weighted": {"edge_exclusion": 10.0}},
            n_shuffles=5,
        )
        with pytest.raises(ValueError, match="has weight"):
            detect_replay(
                env,
                peaked,
                np.arange(6) * DT,
                replay_method=method,
                pseudo_event_pool=pool,
            )
        # Without the exclusion the same method runs.
        unrestricted = ReplayMethod(
            name="farooq_like",
            reference="test",
            score=weighted_correlation,
            nulls=("column_cycle_weighted",),
            n_shuffles=5,
        )
        result = detect_replay(
            env,
            peaked,
            np.arange(6) * DT,
            replay_method=unrestricted,
            pseudo_event_pool=pool,
            rng=0,
        )
        assert np.isfinite(result.p_value)


class TestNullOptions:
    """null_options are checked against the nulls run and the options they take."""

    @pytest.mark.parametrize(
        ("null_options", "match"),
        [
            ({"column_cycle": {"edge_exclusion": 10.0}}, "not among the nulls run"),
            ({"no_such_null": {}}, "not among the nulls run"),
            (
                {"column_cycle_weighted": {"edge_buffer": 5}},
                r"takes \['edge_exclusion'\]",
            ),
            ({"column_cycle_weighted": {"edge_exclusion": -1.0}}, "finite distance"),
            ({"column_cycle_weighted": {"edge_exclusion": True}}, "finite distance"),
            ({"column_cycle_weighted": 10.0}, "must map null names to mappings"),
        ],
    )
    def test_replay_significance_refuses_bad_options(
        self, place_code, null_options, match
    ):
        env, _ = place_code
        posterior = np.eye(env.n_bins)[[1, 2, 3, 4]]
        with pytest.raises(ValueError, match=match):
            replay_significance(
                env,
                posterior,
                np.arange(4) * DT,
                nulls="column_cycle_weighted",
                null_options=null_options,
                pseudo_event_pool=posterior,
                n_shuffles=5,
            )

    def test_options_of_a_null_without_options_are_refused(self):
        with pytest.raises(ValueError, match="it takes no options"):
            ReplayMethod(
                name="x",
                reference="y",
                score=weighted_correlation,
                nulls=("time_bin_permutation",),
                null_options={"time_bin_permutation": {"edge_exclusion": 10.0}},
            )

    def test_a_method_refuses_options_of_nulls_it_does_not_run(self):
        with pytest.raises(ValueError, match="not among the nulls run"):
            ReplayMethod(
                name="x",
                reference="y",
                score=weighted_correlation,
                nulls=("column_cycle",),
                null_options={"column_cycle_weighted": {"edge_exclusion": 10.0}},
            )

    def test_a_methods_options_are_frozen_and_pickle(self):
        options = {"edge_exclusion": 10.0}
        method = ReplayMethod(
            name="x",
            reference="y",
            score=weighted_correlation,
            nulls=("column_cycle_weighted",),
            null_options={"column_cycle_weighted": options},
        )
        options["edge_exclusion"] = 0.0  # the caller's dict is copied
        assert method.null_options["column_cycle_weighted"]["edge_exclusion"] == 10.0
        with pytest.raises(TypeError):
            method.null_options["column_cycle_weighted"]["edge_exclusion"] = 0.0
        with pytest.raises(TypeError):
            method.null_options["column_cycle"] = {}
        clone = pickle.loads(pickle.dumps(method))
        assert clone == method
        assert dict(clone.null_options["column_cycle_weighted"]) == {
            "edge_exclusion": 10.0
        }
