"""Replay scoring of events whose score or shuffles are undefined.

Covers what :func:`compare_replay_methods` and :func:`detect_replay` do with
an undefined (NaN) event score, undefined shuffle scores, counts summed over
cells, a per-time-bin prior, and malformed per-event inputs.
"""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.decoding import DecodingResult, decode_jointly, decode_position
from neurospatial.replay import (
    ReplayMethod,
    ReplaySegmentation,
    compare_replay_methods,
    decode_intervals,
    detect_replay,
    event_properties,
    fit_linear_trajectory,
    replay_significance,
    simulated_null_events,
    weighted_correlation,
)
from neurospatial.replay.correlation import _weighted_correlations
from neurospatial.replay.methods import (
    CHENANI_2019,
    KARLSSON_FRANK_2009,
    MICHON_2019,
    OLAFSDOTTIR_2017,
    SILVA_2015,
    TIROLE_2022,
    WIDLOSKI_FOSTER_2022,
    WU_2017,
    ReplayCriterion,
)

DT = 0.02
N_CELLS = 20


@pytest.fixture(scope="module")
def track():
    """A 100 cm track with 20 place cells evenly along it."""
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    centers = np.linspace(2.5, 97.5, N_CELLS)[:, None]
    rates = 0.5 + 20.0 * np.exp(-0.5 * ((x - centers) / 6.0) ** 2)
    return env, rates


def _event(env, rates, counts, t0=0.0, prior=None):
    counts = np.asarray(counts)
    times = t0 + DT * (np.arange(len(counts)) + 0.5)
    return decode_position(env, counts, rates, dt=DT, times=times, prior=prior)


def _sweep_counts(n_time=10):
    counts = np.zeros((n_time, N_CELLS), dtype=np.int64)
    for t in range(n_time):
        cell = int(t * N_CELLS / n_time)
        counts[t, cell] = 2
        counts[t, min(cell + 1, N_CELLS - 1)] = 1
    return counts


def _stationary_counts(n_time=10):
    """Five neighboring cells fire the same way in every time bin.

    The MAP position never changes, so its R² is undefined, while the event
    has enough active cells for the methods' cell criteria.
    """
    counts = np.zeros((n_time, N_CELLS), dtype=np.int64)
    counts[:, 8:13] = [1, 2, 4, 2, 1]
    return counts


def _few(method, n_shuffles=20):
    return dataclasses.replace(method, n_shuffles=n_shuffles)


class TestUndefinedEventScore:
    def test_detect_replay_gives_no_verdict(self, track):
        env, rates = track
        counts = _stationary_counts()
        event = _event(env, rates, counts)
        result = detect_replay(
            env,
            event.posterior,
            event.times,
            replay_method=_few(WU_2017),
            spike_counts=counts,
            rng=0,
        )
        assert not result.is_replay
        assert np.isnan(result.score) and np.isnan(result.p_value)
        assert not any(result.criteria.values())
        assert np.isnan(result.summary()["n_shuffles_used"])

    def test_compare_records_an_error_row_instead_of_raising(self, track):
        env, rates = track
        good_counts, still_counts = _sweep_counts(), _stationary_counts()
        events = [
            _event(env, rates, good_counts),
            _event(env, rates, still_counts, t0=1.0),
        ]
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[_few(WU_2017), _few(MICHON_2019)],
            spike_counts=[good_counts, still_counts],
            rng=0,
        )
        still = table[table["event"] == 1].set_index("method")
        assert set(still["error"]) == {
            "score undefined (e.g. the decoded position does not change)"
        }
        assert not still["is_replay"].any()
        assert still["p_value"].isna().all()
        criterion = "passes[n_active_cells >= 4]"
        assert still.loc["wu_2017", criterion] is False
        good = table[table["event"] == 0]
        assert (good["error"] == "").all() and good["score"].notna().all()

    def test_stationary_null_events_do_not_abort(self, track):
        # Stationary events: in many, the MAP bin never changes, so R² and the
        # isotonic fit are undefined. They are no-verdict rows, not an abort.
        env = Environment.linear_track(
            endpoints=[[0.0, 0.0], [200.0, 0.0]], bin_size=10.0, units="cm"
        )
        x = env.bin_centers[:, 0]
        rates = 0.1 + 20.0 * np.exp(
            -0.5 * ((x - np.linspace(5, 195, 40)[:, None]) / 10.0) ** 2
        )
        null = simulated_null_events(
            env, rates, n_events=12, kind="stationary", rate_gain=8.0, rng=0
        )
        events, counts = decode_intervals(
            env, null.spike_times, null.intervals, rates, dt=DT
        )
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[_few(WU_2017), _few(MICHON_2019)],
            spike_counts=counts,
            rng=0,
        )
        assert len(table) == 2 * len(events)
        undefined = table["error"].str.startswith("score undefined")
        assert undefined.any()
        assert table.loc[undefined, "score"].isna().all()
        assert not table.loc[undefined, "is_replay"].any()

    def test_replay_significance_names_the_score_that_is_undefined(self, track):
        env, rates = track
        event = _event(env, rates, _stationary_counts())
        with pytest.raises(ValueError, match="fit_linear_trajectory is undefined"):
            replay_significance(
                env,
                event.posterior,
                event.times,
                score=fit_linear_trajectory,
                score_options={"method": "map"},
                nulls="time_bin_permutation",
                n_shuffles=10,
                rng=0,
            )

    def test_no_segment_stays_a_verdict(self, track):
        env, _ = track
        # A posterior that jumps between the track's ends: no trajectory.
        posterior = np.eye(env.n_bins)[[0, env.n_bins - 1] * 10]
        event = DecodingResult(
            posterior=posterior, env=env, times=np.arange(20) * 0.005
        )
        method = _few(WIDLOSKI_FOSTER_2022)
        table = compare_replay_methods(
            env, [event], replay_methods=[method], spike_counts=None, rng=0
        )
        row = table.iloc[0]
        assert row["error"] == "" and row["n_segments"] == 0
        assert not row["is_replay"] and np.isnan(row["score"])


class TestWeightedCorrelationDegeneracy:
    @pytest.mark.parametrize("t0", [0.0, 1.0, 37.3, 3600.123])
    @pytest.mark.parametrize("row_normalization", ["per_time_bin", "none"])
    def test_mass_in_one_time_bin_is_undefined(self, track, t0, row_normalization):
        env, _ = track
        rng = np.random.default_rng(1)
        times = t0 + DT * (np.arange(10) + 0.5)
        for row in range(10):
            posterior = np.zeros((10, env.n_bins))
            posterior[row] = rng.random(env.n_bins)
            score = weighted_correlation(
                env, posterior, times, row_normalization=row_normalization
            ).score
            assert np.isnan(score), (row, score)

    @pytest.mark.parametrize("t0", [0.0, 3600.123])
    @pytest.mark.parametrize("row_normalization", ["per_time_bin", "none"])
    def test_mass_at_one_position_is_undefined(self, track, t0, row_normalization):
        env, _ = track
        rng = np.random.default_rng(2)
        times = t0 + DT * (np.arange(10) + 0.5)
        for column in range(env.n_bins):
            posterior = np.zeros((10, env.n_bins))
            posterior[:, column] = rng.random(10)
            score = weighted_correlation(
                env, posterior, times, row_normalization=row_normalization
            ).score
            assert np.isnan(score), (column, score)

    def test_small_real_variation_is_kept(self, track):
        env, _ = track
        posterior = np.zeros((10, env.n_bins))
        posterior[4, 2] = 1.0
        posterior[5, 3] = 1e-6  # little mass, but a real second time bin
        times = 3600.0 + DT * np.arange(10)
        assert weighted_correlation(
            env, posterior, times, row_normalization="none"
        ).score == pytest.approx(1.0)

    def test_batch_matches_single_scores(self, track):
        env, _ = track
        rng = np.random.default_rng(3)
        times = 100.0 + DT * np.arange(8)
        batch = rng.random((6, 8, env.n_bins))
        batch[0] = 0.0
        batch[0, 2] = rng.random(env.n_bins)  # one time bin: undefined
        positions = env.bin_centers[:, 0]
        batched = _weighted_correlations(batch, times, positions)
        single = [
            weighted_correlation(env, weights, times, row_normalization="none").score
            for weights in batch
        ]
        np.testing.assert_array_equal(batched, single)
        assert np.isnan(batched[0]) and np.isfinite(batched[1:]).all()

    def test_one_spiking_time_bin_with_zeroed_bins_is_an_error_row(self, track):
        env, rates = track
        good_counts = _sweep_counts()
        one_bin = np.zeros((10, N_CELLS), dtype=np.int64)
        one_bin[9, 13:18] = 1
        events = [_event(env, rates, good_counts), _event(env, rates, one_bin, 1.0)]
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[_few(TIROLE_2022)],
            spike_counts=[good_counts, one_bin],
            encoding_models=rates,
            dt=DT,
            rng=0,
        )
        assert table.loc[1, "error"] == "fewer than two time bins with spikes to score"
        assert np.isnan(table.loc[1, "score"]) and not table.loc[1, "is_replay"]
        assert table.loc[0, "error"] == ""


class TestSummedCounts:
    def test_detect_replay_refuses_summed_counts_when_counting_cells(self, track):
        env, rates = track
        counts = _sweep_counts()
        event = _event(env, rates, counts)
        with pytest.raises(ValueError, match="cannot count active cells"):
            detect_replay(
                env,
                event.posterior,
                event.times,
                replay_method=_few(KARLSSON_FRANK_2009),
                spike_counts=counts.sum(axis=1),
                rng=0,
            )

    def test_event_properties_refuses_summed_counts(self, track):
        env, rates = track
        counts = _sweep_counts()
        event = _event(env, rates, counts)
        with pytest.raises(ValueError, match="cannot count active cells"):
            event_properties(env, [event], spike_counts=[counts.sum(axis=1)])


class TestTimeVaryingPrior:
    def test_prior_is_cut_with_the_dropped_time_bins(self, track):
        env, rates = track
        rng = np.random.default_rng(0)
        counts = rng.poisson(0.6, size=(12, N_CELLS))
        counts[3] = 0  # dropped by the method
        stationary = np.ones(env.n_bins)
        per_time_bin = np.ones((12, env.n_bins))
        method = _few(OLAFSDOTTIR_2017)
        results = []
        for prior in (stationary, per_time_bin):
            event = _event(env, rates, counts, prior=prior)
            results.append(
                detect_replay(
                    env,
                    event.posterior,
                    event.times,
                    replay_method=method,
                    spike_counts=counts,
                    encoding_models=rates,
                    dt=DT,
                    prior=prior,
                    rng=0,
                )
            )
        # The same prior in every row: the same test, row for row.
        assert results[1].p_value == results[0].p_value
        assert results[1].score == results[0].score

    def test_the_kept_rows_of_a_varying_prior_are_used(self, track):
        """Another cut would not reproduce the event's posterior, and raise."""
        env, rates = track
        rng = np.random.default_rng(1)
        counts = rng.poisson(0.6, size=(12, N_CELLS))
        counts[3] = 0
        prior = rng.uniform(0.2, 1.0, size=(12, env.n_bins))
        event = _event(env, rates, counts, prior=prior)
        result = detect_replay(
            env,
            event.posterior,
            event.times,
            replay_method=_few(OLAFSDOTTIR_2017),
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            prior=prior,
            rng=0,
        )
        assert np.isfinite(result.p_value)

    def test_the_prior_is_cut_to_the_tested_segment(self, track):
        env, rates = track
        rng = np.random.default_rng(2)
        counts = np.vstack([rng.poisson(0.6, size=(3, N_CELLS)), _sweep_counts(10)])
        prior = rng.uniform(0.2, 1.0, size=(len(counts), env.n_bins))
        event = _event(env, rates, counts, prior=prior)
        method = ReplayMethod(
            name="segment_then_shuffle",
            reference="test",
            score=weighted_correlation,
            nulls=("place_field_circular",),
            n_shuffles=30,
            segmentation=ReplaySegmentation(
                options={"max_jump": 20.0, "min_time_bins": 5}
            ),
        )
        result = detect_replay(
            env,
            event.posterior,
            event.times,
            replay_method=method,
            spike_counts=counts,
            encoding_models=rates,
            dt=DT,
            prior=prior,
            rng=0,
        )
        assert result.segment is not None
        rows = result.segments.rows(result.segment)
        assert rows.stop - rows.start < len(counts)
        assert np.isfinite(result.p_value)

    def test_a_mapping_of_priors_is_cut_for_a_joint_decode(self, track):
        env, rates = track
        rng = np.random.default_rng(3)
        counts = rng.poisson(0.6, size=(12, N_CELLS))
        counts[3] = 0
        maps = {"out": rates, "in": rates[::-1]}
        priors = {name: rng.uniform(0.2, 1.0, size=(12, env.n_bins)) for name in maps}
        times = DT * (np.arange(12) + 0.5)
        joint = decode_jointly(env, counts, maps, dt=DT, times=times, prior=priors)
        # A share of the joint posterior, scored as Tirole et al. (2022) did.
        method = ReplayMethod(
            name="joint_share",
            reference="test",
            score=weighted_correlation,
            score_options={"row_normalization": "none"},
            nulls=("place_field_circular",),
            n_shuffles=30,
            empty_time_bins="drop",
        )
        result = detect_replay(
            env,
            joint,
            times,
            replay_method=method,
            spike_counts=counts,
            encoding_models=maps,
            dt=DT,
            prior=priors,
            template="out",
            rng=0,
        )
        assert np.isfinite(result.p_value)


def _parity_score(env, posterior, times):
    """A score that is undefined when the first row peaks at an even bin."""
    first = int(np.argmax(posterior[0]))
    score = np.nan if first % 2 == 0 else float(np.argmax(posterior[-1]) - first)
    return SimpleNamespace(score=score, summary=lambda: {"score": score})


class TestUndefinedShuffles:
    @pytest.fixture
    def posterior(self, track):
        env, _ = track
        return np.eye(env.n_bins)[[1, 3, 5, 7, 9, 11, 13, 15]]

    def test_counts_used_shuffles_and_names_the_null(self, track, posterior):
        env, _ = track
        times = np.arange(len(posterior)) * DT
        with pytest.warns(UserWarning, match="column_cycle shuffles have an undefined"):
            result = replay_significance(
                env,
                posterior,
                times,
                score=_parity_score,
                nulls="column_cycle",
                n_shuffles=200,
                rng=0,
            )
        test = result.shuffles["column_cycle"]
        finite = np.isfinite(test.null_scores)
        used = result.n_shuffles_used["column_cycle"]
        assert 0 < used < 200 and used == finite.sum()
        assert result.summary()["column_cycle_n_shuffles_used"] == used
        assert np.isfinite(test.z_score)
        observed = test.observed_score
        expected = (np.sum(test.null_scores[finite] >= observed) + 1) / (used + 1)
        assert test.p_value == pytest.approx(expected)

    def test_compare_reports_used_shuffles_and_context(self, track, posterior):
        env, _ = track
        times = np.arange(len(posterior)) * DT
        method = ReplayMethod(
            name="parity",
            reference="test",
            score=_parity_score,
            nulls=("column_cycle",),
            n_shuffles=200,
        )
        event = DecodingResult(posterior=posterior, env=env, times=times)
        with pytest.warns(UserWarning, match=r"event 0, method 'parity'.*column_cycle"):
            table = compare_replay_methods(
                env, [event], replay_methods=[method, SILVA_2015], rng=0
            )
        rows = table.set_index("method")
        assert 0 < rows.loc["parity", "n_shuffles_used"] < 200
        assert np.isnan(rows.loc["silva_2015", "n_shuffles_used"])


class TestCompareInputs:
    def test_error_rows_fail_every_criterion(self, track):
        env, rates = track
        good_counts = _sweep_counts()
        one_bin = np.zeros((10, N_CELLS), dtype=np.int64)
        one_bin[4, [3, 4, 5, 6, 7]] = 2
        events = [_event(env, rates, good_counts), _event(env, rates, one_bin, 1.0)]
        table = compare_replay_methods(
            env,
            events,
            replay_methods=[_few(KARLSSON_FRANK_2009)],
            spike_counts=[good_counts, one_bin],
            rng=0,
        )
        column = "passes[n_active_cells >= 5]"
        assert table.loc[1, "error"] != ""
        assert table.loc[1, column] is False
        assert table[column].astype(bool).tolist() == [True, False]

    def test_transposed_counts_are_refused(self, track):
        env, rates = track
        counts = _sweep_counts()
        event = _event(env, rates, counts)
        with pytest.raises(ValueError, match=r"spike_counts\[0\] has shape \(20, 10\)"):
            compare_replay_methods(
                env,
                [event],
                replay_methods=[_few(KARLSSON_FRANK_2009)],
                spike_counts=[counts.T],
                rng=0,
            )

    def test_valid_time_bins_are_checked_for_every_method(self, track):
        env, rates = track
        event = _event(env, rates, _sweep_counts())
        with pytest.raises(ValueError, match="event 0: valid_time_bins"):
            compare_replay_methods(
                env,
                [event],
                replay_methods=[SILVA_2015],
                valid_time_bins=[np.ones(3, dtype=bool)],
            )

    def test_rng_entropy_reproduces_an_unseeded_run(self, track):
        env, rates = track
        counts = [_sweep_counts(), _sweep_counts()[::-1].copy()]
        events = [_event(env, rates, c, t0=float(i)) for i, c in enumerate(counts)]
        method = _few(KARLSSON_FRANK_2009, n_shuffles=30)
        first = compare_replay_methods(
            env, events, replay_methods=[method], spike_counts=counts
        )
        again = compare_replay_methods(
            env,
            events,
            replay_methods=[method],
            spike_counts=counts,
            rng=first.attrs["rng_entropy"],
        )
        assert first["p_value"].tolist() == again["p_value"].tolist()
        assert first["score"].tolist() == again["score"].tolist()


class TestPooledThreshold:
    def test_threshold_beyond_the_grid_is_explained(self, track):
        env, _ = track
        # 40 cells in field order fire in order; with a tiny alpha the pooled
        # 1 - alpha quantile lies above the grid's end at SSI 4.
        n_cells = 40
        x = env.bin_centers[:, 0]
        rates = 20.0 * np.exp(
            -0.5 * ((x - np.linspace(0.0, 100.0, n_cells)[:, None]) / 4.0) ** 2
        )
        spikes = [np.array([0.01 + 0.002 * i]) for i in range(n_cells)]
        event = DecodingResult(
            posterior=np.full((5, env.n_bins), 1.0 / env.n_bins),
            env=env,
            times=0.01 + DT * np.arange(5),
        )
        method = dataclasses.replace(
            CHENANI_2019,
            name="chenani_tiny_alpha",
            alpha=1e-9,
            n_shuffles=200_000,
            min_epoch_events=None,
        )
        table = compare_replay_methods(
            env,
            [event],
            replay_methods=[method],
            spike_times=spikes,
            encoding_models=rates,
            dt=DT,
            rng=0,
        )
        row = table.iloc[0]
        assert "above 4, the end of the threshold grid" in row["error"]
        assert np.isfinite(row["score"]) and np.isfinite(row["p_value"])
        assert row["n_shuffles_used"] == 200_000
        assert not row["is_replay"]


class TestCustomScoreSignature:
    def test_unreadable_signature_is_refused(self, track):
        env, _ = track

        class Unreadable:
            __signature__ = "not a signature"

            def __call__(self, env, posterior, times):
                return SimpleNamespace(score=1.0)

        posterior = np.eye(env.n_bins)[[1, 3, 5, 7]]
        with pytest.raises(TypeError, match="cannot read the signature"):
            replay_significance(
                env,
                posterior,
                np.arange(4) * DT,
                score=Unreadable(),
                nulls="column_cycle",
                n_shuffles=5,
                rng=0,
            )


def test_no_warnings_leak_without_undefined_shuffles(track):
    env, rates = track
    counts = _sweep_counts()
    event = _event(env, rates, counts)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_replay_methods(
            env,
            [event],
            replay_methods=[_few(KARLSSON_FRANK_2009)],
            spike_counts=[counts],
            rng=0,
        )


class TestScopedNoVerdict:
    """An undefined score withholds the verdict only where the verdict uses it."""

    def test_criteria_that_do_not_read_the_score_still_decide(self, track):
        env, rates = track
        event = _event(env, rates, _stationary_counts())
        method = ReplayMethod(
            name="stays_put",
            reference="test",
            score=fit_linear_trajectory,
            score_options={"method": "map"},
            criteria=(ReplayCriterion("max_map_jump", "<", 5.0),),
        )
        result = detect_replay(env, event.posterior, event.times, replay_method=method)
        assert np.isnan(result.score)
        assert result.is_replay
        thresholded = dataclasses.replace(
            method,
            name="thresholded",
            criteria=(*method.criteria, ReplayCriterion("score", ">", 0.5)),
        )
        table = compare_replay_methods(
            env, [event], replay_methods=[method, thresholded], rng=0
        ).set_index("method")
        assert table.loc["stays_put", "error"] == ""
        assert table.loc["stays_put", "is_replay"]
        assert table.loc["thresholded", "error"].startswith("score undefined")


def _defined_only_unshuffled(original):
    def score(env, posterior, times):
        value = 0.5 if np.array_equal(posterior, original) else np.nan
        return SimpleNamespace(score=value, summary=lambda: {"score": value})

    return score


class TestUndefinedNull:
    def test_a_null_without_defined_shuffles_is_an_error_row(self, track):
        env, rates = track
        event = _event(env, rates, _sweep_counts())
        method = ReplayMethod(
            name="custom",
            reference="test",
            score=_defined_only_unshuffled(event.posterior),
            nulls=("column_cycle",),
            n_shuffles=20,
        )
        result = detect_replay(
            env, event.posterior, event.times, replay_method=method, rng=0
        )
        assert result.score == 0.5 and np.isnan(result.p_value)
        assert not result.is_replay
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            row = compare_replay_methods(
                env, [event], replay_methods=[method], rng=0
            ).iloc[0]
        assert row["error"] == "every shuffle of a null has an undefined score"


def test_n_shuffles_used_is_the_smallest_over_the_nulls(track, monkeypatch):
    from neurospatial.replay import significance

    env, rates = track
    event = _event(env, rates, _sweep_counts())

    def fake(*args, **kwargs):
        return SimpleNamespace(
            observed=SimpleNamespace(score=0.9),
            p_value=0.01,
            n_shuffles_used={"column_cycle": 10, "pseudo_event": 7},
        )

    monkeypatch.setattr(significance, "replay_significance", fake)
    method = ReplayMethod(
        name="two_nulls",
        reference="test",
        score=weighted_correlation,
        nulls=("column_cycle", "pseudo_event"),
        n_shuffles=10,
    )
    result = detect_replay(
        env,
        event.posterior,
        event.times,
        replay_method=method,
        pseudo_event_pool=event.posterior,
        rng=0,
    )
    assert result.n_shuffles_used == 7


def test_a_score_without_a_readable_signature_is_refused_when_made():
    with pytest.raises(TypeError, match="cannot read the signature"):
        ReplayMethod(
            name="builtin", reference="test", score=max, nulls=("column_cycle",)
        )
