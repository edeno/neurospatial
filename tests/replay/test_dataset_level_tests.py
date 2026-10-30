"""Tests for data-set-level replay tests (neurospatial.replay.dataset)."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pandas as pd
import pytest
from replay_helpers import sweeps_and_jumps
from scipy import stats

from neurospatial import Environment
from neurospatial.decoding import DecodingResult
from neurospatial.replay import (
    CriterionGrid,
    ReplayCriterion,
    ReplayMethod,
    compare_replay_methods,
    dataset_test,
    dataset_values,
    distribution_test,
    rank_order_significance,
    relative_rank_template,
    shuffle_events,
    shuffle_posterior_circular,
    template_correlation_test,
)
from neurospatial.replay.dataset import (
    DATASET_RECIPES,
    FAROOQ_2019_TIME_BIN,
    LIU_2019_TEMPLATES,
    SILVA_2015_MATRIX,
    SILVA_2015_RANK_ORDER,
    _dataset_values,
)
from neurospatial.replay.methods import (
    FAROOQ_2019,
    GROSMARK_2016,
    SILVA_2015,
)

DT = 0.02


@pytest.fixture(scope="module")
def track():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )


@pytest.fixture(scope="module")
def session(track):
    """Twelve simulated events, sweeps and random jumps, decoded in 20 ms bins."""
    return sweeps_and_jumps(track, dt=DT)


def _private_values(env, events, method, *, fast, **options):
    defaults = {
        "null": None,
        "n_datasets": 1,
        "shared": False,
        "quantities": (),
        "null_options": None,
        "spike_counts": None,
        "encoding_models": None,
        "dt": None,
        "prior": None,
        "template": None,
        "pseudo_event_pool": None,
        "spike_times": None,
        "rng": None,
    }
    return _dataset_values(env, events, method, fast=fast, **{**defaults, **options})


class TestDatasetTest:
    def test_reproduces_a_hand_built_grid_count_and_p(self):
        """Shares, counts and p of a 2 x 2 grid, checked by hand."""
        observed = pd.DataFrame(
            {
                "score": [0.9, -0.8, 0.65, 0.2, np.nan],
                "max_map_jump": [10.0, 30.0, 10.0, 5.0, 5.0],
                "track_length": 100.0,
            }
        )
        # Null data set k: its events' (|score|, jump).
        null_rows = {
            0: [(0.9, 10.0), (0.1, 10.0), (0.1, 50.0), (0.1, 5.0), (0.1, 5.0)],
            1: [(0.9, 10.0), (0.9, 10.0), (0.7, 10.0), (0.1, 5.0), (0.1, 5.0)],
            2: [(0.95, 25.0), (0.5, 10.0), (0.2, 10.0), (0.1, 5.0), (0.1, 5.0)],
        }
        null = pd.DataFrame(
            [
                {"dataset": k, "score": s, "max_map_jump": j, "track_length": 100.0}
                for k, rows in null_rows.items()
                for s, j in rows
            ]
        )
        grid = [
            CriterionGrid("score", ">", [0.6, 0.7], absolute=True),
            CriterionGrid("max_map_jump", "<", [0.2, 0.4], relative_to="track_length"),
        ]
        result = dataset_test(observed, null, criteria=grid)

        def count(rows, r, jump):
            return sum(abs(s) > r and j < jump * 100.0 for s, j in rows)

        observed_rows = list(
            zip(observed["score"], observed["max_map_jump"], strict=True)
        )
        for i, r in enumerate([0.6, 0.7]):
            for j, jump in enumerate([0.2, 0.4]):
                c_obs = count(observed_rows, r, jump)
                c_null = [count(rows, r, jump) for rows in null_rows.values()]
                assert result.statistic[i, j] == c_obs / 5
                np.testing.assert_array_equal(
                    result.null_statistics[:, i, j], np.array(c_null) / 5
                )
                k = sum(c >= c_obs for c in c_null)
                assert result.p_value[i, j] == pytest.approx((k + 1) / 4)
                assert result.percentile[i, j] == pytest.approx(
                    100 * sum(c < c_obs for c in c_null) / 3
                )
        # (|r| > 0.6, jump < 0.4): 3 observed events; data sets 0-2 give 1, 3, 1.
        assert result.statistic[0, 1] == 0.6
        assert result.p_value[0, 1] == pytest.approx(2 / 4)
        assert result.labels == ("|score| >", "max_map_jump < x track_length")
        assert result.grid_table().shape == (4, 6)

    def test_p_value_is_silvas_n_plus_one_rule(self):
        """p = (n + 1) / (r + 1), n counting null data sets at least as large."""
        observed = pd.DataFrame({"score": [1.0] * 3 + [0.0] * 7})
        passing = np.array([3, 3, 2, 4, 1, 0, 3, 2, 5, 1])  # three ties, two above
        null = pd.DataFrame(
            {
                "dataset": np.repeat(np.arange(10), 10),
                "score": np.concatenate(
                    [[1.0] * n + [0.0] * (10 - n) for n in passing]
                ),
            }
        )
        result = dataset_test(
            observed, null, criteria=[CriterionGrid("score", ">", 0.5)]
        )
        n = int(np.sum(passing >= 3))
        assert n == 5
        assert result.p_value == (n + 1) / (10 + 1)
        assert result.statistic == 0.3
        assert SILVA_2015_MATRIX.paper_p_value(result) == result.p_value
        less = dataset_test(
            observed,
            null,
            criteria=[CriterionGrid("score", ">", 0.5)],
            alternative="less",
        )
        assert less.p_value == (np.sum(passing <= 3) + 1) / 11
        two = dataset_test(
            observed,
            null,
            criteria=[CriterionGrid("score", ">", 0.5)],
            alternative="two-sided",
        )
        assert two.p_value == min(1.0, 2 * min(result.p_value, less.p_value))

    def test_shares_of_data_sets_of_different_sizes_compare_exactly(self):
        """1 of 3 and 2 of 6 are a tie, though 1/3 and 2/6 round alike anyway."""
        observed = pd.DataFrame({"score": [1.0, 0.0, 0.0]})
        null = pd.DataFrame(
            {
                "dataset": [0] * 6 + [1] * 9,
                "score": [1.0, 1.0] + [0.0] * 4 + [1.0] * 2 + [0.0] * 7,
            }
        )
        result = dataset_test(
            observed, null, criteria=[CriterionGrid("score", ">", 0.5)]
        )
        np.testing.assert_allclose(result.null_statistics, [2 / 6, 2 / 9])
        assert result.p_value == 2 / 3  # data set 0 ties, data set 1 is below

    @pytest.mark.parametrize(
        ("recipe", "expected"),
        [(FAROOQ_2019_TIME_BIN, 0.024), (LIU_2019_TEMPLATES, 0.024)],
    )
    def test_percentile_rules(self, recipe, expected):
        """Farooq's 1 - #(lower)/n and Liu's percentile: 24 of 1,000 at least as large."""
        observed = pd.DataFrame({"score": [0.5]})
        values = np.r_[np.full(976, 0.1), np.full(4, 0.5), np.full(20, 0.9)]
        null = pd.DataFrame({"dataset": np.arange(1000), "score": values})
        result = dataset_test(observed, null, statistic=lambda t: t["score"].mean())
        assert result.percentile == pytest.approx(97.6)
        assert recipe.paper_p_value(result) == pytest.approx(expected)
        # The library's rule has the + 1: 25 / 1,001.
        assert result.p_value == pytest.approx(25 / 1001)
        two_tailed = dataclasses.replace(recipe, p_value_rule="percentile_two_tailed")
        assert two_tailed.paper_p_value(result) == pytest.approx(2 * expected)

    def test_refuses_an_undefined_null_statistic(self):
        """Five null data sets whose events have no score: no mean to compare.

        Leaving them out of both counts would give p = 1/2 where the four
        defined data sets alone give 1.
        """
        observed = pd.DataFrame({"score": [0.3, 0.4, 0.5]})
        null = pd.DataFrame(
            {
                "dataset": np.repeat(np.arange(9), 3),
                "score": np.r_[np.full(15, np.nan), np.full(12, 0.9)],
            }
        )
        with pytest.raises(ValueError, match="not finite for 5 of 9 null data sets"):
            dataset_test(observed, null, statistic=lambda t: t["score"].mean())

    @pytest.mark.parametrize("kind", ["criteria", "statistic"])
    def test_refuses_missing_data_set_labels(self, kind):
        """Rows without a data set would form an extra null data set."""
        observed = pd.DataFrame({"score": [0.9, 0.1, 0.1]})
        null = pd.DataFrame(
            {
                "dataset": [0, 0, 0, 1, 1, 1, np.nan, np.nan, np.nan],
                "score": [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            }
        )
        options = (
            {"criteria": [CriterionGrid("score", ">", 0.5)]}
            if kind == "criteria"
            else {"statistic": lambda t: (t["score"] > 0.5).mean()}
        )
        with pytest.raises(ValueError, match="3 rows of null_table have no dataset"):
            dataset_test(observed, null, **options)

    def test_statistic_and_criteria_are_exclusive(self):
        table = pd.DataFrame({"dataset": [0], "score": [1.0]})
        with pytest.raises(ValueError, match="exactly one of criteria and statistic"):
            dataset_test(table, table)
        with pytest.raises(ValueError, match="no 'dataset' column"):
            dataset_test(table, table.drop(columns="dataset"), statistic=len)
        with pytest.raises(ValueError, match="track_length"):
            dataset_test(
                table,
                table,
                criteria=[CriterionGrid("score", "<", 0.5, relative_to="track_length")],
            )

    def test_null_data_sets_are_calibrated(self, track):
        """An observed data set drawn from the null is one more null data set.

        The observed events are sweeps column-cycled once by
        ``shuffle_posterior_circular``; the null data sets are their
        column cycles by ``dataset_values``. Observed and null data sets are
        then exchangeable, so p is uniform on 1/100, ..., 1 for the mean
        |r| and at most uniform for the share passing |r| > 0.5. Over 200
        replicates the share of p <= 0.1 lies within its 99.9% binomial
        range, (0.04, 0.18).
        """
        rng = np.random.default_rng(5)
        times = np.arange(10) * DT
        sweeps = [np.eye(track.n_bins)[start + np.arange(10)] for start in range(0, 10)]
        mean_p, share_p = [], []
        for _ in range(200):
            seed = int(rng.integers(2**32))
            observed_events = [
                DecodingResult(
                    posterior=next(
                        shuffle_posterior_circular(track, sweep, n_shuffles=1, rng=rng)
                    ),
                    env=track,
                    times=times,
                )
                for sweep in sweeps
            ]
            base = [DecodingResult(posterior=s, env=track, times=times) for s in sweeps]
            observed = dataset_values(track, observed_events, replay_method=SILVA_2015)
            null = dataset_values(
                track,
                base,
                replay_method=SILVA_2015,
                null="column_cycle",
                n_datasets=99,
                rng=seed,
            )
            mean_p.append(
                dataset_test(
                    observed, null, statistic=lambda table: table["score"].abs().mean()
                ).p_value
            )
            share_p.append(
                dataset_test(
                    observed,
                    null,
                    criteria=[CriterionGrid("score", ">", 0.5, absolute=True)],
                ).p_value
            )
        mean_p, share_p = np.asarray(mean_p), np.asarray(share_p)
        assert 0.04 < np.mean(mean_p <= 0.1) < 0.18
        assert 0.4 < mean_p.mean() < 0.6
        assert np.mean(share_p <= 0.1) < 0.18

    def test_sequences_are_detected(self, track, session):
        """Workflow: Silva's matrix on simulated sweeps rejects at high |r|."""
        _, _, _, events, _counts = session
        sweeps = events[1::2]
        real = dataset_values(track, sweeps, replay_method=SILVA_2015)
        null = dataset_values(
            track,
            sweeps,
            replay_method=SILVA_2015,
            null="column_cycle",
            n_datasets=99,
            rng=0,
        )
        result = dataset_test(real, null, criteria=SILVA_2015_MATRIX.criteria)
        assert result.grid_shape == (10, 10)
        # |r| > 0.6 with jumps below 0.4 of the track: Silva's trio region.
        assert result.p_value[6, 3] == pytest.approx(0.01)
        # Every event of every data set passes |r| > 0 and jumps below the
        # whole track: a tie, p = 1.
        assert result.statistic[0, -1] == 1.0
        assert result.p_value[0, -1] == 1.0


class TestDatasetValues:
    def test_real_values_match_compare_replay_methods(self, track, session):
        _, _, _, events, counts = session
        real = dataset_values(
            track, events, replay_method=SILVA_2015, spike_counts=counts
        )
        table = compare_replay_methods(
            track, events, replay_methods=[SILVA_2015], spike_counts=counts
        )
        assert (real["dataset"] == 0).all()
        np.testing.assert_array_equal(real["score"], table["score"])
        np.testing.assert_array_equal(
            real["max_map_jump"], table["value[max_map_jump < 0.4 x track_length]"]
        )
        assert (real["track_length"] == 100.0).all()
        assert (real["n_active_cells"] > 0).all()

    @pytest.mark.parametrize(
        ("null", "null_options"),
        [
            ("column_cycle", None),
            ("column_cycle", {"column_cycle": {"exclude_zero_shift": True}}),
            (
                "column_cycle_weighted",
                {"column_cycle_weighted": {"edge_exclusion": 10.0}},
            ),
            ("time_bin_permutation", None),
        ],
    )
    def test_moment_tables_equal_measuring_each_shuffle(
        self, track, session, null, null_options
    ):
        """The fast path gives detect_replay's values on the same shuffles."""
        _, _, _, events, counts = session
        # An event whose decoded position never changes: score undefined.
        frozen = DecodingResult(
            posterior=np.tile(np.eye(track.n_bins)[4], (6, 1)),
            env=track,
            times=np.arange(6) * DT,
        )
        events = [*events, frozen]
        counts = [*counts, np.ones((6, 20), dtype=np.int64)]
        quantities = (
            "max_map_jump",
            "map_path_length",
            "start_to_end_map_distance",
            "map_range",
            "mean_peak_probability",
            "duration",
        )
        options = {
            "null": null,
            "n_datasets": 40,
            "quantities": quantities,
            "null_options": null_options,
            "spike_counts": counts,
            "rng": 3,
        }
        fast = _private_values(track, events, SILVA_2015, fast=True, **options)
        slow = _private_values(track, events, SILVA_2015, fast=False, **options)
        assert list(fast.columns) == list(slow.columns)
        for column in fast.columns:
            a = fast[column].to_numpy(dtype=float)
            b = slow[column].to_numpy(dtype=float)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=column)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-9, err_msg=column)
        if null == "time_bin_permutation":
            # The frozen event's permuted rows are all alike: no score, and
            # no criterion value of a method that thresholds the score.
            frozen_rows = fast[fast["event"] == len(events) - 1]
            assert frozen_rows["score"].isna().all()
            assert frozen_rows["max_map_jump"].isna().all()
            assert frozen_rows["n_active_cells"].eq(20).all()

    def test_fast_path_on_unequal_bins(self):
        """MAP bins by density and track order hold on bins of unequal length."""
        env = Environment.linear_track(
            node_positions=[
                [0.0, 0.0],
                [40.0, 0.0],
                [45.0, 0.0],
                [75.0, 0.0],
                [77.0, 0.0],
            ],
            bin_size=10.0,
            units="cm",
        )
        assert np.ptp(env.bin_sizes) > 0
        rng = np.random.default_rng(1)
        events = []
        for _ in range(8):
            raw = rng.gamma(0.4, size=(7, env.n_bins))
            events.append(
                DecodingResult(
                    posterior=raw / raw.sum(1, keepdims=True),
                    env=env,
                    times=np.arange(7) * DT,
                )
            )
        options = {"null": "column_cycle", "n_datasets": 30, "rng": 0}
        fast = _private_values(env, events, SILVA_2015, fast=True, **options)
        slow = _private_values(env, events, SILVA_2015, fast=False, **options)
        np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-9)

    @pytest.mark.parametrize(
        "null", ["column_cycle", "column_cycle_weighted", "time_bin_permutation"]
    )
    def test_fast_path_weighs_every_time_bin_alike(self, track, null):
        """Rows summing to 1 within DecodingResult's tolerance, not exactly.

        The weighted correlation normalizes each row, so every time bin
        weighs the same; weighing by the row's own mass moved the score by
        about 1e-7.
        """
        rng = np.random.default_rng(0)
        events = []
        for _ in range(4):
            raw = rng.gamma(0.4, size=(8, track.n_bins))
            raw /= raw.sum(axis=1, keepdims=True)
            raw *= 1.0 + rng.uniform(-9e-7, 9e-7, size=(8, 1))
            events.append(
                DecodingResult(posterior=raw, env=track, times=np.arange(8) * DT)
            )
        options = {"null": null, "n_datasets": 25, "rng": 5}
        fast = _private_values(track, events, SILVA_2015, fast=True, **options)
        slow = _private_values(track, events, SILVA_2015, fast=False, **options)
        np.testing.assert_allclose(fast["score"], slow["score"], rtol=0, atol=1e-11)

    def test_each_event_draws_its_own_stream(self, track, session):
        """An event's shuffles do not depend on the other events."""
        _, _, _, events, _ = session
        everyone = dataset_values(
            track,
            events,
            replay_method=SILVA_2015,
            null="column_cycle",
            n_datasets=5,
            rng=7,
        )
        alone = dataset_values(
            track,
            events[:3],
            replay_method=SILVA_2015,
            null="column_cycle",
            n_datasets=5,
            rng=7,
        )
        pd.testing.assert_frame_equal(everyone[everyone["event"] < 3], alone)
        assert everyone.attrs["rng_entropy"] == alone.attrs["rng_entropy"]

    def test_shuffle_events_gives_the_same_shuffles(self, track, session):
        _, _, _, events, counts = session
        values = dataset_values(
            track,
            events,
            replay_method=SILVA_2015,
            null="time_bin_permutation",
            n_datasets=3,
            spike_counts=counts,
            rng=11,
        )
        shuffled = shuffle_events(
            track,
            events,
            null="time_bin_permutation",
            n_datasets=3,
            spike_counts=counts,
            rng=11,
        )
        for k, data in enumerate(shuffled):
            measured = dataset_values(
                track,
                data.events,
                replay_method=SILVA_2015,
                spike_counts=data.spike_counts,
            )
            expected = values[values["dataset"] == k].reset_index(drop=True)
            np.testing.assert_allclose(
                measured["score"], expected["score"], rtol=0, atol=1e-9
            )
            # Counts rows move with the posterior rows.
            for event, swapped, swapped_counts, original in zip(
                events, data.events, data.spike_counts, counts, strict=True
            ):
                order = [
                    int(np.flatnonzero((event.posterior == row).all(axis=1))[0])
                    for row in swapped.posterior
                ]
                np.testing.assert_array_equal(swapped_counts, original[order])

    @pytest.mark.parametrize("null", ["unit_identity", "place_field_circular"])
    def test_shared_template_draws_are_one_per_data_set(self, track, session, null):
        """Two copies of an event get the same values within a shared data set."""
        rates, _, _, events, counts = session
        pair = [events[1], events[1]]
        pair_counts = [counts[1], counts[1]]
        method = dataclasses.replace(GROSMARK_2016, n_shuffles=10)
        common = {
            "null": null,
            "n_datasets": 8,
            "spike_counts": pair_counts,
            "encoding_models": rates,
            "dt": DT,
            "rng": 0,
        }
        shared = dataset_values(
            track, pair, replay_method=method, shared=True, **common
        )
        apart = dataset_values(track, pair, replay_method=method, **common)
        by_event = shared.pivot(index="dataset", columns="event", values="score")
        np.testing.assert_array_equal(by_event[0], by_event[1])
        by_event = apart.pivot(index="dataset", columns="event", values="score")
        assert not np.array_equal(by_event[0], by_event[1])

    def test_shared_is_refused_for_nulls_that_act_on_each_event(self, track, session):
        _, _, _, events, _ = session
        with pytest.raises(ValueError, match="shared=True cannot be used"):
            dataset_values(
                track,
                events,
                replay_method=SILVA_2015,
                null="column_cycle",
                shared=True,
            )
        with pytest.raises(ValueError, match="null_options needs a null"):
            dataset_values(
                track,
                events,
                replay_method=SILVA_2015,
                null_options={"column_cycle": {"exclude_zero_shift": True}},
            )
        with pytest.raises(ValueError, match="not scalars of TrajectoryMetrics"):
            dataset_values(
                track, events, replay_method=SILVA_2015, quantities=("speed",)
            )

    @pytest.mark.parametrize("null", [None, "time_bin_permutation", "column_cycle"])
    def test_count_criteria_need_spike_counts(self, track, session, null):
        """FAROOQ_2019 counts active cells: no counts is an error, not a MAP range.

        With counts, every data set reports the event's active cells, on the
        moment-table path and when each shuffle is measured alike.
        """
        _, _, _, events, counts = session
        options = {} if null is None else {"null": null, "n_datasets": 3, "rng": 0}
        with pytest.raises(ValueError, match="needs spike_counts"):
            dataset_values(
                track,
                events,
                replay_method=FAROOQ_2019,
                quantities=("max_map_jump",),
                **options,
            )
        table = dataset_values(
            track,
            events,
            replay_method=FAROOQ_2019,
            quantities=("max_map_jump",),
            spike_counts=counts,
            **options,
        )
        active = np.array([np.sum(c.sum(axis=0) > 0) for c in counts], dtype=float)
        np.testing.assert_array_equal(table["n_active_cells"], active[table["event"]])

    def test_decoding_nulls_need_their_inputs(self, track, session):
        _, _, _, events, counts = session
        with pytest.raises(ValueError, match="needs \\['encoding_models', 'dt'\\]"):
            dataset_values(
                track,
                events,
                replay_method=SILVA_2015,
                null="unit_identity",
                spike_counts=counts,
            )


class TestOpenArena:
    """Nulls that need no order along a track work on a 2-D arena."""

    # A method that measures the event's MAP path, which an arena has.
    MAP_PATH = ReplayMethod(
        name="open_field_path",
        reference="test",
        score=None,
        criteria=(ReplayCriterion("start_to_end_map_distance", ">=", 40.0),),
        units="cm",
    )

    @pytest.fixture(scope="class")
    def arena(self):
        positions = np.random.default_rng(0).uniform(0.0, 100.0, (2000, 2))
        return Environment.from_samples(positions, bin_size=10.0, units="cm")

    @pytest.fixture(scope="class")
    def events(self, arena):
        rng = np.random.default_rng(1)
        events = []
        for n_time in (6, 7, 9):
            raw = rng.gamma(0.4, size=(n_time, arena.n_bins))
            events.append(
                DecodingResult(
                    posterior=raw / raw.sum(axis=1, keepdims=True),
                    env=arena,
                    times=np.arange(n_time) * DT,
                )
            )
        return events

    @pytest.mark.parametrize("null", ["time_bin_permutation", "pseudo_event"])
    def test_values_are_measured_on_the_shuffled_events(self, arena, events, null):
        """Each value equals a MAP path measured directly on the shuffled rows.

        MAP distances on an arena are Euclidean between bin centers; the
        bins are equal, so the MAP bin is the largest entry of a row.
        """
        assert np.ptp(arena.bin_sizes) == 0
        quantities = ("max_map_jump", "mean_peak_probability")
        table = dataset_values(
            arena,
            events,
            replay_method=self.MAP_PATH,
            null=null,
            n_datasets=4,
            quantities=quantities,
            rng=2,
        )
        shuffled = shuffle_events(arena, events, null=null, n_datasets=4, rng=2)
        pool = np.vstack([event.posterior for event in events])
        assert len(table) == 4 * len(events)
        for row in table.itertuples():
            event = events[row.event]
            rows = shuffled[row.dataset].events[row.event].posterior
            assert rows.shape == event.posterior.shape
            # The rows the null draws from: the event's own rows, each once,
            # or any rows of the pool (all rows differ, so each has one match).
            source = event.posterior if null == "time_bin_permutation" else pool
            matches = (rows[:, None, :] == source[None, :, :]).all(axis=2)
            assert (matches.sum(axis=1) == 1).all()
            if null == "time_bin_permutation":
                drawn = np.argmax(matches, axis=1)
                np.testing.assert_array_equal(np.sort(drawn), np.arange(len(rows)))
            maps = arena.bin_centers[np.argmax(rows, axis=1)]
            steps = np.linalg.norm(np.diff(maps, axis=0), axis=1)
            assert row.start_to_end_map_distance == pytest.approx(
                np.linalg.norm(maps[-1] - maps[0]), abs=1e-9
            )
            assert row.max_map_jump == pytest.approx(steps.max(), abs=1e-9)
            assert row.mean_peak_probability == pytest.approx(
                rows.max(axis=1).mean(), abs=1e-12
            )
        assert "track_length" not in table.columns

    @pytest.mark.parametrize(
        ("method", "null"),
        [
            # The column-cycle null rolls each row along the track.
            (MAP_PATH, "column_cycle"),
            # A weighted correlation is a slope of position along the track.
            (SILVA_2015, "time_bin_permutation"),
            (SILVA_2015, "pseudo_event"),
        ],
    )
    def test_track_order_is_still_required_where_used(
        self, arena, events, method, null
    ):
        with pytest.raises(ValueError, match="needs a position along a track"):
            dataset_values(arena, events, replay_method=method, null=null, rng=0)

    def test_shuffle_events_refuses_a_column_cycle_off_a_track(self, arena, events):
        with pytest.raises(ValueError, match="needs a position along a track"):
            shuffle_events(arena, events, null="column_cycle", rng=0)


class TestRankOrderValues:
    def test_real_values_match_compare_replay_methods(self, track, session):
        rates, trains, _, events, _ = session
        method = SILVA_2015_RANK_ORDER.method
        maps = {"right": rates, "left": rates[::-1].copy()}
        real = dataset_values(
            track,
            events,
            replay_method=method,
            spike_times=trains,
            encoding_models=maps,
            dt=DT,
        )
        assert real.groupby("event").size().eq(2).all()
        for name in maps:
            table = compare_replay_methods(
                track,
                events,
                replay_methods=[method],
                spike_times=trains,
                encoding_models=maps,
                template=name,
                dt=DT,
            )
            mine = real[real["template"] == name]
            np.testing.assert_allclose(mine["score"], table["score"], atol=1e-12)
            np.testing.assert_array_equal(
                mine["n_active_cells"], table["value[n_active_cells >= 3]"]
            )

    def test_cell_order_shuffles_are_rank_order_significance_draws(
        self, track, session
    ):
        """Per event, the shuffled correlations are rank_order_significance's.

        Each event's 4,000 null correlations and the 4,000 shuffles of
        ``rank_order_significance`` (other seeds) are two samples of one
        distribution: a two-sample KS test does not reject for any of four
        events, a sweep among them.
        """
        rates, trains, intervals, _, _ = session
        method = SILVA_2015_RANK_ORDER.method
        null = dataset_values(
            track,
            intervals,
            replay_method=method,
            null="cell_order",
            n_datasets=4000,
            spike_times=trains,
            encoding_models=rates,
            rng=0,
        )
        for event in range(4):
            significance = rank_order_significance(
                track,
                trains,
                intervals[event],
                rates,
                spike_statistic=method.spike_statistic,
                min_active_cells=method.min_active_cells,
                n_shuffles=4000,
                rng=1,
            )
            ours = null.loc[null["event"] == event, "score"].to_numpy()
            theirs = np.asarray(significance.shuffle.null_scores)
            assert np.isfinite(ours).all() and np.isfinite(theirs).all()
            assert stats.ks_2samp(ours, theirs).pvalue > 0.01, event
        real = dataset_values(
            track,
            intervals,
            replay_method=method,
            spike_times=trains,
            encoding_models=rates,
        )
        # The active cells never change, and a sweep beats its shuffles.
        merged = null.merge(real, on="event", suffixes=("", "_real"))
        np.testing.assert_array_equal(
            merged["n_active_cells"], merged["n_active_cells_real"]
        )
        sweep = null[null["event"] == 1]["score"]
        assert np.mean(sweep >= real.loc[1, "score"]) < 0.01

    def test_shared_permutation_is_shared_by_events_and_templates(self, track, session):
        rates, trains, intervals, _, _ = session
        maps = {"right": rates, "left": rates}
        pair = np.array([intervals[1], intervals[1] + 100.0])
        copied = [
            np.concatenate([t, t[(t >= pair[0, 0]) & (t < pair[0, 1])] + 100.0])
            for t in trains
        ]
        null = dataset_values(
            track,
            pair,
            replay_method=SILVA_2015_RANK_ORDER.method,
            null="cell_order",
            shared=True,
            n_datasets=20,
            spike_times=copied,
            encoding_models=maps,
            rng=0,
        )
        scores = null.pivot_table(
            index="dataset", columns=["event", "template"], values="score"
        )
        for column in scores.columns[1:]:
            np.testing.assert_allclose(scores[column], scores.iloc[:, 0])


class TestDistributionTest:
    @pytest.mark.parametrize("test", ["ks", "rank_sum"])
    def test_side_is_stated_in_terms_of_the_values(self, test):
        rng = np.random.default_rng(0)
        observed = pd.DataFrame({"value": rng.normal(0.5, 1.0, 300)})
        null = pd.DataFrame({"value": rng.normal(0.0, 1.0, 3000)})
        greater = distribution_test(
            observed, null, column="value", test=test, alternative="greater"
        )
        less = distribution_test(
            observed, null, column="value", test=test, alternative="less"
        )
        both = distribution_test(observed, null, column="value", test=test)
        assert greater.p_value < 1e-6
        assert less.p_value > 0.5
        assert both.p_value < 1e-5
        # Shifting the other way swaps the sides.
        swapped = distribution_test(
            observed.assign(value=-observed["value"]),
            null,
            column="value",
            test=test,
            alternative="less",
        )
        assert swapped.p_value < 1e-6
        assert greater.median_observed > greater.median_null

    def test_ks_greater_is_scipys_less(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(0.2, 1, 100), rng.normal(0, 1, 500)
        result = distribution_test(
            pd.DataFrame({"x": a}),
            pd.DataFrame({"x": b}),
            column="x",
            alternative="greater",
        )
        expected = stats.ks_2samp(a, b, alternative="less")
        assert result.p_value == expected.pvalue
        assert result.statistic == expected.statistic

    def test_levene_is_two_sided_and_mean_centered(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(0, 2, 100), rng.normal(0, 1, 400)
        result = distribution_test(
            pd.DataFrame({"x": a}), pd.DataFrame({"x": b}), column="x", test="levene"
        )
        assert result.p_value == stats.levene(a, b, center="mean").pvalue
        with pytest.raises(ValueError, match="Levene's test has no side"):
            distribution_test(
                pd.DataFrame({"x": a}),
                pd.DataFrame({"x": b}),
                column="x",
                test="levene",
                alternative="greater",
            )

    def test_nan_values_are_dropped_and_counted(self):
        observed = pd.DataFrame({"x": [1.0, 2.0, np.nan, 3.0]})
        null = pd.DataFrame({"x": [0.0, np.nan, np.nan, 1.0, 2.0]})
        result = distribution_test(observed, null, column="x")
        assert (result.n_observed, result.n_observed_dropped) == (3, 1)
        assert (result.n_null, result.n_null_dropped) == (3, 2)


class TestTemplateCorrelation:
    def test_relative_rank_matches_a_frame_by_frame_loop(self):
        rng = np.random.default_rng(3)
        trains = [np.sort(rng.uniform(0, 20, rng.integers(5, 40))) for _ in range(9)]
        intervals = np.column_stack(
            [np.arange(0, 20, 0.5), np.arange(0, 20, 0.5) + 0.3]
        )
        for statistic in ("mean", "first", "median"):
            template = relative_rank_template(
                trains, intervals, spike_statistic=statistic, min_active_cells=4
            )
            summarize = {"mean": np.mean, "first": np.min, "median": np.median}[
                statistic
            ]
            sums, frames = np.zeros(9), np.zeros(9)
            for start, stop in intervals:
                firing = {
                    cell: summarize(train[(train >= start) & (train < stop)])
                    for cell, train in enumerate(trains)
                    if np.any((train >= start) & (train < stop))
                }
                if len(firing) < 4:
                    continue
                ranks = stats.rankdata(list(firing.values()))
                for cell, rank in zip(firing, ranks, strict=True):
                    sums[cell] += (rank - 1) / (len(firing) - 1)
                    frames[cell] += 1
            expected = np.where(frames > 0, sums / np.maximum(frames, 1), np.nan)
            np.testing.assert_allclose(template, expected, rtol=0, atol=1e-12)

    def test_matches_the_exact_null_by_enumeration(self):
        """Statistic is the mean Spearman; p matches enumerating every shuffle."""
        sleep = [
            np.array([0.1, 0.3, np.nan, 0.2, 0.9]),
            np.array([0.5, 0.1, 0.7, 0.9]),
        ]
        run = [np.array([1.0, 2.0, 3.0, np.nan, 5.0]), np.array([2.0, 1.0, 4.0, 3.0])]
        result = template_correlation_test(sleep, run, n_shuffles=20000, rng=0)
        observed = []
        null_by_unit = []
        for x, y in zip(sleep, run, strict=True):
            both = np.isfinite(x) & np.isfinite(y)
            observed.append(stats.spearmanr(x[both], y[both]).statistic)
            cells = np.flatnonzero(np.isfinite(y))
            values = []
            for order in itertools.permutations(y[cells]):
                shuffled = y.copy()
                shuffled[cells] = order
                values.append(stats.spearmanr(x[both], shuffled[both]).statistic)
            null_by_unit.append(np.asarray(values))
        statistic = np.mean(observed)
        assert result.statistic == pytest.approx(statistic, abs=1e-12)
        exact = np.mean(
            [
                np.mean([a, b]) >= statistic - 1e-12
                for a in null_by_unit[0]
                for b in null_by_unit[1]
            ]
        )
        # Monte Carlo p within 4 standard errors of the exact tail.
        se = np.sqrt(exact * (1 - exact) / 20000)
        assert abs(result.p_value - exact) < 4 * se + 1 / 20001
        assert result.n_observed == 2

    def test_absolute_correlations(self):
        sleep = [np.arange(6.0), np.arange(6.0)[::-1]]
        run = [np.arange(6.0), np.arange(6.0)]
        signed = template_correlation_test(sleep, run, n_shuffles=200, rng=0)
        absolute = template_correlation_test(
            sleep, run, n_shuffles=200, use_absolute=True, rng=0
        )
        assert signed.statistic == pytest.approx(0.0)
        assert absolute.statistic == pytest.approx(1.0)

    @pytest.mark.parametrize("constant", ["template", "reference"])
    def test_refuses_an_undefined_correlation(self, constant):
        """A template constant over the shared cells has no rank correlation."""
        sleep = [np.full(6, 0.5), np.arange(8.0) / 7]
        run = [np.arange(6.0), np.arange(8.0)]
        if constant == "reference":
            sleep[0], run[0] = run[0], np.full(6, 2.0)
        with pytest.raises(
            ValueError, match="template 0 with its reference is not defined"
        ):
            template_correlation_test(sleep, run, n_shuffles=99, rng=0)

    def test_undefined_shuffles_count_as_at_least_as_extreme(self):
        """A shuffle that puts tied reference values on every shared cell.

        Three of the six cells have a sleep value, and three of the six
        reference values are tied, so 1 in 20 shuffles gives the shared cells
        one value and no correlation. Each such shuffle counts against the
        observed correlation on both sides, so p never shrinks.
        """
        sleep = [np.array([0.1, 0.5, np.nan, 0.3, np.nan, np.nan])]
        run = [np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])]
        result = template_correlation_test(sleep, run, n_shuffles=2000, rng=0)
        undefined = np.isnan(result.null_statistics)
        assert 0.03 < undefined.mean() < 0.07
        defined = result.null_statistics[~undefined]
        for alternative, extreme in (
            ("greater", defined >= result.statistic),
            ("less", defined <= result.statistic),
        ):
            sided = template_correlation_test(
                sleep, run, n_shuffles=2000, alternative=alternative, rng=0
            )
            assert sided.p_value == (1 + undefined.sum() + extreme.sum()) / 2001
        assert result.percentile == 100 * np.sum(defined < result.statistic) / 2000

    def test_refuses_templates_without_three_shared_cells(self):
        with pytest.raises(ValueError, match="shares 2 cells"):
            template_correlation_test(
                [np.array([1.0, 2.0, np.nan])], [np.array([1.0, 2.0, 3.0])]
            )


def test_recipes_hold_their_values():
    names = [recipe.name for recipe in DATASET_RECIPES]
    assert len(set(names)) == len(names)
    grid = SILVA_2015_MATRIX.criteria
    np.testing.assert_allclose(grid[0].thresholds, np.arange(10) / 10)
    np.testing.assert_allclose(grid[1].thresholds, np.arange(1, 11) / 10)
    assert SILVA_2015_MATRIX.method is SILVA_2015
    assert SILVA_2015_MATRIX.n_datasets == 5000
    assert FAROOQ_2019_TIME_BIN.n_datasets == 500
    for recipe in DATASET_RECIPES:
        assert recipe.notes, recipe.name
    assert hash(SILVA_2015_MATRIX) == hash(dataclasses.replace(SILVA_2015_MATRIX))


def test_recipe_columns_the_tables_lack_are_made_as_the_notes_say():
    """Silva compared absolute correlations; derived columns have a recipe."""
    assert SILVA_2015_RANK_ORDER.columns == ("abs_score",)
    derived = {"abs_score", "quality", "step"}
    for recipe in DATASET_RECIPES:
        notes = " ".join(recipe.notes)
        for column in derived.intersection(recipe.columns):
            assert f"assign({column}=" in notes or f", {column}=" in notes, (
                recipe.name,
                column,
            )


def test_silva_pooled_tests_run_on_absolute_correlations(track, session):
    """The rank-order recipe's documented pooled tests, on simulated events.

    The sweeps beat shuffled cell orders in |r|: the one-tailed rank-sum
    and KS tests reject.
    """
    rates, trains, intervals, _, _ = session
    recipe = SILVA_2015_RANK_ORDER
    common = {"spike_times": trains, "encoding_models": rates}
    real = dataset_values(track, intervals, replay_method=recipe.method, **common)
    null = dataset_values(
        track,
        intervals,
        replay_method=recipe.method,
        null=recipe.null,
        shared=recipe.shared,
        n_datasets=100,
        rng=0,
        **common,
    )
    (column,) = recipe.columns
    for table in (real, null):
        table[column] = table["score"].abs()
    for test in recipe.distribution:
        side = "two-sided" if test == "levene" else recipe.alternative
        result = distribution_test(
            real, null, column=column, test=test, alternative=side
        )
        if test != "levene":
            assert result.p_value < 0.01, test


@pytest.mark.parametrize("n_axes", [0, 1, 2])
def test_plot_returns_the_axes(n_axes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    observed = pd.DataFrame(
        {"score": rng.uniform(-1, 1, 20), "jump": rng.uniform(0, 1, 20)}
    )
    null = pd.DataFrame(
        {
            "dataset": np.repeat(np.arange(30), 20),
            "score": rng.uniform(-1, 1, 600),
            "jump": rng.uniform(0, 1, 600),
        }
    )
    grids = [
        CriterionGrid("score", ">", [0.2, 0.5, 0.8], absolute=True),
        CriterionGrid("jump", "<", [0.3, 0.6]),
    ][:n_axes]
    result = (
        dataset_test(observed, null, criteria=grids)
        if grids
        else dataset_test(observed, null, statistic=lambda t: t["score"].mean())
    )
    ax = result.plot()
    assert ax is not None
    plt.close("all")
