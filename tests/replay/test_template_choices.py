"""Template choices that test several templates: smallest p and largest score."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from replay_helpers import template_stream as _stream

from neurospatial import Environment
from neurospatial.decoding import decode_jointly, decode_position
from neurospatial.replay import (
    ReplayCriterion,
    ReplayMethod,
    ReplaySegmentation,
    compare_replay_methods,
    detect_replay,
    replay_significance,
    weighted_correlation,
)

DT = 0.02
N_CELLS = 20


def _method(**changes):
    options = {
        "name": "wc_smallest_p",
        "reference": "test",
        "score": weighted_correlation,
        "nulls": ("column_cycle",),
        "n_shuffles": 99,
        "use_absolute": True,
        "template_choice": "smallest_p",
    }
    return ReplayMethod(**{**options, **changes})


@pytest.fixture(scope="module")
def track():
    env = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=5.0, units="cm"
    )
    x = env.bin_centers[:, 0]
    centers = np.linspace(2.5, 97.5, N_CELLS)[:, None]
    rates = 0.5 + 20.0 * np.exp(-0.5 * ((x - centers) / 6.0) ** 2)
    # A second template whose fields alternate between the track's ends in
    # the sweep's cell order (0, 19, 1, 18, ...): no sequence on it.
    zigzag = np.ravel(np.column_stack([np.arange(10), np.arange(19, 9, -1)]))
    return env, {"out": rates, "in": rates[zigzag]}


@pytest.fixture(scope="module")
def sweep(track):
    """A sweep in the 'out' template's field order."""
    _env, _maps = track
    n_time = 12
    counts = np.zeros((n_time, N_CELLS), dtype=np.int64)
    for t in range(n_time):
        cell = int(t * N_CELLS / n_time)
        counts[t, cell] = 3
        counts[t, min(cell + 1, N_CELLS - 1)] = 1
    return counts, DT * (np.arange(n_time) + 0.5)


@pytest.fixture(scope="module")
def joint(track, sweep):
    env, maps = track
    counts, times = sweep
    return decode_jointly(env, counts, maps, dt=DT, times=times)


@pytest.fixture(scope="module")
def separate(track, sweep):
    env, maps = track
    counts, times = sweep
    return {
        name: decode_position(env, counts, rates, dt=DT, times=times)
        for name, rates in maps.items()
    }


class TestSmallestP:
    def _each_template(self, env, joint, times, method, seed, **inputs):
        return {
            name: detect_replay(
                env,
                joint,
                times,
                replay_method=method,
                template=name,
                rng=_stream(seed, name),
                **inputs,
            )
            for name in joint.template_names
        }

    def test_reports_the_template_with_the_smallest_p(self, track, sweep, joint):
        env, _ = track
        _, times = sweep
        method = _method()
        alone = self._each_template(env, joint, times, method, seed=7)
        assert alone["out"].p_value < method.alpha < alone["in"].p_value
        result = detect_replay(env, joint, times, replay_method=method, rng=7)
        assert result.template == "out"
        assert result.p_value == alone["out"].p_value
        assert result.score == alone["out"].score
        assert result.is_replay

    def test_the_verdict_is_the_chosen_template_s(self, track, sweep, joint):
        """A template with the smallest p but a failed criterion is not replay."""
        env, _ = track
        _, times = sweep
        method = _method(criteria=(ReplayCriterion("score", ">", 1.5),))
        result = detect_replay(env, joint, times, replay_method=method, rng=7)
        assert result.template == "out" and result.p_value < method.alpha
        assert not result.criteria["score > 1.5"] and not result.is_replay

    def test_decoding_nulls_test_each_share(self, track, sweep, joint):
        env, maps = track
        counts, times = sweep
        method = _method(nulls=("unit_identity",), n_shuffles=40)
        inputs = {"spike_counts": counts, "encoding_models": maps, "dt": DT}
        alone = self._each_template(env, joint, times, method, seed=2, **inputs)
        result = detect_replay(env, joint, times, replay_method=method, rng=2, **inputs)
        best = min(alone, key=lambda name: alone[name].p_value)
        assert result.template == best == "out"
        assert result.p_value == alone[best].p_value

    def test_template_criteria_are_the_chosen_template_s(self, track, sweep, joint):
        env, _ = track
        _, times = sweep
        method = _method(criteria=(ReplayCriterion("template_fraction", ">", 0.0),))
        result = detect_replay(env, joint, times, replay_method=method, rng=7)
        alone = detect_replay(
            env,
            joint,
            times,
            replay_method=method,
            template=result.template,
            rng=_stream(7, result.template),
        )
        assert result.values == alone.values

    def test_results_do_not_depend_on_template_order(self, track, sweep, joint):
        env, maps = track
        counts, times = sweep
        reordered = decode_jointly(
            env, counts, {"in": maps["in"], "out": maps["out"]}, dt=DT, times=times
        )
        method = _method()
        first = detect_replay(env, joint, times, replay_method=method, rng=11)
        second = detect_replay(env, reordered, times, replay_method=method, rng=11)
        assert (first.template, first.p_value) == (second.template, second.p_value)
        assert first.score == pytest.approx(second.score)

    def test_a_seed_reproduces_the_result(self, track, sweep, joint):
        env, _ = track
        _, times = sweep
        method = _method(nulls=("time_bin_permutation",), n_shuffles=19)
        runs = [
            detect_replay(env, joint, times, replay_method=method, rng=seed).summary()
            for seed in (5, 5)
        ]
        assert runs[0] == runs[1]
        generators = [
            detect_replay(
                env, joint, times, replay_method=method, rng=np.random.default_rng(5)
            ).p_value
            for _ in range(2)
        ]
        assert generators[0] == generators[1]

    def test_a_tie_goes_to_the_first_template(self, track, sweep):
        env, maps = track
        counts, times = sweep
        for order in (("a", "b"), ("b", "a")):
            same = decode_jointly(
                env, counts, dict.fromkeys(order, maps["out"]), dt=DT, times=times
            )
            result = detect_replay(env, same, times, replay_method=_method(), rng=0)
            assert result.p_value == pytest.approx(1 / 100)
            assert result.template == order[0]

    def test_separately_decoded_templates(self, track, sweep, separate):
        env, maps = track
        counts, times = sweep
        method = _method(nulls=("unit_identity",), n_shuffles=30)
        posteriors = {name: result.posterior for name, result in separate.items()}
        inputs = {"spike_counts": counts, "dt": DT}
        result = detect_replay(
            env,
            posteriors,
            times,
            replay_method=method,
            encoding_models=maps,
            rng=4,
            **inputs,
        )
        alone = {
            name: detect_replay(
                env,
                posteriors[name],
                times,
                replay_method=method,
                encoding_models=maps[name],
                rng=_stream(4, name),
                **inputs,
            )
            for name in maps
        }
        assert result.template == "out"
        assert result.p_value == alone["out"].p_value < alone["in"].p_value

    def test_a_template_without_a_p_value_is_chosen_last(self, track, separate):
        """A parked posterior has no weighted correlation, so no p-value.

        It is first in the decode's order, so it would win a tie; it is
        chosen only when no template has a p-value.
        """
        env, _ = track
        times = separate["out"].times
        parked = np.zeros_like(separate["out"].posterior)
        parked[:, 4] = 1.0  # the decoded position never changes
        method = _method()
        result = detect_replay(
            env,
            {"parked": parked, "out": separate["out"].posterior},
            times,
            replay_method=method,
            rng=0,
        )
        assert result.template == "out"
        assert result.p_value < method.alpha and result.is_replay
        neither = detect_replay(
            env, {"a": parked, "b": parked}, times, replay_method=method, rng=0
        )
        assert neither.template == "a"
        assert np.isnan(neither.p_value) and not neither.is_replay

    def test_compare_records_the_chosen_template(self, track, sweep, joint, separate):
        env, _ = track
        table = compare_replay_methods(
            env, [joint, separate], replay_methods=[_method()], rng=0
        )
        assert list(table["template"]) == ["out", "out"]
        assert table["is_replay"].all()


class TestMaxScore:
    def test_detect_replay_tests_separately_decoded_templates(
        self, track, sweep, separate
    ):
        env, maps = track
        counts, times = sweep
        method = _method(
            name="muessig_like",
            nulls=("unit_identity",),
            n_shuffles=40,
            template_choice="max_score",
            criteria=(ReplayCriterion("max_map_jump", "<", 50.0),),
        )
        posteriors = {name: result.posterior for name, result in separate.items()}
        inputs = {"spike_counts": counts, "encoding_models": maps, "dt": DT}
        result = detect_replay(
            env, posteriors, times, replay_method=method, rng=3, **inputs
        )
        tested = replay_significance(
            env,
            posteriors,
            times,
            score=weighted_correlation,
            nulls="unit_identity",
            n_shuffles=40,
            use_absolute=True,
            rng=3,
            **inputs,
        )
        assert result.template == tested.template == "out"
        assert result.p_value == tested.p_value
        assert result.score == pytest.approx(tested.observed.score)
        assert result.is_replay
        out_alone = detect_replay(
            env,
            posteriors["out"],
            times,
            replay_method=dataclasses.replace(method, template_choice=None),
            rng=0,
            **{**inputs, "encoding_models": maps["out"]},
        )
        assert result.values == out_alone.values

    def test_compare_takes_separately_decoded_events(self, track, sweep, separate):
        env, _ = track
        method = _method(name="max_wc", template_choice="max_score")
        table = compare_replay_methods(env, [separate], replay_methods=[method], rng=0)
        assert table.loc[0, "template"] == "out"
        assert table.loc[0, "is_replay"]

    def test_an_undefined_template_score_gives_no_verdict(self, track, separate):
        env, _ = track
        times = separate["out"].times
        parked = np.zeros_like(separate["out"].posterior)
        parked[:, 4] = 1.0  # the decoded position never changes
        posteriors = {"out": separate["out"].posterior, "in": parked}
        method = _method(template_choice="max_score")
        result = detect_replay(env, posteriors, times, replay_method=method, rng=0)
        assert not result.is_replay and np.isnan(result.score)


class TestRefusals:
    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"nulls": ()}, "needs nulls"),
            (
                {
                    "template_choice": "max_score",
                    "nulls": (),
                    "criteria": (ReplayCriterion("score", ">", 0.5),),
                },
                "needs nulls",
            ),
            (
                {"template_choice": "max_score", "alternative": "less"},
                "alternative='greater'",
            ),
            (
                {
                    "template_choice": "max_score",
                    "criteria": (ReplayCriterion("template_fraction", ">", 0.5),),
                },
                "template_fraction",
            ),
            (
                {"template_choice": "max_score", "nulls": ("pseudo_event",)},
                "pseudo_event",
            ),
            (
                {
                    "template_choice": "max_score",
                    "segmentation": ReplaySegmentation({"max_jump": 20.0}),
                    "units": "cm",
                },
                "segmentation",
            ),
            (
                {
                    "template_choice": "max_score",
                    "score": None,
                    "use_absolute": None,
                    "segmentation": ReplaySegmentation({"max_jump": 20.0}),
                    "units": "cm",
                },
                "a score",
            ),
        ],
    )
    def test_method_construction(self, changes, match):
        with pytest.raises(ValueError, match=match):
            _method(**changes)

    def test_separate_templates_need_a_template_choice(self, track, separate):
        env, _ = track
        posteriors = {name: result.posterior for name, result in separate.items()}
        times = separate["out"].times
        for method in (
            _method(template_choice=None),
            _method(template_choice="most_posterior"),
        ):
            with pytest.raises(ValueError, match="decoded separately"):
                detect_replay(env, posteriors, times, replay_method=method)
        with pytest.raises(ValueError, match="template='out'"):
            detect_replay(
                env,
                posteriors,
                times,
                replay_method=_method(template_choice="max_score"),
                template="out",
            )

    def test_max_score_refuses_a_joint_decode(self, track, sweep, joint):
        env, _ = track
        _, times = sweep
        with pytest.raises(ValueError, match="not a joint decode"):
            detect_replay(
                env, joint, times, replay_method=_method(template_choice="max_score")
            )

    def test_inputs_keyed_like_the_templates(self, track, sweep, separate):
        env, maps = track
        counts, times = sweep
        posteriors = {name: result.posterior for name, result in separate.items()}
        method = _method(template_choice="max_score", nulls=("unit_identity",))
        with pytest.raises(ValueError, match="encoding_models has keys"):
            detect_replay(
                env,
                posteriors,
                times,
                replay_method=method,
                spike_counts=counts,
                encoding_models={"out": maps["out"]},
                dt=DT,
            )
        with pytest.raises(ValueError, match="mapping of encoding_models"):
            replay_significance(
                env,
                posteriors,
                times,
                nulls="unit_identity",
                spike_counts=counts,
                encoding_models=maps["out"],
                dt=DT,
            )
        with pytest.raises(ValueError, match="different numbers of time bins"):
            detect_replay(
                env,
                {"out": posteriors["out"], "in": posteriors["in"][1:]},
                times,
                replay_method=_method(),
            )

    def test_compare_refuses_templates_at_different_times(self, track, separate):
        env, _ = track
        shifted = dataclasses.replace(separate["in"], times=separate["in"].times + 1.0)
        with pytest.raises(ValueError, match="decoded at different times"):
            compare_replay_methods(
                env,
                [{"out": separate["out"], "in": shifted}],
                replay_methods=[_method()],
            )
