"""Replay recipes and results: construction checks and pickling."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.replay import (
    ArmReplayMethod,
    ReplayCriterion,
    ReplayDetectionResult,
    ReplayMethod,
    ReplaySegmentation,
    replay_segments,
    replay_significance,
    weighted_correlation,
)
from neurospatial.replay.methods import PUBLISHED_METHODS

DT = 0.02


@pytest.fixture(scope="module")
def env():
    return Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
    )


@pytest.fixture(scope="module")
def sweep(env):
    posterior = np.zeros((20, env.n_bins))
    posterior[np.arange(20), np.arange(20)] = 1.0
    return posterior, np.arange(20) * DT


def _round_trips(obj):
    return pickle.loads(pickle.dumps(obj)), copy.deepcopy(obj)


def _assert_same_fields(clone, original):
    for item in dataclasses.fields(original):
        got, expected = getattr(clone, item.name), getattr(original, item.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(got, expected)
        else:
            assert got == expected, item.name


def _assert_no_defaults(obj):
    """Every field differs from its default, so a reset would be seen."""
    for item in dataclasses.fields(obj):
        if item.default is not dataclasses.MISSING:
            default = item.default
        elif item.default_factory is not dataclasses.MISSING:
            default = item.default_factory()
        else:
            continue
        assert getattr(obj, item.name) != default, item.name


def _every_field_method() -> ReplayMethod:
    return ReplayMethod(
        name="every_field",
        reference="test",
        score=weighted_correlation,
        score_options={"row_normalization": "none"},
        nulls=("column_cycle",),
        n_shuffles=49,
        alpha=0.01,
        alternative="two-sided",
        use_absolute=True,
        empty_time_bins="zero",
        criteria=(
            ReplayCriterion("max_map_jump", "<", 0.4, relative_to="track_length"),
        ),
        segmentation=_every_field_segmentation(),
        units="cm",
        template_choice="most_posterior",
        exclude_end_distance=5.0,
        decoding={"time_bin": 0.02},
        notes=("a note",),
    )


def _every_field_segmentation() -> ReplaySegmentation:
    return ReplaySegmentation(
        {"max_jump": 0.4, "min_peak_probability": 5.0},
        relative_to={"max_jump": "track_length", "min_peak_probability": "chance"},
        choose="any",
        min_active_cells_per_time_bin=4,
        rows="trajectory",
    )


def _detection_result() -> ReplayDetectionResult:
    track = Environment.linear_track(
        endpoints=[[0.0, 0.0], [100.0, 0.0]], bin_size=4.0, units="cm"
    )
    segments = replay_segments(
        track, np.eye(track.n_bins)[:6], np.arange(6) * DT, max_jump=10.0
    )
    return ReplayDetectionResult(
        method="m",
        is_replay=True,
        score=0.5,
        p_value=0.01,
        criteria={"score > 0.1": True},
        values={"score > 0.1": 0.5},
        segments=segments,
        segment=0,
        template="a",
    )


def _every_field_arm_method() -> ArmReplayMethod:
    return ArmReplayMethod(
        name="every_field",
        reference="test",
        peak_threshold=4.0,
        smoothing_sigma=0.02,
        min_duration=0.04,
        min_coverage=0.4,
        min_abs_correlation=0.4,
        joint_min_abs_correlation=0.4,
        subregions="runs",
        merge_max_gap=0.03,
        coverage="mean_range",
        joint="strongest_pair",
        joint_same_sign=False,
        arm_fallback=True,
        decoding={"time_bin": 0.02},
        notes=("a note",),
    )


class TestPickling:
    def test_significance_result_pickles_with_read_only_arrays(self, env, sweep):
        posterior, times = sweep
        result = replay_significance(
            env,
            posterior,
            times,
            score=weighted_correlation,
            nulls="column_cycle",
            n_shuffles=19,
            rng=0,
        )
        for clone in _round_trips(result):
            assert clone.summary() == result.summary()
            null_scores = clone.shuffles["column_cycle"].null_scores
            np.testing.assert_array_equal(
                null_scores, result.shuffles["column_cycle"].null_scores
            )
            assert not null_scores.flags.writeable
            with pytest.raises(TypeError):
                clone.shuffles["x"] = None  # type: ignore[index]

    def test_significance_summary_with_a_custom_score_result(self, env, sweep):
        posterior, times = sweep

        def first_bin_mass(env, posterior, times):
            return SimpleNamespace(score=float(posterior[:, 0].sum()))

        result = replay_significance(
            env,
            posterior,
            times,
            score=first_bin_mass,
            nulls="column_cycle",
            n_shuffles=9,
            rng=0,
        )
        summary = result.summary()
        assert summary["score"] == 1.0
        assert summary["p_value"] == result.p_value
        assert "summary() failed" not in repr(result)

    @pytest.mark.parametrize(
        "build",
        [_every_field_method, _every_field_segmentation, _every_field_arm_method],
        ids=["ReplayMethod", "ReplaySegmentation", "ArmReplayMethod"],
    )
    def test_recipes_keep_every_field(self, build):
        recipe = build()
        _assert_no_defaults(recipe)
        for clone in _round_trips(recipe):
            _assert_same_fields(clone, recipe)

    def test_detection_result_keeps_every_field(self):
        result = _detection_result()
        for clone in _round_trips(result):
            for name in ("method", "is_replay", "score", "p_value", "segment"):
                assert getattr(clone, name) == getattr(result, name)
            assert clone.template == "a"
            assert dict(clone.criteria) == dict(result.criteria)
            assert dict(clone.values) == dict(result.values)
            np.testing.assert_array_equal(
                clone.segments.segments, result.segments.segments
            )

    @pytest.mark.parametrize(
        ("base", "build"),
        [
            (ReplayMethod, _every_field_method),
            (ReplaySegmentation, _every_field_segmentation),
            (ArmReplayMethod, _every_field_arm_method),
            (ReplayDetectionResult, _detection_result),
        ],
        ids=["ReplayMethod", "ReplaySegmentation", "ArmReplayMethod", "Result"],
    )
    def test_a_field_added_by_a_subclass_survives_pickling(self, base, build):
        extended = dataclasses.make_dataclass(
            f"Extended{base.__name__}",
            [("extra", int, dataclasses.field(default=0))],
            bases=(base,),
            frozen=True,
            eq=False,
            module=__name__,
        )
        globals()[extended.__name__] = extended  # importable for pickle
        original = build()
        fields = {
            item.name: getattr(original, item.name) for item in dataclasses.fields(base)
        }
        for clone in _round_trips(extended(**fields, extra=7)):
            assert clone.extra == 7

    def test_published_recipes_round_trip_field_by_field(self):
        for method in PUBLISHED_METHODS:
            for clone in _round_trips(method):
                assert type(clone) is type(method)
                _assert_same_fields(clone, method)
