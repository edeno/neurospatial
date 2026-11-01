"""Tests for the Trial record and the trial boundaries detect_trials reports."""

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, Regions
from neurospatial.behavior import detect_trials


class TestTrialRecord:
    """Trial stores how it ended; success is derived from the outcome."""

    def test_success_is_derived_from_outcome(self):
        from neurospatial.behavior.segmentation import Trial

        assert Trial(0.0, 1.0, "start", "left", "reached").success is True
        for outcome in ("returned", "censored"):
            assert Trial(0.0, 1.0, "start", None, outcome).success is False
        with pytest.raises(TypeError):
            Trial(0.0, 1.0, "start", "left", success=True)  # type: ignore[call-arg]

    def test_scalars_are_plain_floats(self):
        from neurospatial.behavior.segmentation import Trial

        trial = Trial(np.float32(0.5), np.int64(2), "start", None, "returned")
        assert type(trial.start_time) is float
        assert type(trial.stop_time) is float

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"start_time": 2.0, "stop_time": 1.0}, "start_time <= stop_time"),
            ({"stop_time": np.nan}, "start_time <= stop_time"),
            ({"outcome": "success"}, "outcome must be one of"),
            ({"outcome": "timeout"}, "outcome must be one of"),
            ({"outcome": "returned"}, "does not match"),
            ({"end_region": None}, "does not match"),
        ],
    )
    def test_invalid_records_raise(self, kwargs, match):
        from neurospatial.behavior.segmentation import Trial

        fields = {
            "start_time": 0.0,
            "stop_time": 1.0,
            "start_region": "start",
            "end_region": "left",
            "outcome": "reached",
        }
        with pytest.raises(ValueError, match=match):
            Trial(**{**fields, **kwargs})


class TestTrialEnds:
    """Exact trial boundaries on a corridor sampled at bin centers."""

    @staticmethod
    def _trials(x, times):
        xs = np.linspace(0.0, 100.0, 401)
        env = Environment.from_samples(np.c_[xs, np.zeros_like(xs)], bin_size=2.0)
        regions = Regions(env.regions.values())
        regions.add("start", polygon=box(-5.0, -5.0, 5.0, 5.0))  # bins 0-4 cm
        regions.add("goal", polygon=box(95.0, -5.0, 105.0, 5.0))  # bins 96-100 cm
        env = env.with_regions(regions)
        x = np.asarray(x, dtype=float)
        trials = detect_trials(
            env,
            np.asarray(times, dtype=float),
            np.c_[x, np.zeros_like(x)],
            regions=["start", "goal"],
            max_gap=None,
        )
        return [(t.start_time, t.stop_time, t.end_region, t.outcome) for t in trials]

    def test_trial_runs_from_departure_to_arrival(self):
        """A trial starts at the first sample outside the start region and
        stops at the first sample inside the next region (the arrival, which
        is outside the half-open trial)."""
        assert self._trials([0, 0, 30, 60, 98, 98], [0, 1, 2, 6, 11.5, 12]) == [
            (2.0, 11.5, "goal", "reached")
        ]

    def test_returned_trial_ends_at_the_reentry_sample(self):
        x = [0, 0, 10, 20, 10, 4, 30, 60, 98]
        assert self._trials(x, np.arange(len(x))) == [
            (2.0, 5.0, None, "returned"),
            (6.0, 8.0, "goal", "reached"),
        ]

    def test_trial_in_progress_at_the_end_is_censored(self):
        assert self._trials([0, 0, 30, 60], [0, 1, 2, 3]) == [
            (2.0, 3.0, None, "censored")
        ]
