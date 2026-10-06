"""Detectors analyze observed intervals without inventing transitions in pauses."""

import warnings

import numpy as np
import pytest

from neurospatial.behavior import detect_region_crossings


def test_no_crossing_reported_across_pause(pause_track):
    r = pause_track
    assert (
        detect_region_crossings(r.position_bins, r.times, r.env, region_name="target")
        == []
    )


def test_crossing_epochs_equal_slicing(continuous_pause_track):
    r = continuous_pause_track
    selected = r.times <= 100
    expected = detect_region_crossings(
        r.position_bins[selected], r.times[selected], r.env, region_name="target"
    )
    actual = detect_region_crossings(
        r.position_bins, r.times, r.env, region_name="target", epochs=(0, 100)
    )
    assert actual == expected


def test_compatibility_warning_is_emitted_once(pause_track):
    r = pause_track
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detect_region_crossings(r.position_bins, r.times, "target", r.env)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1


def test_invalid_epochs_raise_without_observed_samples(pause_track):
    r = pause_track
    with pytest.raises(ValueError, match="epochs"):
        detect_region_crossings(
            np.array([], dtype=np.int64),
            np.array([]),
            r.env,
            region_name="target",
            epochs=(2, 1),
        )
