"""Canonical classification names, field detection and peak-accessor migration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def trajectory() -> tuple[
    Environment,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
]:
    """A small environment plus times/positions/headings and population spikes."""
    rng = np.random.default_rng(0)
    positions = rng.uniform(10, 90, (800, 2))
    env = Environment.from_samples(positions, bin_size=5.0)
    times = np.linspace(0, 80, 800)
    headings = rng.uniform(-np.pi, np.pi, 800)
    spike_times = [np.sort(rng.uniform(0, 80, n)) for n in (60, 80, 40)]
    return env, times, positions, headings, spike_times


# ---------------------------------------------------------------------------
# Batch boolean classification
# ---------------------------------------------------------------------------


def test_spatialrates_classify_is_bool_place_predicate(trajectory) -> None:
    from neurospatial.encoding.spatial import compute_spatial_rates

    env, times, positions, _headings, spike_times = trajectory
    result = compute_spatial_rates(env, spike_times, times, positions, bandwidth=10.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        is_place = result.classify()

    assert is_place.dtype == np.bool_
    assert is_place.shape == (len(spike_times),)
    # Should agree with the spatial-information threshold it is defined from.
    info = np.asarray(result.spatial_information())
    np.testing.assert_array_equal(is_place, info >= 0.5)


# ---------------------------------------------------------------------------
# Multi-class string labels
# ---------------------------------------------------------------------------


def test_label_cell_types_distinct_from_classify(trajectory) -> None:
    """label_cell_types (str) and classify (bool) are SEPARATE methods."""
    from neurospatial.encoding.spatial import compute_spatial_rates

    env, times, positions, _headings, spike_times = trajectory
    result = compute_spatial_rates(env, spike_times, times, positions, bandwidth=10.0)

    labels = result.label_cell_types()
    is_place = result.classify()
    assert labels.dtype.kind == "U"
    assert is_place.dtype == np.bool_


# ---------------------------------------------------------------------------
# Field detection agrees with detect_place_fields
# ---------------------------------------------------------------------------


def test_has_place_field_method_agrees_with_detect_place_fields(trajectory) -> None:
    from neurospatial.encoding.spatial import compute_spatial_rate, detect_place_fields

    env, times, positions, _headings, spike_times = trajectory
    for spikes in spike_times:
        result = compute_spatial_rate(env, spikes, times, positions, bandwidth=10.0)
        fields = detect_place_fields(env, np.asarray(result.firing_rate))
        assert result.has_place_field() == (len(fields) > 0)


def test_has_place_field_free_function_agrees_with_detect_place_fields(
    trajectory,
) -> None:
    from neurospatial.encoding.spatial import (
        compute_spatial_rate,
        detect_place_fields,
        has_place_field,
    )

    env, times, positions, _headings, spike_times = trajectory
    for spikes in spike_times:
        result = compute_spatial_rate(env, spikes, times, positions, bandwidth=10.0)
        fields = detect_place_fields(env, np.asarray(result.firing_rate))
        assert has_place_field(env, spikes, times, positions, bandwidth=10.0) == (
            len(fields) > 0
        )


def test_has_place_field_exported_from_encoding() -> None:
    import neurospatial.encoding as enc

    assert hasattr(enc, "has_place_field")
