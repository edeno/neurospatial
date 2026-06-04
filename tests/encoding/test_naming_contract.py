"""Tests for the v0.6 naming-contract enforcement (Task 1.4).

Covers:

- ``classify`` batch predicate replacing per-domain ``detect_*`` detectors
  (OVC / view / HD) plus the new ``SpatialRatesResult.classify`` place
  predicate, with deprecation aliases.
- ``detect_cell_types`` -> ``label_cell_types`` rename (multi-class labeler).
- New ``is_place_cell`` free function + ``SpatialRateResult.is_place_cell``
  method, agreeing with ``detect_place_fields``.
- ``peak_view_location`` -> ``peak_location`` / ``peak_locations`` collapse.

Each deprecation gets a test asserting (a) the old form still works AND warns,
(b) the new form is warning-free, and (c) both give the same result.
"""

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
# 1.4a -- classify() batch predicate + deprecated detect_* aliases
# ---------------------------------------------------------------------------


def test_detect_ovcs_deprecated_alias_of_classify(trajectory) -> None:
    from neurospatial.encoding.egocentric import compute_egocentric_rates

    env, times, positions, headings, spike_times = trajectory
    object_positions = np.array([[50.0, 50.0]])
    result = compute_egocentric_rates(
        env, spike_times, times, positions, headings, object_positions
    )

    # New form: warning-free.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = result.classify()

    # Old form: warns.
    with pytest.warns(DeprecationWarning):
        old = result.detect_ovcs()

    np.testing.assert_array_equal(old, new)
    assert new.dtype == np.bool_


def test_detect_view_cells_deprecated_alias_of_classify(trajectory) -> None:
    from neurospatial.encoding.view import compute_view_rates

    env, times, positions, headings, spike_times = trajectory
    result = compute_view_rates(
        env, spike_times, times, positions, headings, view_distance=10.0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = result.classify()

    with pytest.warns(DeprecationWarning):
        old = result.detect_view_cells()

    np.testing.assert_array_equal(old, new)
    assert new.dtype == np.bool_


def test_detect_hd_cells_deprecated_alias_of_classify(trajectory) -> None:
    from neurospatial.encoding.directional import compute_directional_rates

    _env, times, _positions, headings, spike_times = trajectory
    result = compute_directional_rates(spike_times, times, headings)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = result.classify()

    with pytest.warns(DeprecationWarning):
        old = result.detect_hd_cells()

    np.testing.assert_array_equal(old, new)
    assert new.dtype == np.bool_


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
# 1.4b -- detect_cell_types -> label_cell_types (multi-class labeler)
# ---------------------------------------------------------------------------


def test_detect_cell_types_deprecated_alias_of_label_cell_types(trajectory) -> None:
    from neurospatial.encoding.spatial import compute_spatial_rates

    env, times, positions, _headings, spike_times = trajectory
    result = compute_spatial_rates(env, spike_times, times, positions, bandwidth=10.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = result.label_cell_types()

    with pytest.warns(DeprecationWarning):
        old = result.detect_cell_types()

    np.testing.assert_array_equal(old, new)
    # Multi-class string labels (distinct return type from classify()).
    assert new.dtype.kind == "U"
    assert set(new.tolist()).issubset({"place", "grid", "border", "unclassified"})


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
# 1.4c -- is_place_cell free fn + method, agreeing with detect_place_fields
# ---------------------------------------------------------------------------


def test_is_place_cell_method_agrees_with_detect_place_fields(trajectory) -> None:
    from neurospatial.encoding.spatial import compute_spatial_rate, detect_place_fields

    env, times, positions, _headings, spike_times = trajectory
    for spikes in spike_times:
        result = compute_spatial_rate(env, spikes, times, positions, bandwidth=10.0)
        fields = detect_place_fields(env, np.asarray(result.firing_rate))
        assert result.is_place_cell() == (len(fields) > 0)


def test_is_place_cell_free_function_agrees_with_detect_place_fields(
    trajectory,
) -> None:
    from neurospatial.encoding.spatial import (
        compute_spatial_rate,
        detect_place_fields,
        is_place_cell,
    )

    env, times, positions, _headings, spike_times = trajectory
    for spikes in spike_times:
        result = compute_spatial_rate(env, spikes, times, positions, bandwidth=10.0)
        fields = detect_place_fields(env, np.asarray(result.firing_rate))
        assert is_place_cell(env, spikes, times, positions, bandwidth=10.0) == (
            len(fields) > 0
        )


def test_is_place_cell_exported_from_encoding() -> None:
    import neurospatial.encoding as enc

    assert hasattr(enc, "is_place_cell")


# ---------------------------------------------------------------------------
# 1.4d -- peak_view_location -> peak_location / peak_locations
# ---------------------------------------------------------------------------


def test_view_single_peak_view_location_deprecated(trajectory) -> None:
    from neurospatial.encoding.view import compute_view_rate

    env, times, positions, headings, spike_times = trajectory
    result = compute_view_rate(
        env, spike_times[0], times, positions, headings, view_distance=10.0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = result.peak_location()

    with pytest.warns(DeprecationWarning):
        old = result.peak_view_location()

    np.testing.assert_array_equal(old, new)


def test_view_batch_peak_view_location_deprecated(trajectory) -> None:
    from neurospatial.encoding.view import compute_view_rates

    env, times, positions, headings, spike_times = trajectory
    result = compute_view_rates(
        env, spike_times, times, positions, headings, view_distance=10.0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = result.peak_locations()

    with pytest.warns(DeprecationWarning):
        old = result.peak_view_location()

    np.testing.assert_array_equal(old, new)
    assert new.shape == (len(spike_times), env.bin_centers.shape[1])
