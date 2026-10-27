"""Declared recording breaks (``break_before``) in every encoding family.

A break declared before sample ``k`` must behave exactly like a pause longer
than ``max_gap`` between samples ``k - 1`` and ``k``: the interval is left out
of occupancy, its spikes are not counted, speed is not smoothed across it and
shuffle nulls stay within the runs on either side.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from neurospatial import Environment, encoding

OFFSET = 1000.0  # seconds added after the break: far longer than max_gap


@pytest.fixture(scope="module")
def recording():
    """30 Hz tracking with jitter, a break at sample ``k`` and the shifted twin."""
    rng = np.random.default_rng(11)
    n = 900
    times = np.cumsum(np.full(n, 1 / 30) + rng.uniform(-0.004, 0.004, n))
    phase = 2 * np.pi * times / 9.0
    positions = np.column_stack(
        [50 + 35 * np.sin(phase), 50 + 30 * np.cos(0.7 * phase)]
    )
    velocity = np.gradient(positions, times, axis=0)
    headings = np.arctan2(velocity[:, 1], velocity[:, 0])
    k = 437
    spike_rng = np.random.default_rng(12)
    inside = np.linspace(times[k - 1], times[k], 5, endpoint=False)[1:]
    trains = [
        np.sort(np.r_[spike_rng.uniform(times[0], times[-1], count), inside])
        for count in (300, 220)
    ]
    breaks = np.zeros(n, dtype=bool)
    breaks[k] = True
    shifted_times = np.where(np.arange(n) >= k, times + OFFSET, times)
    shifted_trains = [np.where(t >= times[k], t + OFFSET, t) for t in trains]
    env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=5.0, units="cm")
    return SimpleNamespace(
        env=env,
        times=times,
        positions=positions,
        headings=headings,
        objects=np.array([[50.0, 50.0]]),
        labels=np.where(np.cos(phase) > 0, "outbound", "inbound").astype(object),
        trains=trains,
        k=k,
        breaks=breaks,
        shifted_times=shifted_times,
        shifted_trains=shifted_trains,
        inside=inside,
    )


def _family_args(name, r, trains, times):
    """Positional data arguments for one rate family."""
    if name == "directional":
        return (trains, times, r.headings)
    args = (r.env, trains, times, r.positions)
    if name in ("view", "egocentric", "boundary_vector"):
        args += (r.headings,)
    if name in ("object_vector", "egocentric"):
        args += (r.objects,)
    return args


# Unsmoothed estimators, so rate * occupancy recovers the spike counts.
FAMILIES = {
    "spatial": {"method": "binned", "bandwidth": 0.0, "min_occupancy": 0.0},
    "directional": {"bandwidth": None},
    "view": {
        "method": "binned",
        "bandwidth": 0.0,
        "gaze_model": "fixed_distance",
        "view_distance": 5.0,
    },
    "object_vector": {"method": "binned"},
    "egocentric": {"method": "binned"},
    "boundary_vector": {"method": "binned", "direction_frame": "egocentric"},
}


def _compute(name, r, *, plural, shifted=False, **kwargs):
    single = getattr(encoding, f"compute_{name}_rate")
    population = getattr(encoding, f"compute_{name}_rates")
    times = r.shifted_times if shifted else r.times
    trains = r.shifted_trains if shifted else r.trains
    if plural:
        return population(
            *_family_args(name, r, trains, times), **FAMILIES[name], **kwargs
        )
    return single(*_family_args(name, r, trains[0], times), **FAMILIES[name], **kwargs)


def _rates(result):
    rates = result.firing_rates if hasattr(result, "firing_rates") else None
    return np.asarray(result.firing_rate if rates is None else rates, dtype=float)


def _counts(result):
    return np.nan_to_num(_rates(result) * np.asarray(result.occupancy))


@pytest.mark.parametrize("plural", [False, True])
@pytest.mark.parametrize("name", list(FAMILIES))
def test_break_equals_a_pause_longer_than_max_gap(recording, name, plural):
    r = recording
    declared = _compute(name, r, plural=plural, break_before=r.breaks)
    paused = _compute(name, r, plural=plural, shifted=True)

    np.testing.assert_allclose(declared.occupancy, paused.occupancy, atol=1e-12)
    np.testing.assert_allclose(_rates(declared), _rates(paused), equal_nan=True)
    np.testing.assert_allclose(_counts(declared), _counts(paused), atol=1e-9)
    assert declared.analysis_settings["break_times"] == [r.times[r.k]]
    assert paused.analysis_settings["break_times"] is None
    json.dumps(declared.analysis_settings)  # provenance stays JSON-safe


@pytest.mark.parametrize("name", list(FAMILIES))
def test_single_and_population_agree_with_breaks(recording, name):
    r = recording
    single = _compute(name, r, plural=False, break_before=r.breaks)
    population = _compute(name, r, plural=True, break_before=r.breaks)
    np.testing.assert_allclose(
        _rates(single), _rates(population)[0], equal_nan=True, atol=1e-12
    )
    np.testing.assert_allclose(single.occupancy, population.occupancy)
    assert single.analysis_settings == population.analysis_settings


@pytest.mark.parametrize("name", list(FAMILIES))
def test_no_breaks_leaves_results_unchanged(recording, name):
    r = recording
    reference = _compute(name, r, plural=True)
    for value in (None, np.zeros(len(r.times), dtype=bool)):
        result = _compute(name, r, plural=True, break_before=value)
        np.testing.assert_array_equal(result.occupancy, reference.occupancy)
        np.testing.assert_array_equal(_rates(result), _rates(reference))
        expected = {**reference.analysis_settings}
        expected["break_times"] = None if value is None else []
        assert result.analysis_settings == expected


@pytest.mark.parametrize(
    "name", [name for name in FAMILIES if name != "boundary_vector"]
)
def test_break_removes_exactly_its_interval_and_spikes(recording, name):
    # Boundary-vector occupancy counts every ray that meets a wall, so its
    # totals are a multiple of the duration; the pause equivalence covers it.
    r = recording
    continuous = _compute(name, r, plural=True)
    declared = _compute(name, r, plural=True, break_before=r.breaks)
    duration = r.times[r.k] - r.times[r.k - 1]
    assert continuous.occupancy.sum() - declared.occupancy.sum() == pytest.approx(
        duration, abs=1e-9
    )
    removed = _counts(continuous).sum(axis=1) - _counts(declared).sum(axis=1)
    in_interval = [
        np.count_nonzero((t >= r.times[r.k - 1]) & (t < r.times[r.k])) for t in r.trains
    ]
    assert min(in_interval) >= len(r.inside)
    np.testing.assert_allclose(removed, in_interval, atol=1e-9)


@pytest.mark.parametrize(
    "options",
    [{"time_allocation": "linear"}, {"method": "glm"}],
    ids=["linear", "glm"],
)
def test_spatial_estimators_honor_breaks(recording, options):
    r = recording
    kwargs = options if "method" in options else {**FAMILIES["spatial"], **options}
    declared = encoding.compute_spatial_rates(
        r.env, r.trains, r.times, r.positions, **kwargs, break_before=r.breaks
    )
    paused = encoding.compute_spatial_rates(
        r.env, r.shifted_trains, r.shifted_times, r.positions, **kwargs
    )
    np.testing.assert_allclose(declared.occupancy, paused.occupancy, atol=1e-12)
    np.testing.assert_allclose(
        _rates(declared), _rates(paused), equal_nan=True, rtol=1e-7, atol=1e-10
    )


def test_derived_speed_is_not_smoothed_across_a_break(recording):
    """An animal moved by hand between recordings is not counted as running."""
    r = recording
    slow = 5.0 * np.sin(2 * np.pi * r.times / 4.0)  # peaks near 7.9 cm/s
    x = np.where(np.arange(len(r.times)) < r.k, 30.0, 70.0) + slow
    positions = np.column_stack([x, np.full(len(x), 50.0)])
    kwargs = {**FAMILIES["spatial"], "min_speed": 5.0}
    declared = encoding.compute_spatial_rates(
        r.env, r.trains, r.times, positions, **kwargs, break_before=r.breaks
    )
    paused = encoding.compute_spatial_rates(
        r.env, r.shifted_trains, r.shifted_times, positions, **kwargs
    )
    bridged = encoding.compute_spatial_rates(
        r.env, r.trains, r.times, positions, **kwargs
    )
    np.testing.assert_allclose(declared.occupancy, paused.occupancy, atol=1e-12)
    np.testing.assert_allclose(_counts(declared), _counts(paused), atol=1e-9)
    assert declared.occupancy.sum() > 1.0
    # Smoothing over the 40 cm jump would let ~0.3 s of rest pass the gate.
    assert bridged.occupancy.sum() > declared.occupancy.sum() + 0.1


def test_directional_place_fields_honor_breaks(recording):
    r = recording
    declared = encoding.compute_directional_place_fields(
        r.env, r.trains[0], r.times, r.positions, r.labels, break_before=r.breaks
    )
    paused = encoding.compute_directional_place_fields(
        r.env, r.shifted_trains[0], r.shifted_times, r.positions, r.labels
    )
    assert declared.labels == paused.labels
    for label in declared.labels:
        np.testing.assert_allclose(declared.occupancy[label], paused.occupancy[label])
        np.testing.assert_allclose(
            declared.firing_rates[label], paused.firing_rates[label], equal_nan=True
        )


SIGNIFICANCE = {
    "place_cell_significance": "spatial",
    "head_direction_cell_significance": "directional",
    "spatial_view_cell_significance": "view",
    "object_vector_cell_significance": "object_vector",
    "egocentric_object_vector_cell_significance": "egocentric",
}


@pytest.mark.parametrize("function", list(SIGNIFICANCE))
def test_every_shuffle_iteration_honors_breaks(recording, function):
    r = recording
    family = SIGNIFICANCE[function]
    options = {
        key: value
        for key, value in FAMILIES[family].items()
        if key != "direction_frame"
    }
    test = getattr(encoding, function)
    settings = {"n_shuffles": 15, "min_shift": 3.0, "rng": 4, **options}
    declared = test(
        *_family_args(family, r, r.trains, r.times),
        **settings,
        break_before=r.breaks,
    )
    paused = test(
        *_family_args(family, r, r.shifted_trains, r.shifted_times), **settings
    )
    for unit in declared:
        assert declared[unit].observed_score == pytest.approx(
            paused[unit].observed_score
        )
        np.testing.assert_allclose(
            declared[unit].null_scores, paused[unit].null_scores, equal_nan=True
        )


PREDICATES = {
    "has_place_field": ("spatial", "spatial", "compute_spatial_rate", {}),
    "is_place_cell": (
        "spatial",
        "spatial",
        "compute_spatial_rate",
        {"criterion": "threshold"},
    ),
    "is_head_direction_cell": (
        "directional",
        "directional",
        "compute_directional_rate",
        {},
    ),
    "is_spatial_view_cell": ("view", "view", "compute_view_rate", {}),
    "is_object_vector_cell": (
        "object_vector",
        "egocentric",
        "compute_object_vector_rate",
        {},
    ),
    "is_egocentric_object_vector_cell": (
        "egocentric",
        "egocentric",
        "compute_egocentric_rate",
        {},
    ),
}


@pytest.mark.parametrize("function", list(PREDICATES))
def test_predicates_forward_validated_breaks(recording, function, monkeypatch):
    import importlib

    r = recording
    family, module_name, rate_name, extra = PREDICATES[function]
    module = importlib.import_module(f"neurospatial.encoding.{module_name}")
    original = getattr(module, rate_name)
    seen = []

    def track(*args, **kwargs):
        seen.append(kwargs["break_before"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, rate_name, track)
    getattr(encoding, function)(
        *_family_args(family, r, r.trains[0], r.times),
        **extra,
        break_before=r.breaks.tolist(),
    )
    assert len(seen) == 1
    assert seen[0].dtype == np.bool_
    np.testing.assert_array_equal(seen[0], r.breaks)


ALL_FUNCTIONS = [
    *(f"compute_{name}_rate" for name in FAMILIES),
    *(f"compute_{name}_rates" for name in FAMILIES),
    "compute_directional_place_fields",
    *PREDICATES,
    *SIGNIFICANCE,
]


def _call(function, r, break_before):
    """Call a public encoding function on the recording with ``break_before``."""
    plural = function.endswith("rates") or function.endswith("significance")
    if function in PREDICATES:
        family, _, _, extra = PREDICATES[function]
    elif function in SIGNIFICANCE:
        family, extra = SIGNIFICANCE[function], {"n_shuffles": 2, "min_shift": 1.0}
    elif function == "compute_directional_place_fields":
        return encoding.compute_directional_place_fields(
            r.env,
            r.trains[0],
            r.times,
            r.positions,
            r.labels,
            break_before=break_before,
        )
    else:
        family = (
            function.removeprefix("compute_").removesuffix("s").removesuffix("_rate")
        )
        extra = {
            key: value
            for key, value in FAMILIES[family].items()
            if key == "direction_frame"
        }
    trains = r.trains if plural else r.trains[0]
    return getattr(encoding, function)(
        *_family_args(family, r, trains, r.times), **extra, break_before=break_before
    )


@pytest.mark.parametrize("function", ALL_FUNCTIONS)
@pytest.mark.parametrize("problem", ["length", "dtype"])
def test_invalid_breaks_raise_naming_the_function(recording, function, problem):
    r = recording
    bad = r.breaks[:-1] if problem == "length" else r.breaks.astype(int)
    with pytest.raises(ValueError, match=rf"^{function}\(\) needs break_before"):
        _call(function, r, bad)
