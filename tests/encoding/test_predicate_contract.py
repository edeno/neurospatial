"""Cell screens share defaults and boundaries; shuffle verdicts use raw arrays."""

import inspect
from types import MappingProxyType

import numpy as np
import pytest

from neurospatial import encoding


def _names(f):
    free = "is_spatial_view_cell" if f.name == "view" else f"is_{f.name}_cell"
    method = "is_object_vector_cell" if "object_vector" in f.name else free
    return free, method


def test_is_place_cell_requires_criterion(significance_recording):
    r = significance_recording
    for function, args in [
        (encoding.is_place_cell, (r.env, r.trains[0], r.times, r.positions)),
        (
            encoding.compute_spatial_rate(
                r.env, r.trains[0], r.times, r.positions
            ).is_place_cell,
            (),
        ),
    ]:
        parameter = inspect.signature(function).parameters["criterion"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
        with pytest.raises(TypeError, match="criterion"):
            function(*args)
    with pytest.raises(ValueError, match=r"spatial_info.*shuffle") as exc:
        encoding.is_place_cell(
            r.env, r.trains[0], r.times, r.positions, criterion="threshold"
        )
    assert "has_place_field" in str(exc.value)


def test_threshold_free_method_classify_agree(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    free_name, method_name = _names(f)
    predicate = getattr(f.module, free_name)
    single = getattr(f.module, f.compute_name[:-1])
    criteria = {"criterion": "spatial_info" if f.name == "place" else "threshold"}
    method_kw = criteria if f.name == "place" else {}
    rates = f.compute(*f.args(r), **f.defaults)
    for i, train in enumerate(r.trains):
        result = single(*f.args(r, train), **f.defaults)
        assert predicate(*f.args(r, train), **criteria, **f.defaults) == getattr(
            result, method_name
        )(**method_kw)
        assert getattr(result, method_name)(**method_kw) == rates.classify()[i]
        metric = (
            result.mean_vector_length()
            if f.name == "head_direction"
            else (
                result.view_spatial_information()
                if f.name == "view"
                else result.spatial_information()
            )
        )
        threshold = (
            {"min_mvl": metric, "alpha": 1.01}
            if f.name == "head_direction"
            else {"min_info": metric}
        )
        assert getattr(result, method_name)(**method_kw, **threshold)
        assert predicate(*f.args(r, train), **criteria, **threshold, **f.defaults)
    if f.name != "head_direction":
        metric = (
            rates.view_spatial_information()
            if f.name == "view"
            else rates.spatial_information()
        )
        np.testing.assert_array_equal(
            rates.classify(min_info=metric[0]), metric >= metric[0]
        )


def test_mode_keywords_raise(significance_family, significance_recording):
    f, r = significance_family, significance_recording
    predicate = getattr(f.module, _names(f)[0])
    args = f.args(r, r.trains[0])
    threshold = "min_mvl" if f.name == "head_direction" else "min_info"
    with pytest.raises(ValueError, match="Fix:") as exc:
        predicate(*args, criterion="shuffle", **{threshold: 0.3})
    assert threshold in str(exc.value)
    criterion = "spatial_info" if f.name == "place" else "threshold"
    bad = {"n_shuffles": 10, "min_shift": 1, "rng": 0, "unit_id": 3}
    if f.name != "head_direction":
        bad["alpha"] = 0.1
    with pytest.raises(ValueError, match="Fix:") as exc:
        predicate(*args, criterion=criterion, **bad)
    assert all(key in str(exc.value) for key in bad)
    with pytest.raises(ValueError, match="Fix:") as exc:
        predicate(*args, criterion="percentile")
    assert criterion in str(exc.value) and "shuffle" in str(exc.value)


def test_free_predicates_raise_on_bad_input(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    predicate = getattr(f.module, _names(f)[0])
    args = list(f.args(r, r.trains[0]))
    offset = 1 if f.name == "head_direction" else 2
    args[offset], args[offset + 1] = args[offset + 1], args[offset]
    kwargs = {"criterion": "spatial_info"} if f.name == "place" else {}
    with pytest.raises(ValueError):
        predicate(*args, **kwargs)


def test_threshold_constants_are_the_defaults(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    constant_name = {
        "place": "PLACE_SPATIAL_INFO_THRESHOLDS",
        "head_direction": "HEAD_DIRECTION_THRESHOLDS",
        "view": "VIEW_THRESHOLDS",
    }.get(f.name, "OBJECT_VECTOR_THRESHOLDS")
    constant = getattr(f.module, constant_name)
    assert isinstance(constant, MappingProxyType)
    assert constant_name not in f.module.__all__
    with pytest.raises(TypeError):
        constant["changed"] = 1
    free_name, method_name = _names(f)
    predicate = getattr(f.module, free_name)
    rates = f.compute(*f.args(r), **f.defaults)
    criteria = {"criterion": "spatial_info"} if f.name == "place" else {}
    for fn in [predicate, getattr(rates[0], method_name), rates.classify]:
        for name in constant:
            parameter = inspect.signature(fn).parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default is None
    np.testing.assert_array_equal(rates.classify(), rates.classify(**constant))
    assert predicate(*f.args(r, r.trains[0]), **criteria, **f.defaults) == predicate(
        *f.args(r, r.trains[0]), **criteria, **constant, **f.defaults
    )
    method = getattr(rates[0], method_name)
    assert method(**criteria) == method(**criteria, **constant)


def test_methods_have_no_shuffle(significance_family, significance_recording):
    f, r = significance_family, significance_recording
    rates = f.compute(*f.args(r), **f.defaults)
    assert not hasattr(rates, "shuffle_test") and not hasattr(rates[0], "shuffle_test")
    with pytest.raises(TypeError):
        rates.classify(criterion="shuffle")
    if f.name == "place":
        with pytest.raises(ValueError, match="Fix:") as exc:
            rates[0].is_place_cell(criterion="shuffle")
        assert (
            "is_place_cell(env, spike_times, times, positions, criterion='shuffle')"
            in str(exc.value)
        )


def test_has_place_field_flags_noise(
    ou_2min_env, ou_2min, noise_trains, strong_field_spikes
):
    t, p, _ = ou_2min
    trains = [*noise_trains(120), strong_field_spikes]
    for tr in trains:
        result = encoding.compute_spatial_rate(ou_2min_env, tr, t, p, bandwidth=5)
        assert result.has_place_field()
        assert encoding.has_place_field(ou_2min_env, tr, t, p)
    for fn in [encoding.has_place_field, result.has_place_field]:
        for key, value in encoding.spatial.PLACE_FIELD_DETECTION_DEFAULTS.items():
            assert inspect.signature(fn).parameters[key].default == value
    with pytest.raises(ValueError):
        encoding.has_place_field(ou_2min_env, tr, p, t)


def test_label_constants_and_removed_aliases(significance_recording):
    r = significance_recording
    rates = encoding.compute_spatial_rates(r.env, r.trains, r.times, r.positions)
    constant = encoding.spatial.PLACE_GRID_BORDER_THRESHOLDS
    assert isinstance(constant, MappingProxyType)
    scores = {
        "grid_scores": np.array([0.5, 0.4, 0.3]),
        "border_scores": np.array([0.6, 0.1, 0.2]),
    }
    np.testing.assert_array_equal(
        rates.label_cell_types(**scores), rates.label_cell_types(**constant, **scores)
    )
    for name in constant:
        param = inspect.signature(rates.label_cell_types).parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is None
    for cls, alias in [
        (encoding.SpatialRatesResult, "detect_cell_types"),
        (encoding.DirectionalRatesResult, "detect_hd_cells"),
        (encoding.ViewRatesResult, "detect_view_cells"),
        (encoding.ObjectVectorRatesResult, "detect_ovcs"),
    ]:
        assert not hasattr(cls, alias)
    with pytest.raises(TypeError):
        rates.classify(min_spatial_info=0.5)


@pytest.mark.slow
def test_free_and_population_shuffles_agree(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    predicate = getattr(f.module, _names(f)[0])
    ids = [10, 20, 30]
    population = f.function(
        *f.args(r), unit_ids=ids, n_shuffles=50, rng=0, **f.defaults
    )
    reorder = [2, 0, 1]
    reordered = f.function(
        *f.args(r, [r.trains[i] for i in reorder]),
        unit_ids=[ids[i] for i in reorder],
        n_shuffles=50,
        rng=0,
        **f.defaults,
    )
    for i, uid in enumerate(ids):
        alone = f.function(
            *f.args(r, [r.trains[i]]),
            unit_ids=[uid],
            n_shuffles=50,
            rng=0,
            **f.defaults,
        )[uid]
        np.testing.assert_array_equal(population[uid].p_value, alone.p_value)
        np.testing.assert_array_equal(population[uid].p_value, reordered[uid].p_value)
        assert (
            np.min(np.abs(population[uid].null_scores - population[uid].observed_score))
            > 1e-12
        )
        assert predicate(
            *f.args(r, r.trains[i]),
            criterion="shuffle",
            unit_id=uid,
            n_shuffles=50,
            rng=0,
            **f.defaults,
        ) == (population[uid].p_value < 0.05)


def test_is_place_cell_shuffle_rejects_noise(ou_2min_env, ou_2min, noise_trains):
    t, p, _ = ou_2min
    trains = noise_trains(120)[:5]
    assert all(encoding.has_place_field(ou_2min_env, tr, t, p) for tr in trains)
    flags = [
        encoding.is_place_cell(
            ou_2min_env, tr, t, p, criterion="shuffle", n_shuffles=50, rng=0
        )
        for tr in trains
    ]
    assert sum(flags) <= 2


def test_head_direction_alpha_in_both_modes(significance_recording):
    r = significance_recording
    screen = encoding.is_head_direction_cell(
        r.trains[0], r.times, r.headings, alpha=1.0
    )
    assert isinstance(screen, bool)
    sig = encoding.head_direction_cell_significance(
        [r.trains[0]], r.times, r.headings, n_shuffles=20, rng=0
    )[0]
    for alpha in [sig.p_value, 1.01]:
        assert encoding.is_head_direction_cell(
            r.trains[0],
            r.times,
            r.headings,
            criterion="shuffle",
            n_shuffles=20,
            rng=0,
            alpha=alpha,
        ) == (sig.p_value < alpha)


def test_results_do_not_retain_inputs(significance_family, significance_recording):
    from functools import partial

    f, r = significance_family, significance_recording
    single = getattr(f.module, f.compute_name[:-1])(
        *f.args(r, r.trains[0]), **f.defaults
    )
    plural = f.compute(*f.args(r), **f.defaults)
    inputs = [*r.trains, r.times, r.positions, r.headings, r.objects]
    for result in [single, plural, plural[0]]:
        for name, value in vars(result).items():
            if name == "env":
                continue
            assert not callable(value) and not isinstance(value, partial)
            if isinstance(value, np.ndarray):
                assert not any(
                    np.shares_memory(value, input_array) for input_array in inputs
                )
    rates = plural.firing_rates.copy()
    flags = plural.classify().copy()
    r.positions[:] = 0
    r.times[:] = 0
    r.headings[:] = 0
    np.testing.assert_array_equal(plural.firing_rates, rates)
    np.testing.assert_array_equal(plural.classify(), flags)


def test_is_place_cell_spatial_info(ou_2min_env, ou_2min, noise_trains):
    t, p, _ = ou_2min
    trains = noise_trains(120)
    rates = encoding.compute_spatial_rates(ou_2min_env, trains, t, p)
    flags = [
        encoding.is_place_cell(ou_2min_env, tr, t, p, criterion="spatial_info")
        for tr in trains
    ]
    np.testing.assert_array_equal(flags, rates.classify())
    assert sum(flags) == 15
    assert all(
        rates[i].is_place_cell(criterion="spatial_info") == flags[i] for i in range(20)
    )


def test_heading_interpretation_uses_shared_thresholds(
    significance_recording, monkeypatch
):
    from neurospatial.encoding import directional

    r = significance_recording
    result = directional.compute_directional_rate(r.trains[0], r.times, r.headings)
    assert (
        inspect.signature(result.interpretation).parameters["min_mvl"].default is None
    )
    assert result.interpretation() == result.interpretation(min_mvl=0.4)
    monkeypatch.setattr(
        directional,
        "HEAD_DIRECTION_THRESHOLDS",
        MappingProxyType({"min_mvl": 0.0, "alpha": 0.0}),
    )
    text = result.interpretation()
    assert "Not classified as HD cell" in text
    assert "Rayleigh test not significant" in text
    assert ">= 0.0" in text
