"""Free predicates give shuffle verdicts by default; result methods are screens."""

import inspect
import warnings
from types import MappingProxyType

import numpy as np
import pytest

from neurospatial import encoding


def _names(f):
    free = "is_spatial_view_cell" if f.name == "view" else f"is_{f.name}_cell"
    method = "is_object_vector_cell" if "object_vector" in f.name else free
    return free, method


FREE_PREDICATES = (
    "is_place_cell",
    "is_head_direction_cell",
    "is_object_vector_cell",
    "is_egocentric_object_vector_cell",
    "is_spatial_view_cell",
)


def test_free_predicates_default_to_shuffle():
    for name in FREE_PREDICATES:
        function = getattr(encoding, name)
        parameter = inspect.signature(function).parameters["criterion"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default == "shuffle", name
        assert inspect.getdoc(function).startswith("Test whether "), name


def test_invalid_criterion_names_has_place_field(significance_recording):
    r = significance_recording
    with pytest.raises(ValueError, match=r"threshold.*shuffle") as exc:
        encoding.is_place_cell(
            r.env, r.trains[0], r.times, r.positions, criterion="spatial_info"
        )
    assert "has_place_field" in str(exc.value)


def test_result_screens_take_no_criterion(significance_family, significance_recording):
    f, r = significance_family, significance_recording
    method_name = _names(f)[1]
    rates = f.compute(*f.args(r), **f.defaults)
    for screen in (getattr(rates[0], method_name), rates.classify):
        assert "criterion" not in inspect.signature(screen).parameters
        assert inspect.getdoc(screen).startswith("Screen "), screen.__qualname__
    with pytest.raises(TypeError, match="criterion"):
        getattr(rates[0], method_name)(criterion="threshold")
    if f.name == "place":
        rate = encoding.compute_spatial_rate(r.env, r.trains[0], r.times, r.positions)
        for criterion in ("threshold", "shuffle"):
            with pytest.raises(TypeError, match="criterion"):
                rate.is_place_cell(criterion=criterion)


@pytest.mark.filterwarnings("ignore:The animal never faced:UserWarning")
def test_result_screens_refuse_none_thresholds(
    significance_family, significance_recording
):
    """None is not a cutoff: each screen says so instead of a bare TypeError."""
    from neurospatial import InputError

    f, r = significance_family, significance_recording
    method_name = _names(f)[1]
    rates = f.compute(*f.args(r), **f.defaults)
    names = ("min_mvl", "alpha") if f.name == "head_direction" else ("min_info",)
    for name in names:
        for screen in (getattr(rates[0], method_name), rates.classify):
            with pytest.raises(InputError, match=rf"{name}=None is not a threshold"):
                screen(**{name: None})


def test_cell_type_labels_refuse_none_thresholds(significance_recording):
    from neurospatial import InputError

    r = significance_recording
    rates = encoding.compute_spatial_rates(r.env, r.trains, r.times, r.positions)
    for name in ("min_spatial_info", "min_grid_score", "min_border_score"):
        with pytest.raises(InputError, match=rf"{name}=None is not a threshold"):
            rates.label_cell_types(**{name: None})
    args = (r.env, r.trains, r.times, r.positions)
    place = encoding.place_cell_significance(*args, n_shuffles=19, rng=0)
    for name in ("alpha", "min_info"):
        with pytest.raises(InputError, match=rf"{name}=None is not a threshold"):
            place.classify(**{name: None})
    types = encoding.spatial_cell_type_significance(*args, n_shuffles=19, rng=0)
    for name in ("alpha", "min_spatial_info"):
        with pytest.raises(InputError, match=rf"{name}=None is not a threshold"):
            types.label_cell_types(**{name: None})
        for correction in (None, "holm"):
            with pytest.raises(InputError, match=rf"{name}=None is not a threshold"):
                types.summary_table(**{name: None}, correction=correction)
    for name in ("alpha", "min_info"):
        with pytest.raises(InputError, match=rf"{name}=None is not a threshold"):
            place.summary_table(**{name: None})


def test_default_predicate_matches_explicit_shuffle(
    ou_2min_env, ou_2min, noise_trains, strong_field_spikes
):
    """The default verdict is the 1000-shift test, the population rule's own."""
    t, p, _ = ou_2min
    trains = noise_trains(120)
    ids = [1, 17, 99]  # two untuned units significant by chance, one place cell
    population = [trains[1], trains[17], strong_field_spikes]
    significance = encoding.place_cell_significance(
        ou_2min_env, population, t, p, unit_ids=ids, rng=0
    )
    rates = encoding.compute_spatial_rates(ou_2min_env, population, t, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        debiased = rates.spatial_information() - rates.spatial_information_bias()
    verdicts = []
    for uid, train, beyond_bias in zip(ids, population, debiased >= 0.5, strict=True):
        default = encoding.is_place_cell(ou_2min_env, train, t, p, unit_id=uid, rng=0)
        explicit = encoding.is_place_cell(
            ou_2min_env, train, t, p, criterion="shuffle", unit_id=uid, rng=0
        )
        assert default is explicit
        assert significance.sel(unit_id=uid).null_scores.shape == (1000,)
        assert default == bool(
            significance.sel(unit_id=uid).p_value < 0.05 and beyond_bias
        )
        verdicts.append(default)
    assert verdicts == [False, False, True]


@pytest.mark.filterwarnings("ignore:The animal never faced:UserWarning")
def test_threshold_free_method_classify_agree(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    free_name, method_name = _names(f)
    predicate = getattr(f.module, free_name)
    single = getattr(f.module, f.compute_name[:-1])
    criteria = {"criterion": "threshold"}
    method_kw = {}
    rates = f.compute(*f.args(r), **f.defaults)
    for i, train in enumerate(r.trains):
        result = single(*f.args(r, train), **f.defaults)
        assert predicate(*f.args(r, train), **criteria, **f.defaults) == getattr(
            result, method_name
        )(**method_kw)
        assert getattr(result, method_name)(**method_kw) == rates.classify()[i]
        metric = (
            result.mean_vector_length
            if f.name == "head_direction"
            else result.spatial_information() - result.spatial_information_bias()
        )
        if not np.isfinite(metric):
            continue
        threshold = (
            {"min_mvl": metric, "alpha": 1.01}
            if f.name == "head_direction"
            else {"min_info": metric}
        )
        assert getattr(result, method_name)(**method_kw, **threshold)
        assert predicate(*f.args(r, train), **criteria, **threshold, **f.defaults)
    if f.name != "head_direction":
        metric = rates.spatial_information() - rates.spatial_information_bias()
        finite = np.flatnonzero(np.isfinite(metric))
        assert finite.size
        cutoff = metric[finite[0]]
        np.testing.assert_array_equal(rates.classify(min_info=cutoff), metric >= cutoff)


@pytest.mark.filterwarnings("ignore:The animal never faced:UserWarning")
def test_mode_keywords_raise(significance_family, significance_recording):
    f, r = significance_family, significance_recording
    predicate = getattr(f.module, _names(f)[0])
    args = f.args(r, r.trains[0])
    if f.name == "head_direction":
        # The information predicates' min_info applies to both criteria.
        with pytest.raises(ValueError, match="Fix:") as exc:
            predicate(*args, criterion="shuffle", min_mvl=0.3)
        assert "min_mvl" in str(exc.value)
    criterion = "threshold"
    bad = {"n_shuffles": 10, "min_shift": 1, "rng": 0, "unit_id": 3}
    if f.name != "head_direction":
        bad["alpha"] = 0.1
    with pytest.raises(ValueError, match="Fix:") as exc:
        predicate(*args, criterion=criterion, **bad)
    assert all(key in str(exc.value) for key in bad)
    # A keyword spelled at its default changes nothing, so it is accepted.
    at_default = {"n_shuffles": 1000, "min_shift": 20.0}
    assert predicate(
        *args, criterion=criterion, **at_default, **f.defaults
    ) == predicate(*args, criterion=criterion, **f.defaults)
    with pytest.raises(ValueError, match="Fix:") as exc:
        predicate(*args, criterion="percentile")
    assert criterion in str(exc.value)
    assert "shuffle" in str(exc.value)


def test_free_predicates_raise_on_bad_input(
    significance_family, significance_recording
):
    f, r = significance_family, significance_recording
    predicate = getattr(f.module, _names(f)[0])
    args = list(f.args(r, r.trains[0]))
    offset = 1 if f.name == "head_direction" else 2
    args[offset], args[offset + 1] = args[offset + 1], args[offset]
    kwargs = {"criterion": "threshold"}
    # Swapped, a heading series read as times decreases; 2-D positions are not 1-D.
    swapped = (
        r"times must be strictly increasing"
        if f.name == "head_direction"
        else r"times must be 1-D"
    )
    with pytest.raises(ValueError, match=swapped):
        predicate(*args, **kwargs)


@pytest.mark.filterwarnings("ignore:ChainedAssignmentError:FutureWarning")
@pytest.mark.filterwarnings("ignore:The animal never faced:UserWarning")
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
    criteria = {"criterion": "threshold"}
    for fn in [predicate, getattr(rates[0], method_name), rates.classify]:
        for name in constant:
            parameter = inspect.signature(fn).parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
            assert parameter.default == constant[name], (fn.__qualname__, name)
    np.testing.assert_array_equal(rates.classify(), rates.classify(**constant))
    assert predicate(*f.args(r, r.trains[0]), **criteria, **f.defaults) == predicate(
        *f.args(r, r.trains[0]), **criteria, **constant, **f.defaults
    )
    method = getattr(rates[0], method_name)
    assert method() == method(**constant)


def test_shuffle_defaults_have_one_source():
    """Every shuffle test's and verdict's defaults are the shared shuffle defaults."""
    from neurospatial.encoding._significance import _SHUFFLE_DEFAULTS

    tests = [
        getattr(encoding, name)
        for name in encoding.__all__
        if name.endswith("_significance")
    ]
    assert len(tests) == 6
    for fn in tests:
        parameters = inspect.signature(fn).parameters
        for name in ("n_shuffles", "min_shift"):
            assert parameters[name].default == _SHUFFLE_DEFAULTS[name], fn.__name__
    for name in FREE_PREDICATES:
        parameters = inspect.signature(getattr(encoding, name)).parameters
        for key, value in _SHUFFLE_DEFAULTS.items():
            assert parameters[key].default == value, (name, key)


def test_methods_have_no_shuffle(significance_family, significance_recording):
    f, r = significance_family, significance_recording
    rates = f.compute(*f.args(r), **f.defaults)
    assert not hasattr(rates, "shuffle_test")
    assert not hasattr(rates[0], "shuffle_test")
    with pytest.raises(TypeError):
        rates.classify(criterion="shuffle")


def test_has_place_field_flags_noise(
    ou_2min_env, ou_2min, noise_trains, strong_field_spikes
):
    t, p, _ = ou_2min
    noise = noise_trains(120)
    method_flags = [
        encoding.compute_spatial_rate(
            ou_2min_env, tr, t, p, bandwidth=5
        ).has_place_field()
        for tr in noise
    ]
    function_flags = [
        encoding.has_place_field(ou_2min_env, tr, t, p, bandwidth=5) for tr in noise
    ]
    assert method_flags == function_flags
    # Detection is not a verdict: untuned 0.5 Hz units still have fields.
    assert sum(method_flags) >= 5
    assert encoding.has_place_field(ou_2min_env, strong_field_spikes, t, p)
    result = encoding.compute_spatial_rate(ou_2min_env, noise[0], t, p)
    tr = noise[0]
    for fn in [encoding.has_place_field, result.has_place_field]:
        for key, value in encoding.spatial.PLACE_FIELD_DETECTION_DEFAULTS.items():
            assert inspect.signature(fn).parameters[key].default == value
    with pytest.raises(ValueError, match=r"times must be 1-D"):
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
        rates.label_cell_types(**scores, boundary_source="analysis"),
        rates.label_cell_types(**constant, **scores, boundary_source="analysis"),
    )
    for name in constant:
        param = inspect.signature(rates.label_cell_types).parameters[name]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default == constant[name]
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
# An unsmoothed binned map can have more independent bins than a unit has
# spikes; its verdict is then False with this warning, as the rule expects.
@pytest.mark.filterwarnings("ignore:.*fewer spikes than their rate maps:UserWarning")
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
    if f.name == "head_direction":
        beyond_bias = np.ones(len(ids), dtype=bool)
    else:
        # The information verdicts also need bias-corrected information.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rates = f.compute(*f.args(r), **f.defaults)
            debiased = rates.spatial_information() - rates.spatial_information_bias()
        beyond_bias = debiased >= _information_cutoff(f)
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
        ) == (population[uid].p_value < 0.05 and beyond_bias[i])


def _information_cutoff(f):
    """The family's default bias-corrected information cutoff (bits/spike)."""
    constant = {
        "place": "PLACE_SPATIAL_INFO_THRESHOLDS",
        "view": "VIEW_THRESHOLDS",
    }.get(f.name, "OBJECT_VECTOR_THRESHOLDS")
    return getattr(f.module, constant)["min_info"]


@pytest.mark.parametrize(
    "significance_family",
    ["place", "view", "egocentric_object_vector"],
    indirect=True,
)
@pytest.mark.filterwarnings("ignore:.*fewer spikes than their rate maps:UserWarning")
def test_shuffle_verdict_needs_bias_corrected_information(
    significance_family, ou_2min_env, ou_2min, noise_trains, obj
):
    """Untuned units significant by chance in 2 minutes are not tuned cells.

    About alpha of untuned units pass the information test by chance: the
    count of the 20 units with p < 0.05 must lie in the central 99.9% of
    Binomial(20, alpha_eff), alpha_eff = 4/100 the chance that p = (k+1)/100
    falls below 0.05. Those seeded units include at least one chance
    rejection, which the rest of the test needs. Their information beyond
    the untuned-cell bias is far below the cutoff (or unknown, with fewer
    spikes than independent bins), so the shuffle verdict refuses them, as
    the shuffle labels do.
    """
    from types import SimpleNamespace

    from scipy.stats import binom

    f = significance_family
    times, positions, headings = ou_2min
    recording = SimpleNamespace(
        env=ou_2min_env,
        times=times,
        positions=positions,
        headings=headings,
        objects=obj,
        trains=noise_trains(120),
    )
    settings = {"boundary_source": "analysis"} if f.name == "view" else {}
    shuffle = {"n_shuffles": 99, "rng": 0}
    ids = np.arange(len(recording.trains))
    tests = f.function(*f.args(recording), unit_ids=ids, **shuffle, **settings)
    rates = f.compute(*f.args(recording), **settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        debiased = rates.spatial_information() - rates.spatial_information_bias()
    chance = [uid for uid in ids if tests[uid].p_value < 0.05]
    n_values = shuffle["n_shuffles"] + 1
    alpha_eff = (np.ceil(0.05 * n_values) - 1) / n_values
    low, high = binom.ppf([0.0005, 0.9995], len(ids), alpha_eff)
    assert low <= len(chance) <= high
    assert chance  # the precondition for the checks below
    assert not (debiased[chance] >= _information_cutoff(f)).any()
    predicate = getattr(f.module, _names(f)[0])
    for uid in chance:
        arguments = f.args(recording, recording.trains[uid])
        unknown = np.isnan(debiased[uid])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            verdict = predicate(
                *arguments, criterion="shuffle", unit_id=uid, **shuffle, **settings
            )
        assert verdict is False
        assert unknown == any(
            "information bias is unknown" in str(w.message) for w in caught
        )
        # The cutoff is the only thing refusing them.
        assert (
            predicate(
                *arguments,
                criterion="shuffle",
                unit_id=uid,
                min_info=-np.inf,
                **shuffle,
                **settings,
            )
            is not unknown
        )


def test_shuffle_verdict_agrees_with_shuffle_labels(
    ou_2min_env, ou_2min, noise_trains, strong_field_spikes
):
    """One place rule for one unit alone and in a population, as labels do."""
    t, p, _ = ou_2min
    trains = noise_trains(120)
    ids = [1, 17, 99]  # two untuned units significant by chance, one place cell
    population = [trains[1], trains[17], strong_field_spikes]
    shuffle = {"n_shuffles": 99, "rng": 0}
    labels = encoding.spatial_cell_type_significance(
        ou_2min_env,
        population,
        t,
        p,
        unit_ids=ids,
        boundary_source="analysis",
        **shuffle,
    ).summary_table()
    assert (labels["spatial_information_p_value"] < 0.05).all()
    verdicts = [
        encoding.is_place_cell(
            ou_2min_env, train, t, p, criterion="shuffle", unit_id=uid, **shuffle
        )
        for uid, train in zip(ids, population, strict=True)
    ]
    assert verdicts == [False, False, True]
    assert verdicts == labels["passes_spatial_information"].tolist()


def test_is_place_cell_shuffle_rejects_noise(ou_2min_env, ou_2min, noise_trains):
    t, p, _ = ou_2min
    trains = [
        tr
        for tr in noise_trains(120)
        if encoding.has_place_field(ou_2min_env, tr, t, p)
    ][:5]
    assert len(trains) == 5
    flags = [
        encoding.is_place_cell(
            ou_2min_env, tr, t, p, criterion="shuffle", n_shuffles=50, rng=0
        )
        for tr in trains
    ]
    assert sum(flags) <= 2


@pytest.mark.filterwarnings("ignore:The animal never faced:UserWarning")
def test_head_direction_alpha_in_both_modes(significance_recording):
    r = significance_recording
    screen = encoding.is_head_direction_cell(
        r.trains[0], r.times, r.headings, criterion="threshold", alpha=1.0
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


@pytest.mark.filterwarnings("ignore:The animal never faced:UserWarning")
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
            assert not callable(value)
            assert not isinstance(value, partial)
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


def test_is_place_cell_threshold(ou_2min_env, ou_2min, noise_trains):
    t, p, _ = ou_2min
    trains = noise_trains(120)
    rates = encoding.compute_spatial_rates(ou_2min_env, trains, t, p, bandwidth=5.0)
    flags = [
        encoding.is_place_cell(
            ou_2min_env, tr, t, p, criterion="threshold", bandwidth=5.0
        )
        for tr in trains
    ]
    np.testing.assert_array_equal(flags, rates.classify())
    assert all(rates[i].is_place_cell() == flags[i] for i in range(20))
    # Untuned 0.5 Hz units (about 60 spikes): the uncorrected cutoff passes
    # most of them by chance; the bias-corrected screen passes none.
    assert (rates.spatial_information() >= 0.5).sum() >= 10
    assert sum(flags) == 0


@pytest.mark.filterwarnings("ignore:The animal never faced:UserWarning")
def test_heading_interpretation_uses_shared_thresholds(significance_recording):
    from neurospatial.encoding import directional

    r = significance_recording
    result = directional.compute_directional_rate(r.trains[0], r.times, r.headings)
    constant = directional.HEAD_DIRECTION_THRESHOLDS
    parameters = inspect.signature(result.interpretation).parameters
    for name in constant:
        assert parameters[name].default == constant[name], name
    assert result.interpretation() == result.interpretation(**constant)
    text = result.interpretation(min_mvl=0.0, alpha=0.0)
    assert "Not classified as HD cell" in text
    assert "Tuning test not significant" in text
    assert ">= 0.0" in text
