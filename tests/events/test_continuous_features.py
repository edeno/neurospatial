"""Feature metadata validation for continuous event alignment."""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial.events._continuous import _check_category_codes, _resolve_features


def _resolve(n, **kw):
    args = {
        "feature_ids": None,
        "feature_units": None,
        "value_kinds": None,
        "category_labels": None,
        "interpolation": None,
        "call": "align_continuous_to_events",
    }
    args.update(kw)
    return _resolve_features(n, **args)


def _mixed(**kw):
    args = {
        "feature_ids": ["speed", "heading", "zone"],
        "value_kinds": {"heading": "circular", "zone": "categorical"},
        "category_labels": {"zone": {1: "arm", 0: "center"}},
    }
    args.update(kw)
    return _resolve(3, **args)


def test_defaults():
    spec = _resolve(2)
    np.testing.assert_array_equal(spec.feature_ids, [0, 1])
    assert spec.value_kinds == ("linear", "linear")
    assert spec.interpolation == ("linear", "linear")
    assert spec.units == (None, None)
    assert len(spec.category_features) == 0
    assert len(spec.category_codes) == 0
    assert len(spec.category_labels) == 0


def test_mixed_kinds_resolved():
    spec = _mixed(feature_units={"speed": "cm/s"})
    assert spec.value_kinds == ("linear", "circular", "categorical")
    assert spec.interpolation == ("linear", "circular", "previous")
    assert spec.units == ("cm/s", "rad", None)
    np.testing.assert_array_equal(spec.category_codes, [0, 1])
    np.testing.assert_array_equal(spec.category_labels, ["center", "arm"])
    np.testing.assert_array_equal(spec.category_features, [2, 2])
    assert spec.category_codes.dtype == np.int64
    assert spec.category_features.dtype == np.intp


def test_code_zero_is_a_category():
    spec = _mixed()
    assert 0 in spec.category_codes
    assert spec.category_labels[list(spec.category_codes).index(0)] == "center"


def test_valid_explicit_choices():
    spec = _mixed(
        interpolation={"speed": "previous", "heading": "previous"},
        feature_units={"heading": "radians"},
    )
    assert spec.interpolation == ("previous", "previous", "previous")
    assert spec.units[1] == "radians"


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"interpolation": {"zone": "linear"}}, "zone"),
        ({"feature_units": {"heading": "deg"}}, "np.deg2rad"),
        ({"value_kinds": {"nope": "linear"}}, "speed"),
        ({"value_kinds": {"speed": "angular"}}, "angular"),
        ({"interpolation": {"heading": "linear"}}, "heading"),
        ({"feature_units": {"nope": "m"}}, "nope"),
        ({"category_labels": {"speed": {0: "a"}}}, "speed"),
        ({"category_labels": {"zone": {True: "a"}}}, "zone"),
        ({"category_labels": {"zone": {1.5: "a"}}}, "zone"),
        ({"category_labels": {"zone": {0: 3}}}, "zone"),
        ({"category_labels": {"zone": {}}}, "zone"),
    ],
)
def test_metadata_errors(kw, match):
    base = {
        "value_kinds": {"heading": "circular", "zone": "categorical"},
        "category_labels": {"zone": {0: "a"}},
    }
    base.update(kw)
    if "speed" in kw.get("category_labels", {}):
        base["category_labels"] = {**kw["category_labels"], "zone": {0: "a"}}
    with pytest.raises(ValueError, match=match):
        _resolve(3, feature_ids=["speed", "heading", "zone"], **base)


def test_unknown_key_lists_valid_ids():
    with pytest.raises(ValueError, match=r"(?s)nope.*speed"):
        _resolve(1, feature_ids=["speed"], value_kinds={"nope": "linear"})


def test_categorical_without_labels():
    with pytest.raises(ValueError, match=r"zone.*category_labels"):
        _resolve(1, feature_ids=["zone"], value_kinds={"zone": "categorical"})


def test_numpy_bool_code_rejected():
    with pytest.raises(ValueError, match="zone"):
        _resolve(
            1,
            feature_ids=["zone"],
            value_kinds={"zone": "categorical"},
            category_labels={"zone": {np.True_: "a"}},
        )


def test_numpy_integer_code_accepted():
    spec = _resolve(
        1,
        feature_ids=["zone"],
        value_kinds={"zone": "categorical"},
        category_labels={"zone": {np.int32(3): "a"}},
    )
    assert spec.category_codes.tolist() == [3]


def test_interpolation_error_explains_and_fixes():
    with pytest.raises(ValueError) as err:
        _mixed(interpolation={"zone": "linear"})
    text = str(err.value)
    assert "not a category" in text
    assert "interpolation={'zone': 'previous'}" in text


def test_feature_id_errors():
    with pytest.raises(ValueError, match="duplicate"):
        _resolve(2, feature_ids=["a", "a"])
    with pytest.raises(ValueError, match="3 feature_ids"):
        _resolve(2, feature_ids=["a", "b", "c"])


def test_spec_arrays_are_read_only_and_frozen():
    spec = _mixed()
    for array in (
        spec.feature_ids,
        spec.category_features,
        spec.category_codes,
        spec.category_labels,
    ):
        assert array.flags.writeable is False
    with pytest.raises(AttributeError):
        spec.units = ()  # type: ignore[misc]


@pytest.mark.parametrize("code", ["a", None])
def test_non_integer_code_key_raises_actionable_error(code):
    with pytest.raises(
        ValueError, match=r"code .* for feature 'zone' is not an integer"
    ):
        _resolve(
            1,
            feature_ids=["zone"],
            value_kinds={"zone": "categorical"},
            category_labels={"zone": {0: "center", code: "arm"}},
        )


def _spec_zone(codes=(0, 1)):
    return _resolve(
        1,
        feature_ids=["zone"],
        value_kinds={"zone": "categorical"},
        category_labels={"zone": {c: f"c{c}" for c in codes}},
    )


def test_check_codes_passes_and_ignores_nan():
    sig = np.array([[0], [1], [np.nan], [1]], dtype=float)
    _check_category_codes(sig, np.isfinite(sig), _spec_zone(), call="f")


def test_check_codes_unlisted_code_raises():
    sig = np.array([[0.0], [2.0]])
    with pytest.raises(ValueError, match=r"zone.*value 2\.0 at sample 1"):
        _check_category_codes(sig, np.ones((2, 1), bool), _spec_zone(), call="f")


def test_check_codes_non_integer_raises():
    sig = np.array([[0.0], [0.5]])
    with pytest.raises(ValueError, match=r"zone.*value 0\.5 at sample 1"):
        _check_category_codes(sig, np.ones((2, 1), bool), _spec_zone(), call="f")


def test_check_codes_ignores_invalid_samples():
    sig = np.array([[0.0], [7.0], [1.0]])
    valid = np.array([[True], [False], [True]])
    _check_category_codes(sig, valid, _spec_zone(), call="f")


def test_check_codes_only_categorical_columns():
    spec = _mixed()
    sig = np.column_stack([np.array([0.3, 9.9]), np.array([1.0, 2.0]), [0.0, 1.0]])
    _check_category_codes(sig, np.ones((2, 3), bool), spec, call="f")
