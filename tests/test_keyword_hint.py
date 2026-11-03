"""``keyword_hint(missing=...)`` explains a required option the caller left out."""

from __future__ import annotations

import inspect

import pytest

from neurospatial._keyword_hint import keyword_hint


def test_missing_option_raises_with_its_why_and_fix():
    @keyword_hint(missing={"mode": ("the modes differ.", "pass mode='a' or 'b'")})
    def smooth(field: float, *, mode: str) -> str:
        return mode

    with pytest.raises(TypeError) as caught:
        smooth(1.0)

    assert str(caught.value).endswith(
        "smooth() needs mode=.\nWhy: the modes differ.\nFix: pass mode='a' or 'b'"
    )
    assert smooth(1.0, mode="a") == "a"


def test_missing_refuses_an_option_that_has_a_default():
    with pytest.raises(TypeError, match="not required"):

        @keyword_hint(missing={"mode": ("why", "fix")})
        def smooth(field: float, *, mode: str = "a") -> str:
            return mode


class _Env:
    """Stands in for an Environment in the generic env-first tests."""


def _env_first_function():
    @keyword_hint()
    def rate(env, spike_times, times, positions, *, bandwidth=1.0):
        if bandwidth < 0:
            msg = "bandwidth inside the body"
            raise TypeError(msg)
        return (env, positions)

    return rate


def test_env_first_function_without_env_names_the_environment(monkeypatch):
    import neurospatial.environment as environment_module

    monkeypatch.setattr(environment_module, "Environment", _Env)
    rate = _env_first_function()
    with pytest.raises(TypeError) as caught:
        rate([1.0], [0.0, 1.0], [[0.0, 0.0]])
    lines = str(caught.value).splitlines()
    assert "was called without an environment" in lines[0]
    assert lines[-1].endswith(
        "rate(env, spike_times, times, positions, ...), with env first."
    )


def test_env_first_valid_call_and_body_errors_are_unchanged(monkeypatch):
    import neurospatial.environment as environment_module

    monkeypatch.setattr(environment_module, "Environment", _Env)
    rate = _env_first_function()
    env = _Env()
    assert rate(env, [1.0], [0.0], "p") == (env, "p")
    assert rate(env, [1.0], [0.0], positions="p", bandwidth=2.0) == (env, "p")
    # A TypeError raised inside the body is never replaced by the env hint.
    with pytest.raises(TypeError, match="bandwidth inside the body"):
        rate(env, [1.0], [0.0], "p", bandwidth=-1.0)
    # With env given, a different missing argument keeps Python's own message.
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        rate(env, [1.0], [0.0])


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("compute_spatial_rate", "env, spike_times, times, positions"),
        ("compute_spatial_rates", "env, spike_times, times, positions"),
        ("decode_position", "env, spike_counts, encoding_models"),
    ],
)
def test_public_env_first_functions_show_canonical_order(name, canonical):
    import numpy as np

    import neurospatial

    function = getattr(neurospatial, name)
    data = np.zeros(3)
    with pytest.raises(TypeError) as caught:
        function(data, data)
    message = str(caught.value)
    assert "was called without an environment" in message
    assert f"Fix: call {name}({canonical}, ...)" in message
    assert "missing 1 required positional argument" not in message


@pytest.fixture
def env_first(monkeypatch):
    import neurospatial.environment as environment_module

    monkeypatch.setattr(environment_module, "Environment", _Env)

    @keyword_hint()
    def rate(env, spike_times, times, positions, *, dt=None):
        return env

    @keyword_hint()
    def joint(env, counts, models, *, dt):
        return env

    return rate, joint


def _python_message(function, *args, **kwargs):
    """Python's own TypeError text for the undecorated function.

    It includes a "Did you mean" hint only from Python 3.13 on.
    """
    with pytest.raises(TypeError) as expected:
        inspect.unwrap(function)(*args, **kwargs)
    return str(expected.value)


def test_none_env_with_a_typo_keeps_pythons_message(env_first):
    rate, _ = env_first
    with pytest.raises(TypeError) as caught:
        rate(None, [1.0], [0.0], "p", d=1)
    assert str(caught.value) == _python_message(rate, None, [1.0], [0.0], "p", d=1)
    assert "environment" not in str(caught.value)


def test_mapping_env_with_missing_option_keeps_pythons_message(env_first):
    _, joint = env_first
    with pytest.raises(TypeError, match="missing 1 required keyword-only") as caught:
        joint({"a": _Env()}, [1], [2])
    assert "environment" not in str(caught.value)


def test_env_by_keyword_after_positional_data_says_env_goes_first(env_first):
    rate, _ = env_first
    with pytest.raises(TypeError) as caught:
        rate([1.0], [0.0], "p", env=_Env())
    message = str(caught.value)
    assert "not the first positional argument" in message
    assert "without an environment" not in message
    assert message.endswith(
        "(env, spike_times, times, positions, ...), with env first."
    )
    assert isinstance(caught.value.__cause__, TypeError)


def test_env_in_second_position_says_env_goes_first(env_first):
    rate, _ = env_first
    with pytest.raises(TypeError, match="not the first positional argument"):
        rate([1.0], _Env(), [0.0], "p", d=1)


def test_env_first_with_typo_is_not_blamed_on_the_environment(env_first):
    rate, _ = env_first
    with pytest.raises(TypeError) as caught:
        rate(_Env(), [1.0], [0.0], "p", d=1)
    assert str(caught.value) == _python_message(rate, _Env(), [1.0], [0.0], "p", d=1)


@pytest.mark.parametrize("slot", [1, 2, 3])
def test_env_in_any_later_slot_of_a_call_that_binds_says_env_goes_first(
    env_first, slot
):
    """With every positional slot filled the call binds, so the check cannot
    wait for a binding error."""
    rate, _ = env_first
    args = [[1.0], [0.0], "p"]
    args.insert(slot, _Env())
    with pytest.raises(TypeError, match="not the first positional argument"):
        rate(*args)
