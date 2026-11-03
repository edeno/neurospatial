"""``keyword_hint(missing=...)`` explains a required option the caller left out."""

from __future__ import annotations

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
