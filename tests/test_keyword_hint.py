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
