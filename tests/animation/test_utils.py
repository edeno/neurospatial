"""Tests for animation utility functions."""

import pytest

from neurospatial.animation._utils import _pickling_guidance


class TestPicklingGuidance:
    """Tests for _pickling_guidance() helper function."""

    @pytest.mark.parametrize("subject", ["environment", "overlays"])
    def test_guidance_names_fixes_that_work(self, subject):
        guidance = _pickling_guidance(subject)
        assert guidance.startswith("Fix:")
        assert "n_workers=1" in guidance
        assert "backend='html'" in guidance
        assert "lambda" in guidance
        # Cached analyses are never pickled, so clearing them fixes nothing.
        assert "clear_cache" not in guidance

    def test_environment_guidance_points_to_attached_objects(self):
        guidance = _pickling_guidance("environment")
        assert "region metadata" in guidance
        assert "pickle.dumps(env)" in guidance
