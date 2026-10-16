"""Tests for animation utility functions."""

import numpy as np
import pytest

from neurospatial.animation._utils import _pickling_guidance
from neurospatial.animation.overlays import (
    OverlayData,
    PositionData,
    _validate_pickle_ability,
)


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

    def test_overlay_error_does_not_mention_the_environment_cache(self):
        positions = np.zeros((3, 2))
        overlay_data = OverlayData(
            positions=[
                PositionData(data=positions, color="red", size=5.0, trail_length=1)
            ]
        )
        overlay_data._bad_attr = lambda x: x  # cannot be pickled

        with pytest.raises(ValueError, match="OverlayData is not pickle-able") as info:
            _validate_pickle_ability(overlay_data, n_workers=2)

        message = str(info.value)
        assert "pickle.dumps(overlay)" in message
        assert "n_workers=1" in message
        assert "clear_cache" not in message
        assert "cached analyses" not in message
