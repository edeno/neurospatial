"""Test validation and error handling in rendering utilities."""

import numpy as np
import pytest

from neurospatial import Environment


def test_field_to_rgb_for_napari_wrong_shape():
    """Test that field_to_rgb_for_napari validates field shape."""
    pytest.importorskip("matplotlib")

    from neurospatial.animation.rendering import field_to_rgb_for_napari

    rng = np.random.default_rng(42)
    positions = rng.standard_normal((100, 2)) * 50
    env = Environment.from_samples(positions, bin_size=10.0)

    # Create colormap lookup
    from matplotlib import pyplot as plt

    cmap_obj = plt.get_cmap("viridis")
    cmap_lookup = (cmap_obj(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    # Field with wrong number of values
    wrong_field = rng.random(env.n_bins + 10)  # Too many values

    with pytest.raises(ValueError) as exc_info:
        field_to_rgb_for_napari(env, wrong_field, cmap_lookup, vmin=0, vmax=1)

    # Check error message is informative
    error_msg = str(exc_info.value)
    assert "Field has" in error_msg
    assert "values but environment has" in error_msg
    assert str(env.n_bins) in error_msg
    assert "Expected shape:" in error_msg
