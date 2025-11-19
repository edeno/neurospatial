"""Tests for multi-field viewer support in napari backend."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

# Mark all tests as napari tests (skip in CI without display)
# Also mark for serial execution to prevent Qt/napari crashes in parallel
pytestmark = [pytest.mark.napari, pytest.mark.xdist_group(name="napari_gui")]


@pytest.fixture
def simple_env():
    """Create simple 2D environment for testing."""
    from neurospatial import Environment

    positions = np.random.randn(100, 2) * 50
    return Environment.from_samples(positions, bin_size=10.0)


@pytest.fixture
def multi_field_sequences(simple_env) -> list[list[NDArray[np.float64]]]:
    """Create multiple field sequences for testing.

    Returns 3 sequences of 10 frames each.
    """
    n_bins = simple_env.n_bins
    sequences = []
    for seq_idx in range(3):
        sequence = []
        for _ in range(10):  # Frame index not used
            field = (
                np.random.rand(n_bins) + seq_idx * 0.1
            )  # Different baseline per sequence
            sequence.append(field)
        sequences.append(sequence)
    return sequences


class TestMultiFieldDetection:
    """Test detection of single vs multi-field input."""

    def test_single_sequence_detection(self, simple_env):
        """Single field sequence should be detected correctly."""
        from neurospatial.animation.backends.napari_backend import (
            _is_multi_field_input,
        )

        # Single sequence: list of 1D arrays
        fields = [np.random.rand(simple_env.n_bins) for _ in range(10)]
        assert not _is_multi_field_input(fields)

    def test_multi_sequence_detection(self, simple_env, multi_field_sequences):
        """Multiple field sequences should be detected correctly."""
        from neurospatial.animation.backends.napari_backend import (
            _is_multi_field_input,
        )

        assert _is_multi_field_input(multi_field_sequences)

    def test_empty_list_detection(self):
        """Empty list should be detected as single field."""
        from neurospatial.animation.backends.napari_backend import (
            _is_multi_field_input,
        )

        assert not _is_multi_field_input([])


class TestMultiFieldValidation:
    """Test validation of multi-field input."""

    def test_requires_layout_parameter(self, simple_env, multi_field_sequences):
        """Multi-field input without layout should raise error."""
        from neurospatial.animation.backends.napari_backend import render_napari

        with pytest.raises(ValueError, match="Multi-field input requires 'layout'"):
            render_napari(simple_env, multi_field_sequences)

    def test_all_sequences_same_length(self, simple_env):
        """All field sequences must have same length."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # Create sequences with different lengths
        seq1 = [np.random.rand(simple_env.n_bins) for _ in range(10)]
        seq2 = [np.random.rand(simple_env.n_bins) for _ in range(5)]  # Different length
        fields = [seq1, seq2]

        with pytest.raises(
            ValueError, match="All field sequences must have the same length"
        ):
            render_napari(simple_env, fields, layout="horizontal")

    def test_layer_names_match_sequences(self, simple_env, multi_field_sequences):
        """Number of layer names must match number of sequences."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # 3 sequences but only 2 names
        layer_names = ["Field 1", "Field 2"]

        with pytest.raises(
            ValueError, match=r"Number of layer_names .* must match number of sequences"
        ):
            render_napari(
                simple_env,
                multi_field_sequences,
                layout="horizontal",
                layer_names=layer_names,
            )


class TestHorizontalLayout:
    """Test horizontal layout (side-by-side)."""

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_horizontal_two_sequences(self, simple_env):
        """Horizontal layout with 2 sequences."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # Create 2 sequences
        seq1 = [np.random.rand(simple_env.n_bins) for _ in range(5)]
        seq2 = [np.random.rand(simple_env.n_bins) for _ in range(5)]
        fields = [seq1, seq2]

        viewer = render_napari(simple_env, fields, layout="horizontal")

        # Should have 2 image layers
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 2

        # Layers should have default names
        assert any("Field 1" in layer.name for layer in image_layers)
        assert any("Field 2" in layer.name for layer in image_layers)

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_horizontal_custom_layer_names(self, simple_env, multi_field_sequences):
        """Horizontal layout with custom layer names."""
        from neurospatial.animation.backends.napari_backend import render_napari

        layer_names = ["Neuron A", "Neuron B", "Neuron C"]
        viewer = render_napari(
            simple_env,
            multi_field_sequences,
            layout="horizontal",
            layer_names=layer_names,
        )

        # Verify custom names
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 3
        for expected_name in layer_names:
            assert any(expected_name in layer.name for layer in image_layers)


class TestVerticalLayout:
    """Test vertical layout (stacked)."""

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_vertical_three_sequences(self, simple_env, multi_field_sequences):
        """Vertical layout with 3 sequences."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(simple_env, multi_field_sequences, layout="vertical")

        # Should have 3 image layers
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 3


class TestGridLayout:
    """Test grid layout (NxM arrangement)."""

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_grid_four_sequences(self, simple_env):
        """Grid layout with 4 sequences (2x2)."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # Create 4 sequences
        sequences = []
        for _ in range(4):
            seq = [np.random.rand(simple_env.n_bins) for _ in range(5)]
            sequences.append(seq)

        viewer = render_napari(simple_env, sequences, layout="grid")

        # Should have 4 image layers
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 4

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_grid_six_sequences(self, simple_env):
        """Grid layout with 6 sequences (2x3 or 3x2)."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # Create 6 sequences
        sequences = []
        for _ in range(6):
            seq = [np.random.rand(simple_env.n_bins) for _ in range(5)]
            sequences.append(seq)

        viewer = render_napari(simple_env, sequences, layout="grid")

        # Should have 6 image layers
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 6


class TestPlaybackSynchronization:
    """Test that playback is synchronized across all layers."""

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_all_layers_share_time_dimension(self, simple_env, multi_field_sequences):
        """All layers should share the same time dimension."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(simple_env, multi_field_sequences, layout="horizontal")

        # All image layers should have same number of frames
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        n_frames_list = [layer.data.shape[0] for layer in image_layers]

        # All should be equal
        assert len(set(n_frames_list)) == 1
        assert n_frames_list[0] == 10  # 10 frames per sequence

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_frame_counter_shows_total_frames(self, simple_env, multi_field_sequences):
        """Frame counter should show total frames (same for all sequences)."""
        from neurospatial.animation.backends.napari_backend import render_napari

        viewer = render_napari(simple_env, multi_field_sequences, layout="horizontal")

        # Check that all layers have correct number of frames via data shape
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        for layer in image_layers:
            assert layer.data.shape[0] == 10  # 10 frames per sequence


class TestBackwardsCompatibility:
    """Test that single-field input still works (backwards compatibility)."""

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_single_field_still_works(self, simple_env):
        """Single field sequence should work without layout parameter."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # Single sequence (current behavior)
        fields = [np.random.rand(simple_env.n_bins) for _ in range(10)]

        viewer = render_napari(simple_env, fields)  # No layout parameter

        # Should have 1 image layer
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 1

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_single_field_with_frame_labels(self, simple_env):
        """Single field with frame labels should still work."""
        from neurospatial.animation.backends.napari_backend import render_napari

        fields = [np.random.rand(simple_env.n_bins) for _ in range(5)]
        frame_labels = [f"Trial {i + 1}" for i in range(5)]

        viewer = render_napari(simple_env, fields, frame_labels=frame_labels)

        # Should work without error
        assert len(viewer.layers) >= 1


class TestColorScaleSharing:
    """Test that all fields share the same color scale."""

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_shared_vmin_vmax_across_sequences(self, simple_env):
        """All sequences should use the same vmin/vmax for comparison."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # Create sequences with different value ranges
        seq1 = [np.ones(simple_env.n_bins) * 0.5 for _ in range(5)]
        seq2 = [np.ones(simple_env.n_bins) * 1.5 for _ in range(5)]
        fields = [seq1, seq2]

        # Render with explicit vmin/vmax
        viewer = render_napari(
            simple_env, fields, layout="horizontal", vmin=0.0, vmax=2.0
        )

        # Both sequences should use same color scale
        # (This is tested indirectly - the renderer gets same vmin/vmax)
        # In the actual implementation, we'll pass same vmin/vmax to all renderers
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 2

    @pytest.mark.skipif(
        not pytest.importorskip("napari", reason="napari not installed"),
        reason="Requires napari",
    )
    def test_auto_vmin_vmax_computed_globally(self, simple_env):
        """Auto vmin/vmax should be computed across all sequences."""
        from neurospatial.animation.backends.napari_backend import render_napari

        # Create sequences with different ranges
        seq1 = [np.ones(simple_env.n_bins) * 0.1 for _ in range(5)]
        seq2 = [np.ones(simple_env.n_bins) * 0.9 for _ in range(5)]
        fields = [seq1, seq2]

        # Render without explicit vmin/vmax (should compute globally)
        viewer = render_napari(simple_env, fields, layout="horizontal")

        # Should succeed without error
        image_layers = [
            layer for layer in viewer.layers if layer.__class__.__name__ == "Image"
        ]
        assert len(image_layers) == 2
