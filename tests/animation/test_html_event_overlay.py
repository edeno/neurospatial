"""Tests for HTML backend EventOverlay rendering.

Note: As of the matplotlib-based rendering fix, event overlays are rendered
directly in matplotlib (baked into images) rather than via JavaScript canvas.
This fixes coordinate alignment issues and enables full decay mode support.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.backends.html_backend import render_html
from neurospatial.animation.overlays import EventData, OverlayData, PositionData


@pytest.fixture
def simple_env():
    """Create a simple 2D environment for testing."""
    positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    return Environment.from_samples(positions, bin_size=5.0)


@pytest.fixture
def simple_fields(simple_env):
    """Create simple fields for testing."""
    rng = np.random.default_rng(42)
    return [rng.random(simple_env.n_bins) for _ in range(10)]


@pytest.fixture
def event_data_instant():
    """Create EventData with instant mode (no decay)."""
    return EventData(
        event_positions={"spikes": np.array([[5.0, 5.0], [8.0, 8.0], [3.0, 7.0]])},
        event_frame_indices={"spikes": np.array([1, 4, 7])},
        colors={"spikes": "#ff0000"},
        markers={"spikes": "o"},
        size=8.0,
        decay_frames=0,
        border_color="white",
        border_width=0.5,
        opacity=0.7,
    )


@pytest.fixture
def event_data_with_decay():
    """Create EventData with decay enabled."""
    return EventData(
        event_positions={"events": np.array([[5.0, 5.0], [8.0, 8.0]])},
        event_frame_indices={"events": np.array([2, 5])},
        colors={"events": "#00ff00"},
        markers={"events": "s"},
        size=10.0,
        decay_frames=5,  # Decay enabled
        border_color="black",
        border_width=1.0,
        opacity=0.7,
    )


@pytest.fixture
def event_data_multiple_types():
    """Create EventData with multiple event types."""
    return EventData(
        event_positions={
            "cell_001": np.array([[2.0, 3.0], [5.0, 5.0]]),
            "cell_002": np.array([[7.0, 8.0]]),
        },
        event_frame_indices={
            "cell_001": np.array([1, 4]),
            "cell_002": np.array([6]),
        },
        colors={"cell_001": "#1f77b4", "cell_002": "#ff7f0e"},
        markers={"cell_001": "o", "cell_002": "s"},
        size=8.0,
        decay_frames=0,
        border_color="white",
        border_width=0.5,
        opacity=0.7,
    )


class TestHTMLEventOverlayBasics:
    """Test basic event overlay rendering in HTML."""

    def test_event_overlay_renders_successfully(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test event overlay renders without error."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()
        assert path.stat().st_size > 0

    def test_multiple_event_types_render(
        self, simple_env, simple_fields, event_data_multiple_types, tmp_path
    ):
        """Test multiple event types render correctly."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_multiple_types])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()


class TestHTMLEventOverlayDecaySupport:
    """Test decay mode support for event overlays.

    Note: Decay mode is now fully supported via matplotlib rendering.
    """

    def test_decay_mode_renders_successfully(
        self, simple_env, simple_fields, event_data_with_decay, tmp_path
    ):
        """Test that decay mode renders without error (now fully supported)."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_with_decay])

        # Should render successfully - decay is now supported
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()

    def test_instant_mode_renders_successfully(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test that instant mode renders correctly."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()


class TestHTMLEventOverlayWithOtherOverlays:
    """Test event overlays work alongside position and region overlays."""

    def test_events_with_positions(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test events render alongside position overlays."""
        save_path = tmp_path / "test.html"

        rng = np.random.default_rng(47)
        pos_data = PositionData(
            data=rng.random((10, 2)) * 10.0,  # 10 frames to match simple_fields
            color="blue",
            size=10.0,
            trail_length=5,
        )
        overlay_data = OverlayData(positions=[pos_data], events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()

    def test_events_with_regions(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test events render alongside region overlays."""
        simple_env.regions.add("goal", point=np.array([5.0, 5.0]))

        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
            show_regions=True,
        )

        assert path.exists()


class TestHTMLEventOverlayEdgeCases:
    """Test edge cases for HTML event overlay rendering."""

    def test_empty_events_no_crash(self, simple_env, simple_fields, tmp_path):
        """Test empty event data doesn't crash."""
        save_path = tmp_path / "test.html"

        event_data = EventData(
            event_positions={"empty": np.array([]).reshape(0, 2)},
            event_frame_indices={"empty": np.array([], dtype=np.int_)},
            colors={"empty": "red"},
            markers={"empty": "o"},
            size=8.0,
            decay_frames=0,
            border_color="white",
            border_width=0.5,
            opacity=0.7,
        )
        overlay_data = OverlayData(events=[event_data])

        # Should not crash
        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()

    def test_multiple_event_overlays(self, simple_env, simple_fields, tmp_path):
        """Test multiple EventData instances in overlay_data."""
        save_path = tmp_path / "test.html"

        event_data1 = EventData(
            event_positions={"animal1": np.array([[3.0, 3.0]])},
            event_frame_indices={"animal1": np.array([2])},
            colors={"animal1": "#ff0000"},
            markers={"animal1": "o"},
            size=8.0,
            decay_frames=0,
            border_color="white",
            border_width=0.5,
            opacity=0.7,
        )
        event_data2 = EventData(
            event_positions={"animal2": np.array([[7.0, 7.0]])},
            event_frame_indices={"animal2": np.array([5])},
            colors={"animal2": "#0000ff"},
            markers={"animal2": "s"},
            size=8.0,
            decay_frames=0,
            border_color="white",
            border_width=0.5,
            opacity=0.7,
        )
        overlay_data = OverlayData(events=[event_data1, event_data2])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        assert path.exists()


class TestHTMLEventOverlayNonEmbeddedMode:
    """Test event overlay support in non-embedded mode (disk-backed frames)."""

    def test_non_embedded_mode_with_events(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test non-embedded mode supports event overlays."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            embed=False,
            overlay_data=overlay_data,
            n_workers=1,
        )

        assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
