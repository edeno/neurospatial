"""Tests for HTML backend EventOverlay rendering.

Tests Milestone 4 tasks: HTML backend event support with instant mode only.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from neurospatial import Environment
from neurospatial.animation.backends.html_backend import render_html
from neurospatial.animation.overlays import EventData, OverlayData


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
    )


class TestHTMLEventOverlaySerialization:
    """Test event overlay data is serialized to JSON in HTML."""

    def test_event_overlay_serialized_to_json(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test event overlay data is present in HTML JavaScript."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()

        # Check that overlay data includes events
        assert "overlayData" in html_content
        assert "events" in html_content

        # Extract JSON from script tag
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        assert match is not None

        overlay_json = json.loads(match.group(1))
        assert "events" in overlay_json
        assert len(overlay_json["events"]) > 0

    def test_event_overlay_includes_positions(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test event overlay includes position data."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        event_overlay = overlay_json["events"][0]
        assert "spikes" in event_overlay or "event_positions" in str(event_overlay)

    def test_event_overlay_includes_frame_indices(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test event overlay includes frame indices for visibility."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        event_overlay = overlay_json["events"][0]
        # Should include frame indices for visibility logic
        assert "frame_indices" in str(event_overlay) or "frameIndices" in str(
            event_overlay
        )

    def test_event_overlay_includes_styling(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test event overlay includes color, size, marker."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        event_overlay = overlay_json["events"][0]
        # Should include styling info
        assert "colors" in event_overlay or "color" in str(event_overlay)
        assert "size" in event_overlay

    def test_multiple_event_types_serialized(
        self, simple_env, simple_fields, event_data_multiple_types, tmp_path
    ):
        """Test multiple event types are correctly serialized."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_multiple_types])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        # Should have event data with both event types
        assert "events" in overlay_json
        # Check that cell_001 and cell_002 are present in the serialization
        json_str = json.dumps(overlay_json)
        assert "cell_001" in json_str
        assert "cell_002" in json_str


class TestHTMLEventOverlayDecayWarning:
    """Test warning when decay_frames > 0 in HTML backend."""

    def test_decay_frames_emits_warning(
        self, simple_env, simple_fields, event_data_with_decay, tmp_path
    ):
        """Test that decay_frames > 0 emits a warning."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_with_decay])

        with pytest.warns(UserWarning, match="HTML backend only supports instant mode"):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )

    def test_decay_warning_suggests_instant_fallback(
        self, simple_env, simple_fields, event_data_with_decay, tmp_path
    ):
        """Test warning message mentions instant mode fallback."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_with_decay])

        with pytest.warns(UserWarning, match="instant"):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )

    def test_decay_warning_suggests_alternative_backends(
        self, simple_env, simple_fields, event_data_with_decay, tmp_path
    ):
        """Test warning suggests video or napari backend for decay support."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_with_decay])

        with pytest.warns(UserWarning, match="video.*napari"):
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )

    def test_instant_mode_no_warning(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test that instant mode (decay_frames=0) doesn't emit decay warning."""
        import warnings

        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="HTML backend only supports instant mode"
            )
            # Should not raise warning about decay
            render_html(
                simple_env,
                simple_fields,
                str(save_path),
                overlay_data=overlay_data,
            )


class TestHTMLEventOverlayRendering:
    """Test event rendering in HTML canvas."""

    def test_render_events_function_present(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test HTML includes JavaScript function to render events."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()

        # Check for event rendering logic in JavaScript
        assert "events" in html_content.lower()
        # Should have rendering function (renderOverlays or similar)
        assert "renderOverlays" in html_content

    def test_canvas_element_present(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test HTML includes canvas element for overlays."""
        save_path = tmp_path / "test.html"
        overlay_data = OverlayData(events=[event_data_instant])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()

        # Should have canvas for overlay rendering
        assert '<canvas id="overlay-canvas"' in html_content


class TestHTMLEventOverlayWithOtherOverlays:
    """Test event overlays work alongside position and region overlays."""

    def test_events_with_positions(
        self, simple_env, simple_fields, event_data_instant, tmp_path
    ):
        """Test events render alongside position overlays."""
        from neurospatial.animation.overlays import PositionData

        save_path = tmp_path / "test.html"

        pos_data = PositionData(
            data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
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

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        # Should have both positions and events
        assert "positions" in overlay_json
        assert len(overlay_json["positions"]) == 1
        assert "events" in overlay_json
        assert len(overlay_json["events"]) > 0

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

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        # Should have both regions and events
        assert "regions" in overlay_json
        assert len(overlay_json["regions"]) > 0
        assert "events" in overlay_json
        assert len(overlay_json["events"]) > 0


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
        )
        overlay_data = OverlayData(events=[event_data1, event_data2])

        path = render_html(
            simple_env,
            simple_fields,
            str(save_path),
            overlay_data=overlay_data,
        )

        html_content = path.read_text()
        match = re.search(r"const overlayData = ({.*?});", html_content, re.DOTALL)
        overlay_json = json.loads(match.group(1))

        # Should have both event overlays
        assert len(overlay_json["events"]) == 2


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

        html_content = path.read_text()

        # Check overlay data is still in HTML
        assert "overlayData" in html_content
        assert "events" in html_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
