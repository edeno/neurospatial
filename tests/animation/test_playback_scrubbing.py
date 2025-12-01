"""Tests for PlaybackController rapid scrubbing debounce.

This test module validates the debounce feature for rapid frame changes:
- Debounce timer prevents excessive viewer updates when scrubbing rapidly
- Pending frame coalescing (only latest frame applied)
- Immediate feedback maintained (seek feels responsive)
- Debounce can be disabled for testing/video export
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


class TestScrubDebounceInit:
    """Tests for debounce initialization."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_debounce_enabled_by_default(self, mock_viewer: MagicMock):
        """Debounce should be enabled by default."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
        )

        assert controller.scrub_debounce_ms == 16  # ~60 Hz max

    def test_debounce_configurable(self, mock_viewer: MagicMock):
        """Debounce interval should be configurable."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=32,  # ~30 Hz max
        )

        assert controller.scrub_debounce_ms == 32

    def test_debounce_disabled_with_zero(self, mock_viewer: MagicMock):
        """Debounce should be disabled when set to 0."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=0,  # Disabled
        )

        assert controller.scrub_debounce_ms == 0


class TestScrubDebounceValidation:
    """Tests for debounce parameter validation."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_negative_debounce_raises_error(self, mock_viewer: MagicMock):
        """Negative debounce values should raise ValueError."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        with pytest.raises(ValueError, match="scrub_debounce_ms"):
            PlaybackController(
                viewer=mock_viewer,
                n_frames=100,
                fps=30.0,
                scrub_debounce_ms=-10,
            )


class TestScrubDebounceCoalescing:
    """Tests for frame coalescing during rapid scrubbing."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_rapid_scrub_coalesces_to_latest_frame(self, mock_viewer: MagicMock):
        """Rapid go_to_frame calls should coalesce to the latest frame."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        # Simulate rapid scrubbing - multiple calls within debounce window
        controller.go_to_frame(10)
        controller.go_to_frame(20)
        controller.go_to_frame(30)
        controller.go_to_frame(40)
        controller.go_to_frame(50)

        # First call should go through immediately
        # Subsequent calls should be coalesced

        # Wait for debounce to settle
        time.sleep(0.025)  # 25ms > 16ms debounce

        # Flush any pending updates
        controller.flush_pending_frame()

        # Final frame should be 50
        assert controller.current_frame == 50

        # Viewer should NOT have been called 5 times (some coalesced)
        call_count = mock_viewer.dims.set_current_step.call_count
        assert call_count < 5, f"Expected coalescing, got {call_count} calls"

    def test_no_debounce_updates_every_frame(self, mock_viewer: MagicMock):
        """With debounce disabled, every go_to_frame updates immediately."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=0,  # Disabled
        )

        controller.go_to_frame(10)
        controller.go_to_frame(20)
        controller.go_to_frame(30)

        # Every call should update viewer
        assert mock_viewer.dims.set_current_step.call_count == 3
        assert controller.current_frame == 30


class TestScrubDebounceResponsiveness:
    """Tests for responsiveness during scrubbing."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_first_scrub_is_immediate(self, mock_viewer: MagicMock):
        """First frame change should apply immediately for responsiveness."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        controller.go_to_frame(50)

        # First call should update immediately
        mock_viewer.dims.set_current_step.assert_called_once_with(0, 50)
        assert controller.current_frame == 50

    def test_scrub_after_quiet_period_is_immediate(self, mock_viewer: MagicMock):
        """Frame change after quiet period should apply immediately."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        controller.go_to_frame(25)

        # Wait for debounce to fully settle
        time.sleep(0.025)
        controller.flush_pending_frame()

        mock_viewer.dims.set_current_step.reset_mock()

        # This call should be immediate (after quiet period)
        controller.go_to_frame(75)

        mock_viewer.dims.set_current_step.assert_called_with(0, 75)
        assert controller.current_frame == 75


class TestScrubDebounceCallbacks:
    """Tests for callback behavior with debounce."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_callbacks_called_for_coalesced_frame(self, mock_viewer: MagicMock):
        """Callbacks should be called when coalesced frame is applied."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        callback = MagicMock()
        controller.register_callback(callback)

        # Rapid scrubbing
        controller.go_to_frame(10)
        controller.go_to_frame(20)
        controller.go_to_frame(30)

        # Wait and flush
        time.sleep(0.025)
        controller.flush_pending_frame()

        # Callback should have been called for final coalesced frame
        # May have been called for first immediate frame too
        assert callback.call_count >= 1

        # Last call should be for frame 30
        last_call_frame = callback.call_args_list[-1][0][0]
        assert last_call_frame == 30


class TestScrubDebounceMetrics:
    """Tests for metrics with debounce."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_frames_rendered_counts_coalesced(self, mock_viewer: MagicMock):
        """frames_rendered should count actual frame updates, not requests."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        # Rapid scrubbing (5 requests)
        controller.go_to_frame(10)
        controller.go_to_frame(20)
        controller.go_to_frame(30)
        controller.go_to_frame(40)
        controller.go_to_frame(50)

        # Wait and flush
        time.sleep(0.025)
        controller.flush_pending_frame()

        # Fewer renders than requests
        assert controller.frames_rendered < 5


class TestFlushPendingFrame:
    """Tests for flush_pending_frame method."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_flush_applies_pending_frame(self, mock_viewer: MagicMock):
        """flush_pending_frame should apply any pending frame immediately."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        # First call is immediate
        controller.go_to_frame(10)
        initial_count = mock_viewer.dims.set_current_step.call_count

        # Rapid calls queue a pending frame
        controller.go_to_frame(20)
        controller.go_to_frame(30)

        # Flush applies pending immediately
        controller.flush_pending_frame()

        assert controller.current_frame == 30
        # Should have applied pending
        assert mock_viewer.dims.set_current_step.call_count > initial_count

    def test_flush_noop_when_no_pending(self, mock_viewer: MagicMock):
        """flush_pending_frame should be a no-op when no pending frame."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        controller.go_to_frame(50)
        count_before = mock_viewer.dims.set_current_step.call_count

        # Wait for debounce
        time.sleep(0.025)
        controller.flush_pending_frame()
        controller.flush_pending_frame()  # Second flush should be no-op

        # No additional calls after double flush
        count_after = mock_viewer.dims.set_current_step.call_count
        # At most one additional call from first flush
        assert count_after <= count_before + 1


class TestHasPendingFrame:
    """Tests for has_pending_frame property."""

    @pytest.fixture
    def mock_viewer(self) -> MagicMock:
        """Create a mock napari viewer."""
        viewer = MagicMock()
        viewer.dims.set_current_step = MagicMock()
        viewer.dims.current_step = [0]
        return viewer

    def test_has_pending_false_initially(self, mock_viewer: MagicMock):
        """has_pending_frame should be False initially."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        assert controller.has_pending_frame is False

    def test_has_pending_true_after_rapid_scrub(self, mock_viewer: MagicMock):
        """has_pending_frame should be True when frame is pending."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        # First call is immediate, sets baseline
        controller.go_to_frame(10)

        # Second call within debounce window creates pending
        controller.go_to_frame(20)

        assert controller.has_pending_frame is True

    def test_has_pending_false_after_flush(self, mock_viewer: MagicMock):
        """has_pending_frame should be False after flush."""
        from neurospatial.animation.backends.napari_backend import PlaybackController

        controller = PlaybackController(
            viewer=mock_viewer,
            n_frames=100,
            fps=30.0,
            scrub_debounce_ms=16,
        )

        controller.go_to_frame(10)
        controller.go_to_frame(20)

        controller.flush_pending_frame()

        assert controller.has_pending_frame is False
