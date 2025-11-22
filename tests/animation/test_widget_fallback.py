"""Tests for PersistentFigureRenderer fallback behavior.

This module tests the fallback logging and debug behavior when the
set_array optimization cannot be used (e.g., when QuadMesh is None or
_field_to_mesh_array returns None for non-grid layouts).

Phase 4.1: Clarify Fallback in `PersistentFigureRenderer`

The implementation uses QuadMesh.set_array() for efficient updates on grid
layouts, and falls back to full re-render for non-grid layouts (hexagonal,
graph, triangular mesh).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from neurospatial import Environment


@pytest.fixture
def grid_env():
    """Create a grid-compatible environment (RegularGridLayout)."""
    positions = np.random.default_rng(42).random((100, 2)) * 50
    return Environment.from_samples(positions, bin_size=10.0, name="GridEnv")


@pytest.fixture
def hex_env():
    """Create a non-grid-compatible environment (HexagonalLayout)."""
    positions = np.random.default_rng(42).random((100, 2)) * 50
    return Environment.from_samples(
        positions,
        bin_size=10.0,
        layout="Hexagonal",
        name="HexEnv",
        infer_active_bins=True,
        bin_count_threshold=0,
    )


@pytest.fixture
def grid_fields(grid_env):
    """Create sample fields for grid environment."""
    rng = np.random.default_rng(42)
    return [rng.random(grid_env.n_bins) for _ in range(5)]


@pytest.fixture
def hex_fields(hex_env):
    """Create sample fields for hexagonal environment."""
    rng = np.random.default_rng(42)
    return [rng.random(hex_env.n_bins) for _ in range(5)]


# ============================================================================
# Test Fallback Logging
# ============================================================================


class TestFallbackLogging:
    """Tests for logging when fallback occurs."""

    def test_logs_debug_on_fallback_for_hex_layout(self, hex_env, hex_fields, caplog):
        """Fallback should log at DEBUG level for non-grid layouts."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            # First render initializes the figure
            renderer.render(hex_fields[0], frame_idx=0)

            # Clear log to only capture second render's messages
            caplog.clear()

            # Second render should trigger fallback and log (hex is non-grid)
            with caplog.at_level(logging.DEBUG, logger="neurospatial.animation"):
                renderer.render(hex_fields[1], frame_idx=1)

            # Should have logged a message about fallback
            assert any(
                "fallback" in record.message.lower()
                or "re-render" in record.message.lower()
                for record in caplog.records
            ), (
                f"Expected fallback log message, got: {[r.message for r in caplog.records]}"
            )
        finally:
            renderer.close()

    def test_no_fallback_log_for_grid_layout(self, grid_env, grid_fields, caplog):
        """Grid layouts should NOT log fallback (set_array optimization works)."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=grid_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            renderer.render(grid_fields[0], frame_idx=0)
            caplog.clear()

            with caplog.at_level(logging.DEBUG, logger="neurospatial.animation"):
                renderer.render(grid_fields[1], frame_idx=1)

            # Should NOT have any fallback messages (optimization works)
            fallback_messages = [
                r.message
                for r in caplog.records
                if "fallback" in r.message.lower() or "re-render" in r.message.lower()
            ]
            assert len(fallback_messages) == 0, (
                f"Grid layout should use optimization, not fallback: {fallback_messages}"
            )
        finally:
            renderer.close()

    def test_fallback_log_is_informative(self, hex_env, hex_fields, caplog):
        """Fallback log message should be informative for debugging."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            renderer.render(hex_fields[0], frame_idx=0)
            caplog.clear()

            with caplog.at_level(logging.DEBUG, logger="neurospatial.animation"):
                renderer.render(hex_fields[1], frame_idx=1)

            # Log should explain why fallback occurred
            log_text = " ".join(r.message for r in caplog.records)
            # Can be layout type, QuadMesh not found, or other descriptive reason
            assert (
                "quadmesh" in log_text.lower()
                or "layout" in log_text.lower()
                or "not found" in log_text.lower()
                or "not supported" in log_text.lower()
            ), f"Expected informative fallback message, got: {log_text}"
        finally:
            renderer.close()

    def test_first_render_does_not_log_fallback(self, grid_env, grid_fields, caplog):
        """First render should not log fallback (it's the initial setup)."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=grid_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            caplog.clear()

            with caplog.at_level(logging.DEBUG, logger="neurospatial.animation"):
                renderer.render(grid_fields[0], frame_idx=0)

            # First render should NOT log fallback
            fallback_messages = [
                r.message
                for r in caplog.records
                if "fallback" in r.message.lower() or "re-render" in r.message.lower()
            ]
            assert len(fallback_messages) == 0, (
                f"First render should not log fallback: {fallback_messages}"
            )
        finally:
            renderer.close()


# ============================================================================
# Test Debug Flag
# ============================================================================


class TestDebugRaiseOnFallback:
    """Tests for debug flag that raises instead of falling back."""

    def test_raise_on_fallback_flag_raises_for_hex(self, hex_env, hex_fields):
        """When raise_on_fallback=True, should raise for non-grid layouts."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            raise_on_fallback=True,
        )

        try:
            # First render always works
            renderer.render(hex_fields[0], frame_idx=0)

            # Second render should raise (non-grid layout needs fallback)
            with pytest.raises(RuntimeError, match=r"(?i)fallback|re-render"):
                renderer.render(hex_fields[1], frame_idx=1)
        finally:
            renderer.close()

    def test_raise_on_fallback_does_not_raise_for_grid(self, grid_env, grid_fields):
        """When raise_on_fallback=True, should NOT raise for grid layouts."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=grid_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            raise_on_fallback=True,
        )

        try:
            renderer.render(grid_fields[0], frame_idx=0)
            # Should NOT raise for grid layout (set_array optimization works)
            png_bytes = renderer.render(grid_fields[1], frame_idx=1)
            assert isinstance(png_bytes, bytes)
            assert len(png_bytes) > 0
        finally:
            renderer.close()

    def test_raise_on_fallback_default_is_false(self, hex_env, hex_fields):
        """By default, raise_on_fallback should be False (no exception)."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            # No raise_on_fallback parameter - should default to False
        )

        try:
            renderer.render(hex_fields[0], frame_idx=0)
            # Should NOT raise by default (fallback is logged, not raised)
            png_bytes = renderer.render(hex_fields[1], frame_idx=1)
            assert isinstance(png_bytes, bytes)
            assert len(png_bytes) > 0
        finally:
            renderer.close()

    def test_first_render_does_not_raise_even_with_flag(self, hex_env, hex_fields):
        """First render should not raise even with raise_on_fallback=True."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
            raise_on_fallback=True,
        )

        try:
            # First render should not raise
            png_bytes = renderer.render(hex_fields[0], frame_idx=0)
            assert isinstance(png_bytes, bytes)
            assert len(png_bytes) > 0
        finally:
            renderer.close()


# ============================================================================
# Test Fallback Still Produces Valid Output
# ============================================================================


class TestFallbackProducesValidOutput:
    """Tests that fallback path still produces valid PNG output."""

    def test_fallback_produces_valid_png(self, hex_env, hex_fields):
        """Fallback path should still produce valid PNG bytes."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            png_1 = renderer.render(hex_fields[0], frame_idx=0)
            png_2 = renderer.render(hex_fields[1], frame_idx=1)

            # Both should be valid PNG files
            assert png_1[:8] == b"\x89PNG\r\n\x1a\n", "First frame not valid PNG"
            assert png_2[:8] == b"\x89PNG\r\n\x1a\n", "Second frame not valid PNG"

            # Should have reasonable size
            assert len(png_1) > 1000
            assert len(png_2) > 1000
        finally:
            renderer.close()

    def test_fallback_different_fields_produce_different_output(
        self, hex_env, hex_fields
    ):
        """Different fields should produce different PNG output (not cached/stale)."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            # Render two different fields
            png_1 = renderer.render(hex_fields[0], frame_idx=0)
            png_2 = renderer.render(hex_fields[1], frame_idx=1)

            # They should be different (not stale/cached from previous)
            # Note: Can't guarantee different PNG sizes, but bytes should differ
            assert png_1 != png_2, (
                "Different fields should produce different PNG output"
            )
        finally:
            renderer.close()


# ============================================================================
# Test _field_to_image_data Returns None Correctly
# ============================================================================


class TestFieldToImageDataReturnsNone:
    """Tests for _field_to_image_data behavior with different layouts."""

    def test_returns_none_for_hexagonal_layout(self, hex_env, hex_fields):
        """_field_to_image_data should return None for hexagonal layouts."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=hex_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            result = renderer._field_to_image_data(hex_fields[0])
            assert result is None, "Hexagonal layout should return None"
        finally:
            renderer.close()

    def test_returns_array_for_grid_layout(self, grid_env, grid_fields):
        """_field_to_image_data should return array for grid layouts."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=grid_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            result = renderer._field_to_image_data(grid_fields[0])
            assert result is not None, "Grid layout should return array"
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2, "Should return 2D array for imshow"
        finally:
            renderer.close()


# ============================================================================
# Test No Image Artist Fallback
# ============================================================================


class TestNoMeshFallback:
    """Tests for fallback when _mesh is None."""

    def test_logs_when_mesh_missing(self, grid_env, grid_fields, caplog):
        """Should log when _mesh attribute is None (edge case)."""
        from neurospatial.animation.backends.widget_backend import (
            PersistentFigureRenderer,
        )

        renderer = PersistentFigureRenderer(
            env=grid_env,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=50,
        )

        try:
            # First render
            renderer.render(grid_fields[0], frame_idx=0)

            # Manually clear _mesh to simulate edge case
            renderer._mesh = None
            caplog.clear()

            with caplog.at_level(logging.DEBUG, logger="neurospatial.animation"):
                # Should fall back and log
                png = renderer.render(grid_fields[1], frame_idx=1)

            # Should still produce valid output
            assert png[:8] == b"\x89PNG\r\n\x1a\n"

            # Should have logged about fallback
            assert any(
                "fallback" in r.message.lower()
                or "re-render" in r.message.lower()
                or "mesh" in r.message.lower()
                or "quadmesh" in r.message.lower()
                for r in caplog.records
            ), f"Expected fallback log, got: {[r.message for r in caplog.records]}"
        finally:
            renderer.close()
