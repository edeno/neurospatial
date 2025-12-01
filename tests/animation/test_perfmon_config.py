"""Tests for scripts/perfmon_config.json.

This test module validates the napari perfmon configuration file:
- Required fields are present
- Traces critical performance functions (video, timeseries, rendering)
- JSON is valid and loadable
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Path to the perfmon config
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
PERFMON_CONFIG_PATH = SCRIPTS_DIR / "perfmon_config.json"


class TestPerfmonConfigExists:
    """Tests that the perfmon config file exists and is valid JSON."""

    def test_config_file_exists(self):
        """Perfmon config file should exist at scripts/perfmon_config.json."""
        assert PERFMON_CONFIG_PATH.exists(), (
            f"Expected perfmon config at {PERFMON_CONFIG_PATH}"
        )

    def test_config_is_valid_json(self):
        """Config file should be valid JSON."""
        with PERFMON_CONFIG_PATH.open() as f:
            config = json.load(f)
        assert isinstance(config, dict)


class TestPerfmonConfigRequiredFields:
    """Tests for required configuration fields."""

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        """Load the perfmon config."""
        with PERFMON_CONFIG_PATH.open() as f:
            return json.load(f)  # type: ignore[no-any-return]

    def test_has_trace_qt_events(self, config: dict[str, Any]):
        """Config should have trace_qt_events enabled."""
        assert "trace_qt_events" in config
        assert config["trace_qt_events"] is True

    def test_has_trace_file_on_start(self, config: dict[str, Any]):
        """Config should specify output trace file path."""
        assert "trace_file_on_start" in config
        assert isinstance(config["trace_file_on_start"], str)
        assert config["trace_file_on_start"].endswith(".json")

    def test_has_trace_callables(self, config: dict[str, Any]):
        """Config should specify callable lists to trace."""
        assert "trace_callables" in config
        assert isinstance(config["trace_callables"], list)
        assert len(config["trace_callables"]) > 0

    def test_has_callable_lists(self, config: dict[str, Any]):
        """Config should define callable_lists mapping."""
        assert "callable_lists" in config
        assert isinstance(config["callable_lists"], dict)


class TestPerfmonConfigCallables:
    """Tests for traced callables - must include video and timeseries functions."""

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        """Load the perfmon config."""
        with PERFMON_CONFIG_PATH.open() as f:
            return json.load(f)  # type: ignore[no-any-return]

    @pytest.fixture
    def all_traced_callables(self, config: dict[str, Any]) -> list[str]:
        """Get all callables from all callable_lists."""
        callables = []
        for list_name in config.get("trace_callables", []):
            if list_name in config.get("callable_lists", {}):
                callables.extend(config["callable_lists"][list_name])
        return callables

    def test_traces_video_frame_callback(self, all_traced_callables: list[str]):
        """Should trace the video frame callback function.

        This is critical for measuring video overlay performance.
        """
        video_callback_pattern = "_make_video_frame_callback"
        has_video_callback = any(
            video_callback_pattern in c for c in all_traced_callables
        )
        assert has_video_callback, (
            f"Config should trace video frame callback. "
            f"Expected pattern '{video_callback_pattern}' in traced callables: "
            f"{all_traced_callables}"
        )

    def test_traces_timeseries_dock(self, all_traced_callables: list[str]):
        """Should trace the timeseries dock rendering function.

        This is critical for measuring time series dock performance.
        """
        timeseries_pattern = "_add_timeseries_dock"
        has_timeseries = any(timeseries_pattern in c for c in all_traced_callables)
        assert has_timeseries, (
            f"Config should trace timeseries dock function. "
            f"Expected pattern '{timeseries_pattern}' in traced callables: "
            f"{all_traced_callables}"
        )

    def test_traces_field_rendering(self, all_traced_callables: list[str]):
        """Should trace field rendering for performance measurement."""
        # At least one of these should be traced
        rendering_patterns = [
            "field_to_rgb_for_napari",
            "LazyFieldRenderer",
            "ChunkedLazyFieldRenderer",
        ]
        has_rendering = any(
            pattern in c for c in all_traced_callables for pattern in rendering_patterns
        )
        assert has_rendering, (
            f"Config should trace field rendering functions. "
            f"Expected one of {rendering_patterns} in traced callables"
        )

    def test_traces_overlay_rendering(self, all_traced_callables: list[str]):
        """Should trace overlay rendering functions."""
        # At least one overlay should be traced
        overlay_patterns = [
            "_render_position_overlay",
            "_render_bodypart_overlay",
            "_render_head_direction_overlay",
            "_render_event_overlay",
        ]
        has_overlay = any(
            pattern in c for c in all_traced_callables for pattern in overlay_patterns
        )
        assert has_overlay, (
            f"Config should trace overlay rendering functions. "
            f"Expected one of {overlay_patterns} in traced callables"
        )


class TestPerfmonConfigConsistency:
    """Tests for configuration consistency."""

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        """Load the perfmon config."""
        with PERFMON_CONFIG_PATH.open() as f:
            return json.load(f)  # type: ignore[no-any-return]

    def test_trace_callables_reference_existing_lists(self, config: dict[str, Any]):
        """All names in trace_callables should exist in callable_lists."""
        trace_callables = config.get("trace_callables", [])
        callable_lists = config.get("callable_lists", {})

        for name in trace_callables:
            assert name in callable_lists, (
                f"trace_callables references '{name}' but it's not defined in "
                f"callable_lists. Available: {list(callable_lists.keys())}"
            )

    def test_callable_lists_are_non_empty(self, config: dict[str, Any]):
        """Each callable list should contain at least one callable."""
        callable_lists = config.get("callable_lists", {})

        for name, callables in callable_lists.items():
            assert isinstance(callables, list), (
                f"callable_lists['{name}'] should be a list, got {type(callables)}"
            )
            assert len(callables) > 0, f"callable_lists['{name}'] should not be empty"

    def test_callables_are_fully_qualified_names(self, config: dict[str, Any]):
        """Each callable should be a fully qualified Python path."""
        callable_lists = config.get("callable_lists", {})

        for name, callables in callable_lists.items():
            for callable_path in callables:
                assert isinstance(callable_path, str), (
                    f"Callable path should be string, got {type(callable_path)}"
                )
                # Should have at least one dot (module.function)
                assert "." in callable_path, (
                    f"Callable '{callable_path}' in list '{name}' should be a "
                    f"fully qualified name (e.g., 'module.function')"
                )
                # Should start with neurospatial for our own functions
                assert callable_path.startswith("neurospatial"), (
                    f"Callable '{callable_path}' should start with 'neurospatial'"
                )
