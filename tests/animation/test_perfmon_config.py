"""Tests for scripts/perfmon_config.json, the napari perfmon configuration.

napari silently skips a traced callable it cannot import, so a renamed or
deleted function would leave the trace without the frames it should time.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

PERFMON_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "scripts" / "perfmon_config.json"
)


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    with PERFMON_CONFIG_PATH.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def test_config_names_a_trace_file_and_existing_callable_lists(config):
    assert config["trace_qt_events"] is True
    assert config["trace_file_on_start"].endswith(".json")
    assert config["trace_callables"]
    for name in config["trace_callables"]:
        assert config["callable_lists"][name]


def _resolve(dotted: str) -> Any:
    """Import the longest module prefix of ``dotted`` and walk the rest."""
    parts = dotted.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split:]:
            target = getattr(target, attribute)
        return target
    raise ModuleNotFoundError(dotted)


def test_every_traced_callable_exists(config):
    traced = [
        dotted
        for name in config["trace_callables"]
        for dotted in config["callable_lists"][name]
    ]
    for dotted in traced:
        assert callable(_resolve(dotted)), dotted
    # The trace covers preparing and applying a frame, the playback cost.
    assert any(dotted.endswith("NapariScene._prepare") for dotted in traced)
    assert any(dotted.endswith("NapariScene._apply") for dotted in traced)
