"""``docs/llms.txt`` is generated from the API snapshot and ``CLAUDE.md``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    path = ROOT / "scripts" / "generate_llms_txt.py"
    spec = importlib.util.spec_from_file_location("generate_llms_txt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llms_txt_in_sync() -> None:
    generator = _generator()
    committed = (ROOT / "docs" / "llms.txt").read_bytes().decode("utf-8")
    assert committed.replace("\r\n", "\n") == generator.render(), (
        "docs/llms.txt is out of date; regenerate it with "
        "`uv run python scripts/generate_llms_txt.py`."
    )


def test_llms_txt_lists_every_public_name() -> None:
    text = _generator().render()
    snapshot = (ROOT / "tests" / "data" / "public_api.txt").read_text(encoding="utf-8")
    api = text.split("## Public API", 1)[1]
    assert api.count("\n- `") == len(snapshot.splitlines())
    for name in ("decode_session", "compute_spatial_rate", "is_place_cell"):
        assert f"- `{name}`" in text
    assert "Missing is not zero" in text
    assert "## Canonical argument order" in text
