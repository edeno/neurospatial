"""The registries, catalog() and the registry builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from neurospatial import InputError
from neurospatial.datasets import catalog
from neurospatial.datasets._catalog import (
    CATALOG_COLUMNS,
    REGISTRY,
    _read,
    dataset_ids,
    sessions,
)

ROOT = Path(__file__).resolve().parents[2]


def _registry(dataset: str) -> dict:
    return json.loads((REGISTRY / f"{dataset}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("dataset", dataset_ids())
def test_registry_complete(dataset):
    """Every session has a checksum, a size and a license; one is the default."""
    registry = _registry(dataset)
    assert registry["license"].strip()
    for row in registry["sessions"]:
        assert len(row["sha256"]) == 64, row["session"]
        int(row["sha256"], 16)
        assert row["size_bytes"] > 0, row["session"]
    assert sum(row["default"] for row in registry["sessions"]) == 1
    entries = sessions(dataset)
    assert len(entries) == len(registry["sessions"])
    for entry in entries:
        if entry.verified:
            assert entry.verification.strip()
            assert entry.n_units is not None


def test_catalog_lists_every_session():
    table = catalog()
    assert tuple(table.columns) == CATALOG_COLUMNS
    assert len(table) == sum(len(sessions(d)) for d in dataset_ids())
    assert not table.duplicated(["dataset", "session"]).any()
    assert (table.groupby("dataset")["default"].sum() == 1).all()
    sargolini = table[table["dataset"] == "sargolini_2006"].set_index("session")
    assert len(sargolini) == 118
    default = sargolini.loc["sub-10073_ses-17010302"]
    assert default["default"]
    assert default["verified"]
    assert default["n_units"] == 8
    assert default["license"] == "CC-BY-4.0"
    assert default["archive"] == "DANDI 000582"
    assert sargolini["verified"].sum() == 2
    assert sargolini["n_units"].isna().sum() == 116


def test_unknown_dataset_lists_the_known_ones():
    with pytest.raises(InputError, match=r"(?s)No dataset 'nope'.*sargolini_2006"):
        sessions("nope")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["sessions"][1].update(default=True), "2 default sessions"),
        (lambda r: r["sessions"][0].update(sha256="xyz"), "not 64 hex digits"),
        (lambda r: r["sessions"].append(dict(r["sessions"][2])), "repeats a session"),
        (lambda r: r["sessions"][0].pop("asset_id"), "either asset_id or url"),
        (lambda r: r["sessions"][2].update(verified=True), "give no verification"),
        (lambda r: r.pop("license"), "no field 'license'"),
    ],
)
def test_malformed_registry_refused(change, message):
    registry = _registry("sargolini_2006")
    registry["sessions"].sort(key=lambda row: not row["default"])
    change(registry)
    with pytest.raises(ValueError, match=message):
        _read(registry, "sargolini_2006.json")


def _builder():
    path = ROOT / "scripts" / "build_dataset_registry.py"
    spec = importlib.util.spec_from_file_location("build_dataset_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _api(pages: dict[str, object]):
    """A fake DANDI API: each URL's JSON, refusing any URL not listed."""

    def get(url: str) -> object:
        return pages[url]

    return get


def _asset(path: str, digest: str) -> dict:
    return {
        "asset_id": f"id-{path}",
        "path": path,
        "size": 1000,
        "metadata": {"digest": {"dandi:sha2-256": digest}},
    }


def test_builder_pins_a_published_version_and_keeps_hand_fields():
    builder = _builder()
    api = builder.API
    pages = {
        f"{api}/dandisets/000001/versions/?page_size=1000": {
            "results": [
                {"version": "draft", "created": "2026-01-01"},
                {"version": "0.1", "created": "2024-01-01"},
                {"version": "0.2", "created": "2025-01-01"},
            ]
        },
        f"{api}/dandisets/000001/versions/0.2/": {
            "license": ["spdx:CC-BY-4.0"],
            "citation": "Someone (2025) Data. DANDI Archive.",
        },
        f"{api}/dandisets/000001/versions/0.2/assets/?page_size=200&metadata=true": {
            "results": [_asset("sub-1/sub-1_ses-2_behavior+ecephys.nwb", "a" * 64)],
            "next": "page-2",
        },
        "page-2": {
            "results": [_asset("sub-1/sub-1_ses-1_ecephys.nwb", "b" * 64)],
            "next": None,
        },
    }
    get = _api(pages)
    assert builder.latest_published("000001", get) == "0.2"
    written = {
        "dataset": "x_2025",
        "archive": {"dandiset": "000001"},
        "sessions": [
            {"session": "sub-1_ses-2", "verified": True, "default": True, "n_units": 3}
        ],
    }
    built = builder.build(written, "0.2", get)
    assert built["license"] == "CC-BY-4.0"
    assert built["archive"]["version"] == "0.2"
    assert built["archive"]["citation"] == "Someone (2025) Data. DANDI Archive."
    first, second = built["sessions"]
    assert (first["session"], first["sha256"], first["verified"]) == (
        "sub-1_ses-1",
        "b" * 64,
        False,
    )
    assert second["session"] == "sub-1_ses-2"
    assert second["asset_id"] == "id-sub-1/sub-1_ses-2_behavior+ecephys.nwb"
    assert (second["verified"], second["default"], second["n_units"]) == (True, True, 3)
    # A session the archive no longer lists is refused, not dropped.
    written["sessions"].append({"session": "sub-9_ses-9"})
    with pytest.raises(ValueError, match="sub-9_ses-9"):
        builder.build(written, "0.2", get)
