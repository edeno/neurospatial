"""The registries, catalog() and the registry builder."""

from __future__ import annotations

import importlib
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


@pytest.mark.parametrize("dataset", dataset_ids())
def test_each_dataset_has_its_loader(dataset):
    """A loader module with load_session, REQUIRES and WATCH_OUTS for each verified session."""
    module = importlib.import_module(f"neurospatial.datasets.{dataset}")
    assert callable(module.load_session)
    assert all(isinstance(name, str) for name in module.REQUIRES)
    verified = {entry.session for entry in sessions(dataset) if entry.verified}
    assert set(module.WATCH_OUTS) == verified
    assert all(
        isinstance(text, str) and text.strip()
        for texts in module.WATCH_OUTS.values()
        for text in texts
    )


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
        (
            lambda r: r["stated"]["sampling_rate"].update(basis="guessed"),
            "stated.sampling_rate basis 'guessed'",
        ),
        (
            lambda r: r["stated"]["position_span"].update(source=""),
            "stated.position_span gives no source",
        ),
        (lambda r: r["stated"]["sampling_rate"].pop("basis"), "no field 'basis'"),
        (
            lambda r: r["sessions"][0].update(
                members=[{"path": "a.mat", "sha256": "0" * 64, "size_bytes": 1}]
            ),
            "zip members need the zip's url",
        ),
        (
            lambda r: r["sessions"][0].update(
                asset_id=None,
                url="https://example.org/a.zip",
                members=[{"path": "a.mat", "sha256": "xyz", "size_bytes": 1}],
            ),
            "member 'a.mat': sha256 'xyz' is not 64 hex digits",
        ),
        (
            lambda r: r["sessions"][0].update(
                videos=[
                    {
                        "asset_path": "v.avi",
                        "asset_id": "a",
                        "sha256": "xyz",
                        "size_bytes": 1,
                    }
                ]
            ),
            "video 'v.avi': sha256 'xyz' is not 64 hex digits",
        ),
        (
            lambda r: r["sessions"][0].update(
                videos=[
                    {
                        "asset_path": "v.avi",
                        "asset_id": "a",
                        "sha256": "a" * 64,
                        "size_bytes": 1,
                        "led_fit": {"video_sha256": "b" * 64},
                    }
                ]
            ),
            "its led_fit was made on a video with SHA-256",
        ),
        (
            lambda r: r["sessions"][0].update(
                lfp_files=[{"path": "a.mat", "sha256": "xyz", "size_bytes": 1}]
            ),
            "LFP file 'a.mat': sha256 'xyz' is not 64 hex digits",
        ),
    ],
)
def test_malformed_registry_refused(change, message):
    registry = _registry("sargolini_2006")
    registry["sessions"].sort(key=lambda row: not row["default"])
    change(registry)
    with pytest.raises(ValueError, match=message):
        _read(registry, "sargolini_2006.json")


def test_lfp_files_are_read_at_the_sessions_own_location():
    """A row's LFP file takes the session's location unless it names another."""
    registry = _registry("forli_2025")
    row = registry["sessions"][0]
    row["lfp_files"] = [
        {
            "path": "s/LFP.mat",
            "sha256": "a" * 64,
            "size_bytes": 5,
            "member": "z/LFP.mat",
            "channels": ["c1"],
        },
        {
            "path": "other.mat",
            "sha256": "b" * 64,
            "size_bytes": 6,
            "url": "https://example.org/other.mat",
        },
    ]
    first, second = _read(registry, "forli_2025.json")[0].lfp_files
    assert first.url == row["url"]
    assert first.member == "z/LFP.mat"
    assert first.channels == ("c1",)
    assert second.url == "https://example.org/other.mat"
    assert second.channels == ()


def test_a_row_can_state_its_own_facts():
    """A row's ``stated`` entries replace the dataset's for that session only."""
    registry = _registry("forli_2025")
    own = {"value": 60.0, "basis": "inferred", "source": "tracked by other means"}
    registry["sessions"][0]["stated"] = {"sampling_rate": own}
    first, *rest = _read(registry, "forli_2025.json")
    assert first.stated.sampling_rate == 60.0
    assert first.stated.sampling_rate_basis == "inferred"
    assert first.stated.position_span == first.info.stated.position_span
    assert first.info.stated.sampling_rate != 60.0
    assert all(entry.stated is entry.info.stated for entry in rest)


def test_members_of_a_crcns_archive_download_the_archive():
    """Reaching members of a .tar.gz reads the archive from its start: its size counts."""
    from neurospatial.datasets._fetch import crcns_url

    registry = _registry("forli_2025")
    row = registry["sessions"][0]
    row.pop("asset_id", None)
    row["url"] = crcns_url("hc-3/top/s.tar.gz")
    row["size_bytes"] = 1000
    row["members"] = [
        {"path": "top/s/s.res.1", "sha256": "a" * 64, "size_bytes": 10},
        {
            "path": "t.zip",
            "sha256": "b" * 64,
            "size_bytes": 7,
            "url": "https://x/t.zip",
        },
    ]
    entry = _read(registry, "forli_2025.json")[0]
    assert entry.download_bytes == 1000 + 7


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
    # A session from another archive (a url row, written by hand) is kept whole.
    written["sessions"][-1] = {
        "session": "tutorial",
        "archive": "OSF abcde",
        "url": "https://osf.io/abcde/download",
        "license": "not stated",
    }
    built = builder.build(written, "0.2", get)
    assert built["sessions"][-1] == written["sessions"][-1]
    assert [entry["session"] for entry in built["sessions"]] == [
        "sub-1_ses-1",
        "sub-1_ses-2",
        "tutorial",
    ]
    # ... but not under a session id an asset also gives.
    written["sessions"][-1]["session"] = "sub-1_ses-1"
    with pytest.raises(ValueError, match="give one session id"):
        builder.build(written, "0.2", get)


def test_builder_resets_a_session_whose_file_changed(capsys):
    """A new SHA-256 clears the check by hand, keeps the default and says so."""
    builder = _builder()
    api = builder.API
    pages = {
        f"{api}/dandisets/000001/versions/0.3/": {
            "license": ["spdx:CC-BY-4.0"],
            "citation": "Someone (2025) Data. DANDI Archive.",
        },
        f"{api}/dandisets/000001/versions/0.3/assets/?page_size=200&metadata=true": {
            "results": [
                _asset("sub-1/sub-1_ses-1_ecephys.nwb", "c" * 64),
                _asset("sub-1/sub-1_ses-2_ecephys.nwb", "d" * 64),
            ],
            "next": None,
        },
    }
    checked = {"verified": True, "n_units": 3, "verification": "Checked by hand."}
    written = {
        "dataset": "x_2025",
        "archive": {"dandiset": "000001"},
        "sessions": [
            {"session": "sub-1_ses-1", "sha256": "a" * 64, "default": True, **checked},
            {"session": "sub-1_ses-2", "sha256": "d" * 64, "default": False, **checked},
        ],
    }
    changed, same = builder.build(written, "0.3", _api(pages))["sessions"]
    assert (changed["sha256"], changed["default"]) == ("c" * 64, True)
    assert (changed["verified"], changed["n_units"], changed["verification"]) == (
        False,
        None,
        "",
    )
    assert {key: same[key] for key in checked} == checked
    printed = capsys.readouterr().out
    assert "sub-1_ses-1: reset to unverified" in printed
    assert "sub-1_ses-2" not in printed


def test_builder_lists_one_variant_with_its_videos():
    """Only the named variant's NWB files are sessions; their folders' files are videos."""
    builder = _builder()
    api = builder.API
    stem = "sub-1/sub-1_ses-1-no-raw-data_behavior+image"
    pages = {
        f"{api}/dandisets/000002/versions/0.1/": {
            "license": ["spdx:CC-BY-4.0"],
            "citation": "Someone (2023) Data. DANDI Archive.",
        },
        f"{api}/dandisets/000002/versions/0.1/assets/?page_size=200&metadata=true": {
            "results": [
                _asset(f"{stem}.nwb", "a" * 64),
                _asset(f"{stem}/uuid_external_file_1.avi", "c" * 64),
                _asset(f"{stem}/uuid_external_file_0.avi", "b" * 64),
                _asset("sub-1/sub-1_ses-1_behavior+image.nwb", "d" * 64),
                _asset(
                    "sub-1/sub-1_ses-1_behavior+image/uuid_external_file_0.avi",
                    "e" * 64,
                ),
            ],
            "next": None,
        },
    }
    landmarks = {
        "px": [[0, 0], [1, 0], [0, 1]],
        "placed_by": "someone",
        "video_sha256": "b" * 64,
    }
    written = {
        "dataset": "x_2023",
        "archive": {"dandiset": "000002", "variant": "-no-raw-data"},
        "sessions": [
            {
                "session": "sub-1_ses-1",
                "default": True,
                "videos": [
                    {
                        "asset_path": f"{stem}/uuid_external_file_0.avi",
                        "landmarks": landmarks,
                    }
                ],
            }
        ],
    }
    built = builder.build(written, "0.1", _api(pages))
    (session,) = built["sessions"]
    assert (session["session"], session["sha256"], session["default"]) == (
        "sub-1_ses-1",
        "a" * 64,
        True,
    )
    first, second = session["videos"]
    assert first["asset_path"] == f"{stem}/uuid_external_file_0.avi"
    assert (first["sha256"], first["landmarks"]) == ("b" * 64, landmarks)
    assert (second["sha256"], "landmarks" in second) == ("c" * 64, False)
    # Landmarks made on another video are dropped when the video changes.
    written["sessions"][0]["videos"][0]["landmarks"] = {
        **landmarks,
        "video_sha256": "f" * 64,
    }
    (session,) = builder.build(written, "0.1", _api(pages))["sessions"]
    assert "landmarks" not in session["videos"][0]
    # A video with landmarks that the archive no longer lists is refused.
    written["sessions"][0]["videos"][0]["asset_path"] = f"{stem}/gone.avi"
    with pytest.raises(ValueError, match=r"gone\.avi"):
        builder.build(written, "0.1", _api(pages))


def test_builder_lists_a_sibling_file_as_lfp_and_leaves_out_sessions_not_offered():
    builder = _builder()
    registry = {
        "archive": {
            "dandiset": "000552",
            "lfp_variant": "-raw",
            "no_lfp": ["sub-3_ses-3"],
            "not_offered": ["sub-2_ses-2"],
        }
    }
    assets = [
        _asset("sub-1/sub-1_ses-1_behavior+ecephys.nwb", "a" * 64),
        _asset("sub-1/sub-1_ses-1-raw_ecephys.nwb", "b" * 64),
        _asset("sub-2/sub-2_ses-2_behavior+ecephys.nwb", "c" * 64),
        _asset("sub-3/sub-3_ses-3_behavior+ecephys.nwb", "d" * 64),
        _asset("sub-3/sub-3_ses-3-raw_ecephys.nwb", "e" * 64),
    ]
    api = builder.API
    get = _api(
        {
            f"{api}/dandisets/000552/versions/0.1/": {
                "license": ["spdx:CC-BY-4.0"],
                "citation": "A citation",
            },
            f"{api}/dandisets/000552/versions/0.1/assets/"
            "?page_size=200&metadata=true": {"results": assets, "next": None},
        }
    )
    built = builder.build(registry, "0.1", get)
    rows = {row["session"]: row for row in built["sessions"]}
    assert set(rows) == {"sub-1_ses-1", "sub-3_ses-3"}
    assert rows["sub-1_ses-1"]["lfp_files"] == [
        {
            "path": "sub-1_ses-1-raw_ecephys.nwb",
            "asset_id": "id-sub-1/sub-1_ses-1-raw_ecephys.nwb",
            "sha256": "b" * 64,
            "size_bytes": 1000,
        }
    ]
    assert "lfp_files" not in rows["sub-3_ses-3"]


def test_zip_builder_hashes_new_members_and_keeps_hand_fields(capsys):
    """Members listed at their size keep their hash; a changed one resets the check."""
    builder = _builder()
    zipped = {
        "url": "https://example.org/a.zip",
        "file_name": "a.zip",
        "sha256": "e" * 64,
        "size_bytes": 99,
    }
    checked = {"verified": True, "n_units": 3, "verification": "Checked by hand."}
    written = {
        "dataset": "x_2016",
        "archive": {"name": "Zenodo 1", "zip": zipped},
        "sessions": [
            {
                "session": "R1-day",
                "default": True,
                **checked,
                "members": [
                    {"path": "r1/day/t.1", "sha256": "a" * 64, "size_bytes": 10},
                    {"path": "r1/day/t.pos", "sha256": "b" * 64, "size_bytes": 5},
                ],
                "lfp_files": [],
            }
        ],
    }
    sizes = {"r1/day/t.1": 10, "r1/day/t.pos": 5, "r1/day/t.egf": 7, "README.txt": 1}
    hashes = {
        "r1/day/t.1": "f" * 64,
        "r1/day/t.pos": "b" * 64,
        "r1/day/t.egf": "c" * 64,
    }

    def members(names):
        assert names == sorted(sizes)
        return {"R1-day": (["r1/day/t.1", "r1/day/t.pos"], ["r1/day/t.egf"])}

    asked = []

    def sha256_of(name):
        asked.append(name)
        return hashes[name]

    (row,) = builder.build_zip(written, sizes, members, sha256_of)["sessions"]
    assert asked == ["r1/day/t.egf"]
    assert row["members"] == [
        {"path": "r1/day/t.1", "sha256": "a" * 64, "size_bytes": 10},
        {"path": "r1/day/t.pos", "sha256": "b" * 64, "size_bytes": 5},
    ]
    assert row["lfp_files"] == [
        {
            "path": "R1-day/t.egf",
            "sha256": "c" * 64,
            "size_bytes": 7,
            "member": "r1/day/t.egf",
        }
    ]
    assert (row["url"], row["sha256"], row["file_name"]) == (
        zipped["url"],
        zipped["sha256"],
        "a.zip",
    )
    # A member listed at its size keeps its hash, so the check by hand stays.
    assert {key: row[key] for key in checked} == checked
    assert "reset" not in capsys.readouterr().out
    sizes["r1/day/t.1"] = 11
    asked.clear()
    (changed,) = builder.build_zip(written, sizes, members, sha256_of)["sessions"]
    assert "r1/day/t.1" in asked
    assert changed["members"][0]["sha256"] == "f" * 64
    assert (changed["verified"], changed["n_units"], changed["default"]) == (
        False,
        None,
        True,
    )
    assert "R1-day: reset to unverified" in capsys.readouterr().out


def _zenodo_record(names: dict[str, bytes]) -> dict:
    import hashlib

    return {
        "doi": "10.5061/dryad.x",
        "files": [
            {
                "key": name,
                "size": len(data),
                "checksum": "md5:"
                + hashlib.md5(data, usedforsecurity=False).hexdigest(),
                "links": {
                    "self": f"https://zenodo.org/api/records/1/files/{name}/content"
                },
            }
            for name, data in names.items()
        ],
    }


def test_zenodo_builder_hashes_cached_files_checked_against_the_records_md5(tmp_path):
    import hashlib

    builder = _builder()
    data = {"a.mat": b"spikes", "b.mat": b"position", "c.mat": b"lfp"}
    (tmp_path / "ds" / "s1").mkdir(parents=True)
    (tmp_path / "ds" / "a.mat").write_bytes(data["a.mat"])
    (tmp_path / "ds" / "s1" / "b.mat").write_bytes(data["b.mat"])
    (tmp_path / "ds" / "c.mat").write_bytes(data["c.mat"])
    registry = {
        "dataset": "ds",
        "archive": {"zenodo": "1"},
        "sessions": [
            {
                "session": "s1",
                "verified": True,
                "members": [{"path": "a.mat"}, {"path": "b.mat"}],
                "lfp_files": [{"path": "c.mat", "channels": ["x"]}],
            }
        ],
    }
    get = _api({"https://zenodo.org/api/records/1": _zenodo_record(data)})
    built = builder.build_zenodo(registry, tmp_path, get)
    row = built["sessions"][0]
    sha = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    assert [m["sha256"] for m in row["members"]] == [sha["a.mat"], sha["b.mat"]]
    assert (
        row["sha256"]
        == hashlib.sha256(f"{sha['a.mat']}\n{sha['b.mat']}".encode()).hexdigest()
    )
    assert row["size_bytes"] == len(data["a.mat"]) + len(data["b.mat"])
    assert row["lfp_files"][0]["channels"] == ["x"]
    assert row["lfp_files"][0]["sha256"] == sha["c.mat"]
    assert row["verified"] is True
    assert built["archive"]["url"] == "https://doi.org/10.5061/dryad.x"

    (tmp_path / "ds" / "a.mat").write_bytes(b"changed")
    with pytest.raises(ValueError, match="the record states"):
        builder.build_zenodo(registry, tmp_path, get)
