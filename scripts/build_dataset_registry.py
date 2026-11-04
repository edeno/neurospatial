#!/usr/bin/env python
"""Fill a dataset registry's session list from a pinned DANDI version.

A registry, ``src/neurospatial/datasets/registry/<dataset>.json``, holds what
a person wrote about the dataset (paper, citation, apparatus, stated facts)
and one entry per session. This script writes the parts the archive states:
the pinned version, the license and the archive's citation, and each
session's asset path, id, size and ``dandi:sha2-256`` digest. A published
DANDI version never changes, so the digests stay valid.

What a person decided about a session (``verified``, ``default``,
``n_units``, ``verification``) is kept, matched by session id; a new session
starts unverified. Sessions from another archive (rows with a ``url``, such
as the tutorial file a dataset loads by default) are written by hand and
kept as they are. The registry must already exist with ``dataset`` and
``archive.dandiset`` set.

Session ids are the asset file's BIDS entities (``sub-10073_ses-17010302``
for ``sub-10073_ses-17010302_behavior+ecephys.nwb``).

Usage
-----
    uv run python scripts/build_dataset_registry.py sargolini_2006
    uv run python scripts/build_dataset_registry.py sargolini_2006 \
        --version 0.251111.2151

Without ``--version`` the latest published version is pinned (never
``draft``); a change of version is printed, since every session must then be
checked again.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "src" / "neurospatial" / "datasets" / "registry"
API = "https://api.dandiarchive.org/api"
#: Seconds to wait for each API response.
TIMEOUT = 60
#: Session fields a person sets; the script keeps them.
HAND_FIELDS = {"verified": False, "default": False, "n_units": None, "verification": ""}

Fetch = Callable[[str], Any]


def get_json(url: str) -> Any:
    """GET ``url`` and decode its JSON body."""
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return json.load(response)


def latest_published(dandiset: str, get: Fetch = get_json) -> str:
    """The newest published version of ``dandiset``, never ``draft``."""
    versions = get(f"{API}/dandisets/{dandiset}/versions/?page_size=1000")["results"]
    published = [v for v in versions if v["version"] != "draft"]
    if not published:
        msg = f"DANDI {dandiset} has no published version; publish one to pin it"
        raise ValueError(msg)
    return str(max(published, key=lambda v: v["created"])["version"])


def session_id(asset_path: str) -> str:
    """The BIDS entities of an asset's file name, ``sub-X_ses-Y``.

    ``sub-10073/sub-10073_ses-17010302_behavior+ecephys.nwb`` gives
    ``sub-10073_ses-17010302``: the modality suffix, which has no ``-``, is
    dropped.
    """
    stem = Path(asset_path).name.split(".")[0]
    return "_".join(part for part in stem.split("_") if "-" in part)


def list_assets(dandiset: str, version: str, get: Fetch = get_json) -> list[dict]:
    """Every asset of the version with its metadata, following pages."""
    url: str | None = (
        f"{API}/dandisets/{dandiset}/versions/{version}/assets/"
        "?page_size=200&metadata=true"
    )
    assets: list[dict] = []
    while url:
        page = get(url)
        assets.extend(page["results"])
        url = page["next"]
    return assets


def license_text(licenses: list[str]) -> str:
    """The archive's license list as it states it, ``spdx:`` prefix dropped."""
    names = [name.removeprefix("spdx:") for name in licenses]
    return ", ".join(names) or "not stated"


def build(registry: Mapping[str, Any], version: str, get: Fetch = get_json) -> dict:
    """The registry with the archive's facts and session list from ``version``.

    Parameters
    ----------
    registry : Mapping
        The registry as written, with ``archive.dandiset``.
    version : str
        The published DANDI version to pin.

    Returns
    -------
    dict
        ``registry`` with ``license``, ``archive.version``, ``archive.url``,
        ``archive.citation`` and ``sessions`` from the archive; each
        session keeps its hand-set fields, and the rows from other archives
        are kept whole.
    """
    dandiset = registry["archive"]["dandiset"]
    meta = get(f"{API}/dandisets/{dandiset}/versions/{version}/")
    rows = registry.get("sessions", [])
    elsewhere = [entry for entry in rows if "url" in entry]
    known = {entry["session"]: entry for entry in rows if "url" not in entry}
    sessions = list(elsewhere)
    for asset in list_assets(dandiset, version, get):
        session = session_id(asset["path"])
        old = known.get(session, {})
        sessions.append(
            {
                "session": session,
                "archive": f"DANDI {dandiset}",
                "asset_path": asset["path"],
                "asset_id": asset["asset_id"],
                "sha256": asset["metadata"]["digest"]["dandi:sha2-256"],
                "size_bytes": int(asset["size"]),
                **{key: old.get(key, value) for key, value in HAND_FIELDS.items()},
            }
        )
    names = [entry["session"] for entry in sessions]
    if len(set(names)) != len(names):
        msg = (
            f"DANDI {dandiset} {version}: two assets, or an asset and a session "
            "from another archive, give one session id"
        )
        raise ValueError(msg)
    if missing := sorted(set(known) - set(names)):
        msg = f"sessions {missing} are not in DANDI {dandiset} {version}"
        raise ValueError(msg)
    return {
        **registry,
        "license": license_text(meta.get("license", [])),
        "archive": {
            **registry["archive"],
            "version": version,
            "url": f"https://dandiarchive.org/dandiset/{dandiset}/{version}",
            "citation": meta["citation"],
        },
        "sessions": sorted(sessions, key=lambda entry: entry["session"]),
    }


def main() -> None:
    """Rewrite one registry from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="dataset id, e.g. sargolini_2006")
    parser.add_argument("--version", help="DANDI version to pin; default the latest")
    args = parser.parse_args()
    path = REGISTRY / f"{args.dataset}.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    dandiset = registry["archive"]["dandiset"]
    version = args.version or latest_published(dandiset)
    pinned = registry["archive"].get("version")
    if pinned and pinned != version:
        print(f"version changes from {pinned} to {version}: check the sessions again")
    built = build(registry, version)
    path.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: {len(built['sessions'])} sessions, {version}")


if __name__ == "__main__":
    main()
