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
starts unverified. When a session's file has a new SHA-256, its check by hand
was made on another file: it is reset to unverified (``verified``,
``n_units`` and ``verification`` cleared, ``default`` kept, since it names the
session, not the file), and the reset is printed. Sessions from another archive (rows with a ``url``, such
as the tutorial file a dataset loads by default) are written by hand and
kept as they are. The registry must already exist with ``dataset`` and
``archive.dandiset`` set.

Session ids are the asset file's BIDS entities (``sub-10073_ses-17010302``
for ``sub-10073_ses-17010302_behavior+ecephys.nwb``). Only NWB assets are
sessions. Where a dandiset publishes each session in several variants (DANDI
000568: with and without the raw voltage), ``archive.variant`` names the one
to list (``"-no-raw-data"``); only NWB files whose session entity ends with
it are sessions, and it is dropped from the session id. Where a dandiset
publishes a session's LFP in a file of its own (DANDI 000552:
``..._ses-X-raw_ecephys.nwb`` beside ``..._ses-X_behavior+ecephys.nwb``),
``archive.lfp_variant`` names that file's suffix (``"-raw"``): it is not a
session, and it is listed under the session's ``lfp_files`` with its path, id,
size and digest, except for the sessions in ``archive.no_lfp`` (such a file with
only raw voltage). ``archive.not_offered`` lists the session ids of NWB assets the
loader cannot read as sessions (with ``archive.not_offered_why``). Other assets
in a folder named after an NWB file's stem are the external files its
ImageSeries name (videos); each is listed under the session's ``videos``
with its path, id, size and digest, keeping the ``landmarks`` a person
placed on it and the ``led_fit`` that ``fit_video_led_calibration.py``
wrote.

Usage
-----
    uv run python scripts/build_dataset_registry.py sargolini_2006
    uv run python scripts/build_dataset_registry.py sargolini_2006 \
        --version 0.251111.2151

Without ``--version`` the latest published version is pinned (never
``draft``); a change of version is printed, since every session must then be
checked again.

A dataset read from members of one remote zip (Zenodo 5566548 for
``olafsdottir_2016``) gives ``archive.zip`` (its ``url``, ``sha256`` and
``size_bytes``) instead of ``archive.dandiset``. Its loader's
``session_members(names)`` says which members each session reads and which hold
its LFP. The script lists the zip's members by byte range and hashes each (a
member the registry already lists with the same path and size keeps its
SHA-256; ``--known`` gives hashes computed elsewhere, as ``{"members": {path:
{"sha256", "size_bytes"}}}``), then writes each session's ``members`` and
``lfp_files``, keeping the hand-set fields as for DANDI.

    uv run python scripts/build_dataset_registry.py olafsdottir_2016 \
        --known hashes.json

A Zenodo registry (``archive.zenodo`` set to the record id) is filled from
the record instead. Which files make up a session is a person's decision,
so each session row names its files by their keys in the record (``members``
and ``lfp_files``, each with ``path``); the script writes each file's URL and
size from the record and its SHA-256. Zenodo publishes only MD5, so each
file must already be in the cache (``<cache>/<dataset>/<name>`` or
``<cache>/<dataset>/<session>/<name>``): the script checks its MD5 against
the record's and hashes it. A session's ``sha256`` and ``size_bytes`` are
then those of its members (the SHA-256 of their checksums in order, one per
line, and their total size), and its ``url`` is the record's.

    uv run python scripts/build_dataset_registry.py tirole_2022
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import struct
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
#: Hand fields that describe the file's contents; a new SHA-256 resets them.
FILE_HAND_FIELDS = ("verified", "n_units", "verification")
#: Video fields a person sets; the script keeps them where set.
VIDEO_HAND_FIELDS = ("landmarks", "led_fit")

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
    variant = registry["archive"].get("variant", "")
    lfp_variant = registry["archive"].get("lfp_variant")
    not_offered = set(registry["archive"].get("not_offered", ()))
    no_lfp = set(registry["archive"].get("no_lfp", ()))
    meta = get(f"{API}/dandisets/{dandiset}/versions/{version}/")
    rows = registry.get("sessions", [])
    elsewhere = [entry for entry in rows if "url" in entry]
    known = {entry["session"]: entry for entry in rows if "url" not in entry}
    sessions = list(elsewhere)
    assets = list_assets(dandiset, version, get)
    lfp = {
        session_id(asset["path"]).removesuffix(lfp_variant): asset
        for asset in assets
        if lfp_variant and session_id(asset["path"]).endswith(lfp_variant)
    }
    nwb = [
        asset
        for asset in assets
        if asset["path"].endswith(".nwb")
        and session_id(asset["path"]).endswith(variant)
        and session_id(asset["path"]) not in not_offered
        and not (lfp_variant and session_id(asset["path"]).endswith(lfp_variant))
    ]
    for asset in nwb:
        session = session_id(asset["path"]).removesuffix(variant)
        old = known.get(session, {})
        located = _located(asset)
        hand = {key: old.get(key, value) for key, value in HAND_FIELDS.items()}
        if old.get("sha256", located["sha256"]) != located["sha256"]:
            hand.update({key: HAND_FIELDS[key] for key in FILE_HAND_FIELDS})
            print(
                f"{session}: reset to unverified; its file's SHA-256 changed "
                f"from {old['sha256']} to {located['sha256']}, so check it again"
            )
        entry = {
            "session": session,
            "archive": f"DANDI {dandiset}",
            **located,
            **hand,
        }
        if videos := _videos(asset["path"], assets, old.get("videos", [])):
            entry["videos"] = videos
        if session in lfp and session not in no_lfp:
            entry["lfp_files"] = [_lfp_file(lfp[session])]
        sessions.append(entry)
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


def _located(asset: Mapping[str, Any]) -> dict[str, Any]:
    """An asset's path, id, digest and size, as a registry row gives them."""
    return {
        "asset_path": asset["path"],
        "asset_id": asset["asset_id"],
        "sha256": asset["metadata"]["digest"]["dandi:sha2-256"],
        "size_bytes": int(asset["size"]),
    }


def _lfp_file(asset: Mapping[str, Any]) -> dict[str, Any]:
    """A session's LFP file as its ``lfp_files`` row: cached under its file name."""
    return {
        "path": Path(asset["path"]).name,
        "asset_id": asset["asset_id"],
        "sha256": asset["metadata"]["digest"]["dandi:sha2-256"],
        "size_bytes": int(asset["size"]),
    }


def _videos(
    nwb_path: str, assets: list[dict], written: list[dict]
) -> list[dict[str, Any]]:
    """The assets in the folder named after ``nwb_path``'s stem, sorted by path.

    Each keeps the hand-set fields (``VIDEO_HAND_FIELDS``) of the row
    ``written`` gives for the same path, where they were made on a video of
    the same SHA-256 (their ``video_sha256``); others are dropped, since
    they calibrate another file. A written video the archive no longer
    lists is refused, not dropped.
    """
    folder = nwb_path.removesuffix(".nwb") + "/"
    hand = {row["asset_path"]: row for row in written}
    videos = []
    for asset in sorted(assets, key=lambda a: a["path"]):
        if not asset["path"].startswith(folder):
            continue
        row = _located(asset)
        old = hand.get(asset["path"], {})
        for key in VIDEO_HAND_FIELDS:
            if key not in old:
                continue
            if old[key].get("video_sha256") == row["sha256"]:
                row[key] = old[key]
            else:
                print(f"{asset['path']}: dropped its {key}, made on another video")
        videos.append(row)
    if missing := sorted(set(hand) - {row["asset_path"] for row in videos}):
        msg = f"videos {missing} are not in the folder of {nwb_path}"
        raise ValueError(msg)
    return videos


#: Members hashed at once, each by its own HTTP range requests.
HASH_WORKERS = 6
#: Bytes read per range request while hashing a member.
HASH_BLOCK = 16 << 20

ZipSessions = Callable[[list[str]], dict[str, tuple[list[str], list[str]]]]


def zip_listing(url: str) -> dict[str, tuple[int, int, int]]:
    """Each member of the remote zip at ``url``: its size, header offset and method."""
    import zipfile

    import fsspec

    with (
        fsspec.open(url, "rb", block_size=4 << 20) as remote,
        zipfile.ZipFile(remote) as archive,
    ):
        return {
            info.filename: (info.file_size, info.header_offset, info.compress_type)
            for info in archive.infolist()
        }


def _range(url: str, start: int, stop: int) -> bytes:
    """Bytes ``start`` to ``stop`` (exclusive) of ``url``."""
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{stop - 1}", "User-Agent": "neurospatial"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data: bytes = response.read()
    if len(data) != stop - start:
        msg = f"{url}: asked for {stop - start} bytes from {start}, got {len(data)}"
        raise ValueError(msg)
    return data


def member_sha256(url: str, name: str, size: int, offset: int, method: int) -> str:
    """The SHA-256 of one stored (uncompressed) member of the remote zip at ``url``."""
    if method != 0:
        msg = f"{name}: a compressed member cannot be hashed by range; extract it"
        raise ValueError(msg)
    header = _range(url, offset, offset + 30)
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    start = offset + 30 + name_length + extra_length
    digest = hashlib.sha256()
    for at in range(start, start + size, HASH_BLOCK):
        digest.update(_range(url, at, min(at + HASH_BLOCK, start + size)))
    return digest.hexdigest()


def build_zip(
    registry: Mapping[str, Any],
    sizes: Mapping[str, int],
    session_members: ZipSessions,
    sha256_of: Callable[[str], str],
) -> dict:
    """The registry's sessions, each read from members of the archive's zip.

    Parameters
    ----------
    registry : Mapping
        The registry as written, with ``archive.zip``.
    sizes : Mapping of str to int
        Each member of the zip and its size.
    session_members : callable
        The loader's ``session_members``: each session's load and LFP members.
    sha256_of : callable
        The SHA-256 of a member the registry does not already list at its size.

    Returns
    -------
    dict
        ``registry`` with ``sessions`` rebuilt. Each keeps its hand-set fields,
        reset as for DANDI when one of its members changed.
    """
    archive = registry["archive"]
    zipped = archive["zip"]
    rows = {entry["session"]: entry for entry in registry.get("sessions", [])}
    listed = {
        member.get("member", member["path"]): member
        for entry in rows.values()
        for member in [*entry.get("members", ()), *entry.get("lfp_files", ())]
    }
    days = session_members(sorted(sizes))
    wanted = sorted(
        name
        for load, lfp in days.values()
        for name in [*load, *lfp]
        if not (name in listed and listed[name].get("size_bytes") == sizes[name])
    )
    with concurrent.futures.ThreadPoolExecutor(HASH_WORKERS) as pool:
        hashed = dict(zip(wanted, pool.map(sha256_of, wanted), strict=True))

    def member(name: str) -> dict[str, Any]:
        sha256 = hashed[name] if name in hashed else listed[name]["sha256"]
        return {"path": name, "sha256": sha256, "size_bytes": sizes[name]}

    sessions = []
    for session, (load, lfp) in days.items():
        old = rows.get(session, {})
        hand = {key: old.get(key, value) for key, value in HAND_FIELDS.items()}
        members = [member(name) for name in load]
        before = {m["path"]: m["sha256"] for m in old.get("members", ())}
        if before and before != {m["path"]: m["sha256"] for m in members}:
            hand.update({key: HAND_FIELDS[key] for key in FILE_HAND_FIELDS})
            print(f"{session}: reset to unverified; its members changed")
        lfp_files = [
            {**member(name), "path": f"{session}/{Path(name).name}", "member": name}
            for name in lfp
        ]
        sessions.append(
            {
                "session": session,
                "archive": archive["name"],
                "url": zipped["url"],
                "file_name": zipped["file_name"],
                "sha256": zipped["sha256"],
                "size_bytes": zipped["size_bytes"],
                **hand,
                "members": members,
                "lfp_files": lfp_files,
            }
        )
    if missing := sorted(set(rows) - set(days)):
        msg = f"sessions {missing} are not in the zip"
        raise ValueError(msg)
    return {**registry, "sessions": sessions}


def file_digests(path: Path) -> tuple[str, str]:
    """The MD5 and SHA-256 of ``path``, read once."""
    md5, sha256 = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def build_zenodo(
    registry: Mapping[str, Any],
    cache: Path,
    get: Fetch = get_json,
    digests: Callable[[Path], tuple[str, str]] = file_digests,
) -> dict:
    """The registry with each session's files located in a Zenodo record.

    Parameters
    ----------
    registry : Mapping
        The registry as written, with ``archive.zenodo`` and session rows
        naming their ``members`` and ``lfp_files`` by ``path``, the file's key
        in the record.
    cache : Path
        The cache root holding the files, ``<cache>/<dataset>/``.

    Returns
    -------
    dict
        ``registry`` with ``archive.url`` and each file's ``url``, ``size_bytes``
        and ``sha256``, and each session's ``url``, ``sha256`` and
        ``size_bytes``.

    Raises
    ------
    ValueError
        If a file is not in the record, not in the cache, or its MD5 differs
        from the record's.
    """
    record_id = str(registry["archive"]["zenodo"])
    record = get(f"https://zenodo.org/api/records/{record_id}")
    files = {item["key"]: item for item in record["files"]}
    folder = cache / registry["dataset"]
    hashed: dict[str, tuple[str, int]] = {}

    def locate(name: str, session: str) -> dict[str, Any]:
        if name not in files:
            msg = f"{name} is not a file of Zenodo record {record_id}"
            raise ValueError(msg)
        item = files[name]
        if name not in hashed:
            found = [p for p in (folder / name, folder / session / name) if p.exists()]
            if not found:
                msg = (
                    f"{name} is not in {folder}: Zenodo publishes only MD5, so "
                    f"download it there first ({item['links']['self']})"
                )
                raise ValueError(msg)
            md5, sha256 = digests(found[0])
            published = str(item["checksum"]).removeprefix("md5:")
            if md5 != published:
                msg = f"{found[0]} has MD5 {md5}, the record states {published}"
                raise ValueError(msg)
            hashed[name] = (sha256, int(item["size"]))
        sha256, size = hashed[name]
        return {"url": item["links"]["self"], "sha256": sha256, "size_bytes": size}

    sessions = []
    for row in registry["sessions"]:
        members = [
            {**member, **locate(member["path"], row["session"])}
            for member in row.get("members", ())
        ]
        lfp_files = [
            {**item, **locate(item["path"], row["session"])}
            for item in row.get("lfp_files", ())
        ]
        checksums = "\n".join(member["sha256"] for member in members)
        sessions.append(
            {
                **row,
                "url": f"https://zenodo.org/records/{record_id}",
                "sha256": hashlib.sha256(checksums.encode()).hexdigest(),
                "size_bytes": sum(member["size_bytes"] for member in members),
                "members": members,
                **({"lfp_files": lfp_files} if lfp_files else {}),
            }
        )
    return {
        **registry,
        "archive": {
            **registry["archive"],
            "url": f"https://doi.org/{record['doi']}",
        },
        "sessions": sessions,
    }


def main() -> None:
    """Rewrite one registry from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="dataset id, e.g. sargolini_2006")
    parser.add_argument("--version", help="DANDI version to pin; default the latest")
    parser.add_argument("--known", help="JSON of member hashes computed elsewhere")
    args = parser.parse_args()
    path = REGISTRY / f"{args.dataset}.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    if "zenodo" in registry["archive"]:
        from neurospatial.datasets._fetch import cache_root

        built = build_zenodo(registry, cache_root())
        path.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {len(built['sessions'])} sessions")
        return
    if "zip" in registry["archive"]:
        url = registry["archive"]["zip"]["url"]
        known = (
            json.loads(Path(args.known).read_text(encoding="utf-8"))["members"]
            if args.known
            else {}
        )
        loader = importlib.import_module(f"neurospatial.datasets.{args.dataset}")
        listing = zip_listing(url)

        def sha256_of(name: str) -> str:
            size, offset, method = listing[name]
            if name in known and known[name]["size_bytes"] == size:
                return str(known[name]["sha256"])
            return member_sha256(url, name, size, offset, method)

        sizes = {name: row[0] for name, row in listing.items()}
        built = build_zip(registry, sizes, loader.session_members, sha256_of)
        path.write_text(json.dumps(built, indent=2) + "\n", encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {len(built['sessions'])} sessions")
        return
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
