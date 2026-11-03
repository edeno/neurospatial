"""Download public dataset files once into a local cache and verify them.

Every file is pinned by SHA-256. A file is written to a temporary name in its
destination directory and renamed only after its digest matches, so an
interrupted or corrupted download never leaves a file the loaders would trust.
An existing file with a different digest is never overwritten. Files too
large to download (multi-GB DANDI assets) are read by HTTP byte range instead,
pinned by the SHA-256 DANDI publishes for them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any

DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "datasets"


def cache_root(cache: Path | None = None) -> Path:
    """Return the cache directory: ``cache``, ``$NEUROSPATIAL_DATA_DIR`` or ``data/datasets``."""
    root = Path(cache or os.environ.get("NEUROSPATIAL_DATA_DIR", DEFAULT_CACHE))
    root.mkdir(parents=True, exist_ok=True)
    return root


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing(dest: Path, sha256: str) -> Path | None:
    if not dest.exists():
        return None
    if file_sha256(dest) != sha256:
        msg = (
            f"{dest} exists with a different SHA-256 from the pinned file. "
            "Move it aside to download again; it will not be overwritten."
        )
        raise FileExistsError(msg)
    return dest


def _install(tmp: Path, dest: Path, sha256: str) -> Path:
    actual = file_sha256(tmp)
    if actual != sha256:
        tmp.unlink()
        msg = f"{dest.name}: downloaded SHA-256 {actual} does not match the pinned {sha256}."
        raise ValueError(msg)
    tmp.replace(dest)
    return dest


def _download_to_temp(dest: Path, write: Callable[[IO[bytes]], None]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            write(tmp)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    return tmp_path


def fetch(
    url: str, name: str, *, sha256: str, subdir: str, cache: Path | None = None
) -> Path:
    """Return a verified local copy of ``url``, downloading it once."""
    dest = cache_root(cache) / subdir / name
    if (found := _existing(dest, sha256)) is not None:
        return found

    def write(tmp: IO[bytes]) -> None:
        with urllib.request.urlopen(url) as response:
            shutil.copyfileobj(response, tmp)

    return _install(_download_to_temp(dest, write), dest, sha256)


def fetch_zip_member(
    url: str, member: str, *, sha256: str, subdir: str, cache: Path | None = None
) -> Path:
    """Extract one member of a remote zip with HTTP range reads; zipfile checks its CRC-32."""
    import fsspec

    dest = cache_root(cache) / subdir / Path(member).name
    if (found := _existing(dest, sha256)) is not None:
        return found

    def write(tmp: IO[bytes]) -> None:
        with (
            fsspec.open(url, "rb", block_size=4 << 20) as remote,
            zipfile.ZipFile(remote) as archive,
            archive.open(member) as source,
        ):
            shutil.copyfileobj(source, tmp)

    return _install(_download_to_temp(dest, write), dest, sha256)


def dandi_asset_url(asset_id: str) -> str:
    """Return the download URL of a DANDI asset."""
    return f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"


def dandi_stream_url(asset_id: str, *, sha256: str) -> str:
    """Return the asset's download URL after checking DANDI's published SHA-256.

    For files read by byte range, where hashing the whole file would mean
    downloading it.
    """
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/"
    with urllib.request.urlopen(url) as response:
        published = json.load(response)["digest"]["dandi:sha2-256"]
    if published != sha256:
        msg = (
            f"DANDI asset {asset_id} now has SHA-256 {published}, not the pinned"
            f" {sha256}; the file changed, so recheck the loader before updating the pin"
        )
        raise ValueError(msg)
    return dandi_asset_url(asset_id)


@contextlib.contextmanager
def open_nwb(source: str) -> Iterator[Any]:
    """Yield the NWBFile at a local path, or at an http(s) URL read by byte range."""
    import pynwb

    if not source.startswith(("https://", "http://")):
        with pynwb.NWBHDF5IO(source, "r", load_namespaces=True) as io:
            yield io.read()
        return
    import fsspec
    import h5py

    with (
        fsspec.open(
            source, "rb", block_size=4 << 20, cache_type="blockcache"
        ) as remote,
        h5py.File(remote, "r") as h5,
        pynwb.NWBHDF5IO(file=h5, mode="r", load_namespaces=True) as io,
    ):
        yield io.read()
