"""First download of a CRCNS file: check it whole against CRCNS's MD5, then pin members.

The loaders pin each kept file by SHA-256 (``_fetch.fetch_crcns``,
``_fetch.fetch_crcns_tar_members``), but a SHA-256 has to come from a first
download. This script makes that first download trustworthy: it streams the
whole CRCNS file once, computes its MD5 and compares it with the value in the
data set's ``checksums.md5``; members matching ``member_regex`` (or the file
itself, without a regex) are staged and moved into the cache only if the MD5
matches. It prints each kept file's name, size and SHA-256 for the loader's
pins, and nothing else (no portal page, no credential).

Run (one file at a time; the MD5 is copied from the set's ``checksums.md5``)::

    uv run --group datasets python datasets/studies/crcns_first_download.py \\
        hc-3/ec013.38/ec013.669.tar.gz dc3a6538ce24219e4b35b83faead75c9 \\
        crcns_hc3 'ec013\\.669\\.(clu\\.\\d+|res\\.\\d+|whl|xml)$'

``BATCH_CRCNS.md`` lists the six runs and their results.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurospatial.datasets._fetch import _crcns_open, cache_root


class _Hashing:
    """A readable stream that counts and MD5-hashes what passes through it."""

    def __init__(self, raw: object) -> None:
        self.raw, self.md5, self.n = raw, hashlib.md5(), 0

    def read(self, size: int = -1) -> bytes:
        block = self.raw.read(size)
        self.md5.update(block)
        self.n += len(block)
        return block


def first_download(
    path: str, md5: str, subdir: str, member_regex: str | None = None
) -> bool:
    """Stream ``path`` whole, stage the wanted files, keep them if the MD5 matches."""
    root = cache_root() / subdir
    stage = root / ".staging"
    stage.mkdir(parents=True, exist_ok=True)
    wanted = re.compile(member_regex) if member_regex else None
    start, staged = time.time(), []
    with _crcns_open(path) as response:
        stream = _Hashing(response)
        if wanted is None:
            dest = stage / Path(path).name
            with dest.open("wb") as out:
                while block := stream.read(1 << 20):
                    out.write(block)
            staged.append(dest)
        else:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for info in archive:
                    if info.isfile() and wanted.search(info.name):
                        dest = stage / Path(info.name).name
                        source = archive.extractfile(info)
                        if source is None:
                            msg = f"{info.name} has no data"
                            raise ValueError(msg)
                        with source, dest.open("wb") as out:
                            while block := source.read(1 << 20):
                                out.write(block)
                        staged.append(dest)
            while stream.read(1 << 20):  # the rest of the archive, for the MD5
                pass
    matched = stream.md5.hexdigest() == md5
    print(
        f"{path}: {stream.n / 1e6:.1f} MB in {time.time() - start:.0f}s;"
        f" md5 match {matched}"
    )
    if not matched:
        return False
    for file in staged:
        final = root / file.name
        if final.exists():
            print(f"exists, left alone: {final.name}")
            continue
        file.replace(final)
        digest = hashlib.sha256(final.read_bytes()).hexdigest()
        print(f"{final.name}\t{final.stat().st_size}\t{digest}")
    if not any(stage.iterdir()):
        stage.rmdir()
    return True


if __name__ == "__main__":
    sys.exit(0 if first_download(*sys.argv[1:5]) else 1)
