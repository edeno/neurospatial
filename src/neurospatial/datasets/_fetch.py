"""Download public dataset files once into a local cache and verify them.

Every file is pinned by SHA-256. A file is written to a temporary name in its
destination directory and renamed only after its digest matches, so an
interrupted or corrupted download never leaves a file the loaders would trust.
An existing file with a different digest is never overwritten. Files too
large to download (multi-GB DANDI assets) are read by HTTP byte range instead,
pinned by the SHA-256 DANDI publishes for them.

CRCNS files need an account. Their helpers read the login from the
``CRCNS_USERNAME`` and ``CRCNS_PASSWORD`` environment variables and never put
either value, or a page the portal returns (which can echo the account name),
into a message, a log or a file.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.cookiejar
import json
import os
import shutil
import struct
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import IO, Any

#: Seconds to wait for a connection or a read before giving up.
TIMEOUT = 300


def _default_cache() -> Path:
    """The per-user cache: ``%LOCALAPPDATA%\\neurospatial\\datasets`` on Windows, else ``~/.cache/neurospatial/datasets``."""
    platform = sys.platform  # a variable, so mypy checks both branches on any OS
    if platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "neurospatial" / "datasets"
    return Path.home() / ".cache" / "neurospatial" / "datasets"


def cache_root(cache: Path | None = None) -> Path:
    """Return the cache directory, created if absent.

    The first of ``cache``, ``$NEUROSPATIAL_DATA_DIR`` and the per-user
    default: ``%LOCALAPPDATA%\\neurospatial\\datasets`` on Windows,
    ``~/.cache/neurospatial/datasets`` elsewhere.
    """
    if cache is not None:
        root = Path(cache)
    elif env := os.environ.get("NEUROSPATIAL_DATA_DIR"):
        root = Path(env)
    else:
        root = _default_cache()
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
        from neurospatial._exceptions import _format_error

        raise ValueError(
            _format_error(
                f"{dest.name}: downloaded SHA-256 {actual} does not match the "
                f"pinned {sha256}; the download was removed.",
                why="the file was cut short or changed in transit, or the "
                "archive now serves a different file, which the loader was not "
                "checked against.",
                fix="load again to retry the download; if it fails again with "
                "the same SHA-256, the archive's file changed: report it, and do "
                "not update the pin until the session is checked again.",
            )
        )
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
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            shutil.copyfileobj(response, tmp)

    return _install(_download_to_temp(dest, write), dest, sha256)


_DEFLATE64 = 9


_DEFLATE64_CHUNK = 64 << 10  # 64 KiB in; Deflate64 expands at most ~1000x


def _copy_deflate64(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, out: IO[bytes]
) -> None:
    """Inflate a Deflate64 member, which ``zipfile`` cannot read, and check it.

    The local header must name the same member as the central directory, the
    output may not exceed the size the directory states, and its CRC-32 must
    match.
    """
    from neurospatial._optional import require_optional

    inflate64 = require_optional(
        "inflate64", extra="datasets", purpose="reading Deflate64 zip members"
    )
    stream = archive.fp
    assert stream is not None
    stream.seek(info.header_offset)
    header = stream.read(30)
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        msg = f"{info.filename}: no local file header at offset {info.header_offset}"
        raise zipfile.BadZipFile(msg)
    name_length, extra_length = struct.unpack("<HH", header[26:30])
    name = stream.read(name_length)
    if name.decode("utf-8" if info.flag_bits & 0x800 else "cp437") != info.filename:
        msg = f"local header names {name!r}, not {info.filename!r}"
        raise zipfile.BadZipFile(msg)
    stream.seek(extra_length, 1)
    inflater = inflate64.Inflater()
    crc, left, written = 0, info.compress_size, 0
    while left:
        chunk = stream.read(min(left, _DEFLATE64_CHUNK))
        if not chunk:
            msg = f"{info.filename}: archive ended {left} bytes early"
            raise zipfile.BadZipFile(msg)
        left -= len(chunk)
        data = inflater.inflate(chunk)
        written += len(data)
        if written > info.file_size:
            msg = f"{info.filename}: inflates past its stated {info.file_size} bytes"
            raise zipfile.BadZipFile(msg)
        crc = zlib.crc32(data, crc)
        out.write(data)
    if written != info.file_size or crc != info.CRC:
        msg = (
            f"{info.filename}: inflated {written} of {info.file_size} bytes, CRC-32"
            f" {crc:08x} against the archive's {info.CRC:08x}"
        )
        raise zipfile.BadZipFile(msg)


def fetch_zip_member(
    url: str, member: str, *, sha256: str, subdir: str, cache: Path | None = None
) -> Path:
    """Extract one member of a remote zip with HTTP range reads, checking its CRC-32.

    Deflate64 members (compression method 9), which ``zipfile`` cannot read,
    are inflated with ``inflate64``.
    """
    from neurospatial._optional import require_optional

    fsspec = require_optional(
        "fsspec", extra="datasets", purpose="reading zip members by byte range"
    )
    dest = cache_root(cache) / subdir / Path(member).name
    if (found := _existing(dest, sha256)) is not None:
        return found

    def write(tmp: IO[bytes]) -> None:
        with (
            fsspec.open(url, "rb", block_size=4 << 20) as remote,
            zipfile.ZipFile(remote) as archive,
        ):
            info = archive.getinfo(member)
            if info.compress_type == _DEFLATE64:
                _copy_deflate64(archive, info, tmp)
                return
            with archive.open(info) as source:
                shutil.copyfileobj(source, tmp)

    return _install(_download_to_temp(dest, write), dest, sha256)


CRCNS_PORTAL = "https://portal.nersc.gov/project/crcns/download/index.php"


def _crcns_require_account() -> None:
    """Raise ``KeyError`` naming the account variables missing from the environment."""
    if missing := [
        name for name in ("CRCNS_USERNAME", "CRCNS_PASSWORD") if name not in os.environ
    ]:
        msg = (
            f"CRCNS files need a free crcns.org account: {' and '.join(missing)}"
            " not set in the environment (set CRCNS_USERNAME and CRCNS_PASSWORD)"
        )
        raise KeyError(msg)


def _crcns_login_form(path: str) -> bytes:
    """The portal's login form for ``path``, from the environment's account.

    The account is read and encoded in one expression, so no frame that can
    raise holds it in a variable (a traceback that shows local variables,
    such as ``pytest --showlocals``, would print it).
    """
    _crcns_require_account()
    return urllib.parse.urlencode(
        {
            "fn": path,
            "username": os.environ["CRCNS_USERNAME"],
            "password": os.environ["CRCNS_PASSWORD"],
            "submit": "Login",
        }
    ).encode()


def _crcns_open(path: str) -> Any:
    """Open the CRCNS file ``path`` (e.g. ``"hc-31/data/resSave.mat"``) for reading.

    Posts the portal's login form with the account from the environment. Any
    failure of the request (an HTTP status, a refused connection, a server
    that hangs up or times out) names only ``path`` and the status or the
    error's type, and is raised outside the handler so that it carries no
    chained exception: the request's frames hold the password and the
    portal's pages can echo the account name. A ``KeyboardInterrupt`` or
    ``SystemExit`` during the request is raised again as a fresh exception
    of the same type, without those frames; a ``SystemExit`` keeps an integer
    exit code and exits with 1 for any other code (a message could hold the
    form).
    """
    _crcns_require_account()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    failure = None
    stopped: BaseException | None = None
    try:
        response = opener.open(
            CRCNS_PORTAL, data=_crcns_login_form(path), timeout=TIMEOUT
        )
    except urllib.error.HTTPError as exc:
        exc.close()
        failure = f"failed with HTTP {exc.code}"
    except urllib.error.URLError as exc:
        failure = f"failed: {type(exc.reason).__name__}"
    except BaseException as exc:  # any error's frames hold the form, Ctrl-C's too
        if isinstance(exc, KeyboardInterrupt):
            stopped = KeyboardInterrupt()
        elif isinstance(exc, SystemExit):
            stopped = SystemExit(exc.code if isinstance(exc.code, int) else 1)
        else:
            failure = f"failed: {type(exc).__name__}"
    if stopped is not None:
        # A fresh one, raised outside the handler, so Ctrl-C still stops the
        # run without the request's frames.
        raise stopped
    if failure is not None:
        msg = f"CRCNS download of {path} {failure}"
        raise OSError(msg)
    if response.headers.get_content_type() == "text/html":
        response.close()
        msg = (
            f"CRCNS answered {path} with a web page instead of the file: the login"
            " failed (check CRCNS_USERNAME and CRCNS_PASSWORD), the path is wrong,"
            " or the portal asks for a browser step. The page is not shown because"
            " it can contain the account name."
        )
        raise PermissionError(msg)
    return response


def fetch_crcns(
    path: str, *, sha256: str, subdir: str, cache: Path | None = None
) -> Path:
    """Return a verified local copy of the CRCNS file ``path``, downloading it once."""
    dest = cache_root(cache) / subdir / Path(path).name
    if (found := _existing(dest, sha256)) is not None:
        return found

    def write(tmp: IO[bytes]) -> None:
        with _crcns_open(path) as response:
            shutil.copyfileobj(response, tmp)

    return _install(_download_to_temp(dest, write), dest, sha256)


def fetch_crcns_tar_members(
    path: str,
    members: Mapping[str, str],
    *,
    subdir: str,
    cache: Path | None = None,
) -> dict[str, Path]:
    """Return verified local copies of members of the CRCNS ``.tar.gz`` at ``path``.

    ``members`` maps each member's name in the archive to its pinned SHA-256;
    each is stored under its base name. A gzip stream cannot be read by byte
    range, so the archive is read from its start and the download stops once
    every missing member is written; the archive itself is not kept.
    """
    names = [Path(member).name for member in members]
    if len(set(names)) != len(names):
        msg = f"members of {path} share a base name, which is how they are stored: {sorted(members)}"
        raise ValueError(msg)
    root = cache_root(cache) / subdir
    found: dict[str, Path] = {}
    missing: dict[str, tuple[Path, str]] = {}
    for member, digest in members.items():
        dest = root / Path(member).name
        if (cached := _existing(dest, digest)) is not None:
            found[member] = cached
        else:
            missing[member] = (dest, digest)
    if not missing:
        return found
    with (
        _crcns_open(path) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        for info in archive:
            if info.name not in missing:
                continue
            source = archive.extractfile(info)
            if source is None:
                msg = f"{path}: member {info.name} is not a regular file"
                raise ValueError(msg)
            dest, digest = missing.pop(info.name)
            stream: IO[bytes] = source

            def write(tmp: IO[bytes], stream: IO[bytes] = stream) -> None:
                shutil.copyfileobj(stream, tmp)

            found[info.name] = _install(_download_to_temp(dest, write), dest, digest)
            if not missing:
                break
    if missing:
        msg = f"{path} has no member(s) {sorted(missing)}"
        raise FileNotFoundError(msg)
    return found


def dandi_asset_url(asset_id: str) -> str:
    """Return the download URL of a DANDI asset."""
    return f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"


def dandi_stream_url(asset_id: str, *, sha256: str) -> str:
    """Return the asset's download URL after checking DANDI's published SHA-256.

    For files read by byte range, where hashing the whole file would mean
    downloading it.
    """
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
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
    from neurospatial._optional import require_optional

    pynwb = require_optional("pynwb", extra="datasets", purpose="reading NWB files")
    if not source.startswith(("https://", "http://")):
        with pynwb.NWBHDF5IO(source, "r", load_namespaces=True) as io:
            yield io.read()
        return
    fsspec = require_optional(
        "fsspec", extra="datasets", purpose="reading NWB files by byte range"
    )
    h5py = require_optional(
        "h5py", extra="datasets", purpose="reading NWB files by byte range"
    )
    with (
        fsspec.open(
            source, "rb", block_size=4 << 20, cache_type="blockcache"
        ) as remote,
        h5py.File(remote, "r") as h5,
        pynwb.NWBHDF5IO(file=h5, mode="r", load_namespaces=True) as io,
    ):
        yield io.read()
