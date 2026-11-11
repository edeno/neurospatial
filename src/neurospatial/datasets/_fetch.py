"""Download public dataset files once into a local cache and verify them.

Every file is pinned by SHA-256. A file is written to a temporary name in its
destination directory and renamed only after its digest matches, so an
interrupted or corrupted download never leaves a file the loaders would trust.
An existing file with a different digest is never overwritten. Each verified
file gets a record, ``<cache>/.verified/<subdir>/<name>.verified``, of its
digest, size and modification time; a later fetch trusts a file whose size
and modification time still match its record without hashing it again, and
hashes any other. The records sit apart from the files, so a session's
directory holds only its files and keeps its modification time. Files too
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
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

#: Seconds to wait for a connection or a read before giving up.
TIMEOUT = 300
#: Sent with each download: Mendeley Data refuses urllib's default with HTTP 403.
_HEADERS = {"User-Agent": "neurospatial"}


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


#: The directory, under the cache root, of the verified files' records.
VERIFIED = ".verified"


@dataclass(frozen=True)
class _Target:
    """Where a file is cached, and where the record of its verification is."""

    path: Path
    record: Path


def _target(cache: Path | None, subdir: str, name: str) -> _Target:
    """The cache location of file ``name`` in ``subdir``, and of its record."""
    root = cache_root(cache)
    return _Target(root / subdir / name, root / VERIFIED / subdir / f"{name}.verified")


def _stat_record(sha256: str, stat: os.stat_result) -> dict[str, Any]:
    """What a record holds: the file's digest, size and modification time."""
    return {"sha256": sha256, "st_size": stat.st_size, "st_mtime_ns": stat.st_mtime_ns}


def _write_record(target: _Target, sha256: str) -> None:
    """Record that ``target.path``, as it is now, has SHA-256 ``sha256``."""
    record = _stat_record(sha256, target.path.stat())
    # A cache that cannot be written to (a shared, read-only one) keeps no
    # records; its files are hashed on every load, as without records.
    with contextlib.suppress(OSError):
        target.record.parent.mkdir(parents=True, exist_ok=True)
        partial = target.record.with_name(f"{target.record.name}.{os.getpid()}")
        partial.write_text(json.dumps(record))
        partial.replace(target.record)


def _recorded(target: _Target, sha256: str, stat: os.stat_result) -> bool:
    """Whether the record shows the file, at this size and time, has ``sha256``."""
    try:
        record = json.loads(target.record.read_text())
    except (OSError, ValueError):  # no record, or a record cut short
        return False
    return bool(record == _stat_record(sha256, stat))


def _existing(target: _Target, sha256: str) -> Path | None:
    """The cached file if it is there and has ``sha256``; None if it is not there.

    A file whose record matches its size and modification time is trusted;
    any other is hashed, and recorded when it matches.
    """
    dest = target.path
    if not dest.exists():
        return None
    if _recorded(target, sha256, dest.stat()):
        return dest
    if file_sha256(dest) != sha256:
        msg = (
            f"{dest} exists with a different SHA-256 from the pinned file. "
            "Move it aside to download again; it will not be overwritten."
        )
        raise FileExistsError(msg)
    _write_record(target, sha256)
    return dest


def _install(tmp: Path, actual: str, target: _Target, sha256: str) -> Path:
    """Rename the download ``tmp``, whose SHA-256 is ``actual``, into the cache.

    Refuses, and removes ``tmp``, when ``actual`` is not the pinned
    ``sha256``; records the installed file's verification otherwise.
    """
    dest = target.path
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
    _write_record(target, sha256)
    return dest


class _HashingWriter:
    """A binary file that hashes, with SHA-256, every byte written to it."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        """Hash ``data`` and write it."""
        self._digest.update(data)
        return self._stream.write(data)

    def hexdigest(self) -> str:
        """The hex SHA-256 of everything written."""
        return self._digest.hexdigest()


def _download_to_temp(
    dest: Path, write: Callable[[_HashingWriter], None]
) -> tuple[Path, str]:
    """Write a download to a temporary file beside ``dest``, hashing it as it goes.

    Returns the temporary file and its hex SHA-256; on any error the
    temporary file is removed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        hashing = _HashingWriter(tmp)
        try:
            write(hashing)
        except BaseException:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise
    return tmp_path, hashing.hexdigest()


def fetch(
    url: str, name: str, *, sha256: str, subdir: str, cache: Path | None = None
) -> Path:
    """Return a verified local copy of ``url``, downloading it once.

    A CRCNS file's URL (:func:`crcns_url`) is fetched with the account
    (:func:`fetch_crcns`).
    """
    if (path := crcns_path(url)) is not None:
        if Path(path).name != name:
            msg = f"{path} is cached under its own name, not {name!r}"
            raise ValueError(msg)
        return fetch_crcns(path, sha256=sha256, subdir=subdir, cache=cache)
    target = _target(cache, subdir, name)
    if (found := _existing(target, sha256)) is not None:
        return found

    def write(tmp: _HashingWriter) -> None:
        request = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            shutil.copyfileobj(response, tmp)

    return _install(*_download_to_temp(target.path, write), target, sha256)


_DEFLATE64 = 9


_DEFLATE64_CHUNK = 64 << 10  # 64 KiB in; Deflate64 expands at most ~1000x


def _copy_deflate64(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, out: _HashingWriter
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
    target = _target(cache, subdir, Path(member).name)
    if (found := _existing(target, sha256)) is not None:
        return found

    def write(tmp: _HashingWriter) -> None:
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

    return _install(*_download_to_temp(target.path, write), target, sha256)


CRCNS_PORTAL = "https://portal.nersc.gov/project/crcns/download/index.php"
#: How a registry names a CRCNS file: the portal's URL with the file's path.
_CRCNS_FILE = f"{CRCNS_PORTAL}?fn="


def crcns_url(path: str) -> str:
    """The URL a registry gives the CRCNS file ``path`` (``"hc-3/<dir>/<file>"``)."""
    return f"{_CRCNS_FILE}{path}"


def crcns_path(url: str) -> str | None:
    """The CRCNS path of a :func:`crcns_url`; None for any other URL."""
    return url.removeprefix(_CRCNS_FILE) if url.startswith(_CRCNS_FILE) else None


def fetch_members(
    url: str,
    members: Mapping[str, str],
    *,
    subdir: str,
    cache: Path | None = None,
) -> dict[str, Path]:
    """Verified local copies of members of the remote archive at ``url``.

    A CRCNS ``.tar.gz`` (a :func:`crcns_url`) is read once from its start for
    all of ``members`` (:func:`fetch_crcns_tar_members`); a zip anywhere else is
    read member by member by byte range (:func:`fetch_zip_member`).

    Parameters
    ----------
    url : str
        The archive's URL.
    members : Mapping[str, str]
        Each member's path in the archive and its pinned SHA-256.
    subdir : str
        The cache folder the members are stored in, under their base names.
    cache : Path, optional
        The cache root.

    Returns
    -------
    dict of str to Path
        Each member's cached copy, by its path in the archive.
    """
    if (path := crcns_path(url)) is not None:
        return fetch_crcns_tar_members(path, members, subdir=subdir, cache=cache)
    return {
        member: fetch_zip_member(url, member, sha256=sha, subdir=subdir, cache=cache)
        for member, sha in members.items()
    }


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
    target = _target(cache, subdir, Path(path).name)
    if (found := _existing(target, sha256)) is not None:
        return found

    def write(tmp: _HashingWriter) -> None:
        with _crcns_open(path) as response:
            shutil.copyfileobj(response, tmp)

    return _install(*_download_to_temp(target.path, write), target, sha256)


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
    found: dict[str, Path] = {}
    missing: dict[str, tuple[_Target, str]] = {}
    for member, digest in members.items():
        target = _target(cache, subdir, Path(member).name)
        if (cached := _existing(target, digest)) is not None:
            found[member] = cached
        else:
            missing[member] = (target, digest)
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
            target, digest = missing.pop(info.name)
            stream: IO[bytes] = source

            def write(tmp: _HashingWriter, stream: IO[bytes] = stream) -> None:
                shutil.copyfileobj(stream, tmp)

            found[info.name] = _install(
                *_download_to_temp(target.path, write), target, digest
            )
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
    """Yield the NWBFile at a local path, or at an http(s) URL read by byte range.

    Extension types are read with the specification the file itself caches,
    whichever version of the extension the process has imported.

    Parameters
    ----------
    source : str
        A local path, or an http(s) URL.

    Yields
    ------
    pynwb.NWBFile
        Open until the block ends.

    Raises
    ------
    RuntimeError
        If the file caches an extension at another version than the process
        has loaded and the installed hdmf (before 4.1.0) cannot read the file
        with its own version.
    """
    from neurospatial._optional import require_optional

    pynwb = require_optional("pynwb", extra="datasets", purpose="reading NWB files")
    h5py = require_optional("h5py", extra="datasets", purpose="reading NWB files")
    if not source.startswith(("https://", "http://")):
        with h5py.File(source, "r") as h5, _nwb_io(pynwb, h5, source) as io:
            yield io.read()
        return
    fsspec = require_optional(
        "fsspec", extra="datasets", purpose="reading NWB files by byte range"
    )
    with (
        fsspec.open(
            source, "rb", block_size=4 << 20, cache_type="blockcache"
        ) as remote,
        h5py.File(remote, "r") as h5,
        _nwb_io(pynwb, h5, source) as io,
    ):
        yield io.read()


def _nwb_io(pynwb: Any, h5: Any, source: str) -> Any:
    """Open ``NWBHDF5IO`` on an HDF5 file, its extensions as the file specifies them.

    pynwb reads a file with a copy of its process-wide type map, to which it
    adds the namespaces the file caches. Importing an extension package puts
    that package's version in the map, and a namespace the copy already holds
    at another version is not added: hdmf warns that it ignored the file's,
    and every object of a type only the file's version defines is then left
    out of what is read, with no further notice. After ``import ndx_events``
    (0.4.0), a file written with ndx-events 0.2.0 has no ``LabeledEvents``.

    Here such an extension is taken out of the copy first, so the file's
    version is the one added and its types are read as in a process that has
    not imported the package. pynwb's and hdmf's own namespaces stay as
    loaded: their classes belong to the installed schema.

    hdmf has one public call that copies the classes of chosen namespaces,
    ``TypeMap.copy_mappers``. hdmf 5.0 deprecated it for ``TypeMap.merge``,
    which copies every class and so needs the namespace being taken out; from
    5.0 on, hdmf's DeprecationWarning accompanies the read.

    Parameters
    ----------
    pynwb : module
    h5 : h5py.File
        Open for reading.
    source : str
        The file's path or URL, for the error message.

    Returns
    -------
    pynwb.NWBHDF5IO

    Raises
    ------
    RuntimeError
        If an extension must be taken out and hdmf cannot build a type map
        without it: hdmf before 4.1.0 has no
        ``NamespaceCatalog.core_namespaces``.
    """
    import hdmf
    from hdmf.build import BuildManager, TypeMap
    from hdmf.common import CORE_NAMESPACE as HDMF_COMMON
    from hdmf.common import EXP_NAMESPACE as HDMF_EXPERIMENTAL
    from hdmf.common import get_type_map as hdmf_type_map
    from hdmf.spec import NamespaceCatalog

    from neurospatial._exceptions import _format_error

    type_map = pynwb.get_type_map()
    loaded = type_map.namespace_catalog
    own = {pynwb.CORE_NAMESPACE, HDMF_COMMON, HDMF_EXPERIMENTAL}
    # extension -> (the file's version, the loaded version), where they differ
    differing = {
        name: (version, loaded.get_namespace(name)["version"])
        for name, version in pynwb.NWBHDF5IO.get_namespaces(file=h5).items()
        if name not in own
        and name in loaded.namespaces
        and loaded.get_namespace(name)["version"] != version
    }
    if differing:
        try:
            kept = NamespaceCatalog(
                loaded.group_spec_cls,
                loaded.dataset_spec_cls,
                loaded.spec_namespace_cls,
                core_namespaces=list(loaded.core_namespaces),
            )
            for name in loaded.namespaces:
                if name not in differing:
                    kept.add_namespace(name, loaded.get_namespace(name))
            without = TypeMap(kept)
            # hdmf's class generator for table types; copy_mappers copies none.
            without.merge(hdmf_type_map())
            without.copy_mappers(type_map)
        except (AttributeError, TypeError) as error:
            cached = ", ".join(f"{name} {v[0]}" for name, v in differing.items())
            imported = ", ".join(f"{name} {v[1]}" for name, v in differing.items())
            raise RuntimeError(
                _format_error(
                    f"{source}: cannot read the file with the extension version "
                    "it was written with.",
                    why=f"the file caches {cached}, and this process has loaded "
                    f"{imported} (importing an extension package registers its "
                    "version with pynwb). pynwb would ignore the file's version "
                    "and leave out every object of a type only that version "
                    "defines. Setting the loaded version aside needs hdmf>=4.1.0 "
                    "(NamespaceCatalog.core_namespaces and TypeMap.copy_mappers); "
                    f"hdmf {hdmf.__version__} is installed and raised "
                    f"{type(error).__name__}: {error}.",
                    fix='pip install "hdmf>=4.1.0", or read this file in a Python '
                    "process that has not imported the extension package.",
                )
            ) from error
        type_map = without
    pynwb.NWBHDF5IO.load_namespaces(type_map, file=h5)
    return pynwb.NWBHDF5IO(file=h5, mode="r", manager=BuildManager(type_map))
