"""Load one checked session, or download every session of a dataset.

Each dataset is a module ``neurospatial.datasets.<dataset>`` with
``load_session(entry, path_or_stream) -> Recording``, which reads the
session's file into a :class:`Recording` with the dataset's own watch-outs;
``REQUIRES``, the modules it imports from the ``datasets`` extra; and
``WATCH_OUTS``, the pitfalls of each session checked by hand. Its registry,
``registry/<dataset>.json``, lists the sessions. :func:`load` fetches the
file, calls ``load_session`` and runs the automatic checks.
"""

from __future__ import annotations

import logging
import shutil
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from neurospatial._exceptions import InputError, _format_error
from neurospatial.datasets._catalog import SessionEntry, sessions

if TYPE_CHECKING:
    from neurospatial.datasets._recording import Recording

__all__ = ["fetch_all", "fetch_session", "load"]

logger = logging.getLogger("neurospatial.datasets")


def load(
    dataset: str,
    session: str | None = None,
    *,
    cache: Path | None = None,
    stream: bool = False,
    accept_failed_checks: tuple[str, ...] = (),
) -> Recording:
    """Load one session of a public dataset, checked, as the arrays analyses take.

    The first load downloads the session's file into the cache and verifies
    its SHA-256; later loads read the cached file, offline. The session is
    then checked: spikes, events and epochs must be on the tracking clock,
    and the position span and sampling rate must match what the dataset
    states. A failure of either refuses the load, because the arrays would
    be wrong. Other findings (samples at the origin, held tracking, silent
    units) go into ``rec.watch_outs``; no array is changed.

    Parameters
    ----------
    dataset : str
        The dataset id, for example ``"sargolini_2006"``; :func:`catalog`
        lists them.
    session : str, optional
        The archive's session id. Default: the dataset's default session,
        the one the tutorials of other tools use where there is one.
    cache : path, optional
        The cache directory. Default ``$NEUROSPATIAL_DATA_DIR``, else
        ``~/.cache/neurospatial/datasets`` (``%LOCALAPPDATA%`` on Windows).
    stream : bool, default False
        Read a DANDI file by HTTP byte range instead of downloading it,
        after checking that the archive still publishes the pinned SHA-256.
        For files too large to keep; nothing is cached.
    accept_failed_checks : tuple of str, default ()
        Refusing checks (``"clock"``, ``"units"``) to load through. Each
        accepted failure is kept in ``rec.failed_checks`` and printed first
        under "Watch out". Naming a check that passes changes nothing.

    Returns
    -------
    Recording
        The session. Printing it shows the full summary.

    Raises
    ------
    InputError
        If the dataset or session is unknown, ``accept_failed_checks`` names
        an unknown check, or ``stream=True`` for a file outside DANDI.
    DatasetCheckError
        If the session fails the clock or units check and that check is not
        in ``accept_failed_checks``.
    ImportError
        If the ``datasets`` extra is not installed
        (``pip install 'neurospatial[datasets]'``); checked before anything
        is downloaded.
    ValueError
        If the downloaded file's SHA-256 differs from the pinned one.

    See Also
    --------
    catalog : Every session that can be loaded.
    fetch_all : Download every session of a dataset.

    Examples
    --------
    >>> from neurospatial import compute_spatial_rates
    >>> from neurospatial.datasets import load
    >>> rec = load("sargolini_2006")  # doctest: +SKIP
    >>> rates = compute_spatial_rates(  # doctest: +SKIP
    ...     rec.env, rec.spike_times, rec.times, rec.positions
    ... )
    """
    from neurospatial._optional import require_optional
    from neurospatial.datasets._checks import _accepted, run_checks

    _accepted(accept_failed_checks)
    entry = _session_entry(dataset, session)
    module = import_module(f"neurospatial.datasets.{dataset}")
    for name in module.REQUIRES:
        require_optional(name, extra="datasets", purpose=f"loading {dataset}")
    source = _stream_url(entry) if stream else fetch_session(entry, cache=cache)
    rec: Recording = module.load_session(entry, source)
    return run_checks(
        rec, stated=entry.info.stated, accept_failed_checks=accept_failed_checks
    )


def fetch_all(dataset: str, *, cache: Path | None = None) -> list[Path]:
    """Download every session of a dataset into the cache.

    The total size is logged first (logger ``neurospatial.datasets``, level
    INFO), and nothing is downloaded when the cache's disk has less free
    space than the sessions not yet cached. Cached files are verified
    against their pinned SHA-256, not downloaded again.

    Parameters
    ----------
    dataset : str
        The dataset id; :func:`catalog` lists them.
    cache : path, optional
        The cache directory, as for :func:`load`.

    Returns
    -------
    list of Path
        The cached file of each session, in registry order.

    Raises
    ------
    InputError
        If the dataset is unknown.
    OSError
        If the cache's disk has less free space than the download needs.
    ValueError
        If a downloaded file's SHA-256 differs from the pinned one.
    """
    from neurospatial.datasets._fetch import cache_root

    entries = sessions(dataset)
    root = cache_root(cache)
    missing = [e for e in entries if not (root / dataset / e.file_name).exists()]
    total = sum(entry.size_bytes for entry in entries)
    needed = sum(entry.size_bytes for entry in missing)
    logger.info(
        "%s: %d sessions, %s in all; %d to download (%s) into %s",
        dataset,
        len(entries),
        _size(total),
        len(missing),
        _size(needed),
        root,
    )
    free = shutil.disk_usage(root).free
    if needed > free:
        raise OSError(
            _format_error(
                f"Fetching {dataset} needs {_size(needed)} for {len(missing)} "
                f"sessions, but {root} has {_size(free)} free.",
                why="a download that fills the disk stops partway.",
                fix="free space, or pass cache= (or set NEUROSPATIAL_DATA_DIR) "
                "to a disk with room.",
            )
        )
    return [fetch_session(entry, cache=cache) for entry in entries]


def fetch_session(entry: SessionEntry, *, cache: Path | None = None) -> Path:
    """The session's verified file in the cache, downloaded once.

    Files are cached as ``<cache>/<dataset>/<file name>``.
    """
    from neurospatial.datasets._fetch import fetch

    return fetch(
        entry.download_url,
        entry.file_name,
        sha256=entry.sha256,
        subdir=entry.info.dataset,
        cache=cache,
    )


def _stream_url(entry: SessionEntry) -> str:
    """The byte-range URL of a DANDI session, its published digest checked."""
    from neurospatial.datasets._fetch import dandi_stream_url

    if entry.asset_id is None:
        raise InputError(
            _format_error(
                f"Session {entry.session!r} of {entry.info.dataset!r} cannot be "
                f"streamed: it is a file in {entry.archive}, not a DANDI asset.",
                why="streaming reads an NWB file by byte range and checks the "
                "SHA-256 that DANDI publishes for it; other archives publish none.",
                fix="load it with stream=False, which downloads and verifies it.",
            )
        )
    return dandi_stream_url(entry.asset_id, sha256=entry.sha256)


def _session_entry(dataset: str, session: str | None) -> SessionEntry:
    """The registry entry of ``session``, or of the default session."""
    entries = sessions(dataset)
    if session is None:
        return next(entry for entry in entries if entry.default)
    for entry in entries:
        if entry.session == session:
            return entry
    names = [entry.session for entry in entries]
    shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
    raise InputError(
        _format_error(
            f"Dataset {dataset!r} has no session {session!r}.",
            why=f"its registry lists {len(names)} sessions, by the archive's own "
            f"ids ({shown}).",
            fix=f"pick one from catalog().query('dataset == {dataset!r}'), or "
            "pass session=None for the default.",
        )
    )


def _size(n_bytes: float) -> str:
    """``"1.86 GB"``: a byte count in decimal units."""
    for unit in ("B", "kB", "MB", "GB"):
        if n_bytes < 1000:
            return f"{n_bytes:.3g} {unit}"
        n_bytes /= 1000
    return f"{n_bytes:.3g} TB"
