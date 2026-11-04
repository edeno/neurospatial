"""The dataset registries: every session each archive offers, pinned by checksum.

One JSON file per dataset, ``registry/<dataset>.json``, holds what a person
wrote about the dataset (paper, citation, species, apparatus, the stated
facts the checks compare with) and one entry per session: where it lives,
its SHA-256 and size, whether a person has checked it by hand, and which
session :func:`load` gives by default. ``scripts/build_dataset_registry.py``
writes the archive's part of a DANDI registry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neurospatial._exceptions import InputError, _format_error
from neurospatial.datasets._checks import Stated
from neurospatial.datasets._recording import Provenance

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["DatasetInfo", "SessionEntry", "catalog", "dataset_ids", "sessions"]

REGISTRY = Path(__file__).parent / "registry"
CATALOG_COLUMNS = (
    "dataset",
    "session",
    "species",
    "brain_regions",
    "apparatus",
    "n_units",
    "verified",
    "default",
    "license",
    "archive",
    "size_bytes",
    "citation",
)


@dataclass(frozen=True)
class DatasetInfo:
    """What a dataset's registry states about all of its sessions.

    Attributes
    ----------
    dataset : str
        The dataset id, ``<first author>_<year>``.
    paper : str
        Short reference, for example ``"Sargolini et al. (2006)"``.
    citation : str
        The paper's citation, then the archive's citation for the data.
    species, apparatus : str
    brain_regions : tuple of str
    license : str
        The data license as the archive states it.
    archive : str
        The archive and record, with its pinned version where it has one.
    stated : Stated
        The sampling rate and position span the checks compare each session
        with, and a speed limit where a source states one.
    """

    dataset: str
    paper: str
    citation: str
    species: str
    brain_regions: tuple[str, ...]
    apparatus: str
    license: str
    archive: str
    stated: Stated


@dataclass(frozen=True)
class SessionEntry:
    """One session of a registry: where it lives and what is known about it.

    Attributes
    ----------
    info : DatasetInfo
        The dataset the session belongs to.
    session : str
        The archive's own session id.
    archive : str
        The archive holding the file, for example ``"DANDI 000582"``.
    sha256 : str
        The file's SHA-256, which every download is verified against.
    size_bytes : int
    verified : bool
        Whether a person has checked the session by hand.
    default : bool
        Whether :func:`load` gives this session when none is named.
    n_units : int or None
        The number of units, where it is known.
    asset_id, asset_path : str or None
        The DANDI asset's id and path, for a DANDI file.
    url : str or None
        The file's URL, for a file outside DANDI.
    file_name : str
        The name the file is cached under.
    verification : str
        The evidence of the check by hand; empty for an unverified session.
    """

    info: DatasetInfo
    session: str
    archive: str
    sha256: str
    size_bytes: int
    verified: bool
    default: bool
    n_units: int | None
    asset_id: str | None
    asset_path: str | None
    url: str | None
    file_name: str
    verification: str

    @property
    def download_url(self) -> str:
        """The URL the whole file is fetched from."""
        if self.asset_id is not None:
            from neurospatial.datasets._fetch import dandi_asset_url

            return dandi_asset_url(self.asset_id)
        assert self.url is not None  # checked when the registry is read
        return self.url

    def provenance(self, source: str | os.PathLike[str]) -> Provenance:
        """Where the session's file came from, read from ``source``.

        Parameters
        ----------
        source : str or path
            The cached file, whose modification time is when it was fetched,
            or the URL it is streamed from, fetched now.
        """
        from neurospatial._results import software_version

        if isinstance(source, str) and source.startswith(("https://", "http://")):
            fetched = datetime.now(timezone.utc)
        else:
            mtime = Path(source).stat().st_mtime
            fetched = datetime.fromtimestamp(mtime, timezone.utc)
        return Provenance(
            archive=self.info.archive,
            url=self.download_url,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            neurospatial_version=software_version(),
            fetched_at=fetched.isoformat(timespec="seconds"),
        )

    def identity(self, source: str | os.PathLike[str]) -> dict[str, Any]:
        """The :class:`Recording` fields the registry states for this session.

        ``dataset``, ``session``, ``paper``, ``license``, ``citation``,
        ``provenance`` (see :meth:`provenance`) and ``verified``.
        """
        return {
            "dataset": self.info.dataset,
            "session": self.session,
            "paper": self.info.paper,
            "license": self.info.license,
            "citation": self.info.citation,
            "provenance": self.provenance(source),
            "verified": self.verified,
        }


def dataset_ids() -> tuple[str, ...]:
    """The ids of every dataset with a registry, sorted."""
    return tuple(sorted(path.stem for path in REGISTRY.glob("*.json")))


def sessions(dataset: str) -> tuple[SessionEntry, ...]:
    """Every session of ``dataset``, as its registry lists them.

    Raises
    ------
    InputError
        If ``dataset`` has no registry; the message lists those that do.
    ValueError
        If the registry is malformed: a field missing, a checksum that is
        not 64 hexadecimal digits, a repeated session, or not exactly one
        default session.
    """
    known = dataset_ids()
    if dataset not in known:
        raise InputError(
            _format_error(
                f"No dataset {dataset!r}.",
                why="each dataset is listed by its registry file.",
                fix=f"use one of {list(known)}; catalog() lists their sessions.",
            )
        )
    path = REGISTRY / f"{dataset}.json"
    return _read(json.loads(path.read_text(encoding="utf-8")), path.name)


def _read(registry: dict[str, Any], name: str) -> tuple[SessionEntry, ...]:
    """The entries of one registry, after checking its fields."""
    try:
        archive = registry["archive"]
        stated = registry["stated"]
        info = DatasetInfo(
            dataset=registry["dataset"],
            paper=registry["paper"],
            citation=f"{registry['paper_citation']} Data: {archive['citation']}",
            species=registry["species"],
            brain_regions=tuple(registry["brain_regions"]),
            apparatus=registry["apparatus"],
            license=registry["license"],
            archive=_archive_name(archive),
            stated=Stated(
                sampling_rate=stated["sampling_rate"],
                position_span=(
                    None
                    if stated.get("position_span") is None
                    else tuple(stated["position_span"])
                ),
                speed_limit=stated.get("speed_limit"),
                speed_limit_source=stated.get("speed_limit_source", ""),
            ),
        )
        entries = tuple(_entry(info, row) for row in registry["sessions"])
    except KeyError as exc:
        msg = f"registry {name} has no field {exc}"
        raise ValueError(msg) from exc
    if info.dataset != Path(name).stem:
        msg = f"registry {name} names dataset {info.dataset!r}"
        raise ValueError(msg)
    names = [entry.session for entry in entries]
    if len(set(names)) != len(names):
        msg = f"registry {name} repeats a session"
        raise ValueError(msg)
    if (n_default := sum(entry.default for entry in entries)) != 1:
        msg = f"registry {name} has {n_default} default sessions, not one"
        raise ValueError(msg)
    if unchecked := [e.session for e in entries if e.verified and not e.verification]:
        msg = f"registry {name}: verified sessions {unchecked} give no verification"
        raise ValueError(msg)
    return entries


def _archive_name(archive: dict[str, Any]) -> str:
    """``"DANDI 000582 (version 0.251111.2151)"``, or the archive's own name."""
    if "dandiset" in archive:
        return f"DANDI {archive['dandiset']} (version {archive['version']})"
    return str(archive["name"])


def _entry(info: DatasetInfo, row: dict[str, Any]) -> SessionEntry:
    """One session row, with its download location and checksum checked."""
    sha256 = str(row["sha256"])
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        msg = f"session {row['session']!r}: sha256 {sha256!r} is not 64 hex digits"
        raise ValueError(msg)
    asset_id, url = row.get("asset_id"), row.get("url")
    if (asset_id is None) == (url is None):
        msg = f"session {row['session']!r}: give either asset_id or url"
        raise ValueError(msg)
    location = row.get("asset_path") or row.get("file_name") or str(url)
    return SessionEntry(
        info=info,
        session=row["session"],
        archive=row["archive"],
        sha256=sha256,
        size_bytes=int(row["size_bytes"]),
        verified=bool(row["verified"]),
        default=bool(row["default"]),
        n_units=row.get("n_units"),
        asset_id=asset_id,
        asset_path=row.get("asset_path"),
        url=url,
        file_name=Path(location).name,
        verification=row.get("verification", ""),
    )


def catalog() -> pd.DataFrame:
    """List every session of every dataset, one row each.

    Reading the catalog downloads nothing.

    Returns
    -------
    pandas.DataFrame
        One row per session, with columns ``dataset``, ``session``,
        ``species``, ``brain_regions`` (comma-separated), ``apparatus``,
        ``n_units`` (missing where not known before loading), ``verified``
        (checked by hand against the file and its sources), ``default``
        (the session ``load(dataset)`` gives), ``license`` (as the archive
        states it), ``archive``, ``size_bytes`` and ``citation``.

    See Also
    --------
    load : Load one session.
    fetch_all : Download every session of a dataset.

    Examples
    --------
    >>> from neurospatial.datasets import catalog
    >>> table = catalog()
    >>> defaults = table[table["default"]].set_index("dataset")["session"]
    >>> defaults["sargolini_2006"]
    'sub-10073_ses-17010302'
    """
    import pandas as pd

    rows = [
        {
            "dataset": entry.info.dataset,
            "session": entry.session,
            "species": entry.info.species,
            "brain_regions": ", ".join(entry.info.brain_regions),
            "apparatus": entry.info.apparatus,
            "n_units": entry.n_units,
            "verified": entry.verified,
            "default": entry.default,
            "license": entry.info.license,
            "archive": entry.archive,
            "size_bytes": entry.size_bytes,
            "citation": entry.info.citation,
        }
        for dataset in dataset_ids()
        for entry in sessions(dataset)
    ]
    table = pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))
    return table.astype({"n_units": "Int64"})
