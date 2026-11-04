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

__all__ = [
    "DatasetInfo",
    "Member",
    "SessionEntry",
    "catalog",
    "dataset_ids",
    "sessions",
]

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
class Member:
    """One member of a zip archive that a session is read from.

    Attributes
    ----------
    path : str
        The member's path inside the zip.
    sha256 : str
        The extracted member's SHA-256, which every extraction is verified
        against.
    size_bytes : int
        The extracted member's size.
    """

    path: str
    sha256: str
    size_bytes: int


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
    license : str
        The session file's license as its archive states it: the dataset's,
        unless the row names its own (a session from a second archive).
    citation : str
        The paper's citation, then the citation of the session's archive.
    members : tuple of Member
        For a session read from members of a remote zip (``url``), the
        members fetched, each by byte range; empty for a whole file. The
        session's ``sha256`` and ``size_bytes`` are then the zip's.
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
    license: str
    citation: str
    members: tuple[Member, ...] = ()

    @property
    def download_url(self) -> str:
        """The URL the whole file is fetched from."""
        if self.asset_id is not None:
            from neurospatial.datasets._fetch import dandi_asset_url

            return dandi_asset_url(self.asset_id)
        assert self.url is not None  # checked when the registry is read
        return self.url

    @property
    def download_bytes(self) -> int:
        """The bytes a download of the session fetches: its members, or the file."""
        if self.members:
            return sum(member.size_bytes for member in self.members)
        return self.size_bytes

    def provenance(self, source: str | os.PathLike[str]) -> Provenance:
        """Where the session's file came from, read from ``source``.

        The archive is the dataset's, with its pinned version, for a DANDI
        asset, and the row's own archive for a file elsewhere.

        Parameters
        ----------
        source : str or path
            The cached file (or, for zip members, the directory holding
            them), whose modification time is when it was fetched, or the
            URL it is streamed from, fetched now.
        """
        from neurospatial._results import software_version

        if isinstance(source, str) and source.startswith(("https://", "http://")):
            fetched = datetime.now(timezone.utc)
        else:
            mtime = Path(source).stat().st_mtime
            fetched = datetime.fromtimestamp(mtime, timezone.utc)
        return Provenance(
            archive=self.info.archive if self.asset_id is not None else self.archive,
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
            "license": self.license,
            "citation": self.citation,
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
        info = DatasetInfo(
            dataset=registry["dataset"],
            paper=registry["paper"],
            citation=_citation(registry["paper_citation"], archive["citation"]),
            species=registry["species"],
            brain_regions=tuple(registry["brain_regions"]),
            apparatus=registry["apparatus"],
            license=registry["license"],
            archive=_archive_name(archive),
            stated=_stated(registry["stated"], name),
        )
        entries = tuple(
            _entry(info, row, registry["paper_citation"])
            for row in registry["sessions"]
        )
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


def _stated(stated: dict[str, Any], name: str) -> Stated:
    """The registry's ``stated`` block: each value with its basis and source.

    ``sampling_rate`` and ``position_span`` are ``{"value", "basis",
    "source"}``, the basis ``"stated"`` or ``"inferred"`` and the source
    never empty; ``position_span`` may be null for a dataset without
    positions. ``speed_limit`` is null or ``{"value", "source"}``, always
    stated.
    """
    values = {"sampling_rate": stated["sampling_rate"]}
    if stated.get("position_span") is not None:
        values["position_span"] = stated["position_span"]
    fields: dict[str, Any] = {}
    for key, value in values.items():
        if value["basis"] not in ("stated", "inferred"):
            msg = f"registry {name}: stated.{key} basis {value['basis']!r}"
            raise ValueError(msg)
        if not str(value["source"]).strip():
            msg = f"registry {name}: stated.{key} gives no source"
            raise ValueError(msg)
        number = value["value"]
        fields[key] = tuple(number) if key == "position_span" else number
        fields[f"{key}_basis"] = value["basis"]
        fields[f"{key}_source"] = value["source"]
    if (limit := stated.get("speed_limit")) is not None:
        fields["speed_limit"] = limit["value"]
        fields["speed_limit_source"] = limit["source"]
    return Stated(**fields)


def _citation(paper: str, data: str) -> str:
    """The paper's citation, then the data's."""
    return f"{paper} Data: {data}"


def _archive_name(archive: dict[str, Any]) -> str:
    """``"DANDI 000582 (version 0.251111.2151)"``, or the archive's own name."""
    if "dandiset" in archive:
        return f"DANDI {archive['dandiset']} (version {archive['version']})"
    return str(archive["name"])


def _entry(info: DatasetInfo, row: dict[str, Any], paper: str) -> SessionEntry:
    """One session row, with its download location and checksums checked.

    A row from an archive other than the dataset's gives its own
    ``license`` and ``data_citation``; a row read from zip members gives
    ``members``, each with ``path``, ``sha256`` and ``size_bytes``.
    """
    sha256 = _sha256(row["sha256"], f"session {row['session']!r}")
    asset_id, url = row.get("asset_id"), row.get("url")
    if (asset_id is None) == (url is None):
        msg = f"session {row['session']!r}: give either asset_id or url"
        raise ValueError(msg)
    members = tuple(
        Member(
            path=str(member["path"]),
            sha256=_sha256(member["sha256"], f"member {member['path']!r}"),
            size_bytes=int(member["size_bytes"]),
        )
        for member in row.get("members", ())
    )
    if members and url is None:
        msg = f"session {row['session']!r}: zip members need the zip's url"
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
        license=row.get("license", info.license),
        citation=(
            _citation(paper, row["data_citation"])
            if "data_citation" in row
            else info.citation
        ),
        members=members,
    )


def _sha256(value: Any, what: str) -> str:
    """``value`` as a SHA-256, refusing anything but 64 hexadecimal digits."""
    sha256 = str(value)
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        msg = f"{what}: sha256 {sha256!r} is not 64 hex digits"
        raise ValueError(msg)
    return sha256


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
        (the session ``load(dataset)`` gives), ``license`` (as the session's
        archive states it), ``archive``, ``size_bytes`` (what loading it
        downloads: the file, or the zip members it is read from) and
        ``citation``.

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
            "license": entry.license,
            "archive": entry.archive,
            "size_bytes": entry.download_bytes,
            "citation": entry.citation,
        }
        for dataset in dataset_ids()
        for entry in sessions(dataset)
    ]
    table = pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))
    return table.astype({"n_units": "Int64"})
