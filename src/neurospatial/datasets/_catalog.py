"""The dataset registries: every session each archive offers, pinned by checksum.

One JSON file per dataset, ``registry/<dataset>.json``, holds what a person
wrote about the dataset (paper, citation, species, apparatus, the stated
facts the checks compare with) and one entry per session: where it lives,
its SHA-256 and size, whether a person has checked it by hand, and which
session :func:`load` gives by default. ``scripts/build_dataset_registry.py``
writes the archive's part of a DANDI registry.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neurospatial._exceptions import InputError, _format_error
from neurospatial.datasets._checks import Stated
from neurospatial.datasets._lfp import LFPFile
from neurospatial.datasets._recording import Provenance

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.datasets._video import Landmarks, LedFit

__all__ = [
    "DatasetInfo",
    "Member",
    "SessionEntry",
    "VideoAsset",
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
    """One member of a zip archive, or one file, that a session is read from.

    Attributes
    ----------
    path : str
        The member's path inside the zip, or the file's path under the
        session's ``url``.
    sha256 : str
        The extracted member's or the file's SHA-256, which every extraction
        or download is verified against.
    size_bytes : int
        The extracted member's or the file's size.
    """

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class VideoAsset:
    """A session's video file in the archive, beside its NWB file.

    Attributes
    ----------
    asset_path : str
        The video's path in the dandiset: the NWB file's ImageSeries names it
        relative to the NWB file, in a folder named after the file's stem.
    asset_id : str
        The DANDI asset id.
    sha256 : str
        The video's SHA-256, which every download is verified against.
    size_bytes : int
    landmarks : Landmarks or None
        Points a person placed on one frame, with their environment
        positions, for the calibration; None where nobody has.
    led_fit : LedFit or None
        The affine fitted from the LED detected in the video to the
        positions (``scripts/fit_video_led_calibration.py``); None where it
        has not been run.
    """

    asset_path: str
    asset_id: str
    sha256: str
    size_bytes: int
    landmarks: Landmarks | None = None
    led_fit: LedFit | None = None


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
    files : tuple of Member
        For a session read from several files under one folder (``url``), the
        files fetched; empty otherwise. The session's ``sha256`` is then
        :func:`files_sha256` of them and its ``size_bytes`` their total.
    videos : tuple of VideoAsset
        The session's video files, fetched only on request; empty where the
        archive has none.
    lfp_files : tuple of LFPFile
        The files that hold the session's LFP, fetched only on request; empty
        where the archive has none. The loader builds ``rec.lfp`` from them.
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
    files: tuple[Member, ...] = ()
    videos: tuple[VideoAsset, ...] = ()
    lfp_files: tuple[LFPFile, ...] = ()

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


def files_sha256(files: Iterable[Member]) -> str:
    """One SHA-256 for a session read from several files.

    The digest of a line ``<sha256>  <path>`` per file, sorted by path, as
    ``sha256sum`` lists them, so it names every file and its bytes.

    Parameters
    ----------
    files : iterable of Member
    """
    lines = sorted(f"{file.sha256}  {file.path}\n" for file in files)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


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
    positions. ``speed_limit`` is null or ``{"value", "basis", "source"}``,
    the basis always ``"stated"``.
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
        if limit.get("basis", "stated") != "stated":
            msg = (
                f"registry {name}: stated.speed_limit basis {limit['basis']!r}; a "
                "speed limit must be stated by a source"
            )
            raise ValueError(msg)
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
    ``members``, each with ``path``, ``sha256`` and ``size_bytes``. A row with
    an LFP gives ``lfp_files``, each an :class:`LFPFile`'s fields, located at
    the row's own ``url`` or ``asset_id`` unless it names another.
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
    files = tuple(
        Member(
            path=str(file["path"]),
            sha256=_sha256(file["sha256"], f"file {file['path']!r}"),
            size_bytes=int(file["size_bytes"]),
        )
        for file in row.get("files", ())
    )
    if files and (url is None or members):
        msg = f"session {row['session']!r}: files need the folder's url and no members"
        raise ValueError(msg)
    if files and sha256 != files_sha256(files):
        msg = f"session {row['session']!r}: sha256 is not files_sha256 of its files"
        raise ValueError(msg)
    if files and int(row["size_bytes"]) != sum(file.size_bytes for file in files):
        msg = f"session {row['session']!r}: size_bytes is not its files' total"
        raise ValueError(msg)
    videos = tuple(
        VideoAsset(
            asset_path=str(video["asset_path"]),
            asset_id=str(video["asset_id"]),
            sha256=_sha256(video["sha256"], f"video {video['asset_path']!r}"),
            size_bytes=int(video["size_bytes"]),
            landmarks=_landmarks(_keyed(video, "landmarks")),
            led_fit=_led_fit(_keyed(video, "led_fit")),
        )
        for video in row.get("videos", ())
    )
    lfp_files = tuple(
        _lfp_file(item, asset_id=asset_id, url=url) for item in row.get("lfp_files", ())
    )
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
        files=files,
        videos=videos,
        lfp_files=lfp_files,
    )


def _lfp_file(item: dict[str, Any], *, asset_id: Any, url: Any) -> LFPFile:
    """One ``lfp_files`` row; without a location of its own, the session's."""
    own = item.get("asset_id") is not None or item.get("url") is not None
    return LFPFile(
        path=str(item["path"]),
        sha256=_sha256(item["sha256"], f"LFP file {item['path']!r}"),
        size_bytes=int(item["size_bytes"]),
        url=item.get("url") if own else url,
        asset_id=item.get("asset_id") if own else asset_id,
        member=item.get("member"),
        channels=tuple(item.get("channels", ())),
    )


def _keyed(video: dict[str, Any], key: str) -> dict[str, Any] | None:
    """A video's stored ``key`` (landmarks, a fit), refused unless made on this file.

    Each stored calibration names the ``video_sha256`` it was made on, so a
    changed video cannot keep a calibration of the old one.
    """
    value = video.get(key)
    if value is not None and value.get("video_sha256") != video["sha256"]:
        msg = (
            f"video {video['asset_path']!r}: its {key} was made on a video with "
            f"SHA-256 {value.get('video_sha256')!r}, not this one; rerun it"
        )
        raise ValueError(msg)
    return value


def _landmarks(value: dict[str, Any] | None) -> Landmarks | None:
    """A video's ``landmarks`` row as :class:`Landmarks`; None without one.

    The row gives ``px``, ``env``, ``frame_size_px``, ``frame`` and
    ``placed_by``.
    """
    if value is None:
        return None
    from neurospatial.datasets._video import Landmarks

    width, height = value["frame_size_px"]
    return Landmarks(
        px=value["px"],
        env=value["env"],
        frame_size_px=(int(width), int(height)),
        frame=int(value["frame"]),
        placed_by=str(value["placed_by"]),
    )


def _led_fit(value: dict[str, Any] | None) -> LedFit | None:
    """A video's ``led_fit`` row as :class:`LedFit`; None without one."""
    if value is None:
        return None
    from neurospatial.datasets._video import LedFit

    width, height = value["frame_size_px"]
    return LedFit(
        matrix=value["matrix"],
        frame_size_px=(int(width), int(height)),
        method=str(value["method"]),
        frames=str(value["frames"]),
        n_frames=int(value["n_frames"]),
        median_cm=float(value["residual_cm"]["median"]),
        p90_cm=float(value["residual_cm"]["p90"]),
        median_px=float(value["residual_px"]["median"]),
        p90_px=float(value["residual_px"]["p90"]),
        outlier_cm=float(value["outliers"]["cut_cm"]),
        outlier_share=float(value["outliers"]["share"]),
        lag_scan=tuple(
            (int(lag), float(median))
            for lag, median in value["lag_scan_median_cm"].items()
        ),
        fitted_by=str(value["fitted_by"]),
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
        (checked by hand against its archive, its files and the sources its
        page lists), ``default`` (the session ``load(dataset)`` gives),
        ``license`` (as the session's
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
