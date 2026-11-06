"""A session's local field potential: where it is, its channels, and how to read it.

A dataset that publishes the LFP gives each session an :class:`LFPSource`: the
files or zip members that hold it (each with its SHA-256 and size), its
sampling rate, what each channel is, and its unit and clock, each recorded as a
:class:`~neurospatial.datasets.Conversion`. Nothing is downloaded at load, since
LFP is often far larger than the rest of a session;
:meth:`LFPSource.fetch` downloads what the chosen channels need and returns the
samples and their times on the recording's clock.

Archives store LFP in unrelated ways (a file per trial, one MATLAB file in a
zip, a file per channel, a series inside an NWB file), so a loader supplies the
reader that turns its fetched files into arrays. The reader must be a
module-level function, or a :func:`functools.partial` of one, so that a
recording can be pickled.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import InputError, _format_error
from neurospatial._results import reduce_through_constructor
from neurospatial.datasets._errors import DownloadWarning
from neurospatial.datasets._recording import Conversion

__all__ = ["LFPFile", "LFPReader", "LFPSource"]

#: Reads ``(sources, channels)`` into ``(samples, times)``. ``sources`` holds
#: one local path per file fetched, or a URL for a streamed one, in the order of
#: ``LFPSource.files``. ``samples`` has shape ``(n_time, n_channels)``, columns
#: in the order of ``channels``, in the source's ``unit``; ``times`` has shape
#: ``(n_time,)``, seconds on the recording's clock.
LFPReader: TypeAlias = Callable[
    [tuple[str | Path, ...], tuple[str, ...]],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]

#: A step between samples beyond this many sampling periods is a gap. It is the
#: rule ``ripple_detection`` splits its contiguous blocks by.
GAP_PERIODS = 1.5
#: The median step between samples may differ from one sampling period by this
#: share; beyond it, the stated rate does not describe the samples.
RATE_TOLERANCE = 0.01
#: ``fetch()`` with no channels reads every channel only up to this many, unless
#: ripple channels are stated: beyond it a session's files run to hundreds of
#: gigabytes and its samples to more than memory.
_MAX_DEFAULT_CHANNELS = 32


@dataclass(frozen=True)
class LFPFile:
    """One file, or one member of a zip, that holds LFP.

    Parameters
    ----------
    path : str
        Where it is cached, relative to ``<cache>/<dataset>/``. A file the
        session's loader has already fetched gives that file's path, so it is
        not fetched twice.
    sha256 : str
        The file's SHA-256 (the extracted member's, for a zip member), which
        every download is verified against.
    size_bytes : int
        The file's size (the extracted member's, for a zip member).
    url, asset_id : str or None
        The file's URL (the zip's, for a member), or its DANDI asset id;
        exactly one.
    member : str or None
        The member's path inside the zip at ``url``, read by byte range.
    channels : tuple of str, default ()
        The channels this file holds, so a request for one channel fetches
        only the files it needs. Empty when every file is needed for any
        channel (one file per trial).

    Raises
    ------
    InputError
        If the location or checksum is inconsistent; the message names each
        problem.
    """

    path: str
    sha256: str
    size_bytes: int
    url: str | None = None
    asset_id: str | None = None
    member: str | None = None
    channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        problems = []
        if (self.url is None) == (self.asset_id is None):
            problems.append("give exactly one of url and asset_id")
        if self.member is not None and self.url is None:
            problems.append("a zip member needs the zip's url")
        if self.member is not None and Path(self.member).name != Path(self.path).name:
            problems.append(
                f"member {self.member!r} is cached under its own name, not "
                f"{Path(self.path).name!r}"
            )
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            problems.append(f"sha256 {self.sha256!r} is not 64 hexadecimal digits")
        if self.size_bytes < 0:
            problems.append(f"size_bytes {self.size_bytes} is negative")
        if problems:
            raise InputError(
                _format_error(
                    f"LFPFile {self.path!r} cannot be built:\n- "
                    + "\n- ".join(problems),
                    why="the file is fetched and verified from these fields.",
                    fix="correct the loader so each field listed meets the "
                    "contract in the LFPFile docstring.",
                )
            )

    @property
    def download_url(self) -> str:
        """The URL the file (or the zip holding it) is fetched from."""
        if self.asset_id is not None:
            from neurospatial.datasets._fetch import dandi_asset_url

            return dandi_asset_url(self.asset_id)
        assert self.url is not None  # checked at construction
        return self.url


@dataclass(frozen=True)
class LFPSource:
    """A session's LFP: where it is, its channels, and its unit and clock.

    ``rec.lfp`` holds it for a session whose dataset publishes the LFP, and is
    None otherwise. Nothing is downloaded until :meth:`fetch`.

    Parameters
    ----------
    dataset : str
        The dataset id, which names the cache folder.
    files : tuple of LFPFile
        What holds the LFP.
    sampling_rate : float
        Samples per second, Hz.
    channels : Mapping[str, str]
        Each channel's name and what it is, for example
        ``{"TT3": "tetrode 3, CA1 pyramidal layer"}``, in file order.
    unit : str
        The unit :meth:`fetch` returns samples in, for example ``"uV"``, or
        ``"arbitrary"`` where no source gives a scale.
    unit_basis : Conversion
        How stored values become ``unit``, and where the scale comes from.
    times_basis : Conversion
        How the samples' times are placed on the recording's clock.
    reader : LFPReader
        The dataset's reader; see :data:`LFPReader`. A module-level function
        or a :func:`functools.partial` of one.
    ripple_channels : tuple of str, default ()
        The channels the paper chose for ripple detection, where it states
        them; empty otherwise. ``fetch(rec.lfp.ripple_channels)`` reads them.
    cache : path or None
        The cache the session was loaded into; None for the default.

    Raises
    ------
    InputError
        If the files, rate, channels, unit or bases are inconsistent; the
        message names each problem.

    Examples
    --------
    >>> from neurospatial.datasets import load
    >>> rec = load("forli_2025")  # doctest: +SKIP
    >>> samples, times = rec.lfp.fetch(rec.lfp.ripple_channels)  # doctest: +SKIP
    """

    dataset: str
    files: tuple[LFPFile, ...]
    sampling_rate: float
    channels: Mapping[str, str]
    unit: str
    unit_basis: Conversion
    times_basis: Conversion
    reader: LFPReader
    ripple_channels: tuple[str, ...] = ()
    cache: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        files, channels = tuple(self.files), dict(self.channels)
        ripple = tuple(self.ripple_channels)
        problems = []
        if not files:
            problems.append("files: empty; say what holds the LFP")
        if not (np.isfinite(self.sampling_rate) and self.sampling_rate > 0):
            problems.append(f"sampling_rate: {self.sampling_rate!r}, not positive")
        if not channels:
            problems.append("channels: empty; name each channel")
        if blank := [name for name, text in channels.items() if not str(text).strip()]:
            problems.append(f"channels: nothing says what {blank} are")
        if unknown := sorted(set(ripple) - set(channels)):
            problems.append(f"ripple_channels: {unknown} are not channels")
        held = {name for file in files for name in file.channels}
        if unknown := sorted(held - set(channels)):
            problems.append(f"files: hold {unknown}, which are not channels")
        if held and not all(file.channels for file in files):
            problems.append("files: name the channels of every file, or of none")
        if held and (missing := sorted(set(channels) - held)):
            problems.append(f"files: none holds channels {missing}")
        if not str(self.unit).strip():
            problems.append("unit: empty; say what unit fetch returns")
        problems.extend(
            f"{name}: not a Conversion(text, basis, evidence)"
            for name in ("unit_basis", "times_basis")
            if not isinstance(getattr(self, name), Conversion)
        )
        if problems:
            raise InputError(
                _format_error(
                    f"LFPSource of {self.dataset!r} cannot be built:\n- "
                    + "\n- ".join(problems),
                    why="the LFP is fetched, scaled and timed from these fields.",
                    fix="correct the loader so each field listed meets the "
                    "contract in the LFPSource docstring.",
                )
            )
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "channels", MappingProxyType(channels))
        object.__setattr__(self, "ripple_channels", ripple)

    def __reduce__(self) -> Any:
        """Pickle through the constructor; the channel mapping is read-only."""
        return reduce_through_constructor(self)

    @property
    def size_bytes(self) -> int:
        """The bytes of every file that holds the LFP."""
        return sum(file.size_bytes for file in self.files)

    @property
    def _needs_channels(self) -> bool:
        """Whether ``fetch()`` with no channels would read too much to default."""
        return not self.ripple_channels and len(self.channels) > _MAX_DEFAULT_CHANNELS

    def fetch(
        self,
        channels: Sequence[str] | None = None,
        *,
        cache: str | os.PathLike[str] | None = None,
        stream: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Download the LFP once, verified, and read it.

        A download not yet cached is announced first with a
        :class:`~neurospatial.datasets.DownloadWarning` giving its size.

        Parameters
        ----------
        channels : sequence of str, optional
            The channels to read, in the order of the returned columns.
            Default: ``ripple_channels`` where the dataset states them, else
            every channel, in the order of ``channels``, when the source has
            at most 32. A larger source with no stated ripple channel needs
            ``channels``, since its files can run to hundreds of gigabytes:
            ``tuple(rec.lfp.channels)`` reads them all. Only the files
            holding the channels are fetched.
        cache : path, optional
            The cache directory. Default: the one the session was loaded into
            (``$NEUROSPATIAL_DATA_DIR``, else ``~/.cache/neurospatial/datasets``).
        stream : bool, default False
            Read DANDI files by HTTP byte range instead of downloading them,
            after checking that the archive still publishes the pinned
            SHA-256. Nothing is cached.

        Returns
        -------
        samples : ndarray, shape (n_time, n_channels)
            The LFP in ``unit``.
        times : ndarray, shape (n_time,)
            Seconds on the recording's clock, strictly increasing; a gap
            between files or trials stays a gap.

        Raises
        ------
        InputError
            If ``channels`` names an unknown channel, is omitted for a source
            of more than 32 channels with no ripple channels stated, or
            ``stream=True`` for a file outside DANDI.
        ValueError
            If a downloaded file's SHA-256 differs from the pinned one, or the
            reader's arrays break the contract above.
        """
        if channels is None:
            if self._needs_channels:
                from neurospatial.datasets._load import _size

                sizes = sorted(file.size_bytes for file in self.files)
                raise InputError(
                    _format_error(
                        f"{self.dataset}: fetch() needs channels for this LFP.",
                        why=f"it holds {len(self.channels)} channels in "
                        f"{len(self.files)} file(s) of {_size(self.size_bytes)} "
                        f"(largest {_size(sizes[-1])}), and no ripple channels "
                        "are stated, so a default would download and read "
                        "all of them.",
                        fix="pick channels from rec.lfp.channels and call "
                        "rec.lfp.fetch(channels); tuple(rec.lfp.channels) "
                        "reads them all.",
                    )
                )
            chosen = self.ripple_channels or tuple(self.channels)
        else:
            chosen = tuple(channels)
        if unknown := [name for name in chosen if name not in self.channels]:
            raise InputError(
                _format_error(
                    f"{self.dataset}: the LFP has no channels {unknown}.",
                    why=f"it holds {list(self.channels)}.",
                    fix="pick from rec.lfp.channels; tuple(rec.lfp.channels) reads them all.",
                )
            )
        if not chosen:
            raise InputError(
                _format_error(
                    f"{self.dataset}: no LFP channels were asked for.",
                    why="this dataset's paper may state no ripple channel, which "
                    "leaves rec.lfp.ripple_channels empty.",
                    fix="pick from rec.lfp.channels; tuple(rec.lfp.channels) reads them all.",
                )
            )
        files = self._files_of(chosen)
        sources: tuple[str | Path, ...]
        if stream:
            sources = tuple(self._stream_url(file) for file in files)
        else:
            sources = self._download(
                files, Path(cache) if cache is not None else self.cache
            )
        samples, times = self.reader(sources, chosen)
        return self._checked(samples, times, len(chosen))

    def _files_of(self, channels: tuple[str, ...]) -> tuple[LFPFile, ...]:
        """The files that hold ``channels``, in the order of ``files``."""
        wanted = set(channels)
        return tuple(
            file
            for file in self.files
            if not file.channels or wanted & set(file.channels)
        )

    def _stream_url(self, file: LFPFile) -> str:
        """The byte-range URL of a DANDI file, its published digest checked."""
        from neurospatial.datasets._fetch import dandi_stream_url

        if file.asset_id is None:
            raise InputError(
                _format_error(
                    f"{file.path} cannot be streamed: it is not a DANDI asset.",
                    why="streaming checks the SHA-256 DANDI publishes for a file; "
                    "other archives publish none.",
                    fix="fetch it with stream=False, which downloads and verifies it.",
                )
            )
        return dandi_stream_url(file.asset_id, sha256=file.sha256)

    def _download(
        self, files: tuple[LFPFile, ...], cache: Path | None
    ) -> tuple[Path, ...]:
        """Each of ``files`` in the cache, downloaded once."""
        from neurospatial.datasets._fetch import cache_root, fetch, fetch_zip_member
        from neurospatial.datasets._load import _size

        folder = cache_root(cache) / self.dataset
        missing = [file for file in files if not (folder / file.path).exists()]
        if missing:
            total = _size(sum(file.size_bytes for file in missing))
            warnings.warn(
                f"Downloading {self.dataset} LFP: {total} in {len(missing)} "
                f"file(s) from {missing[0].download_url} into {folder}",
                DownloadWarning,
                stacklevel=3,  # _download <- fetch <- the user
            )
        paths = []
        for file in files:
            relative = Path(file.path)
            subdir = str(Path(self.dataset) / relative.parent)
            if file.member is not None:
                assert file.url is not None  # checked at construction
                path = fetch_zip_member(
                    file.url,
                    file.member,
                    sha256=file.sha256,
                    subdir=subdir,
                    cache=cache,
                )
            else:
                path = fetch(
                    file.download_url,
                    relative.name,
                    sha256=file.sha256,
                    subdir=subdir,
                    cache=cache,
                )
            paths.append(path)
        return tuple(paths)

    def _checked(
        self, samples: Any, times: Any, n_channels: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The reader's arrays as float64, refused if they break the contract."""
        data = np.asarray(samples, dtype=np.float64)
        clock = np.asarray(times, dtype=np.float64)
        steps = np.diff(clock)
        problems = []
        if data.ndim != 2 or data.shape[1] != n_channels:
            problems.append(f"samples: shape {data.shape}, not (n_time, {n_channels})")
        if clock.ndim != 1 or clock.size < 2 or len(data) != clock.size:
            problems.append(
                f"times: shape {clock.shape}, not one time per sample of "
                f"{len(data)}, at least two"
            )
        elif not np.all(np.isfinite(clock)) or not np.all(steps > 0):
            problems.append("times: not finite and strictly increasing")
        elif not np.isclose(
            np.median(steps) * self.sampling_rate, 1.0, rtol=RATE_TOLERANCE, atol=0.0
        ):
            problems.append(
                f"times: {1 / np.median(steps):.6g} Hz by the median step, against "
                f"the stated {self.sampling_rate:g} Hz"
            )
        if problems:
            raise ValueError(
                _format_error(
                    f"The {self.dataset} LFP reader returned arrays that break "
                    "its contract:\n- " + "\n- ".join(problems),
                    why="analyses line these samples up with the spikes by these "
                    "times, so a wrong rate or order gives plausible wrong results.",
                    fix="correct the dataset's reader; see LFPReader in "
                    "neurospatial.datasets._lfp.",
                )
            )
        return data, clock
