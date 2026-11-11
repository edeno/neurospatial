"""Every detected spike's waveform features, for decoders that skip spike sorting.

A dataset that keeps each detected spike's waveform, sorted or not, gives the
recording a :class:`SpikeMarks`: per electrode, the time of every spike that
crossed the detection threshold, its peak amplitude on each channel (the
"marks" a clusterless decoder takes in place of unit identities), and the
cluster the authors' sorting gave it. The amplitudes are computed at load from
the waveforms; the waveforms themselves, many times larger, are read on
request with :meth:`SpikeMarks.waveforms` from the files already fetched.

The reader that returns an electrode's waveforms must be a module-level
function, or a :func:`functools.partial` of one, so a recording can be
pickled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import InputError, _format_error
from neurospatial._results import read_only_copy, reduce_through_constructor
from neurospatial.datasets._recording import Conversion

__all__ = ["SpikeMarks", "WaveformReader"]

#: Reads ``(folder, electrode)`` into waveforms of shape ``(n_spikes, n_channels,
#: n_samples)`` in the marks' ``unit``, one row per spike of ``times[electrode]``
#: in its order. ``folder`` is the directory holding the session's fetched files.
WaveformReader: TypeAlias = Callable[[Path, str], NDArray[np.float32]]

#: Cluster id of a spike whose electrode the authors did not sort in its trial.
UNSORTED = -1


@dataclass(frozen=True)
class SpikeMarks:
    """Every detected spike of each electrode: its time, amplitudes and cluster.

    ``rec.marks`` holds it for a dataset that keeps every detected spike's
    waveform, and is None otherwise. ``times`` and ``amplitudes`` are the
    spike times and waveform features, one array per electrode, that a
    clusterless decoder takes.

    Parameters
    ----------
    electrodes : tuple of str
        The electrodes, for example ``("T1", "T2")``.
    times : tuple of ndarray, each shape (n_spikes,)
        Seconds on the recording's clock, sorted, one array per electrode.
    amplitudes : tuple of ndarray, each shape (n_spikes, n_channels)
        Each spike's peak on each channel, in ``unit``.
    clusters : tuple of ndarray of int, each shape (n_spikes,)
        The cluster the authors' sorting gave each spike; cluster ``c`` of
        electrode ``e`` is the unit ``units["electrode"] == e`` and
        ``units["cluster"] == c`` of the recording. The sorting's own noise
        cluster keeps its id (0 for Tint); :data:`UNSORTED` (-1) marks a
        spike of a trial whose electrode was not sorted.
    unit : str
        The unit of ``amplitudes`` and of :meth:`waveforms`, such as ``"uV"``.
    basis : Conversion
        How the amplitudes were computed from the stored samples.
    reader : WaveformReader
        Reads one electrode's waveforms; see :data:`WaveformReader`.
    folder : path
        The directory holding the fetched files the reader reads.
    regions : Mapping[str, str], optional
        Each electrode's brain region.

    Raises
    ------
    InputError
        If the arrays disagree in length or shape, times are not sorted and
        finite, or ``basis`` is not a Conversion.
    """

    electrodes: tuple[str, ...]
    times: tuple[NDArray[np.float64], ...]
    amplitudes: tuple[NDArray[np.float32], ...]
    clusters: tuple[NDArray[np.int64], ...]
    unit: str
    basis: Conversion
    reader: WaveformReader
    folder: Path = field(repr=False)
    regions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        electrodes = tuple(self.electrodes)
        times = tuple(read_only_copy(t, np.float64) for t in self.times)
        amplitudes = tuple(read_only_copy(a, np.float32) for a in self.amplitudes)
        clusters = tuple(read_only_copy(c, np.int64) for c in self.clusters)
        problems = []
        if not electrodes or len(set(electrodes)) != len(electrodes):
            problems.append("electrodes: give each electrode once")
        if not len(times) == len(amplitudes) == len(clusters) == len(electrodes):
            problems.append(
                f"{len(electrodes)} electrodes, {len(times)} time arrays, "
                f"{len(amplitudes)} amplitude arrays and {len(clusters)} cluster "
                "arrays: give one of each per electrode"
            )
        for name, t, a, c in zip(electrodes, times, amplitudes, clusters, strict=False):
            if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
                problems.append(f"times[{name!r}]: not a sorted, finite 1-D array")
            if a.ndim != 2 or len(a) != len(t):
                problems.append(
                    f"amplitudes[{name!r}]: shape {a.shape}, not (n_spikes, n_channels)"
                )
            if c.shape != t.shape:
                problems.append(f"clusters[{name!r}]: shape {c.shape}, not {t.shape}")
        if unknown := sorted(set(self.regions) - set(electrodes)):
            problems.append(f"regions: {unknown} are not electrodes")
        if not str(self.unit).strip():
            problems.append("unit: empty")
        if not isinstance(getattr(self, "basis"), Conversion):  # noqa: B009
            problems.append("basis: not a Conversion(text, basis, evidence)")
        if problems:
            raise InputError(
                _format_error(
                    "SpikeMarks cannot be built:\n- " + "\n- ".join(problems),
                    why="a decoder pairs each spike's time with its amplitudes by "
                    "position in these arrays.",
                    fix="correct the loader so each field listed meets the "
                    "contract in the SpikeMarks docstring.",
                )
            )
        object.__setattr__(self, "electrodes", electrodes)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "amplitudes", amplitudes)
        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "folder", Path(self.folder))
        object.__setattr__(self, "regions", MappingProxyType(dict(self.regions)))

    def __reduce__(self) -> Any:
        """Pickle through the constructor; the region mapping is read-only."""
        return reduce_through_constructor(self)

    @property
    def n_spikes(self) -> int:
        """Every detected spike, over all electrodes."""
        return int(sum(t.size for t in self.times))

    def _index(self, electrode: str) -> int:
        if electrode not in self.electrodes:
            raise InputError(
                _format_error(
                    f"No electrode {electrode!r} in these marks.",
                    why=f"the marks hold {list(self.electrodes)}.",
                    fix="pick one from rec.marks.electrodes.",
                )
            )
        return self.electrodes.index(electrode)

    def select(
        self, electrode: str, intervals: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """One electrode's spike times and amplitudes, within ``intervals``.

        Parameters
        ----------
        electrode : str
        intervals : ndarray, shape (n, 2), optional
            Sorted, disjoint ``[start, stop)`` rows, seconds; every spike by
            default.

        Returns
        -------
        times : ndarray, shape (n_kept,)
        amplitudes : ndarray, shape (n_kept, n_channels)
        """
        k = self._index(electrode)
        times, amplitudes = self.times[k], self.amplitudes[k]
        if intervals is None:
            return times, amplitudes
        rows = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
        row = np.searchsorted(rows[:, 0], times, side="right") - 1
        kept = row >= 0
        kept[kept] = times[kept] < rows[row[kept], 1]
        return times[kept], amplitudes[kept]

    def waveforms(self, electrode: str) -> NDArray[np.float32]:
        """Read one electrode's waveforms from the fetched files, in ``unit``.

        Parameters
        ----------
        electrode : str

        Returns
        -------
        ndarray of float32, shape (n_spikes, n_channels, n_samples)
            One row per spike of ``times[electrode]``, in its order.

        Raises
        ------
        InputError
            If ``electrode`` is unknown.
        ValueError
            If the reader returns another number of spikes or channels.
        """
        k = self._index(electrode)
        waves = np.asarray(self.reader(self.folder, electrode), dtype=np.float32)
        expected = (self.times[k].size, self.amplitudes[k].shape[1])
        if waves.ndim != 3 or waves.shape[:2] != expected:
            msg = (
                f"waveforms of {electrode}: shape {waves.shape}, not "
                f"({expected[0]}, {expected[1]}, n_samples)"
            )
            raise ValueError(msg)
        return waves
