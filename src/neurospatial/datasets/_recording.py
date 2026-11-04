"""A loaded dataset session: plain arrays, where they came from, what was converted."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import InputError, _format_error
from neurospatial._results import (
    ReadOnlyArraysMixin,
    read_only_copy,
    reduce_through_constructor,
)
from neurospatial.datasets._errors import REFUSING_CHECKS

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.figure import Figure

    from neurospatial import Environment, Intervals
    from neurospatial.animation import Animation
    from neurospatial.datasets._video import VideoSource

__all__ = ["Conversion", "Provenance", "Recording"]

POSITION_UNITS = ("cm", "m", "pixel", "arbitrary")
_UNIT_COLUMNS = ("brain_region", "label")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Conversion:
    """One change a loader made to the file's values, and why it is right.

    Parameters
    ----------
    text : str
        What was converted, for example ``"spike times: samples / 20000 Hz"``.
    basis : {"stated", "inferred"}
        ``"stated"`` when the source states the fact the conversion uses (a
        unit attribute, the paper's methods); ``"inferred"`` when the loader
        worked it out from the data.
    evidence : str, default ""
        What the conversion rests on: the source for a stated one, the
        measurement for an inferred one (for example "the file's head_speed
        matches |step| / dt of the stored positions"). Required when
        ``basis="inferred"``.

    Raises
    ------
    InputError
        If ``basis`` is not ``"stated"`` or ``"inferred"``, ``text`` is
        empty, or an inferred conversion has no evidence.

    Examples
    --------
    >>> from neurospatial.datasets import Conversion
    >>> Conversion(
    ...     "positions: pixels * 0.43 cm/pixel",
    ...     "inferred",
    ...     "the tracked span matches the stated 120 cm track",
    ... ).basis
    'inferred'
    """

    text: str
    basis: Literal["stated", "inferred"]
    evidence: str = ""

    def __post_init__(self) -> None:
        problems = []
        if not str(self.text).strip():
            problems.append("text is empty")
        if self.basis not in ("stated", "inferred"):
            problems.append(f"basis is {self.basis!r}, not 'stated' or 'inferred'")
        elif self.basis == "inferred" and not str(self.evidence).strip():
            problems.append("evidence is empty for an inferred conversion")
        if problems:
            raise InputError(
                _format_error(
                    f"Conversion {self.text!r} is incomplete: {'; '.join(problems)}.",
                    why="a conversion without its basis and evidence cannot be "
                    "checked, and a wrong one (ms read as s, pixels labelled cm) "
                    "gives plausible but wrong results.",
                    fix="say whether the source states the fact (basis='stated') "
                    "or the loader inferred it (basis='inferred'), and give the "
                    "evidence, for example evidence='the tracked span matches "
                    "the stated 120 cm track'.",
                )
            )


@dataclass(frozen=True)
class Provenance:
    """Where a session's file came from and when it was fetched.

    Parameters
    ----------
    archive : str
        The archive and its record, for example ``"DANDI 000044"`` or
        ``"Zenodo 23020757"``.
    url : str
        The exact file fetched. For a session read from members of a remote
        zip, the zip.
    sha256 : str
        The file's SHA-256 checksum, 64 lowercase hexadecimal digits, which
        the download was verified against. For a session read from zip
        members, the zip's (each member is verified against its own, pinned
        in the dataset's registry).
    size_bytes : int
        The file's size in bytes; the zip's, for a session read from its
        members.
    neurospatial_version : str
        The neurospatial version that loaded it.
    fetched_at : str
        When the file was fetched, as an ISO 8601 time.

    Raises
    ------
    InputError
        If ``sha256`` is not 64 lowercase hexadecimal digits or
        ``size_bytes`` is negative.
    """

    archive: str
    url: str
    sha256: str
    size_bytes: int
    neurospatial_version: str
    fetched_at: str

    def __post_init__(self) -> None:
        problems = []
        if not _SHA256.fullmatch(str(self.sha256)):
            problems.append(
                f"sha256 {self.sha256!r} is not 64 lowercase hexadecimal digits"
            )
        if self.size_bytes < 0:
            problems.append(f"size_bytes {self.size_bytes!r} is negative")
        if problems:
            raise InputError(
                _format_error(
                    f"Provenance for {self.url!r} is invalid: {'; '.join(problems)}.",
                    why="the checksum and size are what a later download is "
                    "verified against.",
                    fix="pass the file's hashlib.sha256(...).hexdigest() and its "
                    "size in bytes.",
                )
            )


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class Recording(ReadOnlyArraysMixin):
    """One session of a public dataset, as the arrays analyses take.

    Printing a recording shows its summary: who and what was recorded, the
    clock and sampling rate, how well the animal was tracked, the
    environment, the neurons, and every known pitfall of the session under
    "Watch out". The arrays go straight into neurospatial's functions, for
    example ``compute_spatial_rates(rec.env, rec.spike_times, rec.times,
    rec.positions)``.

    Attributes
    ----------
    dataset, session : str
        The dataset id (``<first author>_<year>``) and the archive's own
        session id.
    paper : str
        Short reference to the paper, for example ``"Comrie et al. (2026)"``.
        ``citation`` holds the full one.
    animal : str
        The animal, for example ``"rat j16"``.
    apparatus : str
        What the animal was in, for example ``"spatial bandit maze"``.
    brain_regions : tuple of str
        The recorded brain regions, as the source names them.
    license : str
        The data license, for example ``"CC-BY-4.0"``.
    citation : str
        The full citation to use for the data.
    provenance : Provenance
        Where the file came from: archive, URL, checksum, size, neurospatial
        version and fetch time.
    clock : str
        The clock the times are on, for example ``"the Unix clock"`` or
        ``"the acquisition system's clock, from 0 s at its start"``. Times
        are always seconds.
    times : ndarray, shape (n_time,)
        Tracking sample times in seconds, strictly increasing, at least two.
    spike_times : tuple of ndarray
        One sorted, finite array of spike times in seconds per unit, on the
        clock of ``times``.
    units : pandas.DataFrame
        One row per unit, in ``spike_times`` order, indexed by ``unit_id``
        (unique). Columns: ``brain_region`` and ``label`` (the authors'
        cell-type label), missing values where the file has none;
        ``n_spikes`` and ``fires_in_epochs`` (whether the unit spikes in the
        analysed epochs), both computed from ``spike_times`` and replacing
        any columns of those names given. Reading returns a copy.
    env : Environment or None
        The environment, or None when the dataset has no positions.
    env_kind : {"declared", "inferred"} or None
        Whether the environment is declared from the stated apparatus or
        inferred from the samples; None without an environment.
    env_source : str
        The environment's stated dimensions and their source, for example
        ``"the authors' track graph, 6.5 cm hallways (STAR Methods)"``.
    track_width_quantile : float or None
        Where the track's ``track_width`` was sized from these samples: the
        quantile of the tracked samples' distance from the track that its
        half-width is (0.99 when half the width is their 99th percentile), so
        about ``1 - track_width_quantile`` of them lie outside the
        environment by construction. None when the width is stated, floored
        by a stated width, or the environment is not a track.
    positions : ndarray, shape (n_time, n_dims), or None
        Tracked positions in ``position_units``, ``n_dims`` from 1 to 3, NaN
        where tracking was lost (never infinite); kept as the file stores
        them.
    position_units : {"cm", "m", "pixel", "arbitrary"} or None
        The length unit of ``positions``; None without positions.
    tracking : str
        What was tracked, for example ``"head, LEDs on the headstage"``.
    headings : ndarray, shape (n_time,), or None
        Allocentric head direction in radians in [-π, π] (0 = East, +π/2 =
        North), NaN where missing (never infinite).
    heading_source : str
        Where ``headings`` comes from, for example ``"the file's head
        direction"``; empty without headings.
    epochs : Mapping[str, ndarray]
        Named epochs, each shape (n, 2): [start, stop) rows in seconds with
        start < stop.
    events : Mapping[str, ndarray]
        Named event times, shape (n,), or intervals, shape (n, 2), in
        seconds.
    trials : Intervals or None
        The trials table, where the dataset has trials.
    linearized : ndarray, shape (n_time,), or None
        The authors' linearized position, where the file has one.
    activity : ndarray, shape (n_frames, n_cells), or None
        Imaging activity, where the dataset is imaged.
    activity_times : ndarray, shape (n_frames,), or None
        Frame times of ``activity`` in seconds, strictly increasing.
    params : Mapping[str, Any]
        The analysis choices of the dataset's worked result, for example
        ``bin_size``, ``min_speed``, ``train_epochs`` and ``test_epochs``.
        ``analysis_epochs``, an (n, 2) array, sets the epochs
        ``units["fires_in_epochs"]`` refers to; without it, the tracked span
        from the first to the last sample of ``times``.
    conversions : tuple of Conversion
        Every conversion the loader applied, stated or inferred.
    watch_outs : tuple of str
        Every known pitfall of this session, with its numbers: the accepted
        check failures first, then the dataset's own, then what the
        automatic checks found.
    failed_checks : tuple of str
        The refusing checks (``"clock"``, ``"units"``) this session failed
        and the user accepted with ``accept_failed_checks``; empty otherwise.
    verified : bool
        Whether a person has checked this session by hand against its
        archive, its files and the sources its dataset's page lists.
    video : VideoSource or None
        The session's camera video, where the dataset publishes it: where it
        is, its frame times and its pixel-to-cm calibration, each with its
        basis. Nothing is downloaded until ``rec.video.fetch()`` or
        ``rec.animation(video=True)``. None without a video.

    Raises
    ------
    InputError
        If any field breaks the contract above; the message names every
        field that does.

    Notes
    -----
    The stated wells, ports, platforms and boxes are regions of the
    environment, ``rec.env.regions``; a recording has no regions of its own.

    NumPy arrays, including those in ``spike_times``, ``epochs``,
    ``events`` and at the top level of ``params``, are stored as read-only
    copies; copy with ``np.array(...)`` to modify values.
    """

    dataset: str
    session: str
    paper: str
    animal: str
    apparatus: str
    brain_regions: tuple[str, ...]
    license: str
    citation: str
    provenance: Provenance
    clock: str
    times: NDArray[np.float64]
    spike_times: tuple[NDArray[np.float64], ...]
    units: pd.DataFrame
    env: Environment | None = None
    env_kind: Literal["declared", "inferred"] | None = None
    env_source: str = ""
    track_width_quantile: float | None = None
    positions: NDArray[np.float64] | None = None
    position_units: Literal["cm", "m", "pixel", "arbitrary"] | None = None
    tracking: str = ""
    headings: NDArray[np.float64] | None = None
    heading_source: str = ""
    epochs: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    events: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    trials: Intervals[Any] | None = None
    linearized: NDArray[np.float64] | None = None
    activity: NDArray[np.float64] | None = None
    activity_times: NDArray[np.float64] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    conversions: tuple[Conversion, ...] = ()
    watch_outs: tuple[str, ...] = ()
    failed_checks: tuple[str, ...] = ()
    verified: bool = False
    video: VideoSource | None = None

    #: Physical unit of each ``summary()`` key; ``{units}`` is
    #: ``position_units``.
    _repr_units: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "duration": "s",
            "sampling_rate": "Hz",
            "longest_gap": "s",
            "median_speed": "{units}/s",
            "speed_p95": "{units}/s",
        }
    )

    def __post_init__(self) -> None:
        problems: list[str] = []
        times = _float_array(self.times, "times", problems)
        if times is not None and (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
        ):
            problems.append(_times_problem(times))
            times = None
        n_time = None if times is None else times.size
        object.__setattr__(self, "times", times)

        positions = _optional_float(self.positions, "positions", problems)
        if positions is not None and (
            positions.ndim != 2
            or positions.shape[0] != n_time
            or not 1 <= positions.shape[1] <= 3
        ):
            problems.append(
                f"positions: shape {positions.shape}, not (n_time, n_dims) with "
                f"n_time = {n_time} (the length of times) and n_dims 1 to 3"
            )
        elif positions is not None and (n_inf := int(np.isinf(positions).sum())):
            problems.append(
                f"positions: {n_inf} {'value is' if n_inf == 1 else 'values are'} "
                "infinite; mark a sample where tracking was lost with NaN"
            )
        object.__setattr__(self, "positions", positions)
        self._check_position_fields(problems)

        headings = _optional_float(self.headings, "headings", problems)
        if headings is not None:
            finite = headings[np.isfinite(headings)]
            if headings.shape != (n_time,):
                problems.append(
                    f"headings: shape {headings.shape}, not ({n_time},), one "
                    "per sample of times"
                )
            elif n_inf := int(np.isinf(headings).sum()):
                problems.append(
                    f"headings: {n_inf} {'value is' if n_inf == 1 else 'values are'} "
                    "infinite; mark a sample with no head direction with NaN"
                )
            elif finite.size and (finite.min() < -np.pi or finite.max() > np.pi):
                problems.append(
                    f"headings: values from {finite.min():.3g} to "
                    f"{finite.max():.3g}, not radians in [-π, π] (degrees, or "
                    "not wrapped)"
                )
            if not self.heading_source.strip():
                problems.append("heading_source: empty; say where headings come from")
        object.__setattr__(self, "headings", headings)

        linearized = _optional_float(self.linearized, "linearized", problems)
        if linearized is not None and linearized.shape != (n_time,):
            problems.append(
                f"linearized: shape {linearized.shape}, not ({n_time},), one per "
                "sample of times"
            )
        object.__setattr__(self, "linearized", linearized)

        spike_times = tuple(
            _float_array(train, f"spike_times[{index}]", problems)
            for index, train in enumerate(self.spike_times)
        )
        for index, train in enumerate(spike_times):
            if train is not None and (
                train.ndim != 1
                or not np.all(np.isfinite(train))
                or np.any(np.diff(train) < 0)
            ):
                problems.append(
                    f"spike_times[{index}]: not a 1-D, finite, sorted array of seconds"
                )
        object.__setattr__(self, "spike_times", spike_times)

        epochs = _interval_mapping(self.epochs, "epochs", problems)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "events", _event_mapping(self.events, problems))
        params = _owned_params(self.params, problems)
        object.__setattr__(self, "params", params)
        self._check_activity(problems)
        self._check_records(problems)
        self._check_unit_table(problems)
        if problems:
            raise InputError(
                _format_error(
                    f"Recording {self.dataset}/{self.session} cannot be built:\n- "
                    + "\n- ".join(problems),
                    why="analyses index every array by these fields, so a "
                    "mis-shaped, unsorted or mislabelled array gives wrong "
                    "numbers rather than an error.",
                    fix="correct the loader so each field listed meets the "
                    "contract in the Recording docstring.",
                )
            )
        self._count_unit_spikes()
        self._own_arrays()

    def _check_position_fields(self, problems: list[str]) -> None:
        """Positions, their unit, the environment and its kind agree."""
        has_positions = self.positions is not None
        if has_positions != (self.position_units is not None):
            problems.append(
                f"position_units: {self.position_units!r} with "
                f"{'positions' if has_positions else 'no positions'}; set it "
                "exactly when positions are given"
            )
        elif has_positions and self.position_units not in POSITION_UNITS:
            problems.append(
                f"position_units: {self.position_units!r}, not one of "
                f"{list(POSITION_UNITS)}"
            )
        if has_positions != (self.env is not None):
            problems.append(
                f"env: {'missing' if has_positions else 'given'} with "
                f"{'positions' if has_positions else 'no positions'}; a "
                "recording has an environment exactly when it has positions"
            )
        if (self.env is None) != (self.env_kind is None):
            problems.append(
                f"env_kind: {self.env_kind!r}; set it exactly when env is given"
            )
        elif self.env is not None and self.env_kind not in ("declared", "inferred"):
            problems.append(
                f"env_kind: {self.env_kind!r}, not 'declared' or 'inferred'"
            )
        if self.track_width_quantile is not None and not (
            self.env is not None
            and self.env.is_linearized_track
            and 0.0 < self.track_width_quantile < 1.0
        ):
            problems.append(
                f"track_width_quantile: {self.track_width_quantile!r}; give a "
                "quantile between 0 and 1, and only with a track environment"
            )
        if self.env is not None and not self.env_source.strip():
            problems.append(
                "env_source: empty; give the environment's dimensions and source"
            )
        if (has_positions or self.headings is not None) and not self.tracking.strip():
            problems.append("tracking: empty; say what was tracked")
        if self.env is None or self.positions is None or self.positions.ndim != 2:
            return
        from neurospatial._units import normalize_position_units

        env_units = getattr(self.env, "units", None)
        if (
            env_units
            and self.position_units
            and normalize_position_units(env_units)
            != normalize_position_units(self.position_units)
        ):
            problems.append(
                f"env: units {env_units!r}, but positions are in "
                f"{self.position_units!r}"
            )
        if self.env.n_dims != self.positions.shape[1]:
            problems.append(
                f"env: {self.env.n_dims}-D, but positions are "
                f"{self.positions.shape[1]}-D"
            )

    def _check_activity(self, problems: list[str]) -> None:
        """Imaging activity and its frame times are given together and agree."""
        activity = _optional_float(self.activity, "activity", problems)
        frames = _optional_float(self.activity_times, "activity_times", problems)
        if (activity is None) != (frames is None):
            problems.append("activity, activity_times: give both or neither")
        elif activity is not None and frames is not None:
            if frames.ndim != 1 or not np.all(np.diff(frames) > 0):
                problems.append("activity_times: not 1-D and strictly increasing")
            elif activity.ndim != 2 or activity.shape[0] != frames.size:
                problems.append(
                    f"activity: shape {activity.shape}, not (n_frames, n_cells) "
                    f"with n_frames = {frames.size}"
                )
        object.__setattr__(self, "activity", activity)
        object.__setattr__(self, "activity_times", frames)

    def _check_records(self, problems: list[str]) -> None:
        """The non-array fields: identity, trials, conversions and flags."""
        identity = ("dataset", "session", "paper", "animal", "apparatus", "license")
        problems.extend(
            f"{name}: empty"
            for name in (*identity, "citation", "clock")
            if not str(getattr(self, name)).strip()
        )
        object.__setattr__(self, "brain_regions", tuple(self.brain_regions))
        conversions: tuple[Any, ...] = tuple(self.conversions)
        problems.extend(
            f"conversions[{index}]: a {type(item).__name__}, not a "
            "Conversion(text, basis, evidence)"
            for index, item in enumerate(conversions)
            if not isinstance(item, Conversion)
        )
        object.__setattr__(self, "conversions", conversions)
        object.__setattr__(self, "watch_outs", tuple(self.watch_outs))
        failed = tuple(self.failed_checks)
        if unknown := sorted(set(failed) - set(REFUSING_CHECKS)):
            problems.append(
                f"failed_checks: {unknown} are not refusing checks "
                f"{list(REFUSING_CHECKS)}"
            )
        object.__setattr__(self, "failed_checks", failed)

    def _check_unit_table(self, problems: list[str]) -> None:
        """One row per spike train, indexed by unique ``unit_id``, with its columns."""
        table = vars(self)["_units"]
        if len(table) != len(self.spike_times):
            problems.append(
                f"units: {len(table)} rows for {len(self.spike_times)} spike trains"
            )
        if table.index.name != "unit_id":
            problems.append(f"units: index named {table.index.name!r}, not 'unit_id'")
        repeated = table.index[table.index.duplicated()].unique().tolist()
        if repeated:
            problems.append(f"units: unit ids {repeated} repeat; each must be unique")
        if missing := [name for name in _UNIT_COLUMNS if name not in table.columns]:
            problems.append(f"units: no {missing} column")

    def _count_unit_spikes(self) -> None:
        """Set the ``n_spikes`` and ``fires_in_epochs`` columns from ``spike_times``."""
        analysed = self._analysis_epochs
        table = vars(self)["_units"].copy()
        table["n_spikes"] = np.array([train.size for train in self.spike_times])
        table["fires_in_epochs"] = np.array(
            [_count_within(train, analysed) > 0 for train in self.spike_times],
            dtype=bool,
        )
        object.__setattr__(self, "units", table)

    @property
    def _analysis_epochs(self) -> NDArray[np.float64]:
        """The epochs ``units["fires_in_epochs"]`` refers to, shape (n, 2).

        ``params["analysis_epochs"]`` when given, else one row spanning the
        tracked samples, the last sample included.
        """
        epochs = self.params.get("analysis_epochs")
        if epochs is not None:
            return np.asarray(epochs, dtype=np.float64)
        stop = np.nextafter(self.times[-1], np.inf)
        return np.array([[self.times[0], stop]])

    @property
    def n_units(self) -> int:
        """The number of units, one per spike train."""
        return len(self.spike_times)

    @property
    def duration(self) -> float:
        """Seconds from the first to the last sample of ``times``.

        This includes any untracked epoch between tracked ones (a sleep
        between two runs).
        """
        return float(self.times[-1] - self.times[0])

    @property
    def n_dims(self) -> int:
        """The number of position dimensions; 0 without positions."""
        return 0 if self.positions is None else int(self.positions.shape[1])

    @property
    def sampling_rate(self) -> float:
        """The tracking sampling rate in Hz, one over the median interval."""
        return float(1.0 / np.median(np.diff(self.times)))

    @property
    def tracked_fraction(self) -> float:
        """The share of samples tracked: positions finite, or headings without positions."""
        return float(self._tracked.mean())

    @property
    def longest_gap(self) -> float:
        """The longest time between consecutive tracked samples, in seconds.

        Only gaps inside the epochs that hold tracking samples count, so an
        untracked sleep between two tracked runs is not a gap. Without such
        epochs, every gap from the first to the last sample counts. NaN
        with no gap to measure.
        """
        tracked_times = self.times[self._tracked]
        rows = tracked_epoch_rows(self.times, self.epochs)
        if len(rows) == 0:
            rows = np.array([[self.times[0], self.times[-1]]])
        before, after = tracked_times[:-1], tracked_times[1:]
        starts, stops = merged_rows(rows)
        index = np.searchsorted(starts, before, side="right") - 1
        inside = (index >= 0) & (after <= stops[np.maximum(index, 0)])
        if not np.any(inside):
            return float("nan")
        return float(np.max(after[inside] - before[inside]))

    @property
    def median_speed(self) -> float:
        """Median speed in ``position_units`` per second; NaN without positions.

        The speed is :func:`neurospatial.behavior.compute_speed` with its
        defaults (positions smoothed over 0.1 s, gaps over 1 s not bridged).
        """
        return self._speed_quantiles[0]

    @property
    def speed_p95(self) -> float:
        """95th percentile of speed, as for :attr:`median_speed`."""
        return self._speed_quantiles[1]

    @property
    def _tracked(self) -> NDArray[np.bool_]:
        """Whether each sample is tracked, shape (n_time,)."""
        if self.positions is not None:
            tracked: NDArray[np.bool_] = np.all(np.isfinite(self.positions), axis=1)
            return tracked
        if self.headings is not None:
            return np.isfinite(self.headings)
        return np.zeros(self.times.size, dtype=bool)

    @cached_property
    def _speed_quantiles(self) -> tuple[float, float]:
        """Median and 95th percentile of speed; NaN without a finite speed."""
        if self.positions is None:
            return float("nan"), float("nan")
        from neurospatial.behavior import compute_speed

        try:
            speed = compute_speed(self.times, self.positions)
        except InputError:  # no two tracked samples within 1 s of each other
            return float("nan"), float("nan")
        speed = speed[np.isfinite(speed)]
        if speed.size == 0:
            return float("nan"), float("nan")
        median, p95 = np.percentile(speed, [50, 95])
        return float(median), float(p95)

    def summary(self) -> dict[str, Any]:
        """Return the session's headline numbers, each a property of ``rec``.

        Returns
        -------
        dict
            ``n_units``, ``duration`` (s), ``n_dims``, ``sampling_rate``
            (Hz), ``tracked_fraction``, ``longest_gap`` (s),
            ``median_speed`` and ``speed_p95`` (``position_units`` per
            second) and ``verified``.
        """
        keys = (
            "n_units",
            "duration",
            "n_dims",
            "sampling_rate",
            "tracked_fraction",
            "longest_gap",
            "median_speed",
            "speed_p95",
            "verified",
        )
        return {key: getattr(self, key) for key in keys}

    def _repr_unit(self, key: str) -> str:
        """The unit of summary key ``key``; ``""`` for none or arbitrary units."""
        unit = self._repr_units.get(key, "")
        if "{units}" not in unit:
            return unit
        if self.position_units is None or self.position_units == "arbitrary":
            return ""
        return unit.replace("{units}", self.position_units)

    def __repr__(self) -> str:
        """The full summary text (see :func:`summary_text`)."""
        from neurospatial.datasets._summary import summary_text

        return summary_text(self)

    def plot_overview(self, *, fig: Figure | None = None) -> Figure:
        """Draw six panels that show the session at a glance.

        The environment with the path, occupancy (gray: never visited), the
        speed histogram with ``params["min_speed"]`` marked, a head-direction
        polar histogram beside heading minus direction of travel, units per
        brain region with spikes per unit, and the first 60 s of position
        with the spike raster. A panel the dataset has no data for says "not
        in this dataset". This is not ``plot``: the rules fix ``plot`` to one
        Axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            An empty figure to draw into; a new one when None.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from neurospatial.datasets._overview import plot_overview

        return plot_overview(self, fig)

    def animation(
        self,
        *,
        time_range: tuple[float, float] | None = None,
        trail_length: int | None = None,
        video: bool = False,
    ) -> Animation:
        """Build an interactive or movie scene of the session.

        Occupancy of the whole session over the environment, with the path,
        headings, spike raster, events and epochs of the window on top.
        Open it with ``.show(backend="fastplotlib")`` or write a movie with
        ``.export("session.mp4", time_range=(0, 10), fps=10)``. With
        ``video=True`` the session's camera frames are drawn below, through
        ``rec.video``'s calibration, so the tracked position can be checked
        against what the camera saw; occupancy is drawn at most half opaque,
        and see-through where it is low, so the frames show through every bin.

        Parameters
        ----------
        time_range : (float, float), optional
            Seconds [start, stop) on the recording's clock. Default: the
            first 60 s of tracking, so the first ``show()`` opens fast.
        trail_length : int, optional
            Length of the path trail behind the animal, in frames.
        video : bool, default False
            Draw the session's video below the data. The video is downloaded
            once, verified against its SHA-256, into the cache the session
            was loaded into; it needs the ``video`` extra
            (``pip install 'neurospatial[video]'``).

        Returns
        -------
        neurospatial.animation.Animation

        Raises
        ------
        InputError
            If the session has no positions (it then has ``headings`` and
            ``spike_times`` to use directly), or ``time_range`` holds fewer
            than two samples; with ``video=True``, if the session has no
            video or no calibration maps its pixels to the environment (the
            message names each calibration source tried).
        ImportError
            With ``video=True``, if the ``video`` extra (OpenCV) is not
            installed.
        """
        from neurospatial.datasets._overview import animation

        return animation(
            self, time_range=time_range, trail_length=trail_length, video=video
        )

    def _repr_html_(self) -> str:
        """The summary text, preformatted, for Jupyter."""
        import html

        return f"<pre>{html.escape(repr(self))}</pre>"

    def __reduce__(self) -> Any:
        """Pickle through the constructor; read-only mappings cannot pickle."""
        return reduce_through_constructor(self)


def _read_units(rec: Recording) -> pd.DataFrame:
    table: pd.DataFrame = vars(rec)["_units"].copy()
    return table


def _write_units(rec: Recording, table: pd.DataFrame) -> None:
    vars(rec)["_units"] = table


# The unit table is a DataFrame, which can be edited in place; reads return a
# copy so an edit cannot break its alignment with spike_times.
setattr(Recording, "units", property(_read_units, _write_units))  # noqa: B010


def _float_array(value: Any, name: str, problems: list[str]) -> Any:
    """``value`` as a float64 array; None, with a problem noted, if it is not numeric."""
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        problems.append(f"{name}: not numeric")
        return None


def _optional_float(value: Any, name: str, problems: list[str]) -> Any:
    """As :func:`_float_array`, keeping None."""
    return None if value is None else _float_array(value, name, problems)


def _times_problem(times: NDArray[np.float64]) -> str:
    """Why ``times`` is not a strictly increasing 1-D array of two or more seconds."""
    if times.ndim != 1:
        return f"times: shape {times.shape}, not (n_time,)"
    if times.size < 2:
        return f"times: {times.size} samples; at least two are needed"
    if not np.all(np.isfinite(times)):
        return f"times: {int(np.sum(~np.isfinite(times)))} non-finite samples"
    step = int(np.flatnonzero(np.diff(times) <= 0)[0])
    return (
        f"times: not strictly increasing (sample {step + 1} at "
        f"{times[step + 1]:.6g} s is not after sample {step} at {times[step]:.6g} s)"
    )


def _interval_rows(value: Any, name: str, problems: list[str]) -> Any:
    """``value`` as finite (n, 2) [start, stop) rows with start < stop, or None."""
    rows = _float_array(value, name, problems)
    if rows is None:
        return None
    if rows.size == 0:
        return np.empty((0, 2))
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.all(np.isfinite(rows)):
        problems.append(f"{name}: shape {rows.shape}, not finite (n, 2) rows")
        return None
    if np.any(rows[:, 0] >= rows[:, 1]):
        row = int(np.flatnonzero(rows[:, 0] >= rows[:, 1])[0])
        problems.append(
            f"{name}: row {row} starts at {rows[row, 0]:.6g} s, not before its "
            f"stop at {rows[row, 1]:.6g} s"
        )
        return None
    return rows


def _interval_mapping(
    mapping: Mapping[str, Any], name: str, problems: list[str]
) -> Mapping[str, NDArray[np.float64]]:
    """A read-only mapping of read-only (n, 2) interval tables."""
    owned = {}
    for key, value in mapping.items():
        rows = _interval_rows(value, f"{name}[{key!r}]", problems)
        if rows is not None:
            owned[str(key)] = read_only_copy(rows)
    return MappingProxyType(owned)


def _event_mapping(
    mapping: Mapping[str, Any], problems: list[str]
) -> Mapping[str, NDArray[np.float64]]:
    """A read-only mapping of finite event times (n,) or intervals (n, 2)."""
    owned = {}
    for key, value in mapping.items():
        name = f"events[{key!r}]"
        times = _float_array(value, name, problems)
        if times is None:
            continue
        if times.ndim == 2:
            times = _interval_rows(times, name, problems)
            if times is None:
                continue
        elif times.ndim != 1 or not np.all(np.isfinite(times)):
            problems.append(f"{name}: not finite times (n,) or intervals (n, 2)")
            continue
        owned[str(key)] = read_only_copy(times)
    return MappingProxyType(owned)


def _owned_params(mapping: Mapping[str, Any], problems: list[str]) -> Mapping[str, Any]:
    """A read-only mapping of the analysis choices, top-level arrays read-only."""
    owned = {
        str(key): read_only_copy(value) if isinstance(value, np.ndarray) else value
        for key, value in mapping.items()
    }
    for key in ("analysis_epochs", "train_epochs", "test_epochs"):
        if key in owned:
            rows = _interval_rows(owned[key], f"params[{key!r}]", problems)
            if rows is not None:
                owned[key] = read_only_copy(rows)
    return MappingProxyType(owned)


def _count_within(times: NDArray[np.float64], intervals: NDArray[np.float64]) -> int:
    """How many of the sorted ``times`` lie in the [start, stop) rows of ``intervals``."""
    starts = np.searchsorted(times, intervals[:, 0], side="left")
    stops = np.searchsorted(times, intervals[:, 1], side="left")
    return int(np.sum(stops - starts))


def tracked_epoch_rows(
    times: NDArray[np.float64], epochs: Mapping[str, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """The epoch rows, of every name, holding a tracking sample, shape (n, 2).

    A row holds tracking when a sample of ``times`` lies strictly inside it,
    so a sleep epoch that starts at the last tracking sample is untracked.
    """
    rows = all_epoch_rows(epochs)
    return rows[_samples_strictly_inside(times, rows) > 0]


def untracked_epoch_rows(
    times: NDArray[np.float64], epochs: Mapping[str, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """The epoch rows with no tracking sample strictly inside, shape (n, 2)."""
    rows = all_epoch_rows(epochs)
    untracked: NDArray[np.float64] = rows[_samples_strictly_inside(times, rows) == 0]
    return untracked


def all_epoch_rows(epochs: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Every epoch's rows, stacked, shape (n, 2)."""
    return np.concatenate(list(epochs.values())) if epochs else np.empty((0, 2))


def _samples_strictly_inside(
    times: NDArray[np.float64], rows: NDArray[np.float64]
) -> NDArray[np.intp]:
    """How many of the sorted ``times`` lie in each open (start, stop) row."""
    inside: NDArray[np.intp] = np.searchsorted(
        times, rows[:, 1], side="left"
    ) - np.searchsorted(times, rows[:, 0], side="right")
    return np.maximum(inside, 0)


def interval_rows(table: pd.DataFrame) -> NDArray[np.float64]:
    """The ``(n, 2)`` [start, stop) rows of an NWB interval table, sorted by start.

    Parameters
    ----------
    table : pandas.DataFrame
        An interval table with ``start_time`` and ``stop_time`` columns, in
        seconds.

    Returns
    -------
    ndarray, shape (n, 2)
    """
    rows: NDArray[np.float64] = table[["start_time", "stop_time"]].to_numpy(
        dtype=np.float64
    )
    ordered: NDArray[np.float64] = rows[np.argsort(rows[:, 0], kind="stable")]
    return ordered


def merged_rows(
    rows: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The union of ``[start, stop]`` rows as sorted, disjoint starts and stops."""
    if len(rows) == 0:
        return np.empty(0), np.empty(0)
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    reach = np.maximum.accumulate(rows[:, 1])
    first = np.flatnonzero(np.r_[True, rows[1:, 0] > reach[:-1]])
    return rows[first, 0], np.maximum.reduceat(rows[:, 1], first)
