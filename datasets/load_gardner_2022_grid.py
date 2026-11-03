"""Gardner et al. 2022 rat MEC grid modules in an open field (Figshare 16764508).

What was recorded: rat R, day 1, Neuropixels in medial entorhinal cortex, with
the units the paper assigned to grid modules 1, 2 and 3, during a 2.4 h
recording that holds an open-field session and two wagon-wheel sessions.
Head position and azimuth (head direction) are tracked at 100 Hz.

Where it lives: Figshare 16764508 ("Toroidal topology of population activity
in grid cells"), CC0, one 320 MB zip; two members are read by HTTP byte range:
``rat_r_day1_grid_modules_1_2_3.npz`` (169 MB) and ``rat_r_day1_sessions.txt``.

Reading the arrays safely: ``x``, ``y``, ``z``, ``t`` and ``azimuth`` are plain
arrays. ``spikes_mod1..3`` are 0-d object arrays, which ``np.load`` only reads
with ``allow_pickle=True``, letting the file run code. Without executing them,
their pickles were inspected: they import only
``numpy.core.multiarray._reconstruct``, ``numpy.ndarray`` and ``numpy.dtype``
(dtypes ``O8`` and ``f8``), and hold a dict of unit index to spike times (s).
``read_npz`` lists each pickle's imports first (``pickle_imports``) and
refuses any outside that set, then unpickles with an unpickler that resolves
only those names and returns NumPy's built-in ``O8`` and ``f8`` dtypes.

What the loader converts:

- ``x``, ``y`` are metres (the open-field samples span about 1.5 m, the box
  size in the paper); converted to cm.
- Only the open-field session is kept: ``rat_r_day1_sessions.txt`` gives
  ``open_field_1`` from 7457 to 16045 s, valid 7457–14778 and 14890–16045 s;
  the valid ranges are the analysis epochs. Spikes are kept over the whole
  recording.
- ``azimuth`` is radians in [0, 2π); the loader checks it against the
  direction of movement during running (``heading_vs_movement``) and wraps it to
  [−π, π].

Analysis choices (``params``): ``bin_size`` 2.5 cm; ``dt`` 0.25 s and
alternating 60 s train/test blocks, inside the valid times, for decoding; each unit's grid module in
``params["module"]``.

Environment (``geometry["environment"]``): the open-field arena Gardner et
al. 2022 (Methods, "Open-field foraging task"; open-access PDF read) state,
"a square open-field (OF) arena of size 1.5 × 1.5 m" with 50 cm walls, built
with ``Environment.from_polygon`` at ``bin_size``. Centre and rotation are
fitted to the open-field samples (``_geometry.declared_rectangle``); the
residual of their own fitted sides against 150 cm is in ``conversions``.
"""

from __future__ import annotations

import io
import pickle
import pickletools
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from _fetch import fetch_zip_member
from _geometry import declared_rectangle, with_environment
from _record import (
    Range,
    Recording,
    alternating_blocks,
    check_expectations,
    checked_span,
    heading_vs_movement,
    intersect_intervals,
    validate_recording,
)

DATASET = {
    "key": "gardner_2022_grid",
    "title": "Gardner et al. (2022), rat R day 1, MEC grid modules 1-3, open field",
    "source": "https://figshare.com/articles/dataset/16764508",
    "license": "CC0",
    "download_size": "168.9 MB",
}

# Pre-registered from the paper (1.5 m box, three modules of roughly 100-150
# cells) and the file's array headers (100 Hz) before the first run of this loader.
EXPECTATIONS: dict[str, Any] = {
    "n_units": Range(250, 450),
    "n_dims": 2,
    "sampling_rate_hz": Range(99.0, 101.0),
    "position_extent_cm": lambda rec: Range(140, 160).contains(
        float(
            np.nanmax(np.ptp(rec.positions[np.isfinite(rec.positions).all(1)], axis=0))
        )
    ),
    "headings_span_the_circle": lambda rec: bool(
        np.nanmin(rec.headings) < -3.0 and np.nanmax(rec.headings) > 3.0
    ),
}

_ZIP_URL = "https://ndownloader.figshare.com/files/35078602"
_PREFIX = "Toroidal_topology_grid_cell_data/"
_ARRAYS = "rat_r_day1_grid_modules_1_2_3.npz"
_ARRAYS_SHA256 = "c7e5acbdcf1f7ec9c43dd1032b2a57cda2d69fdc7d0c904a237b2dceac65e5be"
_SESSIONS = "rat_r_day1_sessions.txt"
_SESSIONS_SHA256 = "a08d162130a9df628ea271f8764f651cee081cb9d68c484c7710da9fcd6979dd"
_SESSION = "open_field_1"
_RUNNING_CM_S = 10.0


def _reconstruct() -> Any:
    try:
        from numpy._core.multiarray import _reconstruct as reconstruct
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import _reconstruct as reconstruct
    return reconstruct


# NumPy pickles every array as ``_reconstruct(numpy.ndarray, (0,), b"b")``
# followed by a BUILD that sets shape, dtype and data. The stream may name
# ``numpy.ndarray`` only as that first argument: resolving it to the real class
# would let a REDUCE call ``ndarray(shape, "O", buffer=...)`` and read raw
# bytes as object pointers.
_NDARRAY = object()


def _safe_reconstruct(cls: Any, shape: Any, dtype_code: Any) -> np.ndarray:
    if cls is not _NDARRAY or shape != (0,) or dtype_code != b"b":
        msg = f"refusing _reconstruct({cls!r}, {shape!r}, {dtype_code!r})"
        raise pickle.UnpicklingError(msg)
    return _reconstruct()(np.ndarray, (0,), b"b")


# Built-in dtype objects, whose pickled state cannot be changed: a crafted
# stream cannot build a dtype whose flags read raw bytes as object pointers.
_DTYPES = {"O8": np.dtype("O8"), "f8": np.dtype("f8")}


def _dtype(spec: Any, align: bool = False, _copy: bool = False) -> np.dtype:
    if align or not isinstance(spec, str) or spec not in _DTYPES:
        msg = f"refusing dtype {spec!r} in a spike file (only O8 and f8 are expected)"
        raise pickle.UnpicklingError(msg)
    return _DTYPES[spec]


class _SpikeUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the globals the ``spikes_mod*`` arrays hold."""

    _ALLOWED: ClassVar[dict[tuple[str, str], Any]] = {
        ("numpy.core.multiarray", "_reconstruct"): _safe_reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _safe_reconstruct,
        ("numpy", "ndarray"): _NDARRAY,
        ("numpy", "dtype"): _dtype,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._ALLOWED[module, name]
        except KeyError:
            msg = f"refusing to load {module}.{name} from a spike file"
            raise pickle.UnpicklingError(msg) from None


def pickle_imports(blob: bytes) -> set[str]:
    """Every global a pickle would import, read from its opcodes without running it.

    Module and name strings are followed through the memo, as protocols 2-5
    write repeated strings once. ``"<unnamed>"`` marks a ``STACK_GLOBAL``
    whose names this reading cannot resolve; treat it as unexpected.
    """
    found: set[str] = set()
    memo: dict[int, str | None] = {}
    recent: list[str] = []
    top: str | None = None
    for op, arg, _ in pickletools.genops(blob):
        if op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"}:
            top = str(arg)
            recent = [*recent[-1:], top]
        elif op.name in {"BINPUT", "LONG_BINPUT", "PUT"}:
            memo[int(arg)] = top
        elif op.name == "MEMOIZE":
            memo[len(memo)] = top
        elif op.name in {"BINGET", "LONG_BINGET", "GET"}:
            top = memo.get(int(arg))
            recent = [*recent[-1:], top] if top is not None else []
        elif op.name == "GLOBAL":
            found.add(str(arg).replace(" ", "."))
            recent, top = [], None
        elif op.name == "STACK_GLOBAL":
            found.add(".".join(recent) if len(recent) == 2 else "<unnamed>")
            recent, top = [], None
        else:
            recent, top = [], None
    return found


def read_npz(path: Path) -> dict[str, Any]:
    """Read every array in an ``.npz``; object arrays through the restricted unpickler."""
    from numpy.lib import format as npy

    arrays: dict[str, Any] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            with archive.open(member) as stream:
                version = npy.read_magic(stream)
                read_header = (
                    npy.read_array_header_1_0
                    if version == (1, 0)
                    else npy.read_array_header_2_0
                )
                shape, fortran, dtype = read_header(stream)
                name = member.removesuffix(".npy")
                if dtype.hasobject:
                    blob = stream.read()
                    allowed = {".".join(key) for key in _SpikeUnpickler._ALLOWED}
                    if unexpected := pickle_imports(blob) - allowed:
                        msg = f"refusing {member}: its pickle imports {sorted(unexpected)}"
                        raise pickle.UnpicklingError(msg)
                    arrays[name] = _SpikeUnpickler(io.BytesIO(blob)).load()
                else:
                    count = int(np.prod(shape))
                    data = np.frombuffer(stream.read(count * dtype.itemsize), dtype)
                    arrays[name] = data.reshape(shape, order="F" if fortran else "C")
    return arrays


def _session_ranges(text: str, session: str) -> np.ndarray:
    """``valid_times`` (or start/end) of one session in a ``_sessions.txt`` file, s."""
    for line in text.splitlines():
        if not re.search(rf"\b{session}\b", line):
            continue
        valid = re.search(r"valid_times=\[([^\]]+)\]", line)
        if valid:
            pairs = [pair.split(",") for pair in valid.group(1).split(";")]
            return np.array([[float(a), float(b)] for a, b in pairs])
        start = re.search(r"start=(\d+)", line)
        end = re.search(r"end=(\d+)", line)
        if start and end:
            return np.array([[float(start.group(1)), float(end.group(1))]])
    msg = f"session {session!r} not listed with its times"
    raise ValueError(msg)


def load(cache: Path | None = None) -> Recording:
    """Fetch rat R day 1; open-field positions in cm, headings in radians, spikes."""
    from neurospatial.stats import wrap_angle

    arrays_path = fetch_zip_member(
        _ZIP_URL,
        _PREFIX + _ARRAYS,
        sha256=_ARRAYS_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    sessions_path = fetch_zip_member(
        _ZIP_URL,
        _PREFIX + _SESSIONS,
        sha256=_SESSIONS_SHA256,
        subdir=DATASET["key"],
        cache=cache,
    )
    arrays = read_npz(arrays_path)
    valid = _session_ranges(sessions_path.read_text(), _SESSION)
    t = np.asarray(arrays["t"], dtype=float)
    keep = (t >= valid[0, 0]) & (t <= valid[-1, 1])
    times = t[keep]
    positions = np.column_stack([arrays["x"], arrays["y"]]).astype(float)[keep] * 100.0
    azimuth = np.asarray(arrays["azimuth"], dtype=float)[keep]
    span = checked_span(positions, 140.0, 160.0)
    if azimuth.min() < 0.0 or azimuth.max() > 2 * np.pi + 1e-6:
        msg = "azimuth is no longer in [0, 2pi); recheck its unit"
        raise ValueError(msg)
    offset, spread = heading_vs_movement(
        times, positions, azimuth, min_speed=_RUNNING_CM_S
    )
    if abs(offset) > 0.2 or spread > np.radians(45):
        msg = (
            f"azimuth minus the direction of movement: mean {offset:.2f} rad, spread"
            f" {spread:.2f} rad; recheck its frame"
        )
        raise ValueError(msg)

    trains, unit_ids, modules = [], [], []
    for module in (1, 2, 3):
        spikes = arrays[f"spikes_mod{module}"][()]
        for index in sorted(spikes):
            trains.append(np.sort(np.asarray(spikes[index], dtype=float)))
            unit_ids.append(f"mod{module}_{index}")
            modules.append(module)
    blocks = alternating_blocks(times[0], times[-1], 60.0)
    train, test = (intersect_intervals(b, valid) for b in blocks)
    rec = Recording(
        dataset=DATASET["key"],
        session="rat_r_day1 open_field_1",
        source=f"{DATASET['source']} {_ARRAYS}",
        license=DATASET["license"],
        times=times,
        positions=positions,
        position_units="cm",
        headings=np.asarray(wrap_angle(azimuth)),
        spike_times=tuple(trains),
        unit_ids=np.asarray(unit_ids),
        epochs={"open_field": valid},
        geometry={"shape": "square", "size_cm": 150.0},
        params={
            "bin_size": 2.5,
            "dt": 0.25,
            "train_epochs": train,
            "test_epochs": test,
            "analysis_epochs": valid,
            "module": np.asarray(modules),
        },
        conversions=(
            "spikes: spikes_mod1..3 read with a restricted unpickler (numpy"
            " _reconstruct, ndarray and the O8/f8 dtypes only); dict of unit index"
            " to spike times in s",
            f"positions: x, y in m times 100 -> cm; {span};"
            f" kept {valid[0, 0]:.0f}-{valid[-1, 1]:.0f} s ({_SESSION})",
            "headings: azimuth in [0, 2pi) rad, wrapped to [-pi, pi]; minus the"
            f" direction of movement above {_RUNNING_CM_S:g} cm/s: circular mean"
            f" {np.degrees(offset):.1f} deg, median spread {np.degrees(spread):.1f} deg",
            f"analysis epochs: the session's valid times {valid.tolist()}",
        ),
    )
    return with_environment(
        rec,
        *declared_rectangle(
            rec.positions,
            150.0,
            150.0,
            bin_size=rec.params["bin_size"],
            units="cm",
            name=DATASET["key"],
            source="Gardner et al. 2022 Methods: 'square open-field (OF) arena of size 1.5 x 1.5 m', 50 cm walls",
        ),
        tracked_point="the head (the data's head position and azimuth at 100 Hz); the marker is not stated",
    )


def _headings_as_stored(rec: Recording) -> Recording:
    # The stored azimuth: radians in [0, 2pi) instead of [-pi, pi].
    assert rec.headings is not None
    return replace(rec, headings=np.mod(rec.headings, 2 * np.pi))


HAZARDS = {"headings_as_stored": _headings_as_stored}


if __name__ == "__main__":
    rec = load()
    validate_recording(rec)
    print(
        f"{rec.dataset}: {len(rec.spike_times or ())} units, {rec.times[-1] - rec.times[0]:.0f} s"
    )
    for row in check_expectations(rec, EXPECTATIONS):
        print(f"  {'pass' if row['passed'] else 'FAIL'} {row['key']}: {row['detail']}")
    for line in rec.conversions:
        print(f"  conversion: {line}")
