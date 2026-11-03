"""Check neurospatial's warnings in both directions.

Too aggressive: ``captured`` records every warning emitted while loading or
running an operation on correctly loaded data; each one is a candidate false
alarm. Not aggressive enough: ``probe_hazards`` re-runs a few operations with
one known defect put back into the recording (a hazard) and records whether
neurospatial raised, warned or stayed silent.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from _record import Recording
from operations import OPERATIONS

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")

Hazard = Callable[[Recording], Recording]


def template(message: str) -> str:
    """Group warnings that differ only in the numbers they report."""
    return _NUMBER.sub("#", message)


def _origin(filename: str) -> str:
    # neurospatial warns with a stacklevel that points at the caller (the
    # harness), or at its own modules; anything else raised from an installed
    # package is third-party.
    # A survey loader's own warning (``load_*.py``) is the loader's.
    parts = Path(filename).parts
    if "site-packages" in parts and "neurospatial" not in parts:
        return "third-party"
    if Path(filename).name.startswith("load_"):
        return "loader"
    return "neurospatial"


def captured(
    fn: Callable[..., Any], *args: Any
) -> tuple[Any, BaseException | None, list[dict[str, Any]]]:
    """Call ``fn(*args)``; return (value, exception or None, distinct warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value, error = fn(*args), None
        except Exception as exc:  # the driver records every failure
            value, error = None, exc
    keys = [
        (w.category.__name__, _origin(w.filename), template(str(w.message)))
        for w in caught
    ]
    example: dict[tuple[str, str, str], str] = {}
    for key, w in zip(keys, caught, strict=True):
        example.setdefault(key, str(w.message))
    rows = [
        {
            "category": category,
            "origin": origin,
            "template": text,
            "example": example[category, origin, text],
            "count": n,
        }
        for (category, origin, text), n in Counter(keys).items()
    ]
    return value, error, rows


def _degrees(rec: Recording) -> Recording:
    assert rec.headings is not None
    return replace(rec, headings=np.degrees(rec.headings))


def _missing_as_zero(rec: Recording) -> Recording:
    assert rec.positions is not None
    return replace(rec, positions=np.nan_to_num(rec.positions, nan=0.0))


def _spikes_in_ms(rec: Recording) -> Recording:
    assert rec.spike_times is not None
    return replace(rec, spike_times=tuple(train * 1e3 for train in rec.spike_times))


def _length_off_by_100(rec: Recording) -> Recording:
    # Metres labelled cm, or cm labelled metres. params["bin_size"] stays in
    # the labelled unit, as a user who missed the error would leave it.
    assert rec.positions is not None
    scale = 0.01 if rec.position_units == "cm" else 100.0
    return replace(rec, positions=rec.positions * scale)


# Applied to every recording they fit: (applies?, make hazardous recording).
GENERIC_HAZARDS: dict[str, tuple[Callable[[Recording], bool], Hazard]] = {
    "headings_in_degrees": (lambda r: r.headings is not None, _degrees),
    "missing_positions_as_zero": (
        lambda r: r.positions is not None and bool(np.isnan(r.positions).any()),
        _missing_as_zero,
    ),
    "spike_times_in_ms": (lambda r: r.spike_times is not None, _spikes_in_ms),
    "length_unit_off_by_100": (
        lambda r: r.positions is not None and r.position_units in ("cm", "m"),
        _length_off_by_100,
    ),
}
PROBE_OPS = (
    "env.from_samples",
    "env.occupancy",
    "rates.spatial",
    "hd.tuning",
    # Speed thresholds are where a time or length unit error shows up.
    "behavior.segment_by_velocity",
    "behavior.laps",
    # A mirrored frame swaps left and right turns.
    "behavior.turn_sequence",
)


def hazards_for(module: ModuleType, rec: Recording) -> dict[str, Hazard]:
    """Generic hazards that fit ``rec``, plus the loader's own ``HAZARDS``."""
    hazards = {
        name: make for name, (applies, make) in GENERIC_HAZARDS.items() if applies(rec)
    }
    hazards.update(getattr(module, "HAZARDS", {}))
    return hazards


# Fields every operation reads besides its ``requires``.
ALWAYS_READ = frozenset({"times", "position_units", "epochs", "params"})


def _warning_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(row["category"], row["template"]) for row in rows}


def _row(
    hazard: str, op_id: str, response: str, message: str, **extra: Any
) -> dict[str, Any]:
    return {
        "hazard": hazard,
        "op": op_id,
        "response": response,
        "message": message,
        "repeated_clean_warnings": extra.get("repeated", 0),
        "clean_summary": dict(extra.get("clean_summary", {})),
        "hazard_summary": extra.get("hazard_summary"),
    }


def probe_hazards(
    module: ModuleType,
    rec: Recording,
    clean_operations: Mapping[str, Mapping[str, Any]],
    operations: tuple[Any, ...] = OPERATIONS,
    probe_ops: tuple[str, ...] = PROBE_OPS,
    cache_seed: Mapping[str, Any] | None = None,
    *,
    report_unreached: bool = True,
) -> list[dict[str, Any]]:
    """Re-run ``probe_ops`` on each hazardous version of ``rec``.

    An operation is probed when it reads a field the hazard changed. Its
    response is ``raised``, ``warned`` (a warning the clean run of the same
    operation did not emit), ``n/a`` (it found its input missing) or
    ``silent``. An operation that was not ``ok`` on the clean recording cannot
    respond to the hazard; it gets a ``not probed`` row saying so, as does a
    hazard that reaches no probed operation, so nothing vanishes from the table.
    ``cache_seed`` starts each probe's operation cache (the run's settings,
    such as ``inferred_env``). ``report_unreached=False`` leaves out the
    rows for hazards that reach none of ``probe_ops`` (a run probing a
    subset of the operations cannot tell).
    """
    rows = []
    for name, make in hazards_for(module, rec).items():
        bad: Recording = make(rec)  # not validated: the hazard is the point
        changed = {
            f.name
            for f in fields(Recording)
            if getattr(bad, f.name) is not getattr(rec, f.name)
        }
        cache: dict[str, Any] = dict(cache_seed or {})
        reached = False
        for op in operations:
            if (
                op.op_id not in probe_ops
                or not changed & (set(op.requires) | ALWAYS_READ)
                or any(getattr(bad, f) is None for f in op.requires)
            ):
                continue
            reached = True
            clean = clean_operations.get(op.op_id, {})
            if (status := clean.get("status", "ok")) != "ok":
                rows.append(
                    _row(name, op.op_id, "not probed", f"the clean run was {status}")
                )
                continue
            outcome, error, caught = captured(op.run, bad, cache)
            new = [
                row
                for row in caught
                if (row["category"], row["template"])
                not in _warning_keys(list(clean.get("warnings", [])))
            ]
            if error is not None:
                response, message = "raised", f"{type(error).__name__}: {error}"
            elif outcome is not None and outcome.status == "n/a":
                response, message = "n/a", outcome.note
            elif new:
                response, message = "warned", "\n".join(row["example"] for row in new)
            else:
                response, message = "silent", ""
            rows.append(
                _row(
                    name,
                    op.op_id,
                    response,
                    message,
                    repeated=len(caught) - len(new),
                    clean_summary=clean.get("summary", {}),
                    hazard_summary=None if outcome is None else dict(outcome.summary),
                )
            )
        if not reached and report_unreached:
            rows.append(
                _row(
                    name,
                    "",
                    "not probed",
                    f"changes {sorted(changed)}, which no probed operation reads",
                )
            )
    return rows
