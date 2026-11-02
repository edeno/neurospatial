"""The neurospatial operations run on every recording, and their outcomes.

Each operation calls the public API on a ``Recording`` and returns an
``Outcome``. ``ok`` means neurospatial did it with no harness-side adaptation;
``workaround`` and ``missing`` are decided by the operation itself and point to
a gap-log entry. Exceptions are not caught here: the driver records them as
``error``. The per-recording ``cache`` shares the environment and rate maps
across operations, so the registry lists the operations that build them first.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from _record import Recording

Status = Literal["ok", "workaround", "missing", "error", "n/a"]


@dataclass(frozen=True)
class Outcome:
    """What one operation did on one recording."""

    status: Status
    summary: Mapping[str, float | int | str] = field(default_factory=dict)
    note: str = ""
    gap: str | None = None


@dataclass(frozen=True)
class Operation:
    """A named operation and the ``Recording`` fields it needs."""

    op_id: str
    requires: tuple[str, ...]
    run: Callable[[Recording, dict[str, Any]], Outcome]


def _env(rec: Recording, cache: dict[str, Any]) -> Any:
    from neurospatial import Environment

    if "env" not in cache:
        cache["env"] = Environment.from_samples(
            rec.positions,
            bin_size=rec.params["bin_size"],
            units=rec.position_units,
            name=rec.dataset,
        )
    return cache["env"]


def _rates(rec: Recording, cache: dict[str, Any]) -> Any:
    from neurospatial.encoding import compute_spatial_rates

    if "rates" not in cache:
        cache["rates"] = compute_spatial_rates(
            _env(rec, cache),
            list(rec.spike_times or ()),
            rec.times,
            rec.positions,
            min_speed=rec.params.get("min_speed"),
            epochs=rec.params.get("analysis_epochs"),
        )
    return cache["rates"]


def _finite_stats(values: Any) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n_finite": 0}
    return {
        "median": float(np.median(finite)),
        "max": float(finite.max()),
        "n_finite": int(finite.size),
    }


def env_from_samples(rec: Recording, cache: dict[str, Any]) -> Outcome:
    env = _env(rec, cache)
    extent = np.ptp(np.asarray(env.bin_centers), axis=0)
    return Outcome(
        "ok",
        {"n_bins": env.n_bins, "n_dims": env.n_dims, "max_extent": float(extent.max())},
    )


def occupancy(rec: Recording, cache: dict[str, Any]) -> Outcome:
    occ = _env(rec, cache).occupancy(
        rec.times,
        rec.positions,
        min_speed=rec.params.get("min_speed"),
        epochs=rec.params.get("analysis_epochs"),
    )
    return Outcome(
        "ok",
        {
            "total_s": float(np.nansum(occ)),
            "frac_bins_visited": float(np.mean(occ > 0)),
            "median_s_per_visited_bin": float(np.median(occ[occ > 0]))
            if np.any(occ > 0)
            else 0.0,
            "max_s": float(np.nanmax(occ)),
        },
    )


def rates_spatial(rec: Recording, cache: dict[str, Any]) -> Outcome:
    rates = _rates(rec, cache)
    firing = np.asarray(rates.firing_rates)
    finite = np.where(np.isfinite(firing), firing, np.nan)
    with np.errstate(all="ignore"):
        peaks = np.nanmax(finite, axis=1) if finite.size else np.empty(0)
        means = np.nanmean(finite, axis=1) if finite.size else np.empty(0)
    return Outcome(
        "ok",
        {
            "n_units": firing.shape[0],
            "frac_nan_bins": float(np.isnan(firing).mean()),
            "median_peak_hz": float(np.nanmedian(peaks))
            if peaks.size
            else float("nan"),
            "median_mean_hz": float(np.nanmedian(means))
            if means.size
            else float("nan"),
        },
    )


def spatial_information(rec: Recording, cache: dict[str, Any]) -> Outcome:
    info = _rates(rec, cache).spatial_information()
    return Outcome(
        "ok", {"bits_per_spike_" + k: v for k, v in _finite_stats(info).items()}
    )


def place_screen(rec: Recording, cache: dict[str, Any]) -> Outcome:
    passed = np.asarray(_rates(rec, cache).classify())
    return Outcome("ok", {"n_place_screen": int(passed.sum()), "n_units": passed.size})


def grid_scores(rec: Recording, cache: dict[str, Any]) -> Outcome:
    scores = _rates(rec, cache).grid_scores().scores
    return Outcome("ok", _finite_stats(scores))


def border_scores(rec: Recording, cache: dict[str, Any]) -> Outcome:
    scores = _rates(rec, cache).border_scores().scores
    return Outcome("ok", _finite_stats(scores))


def hd_tuning(rec: Recording, cache: dict[str, Any]) -> Outcome:
    from neurospatial.encoding import compute_directional_rates

    tuning = compute_directional_rates(
        list(rec.spike_times or ()),
        rec.times,
        rec.headings,
        epochs=rec.params.get("analysis_epochs"),
    )
    cache["hd_tuning"] = tuning
    return Outcome(
        "ok",
        {
            "n_units": len(tuning.unit_ids),
            **{
                "mean_vector_length_" + k: v
                for k, v in _finite_stats(tuning.mean_vector_lengths()).items()
            },
        },
    )


def decode_position(rec: Recording, cache: dict[str, Any]) -> Outcome:
    from neurospatial.decoding import BayesianDecoder

    train, test = rec.params.get("train_epochs"), rec.params.get("test_epochs")
    if train is None or test is None:
        return Outcome("n/a", note="params has no train_epochs/test_epochs")
    spikes = list(rec.spike_times or ())
    decoder = BayesianDecoder(_env(rec, cache), dt=rec.params["dt"]).fit(
        spikes,
        rec.times,
        rec.positions,
        min_speed=rec.params.get("min_speed"),
        epochs=train,
    )
    result = decoder.predict(spikes, rec.times, epochs=test)
    error = result.error_against(rec.times, rec.positions)
    # Chance: distance between each decoded bin's true position and the true
    # position of a random other decoded bin (seeded permutation).
    actual = np.column_stack(
        [
            np.interp(result.times, rec.times, column)
            for column in np.asarray(rec.positions).T
        ]
    )
    actual = actual[np.isfinite(actual).all(axis=1)]
    shuffled = actual[np.random.default_rng(0).permutation(len(actual))]
    return Outcome(
        "ok",
        {
            "median_error": float(np.nanmedian(error)),
            "chance_median_error": float(
                np.median(np.linalg.norm(actual - shuffled, axis=1))
            ),
            "units": str(rec.position_units),
            "n_time_bins": int(result.n_time_bins),
        },
    )


def nwb_roundtrip(rec: Recording, cache: dict[str, Any]) -> Outcome:
    import pynwb

    from neurospatial.io.nwb import (
        read_environment,
        read_spatial_rates,
        write_environment,
        write_spatial_rates,
    )

    env, rates = _env(rec, cache), _rates(rec, cache)
    nwbfile = pynwb.NWBFile(
        session_description=rec.session,
        identifier=rec.dataset,
        session_start_time=dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc),
    )
    write_environment(nwbfile, env)
    write_spatial_rates(nwbfile, rates)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.nwb"
        with pynwb.NWBHDF5IO(path, "w") as io:
            io.write(nwbfile)
        with pynwb.NWBHDF5IO(path, "r") as io:
            back = io.read()
            env_back, rates_back = read_environment(back), read_spatial_rates(back)
            checks = {
                "firing_rates": np.allclose(
                    rates.firing_rates, rates_back.firing_rates, equal_nan=True
                ),
                "occupancy": np.allclose(
                    rates.occupancy, rates_back.occupancy, equal_nan=True
                ),
                "bin_centers": np.allclose(env.bin_centers, env_back.bin_centers),
            }
    failed = [name for name, same in checks.items() if not same]
    if failed:
        msg = f"values changed in the NWB round trip: {', '.join(failed)}"
        raise AssertionError(msg)
    return Outcome("ok", {"n_bins": env.n_bins})


_SPIKES_AND_POSITIONS = ("positions", "spike_times")

OPERATIONS: tuple[Operation, ...] = (
    Operation("env.from_samples", ("positions",), env_from_samples),
    Operation("env.occupancy", ("positions",), occupancy),
    Operation("rates.spatial", _SPIKES_AND_POSITIONS, rates_spatial),
    Operation("rates.spatial_information", _SPIKES_AND_POSITIONS, spatial_information),
    Operation("cells.place_screen", _SPIKES_AND_POSITIONS, place_screen),
    Operation("cells.grid_score", _SPIKES_AND_POSITIONS, grid_scores),
    Operation("cells.border_score", _SPIKES_AND_POSITIONS, border_scores),
    Operation("hd.tuning", ("headings", "spike_times"), hd_tuning),
    Operation("decode.position", _SPIKES_AND_POSITIONS, decode_position),
    Operation("io.nwb_roundtrip", _SPIKES_AND_POSITIONS, nwb_roundtrip),
)
