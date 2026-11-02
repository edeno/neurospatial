"""The neurospatial operations run on every recording, and their outcomes.

Each operation calls the public API on a ``Recording`` and returns an
``Outcome``. ``ok`` means neurospatial did it with no harness-side adaptation;
``workaround`` and ``missing`` are decided by the operation itself and point to
a gap-log entry. Exceptions are not caught here: the driver records them as
``error``. The per-recording ``cache`` shares the environment and rate maps
across operations, so the registry lists the operations that build them first.
"""

from __future__ import annotations

import dataclasses
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


def _units(rec: Recording) -> str | None:
    # "arbitrary" means no physical unit, which neurospatial spells units=None.
    return None if rec.position_units == "arbitrary" else rec.position_units


def _env(rec: Recording, cache: dict[str, Any]) -> Any:
    from neurospatial import Environment

    if "env" not in cache:
        cache["env"] = Environment.from_samples(
            rec.positions,
            bin_size=rec.params["bin_size"],
            units=_units(rec),
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


N_SHIFTS = 4
MIN_SHIFT_S = 20.0


def information_vs_shifts(rec: Recording, cache: dict[str, Any]) -> Outcome:
    """Spatial information against circularly shifted spikes (seeded).

    Each unit's spikes inside the tracking span are shifted by a random
    amount between 20 s and the span minus 20 s, wrapping around, ``N_SHIFTS``
    times (seeds 0, 1, …); spikes outside the span stay put. A unit whose
    information does not beat its shifts carries no more than sampling noise.
    """
    from neurospatial.encoding import compute_spatial_rates

    real = np.asarray(_rates(rec, cache).spatial_information(), dtype=float)
    start, stop = float(rec.times[0]), float(rec.times[-1])
    span = stop - start
    shifted_info = []
    for seed in range(N_SHIFTS):
        rng = np.random.default_rng(seed)
        trains = []
        for train in rec.spike_times or ():
            inside = (train >= start) & (train <= stop)
            moved = train.copy()
            shift = rng.uniform(MIN_SHIFT_S, span - MIN_SHIFT_S)
            moved[inside] = (train[inside] - start + shift) % span + start
            trains.append(np.sort(moved))
        rates = compute_spatial_rates(
            _env(rec, cache),
            trains,
            rec.times,
            rec.positions,
            min_speed=rec.params.get("min_speed"),
            epochs=rec.params.get("analysis_epochs"),
        )
        shifted_info.append(np.asarray(rates.spatial_information(), dtype=float))
    shifted = np.vstack(shifted_info)
    finite = np.isfinite(real) & np.isfinite(shifted).all(axis=0)
    return Outcome(
        "ok",
        {
            "median_bits_per_spike": float(np.median(real[finite])),
            "median_shifted_bits_per_spike": float(np.median(shifted[:, finite])),
            "frac_above_every_shift": float(
                np.mean(real[finite] > shifted[:, finite].max(axis=0))
            ),
            "n_units": int(finite.sum()),
            "n_shifts": N_SHIFTS,
        },
    )


def place_screen(rec: Recording, cache: dict[str, Any]) -> Outcome:
    passed = np.asarray(_rates(rec, cache).classify())
    return Outcome("ok", {"n_place_screen": int(passed.sum()), "n_units": passed.size})


def _one_dimensional(rec: Recording) -> bool:
    return rec.positions is not None and rec.positions.shape[1] == 1


def grid_scores(rec: Recording, cache: dict[str, Any]) -> Outcome:
    if _one_dimensional(rec):
        return Outcome("n/a", note="grid score needs 2-D positions")
    scores = _rates(rec, cache).grid_scores().scores
    return Outcome("ok", _finite_stats(scores))


def border_scores(rec: Recording, cache: dict[str, Any]) -> Outcome:
    if _one_dimensional(rec):
        return Outcome("n/a", note="border score needs 2-D positions")
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


def _graph_env(rec: Recording, cache: dict[str, Any]) -> Any | None:
    """The track-graph environment, or None when the source gives no graph.

    Edges are ordered by a depth-first walk from ``geometry["graph_root"]``
    (or the graph's first node), which lays a tree out branch by branch.
    ``geometry["track_width"]``, when the source states it, is the corridor
    width.
    """
    import networkx as nx

    from neurospatial import Environment

    graph = rec.geometry.get("graph")
    if graph is None:
        return None
    if "graph_env" not in cache:
        root = rec.geometry.get("graph_root", next(iter(graph.nodes)))
        cache["graph_env"] = Environment.from_graph(
            graph,
            list(nx.dfs_edges(graph, source=root)),
            bin_size=rec.params.get("graph_bin_size", rec.params["bin_size"]),
            units=_units(rec),
            name=f"{rec.dataset}-graph",
            track_width=rec.geometry.get("track_width"),
        )
    return cache["graph_env"]


def env_from_graph(rec: Recording, cache: dict[str, Any]) -> Outcome:
    env = _graph_env(rec, cache)
    if env is None:
        return Outcome("n/a", note="geometry has no graph")
    return Outcome(
        "ok", {"n_bins": env.n_bins, "n_bin_edges": env.connectivity.number_of_edges()}
    )


def segment_by_velocity(rec: Recording, _cache: dict[str, Any]) -> Outcome:
    from neurospatial.behavior import segment_by_velocity as segment

    if rec.params.get("min_speed") is None:
        return Outcome("n/a", note="params has no min_speed")
    segments = segment(rec.times, rec.positions, min_speed=rec.params["min_speed"])
    windows = np.asarray(segments.windows)
    return Outcome(
        "ok",
        {
            "n_segments": len(windows),
            "running_s": float(np.sum(windows[:, 1] - windows[:, 0]))
            if len(windows)
            else 0.0,
        },
    )


def laps(rec: Recording, cache: dict[str, Any]) -> Outcome:
    from neurospatial.behavior import detect_laps

    env = _graph_env(rec, cache) or _env(rec, cache)
    found = detect_laps(env, rec.times, rec.positions)
    directions = list(found.intervals_table()["direction"]) if len(found.start) else []
    return Outcome(
        "ok",
        {
            "n_laps": len(directions),
            **{f"n_{d}": directions.count(d) for d in sorted(set(directions))},
        },
    )


def turn_sequence(rec: Recording, cache: dict[str, Any]) -> Outcome:
    from neurospatial.behavior import graph_turn_sequence

    env = _graph_env(rec, cache)
    if env is None:
        return Outcome("n/a", note="geometry has no graph")
    turns = [
        turn
        for turn in graph_turn_sequence(env, rec.times, rec.positions).split("-")
        if turn
    ]
    return Outcome(
        "ok",
        {
            "n_turns": len(turns),
            "frac_left": turns.count("left") / len(turns) if turns else float("nan"),
        },
    )


def replay_compare_methods(rec: Recording, cache: dict[str, Any]) -> Outcome:
    from neurospatial.encoding import compute_spatial_rates
    from neurospatial.replay import (
        compare_replay_methods,
        decode_intervals,
        detection_rates,
    )
    from neurospatial.replay.methods import DAVIDSON_2009, GROSMARK_2016

    # The rate maps and spike counts use every unit; GROSMARK_2016's active-cell
    # rule counts place cells in the paper, and both papers decoded each running
    # direction with its own maps. Both are simplifications here.
    ripples = rec.events.get("ripples")
    if ripples is None or rec.params.get("replay_dt") is None:
        return Outcome("n/a", note="needs events['ripples'] and params['replay_dt']")
    env, dt = _env(rec, cache), rec.params["replay_dt"]
    spikes = list(rec.spike_times or ())
    rates = compute_spatial_rates(
        env,
        spikes,
        rec.times,
        rec.positions,
        min_speed=rec.params.get("min_speed"),
        epochs=rec.params.get("train_epochs"),
    )
    events, counts = decode_intervals(env, spikes, ripples, rates, dt=dt)
    methods = [DAVIDSON_2009, GROSMARK_2016]
    if (n_shuffles := rec.params.get("replay_n_shuffles")) is not None:
        # Fewer shuffles than published, for quick checks; real runs omit it.
        methods = [dataclasses.replace(m, n_shuffles=n_shuffles) for m in methods]
    table = compare_replay_methods(
        env,
        events,
        replay_methods=methods,
        spike_counts=counts,
        spike_times=spikes,
        encoding_models=rates,
        dt=dt,
        rng=0,
    )
    summary: dict[str, float | int | str] = {"n_events": len(events)}
    for row in detection_rates(table).itertuples():
        summary[f"{row.method}_rate"] = float(row.rate)
        summary[f"{row.method}_ci"] = f"{row.ci_low:.3f}-{row.ci_high:.3f}"
    return Outcome("ok", summary)


_SPIKES_AND_POSITIONS = ("positions", "spike_times")

OPERATIONS: tuple[Operation, ...] = (
    Operation("env.from_samples", ("positions",), env_from_samples),
    Operation("env.occupancy", ("positions",), occupancy),
    Operation("rates.spatial", _SPIKES_AND_POSITIONS, rates_spatial),
    Operation("rates.spatial_information", _SPIKES_AND_POSITIONS, spatial_information),
    Operation(
        "rates.information_vs_shifts", _SPIKES_AND_POSITIONS, information_vs_shifts
    ),
    Operation("cells.place_screen", _SPIKES_AND_POSITIONS, place_screen),
    Operation("cells.grid_score", _SPIKES_AND_POSITIONS, grid_scores),
    Operation("cells.border_score", _SPIKES_AND_POSITIONS, border_scores),
    Operation("hd.tuning", ("headings", "spike_times"), hd_tuning),
    Operation("decode.position", _SPIKES_AND_POSITIONS, decode_position),
    Operation("io.nwb_roundtrip", _SPIKES_AND_POSITIONS, nwb_roundtrip),
    Operation("env.from_graph", ("positions",), env_from_graph),
    Operation("behavior.segment_by_velocity", ("positions",), segment_by_velocity),
    Operation("behavior.laps", ("positions",), laps),
    Operation("behavior.turn_sequence", ("positions",), turn_sequence),
    Operation("replay.compare_methods", _SPIKES_AND_POSITIONS, replay_compare_methods),
)
