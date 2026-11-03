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
from _geometry import graph_environment
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


def env_kind(rec: Recording, cache: dict[str, Any]) -> str:
    """``"declared"`` when the loader declares the apparatus and the run uses it."""
    if "environment" in rec.geometry and not cache.get("inferred_env"):
        return "declared"
    return "inferred"


def _samples_env(rec: Recording, cache: dict[str, Any]) -> Any:
    from neurospatial import Environment

    if "samples_env" not in cache:
        cache["samples_env"] = Environment.from_samples(
            rec.positions,
            bin_size=rec.params["bin_size"],
            units=_units(rec),
            name=rec.dataset,
        )
    return cache["samples_env"]


def _env(rec: Recording, cache: dict[str, Any]) -> Any:
    """The analysis environment: the declared apparatus when the loader gives one.

    ``rec.geometry["environment"]`` is a zero-argument callable returning the
    environment built from the source's stated apparatus; otherwise, or when
    the run sets ``cache["inferred_env"]``, the environment is built from the
    samples.
    """
    if "env" not in cache:
        if env_kind(rec, cache) == "declared":
            cache["env"] = rec.geometry["environment"]()
        else:
            cache["env"] = _samples_env(rec, cache)
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
    env = _samples_env(rec, cache)
    extent = np.ptp(np.asarray(env.bin_centers), axis=0)
    return Outcome(
        "ok",
        {"n_bins": env.n_bins, "n_dims": env.n_dims, "max_extent": float(extent.max())},
    )


def env_declared(rec: Recording, cache: dict[str, Any]) -> Outcome:
    """The declared environment, and the share of tracked samples it cannot place."""
    if env_kind(rec, cache) != "declared":
        return Outcome("n/a", note="the loader declares no apparatus")
    env = _env(rec, cache)
    extent = np.ptp(np.asarray(env.bin_centers), axis=0)
    tracked = rec.positions[np.isfinite(rec.positions).all(axis=1)]
    outside = float(np.mean(np.asarray(env.bin_at(tracked)) < 0))
    return Outcome(
        "ok",
        {
            "n_bins": env.n_bins,
            "n_dims": env.n_dims,
            "max_extent": float(extent.max()),
            "frac_samples_outside": outside,
        },
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
REPLAY_ON_LINEARIZED_GAP = "G-038"
MIN_SHIFT_S = 20.0


def shifted_trains(rec: Recording, seed: int) -> list[np.ndarray]:
    """Each unit's spikes inside the tracking span, circularly shifted (seeded).

    The shift is uniform between ``MIN_SHIFT_S`` and the span minus
    ``MIN_SHIFT_S``, one per unit, wrapping around; spikes outside the span
    stay put.
    """
    start, stop = float(rec.times[0]), float(rec.times[-1])
    span = stop - start
    rng = np.random.default_rng(seed)
    trains = []
    for train in rec.spike_times or ():
        inside = (train >= start) & (train <= stop)
        moved = train.copy()
        shift = rng.uniform(MIN_SHIFT_S, span - MIN_SHIFT_S)
        moved[inside] = (train[inside] - start + shift) % span + start
        trains.append(np.sort(moved))
    return trains


def information_vs_shifts(rec: Recording, cache: dict[str, Any]) -> Outcome:
    """Spatial information against circularly shifted spikes (seeded).

    Each unit's spikes inside the tracking span are shifted by a random
    amount between 20 s and the span minus 20 s, wrapping around, ``N_SHIFTS``
    times (seeds 0, 1, …); spikes outside the span stay put. A unit whose
    information does not beat its shifts carries no more than sampling noise.
    """
    from neurospatial.encoding import compute_spatial_rates

    real = np.asarray(_rates(rec, cache).spatial_information(), dtype=float)
    shifted_info = []
    for seed in range(N_SHIFTS):
        trains = shifted_trains(rec, seed)
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


def _on_track(rec: Recording, cache: dict[str, Any]) -> bool:
    """Whether the analysis environment is a track graph (positions along a track)."""
    return bool(_env(rec, cache).is_linearized_track)


def grid_scores(rec: Recording, cache: dict[str, Any]) -> Outcome:
    if _one_dimensional(rec):
        return Outcome("n/a", note="grid score needs 2-D positions")
    if _on_track(rec, cache):
        return Outcome("n/a", note="grid score needs a 2-D area, not a track graph")
    scores = _rates(rec, cache).grid_scores().scores
    return Outcome("ok", _finite_stats(scores))


def border_scores(rec: Recording, cache: dict[str, Any]) -> Outcome:
    if _one_dimensional(rec):
        return Outcome("n/a", note="border score needs 2-D positions")
    if _on_track(rec, cache):
        return Outcome("n/a", note="border score needs a 2-D area, not a track graph")
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


def chance_errors(
    env: Any, actual: np.ndarray, *, seed: int
) -> tuple[np.ndarray, float]:
    """Chance decoding errors: each true position against a shuffled other one.

    The true positions at the decoded times are permuted (seeded) within
    each connected part of the environment's bin graph, and each pair's
    distance is measured as ``error_against`` measures by default
    (geodesic, along the bin graph). A permutation across parts would put
    pairs an infinite path apart; their share under one global permutation
    is returned alongside.

    Parameters
    ----------
    env : Environment
    actual : ndarray, shape (n, n_dims)
        Finite true positions at the decoded times.
    seed : int

    Returns
    -------
    errors : ndarray, shape (n,)
        Geodesic distance of each within-part pair.
    frac_disconnected : float
        Share of pairs in different parts under a global permutation.
    """
    import networkx as nx

    from neurospatial.decoding.metrics import decoding_error

    rng = np.random.default_rng(seed)
    part_of_bin = np.full(env.n_bins, -1)
    for label, part in enumerate(nx.connected_components(env.connectivity)):
        part_of_bin[list(part)] = label
    bins = np.asarray(env.bin_at(actual))
    part = np.where(bins >= 0, part_of_bin[np.clip(bins, 0, None)], -1)
    frac_disconnected = float(np.mean(part != part[rng.permutation(len(part))]))
    shuffled = actual.copy()
    for label in np.unique(part):
        members = np.flatnonzero(part == label)
        shuffled[members] = actual[members[rng.permutation(len(members))]]
    errors = decoding_error(env, shuffled, actual, metric="geodesic")
    return np.asarray(errors, dtype=float), frac_disconnected


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
    actual = np.column_stack(
        [
            np.interp(result.times, rec.times, column)
            for column in np.asarray(rec.positions).T
        ]
    )
    chance, disconnected = chance_errors(
        result.env, actual[np.isfinite(actual).all(axis=1)], seed=0
    )
    return Outcome(
        "ok",
        {
            "median_error": float(np.nanmedian(error)),
            "chance_median_error": float(np.nanmedian(chance)),
            "chance_frac_disconnected": disconnected,
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

    Every edge is kept (``_geometry.graph_edge_order``): a depth-first walk
    from ``geometry["graph_root"]`` (or the graph's first node), then the
    edges that close cycles, which ``Environment.from_graph`` would otherwise
    drop without a word.
    ``geometry["track_width"]``, when the source states it, is the corridor
    width. Without ``geometry["graph"]``, a declared environment that is a
    track graph is the graph environment.
    """

    graph = rec.geometry.get("graph")
    if graph is None:
        # A declared track (a graph the loader fitted to a stated layout).
        env = _env(rec, cache) if env_kind(rec, cache) == "declared" else None
        return env if env is not None and env.is_linearized_track else None
    if "graph_env" not in cache:
        cache["graph_env"] = graph_environment(
            graph,
            bin_size=rec.params.get("graph_bin_size", rec.params["bin_size"]),
            track_width=rec.geometry.get("track_width"),
            units=_units(rec),
            name=f"{rec.dataset}-graph",
            root=rec.geometry.get("graph_root"),
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


def _lap_counts(found: Any, prefix: str) -> dict[str, int]:
    directions = list(found.intervals_table()["direction"]) if len(found.start) else []
    return {
        f"n_{prefix}": len(directions),
        **{f"n_{prefix}_{d}": directions.count(d) for d in sorted(set(directions))},
    }


def laps(rec: Recording, cache: dict[str, Any]) -> Outcome:
    """Laps with ``detect_laps``' defaults and, where the loader states one, by start region.

    ``geometry["lap_start_region"]`` is a ``Region`` drawn across the loop
    (where the source puts the start, or a stated cut); the second count is
    ``method="region"`` with it added to the environment's regions, inside
    ``params["analysis_epochs"]``. The default count keeps its earlier call
    (no epochs), so it stays comparable with recordings not rerun.
    """
    from neurospatial.behavior import detect_laps

    env = _graph_env(rec, cache) or _env(rec, cache)
    summary: dict[str, float | int | str] = {}
    region = rec.geometry.get("lap_start_region")
    if region is not None:
        with_start = env.with_regions({**env.regions, region.name: region})
        found = detect_laps(
            with_start,
            rec.times,
            rec.positions,
            method="region",
            start_region=region.name,
            epochs=rec.params.get("analysis_epochs"),
        )
        summary.update(_lap_counts(found, "laps_region"))
    found = detect_laps(env, rec.times, rec.positions)
    summary = {**_lap_counts(found, "laps"), **summary}
    return Outcome("ok", summary)


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
    # Ripples when the source has them, else candidate events the loader found
    # by a stated rule (population bursts).
    kind = "ripples" if "ripples" in rec.events else "candidate_events"
    ripples = rec.events.get(kind)
    if ripples is None or rec.params.get("replay_dt") is None:
        return Outcome(
            "n/a",
            note="needs events['ripples'] or events['candidate_events'], and params['replay_dt']",
        )
    dt = rec.params["replay_dt"]
    positions, status, gap, note = rec.positions, "ok", None, ""
    declared_track = env_kind(rec, cache) == "declared" and bool(
        _env(rec, cache).is_linearized_track
    )
    if declared_track:
        # The declared track graph gives the position along the track.
        env = _env(rec, cache)
        note = "decoded on the declared track graph"
    elif positions.shape[1] > 1 and "linearized" in rec.params:
        # Replay scores need a position along a track; a track recorded in 2-D
        # is decoded on the file's own linearized coordinate instead.
        from neurospatial import Environment

        positions = np.asarray(rec.params["linearized"], dtype=float).reshape(-1, 1)
        env = Environment.from_samples(
            positions,
            bin_size=rec.params["bin_size"],
            units=_units(rec),
            name=f"{rec.dataset}-linearized",
        )
        status, gap = "workaround", REPLAY_ON_LINEARIZED_GAP
        note = "decoded on params['linearized'], the file's track coordinate"
    else:
        env = _env(rec, cache)
    spikes = list(rec.spike_times or ())
    rates = compute_spatial_rates(
        env,
        spikes,
        rec.times,
        positions,
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
    summary: dict[str, float | int | str] = {"n_events": len(events), "events": kind}
    for row in detection_rates(table).itertuples():
        summary[f"{row.method}_rate"] = float(row.rate)
        summary[f"{row.method}_ci"] = f"{row.ci_low:.3f}-{row.ci_high:.3f}"
    return Outcome(status, summary, note=note, gap=gap)


_TO_CM = {"m": 100.0, "cm": 1.0, "mm": 0.1}
_ACCEPTED = ("ok", "absent")


def _reader_result(
    read: Callable[[], Any], compare: Callable[[Any], str], have: bool
) -> str:
    """One reader's result; a reader error other than "not found" propagates."""
    try:
        value = read()
    except KeyError:  # the readers' documented "not found"
        return (
            "not found, but the loader found it" if have else "absent (file has none)"
        )
    if not have:
        return "found data the loader did not use"
    return compare(value)


def _same_times(got: Any, expected: Any) -> bool:
    got, expected = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    return got.shape == expected.shape and np.allclose(got, expected)


def nwb_readers(rec: Recording, _cache: dict[str, Any]) -> Outcome:
    """Run the NWB readers with default arguments and compare with the loader.

    Each reader is ``ok`` when it returns the loader's range-checked arrays
    and times (after unit scaling), ``absent`` when neither finds the data,
    and otherwise a mismatch or a miss, which makes the cell a workaround; a
    reader that raises anything but "not found" makes it an error. When the
    loader itself built a field with that reader (``params["readers_used"]``),
    agreement is reported as "ok (loader used this reader)": it shows the
    reader is consistent, not that it read the file correctly.
    """
    from _fetch import open_nwb

    from neurospatial.io.nwb import read_head_direction, read_position, read_units
    from neurospatial.stats import wrap_angle

    source = rec.params.get("nwb_source")
    if source is None:
        return Outcome("n/a", note="params has no nwb_source (not an NWB dataset)")
    used = set(rec.params.get("readers_used", ()))

    def labelled(name: str, verdict: str) -> str:
        return (
            f"{verdict} (loader used this reader)"
            if verdict == "ok" and name in used
            else verdict
        )

    def position(value: Any) -> str:
        if rec.position_units == "pixel" and value.units == "px":
            scale = 1.0
        elif rec.position_units not in _TO_CM or value.units not in _TO_CM:
            return f"units {value.units!r} not comparable with {rec.position_units!r}"
        else:
            scale = _TO_CM[value.units] / _TO_CM[rec.position_units]
        scaled = np.asarray(value.positions) * scale
        if scaled.shape != np.shape(rec.positions):
            return (
                f"different series (shape {scaled.shape} vs {np.shape(rec.positions)})"
            )
        if not _same_times(value.times, rec.times):
            return "mismatch: times differ"
        same = np.allclose(scaled, rec.positions, equal_nan=True)
        return labelled("position", "ok" if same else "mismatch: values differ")

    def heading(value: Any) -> str:
        got = np.asarray(wrap_angle(np.asarray(value.headings, dtype=float)))
        if got.shape != np.shape(rec.headings):
            return f"different series (shape {got.shape})"
        if not _same_times(value.times, rec.times):
            return "mismatch: times differ"
        diff = np.abs(np.angle(np.exp(1j * (got - rec.headings))))
        worst = float(np.nanmax(diff))
        if worst < 1e-6:
            return labelled("head_direction", "ok")
        return (
            f"mismatch: up to {worst:.3f} rad; reader range "
            f"[{np.nanmin(got):.3f}, {np.nanmax(got):.3f}]"
        )

    def units(value: Any) -> str:
        trains = list(value.spike_times)
        if len(trains) != len(rec.spike_times or ()):
            return f"mismatch: {len(trains)} units vs {len(rec.spike_times or ())}"
        same = all(
            np.array_equal(np.asarray(a), b)
            for a, b in zip(trains, rec.spike_times or (), strict=True)
        )
        return labelled("units", "ok" if same else "mismatch: spike times differ")

    readers = {
        "position": (read_position, position, rec.positions is not None),
        "head_direction": (read_head_direction, heading, rec.headings is not None),
        "units": (read_units, units, rec.spike_times is not None),
    }
    found: dict[str, str] = {}
    errors: list[str] = []
    with open_nwb(source) as nwbfile:
        for name, (read, compare, have) in readers.items():
            try:
                found[name] = _reader_result(
                    lambda read=read: read(nwbfile), compare, have
                )
            except Exception as exc:  # noqa: PERF203 - three readers, each reported
                found[name] = f"raised {type(exc).__name__}"
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        return Outcome("error", found, note=" | ".join(errors))
    problems = [f"{k}: {v}" for k, v in found.items() if not v.startswith(_ACCEPTED)]
    return Outcome("workaround" if problems else "ok", found, note="; ".join(problems))


POSE_BINS_PER_SPAN = 50


def read_pose_operation(rec: Recording, _cache: dict[str, Any]) -> Outcome:
    """Read the file's pose keypoints and feed one to the rate or occupancy API.

    ``read_pose`` runs with default arguments. The summary lists the
    keypoints, the pose unit, and each keypoint's share of samples with any
    NaN coordinate. The keypoint ``params["pose_keypoint"]`` (else the first
    by name) is then passed, with the pose's own times, to
    ``Environment.from_samples`` and ``compute_spatial_rates`` (or
    ``env.occupancy`` for a recording without spikes). The bin size is the
    recording's when the pose shares its unit, else 1/50 of the keypoint's
    largest span. A reader error other than "not found" propagates.
    """
    from _fetch import open_nwb

    from neurospatial import Environment
    from neurospatial.encoding import compute_spatial_rates
    from neurospatial.io.nwb import read_pose

    source = rec.params.get("nwb_source")
    if source is None:
        return Outcome("n/a", note="params has no nwb_source (not an NWB dataset)")
    with open_nwb(source) as nwbfile:
        try:
            pose = read_pose(nwbfile)
        except KeyError:  # the reader's documented "not found"
            return Outcome("n/a", note="the file has no PoseEstimation")
    names = sorted(pose.bodyparts)
    summary: dict[str, float | int | str] = {
        "keypoints": ",".join(names),
        "units": str(pose.units),
        "n_samples": len(pose.times),
    }
    for name in names:
        xy = np.asarray(pose.bodyparts[name], dtype=float)
        summary[f"nan_share_{name}"] = float(np.isnan(xy).any(axis=1).mean())
    chosen = rec.params.get("pose_keypoint", names[0])
    xy = np.asarray(pose.bodyparts[chosen], dtype=float)
    times = np.asarray(pose.times, dtype=float)
    pose_units = (
        None if pose.units is None else {"px": "pixel"}.get(pose.units, pose.units)
    )
    if pose_units == rec.position_units:
        bin_size = rec.params["bin_size"]
    else:
        span = np.nanmax(xy, axis=0) - np.nanmin(xy, axis=0)
        bin_size = float(span.max()) / POSE_BINS_PER_SPAN
    env = Environment.from_samples(xy, bin_size=bin_size, name=f"{rec.dataset}-pose")
    if rec.spike_times is not None:
        compute_spatial_rates(env, list(rec.spike_times), times, xy)
        summary["fed"] = "compute_spatial_rates"
    else:
        env.occupancy(times, xy)
        summary["fed"] = "occupancy"
    summary["keypoint"] = chosen
    return Outcome("ok", summary)


def imaging_rates(_rec: Recording, _cache: dict[str, Any]) -> Outcome:
    return Outcome(
        "missing",
        note=(
            "no public spatial map from a continuous signal: compute_spatial_rates and"
            " compute_tuning_curve(s) take spike times"
        ),
        gap="G-029",
    )


DFF_ONSET_SD = 3.0
DFF_OFFSET_SD = 1.0


def _transient_onsets(trace: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Onset times of ΔF/F transients, one per excursion.

    A transient starts where the trace rises above median + 3 robust SD
    (1.4826 x MAD) and ends when it falls back below median + 1 SD, so noise
    that re-crosses the onset level within one transient counts once. The
    thresholds are a harness heuristic, not taken from a published method.
    """
    median = np.nanmedian(trace)
    sd = 1.4826 * np.nanmedian(np.abs(trace - median))
    onset, offset = median + DFF_ONSET_SD * sd, median + DFF_OFFSET_SD * sd
    starts = []
    inside = False
    for i, value in enumerate(trace):
        if not inside and value > onset:
            starts.append(i)
            inside = True
        elif inside and not value > offset:  # NaN also ends a transient
            inside = False
    return times[np.asarray(starts, dtype=int)]


def imaging_rates_workaround(rec: Recording, cache: dict[str, Any]) -> Outcome:
    """Spatial maps from imaging activity built in the harness, not by neurospatial.

    ``binned_rate``: occupancy-weighted mean of each cell's rate per spatial
    bin. Positions are interpolated at the frame times (NaN outside the
    tracking span), frames are assigned with ``env.bin_at``, frames with NaN
    position or activity are left out, and bins with no frame are NaN.
    ``dff``: one event per transient (see ``_transient_onsets``), passed to
    ``compute_spatial_rates`` as spike times. Both rules are harness
    heuristics, not published methods.
    """
    from neurospatial.encoding import compute_spatial_rates

    env = _env(rec, cache)
    activity = np.asarray(rec.activity, dtype=float)
    frame_times = np.asarray(rec.activity_times, dtype=float)
    if rec.activity_kind == "binned_rate":
        at_frames = np.column_stack(
            [
                np.interp(frame_times, rec.times, column, left=np.nan, right=np.nan)
                for column in np.asarray(rec.positions).T
            ]
        )
        bins = np.asarray(env.bin_at(at_frames))
        maps = np.full((activity.shape[1], env.n_bins), np.nan)
        for cell, values in enumerate(activity.T):
            keep = (bins >= 0) & np.isfinite(values)
            counts = np.bincount(bins[keep], minlength=env.n_bins).astype(float)
            sums = np.bincount(bins[keep], weights=values[keep], minlength=env.n_bins)
            maps[cell] = np.where(
                counts > 0, sums / np.where(counts > 0, counts, 1), np.nan
            )
        rule = "occupancy-weighted mean binned rate per spatial bin (harness)"
    elif rec.activity_kind == "dff":
        events = [_transient_onsets(cell, frame_times) for cell in activity.T]
        cache["imaging_events"] = events
        maps = np.asarray(
            compute_spatial_rates(env, events, rec.times, rec.positions).firing_rates
        )
        rule = (
            f"one event per dF/F transient: onset above median + {DFF_ONSET_SD:g} robust SD"
            f" (1.4826 x MAD), end below median + {DFF_OFFSET_SD:g} SD; then"
            " compute_spatial_rates (harness heuristic, no published source)"
        )
    else:
        return Outcome(
            "n/a", note=f"activity_kind {rec.activity_kind!r} has no workaround"
        )
    cache["imaging_maps"] = maps
    with np.errstate(all="ignore"):
        peaks = np.nanmax(maps, axis=1)
    return Outcome(
        "workaround",
        {
            "n_cells": maps.shape[0],
            "frac_nan_bins": float(np.isnan(maps).mean()),
            "median_peak": float(np.nanmedian(peaks)),
        },
        note=rule,
        gap="G-029",
    )


def egocentric_boundary(rec: Recording, cache: dict[str, Any]) -> Outcome:
    """Egocentric boundary maps: rate by distance and egocentric direction to walls.

    Uses ``compute_boundary_vector_rates`` with ``direction_frame="egocentric"``
    and its defaults (0-50 cm in 5 cm rings, 30 degree bins, unsmoothed). An
    environment built from samples declares no walls, so the walls are the
    edge of the visited bins (``boundary_source="analysis"``). The summary
    counts units whose peak is in the outermost ring, where the docstring
    says untuned cells peak by chance, and units whose peak is on the left
    (egocentric direction in (0, pi)).
    """
    from neurospatial.encoding import compute_boundary_vector_rates

    if rec.positions.shape[1] != 2:
        return Outcome("n/a", note="egocentric boundary maps need 2-D positions")
    if _on_track(rec, cache):
        return Outcome(
            "n/a", note="egocentric boundary maps need a 2-D area, not a track graph"
        )
    rates = compute_boundary_vector_rates(
        _env(rec, cache),
        list(rec.spike_times or ()),
        rec.times,
        rec.positions,
        rec.headings,
        direction_frame="egocentric",
        boundary_source="analysis",
        epochs=rec.params.get("analysis_epochs"),
    )
    cache["egocentric_boundary"] = rates
    table = rates.summary_table()
    distance = np.asarray(table["preferred_distance"], dtype=float)
    direction = np.asarray(table["preferred_direction"], dtype=float)
    low, high = rates.distance_range
    ring = (high - low) / rates.n_distance_bins
    peaked = np.isfinite(distance)
    return Outcome(
        "ok",
        {
            "n_units": len(table),
            "n_with_peak": int(peaked.sum()),
            "median_preferred_distance": float(np.median(distance[peaked]))
            if peaked.any()
            else float("nan"),
            "frac_peak_in_last_ring": float(np.mean(distance[peaked] > high - ring))
            if peaked.any()
            else float("nan"),
            "frac_peak_on_left": float(np.mean(np.sin(direction[peaked]) > 0))
            if peaked.any()
            else float("nan"),
            **{
                "distance_information_" + k: v
                for k, v in _finite_stats(table["distance_information"]).items()
            },
        },
        note="walls = edge of the visited bins (boundary_source='analysis')",
    )


_SPIKES_AND_POSITIONS = ("positions", "spike_times")

OPERATIONS: tuple[Operation, ...] = (
    Operation("env.from_samples", ("positions",), env_from_samples),
    Operation("env.occupancy", ("positions",), occupancy),
    Operation("env.declared", ("positions",), env_declared),
    Operation("rates.spatial", _SPIKES_AND_POSITIONS, rates_spatial),
    Operation("rates.spatial_information", _SPIKES_AND_POSITIONS, spatial_information),
    Operation(
        "rates.information_vs_shifts", _SPIKES_AND_POSITIONS, information_vs_shifts
    ),
    Operation("cells.place_screen", _SPIKES_AND_POSITIONS, place_screen),
    Operation("cells.grid_score", _SPIKES_AND_POSITIONS, grid_scores),
    Operation("cells.border_score", _SPIKES_AND_POSITIONS, border_scores),
    Operation("hd.tuning", ("headings", "spike_times"), hd_tuning),
    Operation(
        "rates.egocentric_boundary",
        ("positions", "headings", "spike_times"),
        egocentric_boundary,
    ),
    Operation("decode.position", _SPIKES_AND_POSITIONS, decode_position),
    Operation("io.nwb_roundtrip", _SPIKES_AND_POSITIONS, nwb_roundtrip),
    Operation("env.from_graph", ("positions",), env_from_graph),
    Operation("behavior.segment_by_velocity", ("positions",), segment_by_velocity),
    Operation("behavior.laps", ("positions",), laps),
    Operation("behavior.turn_sequence", ("positions",), turn_sequence),
    Operation("replay.compare_methods", _SPIKES_AND_POSITIONS, replay_compare_methods),
    Operation("io.nwb_readers", (), nwb_readers),
    Operation("io.read_pose", (), read_pose_operation),
    Operation("imaging.rates", ("activity", "positions"), imaging_rates),
    Operation(
        "imaging.rates_workaround", ("activity", "positions"), imaging_rates_workaround
    ),
)
