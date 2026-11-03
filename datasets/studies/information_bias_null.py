"""Does ``spatial_information_bias`` predict the information of an untuned cell?

``SpatialRatesResult.spatial_information_bias()`` is the expected Skaggs
information of an untuned *Poisson* cell at the unit's mean rate on the
recording's occupancy. On the sparse 3-D bat occupancy, circularly shifted
spikes keep far more information (median 3.55 bits/spike) than that estimate
(median 0.37). Two explanations are separated here:

1. Sparse occupancy breaks the estimate's second-order expansion. Then
   simulated untuned Poisson cells on the bat's own occupancy show much more
   information than the estimate.
2. Temporal structure that circular shifts keep (bursts, firing locked to
   flights or runs) adds information that an untuned Poisson null does not
   have. Then the estimate matches simulated Poisson cells, simulated bursty
   cells show more, and real units' shift null sits above the estimate.

Steps (all seeds fixed; written to ``results/studies/information_bias_null.{md,json}``):

- Simulated untuned cells on each recording's own trajectory and analysis
  settings (``operations._rates``): ``N_SIM`` Poisson units and ``N_SIM``
  bursty units (burst onsets Poisson, a geometric number of spikes per burst
  with mean ``BURST_MEAN``, ``BURST_ISI`` apart) at each rate in ``RATES_HZ``.
  Reported: mean information, mean bias estimate, and their ratio.
- Real units: ``N_SHIFTS`` circular shifts each, by the rule of the
  ``rates.information_vs_shifts`` operation (seeds 0, 1, …). Reported per
  dataset: median of shift-null mean / bias estimate, and the share of units
  whose information minus bias exceeds ``CLASSIFY_CUTOFF`` while their
  information is below the 95th percentile of their shifts.
- Occupancy: share of visited bins with under 0.1 s, median seconds and
  median spikes per visited bin.

Run: ``uv run --group datasets --extra nwb python datasets/studies/information_bias_null.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _record import Recording  # noqa: E402
from operations import MIN_SHIFT_S, _env, _rates  # noqa: E402

RESULTS = HERE.parent / "results" / "studies"
DATASETS = (
    "yartsev_bat_demo",
    "sargolini_000582",
    "gardner_2022_grid",
    "grosmark_000044_circular",
)
RATES_HZ = (0.5, 2.0, 8.0)
N_SIM = 200
N_SHIFTS = 100
BURST_MEAN = 3.0
BURST_ISI = 0.005
CLASSIFY_CUTOFF = 0.1  # bits/spike above bias; the order of is_place_cell's default
SPARSE_S = 0.1


def _rates_for(rec: Recording, cache: dict[str, Any], trains: list[np.ndarray]) -> Any:
    from neurospatial.encoding import compute_spatial_rates

    return compute_spatial_rates(
        _env(rec, cache),
        trains,
        rec.times,
        rec.positions,
        min_speed=rec.params.get("min_speed"),
        epochs=rec.params.get("analysis_epochs"),
    )


def poisson_trains(
    times: np.ndarray, rate: float, n: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Untuned Poisson trains: each sample interval gets Poisson(rate * dt) spikes."""
    dt = np.diff(times, append=times[-1] + np.median(np.diff(times)))
    trains = []
    for _ in range(n):
        counts = rng.poisson(rate * dt)
        starts = np.repeat(times, counts)
        trains.append(
            np.sort(starts + rng.uniform(0, 1, starts.size) * np.repeat(dt, counts))
        )
    return trains


def bursty_trains(
    times: np.ndarray, rate: float, n: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Untuned bursty trains at the same mean rate: Poisson bursts of geometric size."""
    onsets = poisson_trains(times, rate / BURST_MEAN, n, rng)
    trains = []
    for onset in onsets:
        sizes = rng.geometric(1.0 / BURST_MEAN, onset.size)
        offsets = (
            np.concatenate([np.arange(k) * BURST_ISI for k in sizes])
            if sizes.size
            else np.empty(0)
        )
        trains.append(np.sort(np.repeat(onset, sizes) + offsets))
    return trains


def simulate(rec: Recording, cache: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for kind, make in (("poisson", poisson_trains), ("bursty", bursty_trains)):
        for k, rate in enumerate(RATES_HZ):
            rng = np.random.default_rng([k, 0 if kind == "poisson" else 1])
            result = _rates_for(rec, cache, make(rec.times, rate, N_SIM, rng))
            info = np.asarray(result.spatial_information(), dtype=float)
            bias = np.asarray(result.spatial_information_bias(), dtype=float)
            ok = np.isfinite(info) & np.isfinite(bias)
            if not ok.any():
                rows.append(
                    {
                        "kind": kind,
                        "rate_hz": rate,
                        "n_units": 0,
                        "mean_info": np.nan,
                        "mean_bias": np.nan,
                        "ratio": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "kind": kind,
                    "rate_hz": rate,
                    "n_units": int(ok.sum()),
                    "mean_info": float(info[ok].mean()),
                    "mean_bias": float(bias[ok].mean()),
                    "ratio": float(info[ok].mean() / bias[ok].mean()),
                }
            )
    return rows


def shifted(rec: Recording, seed: int) -> list[np.ndarray]:
    """The ``information_vs_shifts`` rule: shift spikes inside the tracking span."""
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


def shift_null(rec: Recording, cache: dict[str, Any]) -> dict[str, Any]:
    real = _rates(rec, cache)
    info = np.asarray(real.spatial_information(), dtype=float)
    bias = np.asarray(real.spatial_information_bias(), dtype=float)
    null = np.vstack(
        [
            np.asarray(
                _rates_for(rec, cache, shifted(rec, seed)).spatial_information(),
                dtype=float,
            )
            for seed in range(N_SHIFTS)
        ]
    )
    ok = (
        np.isfinite(info)
        & np.isfinite(bias)
        & np.isfinite(null).all(axis=0)
        & (bias > 0)
    )
    null_mean = null[:, ok].mean(axis=0)
    p95 = np.percentile(null[:, ok], 95, axis=0)
    passes_bias = info[ok] - bias[ok] > CLASSIFY_CUTOFF
    below_shifts = info[ok] < p95
    return {
        "n_units": int(ok.sum()),
        "median_info": float(np.median(info[ok])),
        "median_bias": float(np.median(bias[ok])),
        "median_null_mean": float(np.median(null_mean)),
        "median_null_over_bias": float(np.median(null_mean / bias[ok])),
        "share_pass_bias_fail_shifts": float(np.mean(passes_bias & below_shifts)),
        "share_pass_bias": float(np.mean(passes_bias)),
        "share_above_p95": float(np.mean(~below_shifts)),
    }


def occupancy_summary(rec: Recording, cache: dict[str, Any]) -> dict[str, Any]:
    result = _rates(rec, cache)
    occupancy = np.asarray(result.occupancy, dtype=float)
    visited = occupancy > 0
    # The default binned rates are unsmoothed: rate times occupancy is the count.
    counts = np.asarray(result.firing_rates, dtype=float) * occupancy
    return {
        "visited_bins": int(visited.sum()),
        "share_under_0p1_s": float(np.mean(occupancy[visited] < SPARSE_S)),
        "median_s_per_visited_bin": float(np.median(occupancy[visited])),
        "median_spikes_per_visited_bin": float(np.nanmedian(counts[:, visited])),
    }


def run(keys: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    import importlib

    from run import LOADERS

    out: dict[str, Any] = {}
    for key in keys:
        started = time.perf_counter()
        rec = importlib.import_module(LOADERS[key]).load()
        cache: dict[str, Any] = {}
        out[key] = {
            "occupancy": occupancy_summary(rec, cache),
            "simulated": simulate(rec, cache),
            "shift_null": shift_null(rec, cache),
        }
        out[key]["seconds"] = round(time.perf_counter() - started, 1)
        print(f"{key}: {out[key]['seconds']} s", flush=True)
    return out


def render(out: dict[str, Any]) -> str:
    lines = [
        "# Spatial-information bias against untuned and shifted nulls",
        "",
        "Generated by `datasets/studies/information_bias_null.py`; see its docstring"
        " for the method. Information in bits/spike.",
        "",
        "## Occupancy",
        "",
        "| Dataset | Visited bins | Share under 0.1 s | Median s per visited bin | Median spikes per visited bin |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, res in out.items():
        o = res["occupancy"]
        lines.append(
            f"| {key} | {o['visited_bins']} | {o['share_under_0p1_s']:.2f} |"
            f" {o['median_s_per_visited_bin']:.2f} | {o['median_spikes_per_visited_bin']:.2f} |"
        )
    lines += [
        "",
        "## Simulated untuned cells",
        "",
        f"{N_SIM} units per row; 'Units' counts those with a defined bias (it is NaN"
        " when a unit has fewer spikes than the map has independent bins). The"
        " ratio is mean information / mean bias estimate over those units.",
        "",
        "| Dataset | Kind | Rate (Hz) | Units | Mean information | Mean bias | Ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, res in out.items():
        for row in res["simulated"]:
            lines.append(
                f"| {key} | {row['kind']} | {row['rate_hz']:g} | {row['n_units']} |"
                f" {row['mean_info']:.3f} |"
                f" {row['mean_bias']:.3f} | {row['ratio']:.2f} |"
            )
    lines += [
        "",
        "## Real units against circular shifts",
        "",
        f"{N_SHIFTS} shifts per unit. 'Pass bias, fail shifts': information minus bias"
        f" above {CLASSIFY_CUTOFF} while information is below the shifts' 95th percentile.",
        "",
        "| Dataset | Units | Median information | Median bias | Median shift-null mean | Median null / bias | Pass bias | Above shift p95 | Pass bias, fail shifts |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, res in out.items():
        s = res["shift_null"]
        lines.append(
            f"| {key} | {s['n_units']} | {s['median_info']:.3f} | {s['median_bias']:.3f} |"
            f" {s['median_null_mean']:.3f} | {s['median_null_over_bias']:.2f} |"
            f" {s['share_pass_bias']:.2f} | {s['share_above_p95']:.2f} |"
            f" {s['share_pass_bias_fail_shifts']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1:] == ["--render-only"]:
        out = json.loads((RESULTS / "information_bias_null.json").read_text())
    else:
        out = run(tuple(sys.argv[1:]) or DATASETS)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "information_bias_null.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )
    (RESULTS / "information_bias_null.md").write_text(render(out))
