"""The same session from two archives: CRCNS hc-11 against DANDI 000044.

Rat Achilles, 25 Oct 2013, is published as ``Achilles_10252013_sessInfo.mat``
in CRCNS hc-11 (``load_hc11_achilles``) and as NWB in DANDI 000044
(``load_grosmark_000044_linear``). This script compares what the two loaders
return:

- **Units**: the NWB unit ids are not the hc-11 cluster ids (shank * 100 +
  cluster), so units are paired by spike train: each NWB unit with the hc-11
  unit of the same spike count whose times differ least. A pair agrees when
  every spike time is within one sample (1/20000 s). Cell types are compared
  for the pairs (NWB ``excitatory``/``inhibitory`` against hc-11
  ``PyrIDs``/``IntIDs``).
- **Position**: sample times, 2-D positions in cm and the linearized
  coordinate in cm, sample by sample, with the NaN patterns.
- **Epochs**: PRE/MAZE/POST bounds, and the sleep-state tables (hc-11 scores
  five states, 000044 three).

Writes ``results/studies/hc11_vs_000044.{md,json}``. Run (streams the 8.7 GB
NWB file by byte range, about a minute):
``uv run --group datasets --extra nwb python datasets/studies/hc11_vs_000044.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import load_grosmark_000044_linear  # noqa: E402
import load_hc11_achilles  # noqa: E402
from _fetch import open_nwb  # noqa: E402

RESULTS = HERE.parent / "results" / "studies"
SAMPLE_S = 1 / 20000


def pair_units(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> list[dict]:
    """Pair each train of ``a`` with the closest equal-length unused train of ``b``."""
    used: set[int] = set()
    pairs = []
    for i, train in enumerate(a):
        best, best_dt = None, np.inf
        for j, other in enumerate(b):
            if j in used or other.size != train.size:
                continue
            dt = float(np.max(np.abs(other - train))) if train.size else 0.0
            if dt < best_dt:
                best, best_dt = j, dt
        if best is not None:
            used.add(best)
        pairs.append({"a": i, "b": best, "max_dt_s": best_dt, "n_spikes": train.size})
    return pairs


def interval_summary(table: np.ndarray) -> dict[str, float | int]:
    return {"n": len(table), "total_s": float(np.sum(table[:, 1] - table[:, 0]))}


def compare() -> dict[str, Any]:
    crcns = load_hc11_achilles.load()
    dandi = load_grosmark_000044_linear.load()
    with open_nwb(dandi.params["nwb_source"]) as nwbfile:
        units = nwbfile.units.to_dataframe()
    nwb_type = units["cell_type"].to_numpy()

    pairs = pair_units(dandi.spike_times, crcns.spike_times)
    paired = [p for p in pairs if p["b"] is not None]
    same_type = [
        {"excitatory": "pyramidal", "inhibitory": "interneuron"}.get(nwb_type[p["a"]])
        == crcns.params["cell_type"][p["b"]]
        for p in paired
    ]
    dt = np.array([p["max_dt_s"] for p in paired])
    order_kept = bool(
        len(paired) == len(pairs) and all(p["a"] == p["b"] for p in paired)
    )

    pos_a, pos_b = dandi.positions, crcns.positions
    same_shape = pos_a.shape == pos_b.shape
    both = np.isfinite(pos_a).all(1) & np.isfinite(pos_b).all(1) if same_shape else None
    lin_a, lin_b = dandi.params["linearized"], crcns.params["linearized"]

    def bounds(rec: Any, name: str) -> list[float]:
        return [float(x) for x in rec.epochs[name][0]]

    return {
        "units": {
            "n_000044": len(dandi.spike_times),
            "n_hc11": len(crcns.spike_times),
            "n_paired_same_count": len(paired),
            "n_within_one_sample": int(np.sum(dt <= SAMPLE_S)),
            "max_dt_s": float(dt.max()) if dt.size else float("nan"),
            "n_same_cell_type": int(np.sum(same_type)),
            "same_order": order_kept,
            "total_spikes_000044": int(sum(t.size for t in dandi.spike_times)),
            "total_spikes_hc11": int(sum(t.size for t in crcns.spike_times)),
        },
        "position": {
            "n_samples_000044": len(dandi.times),
            "n_samples_hc11": len(crcns.times),
            "max_time_diff_s": float(np.max(np.abs(dandi.times - crcns.times)))
            if same_shape
            else None,
            "same_nan_pattern": bool(np.array_equal(np.isnan(pos_a), np.isnan(pos_b)))
            if same_shape
            else None,
            "max_xy_diff_cm": float(np.max(np.abs(pos_a[both] - pos_b[both])))
            if same_shape
            else None,
            "same_linearized_nan_pattern": bool(
                np.array_equal(np.isnan(lin_a), np.isnan(lin_b))
            ),
            "max_linearized_diff_cm": float(np.nanmax(np.abs(lin_a - lin_b))),
        },
        "epochs": {
            name: {"000044": bounds(dandi, name), "hc11": bounds(crcns, name)}
            for name in ("pre", "maze", "post")
        },
        "states": {
            "000044": {
                k: interval_summary(v)
                for k, v in dandi.epochs.items()
                if k not in ("pre", "maze", "post")
            },
            "hc11": {
                k: interval_summary(v)
                for k, v in crcns.epochs.items()
                if k not in ("pre", "maze", "post")
            },
        },
        "candidate_bursts": {
            "000044": len(dandi.events["all_bursts"]),
            "hc11": len(crcns.events["all_bursts"]),
        },
    }


def render(result: dict[str, Any]) -> str:
    u, p = result["units"], result["position"]
    lines = [
        "# CRCNS hc-11 against DANDI 000044: Achilles 10252013",
        "",
        "Generated by `datasets/studies/hc11_vs_000044.py`; see its docstring.",
        "",
        "| Quantity | 000044 (NWB) | hc-11 (sessInfo.mat) | Agreement |",
        "| --- | --- | --- | --- |",
        f"| units | {u['n_000044']} | {u['n_hc11']} | {u['n_paired_same_count']} paired"
        f" by equal spike count; {u['n_within_one_sample']} within one sample"
        f" (largest difference {u['max_dt_s']:.2g} s); same order: {u['same_order']} |",
        f"| spikes | {u['total_spikes_000044']} | {u['total_spikes_hc11']} | |",
        f"| cell type | excitatory/inhibitory | pyramidal/interneuron |"
        f" {u['n_same_cell_type']} of {u['n_paired_same_count']} pairs agree |",
        f"| position samples | {p['n_samples_000044']} | {p['n_samples_hc11']} |"
        f" times differ by at most {p['max_time_diff_s']} s |",
        f"| 2-D position (cm) | | | NaN pattern same: {p['same_nan_pattern']};"
        f" largest difference {p['max_xy_diff_cm']} cm |",
        f"| linearized (cm) | | | NaN pattern same:"
        f" {p['same_linearized_nan_pattern']}; largest difference"
        f" {p['max_linearized_diff_cm']} cm |",
    ]
    for name, row in result["epochs"].items():
        lines.append(
            f"| {name} epoch (s) | {row['000044'][0]:.4f}–{row['000044'][1]:.4f} |"
            f" {row['hc11'][0]:.4f}–{row['hc11'][1]:.4f} |"
            f" {'same' if np.allclose(row['000044'], row['hc11']) else 'differ'} |"
        )
    states = result["states"]
    lines.append(
        "| sleep states | "
        + ", ".join(
            f"{k} {v['n']} ({v['total_s']:.0f} s)" for k, v in states["000044"].items()
        )
        + " | "
        + ", ".join(
            f"{k} {v['n']} ({v['total_s']:.0f} s)" for k, v in states["hc11"].items()
        )
        + " | different scoring |"
    )
    b = result["candidate_bursts"]
    lines += [
        f"| POST non-REM population bursts | {b['000044']} | {b['hc11']} | same rule,"
        " different non-REM epochs |",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = compare()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hc11_vs_000044.json").write_text(json.dumps(result, indent=2) + "\n")
    (RESULTS / "hc11_vs_000044.md").write_text(render(result))
    print(render(result))
