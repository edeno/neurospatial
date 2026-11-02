"""Compare neurospatial with independent tools on the same arrays.

Writes ``datasets/results/reference_checks.md``. Each check fixes both tools'
conventions (bins, range, smoothing), reports the numbers side by side and
lists the conventions that still differ. The thresholds below are triggers for
investigation, not pass/fail tests.

1. Spatial information on DANDI 000582 vs pynapple (in-process and the
   printed table of pynapple's null-distribution tutorial).
2. Grid score on the same binned rate maps vs opexebo.
3. Head-direction decoding on Mouse32 with pynapple's ``decode_bayes``; the
   neurospatial side is reported as missing if there is no public path.

Usage::

    MPLBACKEND=Agg uv run --group datasets --extra nwb --extra pynapple \\
        python datasets/reference_checks.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import load_peyrache_mouse32
import load_sargolini_000582
import numpy as np
from _record import Recording

RESULTS = Path(__file__).resolve().parent / "results"

# Printed in pynapple's tutorial_null_distributions (rendered 2026-11-02), unit 0-5:
# (bits/s, bits/spike).
PYNAPPLE_TUTORIAL_SI = np.array(
    [
        [2.797138, 1.389261],
        [2.408331, 2.670888],
        [2.412850, 3.133467],
        [3.301270, 2.673004],
        [1.965672, 2.290027],
        [1.832783, 2.013981],
    ]
)
RANGE_CM = (-50.0, 50.0)
BIN_CM = 2.5


def _relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def _binned_rates(rec: Recording, bandwidth: float | None = 0.0) -> tuple[Any, Any]:
    """Unsmoothed (``bandwidth=0``) binned rates on the reference grid.

    ``method="binned"`` smooths with a 5 cm bandwidth unless ``bandwidth`` is
    given, so the reference comparison passes ``bandwidth=0.0``.
    """
    from neurospatial import Environment
    from neurospatial.encoding import compute_spatial_rates

    env = Environment.from_samples(
        rec.positions,
        bin_size=BIN_CM,
        dimension_ranges=[RANGE_CM, RANGE_CM],
        infer_active_bins=False,
        units="cm",
    )
    rates = compute_spatial_rates(
        env,
        list(rec.spike_times or ()),
        rec.times,
        rec.positions,
        method="binned",
        bandwidth=bandwidth,
    )
    return env, rates


def _sample_histogram(rec: Recording, spikes: np.ndarray, rule: str) -> np.ndarray:
    """Spike counts per reference bin, assigning each spike to one tracking sample."""
    times, positions = rec.times, rec.positions
    after = np.clip(np.searchsorted(times, spikes, side="right"), 1, len(times) - 1)
    previous = after - 1
    if rule == "nearest":
        closer_after = np.abs(times[after] - spikes) < np.abs(times[previous] - spikes)
        index = np.where(closer_after, after, previous)
    else:
        index = previous
    sampled = positions[index]
    keep = np.isfinite(sampled).all(axis=1)
    edges = np.linspace(*RANGE_CM, 41)
    counts, _, _ = np.histogram2d(
        sampled[keep, 0], sampled[keep, 1], bins=[edges, edges]
    )
    return counts


def spatial_information_check(rec: Recording, env: Any, rates: Any) -> list[str]:
    import pynapple as nap

    from neurospatial.encoding import information_per_second

    firing = np.asarray(rates.firing_rates)
    occupancy = np.asarray(rates.occupancy)
    ns_spike = np.asarray(rates.spatial_information())
    ns_second = np.array(
        [information_per_second(row, occupancy=occupancy) for row in firing]
    )

    units = nap.TsGroup(
        {
            int(uid): nap.Ts(t)
            for uid, t in zip(rec.unit_ids, rec.spike_times or (), strict=True)
        }
    )
    position = nap.TsdFrame(t=rec.times, d=rec.positions, columns=["x", "y"])
    tuning = nap.compute_tuning_curves(
        data=units, features=position, bins=40, range=[RANGE_CM, RANGE_CM]
    )
    info = nap.compute_mutual_information(tuning)
    pn = info[["bits/sec", "bits/spike"]].to_numpy()
    ns = np.column_stack([ns_second, ns_spike])
    default_si = np.asarray(_binned_rates(rec, bandwidth=None)[1].spatial_information())

    lines = [
        "## 1. Spatial information, DANDI 000582 sub-10884_ses-03080402",
        "",
        "Both tools: 40 × 40 bins of 2.5 cm over [−50, 50] cm, no smoothing, no"
        " occupancy threshold, all tracked samples. neurospatial:"
        " `Environment.from_samples(..., dimension_ranges=..., infer_active_bins=False)`,"
        " `compute_spatial_rates(..., method='binned', bandwidth=0.0)`, `spatial_information()` and"
        " `information_per_second`. pynapple: `compute_tuning_curves` +"
        " `compute_mutual_information`, run in-process on the same arrays.",
        "",
        "| unit | ns bits/s | pynapple bits/s | tutorial bits/s | ns bits/spike |"
        " pynapple bits/spike | tutorial bits/spike |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, uid in enumerate(rec.unit_ids):
        lines.append(
            f"| {uid} | {ns[i, 0]:.6f} | {pn[i, 0]:.6f} | {PYNAPPLE_TUTORIAL_SI[i, 0]:.6f} |"
            f" {ns[i, 1]:.6f} | {pn[i, 1]:.6f} | {PYNAPPLE_TUTORIAL_SI[i, 1]:.6f} |"
        )
    lines += [
        "",
        f"- Max relative difference, neurospatial vs pynapple in-process:"
        f" {np.nanmax(_relative(ns, pn)):.3g} (bits/s and bits/spike together).",
        f"- Max relative difference, pynapple in-process vs tutorial table:"
        f" {np.nanmax(_relative(pn, PYNAPPLE_TUTORIAL_SI)):.3g}.",
        f"- Bins with occupancy: neurospatial {int(np.sum(occupancy > 0))},"
        f" pynapple {int(np.isfinite(tuning.to_numpy()[0]).sum())}.",
        *_assignment_lines(rec, env, rates, units, position),
        f"- With the default `method='binned'` (5 cm smoothing, no `bandwidth=`),"
        f" neurospatial gives {default_si[0]:.3f} bits/spike for unit"
        f" {rec.unit_ids[0]} instead of {ns_spike[0]:.3f}; median relative difference"
        f" from pynapple over units {np.median(_relative(default_si, pn[:, 1])):.2f}.",
        "",
    ]
    return lines


def _assignment_lines(
    rec: Recording, env: Any, rates: Any, units: Any, position: Any
) -> list[str]:
    """Which spike-to-sample rule each tool's per-bin spike counts match."""
    import pynapple as nap

    pn_counts = nap.compute_tuning_curves(
        data=units,
        features=position,
        bins=40,
        range=[RANGE_CM, RANGE_CM],
        return_counts=True,
    ).to_numpy()
    ns_counts = np.stack(
        [
            np.nan_to_num(_grid(env, row * np.asarray(rates.occupancy)))
            for row in np.asarray(rates.firing_rates)
        ]
    )
    n_spikes = sum(len(spikes) for spikes in rec.spike_times or ())
    lines = []
    for tool, counts in (
        ("neurospatial", ns_counts),
        ("pynapple", np.nan_to_num(pn_counts)),
    ):
        moved = {
            rule: round(
                float(
                    sum(
                        np.abs(counts[i] - _sample_histogram(rec, spikes, rule)).sum()
                        for i, spikes in enumerate(rec.spike_times or ())
                    )
                )
                / 2
            )
            for rule in ("previous", "nearest")
        }
        lines.append(
            f"- {tool} per-bin spike counts vs a spike-to-sample rule (spikes placed in"
            f" a different bin, of {n_spikes}): sample at or before the spike"
            f" {moved['previous']}, nearest sample {moved['nearest']}."
        )
    return lines


def _grid(env: Any, values: np.ndarray) -> np.ndarray:
    centers = np.asarray(env.bin_centers)
    index = np.round((centers - RANGE_CM[0]) / BIN_CM - 0.5).astype(int)
    n = round((RANGE_CM[1] - RANGE_CM[0]) / BIN_CM)
    grid = np.full((n, n), np.nan)
    grid[index[:, 0], index[:, 1]] = values
    return grid


def grid_score_check(rec: Recording, env: Any, rates: Any) -> list[str]:
    from opexebo.analysis import autocorrelation
    from opexebo.analysis import grid_score as opexebo_grid_score

    from neurospatial.encoding import grid_score, spatial_autocorrelation

    firing = np.asarray(rates.firing_rates)
    ns_acorr = [spatial_autocorrelation(env, row) for row in firing]
    ox_acorr = [autocorrelation(_grid(env, row)) for row in firing]
    ns_scores = np.array([grid_score(a) for a in ns_acorr])
    ns_batch = np.asarray(rates.grid_scores().scores, dtype=float)
    ox_scores = np.array([opexebo_grid_score(a)[0] for a in ox_acorr])
    # neurospatial's scorer on opexebo's autocorrelogram separates the two steps.
    ns_on_ox = np.array([grid_score(a) for a in ox_acorr])
    finite = np.isfinite(ns_scores) & np.isfinite(ox_scores)
    corr = (
        float(np.corrcoef(ns_scores[finite], ox_scores[finite])[0, 1])
        if finite.sum() > 2
        else float("nan")
    )
    lag_rows = []
    for a_ns, a_ox in zip(ns_acorr, ox_acorr, strict=True):
        trim = (a_ns.shape[0] - a_ox.shape[0]) // 2
        central = a_ns[trim : a_ns.shape[0] - trim, trim : a_ns.shape[1] - trim]
        defined = np.isfinite(central)
        lag_rows.append(
            (
                int((~defined).sum()),
                float(np.max(np.abs(central[defined] - a_ox[defined]))),
            )
        )
    lines = [
        "## 2. Grid score, same session and the same binned rate maps",
        "",
        "neurospatial: `grid_score(spatial_autocorrelation(env, rate))` and"
        " `rates.grid_scores()`; opexebo: `grid_score(autocorrelation(rate_map))`"
        " on the same 40 × 40 map (NaN for unvisited bins, no smoothing). The"
        " third column scores opexebo's autocorrelogram with neurospatial's"
        " `grid_score`, which separates the autocorrelogram step from the scoring step.",
        "",
        "| unit | neurospatial | neurospatial batch | ns score on opexebo autocorr | opexebo |"
        " difference | ns undefined lags in opexebo's extent | max acorr diff at shared lags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, uid in enumerate(rec.unit_ids):
        lines.append(
            f"| {uid} | {ns_scores[i]:.3f} | {ns_batch[i]:.3f} | {ns_on_ox[i]:.3f} |"
            f" {ox_scores[i]:.3f} | {ns_scores[i] - ox_scores[i]:+.3f} |"
            f" {lag_rows[i][0]} of {ox_acorr[i].size} | {lag_rows[i][1]:.3f} |"
        )
    lines += [
        "",
        f"- Pearson r over units with both scores: {corr:.3f}; max |difference|:"
        f" {np.nanmax(np.abs(ns_scores - ox_scores)):.3f} (investigate above 0.1).",
        f"- Scoring step alone (ns score on opexebo autocorr vs opexebo): max"
        f" |difference| {np.nanmax(np.abs(ns_on_ox - ox_scores)):.3f}.",
        "",
    ]
    return lines


def _circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


def head_direction_check(rec: Recording) -> list[str]:
    import pynapple as nap
    import pynwb

    with pynwb.NWBHDF5IO(rec.params["nwb_path"], "r", load_namespaces=True) as io:
        locations = io.read().units["location"][:]
    adn = np.asarray(locations) == "adn"
    spikes = nap.TsGroup(
        {
            int(uid): nap.Ts(t)
            for uid, t, keep in zip(
                rec.unit_ids, rec.spike_times or (), adn, strict=True
            )
            if keep
        }
    )
    wake = nap.IntervalSet(start=rec.epochs["wake"][:, 0], end=rec.epochs["wake"][:, 1])
    # pynapple's tutorial bins head direction over [0, 2pi) as stored in the file.
    angle = nap.Tsd(t=rec.times, d=np.mod(rec.headings, 2 * np.pi))
    tuning = nap.compute_tuning_curves(
        data=spikes, features=angle, bins=61, epochs=wake, range=(0, 2 * np.pi)
    )
    decoded, _ = nap.decode_bayes(
        tuning_curves=tuning, data=spikes, epochs=wake, bin_size=0.1
    )
    actual = np.interp(decoded.t, rec.times, np.unwrap(rec.headings))
    error = _circular_error(decoded.to_numpy(), actual)
    return [
        "## 3. Head-direction decoding, Mouse32-140822 wake",
        "",
        "pynapple: ADn units, 61 bins over [0, 2π), wake epoch,"
        " `decode_bayes(..., bin_size=0.1)` (the HD tutorial's settings).",
        "",
        f"- Units: {len(rec.unit_ids)} total, {int(adn.sum())} labelled `adn`"
        " (tutorial: 49 and 31).",
        f"- Decoded samples: {len(decoded)} (tutorial: 19590).",
        f"- pynapple median absolute circular error vs tracked heading:"
        f" {np.degrees(np.median(error)):.1f}° ({np.median(error):.3f} rad).",
        "- neurospatial: no public heading decoder (`neurospatial.decoding` decodes"
        " position on an `Environment`; no documented recipe in `docs/user-guide/`"
        " decodes head direction). Recorded as missing, see GAPS.md.",
        "",
    ]


def main() -> None:
    warnings.simplefilter("ignore", DeprecationWarning)
    sargolini = load_sargolini_000582.load()
    env, rates = _binned_rates(sargolini)
    lines = [
        "# Reference checks",
        "",
        "Generated by `datasets/reference_checks.py`; do not edit.",
        "",
        *spatial_information_check(sargolini, env, rates),
        *grid_score_check(sargolini, env, rates),
        *head_direction_check(load_peyrache_mouse32.load()),
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reference_checks.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
