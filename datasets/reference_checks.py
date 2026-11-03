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

import load_gardner_2022_grid
import load_peyrache_mouse32
import load_sargolini_000582
import load_vollan_of
import numpy as np
from _record import Recording
from run import _versions

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
MAX_TRACKING_GAP_S = 0.2  # about 8 frames at Mouse32's 39 Hz tracking
BIN_CM = 2.5
POPULATION_BANDWIDTH_CM = 5.0
GRID_CUTOFF = 0.3
CONVENTIONS = (
    "neurospatial",
    "full lags, overlap 20 (a)",
    "full lags, overlap 1/4 (b)",
    "zero-filled (c)",
    "opexebo",
)


def _relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def _binned_rates(rec: Recording, bandwidth: float | None = None) -> tuple[Any, Any]:
    """Binned rates on the reference grid, unsmoothed unless ``bandwidth`` is given.

    ``method="binned"`` is the unsmoothed count / occupancy map by default, the
    estimator pynapple's tuning curves use.
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

    lines = [
        "## 1. Spatial information, DANDI 000582 sub-10884_ses-03080402",
        "",
        "Both tools: 40 × 40 bins of 2.5 cm over [−50, 50] cm, no smoothing, no"
        " occupancy threshold, all tracked samples. neurospatial:"
        " `Environment.from_samples(..., dimension_ranges=..., infer_active_bins=False)`,"
        " `compute_spatial_rates(..., method='binned')` (unsmoothed by default),"
        " `spatial_information()` and"
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
        f"- The neurospatial maps above use the default `method='binned'` (no"
        f" `bandwidth=`, so no smoothing): median relative difference in"
        f" bits/spike from pynapple over units"
        f" {np.median(_relative(ns_spike, pn[:, 1])):.3g}, the remainder being the"
        f" spike-to-sample assignment above.",
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
    # opexebo's autocorrelation zero-fills NaN (unvisited bins as 0 Hz); the
    # same map zero-filled on the neurospatial side isolates that convention.
    ns_zero_filled = np.array(
        [grid_score(spatial_autocorrelation(env, np.nan_to_num(row))) for row in firing]
    )
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
        " `grid_score`, which separates the autocorrelogram step from the scoring step."
        " opexebo's `autocorrelation` replaces NaN with 0 (unvisited bins as 0 Hz)"
        " before correlating; the zero-filled column applies that to neurospatial's input.",
        "",
        "| unit | neurospatial | neurospatial batch | ns on zero-filled map |"
        " ns score on opexebo autocorr | opexebo | difference |"
        " ns undefined lags in opexebo's extent | max acorr diff at shared lags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, uid in enumerate(rec.unit_ids):
        lines.append(
            f"| {uid} | {ns_scores[i]:.3f} | {ns_batch[i]:.3f} | {ns_zero_filled[i]:.3f} |"
            f" {ns_on_ox[i]:.3f} |"
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


def nan_aware_autocorrelation(rate_map: np.ndarray, min_overlap: int) -> np.ndarray:
    """Pearson autocorrelation at every lag over the bins visited at both ends.

    Unvisited bins (NaN) are left out, never treated as 0 Hz; a lag is NaN when
    fewer than ``min_overlap`` visited bins overlap, or when either side of the
    overlap is constant. Sums over each lag's
    overlap come from FFT correlations of the masked map and its mask.

    Parameters
    ----------
    rate_map : ndarray, shape (n_x, n_y)
        Rate map with NaN for unvisited bins.
    min_overlap : int
        Fewest overlapping visited bins for a lag to be defined.

    Returns
    -------
    ndarray, shape (2 n_x - 1, 2 n_y - 1)
        Zero lag at the centre.
    """
    from scipy.signal import correlate

    visited = np.isfinite(rate_map).astype(float)
    values = np.where(visited > 0, rate_map, 0.0)

    def overlap_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return correlate(a, b, mode="full", method="fft")

    n = np.rint(overlap_sum(visited, visited))
    sx = overlap_sum(values, visited)
    sy = overlap_sum(visited, values)
    sxx = overlap_sum(values**2, visited)
    syy = overlap_sum(visited, values**2)
    sxy = overlap_sum(values, values)
    var_x = n * sxx - sx**2
    var_y = n * syy - sy**2
    # A side with no variance over the overlap has no correlation; FFT
    # round-off would otherwise leave a tiny denominator and a value of ±1.
    flat = (var_x <= 1e-10 * np.abs(n * sxx)) | (var_y <= 1e-10 * np.abs(n * syy))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n * sxy - sx * sy) / np.sqrt(var_x * var_y)
    r[(n < min_overlap) | flat] = np.nan
    return np.clip(r, -1.0, 1.0)


def _population_maps(rec: Recording) -> tuple[Any, np.ndarray, np.ndarray]:
    """Smoothed binned rates on a full rectangular grid, per bin and as 2-D maps."""
    from neurospatial import Environment
    from neurospatial.encoding import compute_spatial_rates

    low = np.floor(np.nanmin(rec.positions, axis=0) / BIN_CM) * BIN_CM
    high = np.ceil(np.nanmax(rec.positions, axis=0) / BIN_CM) * BIN_CM
    env = Environment.from_samples(
        rec.positions,
        bin_size=BIN_CM,
        dimension_ranges=list(zip(low, high, strict=True)),
        infer_active_bins=False,
        units="cm",
    )
    rates = compute_spatial_rates(
        env,
        list(rec.spike_times or ()),
        rec.times,
        rec.positions,
        method="binned",
        bandwidth=POPULATION_BANDWIDTH_CM,
        min_speed=rec.params.get("min_speed"),
        epochs=rec.params.get("analysis_epochs"),
    )
    shape = tuple(int(k) for k in np.rint((high - low) / BIN_CM))
    index = np.rint((np.asarray(env.bin_centers) - low) / BIN_CM - 0.5).astype(int)
    firing = np.asarray(rates.firing_rates, dtype=float)
    maps = np.full((len(firing), *shape), np.nan)
    maps[:, index[:, 0], index[:, 1]] = firing
    return env, firing, maps


def grid_population_check(
    rec: Recording,
    title: str,
    units: np.ndarray,
    author_medians: dict[int, float] | None = None,
) -> list[str]:
    """Grid-score conventions compared on a population of grid-module units.

    ``units`` selects the units; ``author_medians`` maps a module id in
    ``rec.params["module"]`` to the authors' median unit grid score.
    """
    from opexebo.analysis import autocorrelation
    from opexebo.analysis import grid_score as opexebo_grid_score

    from neurospatial.encoding import grid_score, spatial_autocorrelation

    env, firing, maps = _population_maps(rec)
    visited = int(np.isfinite(maps[0]).sum())
    scores: dict[str, list[float]] = {name: [] for name in CONVENTIONS}
    for i in np.flatnonzero(units):
        rate, grid_map = firing[i], maps[i]
        scores["neurospatial"].append(grid_score(spatial_autocorrelation(env, rate)))
        scores["full lags, overlap 20 (a)"].append(
            grid_score(nan_aware_autocorrelation(grid_map, 20))
        )
        scores["full lags, overlap 1/4 (b)"].append(
            grid_score(nan_aware_autocorrelation(grid_map, visited // 4))
        )
        scores["zero-filled (c)"].append(
            grid_score(spatial_autocorrelation(env, np.nan_to_num(rate)))
        )
        scores["opexebo"].append(opexebo_grid_score(autocorrelation(grid_map))[0])
    arrays = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    reference = arrays["opexebo"]
    lines = [
        f"## {title}",
        "",
        f"{int(units.sum())} grid-module units; binned maps smoothed with a"
        f" {POPULATION_BANDWIDTH_CM:g} cm kernel on a full {BIN_CM:g} cm grid"
        f" ({visited} visited bins); every convention scores the same map with"
        " neurospatial's `grid_score`, except opexebo, which uses its own."
        " (a) and (b) correlate visited bins only at every lag with at least 20"
        " or a quarter of the visited bins overlapping; (c) zero-fills unvisited"
        f" bins as opexebo does. Verdicts use the conventional cutoff {GRID_CUTOFF}.",
        "",
        "| Convention | Median score | r vs opexebo | Max abs. difference | Verdicts that differ from opexebo |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, values in arrays.items():
        both = np.isfinite(values) & np.isfinite(reference)
        r = (
            float(np.corrcoef(values[both], reference[both])[0, 1])
            if both.sum() > 2
            else float("nan")
        )
        differ = int(
            np.sum((values[both] > GRID_CUTOFF) != (reference[both] > GRID_CUTOFF))
        )
        lines.append(
            f"| {name} | {np.nanmedian(values):.3f} | {r:.3f} |"
            f" {np.nanmax(np.abs(values[both] - reference[both])):.3f} | {differ} of {int(both.sum())} |"
        )
    lines.append("")
    if author_medians:
        modules = np.asarray(rec.params["module"])[units]
        lines += [
            "Per module, median score against the authors' median unit grid score"
            " (their own maps and scorer):",
            "",
            "| Module | Units | Authors | " + " | ".join(CONVENTIONS) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in CONVENTIONS) + " |",
        ]
        for module, median in sorted(author_medians.items()):
            member = modules == module
            lines.append(
                f"| {module} | {int(member.sum())} | {median:.2f} | "
                + " | ".join(
                    f"{np.nanmedian(arrays[c][member]):.2f}" for c in CONVENTIONS
                )
                + " |"
            )
        lines.append("")
    return lines


def _circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


def head_direction_check(rec: Recording) -> list[str]:
    import pynapple as nap
    from _fetch import open_nwb

    with open_nwb(rec.params["nwb_source"]) as nwbfile:
        locations = nwbfile.units["location"][:]
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
    # Heading at each decoded bin: interpolate cos and sin (NaN-safe, no unwrap)
    # and drop bins inside tracking gaps longer than MAX_TRACKING_GAP_S, where
    # there is no measured heading to compare with.
    finite = np.isfinite(rec.headings)
    t_ok, h_ok = rec.times[finite], rec.headings[finite]
    actual = np.arctan2(
        np.interp(decoded.t, t_ok, np.sin(h_ok)),
        np.interp(decoded.t, t_ok, np.cos(h_ok)),
    )
    after = np.clip(np.searchsorted(t_ok, decoded.t), 1, len(t_ok) - 1)
    tracked = (t_ok[after] - t_ok[after - 1]) <= MAX_TRACKING_GAP_S
    error = _circular_error(decoded.to_numpy(), actual)
    all_bins = float(np.median(error))
    error = error[tracked]
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
        f" {np.degrees(np.median(error)):.1f}° ({np.median(error):.3f} rad) over the"
        f" {tracked.sum()} bins outside tracking gaps longer than {MAX_TRACKING_GAP_S} s"
        f" ({np.degrees(all_bins):.1f}° over all bins).",
        "- neurospatial: no public heading decoder (`neurospatial.decoding` decodes"
        " position on an `Environment`; no documented recipe in `docs/user-guide/`"
        " decodes head direction). Recorded as missing, see GAPS.md.",
        "",
    ]


def _population_sections() -> list[str]:
    gardner = load_gardner_2022_grid.load()
    vollan = load_vollan_of.load()
    params = vollan.params
    return [
        *grid_population_check(
            gardner,
            "2b. Grid score conventions, Gardner 2022 rat R day 1 open field",
            np.ones(len(gardner.unit_ids), dtype=bool),
        ),
        *grid_population_check(
            vollan,
            "2c. Grid score conventions, Vollan 2025 rat 26018 open field",
            np.asarray(params["is_grid"]),
            author_medians=dict(
                zip(
                    params["module_ids"].tolist(),
                    params["module_median_grid_score"].tolist(),
                    strict=True,
                )
            ),
        ),
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
        "Versions: "
        + ", ".join(f"{name} {version}" for name, version in _versions().items()),
        "",
        *spatial_information_check(sargolini, env, rates),
        *grid_score_check(sargolini, env, rates),
        *_population_sections(),
        *head_direction_check(load_peyrache_mouse32.load()),
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reference_checks.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
