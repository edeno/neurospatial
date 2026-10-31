"""Calibrate the default ``max_gap`` by simulation.

``max_gap`` is the longest stretch of missing tracking, between two valid
samples, that analyses bridge; a longer gap is excluded as a tracking loss
(see ``neurospatial._intervals.validate_max_gap``). This script sweeps
``max_gap`` on simulated linear-track recordings with realistic missing
tracking and scores each value against the true trajectory.

Scenarios
---------
About 15 min (the session ends at the end of the first pause that ends at or
after 900 s) of running back and forth on a 200 cm linear track. The animal
runs between 2 cm and 198 cm; each run has a raised-cosine speed profile with
a mean speed drawn uniformly from 30-50 cm/s, and pauses at the ends last
2-6 s (uniform). Tracking at 30, 60 and 120 Hz, with 1 % or 5 % isolated
dropped frames (rows deleted, no two adjacent): 6 scenarios. Every recording
also has

- DeepLabCut-style low-confidence NaN bursts (NaN rows), 1-15 frames
  (uniform integer length), separated by at least one tracked frame, whose
  start rate per frame is four times higher during the pauses at the ends;
  the base rate is set so the bursts cover about 3 % of frames;
- three tracking losses of 2-5 s (uniform) that start at a uniform time
  inside three distinct runs (NaN rows);
- one 10 s pickup in the middle of a pause between 20 % and 80 % of the
  session: the animal is off the track (no true on-track time, NaN rows) and
  is put back at the same end (``--pickup-return opposite`` puts it back at
  the other end).

Each scenario has ``--replicates`` recordings (default 12), each with a
seed derived from the scenario and replicate, so the run is deterministic.

Metrics (errors against the truth, simulated at 1 kHz)
------------------------------------------------------
1-4. Occupancy from ``env.occupancy(times, positions, max_gap=...)`` on
     2 cm and 5 cm bins over [0, 200] cm, against the true time in each bin
     (on-track time only): total variation distance between the normalized
     occupancies, ``0.5 * sum|p - q|`` (1 when every interval is excluded),
     and L1 distance in seconds, ``sum|occ - true_occ|``.
5.   Rate map of one place cell (Gaussian field, centre 100 cm, sd 10 cm,
     0.5 Hz baseline, 15 Hz peak). Spikes are an inhomogeneous Poisson
     process on the true trajectory; during the pickup the cell fires at its
     0.5 Hz baseline (those spikes fall in untracked time). The estimate is
     ``compute_spatial_rate(..., method="binned", bandwidth=0.0)`` on 5 cm
     bins (counts over occupancy, no smoothing); the error is the RMSE
     against the true rate at the bin centres over the bins with support
     (NaN when no bin has support).
6.   Running bouts: ``|detected - true|``, detected with
     ``segment_by_velocity(min_speed=10 cm/s, min_duration=0.5 s)`` (default
     hysteresis 2 and 0.1 s speed smoothing); the true count is the number
     of runs, one bout per run.
7.   Complete traversals: ``|detected - true|``. A traversal is a change
     from the < 10 cm zone to the > 190 cm zone, or back, between tracked
     samples of one observed stretch: the valid samples joined by gaps no
     longer than ``max_gap`` (the gating every public function applies), so
     a traversal cannot span an excluded gap. The true count is the number of
     complete runs.

Summary
-------
``max_gap`` is swept on a log grid of 8 values per decade from 0.03 s to
13.3 s, with 0.03 s, 0.5 s and 2.0 s added (25 values; the value past 10 s bridges
the pickup). For each of the 42 scenario-metric pairs: the mean error and its Monte-Carlo
standard error (SE) over replicates at each ``max_gap``, the best (lowest)
mean, and whether every grid value in [0.55, 2.0] s stays within two SE of
the best. Two SE definitions are reported: the SE of the mean at the best
value (primary), and the SE of the paired per-replicate difference from the
best value (the replicates share their data across ``max_gap``).

Usage
-----
::

    PYTHONPATH=src python scripts/calibrate_max_gap.py            # full run
    PYTHONPATH=src python scripts/calibrate_max_gap.py --quick    # smoke test
    PYTHONPATH=src python scripts/calibrate_max_gap.py --out curves.json

The full run (6 scenarios x 12 replicates x 25 ``max_gap`` values) takes
about 2 minutes single-process on a laptop; ``--quick`` (2 replicates,
an 11-value grid) takes about 10 seconds.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurospatial import Environment
from neurospatial._intervals import interval_valid_mask
from neurospatial.behavior.segmentation import segment_by_velocity
from neurospatial.encoding import compute_spatial_rate

BASE_SEED = 20261031
SESSION_S = 900.0
TRACK_CM = (0.0, 200.0)
RUN_ENDS_CM = (2.0, 198.0)
RUN_SPEED_CM_S = (30.0, 50.0)
PAUSE_S = (2.0, 6.0)
FRAME_RATES_HZ = (30, 60, 120)
DROP_FRACTIONS = (0.01, 0.05)
BURST_FRAMES = (1, 15)
BURST_COVERAGE = 0.03
PAUSE_BURST_FACTOR = 4.0
N_LOSSES = 3
LOSS_S = (2.0, 5.0)
PICKUP_S = 10.0
TRUTH_HZ = 1000.0
FIELD_CENTER_CM, FIELD_SD_CM = 100.0, 10.0
PEAK_HZ, BASELINE_HZ = 15.0, 0.5
BOUT_MIN_SPEED_CM_S, BOUT_MIN_DURATION_S = 10.0, 0.5
TRAVERSAL_ZONES_CM = (10.0, 190.0)
PLATEAU_S = (0.55, 2.0)
METRICS = (
    "occupancy_tv_2cm",
    "occupancy_tv_5cm",
    "occupancy_l1_2cm",
    "occupancy_l1_5cm",
    "rate_rmse_5cm",
    "bout_count",
    "traversal_count",
)
PAUSE, RUN, PICKUP = 0, 1, 2


@dataclass(frozen=True)
class Truth:
    """True trajectory as segments, sampled at ``TRUTH_HZ``.

    Attributes
    ----------
    starts, stops : ndarray, shape (n_segments,)
        Segment bounds in seconds.
    kind : ndarray of int, shape (n_segments,)
        ``PAUSE``, ``RUN`` or ``PICKUP``.
    x_from, x_to : ndarray, shape (n_segments,)
        Position in cm at the segment's start and end (equal for a pause).
    losses : ndarray, shape (N_LOSSES, 2)
        ``[start, stop)`` of the tracking losses in seconds.
    """

    starts: NDArray[np.float64]
    stops: NDArray[np.float64]
    kind: NDArray[np.int64]
    x_from: NDArray[np.float64]
    x_to: NDArray[np.float64]
    losses: NDArray[np.float64]

    @property
    def duration(self) -> float:
        """Session length in seconds."""
        return float(self.stops[-1])

    def segment_of(self, t: NDArray[np.float64]) -> NDArray[np.intp]:
        """Index of the segment holding each time, shape ``t.shape``."""
        index = np.searchsorted(self.starts, t, side="right") - 1
        return np.clip(index, 0, self.starts.size - 1)

    def position(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """True position in cm at times ``t``; NaN during the pickup.

        Parameters
        ----------
        t : ndarray, shape (n,)
            Times in seconds.

        Returns
        -------
        ndarray, shape (n,)
        """
        seg = self.segment_of(t)
        u = (t - self.starts[seg]) / (self.stops[seg] - self.starts[seg])
        progress = u - np.sin(2.0 * np.pi * u) / (2.0 * np.pi)
        x = self.x_from[seg] + (self.x_to[seg] - self.x_from[seg]) * progress
        return np.where(self.kind[seg] == PICKUP, np.nan, x)


def simulate_truth(rng: np.random.Generator, pickup_return: str) -> Truth:
    """Simulate segments of pauses and runs with one pickup and three losses.

    Parameters
    ----------
    rng : numpy.random.Generator
    pickup_return : {"same", "opposite"}
        End of the track where the animal is put back after the pickup.

    Returns
    -------
    Truth
    """
    n_max = 400  # far more pause/run pairs than 900 s can hold
    run_s = (RUN_ENDS_CM[1] - RUN_ENDS_CM[0]) / rng.uniform(*RUN_SPEED_CM_S, n_max)
    pause_s = rng.uniform(*PAUSE_S, n_max)
    durations = np.column_stack([pause_s, run_s]).ravel()
    kind = np.tile([PAUSE, RUN], n_max)
    stops = np.cumsum(durations)
    # End at the first pause that ends at or after SESSION_S.
    last = np.flatnonzero((kind == PAUSE) & (stops >= SESSION_S))[0]
    durations, kind = durations[: last + 1], kind[: last + 1]
    starts = np.r_[0.0, np.cumsum(durations)[:-1]]

    # Pickup: split a pause starting between 20 % and 80 % of the session.
    eligible = np.flatnonzero(
        (kind == PAUSE) & (starts > 0.2 * SESSION_S) & (starts < 0.8 * SESSION_S)
    )
    p = int(rng.choice(eligible))
    half = durations[p] / 2.0
    durations = np.r_[durations[:p], half, PICKUP_S, half, durations[p + 1 :]]
    kind = np.r_[kind[:p], PAUSE, PICKUP, PAUSE, kind[p + 1 :]]
    starts = np.r_[0.0, np.cumsum(durations)[:-1]]
    stops = starts + durations

    # The animal starts at the low end; each run, and a pickup put back at
    # the opposite end, moves it to the other end.
    flips = (kind == RUN) | ((kind == PICKUP) & (pickup_return == "opposite"))
    end_before = np.r_[0, np.cumsum(flips)[:-1]] % 2
    ends = np.asarray(RUN_ENDS_CM)
    x_from = ends[end_before]
    x_to = np.where(kind == RUN, ends[1 - end_before], x_from)

    runs = np.flatnonzero(kind == RUN)
    chosen = np.sort(rng.choice(runs, N_LOSSES, replace=False))
    loss_start = starts[chosen] + rng.uniform(0, 1, N_LOSSES) * durations[chosen]
    loss_stop = loss_start + rng.uniform(*LOSS_S, N_LOSSES)
    return Truth(
        starts, stops, kind, x_from, x_to, np.column_stack([loss_start, loss_stop])
    )


def true_rate(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Place-cell rate in Hz at positions ``x`` (cm), shape ``x.shape``."""
    gauss = np.exp(-0.5 * ((x - FIELD_CENTER_CM) / FIELD_SD_CM) ** 2)
    return BASELINE_HZ + (PEAK_HZ - BASELINE_HZ) * gauss


def simulate_spikes(
    truth: Truth, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Sample the truth at 1 kHz and draw inhomogeneous Poisson spikes.

    Returns
    -------
    t_fine : ndarray, shape (n_fine,)
        Midpoints of the 1 ms truth bins, in seconds.
    x_fine : ndarray, shape (n_fine,)
        True position at ``t_fine`` (NaN during the pickup).
    spike_times : ndarray, shape (n_spikes,)
        Sorted spike times; the pickup fires at the baseline rate.
    """
    dt = 1.0 / TRUTH_HZ
    n_fine = round(truth.duration * TRUTH_HZ)
    t_fine = (np.arange(n_fine) + 0.5) * dt
    x_fine = truth.position(t_fine)
    rate = np.where(np.isnan(x_fine), BASELINE_HZ, true_rate(x_fine))
    counts = rng.poisson(rate * dt)
    left = np.repeat(t_fine - dt / 2, counts)
    spike_times = np.sort(left + rng.uniform(0, dt, left.size))
    return t_fine, x_fine, spike_times


def nan_bursts(
    paused: NDArray[np.bool_], rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Low-confidence NaN bursts, four times as frequent during pauses.

    Candidate starts are drawn per frame; a candidate that falls inside or
    right after an accepted burst is skipped, so bursts keep their 1-15
    frame length and are separated by at least one tracked frame.

    Parameters
    ----------
    paused : ndarray of bool, shape (n_frames,)
        True at frames during a pause at the track ends.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of bool, shape (n_frames,)
        True at frames inside a burst.
    """
    n = paused.size
    weight = np.where(paused, PAUSE_BURST_FACTOR, 1.0)
    mean_len = 0.5 * (BURST_FRAMES[0] + BURST_FRAMES[1])
    p_start = BURST_COVERAGE / (mean_len * weight.mean()) * weight
    candidates = np.flatnonzero(rng.random(n) < p_start)
    lengths = rng.integers(BURST_FRAMES[0], BURST_FRAMES[1] + 1, candidates.size)
    mask = np.zeros(n, dtype=bool)
    free_from = 0
    for start, length in zip(candidates, lengths, strict=True):  # ~10^3 bursts
        if start < free_from:
            continue
        mask[start : start + length] = True
        free_from = start + length + 1
    return mask


@dataclass(frozen=True)
class Recording:
    """Observed tracking of one simulated session.

    Attributes
    ----------
    times : ndarray, shape (n_kept,)
        Frame times in seconds after the dropped rows are deleted.
    positions : ndarray, shape (n_kept, 1)
        Tracked position in cm; NaN rows are missing samples.
    burst, loss, pickup : ndarray of bool, shape (n_frames,)
        Missing-frame sources on the full frame grid (before deletion).
    dropped : ndarray of bool, shape (n_frames,)
        Deleted rows.
    frame_times : ndarray, shape (n_frames,)
        Every frame time, before deletion.
    """

    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    burst: NDArray[np.bool_]
    loss: NDArray[np.bool_]
    pickup: NDArray[np.bool_]
    dropped: NDArray[np.bool_]
    frame_times: NDArray[np.float64]


def observe(
    truth: Truth, fs: float, drop_fraction: float, rng: np.random.Generator
) -> Recording:
    """Sample the truth at ``fs`` Hz and apply every source of missing data.

    Parameters
    ----------
    truth : Truth
    fs : float
        Frame rate in Hz.
    drop_fraction : float
        Target fraction of isolated dropped (deleted) frames.
    rng : numpy.random.Generator

    Returns
    -------
    Recording
    """
    n = int(np.floor(truth.duration * fs)) + 1
    frame_times = np.arange(n) / fs
    x = truth.position(frame_times)
    kind = truth.kind[truth.segment_of(frame_times)]
    pickup = kind == PICKUP
    loss = np.zeros(n, dtype=bool)
    for start, stop in truth.losses:
        loss |= (frame_times >= start) & (frame_times < stop)
    burst = nan_bursts(kind == PAUSE, rng)
    # Isolated drops: a chosen frame right after another chosen frame is
    # kept, so the realised fraction is q (1 - q) = drop_fraction.
    q = 0.5 * (1.0 - np.sqrt(1.0 - 4.0 * drop_fraction))
    chosen = rng.random(n) < q
    dropped = chosen & ~np.r_[False, chosen[:-1]]
    x = np.where(pickup | loss | burst, np.nan, x)
    keep = ~dropped
    return Recording(
        frame_times[keep], x[keep, None], burst, loss, pickup, dropped, frame_times
    )


def count_traversals(
    times: NDArray[np.float64], x: NDArray[np.float64], max_gap: float | None
) -> int:
    """Complete end-to-end traversals within observed stretches.

    Parameters
    ----------
    times : ndarray, shape (n,)
    x : ndarray, shape (n,)
        Position in cm; NaN is missing.
    max_gap : float or None
        Gaps between valid samples longer than this split the recording.

    Returns
    -------
    int
        Number of changes between the < 10 cm and > 190 cm zones, ignoring
        the middle of the track, inside one observed stretch.
    """
    valid = np.isfinite(x)
    mask = interval_valid_mask(times, max_gap=max_gap, sample_valid=valid)
    stretch = np.cumsum(np.r_[True, ~mask])
    zone = np.where(
        x < TRAVERSAL_ZONES_CM[0], -1, np.where(x > TRAVERSAL_ZONES_CM[1], 1, 0)
    )
    in_zone = valid & (zone != 0)
    z, s = zone[in_zone], stretch[in_zone]
    return int(np.sum((z[1:] != z[:-1]) & (s[1:] == s[:-1])))


def build_envs() -> dict[float, Environment]:
    """1-D track environments with 2 cm and 5 cm bins covering [0, 200] cm."""
    extent = np.array([[TRACK_CM[0]], [TRACK_CM[1]]])
    return {
        size: Environment.from_samples(
            extent,
            bin_size=size,
            dimension_ranges=[TRACK_CM],
            infer_active_bins=False,
            units="cm",
        )
        for size in (2.0, 5.0)
    }


def score_recording(
    truth: Truth,
    rec: Recording,
    spike_times: NDArray[np.float64],
    t_fine: NDArray[np.float64],
    x_fine: NDArray[np.float64],
    envs: dict[float, Environment],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Errors of every metric at every ``max_gap``.

    Returns
    -------
    ndarray, shape (n_metrics, n_grid)
        Rows in the order of ``METRICS``.
    """
    on_track = np.isfinite(x_fine)
    true_occ = {
        size: np.bincount(
            env.bin_at(x_fine[on_track, None]), minlength=env.n_bins
        ).astype(float)
        / TRUTH_HZ
        for size, env in envs.items()
    }
    # Every run is one bout and one complete traversal (the session ends in
    # a pause); describe_inputs checks both counts on clean sampling.
    true_bouts = true_traversals = int(np.sum(truth.kind == RUN))
    rate_truth = true_rate(envs[5.0].bin_centers[:, 0])

    errors = np.full((len(METRICS), grid.size), np.nan)
    with warnings.catch_warnings():
        # Exclusion warnings at very short max_gap are expected here.
        warnings.simplefilter("ignore", UserWarning)
        for j, gap in enumerate(grid):
            g = float(gap)
            row = 0
            for metric in ("tv", "l1"):
                for size in (2.0, 5.0):
                    occ = envs[size].occupancy(rec.times, rec.positions, max_gap=g)
                    truth_occ = true_occ[size]
                    if metric == "tv":
                        if occ.sum() == 0:
                            errors[row, j] = 1.0
                        else:
                            p = occ / occ.sum()
                            errors[row, j] = (
                                0.5 * np.abs(p - truth_occ / truth_occ.sum()).sum()
                            )
                    else:
                        errors[row, j] = np.abs(occ - truth_occ).sum()
                    row += 1
            rate = compute_spatial_rate(
                envs[5.0],
                spike_times,
                rec.times,
                rec.positions,
                method="binned",
                bandwidth=0.0,
                max_gap=g,
                warn_on_drop=False,
            )
            support = rate.spatial_support
            if support.any():
                diff = rate.firing_rate[support] - rate_truth[support]
                errors[4, j] = np.sqrt(np.mean(diff**2))
            bouts = segment_by_velocity(
                rec.times,
                rec.positions,
                min_speed=BOUT_MIN_SPEED_CM_S,
                min_duration=BOUT_MIN_DURATION_S,
                max_gap=g,
            )
            errors[5, j] = abs(len(bouts) - true_bouts)
            detected = count_traversals(rec.times, rec.positions[:, 0], g)
            errors[6, j] = abs(detected - true_traversals)
    return errors


def recording_seed(
    fs: int, drop_fraction: float, replicate: int
) -> np.random.Generator:
    """Generator seeded from the scenario and replicate."""
    seq = np.random.SeedSequence(
        [BASE_SEED, fs, round(drop_fraction * 1000), replicate]
    )
    return np.random.default_rng(seq)


def make_grid(quick: bool) -> NDArray[np.float64]:
    """Log-spaced ``max_gap`` grid from 0.03 s to one step past 10 s.

    The full grid has 8 values per decade anchored at 1 s (0.0316 ... 13.3),
    plus 0.03, 0.5 and 2.0 s: 25 values. The step past 10 s is there
    because the 10 s pickup leaves a gap of 10 s plus one frame between
    valid samples, which ``max_gap=10`` does not bridge. The quick grid has
    2 values per decade (up to 31.6 s) plus 0.03, 0.5, 0.75 and 2.0 s.
    """
    per_decade = 2 if quick else 8
    k = np.arange(int(np.ceil(per_decade * np.log10(0.03))), per_decade + 2)
    extra = [0.03, 0.5, 0.75, 2.0] if quick else [0.03, 0.5, 2.0]
    return np.unique(np.round(np.r_[10.0 ** (k / per_decade), extra], 12))


def describe_inputs(truth: Truth, rec: Recording, fs: float) -> None:
    """Print realised missing-data statistics of one recording."""
    burst = rec.burst
    edges = np.diff(np.r_[0, burst.astype(int), 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    kind = truth.kind[truth.segment_of(rec.frame_times)]
    paused = kind == PAUSE
    nan_rows = np.isnan(rec.positions[:, 0])
    print(f"  session {truth.duration:.1f} s, {rec.frame_times.size} frames at {fs} Hz")
    print(
        f"  dropped rows {rec.dropped.mean():.4f}; NaN rows kept {nan_rows.mean():.4f}"
        f" (bursts {burst.mean():.4f}, losses {rec.loss.mean():.4f},"
        f" pickup {rec.pickup.mean():.4f})"
    )
    print(
        f"  bursts: n={lengths.size}, length min/median/max "
        f"{lengths.min()}/{int(np.median(lengths))}/{lengths.max()} frames;"
        f" burst coverage paused {burst[paused].mean():.4f}"
        f" vs running {burst[kind == RUN].mean():.4f}"
    )
    print(f"  losses (s): {np.round(np.diff(truth.losses, axis=1).ravel(), 2)}")
    gaps = np.diff(rec.times[~nan_rows])
    print(
        f"  longest gaps between valid samples (s): {np.round(np.sort(gaps)[-6:], 3)}"
    )
    n_runs = int(np.sum(truth.kind == RUN))
    clean_t = rec.frame_times
    clean_x = truth.position(clean_t)
    on = np.isfinite(clean_x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clean_bouts = len(
            segment_by_velocity(
                clean_t[on],
                clean_x[on, None],
                min_speed=BOUT_MIN_SPEED_CM_S,
                min_duration=BOUT_MIN_DURATION_S,
                max_gap=None,
            )
        )
    clean_trav = count_traversals(clean_t, clean_x, 1.5 / fs)
    print(
        f"  true runs {n_runs}; on clean {fs} Hz tracking (pickup removed): "
        f"{clean_bouts} bouts, {clean_trav} traversals; pickup at "
        f"{truth.starts[truth.kind == PICKUP][0]:.1f} s"
    )


def summarize(
    errors: NDArray[np.float64],
    grid: NDArray[np.float64],
    scenarios: list[tuple[int, float]],
) -> dict[str, object]:
    """Per-pair curves, best values and plateau checks.

    Parameters
    ----------
    errors : ndarray, shape (n_scenarios, n_metrics, n_replicates, n_grid)

    Returns
    -------
    dict
        JSON-compatible summary.
    """
    n_rep = errors.shape[2]
    mean = errors.mean(axis=2)
    se = errors.std(axis=2, ddof=1) / np.sqrt(n_rep)
    in_window = (grid >= PLATEAU_S[0] - 1e-9) & (grid <= PLATEAU_S[1] + 1e-9)
    pairs = []
    near_best = np.zeros(mean.shape, dtype=bool)
    near_best_paired = np.zeros(mean.shape, dtype=bool)
    for s, (fs, drop) in enumerate(scenarios):
        for m, metric in enumerate(METRICS):
            curve = mean[s, m]
            b = int(np.nanargmin(curve))
            best = curve[b]
            diff = errors[s, m] - errors[s, m][:, [b]]
            se_diff = diff.std(axis=0, ddof=1) / np.sqrt(n_rep)
            near = curve <= best + 2 * se[s, m, b] + 1e-12
            near_paired = diff.mean(axis=0) <= 2 * se_diff + 1e-12
            near_best[s, m] = near
            near_best_paired[s, m] = near_paired
            window_worst = float(np.nanmax(curve[in_window]))
            pairs.append(
                {
                    "frame_rate_hz": fs,
                    "drop_fraction": drop,
                    "metric": metric,
                    "mean": curve.tolist(),
                    "se": se[s, m].tolist(),
                    "best_max_gap": float(grid[b]),
                    "best_mean": float(best),
                    "best_se": float(se[s, m, b]),
                    "plateau": bool(np.all(near[in_window])),
                    "plateau_paired": bool(np.all(near_paired[in_window])),
                    "window_worst_mean": window_worst,
                    "window_worst_pct_above_best": (
                        float(100 * (window_worst - best) / best) if best > 0 else None
                    ),
                    "near_best_max_gaps": grid[near].tolist(),
                }
            )
    return {
        "grid": grid.tolist(),
        "n_replicates": n_rep,
        "pairs": pairs,
        "near_best_count": near_best.sum(axis=(0, 1)).tolist(),
        "near_best_count_paired": near_best_paired.sum(axis=(0, 1)).tolist(),
        "near_best": near_best.tolist(),
        "errors": errors.tolist(),
    }


def contiguous_max(counts: NDArray[np.int64], grid: NDArray[np.float64]) -> tuple:
    """Widest contiguous grid range at the maximum count, and its geometric middle."""
    top = counts == counts.max()
    bounds = np.flatnonzero(np.diff(np.r_[0, top.astype(int), 0]))
    starts, stops = bounds[::2], bounds[1::2] - 1
    widest = int(np.argmax(np.log(grid[stops]) - np.log(grid[starts])))
    lo, hi = grid[starts[widest]], grid[stops[widest]]
    return float(lo), float(hi), float(np.sqrt(lo * hi))


def print_report(summary: dict, scenarios: list[tuple[int, float]]) -> None:
    """Print the per-pair table, per-metric ranges and plateau middle."""
    grid = np.asarray(summary["grid"])
    pairs = summary["pairs"]
    print("\nmax_gap grid (s):", np.round(grid, 3).tolist())
    header = (
        f"{'Hz':>4} {'drop':>5} {'metric':<18} {'best@':>7} {'best':>9} "
        f"{'SE':>8} {'plat':>5} {'pair':>5} {'worst%':>8}  near-best range (s)"
    )
    print("\n" + header)
    for p in pairs:
        near = p["near_best_max_gaps"]
        pct = p["window_worst_pct_above_best"]
        print(
            f"{p['frame_rate_hz']:>4} {p['drop_fraction']:>5.2f} {p['metric']:<18} "
            f"{p['best_max_gap']:>7.3f} {p['best_mean']:>9.4g} {p['best_se']:>8.2g} "
            f"{'yes' if p['plateau'] else 'no':>5} "
            f"{'yes' if p['plateau_paired'] else 'no':>5} "
            f"{'-' if pct is None else f'{pct:.1f}':>8}  "
            f"{min(near):.3f}-{max(near):.3f}"
        )
    n_pairs = len(pairs)
    n_plateau = sum(p["plateau"] for p in pairs)
    n_paired = sum(p["plateau_paired"] for p in pairs)
    print(
        f"\nWithin 2 SE of best over [{PLATEAU_S[0]}, {PLATEAU_S[1]}] s: "
        f"{n_plateau} of {n_pairs} pairs (SE of best mean); "
        f"{n_paired} of {n_pairs} (paired-difference SE)."
    )
    near = np.asarray(summary["near_best"])
    print("\nPer metric: max_gap values where all scenarios are within 2 SE of best")
    for m, metric in enumerate(METRICS):
        all_near = near[:, m].all(axis=0)
        values = grid[all_near]
        text = (
            f"{values.min():.3f}-{values.max():.3f} s ({all_near.sum()} values)"
            if values.size
            else "none"
        )
        print(f"  {metric:<18} {text}")
    for label, key in (
        ("SE of best mean", "near_best_count"),
        ("paired SE", "near_best_count_paired"),
    ):
        counts = np.asarray(summary[key])
        lo, hi, mid = contiguous_max(counts, grid)
        print(
            f"\nPairs near best at each max_gap ({label}): "
            + ", ".join(f"{g:.3g}:{c}" for g, c in zip(grid, counts, strict=True))
        )
        print(
            f"  maximum {counts.max()} of {n_pairs} on {lo:.3f}-{hi:.3f} s; "
            f"geometric middle {mid:.3f} s"
        )
    errors = np.asarray(summary["errors"])
    bout = METRICS.index("bout_count")
    for s, (fs, drop) in enumerate(scenarios):
        if fs != 30:
            continue
        curve = errors[s, bout].mean(axis=0)
        shown = ", ".join(
            f"{g:.3g}:{c:.2f}"
            for g, c in zip(grid, curve, strict=True)
            if 0.3 <= g <= 2.5
        )
        print(f"\n30 Hz, drop {drop:.2f}: mean |bout error| by max_gap: {shown}")


def main() -> None:
    """Run the sweep and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument(
        "--quick", action="store_true", help="2 replicates and a coarse grid"
    )
    parser.add_argument(
        "--frame-rates", type=int, nargs="+", default=list(FRAME_RATES_HZ)
    )
    parser.add_argument(
        "--drop-fractions", type=float, nargs="+", default=list(DROP_FRACTIONS)
    )
    parser.add_argument("--pickup-return", choices=("same", "opposite"), default="same")
    parser.add_argument("--out", help="write the per-pair curves to this JSON file")
    args = parser.parse_args()

    n_rep = 2 if args.quick else args.replicates
    grid = make_grid(args.quick)
    envs = build_envs()
    scenarios = [(fs, d) for fs in args.frame_rates for d in args.drop_fractions]
    errors = np.full((len(scenarios), len(METRICS), n_rep, grid.size), np.nan)
    t0 = time.perf_counter()
    for s, (fs, drop) in enumerate(scenarios):
        for r in range(n_rep):
            rng = recording_seed(fs, drop, r)
            truth = simulate_truth(rng, args.pickup_return)
            t_fine, x_fine, spikes = simulate_spikes(truth, rng)
            rec = observe(truth, fs, drop, rng)
            if r == 0:
                print(f"\nScenario {fs} Hz, {drop:.0%} dropped, replicate 0:")
                describe_inputs(truth, rec, fs)
            errors[s, :, r] = score_recording(
                truth, rec, spikes, t_fine, x_fine, envs, grid
            )
        print(f"  done {fs} Hz / {drop:.2f} at {time.perf_counter() - t0:.0f} s")

    summary = summarize(errors, grid, scenarios)
    summary["settings"] = {
        "replicates": n_rep,
        "pickup_return": args.pickup_return,
        "base_seed": BASE_SEED,
        "runtime_s": time.perf_counter() - t0,
    }
    print_report(summary, scenarios)
    print(f"\nruntime {time.perf_counter() - t0:.0f} s")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
