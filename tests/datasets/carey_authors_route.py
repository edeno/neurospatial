"""Carey, Tanaka & van der Meer (2019): the authors' decoded-sequence route, ported.

A line-by-line port of the code that made the paper's sequence counts, for the
realdata tests that compare neurospatial's ``CAREY_2019`` route with it:

- ``Carey_etal_submitted/decoding_Seq/Generate_DecSeqShuf.m`` with shuffles
  off, and ``getEligibleSequences.m`` (github.com/vandermeerlab/papers at
  3aebc8e);
- the helpers they call, from github.com/vandermeerlab/vandermeerlab at
  ad0bbd4 (the commit the paper's README pins): ``LinearizePos.m``,
  ``getLinSpd.m`` / ``dxdt.m``, ``TSDtoIV.m``, ``restrict.m``,
  ``TuningCurves.m``, ``gausskernel.m``, ``MakeQfromS.m``, ``DecodeZ.m``,
  ``DecSeqDetectZ.m``, ``IntersectIV.m``, ``DifferenceIV.m``,
  ``AddNActiveCellsIV.m``.

It keeps the code's behaviour where the paper's text says otherwise or is
silent, including what is not intended: a position bin with no occupancy has
a NaN tuning curve, ``DecodeZ``'s ``nansum`` then gives it a likelihood of
exactly one, and ``max`` takes the first such bin on a tie, so a run of rows
whose occupied bins all score below one decodes to bin 1 and stays there
(``BIN_ONE_SINK``). The identity-shuffle filter is left out: with its
``shuf_Q(shuf_mask)`` indexing every shuffled bin is a permutation of the
session's first 5 ms bin, and in the authors' run (Octave, 3 shuffles,
R050-2014-03-29) it removed nothing.

The time grid of ``MakeQfromS`` is ``firstSpike:dt:lastSpike+dt``. Two forms
are offered: ``"octave"`` builds element ``k`` as ``fma(k, dt, first)``,
which reproduces the Octave 11 run (arm64) exactly, and ``"matlab"`` follows
MathWorks' ``colonop.m`` (each half from its own end, plain arithmetic). The
50 ms length test sees the last bit of these times.

This is a test helper, not package code: neurospatial does not reproduce the
bin-one artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray = NDArray[np.float64]

#: Generate_DecSeqShuf.m defaults, as ALL_Generate_DecSeq.m calls it.
DT = 0.025
Q_DT = 0.005
Q_BOXCAR = 5
Q_SMOOTH_SD = 0.002
TC_SMOOTH_BINS = 1.0
MIN_SPIKES = 25
N_EDGES = 112
TRIAL_START_OFFSET = -1.0
TRACK_EXCLUDE_START = 20.0
TRACK_EXCLUDE_END = 15.0
N_MIN_NEURONS = 4
MAX_JUMP_CM = 40.0
MIN_SEQ_LENGTH = 10
RUN_SPEED = 5.0
#: getEligibleSequences.m defaults.
MIN_LEN = 0.05
MIN_ACTIVE_CELLS = 4
#: TuningCurves.m: occupancy is samples times 1/30 s; at least one sample.
OCC_DT = 1 / 30
MIN_OCC = 1
#: The NaN-tuning-curve tie of DecodeZ.m, kept here as the code has it.
BIN_ONE_SINK = True


def gausskernel(radius: float, sd: float) -> FloatArray:
    """``gausskernel.m`` for one dimension: a normalized Gaussian on -R..R."""
    sd = 1e-5 if sd == 0 else sd
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    gauss = np.exp(-(grid**2) / (2 * sd**2))
    return gauss / gauss.sum()


def conv_same(x: FloatArray, kernel: FloatArray) -> FloatArray:
    """MATLAB ``conv(x, k, 'same')``: the centre of the full convolution."""
    full = np.convolve(x, kernel)
    start = kernel.size // 2
    return full[start : start + x.size]


def colon(
    first: float, step: float, last: float, kind: Literal["octave", "matlab"]
) -> FloatArray:
    """``first:step:last`` as Octave 11 (arm64) or MATLAB builds it."""
    if kind == "octave":
        n = math.floor((last - first) / step * (1 + np.finfo(float).eps))
        while math.fma(n + 1, step, first) <= last:
            n += 1
        while n > 0 and math.fma(n, step, first) > last:
            n -= 1
        return np.array([math.fma(k, step, first) for k in range(n + 1)])
    eps = np.finfo(float).eps
    tol = 2.0 * eps * max(abs(first), abs(last))
    n = round((last - first) / step)
    if first + n * step - last > tol:
        n -= 1
    end = first + n * step
    if end - last > -tol:
        end = last
    v = np.zeros(n + 1)
    k = np.arange(n // 2 + 1)
    v[k] = first + k * step
    v[n - k] = end - k * step
    if n % 2 == 0:
        v[n // 2] = (first + end) / 2
    return v


def linspace(
    first: float, last: float, n: int, kind: Literal["octave", "matlab"]
) -> FloatArray:
    """``linspace(first, last, n)`` as Octave 11 (arm64) or MATLAB builds it.

    Octave: the first half ``fma(i, delta, first)``, the second half
    ``fma(-(n-1-i), delta, last)``, ends exact; this reproduces the Octave
    run's bin edges exactly. MATLAB: ``first + (0:n-1).*(last-first)./(n-1)``
    with exact ends, from MATLAB's ``linspace.m`` as we understand it; not
    checked against a MATLAB run.
    """
    if kind == "octave":
        delta = (last - first) / (n - 1)
        half = n // 2
        v = np.array(
            [math.fma(i, delta, first) for i in range(half)]
            + [math.fma(-(n - 1 - i), delta, last) for i in range(half, n)]
        )
    else:
        v = first + np.arange(n) * (last - first) / (n - 1)
    v[0], v[-1] = first, last
    return v


def tsd_to_iv(
    times: FloatArray,
    above: NDArray[np.bool_],
    *,
    merge: float = 0.05,
    min_len: float = 0.05,
) -> FloatArray:
    """``TSDtoIV.m`` with method 'raw': runs of ``above``, gaps < ``merge`` joined, length > ``min_len``."""
    detec = np.r_[0, above.astype(np.int8), 0]
    dfs = np.diff(detec)
    up = np.flatnonzero(dfs == 1)
    down = np.flatnonzero(dfs == -1) - 1
    up_t, down_t = times[up], times[down]
    gap = up_t[1:] - down_t[:-1]
    merge_idx = np.flatnonzero(gap < merge)
    down_t = np.delete(down_t, merge_idx)
    up_t = np.delete(up_t, merge_idx + 1)
    keep = (down_t - up_t) > min_len
    return np.column_stack([up_t[keep], down_t[keep]])


def inside(times: FloatArray, rows: FloatArray) -> NDArray[np.bool_]:
    """``restrict.m`` for a ts or tsd: inside any closed ``[start, end]``."""
    keep = np.zeros(times.size, dtype=bool)
    for start, end in rows:
        keep |= (times >= start) & (times <= end)
    return keep


def linearize(xy: FloatArray, coord: FloatArray, pathlength: float) -> FloatArray:
    """``LinearizePos.m`` (nearest drawn point, 1-based) scaled as Generate_DecSeq.m does."""
    pts = np.asarray(coord, float)
    if pts.shape[0] != 2:
        pts = pts.T
    if pts[0, 0] == pts[0, -1] and pts[1, 0] == pts[1, -1]:
        pts = pts[:, :-1]
    # Octave's griddata 'nearest' is dsearchn: sqrt of the summed squared
    # differences, the first point kept on a tie. Ties happen: positions are
    # integer pixels over fixed factors, so take a few candidates and redo it.
    k = min(8, pts.shape[1])
    _, cand = cKDTree(pts.T).query(xy, k=k)
    cand = np.sort(cand, axis=1)
    dx = pts[0][cand] - xy[:, [0]]
    dy = pts[1][cand] - xy[:, [1]]
    dist = np.sqrt(dx * dx + dy * dy)
    nearest = cand[np.arange(cand.shape[0]), np.argmin(dist, axis=1)]
    return (nearest + 1) / pts.shape[1] * pathlength


def dxdt(
    times: FloatArray, x: FloatArray, *, window: float = 1.0, smoothing: float = 0.5
) -> FloatArray:
    """``dxdt.m``: best backward slope over 3..window samples, then a boxcar."""
    step = float(np.nanmean(np.diff(times)))
    n_window = int(np.ceil(window / step))
    best = np.full(x.size, np.inf)
    slope = np.zeros(x.size)

    def lagged(lag: int) -> FloatArray:
        return np.r_[np.full(lag, np.nan), x[:-lag]]

    for span in range(3, n_window + 1):
        candidate = (lagged(span) - x) / span
        error = (
            sum((lagged(k) - x + candidate * k) ** 2 for k in range(1, span + 1)) / span
        )
        better = error < best
        best[better] = error[better]
        slope[better] = candidate[better]
    n_smooth = int(np.ceil(smoothing / step))
    return conv_same(slope / step, np.ones(n_smooth) / n_smooth)


def histc_trimmed(values: FloatArray, edges: FloatArray) -> FloatArray:
    """``trim_histc(histc(values, edges))``: [e_i, e_i+1) bins, the last edge folded in."""
    counts = np.zeros(edges.size)
    ok = (values >= edges[0]) & (values <= edges[-1])
    idx = np.searchsorted(edges, values[ok], side="right") - 1
    np.add.at(counts, idx, 1)
    counts[-2] += counts[-1]
    return counts[:-1]


def tuning_curves(
    spikes: list[FloatArray],
    pos_times: FloatArray,
    linpos: FloatArray,
    edges: FloatArray,
) -> FloatArray:
    """``TuningCurves.m`` (1-D): smoothed spike counts over smoothed occupancy, NaN where unoccupied."""
    kernel = gausskernel(51, TC_SMOOTH_BINS)
    occ = histc_trimmed(linpos, edges)
    no_occ = occ < MIN_OCC
    occ = conv_same(occ, kernel)
    occ[no_occ] = np.nan
    occ = occ * OCC_DT
    tc = np.empty((len(spikes), edges.size - 1))
    for i, train in enumerate(spikes):
        z = np.interp(train, pos_times, linpos, left=np.nan, right=np.nan)
        z = z[(train >= pos_times[0]) & (train <= pos_times[-1])]
        hist = conv_same(histc_trimmed(z, edges), kernel)
        hist[no_occ] = np.nan
        tc[i] = hist / occ
    return tc


def make_q(spikes: list[FloatArray], edges: FloatArray, *, smooth: bool) -> FloatArray:
    """``MakeQfromS.m`` with dt 5 ms and a 5-bin boxcar; Gaussian (2 ms SD) first if ``smooth``. (n_cells, n_time)."""
    q = np.empty((len(spikes), edges.size - 1))
    gk = gausskernel(1 / Q_DT, Q_SMOOTH_SD / Q_DT) if smooth else None
    box = np.ones(Q_BOXCAR)
    for i, train in enumerate(spikes):
        row = histc_trimmed(train, edges)
        if gk is not None:
            row = conv_same(row, gk)
        q[i] = conv_same(row, box)
    return q


def decode_z(q: FloatArray, tc: FloatArray, binsize: float) -> FloatArray:
    """``DecodeZ.m``: uniform prior, ``nansum`` over cells, rows with total Q < 0.025 set NaN. (n_time, n_bins)."""
    n_bins = tc.shape[1]
    p = np.empty((q.shape[1], n_bins))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for b in range(n_bins):
            f = tc[:, b][:, None]
            term = np.log(f**q)
            prod = np.nansum(term, axis=0)
            rate = np.nansum(tc[:, b])
            p[:, b] = np.exp(prod) * np.exp(-binsize * rate) * (1.0 / n_bins)
        p = p / np.nansum(p, axis=1, keepdims=True)
    toss = q.sum(axis=0) < DT  # cfg.nMinSpikes = cfg.dt in Generate_DecSeqShuf.m
    p[toss] = np.nan
    return p


def map_bins(p: FloatArray) -> FloatArray:
    """``[~, z] = max(P.data)``, 1-based, first maximum, NaN where the row is all NaN."""
    allnan = np.all(np.isnan(p), axis=1)
    z = np.nanargmax(np.where(allnan[:, None], 0.0, p), axis=1).astype(float) + 1
    z[allnan] = np.nan
    return z


def detect_sequences(
    times: FloatArray, z: FloatArray, *, max_jump: int, min_length: int
) -> FloatArray:
    """``DecSeqDetectZ.m`` with nMaxNanSkipSequential = 0, nMaxNanSkipTotal = Inf."""
    tstart: list[float] = []
    tend: list[float] = []
    start: int | None = None
    prev = -np.inf
    skip = 0
    n = z.size
    for i in range(n):
        broken = False
        if np.isnan(z[i]):
            if start is not None:
                skip += 1
                if skip > 0:
                    broken = True
                    prev = np.nan
            else:
                prev = np.nan
        jump = abs(z[i] - prev)
        if jump <= max_jump:
            if start is None:
                start = i - 1
            skip = 0
        if not np.isnan(z[i]) and jump > max_jump:
            broken = True
        end_of_data = i == n - 1
        if broken or end_of_data:
            if start is not None:
                length = i - start - skip
                if end_of_data and not broken:
                    length += 1
                if length >= min_length:
                    tstart.append(times[start])
                    cur = skip - 1 if np.isnan(z[i]) else skip
                    adjust = 1 if (end_of_data and not broken) else 0
                    tend.append(times[i - 1 - cur + adjust])
            start = None
        if not np.isnan(z[i]):
            prev = z[i]
    return np.column_stack([tstart, tend]) if tstart else np.empty((0, 2))


def intersect_iv(iv: FloatArray, ev: FloatArray) -> FloatArray:
    """``IntersectIV.m``: an event starts or ends strictly inside, or strictly contains."""
    keep = [
        np.any((ev[:, 0] > s) & (ev[:, 0] < e))
        or np.any((ev[:, 1] > s) & (ev[:, 1] < e))
        or np.any((ev[:, 0] < s) & (ev[:, 1] > e))
        for s, e in iv
    ]
    return iv[np.array(keep, dtype=bool)] if len(iv) else iv


def difference_iv(iv: FloatArray, other: FloatArray) -> FloatArray:
    """``DifferenceIV.m``: drop rows that ``other`` touches (closed) or strictly contains."""
    keep = [
        not (
            np.any((other[:, 0] >= s) & (other[:, 0] <= e))
            or np.any((other[:, 1] >= s) & (other[:, 1] <= e))
            or np.any((other[:, 0] < s) & (other[:, 1] > e))
        )
        for s, e in iv
    ]
    return iv[np.array(keep, dtype=bool)] if len(iv) else iv


def authors_spike_trains(rec, folder) -> tuple[FloatArray, ...]:
    """``rec.spike_times`` as ``LoadSpikes.m`` reads them.

    ``LoadSpikes.m`` reads every MClust file as 32-bit; the files that hold
    64-bit times (their headers do not say so, a ``carey_2019`` watch-out)
    then give a spike at 0 s before each spike. The extra spikes lie outside
    every analysed interval, but they move ``firstSpike`` to 0 and with it
    the phase of the 5 ms decoding grid.
    """
    from pathlib import Path

    from neurospatial.datasets._neuralynx import (
        _spike_words,
        is_64_bit,
        read_spike_times,
    )

    files = sorted(list(Path(folder).glob("*.t")) + list(Path(folder).glob("*._t")))
    trains = list(rec.spike_times)
    for f in files:
        if not is_64_bit(_spike_words(f)):
            continue
        times = read_spike_times(f)
        unit = next(
            i
            for i, s in enumerate(trains)
            if s.size == times.size and np.array_equal(s, times)
        )
        trains[unit] = np.sort(np.r_[np.zeros(times.size), times])
    return tuple(trains)


def _neurospatial_sequences(
    q: FloatArray,
    tc: FloatArray,
    times: FloatArray,
    decoded: NDArray[np.bool_],
    pathlength: float,
    max_jump_bins: int,
) -> tuple[FloatArray, FloatArray]:
    """MAP and raw sequences from neurospatial's ``decode_position`` and ``replay_segments``.

    The environment is the code's 111 position bins along the trajectory;
    unoccupied bins (NaN tuning curves) get zero prior and rate, the
    package's way to leave a bin out. ``replay_segments`` runs within each
    run of decoded rows, as DecSeqDetectZ.m breaks at undecoded ones.
    """
    from neurospatial import Environment
    from neurospatial.decoding import decode_position
    from neurospatial.replay import replay_segments

    n_bins = tc.shape[1]
    env = Environment.linear_track(
        endpoints=[(0.0, 0.0), (pathlength, 0.0)],
        bin_size=pathlength / n_bins,
        units="cm",
    )
    supported = np.isfinite(tc).all(axis=0)
    prior = supported * np.asarray(env.bin_sizes, dtype=np.float64)
    rows = np.flatnonzero(decoded)
    z = np.full(times.size, np.nan)
    if rows.size == 0:
        return z, np.empty((0, 2))
    post = np.asarray(
        decode_position(
            env,
            q[:, rows].T,
            np.nan_to_num(tc, nan=0.0),
            dt=Q_DT,
            prior=prior / prior.sum(),
            times=times[rows],
            fractional_counts=True,
        ).posterior
    )
    z[rows] = np.argmax(post, axis=1) + 1.0
    width = float(env.bin_sizes[0])
    found: list[tuple[float, float]] = []
    for block in np.split(np.arange(rows.size), np.flatnonzero(np.diff(rows) != 1) + 1):
        if block.size < MIN_SEQ_LENGTH:
            continue
        seg = replay_segments(
            env,
            post[block],
            times[rows][block],
            max_jump=(max_jump_bins + 0.5) * width,
            method="map",
            min_time_bins=MIN_SEQ_LENGTH,
        )
        found += list(zip(seg.start_times, seg.end_times, strict=True))
    return z, (np.array(sorted(found)) if found else np.empty((0, 2)))


@dataclass
class Side:
    """One trajectory's decode, as Generate_DecSeqShuf.m's ``expCond`` holds it."""

    units: NDArray[np.intp]
    tc: FloatArray
    times: FloatArray
    map_z: FloatArray
    q_raw: FloatArray
    fine_counts: NDArray[np.int32]
    raw_sequences: FloatArray


def decode_side(
    spikes: tuple[FloatArray, ...],
    pos_times: FloatArray,
    pos_xy: FloatArray,
    coord: FloatArray,
    pathlength: float,
    trials: FloatArray,
    center_pb: FloatArray,
    chew: FloatArray | None,
    *,
    colon_kind: Literal["octave", "matlab"] = "octave",
    decoder: Literal["authors", "authors_no_tie", "neurospatial"] = "authors",
) -> Side:
    """Generate_DecSeqShuf.m for one side (shuffles off).

    ``pos_xy`` are the tracked samples in cm (frames with x or y of 0 dropped,
    as ``LoadPos`` does); ``trials`` the side's matched trials (GetMatchedTrials);
    ``chew`` R042's ``*times.mat`` intervals or R044's spiking seconds, else None.

    ``decoder`` sets what follows the shared preprocessing (units, tuning
    curves, spike density, the rows decoded): ``"authors"`` is DecodeZ.m and
    DecSeqDetectZ.m; ``"authors_no_tie"`` the same with unoccupied bins given
    no likelihood (the bin-one tie removed); ``"neurospatial"`` is
    ``decode_position`` (uniform prior over occupied bins, 5 ms exposure,
    fractional counts) and ``replay_segments`` with ``CAREY_2019``'s settings.
    """
    starts = trials[:, 0]
    # nearest_idx3(runStart, center_pb, 1): the first break strictly after, the
    # first break at or before center_pb[0], the last at or after center_pb[-1].
    idx = np.minimum(
        np.searchsorted(center_pb, starts, side="right"), center_pb.size - 1
    )
    idx = np.where(starts <= center_pb[0], 0, idx)
    trial_iv = np.column_stack([center_pb[idx] + TRIAL_START_OFFSET, trials[:, 1]])

    linpos = linearize(pos_xy, coord, pathlength)
    vx = dxdt(pos_times, pos_xy[:, 0])
    vy = dxdt(pos_times, pos_xy[:, 1])
    spd = np.sqrt(vx**2 + vy**2)
    run_iv = tsd_to_iv(pos_times, spd > RUN_SPEED)
    track1 = tsd_to_iv(pos_times, linpos > TRACK_EXCLUDE_START)
    track2 = tsd_to_iv(pos_times, linpos < pathlength - TRACK_EXCLUDE_END)

    keep_pos = inside(pos_times, track1)
    t_p, l_p = pos_times[keep_pos], linpos[keep_pos]
    keep2 = inside(t_p, track2)
    t_p, l_p = t_p[keep2], l_p[keep2]
    enc = [s[inside(s, track1)] for s in spikes]
    enc = [s[inside(s, track2)] for s in enc]
    for rows in (run_iv, trial_iv) + ((chew,) if chew is not None else ()):
        k = inside(t_p, rows)
        t_p, l_p = t_p[k], l_p[k]
        enc = [s[inside(s, rows)] for s in enc]
    nonempty = np.array([s.size > 0 for s in enc])
    counts = np.array([s.size for s in enc])
    units = np.flatnonzero(nonempty & (counts >= MIN_SPIKES))
    enc = [enc[u] for u in units]
    dec = [spikes[u] for u in units]

    edges_x = linspace(0.0, pathlength, N_EDGES, colon_kind)
    tc = tuning_curves(enc, t_p, l_p, edges_x)

    all_spk = np.concatenate(dec)
    edges_t = colon(float(all_spk.min()), Q_DT, float(all_spk.max()) + Q_DT, colon_kind)
    centers = edges_t[:-1] + Q_DT / 2
    q = make_q(dec, edges_t, smooth=True)
    q_raw = make_q(dec, edges_t, smooth=False)
    fine = np.vstack([histc_trimmed(s, edges_t) for s in dec]).astype(np.int32)
    n_neurons = np.sum(q_raw >= 1, axis=0)
    p = decode_z(q, tc, float(np.median(np.diff(centers))))
    p[n_neurons < N_MIN_NEURONS] = np.nan
    max_jump = round(MAX_JUMP_CM / (pathlength / N_EDGES))
    if decoder == "authors_no_tie":
        # Only decoded rows; a row whose occupied bins are all impossible (an
        # active cell with a zero rate everywhere it was seen) has no answer.
        decoded = ~np.isnan(p).all(axis=1)
        unoccupied = ~np.isfinite(tc).all(axis=0)
        p[np.ix_(decoded, unoccupied)] = 0.0
        p[decoded & ~(p > 0).any(axis=1)] = np.nan
    if decoder == "neurospatial":
        z, seqs = _neurospatial_sequences(
            q, tc, centers, ~np.isnan(p).all(axis=1), pathlength, max_jump
        )
    else:
        z = map_bins(p)
        seqs = detect_sequences(
            centers, z, max_jump=max_jump, min_length=MIN_SEQ_LENGTH
        )
    return Side(
        units=units,
        tc=tc,
        times=centers,
        map_z=z,
        q_raw=q_raw,
        fine_counts=fine,
        raw_sequences=seqs,
    )


def eligible(left: Side, right: Side, events: FloatArray) -> dict[str, FloatArray]:
    """getEligibleSequences.m (shuffles off): candidate overlap, >= 50 ms, >= 4 active cells, cross-side removal.

    Active cells are AddNActiveCellsIV.m's: units with a spike in a 5 ms bin
    whose centre lies in the sequence's closed interval.
    """
    kept = {}
    for name, side in (("left", left), ("right", right)):
        x = intersect_iv(side.raw_sequences, events)
        x = x[(x[:, 1] - x[:, 0]) >= MIN_LEN]
        if len(x):
            n_active = np.array(
                [
                    int(
                        np.sum(
                            side.fine_counts[
                                :, (side.times >= a) & (side.times <= b)
                            ].sum(axis=1)
                            > 0
                        )
                    )
                    for a, b in x
                ]
            )
            x = x[n_active >= MIN_ACTIVE_CELLS]
        kept[name] = x
    return {
        "left": difference_iv(kept["left"], kept["right"]),
        "right": difference_iv(kept["right"], kept["left"]),
    }


def session_sequences(
    rec,
    folder,
    events: FloatArray,
    *,
    colon_kind: Literal["octave", "matlab"] = "octave",
    decoder: Literal["authors", "authors_no_tie", "neurospatial"] = "authors",
) -> tuple[dict[str, FloatArray], dict[str, Side]]:
    """The authors' eligible sequences of one loaded ``carey_2019`` session.

    ``events`` are the paper's candidates ``[tstart, tend]``. R042 keeps only
    its ``*times.mat`` spans (``rec.epochs['spikes_kept']``); R044 only the
    seconds in which any unit fires (Generate_DecSeqShuf.m's ``chew_iv``).
    """
    spikes = authors_spike_trains(rec, folder)
    ok = np.isfinite(rec.positions).all(axis=1)
    chew = None
    if rec.session.startswith("R042"):
        chew = np.asarray(rec.epochs["spikes_kept"])
    elif rec.session.startswith("R044"):
        allspk = np.concatenate(spikes)
        edges = colon(float(allspk.min()), 1.0, float(allspk.max()) + 1.0, colon_kind)
        chew = tsd_to_iv(edges[:-1] + 0.5, histc_trimmed(allspk, edges) > 0.5)
    sides = {
        side: decode_side(
            spikes,
            rec.times[ok],
            rec.positions[ok],
            rec.params[f"{side}_trajectory"],
            float(rec.params["pathlength"]),
            np.asarray(rec.epochs[f"matched_{side}"]),
            np.asarray(rec.events["center_photobeam"]),
            chew,
            colon_kind=colon_kind,
            decoder=decoder,
        )
        for side in ("left", "right")
    }
    return eligible(sides["left"], sides["right"], events), sides
