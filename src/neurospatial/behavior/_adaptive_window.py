"""The Redish lab's adaptive-window rate estimator, used for VTE head sweeps.

Papale et al. (2012) computed the orientation of motion, Phi, as the
arctangent of a velocity from "an adaptive windowing of best-fit velocity
vectors (Janabi-Sharifi, Hayward, & Chen, 2000)", unwrapped it, took dPhi
"using the same Janabi-Sharifi algorithm", and integrated |dPhi| over the
pass. The lab's estimator is ``dxdt.m`` ("ADR 2003", "Based on
Janabi-Sharifi/Hayward/Chen ... But modified extensively"), called by the
lab's zIdPhi code with a 0.5 s window and 0.1 s smoothing. Copies read for
this port, none of them licensed for reuse, so the algorithm here is
written from their behavior and checked against ``dxdt.m`` in GNU Octave
on three synthetic paths, over samples away from the ends of the recording:

- ``andrewpapale/odorTrails`` @ 91954210ec: ``dxdt.m`` (with the
  "CORRECTED ADR 6 August 2012" sign fix) and ``zIdPhi.m``, whose 2012
  header (AndyP, ADR) sets ``dxdtWindow = 0.5`` and ``dxdtSmoothing = 0.1``.
- ``GWDiehl/mPFC-RRow-Recordings`` @ a0d1083214 (GPL-3.0):
  ``compute_IdPhi_AvgdPhi.m``, the same zIdPhi computation with those
  parameters: dx and dy by ``dxdt``, ``phi = atan2(dy, dx)``, ``unwrap``,
  ``dphi = dxdt(uphi)``, and ``IdPhi = sum(abs(dphi))`` over the pass.
- ``vandermeerlab/vandermeerlab`` @ 82ba3fe29c: ``dxdt.m`` before the sign
  fix, which does not change ``|dPhi|``.

At each sample the estimator tries every backward window of 3 to
``ceil(window / dt)`` samples, takes the slope between the window's two ends,
keeps the window with the smallest score, then averages the rates with a
boxcar ``ceil(smoothing / dt)`` samples long. The score is the lab's: the
mean of ``(x[i-k] - x[i] - slope * (t[i] - t[i-k]))**2`` over the window.
That is the squared distance of each earlier sample from the line through
``x[i]`` with slope ``-slope`` rather than ``slope``; on a moving signal it
grows with the window length, so the score picks short windows when the
signal changes and longer ones when it is flat. The lab's code has no
noise-bound parameter.

Four details are neurospatial's. The lab ran ``dxdt`` once over a whole
session, so its edges hardly mattered: here a sample with fewer than 3
earlier samples has no rate (the lab's code returns 0 there, a padding
artifact), and the boxcar averages the rates present instead of padding
with zeros. Slopes use the actual sample times, which equals the lab's
even-spacing arithmetic when the samples are evenly spaced. And the window
and boxcar sample counts are rounded before their ceiling, so float noise
in the sample interval does not add a sample (at exactly 30 Hz the lab's
``ceil(0.5 / dT)`` is 16 or 15 depending on rounding; at video rates such
as 29.97 Hz both give 15).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from neurospatial._stacklevel import _external_stacklevel

#: Longest backward window, in seconds, of the lab's zIdPhi code.
ADAPTIVE_WINDOW = 0.5
#: Length of the boxcar, in seconds, that averages the rates.
ADAPTIVE_SMOOTHING = 0.1
#: Entries of the samples-by-lags arrays that window selection holds at once.
_BLOCK_ELEMENTS = 2**18


def adaptive_window_rate(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    window: float = ADAPTIVE_WINDOW,
    smoothing: float = ADAPTIVE_SMOOTHING,
) -> NDArray[np.float64]:
    """Rate of change of one evenly sampled signal, per second.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Strictly increasing timestamps in seconds, from one continuous run.
    values : ndarray, shape (n_samples,)
        Finite values of the signal at ``times``.
    window : float, default=ADAPTIVE_WINDOW
        Longest backward window in seconds; ``ceil(window / dt)`` samples,
        with ``dt`` the mean sample interval.
    smoothing : float, default=ADAPTIVE_SMOOTHING
        Boxcar length in seconds, ``ceil(smoothing / dt)`` samples; 0 skips
        the smoothing. Both counts are rounded to 9 decimals before the
        ceiling, so float noise in ``dt`` does not add a sample.

    Returns
    -------
    ndarray, shape (n_samples,)
        Rate in value units per second; NaN where no rate was estimated
        (the first 3 samples, before smoothing, and all samples when the
        window holds fewer than 3 intervals).
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n_samples = values.size
    rate = np.full(n_samples, np.nan)
    if n_samples < 4:
        return rate
    dt = float(np.mean(np.diff(times)))
    longest = min(_samples_in(window, dt), n_samples - 1)
    widths = _best_windows(times, values, longest)
    chosen = np.flatnonzero(widths)
    start = chosen - widths[chosen]
    rate[chosen] = (values[chosen] - values[start]) / (times[chosen] - times[start])
    if smoothing > 0:
        rate = _boxcar(rate, _samples_in(smoothing, dt))
    return rate


def _best_windows(
    times: NDArray[np.float64], values: NDArray[np.float64], longest: int
) -> NDArray[np.intp]:
    """Length, in samples, of the backward window with the smallest score.

    The score of the window of ``width`` samples ending at sample ``i`` is
    the lab's: with ``slope`` the slope between the window's two ends, the
    mean over ``k = 1..width`` of
    ``(x[i-k] - x[i] - slope * (t[i] - t[i-k]))**2``, summed in that order.
    On a tie the shorter window is kept, as in the lab.

    Summing every window's score takes ``longest**2 / 2`` terms per sample.
    Instead each score is first bounded. Expanding the square gives
    ``(A + 2 * f * B + f**2 * C) / width`` with ``f = -slope`` and ``A``,
    ``B``, ``C`` the running sums over ``k`` of ``(x[i-k] - x[i])**2``,
    ``(x[i-k] - x[i]) * (t[i] - t[i-k])`` and ``(t[i] - t[i-k])**2``: one
    pass gives every width. This value and the sum in the lab's order
    round differently, by less than ``slack``, so the smallest score can
    only belong to a window whose lower bound is no higher than the
    smallest upper bound. Usually that is one window. Where it is several
    (ties and near-ties: whole-pixel tracking, pauses), those windows are
    summed in the lab's order and compared as before. The window chosen is
    therefore the one that summing every window chooses, ties included.

    Parameters
    ----------
    times, values : ndarray, shape (n_samples,)
        One run's sample times and signal.
    longest : int
        Longest window tried, in samples; the shortest is 3.

    Returns
    -------
    ndarray of int, shape (n_samples,)
        Window length per sample; 0 where no window has a score (fewer than
        3 earlier samples, or ``longest < 3``).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_samples = values.size
    widths = np.zeros(n_samples, dtype=np.intp)
    if longest < 3:
        return widths
    # Arrays below are (longest, n_block): lag k in row k - 1, one sample
    # per column, so sums over lags run down a column.
    lags = np.arange(1, longest + 1)[:, None]
    # Row i of each view holds samples i - longest .. i; before the run's
    # start the first sample is repeated, and those lags are never used.
    value_rows = sliding_window_view(
        np.concatenate([np.full(longest, values[0]), values]), longest + 1
    )
    time_rows = sliding_window_view(
        np.concatenate([np.full(longest, times[0]), times]), longest + 1
    )
    # Rounding of the expanded score and of the lab's sum: each adds
    # ``width`` rounded terms no larger than those of 2 * (A + f**2 * C), so
    # the two differ by less than (width + 5) * eps of it, doubled here
    # (measured: at most 0.14 of ``slack``). ``tiny`` covers underflow.
    eps, tiny = np.finfo(np.float64).eps, np.finfo(np.float64).tiny
    reach = 4 * (lags + 5) * eps / lags
    block = max(1, _BLOCK_ELEMENTS // longest)
    for first in range(3, n_samples, block):
        last = min(first + block, n_samples)
        samples = np.arange(first, last)
        # x[i-k] - x[i] and t[i] - t[i-k].
        behind = value_rows[first:last, -2::-1].T - values[first:last]
        elapsed = times[first:last] - time_rows[first:last, -2::-1].T
        fits = lags <= samples
        fits[:2] = False
        with np.errstate(over="ignore", invalid="ignore"):
            fall = behind / elapsed
            flat = _running_sum(behind * behind)
            cross = _running_sum(behind * elapsed)
            steep = _running_sum(elapsed * elapsed) * fall * fall
            size = flat + steep
            score = (size + 2 * fall * cross) / lags
            slack = reach * size + tiny
            # No score is below 0, so an exact 0 ends a sample's search.
            lowest, highest = np.maximum(score - slack, 0.0), score + slack
        # A sample with a bound that is not finite is summed in full.
        unsure = ~np.isfinite(highest).all(axis=0)
        lowest[:, unsure] = -np.inf
        highest[unsure | ~fits] = np.inf
        candidate = fits & (lowest <= highest.min(axis=0))

        # A lone candidate with finite bounds holds the smallest score.
        lone = (candidate.sum(axis=0) == 1) & ~unsure
        widths[samples[lone]] = candidate.argmax(axis=0)[lone] + 1
        best = np.full(samples.size, np.inf)
        columns = np.flatnonzero(~lone)
        while columns.size:
            # The shortest window of each sample that can still win.
            row = candidate[:, columns].argmax(axis=0)
            width = row + 1
            deepest = width.max()
            end = samples[columns]
            slope = (values[end] - values[end - width]) / (
                times[end] - times[end - width]
            )
            miss = behind[:deepest, columns] - slope * elapsed[:deepest, columns]
            # Added in the lab's order, k = 1..width.
            exact = _running_sum(miss * miss)[row, np.arange(columns.size)]
            exact /= width
            # Strictly smaller: on a tie the shorter window is kept.
            better = exact < best[columns]
            best[columns[better]] = exact[better]
            widths[end[better]] = width[better]
            candidate[row, columns] = False
            candidate[:, columns] &= lowest[:, columns] < best[columns]
            columns = columns[candidate[:, columns].any(axis=0)]
    return widths


def _running_sum(terms: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum down each column in place, row 0 first: row ``k`` holds rows 0 to ``k``.

    One vector addition per row, in the same order as a loop over lags
    (``numpy.cumsum`` gives the same sums several times slower).
    """
    for k in range(1, terms.shape[0]):
        terms[k] += terms[k - 1]
    return terms


def _samples_in(duration: float, dt: float) -> int:
    """``ceil(duration / dt)``, after rounding off float noise in ``dt``.

    At exactly 30 Hz, ``0.5 / (1 / 30)`` is 15.000000000000002, so a bare
    ceiling gives 16 samples or 15 depending on the clock's offset.
    """
    return int(np.ceil(round(duration / dt, 9)))


def _boxcar(rate: NDArray[np.float64], length: int) -> NDArray[np.float64]:
    """Mean of the finite rates in a boxcar placed as MATLAB's ``conv2 'same'``.

    Sample ``i`` averages samples ``i + length // 2 - length + 1`` through
    ``i + length // 2``, those that exist and have a rate; NaN where none do.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    ahead = length // 2
    padded = np.concatenate(
        [np.full(length - 1 - ahead, np.nan), rate, np.full(ahead, np.nan)]
    )
    windows = sliding_window_view(padded, length)
    present = np.isfinite(windows)
    count = present.sum(axis=1)
    total = np.where(present, windows, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def turn_rates(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    headings: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rate of change of the orientation of motion at each sample (dPhi).

    Within each observed run (missing samples skipped like deleted rows),
    the velocity's x and y come from :func:`adaptive_window_rate`, Phi is
    their arctangent, unwrapped over the run, and dPhi is the
    :func:`adaptive_window_rate` of Phi, as in the lab's zIdPhi code.

    Parameters
    ----------
    times : ndarray, shape (n_samples,)
        Strictly increasing timestamps in seconds.
    positions : ndarray, shape (n_samples, n_dims)
        Positions, ``n_dims >= 2``; x and y are used.
    interval_mask : ndarray of bool, shape (n_samples - 1,)
        ``ObservationSupport.interval_mask``; no estimate spans an unusable
        interval.
    headings : ndarray, shape (n_samples,), optional
        The caller's orientation in radians (NaN where not observed), used
        as Phi instead of the direction of the estimated velocity.

    Returns
    -------
    dphi : ndarray, shape (n_samples,)
        Rate of change of Phi in radians per second; NaN where none was
        estimated.
    dt : ndarray, shape (n_samples,)
        Mean sample interval, in seconds, of the samples dPhi was estimated
        from, so ``abs(dphi) * dt`` is the turn each sample contributes;
        NaN where ``dphi`` is.
    """
    from neurospatial._intervals import valid_run_samples

    finite = np.all(np.isfinite(positions[:, :2]), axis=1)
    dphi = np.full(times.size, np.nan)
    dt = np.full(times.size, np.nan)
    coarse: list[float] = []
    for run in valid_run_samples(np.asarray(interval_mask, dtype=bool), finite):
        if run.size >= 4:
            run_dt = float(np.mean(np.diff(times[run])))
            if _samples_in(ADAPTIVE_WINDOW, run_dt) < 3:
                coarse.append(run_dt)
                continue
        if headings is None:
            run_times = times[run]
            phi = np.arctan2(
                adaptive_window_rate(run_times, positions[run, 1]),
                adaptive_window_rate(run_times, positions[run, 0]),
            )
        else:
            phi = headings[run]
        observed = run[np.isfinite(phi)]
        if observed.size < 4:
            continue
        observed_times = times[observed]
        dphi[observed] = adaptive_window_rate(
            observed_times, np.unwrap(phi[np.isfinite(phi)])
        )
        dt[observed] = np.mean(np.diff(observed_times))
    if coarse:
        warnings.warn(
            f"{len(coarse)} run(s) of samples are about {np.median(coarse):.3g} s "
            "apart, so they have no head turn: the VTE velocity estimator's "
            f"{ADAPTIVE_WINDOW} s window holds fewer than 3 sample intervals.\n"
            "Why: IdPhi follows the Redish lab's adaptive-window estimator, "
            "which fits backward windows of at least 3 samples within "
            f"{ADAPTIVE_WINDOW} s.\n"
            "Fix: use tracking sampled faster than "
            f"{2 / ADAPTIVE_WINDOW:g} Hz for head sweeps.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
    dt[np.isnan(dphi)] = np.nan
    return dphi, dt
