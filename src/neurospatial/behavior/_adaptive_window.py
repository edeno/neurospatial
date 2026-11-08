"""The Redish lab's adaptive-window rate estimator, used for VTE head sweeps.

Papale et al. (2012) computed the orientation of motion, Phi, as the
arctangent of a velocity from "an adaptive windowing of best-fit velocity
vectors (Janabi-Sharifi, Hayward, & Chen, 2000)", unwrapped it, took dPhi
"using the same Janabi-Sharifi algorithm", and integrated |dPhi| over the
pass. The lab's estimator is ``dxdt.m`` ("ADR 2003", "Based on
Janabi-Sharifi/Hayward/Chen ... But modified extensively"), called by the
lab's zIdPhi code with a 0.5 s window and 0.1 s smoothing. Copies read for
this port, none of them licensed for reuse, so the algorithm here is
written from their behavior and checked against them in GNU Octave:

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
signal changes and longer ones when it is flat. Janabi-Sharifi et al.
instead take the longest window whose line stays within a noise bound of
every sample; the lab's code has no noise bound.

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
    best = np.full(n_samples, np.inf)
    for width in range(3, longest + 1):
        # Samples i >= width: slope between x[i - width] and x[i].
        end = values[width:]
        slope = (end - values[:-width]) / (times[width:] - times[:-width])
        score = np.zeros(n_samples - width)
        # Summed in the lab's order, k = 1..width, so near-ties break alike.
        for k in range(1, width + 1):
            lagged = values[width - k : n_samples - k]
            elapsed = times[width:] - times[width - k : n_samples - k]
            miss = lagged - end - slope * elapsed
            score += miss * miss
        score /= width
        # Strictly smaller: on a tie the shorter window is kept, as in the lab.
        better = score < best[width:]
        best[width:][better] = score[better]
        rate[width:][better] = slope[better]
    if smoothing > 0:
        rate = _boxcar(rate, _samples_in(smoothing, dt))
    return rate


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
