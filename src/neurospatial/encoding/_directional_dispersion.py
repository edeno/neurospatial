"""Spike-count overdispersion for the directional tuning test.

The occupancy-corrected tuning test treats spikes as independent given
heading (Poisson). Bursts and slow rate drift make spike counts vary more
than that, so the test rejects too often. This module estimates, per unit,
the dispersion ``φ = Var(count) / mean(count)`` beyond what heading explains,
from spike counts in time windows, so the test statistic can be divided by it
(a quasi-Poisson test).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft
from scipy.sparse import coo_matrix

from neurospatial.encoding._spatial_counts import FrameBinning

# Newton iterations for the cosine model; each converges quadratically, so
# this bound is reached only when spikes are perfectly separable by heading.
_MAX_NEWTON_STEPS = 100
_MAX_STEP_HALVINGS = 30
_N_COSINE_PARAMETERS = 3  # intercept, cos θ, sin θ


def heading_correlation_time(
    prepared: FrameBinning, bin_centers: NDArray[np.float64]
) -> float:
    """Two-sided integrated autocorrelation time of the heading vector, seconds.

    Parameters
    ----------
    prepared : FrameBinning
        Heading bin of every sample and the analyzed interval mask.
    bin_centers : ndarray of shape (n_bins,)
        Heading bin centers in radians.

    Returns
    -------
    float
        ``∫ C(τ) dτ`` over ``|τ| < L``, where ``C(τ)`` is the autocorrelation
        at time lag ``τ`` of the occupancy-centered heading unit vector over
        the analyzed time and ``L`` the first lag with ``C(L) <= 0``. Equals
        the sampling interval when headings are uncorrelated from sample to
        sample.

    Notes
    -----
    The heading is a step function of time: each analyzed interval holds the
    heading of the sample that starts it, and a missing sample holds the last
    valid heading, exactly as when its row is deleted. The step function is
    read at the midpoints of a regular grid whose step ``δ`` is the median
    interval between observed samples, and ``∫ C(τ) dτ`` is the sum
    ``δ (1 + 2 Σ_{l=1}^{L/δ-1} C(lδ))``. For regularly sampled headings the
    grid points are the samples themselves. The estimate depends only on the
    heading's time course, so a NaN row and a deleted row give the same
    value, and irregular sampling is measured in seconds, not rows.
    """
    times = prepared.times
    observed = times if prepared.sample_valid is None else times[prepared.sample_valid]
    # Without two observed samples there is no sampling interval.
    step = float(np.median(np.diff(observed))) if observed.size > 1 else np.nan
    valid = prepared.support.mask & (prepared.frame_bins[:-1] >= 0)
    if not valid.any():
        return step
    intervals = np.flatnonzero(valid)
    start, stop = times[intervals[0]], times[intervals[-1] + 1]
    midpoints = start + (np.arange(int(np.ceil((stop - start) / step))) + 0.5) * step
    midpoints = midpoints[midpoints < stop]
    # The interval that holds each grid point, and whether it is analyzed.
    interval = np.searchsorted(times, midpoints, side="right") - 1
    on_support = valid[interval]
    unit = np.stack([np.cos(bin_centers), np.sin(bin_centers)], axis=1)
    durations = np.diff(times)[intervals]
    mean = durations @ unit[prepared.frame_bins[intervals]] / durations.sum()
    bins = np.where(on_support, prepared.frame_bins[interval], 0)
    centered = (unit[bins] - mean) * on_support[:, None]  # (n_grid, 2)

    n = len(centered)
    n_fft = next_fast_len(2 * n - 1, real=True)
    spectrum = rfft(centered, n_fft, axis=0)
    lagged = irfft((spectrum * spectrum.conj()).real.sum(axis=1), n_fft)[:n]
    if lagged[0] <= 0:
        return step
    if on_support.all():
        n_pairs = np.arange(n, 0, -1, dtype=np.float64)
    else:
        weights = rfft(on_support.astype(np.float64), n_fft)
        n_pairs = irfft((weights * weights.conj()).real, n_fft)[:n]
    # Lags with no analyzed pair carry no information; treat them as zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(n_pairs > 0.5, lagged / n_pairs, 0.0)
    correlation = correlation / correlation[0]
    nonpositive = np.flatnonzero(correlation[1:] <= 0)
    end = int(nonpositive[0]) + 1 if nonpositive.size else n
    return step * (1.0 + 2.0 * float(correlation[1:end].sum()))


def cosine_model_rates(
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bin_centers: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Maximum-likelihood rates of ``log rate(θ) = a + b cos θ + c sin θ``.

    Parameters
    ----------
    spike_counts : ndarray of shape (n_units, n_bins)
        Unsmoothed spike counts per heading bin.
    occupancy : ndarray of shape (n_bins,)
        Time per heading bin, seconds (the Poisson exposure).
    bin_centers : ndarray of shape (n_bins,)
        Heading bin centers in radians.

    Returns
    -------
    ndarray of shape (n_units, n_bins)
        Fitted rate in Hz at every bin center. Units without spikes are NaN.
    """
    used = occupancy > 0
    centers = bin_centers[used]
    design = np.stack(
        [np.ones_like(centers), np.cos(centers), np.sin(centers)], axis=1
    )  # (n_used, 3)
    exposure = occupancy[used]
    counts = spike_counts[:, used]
    n_spikes = counts.sum(axis=1)
    has_spikes = n_spikes > 0
    beta = np.zeros((len(counts), _N_COSINE_PARAMETERS))
    beta[has_spikes, 0] = np.log(n_spikes[has_spikes] / exposure.sum())

    def log_likelihood(
        y: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Each unit's log-likelihood and a bound on its rounding error."""
        eta = b @ design.T
        terms = y * eta - exposure * np.exp(eta)
        rounding = len(exposure) * np.finfo(float).eps * np.abs(terms).sum(axis=1)
        return terms.sum(axis=1), rounding

    # Each unit is fitted on its own and stops when its own step is small,
    # so a unit's fit does not depend on the other units in the call.
    fitting = has_spikes.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(_MAX_NEWTON_STEPS):
            units = np.flatnonzero(fitting)
            if units.size == 0:
                break
            y, b = counts[units], beta[units]
            mu = exposure * np.exp(b @ design.T)  # (n_fit, n_used)
            gradient = (y - mu) @ design
            hessian = np.einsum("kb,bi,bj->kij", mu, design, design)
            step = (np.linalg.pinv(hessian) @ gradient[:, :, None])[:, :, 0]
            current, rounding = log_likelihood(y, b)
            scale = np.ones(len(units))
            # Halve each unit's step until its likelihood does not decrease by
            # more than rounding. Near the maximum a Newton step changes the
            # likelihood by less than its rounding error; rejecting those
            # steps would stop the fit short of the maximum, at a point set by
            # rounding noise.
            for _ in range(_MAX_STEP_HALVINGS):
                candidate, _ = log_likelihood(y, b + scale[:, None] * step)
                worse = ~(candidate >= current - rounding)
                if not worse.any():
                    break
                scale[worse] *= 0.5
            change = scale[:, None] * step
            beta[units] = b + change
            fitting[units] = np.any(np.abs(change) > 1e-10, axis=1)
    full_design = np.stack(
        [np.ones_like(bin_centers), np.cos(bin_centers), np.sin(bin_centers)], axis=1
    )
    with np.errstate(over="ignore"):
        rates: NDArray[np.float64] = np.exp(beta @ full_design.T)
    rates[~has_spikes] = np.nan
    return rates


def spike_count_dispersion(
    spike_trains: Sequence[NDArray[np.float64]],
    prepared: FrameBinning,
    spike_counts: NDArray[np.float64],
    occupancy: NDArray[np.float64],
    bin_centers: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float, int | None]:
    """Pearson dispersion of window spike counts around the cosine model.

    Parameters
    ----------
    spike_trains : sequence of ndarray of shape (n_spikes_i,)
        Spike times in seconds, one array per unit.
    prepared : FrameBinning
        Heading bin of every sample and the analyzed interval mask; the same
        support that produced ``spike_counts`` and ``occupancy``.
    spike_counts : ndarray of shape (n_units, n_bins)
        Unsmoothed spike counts per heading bin.
    occupancy : ndarray of shape (n_bins,)
        Time per heading bin, seconds.
    bin_centers : ndarray of shape (n_bins,)
        Heading bin centers in radians.

    Returns
    -------
    dispersion : ndarray of shape (n_units,)
        ``φ = Σ_w (y_w - μ_w)² / μ_w / dof``; NaN for a unit without spikes
        or when fewer than four windows hold analyzed time.
    window : float
        Window length in seconds, :func:`heading_correlation_time`.
    dof : int or None
        Number of windows with analyzed time minus the three cosine-model
        parameters; None when that is less than 1.

    Notes
    -----
    Time is cut into windows of length ``Δ`` starting at the first analyzed
    time; windows without analyzed time are skipped. ``y_w`` is the number
    of counted spikes inside window ``w`` and ``μ_w = ∫_w r(θ(t)) dt`` its
    expected count under the fitted cosine model ``r``
    (:func:`cosine_model_rates`), integrated exactly over the window's
    analyzed time, so headings may change within a window and a sampling
    interval that crosses a window edge is split at the edge. Windows are
    defined in time, so a NaN row and a deleted row give the same counts.

    **Window length.** The tuning statistic's score is
    ``S = Σ_k N_k g_k``, a sum over intervals of spike counts ``N_k`` times
    the centered heading vector ``g_k``. If spikes have excess pair
    correlation ``ρ(τ)`` (bursts, drift) at mean rate ``λ``, then
    ``Var(S) ≈ λ M (1 + λ ∫ ρ(τ) C(τ) dτ)``, with ``M`` the Poisson
    variance and ``C(τ)`` the normalized autocorrelation of ``g``. The Pearson
    dispersion of counts in windows of length ``Δ`` estimates
    ``1 + λ ∫ ρ(τ) (1 - |τ|/Δ)₊ dτ``. Correlations much shorter than both
    kernels (bursts) enter both equally; for correlations slower than both
    (drift), the two agree when the kernels have equal area,
    ``Δ = ∫ C(τ) dτ``, which is :func:`heading_correlation_time`. Shorter
    windows miss slow correlations; longer ones over-correct and cost power.
    """
    times = prepared.times
    valid = prepared.support.mask & (prepared.frame_bins[:-1] >= 0)
    n_units = len(spike_trains)
    window = heading_correlation_time(prepared, bin_centers)
    intervals = np.flatnonzero(valid)
    if intervals.size == 0:
        return np.full(n_units, np.nan), window, None
    starts, stops = times[intervals], times[intervals + 1]
    n_edges = int(np.ceil((stops[-1] - starts[0]) / window)) + 2
    edges = starts[0] + window * np.arange(n_edges)
    # Each analyzed interval is split into one piece per window it overlaps;
    # every piece has positive length.
    first = np.searchsorted(edges, starts, side="right") - 1
    last = np.searchsorted(edges, stops, side="left") - 1
    n_pieces = last - first + 1
    piece_interval = np.repeat(intervals, n_pieces)
    offsets = np.arange(n_pieces.sum()) - np.repeat(
        np.cumsum(n_pieces) - n_pieces, n_pieces
    )
    piece_window = np.repeat(first, n_pieces) + offsets
    piece_length = np.minimum(
        times[piece_interval + 1], edges[piece_window + 1]
    ) - np.maximum(times[piece_interval], edges[piece_window])
    # Renumber the windows that hold analyzed time 0, 1, 2, ...
    windows, piece_window = np.unique(piece_window, return_inverse=True)
    n_windows = len(windows)
    dof = n_windows - _N_COSINE_PARAMETERS
    if dof < 1:
        return np.full(n_units, np.nan), window, None
    # Seconds spent in each heading bin during each window.
    window_occupancy = coo_matrix(
        (piece_length, (piece_window, prepared.frame_bins[piece_interval])),
        shape=(n_windows, len(bin_centers)),
    ).tocsr()
    rates = cosine_model_rates(spike_counts, occupancy, bin_centers)

    dispersion = np.full(n_units, np.nan)
    for unit, spikes in enumerate(spike_trains):
        if not np.isfinite(rates[unit]).all():
            continue
        expected = window_occupancy @ rates[unit]
        spikes = spikes[(spikes >= times[0]) & (spikes < times[-1])]
        spikes = spikes[valid[np.searchsorted(times, spikes, "right") - 1]]
        # A spike in an analyzed interval lies in one of that interval's
        # pieces, so its window holds analyzed time.
        spike_windows = np.searchsorted(
            windows, np.searchsorted(edges, spikes, side="right") - 1
        )
        observed = np.bincount(spike_windows, minlength=n_windows).astype(np.float64)
        # A window the model expects no spikes in adds nothing if it has none.
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = (observed - expected) ** 2 / expected
        pearson = float(np.sum(np.where(observed + expected > 0, terms, 0.0)))
        dispersion[unit] = pearson / dof
    return dispersion, window, dof
