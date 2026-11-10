"""The package's adaptive-window head-turn estimate, written a second way.

This is not the Redish lab's code and not an external reference. It is the
algorithm of ``neurospatial.behavior._adaptive_window``, from the same
reading of the lab's ``dxdt.m``, as a plain loop over samples: at sample
``i`` it tries the backward windows of 3 to ``ceil(window / dt)`` samples,
scores each by ``mean((x[i-k] - x[i] - slope * (t[i] - t[i-k]))**2)`` with
``slope`` the window's end-to-end slope, keeps the first smallest score's
slope, then averages ``ceil(smoothing / dt)`` rates in a boxcar placed as
MATLAB's ``conv2(..., 'same')``. The edge rules are the package's too: no
rate before 3 earlier samples, the boxcar averages the rates present, the
sample counts are rounded to 9 decimals before their ceiling, and the speed
gate is the package's.

So agreement with it checks the package's array code (the choice of window,
the boxcar, runs, gaps and the gate) against a loop that is easy to read. It
does not check that the algorithm is the lab's. The only values from the
lab's own code are in ``test_head_sweep_matches_the_redish_lab_code`` of
``test_vte.py``: IdPhi printed by GNU Octave running ``dxdt.m`` on three
synthetic paths, over samples away from the ends of the recording. Nothing
from the lab's code is stored for real tracking, for a pause of whole-pixel
tracking, or for the first samples of a run, where the package gives no
rate and the lab's code gives 0.
"""

import numpy as np

from ._reference_velocity import reference_velocity


def lab_rate(times, values, window=0.5, smoothing=0.1):
    """Rate of one run, per second, shape (n_samples,).

    The package's algorithm as a loop over samples (see the module
    docstring); the name says whose estimator it follows, not whose code
    produced the values.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    n = values.size
    raw = np.full(n, np.nan)
    if n < 4:
        return raw
    dt = np.mean(np.diff(times))
    longest = min(int(np.ceil(round(window / dt, 9))), n - 1)
    for i in range(3, n):
        best = np.inf
        for width in range(3, min(longest, i) + 1):
            slope = (values[i] - values[i - width]) / (times[i] - times[i - width])
            score = 0.0
            for k in range(1, width + 1):
                miss = values[i - k] - values[i] - slope * (times[i] - times[i - k])
                score += miss * miss
            score /= width
            if score < best:
                best, raw[i] = score, slope
    if smoothing == 0:
        return raw
    length = int(np.ceil(round(smoothing / dt, 9)))
    ahead = length // 2
    smoothed = np.full(n, np.nan)
    for i in range(n):
        box = raw[max(0, i + ahead - length + 1) : i + ahead + 1]
        box = box[np.isfinite(box)]
        if box.size:
            smoothed[i] = box.mean()
    return smoothed


def reference_turns(times, positions, min_speed=0.0, headings=None, speed=None):
    """``|dPhi| * dt`` of each sample of one run; NaN where slow or unestimated.

    Phi is the arctangent of the x and y rates (or ``headings``), unwrapped
    over its observed samples, and dPhi is its rate. The gate is ``speed``,
    by default the speed of the reference Gaussian line-fit velocity (0.1 s),
    as the package's.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if headings is None:
        phi = np.arctan2(
            lab_rate(times, positions[:, 1]), lab_rate(times, positions[:, 0])
        )
    else:
        phi = np.asarray(headings, dtype=float)
    observed = np.isfinite(phi)
    turns = np.full(times.size, np.nan)
    if observed.sum() >= 4:
        dt = np.mean(np.diff(times[observed]))
        dphi = lab_rate(times[observed], np.unwrap(phi[observed]))
        turns[observed] = np.abs(dphi) * dt
    if speed is None:
        velocity = reference_velocity(times, positions, 0.1)
        speed = np.hypot(velocity[:, 0], velocity[:, 1])
    turns[~(speed >= min_speed)] = np.nan
    return turns


def reference_head_sweep(
    times, positions, min_speed=0.0, headings=None, samples=slice(None), speed=None
):
    """IdPhi of ``samples`` of one run; NaN when none has a turn."""
    turns = reference_turns(times, positions, min_speed, headings, speed)[samples]
    turns = turns[np.isfinite(turns)]
    return float(turns.sum()) if turns.size else np.nan
