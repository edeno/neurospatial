"""Vicarious Trial and Error (VTE) detection and analysis.

VTE refers to hesitation behavior at decision points, characterized by:
- **Head sweeping**: Looking back and forth between options (high IdPhi)
- **Pausing**: Slowing down or stopping at the choice point (low speed)

These behaviors are thought to reflect deliberative decision-making,
as opposed to habitual or reflexive choices.

Terminology
-----------
- **IdPhi** (Integrated absolute head rotation): Sum of absolute heading changes
  in a time window. High values indicate "scanning" behavior.
- **zIdPhi**: Z-scored IdPhi relative to session baseline. Standardizes across
  animals and sessions for comparison.
- **VTE index**: Combined measure of head sweeping and slowing.

Example
-------
>>> from neurospatial.metrics import compute_vte_session
>>> result = compute_vte_session(
...     positions,
...     times,
...     trials,
...     decision_region="center",
...     window_duration=1.0,
... )
>>> print(f"VTE trials: {result.n_vte_trials}/{len(result.trial_results)}")
>>> for trial in result.trial_results:
...     if trial.is_vte:
...         print(
...             f"  Trial at {trial.window_end:.1f}s: IdPhi={trial.head_sweep_magnitude:.2f}"
...         )

References
----------
.. [1] Redish, A. D. (2016). Vicarious trial and error. Nat Rev Neurosci.
       DOI: 10.1038/nrn.2015.30
.. [2] Papale, A. E., et al. (2012). Interplay between hippocampal sharp-wave
       ripple events and vicarious trial and error behaviors. Neuron.
       DOI: 10.1016/j.neuron.2012.10.018
.. [3] Muenzinger, K. F. (1938). Vicarious trial and error at a point of choice.
       J Genet Psychol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from neurospatial.environment import Environment
    from neurospatial.segmentation.trials import Trial

__all__ = [
    "VTESessionResult",
    "VTETrialResult",
    "classify_vte",
    "compute_vte_index",
    "compute_vte_session",
    "compute_vte_trial",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
    "integrated_absolute_rotation",
    "normalize_vte_scores",
    "wrap_angle",
]


# =============================================================================
# Utility Functions
# =============================================================================


def wrap_angle(angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angle to (-pi, pi].

    Parameters
    ----------
    angle : NDArray[np.float64]
        Angles in radians.

    Returns
    -------
    NDArray[np.float64]
        Angles wrapped to (-pi, pi].

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.vte import wrap_angle
    >>> wrap_angle(np.array([3 * np.pi / 2]))
    array([-1.57079633])
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Core VTE Functions
# =============================================================================


def head_sweep_magnitude(headings: NDArray[np.float64]) -> float:
    """Compute integrated absolute head rotation (IdPhi).

    Sums the absolute value of heading changes across a trajectory window.
    High values indicate back-and-forth head movements ("scanning").

    Parameters
    ----------
    headings : NDArray[np.float64], shape (n_samples,)
        Heading angles in radians.

    Returns
    -------
    float
        Sum of absolute heading changes in radians.
        Returns 0.0 if fewer than 2 valid samples.

    Notes
    -----
    This is the core VTE metric from Papale et al. (2012) and Redish (2016).
    Also known as "IdPhi" (integrated absolute dphi/dt * dt).

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.vte import head_sweep_magnitude
    >>> # Constant heading: no head sweep
    >>> head_sweep_magnitude(np.zeros(10))
    0.0
    >>> # Alternating headings: high head sweep
    >>> headings = np.array([0, np.pi / 4, 0, np.pi / 4, 0])
    >>> head_sweep_magnitude(headings)  # 4 * pi/4 = pi
    3.141592653589793
    """
    # Filter NaN values
    valid_headings = headings[~np.isnan(headings)]

    if len(valid_headings) < 2:
        return 0.0

    delta = wrap_angle(np.diff(valid_headings))
    return float(np.sum(np.abs(delta)))


# Alias for backward compatibility and paper terminology
integrated_absolute_rotation = head_sweep_magnitude


def head_sweep_from_positions(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    min_speed: float = 5.0,
) -> float:
    """Compute head sweep magnitude (IdPhi) from position trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
        Stationary periods are excluded.

    Returns
    -------
    float
        Sum of absolute heading changes (radians).
        Returns 0.0 if fewer than 2 valid heading samples.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.vte import head_sweep_from_positions
    >>> # Straight line trajectory (low head sweep)
    >>> times = np.linspace(0, 2, 20)
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.ones(20) * 50])
    >>> head_sweep_from_positions(positions, times, min_speed=1.0)  # doctest: +SKIP
    0.0
    """
    from neurospatial.reference_frames import heading_from_velocity

    if len(positions) < 2:
        return 0.0

    # Compute dt from times (heading_from_velocity expects scalar dt)
    # Use median dt to handle irregular sampling
    dt = float(np.median(np.diff(times)))

    # Get headings
    headings = heading_from_velocity(positions, dt, min_speed=min_speed)

    return head_sweep_magnitude(headings)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class VTETrialResult:
    """VTE metrics for a single trial.

    Attributes
    ----------
    head_sweep_magnitude : float
        Sum of |delta_theta| in pre-decision window (radians).
        Also known as IdPhi (Integrated absolute Phi).
        High values indicate looking back and forth between options.
    z_head_sweep : float or None
        Z-scored head sweep magnitude relative to session baseline.
        None if session statistics not computed (single trial analysis).
    mean_speed : float
        Mean speed in pre-decision window (environment units/s).
    min_speed : float
        Minimum speed in pre-decision window (units/s).
        Near-zero indicates a pause.
    z_speed_inverse : float or None
        Z-scored inverse speed (higher = slower, more VTE-like).
        None if session statistics not computed.
    vte_index : float or None
        Combined VTE index: alpha * z_head_sweep + (1-alpha) * z_speed_inverse.
        None if session statistics not computed.
    is_vte : bool or None
        True if vte_index > threshold. None if not classified.
    window_start : float
        Start time of analysis window (seconds).
    window_end : float
        End time of window (decision region entry time, seconds).
    """

    head_sweep_magnitude: float
    z_head_sweep: float | None
    mean_speed: float
    min_speed: float
    z_speed_inverse: float | None
    vte_index: float | None
    is_vte: bool | None
    window_start: float
    window_end: float

    # Aliases for common terminology
    @property
    def idphi(self) -> float:
        """Alias for head_sweep_magnitude (IdPhi terminology)."""
        return self.head_sweep_magnitude

    @property
    def z_idphi(self) -> float | None:
        """Alias for z_head_sweep (zIdPhi terminology)."""
        return self.z_head_sweep

    def summary(self) -> str:
        """Human-readable summary.

        Returns
        -------
        str
            Formatted string with trial VTE metrics.
        """
        vte_str = (
            "VTE"
            if self.is_vte
            else "non-VTE"
            if self.is_vte is not None
            else "unclassified"
        )
        return (
            f"Trial [{self.window_start:.1f}-{self.window_end:.1f}s]: "
            f"IdPhi={self.head_sweep_magnitude:.2f} rad, "
            f"speed={self.mean_speed:.1f}, {vte_str}"
        )


@dataclass(frozen=True)
class VTESessionResult:
    """VTE analysis for an entire session.

    Attributes
    ----------
    trial_results : list[VTETrialResult]
        Per-trial VTE metrics with z-scores computed.
    mean_head_sweep : float
        Session mean of head sweep magnitude (for z-scoring).
    std_head_sweep : float
        Session std of head sweep magnitude.
    mean_speed : float
        Session mean of pre-decision mean speed.
    std_speed : float
        Session std of pre-decision mean speed.
    n_vte_trials : int
        Number of trials classified as VTE.
    vte_fraction : float
        Fraction of trials classified as VTE.
    """

    trial_results: list[VTETrialResult]
    mean_head_sweep: float
    std_head_sweep: float
    mean_speed: float
    std_speed: float
    n_vte_trials: int
    vte_fraction: float

    # Aliases for common terminology
    @property
    def mean_idphi(self) -> float:
        """Alias for mean_head_sweep (IdPhi terminology)."""
        return self.mean_head_sweep

    @property
    def std_idphi(self) -> float:
        """Alias for std_head_sweep (IdPhi terminology)."""
        return self.std_head_sweep

    def summary(self) -> str:
        """Human-readable summary.

        Returns
        -------
        str
            Formatted string with session VTE summary.
        """
        return (
            f"VTE session: {self.n_vte_trials}/{len(self.trial_results)} "
            f"VTE trials ({self.vte_fraction:.1%})\n"
            f"  Head sweep: mean={self.mean_head_sweep:.2f}, std={self.std_head_sweep:.2f}\n"
            f"  Speed: mean={self.mean_speed:.1f}, std={self.std_speed:.1f}"
        )

    def get_vte_trials(self) -> list[VTETrialResult]:
        """Return only trials classified as VTE.

        Returns
        -------
        list[VTETrialResult]
            List of trials where is_vte is True.
        """
        return [t for t in self.trial_results if t.is_vte]


# =============================================================================
# Z-Scoring Functions
# =============================================================================


def normalize_vte_scores(
    head_sweeps: NDArray[np.float64],
    speeds: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score VTE metrics across trials.

    Parameters
    ----------
    head_sweeps : NDArray[np.float64], shape (n_trials,)
        Head sweep magnitude for each trial.
    speeds : NDArray[np.float64], shape (n_trials,)
        Mean speed for each trial.

    Returns
    -------
    z_head_sweeps : NDArray[np.float64], shape (n_trials,)
        Z-scored head sweeps.
    z_speed_inverse : NDArray[np.float64], shape (n_trials,)
        Z-scored inverse speed (higher = slower = more VTE-like).

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Warns
    -----
    UserWarning
        If std is zero (no variation across trials), z-scores will be 0.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.vte import normalize_vte_scores
    >>> head_sweeps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> speeds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> z_hs, z_sp = normalize_vte_scores(head_sweeps, speeds)
    >>> np.abs(np.mean(z_hs)) < 1e-10  # Mean is ~0
    True
    """
    if len(head_sweeps) != len(speeds):
        raise ValueError(
            f"head_sweeps and speeds must have same length. "
            f"Got {len(head_sweeps)} and {len(speeds)}."
        )

    # Z-score head sweeps
    mean_hs = np.mean(head_sweeps)
    std_hs = np.std(head_sweeps)
    if std_hs < 1e-10:
        warnings.warn(
            "No variation in head sweep magnitude across trials (std=0). "
            "All trials have identical head sweep behavior. "
            "Z-scores will be 0, and VTE classification may not be meaningful. "
            "Consider adjusting window_duration or min_speed parameters.",
            stacklevel=2,
        )
        z_head_sweeps = np.zeros_like(head_sweeps)
    else:
        z_head_sweeps = (head_sweeps - mean_hs) / std_hs

    # Z-score inverse speed (invert so slower = higher score)
    mean_spd = np.mean(speeds)
    std_spd = np.std(speeds)
    if std_spd < 1e-10:
        warnings.warn(
            "No variation in speed across trials (std=0). "
            "All trials have identical speed behavior. "
            "Z-scores will be 0, and VTE classification may not be meaningful. "
            "Consider adjusting window_duration or min_speed parameters.",
            stacklevel=2,
        )
        z_speed_inverse = np.zeros_like(speeds)
    else:
        # Invert: higher speed -> lower z, lower speed -> higher z
        z_speed_inverse = -(speeds - mean_spd) / std_spd

    return z_head_sweeps, z_speed_inverse


# =============================================================================
# Classification Functions
# =============================================================================


def compute_vte_index(
    z_head_sweep: float,
    z_speed_inv: float,
    *,
    alpha: float = 0.5,
) -> float:
    """Compute combined VTE index.

    Parameters
    ----------
    z_head_sweep : float
        Z-scored head sweep magnitude.
    z_speed_inv : float
        Z-scored inverse speed.
    alpha : float, default=0.5
        Weight for head sweep (1-alpha for speed).
        Default 0.5 weights both equally.

    Returns
    -------
    float
        Combined VTE index.

    Examples
    --------
    >>> from neurospatial.metrics.vte import compute_vte_index
    >>> compute_vte_index(1.0, 1.0, alpha=0.5)
    1.0
    >>> compute_vte_index(2.0, 0.0, alpha=1.0)  # Head sweep only
    2.0
    """
    return alpha * z_head_sweep + (1 - alpha) * z_speed_inv


def classify_vte(
    vte_index: float,
    *,
    threshold: float = 0.5,
) -> bool:
    """Classify trial as VTE based on VTE index.

    Parameters
    ----------
    vte_index : float
        Combined VTE index.
    threshold : float, default=0.5
        Classification threshold.
        Trial is VTE if vte_index > threshold.

    Returns
    -------
    bool
        True if trial is classified as VTE.

    Examples
    --------
    >>> from neurospatial.metrics.vte import classify_vte
    >>> classify_vte(1.0, threshold=0.5)
    True
    >>> classify_vte(0.3, threshold=0.5)
    False
    """
    return vte_index > threshold


# =============================================================================
# Composite Functions
# =============================================================================


def compute_vte_trial(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    entry_time: float,
    window_duration: float,
    *,
    min_speed: float = 5.0,
) -> VTETrialResult:
    """Compute VTE metrics for a single trial.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates for entire trajectory.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire trajectory.
    entry_time : float
        Time of decision region entry (seconds).
    window_duration : float
        Duration of pre-decision window (seconds).
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).

    Returns
    -------
    VTETrialResult
        VTE metrics for the trial.
        Note: z-scores and classification are None for single trial analysis.
        Use compute_vte_session() for session-level analysis with z-scoring.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.metrics.vte import compute_vte_trial
    >>> times = np.linspace(0, 3, 90)
    >>> positions = np.column_stack([np.linspace(0, 50, 90), np.ones(90) * 50])
    >>> result = compute_vte_trial(
    ...     positions, times, entry_time=2.0, window_duration=1.0
    ... )
    >>> result.window_start
    1.0
    >>> result.window_end
    2.0
    """
    from neurospatial.metrics.decision_analysis import extract_pre_decision_window

    # Extract pre-decision window
    window_positions, window_times = extract_pre_decision_window(
        positions, times, entry_time, window_duration
    )

    window_start = entry_time - window_duration
    window_end = entry_time

    # Compute head sweep magnitude
    head_sweep = head_sweep_from_positions(
        window_positions, window_times, min_speed=min_speed
    )

    # Compute speed statistics
    if len(window_positions) < 2:
        mean_spd = 0.0
        min_spd = 0.0
    else:
        dt = np.diff(window_times)
        velocity = np.diff(window_positions, axis=0) / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocity, axis=1)
        mean_spd = float(np.mean(speeds))
        min_spd = float(np.min(speeds))

    return VTETrialResult(
        head_sweep_magnitude=head_sweep,
        z_head_sweep=None,  # Not computed for single trial
        mean_speed=mean_spd,
        min_speed=min_spd,
        z_speed_inverse=None,
        vte_index=None,
        is_vte=None,
        window_start=window_start,
        window_end=window_end,
    )


def compute_vte_session(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    trials: list[Trial],
    decision_region: str,
    env: Environment,
    *,
    window_duration: float = 1.0,
    min_speed: float = 5.0,
    alpha: float = 0.5,
    vte_threshold: float = 0.5,
) -> VTESessionResult:
    """Compute VTE metrics for all trials in a session.

    Parameters
    ----------
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates for entire session.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session.
    trials : list[Trial]
        Trial segmentation from segment_trials().
    decision_region : str
        Name of decision region in env.regions.
    env : Environment
        Environment with region definitions.
    window_duration : float, default=1.0
        Duration of pre-decision window in seconds.
        Typical values: 0.5-2.0s depending on maze size and task.
    min_speed : float, default=5.0
        Minimum speed for valid heading (units/s).
    alpha : float, default=0.5
        Weight for head sweep in VTE index (1-alpha for speed).
        Default 0.5 weights both equally.
    vte_threshold : float, default=0.5
        Threshold for VTE classification.
        Trial is VTE if vte_index > threshold.

    Returns
    -------
    VTESessionResult
        Session-level VTE analysis with per-trial metrics.

    Examples
    --------
    >>> from neurospatial.metrics import compute_vte_session
    >>> result = compute_vte_session(
    ...     positions,
    ...     times,
    ...     trials,
    ...     decision_region="center",
    ...     env=env,
    ...     window_duration=1.0,
    ... )  # doctest: +SKIP
    >>> print(
    ...     f"VTE trials: {result.n_vte_trials}/{len(result.trial_results)}"
    ... )  # doctest: +SKIP
    """
    from neurospatial.metrics.decision_analysis import (
        decision_region_entry_time,
        extract_pre_decision_window,
    )

    # First pass: compute raw metrics for all trials
    raw_head_sweeps: list[float] = []
    raw_speeds: list[float] = []
    trial_windows: list[tuple[float, float]] = []  # (window_start, window_end)
    raw_min_speeds: list[float] = []

    for trial in trials:
        # Get entry time to decision region
        mask = (times >= trial.start_time) & (times <= trial.end_time)
        trial_positions = positions[mask]
        trial_times = times[mask]

        if len(trial_positions) == 0:
            continue

        trial_bins = env.bin_at(trial_positions)

        try:
            entry_time = decision_region_entry_time(
                trial_bins, trial_times, env, decision_region
            )
        except ValueError:
            # Trial never enters decision region - skip
            continue

        # Extract pre-decision window
        window_positions, window_times = extract_pre_decision_window(
            positions, times, entry_time, window_duration
        )

        if len(window_positions) < 3:
            # Not enough samples for heading analysis - skip
            continue

        # Compute head sweep magnitude
        head_sweep = head_sweep_from_positions(
            window_positions, window_times, min_speed=min_speed
        )

        # Compute speed statistics
        dt = np.diff(window_times)
        velocity = np.diff(window_positions, axis=0) / dt[:, np.newaxis]
        speeds = np.linalg.norm(velocity, axis=1)
        mean_spd = float(np.mean(speeds))
        min_spd = float(np.min(speeds))

        raw_head_sweeps.append(head_sweep)
        raw_speeds.append(mean_spd)
        raw_min_speeds.append(min_spd)
        trial_windows.append((entry_time - window_duration, entry_time))

    # Convert to arrays
    head_sweeps_arr = np.array(raw_head_sweeps)
    speeds_arr = np.array(raw_speeds)

    if len(head_sweeps_arr) == 0:
        # No valid trials
        return VTESessionResult(
            trial_results=[],
            mean_head_sweep=0.0,
            std_head_sweep=0.0,
            mean_speed=0.0,
            std_speed=0.0,
            n_vte_trials=0,
            vte_fraction=0.0,
        )

    # Compute session statistics
    mean_hs = float(np.mean(head_sweeps_arr))
    std_hs = float(np.std(head_sweeps_arr))
    mean_spd = float(np.mean(speeds_arr))
    std_spd = float(np.std(speeds_arr))

    # Second pass: compute z-scores and classify
    z_head_sweeps, z_speed_inv = normalize_vte_scores(head_sweeps_arr, speeds_arr)

    trial_results: list[VTETrialResult] = []
    n_vte = 0

    for i in range(len(head_sweeps_arr)):
        # Compute VTE index
        vte_idx = compute_vte_index(
            float(z_head_sweeps[i]), float(z_speed_inv[i]), alpha=alpha
        )

        # Classify
        is_vte = classify_vte(vte_idx, threshold=vte_threshold)
        if is_vte:
            n_vte += 1

        window_start, window_end = trial_windows[i]

        trial_results.append(
            VTETrialResult(
                head_sweep_magnitude=head_sweeps_arr[i],
                z_head_sweep=float(z_head_sweeps[i]),
                mean_speed=speeds_arr[i],
                min_speed=raw_min_speeds[i],
                z_speed_inverse=float(z_speed_inv[i]),
                vte_index=float(vte_idx),
                is_vte=is_vte,
                window_start=window_start,
                window_end=window_end,
            )
        )

    return VTESessionResult(
        trial_results=trial_results,
        mean_head_sweep=mean_hs,
        std_head_sweep=std_hs,
        mean_speed=mean_spd,
        std_speed=std_spd,
        n_vte_trials=n_vte,
        vte_fraction=n_vte / len(trial_results) if trial_results else 0.0,
    )
