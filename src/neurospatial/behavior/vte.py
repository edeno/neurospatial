"""Vicarious Trial and Error (VTE) detection and analysis.

This module provides tools for detecting and analyzing VTE behavior at choice
points, including head sweeping (IdPhi), pausing, and combined VTE indices.

Imports
-------
All functions are importable from `behavior.vte`:

    from neurospatial.behavior.vte import (
        VTETrialResult,
        VTESessionResult,
        head_sweep_magnitude,
        head_sweep_from_positions,
        normalize_vte_scores,
        compute_vte_index,
        classify_vte,
        compute_vte_trial,
        compute_vte_session,
    )

Key Concepts
------------
VTE (Vicarious Trial and Error)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Head sweeping**: Looking back and forth between options (high IdPhi)
- **Pausing**: Slowing down or stopping at the choice point (low speed)
- **IdPhi**: Integrated absolute head rotation - sum of absolute heading changes
- **zIdPhi**: Z-scored IdPhi relative to session baseline
- **VTE index**: Combined measure of head sweeping and slowing

When to Use
-----------
Use this module when analyzing:
- T-maze or Y-maze choice behavior
- VTE detection (hesitation at choice points)
- Head scanning behavior during deliberation

Example
-------
VTE analysis:

>>> from neurospatial.behavior import compute_vte_session
>>> result = compute_vte_session(
...     env,
...     times,
...     positions,
...     decision_region="center",
...     trials=trials,
...     window_duration=1.0,
... )  # doctest: +SKIP
>>> print(
...     f"VTE trials: {result.n_vte_trials}/{len(result.trial_results)}"
... )  # doctest: +SKIP

References
----------
.. [1] Johnson, A., & Redish, A. D. (2007). Neural ensembles in CA3 transiently
       encode paths forward of the animal at a decision point. J Neurosci.
.. [2] Papale, A. E., Stott, J. J., Powell, N. J., Regier, P. S., & Redish,
       A. D. (2012). Interactions between deliberation and delay-discounting
       in rats. Cogn Affect Behav Neurosci 12(3), 513-526. IdPhi and zIdPhi.
       DOI: 10.3758/s13415-012-0097-7
.. [3] Redish, A. D. (2016). Vicarious trial and error. Nat Rev Neurosci.
       DOI: 10.1038/nrn.2015.30
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pandas as pd

    from neurospatial.behavior.segmentation import Trial
    from neurospatial.environment import Environment

from neurospatial._intervals import DEFAULT_MAX_GAP, run_sample_bounds
from neurospatial._observation import ObservationSupport, validated_observations
from neurospatial._results import ResultMixin
from neurospatial._stacklevel import _external_stacklevel
from neurospatial._velocity import DEFAULT_SPEED_SMOOTHING
from neurospatial.geometry import require_cartesian
from neurospatial.ops.mapping import _require_region_mask
from neurospatial.stats.circular import wrap_angle

__all__ = [
    # VTE dataclasses
    "VTESessionResult",
    "VTETrialResult",
    # VTE functions
    "classify_vte",
    "compute_vte_index",
    "compute_vte_session",
    "compute_vte_trial",
    "head_sweep_from_positions",
    "head_sweep_magnitude",
    "normalize_vte_scores",
]


# =============================================================================
# VTE Data Structures
# =============================================================================


VTETrialStatus = Literal["ok", "no_samples", "no_entry", "too_few_samples"]
_VTE_TRIAL_STATUSES: tuple[str, ...] = get_args(VTETrialStatus)
# Per-trial scalars: window and status, raw metrics, z-scores, verdict last.
_TRIAL_COLUMNS = (
    "window_start",
    "window_end",
    "status",
    "head_sweep_magnitude",
    "mean_speed",
    "min_observed_speed",
    "z_head_sweep",
    "z_speed_inverse",
    "vte_index",
    "is_vte",
)


@dataclass(frozen=True, repr=False)
class VTETrialResult(ResultMixin):
    """VTE metrics for a single trial.

    Attributes
    ----------
    head_sweep_magnitude : float
        Sum of |delta_theta| in pre-decision window (radians).
        Also known as IdPhi (Integrated absolute Phi).
        High values indicate looking back and forth between options.
        NaN when no movement heading was observed in the window (the animal
        was slower than ``min_speed`` throughout, or too few samples).
    z_head_sweep : float or None
        Z-scored head sweep magnitude relative to session baseline.
        None if session statistics not computed (single trial analysis);
        NaN when ``head_sweep_magnitude`` is NaN.
    mean_speed : float
        Mean speed in pre-decision window (environment units/s).
        NaN when the window has no observed interval (fewer than two
        consecutive observed samples); 0.0 is a measured stop.
    min_observed_speed : float
        Minimum observed speed in pre-decision window (units/s).
        Near-zero indicates a pause. NaN like ``mean_speed``. Not to be
        confused with the ``min_speed`` heading gate passed to the analysis.
    z_speed_inverse : float or None
        Negated z-score of ``mean_speed`` relative to the session baseline
        (higher = slower, more VTE-like). None if session statistics not
        computed; NaN when ``mean_speed`` is NaN.
    vte_index : float or None
        Combined VTE index: alpha * z_head_sweep + (1-alpha) * z_speed_inverse.
        None if session statistics not computed; NaN when either z-score is
        NaN.
    is_vte : bool or None
        True if ``vte_index >= threshold``. None if not classified: single
        trial analysis, or a NaN ``vte_index`` (nothing to classify).
    window_start : float
        Requested window start (seconds), clipped at the trial start in
        session analysis. Available samples may start later after a pause.
        NaN when the trial has no decision-region entry.
    window_end : float
        End time of window (decision region entry time, seconds). NaN when
        the trial has no decision-region entry.
    status : {"ok", "no_samples", "no_entry", "too_few_samples"}
        Why the trial was or was not measured. ``"ok"``: the window was
        analyzed (its metrics can still be NaN, for example a slow animal
        with no observed heading). ``"no_samples"``: no sample lies between
        the trial's start and end times. ``"no_entry"``: the trial never
        enters the decision region inside ``epochs``. ``"too_few_samples"``:
        fewer than three observed samples lie in the window. Every metric of
        a trial that is not ``"ok"`` is NaN and it is not classified.

    Examples
    --------
    A trial whose window holds a single observed sample has no speed and no
    heading, so its metrics are NaN and it is not classified:

    >>> import numpy as np
    >>> from neurospatial.behavior.vte import compute_vte_trial
    >>> times = np.linspace(0, 3, 90)
    >>> positions = np.column_stack([np.linspace(0, 50, 90), np.ones(90) * 50])
    >>> result = compute_vte_trial(
    ...     times,
    ...     positions,
    ...     entry_time=2.0,
    ...     window_duration=1.0,
    ...     min_speed=5.0,
    ...     break_before=times == times[59],
    ... )
    >>> result.head_sweep_magnitude, result.mean_speed, result.is_vte
    (nan, nan, None)
    """

    head_sweep_magnitude: float
    z_head_sweep: float | None
    mean_speed: float
    min_observed_speed: float
    z_speed_inverse: float | None
    vte_index: float | None
    is_vte: bool | None
    window_start: float
    window_end: float
    status: VTETrialStatus

    def __post_init__(self) -> None:
        """Store plain Python scalars and check the status and window."""
        if self.status not in _VTE_TRIAL_STATUSES:
            raise ValueError(
                f"VTETrialResult status must be one of {_VTE_TRIAL_STATUSES}; "
                f"got {self.status!r}."
            )
        for name in (
            "head_sweep_magnitude",
            "mean_speed",
            "min_observed_speed",
            "window_start",
            "window_end",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("z_head_sweep", "z_speed_inverse", "vte_index"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.is_vte is not None:
            object.__setattr__(self, "is_vte", bool(self.is_vte))
        if self.window_start > self.window_end:
            raise ValueError(
                f"VTETrialResult window_start ({self.window_start}) is after "
                f"window_end ({self.window_end})."
            )
        if self.status != "ok" and self.is_vte is not None:
            raise ValueError(
                f"A VTE trial with status {self.status!r} was not measured, so "
                "it cannot be classified: is_vte must be None."
            )

    # Aliases for common terminology
    @property
    def idphi(self) -> float:
        """Alias for head_sweep_magnitude (IdPhi terminology)."""
        return self.head_sweep_magnitude

    @property
    def z_idphi(self) -> float | None:
        """Alias for z_head_sweep (zIdPhi terminology)."""
        return self.z_head_sweep

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar VTE metrics for this trial.

        Follows the result-object convention (scalar dict, tabulatable). For a
        human-readable one-line string use ``str(result)``.
        """
        return {name: getattr(self, name) for name in _TRIAL_COLUMNS}

    def __str__(self) -> str:
        if self.status != "ok":
            return f"Trial: not measured ({self.status})"
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


@dataclass(frozen=True, repr=False, eq=False)
class VTESessionResult(ResultMixin):
    """VTE analysis for an entire session.

    Holds one :class:`VTETrialResult` per input trial, in input order, so
    ``trial_results[i]`` describes ``trials[i]``. Session statistics are
    derived from them.

    Attributes
    ----------
    trial_results : tuple of VTETrialResult
        Per-trial VTE metrics with z-scores computed, one per input trial.
    mean_head_sweep : float
        Session mean of head sweep magnitude (the z-scoring baseline), over
        trials with an observed head sweep. NaN when no trial has one.
    std_head_sweep : float
        Session (population) std of head sweep magnitude, over the same
        trials.
    mean_speed : float
        Session mean of pre-decision mean speed, over trials with an
        observed speed. NaN when no trial has one.
    std_speed : float
        Session (population) std of pre-decision mean speed, over the same
        trials.
    n_vte_trials : int
        Number of trials classified as VTE (``is_vte is True``).
    vte_fraction : float
        Fraction of classified trials (``is_vte`` not None) that are VTE.
        NaN when no trial was classified.

    Examples
    --------
    >>> from neurospatial.behavior import VTESessionResult, VTETrialResult
    >>> def trial(head_sweep, is_vte, status="ok"):
    ...     return VTETrialResult(
    ...         head_sweep_magnitude=head_sweep,
    ...         z_head_sweep=None,
    ...         mean_speed=10.0,
    ...         min_observed_speed=8.0,
    ...         z_speed_inverse=None,
    ...         vte_index=None,
    ...         is_vte=is_vte,
    ...         window_start=0.0,
    ...         window_end=1.0,
    ...         status=status,
    ...     )
    >>> session = VTESessionResult(
    ...     trial_results=[
    ...         trial(3.0, True),
    ...         trial(1.0, False),
    ...         trial(float("nan"), None, status="too_few_samples"),
    ...     ]
    ... )
    >>> print(session)
    VTE session: 1/2 classified trials VTE (50.0%), 1 unclassified
      Head sweep: mean=2.00, std=1.00
      Speed: mean=10.0, std=0.0
    >>> session.trials_table()["status"].tolist()
    ['ok', 'ok', 'too_few_samples']
    """

    trial_results: tuple[VTETrialResult, ...]

    def __post_init__(self) -> None:
        """Store the trial results as a tuple."""
        trial_results = tuple(self.trial_results)
        for trial in trial_results:
            if not isinstance(trial, VTETrialResult):
                raise TypeError(
                    "VTESessionResult.trial_results must hold VTETrialResult "
                    f"objects; got {type(trial).__name__}."
                )
        object.__setattr__(self, "trial_results", trial_results)

    def _observed(self, name: str) -> NDArray[np.float64]:
        """Finite values of one per-trial metric."""
        values: NDArray[np.float64] = np.array(
            [getattr(trial, name) for trial in self.trial_results], dtype=np.float64
        )
        observed: NDArray[np.float64] = values[np.isfinite(values)]
        return observed

    @property
    def mean_head_sweep(self) -> float:
        """Session mean head sweep over trials with one; NaN without any."""
        return _mean_or_nan(self._observed("head_sweep_magnitude"))

    @property
    def std_head_sweep(self) -> float:
        """Session std of head sweep over trials with one; NaN without any."""
        return _std_or_nan(self._observed("head_sweep_magnitude"))

    @property
    def mean_speed(self) -> float:
        """Session mean of pre-decision mean speed; NaN without any."""
        return _mean_or_nan(self._observed("mean_speed"))

    @property
    def std_speed(self) -> float:
        """Session std of pre-decision mean speed; NaN without any."""
        return _std_or_nan(self._observed("mean_speed"))

    @property
    def n_vte_trials(self) -> int:
        """Number of trials classified as VTE (``is_vte is True``)."""
        return sum(trial.is_vte is True for trial in self.trial_results)

    @property
    def vte_fraction(self) -> float:
        """``n_vte_trials / n_classified``; NaN when no trial was classified."""
        n_classified = self.n_classified
        return self.n_vte_trials / n_classified if n_classified else np.nan

    # Aliases for common terminology
    @property
    def mean_idphi(self) -> float:
        """Alias for mean_head_sweep (IdPhi terminology)."""
        return self.mean_head_sweep

    @property
    def std_idphi(self) -> float:
        """Alias for std_head_sweep (IdPhi terminology)."""
        return self.std_head_sweep

    @property
    def n_classified(self) -> int:
        """Number of trials classified as VTE or not (``is_vte`` not None).

        ``vte_fraction`` is ``n_vte_trials / n_classified``. A trial without
        an observed head sweep or speed is analyzed but not classified.

        Examples
        --------
        >>> from neurospatial.behavior import VTESessionResult
        >>> VTESessionResult(()).n_classified
        0
        """
        return sum(trial.is_vte is not None for trial in self.trial_results)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar session-level VTE metrics.

        ``n_trials`` counts the input trials and ``n_classified`` those with
        a VTE verdict, the denominator of ``vte_fraction``. For a
        human-readable multi-line string use ``str(result)``.

        Examples
        --------
        >>> from neurospatial.behavior import VTESessionResult
        >>> summary = VTESessionResult(()).summary()
        >>> summary["n_trials"], summary["n_classified"], summary["vte_fraction"]
        (0, 0, nan)
        """
        return {
            "n_trials": len(self.trial_results),
            "n_classified": self.n_classified,
            "n_vte_trials": self.n_vte_trials,
            "vte_fraction": self.vte_fraction,
            "mean_head_sweep": self.mean_head_sweep,
            "std_head_sweep": self.std_head_sweep,
            "mean_speed": self.mean_speed,
            "std_speed": self.std_speed,
        }

    def trials_table(self) -> pd.DataFrame:
        """One row per input trial, in input order.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``trial`` (the position of the trial in the input
            list), with the columns of :meth:`VTETrialResult.summary`:
            window bounds and ``status`` first, then the raw metrics,
            z-scores and VTE index, and the ``is_vte`` verdict last.
        """
        import pandas as pd

        table = pd.DataFrame(
            [trial.summary() for trial in self.trial_results],
            columns=list(_TRIAL_COLUMNS),
        )
        table.index.name = "trial"
        return table

    def __str__(self) -> str:
        n_classified = self.n_classified
        n_unclassified = len(self.trial_results) - n_classified
        if n_classified:
            verdict = (
                f"{self.n_vte_trials}/{n_classified} classified trials VTE "
                f"({self.vte_fraction:.1%})"
            )
            if n_unclassified:
                verdict += f", {n_unclassified} unclassified"
        else:
            verdict = "no trial could be classified"
            if n_unclassified:
                verdict += f" ({n_unclassified} unclassified)"
        return (
            f"VTE session: {verdict}\n"
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


def _mean_or_nan(values: NDArray[np.float64]) -> float:
    return float(np.mean(values)) if values.size else np.nan


def _std_or_nan(values: NDArray[np.float64]) -> float:
    return float(np.std(values)) if values.size else np.nan


# =============================================================================
# VTE Core Functions
# =============================================================================


def head_sweep_magnitude(headings: NDArray[np.float64]) -> float:
    """Compute integrated absolute head rotation (IdPhi).

    Sums the absolute value of heading changes across a trajectory window.
    High values indicate back-and-forth head movements ("scanning").

    Parameters
    ----------
    headings : NDArray[np.float64], shape (n_samples,)
        Heading angles in radians, from one continuous stretch of
        recording. NaN means no heading (for example the animal was too
        slow for its direction of movement to be observed).

    Returns
    -------
    float
        Sum of absolute heading changes in radians.
        NaN if fewer than 2 headings are finite: no head sweep was observed.

    Warns
    -----
    UserWarning
        If ``headings`` look like degrees (values beyond 2π, all within ±360,
        with steps larger than π).

    Notes
    -----
    This is the IdPhi measure of head sweeping at a choice point (integrated
    absolute dphi/dt * dt), as used by Papale et al. (2012) and reviewed by
    Redish (2016).

    NaN entries are skipped, so the change from the last heading before
    them to the first heading after them counts once. Pass one continuous
    run at a time: across a tracking loss or a recording break the heading
    change was not observed. :func:`head_sweep_from_positions` splits runs
    for you.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.vte import head_sweep_magnitude
    >>> # Constant heading: no head sweep
    >>> head_sweep_magnitude(np.zeros(10))
    0.0
    >>> # Alternating headings: high head sweep
    >>> headings = np.array([0, np.pi / 4, 0, np.pi / 4, 0])
    >>> head_sweep_magnitude(headings)  # 4 * pi/4 = pi
    3.141592653589793

    With fewer than two headings nothing was measured:

    >>> head_sweep_magnitude(np.array([0.5, np.nan]))
    nan
    """
    from neurospatial.encoding._validation import warn_if_angles_look_like_degrees

    headings = np.asarray(headings, dtype=np.float64)
    warn_if_angles_look_like_degrees(
        headings, name="headings", context="head_sweep_magnitude()"
    )
    valid_headings = headings[np.isfinite(headings)]

    if len(valid_headings) < 2:
        return np.nan

    delta = wrap_angle(np.diff(valid_headings))
    return float(np.sum(np.abs(delta)))


def head_sweep_from_positions(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    min_speed: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> float:
    """Compute head sweep magnitude (IdPhi) from position trajectory.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps in seconds; finite and strictly increasing (a repeated
        timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates, ``n_dims >= 2``; heading uses x and y. A row
        with a non-finite coordinate (NaN, for a lost frame) is a missing
        sample: it has no heading of its own and is bridged like a deleted
        row when the observed samples on either side are within ``max_gap``.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s
        with positions in cm): this function sees no environment, so it
        cannot know the positions' unit.
        Minimum speed for valid heading (units/s); finite and non-negative.
        Stationary periods are excluded.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. No heading change is counted across a
        declared break.

    Returns
    -------
    float
        Sum of absolute heading changes (radians). NaN when no movement
        heading was observed: the animal was slower than ``min_speed``
        throughout, or there are too few samples.

    Raises
    ------
    ValueError
        If there are no samples, if ``times`` is not finite and strictly
        increasing, if ``break_before`` is not a Boolean array with one entry
        per sample, or if positions has fewer than two coordinates (x and y).

    Notes
    -----
    A missing row is skipped as if it were deleted: when the observed
    samples on either side are no more than ``max_gap`` apart, velocity and
    heading are computed across it. Only a longer gap, an ``epochs`` edge or
    a ``break_before`` splits the recording into separately analyzed runs,
    and no velocity or heading spans such a split.

    Heading is the direction of the smoothed velocity (as in
    :func:`neurospatial.behavior.heading_from_velocity`) and is NaN
    where the animal is slower than ``min_speed`` or still. Within a run, a
    slow stretch is skipped like a missing heading: the heading change from
    the last heading before it to the first after it is counted once. The
    result sums :func:`head_sweep_magnitude` over runs, without connecting
    the last heading of one run to the first of the next; runs without a
    head sweep add nothing.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.vte import head_sweep_from_positions

    A straight eastward run has no head sweep:

    >>> times = np.linspace(0, 2, 20)  # s
    >>> positions = np.column_stack([np.linspace(0, 100, 20), np.ones(20) * 50])
    >>> float(head_sweep_from_positions(times, positions, min_speed=1.0))
    0.0

    A still animal has no direction of movement, so no head sweep was
    observed:

    >>> still = np.full((20, 2), 50.0)  # cm
    >>> head_sweep_from_positions(times, still, min_speed=1.0)
    nan
    """
    from neurospatial.behavior._kinematics import _validate_velocity_positions
    from neurospatial.behavior.decisions import _validated_min_speed

    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        positions,
        call="head_sweep_from_positions",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no head sweep was observed",
    )
    _validate_velocity_positions(raw_positions, call="head_sweep_from_positions")
    min_speed = _validated_min_speed(min_speed, call="head_sweep_from_positions")
    return _head_sweep(times, positions, support.interval_mask, min_speed=min_speed)


def _head_sweep(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    interval_mask: NDArray[np.bool_],
    *,
    min_speed: float,
) -> float:
    """IdPhi summed over runs of usable intervals; NaN when no run has one."""
    from neurospatial.behavior._kinematics import _movement_heading_and_speed

    if not interval_mask.any():
        return np.nan
    heading, speed = _movement_heading_and_speed(
        times, positions, interval_mask, smooth_window=DEFAULT_SPEED_SMOOTHING
    )
    # The velocity direction is observed only where the animal moves.
    heading[~((speed > 0) & (speed >= min_speed))] = np.nan
    sweeps = [
        head_sweep_magnitude(heading[first : last + 1])
        for first, last in run_sample_bounds(interval_mask)
    ]
    observed = [sweep for sweep in sweeps if not np.isnan(sweep)]
    return float(sum(observed)) if observed else np.nan


# =============================================================================
# VTE Z-Scoring Functions
# =============================================================================


def normalize_vte_scores(
    head_sweeps: NDArray[np.float64],
    speeds: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Z-score VTE metrics across trials.

    Parameters
    ----------
    head_sweeps : NDArray[np.float64], shape (n_trials,)
        Head sweep magnitude for each trial; NaN for a trial without one.
    speeds : NDArray[np.float64], shape (n_trials,)
        Mean speed for each trial; NaN for a trial without one.

    Returns
    -------
    z_head_sweeps : NDArray[np.float64], shape (n_trials,)
        Z-scored head sweeps.
    z_speed_inverse : NDArray[np.float64], shape (n_trials,)
        Negated z-scored speed (higher = slower = more VTE-like).

    Raises
    ------
    ValueError
        If arrays have different lengths.

    Warns
    -----
    UserWarning
        If std over the finite entries is zero (no variation across trials);
        their z-scores will be 0.

    Notes
    -----
    Each input is z-scored with the mean and std of its finite entries only.
    A non-finite entry (a trial where the quantity was not observed) gets a
    NaN z-score instead of pulling the session baseline; an input with no
    finite entry gives all NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.vte import normalize_vte_scores
    >>> head_sweeps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> speeds = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    >>> z_hs, z_sp = normalize_vte_scores(head_sweeps, speeds)
    >>> bool(np.abs(np.mean(z_hs)) < 1e-10)  # Mean is ~0
    True

    A trial without an observed head sweep (NaN) is left out of the
    baseline and gets a NaN z-score:

    >>> z_hs, z_sp = normalize_vte_scores([1.0, np.nan, 3.0], [10.0, 20.0, 30.0])
    >>> z_hs.tolist()
    [-1.0, nan, 1.0]
    """
    head_sweeps = np.asarray(head_sweeps, dtype=np.float64)
    speeds = np.asarray(speeds, dtype=np.float64)
    if len(head_sweeps) != len(speeds):
        raise ValueError(
            f"head_sweeps and speeds must have same length. "
            f"Got {len(head_sweeps)} and {len(speeds)}."
        )

    z_head_sweeps = _observed_z_scores(
        head_sweeps,
        no_variation="No variation in head sweep magnitude across trials (std=0). "
        "All trials have identical head sweep behavior. ",
        invert=False,
    )
    # Negate the speed z-score so slower = higher score.
    z_speed_inverse = _observed_z_scores(
        speeds,
        no_variation="No variation in speed across trials (std=0). "
        "All trials have identical speed behavior. ",
        invert=True,
    )
    return z_head_sweeps, z_speed_inverse


def _observed_z_scores(
    values: NDArray[np.float64], *, no_variation: str, invert: bool
) -> NDArray[np.float64]:
    """Z-scores over the finite entries; NaN elsewhere, or everywhere if none."""
    z_scores = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return z_scores
    observed = values[finite]
    mean = np.mean(observed)
    std = np.std(observed)
    if std < 1e-10:
        warnings.warn(
            no_variation
            + "Z-scores will be 0, and VTE classification may not be meaningful. "
            "Consider adjusting window_duration or min_speed parameters.",
            category=UserWarning,
            stacklevel=_external_stacklevel(),
        )
        z_scores[finite] = 0.0
    elif invert:
        z_scores[finite] = -(observed - mean) / std
    else:
        z_scores[finite] = (observed - mean) / std
    return z_scores


# =============================================================================
# VTE Classification Functions
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
        Negated z-scored speed (higher = slower).
    alpha : float, default=0.5
        Weight for head sweep (1-alpha for speed), in [0, 1].
        Default 0.5 weights both equally.

    Returns
    -------
    float
        Combined VTE index; NaN when either z-score is NaN.

    Raises
    ------
    ValueError
        If ``alpha`` is not in [0, 1].

    Examples
    --------
    >>> from neurospatial.behavior.vte import compute_vte_index
    >>> compute_vte_index(1.0, 1.0, alpha=0.5)
    1.0
    >>> compute_vte_index(2.0, 0.0, alpha=1.0)  # Head sweep only
    2.0
    """
    alpha = _validated_alpha(alpha, call="compute_vte_index")
    return alpha * z_head_sweep + (1 - alpha) * z_speed_inv


def _validated_alpha(alpha: float, *, call: str) -> float:
    """A head-sweep weight in [0, 1]."""
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{call}() needs alpha in [0, 1]; got {alpha!r}.\n"
            "Why: alpha weights the head-sweep z-score and 1 - alpha the "
            "slowness z-score, so outside [0, 1] one of them counts against "
            "VTE.\n"
            "Fix: pass alpha=0.5 to weight both equally, 1.0 for head sweep "
            "only or 0.0 for slowness only."
        )
    return value


def classify_vte(
    vte_index: float,
    *,
    threshold: float = 0.5,
) -> bool | None:
    """Classify trial as VTE based on VTE index.

    A threshold screen on the session-relative VTE index, not a statistical
    test.

    Parameters
    ----------
    vte_index : float
        Combined VTE index.
    threshold : float, default=0.5
        Classification threshold (finite). Trial is VTE if
        ``vte_index >= threshold``.

    Returns
    -------
    bool or None
        True if the trial is classified as VTE, False if not, and None for a
        NaN ``vte_index``: nothing was measured, which is not evidence
        against VTE.

    Raises
    ------
    ValueError
        If ``threshold`` is not finite.

    Examples
    --------
    >>> from neurospatial.behavior.vte import classify_vte
    >>> classify_vte(1.0, threshold=0.5)
    True
    >>> classify_vte(0.3, threshold=0.5)
    False
    >>> classify_vte(float("nan")) is None
    True
    """
    if not np.isfinite(threshold):
        raise ValueError(
            f"classify_vte() needs a finite threshold; got {threshold!r}.\n"
            "Why: no VTE index compares meaningfully with a NaN or infinite "
            "threshold.\n"
            "Fix: pass a finite threshold on the VTE index, e.g. threshold=0.5."
        )
    if np.isnan(vte_index):
        return None
    return bool(vte_index >= threshold)


# =============================================================================
# VTE Composite Functions
# =============================================================================


def compute_vte_trial(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    entry_time: float,
    *,
    window_duration: float,
    min_speed: float,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> VTETrialResult:
    """Compute VTE metrics for a single trial.

    Parameters
    ----------
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire trajectory, in seconds; finite and strictly
        increasing (a repeated timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates for entire trajectory, ``n_dims >= 2``. A row
        with a non-finite coordinate (NaN, for a lost frame) is a missing
        sample.
    entry_time : float
        Time of decision region entry, in seconds on the clock of ``times``;
        within ``[times[0], times[-1]]``.
    window_duration : float
        Duration of pre-decision window (seconds); finite and positive.
    min_speed : float
        Required, in position units per second (e.g. ``5.0`` for 5 cm/s
        with positions in cm): this function sees no environment, so it
        cannot know the positions' unit.
        Minimum speed for valid heading (units/s); finite and non-negative.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. The window never reaches back across a
        declared break.

    Returns
    -------
    VTETrialResult
        VTE metrics for the trial.
        Note: z-scores and classification are None for single trial analysis.
        Use compute_vte_session() for session-level analysis with z-scoring.
        ``mean_speed`` and ``min_observed_speed`` are NaN when the window has no
        observed interval, and ``head_sweep_magnitude`` is NaN when no
        movement heading was observed.

    Raises
    ------
    ValueError
        If there are no samples, if ``times`` is not finite and strictly
        increasing, if ``break_before`` is not a Boolean array with one entry
        per sample, if positions has fewer than two coordinates (x and y), if
        ``window_duration`` is not finite and positive, or if ``entry_time``
        is outside the recording.

    Notes
    -----
    A missing row is skipped as if it were deleted: when the observed
    samples on either side are no more than ``max_gap`` apart, velocity and
    heading are computed across it. Only a longer gap, an ``epochs`` edge or
    a ``break_before`` splits the recording into separately analyzed runs,
    and no velocity or heading spans such a split.

    The pre-decision window is the one :func:`extract_pre_decision_window`
    returns: it lies in the time segment containing ``entry_time``. Result
    window bounds describe the requested window; fewer samples may be
    available within that window.

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial.behavior.vte import compute_vte_trial
    >>> times = np.linspace(0, 3, 90)
    >>> positions = np.column_stack([np.linspace(0, 50, 90), np.ones(90) * 50])
    >>> result = compute_vte_trial(
    ...     times, positions, entry_time=2.0, window_duration=1.0, min_speed=5.0
    ... )
    >>> result.window_start
    1.0
    >>> result.window_end
    2.0

    An animal that waits still before the entry moves at a measured 0 cm/s
    but has no direction of movement, so its head sweep is NaN:

    >>> still = np.full((90, 2), 50.0)  # cm
    >>> result = compute_vte_trial(
    ...     times, still, entry_time=2.0, window_duration=1.0, min_speed=5.0
    ... )
    >>> float(result.mean_speed), result.head_sweep_magnitude
    (0.0, nan)

    A new recording that starts one sample before the entry leaves one
    sample in the window, so no speed was observed either:

    >>> result = compute_vte_trial(
    ...     times,
    ...     positions,
    ...     entry_time=2.0,
    ...     window_duration=1.0,
    ...     min_speed=5.0,
    ...     break_before=times == times[59],
    ... )
    >>> result.mean_speed, result.head_sweep_magnitude
    (nan, nan)
    """
    from neurospatial.behavior._kinematics import _validate_velocity_positions
    from neurospatial.behavior.decisions import (
        _pre_decision_window,
        _validated_entry_time,
        _validated_min_speed,
        _validated_window_duration,
    )

    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        positions,
        call="compute_vte_trial",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no VTE metric was observed",
    )
    _validate_velocity_positions(raw_positions, call="compute_vte_trial")
    window_duration = _validated_window_duration(
        window_duration, name="window_duration", call="compute_vte_trial"
    )
    entry_time = _validated_entry_time(times, entry_time, call="compute_vte_trial")
    min_speed = _validated_min_speed(min_speed, call="compute_vte_trial")
    window = _pre_decision_window(
        times, support.interval_gates, entry_time, window_duration
    )
    head_sweep, mean_spd, min_spd = _window_metrics(
        times, positions, support, window, min_speed=min_speed
    )
    return VTETrialResult(
        head_sweep_magnitude=head_sweep,
        z_head_sweep=None,  # Not computed for single trial
        mean_speed=mean_spd,
        min_observed_speed=min_spd,
        z_speed_inverse=None,
        vte_index=None,
        is_vte=None,
        window_start=entry_time - window_duration,
        window_end=entry_time,
        status="ok",
    )


def _window_metrics(
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    support: ObservationSupport,
    window: slice,
    *,
    min_speed: float,
) -> tuple[float, float, float]:
    """Head sweep, mean speed and minimum speed of one pre-decision window.

    The window lies inside one gate segment, so its usable intervals are the
    support's usable intervals between its first and last sample.
    """
    from neurospatial.behavior.decisions import _speed_stats

    window_times = times[window]
    window_positions = positions[window]
    window_mask = support.interval_mask[window.start : max(window.stop - 1, 0)]
    head_sweep = _head_sweep(
        window_times, window_positions, window_mask, min_speed=min_speed
    )
    mean_speed, min_speed_value = _speed_stats(
        window_times, window_positions, window_mask
    )
    return head_sweep, mean_speed, min_speed_value


def compute_vte_session(
    env: Environment,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    *,
    decision_region: str,
    trials: list[Trial],
    window_duration: float = 1.0,
    min_speed: float | None = None,
    alpha: float = 0.5,
    vte_threshold: float = 0.5,
    max_gap: float | None = DEFAULT_MAX_GAP,
    epochs: Any = None,
    break_before: ArrayLike | None = None,
) -> VTESessionResult:
    """Compute VTE metrics for all trials in a session.

    Parameters
    ----------
    env : Environment
        Environment with region definitions.
    times : NDArray[np.float64], shape (n_samples,)
        Timestamps for entire session, in seconds; finite and strictly
        increasing (a repeated timestamp raises).
    positions : NDArray[np.float64], shape (n_samples, n_dims)
        Position coordinates for entire session, ``n_dims >= 2``. A row with
        a non-finite coordinate (NaN, for a lost frame) is a missing sample.
    decision_region : str
        Name of decision region in env.regions.
    trials : list[Trial]
        Trial segmentation from segment_trials().
    window_duration : float, default=1.0
        Duration of pre-decision window in seconds; finite and positive.
        Typical values: 0.5-2.0s depending on maze size and task.
        The window is clipped at the trial start; samples before
        ``trial.start_time`` are never used.
    min_speed : float or None, default=None
        In position units per second. None is 5 cm/s expressed in
        ``env.units`` (``0.05`` for meters, ``50.0`` for millimeters); with
        units that are not a metric length, pass it explicitly.
        Minimum speed for valid heading (units/s); finite and non-negative.
    alpha : float, default=0.5
        Weight for head sweep in VTE index (1-alpha for speed), in [0, 1].
        Default 0.5 weights both equally.
    vte_threshold : float, default=0.5
        Threshold for VTE classification (finite).
        Trial is VTE if ``vte_index >= vte_threshold``.
    max_gap : float or None, default=1.0
        Longest stretch of missing tracking, in seconds, to bridge: the time
        between consecutive valid samples (a dropped frame and a NaN row are
        the same missing sample). Longer gaps are excluded as tracking losses;
        ``None`` bridges every gap. The default is calibrated for typical video
        and DeepLabCut tracking; choose a value longer than routine dropped or
        low-confidence frames and shorter than real tracking losses or pickups.
    epochs : (start, stop), array-like of shape (n, 2), IntervalSet, or None
        Restrict the analysis to these half-open [start, stop) windows (seconds,
        same clock as ``times``). The interval between two consecutive samples
        counts only if it lies entirely inside one window. ``None`` (default)
        means unrestricted.
    break_before : ArrayLike of bool, shape (n_samples,), optional
        ``True`` at sample ``k`` declares a break between samples ``k - 1`` and
        ``k`` (for example where a new recording starts), whatever the elapsed
        time. Entry 0 is ignored. No pre-decision window reaches back across
        a declared break.

    Returns
    -------
    VTESessionResult
        Session-level VTE analysis with one :class:`VTETrialResult` per input
        trial, in input order (``result.trial_results[i]`` is ``trials[i]``;
        ``result.trials_table()`` tabulates them). A trial that could not be
        measured keeps its place with NaN metrics, ``is_vte=None`` and a
        ``status`` saying why; a measured trial without an observed head
        sweep or speed has status ``"ok"``, NaN metrics and ``is_vte=None``.
        Without any measured trial the session statistics and
        ``vte_fraction`` are NaN.

    Raises
    ------
    ValueError
        If ``decision_region`` is not in ``env.regions``, if there are no
        samples, if ``times`` is not finite and strictly increasing, if
        ``break_before`` is not a Boolean array with one entry per sample, if
        positions has fewer than two coordinates (x and y), if
        ``window_duration`` is not finite and positive, if ``alpha`` is not
        in [0, 1], or if ``vte_threshold`` is not finite.

    Warns
    -----
    UserWarning
        For each trial that could not be measured: no samples between its
        start and end times (often trial times on another clock), no entry
        to ``decision_region`` inside ``epochs``, or fewer than three
        observed samples in its window. Also when every trial has the same
        head sweep or speed (z-scores of 0).

    Notes
    -----
    A missing row is skipped as if it were deleted: when the observed
    samples on either side are no more than ``max_gap`` apart, velocity and
    heading are computed across it. Only a longer gap, an ``epochs`` edge or
    a ``break_before`` splits the recording into separately analyzed runs,
    and no velocity or heading spans such a split.

    A trial's decision-region entry is its first sample inside both the
    region and ``epochs``. Each pre-decision window uses only its trial
    samples and the time segment containing that entry. Result window bounds
    describe the requested window clipped at the trial start; fewer samples
    may be available within that window.

    Z-scores use the session mean and std of the trials where the quantity
    was observed (see :func:`normalize_vte_scores`). A trial whose window has
    no observed movement heading (the animal stayed slower than
    ``min_speed``) has a NaN head sweep, so its VTE index is NaN and it is
    not classified rather than scored as "no head sweep".

    Examples
    --------
    >>> import numpy as np
    >>> from neurospatial import Environment, Regions
    >>> from neurospatial.behavior import Trial, compute_vte_session

    A T-maze stem rising to a crossbar, with the choice point at its center:

    >>> stem = np.column_stack([np.full(30, 50.0), np.linspace(0, 50, 30)])
    >>> bar = np.column_stack([np.linspace(0, 100, 60), np.full(60, 55.0)])
    >>> env = Environment.from_samples(np.vstack([stem, bar]), bin_size=5.0)
    >>> regions = Regions()
    >>> _ = regions.add("center", point=(50.0, 55.0))
    >>> env = env.with_regions(regions)

    Three 2 s approaches up the stem at 13 cm/s, scanning left and right
    with growing amplitude, then one where the animal creeps up at 1.5 cm/s:

    >>> times, positions, trials = [], [], []
    >>> for i, scan in enumerate([1.0, 2.0, 3.0, None]):
    ...     t = 3.0 * i + np.arange(60) / 30  # s
    ...     if scan is None:
    ...         xy = np.column_stack([np.full(60, 50.0), 50.0 + 1.5 * (t - t[0])])
    ...     else:
    ...         x = 50.0 + scan * np.sin(np.linspace(0, 10 * np.pi, 60))
    ...         xy = np.column_stack([x, np.linspace(30.0, 56.0, 60)])
    ...     times.append(t)
    ...     positions.append(xy)
    ...     trials.append(Trial(t[0], t[-1], "start", "center", "reached"))
    >>> result = compute_vte_session(
    ...     env,
    ...     np.concatenate(times),
    ...     np.vstack(positions),
    ...     decision_region="center",
    ...     trials=trials,
    ...     window_duration=0.5,
    ...     min_speed=5.0,
    ...     vte_threshold=0.0,
    ... )

    The creeping trial has no observed heading, so it is unclassified, and
    the VTE fraction counts the three classified trials:

    >>> [t.is_vte for t in result.trial_results]
    [False, False, True, None]
    >>> float(result.trial_results[3].head_sweep_magnitude)
    nan
    >>> result.n_vte_trials, round(result.vte_fraction, 3)
    (1, 0.333)

    One row per trial, in input order:

    >>> result.trials_table()[["status", "is_vte"]].values.tolist()
    [['ok', False], ['ok', False], ['ok', True], ['ok', None]]
    """
    require_cartesian(
        env.coordinate_system,
        "compute_vte_session()",
        why="It reads positions as the animal's allocentric positions, with "
        "head sweeps and speeds in the same Cartesian coordinates.",
        fix="use the allocentric environment the positions were recorded in",
    )
    raw_positions = positions
    times, positions, support = validated_observations(
        times,
        positions,
        call="compute_vte_session",
        max_gap=max_gap,
        epochs=epochs,
        break_before=break_before,
        result="no VTE trial was observed",
    )
    _require_region_mask(
        env,
        decision_region,
        operation="compute_vte_session()",
        argument="decision_region",
    )
    from neurospatial.behavior._kinematics import (
        _validate_velocity_positions,
        resolve_min_speed,
    )
    from neurospatial.behavior.decisions import (
        _in_epochs,
        _validated_min_speed,
        _validated_window_duration,
    )

    _validate_velocity_positions(raw_positions, call="compute_vte_session")
    window_duration = _validated_window_duration(
        window_duration, name="window_duration", call="compute_vte_session"
    )
    alpha = _validated_alpha(alpha, call="compute_vte_session")
    if not np.isfinite(vte_threshold):
        raise ValueError(
            f"compute_vte_session() needs a finite vte_threshold; got "
            f"{vte_threshold!r}.\n"
            "Why: no VTE index compares meaningfully with a NaN or infinite "
            "threshold.\n"
            "Fix: pass a finite threshold on the VTE index, e.g. "
            "vte_threshold=0.5."
        )
    min_speed = _validated_min_speed(
        resolve_min_speed(env, min_speed, call="compute_vte_session()"),
        call="compute_vte_session",
    )
    region_bins = env.bins_in_region(decision_region)
    in_epochs = _in_epochs(times, support.epochs)
    measured: list[_TrialMeasurement] = []
    for index, trial in enumerate(trials):
        measured.append(
            _measure_trial(
                index,
                trial,
                times,
                positions,
                support,
                env=env,
                region_bins=region_bins,
                in_epochs=in_epochs,
                decision_region=decision_region,
                window_duration=window_duration,
                min_speed=min_speed,
            )
        )

    # Z-scores over the session: unmeasured trials carry NaN metrics, so they
    # get NaN z-scores and stay out of the baseline.
    head_sweeps = np.array([m.head_sweep for m in measured], dtype=np.float64)
    speeds = np.array([m.mean_speed for m in measured], dtype=np.float64)
    z_head_sweeps, z_speed_inv = normalize_vte_scores(head_sweeps, speeds)

    trial_results = []
    for m, z_head_sweep, z_speed in zip(
        measured, z_head_sweeps, z_speed_inv, strict=True
    ):
        vte_idx = compute_vte_index(float(z_head_sweep), float(z_speed), alpha=alpha)
        trial_results.append(
            VTETrialResult(
                head_sweep_magnitude=m.head_sweep,
                z_head_sweep=float(z_head_sweep),
                mean_speed=m.mean_speed,
                min_observed_speed=m.min_observed_speed,
                z_speed_inverse=float(z_speed),
                vte_index=float(vte_idx),
                is_vte=classify_vte(vte_idx, threshold=vte_threshold),
                window_start=m.window_start,
                window_end=m.window_end,
                status=m.status,
            )
        )
    return VTESessionResult(trial_results=tuple(trial_results))


@dataclass(frozen=True)
class _TrialMeasurement:
    """Raw pre-decision metrics of one session trial, before z-scoring."""

    status: VTETrialStatus
    head_sweep: float = np.nan
    mean_speed: float = np.nan
    min_observed_speed: float = np.nan
    window_start: float = np.nan
    window_end: float = np.nan


def _measure_trial(
    index: int,
    trial: Trial,
    times: NDArray[np.float64],
    positions: NDArray[np.float64],
    support: ObservationSupport,
    *,
    env: Environment,
    region_bins: NDArray[np.int_],
    in_epochs: NDArray[np.bool_],
    decision_region: str,
    window_duration: float,
    min_speed: float,
) -> _TrialMeasurement:
    """Raw metrics of one trial, or the status saying why it has none."""
    from neurospatial.behavior.decisions import _first_entry, _pre_decision_window

    span = f"Trial {index} spanning [{trial.start_time:.3f}, {trial.end_time:.3f}] s"
    # The trial's samples, start_time <= t <= end_time, are contiguous.
    first = int(np.searchsorted(times, trial.start_time, side="left"))
    stop = int(np.searchsorted(times, trial.end_time, side="right"))
    if stop <= first:
        warnings.warn(
            f"{span} has no samples between its start and end times, so its VTE "
            f"metrics are NaN. The recording spans [{times[0]:.3f}, "
            f"{times[-1]:.3f}] s; check that trial times are in seconds on the "
            "same clock as times.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
        return _TrialMeasurement(status="no_samples")
    trial_times = times[first:stop]
    # The entry is the trial's first decision-region sample inside epochs.
    entry = _first_entry(
        env.bin_at(positions[first:stop]), region_bins, in_epochs[first:stop]
    )
    if entry is None:
        warnings.warn(
            f"{span} never enters decision region '{decision_region}' (inside "
            "epochs), so its VTE metrics are NaN.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
        return _TrialMeasurement(status="no_entry")
    entry_time = float(trial_times[entry])
    window_start = max(entry_time - window_duration, float(trial.start_time))

    # The pre-decision window comes from this trial's samples only, so it
    # stops at the trial start; the trial's interval gates are the session's
    # gates between its first and last sample.
    in_trial = _pre_decision_window(
        trial_times,
        support.interval_gates[first : stop - 1],
        entry_time,
        window_duration,
    )
    window = slice(first + in_trial.start, first + in_trial.stop)
    n_observed = int(np.count_nonzero(support.sample_valid[window]))
    if n_observed < 3:
        warnings.warn(
            f"{span}: only {n_observed} observed sample(s) between the trial "
            f"start and its decision-region entry at {entry_time:.3f} s; heading "
            "analysis needs at least 3, so its VTE metrics are NaN.",
            UserWarning,
            stacklevel=_external_stacklevel(),
        )
        return _TrialMeasurement(
            status="too_few_samples", window_start=window_start, window_end=entry_time
        )
    head_sweep, mean_speed, min_observed_speed = _window_metrics(
        times, positions, support, window, min_speed=min_speed
    )
    return _TrialMeasurement(
        status="ok",
        head_sweep=head_sweep,
        mean_speed=mean_speed,
        min_observed_speed=min_observed_speed,
        window_start=window_start,
        window_end=entry_time,
    )
