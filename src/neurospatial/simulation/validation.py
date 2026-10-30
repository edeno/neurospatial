"""Validation functions for comparing detected vs ground truth spatial firing patterns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from neurospatial._exceptions import _format_error
from neurospatial._results import ReadOnlyArraysMixin, ResultMixin

#: Default center-error limit, in true field widths (Gaussian standard
#: deviations), calibrated by simulation; see ``validate_simulation`` Notes.
CENTER_ERROR_WIDTHS = 1.5

#: Default minimum correlation between detected and true field shapes,
#: calibrated by simulation; see ``validate_simulation`` Notes.
MIN_FIELD_CORRELATION = 0.9

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from neurospatial.simulation.session import SimulationSession


def _select_unit_indices(
    session: SimulationSession, unit_ids: list[int] | None, operation: str
) -> list[int]:
    """Resolve unit labels to aligned spike/model rows of an allocentric session."""
    from neurospatial.geometry import require_cartesian
    from neurospatial.simulation.session import SimulationSession

    if not isinstance(session, SimulationSession):
        raise TypeError(
            f"session must be a SimulationSession instance, got {type(session).__name__}.\n"
            "Why: validation and plotting need the simulator's ground truth.\n"
            "Fix: pass the holder returned by simulate_session() or open_field_session()."
        )
    require_cartesian(
        session.env.coordinate_system,
        operation,
        why="It estimates each unit's rate map from the session's positions "
        "with compute_spatial_rate, which needs the animal's allocentric "
        "positions, and measures field centers as straight-line distances.",
        fix="simulate the session on the allocentric environment the animal moves in",
    )
    if unit_ids is None:
        return list(range(len(session.unit_ids)))
    valid_ids = session.unit_ids.tolist()
    unknown = [label for label in unit_ids if label not in valid_ids]
    if unknown:
        raise ValueError(
            f"unit_ids contains unknown labels {unknown}. Valid labels: {valid_ids}.\n"
            "Why: unit_ids selects labels, not row indices.\n"
            "Fix: choose unit_ids from session.unit_ids."
        )
    return [valid_ids.index(label) for label in unit_ids]


def _field_width(truth: Mapping[str, Any]) -> float:
    """A ground truth's place field width (the largest axis), or NaN."""
    if "width" not in truth:
        return np.nan
    return float(np.max(np.atleast_1d(np.asarray(truth["width"], dtype=float))))


def _nan_stat(values: NDArray[np.float64], stat: Any) -> float:
    """``stat`` of the finite values, or NaN when there are none."""
    finite = values[np.isfinite(values)]
    return float(stat(finite)) if finite.size else float("nan")


@dataclass(frozen=True, repr=False, eq=False)
class SimulationValidationResult(ReadOnlyArraysMixin, ResultMixin):
    """Recovery of simulated place fields, returned by ``validate_simulation``.

    One entry per selected unit, in the requested order. A place unit is
    evaluated when both its center error and its field correlation could be
    measured; it passes when both meet their thresholds. Units whose ground
    truth is not a place cell (border, grid and other cells of mixed
    sessions) are excluded: they are listed but do not count toward any
    verdict. Arrays are read-only.

    Attributes
    ----------
    unit_ids : NDArray[np.int64], shape (n_selected,)
        Unit labels, in the requested order.
    center_errors : NDArray[np.float64], shape (n_selected,)
        Distance from the peak of the estimated rate map to the true field
        center, in environment units. NaN when it could not be measured.
    correlations : NDArray[np.float64], shape (n_selected,)
        Pearson correlation between the estimated and true rate maps over
        bins where both are finite. NaN when it could not be measured.
    max_center_errors : NDArray[np.float64], shape (n_selected,)
        Center-error limit applied to each unit, in environment units.
    spike_counts : NDArray[np.int64], shape (n_selected,)
        Number of simulated spikes of each unit.
    unit_status : tuple[str, ...]
        For each unit: ``'passed'`` or ``'failed'`` for an evaluated place
        unit, ``'not_evaluable'`` for a place unit whose metrics could not
        be measured, and ``'excluded'`` for a unit that is not a place cell.
    skip_reasons : tuple[str | None, ...]
        Why a unit is not evaluated (for example ``'no_spikes'``, or
        ``'not_a_place_field'`` for an excluded unit), or None for an
        evaluated unit.
    min_correlation : float
        Correlation threshold applied to every unit.
    center_error_rule : str
        How ``max_center_errors`` was set: ``'1.5 x true field width'`` by
        default, or ``'fixed'`` for a ``max_center_error`` passed by the
        caller.
    max_center_error : float | None
        The fixed center-error limit passed by the caller, in environment
        units, or None when each unit's limit follows its field width.
    method : str
        Rate-map estimator used for the detected fields.
    units : str | None
        Environment length units of the center errors.
    n_session_units : int
        Number of units in the validated session, selected or not.
    """

    unit_ids: NDArray[np.int64]
    center_errors: NDArray[np.float64]
    correlations: NDArray[np.float64]
    max_center_errors: NDArray[np.float64]
    spike_counts: NDArray[np.int64]
    unit_status: tuple[str, ...]
    skip_reasons: tuple[str | None, ...]
    min_correlation: float
    center_error_rule: str
    max_center_error: float | None
    method: str
    units: str | None
    n_session_units: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit_status", tuple(self.unit_status))
        object.__setattr__(self, "skip_reasons", tuple(self.skip_reasons))
        self._own_arrays()

    @property
    def n_selected(self) -> int:
        """Number of selected units."""
        return len(self.unit_ids)

    @property
    def n_evaluated(self) -> int:
        """Number of place units with both metrics measured."""
        return sum(status in ("passed", "failed") for status in self.unit_status)

    @property
    def n_excluded(self) -> int:
        """Number of selected units that are not place cells."""
        return self.unit_status.count("excluded")

    @property
    def passed(self) -> bool:
        """True when at least one place unit was evaluated and every place unit passed.

        Excluded units do not count. A place unit that could not be evaluated
        blocks a pass, and a selection with no evaluated place unit cannot
        pass.
        """
        place = [status for status in self.unit_status if status != "excluded"]
        return bool(place) and all(status == "passed" for status in place)

    def summary(self) -> dict[str, Any]:
        """Flat dict of scalar headline metrics.

        Means and extremes are over the units where each metric was
        measured; NaN when it was measured for none.
        """
        return {
            "n_selected": self.n_selected,
            "n_evaluated": self.n_evaluated,
            "n_excluded": self.n_excluded,
            "n_passed": self.unit_status.count("passed"),
            "passed": self.passed,
            "mean_center_error": _nan_stat(self.center_errors, np.mean),
            "largest_center_error": _nan_stat(self.center_errors, np.max),
            "mean_correlation": _nan_stat(self.correlations, np.mean),
            "smallest_correlation": _nan_stat(self.correlations, np.min),
        }

    def summary_table(self) -> pd.DataFrame:
        """One row per unit, indexed by ``unit_id``.

        Returns
        -------
        pandas.DataFrame
            Columns ``center_error``, ``correlation``, ``max_center_error``,
            ``spike_count``, ``status`` and ``skip_reason``. ``df.attrs``
            holds ``method``, ``units``, ``min_correlation`` and
            ``center_error_rule``.
        """
        import pandas as pd

        df = pd.DataFrame(
            {
                "center_error": self.center_errors,
                "correlation": self.correlations,
                "max_center_error": self.max_center_errors,
                "spike_count": self.spike_counts,
                "status": list(self.unit_status),
                "skip_reason": list(self.skip_reasons),
            },
            index=pd.Index(np.array(self.unit_ids), name="unit_id"),
        )
        df.attrs.update(
            method=self.method,
            units=self.units,
            min_correlation=self.min_correlation,
            center_error_rule=self.center_error_rule,
        )
        return df

    def report(self) -> str:
        """Human-readable summary of the validation.

        Returns
        -------
        str
            Counts of selected, evaluated, unevaluable and excluded units,
            the mean and extreme center errors and correlations with their
            thresholds, and the overall verdict. Each metric's verdict, like
            the overall one, is over the place units only and needs at least
            one of them.
        """
        stats = self.summary()
        units = self.units or "units"
        place = np.array(self.unit_status, dtype=object) != "excluded"
        errors = self.center_errors[place]
        correlations = self.correlations[place]
        errors_pass = bool(place.any()) and bool(
            np.all(np.isfinite(errors) & (errors <= self.max_center_errors[place]))
        )
        corrs_pass = bool(place.any()) and bool(
            np.all(np.isfinite(correlations) & (correlations >= self.min_correlation))
        )
        threshold = (
            f"{CENTER_ERROR_WIDTHS:g} x each cell's true field width"
            if self.max_center_error is None
            else f"{self.max_center_error:.2f}"
        )
        error_std = _nan_stat(self.center_errors, np.std)
        corr_std = _nan_stat(self.correlations, np.std)
        return f"""Validation Results:
===================
Cells selected: {self.n_selected} (of {self.n_session_units} total)
Cells fully evaluated: {self.n_evaluated}
Cells not evaluable: {self.unit_status.count("not_evaluable")}
Cells excluded (not place cells): {self.n_excluded}

Center Errors ({units}):
  Mean: {stats["mean_center_error"]:.2f} ± {error_std:.2f}
  Max:  {stats["largest_center_error"]:.2f}
  Threshold: {threshold}
  Passed: {errors_pass}

Field Correlations:
  Mean: {stats["mean_correlation"]:.3f} ± {corr_std:.3f}
  Min:  {stats["smallest_correlation"]:.3f}
  Threshold: {self.min_correlation:.3f}
  Passed: {corrs_pass}

Overall: {"PASSED" if self.passed else "FAILED"}
"""

    def plot(self, ax: Axes | None = None) -> Axes:
        """Center error against field correlation for each evaluated unit.

        Passed units are drawn in one color and failed units in another;
        the dashed lines mark the correlation threshold and, when every unit
        has the same limit, the center-error limit.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure's axes when None.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        status = np.array(self.unit_status, dtype=object)
        evaluated = (status == "passed") | (status == "failed")
        ax.scatter(
            self.center_errors[evaluated],
            self.correlations[evaluated],
            c=np.where(status[evaluated] == "passed", "tab:blue", "tab:red"),
        )
        ax.axhline(self.min_correlation, color="k", linestyle="--", alpha=0.5)
        limits = np.unique(self.max_center_errors[np.isfinite(self.max_center_errors)])
        if limits.size == 1:
            ax.axvline(limits[0], color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel(f"Center error ({self.units or 'units'})")
        ax.set_ylabel("Field correlation")
        ax.set_title(
            f"{self.n_evaluated} of {self.n_selected} units evaluated, "
            f"{self.summary()['n_passed']} passed"
        )
        return ax


def validate_simulation(
    session: SimulationSession,
    *,
    unit_ids: list[int] | None = None,
    method: Literal["diffusion_kde", "gaussian_kde", "binned"] = "diffusion_kde",
    max_center_error: float | None = None,
    min_correlation: float | None = None,
    **kwargs: Any,
) -> SimulationValidationResult:
    """Validate simulation by comparing detected place fields to ground truth.

    Computes place fields from spike data and compares detected field centers
    and firing patterns to the known ground truth parameters. Useful for
    validating place field detection algorithms and simulation quality.

    Parameters
    ----------
    session : SimulationSession
        Simulated recording with spikes, tracking and ground truth.
    unit_ids : list[int] | None, optional
        Unit labels from session.unit_ids to validate, in requested order.
        None validates all units.
    method : {'diffusion_kde', 'gaussian_kde', 'binned'}, optional
        Smoothing method for computing place fields (default: 'diffusion_kde').
    max_center_error : float | None, optional
        Maximum acceptable center error in environment units, the same for
        every cell. If None, each cell's limit is 1.5 times its true field
        width (the largest axis of ``ground_truth['width']``); a cell whose
        ground truth has no width cannot pass. Calibrated; see Notes.
    min_correlation : float | None, optional
        Minimum acceptable correlation between detected and true fields.
        If None, 0.9. Calibrated; see Notes.
    **kwargs
        Rate-map parameters of ``compute_spatial_rate()``. Every evaluated
        cell's map is estimated in one ``compute_spatial_rates()`` call.

    Returns
    -------
    SimulationValidationResult
        Per-unit center errors, field correlations, limits, spike counts,
        status and skip reasons, with ``n_evaluated``, ``n_excluded``,
        ``passed``, ``summary()``, ``summary_table()``, ``report()`` and
        ``plot()``. Units that are not place cells are ``'excluded'`` and do
        not count toward the verdict. ``passed`` is True only when at least
        one place unit was selected and every selected place unit has both
        metrics measured and meets both thresholds.

    Raises
    ------
    ValueError
        If unit_ids contains unknown labels.
    TypeError
        If a keyword argument is not a parameter of ``compute_spatial_rate()``.
    CartesianOnlyError
        If the session's environment is egocentric polar: rate maps need the
        animal's allocentric positions.

    Warns
    -----
    UserWarning
        If no selected unit could be evaluated, if a selected unit is not a
        place cell (its ground truth ``cell_type`` is not ``'place'``;
        excluded as ``'not_a_place_field'``), or if a place unit's ground
        truth has no ``'center'`` or ``'width'``
        (``'unsupported_ground_truth'``).

    Examples
    --------
    Validate a one-minute linear-track session of three place cells:

    >>> from neurospatial.simulation import linear_track_session, validate_simulation
    >>> session = linear_track_session(
    ...     duration=60.0, n_place_cells=3, n_laps=6, seed=42
    ... )
    >>> result = validate_simulation(session)
    >>> result.n_evaluated, result.passed
    (3, True)
    >>> table = result.summary_table()
    >>> list(table.columns)
    ['center_error', 'correlation', 'max_center_error', 'spike_count', 'status', 'skip_reason']

    A stricter fixed limit applies to every cell:

    >>> strict = validate_simulation(session, max_center_error=0.5)
    >>> strict.summary_table().attrs["center_error_rule"]
    'fixed'

    See Also
    --------
    simulate_session : Create complete simulation session
    compute_spatial_rate : Compute spatial firing rate from spike data
    plot_session_summary : Visualize complete session

    Notes
    -----
    **Validation Workflow**:

    1. For each place cell, compute place field from spike data using
       specified method. Units whose ground truth ``cell_type`` is not
       ``'place'`` (border, grid and other cells of mixed sessions) are
       excluded: listed, but not part of any verdict.
    2. Detect field center as peak of computed rate map
    3. Compare detected center to ground truth center (Euclidean distance)
    4. Compute correlation between detected and true rate maps. The true map
       is rebuilt from the ground truth's ``center``, ``width`` and
       ``metric``, so a geodesic cell is compared with its geodesic field. A
       firing ``condition`` is not applied: its effect on the map depends on
       the trajectory.
    5. Aggregate statistics and determine pass/fail based on thresholds

    **Default Thresholds**:

    - Center error: 1.5 × each cell's true field width (Gaussian standard
      deviation)
    - Correlation: 0.9

    Both were calibrated by simulation: ``open_field_session`` (100 cm
    arena, 2 cm bins, default 12.4 Hz place cells, default
    ``diffusion_kde`` smoothing), 8 seeds × 16 cells for each of 180 s and
    600 s sessions and 6, 11.1 and 17.5 cm field widths. In five of the six
    conditions at least 95% of correctly specified cells pass (95.3-100%),
    while 93-100% of cells whose ground-truth center is moved by one field
    width fail, and every cell whose ground-truth width is tripled fails;
    the correlation threshold catches most mis-specified cells, as a
    one-width shift leaves the center error near 1 width. The exception is
    180 s with 6 cm fields, whose small fields are visited briefly: only
    69% of correct cells pass, and no pair of thresholds that passes 95% of them
    (about 3.4 widths and a correlation of 0.33) rejects mis-specified
    cells. Short sessions with small fields cannot validate each cell;
    lengthen the session or pass thresholds for your own design.
    Missing metrics remain NaN and prevent an overall pass; a partial
    report is not evidence that every selected unit was recovered. Inspect
    ``unit_status``, ``skip_reasons``, and ``spike_counts`` to distinguish failed
    recovery from units that could not be evaluated. A place unit whose
    ground truth lacks ``center`` or ``width`` (``'unsupported_ground_truth'``)
    also blocks a pass: it is declared a place cell, so a pass would claim
    a recovery that was never checked. Deselect it with ``unit_ids`` if it is
    not meant to be validated.

    **Method Selection**:

    - 'diffusion_kde': Graph-based boundary-aware smoothing (recommended)
    - 'gaussian_kde': Standard Gaussian kernel (fast, ignores boundaries)
    - 'binned': Simple binning without smoothing (noisy but unbiased)
    """
    # Import here to avoid circular dependency
    import inspect

    from scipy.stats import pearsonr

    from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates
    from neurospatial.simulation.models import PlaceCellModel

    # The maps come from one batch call, but each is a single-cell estimate:
    # batch-only options (n_jobs, dtype) are refused.
    unknown = sorted(
        set(kwargs) - set(inspect.signature(compute_spatial_rate).parameters)
    )
    if unknown:
        raise TypeError(
            _format_error(
                f"validate_simulation() got unexpected keyword arguments {unknown}.",
                fix="Pass only rate-map parameters of compute_spatial_rate().",
            )
        )

    cell_indices = _select_unit_indices(session, unit_ids, "validate_simulation()")
    env = session.env
    spike_times = session.spike_times
    positions = session.positions
    times = session.times
    ground_truth = session.ground_truth

    # Thresholds: by default each cell's center error is judged against its
    # own true field width (calibrated; see Notes).
    if max_center_error is None:
        center_error_limits = np.array(
            [
                CENTER_ERROR_WIDTHS
                * _field_width(ground_truth[int(session.unit_ids[index])])
                for index in cell_indices
            ]
        )
    else:
        center_error_limits = np.full(len(cell_indices), float(max_center_error))

    if min_correlation is None:
        min_correlation = MIN_FIELD_CORRELATION

    # Keep unavailable metrics missing, with diagnostics aligned to unit labels.
    selected_unit_ids = session.unit_ids[cell_indices]
    spike_counts = np.array(
        [len(spike_times[index]) for index in cell_indices], dtype=np.int64
    )
    center_errors = np.full(len(cell_indices), np.nan)
    correlations = np.full(len(cell_indices), np.nan)
    skip_reasons: list[str | None] = [None] * len(cell_indices)

    # Place units with a ground-truth field and at least one spike can be
    # evaluated; the rest keep the reason they cannot.
    not_place: dict[int, str] = {}
    unsupported: list[int] = []
    evaluable: list[int] = []
    for i, cell_idx in enumerate(cell_indices):
        cell_key = int(session.unit_ids[cell_idx])
        gt = ground_truth[cell_key]
        cell_type = gt.get("cell_type", "place")
        if cell_type != "place":
            not_place[cell_key] = str(cell_type)
            skip_reasons[i] = "not_a_place_field"
        elif "center" not in gt or "width" not in gt:
            unsupported.append(cell_key)
            skip_reasons[i] = "unsupported_ground_truth"
        elif len(spike_times[cell_idx]) == 0:
            skip_reasons[i] = "no_spikes"
        else:
            evaluable.append(i)

    # One batch call shares the occupancy and smoothing across the cells.
    # compute_spatial_rates also takes method="glm"; it then still fits each
    # cell its own penalty, as for a single cell.
    detected_fields = np.empty((0, env.n_bins))
    if evaluable:
        rate_kwargs = {**kwargs, "pooled": False} if str(method) == "glm" else kwargs
        detected_fields = np.asarray(
            compute_spatial_rates(
                env,
                [spike_times[cell_indices[i]] for i in evaluable],
                times,
                positions,
                method=method,
                **rate_kwargs,
            ).firing_rates,
            dtype=np.float64,
        )

    for i, detected_field in zip(evaluable, detected_fields, strict=True):
        gt = ground_truth[int(session.unit_ids[cell_indices[i]])]

        # Find detected center (peak of rate map). Bins the smoothing cannot
        # resolve are NaN; with no finite bin there is no peak to compare.
        if not np.any(np.isfinite(detected_field)):
            skip_reasons[i] = "no_finite_rate_bins"
            continue
        peak_bin = int(
            np.argmax(np.where(np.isfinite(detected_field), detected_field, -np.inf))
        )
        detected_center = env.bin_centers[peak_bin]

        # Compute center error (Euclidean distance)
        center_error = float(np.linalg.norm(detected_center - np.asarray(gt["center"])))
        center_errors[i] = center_error
        if not np.isfinite(center_error):
            skip_reasons[i] = "nonfinite_center_error"
            continue

        # The true field's shape at each bin, rebuilt with the simulated
        # cell's own distance metric. Pearson correlation ignores a field's
        # offset and positive scale, so a unit field on a zero baseline has
        # the same correlation as the cell's own rates.
        true_field = PlaceCellModel(
            env,
            **{key: gt[key] for key in ("center", "width", "metric") if key in gt},
            max_rate=1.0,
            baseline_rate=0.0,
        ).firing_rate(np.zeros(env.n_bins), env.bin_centers)

        # Compute correlation between detected and true fields
        # Correlation needs at least three jointly finite, nonconstant bins.
        valid_mask = np.isfinite(detected_field) & np.isfinite(true_field)
        if np.sum(valid_mask) < 3:
            skip_reasons[i] = "insufficient_finite_bins"
        elif (
            np.ptp(detected_field[valid_mask]) == 0
            or np.ptp(true_field[valid_mask]) == 0
        ):
            skip_reasons[i] = "constant_rate_map"
        else:
            corr, _ = pearsonr(
                detected_field[valid_mask],
                true_field[valid_mask],
            )
            correlations[i] = corr
            if not np.isfinite(corr):
                skip_reasons[i] = "undefined_correlation"

    import warnings

    if not_place:
        warnings.warn(
            f"Units {sorted(not_place)} are not place fields (cell types "
            f"{sorted(set(not_place.values()))}); validate_simulation compares "
            "place fields only, so they are excluded from the verdict "
            "('not_a_place_field').",
            UserWarning,
            stacklevel=2,
        )
    if unsupported:
        warnings.warn(
            f"Units {unsupported}: ground truth has no place field 'center' and "
            "'width' to compare with, so they are marked not evaluable "
            "('unsupported_ground_truth').",
            UserWarning,
            stacklevel=2,
        )

    evaluated = np.isfinite(center_errors) & np.isfinite(correlations)
    meets_thresholds = (center_errors <= center_error_limits) & (
        correlations >= min_correlation
    )
    unit_status = tuple(
        "excluded"
        if reason == "not_a_place_field"
        else "not_evaluable"
        if not complete
        else "passed"
        if meets
        else "failed"
        for reason, complete, meets in zip(
            skip_reasons, evaluated, meets_thresholds, strict=True
        )
    )
    if not evaluated.any():
        warnings.warn(
            f"No cells could be fully validated ({len(cell_indices)} selected). "
            "Inspect skip_reasons and spike_counts for unavailable metrics.",
            UserWarning,
            stacklevel=2,
        )

    return SimulationValidationResult(
        unit_ids=selected_unit_ids,
        center_errors=center_errors,
        correlations=correlations,
        max_center_errors=center_error_limits,
        spike_counts=spike_counts,
        unit_status=unit_status,
        skip_reasons=tuple(skip_reasons),
        min_correlation=float(min_correlation),
        center_error_rule="fixed"
        if max_center_error is not None
        else f"{CENTER_ERROR_WIDTHS:g} x true field width",
        max_center_error=None if max_center_error is None else float(max_center_error),
        method=method,
        units=env.units,
        n_session_units=len(session.unit_ids),
    )


def plot_session_summary(
    session: SimulationSession,
    *,
    unit_ids: list[int] | None = None,
    figsize: tuple[float, float] = (15, 10),
) -> tuple[Figure, NDArray]:
    """Create comprehensive visualization of simulation session.

    Generates a multi-panel figure showing trajectory, spatial rate maps,
    and temporal spiking patterns for a complete simulation session. Useful
    for quickly inspecting simulation quality and spatial tuning properties.

    Parameters
    ----------
    session : SimulationSession
        Complete simulation session from simulate_session().
    unit_ids : list[int] | None, optional
        Unit labels from session.unit_ids to plot. If None, plots first 6 units
        (or fewer if session has < 6 cells). Default: None.
    figsize : tuple[float, float], optional
        Figure size in inches (width, height). Default: (15, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    axes : NDArray
        Array of axis handles for each subplot.

    Raises
    ------
    TypeError
        If session is not a SimulationSession instance.
    ValueError
        If unit_ids contains unknown labels.
    CartesianOnlyError
        If the session's environment is egocentric polar: rate maps need the
        animal's allocentric positions.
    NotImplementedError
        If the environment is 3-D. A flat panel would stack bins at different
        heights; plot each rate map with
        ``env.plot_field(rates, backend='fastplotlib')`` instead.

    Examples
    --------
    Basic usage with default cells:

    >>> import matplotlib.pyplot as plt
    >>> from neurospatial.simulation import open_field_session, plot_session_summary
    >>> session = open_field_session(duration=30.0, n_place_cells=8, seed=42)
    >>> fig, axes = plot_session_summary(session)
    >>> [ax.get_title() for ax in axes[2:8]]
    ['Cell 0', 'Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', 'Cell 5']
    >>> plt.close(fig)

    Plot specific cells, by label:

    >>> fig, axes = plot_session_summary(session, unit_ids=[0, 5, 7])
    >>> [ax.get_title() for ax in axes[2:5]]
    ['Cell 0', 'Cell 5', 'Cell 7']
    >>> plt.close(fig)

    Custom figure size:

    >>> fig, axes = plot_session_summary(session, figsize=(12, 8))
    >>> plt.close(fig)

    See Also
    --------
    simulate_session : Create complete simulation session
    validate_simulation : Validate simulation against ground truth

    Notes
    -----
    **Figure Layout**:

    The figure uses a 4×3 grid layout:

    - Row 0, Cols 0-1: Trajectory colored by time; for a 1-D environment,
      position against time
    - Row 0, Col 2: Session metadata (duration, n_cells, cell_type)
    - Rows 1-2: Rate maps for up to 6 cells (2×3 grid), drawn by
      ``Environment.plot_field``: a heat map over the bins in 2-D, a rate
      curve along the track in 1-D
    - Row 3: Raster plot showing spike times for all cells

    **Rate Map Computation**:

    Rate maps are computed using diffusion_kde method (graph-based
    boundary-aware smoothing) with default bandwidth. This provides
    smooth, accurate spatial tuning curves.

    **Empty Spike Trains**:

    Cells with no spikes will show empty rate maps with a "No spikes"
    message. This is common in short simulations or cells with low rates.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Import here to avoid circular dependency
    from neurospatial.encoding import compute_spatial_rates

    cell_indices = _select_unit_indices(session, unit_ids, "plot_session_summary()")
    if unit_ids is None:
        cell_indices = cell_indices[:6]
    elif len(cell_indices) > 6:
        import warnings

        warnings.warn(
            f"unit_ids has {len(cell_indices)} units. Only first 6 will be plotted.",
            UserWarning,
            stacklevel=2,
        )
        cell_indices = cell_indices[:6]

    env = session.env
    if env.n_dims > 2:
        raise NotImplementedError(
            _format_error(
                f"plot_session_summary() draws 1-D and 2-D sessions; this "
                f"environment is {env.n_dims}-D.",
                why="a flat panel of a 3-D rate map would draw bins at "
                "different heights on top of each other, so each color would "
                "hide the rates beneath it.",
                fix="plot each unit's rate map in 3-D with "
                "env.plot_field(rates, backend='fastplotlib'), using rates "
                "from compute_spatial_rate(env, spike_times, session.times, "
                "session.positions).firing_rate",
            )
        )
    positions = session.positions
    times = session.times
    spike_trains = session.spike_times
    n_cells = len(spike_trains)

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.3)

    # Row 0, Cols 0-1: Trajectory
    ax_traj = fig.add_subplot(gs[0, :2])

    # Row 0, Col 2: Metadata text
    ax_info = fig.add_subplot(gs[0, 2])

    # Rows 1-2: Rate maps (2x3 grid for up to 6 cells)
    axes_rates = []
    for i in range(2):
        for j in range(3):
            axes_rates.append(fig.add_subplot(gs[i + 1, j]))

    # Row 3: Raster plot (spans all columns)
    ax_raster = fig.add_subplot(gs[3, :])

    # Plot 1: Trajectory, colored by time; on a 1-D track, position over time.
    length_units = env.units or "units"
    if env.n_dims == 1:
        scatter = ax_traj.scatter(
            times, positions[:, 0], c=times, cmap="viridis", s=1, alpha=0.5
        )
        ax_traj.set_xlabel("Time (s)")
        ax_traj.set_ylabel(f"Position ({length_units})")
    else:
        scatter = ax_traj.scatter(
            positions[:, 0], positions[:, 1], c=times, cmap="viridis", s=1, alpha=0.5
        )
        ax_traj.set_xlabel(f"X ({length_units})")
        ax_traj.set_ylabel(f"Y ({length_units})")
        ax_traj.set_aspect("equal")
    ax_traj.set_title("Trajectory", fontweight="bold")
    plt.colorbar(scatter, ax=ax_traj, label="Time (s)", fraction=0.046, pad=0.04)

    # Plot 2: Session metadata
    ax_info.axis("off")
    metadata = session.metadata
    info_text = f"""Session Info:
Duration: {times[-1] - times[0]:.1f} s
Bins: {env.n_bins}
Cells: {n_cells}
Type: {metadata.get("cell_type", "mixed")}
Traj: {metadata.get("trajectory_method", "ou")}
"""
    ax_info.text(
        0.1,
        0.9,
        info_text,
        transform=ax_info.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=9,
    )

    # Plots 3-8: Rate maps for selected cells, estimated in one call for the
    # cells that spiked.
    spiking = [cell_idx for cell_idx in cell_indices if len(spike_trains[cell_idx])]
    rate_maps: dict[int, NDArray[np.float64]] = {}
    if spiking:
        rates = compute_spatial_rates(
            env,
            [spike_trains[cell_idx] for cell_idx in spiking],
            times,
            positions,
            method="diffusion_kde",
        ).firing_rates
        rate_maps = dict(zip(spiking, np.asarray(rates, dtype=np.float64), strict=True))
    for ax, cell_idx in zip(axes_rates, cell_indices, strict=False):
        cell_id = int(session.unit_ids[cell_idx])
        if cell_idx not in rate_maps:
            # Empty spike train
            ax.text(
                0.5,
                0.5,
                "No spikes",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Cell {cell_id}", fontsize=10)
            ax.axis("off")
        else:
            env.plot_field(rate_maps[cell_idx], ax=ax, colorbar_label="Rate (Hz)")
            ax.set_title(f"Cell {cell_id}", fontsize=10)

    # Hide unused rate map subplots
    for i in range(len(cell_indices), len(axes_rates)):
        axes_rates[i].axis("off")

    # Plot 9: Raster plot
    for i in range(n_cells):
        spike_times = spike_trains[i]
        if len(spike_times) > 0:
            ax_raster.scatter(
                spike_times, np.full_like(spike_times, i), s=1, c="k", alpha=0.5
            )

    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Unit ID")
    ax_raster.set_yticks(np.arange(n_cells), labels=session.unit_ids)
    ax_raster.set_title("Raster Plot", fontweight="bold")
    ax_raster.set_xlim(times[0], times[-1])
    ax_raster.set_ylim(-0.5, n_cells - 0.5)

    # Overall title
    fig.suptitle("Simulation Session Summary", fontsize=14, fontweight="bold")

    # Store axes in array for return
    all_axes = np.array(
        [ax_traj, ax_info, *axes_rates, ax_raster], dtype=object
    ).reshape(-1)

    return fig, all_axes
