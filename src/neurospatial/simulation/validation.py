"""Validation functions for comparing detected vs ground truth spatial firing patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from neurospatial import Environment
    from neurospatial.simulation.session import SimulationSession


def validate_simulation(
    session: SimulationSession | None = None,
    *,
    env: Environment | None = None,
    spike_trains: list[NDArray[np.float64]] | None = None,
    positions: NDArray[np.float64] | None = None,
    times: NDArray[np.float64] | None = None,
    ground_truth: dict[str, Any] | None = None,
    cell_indices: list[int] | None = None,
    smoothing_method: Literal[
        "diffusion_kde", "gaussian_kde", "binned"
    ] = "diffusion_kde",
    max_center_error: float | None = None,
    min_correlation: float | None = None,
    show_plots: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate simulation by comparing detected place fields to ground truth.

    Computes place fields from spike data and compares detected field centers
    and firing patterns to the known ground truth parameters. Useful for
    validating place field detection algorithms and simulation quality.

    Parameters
    ----------
    session : SimulationSession | None, optional
        Complete simulation session. If provided, extracts all needed parameters.
    env : Environment | None, optional
        Spatial environment (required if session not provided).
    spike_trains : list[NDArray[np.float64]] | None, optional
        List of spike time arrays (required if session not provided).
    positions : NDArray[np.float64] | None, optional
        Trajectory positions, shape (n_time, n_dims) (required if session not provided).
    times : NDArray[np.float64] | None, optional
        Time points, shape (n_time,) (required if session not provided).
    ground_truth : dict[str, Any] | None, optional
        Ground truth parameters for each cell (required if session not provided).
    cell_indices : list[int] | None, optional
        Indices of cells to validate. If None, validates all cells.
    smoothing_method : {'diffusion_kde', 'gaussian_kde', 'binned'}, optional
        Smoothing method for computing place fields (default: 'diffusion_kde').
    max_center_error : float | None, optional
        Maximum acceptable center error in environment units. If None, uses
        2 * mean(bin_sizes) as threshold.
    min_correlation : float | None, optional
        Minimum acceptable correlation between detected and true fields.
        If None, uses 0.5 as threshold.
    show_plots : bool, optional
        If True, creates diagnostic plots (default: False).
    **kwargs
        Additional parameters passed to compute_place_field().

    Returns
    -------
    results : dict[str, Any]
        Dictionary containing:

        - 'center_errors': NDArray[np.float64], shape (n_cells,)
            Euclidean distance between detected and true centers (in env units).
        - 'correlations': NDArray[np.float64], shape (n_cells,)
            Pearson correlation between detected and true rate maps.
        - 'summary': str
            Human-readable summary of validation results.
        - 'passed': bool
            True if all cells meet the specified thresholds.
        - 'plots': Figure | None
            Matplotlib figure with diagnostic plots (only if show_plots=True).

    Raises
    ------
    ValueError
        If neither session nor all individual parameters are provided.
    ValueError
        If ground_truth is missing or incomplete.

    Examples
    --------
    Validate a simulated session:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_session, validate_simulation
    >>>
    >>> # Create environment
    >>> data = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Simulate session with place cells
    >>> session = simulate_session(
    ...     env,
    ...     duration=5.0,
    ...     n_cells=3,
    ...     cell_type="place",
    ...     seed=42,
    ...     show_progress=False,
    ... )
    >>>
    >>> # Validate detected vs true place fields
    >>> results = validate_simulation(session)
    >>> print(results["summary"])
    Validation Results:
    ...
    >>> print(f"All cells passed: {results['passed']}")
    All cells passed: True

    Validate with custom thresholds:

    >>> results = validate_simulation(
    ...     session,
    ...     max_center_error=5.0,  # Maximum 5 cm error
    ...     min_correlation=0.8,  # Minimum 0.8 correlation
    ... )
    >>> print(f"Mean center error: {results['center_errors'].mean():.2f} cm")
    Mean center error: 3.45 cm

    Validate specific cells with diagnostic plots:

    >>> results = validate_simulation(
    ...     session,
    ...     cell_indices=[0, 1, 2],  # Only first 3 cells
    ...     show_plots=True,
    ... )
    >>> # results['plots'] contains matplotlib figure

    See Also
    --------
    simulate_session : Create complete simulation session
    compute_place_field : Compute place field from spike data
    plot_session_summary : Visualize complete session

    Notes
    -----
    **Validation Workflow**:

    1. For each cell, compute place field from spike data using specified method
    2. Detect field center as peak of computed rate map
    3. Compare detected center to ground truth center (Euclidean distance)
    4. Compute correlation between detected and true rate maps
    5. Aggregate statistics and determine pass/fail based on thresholds

    **Default Thresholds**:

    - Center error: 2 * mean(bin_sizes) - accounts for discretization error
    - Correlation: 0.5 - reasonable lower bound for noisy place fields

    **Method Selection**:

    - 'diffusion_kde': Graph-based boundary-aware smoothing (recommended)
    - 'gaussian_kde': Standard Gaussian kernel (fast, ignores boundaries)
    - 'binned': Simple binning without smoothing (noisy but unbiased)
    """
    # Import here to avoid circular dependency
    from neurospatial import compute_place_field

    # Parse input parameters
    if session is not None:
        # Import here to avoid circular dependency
        from neurospatial.simulation.session import SimulationSession

        # Validate session type (proper isinstance check)
        if not isinstance(session, SimulationSession):
            raise TypeError(
                f"session must be a SimulationSession instance, got {type(session).__name__}"
            )

        env = session.env
        spike_trains = session.spike_trains
        positions = session.positions
        times = session.times
        ground_truth = session.ground_truth
    elif (
        env is None
        or spike_trains is None
        or positions is None
        or times is None
        or ground_truth is None
    ):
        raise ValueError(
            "Must provide either 'session' or all of "
            "('env', 'spike_trains', 'positions', 'times', 'ground_truth')"
        )

    # Determine which cells to validate
    n_cells = len(spike_trains)
    if cell_indices is None:
        cell_indices = list(range(n_cells))
    else:
        # Validate cell_indices range
        invalid_indices = [idx for idx in cell_indices if idx < 0 or idx >= n_cells]
        if invalid_indices:
            raise ValueError(
                f"cell_indices contains invalid indices {invalid_indices}. "
                f"Valid range is [0, {n_cells - 1}] for {n_cells} cells."
            )

    # Set default thresholds
    if max_center_error is None:
        # Use 2x mean bin size as threshold (accounts for discretization)
        bin_sizes = env.bin_sizes
        max_center_error = 2.0 * float(np.mean(bin_sizes))

    if min_correlation is None:
        min_correlation = 0.5  # Reasonable threshold for noisy place fields

    # Initialize result arrays
    center_errors = np.zeros(len(cell_indices))
    correlations = np.zeros(len(cell_indices))

    # Validate each cell
    for i, cell_idx in enumerate(cell_indices):
        spike_times = spike_trains[cell_idx]

        # Skip cells with no spikes (cannot compute place field)
        if len(spike_times) == 0:
            center_errors[i] = np.nan
            correlations[i] = np.nan
            continue

        # Compute detected place field
        detected_field = compute_place_field(
            env,
            spike_times,
            times,
            positions,
            smoothing_method=smoothing_method,
            **kwargs,
        )

        # Find detected center (peak of rate map)
        peak_bin = int(np.argmax(detected_field))
        detected_center = env.bin_centers[peak_bin]

        # Get ground truth center
        cell_key = f"cell_{cell_idx}"
        if cell_key not in ground_truth:
            raise ValueError(
                f"Ground truth missing for {cell_key}. "
                f"Available keys: {list(ground_truth.keys())}"
            )

        gt = ground_truth[cell_key]
        if "center" not in gt:
            raise ValueError(
                f"Ground truth for {cell_key} missing 'center' field. "
                f"Available fields: {list(gt.keys())}"
            )

        true_center = np.asarray(gt["center"])

        # Compute center error (Euclidean distance)
        center_error = float(np.linalg.norm(detected_center - true_center))
        center_errors[i] = center_error

        # Compute true firing rate at each bin
        # Use same model type to compute true field
        if "width" in gt and "max_rate" in gt:
            # Place cell - compute Gaussian field
            from neurospatial.simulation.models import PlaceCellModel

            # Create temporary model with ground truth parameters
            temp_model = PlaceCellModel(
                env,
                center=true_center,
                width=gt["width"],
                max_rate=gt["max_rate"],
                baseline_rate=gt.get("baseline_rate", 0.001),
            )

            # Compute true firing rate at bin centers
            true_field = temp_model.firing_rate(env.bin_centers)
        else:
            # For non-place cells, skip correlation (not implemented yet)
            correlations[i] = np.nan
            continue

        # Compute correlation between detected and true fields
        # Remove NaN values before correlation
        valid_mask = ~np.isnan(detected_field) & ~np.isnan(true_field)
        if np.sum(valid_mask) > 2:  # Need at least 3 points for correlation
            corr, _ = pearsonr(
                detected_field[valid_mask],
                true_field[valid_mask],
            )
            correlations[i] = corr
        else:
            correlations[i] = np.nan

    # Compute summary statistics (ignoring NaNs)
    valid_errors = center_errors[~np.isnan(center_errors)]
    valid_corrs = correlations[~np.isnan(correlations)]

    # Check if any cells could be validated
    if len(valid_errors) == 0:
        # No cells could be validated (all empty spike trains)
        import warnings

        warnings.warn(
            f"No cells could be validated ({len(cell_indices)} cells checked). "
            "All cells have empty spike trains. Try increasing simulation duration.",
            UserWarning,
            stacklevel=2,
        )

    mean_error = float(np.mean(valid_errors)) if len(valid_errors) > 0 else np.nan
    std_error = float(np.std(valid_errors)) if len(valid_errors) > 0 else np.nan
    max_error = float(np.max(valid_errors)) if len(valid_errors) > 0 else np.nan
    mean_corr = float(np.mean(valid_corrs)) if len(valid_corrs) > 0 else np.nan
    std_corr = float(np.std(valid_corrs)) if len(valid_corrs) > 0 else np.nan
    min_corr = float(np.min(valid_corrs)) if len(valid_corrs) > 0 else np.nan

    # Determine pass/fail
    # If no cells could be validated, fail validation
    if len(valid_errors) == 0:
        errors_pass = False
        corrs_pass = False
        passed = False
    else:
        errors_pass = bool(np.all(valid_errors <= max_center_error))
        corrs_pass = bool(np.all(valid_corrs >= min_correlation))
        passed = bool(errors_pass and corrs_pass)

    # Generate summary string
    units = env.units if env.units else "units"
    summary = f"""Validation Results:
===================
Cells validated: {len(cell_indices)} (of {n_cells} total)

Center Errors ({units}):
  Mean: {mean_error:.2f} ± {std_error:.2f}
  Max:  {max_error:.2f}
  Threshold: {max_center_error:.2f}
  Passed: {errors_pass}

Field Correlations:
  Mean: {mean_corr:.3f} ± {std_corr:.3f}
  Min:  {min_corr:.3f}
  Threshold: {min_correlation:.3f}
  Passed: {corrs_pass}

Overall: {"✓ PASSED" if passed else "✗ FAILED"}
"""

    # Create results dict
    results = {
        "center_errors": center_errors,
        "correlations": correlations,
        "summary": summary,
        "passed": passed,
    }

    # Optionally create diagnostic plots
    if show_plots:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Center error histogram
        ax = axes[0, 0]
        ax.hist(valid_errors, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(max_center_error, color="r", linestyle="--", label="Threshold")
        ax.set_xlabel(f"Center Error ({units})")
        ax.set_ylabel("Count")
        ax.set_title("Center Error Distribution")
        ax.legend()

        # Plot 2: Correlation histogram
        ax = axes[0, 1]
        ax.hist(valid_corrs, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(min_correlation, color="r", linestyle="--", label="Threshold")
        ax.set_xlabel("Correlation")
        ax.set_ylabel("Count")
        ax.set_title("Field Correlation Distribution")
        ax.legend()

        # Plot 3: Error vs Correlation scatter
        ax = axes[1, 0]
        ax.scatter(valid_errors, valid_corrs, alpha=0.6)
        ax.axhline(min_correlation, color="r", linestyle="--", alpha=0.5)
        ax.axvline(max_center_error, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel(f"Center Error ({units})")
        ax.set_ylabel("Correlation")
        ax.set_title("Error vs Correlation")
        ax.grid(True, alpha=0.3)

        # Plot 4: Text summary
        ax = axes[1, 1]
        ax.axis("off")
        ax.text(
            0.1,
            0.9,
            summary,
            transform=ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
        )

        fig.suptitle("Simulation Validation Report", fontsize=14, fontweight="bold")
        plt.tight_layout()

        results["plots"] = fig
    else:
        results["plots"] = None

    return results


def plot_session_summary(
    session: SimulationSession,
    cell_ids: list[int] | None = None,
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
    cell_ids : list[int] | None, optional
        Specific cells to plot rate maps for. If None, plots first 6 cells
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
        If cell_ids contains indices outside valid range [0, n_cells-1].

    Examples
    --------
    Basic usage with default cells:

    >>> import numpy as np
    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_session, plot_session_summary
    >>>
    >>> # Create environment
    >>> data = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(data, bin_size=2.0)
    >>> env.units = "cm"
    >>>
    >>> # Simulate session
    >>> session = simulate_session(
    ...     env, duration=5.0, n_cells=3, seed=42, show_progress=False
    ... )
    >>>
    >>> # Plot summary
    >>> fig, axes = plot_session_summary(session)
    >>> # Figure shows: trajectory, rate maps for first 6 cells, raster plot

    Plot specific cells:

    >>> fig, axes = plot_session_summary(session, cell_ids=[0, 5, 10, 15])
    >>> # Shows rate maps for cells 0, 5, 10, 15 only

    Custom figure size:

    >>> fig, axes = plot_session_summary(session, figsize=(12, 8))

    See Also
    --------
    simulate_session : Create complete simulation session
    validate_simulation : Validate simulation against ground truth

    Notes
    -----
    **Figure Layout**:

    The figure uses a 4×3 grid layout:

    - Row 0, Cols 0-1: Trajectory plot with color-coded time
    - Row 0, Col 2: Session metadata (duration, n_cells, cell_type)
    - Rows 1-2: Rate maps for up to 6 cells (2×3 grid)
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
    from neurospatial import compute_place_field
    from neurospatial.simulation.session import SimulationSession

    # Validate session type
    if not isinstance(session, SimulationSession):
        raise TypeError(
            f"session must be a SimulationSession instance, got {type(session).__name__}. "
            f"Create a session using simulate_session() or SimulationSession(...)"
        )

    # Extract session data
    env = session.env
    positions = session.positions
    times = session.times
    spike_trains = session.spike_trains
    n_cells = len(spike_trains)

    # Determine which cells to plot
    if cell_ids is None:
        # Default to first 6 cells (or fewer if session has < 6)
        cell_ids = list(range(min(6, n_cells)))
    else:
        # Validate cell_ids range
        invalid_ids = [cid for cid in cell_ids if cid < 0 or cid >= n_cells]
        if invalid_ids:
            raise ValueError(
                f"cell_ids contains invalid indices {invalid_ids}. "
                f"Valid range is [0, {n_cells - 1}] for {n_cells} cells."
            )

    # Limit to 6 cells for visualization clarity
    if len(cell_ids) > 6:
        import warnings

        warnings.warn(
            f"cell_ids has {len(cell_ids)} cells. Only first 6 will be plotted.",
            UserWarning,
            stacklevel=2,
        )
        cell_ids = cell_ids[:6]

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

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

    # Plot 1: Trajectory with color-coded time
    # Use scatter plot for color-mapped trajectory
    scatter = ax_traj.scatter(
        positions[:, 0],
        positions[:, 1],
        c=times,
        cmap="viridis",
        s=1,
        alpha=0.5,
    )
    ax_traj.set_xlabel(f"X ({env.units or 'units'})")
    ax_traj.set_ylabel(f"Y ({env.units or 'units'})")
    ax_traj.set_title("Trajectory", fontweight="bold")
    ax_traj.set_aspect("equal")
    plt.colorbar(scatter, ax=ax_traj, label="Time (s)", fraction=0.046, pad=0.04)

    # Plot 2: Session metadata
    ax_info.axis("off")
    metadata = session.metadata
    info_text = f"""Session Info:
Duration: {times[-1]:.1f} s
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

    # Plots 3-8: Rate maps for selected cells
    for i, cell_id in enumerate(cell_ids):
        if i >= len(axes_rates):
            break

        ax = axes_rates[i]
        spike_times = spike_trains[cell_id]

        if len(spike_times) == 0:
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
            # Compute rate map
            rate_map = compute_place_field(
                env, spike_times, times, positions, smoothing_method="diffusion_kde"
            )

            # Plot as 2D heatmap if possible
            if env.layout is not None and hasattr(env.layout, "grid_shape"):
                grid_shape = env.layout.grid_shape

                # Check if rate_map matches full grid (no inactive bins)
                if grid_shape is not None and len(rate_map) == np.prod(grid_shape):
                    # Can reshape directly
                    rate_map_2d = rate_map.reshape(grid_shape)

                    im = ax.imshow(
                        rate_map_2d.T,
                        origin="lower",
                        aspect="equal",
                        cmap="hot",
                        interpolation="nearest",
                    )
                    ax.set_title(f"Cell {cell_id}", fontsize=10)
                    ax.set_xlabel("X bin")
                    ax.set_ylabel("Y bin")
                    plt.colorbar(im, ax=ax, label="Rate (Hz)", fraction=0.046, pad=0.04)
                else:
                    # Has inactive bins - use scatter plot instead
                    scatter = ax.scatter(
                        env.bin_centers[:, 0],
                        env.bin_centers[:, 1],
                        c=rate_map,
                        cmap="hot",
                        s=10,
                    )
                    ax.set_title(f"Cell {cell_id}", fontsize=10)
                    ax.set_xlabel(f"X ({env.units or 'units'})")
                    ax.set_ylabel(f"Y ({env.units or 'units'})")
                    ax.set_aspect("equal")
                    plt.colorbar(
                        scatter, ax=ax, label="Rate (Hz)", fraction=0.046, pad=0.04
                    )
            else:
                # Fallback: plot as 1D trace
                ax.plot(rate_map)
                ax.set_title(f"Cell {cell_id}", fontsize=10)
                ax.set_xlabel("Bin")
                ax.set_ylabel("Rate (Hz)")

    # Hide unused rate map subplots
    for i in range(len(cell_ids), len(axes_rates)):
        axes_rates[i].axis("off")

    # Plot 9: Raster plot
    for i in range(n_cells):
        spike_times = spike_trains[i]
        if len(spike_times) > 0:
            ax_raster.scatter(
                spike_times, np.full_like(spike_times, i), s=1, c="k", alpha=0.5
            )

    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Cell ID")
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
