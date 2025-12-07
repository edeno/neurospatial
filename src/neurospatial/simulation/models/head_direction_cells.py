"""Head direction cell models for simulating directional firing patterns."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


class HeadDirectionCellModel:
    """Von Mises head direction cell model.

    Simulates neurons that fire preferentially when an animal faces a particular
    direction. Uses a von Mises distribution (circular Gaussian) for directional
    tuning, which is the standard model for head direction cells.

    Parameters
    ----------
    preferred_direction : float
        Preferred firing direction in radians. Convention: 0 = East, π/2 = North,
        π = West, -π/2 = South (standard mathematical convention).
    concentration : float, optional
        Von Mises concentration parameter (κ). Higher values = sharper tuning.
        Default is 2.0.

        Typical values:

        - κ = 1: Broad tuning (~60° half-width)
        - κ = 2: Moderate tuning (~40° half-width)
        - κ = 5: Sharp tuning (~25° half-width)
        - κ = 10: Very sharp tuning (~18° half-width)

        Relationship to tuning width: HWHM ≈ arccos(1 - ln(2)/κ)
    max_rate : float, optional
        Peak firing rate in Hz (default: 40.0). Typical HD cells fire 20-100 Hz.
    baseline_rate : float, optional
        Baseline firing rate outside preferred direction (default: 1.0 Hz).
    seed : int | None, optional
        Random seed for reproducible random direction selection.

    Attributes
    ----------
    preferred_direction : float
        Preferred firing direction in radians.
    concentration : float
        Von Mises concentration parameter.
    max_rate : float
        Peak firing rate in Hz.
    baseline_rate : float
        Baseline firing rate in Hz.

    Examples
    --------
    Simple head direction cell:

    >>> import numpy as np
    >>> from neurospatial.simulation import HeadDirectionCellModel
    >>>
    >>> # Create HD cell preferring North (π/2 radians)
    >>> hd_cell = HeadDirectionCellModel(
    ...     preferred_direction=np.pi / 2,
    ...     concentration=3.0,
    ...     max_rate=50.0,
    ... )
    >>>
    >>> # Generate head directions (full rotation)
    >>> headings = np.linspace(-np.pi, np.pi, 100)
    >>> rates = hd_cell.firing_rate(headings=headings)
    >>>
    >>> # Peak should be near π/2
    >>> peak_idx = np.argmax(rates)
    >>> abs(headings[peak_idx] - np.pi / 2) < 0.1
    True

    Multiple HD cells with uniform preferred directions:

    >>> n_cells = 8
    >>> preferred_dirs = np.linspace(0, 2 * np.pi, n_cells, endpoint=False)
    >>> hd_cells = [
    ...     HeadDirectionCellModel(preferred_direction=d, concentration=2.0)
    ...     for d in preferred_dirs
    ... ]
    >>>
    >>> # Compute population response
    >>> headings = np.random.uniform(-np.pi, np.pi, 1000)
    >>> population_rates = np.column_stack(
    ...     [cell.firing_rate(headings=headings) for cell in hd_cells]
    ... )
    >>> population_rates.shape
    (1000, 8)

    Using with trajectory simulation:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_ou
    >>> from neurospatial.ops.egocentric import heading_from_velocity
    >>>
    >>> # Create environment and trajectory
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> positions, times = simulate_trajectory_ou(env, duration=10.0, seed=42)
    >>>
    >>> # Compute heading from velocity
    >>> headings = heading_from_velocity(positions, times, min_speed=1.0)
    >>>
    >>> # Compute HD cell firing
    >>> hd_cell = HeadDirectionCellModel(preferred_direction=0.0)
    >>> rates = hd_cell.firing_rate(headings=headings)
    >>> len(rates) == len(headings)
    True

    See Also
    --------
    PlaceCellModel : Gaussian place field model
    GridCellModel : Hexagonal grid pattern model
    head_direction_tuning_curve : Compute HD tuning from spike data
    head_direction_metrics : Analyze HD tuning properties

    Notes
    -----
    **Von Mises Distribution**: The firing rate follows:

        rate = baseline + (max - baseline) * exp(κ * cos(θ - θ_pref)) / exp(κ)

    This is normalized so peak rate equals max_rate when θ = θ_pref.

    **Angle Convention**: Uses standard mathematical convention where 0 = East,
    angles increase counterclockwise. This differs from some HD literature that
    uses 0 = North. Convert with: north_convention = (east_convention + π/2) % 2π

    **Biological Realism**: Real HD cells show:

    - Anticipatory firing (~25ms lead of actual head direction)
    - Speed modulation in some cells
    - Context-dependent remapping

    For anticipatory firing, shift spike times or use time-shifted headings.

    References
    ----------
    Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells
        recorded from the postsubiculum in freely moving rats. I. Description
        and quantitative analysis. Journal of Neuroscience, 10(2), 420-435.

    Sargolini, F. et al. (2006). Conjunctive representation of position,
        direction, and velocity in entorhinal cortex. Science, 312, 758-762.
    """

    def __init__(
        self,
        preferred_direction: float | None = None,
        concentration: float = 2.0,
        max_rate: float = 40.0,
        baseline_rate: float = 1.0,
        seed: int | None = None,
    ) -> None:
        # Initialize random number generator
        rng = np.random.default_rng(seed)

        # Set preferred direction
        if preferred_direction is None:
            # Random direction in [-π, π)
            self.preferred_direction = rng.uniform(-np.pi, np.pi)
        else:
            # Wrap to [-π, π)
            self.preferred_direction = float(
                np.arctan2(
                    np.sin(preferred_direction),
                    np.cos(preferred_direction),
                )
            )

        # Validate concentration
        if concentration <= 0:
            raise ValueError(
                f"concentration must be positive, got {concentration}. "
                "Typical values: 1.0 (broad), 2.0 (moderate), 5.0 (sharp)."
            )
        self.concentration = float(concentration)

        # Validate rates
        if max_rate < 0:
            raise ValueError(f"max_rate must be non-negative, got {max_rate}.")
        if baseline_rate < 0:
            raise ValueError(
                f"baseline_rate must be non-negative, got {baseline_rate}."
            )
        if baseline_rate > max_rate:
            raise ValueError(
                f"baseline_rate ({baseline_rate}) cannot exceed max_rate ({max_rate})."
            )

        self.max_rate = float(max_rate)
        self.baseline_rate = float(baseline_rate)

    def firing_rate(
        self,
        positions: NDArray[np.float64] | None = None,
        times: NDArray[np.float64] | None = None,
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute von Mises head direction firing rate.

        Parameters
        ----------
        positions : NDArray[np.float64], shape (n_time, n_dims), optional
            Position coordinates. Not used for HD cells but accepted for
            compatibility with NeuralModel protocol. If headings is None,
            headings will be computed from position velocity.
        times : NDArray[np.float64], shape (n_time,), optional
            Time points in seconds. Used for computing headings from positions.
        headings : NDArray[np.float64], shape (n_time,), optional
            Head direction in radians at each time point.
            If None and positions/times provided, computed from velocity.
            Convention: 0 = East, π/2 = North (counterclockwise positive).

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each time point.

        Raises
        ------
        ValueError
            If neither headings nor positions are provided.
            If positions provided without times.

        Warnings
        --------
        When computing headings from positions, assumes uniform time sampling.
        For non-uniform sampling, compute headings externally and pass directly.

        Notes
        -----
        The von Mises firing rate is computed as:

            rate = baseline + (max - baseline) * exp(κ * (cos(θ - θ_pref) - 1))

        This formulation ensures:

        - rate = max_rate when θ = θ_pref (since cos(0) - 1 = 0)
        - rate → baseline as θ moves away from θ_pref
        """
        # Get headings from input
        if headings is None:
            if positions is None:
                raise ValueError(
                    "Either headings or positions must be provided. "
                    "For HD cells, headings is preferred."
                )
            if times is None:
                raise ValueError(
                    "times must be provided when computing headings from positions."
                )
            # Compute headings from velocity
            from neurospatial.ops.egocentric import heading_from_velocity

            # Compute dt from times (assumes uniform sampling)
            dt = float(np.median(np.diff(times)))
            headings = heading_from_velocity(positions, dt, min_speed=0.0)

        headings = np.asarray(headings, dtype=np.float64).ravel()

        # Compute angular difference (handles circular wraparound)
        angle_diff = headings - self.preferred_direction

        # Von Mises tuning: exp(κ * (cos(θ - θ_pref) - 1))
        # The -1 normalizes so peak = 1 when θ = θ_pref
        tuning = np.exp(self.concentration * (np.cos(angle_diff) - 1))

        # Scale to firing rate range
        rates = self.baseline_rate + (self.max_rate - self.baseline_rate) * tuning

        return rates

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Dictionary with keys:

            - 'preferred_direction': float - preferred direction in radians
            - 'preferred_direction_deg': float - preferred direction in degrees
            - 'concentration': float - von Mises concentration (κ)
            - 'tuning_width': float - approximate HWHM in radians
            - 'tuning_width_deg': float - approximate HWHM in degrees
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz

        Examples
        --------
        >>> import numpy as np
        >>> from neurospatial.simulation import HeadDirectionCellModel
        >>> hd = HeadDirectionCellModel(
        ...     preferred_direction=np.pi / 4,
        ...     concentration=3.0,
        ...     max_rate=50.0,
        ... )
        >>> abs(hd.ground_truth["preferred_direction"] - np.pi / 4) < 0.01
        True
        >>> hd.ground_truth["concentration"]
        3.0
        >>> hd.ground_truth["max_rate"]
        50.0
        """
        # Compute approximate tuning width (HWHM)
        # From von Mises: at half-max, cos(θ) = 1 - ln(2)/κ
        # So HWHM = arccos(1 - ln(2)/κ)
        cos_half = 1 - np.log(2) / self.concentration
        # Clamp to valid range for arccos
        cos_half = np.clip(cos_half, -1, 1)
        tuning_width = np.arccos(cos_half)

        return {
            "preferred_direction": self.preferred_direction,
            "preferred_direction_deg": float(np.degrees(self.preferred_direction)),
            "concentration": self.concentration,
            "tuning_width": float(tuning_width),
            "tuning_width_deg": float(np.degrees(tuning_width)),
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }

    @property
    def mean_vector_length(self) -> float:
        """Theoretical mean vector length for this concentration.

        Returns
        -------
        float
            Expected Rayleigh vector length (0-1) for ideal tuning curve.

        Notes
        -----
        For von Mises distribution with concentration κ:
            MVL = I₁(κ) / I₀(κ)

        where I₀ and I₁ are modified Bessel functions of the first kind.
        """
        from scipy.special import i0, i1

        return float(i1(self.concentration) / i0(self.concentration))
