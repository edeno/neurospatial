"""Head direction cell models for simulating directional firing patterns."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurospatial._angles import wrap_angle
from neurospatial.simulation.models.base import (
    _check_angle,
    _check_positive,
    _draw_rng,
    _model_headings,
    _model_inputs,
    _set_rates,
    _tuned_rate,
)


@dataclass(frozen=True, eq=False)
class HeadDirectionCellModel:
    """Von Mises head direction cell model.

    Simulates neurons that fire preferentially when an animal faces a particular
    direction. Uses a von Mises distribution (circular Gaussian) for directional
    tuning, which is the standard model for head direction cells.

    Parameters
    ----------
    preferred_direction : float, optional
        Preferred firing direction in radians. Convention: 0 = East, π/2 = North,
        π = West, -π/2 = South (standard mathematical convention). If None
        (the default), drawn uniformly from [-π, π) with ``seed``.
    concentration : float, optional
        Von Mises concentration parameter (κ). Higher values = sharper tuning.
        Default is 9.1, a 22.5° half-width at half-maximum: Taube et al.
        (1990) found that postsubicular head-direction cells fire over a
        range of about 90°, falling off roughly linearly on either side of
        the preferred direction, and a triangle 90° wide at its base is half
        its height 22.5° from the peak, so ``κ = ln 2 / (1 - cos 22.5°)``.

        With a zero baseline the rate falls to half its peak where
        ``exp(κ (cos Δ - 1)) = 1/2``, so ``HWHM = arccos(1 - ln(2)/κ)``
        (defined for κ ≥ ln(2)/2):

        - κ = 1: Broad tuning (72° half-width)
        - κ = 2: Moderate tuning (49° half-width)
        - κ = 5: Sharp tuning (31° half-width)
        - κ = 10: Very sharp tuning (21° half-width)
    max_rate : float, optional
        Peak firing rate in Hz (default: 35.0, the mean peak rate of the
        head-direction cells of Taube et al. (1990), whose peak rates ranged
        from 5 to 115 Hz).
    baseline_rate : float, optional
        Firing rate facing away from the preferred direction, in Hz (default:
        0.01). An illustrative value: Taube et al. (1990) report little
        firing outside the directional range but no single rate.
    seed : int | None, optional
        Random seed for reproducible random direction selection. When None
        and the direction is drawn, the model draws a seed from the
        operating system and keeps it in ``seed``, so
        ``dataclasses.replace`` and pickling keep the drawn direction.

    Attributes
    ----------
    ground_truth : dict[str, Any]
        The values the model fires with: read the drawn preferred direction,
        wrapped to (-π, π], here. Every parameter is also an attribute
        holding the argument as passed, so ``preferred_direction`` reads None
        when it is drawn from ``seed``, and ``dataclasses.replace(model,
        seed=2)`` draws again, as the constructor would. Only ``seed``
        differs: a seed drawn for ``seed=None`` is kept there.

    Raises
    ------
    ValueError
        If ``preferred_direction`` is not finite, ``concentration`` is not
        finite and positive, a rate is not finite or is negative, or
        ``baseline_rate`` exceeds ``max_rate``.

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
    >>> # A full turn on the spot, 100 samples over 1 s
    >>> headings = np.linspace(-np.pi, np.pi, 100)
    >>> times = np.linspace(0.0, 1.0, 100)
    >>> positions = np.zeros((100, 2))
    >>> rates = hd_cell.firing_rate(times, positions, headings=headings)
    >>>
    >>> # Peak should be near π/2
    >>> peak_idx = np.argmax(rates)
    >>> bool(abs(headings[peak_idx] - np.pi / 2) < 0.1)
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
    >>> population_rates = np.column_stack(
    ...     [cell.firing_rate(times, positions, headings=headings) for cell in hd_cells]
    ... )
    >>> population_rates.shape
    (100, 8)

    Using with trajectory simulation:

    >>> from neurospatial import Environment
    >>> from neurospatial.simulation import simulate_trajectory_ou
    >>> from neurospatial.behavior import heading_from_velocity
    >>>
    >>> # Create environment and trajectory
    >>> samples = np.random.uniform(0, 100, (1000, 2))
    >>> env = Environment.from_samples(samples, bin_size=2.0)
    >>> env = env.with_metadata(units="cm")  # required for simulate_trajectory_ou
    >>> positions, times = simulate_trajectory_ou(
    ...     env, duration=10.0, seed=42, speed_units="cm"
    ... )
    >>>
    >>> # Heading from velocity, without a speed threshold: with min_speed > 0,
    >>> # slow samples have no heading (NaN) and give NaN rates, which
    >>> # generate_poisson_spikes refuses. generate_population_spikes(...,
    >>> # headings=None) derives a heading that is also defined through stops.
    >>> headings = heading_from_velocity(times, positions)
    >>>
    >>> # Compute HD cell firing
    >>> hd_cell = HeadDirectionCellModel(preferred_direction=0.0)
    >>> rates = hd_cell.firing_rate(times, positions, headings=headings)
    >>> bool(np.isfinite(rates).all())
    True

    See Also
    --------
    PlaceCellModel : Gaussian place field model
    GridCellModel : Hexagonal grid pattern model
    compute_directional_rate : Compute HD tuning from spike data

    Notes
    -----
    **Von Mises Distribution**: The firing rate follows:

        rate = baseline + (max - baseline) * exp(κ * cos(θ - θ_pref)) / exp(κ)

    This is normalized so peak rate equals max_rate when θ = θ_pref.

    **Angle Convention**: Uses standard mathematical convention where 0 = East,
    angles increase counterclockwise. Some HD literature reports compass
    bearings instead (0 = North, increasing clockwise); convert with
    ``bearing = (π/2 - θ) mod 2π``.

    **Biological Realism**: Real HD cells show:

    - Anticipatory firing: anterior thalamic HD cells fire best for the
      direction the head will face about 25 ms later (Blair & Sharp 1995)
    - Speed modulation in some cells
    - Context-dependent remapping

    For anticipatory firing, shift spike times or use time-shifted headings.

    References
    ----------
    Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells
        recorded from the postsubiculum in freely moving rats. I. Description
        and quantitative analysis. Journal of Neuroscience, 10(2), 420-435.

    Blair, H.T., & Sharp, P.E. (1995). Anticipatory head direction signals
        in anterior thalamus: evidence for a thalamocortical circuit that
        integrates angular head motion to compute head direction. Journal of
        Neuroscience, 15(9), 6260-6270.
    """

    _: KW_ONLY
    preferred_direction: float | None = None
    concentration: float = 9.1
    max_rate: float = 35.0
    baseline_rate: float = 0.01
    seed: int | None = None
    _preferred_direction: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        direction = self.preferred_direction
        if direction is None:
            direction = float(_draw_rng(self).uniform(-np.pi, np.pi))
        _check_angle("preferred_direction", direction)
        _check_positive("concentration", self.concentration, units=None)
        _set_rates(self)
        if self.preferred_direction is not None:
            object.__setattr__(self, "preferred_direction", float(direction))
        for name, value in (
            (
                "_preferred_direction",
                float(wrap_angle(np.asarray(direction, dtype=np.float64))),
            ),
            ("concentration", float(self.concentration)),
        ):
            object.__setattr__(self, name, value)

    @property
    def requires_headings(self) -> bool:
        """True: the rate depends on head direction only."""
        return True

    def firing_rate(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        *,
        headings: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute von Mises head direction firing rate.

        Parameters
        ----------
        times : NDArray[np.float64], shape (n_time,)
            Sample times in seconds; the rate does not depend on them.
        positions : NDArray[np.float64], shape (n_time, n_dims)
            Positions; only their number is checked, as position does not
            change a head-direction cell's rate.
        headings : NDArray[np.float64], shape (n_time,)
            Head direction in radians at each time point (0 = East, π/2 =
            North, counterclockwise positive). Required.

        Returns
        -------
        firing_rate : NDArray[np.float64], shape (n_time,)
            Firing rate in Hz at each time point; NaN for a missing heading.

        Raises
        ------
        ValueError
            If ``headings`` is None or not one angle per time, or
            ``positions`` does not have one row per time.

        Notes
        -----
        The von Mises firing rate is computed as:

            rate = baseline + (max - baseline) * exp(κ * (cos(θ - θ_pref) - 1))

        This formulation ensures:

        - rate = max_rate when θ = θ_pref (since cos(0) - 1 = 0)
        - rate → baseline as θ moves away from θ_pref
        """
        call = "HeadDirectionCellModel.firing_rate()"
        times, _positions = _model_inputs(times, positions, None, call=call)
        headings = _model_headings(headings, len(times), call=call)
        tuning = np.exp(
            self.concentration * (np.cos(headings - self._preferred_direction) - 1)
        )
        return _tuned_rate(self, tuning)

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Return ground truth parameters for validation.

        Returns
        -------
        parameters : dict[str, Any]
            Dictionary with keys:

            - 'cell_type': 'head_direction'
            - 'preferred_direction': float - preferred direction in radians
            - 'preferred_direction_deg': float - preferred direction in degrees
            - 'concentration': float - von Mises concentration (κ)
            - 'tuning_width': float - half-width at half-maximum in radians:
              the angle from the preferred direction at which the rate falls
              to ``max_rate / 2``, as ``DirectionalRateResult.tuning_width``
              measures it. NaN when the rate never falls that low.
            - 'tuning_width_deg': float - the same half-width in degrees
            - 'max_rate': float - peak firing rate in Hz
            - 'baseline_rate': float - baseline firing rate in Hz

        Notes
        -----
        The rate is ``b + (m - b) exp(κ (cos Δ - 1))``, so it equals ``m / 2``
        where ``exp(κ (cos Δ - 1)) = q`` with ``q = (m/2 - b) / (m - b)``,
        that is at ``Δ = arccos(1 + ln(q) / κ)``. The rate never falls to
        half the peak when ``b ≥ m / 2`` (``q ≤ 0``) or when even the minimum,
        at ``Δ = π``, is above it (``1 + ln(q) / κ < -1``). With ``b = 0``,
        ``q = 1/2`` and the half-width is ``arccos(1 - ln(2) / κ)``.

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
        peak, baseline = self.max_rate, self.baseline_rate
        tuning_width = np.nan
        if baseline < peak / 2:
            cos_half = 1 + np.log((peak / 2 - baseline) / (peak - baseline)) / (
                self.concentration
            )
            if cos_half >= -1:
                tuning_width = float(np.arccos(cos_half))
        return {
            "cell_type": "head_direction",
            "preferred_direction": self._preferred_direction,
            "preferred_direction_deg": float(np.degrees(self._preferred_direction)),
            "concentration": self.concentration,
            "tuning_width": tuning_width,
            "tuning_width_deg": float(np.degrees(tuning_width)),
            "max_rate": self.max_rate,
            "baseline_rate": self.baseline_rate,
        }

    @property
    def mean_vector_length(self) -> float:
        """Mean vector length of this cell's tuning curve.

        Returns
        -------
        float
            The rate-weighted mean resultant length of the tuning curve over
            a uniform sampling of headings, between 0 and 1.

        Notes
        -----
        The tuning curve is ``r(θ) = b + (m - b) exp(κ (cos(θ - μ) - 1))``.
        Its mean resultant length is ``|∫ r(θ) e^{iθ} dθ| / ∫ r(θ) dθ``; with
        ``∫ e^{κ cos θ} cos θ dθ = 2π I₁(κ)`` and ``∫ e^{κ cos θ} dθ = 2π I₀(κ)``,

            MVL = (m - b) e^{-κ} I₁(κ) / (b + (m - b) e^{-κ} I₀(κ))

        where I₀ and I₁ are modified Bessel functions of the first kind. The
        baseline adds firing in every direction, so it lowers the length;
        with ``b = 0`` it is the von Mises value ``I₁(κ) / I₀(κ)``.
        """
        from scipy.special import i0e, i1e

        tuned = self.max_rate - self.baseline_rate
        kappa = self.concentration
        return float(tuned * i1e(kappa) / (self.baseline_rate + tuned * i0e(kappa)))
