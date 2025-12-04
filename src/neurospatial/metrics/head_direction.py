"""
Head direction cell analysis module.

This module provides tools for analyzing head direction (HD) cells, neurons
that fire preferentially when an animal faces a particular direction.
HD cells are found in various brain regions including the postsubiculum,
anterodorsal thalamus, and lateral mammillary nucleus.

Which Function Should I Use?
----------------------------
**Computing tuning curve from raw data?**
    Use ``head_direction_tuning_curve()`` to compute firing rate as a function
    of head direction from spike times and head direction time series.

**Analyzing tuning curve properties?**
    Use ``head_direction_metrics()`` to compute preferred direction, mean
    vector length, tuning width, and HD cell classification.

**Screening many neurons (100s-1000s)?**
    Use ``is_head_direction_cell()`` for fast boolean filtering without
    manually computing tuning curves and metrics.

**Visualizing HD tuning?**
    Use ``plot_head_direction_tuning()`` for standard polar or linear plots.

Typical Workflow
----------------
1. Compute tuning curve from spike times and head directions::

    >>> bin_centers, firing_rates = head_direction_tuning_curve(
    ...     head_directions, spike_times, position_times,
    ...     bin_size=6.0, angle_unit='deg'
    ... )  # doctest: +SKIP

2. Compute metrics and classify::

    >>> metrics = head_direction_metrics(bin_centers, firing_rates)
    >>> print(metrics)  # Human-readable interpretation  # doctest: +SKIP
    >>> if metrics.is_hd_cell:
    ...     print(f"Preferred direction: {metrics.preferred_direction_deg:.1f}°")
    ...     # doctest: +SKIP

3. Visualize::

    >>> plot_head_direction_tuning(bin_centers, firing_rates, metrics)
    ...     # doctest: +SKIP

Common Parameters
-----------------
Most functions accept ``angle_unit`` parameter: ``'rad'`` (default) or ``'deg'``.
HD research commonly uses degrees, but we default to radians for scipy
compatibility. Use ``angle_unit='deg'`` if your data is in degrees.

References
----------
Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells recorded
    from the postsubiculum in freely moving rats. I. Description and
    quantitative analysis. Journal of Neuroscience, 10(2), 420-435.
Sargolini, F. et al. (2006). Conjunctive representation of position, direction,
    and velocity in entorhinal cortex. Science, 312(5774), 758-762.
"""

from __future__ import annotations

# Mark that circular imports are available for testing
_has_circular_imports = True

__all__: list[str] = []
