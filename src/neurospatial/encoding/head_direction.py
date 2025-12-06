"""
Head direction cell analysis.

This module provides tools for analyzing head direction (HD) cells, neurons
that fire preferentially when an animal faces a particular direction.
HD cells are found in various brain regions including the postsubiculum,
anterodorsal thalamus, and lateral mammillary nucleus.

Imports
-------
All functions can be imported from this module::

    from neurospatial.encoding.head_direction import (
        head_direction_tuning_curve,
        head_direction_metrics,
        is_head_direction_cell,
        plot_head_direction_tuning,
        HeadDirectionMetrics,
        # Re-exports from stats.circular for convenience
        rayleigh_test,
        mean_resultant_length,
        circular_mean,
    )

Or from the encoding package::

    from neurospatial.encoding import (
        head_direction_tuning_curve,
        head_direction_metrics,
    )

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

**Testing for directional selectivity?**
    Use ``rayleigh_test()`` to test whether the distribution of spike directions
    is significantly non-uniform.

Typical Workflow
----------------
1. Compute tuning curve from spike times and head directions::

    >>> bin_centers, firing_rates = head_direction_tuning_curve(  # doctest: +SKIP
    ...     head_directions, spike_times, position_times,
    ...     bin_size=6.0, angle_unit='deg'
    ... )

2. Compute metrics and classify::

    >>> metrics = head_direction_metrics(bin_centers, firing_rates)  # doctest: +SKIP
    >>> print(metrics)  # Human-readable interpretation  # doctest: +SKIP
    >>> if metrics.is_hd_cell:  # doctest: +SKIP
    ...     print(f"Preferred direction: {metrics.preferred_direction_deg:.1f}")

3. Visualize::

    >>> plot_head_direction_tuning(bin_centers, firing_rates, metrics)  # doctest: +SKIP

References
----------
Taube, J.S., Muller, R.U., & Ranck, J.B. (1990). Head-direction cells recorded
    from the postsubiculum in freely moving rats. I. Description and
    quantitative analysis. Journal of Neuroscience, 10(2), 420-435.
Sargolini, F. et al. (2006). Conjunctive representation of position, direction,
    and velocity in entorhinal cortex. Science, 312(5774), 758-762.
"""

# Re-export all from metrics/head_direction.py
from neurospatial.metrics.head_direction import (
    HeadDirectionMetrics,
    head_direction_metrics,
    head_direction_tuning_curve,
    is_head_direction_cell,
    plot_head_direction_tuning,
)

# Re-export from stats/circular.py for HD workflow convenience
from neurospatial.stats.circular import (
    circular_mean,
    mean_resultant_length,
    rayleigh_test,
)

__all__ = [  # noqa: RUF022 - organized by category
    # From metrics/head_direction.py
    "HeadDirectionMetrics",
    "head_direction_metrics",
    "head_direction_tuning_curve",
    "is_head_direction_cell",
    "plot_head_direction_tuning",
    # Re-exports from stats/circular.py
    "rayleigh_test",
    "mean_resultant_length",
    "circular_mean",
]
