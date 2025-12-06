"""
Phase precession analysis for place cells.

This module re-exports phase precession analysis functions from
``neurospatial.metrics.phase_precession`` for convenient access.

Imports
-------
The recommended import pattern is::

    from neurospatial.encoding.phase_precession import (
        phase_precession,
        has_phase_precession,
        plot_phase_precession,
        PhasePrecessionResult,
    )

Or import via the encoding package::

    from neurospatial.encoding import phase_precession, has_phase_precession

Which Function Should I Use?
----------------------------
**Screening many neurons (100s-1000s)?**
    Use ``has_phase_precession()`` for fast boolean filtering.

**Need phase precession slope for publication?**
    Use ``phase_precession()`` for full analysis with slope, offset, and fit quality.
    This is what you want for figures and reporting.

**Visualizing phase precession?**
    Use ``plot_phase_precession()`` for standard doubled-axis visualization.

Common Use Cases
----------------
**Screening many neurons (fast filtering)**::

    from neurospatial.encoding.phase_precession import has_phase_precession

    # Fast boolean check - use for initial filtering
    if has_phase_precession(positions, phases):
        print("Neuron shows phase precession")

**Publication-quality analysis (full metrics)**::

    from neurospatial.encoding.phase_precession import (
        phase_precession,
        plot_phase_precession,
    )

    result = phase_precession(positions, phases)
    print(result)  # Automatic interpretation with slope, correlation, fit
    plot_phase_precession(positions, phases, result)

References
----------
O'Keefe, J. & Recce, M.L. (1993). Phase relationship between hippocampal
    place units and the EEG theta rhythm. Hippocampus, 3(3), 317-330.
Kempter, R. et al. (2012). Quantifying circular-linear associations.
    J Neurosci Methods.
"""

from neurospatial.metrics.phase_precession import (
    PhasePrecessionResult,
    has_phase_precession,
    phase_precession,
    plot_phase_precession,
)

__all__ = [
    "PhasePrecessionResult",
    "has_phase_precession",
    "phase_precession",
    "plot_phase_precession",
]
