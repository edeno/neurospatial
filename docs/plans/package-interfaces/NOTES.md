# Model inputs and package interfaces

**Created:** 2026-10-29
**Status:** Direction notes; no integrations implemented by this document.

These notes record priorities discussed alongside the [project vision](../../vision.md).

## Reusable model inputs

Providing regressors and features is an explicit project goal. Support GLMs,
state-space models, and reinforcement learning through spatial and behavioral
inputs with documented timing, observation support, units, identities, feature
definitions, and history requirements. Reuse the features across recorded and
simulated observations. Keep predictive inputs free of future information and
declare which information an RL agent may observe.

[NeMoS](https://nemos-neuro.org/en/stable/) is a possible consumer. A worked
example could pass spatial and behavioral features to its GLMs, while other
examples could feed a state-space model or RL policy. Feature generation and
the core data interfaces must remain usable without NeMoS or a particular
modeling or RL framework. Package-specific adapters should be optional.

## POT: benchmark an existing calculation

The connection to [POT](https://pythonot.github.io/user_guide.html) is useful and
worth evaluating. The current field Earth Mover's Distance calculation uses
SciPy's general linear-programming solver. Benchmark POT's exact transport
solver against it using independently checked small examples and representative
environments; measure numerical agreement, runtime, and memory.

Preserve bin-measure weighting, common observed support, the chosen distance
metric, and disconnected-component constraints. Regularized transport solves a
different problem and would need an explicit option with documented numerical
behavior. Adoption remains contingent on the comparison; this note does not
commit to replacing the solver or adding a required dependency.

## PyVista: explore mesh and field exchange

[PyVista](https://docs.pyvista.org/user-guide/data_model.html) is an interesting
candidate for inspecting irregular 3D geometry and fields. Start by evaluating
an optional exporter of supported cell geometry and field values, preserving
the mapping to neurospatial bin IDs, coordinate frames, units, and missing data.
Clearly distinguish cell values from values at vertices or bin centers.

A small export-and-inspect example would establish its usefulness before
committing to another animation backend. This is exploratory interest, with
the supported geometries and validation cases still to be scoped.
