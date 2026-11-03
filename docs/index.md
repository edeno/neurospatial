# neurospatial

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://edeno.github.io/neurospatial/)

**neurospatial** is a Python library for spatial neuroscience. It helps scientists
analyze how animals move through space and how neural activity represents it,
with tools for environments, behavior, neural encoding and decoding, simulation,
and visualization.

Our highest design priority is the scientist's experience: common analyses should
be easy to discover, behave as a scientist reasonably expects, and produce
scientifically correct, interpretable results. Clear assumptions, consistent
interfaces, and useful error messages are part of that experience.

The long-term goal is a shared foundation for spatial neuroscience: define an
environment once, save and share it, and use it for behavior, neural analysis,
simulation and visualization, with methods that scientists can trust and
compare. See the [project vision](vision.md) for the goals and the
[design principles](contributing.md#design-principles) for how we evaluate
changes toward them.

## Where does this neuron fire?

You have an animal's position over time and the spike times of a neuron. The
first question is where in the arena the neuron fires. That needs the arena
split into spatial bins, the time the animal spent in each bin (occupancy),
and the spikes in each bin divided by that occupancy: a firing-rate map. The
example below simulates a place cell in a 100 x 100 cm open field and
estimates its rate map. To use your own recording, replace the simulation with
your `spike_times` (seconds), `times` (seconds) and `positions` (same units as
`bin_size`).

```python
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

from neurospatial import Environment, compute_spatial_rate
from neurospatial.simulation import (
    PlaceCellModel,
    generate_poisson_spikes,
    simulate_trajectory_ou,
)

# The arena, binned at 2 cm.
env = Environment.from_polygon(box(0, 0, 100, 100), bin_size=2.0, units="cm")

# 30 minutes of foraging and one place cell tuned to the arena center.
positions, times = simulate_trajectory_ou(env, duration=1800.0, speed_units="cm", rng=1)
cell = PlaceCellModel(env, center=np.array([50.0, 50.0]), width=15.0, max_rate=30.0)
spike_times = generate_poisson_spikes(cell.firing_rate(times, positions), times, rng=2)

# Spikes divided by occupancy, smoothed over the bins the animal visited.
rate_map = compute_spatial_rate(env, spike_times, times, positions, bandwidth=8.0)
rate_map.plot()
plt.show()
```

The map has a single smooth peak near the center of the arena, where the cell
was tuned. The darkest color is the lowest rate: bins the animal visited where
the cell was silent. Bins the animal never visited are NaN (missing), not zero,
and are drawn light grey, so grey means "not sampled" rather than "silent".
From here, ask whether the
neuron is a place cell with `is_place_cell`, find its fields with
`detect_place_fields`, or decode position from a population with
`decode_session`. The user guide's
[Which place-cell function?](user-guide/neuroscience-metrics.md#which-place-cell-function)
table lists the neighboring functions.

neurospatial covers the rest of the path from recording to result:

- **Environments**: bins, connectivity, regions and landmarks for open fields,
  tracks and mazes.
- **Behavior**: speed, laps, trials, goal-directed navigation and decisions.
- **Neural analysis**: spatial, head-direction, object-vector and view
  tuning; decoding; replay; reactivation; peri-event histograms.
- **Simulation and visualization**: simulated sessions, animations and NWB
  input and output.

The [quickstart](getting-started/quickstart.md) walks through this example,
and the [user guide](user-guide/index.md) is organized by scientific task.

## Under the hood: environments and layout engines

Every analysis above sits on an `Environment`: a discretization of space into
bins with a connectivity graph. You can build one from tracking samples, an
arena outline, a mask or a track, with these layout engines:

- **Regular grids and hexagonal tessellations** for open fields
- **Masked regions and polygon-bounded areas** for arenas of any shape
- **Triangular meshes** for irregular boundaries
- **1D linearized tracks** for mazes, with a map between 2D position and
  position along the track

An environment also holds named regions of interest, can be merged with
others, and provides spatial queries (point-to-bin mapping, neighbors, shortest
paths and geodesic distances). Layout engines implement protocols rather than
inheriting from a base class. This smaller example builds an environment
from samples and queries it:

```python
import numpy as np
from neurospatial import Environment

rng = np.random.default_rng(0)
position_data = rng.uniform(0, 100, size=(500, 2))  # (n_samples, 2), in cm

env = Environment.from_samples(position_data, bin_size=10.0, units="cm")
print(f"Environment has {env.n_bins} bins in {env.n_dims}D")

point = np.array([[52.0, 48.0]])
bin_idx = env.bin_at(point)
print(f"Point {point[0]} is in bin {bin_idx[0]}")
print(f"That bin has {len(env.neighbors(bins=bin_idx[0]))} neighbors")
```

## Installation

Install neurospatial using pip:

```bash
pip install neurospatial
```

For development installation with documentation tools:

```bash
git clone https://github.com/edeno/neurospatial.git
cd neurospatial
uv sync --group docs
```

See the [Installation Guide](getting-started/installation.md) for more details.

## Documentation

- **[Getting Started](getting-started/index.md)**: Installation, quickstart, and core concepts
- **[User Guide](user-guide/index.md)**: Detailed guides on using neurospatial features
- **[API Reference](api/index.md)**: Complete API documentation
- **[Examples](examples/index.md)**: Jupyter notebooks with real-world use cases
- **[Contributing](contributing.md)**: Guidelines for contributors
- **[llms.txt](llms.txt)**: A plain-text map of the public API and conventions for AI assistants

## Project Status

**Status**: Alpha (v0.8.0) - API may change. Contributions and feedback welcome!

## Citation

If you use neurospatial in your research, please cite:

```bibtex
@software{neurospatial2026,
  author = {Denovellis, Eric},
  title = {neurospatial: Spatial environment discretization for neuroscience},
  year = {2026},
  url = {https://github.com/edeno/neurospatial},
  version = {0.8.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/edeno/neurospatial/blob/main/LICENSE) file for details.

## Contact

**Eric Denovellis**

- Email: <eric.denovellis@ucsf.edu>
- GitHub: [@edeno](https://github.com/edeno)
- Issues: [GitHub Issues](https://github.com/edeno/neurospatial/issues)
