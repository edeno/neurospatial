# User Guide

Start with the scientific question you want to answer. Each route below uses
public APIs and explains the assumptions needed to interpret the output.

## Choose a Scientific Task

| I want to… | Start here | Check before interpreting |
| --- | --- | --- |
| Analyze my recorded spikes and tracking | [Load NWB data](interoperability.md#nwb-interop) | Position units, clocks, unit labels and recording windows |
| Find where a neuron fires | [First place field](../getting-started/quickstart.md#your-first-place-field) | Occupancy, smoothing scale and missing spatial coverage |
| Compare conditions or trials | [Compare analysis windows](workflows.md#compare-conditions-with-explicit-windows) | Shared geometry, identical analysis settings and observed duration |
| Classify candidate spatial cells | [Cell classification](neuroscience-metrics.md) | Threshold screens versus shuffle evidence |
| Evaluate decoding on new data | [Held-out decoding](workflows.md#evaluate-decoding-on-held-out-data) | Disjoint training/test windows and training spatial support |
| Segment laps and quantify movement | [Trajectory and behavioral analysis](trajectory-and-behavioral-analysis.md) | Tracking gaps and invalid positions |
| Measure responses around events | [Peri-event example](../examples/07_peri_event_analysis.ipynb) | Full event-window coverage and the retained event cohort |
| Save results I can interpret later | [Analysis settings and exports](interoperability.md#analysis-settings-and-exports) | Units, identity, analysis windows and estimator settings |

## Contents

- **[Environments](environments.md)**: Creating and working with Environment objects
- **[Layout Engines](layout-engines.md)**: Understanding different discretization strategies
- **[Regions](regions.md)**: Defining and managing regions of interest
- **[Video Annotation](video-annotation.md)**: Define environments from video frames interactively
- **[Composite Environments](composite-environments.md)**: Merging multiple environments
- **[Alignment & Transforms](alignment.md)**: Transforming spatial representations
- **[Spatial Analysis](spatial-analysis.md)**: Trajectory analysis, movement patterns, and field operations
- **[Performance & Caching](performance.md)**: Memory management, optimization, and benchmarks
- **[Complete Workflows](workflows.md)**: End-to-end analysis examples
- **[Interoperability](interoperability.md)**: Recording holders, SpikeTrains, BayesianDecoder, and pynapple / NWB adapters

## Who This Guide Is For

This guide is for users who:

- Have completed the [Quickstart](../getting-started/quickstart.md)
- Want to understand specific features in depth
- Need to solve complex spatial analysis problems
- Want to customize neurospatial for their use case

## Prerequisites

- Completed [Getting Started](../getting-started/index.md) section
- Familiarity with NumPy arrays and basic Python
- Understanding of spatial coordinates

## Navigation

Each page covers a specific feature area with:

- Conceptual explanations
- Code examples
- Common patterns and best practices
- Troubleshooting tips

## Quick Links

**Common Tasks:**

- [Creating different environment types](environments.md#factory-methods)
- [Choosing the right layout engine](layout-engines.md#quick-decision-guide)
- [Defining regions](regions.md)
- [Annotating video frames](video-annotation.md)
- [Computing occupancy and firing rates](spatial-analysis.md#occupancy)
- [Analyzing movement patterns](spatial-analysis.md#movement-patterns)
- [Computing distance fields](spatial-analysis.md#distance-to)
- [Combining multiple environments](composite-environments.md)
- [Aligning spatial maps](alignment.md)
- [Optimizing performance and managing memory](performance.md)

## See Also

- **[API Reference](../api/index.md)**: Detailed API documentation
- **[Examples](../examples/index.md)**: Real-world use cases
- **[Getting Started](../getting-started/index.md)**: Beginner tutorials
