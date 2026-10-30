# User Guide

Start with the question you want to answer from spikes, tracking, and behavior.
Each route explains the analysis choices and how to interpret the figures.

## Choose a Scientific Task

| I want to… | Start here | Check before interpreting |
| --- | --- | --- |
| Analyze my recorded spikes and tracking | [Load NWB data](interoperability.md#nwb-interop) | Position units, clocks, unit labels and recording windows |
| Find where a neuron fires | [First place field](../getting-started/quickstart.md#your-first-place-field) | Occupancy, smoothing scale and missing spatial coverage |
| Compare conditions or trials | [Place fields on outbound and inbound runs](workflows.md#place-fields-on-outbound-and-inbound-runs) | One environment, identical analysis settings, matched unit IDs and observed time in each condition |
| Classify candidate spatial cells | [Cell classification](neuroscience-metrics.md) | Threshold screens versus shuffle evidence |
| Summarize a population: coverage, remapping, assemblies and reactivation | [Population analysis](population-analysis.md) | Which units passed a spatial screen, bins visited in every condition and one unit selection across periods |
| Test candidate events for replay and compare published methods | [Comparing replay methods](replay-methods.md) | Each method's false-positive rate on several kinds of null data, by event cells and duration |
| Test whether task co-firing returns in rest, as published methods do | [Comparing reactivation methods](reactivation-methods.md) | Rest before the task as the control, pairs on different tetrodes, and rate changes that mimic reactivation |
| Evaluate decoding on new data | [Held-out decoding](workflows.md#decode-held-out-laps) | Separate training/test periods and locations visited during training |
| Segment laps and quantify movement | [Trajectory and behavioral analysis](trajectory-and-behavioral-analysis.md) | Tracking gaps and invalid positions |
| Measure speed, heading or position-based features around events | [Behavior around events](behavior-around-events.md) | Tracking dropouts inside event windows and angle features declared circular |
| Measure firing around cues | [Raster and PSTH example](../examples/07_peri_event_analysis.ipynb) | Which cues have fully recorded before/after windows |
| Save results I can interpret later | [Analysis settings and exports](interoperability.md#analysis-settings-and-exports) | Units, neuron labels, analyzed periods, bin widths and smoothing |

## Contents

- **[Environments](environments.md)**: Creating and working with Environment objects
- **[Layout Engines](layout-engines.md)**: Understanding different discretization strategies
- **[Regions](regions.md)**: Defining and managing regions of interest
- **[Video Annotation](video-annotation.md)**: Define environments from video frames interactively
- **[Composite Environments](composite-environments.md)**: Merging multiple environments
- **[Alignment & Transforms](alignment.md)**: Transforming spatial representations
- **[Spatial Analysis](spatial-analysis.md)**: Trajectory analysis, movement patterns, and field operations
- **[Population Analysis](population-analysis.md)**: Coverage, remapping, cell assemblies, reactivation and decoding across many units
- **[Comparing Replay Methods](replay-methods.md)**: Published replay tests on the same events, their false-positive rates on null data, and comparison at a matched rate
- **[Comparing Reactivation Methods](reactivation-methods.md)**: Published explained-variance and assembly reactivation analyses on one session, and what they report on data without reactivation
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
- [Open-field, linear-track and maze presets](environments.md#presets)
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
