# neurospatial Examples

This directory contains Jupyter notebook tutorials demonstrating the features and capabilities of the neurospatial package. The notebooks follow a progressive learning path, building from basic concepts to advanced workflows.

## Getting Started

### Prerequisites

Ensure you have neurospatial installed:

```bash
# From the project root
uv sync

# Or with pip
pip install -e .
```

### Running the Notebooks

From the project root directory:

```bash
# Launch Jupyter notebook
uv run jupyter notebook examples/

# Or with JupyterLab
uv run jupyter lab examples/
```

## Notebook Series

The tutorials are designed to be completed in order, as each builds on concepts from previous notebooks.

### 01. Introduction to neurospatial: The Basics

**File**: [01_introduction_basics.ipynb](01_introduction_basics.ipynb)
**Time**: 15-20 minutes
**Topics**:

- What is spatial discretization and why it matters
- Creating your first environment from position data
- Understanding bins, bin_size, and active bins
- Basic spatial queries: `bin_at()`, `contains()`, `neighbors()`
- Visualization with `plot()`
- Common pitfalls when choosing bin_size

**Start here** if you're new to neurospatial!

---

### 02. Layout Engines: Choosing the Right Spatial Discretization

**File**: [02_layout_engines.ipynb](02_layout_engines.ipynb)
**Time**: 20-25 minutes
**Topics**:

- Overview of different layout engines
- Regular grids vs hexagonal tessellations
- Polygon-bounded environments (circular arenas, custom shapes)
- Connectivity patterns (orthogonal vs diagonal)
- Comparing layouts side-by-side
- When to use each layout type

**Prerequisites**: Notebook 01

---

### 03. Morphological Operations: Handling Sparse Data

**File**: [03_morphological_operations.ipynb](03_morphological_operations.ipynb)
**Time**: 15-20 minutes
**Topics**:

- Understanding active bin inference
- Morphological operations: `dilate`, `fill_holes`, `close_gaps`
- Controlling `bin_count_threshold`
- Handling sparse vs dense data
- Fixing fragmented environments
- Practical examples with realistic tracking data

**Prerequisites**: Notebook 01

---

### 04. Regions of Interest: Defining and Analyzing Spatial Zones

**File**: [04_regions_of_interest.ipynb](04_regions_of_interest.ipynb)
**Time**: 15-20 minutes
**Topics**:

- Defining point and polygon regions
- Adding, querying, and updating regions
- Region operations: buffering, area calculation, center finding
- Getting bins and masks for regions
- Use case: Morris water maze with platform and quadrants
- JSON serialization for reproducibility

**Prerequisites**: Notebook 01

---

### 05. Track Linearization: From 2D Tracks to 1D Coordinates

**File**: [05_track_linearization.ipynb](05_track_linearization.ipynb)
**Time**: 25-30 minutes
**Topics**:

- Introduction to 1D linearization
- Creating graph-based environments
- Simple linear track example
- Complex plus maze with graph structure
- Converting between 2D and 1D coordinates
- Understanding `edge_order` and `edge_spacing`
- When to use 1D vs N-D environments
- Application: trajectory-dependent place cells

**Prerequisites**: Notebooks 01-02

---

### 06. Composite Environments: Merging Multiple Spaces

**File**: [06_composite_environments.ipynb](06_composite_environments.ipynb)
**Time**: 20-25 minutes
**Topics**:

- Merging multiple environments into unified spaces
- Automatic bridge inference with mutual nearest neighbors
- Multi-room experiments
- Controlling bridge connectivity with `max_mnn_distance`
- Multi-compartment maze analysis
- Working with disconnected spatial components

**Prerequisites**: Notebooks 01-02

---

### 07. Advanced Operations: Paths, Distances, and Transforms

**File**: [07_advanced_operations.ipynb](07_advanced_operations.ipynb)
**Time**: 20-25 minutes
**Topics**:

- Shortest paths through complex environments
- Geodesic vs Euclidean distances
- 2D coordinate transformations (rotation, scaling, translation)
- Composing transformations
- Mapping probability distributions between environments
- Graph analysis: centrality measures and bottleneck detection
- Boundary detection

**Prerequisites**: Notebooks 01-02

---

### 08. Complete Workflow: From Tracking Data to Place Fields

**File**: [08_complete_workflow.ipynb](08_complete_workflow.ipynb)
**Time**: 30-40 minutes
**Topics**:

- End-to-end neuroscience analysis pipeline
- From raw tracking data to spatial discretization
- Computing occupancy-normalized firing rates
- Place field (spatial firing rate map) calculation
- Spatial information metrics
- Place cell detection and classification
- Multi-region analysis
- Population coding analysis
- Publication-quality visualization
- Best practices and performance tips

**Prerequisites**: Notebooks 01, 03, 04 (recommended: all previous notebooks)

This notebook ties together concepts from all previous tutorials into a realistic neuroscience workflow.

---

## Learning Path Recommendations

### Quick Start (1 hour)

If you're just getting started and want to see what neurospatial can do:

1. **01_introduction_basics** (20 min)
2. **04_regions_of_interest** (20 min)
3. **08_complete_workflow** (40 min) - skim for overview

### Comprehensive Learning (3-4 hours)

For a thorough understanding of all features:

- Complete notebooks 01-08 in order
- Try the optional exercises in each notebook

### Specific Use Cases

**Analyzing rodent navigation in open fields:**

- Notebooks 01, 03, 04, 08

**Track/maze experiments (linear tracks, plus mazes, T-mazes):**

- Notebooks 01, 02, 05, 08

**Multi-room or complex environments:**

- Notebooks 01, 02, 06, 07

**Custom arena shapes (circular, polygonal):**

- Notebooks 01, 02, 04

## Getting Help

If you encounter issues with the notebooks:

**Open an issue** on [GitHub](https://github.com/edeno/neurospatial/issues)

## Contributing

Found a typo or have a suggestion for improving these tutorials? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use neurospatial in your research, please cite:

```bibtex
@software{neurospatial2025,
  author = {Denovellis, Eric},
  title = {neurospatial: Spatial environment discretization for neuroscience},
  year = {2025},
  url = {https://github.com/edeno/neurospatial},
  version = {0.1.0}
}
```

## Additional Resources

- **API Documentation**: Coming soon
- **Main README**: [../README.md](../README.md)
- **Developer Guide**: [../CLAUDE.md](../CLAUDE.md)
- **GitHub Repository**: <https://github.com/edeno/neurospatial>
- **Issue Tracker**: <https://github.com/edeno/neurospatial/issues>

---

**Happy learning!** We hope these tutorials help you get the most out of neurospatial for your spatial analysis needs.
