# Examples

Real-world examples demonstrating neurospatial's capabilities through interactive Jupyter notebooks.

## Available Notebooks

### 1. Introduction & Basics

Get started with neurospatial basics:

- Creating environments from data
- Basic spatial queries
- Visualizing environments
- Understanding bin centers and connectivity

**[Open notebook: 01_introduction_basics.ipynb](01_introduction_basics.ipynb)** | **Recommended for**: First-time users

### 2. Layout Engines

Explore different discretization strategies:

- Regular grids
- Hexagonal tessellations
- Triangular meshes
- Comparing layout engines

**[Open notebook: 02_layout_engines.ipynb](02_layout_engines.ipynb)** | **Recommended for**: Understanding spatial discretization options

### 3. Morphological Operations

Master automatic active bin detection:

- Dilation and closing operations
- Filling holes
- Thresholding strategies
- Handling sparse data

**[Open notebook: 03_morphological_operations.ipynb](03_morphological_operations.ipynb)** | **Recommended for**: Working with real experimental data

### 4. Regions of Interest

Define and manage spatial regions:

- Creating point and polygon regions
- Region operations (buffering, area calculation)
- Using regions in analysis
- Region serialization

**[Open notebook: 04_regions_of_interest.ipynb](04_regions_of_interest.ipynb)** | **Recommended for**: Defining experimental zones and ROIs

### 5. Track Linearization

Work with maze and track experiments:

- Creating 1D linearized environments
- Converting between 2D and 1D coordinates
- T-maze and plus maze examples
- Sequential analysis

**[Open notebook: 05_track_linearization.ipynb](05_track_linearization.ipynb)** | **Recommended for**: Track-based experiments

### 6. Composite Environments

Merge multiple environments:

- Creating composite environments
- Automatic bridge inference
- Multi-arena experiments
- Cross-environment queries

**[Open notebook: 06_composite_environments.ipynb](06_composite_environments.ipynb)** | **Recommended for**: Multi-environment studies

### 7. Advanced Operations

Advanced features and techniques:

- Custom spatial queries
- Graph operations
- Performance optimization
- Edge cases and troubleshooting

**[Open notebook: 07_advanced_operations.ipynb](07_advanced_operations.ipynb)** | **Recommended for**: Power users

### 8. Spike & Field Basics

Introduction to place field analysis:

- Generating synthetic trajectories
- Simulating place cell activity
- Computing place fields from spikes
- Validating detection accuracy

**[Open notebook: 08_spike_field_basics.ipynb](08_spike_field_basics.ipynb)** | **Recommended for**: Neural data analysis

### 9. Differential Operators

Spatial derivatives and gradients:

- Computing spatial gradients
- Directional derivatives
- Laplacian operators
- Applications to field analysis

**[Open notebook: 09_differential_operators.ipynb](09_differential_operators.ipynb)** | **Recommended for**: Advanced spatial analysis

### 10. Signal Processing Primitives

Spatial signal processing tools:

- Smoothing and filtering
- Convolution operations
- Kernel methods
- Boundary handling

**[Open notebook: 10_signal_processing_primitives.ipynb](10_signal_processing_primitives.ipynb)** | **Recommended for**: Signal processing workflows

### 11. Place Field Analysis

Complete place field analysis pipeline:

- Trajectory generation
- Place cell models
- Field detection and characterization
- T-maze spatial alternation

**[Open notebook: 11_place_field_analysis.ipynb](11_place_field_analysis.ipynb)** | **Recommended for**: Hippocampal place cell analysis

### 12. Boundary Cell Analysis

Analyzing boundary-tuned neurons:

- Boundary detection
- Distance-to-boundary metrics
- Border cells and boundary vector cells
- Validation metrics

**[Open notebook: 12_boundary_cell_analysis.ipynb](12_boundary_cell_analysis.ipynb)** | **Recommended for**: Border cell analysis

### 13. Trajectory Analysis

Analyzing movement patterns:

- Trajectory metrics
- Speed and acceleration
- Goal-directed behavior
- Path analysis

**[Open notebook: 13_trajectory_analysis.ipynb](13_trajectory_analysis.ipynb)** | **Recommended for**: Behavioral analysis

### 14. Behavioral Segmentation

Segmenting behavior into states:

- State detection algorithms
- Exploratory vs goal-directed behavior
- Transition analysis
- Behavioral bout detection

**[Open notebook: 14_behavioral_segmentation.ipynb](14_behavioral_segmentation.ipynb)** | **Recommended for**: Behavioral state analysis

### 15. Simulation Workflows

Comprehensive simulation tutorial:

- Quick start with pre-configured sessions
- Low-level API (trajectory + models + spikes)
- All cell types (place, boundary, grid)
- Validation and visualization
- Customization examples

**[Open notebook: 15_simulation_workflows.ipynb](15_simulation_workflows.ipynb)** | **Recommended for**: Generating synthetic data for testing

## Viewing on GitHub

All example notebooks are available on GitHub with rendered outputs:

[View examples on GitHub](https://github.com/edeno/neurospatial/tree/main/examples)

## Running Examples

To run the examples locally:

```bash
# Clone the repository
git clone https://github.com/edeno/neurospatial.git
cd neurospatial

# Install with dependencies
uv sync

# Start Jupyter
uv run jupyter notebook examples/
```

## Contributing Examples

Have a useful example? We welcome contributions! See the [Contributing Guide](../contributing.md) for details.

!!! note "For Documentation Contributors"
    The notebooks displayed here are automatically synced from the `examples/` directory.

    **To update notebooks in the documentation:**

    1. Edit notebooks in the `examples/` directory (repository root)
    2. Run `uv run python docs/sync_notebooks.py` before building docs
    3. The GitHub Actions workflow automatically syncs notebooks on deployment

    **Do not** edit `.ipynb` files directly in `docs/examples/` - they will be overwritten.
