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

### 8. Complete Workflow

End-to-end analysis example:

- Loading experimental data
- Environment setup
- Computing spatial statistics
- Visualization and export

**[Open notebook: 08_complete_workflow.ipynb](08_complete_workflow.ipynb)** | **Recommended for**: Seeing it all together

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
