# Installation

This guide covers how to install neurospatial for different use cases.

## Standard Installation

Install neurospatial from PyPI using pip:

```bash
pip install neurospatial
```

This installs neurospatial with all required dependencies for core functionality.

## Development Installation

If you want to contribute to neurospatial or run the latest development version:

```bash
# Clone the repository
git clone https://github.com/edeno/neurospatial.git
cd neurospatial

# Install with uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

The `dev` extra includes testing, linting, and development tools:

- pytest (testing framework)
- pytest-cov (test coverage)
- ruff (linting and formatting)
- pre-commit (git hooks)
- mypy (type checking)

## Documentation Building

To build the documentation locally:

```bash
# Install with documentation dependencies
uv sync --extra docs

# Sync example notebooks
uv run python docs/sync_notebooks.py

# Serve documentation locally
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Optional Dependencies

### OpenCV Support

For advanced image-based masking features:

```bash
pip install neurospatial[opencv]
```

## Verifying Installation

Verify your installation by importing neurospatial:

```python
import neurospatial
print(neurospatial.__version__)
```

Or run a quick test:

```python
from neurospatial import Environment
import numpy as np

# Create a simple environment
data = np.random.uniform(0, 100, (50, 2))
env = Environment.from_samples(data, bin_size=10.0)
print(f"Created environment with {env.n_bins} bins")
```

## Tested Dependency Versions

neurospatial v0.1.0 has been tested with:

| Package | Tested Version |
|---------|---------------|
| Python | 3.13.5 |
| numpy | 2.3.4 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| networkx | 3.5 |
| scipy | 1.16.3 |
| scikit-learn | 1.7.2 |
| shapely | 2.1.2 |
| track-linearization | 2.4.0 |

These versions represent the tested configuration, but neurospatial likely works with a range of versions for each dependency.

## System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.10 or higher
- **Memory**: Recommended 4GB+ for typical use cases
- **Disk**: ~100MB for installation

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError`:

```bash
# Ensure you're in the correct environment
which python  # Should point to your virtual environment

# Reinstall neurospatial
pip install --force-reinstall neurospatial
```

### Using uv

This project uses [uv](https://github.com/astral-sh/uv) for package management. If you're developing:

- Always use `uv run` to execute commands (e.g., `uv run pytest`)
- Never use bare `python`, `pip`, or `pytest` commands

### Dependency Conflicts

If you have dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv neurospatial-env
source neurospatial-env/bin/activate  # On Windows: neurospatial-env\Scripts\activate

# Install neurospatial
pip install neurospatial
```

## Next Steps

Once installed, proceed to the [Quickstart](quickstart.md) guide to create your first environment.
