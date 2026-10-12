# Contributing

Thank you for your interest in contributing to neurospatial! This guide will help you get started.

## Design Principles

The scientist's experience is neurospatial's highest design priority. Scientific
correctness is part of usability: a convenient call that silently changes the
meaning of an analysis is a usability failure. Scientists should be able to
discover an operation, understand its assumptions, and interpret its result
without reading implementation code.

The long-term scope is the operations of spatial neuroscience: loading and
validating recordings, representing environments, analyzing behavior and neural
activity, evaluating statistical evidence, simulating experiments, visualizing
results, and exporting them. Grow this coverage through workflows that work
together and preserve scientific meaning across steps.

Use these principles when designing and reviewing changes:

- **Start with the scientific task.** Describe what the scientist has and wants
  to learn. Use familiar experimental vocabulary and provide a clear recommended
  path for common tasks, with advanced controls available when needed.
- **Make expected behavior scientifically sound.** Choose defensible defaults
  and document their assumptions. Require an explicit choice when alternatives
  change the scientific question and there is no defensible default. Preserve
  the distinction between missing observations and measured zero activity;
  expose how recording gaps, analysis windows, and insufficient coverage affect
  the result.
- **Make the interface learnable across domains.** Keep names, argument order,
  units, shapes, and result methods consistent. Moving from one neuron to a
  population or from encoding to decoding should reuse what the scientist has
  already learned. Follow the existing public API contracts when extending them.
- **Preserve meaning between operations.** Keep timestamps, coordinate frames,
  spatial support, and unit identities aligned through filtering, analysis,
  plotting, and export. Validate incompatible inputs and explain how to fix
  them instead of silently guessing an alignment or conversion.
- **Make results and recovery understandable.** Label plots and tables with
  quantities and units, avoid relying on color alone, and expose the assumptions
  needed to interpret a result. Distinguish descriptive metrics and candidate
  cell labels from statistical evidence. Errors should identify the problem
  and give a concrete next step.
- **Evaluate complete workflows.** For analysis changes, verify the path from
  realistic inputs to an interpretable result using public APIs and runnable
  examples. Test scientific invariants and relevant edge cases, including gaps,
  empty data, and mismatched identities, rather than checking only that a call
  completes. Judge added functionality by whether scientists can use it correctly
  alongside existing operations.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone **your fork** locally (replace `YOUR_USERNAME`, or use
   `gh repo fork edeno/neurospatial --clone` to fork and clone in one step):

```bash
git clone https://github.com/YOUR_USERNAME/neurospatial.git
cd neurospatial
```

3. Install development dependencies:

```bash
uv sync --extra dev
```

4. Set up pre-commit hooks:

```bash
uv run pre-commit install
```

### Development Workflow

This project uses `uv` for package management. Always prefix commands with `uv run`:

```bash
# Run tests
uv run pytest

# Run specific test
uv run pytest tests/test_environment.py::test_name -v

# Check code quality
uv run ruff check .

# Format code
uv run ruff format .

# Type check (optional)
uv run mypy src/neurospatial
```

### Diffusion geometry providers

Smoothing, diffusion, smoothness bases, gradients and divergence all use one
layout capability, defined in `neurospatial.layout._diffusion`:

- `FiniteVolumeGeometryProvider.finite_volume_geometry()` returns a copy of the
  layout's connectivity graph with face measures (`A`) and edge distances
  (`distance`) on every edge, plus cell measures in active-bin order. Gradient
  and divergence index edge fields by `env.connectivity`, so the environment
  rejects a graph whose edges differ from it.
- `FaceGeometry.from_graph` validates those faces into read-only per-edge arrays
  in connectivity edge order. `DiffusionGeometry.from_faces` assembles the
  conductances `W = A / distance`; components come from positive conductances,
  so zero-area (corner-touching) edges and isolated bins form separate
  components.

A layout has the capability only when it defines the method, so `isinstance`
against the protocol is a truthful capability check. A layout that cannot
supply geometry may define `_explain_missing_geometry(operation)` to tailor the
error, as layouts read from non-grid NWB records do.

Grid, hexagonal, track, mesh, and polar layouts own their geometry assembly.
Polar environments are built on `_PolarGridLayout`, a masked grid over
`(distance, angle)` whose cell and face measures are the grid's scaled by
radius. A new layout or subclass does not need registration in the diffusion
solver. These are internal interfaces, not additional scientist-facing factory
requirements.

Layouts read their own `connectivity` and `bin_centers`. Code that replaces the
environment's graph after construction, such as the file and NWB readers, must
assign the same graph to `env.layout`; `copy()` rebuilds from the layout too.

`Environment._face_geometry` caches the validated faces once per state version;
the diffusion snapshot (`_diffusion_geometry`) and the gradient and divergence
operators all derive from it. Consumers read it through
`_require_face_geometry(operation)`, which names the analysis when the layout
has no cell faces. The existing state-version and `clear_cache()` mechanisms
own every cache lifetime.

## Code Standards

### Python Style

- Follow PEP 8 (enforced by ruff)
- Use type hints where practical
- Write descriptive variable names
- Keep functions focused and small

### Docstring Format

**CRITICAL**: All docstrings MUST follow NumPy docstring format:

```python
def function_name(param1, param2):
    """
    Short one-line summary ending with a period.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ValueError
        When this error occurs.

    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    """
```

See [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html) for details.

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(environment): add .info() method
fix(regions): correct area calculation for polygons
docs(quickstart): update installation instructions
test(layout): add tests for hexagonal layout
chore(deps): update numpy to 2.3.4
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Mirror source structure: `src/neurospatial/foo.py` → `tests/test_foo.py`
- Use descriptive test names: `test_environment_from_samples_creates_bins()`
- Use pytest fixtures from `tests/conftest.py`

Example test:

```python
def test_environment_bin_at_maps_points_correctly():
    """Test that bin_at correctly maps points to bin indices."""
    env = Environment.from_samples(sample_data, bin_size=2.0)
    points = np.array([[10.0, 10.0]])
    bin_idx = env.bin_at(points)
    assert bin_idx.shape == (1,)
    assert 0 <= bin_idx[0] < env.n_bins
```

### Running Tests

```bash
# All tests
uv run pytest

# Napari GUI tests (requires napari/Qt display support)
uv run pytest -n 0 -m "napari and not slow"

# Specific test file
uv run pytest tests/test_environment.py

# With coverage
uv run pytest --cov=src/neurospatial

# Verbose output
uv run pytest -v

# Stop at first failure
uv run pytest -x
```

## Pull Request Process

### Before Submitting

1. **Run all tests**: `uv run pytest`
2. **Check code quality**: `uv run ruff check . && uv run ruff format .`
3. **Update documentation**: Add/update docstrings and docs pages
4. **Add tests**: Ensure new code has test coverage
5. **Update CHANGELOG.md**: Add entry under "Unreleased"

### Submitting

1. Create a feature branch:

```bash
git checkout -b feature/amazing-feature
```

2. Make your changes with clear commit messages

3. Push to your fork:

```bash
git push origin feature/amazing-feature
```

4. Open a Pull Request on GitHub

### PR Description

Include in your PR description:

- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Breaking Changes**: Any API changes?
- **Related Issues**: Link to related issues

Example:

```markdown
## Summary
Adds `.info()` method to Environment class for printing summary statistics.

## Motivation
Users frequently need to inspect environment properties. This provides
a convenient way to see all key information at once.

## Testing
- Added tests in `test_environment.py`
- Manually tested with various environment types
- All existing tests pass

## Breaking Changes
None

## Related Issues
Closes #42
```

## Documentation

### Building Documentation

```bash
# Install docs dependencies
uv sync --extra docs

# Serve locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Documentation is at [https://edeno.github.io/neurospatial/](https://edeno.github.io/neurospatial/)

### Adding Documentation

- **New features**: Update relevant user guide pages
- **API changes**: Docstrings are auto-generated
- **Examples**: add a notebook to `examples/` as a jupytext `.py`/`.ipynb` pair
  (editing either keeps the pair in sync via jupytext), then run
  `uv run python docs/sync_notebooks.py` to refresh the committed
  `docs/examples/` mirror. CI verifies both are in sync.

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions handles PyPI release

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/edeno/neurospatial/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/edeno/neurospatial/issues)
- **Contact**: Email <eric.denovellis@ucsf.edu>

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment

## Recognition

Contributors will be acknowledged in:

- CHANGELOG.md
- GitHub contributors page
- Future publications using neurospatial

Thank you for contributing to neurospatial!
