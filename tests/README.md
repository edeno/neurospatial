# Neurospatial Test Suite

This document provides guidance for running and writing tests in the neurospatial project.

## Running Tests

### Quick Development Workflow

```bash
# Fast tests only (excludes slow tests marked with @pytest.mark.slow)
uv run pytest -m "not slow"

# Run specific test file
uv run pytest tests/test_spike_field.py -v

# Run specific test function
uv run pytest tests/test_environment.py::test_bin_at -v

# Run specific test class
uv run pytest tests/metrics/test_place_fields.py::TestSparsityProperties -v

# Run tests matching a pattern
uv run pytest -k "sparsity" -v
```

### Full Test Suite (CI)

```bash
# All tests including slow ones
uv run pytest

# All tests with verbose output
uv run pytest -v

# Stop on first failure (useful for debugging)
uv run pytest -x

# Run with coverage report
uv run pytest --cov=src/neurospatial --cov-report=html --cov-report=term

# Run in parallel (requires pytest-xdist)
uv run pytest -n auto
```

### Performance Benchmarks

```bash
# Run benchmark suite
uv run pytest tests/benchmarks/ -v

# Save baseline benchmark
uv run pytest tests/benchmarks/ --benchmark-save=baseline

# Compare against baseline
uv run pytest tests/benchmarks/ --benchmark-compare=baseline

# Generate benchmark histogram
uv run pytest tests/benchmarks/ --benchmark-histogram
```

## Test Organization

The test suite is organized by functionality:

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures and test constants
├── test_*.py                    # Core functionality tests
├── benchmarks/                  # Performance regression tests
│   └── test_performance.py      # pytest-benchmark tests
├── layout/                      # Layout engine tests
│   ├── test_graph_building.py
│   ├── test_regular_grid_layout.py
│   ├── test_hex_grid_utils.py
│   └── ...
├── metrics/                     # Neuroscience metric tests
│   ├── test_place_fields.py     # Place field detection and metrics
│   ├── test_boundary_cells.py   # Border score and boundary cells
│   ├── test_grid_cells.py       # Grid score
│   └── test_population.py       # Population-level metrics
├── regions/                     # Region functionality tests
│   ├── test_core.py
│   ├── test_ops.py
│   └── test_serialization.py
├── segmentation/                # Behavioral segmentation tests
│   ├── test_laps.py
│   ├── test_regions.py
│   └── test_trials.py
├── simulation/                  # Simulation validation tests
│   └── test_integration.py
└── test_properties.py           # Hypothesis property-based tests
```

## Test Types

### Unit Tests
Test individual functions and methods in isolation.

**Example:**
```python
def test_sparsity_range(simple_env):
    """Test that sparsity is always in [0, 1]."""
    firing_rate = np.random.rand(simple_env.n_bins)
    occupancy = np.ones(simple_env.n_bins)

    sp = sparsity(firing_rate, occupancy)

    assert 0.0 <= sp <= 1.0
```

### Integration Tests
Test interactions between multiple components.

**Example:**
```python
@pytest.mark.integration
def test_place_field_pipeline(medium_2d_env):
    """Test complete place field estimation pipeline."""
    # Create synthetic trajectory
    positions = generate_trajectory(medium_2d_env)
    times = np.linspace(0, 100, len(positions))
    spike_times = generate_spike_times(positions, peak_location)

    # Compute place field
    firing_rate = compute_place_field(
        medium_2d_env, spike_times, times, positions,
        smoothing_method="diffusion_kde", bandwidth=5.0
    )

    # Detect place fields
    fields = detect_place_fields(firing_rate, medium_2d_env)

    assert len(fields) > 0
```

### Property-Based Tests
Use Hypothesis to generate randomized test cases and verify mathematical properties.

**Example:**
```python
from hypothesis import given, strategies as st

@given(valid_firing_rate_and_occupancy())
def test_sparsity_properties(data):
    """Property: sparsity is always in [0, 1] for all valid inputs."""
    firing_rate, occupancy = data
    sp = sparsity(firing_rate, occupancy)
    assert 0.0 <= sp <= 1.0
```

### Performance Benchmarks
Track performance regressions using pytest-benchmark.

**Example:**
```python
def test_environment_creation_performance(benchmark):
    """Benchmark Environment.from_samples() performance."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 100, (1000, 2))

    result = benchmark(Environment.from_samples, positions, bin_size=5.0)
    assert result.n_bins > 0
```

## Writing Tests

### Use Fixtures Instead of Creating Data

**❌ Bad:**
```python
def test_my_feature():
    # Creates new environment every time
    rng = np.random.default_rng(42)
    positions = rng.standard_normal((1000, 2)) * 20
    env = Environment.from_samples(positions, bin_size=2.0)
    # ... test code
```

**✅ Good:**
```python
def test_my_feature(medium_2d_env):
    # Use pre-built session-scoped fixture
    # ... test code using medium_2d_env
```

### Available Fixtures

Import from `conftest.py`:

```python
# Environment fixtures (session-scoped for performance)
small_2d_env         # ~25 bins, 100 samples, for quick tests
medium_2d_env        # ~625 bins, 1000 samples, for standard tests
large_2d_env         # ~2500 bins, 5000 samples, for stress tests
small_1d_env         # ~5 bins, 1D linear track
medium_2d_env_with_diagonal  # ~625 bins, diagonal connectivity

# Legacy fixtures (for compatibility)
simple_env           # Small 2D environment
simple_3d_env        # Small 3D environment with diagonal connectivity
```

### Test Constants

Use constants from `conftest.py` instead of magic numbers:

```python
from conftest import MEDIUM_BIN_SIZE, DEFAULT_SEED, MEDIUM_TOLERANCE

def test_my_feature():
    rng = np.random.default_rng(DEFAULT_SEED)
    positions = rng.uniform(0, 100, (1000, 2))
    env = Environment.from_samples(positions, bin_size=MEDIUM_BIN_SIZE)

    result = compute_something(env)
    expected = 10.0

    assert result == pytest.approx(expected, rel=MEDIUM_TOLERANCE)
```

**Available constants:**
- `SMALL_BIN_SIZE = 1.0`
- `MEDIUM_BIN_SIZE = 2.0`
- `LARGE_BIN_SIZE = 10.0`
- `DEFAULT_SEED = 42`
- `SMALL_N_POSITIONS = 100`
- `MEDIUM_N_POSITIONS = 1000`
- `LARGE_N_POSITIONS = 5000`
- `SMALL_EXTENT = 10.0`
- `MEDIUM_EXTENT = 50.0`
- `LARGE_EXTENT = 100.0`

### Mark Slow Tests

Mark tests that take >0.5 seconds with `@pytest.mark.slow`:

```python
@pytest.mark.slow
def test_expensive_operation(large_2d_env):
    """Test expensive operation with large dataset.

    Marked slow: Creates 5000 samples and computes expensive metric.
    """
    # ... expensive test code
```

**When to mark slow:**
- Test takes >0.5 seconds consistently
- Creates >1000 data points
- Uses computationally expensive methods (gaussian_kde, large matrix operations)
- Computes multiple variations for comparison

### Parametrize Similar Tests

**❌ Bad:**
```python
def test_sparsity_uniform():
    firing_rate = np.ones(100)
    occupancy = np.ones(100)
    sp = sparsity(firing_rate, occupancy)
    assert sp < 0.1

def test_sparsity_sparse():
    firing_rate = np.zeros(100)
    firing_rate[50] = 100.0
    occupancy = np.ones(100)
    sp = sparsity(firing_rate, occupancy)
    assert sp > 0.5
```

**✅ Good:**
```python
@pytest.mark.parametrize("firing_pattern,expected_range", [
    ("uniform", (0.0, 0.1)),
    ("sparse", (0.5, 1.0)),
    ("single_peak", (0.3, 0.7)),
])
def test_sparsity(firing_pattern, expected_range):
    firing_rate = create_pattern(firing_pattern)
    occupancy = np.ones_like(firing_rate)

    sp = sparsity(firing_rate, occupancy)

    assert expected_range[0] <= sp <= expected_range[1]
```

### Use Local RNG Instead of Global Seed

**❌ Bad:**
```python
def test_my_feature():
    np.random.seed(42)  # Global state, affects other tests
    positions = np.random.randn(1000, 2)
```

**✅ Good:**
```python
def test_my_feature():
    rng = np.random.default_rng(42)  # Local RNG, isolated
    positions = rng.standard_normal((1000, 2))
```

### Write Clear Docstrings

Every test should have a docstring explaining:
- What is being tested
- Why this test is important
- Any special conditions (if marked slow, integration, etc.)

```python
def test_skaggs_information_uniform_firing():
    """Test that uniform firing produces zero spatial information.

    Uniform firing means each spike conveys no information about location,
    so Skaggs information should be approximately 0 bits/spike.

    Mathematical property: uniform firing → I ≈ 0
    """
    # ... test code
```

## Test Markers

Available pytest markers:

- `@pytest.mark.slow` - Test takes >0.5 seconds (excluded with `-m "not slow"`)
- `@pytest.mark.integration` - Integration test (tests multiple components)
- `@pytest.mark.skipif(condition, reason="...")` - Conditional skip
- `@pytest.mark.parametrize(...)` - Parametrize test with multiple inputs

## Coverage

View test coverage:

```bash
# Generate HTML coverage report
uv run pytest --cov=src/neurospatial --cov-report=html

# Open report in browser
open htmlcov/index.html

# Terminal report only
uv run pytest --cov=src/neurospatial --cov-report=term

# Find files with low coverage (<90%)
uv run pytest --cov=src/neurospatial --cov-report=term | grep -E "^src.*[0-8][0-9]%"
```

**Coverage targets:**
- Core modules (environment, metrics): >90%
- Layout engines: >85%
- Utilities and helpers: >80%

## Debugging Failed Tests

### Get More Information

```bash
# Show full traceback
uv run pytest tests/test_my_file.py -v --tb=long

# Show local variables in traceback
uv run pytest tests/test_my_file.py -v --showlocals

# Drop into debugger on failure
uv run pytest tests/test_my_file.py --pdb

# Show print statements
uv run pytest tests/test_my_file.py -s
```

### Common Issues

**Issue: Test fails intermittently**
- Likely caused by global random state
- Fix: Use local RNG (`np.random.default_rng()`)

**Issue: Test is slow**
- Mark with `@pytest.mark.slow`
- Consider reducing dataset size
- Use smaller fixtures (`small_2d_env` instead of `large_2d_env`)

**Issue: Fixture not found**
- Ensure fixture is in `conftest.py` or imported
- Check fixture scope matches usage

## Contributing Tests

When adding new tests:

1. **Choose the right location:**
   - Core functionality → `tests/test_*.py`
   - Neuroscience metrics → `tests/metrics/`
   - Layout engines → `tests/layout/`
   - Regions → `tests/regions/`

2. **Use existing fixtures when possible**
   - Check `tests/conftest.py` for available fixtures
   - Add new fixtures to `conftest.py` if reusable

3. **Follow naming conventions:**
   - Test files: `test_*.py`
   - Test functions: `test_<feature>_<condition>`
   - Test classes: `Test<Feature>`

4. **Write clear docstrings:**
   - Explain what is being tested
   - Include mathematical properties for scientific tests
   - Note if test is slow or integration

5. **Mark slow tests:**
   - Add `@pytest.mark.slow` if test takes >0.5 seconds

6. **Parametrize when appropriate:**
   - Use `@pytest.mark.parametrize` for similar test cases
   - Don't parametrize tests with fundamentally different logic

7. **Use assertions effectively:**
   - Use `pytest.approx()` for floating-point comparisons
   - Include helpful assertion messages
   - Test both success and failure cases

## Resources

- **Pytest Documentation**: <https://docs.pytest.org/>
- **Hypothesis Documentation**: <https://hypothesis.readthedocs.io/>
- **pytest-benchmark**: <https://pytest-benchmark.readthedocs.io/>
- **Scientific Python Testing**: <https://scientific-python.org/specs/spec-0001/>
- **NumPy Testing Guidelines**: <https://numpy.org/doc/stable/reference/testing.html>

## Getting Help

If you have questions about testing:

1. Check this README
2. Look at existing tests for similar functionality
3. Review `tests/conftest.py` for available fixtures
4. Consult pytest documentation

---

**Last Updated**: 2025-11-16
**Maintainer**: Development Team
