# SIMULATION_PLAN.md Evaluation & Recommendations

**Evaluators**: Python Best Practices (Raymond Hettinger / Brandon Rhodes Principles)
**Date**: 2025-11-11
**Focus Areas**: API Design, Documentation Integration, Code Consistency

---

## Executive Summary

**Overall Assessment**: The plan is architecturally sound with excellent technical foundations. However, it has **5 critical API design issues** that would confuse users and violate Python best practices. These issues are fixable pre-implementation.

**Key Strengths**:

- ✓ Layered API design (low-level + high-level)
- ✓ Protocol-based architecture (composition over inheritance)
- ✓ Comprehensive mathematical specifications
- ✓ Strong example-driven documentation

**Critical Issues to Address**:

1. Return value inconsistency (dicts vs typed structures)
2. Import path complexity and discoverability
3. API naming inconsistencies
4. Missing integration with existing documentation notebooks
5. Remove/de-emphasize RatInABox integration (per user request)

---

## Principle 1: "There should be one-- and preferably only one --obvious way to do it"

### Issue 1.1: Return Value Inconsistency

**Problem**: The plan has both dict returns AND optional metadata patterns:

```python
# Pattern A: Always returns dict
session = simulate_session(...)  # Returns dict

# Pattern B: Optional metadata via flag
positions, times = simulate_trajectory_ou(...)
positions, times, metadata = simulate_trajectory_laps(..., return_metadata=True)
```

**Hettinger would say**: Pick ONE pattern for consistency. Mixing patterns makes the API harder to learn.

**Recommendation**: Use **NamedTuples or dataclasses** for structured returns:

```python
from typing import NamedTuple
from dataclasses import dataclass

# Option A: NamedTuple (immutable, lightweight)
class TrajectoryResult(NamedTuple):
    """Result of trajectory simulation."""
    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    metadata: dict | None = None

# Option B: dataclass (more flexible, but heavier)
@dataclass(frozen=True)
class SessionResult:
    """Complete simulation session."""
    env: Environment
    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    spike_trains: list[NDArray[np.float64]]
    models: list[NeuralModel]
    ground_truth: dict
    metadata: dict

# Usage - discoverable with IDE autocomplete!
result = simulate_trajectory_ou(env, duration=60.0)
positions = result.positions  # ✓ Discoverable
times = result.times          # ✓ Type-safe

session = simulate_session(env, duration=120.0)
session.spike_trains  # ✓ Autocomplete works
session.ground_truth  # ✓ Clear structure
```

**Benefits**:

- IDE autocomplete works
- Type checkers can verify usage
- Self-documenting (user sees all available fields)
- Can still unpack: `positions, times, _ = result`

### Issue 1.2: API Duplication

**Problem**: The plan has BOTH:

- Top-level functions: `simulate_session(env, ...)`
- Environment methods (optional): `env.create_place_cells(...)`

**Hettinger would say**: Having two ways to do the same thing is confusing. The "optional" qualifier makes it worse - commit or don't.

**Recommendation**: **Remove Environment methods**. Keep simulation as standalone functions.

**Rationale**:

1. Environment is already large (6000+ lines split across mixins)
2. Simulation is a separate concern (generating data vs spatial representation)
3. Functions are more composable and testable
4. Follows existing neurospatial pattern (e.g., `compute_place_field()` is a function, not a method)

---

## Principle 2: "Flat is better than nested"

### Issue 2.1: Import Path Complexity

**Problem**: Multiple import paths for related functionality:

```python
from neurospatial.simulation import simulate_session
from neurospatial.simulation.examples import open_field_session
from neurospatial.simulation.models import PlaceCellModel
from neurospatial.simulation import validate_simulation
```

**Rhodes would say**: Users shouldn't need to memorize 4 different import paths. Make the common case simple.

**Recommendation**: Flatten the public API in `__init__.py`:

```python
# src/neurospatial/simulation/__init__.py

# Trajectory functions
from .trajectory import (
    simulate_trajectory_ou,
    simulate_trajectory_sinusoidal,
    simulate_trajectory_laps,
)

# High-level workflows
from .session import simulate_session
from .validation import validate_simulation, plot_session_summary

# Pre-configured examples (most common imports)
from .examples import (
    open_field_session,
    linear_track_session,
    tmaze_alternation_session,
    boundary_cell_session,
    grid_cell_session,
)

# Neural models (less commonly needed - can import from submodule)
# Users who need models will do:
# from neurospatial.simulation.models import PlaceCellModel

# Spike generation
from .spikes import generate_poisson_spikes, generate_population_spikes

__all__ = [
    # Trajectory
    "simulate_trajectory_ou",
    "simulate_trajectory_sinusoidal",
    "simulate_trajectory_laps",
    # High-level
    "simulate_session",
    "validate_simulation",
    "plot_session_summary",
    # Examples (recommended starting point)
    "open_field_session",
    "linear_track_session",
    "tmaze_alternation_session",
    "boundary_cell_session",
    "grid_cell_session",
    # Spikes
    "generate_poisson_spikes",
    "generate_population_spikes",
]
```

**Usage becomes simple**:

```python
# Common case: 2 imports for 90% of users
from neurospatial import Environment
from neurospatial.simulation import open_field_session, validate_simulation

session = open_field_session(duration=120.0, seed=42)
report = validate_simulation(session)
```

---

## Principle 3: "Explicit is better than implicit"

### Issue 3.1: Magic dict Keys

**Problem**: Session dicts have undocumented keys that users must know:

```python
session = simulate_session(...)
# What keys exist? User must check docs or source
positions = session['positions']  # Hope the key is correct!
spike_trains = session['spike_trains']
```

**Recommendation**: Use **dataclass or NamedTuple** (see Issue 1.1):

```python
@dataclass(frozen=True)
class SimulationSession:
    """Results from simulation session.

    Attributes
    ----------
    env : Environment
        The spatial environment
    positions : NDArray, shape (n_time, n_dims)
        Position trajectory
    times : NDArray, shape (n_time,)
        Time points in seconds
    spike_trains : list[NDArray]
        Spike times for each neuron
    models : list[NeuralModel]
        Neural model instances (for ground truth)
    ground_truth : dict
        True parameters for validation
    metadata : dict
        Session parameters and statistics
    """
    env: Environment
    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    spike_trains: list[NDArray[np.float64]]
    models: list[NeuralModel]
    ground_truth: dict
    metadata: dict

    # Convenience method for common workflow
    def validate(self, method='diffusion_kde', show_plots=False):
        """Validate simulation against ground truth."""
        from .validation import validate_simulation
        return validate_simulation(
            session=self,
            method=method,
            show_plots=show_plots
        )
```

**Usage**:

```python
session = simulate_session(env, duration=120.0)
# IDE shows all available attributes!
session.positions  # ✓ Autocomplete
session.ground_truth  # ✓ Discoverable

# Convenience method
report = session.validate(show_plots=True)
```

---

## Principle 4: "Simple things should be simple, complex things should be possible"

### Issue 4.1: `simulate_session()` has too many parameters

**Problem**: 9 parameters with complex interactions:

```python
def simulate_session(
    env: Environment,
    duration: float,
    n_cells: int = 50,
    cell_type: Literal["place", "boundary", "grid", "mixed"] = "place",
    trajectory_method: Literal["ou", "sinusoidal", "laps"] = "ou",
    coverage: Literal["uniform", "random"] = "uniform",
    show_progress: bool = True,
    seed: int | None = None,
    **kwargs,  # ← Especially this - what can go here?
) -> dict:
```

**Hettinger would say**: `**kwargs` is a code smell in public APIs. Users can't discover what's allowed.

**Recommendation**: Make **pre-configured examples the simple path** (they already exist!), and make `simulate_session()` a power-user tool with explicit parameters:

```python
def simulate_session(
    env: Environment,
    duration: float,
    models: list[NeuralModel],
    trajectory_method: Literal["ou", "sinusoidal", "laps"] = "ou",
    trajectory_params: dict | None = None,
    show_progress: bool = True,
    seed: int | None = None,
) -> SimulationSession:
    """Simulate complete recording session (power-user API).

    For quick starts, use pre-configured examples instead:
    - open_field_session()
    - linear_track_session()
    - tmaze_alternation_session()

    Parameters
    ----------
    env : Environment
        Spatial environment
    duration : float
        Session duration in seconds
    models : list[NeuralModel]
        Pre-configured neural models. For automatic setup, use examples.
    trajectory_method : {'ou', 'sinusoidal', 'laps'}
        Trajectory generation method
    trajectory_params : dict | None
        Parameters for trajectory (e.g., {'speed_mean': 0.08, 'coherence_time': 0.7})
    show_progress : bool
        Show progress bars
    seed : int | None
        Random seed

    Returns
    -------
    session : SimulationSession
        Simulation results with env, positions, times, spike_trains, models,
        ground_truth, and metadata

    See Also
    --------
    open_field_session : Quickstart for open field experiments
    PlaceCellModel : Create individual place cell models
    """
    pass

# Simple case: Use examples
session = open_field_session(duration=120.0, seed=42)

# Complex case: Full control
from neurospatial.simulation.models import PlaceCellModel, BoundaryCellModel

models = [
    PlaceCellModel(env, center=[50, 75], width=10.0),
    PlaceCellModel(env, center=[30, 40], width=8.0),
    BoundaryCellModel(env, distance=15.0, direction='north'),
]
session = simulate_session(
    env,
    duration=180.0,
    models=models,
    trajectory_method='ou',
    trajectory_params={'speed_mean': 0.1, 'coherence_time': 0.8}
)
```

---

## Principle 5: "Readability counts"

### Issue 5.1: Naming Inconsistency

**Problem**: Mixed terminology - "session" vs "simulation":

```python
simulate_session()      # Function uses "session"
open_field_session()    # Example uses "session"
SimulationSession()     # Class uses both!
validate_simulation()   # Function uses "simulation"
```

**Recommendation**: **Use "session" consistently** (it's the more precise term in neuroscience):

```python
# ✓ Consistent naming
from neurospatial.simulation import (
    simulate_session,
    validate_session,
    plot_session_summary,
)

from neurospatial.simulation import (
    open_field_session,
    linear_track_session,
)
```

---

## Integration with Existing Documentation

### Critical Gap: Example Notebooks Not Updated

**Problem**: The plan has extensive API examples but **no plan to integrate with existing notebooks**.

Current `docs/examples/08_complete_workflow.ipynb` contains **200+ lines of hand-written simulation code** that should use the new simulation subpackage!

**Recommendation**: Add new Phase to update documentation:

### Phase 3.5: Documentation Integration (0.5 weeks)

**Goal**: Replace hand-written simulation code in notebooks with simulation subpackage

**Tasks**:

1. ✅ Update `08_complete_workflow.ipynb`:
   - Replace `generate_plus_maze_trajectory()` → `simulate_trajectory_ou()`
   - Replace `generate_place_cell_spikes()` → `PlaceCellModel` + `generate_population_spikes()`
   - Show validation workflow with `validate_session()`
   - Demonstrate both pre-configured (`open_field_session()`) and custom approaches

2. ✅ Create new notebook: `09_simulation_validation.ipynb`:
   - Showcase simulation → detection → validation pipeline
   - Compare different environments (open field, linear track, T-maze)
   - Demonstrate ground truth validation for algorithm benchmarking
   - Show how to test new place field methods with synthetic data

3. ✅ Update `01_introduction_basics.ipynb`:
   - Add "Quick Example" section at the end showing:

     ```python
     from neurospatial.simulation import open_field_session
     session = open_field_session(duration=60.0, seed=42)
     # Now you have env, positions, spike_trains, ground_truth!
     ```

4. ✅ Create simulation gallery:
   - `docs/examples/simulation/` directory
   - `basic_place_cells.ipynb`
   - `boundary_cells.ipynb`
   - `grid_cells.ipynb`
   - `multi_environment_testing.ipynb`
   - `custom_neural_models.ipynb`

**Example Transformation** (`08_complete_workflow.ipynb`):

```python
# BEFORE (current notebook - 200 lines of simulation code):
def generate_plus_maze_trajectory(n_samples=36000, sampling_rate=30.0):
    """Hand-written trajectory simulation..."""
    # 50 lines of code...
    return trajectory, timestamps

def generate_place_cell_spikes(position_data, timestamps, n_neurons=20):
    """Hand-written spike generation..."""
    # 80 lines of code...
    return spike_times, place_field_centers

# AFTER (with simulation subpackage):
from neurospatial import Environment
from neurospatial.simulation import simulate_session, validate_session

# Option 1: Quick start (recommended for learning)
session = open_field_session(duration=1200.0, n_place_cells=20, seed=42)
env = session.env
positions = session.positions
times = session.times
spike_times = session.spike_trains

# Option 2: Custom environment (for advanced users)
env = Environment.from_samples(custom_maze_data, bin_size=3.0)
env.units = "cm"
session = simulate_session(
    env,
    duration=1200.0,
    models=custom_place_cell_models,
    trajectory_method='ou',
    seed=42
)

# Validation is now built-in!
report = session.validate(show_plots=True)
print(f"Validation: {'PASS' if report['passed'] else 'FAIL'}")
```

**Benefits**:

- Notebooks become shorter and more focused on neurospatial features
- Users see real-world usage of simulation subpackage
- Examples serve as additional tests (can be run in CI)
- Reduces maintenance burden (one simulation codebase, not scattered examples)

---

## RatInABox Integration

### Issue: Unnecessary Complexity

**User stated**: "We don't need RatInABox integration"

**Recommendation**: **Remove Phase 5 entirely**. Keep simulation self-contained.

**Rationale**:

1. RatInABox is an external dependency (maintenance burden)
2. neurospatial simulation is sufficient for validation
3. Users who want RatInABox can handle their own conversions
4. Focus resources on core functionality

**Alternative**: If cross-validation against RatInABox is desired, make it an **external validation notebook** (not core functionality):

```python
# docs/examples/validation/compare_with_ratinabox.ipynb
# This is a VALIDATION, not a feature
```

---

## Testing Strategy Improvements

### Current Plan: Unit tests separated from examples

**Rhodes would say**: "Your examples should BE your tests."

**Recommendation**: **Example-driven testing**:

```python
# tests/simulation/test_examples.py

def test_open_field_session_runs():
    """Example 0 from documentation should work."""
    session = open_field_session(duration=60.0, n_place_cells=20, seed=42)
    assert session.env.n_bins > 0
    assert len(session.spike_trains) == 20
    assert session.positions.shape[0] == session.times.shape[0]

def test_validation_workflow():
    """Example 0 validation should work."""
    session = open_field_session(duration=120.0, n_place_cells=30, seed=42)
    report = session.validate()
    # Should pass with good synthetic data
    assert report['passed'], f"Validation failed: {report['summary']}"

def test_example_1_basic_place_cell():
    """Example 1 from documentation should work."""
    # Exact code from Example 1 in docs
    env = Environment.from_samples(arena_data, bin_size=2.0)
    # ...
```

**Benefits**:

- Documentation stays in sync with code (examples break if API changes)
- Users can trust that examples actually work
- Reduces testing boilerplate (examples are tests)

---

## API Design Checklist (Hettinger/Rhodes)

**Before Implementation, Verify:**

- [ ] **One obvious way**: Is there exactly one way to accomplish common tasks?
- [ ] **Discoverability**: Can users find functionality without reading docs?
- [ ] **Type safety**: Do return types support IDE autocomplete?
- [ ] **Consistency**: Do similar functions use the same patterns?
- [ ] **Flat imports**: Can users import what they need in 1-2 lines?
- [ ] **Example-driven**: Do all public functions have working examples?
- [ ] **Test what you document**: Do examples run as tests?

---

## Revised Implementation Priorities

### Phase 1: Core Trajectory + Place Cells (1 week)

- Implement trajectory simulation
- Implement PlaceCellModel
- Implement spike generation
- **Use NamedTuples for return values**
- **Flatten imports in `__init__.py`**

### Phase 2: Boundary Cells + Extended Models (1 week)

- BoundaryCellModel
- Trajectory extensions (laps)
- Condition functions

### Phase 3: Grid Cells + Convenience (1.5 weeks)

- GridCellModel
- **`SimulationSession` dataclass**
- Pre-configured examples (already well-designed!)
- Validation helpers
- **Remove Environment methods** (keep functions only)

### Phase 3.5: Documentation Integration (0.5 weeks) **[NEW]**

- Update `08_complete_workflow.ipynb`
- Create `09_simulation_validation.ipynb`
- Update `01_introduction_basics.ipynb` with quick example
- Create simulation gallery

### Phase 4: Advanced Features (1 week, optional) **[REVISED]**

- State-dependent movement
- Bursting behavior
- Elliptical place fields
- ~~RatInABox integration~~ **REMOVED**

---

## Summary of Changes

**Critical Fixes (Must Do Before Implementation)**:

1. Use NamedTuple/dataclass for return values (not dicts)
2. Flatten imports in `__init__.py`
3. Remove Environment methods (keep functions)
4. Use consistent "session" terminology
5. Make pre-configured examples the recommended path
6. Remove RatInABox integration

**High Value Additions**:
7. Add Phase 3.5 for documentation integration
8. Convert examples to tests (example-driven testing)
9. Add `SimulationSession.validate()` convenience method

**Impact**: These changes make the API **more discoverable, more consistent, and easier to learn** while maintaining the strong technical foundations. Time to implement: +0.5 weeks (for documentation integration).

---

## Conclusion

The SIMULATION_PLAN.md has **excellent scientific and technical foundations**, but needs **API polish** to meet Python best practices. The recommended changes are **straightforward to implement** and will significantly improve user experience.

**Key Insight**: The plan already has the right pieces (layered API, pre-configured examples, validation). It just needs better **organization and presentation** to match the quality of the rest of the neurospatial codebase.

**Next Step**: Incorporate these recommendations into SIMULATION_PLAN.md before starting implementation. This will save significant refactoring time later and ensure a better user experience from day one.
