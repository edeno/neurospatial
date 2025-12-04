# BASIS_TASKS.md

Actionable task breakdown for implementing maze-aware spatial basis functions.

**Plan Source**: [BASIS_PLAN.md](BASIS_PLAN.md)
**Created**: 2025-12-04

---

## Overview

This task list implements the `neurospatial.basis` module providing localized, maze-aware basis functions for spatial regression (GLMs). The implementation follows TDD and builds incrementally from simple to complex components.

**Dependencies**: No new packages required (uses existing numpy, scipy, sklearn, networkx)

---

## Milestone 1: Module Setup & Center Selection

**Goal**: Create module structure and implement center selection utility.

**Success Criteria**:

- `select_basis_centers` returns valid node indices
- All methods (kmeans, farthest_point, random) work correctly
- Tests pass with 100% coverage for this function

### Tasks

- [x] **M1.1**: Create `src/neurospatial/basis.py` with module docstring and `__all__`
  - Add comprehensive module docstring from BASIS_PLAN.md Part 8.1
  - Define `__all__` exports
  - Add TYPE_CHECKING imports
  - **Files**: `src/neurospatial/basis.py` (new)

- [x] **M1.2**: Create `tests/test_basis.py` with pytest fixtures
  - Add fixtures: `simple_2d_env`, `linear_env`, `maze_env_with_wall`, `disconnected_env`
  - Follow patterns from BASIS_PLAN.md Part 9
  - **Files**: `tests/test_basis.py` (new)

- [x] **M1.3**: Implement `select_basis_centers` with kmeans method
  - Input validation (n_centers > 0, n_centers <= n_bins)
  - KMeans clustering on bin_centers
  - Map cluster centroids to nearest bins via cKDTree
  - **Dependencies**: sklearn.cluster.KMeans, scipy.spatial.cKDTree

- [x] **M1.4**: Add farthest_point method to `select_basis_centers`
  - Greedy farthest-point sampling using geodesic distance
  - Start with random point, iteratively add farthest from current set
  - Use scipy.sparse.csgraph.shortest_path for efficiency

- [x] **M1.5**: Add random and grid methods to `select_basis_centers`
  - Random: uniform sampling with random_state support
  - Grid: NotImplementedError (deferred)

- [x] **M1.6**: Write tests for `TestSelectBasisCenters` class
  - test_kmeans_returns_correct_count
  - test_centers_are_valid_nodes
  - test_random_state_reproducibility
  - test_raises_if_too_many_centers
  - test_raises_if_n_centers_zero
  - Run: `uv run pytest tests/test_basis.py::TestSelectBasisCenters -v`

---

## Milestone 2: Normalization Helper & Geodesic RBF Basis

**Goal**: Implement the simplest basis function type using geodesic distances.

**Success Criteria**:

- `geodesic_rbf_basis` returns correct shape (n_centers * n_sigmas, n_bins)
- Basis functions peak at their centers
- Basis respects walls (no leakage through barriers)
- All normalization modes work correctly

### Tasks

- [x] **M2.1**: Implement `_normalize_basis` helper function
  - Support modes: "unit" (L2 norm=1), "max" (max=1), "none"
  - Handle zero-norm rows gracefully
  - **Location**: `src/neurospatial/basis.py`

- [x] **M2.2**: Implement `geodesic_rbf_basis` core logic
  - Handle centers (explicit or auto-select via n_centers)
  - Validate sigma (must be positive)
  - Use scipy.sparse.csgraph.shortest_path with indices parameter
  - Compute RBF: exp(-d²/(2σ²))
  - **Dependencies**: scipy.sparse.csgraph.shortest_path

- [x] **M2.3**: Add multi-sigma support to `geodesic_rbf_basis`
  - Accept single float or sequence of floats
  - Output shape: (n_centers * n_sigmas, n_bins)
  - Row ordering: (center, sigma) - all sigmas for center 0, then center 1, etc.

- [x] **M2.4**: Add disconnected graph detection
  - Check for infinite distances from shortest_path
  - Raise helpful ValueError with diagnostic commands

- [x] **M2.5**: Write tests for `TestGeodesicRBFBasis` class
  - test_output_shape_single_sigma
  - test_output_shape_multi_sigma
  - test_row_ordering_center_major
  - test_center_has_max_value
  - test_respects_walls
  - test_unit_normalization
  - test_raises_if_sigma_non_positive
  - test_raises_if_no_centers_specified
  - test_raises_if_disconnected_graph
  - Run: `uv run pytest tests/test_basis.py::TestGeodesicRBFBasis -v`

---

## Milestone 3: Heat Kernel Wavelet Basis

**Goal**: Implement diffusion-based basis functions using matrix exponential.

**Success Criteria**:

- `heat_kernel_wavelet_basis` returns correct shape
- Larger scales produce wider spatial support
- Heat diffusion respects graph structure (walls)

### Tasks

- [x] **M3.1**: Implement `heat_kernel_wavelet_basis` core logic
  - Get graph Laplacian via nx.laplacian_matrix
  - Create delta vectors at centers
  - Use scipy.sparse.linalg.expm_multiply for efficient exp(-s*L) @ B
  - **Dependencies**: scipy.sparse.linalg.expm_multiply

- [x] **M3.2**: Add multi-scale support
  - Default scales: (0.5, 1.0, 2.0, 4.0)
  - Same row ordering as geodesic_rbf: (center, scale)

- [x] **M3.3**: Add input validation
  - Validate scales (must be positive)
  - Handle centers (explicit or auto-select)

- [x] **M3.4**: Write tests for `TestHeatKernelWaveletBasis` class
  - test_output_shape
  - test_larger_scale_is_wider
  - test_respects_graph_structure
  - test_raises_if_scales_non_positive
  - Run: `uv run pytest tests/test_basis.py::TestHeatKernelWaveletBasis -v`

---

## Milestone 4: Chebyshev Filter Basis

**Goal**: Implement polynomial filter basis with strict k-hop locality.

**Success Criteria**:

- `chebyshev_filter_basis` returns correct shape (n_centers * (max_degree+1), n_bins)
- Degree 0 is identity (delta function)
- Degree k only affects k-hop neighbors
- Spectral radius estimation works robustly

### Tasks

- [x] **M4.1**: Implement `_estimate_spectral_radius` helper
  - Use scipy.sparse.linalg.eigsh with k=1, which='LM'
  - Fall back to max-degree bound (2 * max node degree) if eigsh fails
  - Handle small matrices (n <= 2) directly
  - **Dependencies**: scipy.sparse.linalg.eigsh

- [x] **M4.2**: Implement `chebyshev_filter_basis` core logic
  - Get Laplacian and rescale to [-1, 1]: L_scaled = 2L/λ_max - I
  - Chebyshev recurrence: T₀=1, T₁=x, T_{k+1}=2x*T_k - T_{k-1}
  - Each iteration is sparse matrix-vector product

- [x] **M4.3**: Add proper row ordering
  - Output shape: (n_centers * (max_degree + 1), n_bins)
  - Row ordering: (center, degree) - all degrees for center 0, then center 1

- [x] **M4.4**: Add input validation
  - Validate max_degree >= 0
  - Handle centers (explicit or auto-select)

- [x] **M4.5**: Write tests for `TestChebyshevFilterBasis` class
  - test_output_shape
  - test_degree_0_is_delta
  - test_k_hop_locality
  - test_raises_if_max_degree_negative
  - Run: `uv run pytest tests/test_basis.py::TestChebyshevFilterBasis -v`

---

## Milestone 5: Convenience Function & Visualization

**Goal**: Provide easy-to-use wrapper and visualization helper.

**Success Criteria**:

- `spatial_basis` returns sensible defaults without parameter tuning
- `plot_basis_functions` creates informative visualization
- Coverage settings produce appropriately scaled bases

### Tasks

- [x] **M5.1**: Implement `spatial_basis` convenience function
  - Compute environment extent (geometric mean for multi-dim)
  - Map coverage ("local", "medium", "global") to sigma fractions (5%, 10%, 20%)
  - Create multi-scale basis with 3 octave-spaced sigmas
  - Compute n_centers to achieve target n_features

- [x] **M5.2**: Implement `plot_basis_functions` visualization helper
  - Select random indices if not specified
  - Create subplot grid with env.plot() for each basis
  - Handle empty subplots gracefully
  - Return matplotlib Figure

- [x] **M5.3**: Write tests for `TestSpatialBasis` class
  - test_returns_basis
  - test_coverage_affects_sigma
  - test_n_features_approximate
  - test_random_state_reproducibility
  - Run: `uv run pytest tests/test_basis.py::TestSpatialBasis -v`

- [x] **M5.4**: Write tests for `TestPlotBasisFunctions` class
  - test_returns_figure
  - test_respects_indices
  - Use matplotlib Agg backend for CI
  - Run: `uv run pytest tests/test_basis.py::TestPlotBasisFunctions -v`

---

## Milestone 6: Integration & Exports

**Goal**: Integrate module into package and update exports.

**Success Criteria**:

- All functions importable from `neurospatial` and `neurospatial.basis`
- Type checking passes with mypy
- Full test suite passes

### Tasks

- [x] **M6.1**: Update `src/neurospatial/__init__.py`
  - Add imports for all public functions
  - Add to `__all__` list
  - **Exports**: spatial_basis, select_basis_centers, geodesic_rbf_basis, heat_kernel_wavelet_basis, chebyshev_filter_basis, plot_basis_functions

- [x] **M6.2**: Run type checking
  - `uv run mypy src/neurospatial/basis.py`
  - Fix any type errors
  - Ensure all function signatures have proper type hints

- [x] **M6.3**: Run full test suite
  - `uv run pytest tests/test_basis.py -v --cov=src/neurospatial/basis`
  - Target: 90%+ coverage

- [x] **M6.4**: Run linting and formatting
  - `uv run ruff check src/neurospatial/basis.py tests/test_basis.py`
  - `uv run ruff format src/neurospatial/basis.py tests/test_basis.py`

---

## Milestone 7: Documentation

**Goal**: Update documentation with basis function examples and API reference.

**Success Criteria**:

- QUICKSTART.md has working copy-paste examples
- API_REFERENCE.md lists all imports
- Docstrings are comprehensive with examples

### Tasks

- [x] **M7.1**: Update `.claude/QUICKSTART.md`
  - Add "Spatial Basis Functions for GLMs" section
  - Include quick example with geodesic_rbf_basis
  - Show GLM design matrix creation pattern

- [x] **M7.2**: Update `.claude/API_REFERENCE.md`
  - Add Basis Functions section
  - List all public functions with import statements
  - Group by category (convenience, center selection, basis types, visualization)

- [x] **M7.3**: Verify all docstrings follow NumPy format
  - Check Parameters, Returns, Examples sections
  - Ensure examples are runnable (doctest)
  - Run: `uv run pytest --doctest-modules src/neurospatial/basis.py`

- [x] **M7.4**: Add usage example to module docstring
  - Full GLM workflow example
  - Phase precession example (if appropriate)

---

## Milestone 8: Final Validation

**Goal**: Comprehensive testing and validation before merge.

**Success Criteria**:

- All tests pass
- No regressions in existing functionality
- Performance is acceptable for typical use cases

### Tasks

- [x] **M8.1**: Run complete test suite
  - `uv run pytest` (all tests)
  - Ensure no regressions

- [x] **M8.2**: Performance benchmarking
  - Test with 100x100 grid (10,000 bins), 100 centers
  - geodesic_rbf_basis should complete in < 5s
  - heat_kernel_wavelet_basis should complete in < 15s
  - chebyshev_filter_basis should complete in < 3s

- [x] **M8.3**: Integration test with real-world-like data
  - Create environment from realistic position data
  - Generate basis and create design matrix
  - Verify shapes and values are reasonable

- [x] **M8.4**: Code review checklist
  - [x] All functions have comprehensive docstrings
  - [x] Error messages are helpful (WHAT/WHY/HOW)
  - [x] No bare Environment() calls (use factories)
  - [x] Uses `uv run` for all commands
  - [x] Type hints are complete and correct

---

## Quick Reference

### Running Tests

```bash
# All basis tests
uv run pytest tests/test_basis.py -v

# Specific test class
uv run pytest tests/test_basis.py::TestGeodesicRBFBasis -v

# With coverage
uv run pytest tests/test_basis.py --cov=src/neurospatial/basis --cov-report=term-missing
```

### Type Checking

```bash
uv run mypy src/neurospatial/basis.py
```

### Linting

```bash
uv run ruff check src/neurospatial/basis.py tests/test_basis.py
uv run ruff format src/neurospatial/basis.py tests/test_basis.py
```

---

## Notes

- **TDD approach**: Write tests first (M1.6, M2.5, etc.) before implementing
- **Incremental commits**: Commit after each milestone with conventional format
- **Dependencies**: All deps already in pyproject.toml (scipy, sklearn, networkx)
- **Performance**: Use scipy.sparse.csgraph.shortest_path(indices=...) for batch distances
