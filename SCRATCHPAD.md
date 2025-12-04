# SCRATCHPAD.md

## Current Session: 2025-12-04

### Working On

**Milestones 1-8 COMPLETE** - `neurospatial.basis` module fully implemented.

### Progress

- M1: Module setup & center selection ✅
- M2: Normalization helper & geodesic RBF basis ✅
- M3: Heat kernel wavelet basis ✅
- M4: Chebyshev filter basis ✅
- M5: Convenience function & visualization ✅
- M6: Integration & exports ✅
- M7: Documentation updates ✅
- M8: Final validation ✅

### Summary

Implemented maze-aware spatial basis functions for GLMs:

- `select_basis_centers`: kmeans, farthest_point, random methods
- `geodesic_rbf_basis`: RBF using shortest-path distances
- `heat_kernel_wavelet_basis`: Diffusion-based multi-scale
- `chebyshev_filter_basis`: Polynomial filters with k-bin locality
- `spatial_basis`: Convenience function with automatic parameter selection
- `plot_basis_functions`: Visualization helper

All 38 tests pass. Mypy and ruff clean.

### Notes

- Documentation uses accessible language (e.g., "bins away from center" instead of "hop distance")
- All examples in module docstring have `# doctest: +SKIP`

### Blockers

None.
