# Phase 1 — Finite-volume operator + all geometries

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md) · [contracts](shared-contracts.md)

Replace the diffusion kernel with the finite-volume operator so `bandwidth` is the true
physical σ on every layout. Ships `transition`/`density` modes, the reframed low-level API,
per-geometry builders, `W`-graph components, junction contraction, mesh skew guard, migration
docs, and the full grid-independence + σ-recovery test suite. `binned`/`resample` intensive
paths and the public `mode="average"` are **Phase 2**.

**Inputs to read first:**

- [design-correctness.md](design-correctness.md) §3–§5 — full rationale.
- [shared-contracts.md](shared-contracts.md) — C1–C6 (operator, modes, geometry, M, W-components, low-level API).
- [designs.md](designs.md) — D0 (assembly), D1 (per-geometry), D2 (junctions), D3 (skew), D4 (heat_kernel).
- [ops/smoothing.py:52-234](../../../src/neurospatial/ops/smoothing.py) — the function being rewritten (`compute_diffusion_kernels`, `_assign_gaussian_weights_from_distance`).
- [fields.py:44-148](../../../src/neurospatial/environment/fields.py) — `compute_kernel` (routes here); [fields.py:150-299](../../../src/neurospatial/environment/fields.py) — `smooth` (docstrings).
- [trajectory.py:483-499](../../../src/neurospatial/environment/trajectory.py) — `occupancy(bandwidth=)`; [trajectory.py:820-870](../../../src/neurospatial/environment/trajectory.py) — `transitions(method="diffusion")`.
- [_protocols.py:94-101](../../../src/neurospatial/environment/_protocols.py) — `bin_sizes` docstring to fix; [_protocols.py:50](../../../src/neurospatial/environment/_protocols.py) / [307](../../../src/neurospatial/environment/_protocols.py) — mode Literal/cache-key.
- Layout engines: [regular_grid.py](../../../src/neurospatial/layout/engines/regular_grid.py), [hexagonal.py](../../../src/neurospatial/layout/engines/hexagonal.py), [graph.py](../../../src/neurospatial/layout/engines/graph.py), [triangular_mesh.py](../../../src/neurospatial/layout/engines/triangular_mesh.py); [polar.py:187-238](../../../src/neurospatial/environment/polar.py) (polar `_POLAR` dispatch + sector `bin_sizes` at :413).
- Edge builders where `"A"` is stamped alongside `"distance"`: [graph_building.py:176](../../../src/neurospatial/layout/helpers/graph_building.py) (grids), [graph.py:261,296](../../../src/neurospatial/layout/helpers/graph.py) (track), [triangular_mesh.py:194](../../../src/neurospatial/layout/helpers/triangular_mesh.py) (mesh).
- [appendix.md](appendix.md) — NLD `_neighbor_centers` (junction contraction) reference.

**Contracts referenced:** [C1](shared-contracts.md#c1-finite-volume-operator), [C2](shared-contracts.md#c2-kernel-modes--orientation) (transition+density only this phase), [C3](shared-contracts.md#c3-per-geometry-finite-volume-data), [C4](shared-contracts.md#c4-canonical-m), [C5](shared-contracts.md#c5-component-structure-from-w), [C6](shared-contracts.md#c6-low-level-compute_diffusion_kernels-api) — do not weaken.

**Designs referenced:** [D0](designs.md#d0-module-layout--operator-assembly), [D1](designs.md#d1-per-geometry-finite-volume-builders), [D2](designs.md#d2-graph-junction-contraction), [D3](designs.md#d3-mesh-skew-guard), [D4](designs.md#d4-heat_kernel--per-component-renormalization).

## Tasks

- **New `src/neurospatial/ops/diffusion.py`** (D0/D4): `heat_kernel_from_W(W, volumes,
  sigma, *, mode)` (dense `expm`, clip, **per-mode normalization**, returns `Hᵀ`/`H·M⁻¹`/`H`
  — **all three low-level modes incl. `"average"`** per C6/D4; only public exposure is
  Phase 2), `_components_from_W`, `diffusion_kernel(env, sigma, *, mode)`, `_finite_volume_geometry`
  dispatch, and the five per-geometry builders `_cartesian_fv`/`_hex_fv`/`_polar_fv`/
  `_graph_fv`/`_mesh_fv` (D1). Unsupported layout → clear `NotImplementedError`.
- **Rewrite `ops/smoothing.py::compute_diffusion_kernels`** to `(graph, *, volumes, sigma,
  mode)` reading edge `"A"` (C6: missing `A` raises, `A=0` skips). Delete
  `_assign_gaussian_weights_from_distance` and the Gaussian-weight/`M⁻¹L`-fold path. Keep the
  `_LARGE_KERNEL_THRESHOLD` GB warning. Update the `neurospatial.ops` export + module
  docstring ([ops/__init__.py:99,188](../../../src/neurospatial/ops/__init__.py)).
- **Graph junction contraction** in `_graph_fv` (D2): Dijkstra along-track distance for
  inter-segment edges; intra-segment edges unchanged; straight track is a no-op.
- **Mesh skew guard** in `_mesh_fv` (D3): warn when `>5%` of interior edges exceed `30°`
  non-orthogonality.
- **Route `env.compute_kernel`** ([fields.py:124-142](../../../src/neurospatial/environment/fields.py))
  to `diffusion_kernel(env, sigma, mode=mode)`; modes `{"transition","density"}` this phase.
  The new path reads `env.bin_sizes` internally (C4) — drop the old `bin_sizes if
  mode=="density"` branch.
- **`transitions(method="diffusion")`** ([trajectory.py:820-870](../../../src/neurospatial/environment/trajectory.py)):
  return the **row-stochastic** matrix as `compute_kernel(mode="transition").T` (`= H`), not
  the raw transition kernel — it must be row-stochastic on non-uniform `M`, not rely on
  symmetry. Verify `occupancy(bandwidth=)` ([trajectory.py:495](../../../src/neurospatial/environment/trajectory.py))
  still uses `mode="transition"` (`Hᵀ`, mass-conserving) — correct, no change.
- **Docs (ship with this phase):** CHANGELOG entry (`bandwidth` is now the true physical σ;
  2D place fields smooth ~1.5–1.7× less for the same value + the `√(bin_size)` approximation
  note; **public break** to `neurospatial.ops.compute_diffusion_kernels`; `min_occupancy` is
  seconds-vs-seconds/volume and out of the grid-invariance guarantee). Bump
  `pyproject.toml` version 0.6.0 → 0.7.0. Fix docstrings: `fields.py::compute_kernel`/`smooth`
  ("physical units (σ)" is now accurate; density = **extensive** input → density),
  `ops/smoothing.py` module docstring (drop Gaussian-weight prose), `_protocols.py::bin_sizes`
  (shape `(n_bins,)`, per-bin **volume**, not `(n_dims,)`/widths).

## Deliberately not in this phase

- **PUBLIC `mode="average"` and the intensive-field routing of `binned`/`resample_field`** →
  [Phase 2](phase-2-intensive-fields-and-average-mode.md). The low-level `H`/`"average"` view
  **is** built in Phase 1 (`heat_kernel`/`compute_diffusion_kernels`, C6/D4) — what's deferred
  is exposing `mode="average"` on `env.smooth`/`compute_kernel` (its `Literal`/`valid_modes`)
  and rerouting `binned`/`resample`. Those keep `mode="density"` here (no worse than today's
  pre-existing intensive bias; documented).
- **Performance (cached eigenbasis / spectral truncation)** → PR2, separate plan. Keep the
  `heat_kernel`/`W`/`S`-conjugate seam (C1) so PR2 swaps internals only.
- **Nonuniform-Cartesian `grid_edges`** — excluded (C4); do not touch `_GridMixin.bin_sizes`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_bandwidth_grid_independent_1d` | measured σ ≈ `bandwidth` (rtol 0.02) across bin sizes {0.5,1,2,4}, 1D cartesian |
| `test_bandwidth_grid_independent_2d` | measured σ ≈ `bandwidth` (rtol 0.02) across bin sizes, 2D open field + masked |
| `test_sigma_recovery_hex` | hex: σ recovered across ≥2 hex sizes (a-priori tol, pinned in test) |
| `test_sigma_recovery_polar` | polar: σ recovered across ≥2 (n_dist,n_angle) resolutions; source interior, away from `r=0`/seam; measured in Cartesian coords (spec §4 protocol) |
| `test_sigma_recovery_graph` | linear track incl. a junction: σ recovered across bin sizes; junction not oversmoothed |
| `test_sigma_recovery_mesh` | well-shaped flat triangulation: σ recovered at 2 resolutions |
| `test_transition_is_column_stochastic_polar` | `compute_kernel(mode="transition")` columns sum to 1; `kernel @ counts` conserves `Σ` on non-uniform `M` |
| `test_transitions_row_stochastic_polar` | `transitions(method="diffusion")` rows sum to 1 on polar |
| `test_density_integrates_to_one` | `mode="density"`: `Σ_i M_i K[i,j] = 1` |
| `test_low_level_average_row_stochastic` | `compute_diffusion_kernels(..., mode="average")` returns row-stochastic `H` (rows sum to 1) — the low-level view Phase 2 depends on (C6) |
| `test_raw_heat_operator_m_self_adjoint` | `_raw_heat_operator` (pre-clip/normalize): `H·1 = 1` and `M_i H_ij ≈ M_j H_ji` on a non-uniform-`M` (polar) env — C1's raw invariants (D4 seam) |
| `test_compute_diffusion_kernels_rejects_bad_inputs` | `sigma <= 0`, wrong-shape `volumes`, non-finite/non-positive `volumes`, non-positive `"distance"`, and negative/NaN/inf `"A"` each raise (C6); `A == 0` does not raise |
| `test_apply_kernel_adjoint_nonuniform_M` | `apply_kernel(mode="adjoint", bin_sizes=M)` on the `density` kernel obeys the M-weighted inner-product contract on a non-uniform-`M` env (spec §4; regression, no live caller today) |
| `test_components_from_W_corner_split` | corner-only 8-connected pair → 2 `W`-components; no mass crosses the corner |
| `test_no_leakage_across_masked_wall` | point source beside a masked wall → ~0 mass across it |
| `test_compute_diffusion_kernels_missing_A_raises` | edge without `"A"` → `ValueError` |
| `test_compute_diffusion_kernels_A_zero_no_edge` | `A=0` edge carries no diffusion weight |
| `test_mesh_skew_guard_warns` | skewed mesh fixture → `UserWarning`; well-shaped → none (slow: mark if triangulation build is heavy) |

A-priori tolerances live in the tests as named constants; a geometry that misses is an
operator/spec correction, **not** a tolerance relaxation (spec §4).

## Fixtures

`conftest.py` (deterministic, **explicit construction — no random point sampling**):
1D + 2D regular grids at several bin sizes; a masked grid with a wall/hole; a hexagonal env;
a polar env (`from_polar_egocentric`); a linear track **with a junction** (W or T);
a well-shaped flat triangular mesh and a deliberately-skewed one. A shared point-source
`σ`-measurement helper (physical-coordinate second moment, spec §4 protocol).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task implemented as specified; C1–C6 upheld (esp. `mode="transition"` returns `Hᵀ`,
  components from `W`).
- "Deliberately not in this phase" honored — no **public** `"average"` (no `Literal` /
  `valid_modes` exposure), no `binned`/`resample` reroute. (The **low-level** `H`/`"average"`
  view in `heat_kernel`/`compute_diffusion_kernels` **is** required this phase — C6/D4.)
- Validation slice passes; slow mesh tests marked.
- Tests exercise behavior (measured σ, stochasticity, component counts), not tautologies;
  shared setup in fixtures.
- No plan/phase references in docstrings, test names, or module names.
- Old Gaussian-weight path (`_assign_gaussian_weights_from_distance`, old signature) is
  **removed**, not left beside the new one.
- CHANGELOG / version bump / docstring fixes are in this PR, not deferred.
