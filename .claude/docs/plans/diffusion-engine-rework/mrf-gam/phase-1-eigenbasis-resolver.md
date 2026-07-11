# Phase 1 — Live-component eigenbasis resolver

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#resolver)

Add the rank-based, live-component eigenbasis resolver that produces the reduced-rank penalty
basis for the GAM. Reuses PR2's cached finite-volume geometry and eigensolver — **no operator or
eigensolver change**. Internal only; nothing on the public surface calls it yet (it's scaffolding
phase-2 consumes).

**Inputs to read first:**

- `src/neurospatial/environment/fields.py:512` — `_diffusion_geometry` → `DiffusionGeometry(W, volumes, n_components, labels)`; the resolver reuses `W`/`volumes`/`labels`.
- `src/neurospatial/environment/fields.py:535` — `_diffusion_eigenbasis` cache pattern (`versioned_cached_property`).
- `src/neurospatial/ops/diffusion.py:290` — `_symmetric_conjugate` (builds `S`); `:382` — `_symmetric_eigenbasis` (per-component eigenpairs; global selection `:444-448`; **raises** on `rank < n_components` at `:419`); `:201` — `_components_from_W`.
- `src/neurospatial/environment/core.py:1030` — `n_bins` (active bins; the array length).
- [designs.md#resolver](designs.md#resolver) — the resolver skeleton + null-designation.

**Contracts referenced:**

- [`MRFBasis`](shared-contracts.md#mrfbasis) — the return contract; **do not weaken** its invariants (exactly `n_live_components` structural zeros in `d`, live-bin-only `B`, `r_eff` formula, dead-component exclusion, zero-occupancy → empty basis).

**Designs referenced:** [designs.md#resolver](designs.md#resolver).

## Tasks

- Define `MRFBasis` NamedTuple next to `DiffusionGeometry` (`ops/diffusion.py` or an `environment/_types` leaf), fields per [contract](shared-contracts.md#mrfbasis).
- **Cached global resolver (owns the eigensolve).** Add `Environment._mrf_eigenbasis` as a `versioned_cached_property` returning a **mutable holder** `{"Q": (n_bins, G), "Lam": (G,), "mode_comp": (G,), "G": int}` with **initial empty state `G=0`, empty arrays** (grow-by-replace, mirroring `_diffusion_eigenbasis` at `fields.py:535`; keyed by geometry only — no `(sigma, tol)`). Plus `_ensure_global_modes(holder, S, labels, needed_G) -> (Q, Lam, mode_comp)` ([designs.md → cached resolver](designs.md#resolver)): grows only when short; **recovers each mode's source W-component** from its (component-local, exact-zero-off-component) eigenvector — no change to `_symmetric_eigenbasis`, replacing the previously-undefined `labels_of_modes`. **Transient-dense (Finding 2):** past `dense_fraction·n` it computes a **call-local** dense basis and does **not** persist it to the holder (an `n×n` float64 matrix on the env would be GBs — the design forbids persisting dense bases); the holder stays at its last sparse `G`.
- **Pure selector (no eigensolve).** Add `select_live_basis(Q, Lam, mode_comp, volumes, labels, occupancy, *, rank) -> MRFBasis` to `ops/diffusion.py` (pure, array-only, unit-testable without an `Environment`), per [designs.md → pure selector](designs.md#resolver): live components (`Σ occupancy > 0`), `r_eff = max(n_live_components, min(n_live_bins, R))` (`R=250` when `None`), keep the `r_eff` smallest-λ **live-component** modes, **apply `M^{-1/2}` then restrict to live bins** (`B = (Q / sqrt(volumes)[:,None])[live_bins]` — Finding 2), zero exactly one null per live component. Degenerate `n_live_bins == 0` → `MRFBasis((0,0), (0,), empty, 0)`.
- **Glue with iterative growth.** Add `Environment._mrf_basis(self, occupancy, *, rank) -> MRFBasis` (`environment/fields.py`): fetch `_diffusion_geometry`, compute `r_eff`, **iteratively grow `G`** (dead components crowd out live modes, so the global `G` needed for `r_eff` live modes isn't known up front — Finding 3) via `_ensure_global_modes` until `≥ r_eff` live modes exist (or full rank), then call `select_live_basis`. Only the resolver eigensolves; the selector is pure masking, so a repeat call at the same-or-smaller rank / same live support is a cache hit.

## Deliberately not in this phase

- **The fit / REML** — phase-2. The resolver returns a basis; nothing fits against it yet.
- **The per-component-`eigsh` perf mitigation** (spec §11) — the global-over-request baseline ships; the mitigation is only implemented if the perf-benchmark task shows the `R` cap breaking.
- **Any `compute_spatial_rate` wiring** — phase-3.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_mrfbasis_shapes` | on a 2D open field: `B` is `(n_live_bins, r_eff)`, `d` is `(r_eff,)`, `r_eff == max(n_live_components, min(n_live_bins, R))`. |
| `test_structural_nulls_zeroed` | exactly `n_live_components` entries of `d` are `== 0.0` (bit-exact), one per live component; the rest are `> 0` and finite. |
| `test_live_component_budget` | with one **dead** (unvisited) component present: every `B` column indexes a bin in a live component; the resolver returns `r_eff` **live** modes (a global smallest-rank selection would have spent budget on the dead null). |
| `test_two_3node_paths_rank2` | two disjoint 3-node paths, both visited, `rank=2` → `n_live_components == 2`, `r_eff == 2`, all `d == 0.0` (the all-null basis; feeds the phase-2 `r==0` guard). |
| `test_zero_occupancy_empty_basis` | all-zero occupancy → `B.shape == (0, 0)`, `live_bins.size == 0`, `n_live_components == 0`. |
| `test_rank_clamped_both_ways` | `rank=1` (< `n_live_components`) → `r_eff == n_live_components`; `rank=10**9` → `r_eff == n_live_bins`. |
| `test_basis_applies_inv_sqrt_volume` | on a **nonuniform-volume** layout (polar or mesh): `B` equals `(M^{-1/2}Q)[live_bins]`, **not** `Q[live_bins]` — assert against a direct `eigvecs / sqrt(volumes)[:,None]` recomputation; they differ where volumes vary (guards Finding 2). |
| `test_reuses_pr2_geometry` | `_mrf_basis` does not rebuild `_diffusion_geometry` when called twice (cache hit; assert via a spy/`_state_version` unchanged). |
| `test_reuses_eigensolve` | a second `_mrf_basis(occupancy2, rank=r)` at the same-or-smaller `rank` / same live support does **not** re-run the eigensolver (spy on `_symmetric_eigenbasis` / assert holder `G` unchanged) — distinct from geometry reuse (Finding 6). |
| `test_selector_is_pure` | `select_live_basis(...)` never calls `_symmetric_eigenbasis` (spy asserts 0 calls) — it consumes cached `(Q, Lam, mode_comp)` and only masks (Finding 3 split). |
| `test_growth_past_dead_modes` | a **large dead** component whose low modes outnumber `r_eff` forces `_mrf_basis` to grow `G` iteratively until `≥ r_eff` **live** modes are exposed; the returned `B` still has `r_eff` live columns (guards the "global G not knowable up front" case — Finding 3). |
| `test_mode_component_recovery` | `mode_comp[m]` equals the W-component of every nonzero row of `Q[:, m]` (component-local recovery is correct; replaces the undefined `labels_of_modes`). |
| `test_dense_basis_not_persisted` | a request past `dense_fraction·n` returns a correct `MRFBasis` but leaves `_mrf_eigenbasis["G"]` at its last **sparse** value (the `n×n` dense `Q` is call-local, never stored on the env — Finding 2); the holder's initial state is `G==0`, empty arrays. |

## Fixtures

- Synthesized environments in the encoding `conftest.py`: a 2D open field; a masked env with two disconnected components; two disjoint 3-node linear tracks (via `from_graph`). Occupancy vectors constructed directly (active-bin order) — full and partial (one dead component). No real data needed; the resolver is deterministic geometry.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The resolver honors every [`MRFBasis`](shared-contracts.md#mrfbasis) invariant; nulls are bit-exact `0.0`, not thresholded.
- The operator / `_symmetric_eigenbasis` / `_diffusion_geometry` are **reused, not modified** (diff touches only new functions + `_mrf_basis`).
- "Deliberately not in this phase" honored — no fit, no wiring, no perf-mitigation.
- Tests exercise the invariants (shapes, exact zeros, dead-component exclusion), not tautologies; fixtures are shared.
- No plan/phase references in code or test names.
