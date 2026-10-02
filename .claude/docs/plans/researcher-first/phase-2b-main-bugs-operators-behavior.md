# Phase 2b — Fix the operator, heading and behavior bugs verified on `main`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

**Requires:** [Phase 1](phase-1-port-fixes.md). Independent of [Phase 2a](phase-2a-main-bugs-geometry-io.md). [Phase 3e](phase-3e-time-windows-kinematics.md) requires this phase, because it changes `heading_from_velocity`'s signature in files this phase edits.

Branching, commits, the CHANGELOG rule, the definition of done and the PR workflow are in [executing.md](executing.md). This file adds only what is specific to Phase 2b.

**Inputs to read first:**

- [src/neurospatial/ops/calculus.py](../../../../src/neurospatial/ops/calculus.py) (all 391 lines) and [ops/diffusion.py:1-31, 280-331, 1249-1323, 1386-1422](../../../../src/neurospatial/ops/diffusion.py). Diffusion defines `L = M⁻¹(Deg − W)` with `W_ij = A_ij / d_ij`. `_finite_volume_geometry` returns a copy of `env.connectivity` carrying a face measure `"A"` on every edge, plus the cell volumes `M`. `_graph_fv` also replaces track-junction chord distances with along-track lengths. Layouts outside its dispatch table, such as the `_ReconstructedLayout` of an NWB-reloaded environment, raise `NotImplementedError` (`:1276-1281`). Tasks 1 and 2.
- [src/neurospatial/environment/core.py:51, 1049-1118](../../../../src/neurospatial/environment/core.py) — `get_differential_operator`, cached through `versioned_cached_property`. Task 1.
- [docs/user-guide/differential-operators.md](../../../../docs/user-guide/differential-operators.md) (in the mkdocs nav, `mkdocs.yml:179`) and [examples/09_differential_operators.py](../../../../examples/09_differential_operators.py). Both document the operator Task 1 replaces. Task 1.
- [docs/reviews/REPOSITORY_AND_MATHEMATICAL_REVIEW_2026-10-01.md §2.1](../../../../docs/reviews/REPOSITORY_AND_MATHEMATICAL_REVIEW_2026-10-01.md).
- [src/neurospatial/ops/basis.py:575-757, 759-957, 959-1004](../../../../src/neurospatial/ops/basis.py) — `heat_kernel_wavelet_basis` (scale docs `:599-618`, "Laplacian weighting" Notes `:654-657`, `nx.laplacian_matrix(weight="distance")` at `:722-724`), `chebyshev_filter_basis` (`:902-911`) and `_estimate_spectral_radius`. Task 2.
- [src/neurospatial/ops/egocentric.py:655-817, 820-862, 928](../../../../src/neurospatial/ops/egocentric.py) — `heading_from_velocity` (forward-difference velocity at `:768-771`, low-speed mask at `:784`, interpolation call at `:815`) and `_interpolate_heading_circular` (chord interpolation at `:855-860`; also called by `heading_from_body_orientation` at `:928`). Tasks 3 and 4.
- [src/neurospatial/behavior/navigation.py:1732-1842](../../../../src/neurospatial/behavior/navigation.py) — `instantaneous_goal_alignment` (Returns `:1754-1756`, heading call `:1780`) and `goal_bias` (`min_speed` text `:1809`). Task 4.
- [src/neurospatial/behavior/decisions.py:402-485](../../../../src/neurospatial/behavior/decisions.py) — `pre_decision_heading_stats` (`min_speed` text `:417-418`, NaN filter `:468-469`). Task 5.
- **Files Phase 1 already changed** (re-locate line numbers by symbol): `ops/diffusion.py` (Phase 1 Task 5, docstring only; Tasks 1 and 2 here only call its helpers) and `CHANGELOG.md` (Phase 1 created `### Fixed` under `## [Unreleased]`; append to it).

**Contracts referenced:**

- [Error-message contract](shared-contracts.md#error-message-contract) — the `NotImplementedError` that Tasks 1 and 2 raise for layouts without finite-volume geometry names the operator the caller used (rule 3: "the domain word fits the call"), not "diffusion kernel", and ends with a `Fix:` line.

**Designs referenced:** none (the calculus operator is specified inline in Task 1 and the basis generator in Task 2).

## Tasks

Commits, regression-test-first and the per-commit CHANGELOG bullet follow [executing.md → While you work](executing.md#while-you-work). Phase-specific notes:

- Run a new regression test against the unmodified code with `uv run pytest <nodeid> -n 0` and quote the failure in the commit body. Do not use `git stash` for this.
- Tasks 2–5 were each verified on this branch (`da631a47`) with a minimal probe before being added. The measured numbers are quoted in each task and reused as the regression-test expectations.

1. **Physically scaled discrete calculus (`ops/calculus.py`).** On `main`, `D[·, e] = ∓√d_e`, so `abs(gradient(x)) = d**1.5` (1.0, 2.83, 8.0 at bin sizes 1, 2, 4). `div(grad(·))` is then the distance-weighted combinatorial Laplacian, the sign of `divergence` contradicts its docstring, and the goal bin of `−grad(distance)` comes out as a *source* (positive divergence at every bin size).

   Replace this with the mimetic finite-volume pair built on the geometry `env.smooth` uses. For edge `e = (i → j)`, take the length `d_e` and face measure `A_e` from `_finite_volume_geometry(env)`, and the volumes `M` from the same call:

   - `gradient(f)_e = (f_j − f_i) / d_e`, in field units per length unit;
   - `divergence(q)_i = −(1/M_i) Σ_e B_ie A_e q_e`, where `B` is the oriented incidence matrix. This is positive at sources.
   - Then `divergence(gradient(f)) = −M⁻¹(Deg − W) f = −L f` exactly, with diffusion's `W = A/d`. The continuum limit is ∇²f. `⟨grad f, q⟩_{A·d} = −⟨f, div q⟩_M` (discrete Gauss–Green).

   ```python
   def _fv_edges(env: Environment) -> tuple[nx.Graph, NDArray[np.float64]]:
       from neurospatial.ops.diffusion import _finite_volume_geometry

       try:
           return _finite_volume_geometry(env)  # graph copy (same edge order) + volumes
       except NotImplementedError as err:
           raise NotImplementedError(
               f"gradient/divergence need finite-volume cell geometry, which layout "
               f"{type(env.layout).__name__!r} does not provide ({err}).\n"
               "Fix: build the environment with a factory method, e.g. "
               "Environment.from_samples(positions, bin_size=...)."
           ) from err


   def compute_differential_operator(env: Environment) -> sparse.csc_matrix:
       fv_graph, _ = _fv_edges(env)
       edges = list(fv_graph.edges(data="distance"))
       n_bins, n_edges = env.n_bins, len(edges)
       if n_edges == 0:
           return sparse.csc_matrix((n_bins, 0), dtype=np.float64)
       src = np.fromiter((u for u, _, _ in edges), dtype=np.int64, count=n_edges)
       dst = np.fromiter((v for _, v, _ in edges), dtype=np.int64, count=n_edges)
       inv_d = 1.0 / np.fromiter((d for *_, d in edges), dtype=np.float64, count=n_edges)
       cols = np.arange(n_edges)
       return sparse.csc_matrix(
           (np.concatenate([-inv_d, inv_d]),
            (np.concatenate([src, dst]), np.concatenate([cols, cols]))),
           shape=(n_bins, n_edges),
       )


   def _compute_divergence_operator(env: Environment) -> sparse.csc_matrix:
       fv_graph, volumes = _fv_edges(env)
       grad_t = env.get_differential_operator()  # (n_bins, n_edges), entries ∓1/d_e
       area = np.fromiter((a for *_, a in fv_graph.edges(data="A")), dtype=np.float64,
                          count=grad_t.shape[1])
       length = np.fromiter((d for *_, d in fv_graph.edges(data="distance")),
                            dtype=np.float64, count=grad_t.shape[1])
       # grad_t @ diag(A·d) == B @ diag(A): the face flux of each edge.
       return (-sparse.diags(1.0 / np.asarray(volumes, dtype=np.float64))
               @ grad_t @ sparse.diags(area * length)).tocsc()
   ```

   The distances must come from the finite-volume copy, not from `env.connectivity`. Otherwise graph environments lose consistency with the diffusion generator at track junctions. `gradient` keeps `diff_op.T @ field`. `divergence` uses `env._divergence_operator_cached @ edge_field`. Add `_divergence_operator_cached` next to `_differential_operator_cached` (`core.py:1112-1118`), also as a `versioned_cached_property`. Edge order is `env.connectivity.edges()` order, which `_finite_volume_geometry`'s `.copy()` preserves (asserted by a test).

   **Callers and what changes for them.** No `src/` module calls `gradient` or `divergence`. `compute_differential_operator` is called only by `core.py:1118`, and it remains public in `ops.__all__` with the same signature.
   - **Values change for every user:** gradients are now in field units per length unit, and `div(grad)` changes sign and units, from Hz·cm to Hz/cm².
   - **Layouts without finite-volume geometry now raise.** On `main`, `gradient` and `divergence` work on any connectivity graph, including an NWB-reloaded environment (`_ReconstructedLayout`). After this task they raise the `NotImplementedError` above, the same limitation `env.smooth` has. This is a behavior change and gets its own CHANGELOG line.
   - **Update the docstrings that state the old identity `L = D @ D.T`:** the module docstring (`calculus.py:1-28`), `compute_differential_operator` (`:51-108`, including its doctest comparing with `nx.laplacian_matrix`), `gradient` (`:173-176`, `:213-226`), `divergence` (`:273-276`, `:339-351`), and `get_differential_operator` (`core.py:1069-1078`).
   - **Rewrite the tests that encode the old maths** in `tests/ops/test_differential.py` at `:32-47`, `:60-74`, `:86-107`, `:156-185`, `:239-260` and `:311-333`.
   - `ops/basis.py:722-724, 902-904` builds `nx.laplacian_matrix(weight="distance")` independently and is **not** a caller. Task 2 fixes it separately.

   **User documentation, in the same commit:**
   - **`docs/user-guide/differential-operators.md`** (in the mkdocs nav) states the old operator throughout:
     - `:38-52`: `D_{i,e} = ∓√w_e` and "Why Square Root Weighting?", which claims `L = D·Dᵀ` matches NetworkX's Laplacian. Replace them with the definitions above: `gradient` divides by `d_e`, `divergence` weights by `A_e / M_i`, and `div(grad) = −M⁻¹(Deg − W)`, the operator `env.smooth` uses.
     - `:205-216`: the "Compare with NetworkX Laplacian" block (prints a difference of "~0.0"). Replace it with the continuum check below (`divergence(env, gradient(env, x² + y²)) ≈ 4` in interior bins).
     - `:338-351`: the operator table row `D Dᵀ f = L f`, "Weighted vs. Unweighted Graphs" (√w scaling) and "Sign Convention" (±√w). Rewrite them to match.
     - `:360-401`: the `laplacian_smooth` example steps `f ← f − α·div(grad f)`. With the new sign that **sharpens** the field. It becomes `f ← f + α·div(grad f)`, with `α` in length² units. Explicit stability needs `α < 2/λ_max`, about `h²/4` on a 2-D grid of spacing `h`; state that.
     - Every call on the page is written `gradient(field, env)` / `divergence(edge_field, env)` and imported `from neurospatial import gradient, divergence`. On `main` the functions take `(env, field)` and are not exported at the package root. Fix the argument order and use `from neurospatial.ops import gradient, divergence` (`:10-11`, `:114`, `:132`, `:167`, `:208-209`, `:263`, `:269`, `:365`, `:391-392`, `:437`, `:478-479`).
     - The goal-sink example (`:255-280`, "Should be negative (sink)") becomes true; keep it.
   - **`examples/09_differential_operators.py`.** Change the operator formulas in the markdown at `:46-49`, `:127`, `:246` and `:326`. Replace the "equals NetworkX's Laplacian" verification (`:425-453`) with a check that `divergence(gradient(x² + y²))` is 4 in interior bins. The goal-sink claim at `:271` becomes true. Then run `uv run jupytext --sync examples/09_differential_operators.py`, re-execute the notebook so its stored outputs match (`mkdocs.yml:93` has `execute: false`), and run `uv run python docs/sync_notebooks.py`. CI checks `git diff --exit-code docs/examples`.
   - `.claude/API_REFERENCE.md:251-253` lists the names only, and `docs/user-guide/rl-primitives.md:467` is a comment only; neither needs a change. Search `.claude/` and `docs/` once more for `D @ D.T`, `sqrt(w` and `laplacian_matrix` before committing (exclude `docs/reviews/` and `.claude/docs/plans/`).

2. **Graph bases use the finite-volume diffusion generator (`ops/basis.py`).**
   - **Bug (verified, heat kernel).** `heat_kernel_wavelet_basis` builds `L` with `weight="distance"`, so an edge's *conductance* grows with its length. The Notes (`:654-657`) call these weights "edge conductance (inverse resistance)", and the `scales` docs (`:599-618`) promise bin-relative spread: "scale ≈ σ² / (2·bin_size²)", "scale=1.0: ~2-3 bins radius". Probe `p2extra/b_basis.py` (40 × 40-bin `from_grid_mask` grid, centre bin, `scales=[1.0]`, `normalize="none"`) measures the heat kernel's σ along x as **2.77, 3.91 and 6.17 bins** at bin sizes 1, 2 and 5. It grows as √bin_size, where the documented relation gives √2 = 1.41 bins at every bin size.
   - **Fix (heat kernel).** Use the generator `env.smooth` uses, `M⁻¹(Deg − W)` with `W = A/d`, expressed in bin-spacing units so the documented `scales` semantics become exact:

     ```python
     def _diffusion_generator(env: Environment) -> sparse.csr_matrix:
         """Finite-volume diffusion generator ``M⁻¹(Deg − W)``, in units of bin spacing².

         The same operator ``env.smooth`` uses (``W = A/d``; see ops/diffusion.py),
         multiplied by the squared median nearest-neighbour bin spacing so that
         ``exp(-s L)`` has standard deviation ``sqrt(2 s)`` bins.
         """
         from scipy.spatial import cKDTree

         from neurospatial.ops.binning import _estimate_typical_bin_spacing
         from neurospatial.ops.diffusion import _assemble_W, _finite_volume_geometry

         graph, volumes = _finite_volume_geometry(env)
         W = _assemble_W(graph, env.n_bins)
         degree = sparse.diags(np.asarray(W.sum(axis=1)).ravel())
         spacing = _estimate_typical_bin_spacing(cKDTree(env.bin_centers), env.bin_centers)
         return (spacing**2 * sparse.diags(1.0 / np.asarray(volumes, dtype=np.float64))
                 @ (degree - W)).tocsr()
     ```

     The same probe with this generator (`p2extra/b_basis_fix.py`) gives σ = 1.4142 bins at scale 1 and 2.8284 bins at scale 4, for bin sizes 1, 2 and 5, along both the axis and the diagonal. In `heat_kernel_wavelet_basis`, replace `:722-724` with `laplacian = _diffusion_generator(env)`. Layouts without finite-volume geometry raise a `NotImplementedError` that names `heat_kernel_wavelet_basis` and carries a `Fix:` line, as in Task 1.
   - **Docstrings (heat kernel).** `scales`: "diffusion time in units of bin spacing²; the kernel's standard deviation is `sqrt(2·scale)` bins (0.5 → 1, 1 → 1.41, 2 → 2, 4 → 2.83 bins), independent of `bin_size`". Replace the "~N bins radius" table, the "Adjust if your bin_size is unusual" advice and the "Roughly" relation with that statement. Rewrite "Laplacian weighting" (`:654-657`) as "uses the finite-volume generator of `env.smooth` (face measure over centre distance), so spread is isotropic and in bin units". Update the module-docstring line (`:36`) and the out-of-range error text at `:707-714` to the same numbers.
   - **Chebyshev: deliberately unchanged, documented.** `chebyshev_filter_basis` rescales by `λ_max` (`L_scaled = 2L/λ_max − I`), so a global change of weight scale has no effect. Its documented contract is hop locality: "nonzero only for bins within k steps", "max_degree ≈ desired_radius / bin_size". That holds on this branch (degree-3 support radius = 3 bins, probe `p2extra/b_basis_aniso.py`). Switching it to the finite-volume generator would drop corner-only (diagonal) steps on Cartesian grids, where `A = 0` (`diffusion.py:_cartesian_fv`), and change that contract. Instead, add a Notes sentence: "The operator is the connectivity graph's Laplacian with `distance` edge weights, rescaled by its spectral radius. It defines hop locality only and is not a physical diffusion; use `heat_kernel_wavelet_basis` for bin-size-independent spread."
   - **Tests to update.** `tests/ops/test_basis.py` heat-kernel tests that depend on the old spread (search `heat_kernel_wavelet_basis`). The `_estimate_spectral_radius` tests at `:454-500` build their own distance Laplacian and are unaffected.

3. **Heading interpolation follows the shorter arc, uniformly in angle (`ops/egocentric.py:820-862`).**
   - **Bug (verified).** `_interpolate_heading_circular` interpolates `cos` and `sin` linearly and takes `arctan2`, which moves along the chord, not the arc. Probe `p2extra/a_heading*.py`, 3 masked samples between valid headings `0` and `turn`:

     | turn | interpolated (quarter, half, three-quarter) | uniform in angle | max error |
     | --- | --- | --- | --- |
     | π/2 | 0.322, 0.785, 1.249 | 0.393, 0.785, 1.178 | 0.071 rad |
     | π − 0.1 | 0.050, 1.521, 2.992 | 0.760, 1.521, 2.281 | 0.711 rad |
     | π | 0.000, 1.571, 3.142 | 0.785, 1.571, 2.356 | 0.785 rad |

     Near a 180° turn the filled samples snap to the endpoints. For `π/2 → −π/2` the result is `[π/2, 0, −π/2]`: the midpoint lands on 0 and the quarter points equal the endpoints. Through the public API, an east → stop → west path (`heading_from_velocity(..., min_speed=0.5)`) gives `[0, 0, 0, 1.571, 3.142, …]`, a step instead of a turn.
   - **Fix.** Replace the body after the empty-mask check with:

     ```python
     valid = ~mask & np.isfinite(heading)
     valid_idx = np.flatnonzero(valid)
     if valid_idx.size == 0:
         return heading
     # Unwrap so consecutive valid headings differ by at most pi, interpolate
     # linearly in angle (the shorter arc), then wrap back to (-pi, pi].
     unwrapped = np.unwrap(heading[valid_idx])
     filled = np.interp(np.flatnonzero(mask), valid_idx, unwrapped)
     result = heading.copy()
     result[mask] = np.pi - np.mod(np.pi - filled, 2.0 * np.pi)
     return result
     ```

     Masked samples before the first or after the last valid sample keep taking the nearest valid heading (`np.interp` holds the end values), as today. Unmasked NaN headings are no longer used as interpolation anchors; previously they turned neighbouring filled values into NaN. For an exactly antipodal pair both arcs are equally short; the turn follows the sign of the stored difference (`np.unwrap` leaves a difference of exactly ±π unchanged).
   - **Docstring.** Rewrite the summary ("interpolate along the shorter arc, linearly in angle") and note the tie rule. `heading_from_velocity`'s `min_speed` text and `heading_from_body_orientation`'s NaN text say "interpolated along the shorter arc".

4. **Goal alignment excludes stationary samples, as documented (`behavior/navigation.py:1732-1842`).**
   - **Bug (verified).** `instantaneous_goal_alignment` documents "NaN for stationary periods", and `goal_bias` documents "Stationary periods excluded". But `heading_from_velocity(..., min_speed=…)` *interpolates* low-speed headings, so stationary samples get a fabricated alignment. Probe `p2extra/e_align*.py`, 10 Hz, east for 1 s, stopped for 3 s, north for 1 s, goal due east: 30 stationary samples, **0** of them NaN. `goal_bias` = **0.5862**, against **0.5055** over the moving samples only. With east → stop → west, the 19 stationary samples are all ±1.
   - **Decision: fix the behaviour, not the docstrings.** Both docstrings and the `min_speed` parameter ("Stationary periods excluded") state the same intent, and a stationary animal has no movement direction to align.
   - **Fix.** Add one private helper next to `heading_from_velocity` in `ops/egocentric.py`, and have `heading_from_velocity` call it for its velocity, speed and heading (one implementation):

     ```python
     def _velocity_heading_and_speed(
         positions: NDArray[np.float64], dt: float, *, bandwidth: float = 0.0
     ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
         """Per-sample velocity heading and speed (forward differences, last repeated).

         Returns
         -------
         heading, speed : ndarray, shape (n_samples,)
             Uninterpolated heading in radians and speed in position units per second.
         """
     ```

     `instantaneous_goal_alignment` computes `heading, speed = _velocity_heading_and_speed(positions, dt)`, sets `heading[speed < min_speed] = np.nan`, and returns `np.cos(heading - goal_direction(positions, goal))`. NaN propagates to the stationary samples. `goal_bias` already drops NaN, so it now averages the moving samples only; it still returns NaN when every sample is stationary. The `allow_all_nan` call and its comment go.
   - **Docstrings.** `instantaneous_goal_alignment` Returns: "NaN where speed < `min_speed`". `goal_bias` `min_speed`: "samples slower than this are excluded from the mean".

5. **`pre_decision_heading_stats` excludes stationary samples, as documented (`behavior/decisions.py:402-485`).**
   - **Bug (verified).** The NaN filter at `:468-469` ("Filter out NaN (stationary periods)") never fires unless *every* sample is stationary, because `heading_from_velocity` has already interpolated the stationary headings. Probe `p2extra/f_predec.py`, the same east/stop/north path: on this branch `(mean_direction, circular_variance, mrl) = (0.7854, 0.1849, 0.8151)`; over the 20 moving samples only it is `(0.7854, 0.2929, 0.7071)`.
   - **Fix.** Use Task 4's helper: `heading, speed = _velocity_heading_and_speed(positions, dt)`, then `valid_headings = heading[speed >= min_speed]`. The existing all-stationary return `(0.0, 1.0, 0.0)` is unchanged.

6. **User-facing documentation check (no separate commit unless something is missing).** Per [executing.md](executing.md#while-you-work), each of Tasks 1–5 already appended its own bullet under `## [Unreleased]` → `### Fixed` in `CHANGELOG.md`, with the symptom and the numbers from the Validation slice, and Task 1 shipped the user-guide page and example 09. Before opening the PR, check that all five are present and that these are marked as **behavior changes**:
   - Task 1: the new units, the sign of `divergence`, and `div(grad) = ∇²` (negative semidefinite);
   - Task 1: `gradient`, `divergence` and `compute_differential_operator` raise `NotImplementedError` on layouts without finite-volume geometry, including NWB-reloaded environments (they worked on `main`);
   - Task 2: heat-kernel `scales` now give `sqrt(2·scale)` bins at every bin size; `heat_kernel_wavelet_basis` raises on the same layouts as Task 1;
   - Task 3: masked headings are interpolated along the shorter arc, uniformly in angle;
   - Tasks 4–5: goal alignment and pre-decision heading statistics exclude samples below `min_speed`.

   A missing bullet is added in a `docs: complete phase 2b changelog` commit.

## Deliberately not in this phase

- **The geometry, I/O, simulation and event bugs** (W-maze linearization, polar plots, NWB conversion and units, simulated field width, population PETH, VTE windows) are [Phase 2a](phase-2a-main-bugs-geometry-io.md).
- **Changing `chebyshev_filter_basis`'s operator.** Task 2 documents why it keeps the distance-weighted graph Laplacian (hop locality is its contract, and a global weight scale cancels in its rescaling).
- **Finite-volume geometry for reloaded non-grid NWB environments.** Tasks 1 and 2 make them raise with a fix, like `env.smooth`; supporting them is an overview Non-Goal.
- **Gap-aware (per-run) kinematics.** Tasks 3, 4 and 5 fix single-recording behaviour. [Phase 3e](phase-3e-time-windows-kinematics.md) later runs `heading_from_velocity`, its interpolation and Task 4's speed mask per valid run, with `times` in place of `dt`.
- **Object-vector work and citation fixes** are [Phase 5a](phase-5a-object-vector-frames.md); errors and executable docs are [Phase 4a](phase-4a-errors.md) and [Phase 4b](phase-4b-docs-that-run.md).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/ops/test_differential.py::test_gradient_of_x_is_unit_slope[1,2,4]` | `from_grid_mask` 20×20 cm with edges `np.arange(0, 20+h/2, h)`: on x-axis edges, `np.abs(gradient(env, x)) == 1.0` at atol 1e-12 (`main`: 1.0, 2.8284, 8.0) |
| `…::test_div_grad_quadratic_is_continuum_laplacian[1,2,4]` | `divergence(gradient(x²+y²)) == 4.0` on interior (degree-8) bins (`main`: −15.31, −122.51, −980.08) |
| `…::test_div_grad_equals_negative_diffusion_generator[grid, hex, W-maze graph]` | equals `−(M⁻¹(Deg − W)) f` built from `diffusion._assemble_W(_finite_volume_geometry(env)[0])` for random f, at rtol 1e-10 |
| `…::test_divergence_is_negative_adjoint_of_gradient` | `Σ A·d·grad(f)·q == −Σ M·f·div(q)` at rtol 1e-10 |
| `…::test_goal_is_a_sink` | goal = bin nearest the arena centre, h ∈ {1, 2, 4}: `divergence(−gradient(dist_to_goal))[goal] < 0` (`main`: positive at every h, +12 for the (10.5, 10.5) bin at h = 1) |
| `…::test_operator_edge_order_matches_connectivity` | the finite-volume copy's `list(edges())` equals `list(env.connectivity.edges())`; `gradient(f)[k]` equals `(f[v] − f[u]) / d` for the k-th connectivity edge |
| `…::test_operators_raise_without_finite_volume_geometry` | the Y-track graph environment (nodes `(0,0), (0,100), (-50,150), (50,150)`, `edge_spacing=10`, `bin_size=3`) written with `write_environment` and read back with `read_environment` (`_ReconstructedLayout`) → `gradient` and `divergence` raise `NotImplementedError` whose message names `gradient/divergence` and has a `Fix:` line (`main`, probe: they return arrays of shape `(81,)` and `(82,)`, while `env.smooth` already raises) |
| `uv run pytest --doctest-modules src/neurospatial/ops/calculus.py src/neurospatial/environment/core.py -n 0` | the rewritten doctests pass |
| `examples/09_differential_operators` re-executed; `uv run --extra docs mkdocs build --strict` | the notebook runs, its continuum check prints ≈ 4, and `git diff --exit-code docs/examples` is clean after `docs/sync_notebooks.py` |
| `tests/ops/test_basis.py::test_heat_kernel_spread_is_bin_size_independent[1,2,5]` | 40 × 40-bin `from_grid_mask` grid, centre bin, `normalize="none"`: σ along x and along the diagonal equals `sqrt(2·scale)` bins at rtol 1e-3 for `scale ∈ {1, 4}` (before the fix: 2.77, 3.91, 6.17 bins at scale 1) |
| `…::test_heat_kernel_uses_smooth_generator` | the basis column for a centre equals `expm_multiply(-s · spacing² · M⁻¹(Deg − W), δ)` built from `diffusion._assemble_W(_finite_volume_geometry(env)[0])` at rtol 1e-10, on a hexagonal env and the W maze |
| `…::test_chebyshev_hop_locality_unchanged` (guard) | degree-k column is zero beyond k connectivity hops and identical to the pre-change output (rtol 1e-12) |
| `tests/ops/test_reference_frames.py::test_heading_interpolation_is_uniform_on_shorter_arc[π/2, π−0.1, π]` | 3 masked samples between valid `0` and `turn`: the filled values equal `turn · [0.25, 0.5, 0.75]` at atol 1e-12, with no value equal to an endpoint (before the fix at π − 0.1: `[0.050, 1.521, 2.992]`) |
| `…::test_heading_interpolation_across_pi_wrap` | valid `π − 0.1` and `−π + 0.1` with one masked sample between → `π` (or `−π`) exactly within atol 1e-12, never near 0 (shorter arc through ±π) |
| `…::test_antipodal_turn_has_no_jump_to_zero` | valid `π/2` and `−π/2` with 3 masked samples → `[π/4, 0, −π/4]`, monotone (before the fix: `[π/2, 0, −π/2]`, a step) |
| `…::test_heading_from_velocity_turn_is_gradual` | east → 3 slow samples → west, `min_speed=0.5`: consecutive headings differ by ≤ π/4 + 1e-12 (before the fix: a jump of π/2 then π/2 at the midpoint) |
| `…::test_nan_anchor_not_propagated` | an unmasked NaN heading next to a masked run leaves the masked values finite |
| `tests/behavior/test_behavior_navigation.py::test_goal_alignment_nan_when_stationary` | the Task 4 path: all 30 stationary samples are NaN and moving samples are finite; `goal_bias == 0.5055` at atol 1e-4 (before the fix: 0 NaN, 0.5862) |
| `…::test_goal_bias_all_stationary_is_nan` (guard) | a motionless trajectory → `goal_bias` is NaN without raising |
| `tests/behavior/test_decision_analysis.py::test_pre_decision_heading_stats_excludes_stationary` | the same path: `(0.7854, 0.2929, 0.7071)` at atol 1e-4 (before the fix: `(0.7854, 0.1849, 0.8151)`) |

Run the slice with `-n 0`. Then satisfy [executing.md → Definition of done](executing.md#definition-of-done). Phase-specific extra check: the `test_operators_raise_without_finite_volume_geometry` row needs the `nwb` extra, so run `uv run pytest tests/ops/test_differential.py -n 0 -rs` under `uv sync --all-extras` and confirm it did not skip.

## Fixtures

- **Calculus.** `Environment.from_grid_mask(np.ones((n, n), bool), (e, e))` with exactly uniform edges, so that `d` equals `h` exactly. Use a hexagonal `from_samples` env (seed 0) and the W maze (`bl(0,0) bm(50,0) br(100,0) al(0,50) am(50,50) ar(100,50)`, bin 5, built inline) for the generator-equivalence test. The reloaded environment for the raise test is written to `tmp_path` with `NWBHDF5IO` (`pytest.importorskip("pynwb")`).
- **Basis.** The same `from_grid_mask` construction with `edges = np.arange(41) * h`, centre bin = the bin nearest `(20h, 20h)`; the hexagonal env and W maze above.
- **Headings, goal alignment, pre-decision.** Inline paths copied from the probes: 10 Hz `t = np.arange(50) * 0.1`, `x = np.minimum(t, 1) * 20`, `y = np.maximum(t - 4, 0) * 20` (east, stop, north; goal `(1000, 20)`).
- **Probe scripts** live in the session scratchpad (`p2extra/`); the tests reproduce them inline and do not depend on those files.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- The user guide's `laplacian_smooth` example still smooths (variance of a noisy field decreases) with the new sign.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
